import importlib.util
import os
import sys
import json
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _anthropic_available():
    return importlib.util.find_spec("anthropic") is not None


class TestFormatTimestamp:
    def test_zero_seconds(self):
        from pipeline.llm_analysis import _format_timestamp
        assert _format_timestamp(0) == "00:00:00"

    def test_under_one_minute(self):
        from pipeline.llm_analysis import _format_timestamp
        assert _format_timestamp(44.7) == "00:00:45"

    def test_minutes_and_seconds(self):
        from pipeline.llm_analysis import _format_timestamp
        # spec example: 00:12:24
        assert _format_timestamp(12 * 60 + 24) == "00:12:24"

    def test_hours(self):
        from pipeline.llm_analysis import _format_timestamp
        assert _format_timestamp(3 * 3600 + 6 * 60 + 9) == "03:06:09"

    def test_rounds_to_nearest_second(self):
        from pipeline.llm_analysis import _format_timestamp
        assert _format_timestamp(12.6) == "00:00:13"


class TestClassifySpeakers:
    def test_longest_talker_is_rep(self):
        from pipeline.llm_analysis import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 30.0},
            {"speaker": "SPEAKER_01", "start": 30.0, "end": 40.0},
        ]
        mapping = _classify_speakers(transcript)
        assert mapping == {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT"}

    def test_aggregates_across_segments(self):
        from pipeline.llm_analysis import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 5.0},
            {"speaker": "SPEAKER_00", "start": 5.0, "end": 25.0},
            {"speaker": "SPEAKER_01", "start": 25.0, "end": 35.0},
            {"speaker": "SPEAKER_00", "start": 35.0, "end": 40.0},
        ]
        # SPEAKER_00 = 25s, SPEAKER_01 = 15s
        mapping = _classify_speakers(transcript)
        assert mapping["SPEAKER_00"] == "REP"
        assert mapping["SPEAKER_01"] == "PROSPECT"

    def test_third_speaker_is_other(self):
        from pipeline.llm_analysis import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 50.0},
            {"speaker": "SPEAKER_01", "start": 50.0, "end": 80.0},
            {"speaker": "SPEAKER_02", "start": 80.0, "end": 82.0},
        ]
        mapping = _classify_speakers(transcript)
        assert mapping["SPEAKER_02"] == "OTHER"

    def test_empty_transcript(self):
        from pipeline.llm_analysis import _classify_speakers
        assert _classify_speakers([]) == {}


class TestComputeTalkRatio:
    def test_simple_split(self):
        from pipeline.llm_analysis import _compute_talk_ratio
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 60.0},
            {"speaker": "SPEAKER_01", "start": 60.0, "end": 100.0},
        ]
        speaker_map = {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT"}
        assert _compute_talk_ratio(transcript, speaker_map) == {"rep": 60, "prospect": 40}

    def test_aggregates_across_segments(self):
        from pipeline.llm_analysis import _compute_talk_ratio
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 20.0},
            {"speaker": "SPEAKER_01", "start": 20.0, "end": 60.0},
            {"speaker": "SPEAKER_00", "start": 60.0, "end": 80.0},
        ]
        speaker_map = {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT"}
        # REP = 40s, PROSPECT = 40s -> 50/50
        assert _compute_talk_ratio(transcript, speaker_map) == {"rep": 50, "prospect": 50}

    def test_ignores_other_speakers(self):
        from pipeline.llm_analysis import _compute_talk_ratio
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 80.0},
            {"speaker": "SPEAKER_01", "start": 80.0, "end": 100.0},
            {"speaker": "SPEAKER_02", "start": 100.0, "end": 120.0},
        ]
        speaker_map = {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT", "SPEAKER_02": "OTHER"}
        # only REP + PROSPECT counted: 80 + 20 = 100 -> 80/20
        assert _compute_talk_ratio(transcript, speaker_map) == {"rep": 80, "prospect": 20}

    def test_zero_talk_time_returns_zeros(self):
        from pipeline.llm_analysis import _compute_talk_ratio
        assert _compute_talk_ratio([], {}) == {"rep": 0, "prospect": 0}


class TestFaceForSegment:
    def test_returns_nearest_to_midpoint(self):
        from pipeline.llm_analysis import _face_for_segment
        seg = {"start": 215.0, "end": 225.0}  # midpoint 220
        face = [
            {"timestamp": 210.0, "dominant_emotion": "angry", "scores": {}},
            {"timestamp": 220.0, "dominant_emotion": "happy", "scores": {}},
            {"timestamp": 230.0, "dominant_emotion": "sad", "scores": {}},
        ]
        result = _face_for_segment(seg, face)
        assert result["timestamp"] == 220.0

    def test_empty_face_returns_none(self):
        from pipeline.llm_analysis import _face_for_segment
        seg = {"start": 0.0, "end": 5.0}
        assert _face_for_segment(seg, []) is None


class TestMergeSegments:
    def _inputs(self):
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0,
             "text": " Hello ", "words": [{"word": "Hello"}], "avg_logprob": -0.1},
        ]
        audio = [{"pitch_mean": 180.0, "pitch_std": 12.0, "energy_mean": 0.05,
                  "speech_rate": 3.0, "pause_ratio": 0.1, "zcr": 0.08}]
        voice = [{"valence": 0.4, "arousal": 0.5, "dominance": 0.6}]
        face = [{"timestamp": 2.0, "dominant_emotion": "neutral",
                 "scores": {"neutral": 0.9, "happy": 0.1}}]
        return transcript, audio, voice, face

    def test_relabels_speaker_to_role(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, f = self._inputs()
        merged = _merge_segments(t, a, v, f, {"SPEAKER_00": "REP"})
        assert merged[0]["speaker"] == "REP"

    def test_strips_words_and_avg_logprob(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, f = self._inputs()
        merged = _merge_segments(t, a, v, f, {"SPEAKER_00": "REP"})
        assert "words" not in merged[0]
        assert "avg_logprob" not in merged[0]

    def test_strips_text_whitespace(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, f = self._inputs()
        merged = _merge_segments(t, a, v, f, {"SPEAKER_00": "REP"})
        assert merged[0]["text"] == "Hello"

    def test_attaches_audio_voice_face(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, f = self._inputs()
        merged = _merge_segments(t, a, v, f, {"SPEAKER_00": "REP"})
        assert merged[0]["audio"]["pitch_mean"] == 180.0
        assert merged[0]["voice"]["valence"] == 0.4
        assert merged[0]["face"]["dominant_emotion"] == "neutral"

    def test_no_face_emotion_omits_face_key(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, _ = self._inputs()
        merged = _merge_segments(t, a, v, [], {"SPEAKER_00": "REP"})
        assert "face" not in merged[0]

    def test_index_misalignment_does_not_crash(self):
        from pipeline.llm_analysis import _merge_segments
        t, _, _, _ = self._inputs()
        # audio/voice shorter than transcript
        merged = _merge_segments(t, [], [], [], {"SPEAKER_00": "REP"})
        assert "audio" not in merged[0]
        assert "voice" not in merged[0]


class TestSystemPrompt:
    def test_includes_signal_interpretation_guide(self):
        from pipeline.llm_analysis import SYSTEM_PROMPT
        assert "valence" in SYSTEM_PROMPT
        assert "arousal" in SYSTEM_PROMPT
        assert "dominance" in SYSTEM_PROMPT

    def test_instructs_dissonance_surfacing(self):
        from pipeline.llm_analysis import SYSTEM_PROMPT
        assert "dissonance" in SYSTEM_PROMPT.lower()

    def test_requires_timestamps(self):
        from pipeline.llm_analysis import SYSTEM_PROMPT
        assert "timestamp" in SYSTEM_PROMPT.lower()


class TestAnalysisTool:
    def test_forces_required_fields(self):
        from pipeline.llm_analysis import ANALYSIS_TOOL
        required = ANALYSIS_TOOL["input_schema"]["required"]
        assert set(required) == {
            "engagement_score", "deal_probability",
            "critical_moments", "recommendations",
        }

    def test_critical_moments_has_coaching(self):
        from pipeline.llm_analysis import ANALYSIS_TOOL
        cm = ANALYSIS_TOOL["input_schema"]["properties"]["critical_moments"]["items"]
        assert "coaching" in cm["required"]
        assert "timestamp" in cm["required"]

    def test_tool_name(self):
        from pipeline.llm_analysis import ANALYSIS_TOOL
        assert ANALYSIS_TOOL["name"] == "submit_analysis"


class TestBuildTranscriptPrompt:
    def _merged(self):
        return [{
            "speaker": "PROSPECT", "start": 12.4, "end": 18.1,
            "text": "I'm not sure the pricing makes sense.",
            "audio": {"pitch_mean": 180, "pitch_std": 24, "energy_mean": 0.04,
                      "speech_rate": 3.2, "pause_ratio": 0.18, "zcr": 0.06},
            "voice": {"valence": 0.31, "arousal": 0.22, "dominance": 0.41},
            "face": {"dominant_emotion": "neutral", "scores": {"neutral": 0.71}},
        }]

    def test_includes_speaker_and_text(self):
        from pipeline.llm_analysis import _build_transcript_prompt
        prompt = _build_transcript_prompt(self._merged())
        assert "PROSPECT" in prompt
        assert "I'm not sure the pricing makes sense." in prompt

    def test_includes_timestamp(self):
        from pipeline.llm_analysis import _build_transcript_prompt
        prompt = _build_transcript_prompt(self._merged())
        assert "00:00:12" in prompt

    def test_excludes_modalities(self):
        from pipeline.llm_analysis import _build_transcript_prompt
        prompt = _build_transcript_prompt(self._merged())
        assert "Audio:" not in prompt
        assert "Voice emotion:" not in prompt
        assert "Facial:" not in prompt


class TestBuildMultimodalPrompt:
    def _merged(self):
        return [{
            "speaker": "PROSPECT", "start": 12.4, "end": 18.1,
            "text": "I'm not sure the pricing makes sense.",
            "audio": {"pitch_mean": 180, "pitch_std": 24, "energy_mean": 0.04,
                      "speech_rate": 3.2, "pause_ratio": 0.18, "zcr": 0.06},
            "voice": {"valence": 0.31, "arousal": 0.22, "dominance": 0.41},
            "face": {"dominant_emotion": "neutral", "scores": {"neutral": 0.71}},
        }]

    def test_includes_all_modalities(self):
        from pipeline.llm_analysis import _build_multimodal_prompt
        prompt = _build_multimodal_prompt(self._merged())
        assert "Audio:" in prompt
        assert "Voice emotion:" in prompt
        assert "Facial:" in prompt

    def test_includes_signal_values(self):
        from pipeline.llm_analysis import _build_multimodal_prompt
        prompt = _build_multimodal_prompt(self._merged())
        assert "0.31" in prompt  # valence
        assert "neutral" in prompt

    def test_includes_dissonance_instruction(self):
        from pipeline.llm_analysis import _build_multimodal_prompt
        prompt = _build_multimodal_prompt(self._merged())
        assert "dissonance" in prompt.lower()


def _install_fake_anthropic(fake):
    """Inject a fake `anthropic` module so the lazy import resolves to it."""
    sys.modules["anthropic"] = fake


_COMPLETE_INPUT = {
    "engagement_score": 61,
    "deal_probability": 45,
    "critical_moments": [],
    "recommendations": [],
}


def _make_fake_anthropic(create_side_effect=None, create_return=None, stop_reason="end_turn"):
    fake = MagicMock()
    fake.RateLimitError = type("RateLimitError", (Exception,), {})
    client = MagicMock()
    if create_side_effect is not None:
        client.messages.create.side_effect = create_side_effect
    else:
        resp = create_return if create_return is not None else MagicMock()
        resp.stop_reason = stop_reason
        client.messages.create.return_value = resp
    fake.Anthropic.return_value = client
    return fake


class TestExtractToolInput:
    def test_returns_tool_use_input(self):
        from pipeline.llm_analysis import _extract_tool_input
        block = MagicMock(type="tool_use", input={"engagement_score": 70})
        response = MagicMock(content=[MagicMock(type="text"), block])
        assert _extract_tool_input(response) == {"engagement_score": 70}

    def test_raises_when_no_tool_use_block(self):
        from pipeline.llm_analysis import _extract_tool_input
        response = MagicMock(content=[MagicMock(type="text")])
        with pytest.raises(RuntimeError, match="tool_use"):
            _extract_tool_input(response)


class TestCallClaude:
    def test_extracts_tool_input(self):
        fake = _make_fake_anthropic(
            create_return=MagicMock(content=[MagicMock(type="tool_use", input=_COMPLETE_INPUT)])
        )
        _install_fake_anthropic(fake)
        try:
            from pipeline.llm_analysis import _call_claude
            result = _call_claude("prompt", "key")
        finally:
            del sys.modules["anthropic"]
        assert result == _COMPLETE_INPUT

    def test_uses_forced_tool_choice(self):
        fake = _make_fake_anthropic(
            create_return=MagicMock(content=[MagicMock(type="tool_use", input=_COMPLETE_INPUT)])
        )
        _install_fake_anthropic(fake)
        try:
            from pipeline.llm_analysis import _call_claude
            _call_claude("prompt", "key")
        finally:
            del sys.modules["anthropic"]
        client = fake.Anthropic.return_value
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_analysis"}
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["max_tokens"] == 8192

    def test_retries_once_on_rate_limit(self):
        fake = _make_fake_anthropic()
        RateLimitError = fake.RateLimitError
        client = fake.Anthropic.return_value
        success = MagicMock(
            content=[MagicMock(type="tool_use", input=_COMPLETE_INPUT)],
            stop_reason="end_turn",
        )
        client.messages.create.side_effect = [RateLimitError("limit"), success]
        _install_fake_anthropic(fake)
        try:
            with patch("pipeline.llm_analysis.time.sleep") as sleep:
                from pipeline.llm_analysis import _call_claude
                result = _call_claude("prompt", "key")
        finally:
            del sys.modules["anthropic"]
        assert result == _COMPLETE_INPUT
        assert client.messages.create.call_count == 2
        sleep.assert_called_once_with(10)

    def test_passes_api_key_to_client(self):
        fake = _make_fake_anthropic(
            create_return=MagicMock(content=[MagicMock(type="tool_use", input=_COMPLETE_INPUT)])
        )
        _install_fake_anthropic(fake)
        try:
            from pipeline.llm_analysis import _call_claude
            _call_claude("prompt", "secret-key")
        finally:
            del sys.modules["anthropic"]
        fake.Anthropic.assert_called_once_with(api_key="secret-key")

    def test_raises_on_max_tokens_truncation(self):
        fake = _make_fake_anthropic(
            create_return=MagicMock(content=[MagicMock(type="tool_use", input=_COMPLETE_INPUT)]),
            stop_reason="max_tokens",
        )
        _install_fake_anthropic(fake)
        try:
            from pipeline.llm_analysis import _call_claude
            with pytest.raises(RuntimeError, match="truncat"):
                _call_claude("prompt", "key")
        finally:
            del sys.modules["anthropic"]

    def test_raises_on_incomplete_tool_input(self):
        incomplete = {
            "engagement_score": 1, "deal_probability": 1, "critical_moments": [],
        }
        fake = _make_fake_anthropic(
            create_return=MagicMock(content=[MagicMock(type="tool_use", input=incomplete)]),
            stop_reason="end_turn",
        )
        _install_fake_anthropic(fake)
        try:
            from pipeline.llm_analysis import _call_claude
            with pytest.raises(RuntimeError, match="incomplete"):
                _call_claude("prompt", "key")
        finally:
            del sys.modules["anthropic"]


class TestValidateAnalysis:
    def test_complete_input_does_not_raise(self):
        from pipeline.llm_analysis import _validate_analysis
        _validate_analysis(_COMPLETE_INPUT)

    def test_missing_field_raises(self):
        from pipeline.llm_analysis import _validate_analysis
        incomplete = {"engagement_score": 1, "deal_probability": 1, "critical_moments": []}
        with pytest.raises(RuntimeError, match="incomplete"):
            _validate_analysis(incomplete)


class TestRunAnalysis:
    def _write_inputs(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        transcript = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 60.0, "text": " Hi "}]
        audio = [{"pitch_mean": 180, "pitch_std": 12, "energy_mean": 0.05,
                  "speech_rate": 3.0, "pause_ratio": 0.1, "zcr": 0.08}]
        voice = [{"valence": 0.4, "arousal": 0.5, "dominance": 0.6}]
        face = [{"timestamp": 30.0, "dominant_emotion": "neutral",
                 "scores": {"neutral": 1.0}}]
        for name, data in (("transcript.json", transcript),
                           ("audio_features.json", audio),
                           ("voice_emotion.json", voice),
                           ("face_emotion.json", face)):
            (out / name).write_text(json.dumps(data))
        return out

    def test_writes_analysis_json_with_both_modes(self, tmp_path):
        out = self._write_inputs(tmp_path)
        output_file = out / "analysis.json"
        llm_out = {"engagement_score": 70, "deal_probability": 50,
                   "critical_moments": [], "recommendations": ["x"]}
        with patch("pipeline.llm_analysis._call_claude", return_value=dict(llm_out)):
            from pipeline.llm_analysis import run_analysis
            run_analysis(
                str(out / "transcript.json"),
                str(out / "audio_features.json"),
                str(out / "voice_emotion.json"),
                str(out / "face_emotion.json"),
                str(output_file),
                api_key="key",
            )
        saved = json.loads(output_file.read_text())
        assert set(saved.keys()) == {"transcript_only", "multimodal"}
        assert saved["transcript_only"]["engagement_score"] == 70
        assert saved["multimodal"]["engagement_score"] == 70

    def test_injects_talk_ratio_into_both_modes(self, tmp_path):
        out = self._write_inputs(tmp_path)
        output_file = out / "analysis.json"
        llm_out = {"engagement_score": 1, "deal_probability": 1,
                   "critical_moments": [], "recommendations": []}
        with patch("pipeline.llm_analysis._call_claude", return_value=dict(llm_out)):
            from pipeline.llm_analysis import run_analysis
            run_analysis(
                str(out / "transcript.json"),
                str(out / "audio_features.json"),
                str(out / "voice_emotion.json"),
                str(out / "face_emotion.json"),
                str(output_file),
                api_key="key",
            )
        saved = json.loads(output_file.read_text())
        # single speaker -> REP gets 100% of rep+prospect talk time
        assert saved["transcript_only"]["talk_ratio"] == {"rep": 100, "prospect": 0}
        assert saved["multimodal"]["talk_ratio"] == {"rep": 100, "prospect": 0}

    def test_calls_claude_twice_with_different_prompts(self, tmp_path):
        out = self._write_inputs(tmp_path)
        output_file = out / "analysis.json"
        llm_out = {"engagement_score": 1, "deal_probability": 1,
                   "critical_moments": [], "recommendations": []}
        with patch("pipeline.llm_analysis._call_claude", return_value=dict(llm_out)) as mock:
            from pipeline.llm_analysis import run_analysis
            run_analysis(
                str(out / "transcript.json"),
                str(out / "audio_features.json"),
                str(out / "voice_emotion.json"),
                str(out / "face_emotion.json"),
                str(output_file),
                api_key="key",
            )
        assert mock.call_count == 2
        first_prompt = mock.call_args_list[0].args[0]
        second_prompt = mock.call_args_list[1].args[0]
        assert "TRANSCRIPT" in first_prompt
        assert "MEETING" in second_prompt

    def test_raises_on_missing_input(self, tmp_path):
        from pipeline.llm_analysis import run_analysis
        with pytest.raises(FileNotFoundError, match="not found"):
            run_analysis(
                str(tmp_path / "nope.json"),
                str(tmp_path / "nope2.json"),
                str(tmp_path / "nope3.json"),
                str(tmp_path / "nope4.json"),
                str(tmp_path / "out.json"),
                api_key="key",
            )

    def test_raises_when_api_key_missing(self, tmp_path):
        out = self._write_inputs(tmp_path)
        from pipeline.llm_analysis import run_analysis
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            run_analysis(
                str(out / "transcript.json"),
                str(out / "audio_features.json"),
                str(out / "voice_emotion.json"),
                str(out / "face_emotion.json"),
                str(out / "analysis.json"),
                api_key=None,
            )


def _real_outputs_available():
    return all(
        os.path.exists(f"output/{name}.json")
        for name in ("transcript", "audio_features", "voice_emotion", "face_emotion")
    )


def _load_real_outputs():
    outputs = {}
    for name in ("transcript", "audio_features", "voice_emotion", "face_emotion"):
        with open(f"output/{name}.json") as f:
            outputs[name] = json.load(f)
    return outputs


@pytest.mark.skipif(not _real_outputs_available(), reason="requires the four real output/*.json")
class TestClassifySpeakersRealData:
    def test_longest_talker_is_rep(self):
        from pipeline.llm_analysis import _classify_speakers
        mapping = _classify_speakers(_load_real_outputs()["transcript"])
        assert mapping["SPEAKER_00"] == "REP"
        assert mapping["SPEAKER_01"] == "PROSPECT"
        assert mapping["SPEAKER_02"] == "OTHER"


@pytest.mark.skipif(not _real_outputs_available(), reason="requires the four real output/*.json")
class TestComputeTalkRatioRealData:
    def test_measured_ratio_matches_real_talk_time(self):
        from pipeline.llm_analysis import _classify_speakers, _compute_talk_ratio
        d = _load_real_outputs()
        sm = _classify_speakers(d["transcript"])
        # SPEAKER_00 ~494s, SPEAKER_01 ~225s -> 69/31
        assert _compute_talk_ratio(d["transcript"], sm) == {"rep": 69, "prospect": 31}


@pytest.mark.skipif(not _real_outputs_available(), reason="requires the four real output/*.json")
class TestMergeSegmentsRealData:
    def _merged(self):
        from pipeline.llm_analysis import _classify_speakers, _merge_segments
        d = _load_real_outputs()
        sm = _classify_speakers(d["transcript"])
        return _merge_segments(d["transcript"], d["audio_features"], d["voice_emotion"], d["face_emotion"], sm)

    def test_one_merged_record_per_transcript_segment(self):
        merged = self._merged()
        assert len(merged) == 197

    def test_drops_word_timestamps_and_avg_logprob(self):
        merged = self._merged()
        assert not any("words" in x for x in merged)
        assert not any("avg_logprob" in x for x in merged)

    def test_every_segment_has_audio_voice_and_face(self):
        merged = self._merged()
        assert all("audio" in x and "voice" in x and "face" in x for x in merged)

    def test_speakers_relabeled_to_roles(self):
        merged = self._merged()
        assert set(x["speaker"] for x in merged) == {"REP", "PROSPECT", "OTHER"}


@pytest.mark.skipif(not _real_outputs_available(), reason="requires the four real output/*.json")
class TestPromptsRealData:
    def _merged(self):
        from pipeline.llm_analysis import _classify_speakers, _merge_segments
        d = _load_real_outputs()
        sm = _classify_speakers(d["transcript"])
        return _merge_segments(d["transcript"], d["audio_features"], d["voice_emotion"], d["face_emotion"], sm)

    def test_transcript_prompt_excludes_all_modalities(self):
        from pipeline.llm_analysis import _build_transcript_prompt
        prompt = _build_transcript_prompt(self._merged())
        assert "Audio:" not in prompt
        assert "Voice emotion:" not in prompt
        assert "Facial:" not in prompt

    def test_multimodal_prompt_contains_signal_fields(self):
        from pipeline.llm_analysis import _build_multimodal_prompt
        prompt = _build_multimodal_prompt(self._merged())
        assert "Audio:" in prompt
        assert "Voice emotion:" in prompt
        assert "valence=" in prompt
        assert "Facial:" in prompt

    def test_multimodal_prompt_contains_real_dissonance_moment(self):
        from pipeline.llm_analysis import _build_multimodal_prompt
        prompt = _build_multimodal_prompt(self._merged())
        # 03:39 prospect question: text reads positive, voice valence ~0.39 (hidden concern)
        assert "What would that look like for the company" in prompt


@pytest.mark.skipif(
    not os.path.exists("output/analysis.json"),
    reason="requires output/analysis.json (run: python pipeline/05_llm_analysis.py)",
)
class TestRealAnalysisArtifact:
    def _load(self):
        with open("output/analysis.json") as f:
            return json.load(f)

    def test_both_modes_have_full_schema(self):
        d = self._load()
        for mode in ("transcript_only", "multimodal"):
            assert set(d[mode].keys()) == {
                "engagement_score", "deal_probability", "talk_ratio",
                "critical_moments", "recommendations",
            }

    def test_scores_are_ints_in_range(self):
        d = self._load()
        for mode in ("transcript_only", "multimodal"):
            assert isinstance(d[mode]["engagement_score"], int)
            assert isinstance(d[mode]["deal_probability"], int)
            assert 0 <= d[mode]["engagement_score"] <= 100
            assert 0 <= d[mode]["deal_probability"] <= 100

    def test_measured_talk_ratio_injected_into_both_modes(self):
        d = self._load()
        for mode in ("transcript_only", "multimodal"):
            assert d[mode]["talk_ratio"] == {"rep": 69, "prospect": 31}

    def test_multimodal_sees_at_least_as_many_moments_as_transcript_only(self):
        d = self._load()
        assert len(d["multimodal"]["critical_moments"]) >= len(d["transcript_only"]["critical_moments"])

    def test_both_modes_have_recommendations(self):
        d = self._load()
        assert len(d["transcript_only"]["recommendations"]) >= 1
        assert len(d["multimodal"]["recommendations"]) >= 1


class TestRunAnalysisIntegration:
    @pytest.mark.skipif(
        not _anthropic_available()
        or not os.path.exists("output/transcript.json")
        or not os.path.exists("output/audio_features.json")
        or not os.path.exists("output/voice_emotion.json")
        or not os.path.exists("output/face_emotion.json"),
        reason="requires anthropic SDK + all four upstream output JSONs",
    )
    def test_with_real_outputs(self, tmp_path):
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        from pipeline.llm_analysis import run_analysis
        out = tmp_path / "analysis.json"
        result = run_analysis(
            "output/transcript.json",
            "output/audio_features.json",
            "output/voice_emotion.json",
            "output/face_emotion.json",
            str(out),
            api_key=api_key,
        )
        assert set(result.keys()) == {"transcript_only", "multimodal"}
        for mode in ("transcript_only", "multimodal"):
            assert isinstance(result[mode]["engagement_score"], int)
            assert isinstance(result[mode]["deal_probability"], int)
            assert 0 <= result[mode]["engagement_score"] <= 100
            assert 0 <= result[mode]["deal_probability"] <= 100
            assert isinstance(result[mode]["critical_moments"], list)
            assert isinstance(result[mode]["recommendations"], list)
            assert len(result[mode]["recommendations"]) >= 1
            # talk_ratio is measured deterministically (not LLM-generated) and identical in both modes
            assert result[mode]["talk_ratio"] == {"rep": 69, "prospect": 31}
        # killer feature: multimodal surfaces >= as many moments as text-only
        assert len(result["multimodal"]["critical_moments"]) >= len(result["transcript_only"]["critical_moments"])
        # multimodal actually used the voice/face signals transcript-only structurally cannot see
        mm_text = json.dumps(result["multimodal"]).lower()
        to_text = json.dumps(result["transcript_only"]).lower()
        assert any(term in mm_text for term in ("valence", "arousal", "dominance", "facial"))
        for term in ("valence", "arousal"):
            assert term not in to_text
