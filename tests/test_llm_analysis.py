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
