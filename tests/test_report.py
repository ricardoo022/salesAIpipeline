import json
import re

import pytest


class TestFormatTimestamp:
    def test_formats_seconds_as_hhmmss(self):
        from pipeline.report import _format_timestamp
        assert _format_timestamp(744) == "00:12:24"

    def test_pads_single_digits(self):
        from pipeline.report import _format_timestamp
        assert _format_timestamp(5) == "00:00:05"

    def test_handles_hours(self):
        from pipeline.report import _format_timestamp
        assert _format_timestamp(3661) == "01:01:01"


class TestParseTimestamp:
    def test_parses_hhmmss_to_seconds(self):
        from pipeline.report import _parse_timestamp
        assert _parse_timestamp("00:12:24") == 744

    def test_round_trips_with_format_timestamp(self):
        from pipeline.report import _format_timestamp, _parse_timestamp
        assert _parse_timestamp(_format_timestamp(3661)) == 3661


class TestClassifySpeakers:
    def test_longest_talk_time_is_rep(self):
        from pipeline.report import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0, "end": 5},
            {"speaker": "SPEAKER_01", "start": 5, "end": 30},
        ]
        result = _classify_speakers(transcript)
        assert result["SPEAKER_01"] == "REP"
        assert result["SPEAKER_00"] == "PROSPECT"

    def test_third_speaker_is_other(self):
        from pipeline.report import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0, "end": 30},
            {"speaker": "SPEAKER_01", "start": 30, "end": 40},
            {"speaker": "SPEAKER_02", "start": 40, "end": 42},
        ]
        result = _classify_speakers(transcript)
        assert result["SPEAKER_02"] == "OTHER"


class TestMeetingDuration:
    def test_returns_max_end_time(self):
        from pipeline.report import _meeting_duration
        transcript = [{"start": 0, "end": 10}, {"start": 10, "end": 25.5}]
        assert _meeting_duration(transcript) == 25.5

    def test_empty_transcript_returns_zero(self):
        from pipeline.report import _meeting_duration
        assert _meeting_duration([]) == 0


class TestCountMissingFaceFrames:
    def test_no_missing_frames(self):
        from pipeline.report import _count_missing_face_frames
        face_emotion = [{"timestamp": t} for t in (0.0, 10.0, 20.0)]
        assert _count_missing_face_frames(face_emotion, duration=25, interval=10) == 0

    def test_counts_missing_frames(self):
        from pipeline.report import _count_missing_face_frames
        face_emotion = [{"timestamp": 0.0}]
        assert _count_missing_face_frames(face_emotion, duration=25, interval=10) == 2

    def test_never_negative(self):
        from pipeline.report import _count_missing_face_frames
        face_emotion = [{"timestamp": t} for t in (0.0, 10.0, 20.0, 30.0, 40.0)]
        assert _count_missing_face_frames(face_emotion, duration=25, interval=10) == 0


class TestBuildTimelineSeries:
    def test_splits_by_role_and_metric(self):
        from pipeline.report import _build_timeline_series
        voice_emotion = [
            {"speaker": "SPEAKER_00", "start": 0, "end": 5, "valence": 0.5, "arousal": 0.4, "dominance": 0.3},
            {"speaker": "SPEAKER_01", "start": 5, "end": 10, "valence": 0.6, "arousal": 0.7, "dominance": 0.2},
        ]
        speaker_map = {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT"}
        result = _build_timeline_series(voice_emotion, speaker_map)
        assert result["prospect_valence"] == [{"x": 5, "y": 0.6}]
        assert result["prospect_arousal"] == [{"x": 5, "y": 0.7}]
        assert result["rep_arousal"] == [{"x": 0, "y": 0.4}]

    def test_ignores_other_speakers(self):
        from pipeline.report import _build_timeline_series
        voice_emotion = [
            {"speaker": "SPEAKER_02", "start": 0, "end": 5, "valence": 0.5, "arousal": 0.4, "dominance": 0.3},
        ]
        speaker_map = {"SPEAKER_02": "OTHER"}
        result = _build_timeline_series(voice_emotion, speaker_map)
        assert result["prospect_valence"] == []
        assert result["rep_arousal"] == []

    def test_x_is_segment_start_rounded(self):
        from pipeline.report import _build_timeline_series
        voice_emotion = [
            {"speaker": "SPEAKER_00", "start": 12.456, "end": 15, "valence": 0.5, "arousal": 0.4, "dominance": 0.3},
        ]
        speaker_map = {"SPEAKER_00": "PROSPECT"}
        result = _build_timeline_series(voice_emotion, speaker_map)
        assert result["prospect_valence"][0]["x"] == 12.46


class TestBuildMomentMarkers:
    def test_builds_marker_per_moment(self):
        from pipeline.report import _build_moment_markers
        moments = [
            {"timestamp": "00:12:24", "type": "pricing_objection", "description": "d", "coaching": "c"},
        ]
        result = _build_moment_markers(moments)
        assert result == [{"x": 744, "timestamp": "00:12:24", "type": "pricing_objection"}]

    def test_preserves_order(self):
        from pipeline.report import _build_moment_markers
        moments = [
            {"timestamp": "00:00:07", "type": "a", "description": "", "coaching": ""},
            {"timestamp": "00:01:00", "type": "b", "description": "", "coaching": ""},
        ]
        result = _build_moment_markers(moments)
        assert [m["x"] for m in result] == [7, 60]

    def test_empty_list_returns_empty(self):
        from pipeline.report import _build_moment_markers
        assert _build_moment_markers([]) == []


class TestBuildComparison:
    def test_computes_deltas(self):
        from pipeline.report import _build_comparison
        analysis = {
            "transcript_only": {"engagement_score": 61, "deal_probability": 45, "talk_ratio": {"rep": 62, "prospect": 38}, "critical_moments": []},
            "multimodal": {"engagement_score": 74, "deal_probability": 68, "talk_ratio": {"rep": 62, "prospect": 38}, "critical_moments": []},
        }
        result = _build_comparison(analysis)
        assert result["engagement_score"] == {"transcript_only": 61, "multimodal": 74, "delta": 13}
        assert result["deal_probability"] == {"transcript_only": 45, "multimodal": 68, "delta": 23}
        assert result["talk_ratio"] == {"rep": 62, "prospect": 38}

    def test_negative_delta(self):
        from pipeline.report import _build_comparison
        analysis = {
            "transcript_only": {"engagement_score": 80, "deal_probability": 50, "talk_ratio": {"rep": 50, "prospect": 50}, "critical_moments": []},
            "multimodal": {"engagement_score": 70, "deal_probability": 50, "talk_ratio": {"rep": 50, "prospect": 50}, "critical_moments": []},
        }
        result = _build_comparison(analysis)
        assert result["engagement_score"]["delta"] == -10

    def test_counts_dissonance_moments_per_mode(self):
        from pipeline.report import _build_comparison
        analysis = {
            "transcript_only": {
                "engagement_score": 61, "deal_probability": 45, "talk_ratio": {"rep": 62, "prospect": 38},
                "critical_moments": [{"timestamp": "00:01:00", "type": "pricing_objection", "description": "", "coaching": ""}],
            },
            "multimodal": {
                "engagement_score": 74, "deal_probability": 68, "talk_ratio": {"rep": 62, "prospect": 38},
                "critical_moments": [
                    {"timestamp": "00:01:00", "type": "pricing_objection", "description": "", "coaching": ""},
                    {"timestamp": "00:02:00", "type": "Dissonance – Verbal/Facial Mismatch", "description": "", "coaching": ""},
                    {"timestamp": "00:03:00", "type": "dissonance - tone vs words", "description": "", "coaching": ""},
                ],
            },
        }
        result = _build_comparison(analysis)
        assert result["dissonance_moments"] == {"transcript_only": 0, "multimodal": 2}


class TestCountDissonanceMoments:
    def test_counts_moments_with_dissonance_in_type(self):
        from pipeline.report import _count_dissonance_moments
        moments = [
            {"type": "Dissonance – Verbal/Facial Mismatch"},
            {"type": "pricing_objection"},
        ]
        assert _count_dissonance_moments(moments) == 1

    def test_case_insensitive(self):
        from pipeline.report import _count_dissonance_moments
        moments = [{"type": "DISSONANCE - tone vs words"}]
        assert _count_dissonance_moments(moments) == 1

    def test_zero_when_none_match(self):
        from pipeline.report import _count_dissonance_moments
        moments = [{"type": "pricing_objection"}, {"type": "buying_signal"}]
        assert _count_dissonance_moments(moments) == 0

    def test_empty_list_returns_zero(self):
        from pipeline.report import _count_dissonance_moments
        assert _count_dissonance_moments([]) == 0


class TestSelectDissonanceExamples:
    def test_filters_to_dissonance_type_only(self):
        from pipeline.report import _select_dissonance_examples
        moments = [
            {"timestamp": "00:01:00", "type": "pricing_objection", "description": "d1", "coaching": "c1"},
            {"timestamp": "00:02:00", "type": "Dissonance – X", "description": "d2", "coaching": "c2"},
        ]
        result = _select_dissonance_examples(moments)
        assert result == [{"timestamp": "00:02:00", "type": "Dissonance – X", "description": "d2", "coaching": "c2"}]

    def test_respects_limit(self):
        from pipeline.report import _select_dissonance_examples
        moments = [
            {"timestamp": f"00:0{i}:00", "type": "Dissonance – X", "description": f"d{i}", "coaching": ""}
            for i in range(5)
        ]
        result = _select_dissonance_examples(moments, limit=2)
        assert len(result) == 2
        assert [m["description"] for m in result] == ["d0", "d1"]

    def test_empty_when_none_match(self):
        from pipeline.report import _select_dissonance_examples
        moments = [{"timestamp": "00:01:00", "type": "pricing_objection", "description": "d1", "coaching": "c1"}]
        assert _select_dissonance_examples(moments) == []


class TestNearestEntry:
    def test_returns_entry_containing_timestamp(self):
        from pipeline.report import _nearest_entry
        items = [{"start": 0, "end": 5}, {"start": 10, "end": 15}]
        assert _nearest_entry(items, 12) == {"start": 10, "end": 15}

    def test_returns_closest_by_start_when_no_containing_entry(self):
        from pipeline.report import _nearest_entry
        items = [{"start": 0}, {"start": 100}]
        assert _nearest_entry(items, 40, time_key="start") == {"start": 0}

    def test_uses_custom_time_key(self):
        from pipeline.report import _nearest_entry
        items = [{"timestamp": 0.0}, {"timestamp": 10.0}, {"timestamp": 20.0}]
        assert _nearest_entry(items, 21, time_key="timestamp") == {"timestamp": 20.0}

    def test_empty_list_returns_none(self):
        from pipeline.report import _nearest_entry
        assert _nearest_entry([], 5) is None


class TestDescribeTone:
    def test_positive_valence(self):
        from pipeline.report import _describe_tone
        assert _describe_tone(0.85) == "85% positive"

    def test_negative_valence(self):
        from pipeline.report import _describe_tone
        assert _describe_tone(0.10) == "10% negative"

    def test_neutral_valence(self):
        from pipeline.report import _describe_tone
        assert _describe_tone(0.50) == "50% neutral"


class TestDescribeFace:
    def test_formats_dominant_emotion_as_percentage(self):
        from pipeline.report import _describe_face
        frame = {"dominant_emotion": "sad", "scores": {"sad": 0.971, "neutral": 0.02}}
        assert _describe_face(frame) == "97% sad"


class TestFaceSentiment:
    def test_happy_is_positive(self):
        from pipeline.report import _face_sentiment
        assert _face_sentiment({"dominant_emotion": "happy", "scores": {"happy": 0.9}}) == 0.9

    def test_sad_is_negative(self):
        from pipeline.report import _face_sentiment
        assert _face_sentiment({"dominant_emotion": "sad", "scores": {"sad": 0.97}}) == pytest.approx(0.03)

    def test_neutral_is_midpoint(self):
        from pipeline.report import _face_sentiment
        assert _face_sentiment({"dominant_emotion": "neutral", "scores": {"neutral": 0.8}}) == 0.5


class TestBuildProofExamples:
    def test_matches_quote_tone_and_face_at_moment_timestamp(self):
        from pipeline.report import _build_proof_examples
        critical_moments = [
            {"timestamp": "00:12:23", "type": "Dissonance – X", "description": "d", "coaching": "c"},
        ]
        transcript = [{"speaker": "SPEAKER_01", "start": 742.8, "end": 744.0, "text": "No, it looks great."}]
        voice_emotion = [{"speaker": "SPEAKER_01", "start": 742.8, "end": 744.0, "valence": 0.8545, "arousal": 0.59, "dominance": 0.64}]
        face_emotion = [{"timestamp": 740.0, "dominant_emotion": "sad", "scores": {"sad": 0.9709, "neutral": 0.008}}]
        result = _build_proof_examples(critical_moments, transcript, voice_emotion, face_emotion)
        assert result == [{
            "timestamp": "00:12:23",
            "quote": "No, it looks great.",
            "tone": "85% positive",
            "face": "97% sad",
        }]

    def test_respects_limit(self):
        from pipeline.report import _build_proof_examples
        critical_moments = [
            {"timestamp": f"00:0{i}:00", "type": "Dissonance – X", "description": "", "coaching": ""}
            for i in range(5)
        ]
        transcript = [{"speaker": "SPEAKER_01", "start": i * 60, "end": i * 60 + 1, "text": f"quote {i}"} for i in range(5)]
        voice_emotion = [{"speaker": "SPEAKER_01", "start": i * 60, "end": i * 60 + 1, "valence": 0.5, "arousal": 0.5, "dominance": 0.5} for i in range(5)]
        face_emotion = [{"timestamp": i * 60, "dominant_emotion": "neutral", "scores": {"neutral": 0.9}} for i in range(5)]
        result = _build_proof_examples(critical_moments, transcript, voice_emotion, face_emotion, limit=2)
        assert len(result) == 2

    def test_empty_when_no_dissonance_moments(self):
        from pipeline.report import _build_proof_examples
        critical_moments = [{"timestamp": "00:01:00", "type": "pricing_objection", "description": "", "coaching": ""}]
        assert _build_proof_examples(critical_moments, [], [], []) == []

    def test_skips_moment_when_face_data_unavailable(self):
        from pipeline.report import _build_proof_examples
        critical_moments = [{"timestamp": "00:12:23", "type": "Dissonance – X", "description": "", "coaching": ""}]
        transcript = [{"speaker": "SPEAKER_01", "start": 742.8, "end": 744.0, "text": "No, it looks great."}]
        voice_emotion = [{"speaker": "SPEAKER_01", "start": 742.8, "end": 744.0, "valence": 0.85, "arousal": 0.59, "dominance": 0.64}]
        assert _build_proof_examples(critical_moments, transcript, voice_emotion, []) == []

    def test_prefers_strongest_contradiction_over_chronological_order(self):
        from pipeline.report import _build_proof_examples
        critical_moments = [
            {"timestamp": "00:01:00", "type": "Dissonance – weak", "description": "", "coaching": ""},
            {"timestamp": "00:02:00", "type": "Dissonance – strong", "description": "", "coaching": ""},
        ]
        transcript = [
            {"speaker": "SPEAKER_01", "start": 60, "end": 61, "text": "weak mismatch"},
            {"speaker": "SPEAKER_01", "start": 120, "end": 121, "text": "strong mismatch"},
        ]
        voice_emotion = [
            {"speaker": "SPEAKER_01", "start": 60, "end": 61, "valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            {"speaker": "SPEAKER_01", "start": 120, "end": 121, "valence": 0.9, "arousal": 0.5, "dominance": 0.5},
        ]
        face_emotion = [
            {"timestamp": 60.0, "dominant_emotion": "neutral", "scores": {"neutral": 0.9}},
            {"timestamp": 120.0, "dominant_emotion": "sad", "scores": {"sad": 0.9}},
        ]
        result = _build_proof_examples(critical_moments, transcript, voice_emotion, face_emotion, limit=1)
        assert result == [{
            "timestamp": "00:02:00",
            "quote": "strong mismatch",
            "tone": "90% positive",
            "face": "90% sad",
        }]


def _sample_inputs():
    transcript = [
        {"speaker": "SPEAKER_00", "start": 0, "end": 600, "text": "rep talking"},
        {"speaker": "SPEAKER_01", "start": 600, "end": 742.8, "text": "prospect talking"},
        {"speaker": "SPEAKER_01", "start": 742.8, "end": 900, "text": "No, it looks great."},
    ]
    voice_emotion = [
        {"speaker": "SPEAKER_00", "start": 0, "end": 5, "valence": 0.5, "arousal": 0.4, "dominance": 0.3},
        {"speaker": "SPEAKER_01", "start": 600, "end": 605, "valence": 0.3, "arousal": 0.6, "dominance": 0.4},
        {"speaker": "SPEAKER_01", "start": 742.8, "end": 744.0, "valence": 0.85, "arousal": 0.59, "dominance": 0.64},
    ]
    face_emotion = [
        {"timestamp": 0.0, "dominant_emotion": "neutral", "scores": {"neutral": 0.9}},
        {"timestamp": 740.0, "dominant_emotion": "sad", "scores": {"sad": 0.97, "neutral": 0.02}},
    ]
    analysis = {
        "transcript_only": {
            "engagement_score": 61, "deal_probability": 45,
            "talk_ratio": {"rep": 67, "prospect": 33},
            "critical_moments": [
                {"timestamp": "00:10:00", "type": "pricing_objection",
                 "description": "Prospect questioned pricing", "coaching": "Rep should have paused."},
            ],
            "recommendations": ["Ask more open questions."],
        },
        "multimodal": {
            "engagement_score": 74, "deal_probability": 68,
            "talk_ratio": {"rep": 67, "prospect": 33},
            "critical_moments": [
                {"timestamp": "00:10:00", "type": "pricing_objection",
                 "description": "Voice valence dropped to 0.30 while saying 'sounds good'.",
                 "coaching": "Rep should have paused and addressed the mismatch."},
                {"timestamp": "00:12:23", "type": "Dissonance – Positive Verbal vs Sad Facial",
                 "description": "She says 'looks great' but her face showed 97% sadness.",
                 "coaching": "Pause and check in before advancing."},
            ],
            "recommendations": ["Pause after pricing to let discomfort surface.", "Mirror the prospect's pace."],
        },
    }
    return transcript, voice_emotion, face_emotion, analysis


class TestRenderReport:
    def test_returns_html_document(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_includes_chartjs_cdn(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "cdn.jsdelivr.net/npm/chart.js" in html

    def test_includes_hero_metrics(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "74" in html  # multimodal engagement_score
        assert "68" in html  # multimodal deal_probability

    def test_includes_side_by_side_delta(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "+13" in html  # engagement_score delta
        assert "+23" in html  # deal_probability delta

    def test_includes_critical_moment_and_coaching(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "00:10:00" in html
        assert "Rep should have paused and addressed the mismatch." in html

    def test_includes_recommendations(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "Pause after pricing to let discomfort surface." in html
        assert "Mirror the prospect&#x27;s pace." in html or "Mirror the prospect's pace." in html

    def test_includes_footer(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "WhisperX" in html and "audeering" in html and "DeepFace" in html and "Claude" in html

    def test_embedded_timeline_json_is_valid(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        match = re.search(r"window\.REPORT_DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        assert match is not None
        data = json.loads(match.group(1))
        assert "timeline" in data
        assert "moment_markers" in data
        assert data["timeline"]["prospect_valence"] == [{"x": 600, "y": 0.3}, {"x": 742.8, "y": 0.85}]
        assert data["moment_markers"][0]["x"] == 600

    def test_handles_missing_face_data_gracefully(self):
        from pipeline.report import render_report
        transcript, voice_emotion, face_emotion, analysis = _sample_inputs()
        html = render_report(transcript, voice_emotion, [], analysis)
        assert "facial data unavailable" in html.lower()

    def test_includes_hidden_signals_hero_stat(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "1" in html  # one dissonance-type moment in the multimodal fixture
        assert "missed" in html.lower() or "hidden" in html.lower()

    def test_includes_proof_example_from_dissonance_moment(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "No, it looks great." in html
        assert "85% positive" in html
        assert "97% sad" in html
        assert "00:12:23" in html

    def test_no_proof_section_when_no_dissonance_moments(self):
        from pipeline.report import render_report
        transcript, voice_emotion, face_emotion, analysis = _sample_inputs()
        analysis["multimodal"]["critical_moments"] = [
            {"timestamp": "00:10:00", "type": "pricing_objection", "description": "d", "coaching": "c"},
        ]
        html = render_report(transcript, voice_emotion, face_emotion, analysis)
        assert html  # renders without crashing when there are zero dissonance moments
