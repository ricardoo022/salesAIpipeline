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
