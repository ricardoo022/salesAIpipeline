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
