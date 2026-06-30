import json
import os
import sys
import subprocess
import pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def test_exits_when_input_video_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "pipeline/01_transcribe.py")],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "not found" in result.stdout


def test_exits_when_hf_token_missing(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "meeting.mp4").touch()
    env = {**os.environ, "HF_TOKEN": ""}
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "pipeline/01_transcribe.py")],
        capture_output=True, text=True,
        cwd=str(tmp_path),
        env=env,
    )
    assert result.returncode == 1
    assert "HF_TOKEN" in result.stdout


class TestTranscribeAudio:
    def test_raises_when_audio_file_missing(self):
        from pipeline.transcribe import transcribe_audio
        with pytest.raises(FileNotFoundError, match="not found"):
            transcribe_audio("nonexistent.wav")

    def test_loads_large_v2_on_cpu_with_int8(self, tmp_path):
        from pipeline.transcribe import transcribe_audio
        audio = tmp_path / "test.wav"
        audio.touch()
        with patch("pipeline.transcribe.whisperx") as mock_wx:
            mock_wx.load_model.return_value.transcribe.return_value = {"segments": [], "language": "en"}
            mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
            mock_wx.align.return_value = {"segments": []}
            transcribe_audio(str(audio))
        mock_wx.load_model.assert_called_once_with("large-v2", device="cpu", compute_type="int8")

    def test_aligns_using_detected_language(self, tmp_path):
        from pipeline.transcribe import transcribe_audio
        audio = tmp_path / "test.wav"
        audio.touch()
        raw_segments = [{"text": "Olá"}]
        with patch("pipeline.transcribe.whisperx") as mock_wx:
            mock_wx.load_model.return_value.transcribe.return_value = {
                "segments": raw_segments, "language": "pt"
            }
            align_model, metadata = MagicMock(), MagicMock()
            mock_wx.load_align_model.return_value = (align_model, metadata)
            mock_wx.align.return_value = {"segments": []}
            transcribe_audio(str(audio))
        mock_wx.load_align_model.assert_called_once_with(language_code="pt", device="cpu")
        mock_wx.align.assert_called_once_with(
            raw_segments, align_model, metadata, str(audio), device="cpu"
        )

    def test_returns_aligned_segments(self, tmp_path):
        from pipeline.transcribe import transcribe_audio
        audio = tmp_path / "test.wav"
        audio.touch()
        expected = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "words": [{"word": "Hello", "start": 0.0, "end": 1.0}]}
        ]
        with patch("pipeline.transcribe.whisperx") as mock_wx:
            mock_wx.load_model.return_value.transcribe.return_value = {"segments": [], "language": "en"}
            mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
            mock_wx.align.return_value = {"segments": expected}
            result = transcribe_audio(str(audio))
        assert result == expected


class TestMergeSpeakerLabels:
    def test_assigns_speaker_by_maximum_overlap(self):
        from pipeline.transcribe import merge_speaker_labels
        segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        diarization = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5},
            {"speaker": "SPEAKER_01", "start": 1.5, "end": 3.0},
        ]
        result = merge_speaker_labels(segments, diarization)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_assigns_unknown_when_no_diarization_overlap(self):
        from pipeline.transcribe import merge_speaker_labels
        segments = [{"start": 5.0, "end": 6.0, "text": "Hello"}]
        diarization = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0}]
        result = merge_speaker_labels(segments, diarization)
        assert result[0]["speaker"] == "UNKNOWN"

    def test_handles_empty_diarization(self):
        from pipeline.transcribe import merge_speaker_labels
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "World"},
        ]
        result = merge_speaker_labels(segments, [])
        assert all(s["speaker"] == "UNKNOWN" for s in result)

    def test_does_not_mutate_input_segments(self):
        from pipeline.transcribe import merge_speaker_labels
        segments = [{"start": 0.0, "end": 1.0, "text": "Hi"}]
        diarization = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]
        merge_speaker_labels(segments, diarization)
        assert "speaker" not in segments[0]
