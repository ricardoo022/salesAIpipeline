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
