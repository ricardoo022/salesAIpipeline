import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from pipeline.audio import extract_audio, AUDIO_SAMPLE_RATE


class TestExtractAudio:
    def test_creates_output_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "out.wav"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            extract_audio(str(tmp_path / "in.mp4"), str(nested))
        assert os.path.exists(str(nested.parent))

    def test_passes_correct_ffmpeg_args(self, tmp_path):
        output = tmp_path / "out.wav"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            extract_audio(str(tmp_path / "in.mp4"), str(output))

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert cmd[cmd.index("-ar") + 1] == str(AUDIO_SAMPLE_RATE)
        assert cmd[cmd.index("-ac") + 1] == "1"
        assert cmd[cmd.index("-sample_fmt") + 1] == "s16"

    def test_returns_output_path(self, tmp_path):
        output = tmp_path / "out.wav"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = extract_audio(str(tmp_path / "in.mp4"), str(output))
        assert result == str(output)

    def test_raises_on_ffmpeg_failure(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            with pytest.raises(RuntimeError, match="ffmpeg failed"):
                extract_audio(str(tmp_path / "in.mp4"), str(tmp_path / "out.wav"))

    def test_raises_on_missing_ffmpeg(self, tmp_path):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                extract_audio(str(tmp_path / "in.mp4"), str(tmp_path / "out.wav"))

    def test_uses_default_output_path(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = extract_audio("input.mp4")
        from pipeline.audio import AUDIO_TEMP_FILE
        assert result == AUDIO_TEMP_FILE
