import sys
import pytest
from unittest.mock import MagicMock

# pyannote.audio imports torch which crashes in CPU-only env (no CUDA libs).
# Inject a mock before pipeline.diarize is imported so the deferred
# `from pyannote.audio import Pipeline` inside diarize_audio() resolves safely.
_mock_pyannote = MagicMock()
sys.modules.setdefault("pyannote", _mock_pyannote)
sys.modules.setdefault("pyannote.audio", _mock_pyannote)


class TestDiarizeAudio:
    def setup_method(self):
        _mock_pyannote.reset_mock(return_value=True)

    def test_raises_when_audio_file_missing(self):
        from pipeline.diarize import diarize_audio
        with pytest.raises(FileNotFoundError, match="not found"):
            diarize_audio("nonexistent.wav", hf_token="tok")

    def test_raises_when_hf_token_empty(self, tmp_path):
        from pipeline.diarize import diarize_audio
        audio = tmp_path / "test.wav"
        audio.touch()
        with pytest.raises(ValueError, match=".env"):
            diarize_audio(str(audio), hf_token="")

    def test_loads_pyannote_model_with_hf_token(self, tmp_path):
        from pipeline.diarize import diarize_audio, DIARIZATION_MODEL
        audio = tmp_path / "test.wav"
        audio.touch()
        _mock_pyannote.Pipeline.from_pretrained.return_value.return_value.speaker_diarization.itertracks.return_value = []
        diarize_audio(str(audio), hf_token="my-token")
        _mock_pyannote.Pipeline.from_pretrained.assert_called_once_with(
            DIARIZATION_MODEL, token="my-token"
        )

    def test_returns_speaker_segments(self, tmp_path):
        from pipeline.diarize import diarize_audio
        audio = tmp_path / "test.wav"
        audio.touch()
        turn1, turn2 = MagicMock(), MagicMock()
        turn1.start, turn1.end = 0.0, 5.123
        turn2.start, turn2.end = 5.5, 10.0
        mock_diarization = MagicMock()
        mock_diarization.speaker_diarization.itertracks.return_value = [
            (turn1, None, "SPEAKER_00"),
            (turn2, None, "SPEAKER_01"),
        ]
        _mock_pyannote.Pipeline.from_pretrained.return_value.return_value = mock_diarization
        result = diarize_audio(str(audio), hf_token="tok")
        assert result == [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.123},
            {"speaker": "SPEAKER_01", "start": 5.5, "end": 10.0},
        ]
