import os
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

AUDIO_PATH = "output/audio_temp.wav"
SAMPLE_RATE = 16000
MAX_CHUNK_DURATION = 15


def _make_segment(overrides=None):
    seg = {
        "speaker": "SPEAKER_00",
        "start": 0.0,
        "end": 2.0,
        "text": "Hello world",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 0.4, "end": 0.8},
        ],
    }
    if overrides:
        seg.update(overrides)
    return seg


class TestExtractVoiceEmotion:
    def test_raises_when_audio_missing(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        transcript = [_make_segment()]
        with pytest.raises(FileNotFoundError, match="not found"):
            extract_voice_emotion(transcript, "nonexistent.wav")

    def test_raises_when_segments_empty(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        with pytest.raises(ValueError, match="empty"):
            extract_voice_emotion([], str(audio))

    def test_returns_valence_arousal_dominance(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment()]
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            # Model returns raw regression logits [arousal, dominance, valence]
            # directly (no .logits attribute — custom class forward returns tensor).
            mock_model.return_value = torch.tensor([[0.5, 0.3, 0.7]])
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
                result = extract_voice_emotion(transcript, str(audio))
        assert len(result) == 1
        # Model order is [arousal=0.5, dominance=0.3, valence=0.7]; output dict
        # reorders to {valence, arousal, dominance}. No sigmoid — raw clipped to [0,1].
        assert result[0]["valence"] == pytest.approx(0.7, abs=0.001)
        assert result[0]["arousal"] == pytest.approx(0.5, abs=0.001)
        assert result[0]["dominance"] == pytest.approx(0.3, abs=0.001)

    def test_preserves_speaker_start_end(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({"speaker": "SPEAKER_01", "start": 5.0, "end": 7.5})]
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            mock_model.return_value = torch.tensor([[0.5, 0.3, 0.7]])
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE * 10, dtype=np.float32), SAMPLE_RATE)
                result = extract_voice_emotion(transcript, str(audio))
        assert result[0]["speaker"] == "SPEAKER_01"
        assert result[0]["start"] == 5.0
        assert result[0]["end"] == 7.5

    def test_empty_segment_returns_zero_emotion(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({"start": 0.0, "end": 0.0})]
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
                result = extract_voice_emotion(transcript, str(audio))
        assert result[0]["valence"] == 0.0
        assert result[0]["arousal"] == 0.0
        assert result[0]["dominance"] == 0.0

    def test_chunks_long_segments_and_averages(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({"start": 0.0, "end": 20.0})]
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            # Raw regression tensors per chunk: [arousal, dominance, valence].
            mock_model.side_effect = [
                torch.tensor([[0.6, 0.2, 0.8]]),
                torch.tensor([[0.4, 0.4, 0.6]]),
            ]
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE * 25, dtype=np.float32), SAMPLE_RATE)
                result = extract_voice_emotion(transcript, str(audio))
        # 20s segment with 15s max chunk -> exactly two model calls.
        assert mock_model.call_count == 2
        # mean of [0.6,0.4]=0.5 (arousal), [0.2,0.4]=0.3 (dominance), [0.8,0.6]=0.7 (valence)
        assert result[0]["valence"] == pytest.approx(0.7, abs=0.001)
        assert result[0]["arousal"] == pytest.approx(0.5, abs=0.001)
        assert result[0]["dominance"] == pytest.approx(0.3, abs=0.001)

    def test_processes_multiple_segments(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [
            _make_segment({"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}),
            _make_segment({"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0}),
        ]
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            mock_model.return_value = torch.tensor([[0.5, 0.3, 0.7]])
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
                result = extract_voice_emotion(transcript, str(audio))
        assert len(result) == 2

    def test_does_not_mutate_input_segments(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment()]
        original_keys = set(transcript[0].keys())
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            mock_model.return_value = torch.tensor([[0.5, 0.3, 0.7]])
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
                extract_voice_emotion(transcript, str(audio))
        assert set(transcript[0].keys()) == original_keys


class TestExtractVoiceEmotionIntegration:
    @pytest.mark.skipif(
        not os.path.exists(
            "models/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim"
        ) and not os.path.exists(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim"
            )
        ),
        reason="audeering model not cached; run pipeline step 3 first"
    )
    def test_with_real_sine_tone(self, tmp_path):
        import soundfile as sf
        from pipeline.emotion_voice import extract_voice_emotion

        sr = SAMPLE_RATE
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 200 * t)
        audio = tmp_path / "tone.wav"
        sf.write(str(audio), tone, sr)
        transcript = [{
            "speaker": "S", "start": 0.0, "end": 2.0,
            "words": [
                {"start": 0.0, "end": 0.5},
                {"start": 1.0, "end": 2.0},
            ],
        }]
        cache = str(tmp_path / "models")
        result = extract_voice_emotion(transcript, str(audio), model_path=cache)
        assert len(result) == 1
        assert 0 <= result[0]["valence"] <= 1
        assert 0 <= result[0]["arousal"] <= 1
        assert 0 <= result[0]["dominance"] <= 1
        assert result[0]["valence"] != 0.0
        assert isinstance(result[0]["valence"], float)


class TestModelLoading:
    def test_load_model_creates_cache_dir(self, tmp_path):
        from pipeline.emotion_voice import _load_model
        cache = str(tmp_path / "models")
        with patch("pipeline.emotion_voice.Wav2Vec2FeatureExtractor") as mock_fe:
            with patch("pipeline.emotion_voice.Wav2Vec2ForSpeechClassification") as mock_model:
                mock_fe.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()
                fe, model = _load_model(cache_dir=cache)
                mock_fe.from_pretrained.assert_called_once_with(
                    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", cache_dir=cache
                )
                mock_model.from_pretrained.assert_called_once_with(
                    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", cache_dir=cache
                )
                assert model.eval.called
