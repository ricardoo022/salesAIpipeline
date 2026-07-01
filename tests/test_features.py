import numpy as np
import pytest
from unittest.mock import patch, MagicMock

AUDIO_PATH = "output/audio_temp.wav"
SAMPLE_RATE = 16000


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


class TestExtractAudioFeatures:
    def test_raises_when_audio_missing(self, tmp_path):
        from pipeline.features import extract_audio_features
        transcript = [_make_segment()]
        with pytest.raises(FileNotFoundError, match="not found"):
            extract_audio_features(transcript, "nonexistent.wav")

    def test_raises_when_transcript_empty(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        with pytest.raises(ValueError, match="empty"):
            extract_audio_features([], str(audio))

    def test_raises_when_transcript_none(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        with pytest.raises(ValueError, match="empty"):
            extract_audio_features(None, str(audio))

    def test_computes_pitch_mean_std(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment()]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (
                np.array([182.0, 184.0, np.nan, 180.0, 186.0, np.nan]),
                None,
                None,
            )
            mock_l.feature.rms.return_value = np.array([[0.04, 0.05]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06, 0.07]])

            result = extract_audio_features(transcript, str(audio))

        assert len(result) == 1
        assert result[0]["pitch_mean"] == pytest.approx(183.0, abs=0.01)
        assert result[0]["pitch_std"] == pytest.approx(2.236, abs=0.01)

    def test_pitch_zero_when_all_unvoiced(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment()]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (
                np.array([np.nan, np.nan, np.nan]),
                None,
                None,
            )
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            result = extract_audio_features(transcript, str(audio))

        assert result[0]["pitch_mean"] == 0.0
        assert result[0]["pitch_std"] == 0.0

    def test_computes_energy_mean(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment()]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04, 0.06, 0.08]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            result = extract_audio_features(transcript, str(audio))

        assert result[0]["energy_mean"] == pytest.approx(0.06, abs=0.01)

    def test_computes_speech_rate_from_word_timestamps(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({"start": 0.0, "end": 2.0, "words": [{"start": 0, "end": 0.2}] * 5})]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            result = extract_audio_features(transcript, str(audio))

        assert result[0]["speech_rate"] == pytest.approx(2.5, abs=0.01)

    def test_computes_pause_ratio(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({
            "start": 0.0, "end": 2.0,
            "words": [
                {"word": "A", "start": 0.0, "end": 0.2},
                {"word": "B", "start": 0.6, "end": 0.8},
                {"word": "C", "start": 0.8, "end": 1.0},
            ],
        })]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            result = extract_audio_features(transcript, str(audio))

        # gap A→B = 0.4, B→C = 0.0, total gap = 0.4, duration = 2.0
        assert result[0]["pause_ratio"] == pytest.approx(0.2, abs=0.01)

    def test_pause_ratio_zero_for_single_word(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({
            "start": 0.0, "end": 2.0,
            "words": [{"word": "Hi", "start": 0.0, "end": 0.5}],
        })]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            result = extract_audio_features(transcript, str(audio))

        assert result[0]["pause_ratio"] == 0.0

    def test_computes_zcr(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment()]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06, 0.08, 0.10]])

            result = extract_audio_features(transcript, str(audio))

        assert result[0]["zcr"] == pytest.approx(0.08, abs=0.01)

    def test_preserves_speaker_start_end_from_transcript(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({"speaker": "SPEAKER_01", "start": 5.0, "end": 7.5})]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 10, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            result = extract_audio_features(transcript, str(audio))

        assert result[0]["speaker"] == "SPEAKER_01"
        assert result[0]["start"] == 5.0
        assert result[0]["end"] == 7.5

    def test_segment_with_no_words_defaults_to_zero(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({"words": []})]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            result = extract_audio_features(transcript, str(audio))

        assert result[0]["speech_rate"] == 0.0
        assert result[0]["pause_ratio"] == 0.0

    def test_does_not_mutate_input_segments(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment()]
        original_keys = set(transcript[0].keys())
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            extract_audio_features(transcript, str(audio))

        assert set(transcript[0].keys()) == original_keys

    def test_processes_multiple_segments(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [
            _make_segment({"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}),
            _make_segment({"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0}),
        ]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            result = extract_audio_features(transcript, str(audio))

        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_overlapping_words_ignores_negative_gaps(self, tmp_path):
        from pipeline.features import extract_audio_features
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({
            "start": 0.0, "end": 2.0,
            "words": [
                {"start": 0.0, "end": 0.8},
                {"start": 0.6, "end": 1.4},
                {"start": 1.2, "end": 2.0},
            ],
        })]
        with patch("pipeline.features.librosa") as mock_l:
            mock_l.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
            mock_l.pyin.return_value = (np.array([180.0]), None, None)
            mock_l.feature.rms.return_value = np.array([[0.04]])
            mock_l.feature.zero_crossing_rate.return_value = np.array([[0.06]])

            result = extract_audio_features(transcript, str(audio))

        assert result[0]["pause_ratio"] == 0.0


class TestExtractAudioFeaturesIntegration:
    def test_with_real_sine_tone(self, tmp_path):
        import soundfile as sf
        from pipeline.features import extract_audio_features

        sr = SAMPLE_RATE
        t = np.linspace(0, 1, sr, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 200 * t)
        audio = tmp_path / "tone.wav"
        sf.write(str(audio), tone, sr)
        transcript = [{
            "speaker": "S", "start": 0.0, "end": 1.0,
            "words": [
                {"start": 0.0, "end": 0.3},
                {"start": 0.5, "end": 1.0},
            ],
        }]
        result = extract_audio_features(transcript, str(audio))
        assert len(result) == 1
        assert result[0]["pitch_mean"] == pytest.approx(200, abs=15)
        assert result[0]["energy_mean"] > 0
        assert result[0]["speech_rate"] == pytest.approx(2.0, abs=0.01)
        assert result[0]["pause_ratio"] == pytest.approx(0.2, abs=0.01)

    def test_with_silent_audio(self, tmp_path):
        import soundfile as sf
        from pipeline.features import extract_audio_features

        sr = SAMPLE_RATE
        silent = np.zeros(sr, dtype=np.float32)
        audio = tmp_path / "silent.wav"
        sf.write(str(audio), silent, sr)
        transcript = [{
            "speaker": "S", "start": 0.0, "end": 1.0,
            "words": [
                {"start": 0.0, "end": 0.3},
                {"start": 0.5, "end": 1.0},
            ],
        }]
        result = extract_audio_features(transcript, str(audio))
        assert len(result) == 1
        assert result[0]["pitch_mean"] == 0.0
        assert result[0]["pitch_std"] == 0.0
        assert result[0]["energy_mean"] == 0.0
        assert result[0]["zcr"] == 0.0
        assert result[0]["speech_rate"] == pytest.approx(2.0, abs=0.01)
        assert result[0]["pause_ratio"] == pytest.approx(0.2, abs=0.01)
