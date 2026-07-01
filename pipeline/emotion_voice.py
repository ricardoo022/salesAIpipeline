"""Voice emotion extraction for step 3.

Uses audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim to extract
valence, arousal, and dominance from audio segments per the sales coach
pipeline specification.
"""

import os
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

try:
    from pipeline.audio import AUDIO_SAMPLE_RATE
except ImportError:
    from audio import AUDIO_SAMPLE_RATE

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
MODEL_CACHE_DIR = "models"
MAX_CHUNK_DURATION = 15


def _load_model(cache_dir: str = MODEL_CACHE_DIR):
    """Load the emotion model and feature extractor, cached in models/."""
    os.makedirs(cache_dir, exist_ok=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir
    )
    model.eval()
    return feature_extractor, model


def _predict_chunk(audio_chunk: np.ndarray, feature_extractor, model) -> np.ndarray:
    """Run emotion model on a single audio chunk, returning [valence, arousal, dominance]."""
    inputs = feature_extractor(
        audio_chunk, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).numpy()[0]
    return probs


def extract_voice_emotion(
    segments: list[dict],
    audio_path: str,
    model_path: str = None,
) -> list[dict]:
    """Extract voice emotion for each transcript segment.

    For each segment, loads audio, runs the audeering wav2vec2 model,
    and returns valence, arousal, dominance averaged over < =15s chunks.

    Returns a new list preserving speaker/start/end with added emotion fields.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not segments:
        raise ValueError("Segments list is empty")

    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
    cache_dir = model_path or MODEL_CACHE_DIR
    feature_extractor, model = _load_model(cache_dir)
    max_chunk_samples = MAX_CHUNK_DURATION * AUDIO_SAMPLE_RATE

    result = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        segment_audio = y[start_sample:end_sample]

        if len(segment_audio) == 0:
            result.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
            })
            continue

        # Split long segments into <=MAX_CHUNK_DURATION-second chunks
        if len(segment_audio) > max_chunk_samples:
            chunk_probs = []
            for chunk_start in range(0, len(segment_audio), max_chunk_samples):
                chunk_end = min(chunk_start + max_chunk_samples, len(segment_audio))
                chunk = segment_audio[chunk_start:chunk_end]
                chunk_probs.append(_predict_chunk(chunk, feature_extractor, model))
            probs = np.mean(chunk_probs, axis=0)
        else:
            probs = _predict_chunk(segment_audio, feature_extractor, model)

        result.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "valence": round(float(probs[0]), 4),
            "arousal": round(float(probs[1]), 4),
            "dominance": round(float(probs[2]), 4),
        })

    return result
