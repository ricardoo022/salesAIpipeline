"""Voice emotion extraction for step 3.

Uses audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim to extract
valence, arousal, and dominance from audio segments per the sales coach
pipeline specification.

The audeering model ships a `Wav2Vec2ForSpeechClassification` head (mean-pooled
regression over wav2vec2 hidden states). That class was removed from modern
`transformers` releases, so we reconstruct it here from the head's saved weights
(`classifier.dense` + `classifier.out_proj`). Loading the model with the stock
`Wav2Vec2ForSequenceClassification` silently random-inits the head — the bug that
originally produced the flat-line VAD output.
"""

import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2FeatureExtractor,
)

try:
    from pipeline.audio import AUDIO_SAMPLE_RATE
except ImportError:
    from audio import AUDIO_SAMPLE_RATE

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
MODEL_CACHE_DIR = "models"
MAX_CHUNK_DURATION = 15
# The audeering model's id2label ordering: 0=arousal, 1=dominance, 2=valence.
AROUSAL_IDX = 0
DOMINANCE_IDX = 1
VALENCE_IDX = 2


class Wav2Vec2ClassificationHead(nn.Module):
    """Head saved in the checkpoint as `classifier.dense` + `classifier.out_proj`."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Reconstructed audeering head: mean-pool hidden states → MLP -> regression."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def forward(self, input_values, attention_mask=None):
        # attention_mask from the feature extractor is per-input-sample (pre-encoder);
        # resampling it for the downsampled encoder time dim is fiddly, and we run
        # one chunk at a time (no padding), so we simply mean-pool over the full
        # encoder output — matching the model's configured `pooling_mode: "mean"`.
        outputs = self.wav2vec2(input_values, attention_mask=None)
        hidden_states = outputs[0]
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)


def _load_model(cache_dir: str = MODEL_CACHE_DIR):
    """Load the emotion model and feature extractor, cached in models/."""
    os.makedirs(cache_dir, exist_ok=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir
    )
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir
    )
    model.eval()
    return feature_extractor, model


def _predict_chunk(audio_chunk: np.ndarray, feature_extractor, model) -> np.ndarray:
    """Run the emotion model on a single audio chunk.

    Returns [arousal, dominance, valence] (raw regression outputs clipped to [0, 1]).
    No sigmoid: this is a regression head, not a classifier.
    """
    inputs = feature_extractor(
        audio_chunk, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs)
    probs = logits.numpy()[0]
    probs = np.clip(probs, 0.0, 1.0)
    return probs


def extract_voice_emotion(
    segments: list[dict],
    audio_path: str,
    model_path: str = None,
) -> list[dict]:
    """Extract voice emotion for each transcript segment.

    For each segment, loads audio, runs the audeering wav2vec2 model, and
    returns valence, arousal, dominance (continuous 0-1) averaged over
    <=15s chunks for long segments.

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

        # probs ordering from the model: [arousal, dominance, valence]
        result.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "valence": round(float(probs[VALENCE_IDX]), 4),
            "arousal": round(float(probs[AROUSAL_IDX]), 4),
            "dominance": round(float(probs[DOMINANCE_IDX]), 4),
        })

    return result