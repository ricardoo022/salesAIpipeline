# Step 3 — Voice Emotion Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract valence, arousal, and dominance per transcript segment using audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, with 15s chunking for long segments.

**Architecture:** Follow established project pattern: a shared module (`pipeline/emotion_voice.py`) contains `extract_voice_emotion()`, a CLI script (`pipeline/03_emotion_voice.py`) reads JSON from step 2 and writes `output/voice_emotion.json`, and tests follow the existing mock/unit/integration pattern with TDD discipline.

**Tech Stack:** transformers 4.x (Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor), torch 2.x, numpy, librosa (for audio loading)

**Skills applied:**
- **test-driven-development** — all production code has a failing test first
- **writing-plans** — plan structure, task decomposition, completeness check
- **ml-pipeline-workflow** — modular stage design, idempotency, observability
- **recsys-pipeline-architect** — does not apply (this is single-model inference, not a recommendation pipeline)

---

## Task Structure

### Task 1: Create the shared module — `pipeline/emotion_voice.py`

**Files:**
- Create: `pipeline/emotion_voice.py`
- Test: `tests/test_emotion_voice.py`

- [ ] **Step 1: Write the failing test for module imports and file-not-found guard**

Write `tests/test_emotion_voice.py` starting with:

```python
import numpy as np
import pytest
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_emotion_voice.py::TestExtractVoiceEmotion -v`
Expected: FAIL with `ModuleNotFoundError` or `FunctionNotDefined`

- [ ] **Step 3: Write minimal module scaffolding**

Create `pipeline/emotion_voice.py`:

```python
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
    and returns valence, arousal, dominance averaged over ≤15s chunks.

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

        # Split long segments into ≤MAX_CHUNK_DURATION-second chunks
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_emotion_voice.py::TestExtractVoiceEmotion -v`
Expected: 2 PASS

- [ ] **Step 5: Write test for model loading**

```python
class TestModelLoading:
    def test_load_model_creates_cache_dir(self, tmp_path):
        from pipeline.emotion_voice import _load_model
        cache = str(tmp_path / "models")
        with patch("pipeline.emotion_voice.Wav2Vec2FeatureExtractor") as mock_fe:
            with patch("pipeline.emotion_voice.Wav2Vec2ForSequenceClassification") as mock_model:
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
```

- [ ] **Step 6: Run test to verify it fails first**

Run: `python -m pytest tests/test_emotion_voice.py::TestModelLoading -v`
Expected: FAIL with `AttributeError` or similar (class not found yet — `_load_model` already exists but may need adjustment for the test mock expectations)

If the test passes immediately, verify it's testing the real behavior and not accidentally passing.

- [ ] **Step 7: No code changes needed** — `_load_model` already exists in step 3, verify it passes

Run: `python -m pytest tests/test_emotion_voice.py -v`
Expected: all 3 PASS

- [ ] **Step 8: Write the core behavior tests (mocked)**

```python
class TestExtractVoiceEmotion:
    # ... (previous tests from step 1)

    def test_returns_valence_arousal_dominance(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment()]
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            mock_logits = MagicMock()
            mock_logits.logits = torch.tensor([[0.5, 0.3, 0.7]])
            mock_model.return_value = MagicMock()
            mock_model.return_value.__enter__.return_value = mock_model
            mock_model.__call__.return_value = mock_logits
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
                result = extract_voice_emotion(transcript, str(audio))
        assert len(result) == 1
        assert "valence" in result[0]
        assert "arousal" in result[0]
        assert "dominance" in result[0]
        # sigmoid(0.5)=0.6225, sigmoid(0.3)=0.5744, sigmoid(0.7)=0.6682
        assert result[0]["valence"] == pytest.approx(0.6225, abs=0.001)
        assert result[0]["arousal"] == pytest.approx(0.5744, abs=0.001)
        assert result[0]["dominance"] == pytest.approx(0.6682, abs=0.001)
```

Actually, let me fix the mock setup — `model()` returns the output directly (no `__enter__` needed):

```python
class TestExtractVoiceEmotion:
    # ... (previous tests from step 1)

    def test_returns_valence_arousal_dominance(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment()]
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            mock_logits = MagicMock()
            mock_logits.logits = torch.tensor([[0.5, 0.3, 0.7]])
            mock_model.return_value = mock_logits
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
                result = extract_voice_emotion(transcript, str(audio))
        assert len(result) == 1
        assert result[0]["valence"] == pytest.approx(0.6225, abs=0.001)
        assert result[0]["arousal"] == pytest.approx(0.5744, abs=0.001)
        assert result[0]["dominance"] == pytest.approx(0.6682, abs=0.001)

    def test_preserves_speaker_start_end(self, tmp_path):
        from pipeline.emotion_voice import extract_voice_emotion
        audio = tmp_path / "test.wav"
        audio.touch()
        transcript = [_make_segment({"speaker": "SPEAKER_01", "start": 5.0, "end": 7.5})]
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            mock_logits = MagicMock()
            mock_logits.logits = torch.tensor([[0.5, 0.3, 0.7]])
            mock_model.return_value = mock_logits
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
        # 20-second segment → must chunk into 15s + 5s
        transcript = [_make_segment({"start": 0.0, "end": 20.0})]
        with patch("pipeline.emotion_voice._load_model") as mock_load:
            mock_fe, mock_model = MagicMock(), MagicMock()
            mock_load.return_value = (mock_fe, mock_model)
            mock_logits = MagicMock()
            mock_logits.logits = torch.tensor([[0.5, 0.3, 0.7]])
            mock_model.return_value = mock_logits
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE * 25, dtype=np.float32), SAMPLE_RATE)
                result = extract_voice_emotion(transcript, str(audio))
        # Two chunks, same prediction, average = same
        assert result[0]["valence"] == pytest.approx(0.6225, abs=0.001)
        assert result[0]["arousal"] == pytest.approx(0.5744, abs=0.001)

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
            mock_logits = MagicMock()
            mock_logits.logits = torch.tensor([[0.5, 0.3, 0.7]])
            mock_model.return_value = mock_logits
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
            mock_logits = MagicMock()
            mock_logits.logits = torch.tensor([[0.5, 0.3, 0.7]])
            mock_model.return_value = mock_logits
            with patch("pipeline.emotion_voice.librosa") as mock_lib:
                mock_lib.load.return_value = (np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE)
                extract_voice_emotion(transcript, str(audio))
        assert set(transcript[0].keys()) == original_keys
```

- [ ] **Step 9: Run tests to verify they fail first**

Run: `python -m pytest tests/test_emotion_voice.py -v`
Expected: all new tests FAIL with function not returning expected shape / wrong attribute names

**IMPORTANT:** Watch each test fail. Confirm the failure reason is "feature missing" not "typo in test".

- [ ] **Step 10: Run tests now that implementation exists** — verify all pass

Run: `python -m pytest tests/test_emotion_voice.py -v`
Expected: all tests PASS

- [ ] **Step 11: Write integration test with real sine tone**

Add to `tests/test_emotion_voice.py`:

```python
class TestExtractVoiceEmotionIntegration:
    def test_with_real_sine_tone(self, tmp_path):
        import soundfile as sf
        from pipeline.emotion_voice import extract_voice_emotion, _load_model

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
        # This integration test downloads the real model (~1.3GB).
        # Skip if running in CI without model cache.
        result = extract_voice_emotion(transcript, str(audio), model_path=cache)
        assert len(result) == 1
        assert 0 <= result[0]["valence"] <= 1
        assert 0 <= result[0]["arousal"] <= 1
        assert 0 <= result[0]["dominance"] <= 1
        assert isinstance(result[0]["valence"], float)
```

- [ ] **Step 12: Run integration test** (note: downloads real model — may be slow)

Run: `python -m pytest tests/test_emotion_voice.py::TestExtractVoiceEmotionIntegration -v -s`
Expected: PASS (or skip if model download fails — mark with `@pytest.mark.skipif` if needed)

- [ ] **Step 13: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v`
Expected: all existing 39 tests + new tests all PASS

- [ ] **Step 14: Commit**

```bash
git add pipeline/emotion_voice.py tests/test_emotion_voice.py
git commit -m "feat(step3): add voice emotion extraction module with audeering wav2vec2"
```

---

### Task 2: Create CLI script — `pipeline/03_emotion_voice.py`

**Files:**
- Create: `pipeline/03_emotion_voice.py`
- Test: `tests/test_03_emotion_voice.py`

- [ ] **Step 1: Write failing test for CLI guards**

Create `tests/test_03_emotion_voice.py`:

```python
import os
import sys
import subprocess

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def test_exits_when_segments_input_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "pipeline/03_emotion_voice.py")],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "audio_features.json" in result.stdout


def test_exits_when_audio_missing(tmp_path):
    (tmp_path / "output").mkdir()
    with open(tmp_path / "output" / "audio_features.json", "w") as f:
        f.write("[]")
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "pipeline/03_emotion_voice.py")],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "audio_temp.wav" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_03_emotion_voice.py -v`
Expected: FAIL with `FileNotFoundError` or `returncode 1` not matching

- [ ] **Step 3: Write the CLI script**

Create `pipeline/03_emotion_voice.py`:

```python
#!/usr/bin/env python3
"""Step 3: Extract voice emotion from transcript segments.

Uses audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim to extract
valence, arousal, and dominance per segment. Long segments (>15s) are
split into chunks and results averaged.

Output: output/voice_emotion.json
"""

import json
import os
import sys

from emotion_voice import extract_voice_emotion

SEGMENTS_FILE = "output/audio_features.json"
AUDIO_FILE = "output/audio_temp.wav"
OUTPUT_FILE = "output/voice_emotion.json"


def main():
    if not os.path.exists(SEGMENTS_FILE):
        print(f"ERROR: {SEGMENTS_FILE} not found. Run step 2 first.")
        sys.exit(1)

    if not os.path.exists(AUDIO_FILE):
        print(f"ERROR: {AUDIO_FILE} not found. Run step 1 first.")
        sys.exit(1)

    with open(SEGMENTS_FILE) as f:
        segments = json.load(f)

    print("→ Extracting voice emotion (valence, arousal, dominance) from audio segments...")
    emotions = extract_voice_emotion(segments, AUDIO_FILE)

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(emotions, f, indent=2)

    print(f"✓ Voice emotion saved to {OUTPUT_FILE} ({len(emotions)} segments)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_03_emotion_voice.py -v`
Expected: 2 PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/03_emotion_voice.py tests/test_03_emotion_voice.py
git commit -m "feat(step3): add CLI entry point for voice emotion extraction"
```

---

### Task 3: Update `__init__.py`

**Files:**
- Modify: `pipeline/__init__.py`

- [ ] **Step 1: Write failing test for import**

Add to any existing test file (e.g., `tests/test_emotion_voice.py`):

```python
def test_module_exports_extract_function():
    from pipeline import emotion_voice
    assert hasattr(emotion_voice, "extract_voice_emotion")
```

Run: `python -m pytest tests/test_emotion_voice.py::test_module_exports_extract_function -v`
Expected: PASS (the function is in the module, no change needed — this test just asserts the public API contract)

Actually, looking at the existing `__init__.py`, it only exports `extract_audio` and `AUDIO_SAMPLE_RATE`. The other modules (features.py, transcribe.py, diarize.py) are not exported from `__init__.py`. The pattern is that tests import directly from `pipeline.features`, `pipeline.transcribe`, etc. So no `__init__.py` change needed.

**Skip this task** — existing pattern does not require adding to `__init__.py`.

---

## Self-Review

### 1. Spec coverage

| Spec requirement | Task(s) |
|---|---|
| Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | Task 1, Step 3 — `MODEL_NAME` constant |
| Outputs: valence, arousal, dominance (continuous 0–1 per segment) | Task 1, Step 3 — sigmoid on logits, 4-decimal rounding |
| Segments longer than 15s split into ≤15s chunks; chunk results averaged | Task 1, Step 3 — `MAX_CHUNK_DURATION` + chunk loop |
| Model cached in `models/` directory | Task 1, Step 3 — `MODEL_CACHE_DIR`, `cache_dir` param |
| Output schema: `{speaker, start, end, valence, arousal, dominance}` | Task 1, Step 3 — dict construction |
| CLI reads segments JSON, writes voice_emotion.json | Task 2, Step 3 — `main()` |
| CLI guards: missing input files → exit with message | Task 2, Step 3 — `os.path.exists` checks |
| Empty segment → safe default (0.0 values) | Task 1, Step 8 — `len(segment_audio) == 0` branch |
| No mutation of input segments | Task 1, Step 8 — mutation test |
| Multiple speakers preserved | Task 1, Step 8 — multiple segments test |
| Import duality (bare vs package imports) | Task 1, Step 3 — try/except pattern |

### 2. Placeholder scan

No TBD, TODO, "implement later", or other placeholder patterns found. Every step includes complete code, exact file paths, and exact commands.

### 3. Type consistency

- `_make_segment()` returns `dict` with `speaker: str, start: float, end: float, text: str, words: list[dict]` — consistent across all test cases
- `extract_voice_emotion()` signature: `(segments: list[dict], audio_path: str, model_path: str = None) -> list[dict]` — consistent across module and CLI
- `_predict_chunk()` returns `np.ndarray` of shape `(3,)` with `[valence, arousal, dominance]`
- Index 0 = valence, index 1 = arousal, index 2 = dominance — consistent in `_predict_chunk` return, `extract_voice_emotion` dict construction, and all assertions
- Mock return values: `torch.tensor([[0.5, 0.3, 0.7]])` → expected sigmoid values `[0.6225, 0.5744, 0.6682]` — verified

### Gaps found and fixed

- None. Plan covers all spec requirements for Step 3.
