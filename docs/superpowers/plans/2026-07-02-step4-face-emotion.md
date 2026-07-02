# Step 4 — Facial Emotion Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample one frame every 10 seconds from `input/meeting.mp4`, run DeepFace emotion analysis on each frame, skip frames with no detected face, and write `output/face_emotion.json`.

**Architecture:** Follow the established project module/script split — a shared module (`pipeline/emotion_face.py`) holds `extract_face_emotion()` plus thin seams for the two heavy deps, and a CLI script (`pipeline/04_emotion_face.py`) reads the video and writes JSON. `cv2` and `deepface` are lazy-imported inside the functions that need them (same pattern as `pipeline/diarize.py` lazy-imports pyannote) so the module loads cleanly and the pure-logic unit tests run without those heavy/optional dependencies installed.

**Tech Stack:** opencv-python 4.x (frame sampling via `cv2.VideoCapture`), DeepFace ≥0.0.90 (emotion analysis), numpy (test fixtures), pytest.

**Skills applied:**
- **test-driven-development** — every function has a failing test first; RED → GREEN → commit per task
- **writing-plans** — plan structure, bite-sized tasks, completeness/self-review
- **ml-pipeline-workflow** — does not apply (this is a single inference step in a flat-script demo pipeline; no training/validation/registry/serving)
- **recsys-pipeline-architect** — does not apply (facial emotion detection is not a "top-K items for a user/context" ranking pipeline; no sources/hydrators/filters/scorers/selectors)

---

## Prerequisites

`cv2` and `deepface` are listed in `requirements.txt` but are **not** installed in the venv (verified). The module's lazy imports mean the test suite is green without them, but to actually run the cv2 integration tests and the real pipeline you need at least opencv-python:

```bash
source venv/bin/activate
pip install opencv-python          # required for _iter_frames integration tests + the real pipeline
pip install deepface               # optional: only needed for the real-video integration test + step 4 of run.py
```

No changes to `requirements.txt` (deepface + opencv-python already listed) or `run.py` (step 4 already wired at `run.py:25-30`).

---

## File Structure

| File | Responsibility |
|---|---|
| `pipeline/emotion_face.py` | Shared module: `_shape_emotion_result`, `_analyze_frame`, `_iter_frames`, `extract_face_emotion`. Lazy-imports cv2 + deepface. |
| `pipeline/04_emotion_face.py` | CLI entry point: guard on missing video, call module, write `output/face_emotion.json`. |
| `tests/test_emotion_face.py` | Unit tests (mocked seams) + integration tests (skip-guarded on cv2/deepface). |
| `tests/test_04_emotion_face.py` | Subprocess guard test for the CLI (missing video → exit 1). |

`pipeline/__init__.py` and `tests/__init__.py` already exist; no package changes needed (the existing pattern imports directly from `pipeline.emotion_face`).

---

## Task 1: Module scaffold + `_shape_emotion_result` (pure helper)

**Files:**
- Create: `pipeline/emotion_face.py`
- Create: `tests/test_emotion_face.py`

- [ ] **Step 1: Write the failing tests + test-file header**

Create `tests/test_emotion_face.py`:

```python
import importlib.util
import os
import pytest
from unittest.mock import patch, MagicMock

SAMPLE_INTERVAL = 10


def _cv2_available():
    return importlib.util.find_spec("cv2") is not None


def _deepface_available():
    return importlib.util.find_spec("deepface") is not None


class TestShapeEmotionResult:
    def test_extracts_dominant_and_scores(self):
        from pipeline.emotion_face import _shape_emotion_result
        raw = {"dominant_emotion": "happy", "emotion": {"happy": 0.9123, "sad": 0.0877}}
        result = _shape_emotion_result(raw)
        assert result["dominant_emotion"] == "happy"
        assert result["scores"]["happy"] == 0.9123
        assert result["scores"]["sad"] == 0.0877

    def test_rounds_scores_to_four_decimals(self):
        from pipeline.emotion_face import _shape_emotion_result
        raw = {"dominant_emotion": "neutral", "emotion": {"neutral": 0.712345, "happy": 0.031111}}
        result = _shape_emotion_result(raw)
        assert result["scores"]["neutral"] == 0.7123
        assert result["scores"]["happy"] == 0.0311

    def test_unwraps_list_result(self):
        from pipeline.emotion_face import _shape_emotion_result
        raw = [{"dominant_emotion": "sad", "emotion": {"sad": 0.6, "neutral": 0.4}}]
        result = _shape_emotion_result(raw)
        assert result["dominant_emotion"] == "sad"
        assert result["scores"]["sad"] == 0.6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_emotion_face.py::TestShapeEmotionResult -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'pipeline.emotion_face'` (module does not exist yet).

- [ ] **Step 3: Write the module scaffold + `_shape_emotion_result`**

Create `pipeline/emotion_face.py`:

```python
"""Facial emotion extraction for step 4.

Samples one frame every SAMPLE_INTERVAL seconds from the meeting video and
runs DeepFace emotion analysis on each frame. Frames with no detected face
are skipped (no crash) per the sales coach pipeline specification.

cv2 and deepface are imported lazily inside the functions that need them, so
the module loads cleanly in environments where those heavy/optional deps are
not installed (mirrors the pyannote lazy-import pattern in pipeline/diarize.py).
"""

import os

SAMPLE_INTERVAL = 10


def _shape_emotion_result(raw):
    """Normalize a DeepFace emotion analysis result into our output shape.

    DeepFace.analyze returns either a dict or a list of dicts (one per face);
    we take the first face. Returns {dominant_emotion, scores} with scores
    rounded to 4 decimals for consistency with the rest of the pipeline.
    """
    if isinstance(raw, list):
        raw = raw[0]
    emotions = raw["emotion"]
    dominant = raw["dominant_emotion"]
    scores = {k: round(float(v), 4) for k, v in emotions.items()}
    return {"dominant_emotion": dominant, "scores": scores}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_emotion_face.py::TestShapeEmotionResult -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/emotion_face.py tests/test_emotion_face.py
git commit -m "feat(step4): add emotion_face module scaffold with _shape_emotion_result"
```

---

## Task 2: `_analyze_frame` (DeepFace seam, lazy import)

**Files:**
- Modify: `pipeline/emotion_face.py`
- Modify: `tests/test_emotion_face.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_emotion_face.py`)**

```python
class TestAnalyzeFrame:
    def test_returns_none_when_no_face(self):
        import sys
        fake_df = MagicMock()
        fake_df.DeepFace.analyze.side_effect = ValueError("Face could not be detected")
        sys.modules["deepface"] = fake_df
        try:
            from pipeline.emotion_face import _analyze_frame
            assert _analyze_frame("frame") is None
        finally:
            del sys.modules["deepface"]

    def test_shapes_deepface_output(self):
        import sys
        fake_df = MagicMock()
        fake_df.DeepFace.analyze.return_value = [
            {"dominant_emotion": "happy", "emotion": {"happy": 0.9, "neutral": 0.1}}
        ]
        sys.modules["deepface"] = fake_df
        try:
            from pipeline.emotion_face import _analyze_frame
            result = _analyze_frame("frame")
        finally:
            del sys.modules["deepface"]
        assert result["dominant_emotion"] == "happy"
        assert result["scores"]["happy"] == 0.9
```

These tests inject a fake `deepface` module into `sys.modules` (the same stub-injection pattern documented in CLAUDE.md for pyannote), so they run **without** real deepface/TF installed. The lazy `from deepface import DeepFace` inside `_analyze_frame` resolves to the fake.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_emotion_face.py::TestAnalyzeFrame -v`
Expected: ERROR — `ImportError: cannot import name '_analyze_frame' from 'pipeline.emotion_face'` (function not defined yet).

- [ ] **Step 3: Write `_analyze_frame` (append to `pipeline/emotion_face.py`, after `_shape_emotion_result`)**

```python
def _analyze_frame(frame):
    """Run DeepFace emotion analysis on a single frame.

    Returns {dominant_emotion, scores} or None if no face is detected.
    DeepFace with enforce_detection=True raises ValueError when no face is
    found — we treat that as a skip (return None), per spec.
    """
    from deepface import DeepFace
    try:
        raw = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=True)
    except ValueError:
        return None
    return _shape_emotion_result(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_emotion_face.py::TestAnalyzeFrame -v`
Expected: 2 PASS (no real deepface loaded — the sys.modules fake is used)

- [ ] **Step 5: Commit**

```bash
git add pipeline/emotion_face.py tests/test_emotion_face.py
git commit -m "feat(step4): add _analyze_frame DeepFace seam with no-face skip"
```

---

## Task 3: `_iter_frames` (cv2 seam, lazy import) + integration tests

**Files:**
- Modify: `pipeline/emotion_face.py`
- Modify: `tests/test_emotion_face.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_emotion_face.py`)**

```python
@pytest.mark.skipif(not _cv2_available(), reason="opencv-python not installed")
class TestIterFramesIntegration:
    def test_invalid_video_yields_nothing(self, tmp_path):
        from pipeline.emotion_face import _iter_frames
        assert list(_iter_frames(str(tmp_path / "nope.mp4"))) == []

    def test_samples_frames_at_interval(self, tmp_path):
        import cv2
        import numpy as np
        from pipeline.emotion_face import _iter_frames
        video = tmp_path / "synthetic.mp4"
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video), fourcc, fps, (64, 64))
        for _ in range(35):  # 3.5s of video at 10fps
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        cap = cv2.VideoCapture(str(video))
        readback = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if readback <= 0:
            pytest.skip("cv2 could not read back the synthetic video; codec unavailable")
        timestamps = [t for t, _ in _iter_frames(str(video), interval=1)]
        assert timestamps == [0.0, 1.0, 2.0, 3.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_emotion_face.py::TestIterFramesIntegration -v`
Expected: if opencv installed → ERROR `ImportError: cannot import name '_iter_frames'`; if not installed → 2 SKIPPED (install `opencv-python` to get the real RED, per Prerequisites).

- [ ] **Step 3: Write `_iter_frames` (append to `pipeline/emotion_face.py`, after `_analyze_frame`)**

```python
def _iter_frames(video_path, interval=SAMPLE_INTERVAL):
    """Yield (timestamp_seconds, frame_ndarray) pairs sampled every `interval` seconds.

    Seeks by frame index (CAP_PROP_POS_FRAMES) for speed on long meeting
    videos. Videos cv2 cannot parse (fps=0 or no frames) yield nothing.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or total_frames <= 0:
            return
        duration = total_frames / fps
        timestamp = 0.0
        while timestamp < duration:
            frame_idx = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                yield timestamp, frame
            timestamp += interval
    finally:
        cap.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_emotion_face.py::TestIterFramesIntegration -v`
Expected: 2 PASS (with opencv installed) — the synthetic 3.5s video yields exactly timestamps `[0.0, 1.0, 2.0, 3.0]`.

- [ ] **Step 5: Commit**

```bash
git add pipeline/emotion_face.py tests/test_emotion_face.py
git commit -m "feat(step4): add _iter_frames cv2 frame sampler"
```

---

## Task 4: `extract_face_emotion` (orchestration) + unit + integration tests

**Files:**
- Modify: `pipeline/emotion_face.py`
- Modify: `tests/test_emotion_face.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_emotion_face.py`)**

```python
class TestExtractFaceEmotion:
    def test_raises_when_video_missing(self):
        from pipeline.emotion_face import extract_face_emotion
        with pytest.raises(FileNotFoundError, match="not found"):
            extract_face_emotion("nonexistent.mp4")

    def test_returns_one_record_per_detected_face(self):
        from pipeline.emotion_face import extract_face_emotion
        frames = [(0.0, "frame0"), (10.0, "frame1")]
        analysis = {"dominant_emotion": "happy", "scores": {"happy": 0.9, "neutral": 0.1}}
        with patch("pipeline.emotion_face._iter_frames", return_value=iter(frames)):
            with patch("pipeline.emotion_face._analyze_frame", return_value=analysis):
                result = extract_face_emotion("video.mp4")
        assert len(result) == 2
        assert result[0]["timestamp"] == 0.0
        assert result[0]["dominant_emotion"] == "happy"
        assert result[0]["scores"]["happy"] == 0.9

    def test_skips_frames_with_no_face(self):
        from pipeline.emotion_face import extract_face_emotion
        frames = [(0.0, "f0"), (10.0, "f1"), (20.0, "f2")]

        def fake_analyze(frame):
            return None if frame == "f1" else {"dominant_emotion": "neutral", "scores": {"neutral": 1.0}}

        with patch("pipeline.emotion_face._iter_frames", return_value=iter(frames)):
            with patch("pipeline.emotion_face._analyze_frame", side_effect=fake_analyze):
                result = extract_face_emotion("video.mp4")
        assert len(result) == 2
        assert [r["timestamp"] for r in result] == [0.0, 20.0]

    def test_record_has_required_keys(self):
        from pipeline.emotion_face import extract_face_emotion
        frames = [(5.0, "f0")]
        analysis = {"dominant_emotion": "surprise", "scores": {"surprise": 0.5, "neutral": 0.5}}
        with patch("pipeline.emotion_face._iter_frames", return_value=iter(frames)):
            with patch("pipeline.emotion_face._analyze_frame", return_value=analysis):
                result = extract_face_emotion("video.mp4")
        assert set(result[0].keys()) == {"timestamp", "dominant_emotion", "scores"}

    def test_empty_video_returns_empty_list(self):
        from pipeline.emotion_face import extract_face_emotion
        with patch("pipeline.emotion_face._iter_frames", return_value=iter([])):
            with patch("pipeline.emotion_face._analyze_frame") as mock_analyze:
                result = extract_face_emotion("video.mp4")
        assert result == []
        mock_analyze.assert_not_called()

    def test_timestamps_are_rounded_to_two_decimals(self):
        from pipeline.emotion_face import extract_face_emotion
        frames = [(10.123456, "f0")]
        analysis = {"dominant_emotion": "neutral", "scores": {"neutral": 1.0}}
        with patch("pipeline.emotion_face._iter_frames", return_value=iter(frames)):
            with patch("pipeline.emotion_face._analyze_frame", return_value=analysis):
                result = extract_face_emotion("video.mp4")
        assert result[0]["timestamp"] == 10.12


class TestExtractFaceEmotionIntegration:
    @pytest.mark.skipif(
        not _cv2_available()
        or not _deepface_available()
        or not os.path.exists("input/meeting.mp4")
        or not os.path.exists(os.path.expanduser("~/.deepface/weights")),
        reason="requires opencv-python, deepface, input/meeting.mp4, and downloaded DeepFace weights",
    )
    def test_with_real_meeting_video(self):
        from pipeline.emotion_face import extract_face_emotion
        result = extract_face_emotion("input/meeting.mp4", interval=10)
        assert len(result) > 0
        assert "dominant_emotion" in result[0]
        assert "scores" in result[0]
        assert "timestamp" in result[0]
        dominant = result[0]["dominant_emotion"]
        assert 0 <= result[0]["scores"][dominant] <= 1
```

The 6 `TestExtractFaceEmotion` unit tests patch `_iter_frames` + `_analyze_frame`, so they run with **no** cv2/deepface loaded. The integration test is skip-guarded on all four conditions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_emotion_face.py::TestExtractFaceEmotion -v`
Expected: ERROR — `ImportError: cannot import name 'extract_face_emotion' from 'pipeline.emotion_face'`.

- [ ] **Step 3: Write `extract_face_emotion` (append to `pipeline/emotion_face.py`, after `_iter_frames`)**

```python
def extract_face_emotion(video_path, interval=SAMPLE_INTERVAL):
    """Extract facial emotion for sampled frames of the meeting video.

    Samples one frame every `interval` seconds, runs DeepFace on each, and
    skips frames with no detected face (no crash). Returns a list of
    {timestamp, dominant_emotion, scores} records.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    results = []
    for timestamp, frame in _iter_frames(video_path, interval):
        analysis = _analyze_frame(frame)
        if analysis is None:
            print(f"  ⚠ no face detected at {timestamp:.1f}s, skipping")
            continue
        results.append({
            "timestamp": round(float(timestamp), 2),
            "dominant_emotion": analysis["dominant_emotion"],
            "scores": analysis["scores"],
        })
    return results
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/test_emotion_face.py::TestExtractFaceEmotion -v`
Expected: 6 PASS

- [ ] **Step 5: Run the full module test file**

Run: `python -m pytest tests/test_emotion_face.py -v`
Expected: 11 unit tests PASS (3 shape + 2 analyze + 6 extract) + 2 cv2 integration tests PASS/skip + 1 full integration test skipped (deepface/weights absent) or PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/emotion_face.py tests/test_emotion_face.py
git commit -m "feat(step4): add extract_face_emotion orchestration with no-face skip"
```

---

## Task 5: CLI script `pipeline/04_emotion_face.py` + guard test

**Files:**
- Create: `pipeline/04_emotion_face.py`
- Create: `tests/test_04_emotion_face.py`

- [ ] **Step 1: Write the failing test + a stub script (honest RED)**

Create `tests/test_04_emotion_face.py` (mirrors `tests/test_03_emotion_voice.py`):

```python
import os
import sys
import subprocess

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def test_exits_when_video_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "pipeline/04_emotion_face.py")],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "meeting.mp4" in result.stdout
```

Create `pipeline/04_emotion_face.py` as a minimal stub (no guard yet):

```python
#!/usr/bin/env python3
"""Step 4: Extract facial emotion from video frames."""


def main():
    pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_04_emotion_face.py -v`
Expected: FAIL — `assert result.returncode == 1` fails because the stub's `main()` exits 0.

- [ ] **Step 3: Write the full CLI (replace the stub body of `main()`)**

Replace `pipeline/04_emotion_face.py` with:

```python
#!/usr/bin/env python3
"""Step 4: Extract facial emotion from video frames.

Samples one frame every 10 seconds from input/meeting.mp4, runs DeepFace
emotion analysis on each frame, and skips frames with no detected face.

Output: output/face_emotion.json
"""

import json
import os
import sys

VIDEO_FILE = "input/meeting.mp4"
OUTPUT_FILE = "output/face_emotion.json"


def main():
    if not os.path.exists(VIDEO_FILE):
        print(f"ERROR: {VIDEO_FILE} not found. Place the meeting video at input/meeting.mp4.")
        sys.exit(1)

    from emotion_face import extract_face_emotion, SAMPLE_INTERVAL

    print(f"→ Extracting facial emotion (one frame every {SAMPLE_INTERVAL}s)...")
    emotions = extract_face_emotion(VIDEO_FILE, interval=SAMPLE_INTERVAL)

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(emotions, f, indent=2)

    print(f"✓ Facial emotion saved to {OUTPUT_FILE} ({len(emotions)} frames)")


if __name__ == "__main__":
    main()
```

Note: `from emotion_face import extract_face_emotion, SAMPLE_INTERVAL` is a bare import (works because Python puts the script's own `pipeline/` dir on `sys.path` when run directly), mirroring step 3's `from emotion_voice import extract_voice_emotion`. It's placed inside `main()` so the heavy module loads only after the video-exists guard passes.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_04_emotion_face.py -v`
Expected: 1 PASS

- [ ] **Step 5: Run the full test suite for regressions**

Run: `python -m pytest tests/ -v`
Expected: all previous tests PASS + new step-4 tests PASS/skip. No regressions.

- [ ] **Step 6: Commit**

```bash
git add pipeline/04_emotion_face.py tests/test_04_emotion_face.py
git commit -m "feat(step4): add CLI entry point for facial emotion extraction"
```

---

## Final Verification

- [ ] Confirm `run.py` already wires step 4 (`run.py:25-30`) — no change needed.
- [ ] Confirm `requirements.txt` already lists `deepface` + `opencv-python` (`requirements.txt:14-16`) — no change needed.
- [ ] Run full suite: `python -m pytest tests/ -v`
- [ ] (Optional, with deepface installed + `input/meeting.mp4` present) end-to-end: `rm -f output/face_emotion.json && python run.py` and inspect `output/face_emotion.json` matches the spec schema: `[{timestamp, dominant_emotion, scores{}}]`.

---

## Self-Review

### 1. Spec coverage

| Spec requirement (Step 4) | Task(s) |
|---|---|
| Sample one frame every 10 seconds from `meeting.mp4` via OpenCV | Task 3 — `_iter_frames` (cv2, `SAMPLE_INTERVAL=10`, seek by frame index) |
| Run DeepFace on each frame | Task 2 — `_analyze_frame` (lazy `from deepface import DeepFace`) |
| If no face detected: skip frame, log warning, continue (no crash) | Task 2 (ValueError→None) + Task 4 (`extract_face_emotion` skips None, prints `⚠ no face detected…`) |
| Output schema: `[{timestamp, dominant_emotion, scores{}}]` | Task 1 (`_shape_emotion_result` → scores) + Task 4 (record shape + timestamp rounding) |
| Report handles missing face data gracefully | Module never crashes on no-face; CLI writes whatever frames succeeded |
| Orchestration: `run.py` skips step if output exists | Already wired in `run.py:25-30` (no change) |
| CLI guard: missing video → exit with message | Task 5 — `main()` guard + `tests/test_04_emotion_face.py` |
| Frames sampled every 10s (default) | `SAMPLE_INTERVAL = 10` in module + imported by CLI |

### 2. Placeholder scan

No TBD / TODO / "implement later" / "handle edge cases" / "similar to Task N" anywhere. Every code step contains complete, runnable code. Every command step contains the exact command and expected result.

### 3. Type consistency

- `_shape_emotion_result(raw) -> {"dominant_emotion": str, "scores": dict[str, float]}` — used identically by `_analyze_frame` (Task 2) and asserted in `TestShapeEmotionResult` (Task 1) and the orchestration tests (Task 4).
- `_analyze_frame(frame) -> dict | None` — `None` contract on no-face is what `extract_face_emotion` (Task 4) checks; the `TestAnalyzeFrame` tests assert both branches.
- `_iter_frames(video_path, interval) -> Iterator[tuple[float, ndarray]]` — `extract_face_emotion` unpacks `for timestamp, frame in _iter_frames(...)`; `TestIterFramesIntegration` asserts the yielded timestamps.
- `extract_face_emotion(video_path, interval=SAMPLE_INTERVAL) -> list[dict]` — signature consistent across module (Task 4), CLI call `extract_face_emotion(VIDEO_FILE, interval=SAMPLE_INTERVAL)` (Task 5), and integration test (Task 4).
- Output record keys `{"timestamp", "dominant_emotion", "scores"}` — identical in `extract_face_emotion` (Task 4), the `test_record_has_required_keys` assertion, the integration test, and the spec schema.
- `SAMPLE_INTERVAL = 10` defined once in the module; imported (not redefined) by the CLI — no magic-number duplication.

### Gaps found and fixed

- Initial design imported `cv2` at module top; the environment check showed cv2/deepface are not installed, which would break every unit test at import time. Fixed by lazy-importing both inside the functions that need them (pyannote pattern), and skip-guarding the cv2/deepface-touching integration tests with `importlib.util.find_spec` (cheap, no module execution, no TF load at collection time).
