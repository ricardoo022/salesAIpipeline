# docs/

Project documentation and design specifications.

## Contents

### `superpowers/specs/`

Design specs written before implementation. These are the authoritative reference for intended behaviour when the code and the spec disagree.

| File | Description |
|------|-------------|
| `2026-06-30-sales-coach-mvp-design.md` | Full MVP design spec — pipeline architecture, per-step schemas, report layout, success criteria |

### `superpowers/plans/`

Implementation plans with task breakdowns, written before each development session.

| File | Description |
|------|-------------|
| `2026-06-30-audio-extraction-and-setup.md` | Audio extraction setup, venv, video download — tasks 0–4 |
| `2026-07-01-step-3-voice-emotion.md` | Step 3 (Voice Emotion) implementation plan — audeering wav2vec2 module, CLI, TDD tasks |
| `2026-07-02-step4-face-emotion.md` | Step 4 (Facial Emotion) implementation plan — DeepFace + OpenCV module, CLI, TDD tasks |
| `2026-07-02-step5-llm-analysis.md` | Step 5 (LLM Analysis) implementation plan — Claude tool-use, transcript-only vs multimodal, truncation guard, TDD tasks |

### `steps/`

Line-by-line walkthroughs of each pipeline step, written in Portuguese for the Scale Labs team.

| File | Description |
|------|-------------|
| `step1-walkthrough.md` | Step 1 — Transcription + Diarization: ffmpeg, WhisperX, pyannote, speaker merge |
| `step2-walkthrough.md` | Step 2 — Audio Features: pitch, energy, speech rate, pauses, ZCR with librosa |
| `step3-walkthrough.md` | Step 3 — Voice Emotion: audeering wav2vec2 VAD extraction, plus bugs found via statistical validation and the fixes |
| `step4-walkthrough.md` | Step 4 — Facial Emotion: DeepFace + OpenCV frame sampling, plus bugs found running on the real video (broken opencv detector backend, 0–100 vs 0–1 scores) |
