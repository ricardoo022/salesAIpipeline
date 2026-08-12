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
| `step5-walkthrough.md` | Step 5 — LLM Analysis: Claude tool-use, transcript-only vs multimodal side-by-side, the killer-feature dissonance moments, plus the MAX_TOKENS truncation bug found via code review |

### `Problem/`

Design documents for the BANT evidence-extraction subsystem. Epic 1's
hierarchical chunking (transcript sections and bounded extraction chunks,
US-1.1–US-1.3) is implemented in `pipeline/qualification/`; the topic-agent
harness and later epics remain planned.

| File | Description |
|------|-------------|
| `ARCHITECTURE.md` | Scope, responsibilities, and evidence flow |
| `HIERARCHICAL-CHUNKING.md` | Transcript sections, bounded chunks, overlap, and source traceability (implemented, US-1.1–US-1.3) |
| `HARNESS.md` | Topic agents, validation, assembly, retries, and signal linking (planned) |
| `FOLDER-ARCHITECTURE.md` | Qualification package and runtime artifacts |
| `EPICS-AND-USER-STORIES.md` | Epics, schemas, acceptance criteria, and chunking evaluation plan |
