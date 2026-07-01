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

### `steps/`

Line-by-line walkthroughs of each pipeline step, written in Portuguese for the Scale Labs team.

| File | Description |
|------|-------------|
| `step1-walkthrough.md` | Step 1 — Transcription + Diarization: ffmpeg, WhisperX, pyannote, speaker merge |
| `step2-walkthrough.md` | Step 2 — Audio Features: pitch, energy, speech rate, pauses, ZCR with librosa |
