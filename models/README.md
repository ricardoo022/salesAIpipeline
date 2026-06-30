# models/

Local cache for downloaded model weights.

## What gets cached here

**`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`** — voice emotion model used by `pipeline/03_emotion_voice.py`. Downloaded automatically from HuggingFace on first run (~1.5 GB).

The model directory is passed to the `transformers` pipeline via `cache_dir`. Subsequent runs load from here without re-downloading.

This directory is gitignored. Do not commit model weights.
