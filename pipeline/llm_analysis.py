"""LLM analysis for step 5.

Loads the four upstream JSON outputs (transcript, audio_features, voice_emotion,
face_emotion), projects them into compact per-segment blocks, and calls the
Claude API twice -- once transcript-only, once multimodal -- to produce the
side-by-side analysis that is the demo's killer feature.

`anthropic` is lazy-imported inside the call path (same pattern as pyannote in
`diarize.py` and cv2/deepface in `emotion_face.py`) so the module loads cleanly
and the pure-logic unit tests run without the SDK installed.
"""

import json
import os
import time

MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS = 4096
RATE_LIMIT_WAIT = 10  # seconds; spec: retry once after 10s

TRANSCRIPT_FILE = "output/transcript.json"
AUDIO_FEATURES_FILE = "output/audio_features.json"
VOICE_EMOTION_FILE = "output/voice_emotion.json"
FACE_EMOTION_FILE = "output/face_emotion.json"
OUTPUT_FILE = "output/analysis.json"


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS (spec uses 00:12:24 style)."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _classify_speakers(transcript: list[dict]) -> dict[str, str]:
    """Map diarization labels to REP / PROSPECT / OTHER by total talk time.

    Longest total talk time -> REP, second -> PROSPECT, rest -> OTHER.
    Deterministic default (no extra API call); the LLM still reasons over both
    speakers' content regardless of label. Single swap point for an LLM-infer
    call later.
    """
    totals: dict[str, float] = {}
    for seg in transcript:
        sp = seg.get("speaker", "UNKNOWN")
        totals[sp] = totals.get(sp, 0.0) + (seg.get("end", 0) - seg.get("start", 0))
    ranked = sorted(totals, key=lambda s: totals[s], reverse=True)
    mapping: dict[str, str] = {}
    for i, sp in enumerate(ranked):
        mapping[sp] = "REP" if i == 0 else ("PROSPECT" if i == 1 else "OTHER")
    return mapping
