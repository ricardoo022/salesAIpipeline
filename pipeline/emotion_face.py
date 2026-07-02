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
