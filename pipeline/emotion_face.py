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
