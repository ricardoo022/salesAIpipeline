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
# DeepFace's default `opencv` detector backend needs the haarcascade XMLs in
# cv2/data/, but opencv-python 5.x ships that dir empty (only __init__.py), so
# the default backend raises ValueError on every frame — silently swallowed as
# "no face" by the ValueError handler below. retinaface is a deepface dependency
# and ships its own weights (auto-downloaded to ~/.deepface/weights), so it works
# wherever deepface is installed.
DETECTOR_BACKEND = "retinaface"


def _shape_emotion_result(raw: dict) -> dict:
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


def _analyze_frame(frame) -> dict | None:
    """Run DeepFace emotion analysis on a single frame.

    Returns {dominant_emotion, scores} or None if no face is detected.
    DeepFace with enforce_detection=True raises ValueError when no face is
    found — we treat that as a skip (return None), per spec.
    """
    from deepface import DeepFace
    try:
        raw = DeepFace.analyze(
            frame, actions=["emotion"], enforce_detection=True, detector_backend=DETECTOR_BACKEND
        )
    except ValueError:
        return None
    return _shape_emotion_result(raw)


def _iter_frames(video_path: str, interval: int = SAMPLE_INTERVAL):
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


def extract_face_emotion(video_path: str, interval: int = SAMPLE_INTERVAL) -> list[dict]:
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
