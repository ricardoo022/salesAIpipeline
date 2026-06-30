"""Pyannote speaker diarization for step 1."""
import os

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def diarize_audio(audio_path: str, hf_token: str) -> list[dict]:
    """Run pyannote speaker diarization on audio.

    Returns list of {speaker, start, end} segments.
    Requires HF_TOKEN with access to pyannote model terms accepted at:
    huggingface.co/pyannote/speaker-diarization-3.1
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not hf_token:
        raise ValueError("HF_TOKEN is required for diarization. Set it in .env")

    # Deferred import: pyannote.audio loads torch at import time and crashes
    # in CPU-only environments without CUDA libs.
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
    diarization = pipeline(audio_path)

    return [
        {"speaker": speaker, "start": round(turn.start, 3), "end": round(turn.end, 3)}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
