"""WhisperX word-level transcription for step 1."""
import os
import whisperx

WHISPER_MODEL = "large-v2"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def transcribe_audio(audio_path: str) -> list[dict]:
    """Return aligned segments from WhisperX.

    Each segment: {start, end, text, words[{word, start, end}]}.
    Note: `speaker` field is absent — added by pyannote diarization (step 1 TODO).
    May raise exceptions from whisperx if the audio is malformed or the language
    has no alignment model.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = whisperx.load_model(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    result = model.transcribe(audio_path)

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device=DEVICE)

    return aligned["segments"]
