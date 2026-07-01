import subprocess
import os

AUDIO_SAMPLE_RATE = 16000
AUDIO_TEMP_FILE = "output/audio_temp.wav"


def extract_audio(video_path: str, output_path: str = None) -> str:
    """Extract audio from video file as 16kHz mono WAV using ffmpeg."""
    if output_path is None:
        output_path = AUDIO_TEMP_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: sudo apt-get install ffmpeg"
        )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return output_path
