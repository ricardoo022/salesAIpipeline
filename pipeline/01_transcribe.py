#!/usr/bin/env python3
"""Step 1: Transcribe + diarize meeting audio.

Extracts audio from input/meeting.mp4 via ffmpeg, then runs WhisperX
for word-level transcription and pyannote for speaker diarization.

Output: output/transcript.json
"""

import json
import os

from audio import extract_audio, AUDIO_TEMP_FILE

INPUT_VIDEO = "input/meeting.mp4"
OUTPUT_FILE = "output/transcript.json"


def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: {INPUT_VIDEO} not found.")
        sys.exit(1)

    os.makedirs("output", exist_ok=True)

    print("→ Extracting audio via ffmpeg...")
    audio_path = extract_audio(INPUT_VIDEO)
    print(f"  Audio saved to {audio_path}")

    # TODO: WhisperX transcription
    # TODO: pyannote speaker diarization
    # TODO: Merge words with speaker labels

    # Placeholder output
    transcript = []
    with open(OUTPUT_FILE, "w") as f:
        json.dump(transcript, f, indent=2)

    print(f"✓ Transcript saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
