#!/usr/bin/env python3
"""Step 1: Transcribe + diarize meeting audio.

Extracts audio from input/meeting.mp4 via ffmpeg, then runs WhisperX
for word-level transcription and pyannote for speaker diarization.

Output: output/transcript.json
"""

import json
import os
import sys

from dotenv import load_dotenv

from audio import extract_audio
from diarize import diarize_audio
from transcribe import transcribe_audio, merge_speaker_labels

INPUT_VIDEO = "input/meeting.mp4"
OUTPUT_FILE = "output/transcript.json"


def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: {INPUT_VIDEO} not found.")
        sys.exit(1)

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN is required for diarization. Set it in .env")
        sys.exit(1)

    os.makedirs("output", exist_ok=True)

    print("→ Extracting audio via ffmpeg...")
    audio_path = extract_audio(INPUT_VIDEO)
    print(f"  Audio saved to {audio_path}")

    print("→ Running WhisperX transcription (large-v2)...")
    segments = transcribe_audio(audio_path)
    print(f"  Transcribed {len(segments)} segments")

    print("→ Running pyannote speaker diarization...")
    diarization = diarize_audio(audio_path, hf_token=hf_token)
    print(f"  Found {len(set(d['speaker'] for d in diarization))} speakers")

    transcript = merge_speaker_labels(segments, diarization)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(transcript, f, indent=2)

    print(f"✓ Transcript saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
