#!/usr/bin/env python3
"""Step 2: Extract audio features from transcript segments.

Loads the transcript from step 1 and the 16kHz WAV audio, then for each
diarized segment computes pitch mean/std, energy mean, speech rate,
pause ratio, and zero crossing rate.

Output: output/audio_features.json
"""

import json
import os
import sys

from features import extract_audio_features

TRANSCRIPT_FILE = "output/transcript.json"
AUDIO_FILE = "output/audio_temp.wav"
OUTPUT_FILE = "output/audio_features.json"


def main():
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"ERROR: {TRANSCRIPT_FILE} not found. Run step 1 first.")
        sys.exit(1)

    if not os.path.exists(AUDIO_FILE):
        print(f"ERROR: {AUDIO_FILE} not found. Run step 1 first.")
        sys.exit(1)

    with open(TRANSCRIPT_FILE) as f:
        transcript = json.load(f)

    print("→ Extracting audio features (pitch, energy, speech rate, pauses, ZCR)...")
    features = extract_audio_features(transcript, AUDIO_FILE)

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(features, f, indent=2)

    print(f"✓ Audio features saved to {OUTPUT_FILE} ({len(features)} segments)")


if __name__ == "__main__":
    main()
