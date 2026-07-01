#!/usr/bin/env python3
"""Step 3: Extract voice emotion from transcript segments.

Uses audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim to extract
valence, arousal, and dominance per segment. Long segments (>15s) are
split into chunks and results averaged.

Output: output/voice_emotion.json
"""

import json
import os
import sys

from emotion_voice import extract_voice_emotion

SEGMENTS_FILE = "output/audio_features.json"
AUDIO_FILE = "output/audio_temp.wav"
OUTPUT_FILE = "output/voice_emotion.json"


def main():
    if not os.path.exists(SEGMENTS_FILE):
        print(f"ERROR: {SEGMENTS_FILE} not found. Run step 2 first.")
        sys.exit(1)

    if not os.path.exists(AUDIO_FILE):
        print(f"ERROR: {AUDIO_FILE} not found. Run step 1 first.")
        sys.exit(1)

    with open(SEGMENTS_FILE) as f:
        segments = json.load(f)

    print("→ Extracting voice emotion (valence, arousal, dominance) from audio segments...")
    emotions = extract_voice_emotion(segments, AUDIO_FILE)

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(emotions, f, indent=2)

    print(f"✓ Voice emotion saved to {OUTPUT_FILE} ({len(emotions)} segments)")


if __name__ == "__main__":
    main()
