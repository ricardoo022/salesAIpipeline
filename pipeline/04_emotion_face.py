#!/usr/bin/env python3
"""Step 4: Extract facial emotion from video frames.

Samples one frame every 10 seconds from input/meeting.mp4, runs DeepFace
emotion analysis on each frame, and skips frames with no detected face.

Output: output/face_emotion.json
"""

import json
import os
import sys

VIDEO_FILE = "input/meeting.mp4"
OUTPUT_FILE = "output/face_emotion.json"


def main():
    if not os.path.exists(VIDEO_FILE):
        print(f"ERROR: {VIDEO_FILE} not found. Place the meeting video at input/meeting.mp4.")
        sys.exit(1)

    from emotion_face import extract_face_emotion, SAMPLE_INTERVAL

    print(f"→ Extracting facial emotion (one frame every {SAMPLE_INTERVAL}s)...")
    emotions = extract_face_emotion(VIDEO_FILE, interval=SAMPLE_INTERVAL)

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(emotions, f, indent=2)

    print(f"✓ Facial emotion saved to {OUTPUT_FILE} ({len(emotions)} frames)")


if __name__ == "__main__":
    main()
