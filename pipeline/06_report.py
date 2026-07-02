#!/usr/bin/env python3
"""Step 6: Generate the HTML report.

Reads the four upstream JSONs (transcript, voice_emotion, face_emotion,
analysis) and writes a single self-contained output/report.html.

Output: output/report.html
"""

import json
import os
import sys

TRANSCRIPT_FILE = "output/transcript.json"
VOICE_EMOTION_FILE = "output/voice_emotion.json"
FACE_EMOTION_FILE = "output/face_emotion.json"
ANALYSIS_FILE = "output/analysis.json"
OUTPUT_FILE = "output/report.html"

REQUIRED_INPUTS = (
    TRANSCRIPT_FILE,
    VOICE_EMOTION_FILE,
    FACE_EMOTION_FILE,
    ANALYSIS_FILE,
)


def main():
    missing = [p for p in REQUIRED_INPUTS if not os.path.exists(p)]
    if missing:
        print(f"ERROR: missing input(s): {', '.join(missing)}")
        print("Run the earlier pipeline steps first (python run.py).")
        sys.exit(1)

    from report import render_report

    with open(TRANSCRIPT_FILE) as f:
        transcript = json.load(f)
    with open(VOICE_EMOTION_FILE) as f:
        voice_emotion = json.load(f)
    with open(FACE_EMOTION_FILE) as f:
        face_emotion = json.load(f)
    with open(ANALYSIS_FILE) as f:
        analysis = json.load(f)

    print("→ Rendering report...")
    html = render_report(transcript, voice_emotion, face_emotion, analysis)

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(html)

    print(f"✓ Report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
