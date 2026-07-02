#!/usr/bin/env python3
"""Step 5: LLM analysis.

Merges the four upstream JSONs (transcript, audio_features, voice_emotion,
face_emotion) and calls Claude twice -- transcript-only then multimodal -- to
produce the side-by-side analysis that is the demo's killer feature.

Output: output/analysis.json
"""

import os
import sys

TRANSCRIPT_FILE = "output/transcript.json"
AUDIO_FEATURES_FILE = "output/audio_features.json"
VOICE_EMOTION_FILE = "output/voice_emotion.json"
FACE_EMOTION_FILE = "output/face_emotion.json"
OUTPUT_FILE = "output/analysis.json"

REQUIRED_INPUTS = (
    TRANSCRIPT_FILE,
    AUDIO_FEATURES_FILE,
    VOICE_EMOTION_FILE,
    FACE_EMOTION_FILE,
)


def main():
    missing = [p for p in REQUIRED_INPUTS if not os.path.exists(p)]
    if missing:
        print(f"ERROR: missing input(s): {', '.join(missing)}")
        print("Run the earlier pipeline steps first (python run.py).")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env (see .env.example).")
        sys.exit(1)

    from llm_analysis import run_analysis

    print("→ Running LLM analysis (transcript-only + multimodal)...")
    run_analysis(
        TRANSCRIPT_FILE,
        AUDIO_FEATURES_FILE,
        VOICE_EMOTION_FILE,
        FACE_EMOTION_FILE,
        OUTPUT_FILE,
        api_key=api_key,
    )
    print(f"✓ Analysis saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
