#!/usr/bin/env python3
import subprocess
import sys
import os

STEPS = [
    {
        "n": 1,
        "script": "pipeline/01_transcribe.py",
        "output": "output/transcript.json",
        "name": "Transcription + Diarization",
    },
    {
        "n": 2,
        "script": "pipeline/02_audio_features.py",
        "output": "output/audio_features.json",
        "name": "Audio Features",
    },
    {
        "n": 3,
        "script": "pipeline/03_emotion_voice.py",
        "output": "output/voice_emotion.json",
        "name": "Voice Emotion",
    },
    {
        "n": 4,
        "script": "pipeline/04_emotion_face.py",
        "output": "output/face_emotion.json",
        "name": "Facial Emotion",
    },
    {
        "n": 5,
        "script": "pipeline/05_llm_analysis.py",
        "output": "output/analysis.json",
        "name": "LLM Analysis",
    },
    {
        "n": 6,
        "script": "pipeline/06_report.py",
        "output": "output/report.html",
        "name": "Report Generation",
    },
]


def main():
    if not os.path.exists("input/meeting.mp4"):
        print("ERROR: input/meeting.mp4 not found.")
        print("Download the demo video with:")
        print("  yt-dlp https://www.youtube.com/watch?v=2zy6KTIllY8 -o input/meeting.mp4")
        sys.exit(1)

    for step in STEPS:
        n = step["n"]
        name = step["name"]
        output = step["output"]
        script = step["script"]

        if os.path.exists(output):
            print(f"✓ Step {n} ({name}) already done, skipping")
            continue

        print(f"→ Running step {n} ({name})...")
        result = subprocess.run([sys.executable, script])

        if result.returncode != 0:
            print(f"\nERROR: Step {n} ({name}) failed.")
            print(f"  Fix the issue above, then delete {output} (if it was partially written) to retry:")
            print(f"  rm -f {output} && python run.py")
            sys.exit(1)

    print("\n✓ Pipeline complete. Open output/report.html in your browser.")


if __name__ == "__main__":
    main()
