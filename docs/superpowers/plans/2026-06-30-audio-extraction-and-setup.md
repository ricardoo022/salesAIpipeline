# Audio Extraction & Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Set up the development environment, download the demo video, and implement audio extraction as part of pipeline step 1 (`01_transcribe.py`).

**Architecture:** Follows the existing project architecture where `pipeline/01_transcribe.py` handles both ffmpeg audio extraction + WhisperX transcription + pyannote diarization. This plan covers setup, video download, and the audio extraction portion of the script. The full transcription/diarization logic will be added in a follow-up plan.

**Tech Stack:** ffmpeg (audio extraction), yt-dlp (video download), Python subprocess (ffmpeg invocation), soundfile/librosa (optional audio verification)

---

### Task 0: Set Up Virtual Environment

**Files:**
- Create: `venv/` — Python virtual environment (not committed)

- [x] **Step 1: Create Python virtual environment**

```bash
python3 -m venv venv
```

Verify: `ls -ld venv/` shows the directory exists.

- [x] **Step 2: Install Python dependencies in the venv**

```bash
source venv/bin/activate && pip install -r requirements.txt
```

Verify: `source venv/bin/activate && which pip` points inside `venv/`.

> **Note:** All subsequent `python` and `pip` commands assume the venv is activated: `source venv/bin/activate`

---

### Task 1: Install System Dependencies & Download Demo Video

**Files:**
- Modify: `requirements.txt` (already has yt-dlp)
- Input: `input/meeting.mp4` (to be downloaded)

- [x] **Step 1: Install ffmpeg system dependency**

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Verify: `ffmpeg -version` should print version info.

- [x] **Step 2: Install yt-dlp in the venv**

```bash
source venv/bin/activate && pip install yt-dlp
```

Verify: `source venv/bin/activate && yt-dlp --version` should print version info.

- [x] **Step 3: Download the demo video (with yt-dlp from venv)**

```bash
source venv/bin/activate && yt-dlp https://www.youtube.com/watch?v=2zy6KTIllY8 -o input/meeting.mp4
```

Verify: `ls -lh input/meeting.mp4` shows the file exists and has reasonable size.

---

### Task 2: Implement Audio Extraction Utility

**Files:**
- Create: `pipeline/audio.py` — shared audio utilities used by multiple pipeline steps
- Modify: `pipeline/__init__.py` — ensure package exports

- [x] **Step 1: Create `pipeline/audio.py` with ffmpeg extraction function**

```python
import subprocess
import os

AUDIO_SAMPLE_RATE = 16000
AUDIO_TEMP_FILE = "output/audio_temp.wav"


def extract_audio(video_path: str, output_path: str = AUDIO_TEMP_FILE) -> str:
    """Extract audio from video file as 16kHz mono WAV using ffmpeg.

    Returns the path to the extracted audio file.
    """
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

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return output_path
```

- [x] **Step 2: Create a quick smoke test**

```python
# scripts/test_audio_extraction.py
import sys
sys.path.insert(0, "pipeline")
from audio import extract_audio

path = extract_audio("input/meeting.mp4")
print(f"Audio extracted to: {path}")
```

Run: `python scripts/test_audio_extraction.py`
Expected: Writes `output/audio_temp.wav` without error.

---

### Task 3: Integrate Audio Extraction into 01_transcribe.py

**Files:**
- Create: `pipeline/01_transcribe.py` — first pipeline script
- Create: `output/transcript.json` — output placeholder (actual transcription logic TBD)

- [x] **Step 1: Create `pipeline/01_transcribe.py` with audio extraction + transcription scaffolding**

```python
#!/usr/bin/env python3
"""Step 1: Transcribe + diarize meeting audio.

Extracts audio from input/meeting.mp4 via ffmpeg, then runs WhisperX
for word-level transcription and pyannote for speaker diarization.

Output: output/transcript.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
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
```

- [x] **Step 2: Test the script runs without errors**

Run: `python pipeline/01_transcribe.py`
Expected: Writes `output/audio_temp.wav` and `output/transcript.json` (empty array for now).

- [x] **Step 3: Run `run.py` to verify orchestration works**

Run: `python run.py`
Expected: Step 1 runs (output/transcript.json doesn't exist yet), completes successfully.

---

### Task 4: Clean Up & Verify

**Files:**
- Delete: `scripts/test_audio_extraction.py` (if created) — no longer needed
- Verify: `output/audio_temp.wav` and `output/transcript.json` exist

- [x] **Step 1: Remove test script**

```bash
rm -f scripts/test_audio_extraction.py
rmdir scripts 2>/dev/null; true
```

- [x] **Step 2: Verify final state**

```bash
ls -lh input/meeting.mp4 output/audio_temp.wav output/transcript.json
```

Expected: All three files exist.
