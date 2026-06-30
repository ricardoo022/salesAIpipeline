# Sales AI Pipeline

Multimodal sales meeting analysis pipeline. Takes a B2B sales meeting video and produces a self-contained HTML report with engagement scores, deal probability, talk ratio, emotional timeline, critical moments, and coaching recommendations.

The centrepiece is a **side-by-side comparison** between transcript-only analysis and full multimodal analysis — demonstrating what audio + facial signals reveal that text alone misses. Built as a demo for [Scale Labs](https://scalelabs.pt), a Portuguese B2B sales consulting company.

---

## Quick Start

```bash
# 1. Create and activate virtualenv
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add credentials
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and HF_TOKEN

# 4. Download demo video
yt-dlp https://www.youtube.com/watch?v=2zy6KTIllY8 -o input/meeting.mp4

# 5. Run the pipeline
python run.py

# 6. Open the report
# output/report.html — double-click to open in any browser
```

---

## Pipeline

Six sequential steps. Each reads JSON from the previous step and writes JSON for the next. `run.py` orchestrates them and skips steps whose output already exists.

```
input/meeting.mp4
  │
  ├─ 01_transcribe.py      → output/transcript.json       WhisperX + pyannote diarization
  ├─ 02_audio_features.py  → output/audio_features.json   librosa: pitch, energy, speech rate, pauses
  ├─ 03_emotion_voice.py   → output/voice_emotion.json    audeering wav2vec2: valence/arousal/dominance
  ├─ 04_emotion_face.py    → output/face_emotion.json     DeepFace, one frame every 10s
  ├─ 05_llm_analysis.py    → output/analysis.json         Claude API (run twice: text-only + multimodal)
  └─ 06_report.py          → output/report.html           Self-contained HTML + Chart.js
```

To re-run from a specific step, delete its output file:

```bash
rm output/analysis.json && python run.py   # re-runs steps 5 and 6
```

---

## Environment Variables

Create a `.env` file at the project root:

```
ANTHROPIC_API_KEY=...    # Claude API — used by step 5
HF_TOKEN=...             # HuggingFace — used by step 1 (pyannote diarization)
```

For `HF_TOKEN` you must also accept model terms at:
- huggingface.co/pyannote/speaker-diarization-3.1
- huggingface.co/pyannote/segmentation-3.0

---

## Directory Structure

```
input/       Place meeting.mp4 here before running
pipeline/    The six analysis scripts
output/      Generated JSON files and final report.html
models/      Cached model weights (audeering wav2vec2, auto-downloaded on first run)
docs/        Design specs and project documentation
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| WhisperX | Word-level transcription + speaker diarization |
| pyannote.audio | Speaker segmentation |
| librosa | Audio feature extraction |
| audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | Voice emotion (valence/arousal/dominance) |
| DeepFace | Facial emotion detection |
| Anthropic Claude API (`claude-sonnet-4-6`) | LLM analysis — runs twice per meeting |
| Chart.js (CDN) | Engagement timeline in the HTML report |
| python-dotenv | `.env` loading |
| yt-dlp | Demo video download |

---

## Notes

- Runs entirely on CPU (no GPU required). Steps 1 and 3 are slow on long videos.
- The report is fully self-contained — no server needed, just open `output/report.html` in a browser.
- Step 5 retries once after 10 seconds on rate limit errors.
- Step 4 silently skips frames where no face is detected.
