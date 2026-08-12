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
yt-dlp https://youtu.be/N0SF2nZS-S8 -o input/meeting.mp4

# 5. Run the pipeline
python run.py
```

---

## Pipeline

Sequential steps. Each reads JSON from the previous step and writes JSON for the next. `run.py` orchestrates them and skips steps whose output already exists.

```
input/meeting.mp4
  │
  ├─ 01_transcribe.py      → output/transcript.json       WhisperX + pyannote diarization
  ├─ 02_audio_features.py  → output/audio_features.json   librosa: pitch, energy, speech rate, pauses, ZCR
  ├─ 03_emotion_voice.py   → output/voice_emotion.json    audeering wav2vec2: valence, arousal, dominance
  ├─ 04_emotion_face.py    → output/face_emotion.json      DeepFace facial emotion (every 10s)
  ├─ 05_llm_analysis.py    → output/analysis.json          Claude API (transcript-only + multimodal)
  └─ 06_report.py          → output/report.html             self-contained HTML report
```

`run.py` orchestrates all six steps sequentially, skipping any step whose output already exists.

---

## Planned BANT Evidence Extraction

The planned qualification subsystem runs after the six analysis steps. It
extracts Budget, Authority, Need, and Timeline evidence from the existing
transcript and links the existing audio, voice, and facial measurements. It
does not decide whether a lead qualifies and does not rerun signal analysis.

Epic 1 uses deterministic, custom Python chunking rather than vector search:

```text
transcript segments
  -> chronological sections
  -> bounded overlapping chunks
  -> four topic agents
  -> grounded BANT evidence
```

Chunks contain rendered speaker-labeled text for the LLM plus stable source
segment IDs for traceability. Sections use approximate time windows, chunks use
a token limit, and overlap keeps complete speaker turns. Candidate
configurations are compared with manually reviewed question-answer cases
before selecting a strategy.

See `docs/Problem/EPICS-AND-USER-STORIES.md` for the schemas, acceptance
criteria, and evaluation plan.

---

## Environment Variables

Create a `.env` file at the project root:

```
ANTHROPIC_API_KEY=...    # Claude API
HF_TOKEN=...             # HuggingFace — required for pyannote diarization
```

For `HF_TOKEN` you must also accept model terms at:
- huggingface.co/pyannote/speaker-diarization-3.1
- huggingface.co/pyannote/segmentation-3.0

---

## Directory Structure

```
input/       Place meeting.mp4 here before running
pipeline/    Analysis scripts and shared modules
output/      Generated JSON files
tests/       Pytest test suite
docs/        Design specs and project documentation
```

## Testing

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

Pull requests also run `.github/workflows/ci.yml`. The workflow uses `uv` and
runs only tests under `tests/qualification/`, including unit and future
qualification integration tests. It does not run the real model pipeline
behind steps `01` through `06`.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| WhisperX | Word-level transcription + forced alignment |
| pyannote.audio | Speaker diarization |
| librosa | Audio feature extraction (pitch, energy, speech rate, pauses, ZCR) |
| audeering wav2vec2 | Voice emotion: valence / arousal / dominance per segment |
| DeepFace | Facial emotion per frame (sampled every 10s) |
| Anthropic Claude API | LLM insights — transcript-only vs multimodal comparison |
| Chart.js (CDN) | Engagement timeline chart in HTML report |
| python-dotenv | `.env` loading |
| yt-dlp | Demo video download |
