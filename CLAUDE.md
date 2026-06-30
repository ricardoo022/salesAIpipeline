# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Multimodal sales meeting analysis pipeline. Takes a B2B sales meeting video and produces an HTML report with engagement scores, deal probability, talk ratio, emotional timeline, critical moments, and coaching recommendations. The killer feature is a side-by-side comparison between transcript-only analysis vs full multimodal analysis — this is a demo aimed at Portuguese B2B sales consulting company Scale Labs.

## Pipeline Architecture

Six sequential steps, each a standalone Python script. Output of each step is a JSON file consumed by the next:

```
input/meeting.mp4
  → pipeline/01_transcribe.py      → output/transcript.json      (WhisperX large-v2 ✓ | pyannote diarization TODO)
  → pipeline/02_audio_features.py  → output/audio_features.json  (librosa: pitch, energy, speech rate, pauses)
  → pipeline/03_emotion_voice.py   → output/voice_emotion.json   (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)
  → pipeline/04_emotion_face.py    → output/face_emotion.json    (DeepFace, sampled every 10s)
  → pipeline/05_llm_analysis.py    → output/analysis.json        (Claude API, run twice: transcript-only + multimodal)
  → pipeline/06_report.py          → output/report.html           (pure HTML + Chart.js via CDN)
```

`run.py` at root orchestrates all six steps in sequence, skipping any step whose output file already exists.

### Module/script split

The numbered scripts (`01_transcribe.py` etc.) are CLI entry points — they handle file paths, `sys.exit`, and print progress. Shared logic lives in importable modules alongside them:

- `pipeline/audio.py` — `extract_audio()` — ffmpeg wrapper
- `pipeline/transcribe.py` — `transcribe_audio()` — WhisperX load + align

Tests import the modules directly with mocks; they never call the numbered scripts (except the one subprocess test for the missing-video guard). New logic for each step should follow this pattern.

## Project Structure

```
├── .claude/          Agent skills and configurations
├── docs/             Design specs and implementation plans
├── input/            Place meeting.mp4 here
├── models/           Model weights cache (auto-downloaded)
├── output/           Generated JSON files and report.html
├── pipeline/         The six analysis scripts
├── tests/            Pytest test suite
├── run.py            Orchestrator
└── requirements.txt  Python dependencies
```

## Testing

```bash
source venv/bin/activate
python -m pytest tests/ -v                      # all tests
python -m pytest tests/test_transcribe.py -v    # single file
python -m pytest tests/ -v -k "test_loads"      # single test by name
```

11 tests: `pipeline/audio.py` (extract_audio, 6 tests), `pipeline/transcribe.py` (transcribe_audio, 4 tests), `pipeline/01_transcribe.py` subprocess guard (1 test).

## Key Design Decisions

**Voice emotion model**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` — outputs valence, arousal, dominance (continuous 0–1) per segment. Better than training on RAVDESS for naturalistic speech.

**Segment granularity**: Speaker turns from diarization, capped at 15 seconds. Longer turns split into 15s chunks. This gives natural speech units at consistent granularity for the emotion model.

**LLM analysis runs twice**: Step 5 calls Claude API with two different prompts — once with text only, once with all modalities — and saves both outputs. The report renders them side-by-side.

**No GPU assumed**: Pipeline runs on CPU (WSL Ubuntu). WhisperX and the audeering model will be slow on long videos.

**Model cache**: The audeering wav2vec2 model is downloaded once and cached in `models/` at project root.

## Commands

```bash
# Setup (first time)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download demo video
yt-dlp https://youtu.be/N0SF2nZS-S8 -o input/meeting.mp4

# Run tests
python -m pytest tests/ -v

# Run full pipeline
python run.py

# Re-run from a specific step (delete its output file)
rm output/analysis.json && python run.py   # re-runs steps 5 and 6 only

# Run individual steps (for debugging)
python pipeline/01_transcribe.py
python pipeline/02_audio_features.py
python pipeline/03_emotion_voice.py
python pipeline/04_emotion_face.py
python pipeline/05_llm_analysis.py
python pipeline/06_report.py

# View report
# Open output/report.html in browser
```

## Environment Variables Required

Store in `.env` at the project root — every script that needs a key loads it via `python-dotenv`.

```
ANTHROPIC_API_KEY=...       # Claude API key for step 5
HF_TOKEN=...                # HuggingFace token — required for pyannote diarization (step 1, TODO)
                            # Audio extraction and WhisperX work without it.
                            # Must also accept model terms at:
                            # huggingface.co/pyannote/speaker-diarization-3.1
                            # huggingface.co/pyannote/segmentation-3.0
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| WhisperX | Transcription + speaker diarization (who said what, when) |
| librosa | Audio feature extraction (pitch, energy, speech rate, pauses, ZCR) |
| transformers (audeering model) | Voice emotion: valence / arousal / dominance per segment |
| DeepFace | Facial emotion per frame (sampled every 10s) |
| Anthropic Claude API | LLM insights — model: claude-sonnet-4-6 |
| Chart.js (CDN) | Engagement timeline chart in HTML report |

## Output JSON Schemas

**transcript.json**: Array of `{speaker, start, end, text, words[]}` — word-level timestamps from WhisperX. `speaker` field is absent until pyannote diarization is implemented.

**audio_features.json**: Array of `{speaker, start, end, pitch_mean, pitch_std, energy_mean, speech_rate, pause_ratio, zcr}` — one entry per diarized segment.

**voice_emotion.json**: Array of `{speaker, start, end, valence, arousal, dominance}` — one entry per segment (max 15s chunks).

**face_emotion.json**: Array of `{timestamp, dominant_emotion, scores{}}` — one entry per 10s sample. Frames with no detected face are skipped (no crash).

**analysis.json**: `{transcript_only: {...}, multimodal: {...}}` — both LLM outputs with engagement_score, deal_probability, critical_moments[], coaching_recommendations[], talk_ratio.
