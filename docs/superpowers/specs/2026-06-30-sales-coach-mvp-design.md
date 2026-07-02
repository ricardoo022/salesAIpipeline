# Sales Coach MVP — Design Spec

**Date:** 2026-06-30
**Goal:** Multimodal sales meeting analysis pipeline that produces a demo HTML report for outreach to Scale Labs (Portuguese B2B sales consulting company).

---

## Purpose

Scale Labs has a product that takes a meeting transcript and feeds it into an LLM. This MVP demonstrates what that approach misses by doing audio + video + transcript → multimodal analysis → specific insights with timestamps. The report is the pitch artefact. It needs to make a sales consulting company look at it and think "our clients need this."

---

## Architecture

Flat Python scripts, each reads JSON and writes JSON. No framework, no shared state object. `run.py` orchestrates sequentially, skipping steps whose output already exists.

```
input/
  meeting.mp4           [x] downloaded

pipeline/
  01_transcribe.py      → output/transcript.json   [x] complete (WhisperX + pyannote + speaker merge)
  02_audio_features.py  → output/audio_features.json   [x] complete (librosa: pitch, energy, speech rate, pauses, ZCR)
  03_emotion_voice.py   → output/voice_emotion.json   [x] complete (audeering wav2vec2: valence, arousal, dominance per segment)
  04_emotion_face.py    → output/face_emotion.json   [x] complete (DeepFace retinaface, every 10s, scores normalized 0–1)
  05_llm_analysis.py    → output/analysis.json
  06_report.py          → output/report.html

output/                                       [x] created, gitignored
models/
run.py                                        [x] orchestrator working
requirements.txt                              [x] all deps installed in venv
.env                                          [x] .env.example created
tests/                                        [x] pytest suite (68 tests)
```

Force re-run from a specific step by deleting its output file:
```bash
rm output/analysis.json && python run.py  # re-runs steps 5 and 6 only
```

---

## Pipeline Steps

### Step 1 — Transcription + Diarization (`01_transcribe.py`)

- [x] Extract audio from `input/meeting.mp4` via ffmpeg
- [x] Run WhisperX (`large-v2`) for word-level transcription
- [x] Run pyannote `speaker-diarization-3.1` for speaker labels
- [x] Merge word timestamps with speaker labels

**Output schema:**
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 12.4,
    "end": 18.1,
    "text": "I'm not sure the pricing makes sense for us.",
    "words": [{"word": "I'm", "start": 12.4, "end": 12.6}, ...]
  }
]
```

**Failure mode:** Missing `HF_TOKEN` → clear message pointing to `.env`. Requires manual acceptance of pyannote model terms at huggingface.co/pyannote/speaker-diarization-3.1 and huggingface.co/pyannote/segmentation-3.0.

---

### Step 2 — Audio Features (`02_audio_features.py`)

- [x] Load audio with librosa
- [x] For each diarized segment: extract pitch mean/std (librosa.pyin), energy mean (librosa.feature.rms), speech rate (words/sec from word timestamps), pause ratio (word-gap based), zero crossing rate

**Output schema:**
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 12.4,
    "end": 18.1,
    "pitch_mean": 182.3,
    "pitch_std": 24.1,
    "energy_mean": 0.042,
    "speech_rate": 3.2,
    "pause_ratio": 0.18,
    "zcr": 0.061
  }
]
```

---

### Step 3 — Voice Emotion (`03_emotion_voice.py`)

- [x] Model: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- [x] Outputs: valence, arousal, dominance (continuous 0–1) per segment
- [x] Segments longer than 15 seconds are split into ≤15s chunks; chunk results averaged back to segment level
- [x] Model downloaded once and cached in `models/`

**Output schema:**
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 12.4,
    "end": 18.1,
    "valence": 0.31,
    "arousal": 0.22,
    "dominance": 0.41
  }
]
```

**Sales interpretation:**
- Low arousal + low valence = flat, disengaged — danger zone
- High arousal + high valence = excited, engaged
- High dominance shift in prospect = pushback or strong objection forming

---

### Step 4 — Facial Emotion (`04_emotion_face.py`)

- [x] Sample one frame every 10 seconds from `meeting.mp4` via OpenCV
- [x] Run DeepFace on each frame
- [x] If no face detected: skip frame, log warning, continue (no crash)

**Output schema:**
```json
[
  {
    "timestamp": 10.0,
    "dominant_emotion": "neutral",
    "scores": {
      "happy": 0.03,
      "sad": 0.12,
      "angry": 0.04,
      "neutral": 0.71,
      "fear": 0.02,
      "disgust": 0.05,
      "surprise": 0.03
    }
  }
]
```

Report handles missing face data gracefully: shows "facial data unavailable for N frames" rather than crashing.

---

### Step 5 — LLM Analysis (`05_llm_analysis.py`)

- Merges all four JSONs into a structured prompt (see prompt design below)
- Calls Claude API **twice**: transcript-only mode, then multimodal mode
- Model: `claude-sonnet-4-6`
- On rate limit: retry once after 10 seconds
- Missing `ANTHROPIC_API_KEY` → fail immediately with clear message

**Prompt structure (multimodal mode):**
```
SEGMENT [00:12:24]
Speaker: PROSPECT
Text: "I'm not sure the pricing makes sense for us right now."
Audio: pitch_std=high (vocal stress), energy=low, speech_rate=slow, pause_ratio=0.31
Voice emotion: valence=0.31 (negative), arousal=0.22 (low energy), dominance=0.41
Facial: neutral (0.71 confidence), slight sad (0.12)
```

Transcript-only mode sends only the `Text` field. Both prompts request the same output structure.

**Output schema:**
```json
{
  "transcript_only": {
    "engagement_score": 61,
    "deal_probability": 45,
    "talk_ratio": {"rep": 62, "prospect": 38},
    "critical_moments": [
      {
        "timestamp": "00:12:24",
        "type": "pricing_objection",
        "description": "Prospect questioned pricing",
        "coaching": "Rep should have..."
      }
    ],
    "recommendations": ["...", "..."]
  },
  "multimodal": {
    "engagement_score": 74,
    "deal_probability": 68,
    "talk_ratio": {"rep": 62, "prospect": 38},
    "critical_moments": [...],
    "recommendations": ["...", "..."]
  }
}
```

---

### Step 6 — Report (`06_report.py`)

Generates a single self-contained `output/report.html`. All data embedded as a JavaScript object in a `<script>` tag. Opens by double-clicking — no server needed.

---

## Report Layout

Dark theme, single page, narrative scroll top to bottom.

**1. Header bar**
Meeting title, date, duration, speaker labels (REP / PROSPECT).

**2. Hero metrics row**
Four cards: Engagement Score, Deal Probability, Talk Ratio, Duration. Multimodal numbers. Large font, instant read.

**3. Side-by-side comparison** ← killer feature
Two columns, full width:
- Top: metric cards with transcript-only vs multimodal values + delta indicator (e.g. `+13 pts`, `+23%`)
- Below: first 3–4 items from `recommendations[]` per column rendered as insight cards. Left (transcript-only) = thin, generic. Right (multimodal) = specific with timestamps and signal references. Visual weight deliberately heavier on the right.

**4. Engagement Timeline**
Chart.js line chart, three lines:
- Prospect Valence (blue) — from `voice_emotion.json` where speaker = PROSPECT
- Prospect Arousal (orange) — from `voice_emotion.json` where speaker = PROSPECT
- Rep Arousal (grey) — from `voice_emotion.json` where speaker = REP (labelled "Rep Energy" in UI)

Critical moment markers as vertical dotted lines with labels. Clicking a label scrolls to that moment's detail card.

**5. Critical Moments list**
Cards below the chart: `[timestamp]` | moment type | what happened | coaching note. Clicking a timestamp highlights the chart marker.

**6. Coaching Recommendations**
Numbered list. Specific and actionable — no generic advice.

**7. Footer**
`Powered by multimodal analysis: WhisperX · audeering · DeepFace · Claude`

---

## Orchestration

`run.py` per-step logic:
1. Check if output file exists → if yes, print `✓ Step N already done, skipping` → continue
2. Print `→ Running step N...`
3. Run step → on failure: print step name + error + which file to delete to retry → `exit(1)`

Error handling:
- Step 1: missing `HF_TOKEN` → named error
- Step 4: no face in frame → skip, log, continue
- Step 5: missing API key → exit immediately; rate limit → retry once after 10s
- All steps: unhandled exceptions print step name + traceback, exit code 1

`.env` loaded via `python-dotenv` at top of every script that needs a key.

---

## Tech Stack

| Tool | Version target | Purpose |
|------|---------------|---------|
| WhisperX | latest | Transcription + diarization |
| pyannote.audio | 3.x | Speaker diarization |
| librosa | 0.10+ | Audio feature extraction |
| transformers | 4.x | audeering emotion model |
| torch | 2.x (CPU) | Model inference |
| DeepFace | latest | Facial emotion |
| opencv-python | 4.x | Frame sampling |
| anthropic | latest | Claude API |
| python-dotenv | latest | Env var loading |
| Chart.js | 4.x (CDN) | Timeline chart |
| yt-dlp | latest | Demo video download |

---

## Environment Variables

```
ANTHROPIC_API_KEY=...    # Claude API (step 5)
HF_TOKEN=...             # HuggingFace (step 1, pyannote)
```

---

## Success Criteria

`output/report.html` opens in a browser and:
1. A non-technical person understands the meeting outcome in under 10 seconds (hero metrics)
2. The side-by-side comparison makes transcript-only analysis look visually thin
3. At least one critical moment has a timestamp that matches something real in the video
4. The engagement timeline shows a meaningful signal (not a flat line)
