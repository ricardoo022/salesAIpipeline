# pipeline/

The six analysis scripts that form the pipeline. Each is a standalone Python script — no shared state, no imports between scripts. Data flows exclusively through JSON files in `output/`.

## Scripts

### `01_transcribe.py` → `output/transcript.json`
Extracts audio from `input/meeting.mp4` via ffmpeg (16kHz mono WAV), runs WhisperX (`large-v2`) for word-level transcription and alignment via `pipeline/transcribe.py`, then runs pyannote `speaker-diarization-3.1` via `pipeline/diarize.py`. Speaker labels are merged into segments by maximum time overlap (`merge_speaker_labels`).

Requires `HF_TOKEN` in `.env` (script exits early with a clear error if missing) and accepted model terms on HuggingFace:
- huggingface.co/pyannote/speaker-diarization-3.1
- huggingface.co/pyannote/segmentation-3.0

Output schema: `[{speaker, start, end, text, words: [{word, start, end}]}]`

---

### `02_audio_features.py` → `output/audio_features.json`
Loads audio with librosa. For each diarized segment from `transcript.json`: extracts pitch mean/std, energy mean, speech rate (words/sec from word timestamps), pause ratio, and zero crossing rate.

Output schema: `[{speaker, start, end, pitch_mean, pitch_std, energy_mean, speech_rate, pause_ratio, zcr}]`

---

### `03_emotion_voice.py` → `output/voice_emotion.json`
Runs `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` on each segment. Segments longer than 15s are split into chunks and their results averaged. Model cached in `models/`.

Output schema: `[{speaker, start, end, valence, arousal, dominance}]` — all values 0–1.

Sales signal guide:
- Low arousal + low valence = flat, disengaged
- High arousal + high valence = excited, engaged
- High dominance shift in prospect = pushback forming

---

### `04_emotion_face.py` → `output/face_emotion.json`
Samples one frame every 10 seconds from `input/meeting.mp4` via OpenCV, runs DeepFace on each. Frames where no face is detected are silently skipped (logged, no crash).

Output schema: `[{timestamp, dominant_emotion, scores: {happy, sad, angry, neutral, fear, disgust, surprise}}]`

---

### `05_llm_analysis.py` → `output/analysis.json`
Merges all four JSONs into a structured prompt and calls the Claude API **twice**: once with text only, once with all modalities. Uses `claude-sonnet-4-6`. Retries once after 10s on rate limit.

Prompt format per segment (multimodal mode):
```
SEGMENT [HH:MM:SS]
Speaker: PROSPECT
Text: "..."
Audio: pitch_std=high, energy=low, speech_rate=slow, pause_ratio=0.31
Voice emotion: valence=0.31, arousal=0.22, dominance=0.41
Facial: neutral (0.71), slight sad (0.12)
```

Output schema:
```json
{
  "transcript_only": {
    "engagement_score": 61,
    "deal_probability": 45,
    "talk_ratio": {"rep": 62, "prospect": 38},
    "critical_moments": [{timestamp, type, description, coaching}],
    "recommendations": ["..."]
  },
  "multimodal": { ...same structure... }
}
```

---

### `06_report.py` → `output/report.html`
Generates a self-contained HTML file. All data is embedded as a JavaScript object in a `<script>` tag — no server required, opens by double-clicking.

Report sections:
1. Header (meeting metadata)
2. Hero metrics (engagement score, deal probability, talk ratio, duration)
3. Side-by-side comparison — transcript-only vs multimodal, with delta indicators
4. Engagement timeline — Chart.js line chart (prospect valence, prospect arousal, rep arousal) with critical moment markers
5. Critical moments cards with timestamps
6. Coaching recommendations
