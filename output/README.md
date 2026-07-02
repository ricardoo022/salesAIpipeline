# output/

Generated files from each pipeline step. All files here are derived outputs — safe to delete and regenerate by re-running `python run.py`.

## Files

| File | Produced by | Description |
|------|-------------|-------------|
| `transcript.json` | `01_transcribe.py` | Speaker-diarized transcript with word-level timestamps |
| `audio_features.json` | `02_audio_features.py` | Pitch mean/std, energy mean, speech rate, pause ratio, ZCR per diarized segment |
| `voice_emotion.json` | `03_emotion_voice.py` | Valence, arousal, dominance per segment (max 15s chunks) |
| `face_emotion.json` | `04_emotion_face.py` | Dominant emotion + scores per frame sampled every 10s |
| `analysis.json` | `05_llm_analysis.py` | LLM output — both `transcript_only` and `multimodal` analyses |
| `report.html` | `06_report.py` | Final self-contained report — open in any browser |
| `audio_temp.wav` | `01_transcribe.py` | Temporary 16kHz mono WAV file, overwritten each run |

## transcript.json field reference

`transcript.json` is a JSON array — one object per speech segment, in chronological order. Example segment:

```json
{
  "start": 0.251,
  "end": 0.771,
  "text": " How are you?",
  "words": [
    {"word": "How", "start": 0.251, "end": 0.391, "score": 0.603},
    {"word": "are", "start": 0.411, "end": 0.531, "score": 0.514},
    {"word": "you?", "start": 0.551, "end": 0.771, "score": 0.768}
  ],
  "avg_logprob": -0.184,
  "speaker": "SPEAKER_01"
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `start` | float (seconds) | When this segment starts in the audio |
| `end` | float (seconds) | When this segment ends |
| `text` | string | What was said, as transcribed by WhisperX (leading space is normal — Whisper's tokenizer includes it) |
| `words` | array | Word-level breakdown of `text`, from WhisperX's forced-alignment pass |
| `words[].word` | string | A single word (or punctuation-attached token) |
| `words[].start` / `words[].end` | float (seconds) | Start/end time of that specific word — finer-grained than the segment timestamps |
| `words[].score` | float 0–1 | Alignment confidence for that word. Lower (< 0.3) usually means background noise, overlapping speech, or a mumbled/cut-off word — not necessarily wrong, just less certain |
| `avg_logprob` | float, usually negative | Whisper's average log-probability for this segment's transcription. Closer to 0 = more confident; more negative (e.g. below -1) = the model was unsure, worth a manual sanity-check |
| `speaker` | string | Speaker label assigned by matching this segment against pyannote's diarization turns (`SPEAKER_00`, `SPEAKER_01`, ...). `UNKNOWN` if no diarized turn overlapped this segment's time range |

Note: `speaker` labels (`SPEAKER_00`, `SPEAKER_01`, ...) are arbitrary IDs assigned by the diarization model based on voice clustering — they don't map to "customer" vs "sales rep" automatically. You'd match that up by listening to a sample of each speaker's segments.

## audio_features.json field reference

`audio_features.json` is a JSON array — one object per speech segment, parallel to `transcript.json`. Example segment:

```json
{
  "speaker": "SPEAKER_01",
  "start": 0.251,
  "end": 0.771,
  "pitch_mean": 196.729,
  "pitch_std": 15.1103,
  "energy_mean": 0.0561,
  "speech_rate": 5.7692,
  "pause_ratio": 0.0769,
  "zcr": 0.088
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `speaker` | string | Speaker label from the transcript — matches the same segment in `transcript.json` |
| `start` | float (seconds) | When this segment starts — copy of the transcript value |
| `end` | float (seconds) | When this segment ends — copy of the transcript value |
| `pitch_mean` | float (Hz) | Average vocal pitch (fundamental frequency F0) of voiced frames in the segment. Low ~85–180 Hz (male), high ~165–255 Hz (female). Deviations from the speaker's baseline can signal stress or excitement |
| `pitch_std` | float (Hz) | Standard deviation of pitch — how much the voice varied. High = animated/emotional; low = monotone/disengaged |
| `energy_mean` | float (RMS) | Average volume (root-mean-square energy). High = emphasis or enthusiasm; low = tired or disengaged. Near-zero means silence |
| `speech_rate` | float (words/sec) | Words spoken per second. Fast = nervous or excited; slow = hesitant, thoughtful, or careful explanation. Computed from word timestamps, not from audio |
| `pause_ratio` | float 0–1 | Fraction of the segment spent in pauses between words. 0.2 = 20% of the time was silence. High = hesitation or strategic pausing; low = fluent or rushed. Computed from gaps between consecutive word timestamps |
| `zcr` | float | Zero crossing rate — how often the audio signal crosses zero. Higher values correlate with roughness/noise in the voice (fricatives, emotion); lower values with smooth periodic sounds (vowels) |

## voice_emotion.json field reference

`voice_emotion.json` is a JSON array — one object per speech segment, parallel to `transcript.json` and `audio_features.json`. Example segment:

```json
{
  "speaker": "SPEAKER_01",
  "start": 12.4,
  "end": 18.1,
  "valence": 0.6513,
  "arousal": 0.5250,
  "dominance": 0.5701
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `speaker` | string | Speaker label from the transcript — matches the same segment in `transcript.json` and `audio_features.json` |
| `start` / `end` | float (seconds) | When this segment starts/ends — copy of the transcript value |
| `valence` | float 0–1 | How positive (near 1) or negative (near 0) the voice sounded. Low valence = discomfort, skepticism, or dissatisfaction. High valence = enthusiasm, agreement, interest. A prospect who says "yes" with low valence is not truly convinced |
| `arousal` | float 0–1 | How calm (near 0) or excited (near 1) the voice sounded. Low arousal + low valence = disengagement (danger zone). High arousal + high valence = engagement and enthusiasm. A sudden arousal spike can signal a strong objection forming |
| `dominance` | float 0–1 | How dominant/assertive (near 1) or submissive/passive (near 0) the voice sounded. A dominance shift in the prospect = pushback or strong objection forming. High dominance from the rep = controlling the conversation |

> **Model:** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` — a wav2vec2-large model fine-tuned on the MSP-Dim speech emotion dataset. The head is a **regression** head (not a classifier), so values are continuous VAD dimensions directly — no sigmoid applied. The model returns `[arousal, dominance, valence]` per its `config.json`; the output dict reorders to `{valence, arousal, dominance}`. See `docs/steps/step3-walkthrough.md` for the full implementation story, including the bugs found via statistical validation.

## face_emotion.json field reference

`face_emotion.json` is a JSON array — one object per sampled video frame (every 10 seconds), independent of the per-segment structure of `transcript.json`/`audio_features.json`/`voice_emotion.json`. Example frame:

```json
{
  "timestamp": 100.0,
  "dominant_emotion": "neutral",
  "scores": {
    "angry": 0.0035,
    "disgust": 0.0,
    "fear": 0.0037,
    "happy": 0.205,
    "sad": 0.0279,
    "surprise": 0.0,
    "neutral": 0.7598
  }
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `timestamp` | float (seconds) | When in the video this frame was sampled. One frame every 10s (`0.0, 10.0, 20.0, …`), rounded to 2 decimals. Independent of the diarized segments — it samples the raw video timeline |
| `dominant_emotion` | string | The emotion with the highest `score` — one of `angry, disgust, fear, happy, sad, surprise, neutral`. Quick at-a-glance reading of the facial state at that moment |
| `scores` | object | All 7 emotion probabilities, each a float 0–1 (normalized from DeepFace's 0–100 percentages, rounded to 4 decimals). Sums to ~1.0. Gives the step-5 LLM granularity beyond just the dominant label |

Frames where DeepFace detects no face are **skipped** (not included in the array) — the pipeline never crashes on a face-less frame. On the demo video (~15 min) this yields ~90 frames.

> **Model:** DeepFace with `detector_backend="retinaface"` and `actions=["emotion"]`. The default `opencv` detector backend is broken under `opencv-python` 5.x (haarcascade XMLs missing from `cv2/data/`), so retinaface is used instead. DeepFace returns scores as 0–100 percentages; the module normalizes to 0–1. See `docs/steps/step4-walkthrough.md` for the full implementation story, including the bugs found running on the real video.

## Re-running from a specific step

Delete the output file for the step you want to re-run. `run.py` skips steps whose output already exists, so only the deleted step and everything after it will re-execute:

```bash
rm output/analysis.json && python run.py   # re-runs steps 5 and 6
rm output/voice_emotion.json && python run.py  # re-runs steps 3 through 6
```

This directory is gitignored.
