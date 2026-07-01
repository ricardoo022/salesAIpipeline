# output/

Generated files from each pipeline step. All files here are derived outputs — safe to delete and regenerate by re-running `python run.py`.

## Files

| File | Produced by | Description |
|------|-------------|-------------|
| `transcript.json` | `01_transcribe.py` | Speaker-diarized transcript with word-level timestamps |
| `audio_features.json` | `02_audio_features.py` | Pitch, energy, speech rate, pause ratio, ZCR per segment |
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

## Re-running from a specific step

Delete the output file for the step you want to re-run. `run.py` skips steps whose output already exists, so only the deleted step and everything after it will re-execute:

```bash
rm output/analysis.json && python run.py   # re-runs steps 5 and 6
rm output/voice_emotion.json && python run.py  # re-runs steps 3 through 6
```

This directory is gitignored.
