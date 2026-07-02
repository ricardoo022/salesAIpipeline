# pipeline/

The analysis scripts that form the pipeline. Each is a standalone Python script — no shared state, no imports between scripts. Data flows exclusively through JSON files in `output/`.

## Workflow

```
input/meeting.mp4
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  01_transcribe.py                                           │
│                                                             │
│  ┌─────────────┐     16kHz mono WAV     ┌───────────────┐  │
│  │  audio.py   │ ───────────────────── ▶│ transcribe.py │  │
│  │ (ffmpeg)    │                         │  (WhisperX)   │  │
│  └─────────────┘          │              └───────┬───────┘  │
│                           │                      │           │
│                           │              word-level segments │
│                           │              (what + when)       │
│                           │                      │           │
│                           ▼              ┌───────▼───────┐  │
│                    ┌─────────────┐       │  transcribe.py│  │
│                    │  diarize.py │       │  merge_speaker│  │
│                    │ (pyannote)  │──────▶│  _labels()    │  │
│                    └─────────────┘       └───────┬───────┘  │
│                    who + when                     │           │
│                                                   │           │
└───────────────────────────────────────────────────┼──────────┘
                                                    │
                                                    ▼
                                        output/transcript.json
                          [{speaker, start, end, text, words[]}]
```

WhisperX and pyannote run **in parallel on the same WAV** — WhisperX answers *what was said* (with word-level timestamps), pyannote answers *who was speaking* (and when). `merge_speaker_labels()` joins them by maximum time overlap to produce a single enriched transcript.

## Scripts

## Modules

### `audio.py`
`extract_audio(video_path, output_path=None)` — ffmpeg wrapper. Extracts 16kHz mono WAV from a video file. Returns the output path.

### `transcribe.py`
- `transcribe_audio(audio_path)` — loads WhisperX large-v2 on CPU (int8), transcribes and aligns. Returns aligned segments.
- `merge_speaker_labels(segments, diarization)` — assigns speaker labels to transcript segments by maximum time overlap. Unmatched segments get `speaker: "UNKNOWN"`.

### `features.py`
`extract_audio_features(transcript, audio_path)` — loads audio with `librosa.load`, iterates each diarized segment and computes pitch (librosa.pyin), energy (librosa.feature.rms), speech rate (word count / duration), pause ratio (inter-word gaps / duration), and zero crossing rate. Edge cases handled: empty segments, all-unvoiced frames, overlapping words, missing words array.

### `diarize.py`
`diarize_audio(audio_path, hf_token)` — runs pyannote `speaker-diarization-3.1` on the audio file. Answers *who spoke when*, not *what* was said. Returns a list of turns `[{speaker, start, end}]` with generic labels (`SPEAKER_00`, `SPEAKER_01`, …). The `hf_token` (HuggingFace access token) is required — pyannote's model is gated and must be accepted at huggingface.co/pyannote/speaker-diarization-3.1 before use.

> **Note:** `from pyannote.audio import Pipeline` is imported *inside* the function body, not at module level. pyannote loads PyTorch on import and crashes in CPU-only environments without CUDA libs. The deferred import lets the module load cleanly and fail early on missing inputs before touching pyannote.

### `emotion_voice.py`
`extract_voice_emotion(segments, audio_path, model_path=None)` — loads the audeering `wav2vec2-large-robust-12-ft-emotion-msp-dim` model and runs it per segment to produce valence, arousal, dominance (continuous 0–1). Segments >15s are split into ≤15s chunks; chunk predictions are averaged. Empty segments default to 0.0.

> **Architecture caveat:** The model's `Wav2Vec2ForSpeechClassification` head class was **removed from modern `transformers`** (4.57.6). Loading with stock `Wav2Vec2ForSequenceClassification` silently random-initializes the head (wrong weight names) and produces a flat-line output. The module reconstructs `Wav2Vec2ForSpeechClassification` + `Wav2Vec2ClassificationHead` as custom classes matching the saved weight names (`classifier.dense`, `classifier.out_proj`) so `from_pretrained` loads the real trained weights.
>
> **Post-processing caveat:** This is a **regression head** (`problem_type: "regression"`), not a classifier. Do **not** apply `torch.sigmoid()` — outputs are VAD dimensions directly. Clip to [0,1] with `np.clip`. Applying sigmoid compresses all variation to a narrow band around 0.5, defeating the model.
>
> **Output order caveat:** Model returns `[arousal, dominance, valence]` per `config.json` (`id2label: {0: arousal, 1: dominance, 2: valence}`). The output dict reorders to `{valence, arousal, dominance}` via `VALENCE_IDX=2`, `AROUSAL_IDX=0`, `DOMINANCE_IDX=1`. Getting this wrong silently mislabels every field.
