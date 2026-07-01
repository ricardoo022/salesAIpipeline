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

### `01_transcribe.py` → `output/transcript.json`
Extracts audio from `input/meeting.mp4` via ffmpeg (16kHz mono WAV), runs WhisperX (`large-v2`) for word-level transcription and alignment via `pipeline/transcribe.py`, then runs pyannote `speaker-diarization-3.1` via `pipeline/diarize.py`. Speaker labels are merged into segments by maximum time overlap (`merge_speaker_labels`).

Requires `HF_TOKEN` in `.env` (script exits early with a clear error if missing) and accepted model terms on HuggingFace:
- huggingface.co/pyannote/speaker-diarization-3.1
- huggingface.co/pyannote/segmentation-3.0

Output schema: `[{speaker, start, end, text, words: [{word, start, end}]}]`

## Modules

### `audio.py`
`extract_audio(video_path, output_path=None)` — ffmpeg wrapper. Extracts 16kHz mono WAV from a video file. Returns the output path.

### `transcribe.py`
- `transcribe_audio(audio_path)` — loads WhisperX large-v2 on CPU (int8), transcribes and aligns. Returns aligned segments.
- `merge_speaker_labels(segments, diarization)` — assigns speaker labels to transcript segments by maximum time overlap. Unmatched segments get `speaker: "UNKNOWN"`.

### `diarize.py`
`diarize_audio(audio_path, hf_token)` — runs pyannote `speaker-diarization-3.1` on the audio file. Answers *who spoke when*, not *what* was said. Returns a list of turns `[{speaker, start, end}]` with generic labels (`SPEAKER_00`, `SPEAKER_01`, …). The `hf_token` (HuggingFace access token) is required — pyannote's model is gated and must be accepted at huggingface.co/pyannote/speaker-diarization-3.1 before use.

> **Note:** `from pyannote.audio import Pipeline` is imported *inside* the function body, not at module level. pyannote loads PyTorch on import and crashes in CPU-only environments without CUDA libs. The deferred import lets the module load cleanly and fail early on missing inputs before touching pyannote.
