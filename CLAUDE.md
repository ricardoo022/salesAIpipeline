# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Multimodal sales meeting analysis pipeline. Takes a B2B sales meeting video and produces an HTML report with engagement scores, deal probability, talk ratio, emotional timeline, critical moments, and coaching recommendations. The killer feature is a side-by-side comparison between transcript-only analysis vs full multimodal analysis — this is a demo aimed at Portuguese B2B sales consulting company Scale Labs.

## Pipeline Architecture

Six sequential steps, each a standalone Python script. Output of each step is a JSON file consumed by the next:

```
input/meeting.mp4
  → pipeline/01_transcribe.py      → output/transcript.json      (WhisperX large-v2 ✓ | pyannote diarization ✓ | merge speaker labels ✓)
  → pipeline/02_audio_features.py  → output/audio_features.json  (librosa: pitch mean/std, energy mean, speech rate, pause ratio, ZCR)
  → pipeline/03_emotion_voice.py   → output/voice_emotion.json   (audeering wav2vec2: valence, arousal, dominance per segment ✓)
  → pipeline/04_emotion_face.py    → output/face_emotion.json    (DeepFace retinaface, every 10s, scores normalized 0–1 ✓)
  → pipeline/05_llm_analysis.py    → output/analysis.json        (Claude API, run twice: transcript-only + multimodal ✓)
  → pipeline/06_report.py          → output/report.html           (pure HTML + Chart.js via CDN)
```

`run.py` at root orchestrates all six steps in sequence, skipping any step whose output file already exists.

### Module/script split

The numbered scripts (`01_transcribe.py` etc.) are CLI entry points — they handle file paths, `sys.exit`, and print progress. Shared logic lives in importable modules alongside them:

- `pipeline/audio.py` — `extract_audio()` — ffmpeg wrapper
- `pipeline/transcribe.py` — `transcribe_audio()`, `merge_speaker_labels()` — WhisperX load + align; speaker label assignment by max time overlap
- `pipeline/diarize.py` — `diarize_audio()` — pyannote speaker-diarization-3.1 wrapper
- `pipeline/features.py` — `extract_audio_features()` — librosa pitch, energy, speech rate, pause ratio, ZCR per segment
- `pipeline/emotion_voice.py` — `extract_voice_emotion()` — audeering wav2vec2 VAD per segment; reconstructs `Wav2Vec2ForSpeechClassification` head (class removed from modern transformers)
- `pipeline/emotion_face.py` — `extract_face_emotion()` — DeepFace emotion per sampled frame (every 10s); retinaface detector, scores normalized 0–1, no-face frames skipped
- `pipeline/llm_analysis.py` — `run_analysis()` — merges the 4 upstream JSONs, calls Claude twice (transcript-only + multimodal) with forced tool-use; deterministic talk_ratio; lazy-imports anthropic

Tests import the modules directly with mocks; they never call the numbered scripts (except the subprocess tests for the CLI guards). New logic for each step should follow this pattern.

### Import path duality

Scripts use bare imports (`from audio import extract_audio`) because Python adds the script's own directory to `sys.path` when run directly. Tests use package imports (`from pipeline.audio import extract_audio`) because pytest runs from the project root and `pipeline/__init__.py` makes it a package. Both work; don't add `sys.path` manipulation to either.

### pyannote lazy import

`pipeline/diarize.py` does `from pyannote.audio import Pipeline` **inside** `diarize_audio()`, not at module level. This is intentional: pyannote imports torch at module load time and crashes in CPU-only environments that lack CUDA libs. The deferred import lets the CLI scripts load cleanly and fail early on missing video/token before touching pyannote.

Tests stub pyannote by injecting into `sys.modules` before the module is first imported:
```python
_mock_pyannote = MagicMock()
sys.modules.setdefault("pyannote", _mock_pyannote)
sys.modules.setdefault("pyannote.audio", _mock_pyannote)
```

### audeering head reconstruction (Step 3)

`pipeline/emotion_voice.py` reconstructs `Wav2Vec2ForSpeechClassification` + `Wav2Vec2ClassificationHead` as custom classes. The audeering model ships this head architecture (mean-pool hidden states → 2-layer MLP → regression), but the class was **removed from modern transformers** (4.57.6). Loading with stock `Wav2Vec2ForSequenceClassification` silently random-initializes the head (wrong weight names) and produces a flat-line output. The custom class matches the saved weight names (`classifier.dense`, `classifier.out_proj`) so `from_pretrained` loads the real trained weights.

The model is a **regression head** (`problem_type: "regression"`), not a classifier. Do **not** apply `torch.sigmoid()` — outputs are VAD dimensions directly. Clip to [0,1] with `np.clip` instead.

Output order is `{0: arousal, 1: dominance, 2: valence}` per `config.json`. The output dict reorders to `{valence, arousal, dominance}` via `VALENCE_IDX=2`, `AROUSAL_IDX=0`, `DOMINANCE_IDX=1`. Getting this wrong silently mislabels every field.

### DeepFace detector backend + score normalization (Step 4)

`pipeline/emotion_face.py` sets `DETECTOR_BACKEND = "retinaface"` explicitly. DeepFace's default `opencv` detector backend needs the haarcascade XMLs in `cv2/data/`, but **`opencv-python` 5.x ships that directory empty** (only `__init__.py`), so the default backend raises `ValueError` on every frame — silently swallowed by the `except ValueError` (no-face handler) as "no face", producing 0 frames. retinaface is a deepface dependency and ships its own weights (auto-downloaded to `~/.deepface/weights/`).

DeepFace returns emotion scores as **0–100 percentages**, not 0–1. `_shape_emotion_result` divides by 100 (then rounds to 4 decimals) to match the spec schema and step 3's VAD range. Getting this wrong feeds 0–100 values into the step-5 LLM prompt (written for 0–1).

`cv2` and `deepface` are lazy-imported inside `_iter_frames`/`_analyze_frame` (same pattern as pyannote in `diarize.py`), so the module loads without TensorFlow and the 13 unit tests run dep-free (patching the seams / injecting a `sys.modules` fake).

### Claude tool-use + truncation guard (Step 5)

`pipeline/llm_analysis.py` forces structured output via Claude **tool-use** (`tool_choice={"type": "tool", "name": "submit_analysis"}`) rather than parsing free text — the model is constrained to emit JSON matching `ANALYSIS_TOOL.input_schema`, so the analysis is always well-formed. `anthropic` is lazy-imported inside `_call_claude()` (same pattern as pyannote/cv2-deepface) so the unit tests run without the SDK, injecting a fake into `sys.modules["anthropic"]`.

**MAX_TOKENS truncation gotcha:** with `max_tokens=4096`, the multimodal call surfaced 13–15 detailed `critical_moments` that consumed the entire budget before `recommendations` was emitted — the tool_use block closed `critical_moments` and the partial JSON parsed into a *valid-but-incomplete* dict (missing `recommendations`), which silently flowed into `analysis.json`. Forced `tool_choice` guarantees a tool_use *block*, not a *complete* one. The guard is three-layered: `MAX_TOKENS=8192`, a `stop_reason == "max_tokens"` check, and `_validate_analysis()` asserting all required fields present. Getting this wrong ships a broken multimodal block that step 6 renders as if complete.

**Deterministic talk_ratio:** `talk_ratio` is computed from transcript talk-time (`_compute_talk_ratio`), not LLM-generated — LLMs are unreliable at exact arithmetic. It's injected into both `transcript_only` and `multimodal` outputs, so the ratio is identical and correct across modes (the tool schema deliberately omits it).

**Speaker → REP/PROSPECT:** diarization yields `SPEAKER_00/01/02`; `_classify_speakers()` maps by talk time (longest = REP, second = PROSPECT, rest = OTHER) — a deterministic, testable default, not an extra API call.

## Project Structure

```
├── .claude/          Agent skills and configurations
├── docs/             Design specs and implementation plans
├── input/            Place meeting.mp4 here
├── models/           Model weights cache (auto-downloaded)
├── output/           Generated JSON files and report.html
├── pipeline/         The six analysis scripts + shared modules
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

133 tests: `pipeline/audio.py` (extract_audio, 6 tests), `pipeline/transcribe.py` (transcribe_audio, 4 tests; merge_speaker_labels, 4 tests), `pipeline/diarize.py` (diarize_audio, 4 tests), `pipeline/features.py` (extract_audio_features — 17 tests: 14 mocked + 3 integration with real WAV), `pipeline/emotion_voice.py` (extract_voice_emotion — 10 tests: 9 mocked + 1 integration with real model, skip-guarded if cache missing), `pipeline/emotion_face.py` (extract_face_emotion — 16 tests: 13 mocked/pure unit + 2 cv2 integration + 1 skip-guarded full DeepFace integration), `pipeline/01_transcribe.py` subprocess guards (2 tests: missing video, missing HF_TOKEN), `pipeline/02_audio_features.py` subprocess guards (2 tests: missing transcript, missing audio), `pipeline/03_emotion_voice.py` subprocess guards (2 tests: missing segments, missing audio), `pipeline/04_emotion_face.py` subprocess guard (1 test: missing video), `pipeline/llm_analysis.py` (run_analysis — 63 tests: 35 pure unit + 8 mocked-seam with a fake `anthropic` injected into `sys.modules` + 5 orchestration + 14 real-data tests on the actual `output/*.json` + 1 skip-guarded live Claude integration), `pipeline/05_llm_analysis.py` subprocess guards (2 tests: missing inputs, missing `ANTHROPIC_API_KEY`).

## Key Design Decisions

**Voice emotion model**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` — outputs valence, arousal, dominance (continuous 0–1) per segment. Better than training on RAVDESS for naturalistic speech. Regression head — no sigmoid; clip to [0,1]. Model outputs `[arousal, dominance, valence]` per config.json; the module reorders to `{valence, arousal, dominance}` in the output dict.

**Facial emotion**: DeepFace with `detector_backend="retinaface"` (the default `opencv` backend is broken — `opencv-python` 5.x omits the haarcascade XMLs from `cv2/data/`, so it raises `ValueError` on every frame and the no-face handler swallows it → 0 frames). One frame every 10s via OpenCV seek. DeepFace returns 0–100 percentages; the module normalizes to 0–1 to match the spec and step 3's VAD range. Frames with no detected face are skipped (no crash).

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
HF_TOKEN=...                # HuggingFace token — required for pyannote diarization (step 1)
                            # Step 1 exits early with a clear error if this is missing.
                            # Must also accept model terms at:
                            # huggingface.co/pyannote/speaker-diarization-3.1
                            # huggingface.co/pyannote/segmentation-3.0
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| WhisperX | Word-level transcription + forced alignment |
| pyannote.audio | Speaker diarization (who said what, when) |
| librosa | Audio feature extraction (pitch, energy, speech rate, pauses, ZCR) |
| transformers (audeering model) | Voice emotion: valence / arousal / dominance per segment |
| DeepFace | Facial emotion per frame (sampled every 10s) |
| Anthropic Claude API | LLM insights — model: claude-sonnet-4-6 |
| Chart.js (CDN) | Engagement timeline chart in HTML report |

## Output JSON Schemas

**transcript.json**: Array of `{speaker, start, end, text, words[]}` — word-level timestamps from WhisperX, speaker label from pyannote via `merge_speaker_labels()`. Unmatched segments get `speaker: "UNKNOWN"`.

**audio_features.json**: Array of `{speaker, start, end, pitch_mean, pitch_std, energy_mean, speech_rate, pause_ratio, zcr}` — one entry per diarized segment.

**voice_emotion.json**: Array of `{speaker, start, end, valence, arousal, dominance}` — one entry per segment (max 15s chunks).

**face_emotion.json**: Array of `{timestamp, dominant_emotion, scores{}}` — one entry per 10s video sample. `scores` are 7 emotion probabilities normalized to 0–1 (DeepFace returns 0–100). Frames with no detected face are skipped (no crash).

**analysis.json**: `{transcript_only: {...}, multimodal: {...}}` — both LLM outputs with engagement_score, deal_probability, critical_moments[], recommendations[], talk_ratio.
