# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Multimodal sales meeting analysis pipeline. Takes a B2B sales meeting video and produces an HTML report with engagement scores, deal probability, talk ratio, emotional timeline, critical moments, and coaching recommendations. The killer feature is a side-by-side comparison between transcript-only analysis vs full multimodal analysis — this is a demo aimed at Portuguese B2B sales consulting company Scale Labs.

## Finding context

- Prefer `.codegraph/` (via the `codegraph_explore` MCP tool) over grepping/reading files to locate a symbol, trace who calls a function, or see a function's blast radius before editing it — it's a pre-built index of the repo's symbols and call graph, cheaper and more accurate than re-deriving that by hand.
- `docs/steps/*.md` are line-by-line walkthroughs of each pipeline step (written in Portuguese), including the real bugs found via statistical validation on the actual demo video and how they were fixed. This is the authoritative source for *why* a non-obvious choice was made (regression head vs. classifier, detector backend, truncation guard, etc.) — check here before assuming a gotcha below is the whole story.
- `docs/superpowers/specs/2026-06-30-sales-coach-mvp-design.md` is the original design spec — the reference for intended behavior when code and spec disagree.

## Pipeline Architecture

Six sequential steps, each a standalone Python script. Output of each step is a JSON file consumed by the next:

```
input/meeting.mp4
  → pipeline/01_transcribe.py      → output/transcript.json      (WhisperX large-v2 ✓ | pyannote diarization ✓ | merge speaker labels ✓)
  → pipeline/02_audio_features.py  → output/audio_features.json  (librosa: pitch mean/std, energy mean, speech rate, pause ratio, ZCR)
  → pipeline/03_emotion_voice.py   → output/voice_emotion.json   (audeering wav2vec2: valence, arousal, dominance per segment ✓)
  → pipeline/04_emotion_face.py    → output/face_emotion.json    (DeepFace retinaface, every 10s, scores normalized 0–1 ✓)
  → pipeline/05_llm_analysis.py    → output/analysis.json        (Claude API, run twice: transcript-only + multimodal ✓)
  → pipeline/06_report.py          → output/report.html           (pure HTML + Chart.js via CDN ✓)
```

`run.py` at root orchestrates all six steps in sequence, skipping any step whose output file already exists. On failure it prints which step failed and how to retry (`rm -f <output> && python run.py`).

### Module/script split

The numbered scripts (`01_transcribe.py` etc.) are CLI entry points — they handle file paths, `sys.exit`, and print progress. Shared logic lives in importable modules alongside them:

- `pipeline/audio.py` — `extract_audio()` — ffmpeg wrapper
- `pipeline/transcribe.py` — `transcribe_audio()`, `merge_speaker_labels()` — WhisperX load + align; speaker label assignment by max time overlap
- `pipeline/diarize.py` — `diarize_audio()` — pyannote speaker-diarization-3.1 wrapper
- `pipeline/features.py` — `extract_audio_features()` — librosa pitch, energy, speech rate, pause ratio, ZCR per segment
- `pipeline/emotion_voice.py` — `extract_voice_emotion()` — audeering wav2vec2 VAD per segment; reconstructs `Wav2Vec2ForSpeechClassification` head (class removed from modern transformers); segments >15s are split into ≤15s chunks and averaged
- `pipeline/emotion_face.py` — `extract_face_emotion()` — DeepFace emotion per sampled frame (every 10s); retinaface detector, scores normalized 0–1, no-face frames skipped
- `pipeline/llm_analysis.py` — `run_analysis()` — merges the 4 upstream JSONs, calls Claude twice (transcript-only + multimodal) with forced tool-use; deterministic talk_ratio; lazy-imports anthropic
- `pipeline/report.py` — `render_report(transcript, audio_features, voice_emotion, face_emotion, analysis, meeting_title=...)` — assembles the self-contained HTML string; pure stdlib (`json` + string building), no lazy imports needed. Data-shaping helpers (`_build_timeline_series`, `_build_comparison`, `_build_proof_examples`, `_prosody_by_role`, `_vad_by_role`, `_face_emotion_distribution`, `_compare_role_stat`, `_moment_tag`, etc.) are the testable seams. Report copy is European Portuguese, written for a non-technical sales audience (see "Signal glossary" below); the LLM's own generated text (`critical_moments`, `recommendations`) is rendered verbatim in whatever language step 5 produced it in, same treatment as transcript quotes.

### BANT evidence extraction (US-1.1 complete; remaining work planned)

The next backend feature is a separate qualification subsystem. It extracts Budget, Authority, Need, and Timeline evidence from an already-processed transcript and links the existing audio, voice, and facial measurements to each evidence moment. It does **not** decide whether a lead qualifies, apply company-specific KPIs, or interpret the measurements.

Planned structure:

```
pipeline/07_qualification.py          # CLI entry point
pipeline/qualification/
├── graph.py                          # harness workflow
├── state.py                          # run state
├── schemas.py                        # immutable transcript source units (US-1.1); later chunk/evidence shapes
├── chunking.py                       # hierarchical transcript chunks
├── grounding.py                      # source and quote validation
├── assembly.py                       # merge and deduplicate evidence
├── signals.py                        # link existing measurements
├── storage.py                        # intermediate run artifacts
└── agents/
    ├── budget.py
    ├── authority.py
    ├── need.py
    └── timeline.py
```

The harness uses four parallel topic agents. Each agent scans the complete hierarchical transcript, extracts all relevant evidence for its topic, and validates the extraction against the original transcript. A deterministic assembler combines the four topic outputs, preserving cross-topic evidence when one statement is relevant to more than one BANT field. Signal linking happens afterward and reuses existing JSON outputs; the signal pipeline is not run again.

Only the transcript is chunked. It is organized into chronological parent sections and bounded child chunks made from complete transcript segments and speaker turns, with overlap and stable segment IDs. `audio_features.json`, `voice_emotion.json`, and `face_emotion.json` remain timestamped measurement sources and are linked after extraction.

#### Epic 1 chunking contract

Epic 1 uses custom deterministic Python logic, not a vector database or a
generic retrieval splitter. The qualification layer normalizes the existing
transcript without modifying `output/transcript.json` and defines these data
contracts:

- `TranscriptSegment` — one immutable source utterance with `segment_id`, speaker, timestamps, exact text, and optional word timestamps.
- `ConversationSection` — chronological parent grouping, normally containing both speakers, with intentional section overlap.
- `ExtractionChunk` — bounded LLM input with rendered speaker-labeled text, timestamps, ordered segment IDs, parent section ID, and `token_count`.
- `CoverageRecord` — proof that every source segment appears in at least one extraction chunk, with omissions and intentional shared context recorded separately.
- `ChunkingConfiguration` — section target/overlap duration, maximum chunk tokens, complete-turn overlap target, overlap token target, and tokenizer identity.

The initial strategy is conversation-aware hybrid: approximately 8-minute
sections with 30 seconds of section overlap; chunks bounded at 1,200 tokens;
and overlap of two complete speaker turns with a 250-token target. Time is for
section organization, tokens are for LLM safety, and complete speaker turns
are for chunk overlap. Normal transcript segments are never cut arbitrarily.

Every chunk is processed. Vector search is explicitly not used to select a
subset because Epic 1 requires complete transcript coverage. Candidate
configurations are evaluated with manually reviewed `EvaluationQuestion`
cases containing expected answers and required source segment IDs. Evaluation
must measure evidence coverage, question-answer context sufficiency,
extraction recall, grounding precision, token cost, and overlap duplication.

Planned runtime artifacts are separate from source code:

```
output/qualification.json
output/qualification_runs/<run_id>/
├── manifest.json
├── sections.jsonl
├── chunks.jsonl
├── budget_results.jsonl
├── authority_results.jsonl
├── need_results.jsonl
├── timeline_results.jsonl
├── validation.jsonl
└── qualification.json
```

The design and user stories are documented in `docs/Problem/ARCHITECTURE.md`, `docs/Problem/HIERARCHICAL-CHUNKING.md`, `docs/Problem/HARNESS.md`, `docs/Problem/FOLDER-ARCHITECTURE.md`, and `docs/Problem/EPICS-AND-USER-STORIES.md`.

US-1.1, US-1.2, and US-1.3 are implemented in `pipeline/qualification/schemas.py`
(immutable `TranscriptSegment`, `ChunkingConfiguration`, `ConversationSection`,
and `ExtractionChunk` contracts plus the pure `normalize_transcript()`
boundary) and `pipeline/qualification/chunking.py` (`create_sections()` —
deterministic time-window sections striding by `target - overlap`; membership
by segment start guarantees full coverage; empty windows are skipped and
sections renumbered — and `create_chunks()` — per-section chunking of
speaker turns, greedily packed under `max_chunk_tokens` with a segment-level
fallback for any single oversized turn; neighboring chunks overlap by up to
`chunk_overlap_turns` complete trailing turns bounded by `max_overlap_tokens`,
always keeping the nearest turn intact even if it alone exceeds that bound).
`normalize_transcript()` assigns deterministic `seg_000001`-style source IDs,
preserves speaker/timestamps/exact text/word timestamps, and does not modify
`output/transcript.json`. Oversized segments are split only at complete word
boundaries; pieces retain the original `segment_id` and record `piece_index`,
`word_start`, and `word_end`. Word metadata is recursively immutable.
US-1.4 adds a `ResolvedChunk` contract to `schemas.py` (frozen `chunk`/
`section`/`segments` triple) and a resolution API to `chunking.py`:
`resolve_chunk()` resolves one `ExtractionChunk` to its parent
`ConversationSection` and ordered source `TranscriptSegment`s, raising
`ChunkHierarchyError` on any broken reference; `reconstruct_chunk_text()`
rebuilds a chunk's text from those resolved segments and must equal the
chunk's own `text` byte-for-byte; `iter_chunks_for_processing()` returns
every chunk unfiltered, in original order.
Coverage validation (`CoverageRecord`, US-1.5) is not yet implemented.

Tests import the modules directly with mocks; they never call the numbered scripts (except the subprocess tests for the CLI guards). New logic for each step should follow this pattern.

Qualification tests are separated under `tests/qualification/unit/` and
`tests/qualification/integration/`. Unit tests cover schemas and individual
qualification boundaries. Integration tests use fixture JSON outputs and a
fake LLM boundary to exercise the qualification flow without running pipeline
steps `01` through `06` or calling external APIs. GitHub Actions runs only
these qualification tests through `uv` on pushes and pull requests.

### Import path duality

Scripts use bare imports (`from audio import extract_audio`) because Python adds the script's own directory to `sys.path` when run directly. Tests use package imports (`from pipeline.audio import extract_audio`) because pytest runs from the project root and `pipeline/__init__.py` makes it a package. Both work; don't add `sys.path` manipulation to either.

### Lazy-import pattern for heavy/gated dependencies

Three modules defer an expensive or environment-fragile import to inside the function that needs it, rather than at module level, so the module loads instantly and CLI scripts fail fast on missing files/env vars *before* paying the import cost:

- `pipeline/diarize.py`: `from pyannote.audio import Pipeline` inside `diarize_audio()` — pyannote imports torch at module load time and crashes in CPU-only environments without CUDA libs.
- `pipeline/emotion_face.py`: `cv2`/`deepface` inside `_iter_frames`/`_analyze_frame` — keeps the module import instant so unit tests run without TensorFlow.
- `pipeline/llm_analysis.py`: `anthropic` inside `_call_claude()` — unit tests run without the SDK installed.

Tests stub these by injecting a fake into `sys.modules` before the module is first imported:
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

### Claude tool-use + truncation guard (Step 5)

`pipeline/llm_analysis.py` forces structured output via Claude **tool-use** (`tool_choice={"type": "tool", "name": "submit_analysis"}`) rather than parsing free text — the model is constrained to emit JSON matching `ANALYSIS_TOOL.input_schema`, so the analysis is always well-formed.

**MAX_TOKENS truncation gotcha:** with `max_tokens=4096`, the multimodal call surfaced 13–15 detailed `critical_moments` that consumed the entire budget before `recommendations` was emitted — the tool_use block closed `critical_moments` and the partial JSON parsed into a *valid-but-incomplete* dict (missing `recommendations`), which silently flowed into `analysis.json`. Forced `tool_choice` guarantees a tool_use *block*, not a *complete* one. The guard is three-layered: `MAX_TOKENS=8192`, a `stop_reason == "max_tokens"` check, and `_validate_analysis()` asserting all required fields present. Getting this wrong ships a broken multimodal block that step 6 renders as if complete.

**Deterministic talk_ratio:** `talk_ratio` is computed from transcript talk-time (`_compute_talk_ratio`), not LLM-generated — LLMs are unreliable at exact arithmetic. It's injected into both `transcript_only` and `multimodal` outputs, so the ratio is identical and correct across modes (the tool schema deliberately omits it).

**Speaker → REP/PROSPECT:** diarization yields `SPEAKER_00/01/02`; `_classify_speakers()` maps by talk time (longest = REP, second = PROSPECT, rest = OTHER) — a deterministic, testable default, not an extra API call.

### Hidden-signals proof section (Step 6)

The report's most persuasive element isn't the headline `engagement_score`/`deal_probability` — on real calls those often land identical between `transcript_only` and `multimodal` (the LLM's holistic scoring is inherently noisy/similar across both prompts), which undercuts the "multimodal is better" pitch if that's the only evidence shown. Instead, `pipeline/report.py` leads with a structurally-guaranteed differentiator: `_count_dissonance_moments()` counts `critical_moments` the LLM tagged as `"Dissonance – ..."` (matched on the English substring the LLM actually generates — see Step 5's system prompt) — words-vs-tone/face mismatches, which the multimodal prompt explicitly asks it to surface (`_build_multimodal_prompt` in `llm_analysis.py`: *"Surface moments where the words and the voice/face disagree"*). Transcript-only analysis structurally cannot produce this category — it has no voice/face signal to contrast against — so the comparison (e.g. "transcript-only caught 0, multimodal caught 7") is real and non-technical-audience-friendly, unlike a flat score delta. The report displays this category as "Contradição" (`_moment_tag()`), not a literal translation of "dissonance" — plainer word, same underlying `"dissonance"` match.

The proof cards themselves are numbers-first, not LLM prose: `_build_proof_examples()` re-derives the evidence from the raw `transcript`/`voice_emotion`/`face_emotion` JSONs (via `_nearest_entry()`, a generic nearest-timestamp lookup) rather than reusing the LLM's paraphrased `description` field — an exact quote + measured `"85% positivo"` voice tone + measured `"97% triste"` face reading is self-verifying evidence for a reader who will never watch the source video, whereas prose asks them to trust the model's summary. Examples are ranked by `_face_sentiment()`-vs-`valence` contradiction strength (not chronological order); the strongest one becomes the report's hero moment, the rest render as a shorter proof log.

### Signal glossary and Portuguese report copy (Step 6)

The report is written in European Portuguese for a non-technical sales audience (Scale Labs' own consultants), and its first content section is a glossary table that translates every one of the 9 continuous signals (6 prosody fields from `audio_features.json` — `pitch_mean`, `pitch_std`, `energy_mean`, `speech_rate`, `pause_ratio`, `zcr` — plus the 3 voice VAD fields from `voice_emotion.json`) into a plain "what a high reading means / what a low reading means" pair, followed by the real REP-vs-PROSPECT number for that specific call. This exists because the raw field names (and even translated labels like "Assertividade") mean nothing to a reader who doesn't know the pipeline — a bare "58%" answers nothing until it's anchored to "confident, or an objection forming" and to the actual number from this call. `render_report()` takes `audio_features` as a required argument (previously report.py never read `audio_features.json` at all) to power this table and a "Como analisamos esta chamada" mechanism strip that shows all 5 pipeline stages (4 measurement stages + the LLM interpretation stage), making explicit that the VAD/prosody numbers are direct measurements that never pass through the LLM — only `engagement_score`/`deal_probability`/`critical_moments` do.

`_compare_role_stat()` is the generic templated-sentence generator behind every glossary row's "nesta chamada" column: given a REP value, a PROSPECT value, and three sentence templates (prospect-higher / rep-higher / similar-within-15%), it picks the right one so the same row reads correctly regardless of which role happens to score higher in a given meeting — the glossary isn't hardcoded to this one demo call.

Facial emotion also gets a whole-call distribution (`_face_emotion_distribution()`), not just the isolated proof-moment readings — all 7 DeepFace categories are shown, including the ones that scored zero for this call, so "what we measure" reads as complete rather than cherry-picked.

### `overleaf/report.tex` is a separate, static deliverable

`overleaf/report.tex` is a hand-authored LaTeX document (compile on Overleaf, pdfLaTeX) that mirrors the same content as the HTML report — signal glossary, mechanism strip, hero contradiction moment, comparison table, critical-moments table, recommendations. It is **not** generated by `pipeline/report.py` and is **not** wired into `run.py` or the test suite — it's a one-off snapshot of one specific meeting's numbers, built for a printable/shareable PDF version of the demo. If the underlying `output/*.json` changes, this file does not update automatically; regenerating it means re-deriving the numbers by hand (or building a `.tex`-emitting sibling to `render_report()`, which doesn't exist yet).

## Project Structure

```
├── .claude/          Agent skills and configurations
├── docs/             Design specs, implementation plans, and per-step walkthroughs (docs/steps/)
├── input/            Place meeting.mp4 here
├── models/           Model weights cache (auto-downloaded)
├── output/           Generated JSON files, report.html, and other derived artifacts
├── overleaf/         Static LaTeX snapshot report (see above) — not pipeline-generated
├── pipeline/         The six analysis scripts + shared modules; planned qualification subsystem
├── tests/            Pytest test suite
├── run.py            Orchestrator
└── requirements.txt  Python dependencies
```

## Commands

```bash
# Setup (first time)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and HF_TOKEN

# Download demo video
yt-dlp https://youtu.be/N0SF2nZS-S8 -o input/meeting.mp4

# Run the full pipeline (skips any step whose output already exists)
python run.py

# Re-run from a specific step (delete its output file, everything after it re-runs too)
rm output/analysis.json && python run.py   # re-runs steps 5 and 6 only

# Run an individual step directly (for debugging)
python pipeline/01_transcribe.py   # ... through 06_report.py

# Tests
python -m pytest tests/ -v                      # all tests
python -m pytest tests/test_transcribe.py -v    # single file
python -m pytest tests/ -v -k "test_loads"      # single test by name

# View report
# Open output/report.html in a browser

# Planned qualification subsystem
python pipeline/07_qualification.py
```

271 tests: existing pipeline tests plus 59 qualification tests for US-1.1 through US-1.4 in `tests/qualification/unit/` and `tests/qualification/integration/`. The qualification tests cover deterministic source IDs, exact source preservation, recursive immutability, oversized word-boundary splitting, fixture traceability, chunking configuration defaults and validation, deterministic overlapping conversation sections with full segment coverage, bounded extraction chunks — turn-preserving packing, the segment-level fallback for oversized turns, turn-based overlap (including overlap immediately after a fallback-produced chunk), deterministic global chunk IDs with per-section sequencing, and end-to-end coverage from raw transcript through sections to chunks — and chunk hierarchy resolution (US-1.4): resolving a chunk to its parent section and ordered source segments, every `ChunkHierarchyError` failure mode (unknown chunk ID, missing parent section, broken parent-child link, unresolvable/miscounted segments including the oversized-piece occurrence case), byte-for-byte text reconstruction from resolved segments (including overlap chunks), and unfiltered full-coverage chunk iteration. Existing pipeline test coverage remains as previously listed below.

Two real-model integration tests (`test_emotion_face.py::TestExtractFaceEmotionIntegration::test_with_real_meeting_video`, `test_emotion_voice.py::TestExtractVoiceEmotionIntegration::test_with_real_sine_tone`) segfault if run in the same process — a native library conflict between TensorFlow (DeepFace) and PyTorch (transformers) when both load real models in one pytest run, not a code bug. Deselect one when running the full suite together: `python -m pytest tests/ --deselect tests/test_emotion_face.py::TestExtractFaceEmotionIntegration::test_with_real_meeting_video`.

## Environment

**Required variables** — store in `.env` at the project root; every script that needs a key loads it via `python-dotenv`:

```
ANTHROPIC_API_KEY=...       # Claude API key for step 5
HF_TOKEN=...                # HuggingFace token — required for pyannote diarization (step 1)
                            # Step 1 exits early with a clear error if this is missing.
                            # Must also accept model terms at:
                            # huggingface.co/pyannote/speaker-diarization-3.1
                            # huggingface.co/pyannote/segmentation-3.0
```

**No GPU assumed** — the pipeline runs on CPU (WSL Ubuntu). WhisperX and the audeering model will be slow on long videos.

**Model cache** — the audeering wav2vec2 model is downloaded once and cached in `models/` at the project root; DeepFace caches its own weights to `~/.deepface/weights/`.

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

**analysis.json**: `{transcript_only: {...}, multimodal: {...}}` — both LLM outputs with engagement_score, deal_probability, critical_moments[], recommendations[], talk_ratio. Each `critical_moments[]` entry is `{timestamp: "HH:MM:SS", type, description, coaching}`; `type` values prefixed `"Dissonance – ..."` are what `pipeline/report.py` counts/surfaces as hidden-signal proof (see Step 6 above).
