# Step 5 — LLM Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the four upstream JSONs (transcript, audio_features, voice_emotion, face_emotion) into compact per-segment blocks, call the Claude API twice — transcript-only then multimodal — and write `output/analysis.json` with the side-by-side analysis that is the demo's killer feature.

**Architecture:** Follow the established module/script split — a shared module (`pipeline/llm_analysis.py`) holds the pure projection/merge/prompt logic plus a lazy-imported Claude call, and a CLI script (`pipeline/05_llm_analysis.py`) guards on missing inputs + missing API key, then writes JSON. `anthropic` is lazy-imported inside the call path (same pattern as pyannote in `diarize.py`, cv2/deepface in `emotion_face.py`) so the module loads cleanly and the pure-logic unit tests run without the SDK installed. Structured output is forced via Claude tool-use (`tool_choice`), guaranteeing the analysis schema instead of parsing free text. `talk_ratio` is computed deterministically (not LLM-generated) and injected into both outputs — LLMs are bad at exact arithmetic and talk ratio is a measured fact.

**Tech Stack:** anthropic ≥0.40.0 (Claude Messages API + tool-use), python-dotenv (env loading), pytest.

**Skills applied:**
- **test-driven-development** — every function has a failing test first; RED → GREEN → commit per task
- **writing-plans** — plan structure, bite-sized tasks, completeness/self-review
- **brainstorming** — design was validated against the project goal (the side-by-side contrast IS the pitch) before this plan; the prompt is engineered to surface cross-modal dissonance (words vs. voice/face), which transcript-only structurally cannot see
- **ml-pipeline-workflow** — does not apply (single LLM inference step in a flat-script demo pipeline; no training/validation/registry/serving)
- **recsys-pipeline-architect** — does not apply (not a top-K ranking pipeline)

---

## Prerequisites

`anthropic` is listed in `requirements.txt` but is **not installed** in the venv (verified). The module's lazy import means the test suite is green without it, but to actually run step 5 you must install it:

```bash
source venv/bin/activate
pip install anthropic          # required for the real pipeline + the integration test
```

No changes to `requirements.txt` (anthropic already listed at `requirements.txt:19`) or `run.py` (step 5 already wired at `run.py:31-36`).

`ANTHROPIC_API_KEY` is already set in `.env` (verified).

---

## Spec refinements (resolving gaps in `docs/superpowers/specs/2026-06-30-sales-coach-mvp-design.md:152-197`)

The MVP spec left two mechanical details open. This plan resolves them as follows:

1. **Speaker → REP/PROSPECT mapping** (spec shows `Speaker: PROSPECT` but never says how): a deterministic pure function `_classify_speakers()` maps by total talk time (longest = REP, second = PROSPECT, rest = OTHER). No extra API call, fully testable, and the LLM still reasons over both speakers' content regardless of label. This is the pragmatic default; it's a single swap point if an LLM-infer call is wanted later.

2. **Reliable JSON output** (spec shows the output schema but not the mechanism): force structured output via Claude **tool-use** with `tool_choice={"type": "tool", "name": "submit_analysis"}`. The model is constrained to emit JSON matching the tool's `input_schema` — no fragile text parsing. The tool returns `engagement_score`, `deal_probability`, `critical_moments`, `recommendations`; `talk_ratio` is added by us (see above).

3. **"Transcript-only sends only the Text field"** (spec `:170`): interpreted as *the textual transcript* (speaker role + text + timing) — i.e. what a company like Scale Labs actually has today. Literally sending only the words (no speaker/timing) would make `talk_ratio` impossible, which the output schema requires for both modes. So transcript-only = speaker + text + timestamp; multimodal = + audio + voice + face.

4. **Per-segment block format** (spec `:160-168` shows qualifiers like `pitch_std=high`): generating those requires brittle thresholds, so blocks emit raw numbers and the *system prompt* carries the interpretation guide (`valence < 0.4 = negative`, etc.). More honest and more testable.

---

## File Structure

| File | Responsibility |
|---|---|
| `pipeline/llm_analysis.py` | Shared module: `_format_timestamp`, `_classify_speakers`, `_compute_talk_ratio`, `_face_for_segment`, `_merge_segments`, `_build_transcript_prompt`, `_build_multimodal_prompt`, `_create_message`, `_extract_tool_input`, `_call_claude`, `run_analysis`, plus `SYSTEM_PROMPT` / `ANALYSIS_TOOL` constants. Lazy-imports anthropic. |
| `pipeline/05_llm_analysis.py` | CLI entry point: guard on missing inputs + missing `ANTHROPIC_API_KEY`, call `run_analysis`, write `output/analysis.json`. |
| `tests/test_llm_analysis.py` | Unit tests (pure logic + mocked Claude seams) + skip-guarded integration test. |
| `tests/test_05_llm_analysis.py` | Subprocess guard tests for the CLI (missing inputs → exit 1; missing key → exit 1). |

`pipeline/__init__.py` and `tests/__init__.py` already exist; no package changes needed (the existing pattern imports directly from `pipeline.llm_analysis`).

---

## Task 1: Module scaffold + `_format_timestamp` (pure helper)

**Files:**
- Create: `pipeline/llm_analysis.py`
- Create: `tests/test_llm_analysis.py`

- [ ] **Step 1: Write the failing tests + test-file header**

Create `tests/test_llm_analysis.py`:

```python
import importlib.util
import os
import sys
import json
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _anthropic_available():
    return importlib.util.find_spec("anthropic") is not None


class TestFormatTimestamp:
    def test_zero_seconds(self):
        from pipeline.llm_analysis import _format_timestamp
        assert _format_timestamp(0) == "00:00:00"

    def test_under_one_minute(self):
        from pipeline.llm_analysis import _format_timestamp
        assert _format_timestamp(44.7) == "00:00:45"

    def test_minutes_and_seconds(self):
        from pipeline.llm_analysis import _format_timestamp
        # spec example: 00:12:24
        assert _format_timestamp(12 * 60 + 24) == "00:12:24"

    def test_hours(self):
        from pipeline.llm_analysis import _format_timestamp
        assert _format_timestamp(3 * 3600 + 6 * 60 + 9) == "03:06:09"

    def test_rounds_to_nearest_second(self):
        from pipeline.llm_analysis import _format_timestamp
        assert _format_timestamp(12.6) == "00:00:13"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_analysis.py::TestFormatTimestamp -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.llm_analysis'`

- [ ] **Step 3: Write minimal implementation**

Create `pipeline/llm_analysis.py`:

```python
"""LLM analysis for step 5.

Loads the four upstream JSON outputs (transcript, audio_features, voice_emotion,
face_emotion), projects them into compact per-segment blocks, and calls the
Claude API twice -- once transcript-only, once multimodal -- to produce the
side-by-side analysis that is the demo's killer feature.

`anthropic` is lazy-imported inside the call path (same pattern as pyannote in
`diarize.py` and cv2/deepface in `emotion_face.py`) so the module loads cleanly
and the pure-logic unit tests run without the SDK installed.
"""

import json
import os
import time

MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS = 4096
RATE_LIMIT_WAIT = 10  # seconds; spec: retry once after 10s

TRANSCRIPT_FILE = "output/transcript.json"
AUDIO_FEATURES_FILE = "output/audio_features.json"
VOICE_EMOTION_FILE = "output/voice_emotion.json"
FACE_EMOTION_FILE = "output/face_emotion.json"
OUTPUT_FILE = "output/analysis.json"


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS (spec uses 00:12:24 style)."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_analysis.py::TestFormatTimestamp -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_analysis.py tests/test_llm_analysis.py
git commit -m "feat(step5): add llm_analysis module scaffold with _format_timestamp"
```

---

## Task 2: `_classify_speakers` (pure helper)

**Files:**
- Modify: `pipeline/llm_analysis.py`
- Modify: `tests/test_llm_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_analysis.py`:

```python
class TestClassifySpeakers:
    def test_longest_talker_is_rep(self):
        from pipeline.llm_analysis import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 30.0},
            {"speaker": "SPEAKER_01", "start": 30.0, "end": 40.0},
        ]
        mapping = _classify_speakers(transcript)
        assert mapping == {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT"}

    def test_aggregates_across_segments(self):
        from pipeline.llm_analysis import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 5.0},
            {"speaker": "SPEAKER_00", "start": 5.0, "end": 25.0},
            {"speaker": "SPEAKER_01", "start": 25.0, "end": 35.0},
            {"speaker": "SPEAKER_00", "start": 35.0, "end": 40.0},
        ]
        # SPEAKER_00 = 25s, SPEAKER_01 = 15s
        mapping = _classify_speakers(transcript)
        assert mapping["SPEAKER_00"] == "REP"
        assert mapping["SPEAKER_01"] == "PROSPECT"

    def test_third_speaker_is_other(self):
        from pipeline.llm_analysis import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 50.0},
            {"speaker": "SPEAKER_01", "start": 50.0, "end": 80.0},
            {"speaker": "SPEAKER_02", "start": 80.0, "end": 82.0},
        ]
        mapping = _classify_speakers(transcript)
        assert mapping["SPEAKER_02"] == "OTHER"

    def test_empty_transcript(self):
        from pipeline.llm_analysis import _classify_speakers
        assert _classify_speakers([]) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_analysis.py::TestClassifySpeakers -v`
Expected: FAIL with `AttributeError: module 'pipeline.llm_analysis' has no attribute '_classify_speakers'`

- [ ] **Step 3: Write minimal implementation**

Append to `pipeline/llm_analysis.py`:

```python
def _classify_speakers(transcript: list[dict]) -> dict[str, str]:
    """Map diarization labels to REP / PROSPECT / OTHER by total talk time.

    Longest total talk time -> REP, second -> PROSPECT, rest -> OTHER.
    Deterministic default (no extra API call); the LLM still reasons over both
    speakers' content regardless of label. Single swap point for an LLM-infer
    call later.
    """
    totals: dict[str, float] = {}
    for seg in transcript:
        sp = seg.get("speaker", "UNKNOWN")
        totals[sp] = totals.get(sp, 0.0) + (seg.get("end", 0) - seg.get("start", 0))
    ranked = sorted(totals, key=lambda s: totals[s], reverse=True)
    mapping: dict[str, str] = {}
    for i, sp in enumerate(ranked):
        mapping[sp] = "REP" if i == 0 else ("PROSPECT" if i == 1 else "OTHER")
    return mapping
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_analysis.py::TestClassifySpeakers -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_analysis.py tests/test_llm_analysis.py
git commit -m "feat(step5): add _classify_speakers talk-time heuristic"
```

---

## Task 3: `_compute_talk_ratio` (pure helper)

**Files:**
- Modify: `pipeline/llm_analysis.py`
- Modify: `tests/test_llm_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_analysis.py`:

```python
class TestComputeTalkRatio:
    def test_simple_split(self):
        from pipeline.llm_analysis import _compute_talk_ratio
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 60.0},
            {"speaker": "SPEAKER_01", "start": 60.0, "end": 100.0},
        ]
        speaker_map = {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT"}
        assert _compute_talk_ratio(transcript, speaker_map) == {"rep": 60, "prospect": 40}

    def test_aggregates_across_segments(self):
        from pipeline.llm_analysis import _compute_talk_ratio
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 20.0},
            {"speaker": "SPEAKER_01", "start": 20.0, "end": 60.0},
            {"speaker": "SPEAKER_00", "start": 60.0, "end": 80.0},
        ]
        speaker_map = {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT"}
        # REP = 40s, PROSPECT = 40s -> 50/50
        assert _compute_talk_ratio(transcript, speaker_map) == {"rep": 50, "prospect": 50}

    def test_ignores_other_speakers(self):
        from pipeline.llm_analysis import _compute_talk_ratio
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 80.0},
            {"speaker": "SPEAKER_01", "start": 80.0, "end": 100.0},
            {"speaker": "SPEAKER_02", "start": 100.0, "end": 120.0},
        ]
        speaker_map = {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT", "SPEAKER_02": "OTHER"}
        # only REP + PROSPECT counted: 80 + 20 = 100 -> 80/20
        assert _compute_talk_ratio(transcript, speaker_map) == {"rep": 80, "prospect": 20}

    def test_zero_talk_time_returns_zeros(self):
        from pipeline.llm_analysis import _compute_talk_ratio
        assert _compute_talk_ratio([], {}) == {"rep": 0, "prospect": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_analysis.py::TestComputeTalkRatio -v`
Expected: FAIL with `AttributeError: ... has no attribute '_compute_talk_ratio'`

- [ ] **Step 3: Write minimal implementation**

Append to `pipeline/llm_analysis.py`:

```python
def _compute_talk_ratio(transcript: list[dict], speaker_map: dict[str, str]) -> dict[str, int]:
    """Compute rep/prospect talk-time percentages (measured, not LLM-generated).

    LLMs are unreliable at exact arithmetic; talk ratio is a measured fact, so
    we compute it deterministically and inject it into both LLM outputs.
    """
    role_totals: dict[str, float] = {"REP": 0.0, "PROSPECT": 0.0}
    for seg in transcript:
        role = speaker_map.get(seg.get("speaker", ""), "OTHER")
        dur = seg.get("end", 0) - seg.get("start", 0)
        if role in role_totals:
            role_totals[role] += dur
    total = role_totals["REP"] + role_totals["PROSPECT"]
    if total <= 0:
        return {"rep": 0, "prospect": 0}
    return {
        "rep": round(role_totals["REP"] / total * 100),
        "prospect": round(role_totals["PROSPECT"] / total * 100),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_analysis.py::TestComputeTalkRatio -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_analysis.py tests/test_llm_analysis.py
git commit -m "feat(step5): add _compute_talk_ratio deterministic measurement"
```

---

## Task 4: `_face_for_segment` + `_merge_segments` (pure projection)

**Files:**
- Modify: `pipeline/llm_analysis.py`
- Modify: `tests/test_llm_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_analysis.py`:

```python
class TestFaceForSegment:
    def test_returns_nearest_to_midpoint(self):
        from pipeline.llm_analysis import _face_for_segment
        seg = {"start": 215.0, "end": 225.0}  # midpoint 220
        face = [
            {"timestamp": 210.0, "dominant_emotion": "angry", "scores": {}},
            {"timestamp": 220.0, "dominant_emotion": "happy", "scores": {}},
            {"timestamp": 230.0, "dominant_emotion": "sad", "scores": {}},
        ]
        result = _face_for_segment(seg, face)
        assert result["timestamp"] == 220.0

    def test_empty_face_returns_none(self):
        from pipeline.llm_analysis import _face_for_segment
        seg = {"start": 0.0, "end": 5.0}
        assert _face_for_segment(seg, []) is None


class TestMergeSegments:
    def _inputs(self):
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0,
             "text": " Hello ", "words": [{"word": "Hello"}], "avg_logprob": -0.1},
        ]
        audio = [{"pitch_mean": 180.0, "pitch_std": 12.0, "energy_mean": 0.05,
                  "speech_rate": 3.0, "pause_ratio": 0.1, "zcr": 0.08}]
        voice = [{"valence": 0.4, "arousal": 0.5, "dominance": 0.6}]
        face = [{"timestamp": 2.0, "dominant_emotion": "neutral",
                 "scores": {"neutral": 0.9, "happy": 0.1}}]
        return transcript, audio, voice, face

    def test_relabels_speaker_to_role(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, f = self._inputs()
        merged = _merge_segments(t, a, v, f, {"SPEAKER_00": "REP"})
        assert merged[0]["speaker"] == "REP"

    def test_strips_words_and_avg_logprob(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, f = self._inputs()
        merged = _merge_segments(t, a, v, f, {"SPEAKER_00": "REP"})
        assert "words" not in merged[0]
        assert "avg_logprob" not in merged[0]

    def test_strips_text_whitespace(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, f = self._inputs()
        merged = _merge_segments(t, a, v, f, {"SPEAKER_00": "REP"})
        assert merged[0]["text"] == "Hello"

    def test_attaches_audio_voice_face(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, f = self._inputs()
        merged = _merge_segments(t, a, v, f, {"SPEAKER_00": "REP"})
        assert merged[0]["audio"]["pitch_mean"] == 180.0
        assert merged[0]["voice"]["valence"] == 0.4
        assert merged[0]["face"]["dominant_emotion"] == "neutral"

    def test_no_face_emotion_omits_face_key(self):
        from pipeline.llm_analysis import _merge_segments
        t, a, v, _ = self._inputs()
        merged = _merge_segments(t, a, v, [], {"SPEAKER_00": "REP"})
        assert "face" not in merged[0]

    def test_index_misalignment_does_not_crash(self):
        from pipeline.llm_analysis import _merge_segments
        t, _, _, _ = self._inputs()
        # audio/voice shorter than transcript
        merged = _merge_segments(t, [], [], [], {"SPEAKER_00": "REP"})
        assert "audio" not in merged[0]
        assert "voice" not in merged[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_analysis.py::TestFaceForSegment tests/test_llm_analysis.py::TestMergeSegments -v`
Expected: FAIL with `AttributeError: ... has no attribute '_face_for_segment'`

- [ ] **Step 3: Write minimal implementation**

Append to `pipeline/llm_analysis.py`:

```python
def _face_for_segment(segment: dict, face_emotion: list[dict]) -> dict | None:
    """Return the face sample nearest the segment midpoint, or None.

    Face emotion is sampled every 10s with no speaker attribution, so each
    segment takes the nearest sample (always within ~5s for a 10s cadence).
    The LLM interprets the (speaker-ambiguous) reading; face is the weakest
    signal and the system prompt says so.
    """
    if not face_emotion:
        return None
    midpoint = (segment["start"] + segment["end"]) / 2
    return min(face_emotion, key=lambda fr: abs(fr["timestamp"] - midpoint))


def _merge_segments(
    transcript: list[dict],
    audio_features: list[dict],
    voice_emotion: list[dict],
    face_emotion: list[dict],
    speaker_map: dict[str, str],
) -> list[dict]:
    """Merge the three index-aligned per-segment arrays + nearest face sample.

    Drops word timestamps / avg_logprob (LLM-irrelevant, ~200KB of the 330KB
    transcript). Relabels speakers to roles. Returns compact per-segment dicts.
    """
    merged = []
    for i, seg in enumerate(transcript):
        role = speaker_map.get(seg.get("speaker", ""), "OTHER")
        item = {
            "speaker": role,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", "").strip(),
        }
        if i < len(audio_features):
            a = audio_features[i]
            item["audio"] = {
                "pitch_mean": a.get("pitch_mean"),
                "pitch_std": a.get("pitch_std"),
                "energy_mean": a.get("energy_mean"),
                "speech_rate": a.get("speech_rate"),
                "pause_ratio": a.get("pause_ratio"),
                "zcr": a.get("zcr"),
            }
        if i < len(voice_emotion):
            v = voice_emotion[i]
            item["voice"] = {
                "valence": v.get("valence"),
                "arousal": v.get("arousal"),
                "dominance": v.get("dominance"),
            }
        face = _face_for_segment(seg, face_emotion)
        if face is not None:
            item["face"] = {
                "dominant_emotion": face.get("dominant_emotion"),
                "scores": face.get("scores"),
                "timestamp": face.get("timestamp"),
            }
        merged.append(item)
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_analysis.py::TestFaceForSegment tests/test_llm_analysis.py::TestMergeSegments -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_analysis.py tests/test_llm_analysis.py
git commit -m "feat(step5): add _merge_segments projection dropping word timestamps"
```

---

## Task 5: System prompt + tool schema + prompt builders

**Files:**
- Modify: `pipeline/llm_analysis.py`
- Modify: `tests/test_llm_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_analysis.py`:

```python
class TestSystemPrompt:
    def test_includes_signal_interpretation_guide(self):
        from pipeline.llm_analysis import SYSTEM_PROMPT
        assert "valence" in SYSTEM_PROMPT
        assert "arousal" in SYSTEM_PROMPT
        assert "dominance" in SYSTEM_PROMPT

    def test_instructs_dissonance_surfacing(self):
        from pipeline.llm_analysis import SYSTEM_PROMPT
        assert "dissonance" in SYSTEM_PROMPT.lower()

    def test_requires_timestamps(self):
        from pipeline.llm_analysis import SYSTEM_PROMPT
        assert "timestamp" in SYSTEM_PROMPT.lower()


class TestAnalysisTool:
    def test_forces_required_fields(self):
        from pipeline.llm_analysis import ANALYSIS_TOOL
        required = ANALYSIS_TOOL["input_schema"]["required"]
        assert set(required) == {
            "engagement_score", "deal_probability",
            "critical_moments", "recommendations",
        }

    def test_critical_moments_has_coaching(self):
        from pipeline.llm_analysis import ANALYSIS_TOOL
        cm = ANALYSIS_TOOL["input_schema"]["properties"]["critical_moments"]["items"]
        assert "coaching" in cm["required"]
        assert "timestamp" in cm["required"]

    def test_tool_name(self):
        from pipeline.llm_analysis import ANALYSIS_TOOL
        assert ANALYSIS_TOOL["name"] == "submit_analysis"


class TestBuildTranscriptPrompt:
    def _merged(self):
        return [{
            "speaker": "PROSPECT", "start": 12.4, "end": 18.1,
            "text": "I'm not sure the pricing makes sense.",
            "audio": {"pitch_mean": 180, "pitch_std": 24, "energy_mean": 0.04,
                      "speech_rate": 3.2, "pause_ratio": 0.18, "zcr": 0.06},
            "voice": {"valence": 0.31, "arousal": 0.22, "dominance": 0.41},
            "face": {"dominant_emotion": "neutral", "scores": {"neutral": 0.71}},
        }]

    def test_includes_speaker_and_text(self):
        from pipeline.llm_analysis import _build_transcript_prompt
        prompt = _build_transcript_prompt(self._merged())
        assert "PROSPECT" in prompt
        assert "I'm not sure the pricing makes sense." in prompt

    def test_includes_timestamp(self):
        from pipeline.llm_analysis import _build_transcript_prompt
        prompt = _build_transcript_prompt(self._merged())
        assert "00:00:12" in prompt

    def test_excludes_modalities(self):
        from pipeline.llm_analysis import _build_transcript_prompt
        prompt = _build_transcript_prompt(self._merged())
        assert "Audio:" not in prompt
        assert "Voice emotion:" not in prompt
        assert "Facial:" not in prompt


class TestBuildMultimodalPrompt:
    def _merged(self):
        return [{
            "speaker": "PROSPECT", "start": 12.4, "end": 18.1,
            "text": "I'm not sure the pricing makes sense.",
            "audio": {"pitch_mean": 180, "pitch_std": 24, "energy_mean": 0.04,
                      "speech_rate": 3.2, "pause_ratio": 0.18, "zcr": 0.06},
            "voice": {"valence": 0.31, "arousal": 0.22, "dominance": 0.41},
            "face": {"dominant_emotion": "neutral", "scores": {"neutral": 0.71}},
        }]

    def test_includes_all_modalities(self):
        from pipeline.llm_analysis import _build_multimodal_prompt
        prompt = _build_multimodal_prompt(self._merged())
        assert "Audio:" in prompt
        assert "Voice emotion:" in prompt
        assert "Facial:" in prompt

    def test_includes_signal_values(self):
        from pipeline.llm_analysis import _build_multimodal_prompt
        prompt = _build_multimodal_prompt(self._merged())
        assert "0.31" in prompt  # valence
        assert "neutral" in prompt

    def test_includes_dissonance_instruction(self):
        from pipeline.llm_analysis import _build_multimodal_prompt
        prompt = _build_multimodal_prompt(self._merged())
        assert "dissonance" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_analysis.py::TestSystemPrompt tests/test_llm_analysis.py::TestAnalysisTool tests/test_llm_analysis.py::TestBuildTranscriptPrompt tests/test_llm_analysis.py::TestBuildMultimodalPrompt -v`
Expected: FAIL with `ImportError: cannot import name 'SYSTEM_PROMPT'`

- [ ] **Step 3: Write minimal implementation**

Append to `pipeline/llm_analysis.py`:

```python
SYSTEM_PROMPT = """You are a senior B2B sales coach analyzing a recorded sales meeting.

Read the meeting segment-by-segment and produce a structured analysis: an
engagement score (0-100), a deal probability (0-100), the critical moments
(each with a timestamp, a type, a description of what happened, and a coaching
note), and actionable coaching recommendations.

Rules:
- Be specific and concrete. Cite exact timestamps and signal values.
- No generic advice. Every recommendation must reference something that
  actually happened in this meeting.
- Timestamps must use HH:MM:SS and match moments in the provided segments.

How to read the voice/face signals (multimodal mode only):
- valence < 0.4 = negative affect; > 0.6 = positive.
- arousal > 0.6 = energized/agitated; < 0.4 = flat/disengaged.
- dominance rising in the prospect = pushback or an objection forming.
- When the words say one thing but the voice/face say another, that
  dissonance is the most important signal -- surface it as a critical moment
  with the timestamp and the conflicting signal values.
"""


ANALYSIS_TOOL = {
    "name": "submit_analysis",
    "description": "Submit the structured sales-meeting analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "engagement_score": {"type": "integer"},
            "deal_probability": {"type": "integer"},
            "critical_moments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                        "coaching": {"type": "string"},
                    },
                    "required": ["timestamp", "type", "description", "coaching"],
                },
            },
            "recommendations": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "engagement_score",
            "deal_probability",
            "critical_moments",
            "recommendations",
        ],
    },
}


def _build_transcript_prompt(merged: list[dict]) -> str:
    """Transcript-only user message: speaker role + text + timing per segment.

    No audio/voice/face -- the honest 'what a transcript gives you' baseline.
    The contrast with the multimodal call IS the demo's pitch.
    """
    lines = ["TRANSCRIPT (speaker-labeled; no audio/voice/face signals):", ""]
    for seg in merged:
        ts = _format_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['speaker']}: {seg['text']}")
    lines.append("")
    lines.append("Analyze this transcript and submit your analysis via the submit_analysis tool.")
    return "\n".join(lines)


def _build_multimodal_prompt(merged: list[dict]) -> str:
    """Multimodal user message: full per-segment blocks with audio/voice/face."""
    lines = [
        "MEETING (transcript + audio + voice emotion + facial emotion per segment):",
        "",
    ]
    for seg in merged:
        ts = _format_timestamp(seg["start"])
        lines.append(f"SEGMENT [{ts}]")
        lines.append(f"Speaker: {seg['speaker']}")
        lines.append(f"Text: {seg['text']}")
        a = seg.get("audio")
        if a:
            lines.append(
                f"Audio: pitch_mean={a['pitch_mean']} pitch_std={a['pitch_std']} "
                f"energy={a['energy_mean']} speech_rate={a['speech_rate']} "
                f"pause_ratio={a['pause_ratio']} zcr={a['zcr']}"
            )
        v = seg.get("voice")
        if v:
            lines.append(
                f"Voice emotion: valence={v['valence']} arousal={v['arousal']} "
                f"dominance={v['dominance']}"
            )
        f = seg.get("face")
        if f:
            lines.append(
                f"Facial: {f['dominant_emotion']} (dominant), scores={f['scores']}"
            )
        lines.append("")
    lines.append("Analyze this meeting and submit your analysis via the submit_analysis tool.")
    lines.append(
        "Surface moments where the words and the voice/face disagree -- "
        "that dissonance is the key signal transcript-only analysis cannot see."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_analysis.py::TestSystemPrompt tests/test_llm_analysis.py::TestAnalysisTool tests/test_llm_analysis.py::TestBuildTranscriptPrompt tests/test_llm_analysis.py::TestBuildMultimodalPrompt -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_analysis.py tests/test_llm_analysis.py
git commit -m "feat(step5): add system prompt, analysis tool schema, prompt builders"
```

---

## Task 6: `_create_message` + `_extract_tool_input` + `_call_claude` (mocked)

**Files:**
- Modify: `pipeline/llm_analysis.py`
- Modify: `tests/test_llm_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_analysis.py`:

```python
def _install_fake_anthropic(fake):
    """Inject a fake `anthropic` module so the lazy import resolves to it."""
    sys.modules["anthropic"] = fake


def _make_fake_anthropic(create_side_effect=None, create_return=None):
    fake = MagicMock()
    fake.RateLimitError = type("RateLimitError", (Exception,), {})
    client = MagicMock()
    if create_side_effect is not None:
        client.messages.create.side_effect = create_side_effect
    else:
        client.messages.create.return_value = create_return
    fake.Anthropic.return_value = client
    return fake


class TestExtractToolInput:
    def test_returns_tool_use_input(self):
        from pipeline.llm_analysis import _extract_tool_input
        block = MagicMock(type="tool_use", input={"engagement_score": 70})
        response = MagicMock(content=[MagicMock(type="text"), block])
        assert _extract_tool_input(response) == {"engagement_score": 70}

    def test_raises_when_no_tool_use_block(self):
        from pipeline.llm_analysis import _extract_tool_input
        response = MagicMock(content=[MagicMock(type="text")])
        with pytest.raises(RuntimeError, match="tool_use"):
            _extract_tool_input(response)


class TestCallClaude:
    def test_extracts_tool_input(self):
        tool_input = {"engagement_score": 61, "deal_probability": 45,
                      "critical_moments": [], "recommendations": []}
        fake = _make_fake_anthropic(
            create_return=MagicMock(content=[MagicMock(type="tool_use", input=tool_input)])
        )
        _install_fake_anthropic(fake)
        try:
            from pipeline.llm_analysis import _call_claude
            result = _call_claude("prompt", "key")
        finally:
            del sys.modules["anthropic"]
        assert result == tool_input

    def test_uses_forced_tool_choice(self):
        fake = _make_fake_anthropic(
            create_return=MagicMock(content=[MagicMock(type="tool_use", input={})])
        )
        _install_fake_anthropic(fake)
        try:
            from pipeline.llm_analysis import _call_claude
            _call_claude("prompt", "key")
        finally:
            del sys.modules["anthropic"]
        client = fake.Anthropic.return_value
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_analysis"}
        assert kwargs["model"] == "claude-sonnet-4-6"

    def test_retries_once_on_rate_limit(self):
        tool_input = {"engagement_score": 1}
        fake = _make_fake_anthropic(
            create_side_effect=[
                fake_err := None,  # placeholder, replaced below
            ]
        )
        # Build the side_effect with the fake's real exception class.
        RateLimitError = fake.RateLimitError
        client = fake.Anthropic.return_value
        client.messages.create.side_effect = [
            RateLimitError("limit"),
            MagicMock(content=[MagicMock(type="tool_use", input=tool_input)]),
        ]
        _install_fake_anthropic(fake)
        try:
            with patch("pipeline.llm_analysis.time.sleep") as sleep:
                from pipeline.llm_analysis import _call_claude
                result = _call_claude("prompt", "key")
        finally:
            del sys.modules["anthropic"]
        assert result == tool_input
        assert client.messages.create.call_count == 2
        sleep.assert_called_once_with(10)

    def test_passes_api_key_to_client(self):
        fake = _make_fake_anthropic(
            create_return=MagicMock(content=[MagicMock(type="tool_use", input={})])
        )
        _install_fake_anthropic(fake)
        try:
            from pipeline.llm_analysis import _call_claude
            _call_claude("prompt", "secret-key")
        finally:
            del sys.modules["anthropic"]
        fake.Anthropic.assert_called_once_with(api_key="secret-key")
```

> Note: the `test_retries_once_on_rate_limit` body assigns `create_side_effect=[fake_err := None, ...]` then immediately overwrites `client.messages.create.side_effect` with the real exception list. The first assignment is harmless dead code kept so the helper call signature is uniform; the second assignment is what actually takes effect. If your linter complains, delete the `create_side_effect` kwarg from the `_make_fake_anthropic(...)` call in that test and rely solely on the explicit `side_effect` assignment.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_analysis.py::TestExtractToolInput tests/test_llm_analysis.py::TestCallClaude -v`
Expected: FAIL with `AttributeError: ... has no attribute '_extract_tool_input'`

- [ ] **Step 3: Write minimal implementation**

Append to `pipeline/llm_analysis.py`:

```python
def _create_message(client, user_prompt: str):
    """Send one Messages-API call forcing the analysis tool. Single seam."""
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[ANALYSIS_TOOL],
        tool_choice={"type": "tool", "name": "submit_analysis"},
        messages=[{"role": "user", "content": user_prompt}],
    )


def _extract_tool_input(response) -> dict:
    """Pull the first tool_use block's input out of the response content."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input
    raise RuntimeError("Claude did not return a tool_use block")


def _call_claude(user_prompt: str, api_key: str) -> dict:
    """Call Claude with forced tool_use to guarantee the analysis schema.

    Retries once after RATE_LIMIT_WAIT seconds on a rate-limit error (spec).
    `anthropic` is lazy-imported here so unit tests run without the SDK by
    injecting a fake into sys.modules['anthropic'] (same pattern as pyannote).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = _create_message(client, user_prompt)
    except anthropic.RateLimitError:
        time.sleep(RATE_LIMIT_WAIT)
        response = _create_message(client, user_prompt)
    return _extract_tool_input(response)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_analysis.py::TestExtractToolInput tests/test_llm_analysis.py::TestCallClaude -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_analysis.py tests/test_llm_analysis.py
git commit -m "feat(step5): add _call_claude with forced tool_use and rate-limit retry"
```

---

## Task 7: `run_analysis` orchestration

**Files:**
- Modify: `pipeline/llm_analysis.py`
- Modify: `tests/test_llm_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_analysis.py`:

```python
class TestRunAnalysis:
    def _write_inputs(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        transcript = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 60.0, "text": " Hi "}]
        audio = [{"pitch_mean": 180, "pitch_std": 12, "energy_mean": 0.05,
                  "speech_rate": 3.0, "pause_ratio": 0.1, "zcr": 0.08}]
        voice = [{"valence": 0.4, "arousal": 0.5, "dominance": 0.6}]
        face = [{"timestamp": 30.0, "dominant_emotion": "neutral",
                 "scores": {"neutral": 1.0}}]
        for name, data in (("transcript.json", transcript),
                           ("audio_features.json", audio),
                           ("voice_emotion.json", voice),
                           ("face_emotion.json", face)):
            (out / name).write_text(json.dumps(data))
        return out

    def test_writes_analysis_json_with_both_modes(self, tmp_path):
        out = self._write_inputs(tmp_path)
        output_file = out / "analysis.json"
        llm_out = {"engagement_score": 70, "deal_probability": 50,
                   "critical_moments": [], "recommendations": ["x"]}
        with patch("pipeline.llm_analysis._call_claude", return_value=dict(llm_out)):
            from pipeline.llm_analysis import run_analysis
            run_analysis(
                str(out / "transcript.json"),
                str(out / "audio_features.json"),
                str(out / "voice_emotion.json"),
                str(out / "face_emotion.json"),
                str(output_file),
                api_key="key",
            )
        saved = json.loads(output_file.read_text())
        assert set(saved.keys()) == {"transcript_only", "multimodal"}
        assert saved["transcript_only"]["engagement_score"] == 70
        assert saved["multimodal"]["engagement_score"] == 70

    def test_injects_talk_ratio_into_both_modes(self, tmp_path):
        out = self._write_inputs(tmp_path)
        output_file = out / "analysis.json"
        llm_out = {"engagement_score": 1, "deal_probability": 1,
                   "critical_moments": [], "recommendations": []}
        with patch("pipeline.llm_analysis._call_claude", return_value=dict(llm_out)):
            from pipeline.llm_analysis import run_analysis
            run_analysis(
                str(out / "transcript.json"),
                str(out / "audio_features.json"),
                str(out / "voice_emotion.json"),
                str(out / "face_emotion.json"),
                str(output_file),
                api_key="key",
            )
        saved = json.loads(output_file.read_text())
        # single speaker -> REP gets 100% of rep+prospect talk time
        assert saved["transcript_only"]["talk_ratio"] == {"rep": 100, "prospect": 0}
        assert saved["multimodal"]["talk_ratio"] == {"rep": 100, "prospect": 0}

    def test_calls_claude_twice_with_different_prompts(self, tmp_path):
        out = self._write_inputs(tmp_path)
        output_file = out / "analysis.json"
        llm_out = {"engagement_score": 1, "deal_probability": 1,
                   "critical_moments": [], "recommendations": []}
        with patch("pipeline.llm_analysis._call_claude", return_value=dict(llm_out)) as mock:
            from pipeline.llm_analysis import run_analysis
            run_analysis(
                str(out / "transcript.json"),
                str(out / "audio_features.json"),
                str(out / "voice_emotion.json"),
                str(out / "face_emotion.json"),
                str(output_file),
                api_key="key",
            )
        assert mock.call_count == 2
        first_prompt = mock.call_args_list[0].args[0]
        second_prompt = mock.call_args_list[1].args[0]
        assert "TRANSCRIPT" in first_prompt
        assert "MEETING" in second_prompt

    def test_raises_on_missing_input(self, tmp_path):
        from pipeline.llm_analysis import run_analysis
        with pytest.raises(FileNotFoundError, match="not found"):
            run_analysis(
                str(tmp_path / "nope.json"),
                str(tmp_path / "nope2.json"),
                str(tmp_path / "nope3.json"),
                str(tmp_path / "nope4.json"),
                str(tmp_path / "out.json"),
                api_key="key",
            )

    def test_raises_when_api_key_missing(self, tmp_path):
        out = self._write_inputs(tmp_path)
        from pipeline.llm_analysis import run_analysis
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            run_analysis(
                str(out / "transcript.json"),
                str(out / "audio_features.json"),
                str(out / "voice_emotion.json"),
                str(out / "face_emotion.json"),
                str(out / "analysis.json"),
                api_key=None,
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_analysis.py::TestRunAnalysis -v`
Expected: FAIL with `AttributeError: ... has no attribute 'run_analysis'`

- [ ] **Step 3: Write minimal implementation**

Append to `pipeline/llm_analysis.py`:

```python
def run_analysis(
    transcript_path: str = TRANSCRIPT_FILE,
    audio_features_path: str = AUDIO_FEATURES_FILE,
    voice_emotion_path: str = VOICE_EMOTION_FILE,
    face_emotion_path: str = FACE_EMOTION_FILE,
    output_path: str = OUTPUT_FILE,
    api_key: str = None,
) -> dict:
    """Run both LLM analyses (transcript-only + multimodal) and write analysis.json.

    Loads the four upstream JSONs, projects them into compact per-segment blocks,
    calls Claude twice with different prompts, injects the measured talk_ratio
    into both outputs, and writes {transcript_only, multimodal} to disk.
    """
    for p in (transcript_path, audio_features_path, voice_emotion_path, face_emotion_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input not found: {p}")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required")

    with open(transcript_path) as f:
        transcript = json.load(f)
    with open(audio_features_path) as f:
        audio_features = json.load(f)
    with open(voice_emotion_path) as f:
        voice_emotion = json.load(f)
    with open(face_emotion_path) as f:
        face_emotion = json.load(f)

    speaker_map = _classify_speakers(transcript)
    talk_ratio = _compute_talk_ratio(transcript, speaker_map)
    merged = _merge_segments(
        transcript, audio_features, voice_emotion, face_emotion, speaker_map
    )

    transcript_llm = _call_claude(_build_transcript_prompt(merged), api_key)
    multimodal_llm = _call_claude(_build_multimodal_prompt(merged), api_key)

    analysis = {
        "transcript_only": {**transcript_llm, "talk_ratio": talk_ratio},
        "multimodal": {**multimodal_llm, "talk_ratio": talk_ratio},
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    return analysis
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_analysis.py::TestRunAnalysis -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run the whole module test file to confirm no regressions**

Run: `python -m pytest tests/test_llm_analysis.py -v`
Expected: PASS (all tasks green so far)

- [ ] **Step 6: Commit**

```bash
git add pipeline/llm_analysis.py tests/test_llm_analysis.py
git commit -m "feat(step5): add run_analysis orchestration writing analysis.json"
```

---

## Task 8: CLI entry `pipeline/05_llm_analysis.py` + subprocess guards

**Files:**
- Create: `pipeline/05_llm_analysis.py`
- Create: `tests/test_05_llm_analysis.py`

- [ ] **Step 1: Write the failing CLI guard tests**

Create `tests/test_05_llm_analysis.py`:

```python
import os
import sys
import subprocess

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(PROJECT_ROOT, "pipeline/05_llm_analysis.py")


def test_exits_when_inputs_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "transcript.json" in result.stdout


def test_exits_when_api_key_missing(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    for name in ("transcript.json", "audio_features.json",
                 "voice_emotion.json", "face_emotion.json"):
        (out / name).write_text("[]")
    env = {**os.environ, "ANTHROPIC_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    assert result.returncode == 1
    assert "ANTHROPIC_API_KEY" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_05_llm_analysis.py -v`
Expected: FAIL (`pipeline/05_llm_analysis.py` does not exist; script errors)

- [ ] **Step 3: Write the CLI script**

Create `pipeline/05_llm_analysis.py`:

```python
#!/usr/bin/env python3
"""Step 5: LLM analysis.

Merges the four upstream JSONs (transcript, audio_features, voice_emotion,
face_emotion) and calls Claude twice -- transcript-only then multimodal -- to
produce the side-by-side analysis that is the demo's killer feature.

Output: output/analysis.json
"""

import os
import sys

TRANSCRIPT_FILE = "output/transcript.json"
AUDIO_FEATURES_FILE = "output/audio_features.json"
VOICE_EMOTION_FILE = "output/voice_emotion.json"
FACE_EMOTION_FILE = "output/face_emotion.json"
OUTPUT_FILE = "output/analysis.json"

REQUIRED_INPUTS = (
    TRANSCRIPT_FILE,
    AUDIO_FEATURES_FILE,
    VOICE_EMOTION_FILE,
    FACE_EMOTION_FILE,
)


def main():
    missing = [p for p in REQUIRED_INPUTS if not os.path.exists(p)]
    if missing:
        print(f"ERROR: missing input(s): {', '.join(missing)}")
        print("Run the earlier pipeline steps first (python run.py).")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env (see .env.example).")
        sys.exit(1)

    from llm_analysis import run_analysis

    print("→ Running LLM analysis (transcript-only + multimodal)...")
    run_analysis(
        TRANSCRIPT_FILE,
        AUDIO_FEATURES_FILE,
        VOICE_EMOTION_FILE,
        FACE_EMOTION_FILE,
        OUTPUT_FILE,
        api_key=api_key,
    )
    print(f"✓ Analysis saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_05_llm_analysis.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/05_llm_analysis.py tests/test_05_llm_analysis.py
git commit -m "feat(step5): add CLI entry point with input + API key guards"
```

---

## Task 9: Skip-guarded integration test + run the real step

**Files:**
- Modify: `tests/test_llm_analysis.py`

- [ ] **Step 1: Write the skip-guarded integration test**

Append to `tests/test_llm_analysis.py`:

```python
class TestRunAnalysisIntegration:
    @pytest.mark.skipif(
        not _anthropic_available()
        or not os.path.exists("output/transcript.json")
        or not os.path.exists("output/audio_features.json")
        or not os.path.exists("output/voice_emotion.json")
        or not os.path.exists("output/face_emotion.json"),
        reason="requires anthropic SDK + all four upstream output JSONs",
    )
    def test_with_real_outputs(self, tmp_path):
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        from pipeline.llm_analysis import run_analysis
        out = tmp_path / "analysis.json"
        result = run_analysis(
            "output/transcript.json",
            "output/audio_features.json",
            "output/voice_emotion.json",
            "output/face_emotion.json",
            str(out),
            api_key=api_key,
        )
        assert set(result.keys()) == {"transcript_only", "multimodal"}
        for mode in ("transcript_only", "multimodal"):
            assert 0 <= result[mode]["engagement_score"] <= 100
            assert 0 <= result[mode]["deal_probability"] <= 100
            assert "talk_ratio" in result[mode]
            assert isinstance(result[mode]["critical_moments"], list)
            assert isinstance(result[mode]["recommendations"], list)
```

- [ ] **Step 2: Run the full test suite (integration test skips without the SDK)**

Run: `python -m pytest tests/ -v`
Expected: PASS — all new step-5 unit/CLI tests green; the integration test is SKIPPED (anthropic not installed). Existing 68 tests still green.

- [ ] **Step 3: Install anthropic and run the real step**

```bash
source venv/bin/activate
pip install anthropic
python pipeline/05_llm_analysis.py
```
Expected: prints `→ Running LLM analysis...` then `✓ Analysis saved to output/analysis.json`. Inspect `output/analysis.json` — both `transcript_only` and `multimodal` blocks present with all schema fields; `talk_ratio` injected into both.

- [ ] **Step 4: Run the integration test for real**

Run: `python -m pytest tests/test_llm_analysis.py::TestRunAnalysisIntegration -v`
Expected: PASS (hits the real Claude API; costs tokens)

- [ ] **Step 5: Sanity-check the killer feature**

Open `output/analysis.json` and confirm the contrast is real: the multimodal `critical_moments` should cite timestamps + signal values (e.g. valence/face) that the `transcript_only` block cannot reference. If the contrast is thin, tune `SYSTEM_PROMPT` (Task 5) to push harder on dissonance surfacing — that is the single lever, per the design discussion.

- [ ] **Step 6: Commit**

```bash
git add tests/test_llm_analysis.py
git commit -m "test(step5): add skip-guarded real-output integration test"
```

---

## Self-Review (run after writing, fix inline — already applied)

1. **Spec coverage:** MVP spec `:152-197` — two Claude calls (Task 7), transcript-only + multimodal (Tasks 5+7), model `claude-sonnet-4-6` (Task 6 constant), rate-limit retry once after 10s (Task 6), missing-key fail-fast (Task 8), output schema `{transcript_only, multimodal}` with engagement/deal/talk_ratio/critical_moments/recommendations (Task 7). Speaker mapping + JSON mechanism resolved per "Spec refinements" above. ✓
2. **Placeholder scan:** no TBD/TODO; every code step has full code; no "similar to Task N". ✓
3. **Type/name consistency:** `submit_analysis` tool name used in `ANALYSIS_TOOL`, `_create_message` `tool_choice`, and tests — consistent. `_call_claude(user_prompt, api_key)` signature matches `run_analysis`'s call and the test patches. `run_analysis` default args match the CLI's positional call. ✓
