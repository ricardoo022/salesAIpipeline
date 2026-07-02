# Step 6 — HTML Report Generation Implementation Plan

**Goal:** Read `transcript.json`, `voice_emotion.json`, `face_emotion.json`, and `analysis.json`, and produce a single self-contained `output/report.html` — dark theme, single page, embedded data, Chart.js via CDN, no server needed.

**Architecture:** Follow the established module/script split — `pipeline/report.py` holds pure data-shaping helpers plus `render_report()` which assembles the final HTML string, and `pipeline/06_report.py` is the thin CLI entry point (guard on missing inputs, call the module, write the file). No heavy/optional deps here (no cv2/deepface/torch), so no lazy-import seam is needed — everything is stdlib `json` + string building.

**Tech Stack:** stdlib `json`, Chart.js 4.x (CDN, referenced by URL only — not fetched at build time), pytest.

**Skills applied:**
- **test-driven-development** — every function has a failing test first; RED → GREEN → commit per task
- **writing-plans** — plan structure, bite-sized tasks, completeness/self-review

---

## Reference: existing output shapes (verified against files already on disk)

- `transcript.json`: `[{speaker, start, end, text, words[], ...}]`, speakers are `SPEAKER_00/01/02`.
- `voice_emotion.json`: `[{speaker, start, end, valence, arousal, dominance}]`.
- `face_emotion.json`: `[{timestamp, dominant_emotion, scores{}}]`, one entry per ~10s sample (fewer than expected if frames were skipped for no-face).
- `analysis.json`: `{transcript_only: {engagement_score, deal_probability, talk_ratio{rep,prospect}, critical_moments[{timestamp:"HH:MM:SS", type, description, coaching}], recommendations[]}, multimodal: {...same shape...}}`.

`_classify_speakers` (talk-time ranking → REP/PROSPECT/OTHER) and `_format_timestamp` (seconds → `HH:MM:SS`) already exist as private helpers in `pipeline/llm_analysis.py`. Per this project's established pattern (every module is self-contained, nothing cross-imports a sibling pipeline module), `pipeline/report.py` defines its own small local copies rather than importing from `llm_analysis.py`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pipeline/report.py` | Shared module: time helpers, data-shaping helpers, `render_report()`. Pure Python, no lazy imports needed. |
| `pipeline/06_report.py` | CLI entry point: guard on missing inputs, call module, write `output/report.html`. |
| `tests/test_report.py` | Unit tests for every helper + `render_report()` content assertions. |
| `tests/test_06_report.py` | Subprocess guard test for the CLI (missing inputs → exit 1). |

---

## Task 1: Module scaffold + time helpers

**Files:** Create `pipeline/report.py`, Create `tests/test_report.py`

- [ ] **Step 1: Write failing tests**

```python
import json
import re

import pytest


class TestFormatTimestamp:
    def test_formats_seconds_as_hhmmss(self):
        from pipeline.report import _format_timestamp
        assert _format_timestamp(744) == "00:12:24"

    def test_pads_single_digits(self):
        from pipeline.report import _format_timestamp
        assert _format_timestamp(5) == "00:00:05"

    def test_handles_hours(self):
        from pipeline.report import _format_timestamp
        assert _format_timestamp(3661) == "01:01:01"


class TestParseTimestamp:
    def test_parses_hhmmss_to_seconds(self):
        from pipeline.report import _parse_timestamp
        assert _parse_timestamp("00:12:24") == 744

    def test_round_trips_with_format_timestamp(self):
        from pipeline.report import _format_timestamp, _parse_timestamp
        assert _parse_timestamp(_format_timestamp(3661)) == 3661


class TestClassifySpeakers:
    def test_longest_talk_time_is_rep(self):
        from pipeline.report import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0, "end": 5},
            {"speaker": "SPEAKER_01", "start": 5, "end": 30},
        ]
        result = _classify_speakers(transcript)
        assert result["SPEAKER_01"] == "REP"
        assert result["SPEAKER_00"] == "PROSPECT"

    def test_third_speaker_is_other(self):
        from pipeline.report import _classify_speakers
        transcript = [
            {"speaker": "SPEAKER_00", "start": 0, "end": 30},
            {"speaker": "SPEAKER_01", "start": 30, "end": 40},
            {"speaker": "SPEAKER_02", "start": 40, "end": 42},
        ]
        result = _classify_speakers(transcript)
        assert result["SPEAKER_02"] == "OTHER"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'pipeline.report'`.

- [ ] **Step 3: Write the module scaffold + time helpers**

Create `pipeline/report.py`:

```python
"""HTML report generation for step 6.

Reads the four upstream JSONs and renders a single self-contained
output/report.html: dark theme, embedded data, Chart.js via CDN, no server
needed. Pure stdlib (json + string building) -- no heavy deps, no lazy
imports required.
"""

import json


def _format_timestamp(seconds):
    """Format seconds as HH:MM:SS (matches step 5's critical_moments format)."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_timestamp(ts):
    """Parse an HH:MM:SS string back to seconds (inverse of _format_timestamp)."""
    h, m, s = (int(part) for part in ts.split(":"))
    return h * 3600 + m * 60 + s


def _classify_speakers(transcript):
    """Map diarization labels to REP / PROSPECT / OTHER by total talk time.

    Same deterministic ranking as pipeline/llm_analysis.py's
    _classify_speakers -- duplicated locally per this project's
    self-contained-module convention (see pipeline/report.py module docstring
    and CLAUDE.md's module/script split).
    """
    totals = {}
    for seg in transcript:
        sp = seg.get("speaker", "UNKNOWN")
        totals[sp] = totals.get(sp, 0.0) + (seg.get("end", 0) - seg.get("start", 0))
    ranked = sorted(totals, key=lambda s: totals[s], reverse=True)
    mapping = {}
    for i, sp in enumerate(ranked):
        mapping[sp] = "REP" if i == 0 else ("PROSPECT" if i == 1 else "OTHER")
    return mapping
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/report.py tests/test_report.py
git commit -m "feat(step6): add report module scaffold with timestamp + speaker helpers"
```

---

## Task 2: `_meeting_duration` + `_count_missing_face_frames`

**Files:** Modify `pipeline/report.py`, Modify `tests/test_report.py`

- [ ] **Step 1: Write failing tests (append to `tests/test_report.py`)**

```python
class TestMeetingDuration:
    def test_returns_max_end_time(self):
        from pipeline.report import _meeting_duration
        transcript = [{"start": 0, "end": 10}, {"start": 10, "end": 25.5}]
        assert _meeting_duration(transcript) == 25.5

    def test_empty_transcript_returns_zero(self):
        from pipeline.report import _meeting_duration
        assert _meeting_duration([]) == 0


class TestCountMissingFaceFrames:
    def test_no_missing_frames(self):
        from pipeline.report import _count_missing_face_frames
        face_emotion = [{"timestamp": t} for t in (0.0, 10.0, 20.0)]
        assert _count_missing_face_frames(face_emotion, duration=25, interval=10) == 0

    def test_counts_missing_frames(self):
        from pipeline.report import _count_missing_face_frames
        face_emotion = [{"timestamp": 0.0}]
        assert _count_missing_face_frames(face_emotion, duration=25, interval=10) == 2

    def test_never_negative(self):
        from pipeline.report import _count_missing_face_frames
        face_emotion = [{"timestamp": t} for t in (0.0, 10.0, 20.0, 30.0, 40.0)]
        assert _count_missing_face_frames(face_emotion, duration=25, interval=10) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py::TestMeetingDuration tests/test_report.py::TestCountMissingFaceFrames -v`
Expected: ERROR — `ImportError: cannot import name '_meeting_duration'`.

- [ ] **Step 3: Write the helpers (append to `pipeline/report.py`, after `_classify_speakers`)**

```python
def _meeting_duration(transcript):
    """Total meeting duration in seconds: the latest segment end time."""
    if not transcript:
        return 0
    return max(seg.get("end", 0) for seg in transcript)


def _count_missing_face_frames(face_emotion, duration, interval=10):
    """Expected sample count (one every `interval`s, starting at 0) minus actual.

    Never negative -- extra/duplicate samples don't count as "missing".
    """
    expected = int(duration // interval) + 1
    return max(0, expected - len(face_emotion))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py::TestMeetingDuration tests/test_report.py::TestCountMissingFaceFrames -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/report.py tests/test_report.py
git commit -m "feat(step6): add meeting duration and missing-face-frame helpers"
```

---

## Task 3: `_build_timeline_series`

**Files:** Modify `pipeline/report.py`, Modify `tests/test_report.py`

- [ ] **Step 1: Write failing tests (append to `tests/test_report.py`)**

```python
class TestBuildTimelineSeries:
    def test_splits_by_role_and_metric(self):
        from pipeline.report import _build_timeline_series
        voice_emotion = [
            {"speaker": "SPEAKER_00", "start": 0, "end": 5, "valence": 0.5, "arousal": 0.4, "dominance": 0.3},
            {"speaker": "SPEAKER_01", "start": 5, "end": 10, "valence": 0.6, "arousal": 0.7, "dominance": 0.2},
        ]
        speaker_map = {"SPEAKER_00": "REP", "SPEAKER_01": "PROSPECT"}
        result = _build_timeline_series(voice_emotion, speaker_map)
        assert result["prospect_valence"] == [{"x": 5, "y": 0.6}]
        assert result["prospect_arousal"] == [{"x": 5, "y": 0.7}]
        assert result["rep_arousal"] == [{"x": 0, "y": 0.4}]

    def test_ignores_other_speakers(self):
        from pipeline.report import _build_timeline_series
        voice_emotion = [
            {"speaker": "SPEAKER_02", "start": 0, "end": 5, "valence": 0.5, "arousal": 0.4, "dominance": 0.3},
        ]
        speaker_map = {"SPEAKER_02": "OTHER"}
        result = _build_timeline_series(voice_emotion, speaker_map)
        assert result["prospect_valence"] == []
        assert result["rep_arousal"] == []

    def test_x_is_segment_start_rounded(self):
        from pipeline.report import _build_timeline_series
        voice_emotion = [
            {"speaker": "SPEAKER_00", "start": 12.456, "end": 15, "valence": 0.5, "arousal": 0.4, "dominance": 0.3},
        ]
        speaker_map = {"SPEAKER_00": "PROSPECT"}
        result = _build_timeline_series(voice_emotion, speaker_map)
        assert result["prospect_valence"][0]["x"] == 12.46
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py::TestBuildTimelineSeries -v`
Expected: ERROR — `ImportError: cannot import name '_build_timeline_series'`.

- [ ] **Step 3: Write the helper (append to `pipeline/report.py`, after `_count_missing_face_frames`)**

```python
def _build_timeline_series(voice_emotion, speaker_map):
    """Shape voice_emotion into the three Chart.js series the spec calls for:
    Prospect Valence, Prospect Arousal, Rep Arousal (labelled "Rep Energy" in UI).

    x = segment start time (seconds, rounded to 2dp); y = the metric value.
    """
    series = {"prospect_valence": [], "prospect_arousal": [], "rep_arousal": []}
    for seg in voice_emotion:
        role = speaker_map.get(seg.get("speaker", ""), "OTHER")
        x = round(seg.get("start", 0), 2)
        if role == "PROSPECT":
            series["prospect_valence"].append({"x": x, "y": seg["valence"]})
            series["prospect_arousal"].append({"x": x, "y": seg["arousal"]})
        elif role == "REP":
            series["rep_arousal"].append({"x": x, "y": seg["arousal"]})
    return series
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py::TestBuildTimelineSeries -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/report.py tests/test_report.py
git commit -m "feat(step6): add engagement timeline series builder"
```

---

## Task 4: `_build_moment_markers`

**Files:** Modify `pipeline/report.py`, Modify `tests/test_report.py`

- [ ] **Step 1: Write failing tests (append to `tests/test_report.py`)**

```python
class TestBuildMomentMarkers:
    def test_builds_marker_per_moment(self):
        from pipeline.report import _build_moment_markers
        moments = [
            {"timestamp": "00:12:24", "type": "pricing_objection", "description": "d", "coaching": "c"},
        ]
        result = _build_moment_markers(moments)
        assert result == [{"x": 744, "timestamp": "00:12:24", "type": "pricing_objection"}]

    def test_preserves_order(self):
        from pipeline.report import _build_moment_markers
        moments = [
            {"timestamp": "00:00:07", "type": "a", "description": "", "coaching": ""},
            {"timestamp": "00:01:00", "type": "b", "description": "", "coaching": ""},
        ]
        result = _build_moment_markers(moments)
        assert [m["x"] for m in result] == [7, 60]

    def test_empty_list_returns_empty(self):
        from pipeline.report import _build_moment_markers
        assert _build_moment_markers([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py::TestBuildMomentMarkers -v`
Expected: ERROR — `ImportError: cannot import name '_build_moment_markers'`.

- [ ] **Step 3: Write the helper (append to `pipeline/report.py`, after `_build_timeline_series`)**

```python
def _build_moment_markers(critical_moments):
    """Chart annotation points for critical moments: {x (seconds), timestamp, type}."""
    return [
        {"x": _parse_timestamp(m["timestamp"]), "timestamp": m["timestamp"], "type": m["type"]}
        for m in critical_moments
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py::TestBuildMomentMarkers -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/report.py tests/test_report.py
git commit -m "feat(step6): add critical moment chart marker builder"
```

---

## Task 5: `_build_comparison`

**Files:** Modify `pipeline/report.py`, Modify `tests/test_report.py`

- [ ] **Step 1: Write failing tests (append to `tests/test_report.py`)**

```python
class TestBuildComparison:
    def test_computes_deltas(self):
        from pipeline.report import _build_comparison
        analysis = {
            "transcript_only": {"engagement_score": 61, "deal_probability": 45, "talk_ratio": {"rep": 62, "prospect": 38}},
            "multimodal": {"engagement_score": 74, "deal_probability": 68, "talk_ratio": {"rep": 62, "prospect": 38}},
        }
        result = _build_comparison(analysis)
        assert result["engagement_score"] == {"transcript_only": 61, "multimodal": 74, "delta": 13}
        assert result["deal_probability"] == {"transcript_only": 45, "multimodal": 68, "delta": 23}
        assert result["talk_ratio"] == {"rep": 62, "prospect": 38}

    def test_negative_delta(self):
        from pipeline.report import _build_comparison
        analysis = {
            "transcript_only": {"engagement_score": 80, "deal_probability": 50, "talk_ratio": {"rep": 50, "prospect": 50}},
            "multimodal": {"engagement_score": 70, "deal_probability": 50, "talk_ratio": {"rep": 50, "prospect": 50}},
        }
        result = _build_comparison(analysis)
        assert result["engagement_score"]["delta"] == -10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py::TestBuildComparison -v`
Expected: ERROR — `ImportError: cannot import name '_build_comparison'`.

- [ ] **Step 3: Write the helper (append to `pipeline/report.py`, after `_build_moment_markers`)**

```python
def _build_comparison(analysis):
    """Hero + side-by-side comparison data: transcript-only vs multimodal, with deltas.

    talk_ratio is identical in both modes (deterministic, see CLAUDE.md), so it
    is reported once rather than duplicated per mode.
    """
    t = analysis["transcript_only"]
    m = analysis["multimodal"]
    return {
        "engagement_score": {
            "transcript_only": t["engagement_score"],
            "multimodal": m["engagement_score"],
            "delta": m["engagement_score"] - t["engagement_score"],
        },
        "deal_probability": {
            "transcript_only": t["deal_probability"],
            "multimodal": m["deal_probability"],
            "delta": m["deal_probability"] - t["deal_probability"],
        },
        "talk_ratio": m["talk_ratio"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py::TestBuildComparison -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/report.py tests/test_report.py
git commit -m "feat(step6): add transcript-only vs multimodal comparison builder"
```

---

## Task 6: `render_report` (orchestration + HTML assembly)

**Files:** Modify `pipeline/report.py`, Modify `tests/test_report.py`

- [ ] **Step 1: Write failing tests (append to `tests/test_report.py`)**

```python
def _sample_inputs():
    transcript = [
        {"speaker": "SPEAKER_00", "start": 0, "end": 600, "text": "rep talking"},
        {"speaker": "SPEAKER_01", "start": 600, "end": 900, "text": "prospect talking"},
    ]
    voice_emotion = [
        {"speaker": "SPEAKER_00", "start": 0, "end": 5, "valence": 0.5, "arousal": 0.4, "dominance": 0.3},
        {"speaker": "SPEAKER_01", "start": 600, "end": 605, "valence": 0.3, "arousal": 0.6, "dominance": 0.4},
    ]
    face_emotion = [{"timestamp": 0.0, "dominant_emotion": "neutral", "scores": {"neutral": 0.9}}]
    analysis = {
        "transcript_only": {
            "engagement_score": 61, "deal_probability": 45,
            "talk_ratio": {"rep": 67, "prospect": 33},
            "critical_moments": [
                {"timestamp": "00:10:00", "type": "pricing_objection",
                 "description": "Prospect questioned pricing", "coaching": "Rep should have paused."},
            ],
            "recommendations": ["Ask more open questions."],
        },
        "multimodal": {
            "engagement_score": 74, "deal_probability": 68,
            "talk_ratio": {"rep": 67, "prospect": 33},
            "critical_moments": [
                {"timestamp": "00:10:00", "type": "pricing_objection",
                 "description": "Voice valence dropped to 0.30 while saying 'sounds good'.",
                 "coaching": "Rep should have paused and addressed the mismatch."},
            ],
            "recommendations": ["Pause after pricing to let discomfort surface.", "Mirror the prospect's pace."],
        },
    }
    return transcript, voice_emotion, face_emotion, analysis


class TestRenderReport:
    def test_returns_html_document(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_includes_chartjs_cdn(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "cdn.jsdelivr.net/npm/chart.js" in html

    def test_includes_hero_metrics(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "74" in html  # multimodal engagement_score
        assert "68" in html  # multimodal deal_probability

    def test_includes_side_by_side_delta(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "+13" in html  # engagement_score delta
        assert "+23" in html  # deal_probability delta

    def test_includes_critical_moment_and_coaching(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "00:10:00" in html
        assert "Rep should have paused and addressed the mismatch." in html

    def test_includes_recommendations(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "Pause after pricing to let discomfort surface." in html
        assert "Mirror the prospect&#x27;s pace." in html or "Mirror the prospect's pace." in html

    def test_includes_footer(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        assert "WhisperX" in html and "audeering" in html and "DeepFace" in html and "Claude" in html

    def test_embedded_timeline_json_is_valid(self):
        from pipeline.report import render_report
        html = render_report(*_sample_inputs())
        match = re.search(r"window\.REPORT_DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        assert match is not None
        data = json.loads(match.group(1))
        assert "timeline" in data
        assert "moment_markers" in data
        assert data["timeline"]["prospect_valence"] == [{"x": 600, "y": 0.3}]
        assert data["moment_markers"][0]["x"] == 600

    def test_handles_missing_face_data_gracefully(self):
        from pipeline.report import render_report
        transcript, voice_emotion, face_emotion, analysis = _sample_inputs()
        html = render_report(transcript, voice_emotion, [], analysis)
        assert "facial data unavailable" in html.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py::TestRenderReport -v`
Expected: ERROR — `ImportError: cannot import name 'render_report'`.

- [ ] **Step 3: Write `render_report` (append to `pipeline/report.py`, after `_build_comparison`)**

```python
def render_report(transcript, voice_emotion, face_emotion, analysis, meeting_title="Sales Meeting"):
    """Assemble the single self-contained output/report.html.

    All chart-facing data (timeline series + critical moment markers) is
    embedded as window.REPORT_DATA and rendered client-side by Chart.js
    (CDN). Everything else (hero metrics, comparison cards, critical moment
    cards, recommendations, footer) is rendered directly into the HTML
    string -- no template engine, per the project's "no framework" design.
    """
    speaker_map = _classify_speakers(transcript)
    duration = _meeting_duration(transcript)
    missing_faces = _count_missing_face_frames(face_emotion, duration)
    timeline = _build_timeline_series(voice_emotion, speaker_map)
    markers = _build_moment_markers(analysis["multimodal"]["critical_moments"])
    comparison = _build_comparison(analysis)

    report_data = json.dumps({"timeline": timeline, "moment_markers": markers})

    def esc(s):
        return (
            str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#x27;")
        )

    def fmt_delta(n):
        return f"+{n}" if n >= 0 else str(n)

    def moment_cards(moments):
        return "\n".join(
            f"""<div class="moment-card">
  <div class="moment-timestamp">[{esc(m['timestamp'])}]</div>
  <div class="moment-type">{esc(m['type'])}</div>
  <div class="moment-description">{esc(m['description'])}</div>
  <div class="moment-coaching">{esc(m['coaching'])}</div>
</div>"""
            for m in moments
        )

    def recommendation_items(recs):
        return "\n".join(f"<li>{esc(r)}</li>" for r in recs)

    def insight_cards(recs, weight_class):
        items = "".join(f'<div class="insight-card {weight_class}">{esc(r)}</div>' for r in recs[:4])
        return items

    face_note = (
        f'<p class="face-note">facial data unavailable for {missing_faces} frame(s)</p>'
        if missing_faces > 0
        else ""
    )

    m = analysis["multimodal"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{esc(meeting_title)} — Sales Meeting Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ background: #0e0e12; color: #e4e4e7; font-family: system-ui, sans-serif; margin: 0; padding: 0 24px 48px; }}
  header {{ padding: 24px 0; border-bottom: 1px solid #2a2a33; }}
  .hero {{ display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }}
  .hero-card {{ background: #17171f; border-radius: 12px; padding: 20px; flex: 1; min-width: 160px; }}
  .hero-card .value {{ font-size: 2.2rem; font-weight: 700; }}
  .comparison {{ display: flex; gap: 24px; margin: 32px 0; }}
  .comparison-col {{ flex: 1; }}
  .comparison-col.transcript-only {{ opacity: 0.6; }}
  .comparison-col.multimodal {{ opacity: 1; }}
  .insight-card {{ background: #17171f; border-radius: 8px; padding: 14px; margin-bottom: 10px; }}
  .insight-card.thin {{ padding: 8px 14px; font-size: 0.85rem; color: #9a9aa5; }}
  .insight-card.heavy {{ padding: 18px; font-size: 1rem; border-left: 3px solid #6366f1; }}
  .moment-card {{ background: #17171f; border-radius: 8px; padding: 16px; margin-bottom: 12px; }}
  .moment-timestamp {{ color: #6366f1; font-weight: 600; }}
  .face-note {{ color: #9a9aa5; font-size: 0.85rem; }}
  footer {{ margin-top: 48px; padding: 24px 0; border-top: 1px solid #2a2a33; color: #9a9aa5; font-size: 0.85rem; }}
</style>
</head>
<body>
<header>
  <h1>{esc(meeting_title)}</h1>
  <p>Duration: {esc(_format_timestamp(duration))} · REP vs PROSPECT</p>
</header>

<section class="hero">
  <div class="hero-card"><div class="label">Engagement Score</div><div class="value">{m['engagement_score']}</div></div>
  <div class="hero-card"><div class="label">Deal Probability</div><div class="value">{m['deal_probability']}</div></div>
  <div class="hero-card"><div class="label">Talk Ratio</div><div class="value">{comparison['talk_ratio']['rep']}/{comparison['talk_ratio']['prospect']}</div></div>
  <div class="hero-card"><div class="label">Duration</div><div class="value">{esc(_format_timestamp(duration))}</div></div>
</section>

<section class="comparison">
  <div class="comparison-col transcript-only">
    <h2>Transcript-Only</h2>
    <div class="hero-card"><div class="label">Engagement</div><div class="value">{comparison['engagement_score']['transcript_only']}</div></div>
    <div class="hero-card"><div class="label">Deal Probability</div><div class="value">{comparison['deal_probability']['transcript_only']}</div></div>
    {insight_cards(analysis['transcript_only']['recommendations'], 'thin')}
  </div>
  <div class="comparison-col multimodal">
    <h2>Multimodal</h2>
    <div class="hero-card"><div class="label">Engagement</div><div class="value">{comparison['engagement_score']['multimodal']} <span class="delta">({fmt_delta(comparison['engagement_score']['delta'])} pts)</span></div></div>
    <div class="hero-card"><div class="label">Deal Probability</div><div class="value">{comparison['deal_probability']['multimodal']} <span class="delta">({fmt_delta(comparison['deal_probability']['delta'])}%)</span></div></div>
    {insight_cards(analysis['multimodal']['recommendations'], 'heavy')}
  </div>
</section>

<section class="timeline">
  <h2>Engagement Timeline</h2>
  {face_note}
  <canvas id="timelineChart" height="100"></canvas>
</section>

<section class="critical-moments">
  <h2>Critical Moments</h2>
  {moment_cards(m['critical_moments'])}
</section>

<section class="recommendations">
  <h2>Coaching Recommendations</h2>
  <ol>
  {recommendation_items(m['recommendations'])}
  </ol>
</section>

<footer>
  Powered by multimodal analysis: WhisperX · audeering · DeepFace · Claude
</footer>

<script>
window.REPORT_DATA = {report_data};
const ctx = document.getElementById('timelineChart');
new Chart(ctx, {{
  type: 'line',
  data: {{
    datasets: [
      {{ label: 'Prospect Valence', data: window.REPORT_DATA.timeline.prospect_valence, borderColor: '#3b82f6', parsing: false }},
      {{ label: 'Prospect Arousal', data: window.REPORT_DATA.timeline.prospect_arousal, borderColor: '#f59e0b', parsing: false }},
      {{ label: 'Rep Energy', data: window.REPORT_DATA.timeline.rep_arousal, borderColor: '#9ca3af', parsing: false }},
    ],
  }},
  options: {{ scales: {{ x: {{ type: 'linear' }} }} }},
}});
</script>
</body>
</html>"""
    return html
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py::TestRenderReport -v`
Expected: 9 PASS

- [ ] **Step 5: Run the full module test file**

Run: `python -m pytest tests/test_report.py -v`
Expected: all tests PASS (5 + 5 + 3 + 3 + 2 + 9 = 27)

- [ ] **Step 6: Commit**

```bash
git add pipeline/report.py tests/test_report.py
git commit -m "feat(step6): add render_report HTML assembly"
```

---

## Task 7: CLI script `pipeline/06_report.py` + guard test

**Files:** Create `pipeline/06_report.py`, Create `tests/test_06_report.py`

- [ ] **Step 1: Write the failing test + a stub script (honest RED)**

Create `tests/test_06_report.py` (mirrors `tests/test_05_llm_analysis.py`):

```python
import os
import sys
import subprocess

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(PROJECT_ROOT, "pipeline/06_report.py")


def test_exits_when_inputs_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "analysis.json" in result.stdout


def test_writes_report_html(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "transcript.json").write_text('[{"speaker": "SPEAKER_00", "start": 0, "end": 5, "text": "hi"}]')
    (out / "voice_emotion.json").write_text('[]')
    (out / "face_emotion.json").write_text('[]')
    (out / "analysis.json").write_text(
        '{"transcript_only": {"engagement_score": 50, "deal_probability": 50, '
        '"talk_ratio": {"rep": 100, "prospect": 0}, "critical_moments": [], "recommendations": []}, '
        '"multimodal": {"engagement_score": 50, "deal_probability": 50, '
        '"talk_ratio": {"rep": 100, "prospect": 0}, "critical_moments": [], "recommendations": []}}'
    )
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 0
    assert (out / "report.html").exists()
```

Create `pipeline/06_report.py` as a minimal stub (no guard yet):

```python
#!/usr/bin/env python3
"""Step 6: Generate the HTML report."""


def main():
    pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_06_report.py -v`
Expected: both FAIL — the stub's `main()` does nothing (exits 0, writes no file).

- [ ] **Step 3: Write the full CLI (replace the stub body of `main()`)**

Replace `pipeline/06_report.py` with:

```python
#!/usr/bin/env python3
"""Step 6: Generate the HTML report.

Reads the four upstream JSONs (transcript, voice_emotion, face_emotion,
analysis) and writes a single self-contained output/report.html.

Output: output/report.html
"""

import json
import os
import sys

TRANSCRIPT_FILE = "output/transcript.json"
VOICE_EMOTION_FILE = "output/voice_emotion.json"
FACE_EMOTION_FILE = "output/face_emotion.json"
ANALYSIS_FILE = "output/analysis.json"
OUTPUT_FILE = "output/report.html"

REQUIRED_INPUTS = (
    TRANSCRIPT_FILE,
    VOICE_EMOTION_FILE,
    FACE_EMOTION_FILE,
    ANALYSIS_FILE,
)


def main():
    missing = [p for p in REQUIRED_INPUTS if not os.path.exists(p)]
    if missing:
        print(f"ERROR: missing input(s): {', '.join(missing)}")
        print("Run the earlier pipeline steps first (python run.py).")
        sys.exit(1)

    from report import render_report

    with open(TRANSCRIPT_FILE) as f:
        transcript = json.load(f)
    with open(VOICE_EMOTION_FILE) as f:
        voice_emotion = json.load(f)
    with open(FACE_EMOTION_FILE) as f:
        face_emotion = json.load(f)
    with open(ANALYSIS_FILE) as f:
        analysis = json.load(f)

    print("→ Rendering report...")
    html = render_report(transcript, voice_emotion, face_emotion, analysis)

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(html)

    print(f"✓ Report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

Note: `from report import render_report` is a bare import (script's own `pipeline/` dir is on `sys.path` when run directly), mirroring steps 3–5.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_06_report.py -v`
Expected: 2 PASS

- [ ] **Step 5: Run the full test suite for regressions**

Run: `python -m pytest tests/ -v`
Expected: all previous tests PASS + new step-6 tests PASS. No regressions.

- [ ] **Step 6: Commit**

```bash
git add pipeline/06_report.py tests/test_06_report.py
git commit -m "feat(step6): add CLI entry point for report generation"
```

---

## Final Verification

- [ ] Confirm `run.py` already wires step 6 (`run.py` STEPS list, step `n=6`) — no change needed.
- [ ] Run full suite: `python -m pytest tests/ -v`
- [ ] End-to-end: since `output/transcript.json`, `output/voice_emotion.json`, `output/face_emotion.json`, `output/analysis.json` already exist on disk from prior steps, run `python pipeline/06_report.py` directly and inspect `output/report.html` opens and contains real data (engagement score, deal probability, critical moments, recommendations, timeline chart).

---

## Self-Review

### 1. Spec coverage

| Spec requirement (Step 6 / Report Layout) | Task(s) |
|---|---|
| Header bar: title, duration, speaker labels | Task 6 — `render_report` header section |
| Hero metrics row (4 cards, multimodal numbers) | Task 5 (`_build_comparison`) + Task 6 (hero section) |
| Side-by-side comparison with delta indicators | Task 5 (`_build_comparison` deltas) + Task 6 (comparison section, thin vs heavy insight cards) |
| Engagement Timeline: 3 Chart.js lines (Prospect Valence/Arousal, Rep Arousal="Rep Energy") | Task 3 (`_build_timeline_series`) + Task 6 (chart init JS) |
| Critical moment markers on timeline | Task 4 (`_build_moment_markers`) + Task 6 (embedded in REPORT_DATA) |
| Critical Moments list (cards: timestamp/type/what happened/coaching) | Task 6 — `moment_cards()` |
| Coaching Recommendations numbered list | Task 6 — `recommendation_items()` |
| Footer credit line | Task 6 — footer section |
| Report handles missing face data gracefully | Task 2 (`_count_missing_face_frames`) + Task 6 (`face_note`) |
| Single self-contained HTML, embedded data, Chart.js via CDN, no server | Task 6 — `render_report` returns one HTML string, `window.REPORT_DATA` embedded JSON, CDN `<script src>` |
| CLI guard: missing inputs → exit 1 with message | Task 7 |
| `run.py` orchestration | Already wired (verified in Final Verification) |

### 2. Placeholder scan

No TBD / TODO / "implement later" anywhere. Every code step is complete, runnable code.

### 3. Type consistency

- `_format_timestamp(seconds) -> str "HH:MM:SS"` / `_parse_timestamp(str) -> int seconds` — inverse pair, round-trip tested in Task 1.
- `_classify_speakers(transcript) -> dict[str,str]` — consumed by `_build_timeline_series` (Task 3) and `render_report` (Task 6).
- `_build_timeline_series(...) -> {"prospect_valence": [{x,y}], "prospect_arousal": [...], "rep_arousal": [...]}` — consumed directly by `render_report`'s embedded `REPORT_DATA.timeline` and the Chart.js dataset config.
- `_build_moment_markers(critical_moments) -> [{"x", "timestamp", "type"}]` — embedded as `REPORT_DATA.moment_markers`.
- `_build_comparison(analysis) -> {"engagement_score": {...}, "deal_probability": {...}, "talk_ratio": {...}}` — consumed by `render_report`'s hero + comparison sections.
- `render_report(transcript, voice_emotion, face_emotion, analysis, meeting_title=...) -> str` — signature matches the CLI's call in Task 7.

### Gaps found and fixed

- Initial design considered importing `_classify_speakers`/`_format_timestamp` from `pipeline/llm_analysis.py`, but every existing pipeline module is self-contained (verified via `grep` — no module imports a sibling pipeline module). Fixed by duplicating the two small pure functions locally in `pipeline/report.py`, consistent with the established pattern.
- `_count_missing_face_frames` needed a duration argument (not derivable from `face_emotion.json` alone, since a fully-empty result and a fully-quiet meeting look identical) — sourced from `_meeting_duration(transcript)` in `render_report`.
