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


def _build_moment_markers(critical_moments):
    """Chart annotation points for critical moments: {x (seconds), timestamp, type}."""
    return [
        {"x": _parse_timestamp(m["timestamp"]), "timestamp": m["timestamp"], "type": m["type"]}
        for m in critical_moments
    ]
