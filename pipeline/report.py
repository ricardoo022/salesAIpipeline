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
        return "".join(f'<div class="insight-card {weight_class}">{esc(r)}</div>' for r in recs[:4])

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
