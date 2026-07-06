"""HTML report generation for step 6.

Reads the five upstream JSONs and renders a single self-contained
output/report.html: dark theme, embedded data, Chart.js via CDN, no server
needed. Pure stdlib (json + string building) -- no heavy deps, no lazy
imports required.

Report copy is European Portuguese, written for a non-technical sales
audience. Every measured signal (prosody, voice VAD, facial emotion) is
translated into a plain "o que significa quando esta alto/baixo" reading,
not left as a raw statistic -- see the GLOSSARY_SPEC and FACE_MEANINGS_PT
tables below. The LLM's own generated text (critical_moments, recommendations)
is rendered verbatim in whatever language step 5 produced it in, the same
way transcript quotes are kept verbatim: it is evidence, not report copy.
"""

import json

FACE_LABELS_PT = {
    "neutral": "neutro",
    "happy": "feliz",
    "sad": "triste",
    "angry": "zangado",
    "surprise": "surpreendido",
    "fear": "com medo",
    "disgust": "com nojo",
}

FACE_CATEGORIES = ("neutral", "sad", "happy", "angry", "surprise", "disgust", "fear")

FACE_MEANINGS_PT = {
    "neutral": "Está a ouvir, sem confirmar nem negar. É comum numa reunião de descoberta.",
    "sad": "Preocupação ou ceticismo, não é raiva. Pede uma pergunta aberta, não uma réplica.",
    "happy": "Interesse genuíno e boa receção ao que está a ser dito.",
    "angry": "Uma objeção forte a formar-se, normalmente a volta de perguntas de risco ou preço.",
    "surprise": "Reação a algo inesperado.",
    "disgust": "Rejeição forte a algo que foi dito.",
    "fear": "Insegurança aguda.",
}

# Each row: (signal label with its raw field name, what a high reading means,
# what a low reading means, which per-role table it comes from, the field
# name in that table, a format string, and the three comparison sentence
# templates used by _compare_role_stat).
GLOSSARY_SPEC = [
    {
        "signal": "Tom de voz (Pitch Mean)",
        "high": "Voz mais aguda, sinal de entusiasmo ou tensão",
        "low": "Voz mais grave, sinal de calma ou cansaço",
        "source": "prosody", "field": "pitch_mean", "fmt": "{:.0f} Hz",
        "prospect_higher": "O prospect fala mais agudo que o consultor ({prospect} contra {rep})",
        "rep_higher": "O consultor fala mais agudo que o prospect ({rep} contra {prospect})",
        "similar": "Tom semelhante nos dois falantes ({rep} consultor, {prospect} prospect)",
    },
    {
        "signal": "Variação do tom (Pitch Std)",
        "high": "O tom sobe e desce muito, sinal de forte envolvimento emocional",
        "low": "O tom fica quase plano, sinal de discurso ensaiado ou desligado",
        "source": "prosody", "field": "pitch_std", "fmt": "{:.1f}",
        "prospect_higher": "O prospect varia o tom {ratio}x mais que o consultor ({prospect} contra {rep})",
        "rep_higher": "O consultor varia o tom {ratio}x mais que o prospect ({rep} contra {prospect})",
        "similar": "Variação de tom semelhante nos dois ({rep} consultor, {prospect} prospect)",
    },
    {
        "signal": "Energia da voz (Energy Mean)",
        "high": "Ênfase e entusiasmo, por vezes agressividade",
        "low": "Cansaço ou desinteresse",
        "source": "prosody", "field": "energy_mean", "fmt": "{:.3f}",
        "prospect_higher": "O prospect fala com mais energia na voz ({prospect} contra {rep})",
        "rep_higher": "O consultor fala com mais energia na voz ({rep} contra {prospect})",
        "similar": "Energia semelhante nos dois falantes ({rep} consultor, {prospect} prospect)",
    },
    {
        "signal": "Ritmo de fala (Speech Rate)",
        "high": "Ansiedade ou entusiasmo",
        "low": "Hesitação ou cautela a pensar na resposta",
        "source": "prosody", "field": "speech_rate", "fmt": "{:.1f} palavras/seg",
        "prospect_higher": "O prospect fala mais depressa que o consultor ({prospect} contra {rep})",
        "rep_higher": "O consultor fala mais depressa que o prospect ({rep} contra {prospect})",
        "similar": "Ritmo semelhante nos dois ({rep} consultor, {prospect} prospect)",
    },
    {
        "signal": "Pausas entre palavras (Pause Ratio)",
        "high": "Nervosismo, ou simplesmente tempo a pensar",
        "low": "Discurso ensaiado ou dito sob pressão",
        "source": "prosody", "field": "pause_ratio", "fmt": "{:.0%}",
        "prospect_higher": "O prospect pausa mais que o consultor ({prospect} do tempo contra {rep})",
        "rep_higher": "O consultor pausa mais que o prospect ({rep} do tempo contra {prospect})",
        "similar": "Pausas semelhantes nos dois ({rep} consultor, {prospect} prospect)",
    },
    {
        "signal": "Aspereza da voz (ZCR)",
        "high": "Voz mais tensa e agitada",
        "low": "Voz suave e calma",
        "source": "prosody", "field": "zcr", "fmt": "{:.3f}",
        "prospect_higher": "A voz do prospect soa mais tensa que a do consultor ({prospect} contra {rep})",
        "rep_higher": "A voz do consultor soa mais tensa que a do prospect ({rep} contra {prospect})",
        "similar": "Aspereza semelhante nas duas vozes ({rep} consultor, {prospect} prospect)",
    },
    {
        "signal": "Positividade da voz (Valência)",
        "high": "Concordância genuína",
        "low": "Desconforto ou ceticismo, mesmo quando a pessoa diz que sim",
        "source": "vad", "field": "valence", "fmt": "{:.0%}",
        "prospect_higher": "O prospect soa mais positivo que o consultor ({prospect} contra {rep})",
        "rep_higher": "O consultor soa mais positivo que o prospect ({rep} contra {prospect})",
        "similar": "Positividade semelhante nos dois ({rep} consultor, {prospect} prospect)",
    },
    {
        "signal": "Energia emocional (Arousal)",
        "high": "Envolvido e atento",
        "low": "Desligado, a ouvir por educação",
        "source": "vad", "field": "arousal", "fmt": "{:.0%}",
        "prospect_higher": "O prospect soa mais desperto que o consultor ({prospect} contra {rep})",
        "rep_higher": "O consultor soa mais desperto que o prospect ({rep} contra {prospect})",
        "similar": "Energia emocional semelhante nos dois ({rep} consultor, {prospect} prospect)",
    },
    {
        "signal": "Assertividade (Dominance)",
        "high": "Confiante, ou uma objeção a começar a formar-se",
        "low": "Submisso ou inseguro",
        "source": "vad", "field": "dominance", "fmt": "{:.0%}",
        "prospect_higher": "O prospect soa mais assertivo que o consultor ({prospect} contra {rep})",
        "rep_higher": "O consultor soa mais assertivo que o prospect ({rep} contra {prospect})",
        "similar": "Assertividade semelhante nos dois ({rep} consultor, {prospect} prospect)",
    },
]


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


def _count_dissonance_moments(critical_moments):
    """Count moments the LLM tagged as words-vs-tone/face mismatches.

    The multimodal prompt (pipeline/llm_analysis.py's _build_multimodal_prompt)
    explicitly instructs the model to surface these as "Dissonance" moments --
    the one category transcript-only analysis structurally cannot produce,
    since it has no voice/face signal to contrast against the words.
    """
    return sum(1 for m in critical_moments if "dissonance" in m["type"].lower())


def _select_dissonance_examples(critical_moments, limit=2):
    """Pick the first `limit` dissonance-type moments, in chronological order."""
    return [m for m in critical_moments if "dissonance" in m["type"].lower()][:limit]


def _nearest_entry(items, timestamp, time_key="start"):
    """Find the item whose time range contains `timestamp`, else the closest by start.

    Generic lookup used to pull the real transcript quote / voice reading /
    face reading nearest a critical moment's timestamp, straight from the
    upstream JSONs -- not from the LLM's paraphrased description.
    """
    if not items:
        return None
    for item in items:
        if "end" in item and item[time_key] <= timestamp <= item["end"]:
            return item
    return min(items, key=lambda item: abs(item[time_key] - timestamp))


def _describe_tone(valence):
    """Translate a raw valence float into a plain-language, percentage-first label."""
    pct = round(valence * 100)
    if valence > 0.55:
        label = "positivo"
    elif valence < 0.45:
        label = "negativo"
    else:
        label = "neutro"
    return f"{pct}% {label}"


def _describe_face(face_frame):
    """Format the dominant facial emotion as a plain percentage, e.g. '97% triste'."""
    dominant = face_frame["dominant_emotion"]
    pct = round(face_frame["scores"][dominant] * 100)
    label = FACE_LABELS_PT.get(dominant, dominant)
    return f"{pct}% {label}"


def _face_sentiment(face_frame):
    """Map a facial reading to a 0-1 sentiment scale, comparable to voice valence.

    happy -> its own score (positive); sad/angry/fear/disgust -> inverted score
    (negative); neutral/surprise -> 0.5 (no signal either way).
    """
    dominant = face_frame["dominant_emotion"]
    score = face_frame["scores"][dominant]
    if dominant == "happy":
        return score
    if dominant in ("sad", "angry", "fear", "disgust"):
        return 1 - score
    return 0.5


def _build_proof_examples(critical_moments, transcript, voice_emotion, face_emotion, limit=3):
    """Numbers-first proof points: real quote + measured voice tone + measured face,
    for the `limit` dissonance-type moments with the starkest words-vs-signal gap.

    Deliberately pulls from the raw upstream JSONs (not the LLM's prose
    description) so the proof is the actual measured data, not narrative text.
    Ranked by contradiction strength (not chronological order) so the first
    entry (used as the report's hero moment) is the most persuasive one.
    Also carries arousal/dominance so the full VAD reading is available, not
    just valence.
    """
    candidates = []
    for moment in _select_dissonance_examples(critical_moments, limit=len(critical_moments)):
        t = _parse_timestamp(moment["timestamp"])
        quote_seg = _nearest_entry(transcript, t, time_key="start")
        voice_seg = _nearest_entry(voice_emotion, t, time_key="start")
        face_frame = _nearest_entry(face_emotion, t, time_key="timestamp")
        if quote_seg is None or voice_seg is None or face_frame is None:
            continue
        contradiction = abs(voice_seg["valence"] - _face_sentiment(face_frame))
        candidates.append((contradiction, {
            "timestamp": moment["timestamp"],
            "quote": quote_seg["text"],
            "tone": _describe_tone(voice_seg["valence"]),
            "face": _describe_face(face_frame),
            "valence": voice_seg["valence"],
            "arousal": round(voice_seg["arousal"] * 100),
            "dominance": round(voice_seg["dominance"] * 100),
            "face_sentiment": _face_sentiment(face_frame),
        }))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [example for _, example in candidates[:limit]]


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
        "dissonance_moments": {
            "transcript_only": _count_dissonance_moments(t["critical_moments"]),
            "multimodal": _count_dissonance_moments(m["critical_moments"]),
        },
    }


def _prosody_by_role(audio_features, speaker_map):
    """Average pitch/energy/speech-rate/pause/ZCR per role (REP vs PROSPECT).

    Feeds the glossary table's "nesta chamada" column -- the real acoustic
    numbers for this specific meeting, not a generic description.
    """
    fields = ("pitch_mean", "pitch_std", "energy_mean", "speech_rate", "pause_ratio", "zcr")
    sums = {"REP": dict.fromkeys(fields, 0.0), "PROSPECT": dict.fromkeys(fields, 0.0)}
    counts = {"REP": 0, "PROSPECT": 0}
    for seg in audio_features:
        role = speaker_map.get(seg.get("speaker", ""), "OTHER")
        if role not in sums:
            continue
        counts[role] += 1
        for f in fields:
            sums[role][f] += seg.get(f, 0.0)
    result = {}
    for role in ("REP", "PROSPECT"):
        n = counts[role] or 1
        result[role] = {f: sums[role][f] / n for f in fields}
    return result


def _vad_by_role(voice_emotion, speaker_map):
    """Average valence/arousal/dominance per role (REP vs PROSPECT)."""
    fields = ("valence", "arousal", "dominance")
    sums = {"REP": dict.fromkeys(fields, 0.0), "PROSPECT": dict.fromkeys(fields, 0.0)}
    counts = {"REP": 0, "PROSPECT": 0}
    for seg in voice_emotion:
        role = speaker_map.get(seg.get("speaker", ""), "OTHER")
        if role not in sums:
            continue
        counts[role] += 1
        for f in fields:
            sums[role][f] += seg.get(f, 0.0)
    result = {}
    for role in ("REP", "PROSPECT"):
        n = counts[role] or 1
        result[role] = {f: sums[role][f] / n for f in fields}
    return result


def _compare_role_stat(rep_value, prospect_value, fmt, prospect_higher, rep_higher, similar, similar_ratio=1.15):
    """Turn a rep-vs-prospect number pair into one plain-language sentence.

    Picks whichever of the three templates fits (prospect higher, rep higher,
    or close enough to call similar), so the same glossary row reads correctly
    no matter which role happens to score higher in a given meeting.
    """
    lo = min(rep_value, prospect_value)
    hi = max(rep_value, prospect_value)
    ratio = (hi / lo) if lo > 0 else float("inf")
    rep_s = fmt.format(rep_value)
    prospect_s = fmt.format(prospect_value)
    if ratio < similar_ratio:
        return similar.format(rep=rep_s, prospect=prospect_s)
    ratio_display = round(ratio, 1)
    if prospect_value > rep_value:
        return prospect_higher.format(rep=rep_s, prospect=prospect_s, ratio=ratio_display)
    return rep_higher.format(rep=rep_s, prospect=prospect_s, ratio=ratio_display)


def _face_emotion_distribution(face_emotion):
    """Count + percentage of every one of the 7 facial categories across the call.

    Includes categories that never occurred (count 0) so the report can state
    plainly that they did not happen, rather than omitting them.
    """
    counts = dict.fromkeys(FACE_CATEGORIES, 0)
    for frame in face_emotion:
        dominant = frame.get("dominant_emotion")
        if dominant in counts:
            counts[dominant] += 1
    total = len(face_emotion)
    percentages = {c: (round(counts[c] / total * 100) if total else 0) for c in FACE_CATEGORIES}
    return {"total": total, "counts": counts, "percentages": percentages}


def _moment_tag(type_str):
    """Classify a critical_moment's LLM-generated `type` into a broad, translated
    category for the compact moments table (dissonance / buying / risk / close).
    """
    lowered = type_str.lower()
    if "dissonance" in lowered:
        return "dissonance", "Contradição"
    if "buying" in lowered or "compra" in lowered:
        return "buying", "Sinal de Compra"
    if "risk" in lowered or "objection" in lowered or "objeção" in lowered:
        return "risk", "Risco/Objeção"
    if "close" in lowered or "commitment" in lowered or "fecho" in lowered:
        return "close", "Fecho"
    return "other", "Outro"


def render_report(transcript, audio_features, voice_emotion, face_emotion, analysis, meeting_title="Reunião de Vendas"):
    """Assemble the single self-contained output/report.html.

    All chart-facing data (timeline series + critical moment markers) is
    embedded as window.REPORT_DATA and rendered client-side by Chart.js
    (CDN). Everything else (glossary, mechanism strip, hero proof, comparison,
    proof log, facial distribution, critical moments, recommendations,
    footer) is rendered directly into the HTML string -- no template engine,
    per the project's "no framework" design.
    """
    speaker_map = _classify_speakers(transcript)
    duration = _meeting_duration(transcript)
    missing_faces = _count_missing_face_frames(face_emotion, duration)
    timeline = _build_timeline_series(voice_emotion, speaker_map)
    markers = _build_moment_markers(analysis["multimodal"]["critical_moments"])
    comparison = _build_comparison(analysis)
    proof_examples = _build_proof_examples(
        analysis["multimodal"]["critical_moments"], transcript, voice_emotion, face_emotion
    )
    prosody = _prosody_by_role(audio_features, speaker_map)
    vad = _vad_by_role(voice_emotion, speaker_map)
    face_dist = _face_emotion_distribution(face_emotion)

    report_data = json.dumps({"timeline": timeline, "moment_markers": markers})

    def esc(s):
        return (
            str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#x27;")
        )

    def fmt_delta(n):
        return f"+{n}" if n >= 0 else str(n)

    def glossary_rows():
        rows = []
        for spec in GLOSSARY_SPEC:
            table = prosody if spec["source"] == "prosody" else vad
            rep_val = table["REP"][spec["field"]]
            prospect_val = table["PROSPECT"][spec["field"]]
            here = _compare_role_stat(
                rep_val, prospect_val, spec["fmt"],
                spec["prospect_higher"], spec["rep_higher"], spec["similar"],
            )
            rows.append(f"""<tr>
  <td class="signal">{esc(spec['signal'])}</td>
  <td class="high">{esc(spec['high'])}</td>
  <td class="low">{esc(spec['low'])}</td>
  <td class="here">{esc(here)}</td>
</tr>""")
        return "\n".join(rows)

    def face_glossary_cards():
        ordered = sorted(FACE_CATEGORIES, key=lambda c: face_dist["counts"][c], reverse=True)
        cards = []
        for cat in ordered:
            pct = face_dist["percentages"][cat]
            meaning = FACE_MEANINGS_PT[cat]
            if face_dist["counts"][cat] == 0:
                meaning = meaning + " Não ocorreu nesta chamada."
            css_class = cat if cat in ("happy", "sad", "angry") else "neutral"
            cards.append(f"""<div class="fg-item {css_class}">
  <div class="fg-pct">{pct}%</div>
  <div class="fg-name">Rosto {esc(FACE_LABELS_PT[cat].capitalize())}</div>
  <div class="fg-meaning">{esc(meaning)}</div>
</div>""")
        return "\n".join(cards)

    def mechanism_strip():
        return f"""<div class="mechanism">
  <div class="mech-step">
    <div class="kind">1 - Medição</div>
    <div class="name">Transcrição + Interlocutor</div>
    <div class="figure">{len(transcript)} segmentos</div>
  </div>
  <div class="mech-step">
    <div class="kind">2 - Medição</div>
    <div class="name">Prosódia (Tom, Energia, Ritmo)</div>
    <div class="figure">{len(audio_features)} segmentos</div>
  </div>
  <div class="mech-step">
    <div class="kind">3 - Medição</div>
    <div class="name">Tom de Voz (emoção)</div>
    <div class="figure">{len(voice_emotion)} leituras</div>
  </div>
  <div class="mech-step">
    <div class="kind">4 - Medição</div>
    <div class="name">Expressão Facial</div>
    <div class="figure">{len(face_emotion)} leituras</div>
  </div>
  <div class="mech-step llm">
    <div class="kind">5 - Interpretação</div>
    <div class="name">Scores + Recomendações</div>
    <div class="figure">só aqui entra o LLM</div>
  </div>
</div>"""

    def hero_line(example):
        voice_positive = example["valence"] >= 0.5
        face_positive = example["face_sentiment"] >= 0.5
        if voice_positive and not face_positive:
            return "A voz disse que sim. A cara já tinha respondido que não."
        if not voice_positive and face_positive:
            return "A voz disse que não. A cara já tinha respondido que sim."
        return "A voz e a cara não contam a mesma história."

    def hero_section(hero):
        return f"""<div class="hero-proof">
  <div class="hero-kicker">Captado aos <b>{esc(hero['timestamp'])}</b>, ninguém na chamada reparou</div>
  <p class="hero-quote">&ldquo;{esc(hero['quote'])}&rdquo;</p>
  <div class="hero-numbers">
    <div class="hero-num voice"><div class="n">{esc(hero['tone'].split('%')[0])}%</div><div class="lbl">Voz</div></div>
    <div class="hero-num face"><div class="n">{esc(hero['face'].split('%')[0])}%</div><div class="lbl">Rosto</div></div>
  </div>
  <p class="hero-line">{esc(hero_line(hero))}</p>
  <p class="hero-scale">{comparison['dissonance_moments']['multimodal']} contradições encontradas em {len(voice_emotion)} leituras de tom de voz e {len(face_emotion)} leituras faciais ao longo dos {esc(_format_timestamp(duration))} de chamada. Nenhuma seria visível só com o texto.</p>
</div>"""

    def proof_log_entries(examples):
        parts = []
        for ex in examples:
            parts.append(f"""<div class="proof-entry">
  <div class="proof-ts">{esc(ex['timestamp'])}</div>
  <div class="proof-quote">&ldquo;{esc(ex['quote'])}&rdquo;</div>
  <div class="proof-readout">
    <div class="readout-labels"><span class="voice-lbl">Voz: {esc(ex['tone'])}</span><span class="face-lbl">Rosto: {esc(ex['face'])}</span></div>
    <div class="gap-extra">Energia emocional {ex['arousal']}% . Assertividade {ex['dominance']}%</div>
  </div>
</div>""")
        return "\n".join(parts)

    def moment_rows(moments):
        rows = []
        for m in moments:
            css_class, label = _moment_tag(m["type"])
            rows.append(f"""<tr>
  <td class="ts">{esc(m['timestamp'])}</td>
  <td><span class="tag {css_class}">{esc(label)}</span></td>
  <td>
    <div class="moment-desc">{esc(m['description'])}</div>
    <div class="moment-coach">{esc(m['coaching'])}</div>
  </td>
</tr>""")
        return "\n".join(rows)

    def recommendation_items(recs):
        return "\n".join(f"<li>{esc(r)}</li>" for r in recs)

    proof_section = ""
    if proof_examples:
        hero, log_examples = proof_examples[0], proof_examples[1:]
        proof_section = hero_section(hero)
        if log_examples:
            proof_section += f"""
<section>
  <div class="section-head">
    <h2>Mais contradições que a transcrição não via</h2>
    <div class="section-sub">As palavras concordam. A voz e o rosto, medidos, não concordam.</div>
  </div>
  {proof_log_entries(log_examples)}
</section>"""

    face_note = (
        f'<p class="face-note">dados faciais indisponíveis em {missing_faces} imagem(ns)</p>'
        if missing_faces > 0
        else ""
    )

    m = analysis["multimodal"]

    html = f"""<!DOCTYPE html>
<html lang="pt-PT">
<head>
<meta charset="UTF-8">
<title>{esc(meeting_title)} - Análise de Reunião de Vendas</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root{{
    --ink:#0C0E13; --panel:#151822; --panel-raised:#191D28; --line:#262B38;
    --gold:#C9A233; --gold-dim:#8a6f24; --slate:#5B6272; --pos:#4FA876; --neg:#C2564F;
    --text:#EAEAEE; --muted:#8A90A0; --muted-dim:#5f6472;
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--ink); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 15px; line-height: 1.55; margin: 0; padding: 0 24px 64px; }}
  .mono {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }}
  .eyebrow {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; text-transform: uppercase; letter-spacing: .14em; font-size: 11.5px; color: var(--gold); font-weight: 600; }}
  h1 {{ font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; font-weight: 400; font-size: 2.1rem; margin: 10px 0 6px; }}
  h2 {{ font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; font-weight: 400; font-size: 1.3rem; margin: 0 0 4px; }}
  .meta {{ color: var(--muted); font-size: 13.5px; }}
  .meta b {{ color: var(--text); font-weight: 600; }}
  header {{ padding: 28px 0 24px; border-bottom: 1px solid var(--line); margin-bottom: 28px; }}
  .intro {{ font-size: 14.5px; max-width: 68ch; margin: 0 0 40px; }}
  section {{ margin: 48px 0; }}
  .section-head {{ margin-bottom: 18px; }}
  .section-sub {{ color: var(--muted); font-size: 13.5px; margin-top: 4px; max-width: 62ch; }}
  .scroll-x {{ overflow-x: auto; }}

  table.glossary {{ width: 100%; border-collapse: collapse; }}
  table.glossary th {{ text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; padding: 0 14px 10px; border-bottom: 1px solid var(--line); }}
  table.glossary td {{ padding: 13px 14px; border-bottom: 1px solid var(--line); font-size: 13.4px; vertical-align: top; }}
  table.glossary td.signal {{ font-weight: 600; color: var(--text); white-space: nowrap; }}
  table.glossary td.high {{ color: var(--pos); }}
  table.glossary td.low {{ color: var(--neg); }}
  table.glossary td.here {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; color: var(--muted); font-size: 12.5px; }}

  .face-glossary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 22px; }}
  .face-glossary .fg-item {{ background: var(--panel); border: 1px solid var(--line); padding: 14px 16px; }}
  .face-glossary .fg-pct {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 1.6rem; font-weight: 700; }}
  .face-glossary .fg-item.happy .fg-pct {{ color: var(--pos); }}
  .face-glossary .fg-item.sad .fg-pct {{ color: var(--neg); }}
  .face-glossary .fg-item.angry .fg-pct {{ color: var(--gold-dim); }}
  .face-glossary .fg-item.neutral .fg-pct {{ color: var(--slate); }}
  .face-glossary .fg-name {{ font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin: 4px 0 8px; }}
  .face-glossary .fg-meaning {{ font-size: 12.5px; }}

  .mechanism {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); margin-bottom: 10px; }}
  .mech-step {{ background: var(--panel); padding: 14px 12px; }}
  .mech-step .kind {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 10px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted-dim); margin-bottom: 8px; border-top: 2px solid var(--slate); padding-top: 8px; margin-top: -14px; }}
  .mech-step.llm .kind {{ color: #c8b787; border-top-color: var(--gold); }}
  .mech-step .name {{ font-size: 13px; font-weight: 600; margin-bottom: 6px; line-height: 1.3; }}
  .mech-step .figure {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 12px; color: var(--muted); }}
  .mechanism-note {{ font-size: 12.5px; color: var(--muted); max-width: 68ch; }}

  .hero-proof {{ background: var(--panel-raised); border: 1px solid var(--line); border-left: 3px solid var(--gold); padding: 30px 32px; margin: 0 0 24px; }}
  .hero-kicker {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; text-transform: uppercase; letter-spacing: .12em; font-size: 11px; color: var(--muted); margin-bottom: 14px; }}
  .hero-kicker b {{ color: var(--gold); }}
  .hero-quote {{ font-family: Georgia, "Iowan Old Style", serif; font-style: italic; font-size: 1.6rem; line-height: 1.35; margin: 0 0 22px; }}
  .hero-numbers {{ display: flex; gap: 40px; align-items: flex-end; margin-bottom: 18px; }}
  .hero-num .n {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 3rem; font-weight: 700; line-height: 1; }}
  .hero-num.voice .n {{ color: var(--gold); }}
  .hero-num.face .n {{ color: var(--neg); }}
  .hero-num .lbl {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-top: 6px; }}
  .hero-line {{ font-size: 1.05rem; font-weight: 600; margin-bottom: 10px; }}
  .hero-scale {{ font-size: 12px; color: var(--muted); }}

  .hero-card {{ display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); }}
  .stat {{ background: var(--panel); padding: 16px 18px; }}
  .stat .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 8px; }}
  .stat .value {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 1.9rem; font-weight: 600; }}
  .stat.highlight {{ background: var(--panel-raised); }}
  .stat.highlight .value {{ color: var(--gold); }}
  .stat .sub {{ font-size: 11.5px; color: var(--muted-dim); margin-top: 4px; }}

  table.compare {{ width: 100%; border-collapse: collapse; }}
  table.compare th {{ text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); font-weight: 600; padding: 0 14px 10px; border-bottom: 1px solid var(--line); }}
  table.compare th.num, table.compare td.num {{ text-align: right; }}
  table.compare td {{ padding: 13px 14px; border-bottom: 1px solid var(--line); font-size: 14.5px; }}
  table.compare td.num, table.compare td.delta {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; }}
  table.compare td.tonly {{ color: var(--slate); }}
  table.compare tr.hero td {{ background: rgba(201,162,51,0.07); }}
  table.compare tr.hero td.metric {{ font-weight: 600; }}
  table.compare td.delta {{ font-weight: 700; }}
  table.compare tr.hero td.delta {{ color: var(--gold); }}
  table.compare td.delta.zero {{ color: var(--muted-dim); font-weight: 400; }}

  .proof-entry {{ display: grid; grid-template-columns: 90px 1fr 260px; gap: 20px; align-items: center; padding: 16px 0; border-bottom: 1px solid var(--line); }}
  .proof-entry:last-child {{ border-bottom: none; }}
  .proof-ts {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; color: var(--gold); font-size: 13px; font-weight: 600; }}
  .proof-quote {{ font-family: Georgia, "Iowan Old Style", serif; font-style: italic; font-size: 1.05rem; }}
  .readout-labels {{ display: flex; gap: 18px; font-size: 12px; color: var(--muted); }}
  .readout-labels .voice-lbl:before {{ content: "\\25CF "; color: var(--gold); }}
  .readout-labels .face-lbl:before {{ content: "\\25CF "; color: var(--neg); }}
  .gap-extra {{ font-size: 11px; color: var(--muted-dim); margin-top: 6px; }}

  .dist-bar {{ display: flex; width: 100%; height: 28px; border-radius: 2px; overflow: hidden; }}
  .dist-bar .seg {{ display: flex; align-items: center; justify-content: center; font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 11px; font-weight: 600; color: var(--ink); }}
  .dist-bar .seg.neutral {{ background: var(--slate); }}
  .dist-bar .seg.sad {{ background: var(--neg); }}
  .dist-bar .seg.happy {{ background: var(--pos); }}
  .dist-bar .seg.angry {{ background: var(--gold-dim); color: var(--text); }}

  table.moments {{ width: 100%; border-collapse: collapse; }}
  table.moments th {{ text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); font-weight: 600; padding: 0 12px 10px; border-bottom: 1px solid var(--line); }}
  table.moments td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); font-size: 13.6px; vertical-align: top; }}
  table.moments td.ts {{ font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; color: var(--muted); white-space: nowrap; }}
  .moment-coach {{ color: var(--muted); font-size: 12.5px; margin-top: 4px; }}
  .tag {{ display: inline-block; font-size: 10.5px; text-transform: uppercase; letter-spacing: .05em; font-weight: 700; padding: 2px 7px; border-radius: 2px; white-space: nowrap; }}
  .tag.dissonance {{ background: rgba(201,162,51,0.14); color: var(--gold); }}
  .tag.buying {{ background: rgba(79,168,118,0.14); color: var(--pos); }}
  .tag.risk {{ background: rgba(194,86,79,0.14); color: var(--neg); }}
  .tag.close {{ background: rgba(139,144,160,0.14); color: var(--muted); }}
  .tag.other {{ background: rgba(139,144,160,0.14); color: var(--muted); }}

  .recs {{ list-style: none; margin: 0; padding: 0; }}
  .recs li {{ padding: 12px 0; border-bottom: 1px solid var(--line); font-size: 14px; }}
  .recs li:last-child {{ border-bottom: none; }}

  .face-note {{ color: var(--muted); font-size: 0.85rem; }}
  footer {{ margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--muted-dim); font-size: 11.5px; font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; }}
</style>
</head>
<body>
<header>
  <div class="eyebrow">Relatório de Análise Multimodal</div>
  <h1>{esc(meeting_title)}</h1>
  <div class="meta">Duração <b class="mono">{esc(_format_timestamp(duration))}</b> . Consultor <b class="mono">{comparison['talk_ratio']['rep']}%</b> do tempo de palavra . Prospect <b class="mono">{comparison['talk_ratio']['prospect']}%</b></div>
</header>

<p class="intro">Esta chamada foi processada e analisada duas vezes pelo mesmo modelo de IA: uma vez só com o texto da transcrição, outra vez também com o tom de voz e a expressão facial captados no vídeo. Este relatório mostra, sinal a sinal, o que cada abordagem viu e o que só a segunda conseguiu captar.</p>

<section>
  <div class="section-head">
    <h2>Que sinais medimos, e o que significam</h2>
    <div class="section-sub">Nove sinais de voz, medidos ao longo de toda a chamada. Para cada um: o que significa quando está alto, o que significa quando está baixo, e o que aconteceu nesta chamada em concreto.</div>
  </div>
  <div class="scroll-x">
    <table class="glossary">
      <thead>
        <tr><th style="width:190px">Sinal</th><th>Alto significa</th><th>Baixo significa</th><th style="width:240px">Nesta chamada</th></tr>
      </thead>
      <tbody>
{glossary_rows()}
      </tbody>
    </table>
  </div>
  <div class="face-glossary">
{face_glossary_cards()}
  </div>
</section>

<section>
  <div class="section-head">
    <h2>Como analisamos esta chamada</h2>
    <div class="section-sub">Os sinais acima vêm de 4 medições independentes sobre o áudio e o vídeo. O LLM (Claude) nunca vê o áudio nem o vídeo, só entra no fim, para interpretar o que já foi medido.</div>
  </div>
  {mechanism_strip()}
  <p class="mechanism-note">Os passos 1 a 4 são medições diretas, nunca passam pelo LLM. O LLM só entra no passo 5, para ler os números já medidos e escrever os scores e as recomendações.</p>
</section>

{proof_section}

<section class="hero-card">
  <div class="stats" style="width:100%">
    <div class="stat"><div class="label">Pontuação de Envolvimento</div><div class="value">{m['engagement_score']}</div></div>
    <div class="stat"><div class="label">Probabilidade de Fecho</div><div class="value">{m['deal_probability']}%</div></div>
    <div class="stat"><div class="label">Duração</div><div class="value">{esc(_format_timestamp(duration))}</div></div>
    <div class="stat highlight"><div class="label">Contradições Detetadas</div><div class="value">{comparison['dissonance_moments']['multimodal']}</div><div class="sub">só transcrição: {comparison['dissonance_moments']['transcript_only']}</div></div>
  </div>
</section>

<section>
  <div class="section-head">
    <h2>Só transcrição vs. multimodal</h2>
    <div class="section-sub">Mesmo modelo, duas entradas diferentes: uma só lê o texto, a outra lê voz, prosódia, rosto e texto em conjunto.</div>
  </div>
  <div class="scroll-x">
    <table class="compare">
      <thead>
        <tr><th>Métrica</th><th class="num">Só Transcrição</th><th class="num">Multimodal</th><th class="num">Diferença</th></tr>
      </thead>
      <tbody>
        <tr>
          <td class="metric">Pontuação de Envolvimento</td>
          <td class="num tonly">{comparison['engagement_score']['transcript_only']}</td>
          <td class="num">{comparison['engagement_score']['multimodal']}</td>
          <td class="num delta {'zero' if comparison['engagement_score']['delta'] == 0 else ''}">{fmt_delta(comparison['engagement_score']['delta'])}</td>
        </tr>
        <tr>
          <td class="metric">Probabilidade de Fecho</td>
          <td class="num tonly">{comparison['deal_probability']['transcript_only']}%</td>
          <td class="num">{comparison['deal_probability']['multimodal']}%</td>
          <td class="num delta {'zero' if comparison['deal_probability']['delta'] == 0 else ''}">{fmt_delta(comparison['deal_probability']['delta'])}</td>
        </tr>
        <tr class="hero">
          <td class="metric">Contradições palavra contra tom/rosto</td>
          <td class="num tonly">{comparison['dissonance_moments']['transcript_only']}</td>
          <td class="num">{comparison['dissonance_moments']['multimodal']}</td>
          <td class="num delta">{fmt_delta(comparison['dissonance_moments']['multimodal'] - comparison['dissonance_moments']['transcript_only'])}</td>
        </tr>
        <tr>
          <td class="metric">Sinais analisados</td>
          <td class="tonly">Só texto</td>
          <td>Voz + Prosódia + Rosto + Texto</td>
          <td class="delta">-</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<section class="timeline">
  <h2>Cronologia de Envolvimento</h2>
  {face_note}
  <canvas id="timelineChart" height="100"></canvas>
</section>

<section>
  <div class="section-head">
    <h2>Expressão facial ao longo da chamada</h2>
    <div class="section-sub">{face_dist['total']} leituras, uma a cada 10 segundos, do início ao fim da chamada, não só os momentos de contradição.</div>
  </div>
  <div class="dist-bar">
    <div class="seg neutral" style="width:{face_dist['percentages']['neutral']}%">{face_dist['percentages']['neutral']}%</div>
    <div class="seg sad" style="width:{face_dist['percentages']['sad']}%">{face_dist['percentages']['sad']}%</div>
    <div class="seg happy" style="width:{face_dist['percentages']['happy']}%"></div>
    <div class="seg angry" style="width:{face_dist['percentages']['angry']}%"></div>
  </div>
</section>

<section>
  <div class="section-head">
    <h2>Registo de momentos críticos</h2>
    <div class="section-sub">Todos os {len(m['critical_moments'])} momentos que o modelo multimodal identificou, em formato de tabela, não de parágrafo.</div>
  </div>
  <div class="scroll-x">
    <table class="moments">
      <thead>
        <tr><th style="width:76px">Hora</th><th style="width:150px">Tipo</th><th>Sinal</th></tr>
      </thead>
      <tbody>
{moment_rows(m['critical_moments'])}
      </tbody>
    </table>
  </div>
</section>

<section class="recommendations">
  <h2>Recomendações</h2>
  <ul class="recs">
  {recommendation_items(m['recommendations'])}
  </ul>
</section>

<footer>
  WhisperX . pyannote . librosa . audeering wav2vec2 . DeepFace . Claude
</footer>

<script>
window.REPORT_DATA = {report_data};
const ctx = document.getElementById('timelineChart');
new Chart(ctx, {{
  type: 'line',
  data: {{
    datasets: [
      {{ label: 'Valencia do Prospect', data: window.REPORT_DATA.timeline.prospect_valence, borderColor: '#3b82f6', parsing: false }},
      {{ label: 'Energia do Prospect', data: window.REPORT_DATA.timeline.prospect_arousal, borderColor: '#f59e0b', parsing: false }},
      {{ label: 'Energia do Consultor', data: window.REPORT_DATA.timeline.rep_arousal, borderColor: '#9ca3af', parsing: false }},
    ],
  }},
  options: {{ scales: {{ x: {{ type: 'linear' }} }} }},
}});
</script>
</body>
</html>"""
    return html
