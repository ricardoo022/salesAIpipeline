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
MAX_TOKENS = 8192
RATE_LIMIT_WAIT = 10  # seconds; spec: retry once after 10s
REQUIRED_FIELDS = ("engagement_score", "deal_probability", "critical_moments", "recommendations")

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


def _validate_analysis(result: dict) -> None:
    """Ensure the tool-use input has all required schema fields.

    Forced tool_choice guarantees a tool_use block exists, but max_tokens
    truncation can yield a valid-but-incomplete input (e.g. critical_moments
    closed before recommendations emitted). This turns a silent schema
    violation into a loud failure instead of writing broken analysis.json.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in result]
    if missing:
        raise RuntimeError(
            f"Claude analysis incomplete (missing: {', '.join(missing)}); "
            "likely max_tokens truncation"
        )


def _call_claude(user_prompt: str, api_key: str) -> dict:
    """Call Claude with forced tool_use to guarantee the analysis schema.

    Retries once after RATE_LIMIT_WAIT seconds on a rate-limit error (spec).
    Raises RuntimeError on max_tokens truncation or an incomplete tool input.
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
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise RuntimeError(
            "Claude response truncated (stop_reason=max_tokens); increase MAX_TOKENS"
        )
    result = _extract_tool_input(response)
    _validate_analysis(result)
    return result


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
