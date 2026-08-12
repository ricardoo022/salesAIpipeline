"""Immutable source schemas for qualification transcript processing."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping


def _count_words(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class TranscriptSegment:
    """One original transcript segment or a word-bounded piece of one."""

    segment_id: str
    speaker: str
    start: float
    end: float
    text: str
    words: tuple[Mapping[str, object], ...]
    piece_index: int | None = None
    word_start: int | None = None
    word_end: int | None = None


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _immutable_words(words: list[dict]) -> tuple[Mapping[str, object], ...]:
    return tuple(_freeze(word) for word in words)  # type: ignore[return-value]


def _source_segment(segment_id: str, source: dict) -> TranscriptSegment:
    missing = [field for field in ("speaker", "start", "end", "text") if field not in source]
    if missing:
        raise ValueError(f"transcript segment missing required field(s): {', '.join(missing)}")

    words = source.get("words") or []
    if not isinstance(words, list) or any(not isinstance(word, dict) for word in words):
        raise ValueError("transcript segment words must be a list of mappings")

    return TranscriptSegment(
        segment_id=segment_id,
        speaker=source["speaker"],
        start=source["start"],
        end=source["end"],
        text=source["text"],
        words=_immutable_words(words),
    )


def _split_segment(
    segment_id: str,
    source: dict,
    max_tokens: int,
    tokenizer: Callable[[str], int],
) -> list[TranscriptSegment]:
    words = source.get("words") or []
    if not words:
        raise ValueError(
            f"oversized transcript segment {segment_id} requires word-level timestamps"
        )

    pieces: list[TranscriptSegment] = []
    current: list[dict] = []
    current_start = 0

    def emit(piece_words: list[dict], word_start: int) -> None:
        piece_index = len(pieces)
        pieces.append(
            TranscriptSegment(
                segment_id=segment_id,
                speaker=source["speaker"],
                start=source["start"],
                end=source["end"],
                text=" ".join(word["word"] for word in piece_words),
                words=_immutable_words(piece_words),
                piece_index=piece_index,
                word_start=word_start,
                word_end=word_start + len(piece_words),
            )
        )

    for index, word in enumerate(words):
        if "word" not in word:
            raise ValueError(f"oversized transcript segment {segment_id} has a word without text")
        candidate = current + [word]
        if current and tokenizer(" ".join(item["word"] for item in candidate)) > max_tokens:
            emit(current, current_start)
            current = []
            current_start = index
            candidate = [word]

        if tokenizer(word["word"]) > max_tokens:
            emit([word], index)
            current_start = index + 1
        else:
            current = candidate

    if current:
        emit(current, current_start)
    return pieces


def normalize_transcript(
    transcript: list[dict],
    *,
    max_tokens: int = 1200,
    tokenizer: Callable[[str], int] = _count_words,
) -> list[TranscriptSegment]:
    """Normalize transcript entries without mutating the source list."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    normalized: list[TranscriptSegment] = []
    for index, source in enumerate(transcript, start=1):
        if not isinstance(source, dict):
            raise ValueError("each transcript segment must be a mapping")
        segment_id = f"seg_{index:06d}"
        segment = _source_segment(segment_id, source)
        if tokenizer(segment.text) <= max_tokens:
            normalized.append(segment)
        else:
            normalized.extend(_split_segment(segment_id, source, max_tokens, tokenizer))
    return normalized


@dataclass(frozen=True)
class ChunkingConfiguration:
    """Explicit, reproducible hierarchy settings (Epic 1 schema).

    Time organizes sections, tokens bound LLM chunks, and complete speaker
    turns define chunk overlap. ``tokenizer`` records the identity of the
    tokenizer used for token counting (the default is the whitespace word
    counter used by ``normalize_transcript``).
    """

    section_target_seconds: float = 480.0
    section_overlap_seconds: float = 30.0
    max_chunk_tokens: int = 1200
    chunk_overlap_turns: int = 2
    max_overlap_tokens: int = 250
    tokenizer: str = "whitespace-word-count-v1"

    def __post_init__(self) -> None:
        if self.section_target_seconds <= 0:
            raise ValueError("section_target_seconds must be greater than zero")
        if self.section_overlap_seconds < 0:
            raise ValueError("section_overlap_seconds must not be negative")
        if self.section_overlap_seconds >= self.section_target_seconds:
            raise ValueError(
                "section_overlap_seconds must be smaller than section_target_seconds"
            )
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be greater than zero")
        if self.chunk_overlap_turns < 0:
            raise ValueError("chunk_overlap_turns must not be negative")
        if self.max_overlap_tokens < 0:
            raise ValueError("max_overlap_tokens must not be negative")
        if not isinstance(self.tokenizer, str) or not self.tokenizer:
            raise ValueError("tokenizer must be a non-empty tokenizer identity")


@dataclass(frozen=True)
class ConversationSection:
    """Chronological parent container for source segments (Epic 1 schema).

    Organizational only: sections carry no BANT label and no transcript text.
    ``chunk_ids`` stays empty until extraction chunks exist (US-1.3).
    """

    section_id: str
    sequence: int
    start: float
    end: float
    segment_ids: tuple[str, ...]
    overlap_segment_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractionChunk:
    """Bounded, speaker-labeled text chunk for LLM extraction (US-1.3 schema).

    Organizational only: chunks carry no BANT label. Text is rendered from
    original source segments with speaker labels and timestamps.
    """

    chunk_id: str
    section_id: str
    sequence: int
    start: float
    end: float
    segment_ids: tuple[str, ...]
    overlap_segment_ids: tuple[str, ...]
    token_count: int
    text: str
