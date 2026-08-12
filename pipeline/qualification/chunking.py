"""Hierarchical transcript chunking: conversation sections (US-1.2) and
bounded extraction chunks (US-1.3)."""

import dataclasses
from typing import Callable

from .schemas import (
    ChunkingConfiguration,
    ConversationSection,
    ExtractionChunk,
    TranscriptSegment,
    _count_words,
)


def create_sections(
    segments: list[TranscriptSegment] | tuple[TranscriptSegment, ...],
    config: ChunkingConfiguration | None = None,
) -> list[ConversationSection]:
    """Group normalized segments into chronological conversation sections.

    Windows of ``section_target_seconds`` stride forward by
    ``section_target_seconds - section_overlap_seconds`` starting at the
    earliest segment start, so neighboring windows overlap by exactly the
    configured value. A segment belongs to every window containing its start
    time, which guarantees every segment lands in at least one section.
    Windows with no members (long silences) are skipped and the emitted
    sections are renumbered, keeping IDs and sequences deterministic.
    Sections are organizational only and carry no BANT label.
    """
    config = config or ChunkingConfiguration()
    items = list(segments)
    if not items:
        return []

    stride = config.section_target_seconds - config.section_overlap_seconds
    first_start = min(item.start for item in items)
    last_start = max(item.start for item in items)

    windows: list[tuple[float, float]] = []
    window_start = first_start
    while window_start <= last_start:
        windows.append((window_start, window_start + config.section_target_seconds))
        window_start += stride

    emitted = [
        [index for index, item in enumerate(items) if start <= item.start < end]
        for start, end in windows
    ]
    emitted = [members for members in emitted if members]

    membership_counts: dict[int, int] = {}
    for members in emitted:
        for index in members:
            membership_counts[index] = membership_counts.get(index, 0) + 1

    sections: list[ConversationSection] = []
    for sequence, members in enumerate(emitted, start=1):
        member_segments = [items[index] for index in members]
        sections.append(
            ConversationSection(
                section_id=f"section_{sequence:06d}",
                sequence=sequence,
                start=min(segment.start for segment in member_segments),
                end=max(segment.end for segment in member_segments),
                segment_ids=tuple(segment.segment_id for segment in member_segments),
                overlap_segment_ids=tuple(
                    items[index].segment_id
                    for index in members
                    if membership_counts[index] > 1
                ),
            )
        )
    return sections


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _render_segment(segment: TranscriptSegment) -> str:
    return f"{segment.speaker} [{_format_timestamp(segment.start)}]: {segment.text}"


def _render(segments: list[TranscriptSegment]) -> str:
    return "\n".join(_render_segment(segment) for segment in segments)


def _group_into_turns(
    segments: list[TranscriptSegment],
) -> list[list[TranscriptSegment]]:
    turns: list[list[TranscriptSegment]] = []
    for segment in segments:
        if turns and turns[-1][-1].speaker == segment.speaker:
            turns[-1].append(segment)
        else:
            turns.append([segment])
    return turns


def _pack_segments_individually(
    turn_segments: list[TranscriptSegment],
    config: ChunkingConfiguration,
    tokenizer: Callable[[str], int],
) -> list[list[TranscriptSegment]]:
    groups: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    for segment in turn_segments:
        candidate = current + [segment]
        if current and tokenizer(_render(candidate)) > config.max_chunk_tokens:
            groups.append(current)
            current = [segment]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _pack_turns(
    turns: list[list[TranscriptSegment]],
    config: ChunkingConfiguration,
    tokenizer: Callable[[str], int],
) -> list[list[list[TranscriptSegment]]]:
    chunks: list[list[list[TranscriptSegment]]] = []
    current: list[list[TranscriptSegment]] = []
    current_flat: list[TranscriptSegment] = []

    for turn in turns:
        if tokenizer(_render(turn)) > config.max_chunk_tokens:
            if current:
                chunks.append(current)
                current = []
                current_flat = []
            chunks.extend([group] for group in _pack_segments_individually(turn, config, tokenizer))
            continue

        if not current:
            current = [turn]
            current_flat = list(turn)
            continue

        candidate_flat = current_flat + turn
        if tokenizer(_render(candidate_flat)) <= config.max_chunk_tokens:
            current.append(turn)
            current_flat = candidate_flat
        else:
            chunks.append(current)
            current = [turn]
            current_flat = list(turn)

    if current:
        chunks.append(current)

    return chunks


def _select_overlap(
    previous_turns: list[list[TranscriptSegment]],
    config: ChunkingConfiguration,
    tokenizer: Callable[[str], int],
) -> list[list[TranscriptSegment]]:
    if not previous_turns or config.chunk_overlap_turns <= 0:
        return []

    selected = list(previous_turns[-config.chunk_overlap_turns :])
    while len(selected) > 1:
        flat = [segment for turn in selected for segment in turn]
        if tokenizer(_render(flat)) <= config.max_overlap_tokens:
            break
        selected.pop(0)
    return selected


def _build_section_chunks(
    section: ConversationSection,
    section_segments: list[TranscriptSegment],
    config: ChunkingConfiguration,
    tokenizer: Callable[[str], int],
    start_counter: int,
) -> tuple[list[ExtractionChunk], int]:
    turns = _group_into_turns(section_segments)
    packed = _pack_turns(turns, config, tokenizer)

    chunks: list[ExtractionChunk] = []
    counter = start_counter
    previous_own_turns: list[list[TranscriptSegment]] | None = None

    for sequence, own_turns in enumerate(packed, start=1):
        own_segments = [segment for turn in own_turns for segment in turn]
        overlap_turns = (
            _select_overlap(previous_own_turns, config, tokenizer)
            if sequence > 1 and previous_own_turns is not None
            else []
        )
        overlap_segments = [segment for turn in overlap_turns for segment in turn]
        overlap_segment_ids = tuple(segment.segment_id for segment in overlap_segments)

        final_segments = overlap_segments + own_segments
        text = _render(final_segments)

        chunks.append(
            ExtractionChunk(
                chunk_id=f"chunk_{counter:06d}",
                section_id=section.section_id,
                sequence=sequence,
                start=min(segment.start for segment in final_segments),
                end=max(segment.end for segment in final_segments),
                segment_ids=tuple(segment.segment_id for segment in final_segments),
                overlap_segment_ids=overlap_segment_ids,
                token_count=tokenizer(text),
                text=text,
            )
        )
        counter += 1
        previous_own_turns = own_turns

    return chunks, counter


def _segments_for_section(
    section: ConversationSection,
    by_id: dict[str, list[TranscriptSegment]],
) -> list[TranscriptSegment]:
    occurrence: dict[str, int] = {}
    result: list[TranscriptSegment] = []
    for segment_id in section.segment_ids:
        index = occurrence.get(segment_id, 0)
        candidates = by_id.get(segment_id, [])
        if index < len(candidates):
            result.append(candidates[index])
            occurrence[segment_id] = index + 1
    return result


def create_chunks(
    segments: list[TranscriptSegment] | tuple[TranscriptSegment, ...],
    sections: list[ConversationSection] | tuple[ConversationSection, ...],
    config: ChunkingConfiguration | None = None,
    *,
    tokenizer: Callable[[str], int] = _count_words,
) -> tuple[list[ConversationSection], list[ExtractionChunk]]:
    """Chunk each section's member segments into bounded, speaker-labeled,
    token-counted extraction chunks.

    Chunking is per-section: chunks never span two sections. Within a
    section, segments are grouped into speaker turns and greedily packed
    into chunks bounded by ``config.max_chunk_tokens``; a turn that alone
    exceeds the limit falls back to segment-level packing, and a single
    oversized segment becomes its own chunk exceeding the limit (the only
    unavoidable exception). Neighboring chunks within a section overlap by
    up to ``config.chunk_overlap_turns`` complete trailing turns taken from
    the previous chunk's own pre-overlap turns, bounded by
    ``config.max_overlap_tokens`` except that the nearest turn is always
    included in full. Normal transcript segments are never split.
    """
    config = config or ChunkingConfiguration()

    by_id: dict[str, list[TranscriptSegment]] = {}
    for segment in segments:
        by_id.setdefault(segment.segment_id, []).append(segment)

    updated_sections: list[ConversationSection] = []
    all_chunks: list[ExtractionChunk] = []
    counter = 1

    for section in sections:
        section_segments = _segments_for_section(section, by_id)
        if not section_segments:
            updated_sections.append(dataclasses.replace(section, chunk_ids=()))
            continue

        section_chunks, counter = _build_section_chunks(
            section, section_segments, config, tokenizer, counter
        )
        all_chunks.extend(section_chunks)
        updated_sections.append(
            dataclasses.replace(
                section,
                chunk_ids=tuple(chunk.chunk_id for chunk in section_chunks),
            )
        )

    return updated_sections, all_chunks
