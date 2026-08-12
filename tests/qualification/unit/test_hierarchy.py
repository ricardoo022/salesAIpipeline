import dataclasses

import pytest

from pipeline.qualification.chunking import (
    ChunkHierarchyError,
    iter_chunks_for_processing,
    reconstruct_chunk_text,
    resolve_chunk,
)
from pipeline.qualification.schemas import (
    ConversationSection,
    ExtractionChunk,
    ResolvedChunk,
    TranscriptSegment,
)


def make_segments(
    count,
    *,
    start=0.0,
    first_index=1,
    step=10.0,
    duration=5.0,
    speakers=None,
    text=None,
):
    segments = []
    for offset in range(count):
        index = first_index + offset
        seg_start = start + offset * step
        if speakers is not None:
            speaker = speakers[offset]
        else:
            speaker = "SPEAKER_00" if index % 2 == 1 else "SPEAKER_01"
        segment_text = text[offset] if text is not None else f"utterance {index}"
        segments.append(
            TranscriptSegment(
                segment_id=f"seg_{index:06d}",
                speaker=speaker,
                start=seg_start,
                end=seg_start + duration,
                text=segment_text,
                words=(),
            )
        )
    return segments


def make_section(segments, *, section_id="section_000001", sequence=1, overlap_segment_ids=(), chunk_ids=()):
    return ConversationSection(
        section_id=section_id,
        sequence=sequence,
        start=min(segment.start for segment in segments),
        end=max(segment.end for segment in segments),
        segment_ids=tuple(segment.segment_id for segment in segments),
        overlap_segment_ids=tuple(overlap_segment_ids),
        chunk_ids=tuple(chunk_ids),
    )


def _render_text(segments):
    lines = []
    for segment in segments:
        total = int(segment.start)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        timestamp = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        lines.append(f"{segment.speaker} [{timestamp}]: {segment.text}")
    return "\n".join(lines)


def make_chunk(
    segments,
    *,
    chunk_id="chunk_000001",
    section_id="section_000001",
    sequence=1,
    overlap_segment_ids=(),
    token_count=None,
):
    text = _render_text(segments)
    return ExtractionChunk(
        chunk_id=chunk_id,
        section_id=section_id,
        sequence=sequence,
        start=min(segment.start for segment in segments),
        end=max(segment.end for segment in segments),
        segment_ids=tuple(segment.segment_id for segment in segments),
        overlap_segment_ids=tuple(overlap_segment_ids),
        token_count=token_count if token_count is not None else len(text.split()),
        text=text,
    )


# --- resolve_chunk: happy path ---


def test_resolve_chunk_happy_path_returns_resolved_chunk_with_ordered_segments():
    segments = make_segments(3)
    section = make_section(segments, chunk_ids=("chunk_000001",))
    chunk = make_chunk(segments, section_id=section.section_id)

    resolved = resolve_chunk(chunk.chunk_id, [chunk], [section], segments)

    assert isinstance(resolved, ResolvedChunk)
    assert resolved.chunk == chunk
    assert resolved.section == section
    assert resolved.segments == tuple(segments)
    assert [s.segment_id for s in resolved.segments] == list(chunk.segment_ids)


def test_resolve_chunk_orders_segments_matching_chunk_segment_ids_with_repeats():
    # Oversized-piece scenario: two pieces share the same segment_id.
    pieces = [
        TranscriptSegment(
            segment_id="seg_000001", speaker="SPEAKER_00", start=0.0, end=15.0,
            text="part one", words=(), piece_index=0, word_start=0, word_end=2,
        ),
        TranscriptSegment(
            segment_id="seg_000001", speaker="SPEAKER_00", start=0.0, end=15.0,
            text="part two", words=(), piece_index=1, word_start=2, word_end=4,
        ),
    ]
    section = make_section(pieces, chunk_ids=("chunk_000001",))
    chunk = make_chunk(pieces, section_id=section.section_id)

    resolved = resolve_chunk(chunk.chunk_id, [chunk], [section], pieces)

    assert resolved.segments == tuple(pieces)


# --- resolve_chunk: failure modes ---


def test_resolve_chunk_raises_on_unknown_chunk_id():
    segments = make_segments(2)
    section = make_section(segments, chunk_ids=("chunk_000001",))
    chunk = make_chunk(segments, section_id=section.section_id)

    with pytest.raises(ChunkHierarchyError):
        resolve_chunk("chunk_999999", [chunk], [section], segments)


def test_resolve_chunk_raises_on_missing_parent_section():
    segments = make_segments(2)
    section = make_section(segments, section_id="section_000001", chunk_ids=("chunk_000001",))
    chunk = make_chunk(segments, section_id="section_000099")

    with pytest.raises(ChunkHierarchyError):
        resolve_chunk(chunk.chunk_id, [chunk], [section], segments)


def test_resolve_chunk_raises_on_broken_parent_link():
    segments = make_segments(2)
    section = make_section(segments, chunk_ids=("some_other_chunk",))
    chunk = make_chunk(segments, section_id=section.section_id)

    with pytest.raises(ChunkHierarchyError):
        resolve_chunk(chunk.chunk_id, [chunk], [section], segments)


def test_resolve_chunk_raises_when_segment_ids_unresolvable():
    segments = make_segments(2)
    section = make_section(segments, chunk_ids=("chunk_000001",))
    chunk = make_chunk(segments, section_id=section.section_id)
    # Drop one segment from the pool so resolution comes up short.
    incomplete_segments = segments[:1]

    with pytest.raises(ChunkHierarchyError):
        resolve_chunk(chunk.chunk_id, [chunk], [section], incomplete_segments)


def test_resolve_chunk_raises_when_oversized_piece_occurrence_exhausted():
    pieces = [
        TranscriptSegment(
            segment_id="seg_000001", speaker="SPEAKER_00", start=0.0, end=15.0,
            text="part one", words=(), piece_index=0, word_start=0, word_end=2,
        ),
        TranscriptSegment(
            segment_id="seg_000001", speaker="SPEAKER_00", start=0.0, end=15.0,
            text="part two", words=(), piece_index=1, word_start=2, word_end=4,
        ),
    ]
    section = make_section(pieces, chunk_ids=("chunk_000001",))
    chunk = make_chunk(pieces, section_id=section.section_id)
    # Only one occurrence of seg_000001 available even though chunk expects two.
    only_one_piece = pieces[:1]

    with pytest.raises(ChunkHierarchyError):
        resolve_chunk(chunk.chunk_id, [chunk], [section], only_one_piece)


# --- reconstruct_chunk_text ---


def test_reconstruct_chunk_text_matches_hand_built_chunk_text():
    segments = make_segments(3)
    section = make_section(segments, chunk_ids=("chunk_000001",))
    chunk = make_chunk(segments, section_id=section.section_id)
    resolved = resolve_chunk(chunk.chunk_id, [chunk], [section], segments)

    assert reconstruct_chunk_text(resolved) == chunk.text


# --- iter_chunks_for_processing ---


def test_iter_chunks_for_processing_preserves_order_and_count():
    segments = make_segments(4)
    chunks = [
        make_chunk(segments[:2], chunk_id="chunk_000001"),
        make_chunk(segments[2:], chunk_id="chunk_000002"),
    ]

    result = iter_chunks_for_processing(chunks)

    assert result == tuple(chunks)
    assert [c.chunk_id for c in result] == ["chunk_000001", "chunk_000002"]


def test_iter_chunks_for_processing_returns_defensive_copy():
    segments = make_segments(2)
    chunks = [make_chunk(segments, chunk_id="chunk_000001")]

    result = iter_chunks_for_processing(chunks)
    chunks.append(make_chunk(segments, chunk_id="chunk_000002"))

    assert len(result) == 1
    assert result[0].chunk_id == "chunk_000001"
