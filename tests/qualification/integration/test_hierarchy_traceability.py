"""Integration tests for chunk hierarchy resolution (US-1.4).

Follows test_extraction_chunks.py's convention: real pipeline run
(normalize_transcript -> create_sections -> create_chunks), no mocks.

These tests target the resolution API planned for
`pipeline/qualification/chunking.py` -- `resolve_chunk`,
`reconstruct_chunk_text`, `iter_chunks_for_processing`, and
`ChunkHierarchyError` (US-1.4, Task 2). That API does not exist yet in this
worktree, so this module currently fails to import with an ImportError.
That is the expected starting state for this file (Task 2 runs in a
parallel worktree), not a bug in these tests -- it should turn green once
Task 2 merges.
"""

import json
from pathlib import Path

import pytest

from pipeline.qualification.chunking import (
    ChunkHierarchyError,
    create_chunks,
    create_sections,
    iter_chunks_for_processing,
    reconstruct_chunk_text,
    resolve_chunk,
)
from pipeline.qualification.schemas import (
    ChunkingConfiguration,
    ResolvedChunk,
    normalize_transcript,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "transcript.json"


def _assert_full_hierarchy_traceable(segments, sections, chunks):
    """Resolve every chunk in `chunks` and assert the full hierarchy holds:
    section linkage is consistent both ways, resolved segments match the
    normalized source segments field-for-field, and the reconstructed text
    matches the chunk's own rendered text byte-for-byte."""
    segments_by_id = {segment.segment_id: segment for segment in segments}
    sections_by_id = {section.section_id: section for section in sections}

    assert chunks  # sanity: nothing to prove if there is nothing to resolve

    for chunk in chunks:
        resolved = resolve_chunk(chunk.chunk_id, chunks, sections, segments)

        assert isinstance(resolved, ResolvedChunk)
        assert resolved.chunk == chunk
        assert resolved.section == sections_by_id[chunk.section_id]
        assert resolved.section.section_id == chunk.section_id
        assert chunk.chunk_id in resolved.section.chunk_ids

        assert len(resolved.segments) == len(chunk.segment_ids)
        for resolved_segment, segment_id in zip(resolved.segments, chunk.segment_ids):
            source_segment = segments_by_id[segment_id]
            assert resolved_segment.segment_id == segment_id
            assert resolved_segment.speaker == source_segment.speaker
            assert resolved_segment.start == source_segment.start
            assert resolved_segment.end == source_segment.end
            assert resolved_segment.text == source_segment.text

        assert reconstruct_chunk_text(resolved) == chunk.text


def test_fixture_chunks_resolve_to_traceable_hierarchy():
    raw = json.loads(FIXTURE.read_bytes())
    segments = normalize_transcript(raw)
    sections = create_sections(segments)
    updated_sections, chunks = create_chunks(segments, sections)

    _assert_full_hierarchy_traceable(segments, updated_sections, chunks)


def _build_long_call_transcript():
    pattern = [
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_00",
    ]
    raw = []
    for index in range(90):
        speaker = pattern[index % len(pattern)]
        start = index * 10.0
        raw.append(
            {
                "speaker": speaker,
                "start": start,
                "end": start + 5.0,
                "text": f"turn {index}",
                "words": [{"word": "turn", "start": start, "end": start + 2.0}],
            }
        )
    return raw


def test_long_call_chunks_resolve_to_traceable_hierarchy_with_overlap():
    raw = _build_long_call_transcript()
    config = ChunkingConfiguration(
        section_target_seconds=100.0,
        section_overlap_seconds=10.0,
        max_chunk_tokens=15,
        chunk_overlap_turns=2,
        max_overlap_tokens=50,
    )

    segments = normalize_transcript(raw)
    sections = create_sections(segments, config)
    updated_sections, chunks = create_chunks(segments, sections, config)

    # This transcript must actually exercise multiple sections and overlap
    # chunks -- otherwise this test would add nothing beyond the (single
    # section, no overlap) fixture case above.
    assert len(updated_sections) > 1
    assert any(len(section.chunk_ids) >= 2 for section in updated_sections)
    assert any(chunk.overlap_segment_ids for chunk in chunks)

    _assert_full_hierarchy_traceable(segments, updated_sections, chunks)


def test_resolve_chunk_raises_on_unknown_chunk_id():
    raw = json.loads(FIXTURE.read_bytes())
    segments = normalize_transcript(raw)
    sections = create_sections(segments)
    updated_sections, chunks = create_chunks(segments, sections)

    with pytest.raises(ChunkHierarchyError):
        resolve_chunk("chunk_999999", chunks, updated_sections, segments)


def test_iter_chunks_for_processing_covers_every_chunk_in_order():
    raw = json.loads(FIXTURE.read_bytes())
    segments = normalize_transcript(raw)
    sections = create_sections(segments)
    updated_sections, chunks = create_chunks(segments, sections)

    processed = iter_chunks_for_processing(chunks)

    assert isinstance(processed, tuple)
    assert processed == tuple(chunks)
    assert len(processed) == len(chunks)
