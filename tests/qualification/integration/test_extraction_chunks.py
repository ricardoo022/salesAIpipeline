import json
from pathlib import Path

from pipeline.qualification.chunking import create_chunks, create_sections
from pipeline.qualification.schemas import (
    ChunkingConfiguration,
    ExtractionChunk,
    normalize_transcript,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "transcript.json"


def test_fixture_transcript_forms_traceable_chunks():
    raw_before = FIXTURE.read_bytes()
    segments = normalize_transcript(json.loads(raw_before))
    sections = create_sections(segments)

    updated_sections, chunks = create_chunks(segments, sections)

    assert FIXTURE.read_bytes() == raw_before

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, ExtractionChunk)
    assert chunk.chunk_id == "chunk_000001"
    assert chunk.section_id == "section_000001"
    assert chunk.sequence == 1
    assert chunk.start == 0.0
    assert chunk.end == 2.5
    assert chunk.segment_ids == ("seg_000001", "seg_000002")
    assert chunk.overlap_segment_ids == ()

    expected_text = (
        "SPEAKER_00 [00:00:00]: Who approves this internally?\n"
        "SPEAKER_01 [00:00:01]: Our VP of Finance."
    )
    assert chunk.text == expected_text
    assert "Who approves this internally?" in chunk.text
    assert "Our VP of Finance." in chunk.text
    assert chunk.token_count == len(expected_text.split())

    all_segment_ids = {segment.segment_id for segment in segments}
    covered = {segment_id for c in chunks for segment_id in c.segment_ids}
    assert covered == all_segment_ids

    assert len(updated_sections) == 1
    assert updated_sections[0].section_id == "section_000001"
    assert updated_sections[0].chunk_ids == ("chunk_000001",)

    again_sections, again_chunks = create_chunks(segments, create_sections(segments))
    assert again_chunks == chunks
    assert again_sections == updated_sections


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


def test_long_call_chunks_cover_every_source_segment_with_overlap():
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

    assert len(sections) > 1

    all_segment_ids = {segment.segment_id for segment in segments}
    covered = {segment_id for chunk in chunks for segment_id in chunk.segment_ids}
    assert covered == all_segment_ids

    chunks_by_section: dict[str, list[ExtractionChunk]] = {}
    for chunk in chunks:
        chunks_by_section.setdefault(chunk.section_id, []).append(chunk)

    for section in updated_sections:
        section_chunks = chunks_by_section.get(section.section_id, [])
        assert section.chunk_ids != ()
        assert set(section.chunk_ids) == {chunk.chunk_id for chunk in section_chunks}
        assert [chunk.sequence for chunk in section_chunks] == list(
            range(1, len(section_chunks) + 1)
        )

    multi_chunk_sections = {
        section_id: section_chunks
        for section_id, section_chunks in chunks_by_section.items()
        if len(section_chunks) >= 2
    }
    assert multi_chunk_sections

    found_overlap = any(
        chunk.overlap_segment_ids
        for section_chunks in multi_chunk_sections.values()
        for chunk in section_chunks
        if chunk.sequence > 1
    )
    assert found_overlap

    for section_chunks in multi_chunk_sections.values():
        assert section_chunks[0].overlap_segment_ids == ()

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))

    again_sections, again_chunks = create_chunks(
        segments, create_sections(segments, config), config
    )
    assert again_chunks == chunks
    assert again_sections == updated_sections
