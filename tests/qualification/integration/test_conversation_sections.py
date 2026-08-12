import json
from pathlib import Path

from pipeline.qualification.chunking import create_sections
from pipeline.qualification.schemas import (
    ChunkingConfiguration,
    normalize_transcript,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "transcript.json"


def test_fixture_transcript_forms_a_single_traceable_section():
    raw_before = FIXTURE.read_bytes()
    segments = normalize_transcript(json.loads(raw_before))

    sections = create_sections(segments)

    assert len(sections) == 1
    section = sections[0]
    assert section.section_id == "section_000001"
    assert section.sequence == 1
    assert section.start == 0.0
    assert section.end == 2.5
    assert section.segment_ids == ("seg_000001", "seg_000002")
    assert section.overlap_segment_ids == ()
    assert section.chunk_ids == ()
    assert FIXTURE.read_bytes() == raw_before
    assert sections == create_sections(normalize_transcript(json.loads(raw_before)))


def test_long_call_normalize_then_sections_covers_every_source_segment():
    raw = [
        {
            "speaker": "SPEAKER_00" if index % 2 == 0 else "SPEAKER_01",
            "start": index * 10.0,
            "end": index * 10.0 + 5.0,
            "text": f"turn {index}",
            "words": [
                {"word": "turn", "start": index * 10.0, "end": index * 10.0 + 2.0}
            ],
        }
        for index in range(120)  # 20 minutes of conversation
    ]

    segments = normalize_transcript(raw)
    sections = create_sections(segments, ChunkingConfiguration())

    assert len(sections) == 3
    covered = [sid for section in sections for sid in section.segment_ids]
    assert set(covered) == {segment.segment_id for segment in segments}
    assert [section.sequence for section in sections] == [1, 2, 3]
    starts = [section.start for section in sections]
    assert starts == sorted(starts)
    by_id = {segment.segment_id: segment for segment in segments}
    for section in sections:
        speakers = {by_id[sid].speaker for sid in set(section.segment_ids)}
        assert speakers == {"SPEAKER_00", "SPEAKER_01"}
