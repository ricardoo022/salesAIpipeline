from pipeline.qualification.chunking import create_sections
from pipeline.qualification.schemas import (
    ChunkingConfiguration,
    TranscriptSegment,
)

CONFIG = ChunkingConfiguration(section_target_seconds=100, section_overlap_seconds=20)


def make_segments(count, *, start=0.0, first_index=1, step=10.0, duration=5.0):
    segments = []
    for offset in range(count):
        index = first_index + offset
        seg_start = start + offset * step
        segments.append(
            TranscriptSegment(
                segment_id=f"seg_{index:06d}",
                speaker="SPEAKER_00" if index % 2 == 1 else "SPEAKER_01",
                start=seg_start,
                end=seg_start + duration,
                text=f"utterance {index}",
                words=(),
            )
        )
    return segments


def test_short_transcript_yields_a_single_section():
    sections = create_sections(make_segments(5), CONFIG)

    assert len(sections) == 1
    section = sections[0]
    assert section.section_id == "section_000001"
    assert section.sequence == 1
    assert section.start == 0.0
    assert section.end == 45.0
    assert section.segment_ids == tuple(f"seg_{i:06d}" for i in range(1, 6))
    assert section.overlap_segment_ids == ()
    assert section.chunk_ids == ()


def test_sections_are_chronological_with_deterministic_ids_and_sequences():
    sections = create_sections(make_segments(20), CONFIG)

    assert [s.section_id for s in sections] == [
        "section_000001",
        "section_000002",
        "section_000003",
    ]
    assert [s.sequence for s in sections] == [1, 2, 3]
    assert [s.start for s in sections] == [0.0, 80.0, 160.0]
    assert [s.end for s in sections] == [95.0, 175.0, 195.0]
    assert sections == create_sections(make_segments(20), CONFIG)


def test_every_segment_belongs_to_at_least_one_section():
    segments = make_segments(20)

    sections = create_sections(segments, CONFIG)

    covered = [sid for section in sections for sid in section.segment_ids]
    assert set(covered) == {segment.segment_id for segment in segments}


def test_overlap_matches_configured_overlap_window():
    sections = create_sections(make_segments(20), CONFIG)

    assert sections[0].segment_ids == tuple(f"seg_{i:06d}" for i in range(1, 11))
    assert sections[0].overlap_segment_ids == ("seg_000009", "seg_000010")
    assert sections[1].overlap_segment_ids == (
        "seg_000009",
        "seg_000010",
        "seg_000017",
        "seg_000018",
    )
    assert sections[2].overlap_segment_ids == ("seg_000017", "seg_000018")
    assert "seg_000009" in sections[1].segment_ids
    assert "seg_000017" in sections[1].segment_ids
    assert "seg_000018" in sections[2].segment_ids


def test_overlap_follows_the_configured_value():
    no_overlap = create_sections(
        make_segments(20),
        ChunkingConfiguration(section_target_seconds=100, section_overlap_seconds=0),
    )
    assert all(section.overlap_segment_ids == () for section in no_overlap)

    wide = create_sections(
        make_segments(20),
        ChunkingConfiguration(section_target_seconds=100, section_overlap_seconds=30),
    )
    assert wide[0].overlap_segment_ids == ("seg_000008", "seg_000009", "seg_000010")


def test_sections_contain_multiple_speakers():
    segments = make_segments(20)
    by_id = {segment.segment_id: segment for segment in segments}

    sections = create_sections(segments, CONFIG)

    for section in sections:
        speakers = {by_id[sid].speaker for sid in section.segment_ids}
        assert speakers == {"SPEAKER_00", "SPEAKER_01"}


def test_empty_transcript_yields_no_sections():
    assert create_sections([], CONFIG) == []


def test_long_silence_produces_no_empty_sections():
    segments = make_segments(5) + make_segments(5, start=500.0, first_index=6)

    sections = create_sections(segments, CONFIG)

    assert len(sections) == 2
    assert [section.sequence for section in sections] == [1, 2]
    assert sections[1].section_id == "section_000002"
    assert sections[1].start == 500.0
    assert sections[1].end == 545.0


def test_split_segment_pieces_keep_their_shared_source_id_in_membership():
    pieces = [
        TranscriptSegment(
            segment_id="seg_000001", speaker="SPEAKER_00", start=0.0, end=30.0,
            text="part one", words=(), piece_index=0, word_start=0, word_end=2,
        ),
        TranscriptSegment(
            segment_id="seg_000001", speaker="SPEAKER_00", start=0.0, end=30.0,
            text="part two", words=(), piece_index=1, word_start=2, word_end=4,
        ),
    ]

    sections = create_sections(pieces, CONFIG)

    assert len(sections) == 1
    assert sections[0].segment_ids == ("seg_000001", "seg_000001")
    assert sections[0].overlap_segment_ids == ()
