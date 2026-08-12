from pipeline.qualification.chunking import create_chunks, create_sections
from pipeline.qualification.schemas import (
    ChunkingConfiguration,
    ConversationSection,
    TranscriptSegment,
)

CONFIG = ChunkingConfiguration(section_target_seconds=100, section_overlap_seconds=20)


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


def make_section(segments, *, section_id="section_000001", sequence=1, overlap_segment_ids=()):
    return ConversationSection(
        section_id=section_id,
        sequence=sequence,
        start=min(segment.start for segment in segments),
        end=max(segment.end for segment in segments),
        segment_ids=tuple(segment.segment_id for segment in segments),
        overlap_segment_ids=tuple(overlap_segment_ids),
    )


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


# --- create_chunks (US-1.3) ---


def test_chunks_never_cut_a_segment():
    segments = make_segments(6)
    section = make_section(segments)
    config = ChunkingConfiguration(max_chunk_tokens=9, chunk_overlap_turns=0)

    _, chunks = create_chunks(segments, [section], config)

    by_id = {segment.segment_id: segment for segment in segments}
    seen_ids = set()
    for chunk in chunks:
        for segment_id in chunk.segment_ids:
            assert segment_id in by_id
            seen_ids.add(segment_id)
            assert by_id[segment_id].text in chunk.text
    assert seen_ids == {segment.segment_id for segment in segments}


def test_related_speaker_turns_are_kept_together_when_possible():
    turn1 = make_segments(3, first_index=1, speakers=["SPEAKER_00"] * 3, start=0.0)
    turn2 = make_segments(3, first_index=4, speakers=["SPEAKER_01"] * 3, start=30.0)
    segments = turn1 + turn2
    section = make_section(segments)
    config = ChunkingConfiguration(max_chunk_tokens=1000, chunk_overlap_turns=0)

    _, chunks = create_chunks(segments, [section], config)

    assert len(chunks) == 1
    assert chunks[0].segment_ids == tuple(segment.segment_id for segment in segments)


def test_chunk_respects_max_chunk_tokens():
    segments = make_segments(6)
    section = make_section(segments)
    config = ChunkingConfiguration(max_chunk_tokens=9, chunk_overlap_turns=0)

    _, chunks = create_chunks(segments, [section], config)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= config.max_chunk_tokens


def test_oversized_single_turn_becomes_its_own_chunk_and_exceeds_limit():
    long_text = " ".join(f"word{i}" for i in range(20))
    segments = make_segments(1, text=[long_text])
    section = make_section(segments)
    config = ChunkingConfiguration(max_chunk_tokens=10, chunk_overlap_turns=0)

    _, chunks = create_chunks(segments, [section], config)

    assert len(chunks) == 1
    assert chunks[0].segment_ids == (segments[0].segment_id,)
    assert chunks[0].token_count > config.max_chunk_tokens


def test_neighboring_chunks_within_a_section_include_overlap():
    segments = make_segments(6)
    section = make_section(segments)
    config = ChunkingConfiguration(
        max_chunk_tokens=9, chunk_overlap_turns=1, max_overlap_tokens=10
    )

    _, chunks = create_chunks(segments, [section], config)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert current.overlap_segment_ids != ()
        for segment_id in current.overlap_segment_ids:
            assert segment_id in previous.segment_ids


def test_overlap_uses_complete_turns_and_respects_max_overlap_tokens():
    segments = make_segments(
        6, speakers=["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_00", "SPEAKER_00"]
    )
    section = make_section(segments)
    config = ChunkingConfiguration(
        max_chunk_tokens=9, chunk_overlap_turns=1, max_overlap_tokens=10
    )

    _, chunks = create_chunks(segments, [section], config)

    assert len(chunks) == 3
    assert chunks[1].overlap_segment_ids == ("seg_000001", "seg_000002")
    assert chunks[2].overlap_segment_ids == ("seg_000003", "seg_000004")


def test_overlap_always_includes_nearest_turn_even_if_it_exceeds_max_overlap_tokens():
    segments = make_segments(
        6, speakers=["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_00", "SPEAKER_00"]
    )
    section = make_section(segments)
    config = ChunkingConfiguration(
        max_chunk_tokens=9, chunk_overlap_turns=1, max_overlap_tokens=1
    )

    _, chunks = create_chunks(segments, [section], config)

    assert len(chunks) == 3
    assert chunks[1].overlap_segment_ids == ("seg_000001", "seg_000002")


def test_normal_segments_are_never_cut_for_overlap():
    segments = make_segments(6)
    section = make_section(segments)
    config = ChunkingConfiguration(
        max_chunk_tokens=9, chunk_overlap_turns=1, max_overlap_tokens=10
    )

    by_id = {segment.segment_id: segment for segment in segments}
    _, chunks = create_chunks(segments, [section], config)

    for chunk in chunks:
        for segment_id in chunk.overlap_segment_ids:
            assert segment_id in by_id
            assert by_id[segment_id].text in chunk.text


def test_chunk_ids_are_globally_unique_and_sequence_resets_per_section():
    section_a_segments = make_segments(2, first_index=1, start=0.0)
    section_b_segments = make_segments(2, first_index=3, start=100.0)
    section_a = make_section(section_a_segments, section_id="section_000001", sequence=1)
    section_b = make_section(section_b_segments, section_id="section_000002", sequence=2)
    all_segments = section_a_segments + section_b_segments
    config = ChunkingConfiguration(max_chunk_tokens=4, chunk_overlap_turns=0)

    _, chunks = create_chunks(all_segments, [section_a, section_b], config)

    assert [chunk.chunk_id for chunk in chunks] == [
        "chunk_000001",
        "chunk_000002",
        "chunk_000003",
        "chunk_000004",
    ]
    section_a_chunks = [chunk for chunk in chunks if chunk.section_id == "section_000001"]
    section_b_chunks = [chunk for chunk in chunks if chunk.section_id == "section_000002"]
    assert [chunk.sequence for chunk in section_a_chunks] == [1, 2]
    assert [chunk.sequence for chunk in section_b_chunks] == [1, 2]


def test_every_chunk_retains_section_id_and_ordered_segment_ids():
    turn1 = make_segments(3, first_index=1, speakers=["SPEAKER_00"] * 3, start=0.0)
    turn2 = make_segments(3, first_index=4, speakers=["SPEAKER_01"] * 3, start=30.0)
    segments = turn1 + turn2
    section = make_section(segments, section_id="section_000005", sequence=5)
    config = ChunkingConfiguration(max_chunk_tokens=1000, chunk_overlap_turns=0)

    _, chunks = create_chunks(segments, [section], config)

    assert len(chunks) == 1
    assert chunks[0].section_id == "section_000005"
    assert chunks[0].segment_ids == tuple(segment.segment_id for segment in segments)


def test_token_count_matches_rendered_text_via_tokenizer():
    segments = make_segments(3)
    section = make_section(segments)
    config = ChunkingConfiguration(max_chunk_tokens=1000, chunk_overlap_turns=0)
    custom_tokenizer = lambda text: len(text)

    _, chunks = create_chunks(segments, [section], config, tokenizer=custom_tokenizer)

    assert len(chunks) == 1
    assert chunks[0].token_count == custom_tokenizer(chunks[0].text)


def test_rendered_text_includes_speaker_labels_and_timestamps():
    segments = make_segments(1, text=["utterance 1"])
    section = make_section(segments)
    config = ChunkingConfiguration(max_chunk_tokens=1000, chunk_overlap_turns=0)

    _, chunks = create_chunks(segments, [section], config)

    assert chunks[0].text == "SPEAKER_00 [00:00:00]: utterance 1"


def test_sections_are_returned_with_populated_chunk_ids():
    section_a_segments = make_segments(2, first_index=1, start=0.0)
    section_b_segments = make_segments(2, first_index=3, start=100.0)
    section_a = make_section(section_a_segments, section_id="section_000001", sequence=1)
    section_b = make_section(section_b_segments, section_id="section_000002", sequence=2)
    all_segments = section_a_segments + section_b_segments
    config = ChunkingConfiguration(max_chunk_tokens=4, chunk_overlap_turns=0)

    updated_sections, chunks = create_chunks(all_segments, [section_a, section_b], config)

    for section in updated_sections:
        expected = tuple(
            chunk.chunk_id for chunk in chunks if chunk.section_id == section.section_id
        )
        assert section.chunk_ids == expected


def test_create_chunks_is_deterministic():
    segments = make_segments(6)
    section = make_section(segments)
    config = ChunkingConfiguration(max_chunk_tokens=9, chunk_overlap_turns=1, max_overlap_tokens=10)

    first_sections, first_chunks = create_chunks(segments, [section], config)
    second_sections, second_chunks = create_chunks(segments, [section], config)

    assert first_sections == second_sections
    assert first_chunks == second_chunks


def test_empty_sections_yield_no_chunks():
    assert create_chunks([], [], CONFIG) == ([], [])
