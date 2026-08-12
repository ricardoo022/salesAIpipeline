"""Hierarchical transcript chunking: conversation sections (US-1.2)."""

from .schemas import (
    ChunkingConfiguration,
    ConversationSection,
    TranscriptSegment,
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
