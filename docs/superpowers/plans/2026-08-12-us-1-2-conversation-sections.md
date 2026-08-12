# US-1.2 Create Conversation Sections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group normalized transcript segments into chronological, deterministically overlapping conversation sections (the Epic 1 `ConversationSection` schema), driven by an explicit `ChunkingConfiguration`, without modifying `output/transcript.json`.

**Architecture:** Pure standard-library code, same boundary style as US-1.1. `ChunkingConfiguration` and `ConversationSection` are frozen dataclasses added to `pipeline/qualification/schemas.py` (the documented home for Epic 1 data contracts). The sectioning algorithm lives in a new `pipeline/qualification/chunking.py` (the documented home for hierarchical chunking): fixed time windows of `section_target_seconds` stride forward by `target - overlap` from the earliest segment start, so neighboring windows overlap by exactly the configured value. A segment belongs to every window containing its start time, which guarantees complete coverage; empty windows (long silences) are skipped and emitted sections are renumbered, keeping IDs deterministic.

**Tech Stack:** Python standard library dataclasses, pytest, JSON fixture data.

**Execution notes:**
- No git commits. Repository policy requires explicit user approval for git mutations; leave all changes in the working tree.
- Task 1 must complete first (it defines the types). Tasks 2 and 3 run in parallel after Task 1; their files are disjoint. Task 3's integration tests stay RED (ImportError) until Task 2 lands — that is expected TDD red; the final judge run turns them green.
- US-1.2 scope boundary: `chunk_ids` exists on the schema but stays empty until US-1.3 creates chunks. No CLI entry point, no coverage record, no runtime artifacts yet.

---

### Task 1: Add ChunkingConfiguration and ConversationSection schemas

**Files:**
- Modify: `pipeline/qualification/schemas.py` (append; do not touch existing US-1.1 code)
- Test: `tests/qualification/unit/test_schemas.py` (append)

- [ ] **Step 1: Write the failing schema tests**

In `tests/qualification/unit/test_schemas.py`, replace the existing import line
`from pipeline.qualification.schemas import normalize_transcript` with:

```python
from pipeline.qualification.schemas import (
    ChunkingConfiguration,
    ConversationSection,
    normalize_transcript,
)
```

Append to the same file:

```python
def test_chunking_configuration_defaults_match_documented_strategy():
    config = ChunkingConfiguration()

    assert config.section_target_seconds == 480.0
    assert config.section_overlap_seconds == 30.0
    assert config.max_chunk_tokens == 1200
    assert config.chunk_overlap_turns == 2
    assert config.max_overlap_tokens == 250
    assert isinstance(config.tokenizer, str) and config.tokenizer


def test_chunking_configuration_rejects_invalid_section_geometry():
    with pytest.raises(ValueError, match="section_target_seconds"):
        ChunkingConfiguration(section_target_seconds=0)
    with pytest.raises(ValueError, match="section_overlap_seconds"):
        ChunkingConfiguration(section_overlap_seconds=-1)
    with pytest.raises(ValueError, match="section_overlap_seconds"):
        ChunkingConfiguration(section_target_seconds=100, section_overlap_seconds=100)
    with pytest.raises(ValueError, match="max_chunk_tokens"):
        ChunkingConfiguration(max_chunk_tokens=0)


def test_chunking_configuration_is_immutable():
    config = ChunkingConfiguration()

    with pytest.raises(AttributeError):
        config.section_target_seconds = 60


def test_conversation_section_records_membership_and_is_immutable():
    section = ConversationSection(
        section_id="section_000001",
        sequence=1,
        start=0.0,
        end=8.0,
        segment_ids=("seg_000001", "seg_000002"),
        overlap_segment_ids=("seg_000002",),
        chunk_ids=(),
    )

    assert section.segment_ids == ("seg_000001", "seg_000002")
    assert section.overlap_segment_ids == ("seg_000002",)
    assert section.chunk_ids == ()
    assert not hasattr(section, "topic")
    assert not hasattr(section, "bant")
    with pytest.raises(AttributeError):
        section.sequence = 2
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/qualification/unit/test_schemas.py -q`

Expected: FAIL with `ImportError: cannot import name 'ChunkingConfiguration'`.

- [ ] **Step 3: Implement the schemas**

Append to `pipeline/qualification/schemas.py`:

```python
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
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/qualification/unit/test_schemas.py -q`

Expected: all tests pass (5 existing + 4 new).

### Task 2: Implement create_sections (depends on Task 1)

**Files:**
- Create: `pipeline/qualification/chunking.py`
- Test: `tests/qualification/unit/test_chunking.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/qualification/unit/test_chunking.py`:

```python
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
```

Expected window math for `make_segments(20)` with target=100, overlap=20 (stride=80): windows `[0,100)`, `[80,180)`, `[160,260)`; members by start time are idx 0-9, idx 8-17, idx 16-19; shared members are idx 8-9 (`seg_000009`, `seg_000010`) and idx 16-17 (`seg_000017`, `seg_000018`). With overlap=30 (stride=70): windows `[0,100)`, `[70,170)`, `[140,240)`; the first section shares idx 7-9.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/qualification/unit/test_chunking.py -q`

Expected: collection failure because `pipeline.qualification.chunking` does not yet exist.

- [ ] **Step 3: Implement create_sections**

Create `pipeline/qualification/chunking.py`:

```python
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
```

Note: the intra-package import is relative (`from .schemas import ...`) so the module works both as `pipeline.qualification.chunking` (tests, `PYTHONPATH=.`) and as `qualification.chunking` (a future `pipeline/07_qualification.py` script puts `pipeline/` on `sys.path`). This matches the documented import-path duality.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/qualification/unit/test_chunking.py -q`

Expected: all 9 tests pass.

### Task 3: Integration tests and docs (parallel with Task 2)

**Files:**
- Create: `tests/qualification/integration/test_conversation_sections.py`
- Modify: `docs/Problem/EPICS-AND-USER-STORIES.md` (US-1.2 status block)
- Modify: `CLAUDE.md` (US-1.1 paragraph → cover US-1.2)

- [ ] **Step 1: Write the failing integration tests**

Create `tests/qualification/integration/test_conversation_sections.py`:

```python
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
```

Default config (480s target, 30s overlap, 450s stride) on 120 segments starting every 10s (0-1190) yields windows `[0,480)`, `[450,930)`, `[900,1380)` — 3 sections, full coverage, shared segments in both overlap bands.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/qualification/integration/test_conversation_sections.py -q`

Expected: collection failure (ImportError) until Task 2 lands. This is the expected TDD red for the parallel track; do NOT implement `chunking.py` from this task.

- [ ] **Step 3: Mark US-1.2 done in the epics doc**

In `docs/Problem/EPICS-AND-USER-STORIES.md`, directly under the `### US-1.2: Create Conversation Sections` heading, insert (mirroring the US-1.1 status block):

```markdown
**Status: Done**

Implemented by `pipeline/qualification/chunking.py` (`create_sections`) and the
`ChunkingConfiguration` / `ConversationSection` schemas in
`pipeline/qualification/schemas.py`, covered by unit and integration tests in
`tests/qualification/`. Section windows are deterministic, overlap by exactly
the configured `section_overlap_seconds`, and every source segment belongs to
at least one section.
```

- [ ] **Step 4: Update CLAUDE.md status**

In `CLAUDE.md`, replace the paragraph beginning `US-1.1 is implemented in `pipeline/qualification/schemas.py`.` with:

```markdown
US-1.1 and US-1.2 are implemented in `pipeline/qualification/schemas.py`
(immutable `TranscriptSegment`, `ChunkingConfiguration`, and
`ConversationSection` contracts plus the pure `normalize_transcript()`
boundary) and `pipeline/qualification/chunking.py` (`create_sections()` —
deterministic time-window sections striding by `target - overlap`; membership
by segment start guarantees full coverage; empty windows are skipped and
sections renumbered). `normalize_transcript()` assigns deterministic
`seg_000001`-style source IDs, preserves speaker/timestamps/exact text/word
timestamps, and does not modify `output/transcript.json`. Oversized segments
are split only at complete word boundaries; pieces retain the original
`segment_id` and record `piece_index`, `word_start`, and `word_end`. Word
metadata is recursively immutable. Extraction chunks and coverage validation
are not yet implemented.
```

Also update the test-count line in `CLAUDE.md` ("218 tests: ... plus six qualification tests") to reflect the new qualification test count (6 existing + 4 schema + 9 chunking + 2 integration = 21 qualification tests; 233 total).

### Task 4: Judge the acceptance criteria and regression safety

**Files:**
- Modify: none unless test failures identify a required correction

- [ ] **Step 1: Run the qualification suite exactly as CI does**

Run: `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest pytest tests/qualification/unit/test_*.py tests/qualification/integration/test_*.py -v`

Expected: all qualification unit and integration tests pass (21 tests).

- [ ] **Step 2: Run all repository tests**

Run: `python -m pytest tests/ -q --deselect tests/test_emotion_face.py::TestExtractFaceEmotionIntegration::test_with_real_meeting_video`

Expected: existing suite and qualification unit/integration tests pass, with the documented native-library integration test deselected. (If the local venv lacks heavy pipeline deps, fall back to `PYTHONPATH=. python -m pytest tests/qualification tests/test_emotion_voice.py -q` style scoped runs and record what was executed.)

- [ ] **Step 3: Check each US-1.2 acceptance criterion**

Map each criterion to its evidence before stopping:

| Criterion | Evidence |
|---|---|
| Sections preserve chronological order | `test_sections_are_chronological_with_deterministic_ids_and_sequences`; integration `starts == sorted(starts)` |
| Every transcript segment belongs to at least one section | `test_every_segment_belongs_to_at_least_one_section`; integration coverage assert |
| Sections may overlap where needed | `test_overlap_matches_configured_overlap_window` |
| Sections are organizational, not qualification labels | schema has no topic/BANT field; `test_conversation_section_records_membership_and_is_immutable` (`not hasattr`) |
| Section IDs and sequence numbers deterministic | repeat-run equality asserts in unit + integration tests |
| Section records start/end, ordered segment IDs, overlap IDs, chunk IDs | `test_short_transcript_yields_a_single_section`; schema test |
| Overlap created per configured value | `test_overlap_follows_the_configured_value` (0 vs 20 vs 30) |
| Section may contain multiple speakers | `test_sections_contain_multiple_speakers`; integration speaker assert |

Stop only when unit and integration tests pass and every criterion is evidenced.
