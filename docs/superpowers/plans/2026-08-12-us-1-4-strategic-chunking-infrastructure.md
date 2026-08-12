# US-1.4 Support Strategic Chunking Infrastructure Implementation Plan

**Goal:** Give the extraction harness a real, loudly-failing way to walk the
transcript hierarchy in both directions — chunk → parent section → original
source segments, and the full chunk set → downstream processing — reusing
the exact rendering logic already used to produce `ExtractionChunk.text`,
so a rendered chunk can be independently reconstructed and verified against
its source segments, without introducing any relevance-based (vector)
selection step.

This story adds no new chunking *behavior*: sections and chunks were
already fully linked via `ConversationSection.chunk_ids` /
`ExtractionChunk.section_id` / `ExtractionChunk.segment_ids` (US-1.2/1.3).
It adds a **resolution and reconstruction API** on top of that existing,
unmodified structure, plus explicit, tested failure modes so a broken
reference raises rather than silently drops data.

**Architecture:** All new code lives in `pipeline/qualification/chunking.py`
(no new module — the planned module list in `CLAUDE.md` has no "hierarchy"
entry, and the new logic is a thin layer directly on top of `chunking.py`'s
existing private helpers). One new frozen dataclass, `ResolvedChunk` (chunk,
section, ordered segments), was added to `pipeline/qualification/schemas.py`.
`chunking.py` gained: `ChunkHierarchyError(ValueError)`; a
`_resolve_segment_ids(segment_ids, by_id)` helper extracted (behavior-
preserving) from the old `_segments_for_section` body, with
`_segments_for_section` reduced to a one-line wrapper over it;
`resolve_chunk(chunk_id, chunks, sections, segments) -> ResolvedChunk`,
which raises `ChunkHierarchyError` for an unknown chunk_id, a missing
parent section, a section that doesn't list the chunk back in its
`chunk_ids`, or a segment-resolution count mismatch (one combined check,
covering both "segment_id absent" and "oversized-piece occurrence
exhausted"); `reconstruct_chunk_text(resolved) -> str`, which calls the
existing private `_render` directly on `resolved.segments`; and
`iter_chunks_for_processing(chunks) -> tuple[ExtractionChunk, ...]`, a
deliberately trivial defensive-copy pass-through with no scoring/filtering
parameters — the concrete answer to "no vector search."

**Tech Stack:** Python standard library only (`dataclasses`, `typing`),
matching US-1.1–1.3 — no new dependency.

**Execution notes (as run):**
- Executed via `superpowers:subagent-driven-development` on branch
  `feat/us-1-4-strategic-chunking-infrastructure`, cut from `main` (which
  already carried US-1.1/1.2/1.3 merged).
- Task 1 (`ResolvedChunk` schema, `general-purpose`/haiku) ran first and
  landed before Tasks 2 and 3, since both import `ResolvedChunk`.
- Tasks 2 (`resolve_chunk` implementation + unit tests,
  `general-purpose`/sonnet) and 3 (integration tests + docs,
  `general-purpose`/sonnet) ran in parallel in separate isolated git
  worktrees (`pipeline/qualification/chunking.py` +
  `tests/qualification/unit/test_hierarchy.py` vs.
  `tests/qualification/integration/test_hierarchy_traceability.py` +
  three READMEs/CLAUDE.md — disjoint file sets) and were merged back with
  two conflict-free `--no-ff` merges. Task 3's integration tests were
  RED (`ImportError: cannot import name 'ChunkHierarchyError'`) until
  Task 2's implementation merged — Task 3's implementer independently
  verified its test logic against a scratch, uncommitted reference
  implementation before finalizing, then went green for real once merged.
  Expected TDD red, not a defect, same sequencing as US-1.3.
- Task 2's review found the "Global Constraints" block was accidentally
  left out of the `task-brief` extraction (the constraints paragraph sat
  above the first `### Task N` heading rather than inside a task section);
  the implementer wrote a reasonable `ChunkHierarchyError` docstring from
  context anyway and the reviewer confirmed all constraints were still met
  in the code. No fix dispatch was needed — noted here so a future plan
  keeps global constraints inside (or duplicated into) each task heading.
- Both task reviews came back clean (Approved, no Critical/Important
  findings) on the first pass — no fix/re-review loop was needed.
- Task 4 (this judge pass) ran directly by the orchestrator, not delegated.

---

## Acceptance criteria → tests (final judge pass)

| Acceptance criterion (`docs/Problem/EPICS-AND-USER-STORIES.md` US-1.4) | Proven by |
|---|---|
| Parent sections contain child extraction chunks | Pre-existing US-1.3 `chunk_ids` behavior (unchanged, still covered by `test_sections_are_returned_with_populated_chunk_ids` in `tests/qualification/unit/test_chunking.py`); re-verified end-to-end by every `_assert_full_hierarchy_traceable` call in `tests/qualification/integration/test_hierarchy_traceability.py` |
| Child chunks retain original transcript segment references | `test_resolve_chunk_happy_path_returns_resolved_chunk_with_ordered_segments`, `test_resolve_chunk_orders_segments_matching_chunk_segment_ids_with_repeats` (unit, `tests/qualification/unit/test_hierarchy.py`); `test_fixture_chunks_resolve_to_traceable_hierarchy`, `test_long_call_chunks_resolve_to_traceable_hierarchy_with_overlap` (integration) |
| Parent-child relationships are available to downstream processing | `resolve_chunk()` itself (`pipeline/qualification/chunking.py:314-357`) plus `test_resolve_chunk_raises_on_broken_parent_link` (unit — proves the `chunk_ids` back-link is checked, not assumed) and the `chunk.chunk_id in resolved.section.chunk_ids` assertion inside `_assert_full_hierarchy_traceable` (integration, exercised for every chunk in both fixture and synthetic transcripts) |
| Framework behavior cannot silently discard transcript metadata | `test_resolve_chunk_raises_on_unknown_chunk_id`, `test_resolve_chunk_raises_on_missing_parent_section`, `test_resolve_chunk_raises_on_broken_parent_link`, `test_resolve_chunk_raises_when_segment_ids_unresolvable`, `test_resolve_chunk_raises_when_oversized_piece_occurrence_exhausted` (all unit — every failure mode raises `ChunkHierarchyError`, never returns `None`/partial); `test_resolve_chunk_raises_on_unknown_chunk_id` (integration, against real pipeline output) |
| The hierarchy preserves speakers, timestamps, and exact text | Field-by-field equality assertions in `_assert_full_hierarchy_traceable` (integration: `resolved_segment.speaker`/`.start`/`.end`/`.text` vs. the normalized source segment, for every chunk in both transcripts) |
| Every chunk can be resolved from its chunk ID to its parent section and original source segments | `test_fixture_chunks_resolve_to_traceable_hierarchy` and `test_long_call_chunks_resolve_to_traceable_hierarchy_with_overlap` (integration — loop every chunk `create_chunks` produces over the fixture transcript and a synthetic multi-section/overlap transcript; every one resolves) |
| Rendered chunk text can be reconstructed from the referenced source segments | `test_reconstruct_chunk_text_matches_hand_built_chunk_text` (unit); the `reconstruct_chunk_text(resolved) == chunk.text` assertion in `_assert_full_hierarchy_traceable` (integration, byte-for-byte, including overlap chunks in `test_long_call_chunks_resolve_to_traceable_hierarchy_with_overlap`, which asserts `any(chunk.overlap_segment_ids for chunk in chunks)` before running the check) |
| The hierarchy does not require vector search to select which transcript chunks are processed | `test_iter_chunks_for_processing_preserves_order_and_count`, `test_iter_chunks_for_processing_returns_defensive_copy` (unit); `test_iter_chunks_for_processing_covers_every_chunk_in_order` (integration, against real `create_chunks` output) |

## Test results

- `tests/qualification/` (unit + integration): 59/59 passed (43
  pre-existing US-1.1/1.2/1.3 tests untouched and still green + 2 for
  `ResolvedChunk` (Task 1) + 10 unit + 4 integration for US-1.4).
- Full repo suite (`tests/`, deselecting the two documented DeepFace/torch
  and audeering/torch segfault-pair tests): 268 passed, 2 deselected, 1
  failed. The one failure —
  `tests/test_llm_analysis.py::TestRunAnalysisIntegration::test_with_real_outputs`
  — is the same pre-existing real-API integration test flagged in the
  US-1.3 judge pass, failing on `anthropic.BadRequestError: credit balance
  too low`, an environment/billing condition, not a code regression.
