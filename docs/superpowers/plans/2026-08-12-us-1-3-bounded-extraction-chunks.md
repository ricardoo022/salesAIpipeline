# US-1.3 Create Bounded Extraction Chunks Implementation Plan

**Goal:** Turn each `ConversationSection`'s member segments into one or more
bounded, speaker-labeled, token-counted `ExtractionChunk`s (the Epic 1
`ExtractionChunk` schema) — the input a BANT topic agent actually reads —
without cutting a normal transcript segment and without exceeding the
configured token budget except for the one unavoidable oversized-atom case.

**Architecture:** Pure standard-library code, same boundary style as US-1.1
and US-1.2. `ExtractionChunk` is a frozen dataclass added to
`pipeline/qualification/schemas.py`. Chunking logic lives in
`pipeline/qualification/chunking.py`'s new `create_chunks()`, alongside
US-1.2's `create_sections()`. Chunking runs per section (chunks never span
two sections): segments are grouped into speaker turns (maximal runs of
consecutive same-speaker segments), turns are greedily packed into chunks
bounded by `max_chunk_tokens`, and a turn that alone exceeds the limit falls
back to segment-level packing (a single oversized segment becomes its own
chunk, the only unavoidable exception). Neighboring chunks within a section
overlap by up to `chunk_overlap_turns` complete trailing turns taken from the
previous chunk's own pre-overlap turns, bounded by `max_overlap_tokens`,
except the nearest turn is always kept intact even if it alone exceeds that
bound — including when the "nearest turn" is itself a fallback-produced
fragment. Each section is returned with `chunk_ids` populated
(`dataclasses.replace`, since sections stay frozen).

**Tech Stack:** Python standard library dataclasses, pytest, JSON fixture
data. No new dependency — token counting reuses `normalize_transcript`'s
existing `tokenizer: Callable[[str], int] = _count_words` (whitespace word
count) injection pattern; `ChunkingConfiguration.tokenizer` remains an
identity string, unchanged from US-1.2.

**Execution notes (as run):**
- Executed via `superpowers:subagent-driven-development` on branch
  `feat/us-1-3-bounded-extraction-chunks`, cut from the `crm-deal-qualification`
  tag.
- Task 1 (schema) ran first and landed before Tasks 2 and 3. Tasks 2
  (`create_chunks()` + unit tests) and 3 (integration tests + README) ran in
  parallel in separate isolated git worktrees (their file sets are disjoint)
  and were merged back with two conflict-free merge commits. Task 3's
  integration tests were RED (`ImportError: cannot import name 'create_chunks'`)
  until Task 2's implementation merged, then went green — expected TDD red,
  not a defect.
- Task 2's review surfaced one Important finding: no test exercised overlap
  selection immediately after a fallback-produced chunk (an oversized turn
  packed at segment granularity). A follow-up fix task traced the actual
  behavior, found `_select_overlap` already degrades correctly (the fallback
  fragment is treated as the "nearest turn" and kept intact), and added two
  tests pinning exact `overlap_segment_ids` for that scenario — no production
  code change was needed.
- Task 4 (this judge pass) ran directly by the orchestrator, not delegated.

---

## Acceptance criteria → tests (final judge pass)

| Acceptance criterion (`docs/Problem/EPICS-AND-USER-STORIES.md` US-1.3) | Proven by |
|---|---|
| Chunks do not arbitrarily cut normal transcript segments | `test_chunks_never_cut_a_segment`, `test_normal_segments_are_never_cut_for_overlap` (`tests/qualification/unit/test_chunking.py`) |
| Related speaker turns are kept together when possible | `test_related_speaker_turns_are_kept_together_when_possible` |
| Every chunk has a maximum size | `test_chunk_respects_max_chunk_tokens`; exception case: `test_oversized_single_turn_becomes_its_own_chunk_and_exceeds_limit` |
| Neighboring chunks include overlap | `test_neighboring_chunks_within_a_section_include_overlap`; end-to-end: `test_long_call_chunks_cover_every_source_segment_with_overlap` (`tests/qualification/integration/test_extraction_chunks.py`) |
| Every chunk retains its section ID and source segment IDs | `test_every_chunk_retains_section_id_and_ordered_segment_ids` |
| Every chunk has start and end timestamps | `test_fixture_transcript_forms_traceable_chunks` (integration, asserts `chunk.start`/`chunk.end` against the source fixture) |
| Chunk text is rendered from the original source segments with speaker labels and timestamps | `test_rendered_text_includes_speaker_labels_and_timestamps`; verbatim-quote check in `test_fixture_transcript_forms_traceable_chunks` |
| Each chunk records deterministic ID, sequence, ordered segment IDs, overlap segment IDs, rendered text, `token_count` | `test_chunk_ids_are_globally_unique_and_sequence_resets_per_section`, `test_create_chunks_is_deterministic`, `test_token_count_matches_rendered_text_via_tokenizer` |
| The configured maximum token limit is enforced for rendered chunk text | `test_chunk_respects_max_chunk_tokens` |
| Chunk overlap uses complete speaker turns and respects the configured overlap target where possible | `test_overlap_uses_complete_turns_and_respects_max_overlap_tokens`, `test_overlap_always_includes_nearest_turn_even_if_it_exceeds_max_overlap_tokens`, `test_overlap_after_oversized_turn_fallback_uses_immediately_preceding_fragment`, `test_overlap_after_fallback_still_includes_nearest_fragment_below_max_overlap_tokens` |
| Normal transcript segments are never cut to satisfy an overlap limit | `test_normal_segments_are_never_cut_for_overlap` |

Also verified: `ConversationSection.chunk_ids` is populated end-to-end
(`test_sections_are_returned_with_populated_chunk_ids`,
`test_fixture_transcript_forms_traceable_chunks`) — the forward reference left
by US-1.2's schema docstring is now resolved.

## Test results

- `tests/qualification/` (unit + integration): 43/43 passed (21 pre-existing
  US-1.1/US-1.2 tests untouched and still green + 22 new for US-1.3).
- Full repo suite (`tests/`, deselecting the documented DeepFace/torch
  segfault pair): 253 passed, 1 deselected, 1 failed. The one failure —
  `tests/test_llm_analysis.py::TestRunAnalysisIntegration::test_with_real_outputs`
  — is a pre-existing real-API integration test unrelated to this story,
  failing on `anthropic.BadRequestError: credit balance too low`, an
  environment/billing condition, not a code regression.
