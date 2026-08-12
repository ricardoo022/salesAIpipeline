# Qualification Subsystem

This directory contains the backend BANT evidence-extraction subsystem.

## Current status

Epic 1, US-1.1 through US-1.4 are complete. `schemas.py` provides the
immutable `TranscriptSegment` source boundary and `normalize_transcript()`
assigns deterministic source IDs without modifying `output/transcript.json`.
Oversized segments are split only at complete word boundaries; each piece
retains the original `segment_id` and records its deterministic piece index and
word range. `chunking.py`'s `create_sections()` groups normalized segments
into chronological, deterministically overlapping `ConversationSection`s, and
`create_chunks()` bounds each section's segments into speaker-labeled
`ExtractionChunk`s: whole speaker turns are packed under `max_chunk_tokens`,
oversized turns fall back to per-segment packing, and neighboring chunks
within a section share overlap made of complete trailing turns. Each section's
`chunk_ids` is populated with its own chunks' IDs.

US-1.4 adds a chunk-hierarchy resolution API to `chunking.py`:
`resolve_chunk(chunk_id, chunks, sections, segments)` resolves one
`ExtractionChunk` back to its parent `ConversationSection` and the ordered
source `TranscriptSegment`s it was built from, returning a `ResolvedChunk`
(the frozen `chunk`/`section`/`segments` dataclass added to `schemas.py`),
and raises `ChunkHierarchyError` on any broken or missing reference — an
unknown `chunk_id`, a missing parent section, a section whose `chunk_ids`
does not list the chunk, or a resolved segment count that does not match
the chunk's `segment_ids`. `reconstruct_chunk_text()` rebuilds a chunk's
rendered text purely from `resolved.segments` and equals the chunk's own
`text` byte-for-byte for any chunk `create_chunks` produces, including
chunks with overlap. `iter_chunks_for_processing()` returns every chunk
unfiltered, in original order, making explicit that Epic 1 processes the
full chunk set rather than a retrieved subset. Chunk coverage validation
(`CoverageRecord`, US-1.5) remains planned work.

Planned responsibilities:

- Chunk coverage validation
- Evaluation of candidate chunking configurations with gold question-answer cases
- Four parallel BANT topic agents
- Evidence grounding and validation
- Evidence assembly and deduplication
- Linking existing audio, voice, and facial measurements
- Harness state and intermediate run storage

The subsystem consumes the existing pipeline outputs. It does not rerun transcription, audio analysis, voice analysis, or facial analysis.

Qualification decisions and company-specific KPIs remain outside this subsystem.

## Epic 1 chunking contract

The qualification layer normalizes the existing transcript into immutable
`TranscriptSegment` records without modifying `output/transcript.json`.
Segments are grouped into chronological `ConversationSection` parents and
bounded `ExtractionChunk` children. A chunk contains rendered speaker-labeled
text for the LLM plus ordered source segment IDs, timestamps, and `token_count`.

The initial configuration is a conversation-aware hybrid:

```text
section target: 8 minutes
section overlap: 30 seconds
maximum chunk: 1,200 tokens
chunk overlap: 2 complete speaker turns
overlap target: 250 tokens
```

Sections use time for broad organization, chunks use tokens for LLM safety, and
chunk overlap uses complete speaker turns rather than cutting at arbitrary
token or timestamp boundaries. `CoverageRecord` validation runs before agents
receive chunks.

No vector database is required for Epic 1. Every chunk is processed so that
evidence is not lost through retrieval selection. A manually reviewed
`EvaluationQuestion` dataset compares candidate configurations using evidence
coverage, context sufficiency, extraction recall, grounding precision, token
cost, and overlap duplication.

Qualification tests run in GitHub Actions on every push and pull request via
`uv`. The tests are separated into `tests/qualification/unit/` and
`tests/qualification/integration/`; CI does not execute the real model tests
for pipeline steps `01` through `06`.
