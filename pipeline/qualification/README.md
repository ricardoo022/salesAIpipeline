# Qualification Subsystem

This directory contains the backend BANT evidence-extraction subsystem.

## Current status

Epic 1, US-1.1 is complete. `schemas.py` provides the immutable
`TranscriptSegment` source boundary and `normalize_transcript()` assigns
deterministic source IDs without modifying `output/transcript.json`.
Oversized segments are split only at complete word boundaries; each piece
retains the original `segment_id` and records its deterministic piece index and
word range. Conversation sections and extraction chunks remain planned work.

Planned responsibilities:

- Hierarchical transcript chunking
- Deterministic section and chunk schemas with source traceability
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
