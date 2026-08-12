# Qualification Tests

Qualification tests are isolated from the existing pipeline tests for steps
`01` through `06`. 43 tests currently cover US-1.1 (transcript source
normalization), US-1.2 (conversation sections), and US-1.3 (bounded
extraction chunks) — deterministic source IDs, exact field and word
preservation, non-destructive normalization, recursive immutability,
oversized word-boundary pieces, deterministic overlapping sections with full
segment coverage, turn-preserving chunk packing (including the segment-level
fallback for oversized turns), turn-based chunk overlap, and fixture-based
integration traceability from raw transcript through sections to chunks.

## Unit tests

Focused tests for schemas and chunking live in `unit/` (`test_schemas.py`,
`test_chunking.py`); coverage, grounding, assembly, and signal linking tests
(US-1.4 onward) will join them here. These import `pipeline.qualification.schemas`
and `pipeline.qualification.chunking` directly and do not require model
dependencies.

## Integration tests

End-to-end qualification tests live in `integration/`
(`test_transcript_normalization.py`, `test_conversation_sections.py`,
`test_extraction_chunks.py`). They use the fixture transcript in
`fixtures/transcript.json` to verify the full normalize → sections → chunks
chain is stable, non-destructive, and traceable back to the source. Future
tests here should use fixture JSON outputs and a fake LLM boundary; they must
not run transcription, diarization, audio analysis, voice analysis, facial
analysis, or Claude API calls.
