# Qualification Tests

Qualification tests are isolated from the existing pipeline tests for steps
`01` through `06`. 59 tests currently pass, covering US-1.1 (transcript
source normalization), US-1.2 (conversation sections), US-1.3 (bounded
extraction chunks), and US-1.4 (chunk hierarchy resolution) — deterministic
source IDs, exact field and word preservation, non-destructive normalization,
recursive immutability, oversized word-boundary pieces, deterministic
overlapping sections with full segment coverage, turn-preserving chunk
packing (including the segment-level fallback for oversized turns),
turn-based chunk overlap, fixture-based integration traceability from raw
transcript through sections to chunks, and resolving a chunk back to its
parent section and source segments (`resolve_chunk`, `reconstruct_chunk_text`,
`iter_chunks_for_processing`, `ChunkHierarchyError`) — including every
failure mode raising loudly instead of dropping data, and byte-for-byte text
reconstruction over both the fixture and a synthetic multi-section/overlap
transcript.

## Unit tests

Focused tests for schemas and chunking live in `unit/` (`test_schemas.py`,
`test_chunking.py`, joined by `test_hierarchy.py` for the US-1.4
`resolve_chunk`/`reconstruct_chunk_text`/`iter_chunks_for_processing`/
`ChunkHierarchyError` resolution API); coverage, grounding, assembly, and
signal linking tests (beyond US-1.4) will join them here. These import
`pipeline.qualification.schemas` and `pipeline.qualification.chunking`
directly and do not require model dependencies.

## Integration tests

End-to-end qualification tests live in `integration/`
(`test_transcript_normalization.py`, `test_conversation_sections.py`,
`test_extraction_chunks.py`, `test_hierarchy_traceability.py`). They use the
fixture transcript in `fixtures/transcript.json` to verify the full
normalize → sections → chunks chain is stable, non-destructive, and
traceable back to the source; `test_hierarchy_traceability.py` extends that
chain to prove every chunk resolves back to its parent section and source
segments, and that its rendered text reconstructs byte-for-byte, over both
the fixture and a synthetic multi-section/overlap transcript. Future tests
here should use fixture JSON outputs and a fake LLM boundary; they must not
run transcription, diarization, audio analysis, voice analysis, facial
analysis, or Claude API calls.
