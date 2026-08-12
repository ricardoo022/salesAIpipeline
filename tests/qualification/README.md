# Qualification Tests

Qualification tests are isolated from the existing pipeline tests for steps
`01` through `06`.

US-1.1 currently has six tests covering deterministic source IDs, exact field
and word preservation, non-destructive normalization, recursive immutability,
oversized word-boundary pieces, and fixture-based integration traceability.

## Unit tests

Place focused tests for schemas, chunking, coverage, grounding, assembly, and
signal linking in `unit/`. The source schema tests import
`pipeline.qualification.schemas` directly and do not require model
dependencies.

## Integration tests

Place end-to-end qualification tests in `integration/`. US-1.1 uses a fixture
transcript to verify stable normalization and confirms the fixture remains
unchanged. These tests should use fixture JSON outputs and a fake LLM boundary.
They must not run transcription, diarization, audio analysis, voice analysis,
facial analysis, or Claude API calls.
