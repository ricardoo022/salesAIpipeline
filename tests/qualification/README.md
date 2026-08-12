# Qualification Tests

Qualification tests are isolated from the existing pipeline tests for steps
`01` through `06`.

## Unit tests

Place focused tests for schemas, chunking, coverage, grounding, assembly, and
signal linking in `unit/`.

## Integration tests

Place end-to-end qualification tests in `integration/`. These tests should use
fixture JSON outputs and a fake LLM boundary. They must not run transcription,
diarization, audio analysis, voice analysis, facial analysis, or Claude API
calls.
