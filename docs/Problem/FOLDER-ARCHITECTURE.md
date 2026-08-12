# BANT Extraction Folder Architecture

## Purpose

The BANT evidence-extraction feature is organized as a separate backend subsystem. It consumes the existing pipeline outputs and produces evidence for a company's own qualification process.

The subsystem does not decide whether a lead qualifies and does not apply company-specific KPIs.

## Source Folder Structure

```text
pipeline/
├── 07_qualification.py          # CLI entry point
├── qualification/
│   ├── __init__.py
│   ├── graph.py                  # harness workflow
│   ├── state.py                  # run state
│   ├── schemas.py                # chunks and evidence shapes
│   ├── chunking.py               # hierarchical transcript chunks
│   ├── grounding.py              # source and quote validation
│   ├── assembly.py               # merge and deduplicate evidence
│   ├── signals.py                # link existing measurements
│   ├── storage.py                # intermediate run artifacts
│   └── agents/
│       ├── __init__.py
│       ├── budget.py
│       ├── authority.py
│       ├── need.py
│       └── timeline.py
```

The source files above define the planned boundaries. The implementation will be created through the implementation plan after this architecture is approved.

## Responsibility Boundaries

### Existing Pipeline

The existing modules keep their current responsibilities:

- `transcribe.py` produces the speaker-labelled transcript.
- `features.py` produces audio measurements.
- `emotion_voice.py` produces voice measurements.
- `emotion_face.py` produces facial measurements.

The qualification subsystem reads these outputs. It does not execute those analysis steps again.

### Qualification Subsystem

The `pipeline/qualification/` package owns:

- Transcript hierarchy and chunking
- Four BANT extraction agents
- Source grounding and quote validation
- Evidence assembly and deduplication
- Signal linking
- Harness state
- Intermediate run storage

### CLI Entry Point

`pipeline/07_qualification.py` will be the command-line entry point for running the qualification subsystem after the existing six pipeline steps have completed.

## Test Structure

Qualification tests are isolated from the existing pipeline tests and divided
by scope:

```text
tests/
└── qualification/
    ├── unit/
    │   ├── test_schemas.py
    │   ├── test_chunking.py
    │   ├── test_grounding.py
    │   ├── test_assembly.py
    │   └── test_signals.py
    └── integration/
        └── test_qualification_flow.py
```

Unit tests cover each qualification boundary independently. Integration tests
use fixture JSON outputs and a fake LLM boundary to verify the qualification
flow without running pipeline steps `01` through `06` or making external API
calls.

## Runtime Artifact Structure

Generated data remains separate from source code:

```text
output/
├── qualification.json
└── qualification_runs/
    └── <run_id>/
        ├── manifest.json
        ├── sections.jsonl
        ├── chunks.jsonl
        ├── budget_results.jsonl
        ├── authority_results.jsonl
        ├── need_results.jsonl
        ├── timeline_results.jsonl
        ├── validation.jsonl
        └── qualification.json
```

`output/qualification.json` is the stable consumer-facing artifact. The run directory contains intermediate data used for retries, debugging, auditing, and traceability.

## Dependency Direction

```text
Existing pipeline outputs
            |
            v
pipeline/qualification/
            |
            +--> qualification.json
            +--> qualification_runs/<run_id>/
```

The qualification subsystem may consume existing pipeline modules and output files. Existing transcription, feature, emotion, and report modules should not depend on the qualification subsystem.

## Explicit Non-Responsibilities

This folder architecture does not include:

- Qualification scoring
- Qualified or unqualified decisions
- Company KPI configuration
- CRM export
- Frontend or report changes
- Re-running signal extraction
- Cross-call account analysis
- Embeddings or vector database infrastructure
