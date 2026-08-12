# US-1.1 Preserve Transcript Source Units

## Goal

Normalize the existing transcript into deterministic, immutable source records
without changing `output/transcript.json`, so later qualification stages can
ground every extraction result in the original call.

## Scope

This story adds only the source-unit schema and normalization boundary. It does
not create conversation sections, extraction chunks, coverage records, or BANT
agents. It does include deterministic handling of an individual oversized
segment because that behavior is explicitly part of US-1.1.

## Design

`pipeline/qualification/schemas.py` will define a frozen `TranscriptSegment`
dataclass containing the stable source identity, copied speaker and timing
fields, exact text, preserved word timestamps, and optional piece metadata.
Normal source entries use `segment_id` values `seg_000001`, `seg_000002`, and so
on. The normalizer is a pure function and never mutates its input.

The normalizer accepts a configurable maximum token count and a tokenizer
callable. A normal segment becomes one immutable record. When a segment exceeds
the limit, it is split only at complete word boundaries using its `words`
entries. Every piece retains the original `segment_id`, and `piece_index` plus
`word_start`/`word_end` make the piece identity deterministic as
`(segment_id, piece_index)`. Piece text is reconstructed from the original word
values, and piece words retain their original timestamp dictionaries. If an
oversized segment has no usable word-level timestamps, normalization raises a
clear `ValueError` rather than slicing transcript text arbitrarily.

The original source segment's speaker, start, and end values are preserved
exactly on every piece. This avoids inventing timing precision while keeping
word-level timestamps available for downstream consumers. The source input is
not rewritten or persisted by this module.

## Testing

Unit tests will cover deterministic and unique unsplit IDs, exact field and
word preservation, input immutability, oversized word-boundary splitting and
range metadata, deterministic repeated output, and rejection of oversized
segments without words. A qualification integration test will normalize a
fixture transcript and verify stable references plus byte-for-byte unchanged
fixture data. Tests use only the standard library and do not invoke pipeline
models, external APIs, or the numbered scripts.

## Resolved Identity Rule

The source contract requires split pieces to retain the original `segment_id`
while also describing IDs as unique. This design treats `segment_id` as the
stable identity of the original transcript source unit. Unsplit records are
identified by `segment_id`; split records are uniquely identified by the pair
`(segment_id, piece_index)`, with the word range recorded explicitly.
