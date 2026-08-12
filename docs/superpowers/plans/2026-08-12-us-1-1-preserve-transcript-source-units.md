# US-1.1 Preserve Transcript Source Units Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, immutable transcript normalizer that preserves source fields and supports oversized word-range pieces without modifying the source JSON.

**Architecture:** Keep the qualification source-unit boundary in `pipeline/qualification/schemas.py` as pure standard-library code. A frozen dataclass represents one source record or one deterministic piece of an oversized record; normalization accepts an injectable tokenizer and returns records without side effects.

**Tech Stack:** Python standard library dataclasses, pytest, JSON fixture data.

---

### Task 1: Define the source schema and normalization tests

**Files:**
- Create: `tests/qualification/unit/test_schemas.py`
- Create: `tests/qualification/integration/test_transcript_normalization.py`
- Create: `tests/qualification/fixtures/transcript.json`

- [ ] **Step 1: Write unit tests for stable identity and preservation**

```python
def test_normalize_assigns_deterministic_ids_and_preserves_source_fields():
    source = [{"speaker": "SPEAKER_00", "start": 1.25, "end": 2.5,
               "text": "We need approval.",
               "words": [{"word": "We", "start": 1.25, "end": 1.5}]}]

    result = normalize_transcript(source)

    assert result[0].segment_id == "seg_000001"
    assert result[0].speaker == source[0]["speaker"]
    assert result[0].start == source[0]["start"]
    assert result[0].end == source[0]["end"]
    assert result[0].text == source[0]["text"]
    assert result[0].words == tuple(source[0]["words"])
    assert result[0].piece_index is None


def test_ids_are_deterministic_and_input_is_not_mutated():
    source = [{"speaker": "A", "start": 0, "end": 1, "text": "One", "words": []},
              {"speaker": "B", "start": 1, "end": 2, "text": "Two", "words": []}]
    snapshot = copy.deepcopy(source)

    first = normalize_transcript(source)
    second = normalize_transcript(source)

    assert first == second
    assert source == snapshot
```

- [ ] **Step 2: Write unit tests for oversized word-boundary pieces**

```python
def test_oversized_segment_keeps_source_id_and_records_word_ranges():
    source = [{
        "speaker": "SPEAKER_01", "start": 10.0, "end": 20.0,
        "text": "Alpha beta gamma delta", "words": [
            {"word": "Alpha", "start": 10.0, "end": 11.0},
            {"word": "beta", "start": 11.0, "end": 12.0},
            {"word": "gamma", "start": 12.0, "end": 13.0},
            {"word": "delta", "start": 13.0, "end": 14.0},
        ],
    }]

    pieces = normalize_transcript(source, max_tokens=2,
                                  tokenizer=lambda text: len(text.split()))

    assert [piece.segment_id for piece in pieces] == ["seg_000001"] * 2
    assert [piece.piece_index for piece in pieces] == [0, 1]
    assert [(piece.word_start, piece.word_end) for piece in pieces] == [(0, 2), (2, 4)]
    assert [piece.text for piece in pieces] == ["Alpha beta", "gamma delta"]
    assert pieces[0].words == tuple(source[0]["words"][:2])
    assert pieces[1].words == tuple(source[0]["words"][2:])


def test_oversized_segment_without_words_fails_instead_of_cutting_text():
    source = [{"speaker": "A", "start": 0, "end": 1,
               "text": "too large", "words": []}]

    with pytest.raises(ValueError, match="word-level"):
        normalize_transcript(source, max_tokens=1,
                             tokenizer=lambda text: len(text.split()))
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run: `python -m pytest tests/qualification/unit/test_schemas.py -q`

Expected: collection failure because `pipeline.qualification.schemas` does not yet exist.

- [ ] **Step 4: Add the fixture and integration contract test**

The fixture contains two ordinary source entries with words. The test reads
the fixture bytes, loads JSON, normalizes it twice, asserts the expected IDs,
fields, and reconstructed source references, then asserts the fixture bytes are
unchanged.

```python
def test_fixture_normalization_is_traceable_and_non_destructive():
    raw_before = FIXTURE.read_bytes()
    source = json.loads(raw_before)
    result = normalize_transcript(source)
    assert [item.segment_id for item in result] == ["seg_000001", "seg_000002"]
    assert [item.text for item in result] == [item["text"] for item in source]
    assert FIXTURE.read_bytes() == raw_before
    assert result == normalize_transcript(json.loads(raw_before))
```

### Task 2: Implement the immutable source normalizer

**Files:**
- Create: `pipeline/qualification/schemas.py`
- Create: `pipeline/qualification/__init__.py`

- [ ] **Step 1: Implement immutable record types and defaults**

Use a frozen dataclass with tuple-backed words. Preserve each word mapping by
copying it into a `MappingProxyType` exposed through a tuple; callers must not
be able to mutate the normalizer's internal source.

- [ ] **Step 2: Implement deterministic normal-segment normalization**

Enumerate input entries from one, format IDs as `seg_{index:06d}`, copy exact
speaker/timing/text values, preserve all words, and return a new list. Reject
missing required source fields with a clear `ValueError`.

- [ ] **Step 3: Implement oversized splitting by complete words**

Count each source text with the injected tokenizer. If it exceeds the maximum,
require a non-empty list of word dictionaries, accumulate complete words until
the next word would exceed the limit, and emit pieces with deterministic
`piece_index`, `word_start`, `word_end`, reconstructed text, and copied word
timestamps. A single word over the limit remains intact and is represented as
an explicit oversized piece rather than being cut.

- [ ] **Step 4: Run unit and integration tests**

Run: `python -m pytest tests/qualification/unit tests/qualification/integration -q`

Expected: all US-1.1 tests pass.

### Task 3: Judge the acceptance criteria and regression safety

**Files:**
- Modify: none unless test failures identify a required correction

- [ ] **Step 1: Run all repository tests**

Run: `python -m pytest tests/ -q --deselect tests/test_emotion_face.py::TestExtractFaceEmotionIntegration::test_with_real_meeting_video`

Expected: existing suite and qualification unit/integration tests pass, with
the documented native-library integration test deselected.

- [ ] **Step 2: Check each US-1.1 acceptance criterion**

Confirm stable deterministic IDs, exact field/text preservation, word
timestamps, non-destructive input behavior, and oversized piece identity/range
metadata from the focused test output and implementation inspection. Stop only
when unit and integration tests pass and every criterion is evidenced.
