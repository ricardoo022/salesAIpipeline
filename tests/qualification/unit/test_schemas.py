import copy

import pytest

from pipeline.qualification.schemas import normalize_transcript


def test_normalize_assigns_deterministic_ids_and_preserves_source_fields():
    source = [{
        "speaker": "SPEAKER_00",
        "start": 1.25,
        "end": 2.5,
        "text": "We need approval.",
        "words": [{"word": "We", "start": 1.25, "end": 1.5}],
    }]

    result = normalize_transcript(source)

    assert result[0].segment_id == "seg_000001"
    assert result[0].speaker == source[0]["speaker"]
    assert result[0].start == source[0]["start"]
    assert result[0].end == source[0]["end"]
    assert result[0].text == source[0]["text"]
    assert result[0].words == tuple(source[0]["words"])
    assert result[0].piece_index is None


def test_ids_are_deterministic_and_input_is_not_mutated():
    source = [
        {"speaker": "A", "start": 0, "end": 1, "text": "One", "words": []},
        {"speaker": "B", "start": 1, "end": 2, "text": "Two", "words": []},
    ]
    snapshot = copy.deepcopy(source)

    first = normalize_transcript(source)
    second = normalize_transcript(source)

    assert first == second
    assert source == snapshot


def test_normalized_source_and_nested_word_metadata_are_immutable():
    source = [{
        "speaker": "A",
        "start": 0,
        "end": 1,
        "text": "One",
        "words": [{"word": "One", "meta": {"score": 1}}],
    }]

    result = normalize_transcript(source)

    with pytest.raises(TypeError):
        result[0].words[0]["meta"]["score"] = 2
    with pytest.raises(AttributeError):
        result[0].text = "Changed"
    assert source[0]["words"][0]["meta"]["score"] == 1


def test_oversized_segment_keeps_source_id_and_records_word_ranges():
    source = [{
        "speaker": "SPEAKER_01",
        "start": 10.0,
        "end": 20.0,
        "text": "Alpha beta gamma delta",
        "words": [
            {"word": "Alpha", "start": 10.0, "end": 11.0},
            {"word": "beta", "start": 11.0, "end": 12.0},
            {"word": "gamma", "start": 12.0, "end": 13.0},
            {"word": "delta", "start": 13.0, "end": 14.0},
        ],
    }]

    pieces = normalize_transcript(
        source,
        max_tokens=2,
        tokenizer=lambda text: len(text.split()),
    )

    assert [piece.segment_id for piece in pieces] == ["seg_000001"] * 2
    assert [piece.piece_index for piece in pieces] == [0, 1]
    assert [(piece.word_start, piece.word_end) for piece in pieces] == [(0, 2), (2, 4)]
    assert [piece.text for piece in pieces] == ["Alpha beta", "gamma delta"]
    assert pieces[0].words == tuple(source[0]["words"][:2])
    assert pieces[1].words == tuple(source[0]["words"][2:])


def test_oversized_segment_without_words_fails_instead_of_cutting_text():
    source = [{
        "speaker": "A",
        "start": 0,
        "end": 1,
        "text": "too large",
        "words": [],
    }]

    with pytest.raises(ValueError, match="word-level"):
        normalize_transcript(
            source,
            max_tokens=1,
            tokenizer=lambda text: len(text.split()),
        )
