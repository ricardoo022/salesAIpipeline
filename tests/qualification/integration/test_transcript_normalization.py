import json
from pathlib import Path

from pipeline.qualification.schemas import normalize_transcript


FIXTURE = Path(__file__).parents[1] / "fixtures" / "transcript.json"


def test_fixture_normalization_is_traceable_and_non_destructive():
    raw_before = FIXTURE.read_bytes()
    source = json.loads(raw_before)

    result = normalize_transcript(source)

    assert [item.segment_id for item in result] == ["seg_000001", "seg_000002"]
    assert [item.text for item in result] == [item["text"] for item in source]
    assert [item.speaker for item in result] == [item["speaker"] for item in source]
    assert FIXTURE.read_bytes() == raw_before
    assert result == normalize_transcript(json.loads(raw_before))
