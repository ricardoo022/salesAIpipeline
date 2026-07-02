import pytest


class TestShapeEmotionResult:
    def test_extracts_dominant_and_scores(self):
        from pipeline.emotion_face import _shape_emotion_result
        raw = {"dominant_emotion": "happy", "emotion": {"happy": 0.9123, "sad": 0.0877}}
        result = _shape_emotion_result(raw)
        assert result["dominant_emotion"] == "happy"
        assert result["scores"]["happy"] == 0.9123
        assert result["scores"]["sad"] == 0.0877

    def test_rounds_scores_to_four_decimals(self):
        from pipeline.emotion_face import _shape_emotion_result
        raw = {"dominant_emotion": "neutral", "emotion": {"neutral": 0.712345, "happy": 0.031111}}
        result = _shape_emotion_result(raw)
        assert result["scores"]["neutral"] == 0.7123
        assert result["scores"]["happy"] == 0.0311

    def test_unwraps_list_result(self):
        from pipeline.emotion_face import _shape_emotion_result
        raw = [{"dominant_emotion": "sad", "emotion": {"sad": 0.6, "neutral": 0.4}}]
        result = _shape_emotion_result(raw)
        assert result["dominant_emotion"] == "sad"
        assert result["scores"]["sad"] == 0.6
