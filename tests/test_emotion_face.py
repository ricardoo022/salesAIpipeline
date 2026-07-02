import importlib.util
import pytest
from unittest.mock import MagicMock


def _cv2_available():
    return importlib.util.find_spec("cv2") is not None


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


class TestAnalyzeFrame:
    def test_returns_none_when_no_face(self):
        import sys
        fake_df = MagicMock()
        fake_df.DeepFace.analyze.side_effect = ValueError("Face could not be detected")
        sys.modules["deepface"] = fake_df
        try:
            from pipeline.emotion_face import _analyze_frame
            assert _analyze_frame("frame") is None
        finally:
            del sys.modules["deepface"]

    def test_shapes_deepface_output(self):
        import sys
        fake_df = MagicMock()
        fake_df.DeepFace.analyze.return_value = [
            {"dominant_emotion": "happy", "emotion": {"happy": 0.9, "neutral": 0.1}}
        ]
        sys.modules["deepface"] = fake_df
        try:
            from pipeline.emotion_face import _analyze_frame
            result = _analyze_frame("frame")
        finally:
            del sys.modules["deepface"]
        assert result["dominant_emotion"] == "happy"
        assert result["scores"]["happy"] == 0.9


@pytest.mark.skipif(not _cv2_available(), reason="opencv-python not installed")
class TestIterFramesIntegration:
    def test_invalid_video_yields_nothing(self, tmp_path):
        from pipeline.emotion_face import _iter_frames
        assert list(_iter_frames(str(tmp_path / "nope.mp4"))) == []

    def test_samples_frames_at_interval(self, tmp_path):
        import cv2
        import numpy as np
        from pipeline.emotion_face import _iter_frames
        video = tmp_path / "synthetic.mp4"
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video), fourcc, fps, (64, 64))
        for _ in range(35):  # 3.5s of video at 10fps
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        cap = cv2.VideoCapture(str(video))
        readback = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if readback <= 0:
            pytest.skip("cv2 could not read back the synthetic video; codec unavailable")
        timestamps = [t for t, _ in _iter_frames(str(video), interval=1)]
        assert timestamps == [0.0, 1.0, 2.0, 3.0]
