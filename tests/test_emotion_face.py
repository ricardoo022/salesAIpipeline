import importlib.util
import os
import pytest
from unittest.mock import MagicMock, patch


def _cv2_available():
    return importlib.util.find_spec("cv2") is not None


def _deepface_available():
    return importlib.util.find_spec("deepface") is not None


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

    def test_enforces_face_detection(self):
        import sys
        fake_df = MagicMock()
        fake_df.DeepFace.analyze.return_value = [
            {"dominant_emotion": "happy", "emotion": {"happy": 0.9, "neutral": 0.1}}
        ]
        sys.modules["deepface"] = fake_df
        try:
            from pipeline.emotion_face import _analyze_frame
            _analyze_frame("frame")
        finally:
            del sys.modules["deepface"]
        fake_df.DeepFace.analyze.assert_called_once_with(
            "frame", actions=["emotion"], enforce_detection=True, detector_backend="retinaface"
        )


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


class TestExtractFaceEmotion:
    def test_raises_when_video_missing(self, tmp_path):
        from pipeline.emotion_face import extract_face_emotion
        with pytest.raises(FileNotFoundError, match="not found"):
            extract_face_emotion(str(tmp_path / "nonexistent.mp4"))

    def test_returns_one_record_per_detected_face(self, tmp_path):
        from pipeline.emotion_face import extract_face_emotion
        video = tmp_path / "video.mp4"
        video.touch()
        frames = [(0.0, "frame0"), (10.0, "frame1")]
        analysis = {"dominant_emotion": "happy", "scores": {"happy": 0.9, "neutral": 0.1}}
        with patch("pipeline.emotion_face._iter_frames", return_value=iter(frames)):
            with patch("pipeline.emotion_face._analyze_frame", return_value=analysis):
                result = extract_face_emotion(str(video))
        assert len(result) == 2
        assert result[0]["timestamp"] == 0.0
        assert result[0]["dominant_emotion"] == "happy"
        assert result[0]["scores"]["happy"] == 0.9

    def test_skips_frames_with_no_face(self, tmp_path):
        from pipeline.emotion_face import extract_face_emotion
        video = tmp_path / "video.mp4"
        video.touch()
        frames = [(0.0, "f0"), (10.0, "f1"), (20.0, "f2")]

        def fake_analyze(frame):
            return None if frame == "f1" else {"dominant_emotion": "neutral", "scores": {"neutral": 1.0}}

        with patch("pipeline.emotion_face._iter_frames", return_value=iter(frames)):
            with patch("pipeline.emotion_face._analyze_frame", side_effect=fake_analyze):
                result = extract_face_emotion(str(video))
        assert len(result) == 2
        assert [r["timestamp"] for r in result] == [0.0, 20.0]

    def test_record_has_required_keys(self, tmp_path):
        from pipeline.emotion_face import extract_face_emotion
        video = tmp_path / "video.mp4"
        video.touch()
        frames = [(5.0, "f0")]
        analysis = {"dominant_emotion": "surprise", "scores": {"surprise": 0.5, "neutral": 0.5}}
        with patch("pipeline.emotion_face._iter_frames", return_value=iter(frames)):
            with patch("pipeline.emotion_face._analyze_frame", return_value=analysis):
                result = extract_face_emotion(str(video))
        assert set(result[0].keys()) == {"timestamp", "dominant_emotion", "scores"}

    def test_empty_video_returns_empty_list(self, tmp_path):
        from pipeline.emotion_face import extract_face_emotion
        video = tmp_path / "video.mp4"
        video.touch()
        with patch("pipeline.emotion_face._iter_frames", return_value=iter([])):
            with patch("pipeline.emotion_face._analyze_frame") as mock_analyze:
                result = extract_face_emotion(str(video))
        assert result == []
        mock_analyze.assert_not_called()

    def test_timestamps_are_rounded_to_two_decimals(self, tmp_path):
        from pipeline.emotion_face import extract_face_emotion
        video = tmp_path / "video.mp4"
        video.touch()
        frames = [(10.123456, "f0")]
        analysis = {"dominant_emotion": "neutral", "scores": {"neutral": 1.0}}
        with patch("pipeline.emotion_face._iter_frames", return_value=iter(frames)):
            with patch("pipeline.emotion_face._analyze_frame", return_value=analysis):
                result = extract_face_emotion(str(video))
        assert result[0]["timestamp"] == 10.12


class TestExtractFaceEmotionIntegration:
    @pytest.mark.skipif(
        not _cv2_available()
        or not _deepface_available()
        or not os.path.exists("input/meeting.mp4")
        or not os.path.exists(os.path.expanduser("~/.deepface/weights")),
        reason="requires opencv-python, deepface, input/meeting.mp4, and downloaded DeepFace weights",
    )
    def test_with_real_meeting_video(self):
        from pipeline.emotion_face import extract_face_emotion
        result = extract_face_emotion("input/meeting.mp4", interval=10)
        assert len(result) > 0
        assert "dominant_emotion" in result[0]
        assert "scores" in result[0]
        assert "timestamp" in result[0]
        dominant = result[0]["dominant_emotion"]
        assert 0 <= result[0]["scores"][dominant] <= 1
