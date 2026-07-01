import os
import sys
import subprocess

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def test_exits_when_segments_input_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "pipeline/03_emotion_voice.py")],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "audio_features.json" in result.stdout


def test_exits_when_audio_missing(tmp_path):
    (tmp_path / "output").mkdir()
    with open(tmp_path / "output" / "audio_features.json", "w") as f:
        f.write("[]")
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "pipeline/03_emotion_voice.py")],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "audio_temp.wav" in result.stdout
