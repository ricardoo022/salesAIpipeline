import os
import sys
import subprocess

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(PROJECT_ROOT, "pipeline/05_llm_analysis.py")


def test_exits_when_inputs_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "transcript.json" in result.stdout


def test_exits_when_api_key_missing(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    for name in ("transcript.json", "audio_features.json",
                 "voice_emotion.json", "face_emotion.json"):
        (out / name).write_text("[]")
    env = {**os.environ, "ANTHROPIC_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    assert result.returncode == 1
    assert "ANTHROPIC_API_KEY" in result.stdout
