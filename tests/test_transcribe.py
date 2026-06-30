import json
import os
import sys
import subprocess
import pytest

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def test_exits_when_input_video_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "pipeline/01_transcribe.py")],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "not found" in result.stdout
