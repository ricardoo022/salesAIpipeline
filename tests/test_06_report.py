import os
import sys
import subprocess

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(PROJECT_ROOT, "pipeline/06_report.py")


def test_exits_when_inputs_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1
    assert "analysis.json" in result.stdout


def test_writes_report_html(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "transcript.json").write_text('[{"speaker": "SPEAKER_00", "start": 0, "end": 5, "text": "hi"}]')
    (out / "audio_features.json").write_text('[]')
    (out / "voice_emotion.json").write_text('[]')
    (out / "face_emotion.json").write_text('[]')
    (out / "analysis.json").write_text(
        '{"transcript_only": {"engagement_score": 50, "deal_probability": 50, '
        '"talk_ratio": {"rep": 100, "prospect": 0}, "critical_moments": [], "recommendations": []}, '
        '"multimodal": {"engagement_score": 50, "deal_probability": 50, '
        '"talk_ratio": {"rep": 100, "prospect": 0}, "critical_moments": [], "recommendations": []}}'
    )
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0
    assert (out / "report.html").exists()
