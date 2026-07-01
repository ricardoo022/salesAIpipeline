"""Audio feature extraction for step 2.

Extracts pitch, energy, speech rate, pause ratio, and zero crossing rate
from diarized transcript segments.
"""

import os
import librosa
import numpy as np

try:
    from pipeline.audio import AUDIO_SAMPLE_RATE
except ImportError:
    from audio import AUDIO_SAMPLE_RATE


def extract_audio_features(transcript: list[dict], audio_path: str) -> list[dict]:
    """Extract audio features for each diarized segment.

    For each segment in transcript:
      - pitch_mean / pitch_std  via librosa.pyin (F0 estimation)
      - energy_mean             via librosa.feature.rms
      - speech_rate             word_count / segment_duration (from words[] timestamps)
      - pause_ratio             total_gap / segment_duration (gaps between words)
      - zcr                     via librosa.feature.zero_crossing_rate

    Returns a new list with one dict per input segment, preserving
    speaker / start / end and adding the five feature fields.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not transcript:
        raise ValueError("Transcript is empty")

    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)

    result = []
    for seg in transcript:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        segment_audio = y[start_sample:end_sample]

        if len(segment_audio) == 0:
            pitch_mean = 0.0
            pitch_std = 0.0
            energy_mean = 0.0
            zcr = 0.0
        else:
            f0, _voiced_flag, _voiced_probs = librosa.pyin(
                segment_audio, sr=sr,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
            )
            voiced_f0 = f0[~np.isnan(f0)]
            if len(voiced_f0) > 0:
                pitch_mean = float(np.mean(voiced_f0))
                pitch_std = float(np.std(voiced_f0))
            else:
                pitch_mean = 0.0
                pitch_std = 0.0

            rms = librosa.feature.rms(y=segment_audio)
            energy_mean = float(np.mean(rms))

            zcr_array = librosa.feature.zero_crossing_rate(y=segment_audio)
            zcr = float(np.mean(zcr_array))

        words = seg.get("words", [])
        duration = seg["end"] - seg["start"]
        if duration > 0 and words:
            speech_rate = len(words) / duration
        else:
            speech_rate = 0.0

        if duration > 0 and len(words) > 1:
            gaps = 0.0
            for i in range(len(words) - 1):
                gap = words[i + 1]["start"] - words[i]["end"]
                if gap > 0:
                    gaps += gap
            pause_ratio = gaps / duration
        else:
            pause_ratio = 0.0

        result.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "pitch_mean": round(pitch_mean, 4),
            "pitch_std": round(pitch_std, 4),
            "energy_mean": round(energy_mean, 4),
            "speech_rate": round(speech_rate, 4),
            "pause_ratio": round(pause_ratio, 4),
            "zcr": round(zcr, 4),
        })

    return result
