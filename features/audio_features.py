"""Minimal numerical features via librosa.

Scenario A (discovery / chill) doesn't need BPM-driven filters or DJ matching.
The audio embedding does the heavy lifting; numerical features are kept only
for UI display ("3:42, 86 BPM, A minor") and optional sanity sorting.

Removed from previous version: spectral_*, MFCC×13, ZCR, onset_density,
loudness_db, n_beats. They added 30+ columns of noise that recsys ignored
anyway.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

# Krumhansl-Schmuckler key profiles
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(chroma: np.ndarray) -> tuple[str, float]:
    profile = chroma.mean(axis=1)
    profile = profile / (profile.sum() + 1e-12)
    scores = []
    for i in range(12):
        maj = np.corrcoef(np.roll(_KS_MAJOR, i), profile)[0, 1]
        mn = np.corrcoef(np.roll(_KS_MINOR, i), profile)[0, 1]
        scores.append((maj, f"{_PITCH_NAMES[i]}_major"))
        scores.append((mn, f"{_PITCH_NAMES[i]}_minor"))
    scores.sort(reverse=True)
    best, name = scores[0]
    second = scores[1][0]
    confidence = float(min(max(best - second, 0.0) * 4.0, 1.0))
    return name, confidence


def extract(path: str | Path, sr: int = 22050) -> dict:
    """Numerical features. All NaN/inf are zeroed."""
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    duration = float(len(y) / sr)
    if duration < 1.0:
        return {"duration_sec": duration, "error": "too_short"}

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    # librosa often returns the hi-hat subdivision (e.g. 172 instead of 86).
    # Cap to a perceptual pulse range via octave folding so the number is
    # consistent across tracks for UI display.
    bpm_perceived = bpm
    while bpm_perceived > 160:
        bpm_perceived /= 2
    while 0 < bpm_perceived < 60:
        bpm_perceived *= 2

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, key_conf = _estimate_key(chroma)

    rms_mean = float(librosa.feature.rms(y=y)[0].mean())

    out = {
        "duration_sec": duration,
        "bpm": bpm,
        "bpm_perceived": bpm_perceived,
        "key": key,
        "key_confidence": key_conf,
        "rms_mean": rms_mean,
    }
    return {k: (0.0 if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v)
            for k, v in out.items()}
