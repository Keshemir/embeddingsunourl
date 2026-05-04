"""Corpus-level calibration of zero-shot tag scores.

Two transforms operate on the full (N tracks × T tags) raw-cosine matrix:

1. **Z-score per tag** kills hubness — tags whose text vector lives in a dense
   region of the joint space get systematically high cosines with everything.
   Subtracting per-tag mean and dividing by per-tag std centres each tag
   relative to the corpus, so "dark = 0.7" becomes "this track is 1.5σ above
   the average dark-cosine," which is way more informative than a raw 0.7.

2. **Per-tag percentile thresholds** turn the calibrated scores into hard
   labels without supervised data. For each tag, the K-th percentile (default
   90) of its scores across the corpus becomes the "active" threshold —
   anything above is tagged. Tunable per group: rare fusion genres get a
   higher percentile (top-2%), common moods a lower one (top-20%).

Usage:
    cal = calibrate(raw_cos_matrix)          # fit
    z = cal.zscore(raw_cos_matrix)           # apply (e.g. on a new track)
    is_active = cal.activate(z, percentile=90)  # bool matrix
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Calibrator:
    """Fitted on the corpus, applied at scoring time."""

    mean: np.ndarray  # (T,) per-tag mean cosine across corpus
    std: np.ndarray   # (T,) per-tag std cosine across corpus

    def zscore(self, raw: np.ndarray) -> np.ndarray:
        """Center and scale per tag. `raw` is (T,) or (N, T)."""
        return ((raw - self.mean) / (self.std + 1e-9)).astype(np.float32)

    def percentile_thresholds(
        self,
        raw_corpus: np.ndarray,
        percentile: float = 90.0,
    ) -> np.ndarray:
        """Per-tag threshold: the percentile-th value across the corpus."""
        return np.percentile(raw_corpus, percentile, axis=0).astype(np.float32)


def fit(raw_corpus: np.ndarray) -> Calibrator:
    """Fit per-tag mean/std on the corpus's raw cosine matrix.

    raw_corpus: (N tracks, T tags) — output of zeroshot.score's `raw_cos`.
    """
    return Calibrator(
        mean=raw_corpus.mean(axis=0).astype(np.float32),
        std=raw_corpus.std(axis=0).astype(np.float32),
    )


def cosine_histogram(audio_vecs: np.ndarray, bins: int = 50) -> dict:
    """Distribution of pairwise cosines across the corpus.

    Use the percentile to set a "minimum similarity" cutoff in search:
    anything below the corpus's 95th percentile of pairwise cosine is
    statistically random, not a real match.

    Returns: dict with hist, edges, and key percentiles.
    """
    n = len(audio_vecs)
    if n < 2:
        return {"n_pairs": 0, "percentiles": {}}
    # All N×N cosines; vectors are L2-normalized so this is just a matmul
    sims = audio_vecs @ audio_vecs.T
    # Upper triangle (exclude self-similarity which is 1.0)
    iu = np.triu_indices(n, k=1)
    pairs = sims[iu]
    hist, edges = np.histogram(pairs, bins=bins)
    pct = {p: float(np.percentile(pairs, p)) for p in (50, 75, 90, 95, 98, 99)}
    return {
        "n_pairs": int(len(pairs)),
        "hist": hist.tolist(),
        "edges": edges.tolist(),
        "percentiles": pct,
        "mean": float(pairs.mean()),
        "std": float(pairs.std()),
    }
