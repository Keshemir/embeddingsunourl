"""Zero-shot tagging via MuQ-MuLan with research-backed precision boosters.

Three things this module does that the v1 didn't:

1. **Prompt ensembling** — each tag is encoded under N different templates
   (`"{tag}"`, `"a recording of {tag} music"`, ...) and the L2-normalized
   text embeddings are averaged. CLIP literature reports +3-4% top-1 from
   this; for music CLAP / MuLan it's typically +1-3% mAP. Costs N text
   forward passes per tag, done once at startup.

2. **Multiple score representations** — for every (track, tag) we store:
     - `raw_cos`: signed cosine in [-1, 1], no temperature, no group context
     - `sigmoid`: monotonic transform for thresholding without group bias
     - `softmax`: kept for backwards-compat / single-best-per-group display
   Sigmoid + percentile threshold across the corpus is the recommended
   way to do multi-label tagging when tags within a group are not
   mutually exclusive (mood: "dark" and "energetic" can both be true).

3. **Hubness removal** — some text vectors live in dense regions of the
   joint space and rack up high cosine with everything. We keep the raw
   matrix so `features.calibrate` can z-score per tag across the corpus
   and kill that bias in postprocessing.

The output schema is wider but the heavy lifting (matrix multiplies)
is the same — overhead is the prompt ensembling at startup, not per track.
"""
from __future__ import annotations

import numpy as np

from embed.audio import AudioEmbedder
from features.vocab import TAG_GROUPS

# Templates inspired by CLIP zero-shot ensemble (Radford 2021) + MuLan
# §4.2. Order doesn't matter — embeddings are averaged.
PROMPT_TEMPLATES: list[str] = [
    "{tag}",
    "a recording of {tag} music",
    "a {tag} track",
    "music in the style of {tag}",
    "this song is {tag}",
]


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return (v / n).astype(np.float32)


def _softmax(x: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    z = x / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def _sigmoid(x: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / temperature))


# Cache: one per Embedder instance (keyed by id, since Embedder isn't hashable)
_BANK: dict[int, tuple[list[tuple[str, str]], np.ndarray]] = {}


def _ensure_bank(em: AudioEmbedder) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Build (or reuse) the (group, tag) → ensemble-averaged text-vec table."""
    key = id(em)
    if key in _BANK:
        return _BANK[key]
    if not em.supports_text:
        raise RuntimeError(
            f"{type(em).__name__} has no text encoder — zero-shot tagging needs "
            "MuQ-MuLan. Set AUDIO_MODEL=muq."
        )
    flat: list[tuple[str, str]] = []
    rows: list[np.ndarray] = []
    for group, tags in TAG_GROUPS.items():
        for tag in tags:
            flat.append((group, tag))
            # Encode every template, average, then L2-normalize the average.
            # Averaging unit vectors before normalizing is the CLIP recipe.
            ensemble = np.stack([
                em.embed_text(template.format(tag=tag))
                for template in PROMPT_TEMPLATES
            ])
            rows.append(_l2(ensemble.mean(axis=0)))
    M = np.stack(rows).astype(np.float32)
    _BANK[key] = (flat, M)
    return _BANK[key]


def score(audio_vec: np.ndarray, em: AudioEmbedder, temperature: float = 0.07) -> dict:
    """Score a single audio vector against the prompt-ensembled tag bank.

    Returns:
        flat: [(group, tag), ...]                 — vocabulary order
        raw_cos: (T,) float32                     — signed cosine
        sigmoid: {group: {tag: sigmoid_score}}    — multi-label friendly
        softmax: {group: {tag: softmax_prob}}     — single-best-per-group
        top:     {group: [(tag, sigmoid), ...]}   — top-5 per group, sigmoid-ranked
        best_per_group: {group: tag}              — argmax per group
    """
    flat, M = _ensure_bank(em)
    a = _l2(audio_vec.astype(np.float32))
    sims = (M @ a).astype(np.float32)  # (T,)

    raw: dict[str, dict[str, float]] = {}
    sm: dict[str, dict[str, float]] = {}
    sg: dict[str, dict[str, float]] = {}
    top: dict[str, list[tuple[str, float]]] = {}
    best: dict[str, str] = {}

    group_to_indices: dict[str, list[int]] = {}
    for i, (g, _) in enumerate(flat):
        group_to_indices.setdefault(g, []).append(i)

    for g, idxs in group_to_indices.items():
        gs = sims[idxs]
        probs = _softmax(gs, temperature=temperature)
        sigs = _sigmoid(gs, temperature=temperature)
        tags = [flat[i][1] for i in idxs]
        raw[g] = {t: float(s) for t, s in zip(tags, gs)}
        sm[g] = {t: float(p) for t, p in zip(tags, probs)}
        sg[g] = {t: float(s) for t, s in zip(tags, sigs)}
        # Rank by sigmoid (better for multi-label) for the "top" view
        ranked_sig = sorted(zip(tags, sigs.tolist()), key=lambda x: -x[1])
        top[g] = ranked_sig[:5]
        # Best-per-group still uses softmax because it's argmax — same answer
        ranked_sm = sorted(zip(tags, probs.tolist()), key=lambda x: -x[1])
        best[g] = ranked_sm[0][0]

    return {
        "flat": flat,
        "raw_cos": sims,
        "raw": raw,
        "softmax": sm,
        "sigmoid": sg,
        "top": top,
        "best_per_group": best,
    }
