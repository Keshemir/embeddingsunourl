"""Personalized feed = Rocchio taste vector + cold-start fallback.

Reads events from `recsys.events`, builds a per-user taste vector by
weighted-averaging the audio embeddings of liked/played tracks, subtracts
a fraction of the disliked-vector centroid (Rocchio), and runs cosine
top-k against the corpus with MMR + dedup on the result.

If the user has no interactions yet → cold-start: a diverse pool covering
many `best::genre` buckets so they have something to react to.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import INDEX_DIR
from recsys import events

# Action weights — tuned by hand; revise once we have real interactions.
WEIGHTS = {
    "like":     +1.0,
    "save":     +0.8,
    "share":    +0.5,
    "complete": +0.5,
    "play":     +0.1,    # only if completion_pct >= 0.30
    "skip":     -0.3,    # only if completion_pct <  0.30
    "dislike":  -1.0,
    "unlike":    0.0,    # informational; cancels nothing automatically
}

DEFAULT_HALF_LIFE_DAYS = 30.0
NEG_PULL = 0.3              # how much the dislike centroid pushes the taste away


def _decay(ts: float, now: float, half_life_days: float) -> float:
    days = max(0.0, (now - ts) / 86400.0)
    return math.exp(-math.log(2.0) * days / half_life_days)


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


def _track_index(meta: pd.DataFrame) -> dict[str, int]:
    return {t: i for i, t in enumerate(meta["track_id"].tolist())}


def taste_vector(
    user_id: str,
    audio: np.ndarray,
    meta: pd.DataFrame,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    now: float | None = None,
) -> np.ndarray | None:
    """Returns a unit vector, or None if there's no usable history."""
    df = events.load(user_id)
    if df.empty:
        return None
    now = now if now is not None else time.time()
    idx = _track_index(meta)

    pos_sum = np.zeros(audio.shape[1], dtype=np.float32)
    neg_sum = np.zeros(audio.shape[1], dtype=np.float32)
    pos_w = 0.0
    neg_w = 0.0
    for _, e in df.iterrows():
        action = e["action"]
        if action not in WEIGHTS:
            continue
        # Filter ambiguous plays — only count plays that actually played
        if action == "play":
            comp = e.get("completion_pct")
            if comp is None or (isinstance(comp, float) and (np.isnan(comp) or comp < 0.30)):
                continue
        if action == "skip":
            comp = e.get("completion_pct")
            # only count skips that aborted early (otherwise it's just a finish)
            if comp is None or (isinstance(comp, float) and (np.isnan(comp) or comp >= 0.30)):
                continue
        w = WEIGHTS[action] * _decay(float(e["ts"]), now, half_life_days)
        pos_idx = idx.get(e["track_id"])
        if pos_idx is None:
            continue
        v = audio[pos_idx]
        if w > 0:
            pos_sum += w * v
            pos_w += w
        elif w < 0:
            neg_sum += abs(w) * v
            neg_w += abs(w)

    if pos_w == 0 and neg_w == 0:
        return None
    if pos_w == 0:
        # Only negatives — push *away* from those, but we still need a base
        # vector. Use the mean of the corpus as the "neutral" starting point.
        base = audio.mean(axis=0)
        return _l2(base - NEG_PULL * (neg_sum / neg_w))
    pos_v = pos_sum / pos_w
    if neg_w == 0:
        return _l2(pos_v)
    neg_v = neg_sum / neg_w
    return _l2(pos_v - NEG_PULL * neg_v)


def cold_start(
    audio: np.ndarray,
    meta: pd.DataFrame,
    k: int = 20,
    rng_seed: int | None = 0,
) -> list[int]:
    """Diverse-by-best::genre starter pool.

    Pick one random track from each unique best::genre bucket, then fill the
    rest from the largest buckets. Cheap, deterministic with rng_seed."""
    rng = np.random.default_rng(rng_seed)
    if "best::genre" not in meta.columns:
        return list(rng.choice(len(meta), size=min(k, len(meta)), replace=False))
    out: list[int] = []
    seen_genres: set[str] = set()
    # round 1: one from each unique genre
    by_genre = meta.groupby("best::genre", sort=False).indices
    genres = list(by_genre.keys())
    rng.shuffle(genres)
    for g in genres:
        if len(out) >= k:
            break
        idxs = list(by_genre[g])
        out.append(int(rng.choice(idxs)))
        seen_genres.add(g)
    # round 2: fill remaining slots with the most populous buckets, no duplicates
    if len(out) < k:
        sizes = sorted(genres, key=lambda g: -len(by_genre[g]))
        for g in sizes:
            if len(out) >= k:
                break
            for tid in by_genre[g]:
                if tid not in out:
                    out.append(int(tid))
                    if len(out) >= k:
                        break
    return out[:k]


def recommend_for_user(
    user_id: str,
    audio: np.ndarray,
    meta: pd.DataFrame,
    *,
    k: int = 20,
    pool: int = 100,
    mmr_lambda: float = 0.65,
    dedup_threshold: float = 0.92,
    exclude_recent_seconds: float = 6 * 3600,
) -> tuple[list[int], dict]:
    """Returns (track_indices, debug_info)."""
    debug: dict = {"mode": None, "pool": pool, "k": k}
    q = taste_vector(user_id, audio, meta)
    if q is None:
        debug["mode"] = "cold_start"
        return cold_start(audio, meta, k=k), debug

    debug["mode"] = "personalized"
    sims = (audio @ q).astype(np.float32)

    # Drop recent listens
    excl = events.recent_track_ids(user_id, within_seconds=exclude_recent_seconds)
    if excl:
        idx = _track_index(meta)
        for t in excl:
            i = idx.get(t)
            if i is not None:
                sims[i] = -np.inf
        debug["excluded_recent"] = len(excl)

    pool_idx = [int(i) for i in np.argsort(-sims)[:pool] if sims[i] != -np.inf]
    if not pool_idx:
        return cold_start(audio, meta, k=k), {**debug, "mode": "cold_start_fallback"}

    # Dedup
    kept: list[int] = []
    for i in pool_idx:
        if all(float(audio[i] @ audio[k]) < dedup_threshold for k in kept):
            kept.append(i)
        if len(kept) >= max(k * 3, 30):
            break
    pool_idx = kept

    # MMR rerank
    selected: list[int] = []
    remaining = list(pool_idx)
    while len(selected) < k and remaining:
        if not selected:
            i = max(remaining, key=lambda c: sims[c])
        else:
            def score(c: int) -> float:
                rel = float(sims[c])
                div = max(float(audio[c] @ audio[s]) for s in selected)
                return mmr_lambda * rel - (1 - mmr_lambda) * div
            i = max(remaining, key=score)
        selected.append(i)
        remaining.remove(i)
    return selected, debug


def profile_summary(
    user_id: str,
    audio: np.ndarray,
    meta: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    """Top genres/moods/instruments aligned with the taste vector — for /api/profile."""
    q = taste_vector(user_id, audio, meta)
    if q is None:
        return {"taste": None, "n_events": int(len(events.load(user_id)))}
    summary: dict = {"taste": "personalized", "n_events": int(len(events.load(user_id)))}
    # For each best::group column, find the most-aligned group label by
    # measuring the mean cosine of taste vec with tracks in that bucket.
    for col in ("best::genre", "best::mood", "best::instrument"):
        if col not in meta.columns:
            continue
        sims_to_q = audio @ q
        means = (
            meta.assign(_s=sims_to_q)
            .groupby(col)["_s"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )
        summary[col] = [(label, float(s)) for label, s in means.items()]
    return summary
