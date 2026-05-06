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
    "play":     +0.1,    # baseline; bumped to +0.5 if completion_pct >= 0.30
    "skip":     -0.3,    # only if completion_pct < 0.30 explicitly
    "dislike":  -1.0,
    "unlike":   -1.0,    # subtract the prior like; see _resolve_unlikes()
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
    return _taste_vector_from_events(df, audio, meta,
                                      half_life_days=half_life_days, now=now)


def taste_vector_style(
    user_id: str,
    style_vecs: np.ndarray,
    meta: pd.DataFrame,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    now: float | None = None,
) -> np.ndarray | None:
    """Rocchio taste vector but in **style space** (BGE-M3 1024-d).

    Mirror of `taste_vector` (audio space) — same weights, same recency
    decay, same NaN-safe play/skip handling — just operates on style.npy
    rows instead of audio.npy rows. Used by the Wave mixer to combine
    user taste with slider direction in a single embedding space.

    Returns a unit vector or None if the user has no usable history.
    """
    df = events.load(user_id)
    return _taste_vector_from_events(df, style_vecs, meta,
                                      half_life_days=half_life_days, now=now)


def _taste_vector_from_events(
    df: pd.DataFrame,
    audio: np.ndarray,
    meta: pd.DataFrame,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    now: float | None = None,
) -> np.ndarray | None:
    """Same Rocchio computation but takes the events DataFrame directly,
    so callers like profile_summary can read the log once and reuse it.
    Works in any embedding space — the `audio` argument can be either
    audio.npy (default for taste_vector) or style.npy (for taste_vector_style).
    """
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
        # play/skip require thoughtful completion-pct handling — see C4 in audit.
        # Default `play` weight is +0.1; if the player reported >=30% completion
        # we upgrade it to "complete-ish" weight (+0.5). Missing completion_pct
        # is OK for play (browser tab unfocused, etc.) — keep the small +0.1 signal.
        # For skip, we only count it as a *negative* signal when completion_pct
        # is explicitly < 0.30; without that evidence we ignore it (we don't
        # know if the user skipped early or just listened through).
        comp = e.get("completion_pct")
        comp_known = comp is not None and not (isinstance(comp, float) and np.isnan(comp))

        if action == "play":
            base = WEIGHTS["complete"] if (comp_known and comp >= 0.30) else WEIGHTS["play"]
        elif action == "skip":
            if not comp_known or comp >= 0.30:
                continue   # not a real abort signal
            base = WEIGHTS["skip"]
        else:
            base = WEIGHTS[action]

        w = base * _decay(float(e["ts"]), now, half_life_days)
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
    rng_seed: int | None = None,
    user_id: str | None = None,
) -> list[int]:
    """Diverse-by-best::genre starter pool.

    Pick one random track from each unique best::genre bucket, then fill the
    rest from the largest buckets.

    Stability: if `user_id` is provided, we hash it into a deterministic seed
    so the same user always sees the same cold-start order across reloads
    (until they have likes). Different users see different starts. If neither
    `rng_seed` nor `user_id` is set, the order is fresh on every call.
    """
    if rng_seed is None and user_id:
        # Stable per-user but distinct across users.
        import hashlib as _h
        rng_seed = int(_h.sha1(user_id.encode("utf-8")).hexdigest()[:8], 16)
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
        return cold_start(audio, meta, k=k, user_id=user_id), debug

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
        return cold_start(audio, meta, k=k, user_id=user_id), {**debug, "mode": "cold_start_fallback"}

    # MMR rerank first — its diversity penalty already discourages picking
    # near-clones. Then a final dedup pass collapses the remaining
    # near-identical Suno takes that slipped through MMR's diversity term.
    selected: list[int] = []
    remaining = list(pool_idx)
    target = k + 5  # over-pick a bit so dedup has slack
    while len(selected) < target and remaining:
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

    # Final dedup against actual waveform duplicates (multiple Suno takes
    # of the same prompt have audio cosine ≥ 0.92 even if MMR ranked them
    # apart on relevance).
    final: list[int] = []
    for i in selected:
        if all(float(audio[i] @ audio[k_]) < dedup_threshold for k_ in final):
            final.append(i)
        if len(final) >= k:
            break
    return final, debug


def profile_summary(
    user_id: str,
    audio: np.ndarray,
    meta: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    """Top genres/moods/instruments aligned with the taste vector.

    One read of events.parquet (was three: one inside taste_vector +
    two for n_events).
    """
    df = events.load(user_id)
    n_events = int(len(df))
    q = _taste_vector_from_events(df, audio, meta)
    if q is None:
        return {"taste": None, "n_events": n_events}
    summary: dict = {"taste": "personalized", "n_events": n_events}
    sims_to_q = audio @ q  # compute once, reuse across groups
    for col in ("best::genre", "best::mood", "best::instrument"):
        if col not in meta.columns:
            continue
        means = (
            meta.assign(_s=sims_to_q)
            .groupby(col)["_s"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )
        summary[col] = [(label, float(s)) for label, s in means.items()]
    return summary
