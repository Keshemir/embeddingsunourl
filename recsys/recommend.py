"""Prototype recsys on top of the audio+text index.

- Taste vector = mean(liked) - β * mean(disliked)   (Rocchio)
- Retrieve top-K by late-fusion cosine, then MMR rerank for diversity.

For <1k tracks and <100 users we don't train anything: BPR / two-tower would
overfit instantly. Revisit when interactions cross ~10k.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from index.store import Index, search


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


def _stack(idx: Index, ids: Iterable[str], kind: str) -> np.ndarray:
    arr = idx.audio if kind == "audio" else idx.text
    pos = idx.meta.index[idx.meta["track_id"].isin(list(ids))].tolist()
    if not pos:
        return np.zeros((0, arr.shape[1]), dtype=np.float32)
    return arr[pos]


def taste_vector(
    idx: Index,
    liked_ids: Iterable[str],
    disliked_ids: Iterable[str] = (),
    kind: str = "audio",
    beta: float = 0.3,
) -> np.ndarray | None:
    """Rocchio-style taste vector for a single modality."""
    L = _stack(idx, liked_ids, kind)
    if len(L) == 0:
        return None
    v = L.mean(axis=0)
    D = _stack(idx, disliked_ids, kind)
    if len(D) > 0:
        v = v - beta * D.mean(axis=0)
    return _l2(v)


def mmr_rerank(
    candidates: pd.DataFrame,
    cand_vecs: np.ndarray,
    k: int,
    lam: float = 0.7,
) -> pd.DataFrame:
    """Maximal Marginal Relevance — pick top-k balancing relevance vs diversity.

    candidates must carry a "score" column; cand_vecs is the matrix aligned
    with the candidates' rows (use audio embeddings for "sounds different").
    """
    if len(candidates) <= k:
        return candidates.reset_index(drop=True)
    scores = candidates["score"].to_numpy()
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    sims = cand_vecs @ cand_vecs.T  # (M, M); rows are L2-normalized
    while len(selected) < k and remaining:
        if not selected:
            i = int(np.argmax(scores[remaining]))
            selected.append(remaining.pop(i))
            continue
        max_sim_to_selected = sims[remaining][:, selected].max(axis=1)
        mmr = lam * scores[remaining] - (1 - lam) * max_sim_to_selected
        i = int(np.argmax(mmr))
        selected.append(remaining.pop(i))
    return candidates.iloc[selected].reset_index(drop=True)


def recommend(
    idx: Index,
    liked_ids: Iterable[str],
    disliked_ids: Iterable[str] = (),
    k: int = 10,
    pool: int = 100,
    w_audio: float = 0.7,
    w_text: float = 0.3,
    mmr_lambda: float = 0.7,
) -> pd.DataFrame:
    liked_ids = list(liked_ids)
    disliked_ids = list(disliked_ids)

    audio_q = taste_vector(idx, liked_ids, disliked_ids, kind="audio")
    text_q = taste_vector(idx, liked_ids, disliked_ids, kind="text")

    exclude = set(liked_ids) | set(disliked_ids)
    pool_df = search(
        idx,
        audio_query=audio_q,
        text_query=text_q,
        w_audio=w_audio,
        w_text=w_text,
        k=pool,
        exclude_ids=exclude,
    )

    # Fetch audio vectors for the candidate pool, in the pool's row order
    id_to_pos = {tid: i for i, tid in enumerate(idx.meta["track_id"].tolist())}
    pool_pos = [id_to_pos[t] for t in pool_df["track_id"]]
    cand_vecs = idx.audio[pool_pos]

    return mmr_rerank(pool_df, cand_vecs, k=k, lam=mmr_lambda)
