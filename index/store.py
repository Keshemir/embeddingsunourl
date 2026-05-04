"""Vector store for the prototype: numpy + parquet, cosine via dot product on
L2-normalized vectors. Late-fusion search across (audio, text) indices.

Layout under DATA_DIR/index/:
    audio.npy   — (N, audio_dim) float32, L2-normalized
    text.npy    — (N, text_dim) float32, L2-normalized
    meta.parquet — track metadata, row order matches the .npy files
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import INDEX_DIR


@dataclass
class Index:
    audio: np.ndarray  # (N, Da)
    text: np.ndarray   # (N, Dt)
    meta: pd.DataFrame  # index aligned with rows

    def __len__(self) -> int:
        return len(self.meta)


def save(idx: Index, root: Path = INDEX_DIR) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "audio.npy", idx.audio)
    np.save(root / "text.npy", idx.text)
    idx.meta.to_parquet(root / "meta.parquet", index=False)


def load(root: Path = INDEX_DIR) -> Index:
    root = Path(root)
    return Index(
        audio=np.load(root / "audio.npy"),
        text=np.load(root / "text.npy"),
        meta=pd.read_parquet(root / "meta.parquet"),
    )


def _cos(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # query: (D,) or (Q, D); mat: (N, D); both L2-normalized → dot = cosine
    if query.ndim == 1:
        return mat @ query
    return mat @ query.T  # (N, Q)


def search(
    idx: Index,
    audio_query: np.ndarray | None = None,
    text_query: np.ndarray | None = None,
    w_audio: float = 0.7,
    w_text: float = 0.3,
    k: int = 20,
    exclude_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Late-fusion top-k. Either query may be None — weights renormalize."""
    if audio_query is None and text_query is None:
        raise ValueError("at least one of audio_query / text_query must be set")

    scores = np.zeros(len(idx), dtype=np.float32)
    wa = w_audio if audio_query is not None else 0.0
    wt = w_text if text_query is not None else 0.0
    total = wa + wt
    if total == 0:
        raise ValueError("both weights are zero")
    wa, wt = wa / total, wt / total

    if audio_query is not None:
        scores += wa * _cos(audio_query.astype(np.float32), idx.audio)
    if text_query is not None:
        scores += wt * _cos(text_query.astype(np.float32), idx.text)

    if exclude_ids:
        mask = idx.meta["track_id"].isin(exclude_ids).to_numpy()
        scores[mask] = -np.inf

    top = np.argsort(-scores)[:k]
    out = idx.meta.iloc[top].copy()
    out["score"] = scores[top]
    return out.reset_index(drop=True)


def get_vector(idx: Index, track_id: str, kind: str = "audio") -> np.ndarray:
    pos = idx.meta.index[idx.meta["track_id"] == track_id]
    if len(pos) == 0:
        raise KeyError(track_id)
    arr = idx.audio if kind == "audio" else idx.text
    return arr[int(pos[0])]
