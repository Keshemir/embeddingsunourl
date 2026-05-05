"""FastAPI server: track list, similar, search, feed, event log.

Usage:
    uvicorn scripts.serve:app --reload --port 8000
    # or
    python -m scripts.serve

Then open http://localhost:8000 — static lander served from `static/`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INDEX_DIR, ROOT
from embed.audio import AudioEmbedder
from recsys import events, feed

STATIC_DIR = ROOT / "static"

# ----- one-time globals (loaded at startup) ---------------------------------
_audio: np.ndarray | None = None
_meta: pd.DataFrame | None = None
_idx: dict[str, int] | None = None
_embedder: AudioEmbedder | None = None


def _load_index() -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    global _audio, _meta, _idx
    if _audio is None:
        _audio = np.load(INDEX_DIR / "audio.npy")
        _meta = pd.read_parquet(INDEX_DIR / "tracks.parquet")
        _idx = {t: i for i, t in enumerate(_meta["track_id"].tolist())}
    assert _audio is not None and _meta is not None and _idx is not None
    return _audio, _meta, _idx


def _ensure_embedder() -> AudioEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = AudioEmbedder.load()
    return _embedder


def _top_tags_for_row(meta: pd.DataFrame, i: int, n: int = 5) -> list[dict]:
    """Return the top-N tags for a track using **z-score** (hubness-corrected).

    Why not sigmoid: many tag text-vectors live in dense regions of the joint
    space and rack up sigmoid ≈ 0.99 across the entire corpus (e.g.
    `tag::instrument::duduk` had mean 0.986, std 0.011 — useless for ranking).
    Z-score subtracts the corpus mean and divides by std per tag, so the
    surfaced tags are the ones where THIS track is actually unusual.

    Falls back to sigmoid if no z::* columns exist (single-track index).
    """
    row = meta.iloc[i]
    z_cols = [c for c in meta.columns if c.startswith("z::")]
    if z_cols:
        cols, prefix = z_cols, "z::"
    else:
        cols, prefix = [c for c in meta.columns if c.startswith("tag::")], "tag::"
    if not cols:
        return []
    scores = [(c, float(row[c])) for c in cols if not np.isnan(row[c])]
    scores.sort(key=lambda x: -x[1])
    out = []
    for col, val in scores[:n]:
        parts = col.split("::", 2)
        if len(parts) != 3:
            continue
        out.append({
            "group": parts[1],
            "tag": parts[2].replace("_", " "),
            "score": round(val, 3),  # z-score (~ ±2 = significant)
            "metric": "z" if prefix == "z::" else "sigmoid",
        })
    return out


def _best_per_group_zscored(meta: pd.DataFrame, i: int) -> dict[str, str]:
    """Recompute best::group via argmax over z-score columns.

    The original best::* in parquet was argmax of softmax — fine for tags that
    aren't biased by hubness, but for groups where one tag dominates (e.g.
    "instrumental track without vocals" is high for everything) it lies.
    Falling back per-group: if a z::group::* column exists, take argmax over
    those; otherwise keep whatever sits in best::group.
    """
    out: dict[str, str] = {}
    row = meta.iloc[i]
    z_cols = [c for c in meta.columns if c.startswith("z::")]
    by_group: dict[str, list[str]] = {}
    for c in z_cols:
        parts = c.split("::", 2)
        if len(parts) == 3:
            by_group.setdefault(parts[1], []).append(c)
    for g, cols in by_group.items():
        vals = [(c, float(row[c])) for c in cols if not np.isnan(row[c])]
        if not vals:
            continue
        col, _ = max(vals, key=lambda x: x[1])
        out[g] = col.split("::", 2)[2].replace("_", " ")
    # Fallback: legacy best::* columns for groups not covered (e.g. fusion if it
    # wasn't z-scored, or single-track index where calibration didn't run).
    for col in meta.columns:
        if col.startswith("best::"):
            g = col.split("::", 1)[1]
            if g not in out:
                v = row[col]
                if isinstance(v, str) and v:
                    out[g] = v
    return out


def _track_card(meta: pd.DataFrame, i: int) -> dict[str, Any]:
    r = meta.iloc[i]
    def s(col, default=""):
        v = r.get(col, default)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v
    best = _best_per_group_zscored(meta, i)
    return {
        "track_id": str(s("track_id")),
        "title": str(s("title")) or "(untitled)",
        "source": str(s("source")),
        "duration_sec": float(s("duration_sec", 0.0) or 0.0),
        "bpm": float(s("bpm_perceived", 0.0) or 0.0),
        "key": str(s("key")),
        "best_genre": best.get("genre", str(s("best::genre"))),
        "best_mood": best.get("mood", str(s("best::mood"))),
        "best_instrument": best.get("instrument", str(s("best::instrument"))),
        "best_vocal": best.get("vocal", str(s("best::vocal"))),
        "best_fusion": best.get("fusion", ""),
        "prompt": str(s("prompt")),
        "top_tags": _top_tags_for_row(meta, i, n=5),
    }


# ----- app ------------------------------------------------------------------

app = FastAPI(title="ozenref")

# Single-user prototype — wide-open CORS is fine. Tighten before going public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    _load_index()  # warm up
    print(f"[serve] loaded {len(_meta)} tracks, dim={_audio.shape[1]}")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Mount static AFTER /, so / hits the route, not the file
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/tracks")
def api_tracks(limit: int = 200, offset: int = 0) -> dict:
    audio, meta, _ = _load_index()
    end = min(offset + limit, len(meta))
    return {
        "total": int(len(meta)),
        "tracks": [_track_card(meta, i) for i in range(offset, end)],
    }


@app.get("/api/similar/{track_id}")
def api_similar(track_id: str, k: int = 10) -> dict:
    audio, meta, idx = _load_index()
    i = idx.get(track_id)
    if i is None:
        raise HTTPException(404, f"track_id {track_id!r} not found")
    q = audio[i]
    sims = (audio @ q).astype(np.float32)
    sims[i] = -np.inf
    # dedup
    pool = [int(j) for j in np.argsort(-sims)[: max(k * 5, 50)] if sims[j] != -np.inf]
    kept: list[int] = []
    for j in pool:
        if all(float(audio[j] @ audio[k_]) < 0.92 for k_ in kept):
            kept.append(j)
        if len(kept) >= k:
            break
    return {
        "seed": _track_card(meta, i),
        "tracks": [
            {**_track_card(meta, j), "sim": float(sims[j])}
            for j in kept[:k]
        ],
    }


_QUERY_TEMPLATES = [
    "{q}",
    "a recording of {q} music",
    "a {q} track",
    "music in the style of {q}",
    "this song is {q}",
]


def _embed_query_ensemble(em: AudioEmbedder, q: str) -> np.ndarray:
    """Average L2-normed embeddings across N templates → re-normalize.
    Same trick as prompt ensembling on the tag side; +3-5% retrieval on
    short single-word queries like "arabian"."""
    vecs = np.stack([em.embed_text(t.format(q=q)) for t in _QUERY_TEMPLATES])
    avg = vecs.mean(axis=0)
    return avg / (np.linalg.norm(avg) + 1e-12)


def _style_substring_hits(meta: pd.DataFrame, q: str) -> list[int]:
    """Boolean substring search over Suno's own style description.

    `style` is the rich text Suno authored at generation time
    (e.g. "Russian pop-rock track featuring..."), pulled by ingest.suno_url.
    For unambiguous queries like "trap" or "arabic" this is way more reliable
    than any zero-shot tag or audio-text cosine — the words came from Suno
    itself.

    Returns indices where ANY of the query's tokens appears in style as a
    word-boundary substring (case-insensitive).
    """
    style = meta["style"].fillna("").astype(str).str.lower()
    tokens = [w for w in q.lower().split() if len(w) >= 3]
    if not tokens:
        return []
    # Word-boundary contains. Multiple tokens → OR.
    import re as _re
    pat = "|".join(rf"\b{_re.escape(t)}" for t in tokens)
    mask = style.str.contains(pat, regex=True, na=False)
    return [int(i) for i in meta.index[mask]]


@app.get("/api/search")
def api_search(q: str, k: int = 10, ensemble: bool = True) -> dict:
    """Two-stage hybrid search:
       1. Boolean substring match in Suno's `style` text (the source of truth).
       2. Audio-text cosine in MuQ-MuLan space (catches cases where the user
          phrased it differently from what Suno wrote).
       Stage-1 hits are surfaced first, then stage-2 fills the rest, with
       near-duplicate collapse on top.
    """
    if not q.strip():
        raise HTTPException(400, "empty query")
    audio, meta, _ = _load_index()
    em = _ensure_embedder()
    if not em.supports_text:
        raise HTTPException(500, "audio model has no text encoder")

    # Stage 1: substring hits in Suno style text (sorted by audio sim within)
    qv = _embed_query_ensemble(em, q) if ensemble else em.embed_text(q)
    sims = (audio @ qv).astype(np.float32)
    style_hits = _style_substring_hits(meta, q)
    style_hits.sort(key=lambda i: -sims[i])

    # Stage 2: cosine pool excluding what's already in stage 1
    seen = set(style_hits)
    cosine_pool = [int(j) for j in np.argsort(-sims) if int(j) not in seen][: max(k * 5, 50)]

    # Dedup near-identical takes
    def _take(idx_list, limit):
        kept = []
        for j in idx_list:
            if all(float(audio[j] @ audio[k_]) < 0.92 for k_ in kept):
                kept.append(j)
            if len(kept) >= limit:
                break
        return kept

    style_kept = _take(style_hits, k)
    if len(style_kept) < k:
        cosine_kept = _take(cosine_pool, k - len(style_kept))
        ordered = style_kept + cosine_kept
    else:
        ordered = style_kept

    return {
        "query": q,
        "ensemble": ensemble,
        "n_style_hits": len(style_hits),
        "tracks": [
            {**_track_card(meta, j), "sim": float(sims[j]),
             "match": "style" if j in seen else "audio"}
            for j in ordered[:k]
        ],
    }


@app.get("/api/feed")
def api_feed(user_id: str, k: int = 20) -> dict:
    audio, meta, _ = _load_index()
    selected, debug = feed.recommend_for_user(user_id, audio, meta, k=k)
    return {
        "user_id": user_id,
        "debug": debug,
        "tracks": [_track_card(meta, j) for j in selected],
    }


class EventIn(BaseModel):
    user_id: str
    track_id: str
    action: str
    session_id: str = ""
    completion_pct: float | None = None
    source: str = "manual"
    ts: float | None = Field(default=None, description="unix seconds; server fills if absent")


@app.post("/api/event")
def api_event(ev: EventIn) -> dict:
    if ev.action not in events.VALID_ACTIONS:
        raise HTTPException(400, f"unknown action {ev.action!r}")
    _, meta, idx = _load_index()
    if ev.track_id not in idx:
        raise HTTPException(404, f"track_id {ev.track_id!r} not found")
    row = events.append(
        ev.user_id, ev.track_id, ev.action,
        ts=ev.ts, session_id=ev.session_id,
        completion_pct=ev.completion_pct, source=ev.source,
    )
    return {"ok": True, "event": row}


@app.get("/api/profile/{user_id}")
def api_profile(user_id: str) -> dict:
    audio, meta, _ = _load_index()
    return feed.profile_summary(user_id, audio, meta)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
