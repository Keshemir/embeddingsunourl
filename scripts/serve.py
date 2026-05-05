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


def _track_card(meta: pd.DataFrame, i: int) -> dict[str, Any]:
    r = meta.iloc[i]
    def s(col, default=""):
        v = r.get(col, default)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v
    return {
        "track_id": str(s("track_id")),
        "title": str(s("title")) or "(untitled)",
        "source": str(s("source")),
        "duration_sec": float(s("duration_sec", 0.0) or 0.0),
        "bpm": float(s("bpm_perceived", 0.0) or 0.0),
        "key": str(s("key")),
        "best_genre": str(s("best::genre")),
        "best_mood": str(s("best::mood")),
        "best_instrument": str(s("best::instrument")),
        "best_vocal": str(s("best::vocal")),
        "prompt": str(s("prompt")),
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


@app.get("/api/search")
def api_search(q: str, k: int = 10) -> dict:
    if not q.strip():
        raise HTTPException(400, "empty query")
    audio, meta, _ = _load_index()
    em = _ensure_embedder()
    if not em.supports_text:
        raise HTTPException(500, "audio model has no text encoder")
    qv = em.embed_text(q)
    sims = (audio @ qv).astype(np.float32)
    pool = [int(j) for j in np.argsort(-sims)[: max(k * 5, 50)]]
    kept: list[int] = []
    for j in pool:
        if all(float(audio[j] @ audio[k_]) < 0.92 for k_ in kept):
            kept.append(j)
        if len(kept) >= k:
            break
    return {
        "query": q,
        "tracks": [
            {**_track_card(meta, j), "sim": float(sims[j])}
            for j in kept[:k]
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
