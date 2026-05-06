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
import json
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
from embed.style import StyleEncoder
from features import audio_features
from features.suno_tags import parse as parse_suno_tags, flat_tags as flat_suno_tags
from ingest.suno import load_track
from ingest.suno_url import resolve as resolve_suno_url
from recsys import events, feed

STATIC_DIR = ROOT / "static"

# ----- one-time globals (loaded at startup) ---------------------------------
_audio: np.ndarray | None = None
_style: np.ndarray | None = None       # (N, 1024) BGE-M3 embeddings of Suno style text
_meta: pd.DataFrame | None = None
_idx: dict[str, int] | None = None
_audio_emb: AudioEmbedder | None = None
_style_emb: StyleEncoder | None = None


def _load_index() -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    global _audio, _style, _meta, _idx
    if _audio is None:
        _audio = np.load(INDEX_DIR / "audio.npy")
        _meta = pd.read_parquet(INDEX_DIR / "tracks.parquet")
        _idx = {t: i for i, t in enumerate(_meta["track_id"].tolist())}
        # Style embeddings are optional — server still runs without them, just
        # falling back to the audio path. Built by `python -m scripts.embed_styles`.
        sp = INDEX_DIR / "style.npy"
        if sp.exists():
            _style = np.load(sp)
            print(f"[serve] style index loaded: {_style.shape}")
        else:
            print(f"[serve] style.npy missing — text search will use audio cosine only")
    assert _audio is not None and _meta is not None and _idx is not None
    return _audio, _meta, _idx


def _ensure_audio_embedder() -> AudioEmbedder:
    global _audio_emb
    if _audio_emb is None:
        _audio_emb = AudioEmbedder.load()
    return _audio_emb


def _ensure_style_encoder() -> StyleEncoder | None:
    global _style_emb
    if _style is None:
        return None
    if _style_emb is None:
        _style_emb = StyleEncoder()
    return _style_emb


def _top_tags_for_row(meta: pd.DataFrame, i: int, n: int = 5) -> list[dict]:
    """Tags shown on the track card.

    Priority:
      1. **Suno's own parsed tags** (from `suno_tags_flat` JSON column —
         genre/instrument/mood phrases that Suno itself wrote in the style
         description). These are ground truth.
      2. **Z-score zero-shot tags** (from z::* columns) as fallback when
         Suno tags are missing. Hubness-corrected.
      3. Sigmoid (tag::*) — last resort, raw cosines.
    """
    row = meta.iloc[i]
    # 1. Suno parsed tags
    if "suno_tags_flat" in meta.columns:
        raw = row.get("suno_tags_flat")
        if isinstance(raw, str) and raw.strip():
            try:
                tags = json.loads(raw)
                if tags:
                    return [{**t, "metric": "suno"} for t in tags[:n]]
            except json.JSONDecodeError:
                pass

    # 2. Z-score fallback
    z_cols = [c for c in meta.columns if c.startswith("z::")]
    if z_cols:
        cols, prefix = z_cols, "z::"
    else:
        cols, prefix = [c for c in meta.columns if c.startswith("tag::")], "tag::"
    if not cols:
        return []
    scores = [(c, float(row[c])) for c in cols if not np.isnan(row[c])]
    scores.sort(key=lambda x: -x[1])
    out: list[dict] = []
    for col, val in scores[:n]:
        parts = col.split("::", 2)
        if len(parts) != 3:
            continue
        out.append({
            "group": parts[1],
            "tag": parts[2].replace("_", " "),
            "score": round(val, 3),
            "metric": "z" if prefix == "z::" else "sigmoid",
        })
    return out


def _best_per_group_zscored(meta: pd.DataFrame, i: int) -> dict[str, str]:
    """Best label per group, prioritizing Suno's own parsed tags.

    Order:
      1. First entry in `suno_tags` group list (e.g. parsed["genre"][0]).
         These are the words Suno itself wrote, so they're ground truth.
      2. Argmax over z-score columns for groups not covered by Suno tags.
      3. Legacy best::* string columns as ultimate fallback.
    """
    out: dict[str, str] = {}
    row = meta.iloc[i]

    # 1. Suno parsed tags
    if "suno_tags" in meta.columns:
        raw = row.get("suno_tags")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                for g, tags in parsed.items():
                    if tags:
                        out[g] = str(tags[0]).lower()
            except json.JSONDecodeError:
                pass

    # 2. Z-score argmax for groups not covered
    z_cols = [c for c in meta.columns if c.startswith("z::")]
    by_group: dict[str, list[str]] = {}
    for c in z_cols:
        parts = c.split("::", 2)
        if len(parts) == 3 and parts[1] not in out:
            by_group.setdefault(parts[1], []).append(c)
    for g, cols in by_group.items():
        vals = [(c, float(row[c])) for c in cols if not np.isnan(row[c])]
        if not vals:
            continue
        col, _ = max(vals, key=lambda x: x[1])
        out[g] = col.split("::", 2)[2].replace("_", " ")

    # 3. Legacy best::* fallback
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


@app.get("/api/search")
def api_search(q: str, k: int = 10) -> dict:
    """Semantic text-text search over Suno style descriptions.

    Primary signal is BGE-M3 (multilingual semantic encoder) cosine
    between the query and each track's style embedding (style.npy).
    A query "arabian" matches tracks described with "oud", "ney flute",
    "oriental percussion" etc. without literal substring overlap.
    Russian queries ("арабский", "минор") work the same way — BGE-M3
    is multilingual.

    Audio cosine is kept only as a fallback when style.npy isn't built
    yet (legacy MuQ-MuLan text path; less accurate on a one-author
    Suno corpus).

    Near-duplicate Suno takes (audio cos >= 0.92) are collapsed.
    """
    if not q.strip():
        raise HTTPException(400, "empty query")
    audio, meta, _ = _load_index()

    style_enc = _ensure_style_encoder()
    if style_enc is not None and _style is not None:
        # Primary path: text-text semantic search via BGE-M3
        qv = style_enc.encode(q)                # (1024,) L2-normalized
        sims = (_style @ qv).astype(np.float32)
        match_kind = "style"
    else:
        # Fallback: MuQ-MuLan audio-text cosine (legacy)
        em = _ensure_audio_embedder()
        if not em.supports_text:
            raise HTTPException(500, "no style.npy and audio model has no text encoder — run scripts.embed_styles first")
        qv = _embed_query_ensemble(em, q)
        sims = (audio @ qv).astype(np.float32)
        match_kind = "audio"

    pool = [int(j) for j in np.argsort(-sims)[: max(k * 5, 50)]]
    kept: list[int] = []
    for j in pool:
        # dedup using audio cosine (Suno emits many takes of the same prompt)
        if all(float(audio[j] @ audio[k_]) < 0.92 for k_ in kept):
            kept.append(j)
        if len(kept) >= k:
            break

    return {
        "query": q,
        "match_kind": match_kind,
        "tracks": [
            {**_track_card(meta, j), "sim": float(sims[j]), "match": match_kind}
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


# ---------------- single-track ingest ---------------------------------------

class IngestIn(BaseModel):
    url: str
    keep_audio: bool = False


def _append_track(track: dict, audio_vec: np.ndarray, style_vec: np.ndarray) -> int:
    """Append one track row + 2 vectors to the on-disk index, return its row index."""
    global _audio, _style, _meta, _idx
    audio_path = INDEX_DIR / "audio.npy"
    style_path = INDEX_DIR / "style.npy"
    tracks_path = INDEX_DIR / "tracks.parquet"

    audio_existing = np.load(audio_path)
    style_existing = np.load(style_path) if style_path.exists() else None
    df = pd.read_parquet(tracks_path)

    # Make the new row schema-compatible (fill missing columns with NA)
    new_row = {c: track.get(c, None) for c in df.columns}
    for k, v in track.items():
        if k not in new_row:
            new_row[k] = v
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    audio_new = np.concatenate([audio_existing, audio_vec[None, :]], axis=0).astype(np.float32)
    np.save(audio_path, audio_new)
    if style_existing is not None:
        style_new = np.concatenate([style_existing, style_vec[None, :]], axis=0).astype(np.float32)
        np.save(style_path, style_new)
    df.to_parquet(tracks_path, index=False)

    # Invalidate the in-memory cache so next request re-loads
    _audio = None
    _style = None
    _meta = None
    _idx = None
    return len(df) - 1


@app.post("/api/ingest")
def api_ingest(req: IngestIn) -> dict:
    """Add a Suno share URL to the index in real time.

    Pipeline (~10-30 sec depending on download speed):
        URL → fetch share-page → mp3 download → MuQ audio embed
            → BGE-M3 style embed (on Suno style description)
            → librosa numerical features → Suno tag parse
            → append to audio.npy + style.npy + tracks.parquet
    """
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "expected http(s) URL")
    audio_arr, meta, idx_map = _load_index()

    # 1. fetch share page + download mp3
    try:
        mp3_path, sm = resolve_suno_url(req.url)
    except Exception as e:
        raise HTTPException(400, f"could not resolve URL: {e}")

    # 2. de-dup: if this Suno UUID is already in the index, return it
    if sm.suno_id and sm.suno_id in idx_map:
        if not req.keep_audio:
            mp3_path.unlink(missing_ok=True)
            mp3_path.with_suffix(".json").unlink(missing_ok=True)
        return {"track_id": sm.suno_id, "status": "already_indexed",
                "row": idx_map[sm.suno_id]}

    # 3. enrich Track from local mp3 + Suno meta
    track = load_track(mp3_path)
    if sm.suno_id: track.track_id = sm.suno_id
    if sm.title:   track.title = sm.title
    if sm.prompt:  track.prompt = sm.prompt
    if sm.tags:    track.style = sm.tags
    if sm.lyrics:  track.lyrics = sm.lyrics

    # 4. embed audio
    aud_em = _ensure_audio_embedder()
    audio_vec = aud_em.embed_audio(str(mp3_path))

    # 5. embed style text via BGE-M3
    style_text = (track.style or track.title or "music").strip()
    if track.title and track.title not in style_text:
        style_text = style_text + "\n\n" + track.title
    sty_em = StyleEncoder() if _style_emb is None else _style_emb
    style_vec = sty_em.encode(style_text)

    # 6. librosa numerical features
    try:
        feats = audio_features.extract(str(mp3_path))
    except Exception:
        feats = {}

    # 7. parse Suno tags
    parsed = parse_suno_tags(track.style or "")
    flat = flat_suno_tags(parsed)

    row: dict = {
        "track_id": track.track_id,
        "source": req.url,
        "path": str(mp3_path) if req.keep_audio else "",
        "title": track.title,
        "prompt": track.prompt,
        "style": track.style,
        "lyrics": track.lyrics,
        "bpm_meta": track.bpm,
        "key_meta": track.key,
        "suno_tags": json.dumps(parsed, ensure_ascii=False),
        "suno_tags_flat": json.dumps(flat, ensure_ascii=False),
    }
    row.update(feats)

    # 8. append to indices
    new_idx = _append_track(row, audio_vec, style_vec)

    # 9. clean up downloaded mp3 unless asked to keep
    if not req.keep_audio:
        Path(mp3_path).unlink(missing_ok=True)
        Path(mp3_path).with_suffix(".json").unlink(missing_ok=True)

    return {
        "track_id": track.track_id,
        "row": new_idx,
        "status": "ingested",
        "title": track.title,
        "suno_tags": flat,
        "audio_dim": int(audio_vec.shape[0]),
        "style_dim": int(style_vec.shape[0]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
