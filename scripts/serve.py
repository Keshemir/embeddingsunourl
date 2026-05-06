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
import os
import secrets
import threading
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INDEX_DIR, ROOT
from embed.audio import AudioEmbedder
from embed.style import StyleEncoder, build_style_text
from features import audio_features
from features import calibrate as cal_mod
from features.suno_tags import parse as parse_suno_tags, flat_tags as flat_suno_tags
from features import zeroshot
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


_INDEX_LOCK = threading.RLock()  # serialize load + append (M6)


def _load_index() -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    global _audio, _style, _meta, _idx
    with _INDEX_LOCK:
        if _audio is None:
            _audio = np.load(INDEX_DIR / "audio.npy")
            _meta = pd.read_parquet(INDEX_DIR / "tracks.parquet")
            _idx = {t: i for i, t in enumerate(_meta["track_id"].tolist())}
            sp = INDEX_DIR / "style.npy"
            if sp.exists():
                _style = np.load(sp)
                print(f"[serve] style index loaded: {_style.shape}")
            else:
                print(f"[serve] style.npy missing — text search will use audio cosine only")
        # Hard consistency check — ingest was supposed to keep these in sync.
        # If any step failed mid-write, surface it loudly instead of returning
        # mismatched arrays that silently mis-pair tracks at search time.
        n = len(_meta)
        assert _audio.shape[0] == n, (
            f"index mismatch: audio.npy has {_audio.shape[0]} rows, "
            f"tracks.parquet has {n}. Last ingest likely failed mid-write."
        )
        if _style is not None:
            assert _style.shape[0] == n, (
                f"index mismatch: style.npy has {_style.shape[0]} rows, "
                f"tracks.parquet has {n}. Run scripts.embed_styles to rebuild."
            )
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


UID_COOKIE = "ozenref_uid"
COOKIE_TTL = 365 * 24 * 3600


def _ensure_uid_cookie(request: Request, response: Response) -> str:
    """Return the request's user id, minting one and setting the cookie if
    absent. We're not authenticating against a real account — this is just
    a stable per-browser identifier so the recsys can keep separate taste
    vectors. Replace with proper auth (OAuth, signed JWT) for production."""
    uid = request.cookies.get(UID_COOKIE)
    if not uid or len(uid) < 8:
        uid = "u_" + secrets.token_urlsafe(12)
        response.set_cookie(
            UID_COOKIE, uid,
            max_age=COOKIE_TTL, httponly=True, samesite="lax",
        )
    return uid


@app.get("/")
def root(request: Request, response: Response) -> FileResponse:
    _ensure_uid_cookie(request, response)
    fr = FileResponse(STATIC_DIR / "index.html")
    # Mirror the cookie into the FileResponse — Starlette ignores set_cookie
    # on `response` since we return a different object.
    if UID_COOKIE not in request.cookies:
        fr.set_cookie(
            UID_COOKIE,
            request.cookies.get(UID_COOKIE) or response.headers.get("set-cookie", "").split("=", 1)[-1].split(";", 1)[0]
            or "u_" + secrets.token_urlsafe(12),
            max_age=COOKIE_TTL, httponly=True, samesite="lax",
        )
    return fr


@app.get("/api/me")
def api_me(request: Request, response: Response) -> dict:
    """Return the caller's user id, minting one if absent. Frontend calls
    this on load to learn its uid (httponly cookie isn't readable from JS)."""
    uid = _ensure_uid_cookie(request, response)
    return {"user_id": uid}


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
        # Dedup in the same space we ranked in: two tracks the search
        # engine considers "the same hit" must also be near each other in
        # the embedding it judged on. Suno also emits many takes of the
        # same prompt, so we additionally check audio similarity as a
        # belt-and-braces filter against pure waveform duplicates.
        dedup_mat = _style
        text_threshold = 0.97   # BGE-M3 cosines run high — be strict
        audio_threshold = 0.92  # audio dups (same prompt, multiple takes)
        match_kind = "style"
    else:
        em = _ensure_audio_embedder()
        if not em.supports_text:
            raise HTTPException(500, "no style.npy and audio model has no text encoder — run scripts.embed_styles first")
        qv = _embed_query_ensemble(em, q)
        sims = (audio @ qv).astype(np.float32)
        dedup_mat = audio
        text_threshold = 0.92
        audio_threshold = 0.92  # same matrix; only one threshold applies
        match_kind = "audio"

    pool = [int(j) for j in np.argsort(-sims)[: max(k * 5, 50)]]
    kept: list[int] = []
    for j in pool:
        # 1) ranking-space dedup (text↔text or audio↔audio)
        if any(float(dedup_mat[j] @ dedup_mat[k_]) >= text_threshold for k_ in kept):
            continue
        # 2) belt-and-braces audio-waveform dedup (Suno multi-take)
        if any(float(audio[j] @ audio[k_]) >= audio_threshold for k_ in kept):
            continue
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
def api_event(ev: EventIn, request: Request, response: Response) -> dict:
    """Log an interaction. user_id in the body must match the caller's
    cookie — prevents one client from polluting another's taste vector by
    spoofing user_id in the request body."""
    cookie_uid = _ensure_uid_cookie(request, response)
    if ev.user_id != cookie_uid:
        raise HTTPException(403, "user_id does not match session cookie")
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


def _recalibrate_zscores() -> None:
    """Recompute per-tag z-score across the current corpus and write z::*
    columns back into tracks.parquet.

    Called after each /api/ingest so newcomers don't show NaN z-scores in
    the UI and so mean/std reflect the latest corpus state. Sub-second on
    200-1000 tracks. Atomically rewrites tracks.parquet via .new tempfile.
    """
    global _audio, _style, _meta, _idx
    tracks_path = INDEX_DIR / "tracks.parquet"
    cal_path = INDEX_DIR / "calibration.json"
    with _INDEX_LOCK:
        df = pd.read_parquet(tracks_path)
        raw_cols = sorted(c for c in df.columns if c.startswith("raw::"))
        if not raw_cols:
            return
        raw_mat = df[raw_cols].to_numpy(dtype=np.float32)
        # Skip rows that are all-NaN (e.g. older tracks ingested before zero-shot)
        valid = ~np.all(np.isnan(raw_mat), axis=1)
        if valid.sum() < 2:
            return
        cal = cal_mod.fit(np.nan_to_num(raw_mat[valid], nan=0.0))
        z = cal.zscore(np.nan_to_num(raw_mat, nan=0.0))
        z_cols = [c.replace("raw::", "z::", 1) for c in raw_cols]
        for i, c in enumerate(z_cols):
            df[c] = z[:, i]
        # Atomic write
        _atomic_replace(tracks_path, lambda p: df.to_parquet(p, index=False))
        # Update calibration.json snapshot for analytics tools (not used at request-time)
        try:
            audio_arr = _audio if _audio is not None else np.load(INDEX_DIR / "audio.npy")
            hist = cal_mod.cosine_histogram(audio_arr)
        except Exception:
            hist = {}
        cal_path.write_text(json.dumps({
            "n_tracks": int(len(df)),
            "n_tags": int(len(raw_cols)),
            "tag_means": cal.mean.tolist(),
            "tag_stds": cal.std.tolist(),
            "tag_columns": raw_cols,
            "pairwise_cosine": hist,
        }, indent=2))
        # Bust meta cache so next /api/tracks reads the updated z::*
        _meta = None
        _idx = None


def _atomic_replace(target: Path, write_fn) -> None:
    """Write to target via a sibling .new tempfile, then os.replace().

    `os.replace` is atomic on POSIX — either the new file is in place or the
    old one is. If write_fn raises mid-flight, the .new tempfile is removed
    and the original is untouched.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".new")
    try:
        write_fn(tmp)
        os.replace(tmp, target)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _append_track(track: dict, audio_vec: np.ndarray,
                   style_vec: np.ndarray | None) -> int:
    """Append one track row + 2 vectors to the on-disk index atomically.

    Strategy: write all three files to .new siblings first, then os.replace
    each (or roll back all .new tempfiles on any error). The cache is
    invalidated only after all three succeed, so concurrent reads either see
    the old consistent state or the new consistent state — never a mix.

    Returns the new row index on success.
    """
    global _audio, _style, _meta, _idx
    audio_path = INDEX_DIR / "audio.npy"
    style_path = INDEX_DIR / "style.npy"
    tracks_path = INDEX_DIR / "tracks.parquet"

    with _INDEX_LOCK:
        audio_existing = np.load(audio_path)
        style_existing = np.load(style_path) if style_path.exists() else None
        df = pd.read_parquet(tracks_path)

        # Sanity: existing files must agree before we extend them.
        n_old = len(df)
        if audio_existing.shape[0] != n_old:
            raise RuntimeError(
                f"refusing to append: audio.npy ({audio_existing.shape[0]}) "
                f"already mismatches parquet ({n_old}). Fix the index first."
            )
        if style_existing is not None and style_existing.shape[0] != n_old:
            raise RuntimeError(
                f"refusing to append: style.npy ({style_existing.shape[0]}) "
                f"already mismatches parquet ({n_old}). Fix the index first."
            )

        # Build the new row schema-compatibly (fill missing columns with None)
        new_row = {c: track.get(c, None) for c in df.columns}
        for k, v in track.items():
            if k not in new_row:
                new_row[k] = v
        df_new = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        audio_new = np.concatenate(
            [audio_existing, audio_vec[None, :]], axis=0
        ).astype(np.float32)

        # H3: always create style.npy on first ingest, even if missing on disk.
        if style_existing is None:
            style_combined = style_vec[None, :].astype(np.float32) if style_vec is not None else None
        elif style_vec is not None:
            style_combined = np.concatenate(
                [style_existing, style_vec[None, :]], axis=0
            ).astype(np.float32)
        else:
            # Existing file but no new vector — pad with zeros to keep shape sync
            zero = np.zeros((1, style_existing.shape[1]), dtype=np.float32)
            style_combined = np.concatenate([style_existing, zero], axis=0).astype(np.float32)

        # All three writes go to .new tempfiles first, then we replace atomically.
        # If any individual replace fails, prior replaces stay (we treat replace
        # as the commit point) — but the temp-file write step is rolled back.
        _atomic_replace(audio_path, lambda p: np.save(p, audio_new))
        if style_combined is not None:
            _atomic_replace(style_path, lambda p: np.save(p, style_combined))
        _atomic_replace(tracks_path, lambda p: df_new.to_parquet(p, index=False))

        # Invalidate cache so next read picks up new files (under the same lock)
        _audio = None
        _style = None
        _meta = None
        _idx = None

        return n_old


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

    # 5. embed style text via BGE-M3 — exact same text construction as
    # scripts/embed_styles.py (build_style_text) so a track ingested live
    # gets the same vector as if it were extracted in bulk.
    style_text = build_style_text(
        style=track.style, title=track.title,
        prompt=track.prompt, lyrics=track.lyrics,
    )
    sty_em = StyleEncoder() if _style_emb is None else _style_emb
    style_vec = sty_em.encode(style_text)

    # 6. librosa numerical features
    try:
        feats = audio_features.extract(str(mp3_path))
    except Exception:
        feats = {}

    # 7. parse Suno tags (parser handles literal \n)
    parsed = parse_suno_tags(track.style or "")
    flat = flat_suno_tags(parsed)

    # 8. zero-shot tags via MuQ-MuLan — same pipeline as scripts/extract.py.
    # Without this, freshly ingested tracks have empty tag::*/raw::*/z::*
    # columns and their UI cards fall back to suno_tags only. With it,
    # all tracks share the same scoring grid.
    raw_per_tag: dict[str, float] = {}
    sigmoid_per_tag: dict[str, float] = {}
    if aud_em.supports_text:
        try:
            zs = zeroshot.score(audio_vec, aud_em)
            flat_pairs = zs["flat"]                  # [(group, tag), ...]
            raw_cos = zs["raw_cos"]                  # (T,) signed cosines
            for (g, tag), val in zip(flat_pairs, raw_cos.tolist()):
                key_safe = f"{g}::{tag.replace(' ', '_').replace('-', '_')}"
                raw_per_tag[f"raw::{key_safe}"] = float(val)
            for g, m in zs["sigmoid"].items():
                for tag, val in m.items():
                    key_safe = f"{g}::{tag.replace(' ', '_').replace('-', '_')}"
                    sigmoid_per_tag[f"tag::{key_safe}"] = float(val)
        except Exception as e:
            print(f"[ingest] zero-shot scoring failed: {e}")

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
    row.update(raw_per_tag)
    row.update(sigmoid_per_tag)

    # 9. append to indices (atomic — see _append_track)
    new_idx = _append_track(row, audio_vec, style_vec)

    # 10. recompute z-score per tag across the (now N+1) corpus + write back.
    # Hubness depends on what's in the corpus, so each new track shifts the
    # mean/std slightly. For 200-1000 tracks this is a sub-second op.
    try:
        _recalibrate_zscores()
    except Exception as e:
        print(f"[ingest] z-score recalibration failed: {e}")

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
