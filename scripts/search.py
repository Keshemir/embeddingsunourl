"""Search the index — same pipeline as the HTTP /api/search endpoint.

Stage 1 — semantic retrieval:
   * For --text queries we embed via BGE-M3 (multilingual) and cosine
     against `style.npy` (Suno description embeddings). This is the
     ground-truth text↔text path. If style.npy is absent, we fall back
     to MuQ-MuLan audio↔text — but you should rebuild the style index
     (`python -m scripts.embed_styles`) instead.
   * For --track queries we cosine the seed track's `audio.npy` row
     against the rest of the corpus (audio↔audio).

Stage 2 — boolean tag filters (--no-vocals, --min-tag, --min-z),
near-duplicate dedup, MMR diversity rerank, optional hard cutoff
against the corpus's pairwise-cosine percentile so a no-match query
returns "nothing" instead of random.

Score is never bent by filters — filter is yes/no, rank is cosine.

Examples:
    python -m scripts.search --text "chill late-night drive"
    python -m scripts.search --track <track_id>
    python -m scripts.search --text "study music" --no-vocals
    python -m scripts.search --text "chill" --min-z genre::lo_fi_hip_hop=1.5
    python -m scripts.search --text "drill" --no-mmr        # disable diversity rerank
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INDEX_DIR
from embed.audio import AudioEmbedder
from embed.style import StyleEncoder

DEDUP_THRESHOLD = 0.92      # cosine >= this → near-duplicates (collapse)
DEFAULT_MMR_LAMBDA = 0.65   # 1.0 = pure relevance, 0.0 = pure diversity
DEFAULT_POOL = 50           # candidate pool before MMR / dedup


def _parse_kv_list(specs: list[str], col_prefix: str) -> list[tuple[str, float]]:
    """`genre::drill=0.2 mood::dark=1.5` → [('<prefix>::genre::drill', 0.2), ...]"""
    out = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"bad filter: {s!r}, expected group::name=value")
        key, val = s.split("=", 1)
        out.append((f"{col_prefix}::" + key, float(val)))
    return out


def _dedup(top: list[int], vecs: np.ndarray, threshold: float) -> tuple[list[int], list[list[int]]]:
    """Single-link cluster: collapse near-duplicates into the highest-ranked rep.
    Returns (kept_indices, dropped_clusters_per_kept)."""
    kept: list[int] = []
    dropped: list[list[int]] = []
    for i in top:
        merged = False
        for j_idx, k in enumerate(kept):
            if float(vecs[i] @ vecs[k]) >= threshold:
                dropped[j_idx].append(i)
                merged = True
                break
        if not merged:
            kept.append(i)
            dropped.append([])
    return kept, dropped


def _mmr(candidates: list[int], sims: np.ndarray, vecs: np.ndarray,
         k: int, lam: float) -> list[int]:
    """Carbonell-Goldstein MMR re-rank."""
    selected: list[int] = []
    remaining = list(candidates)
    while len(selected) < k and remaining:
        if not selected:
            i = max(remaining, key=lambda c: sims[c])
        else:
            def mmr_score(c: int) -> float:
                rel = float(sims[c])
                div = max(float(vecs[c] @ vecs[s]) for s in selected)
                return lam * rel - (1 - lam) * div
            i = max(remaining, key=mmr_score)
        selected.append(i)
        remaining.remove(i)
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="natural-language style query")
    ap.add_argument("--track", help="seed track_id")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL,
                    help="candidate pool size before re-ranking")
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--no-vocals", action="store_true",
                    help="filter to instrumental tracks")
    ap.add_argument("--min-tag", action="append", default=[],
                    help="filter on sigmoid score, e.g. genre::drill=0.5 (repeatable)")
    ap.add_argument("--min-z", action="append", default=[],
                    help="filter on z-score, e.g. mood::dark=1.5 (repeatable)")
    ap.add_argument("--no-mmr", action="store_true", help="skip MMR diversity re-rank")
    ap.add_argument("--mmr-lambda", type=float, default=DEFAULT_MMR_LAMBDA)
    ap.add_argument("--no-dedup", action="store_true", help="skip near-duplicate collapse")
    ap.add_argument("--dedup-threshold", type=float, default=DEDUP_THRESHOLD)
    ap.add_argument("--cutoff-percentile", type=float, default=None,
                    help="reject results with cosine below the corpus's P-th pairwise percentile")
    args = ap.parse_args()

    if not args.text and not args.track:
        ap.error("provide --text or --track")

    idx = Path(args.index_dir)
    audio = np.load(idx / "audio.npy")
    style_path = idx / "style.npy"
    style = np.load(style_path) if style_path.exists() else None
    meta = pd.read_parquet(idx / "tracks.parquet")
    print(f"loaded {len(meta)} tracks  audio_dim={audio.shape[1]}"
          + (f" style_dim={style.shape[1]}" if style is not None else "  style.npy=missing"))

    # ---- Stage 1: semantic ranking ----
    # text query → BGE-M3 cosine on style.npy (same as HTTP /api/search)
    # seed track → audio cosine on audio.npy (track-to-track similarity)
    if args.text:
        if style is not None:
            enc = StyleEncoder()
            q = enc.encode(args.text)
            vecs = style
            print(f"[search] text mode → BGE-M3 vs style.npy ({style.shape[1]}-d)")
        else:
            em = AudioEmbedder.load()
            if not em.supports_text:
                print(f"[warn] {type(em).__name__} can't embed text and style.npy is missing — "
                      f"run `python -m scripts.embed_styles`")
                return
            q = em.embed_text(args.text)
            vecs = audio
            print(f"[search] text mode → MuQ-MuLan fallback (no style.npy). "
                  f"Run scripts.embed_styles for better results.")
    else:
        pos = meta.index[meta["track_id"] == args.track]
        if len(pos) == 0:
            print(f"track_id {args.track!r} not found")
            return
        q = audio[int(pos[0])]
        vecs = audio  # seed→audio cosine

    sims = (vecs @ q).astype(np.float32)
    if args.track:
        sims[meta.index[meta["track_id"] == args.track]] = -np.inf

    # ---- Stage 2a: boolean filters ----
    mask = np.ones(len(meta), dtype=bool)

    if args.no_vocals:
        col = "tag::vocal::instrumental_track_without_vocals"
        if col in meta.columns:
            mask &= meta[col].to_numpy() > 0.5
            print(f"[filter] no-vocals (sigmoid>0.5): {mask.sum()}/{len(meta)} pass")

    for col, threshold in _parse_kv_list(args.min_tag, "tag"):
        if col not in meta.columns:
            print(f"[warn] {col!r} not in index — skipping filter"); continue
        mask &= meta[col].to_numpy() > threshold
        print(f"[filter] {col} > {threshold}: {mask.sum()}/{len(meta)} pass")
    for col, threshold in _parse_kv_list(args.min_z, "z"):
        if col not in meta.columns:
            print(f"[warn] {col!r} not in index — skipping filter (run extract on a multi-track corpus first)")
            continue
        mask &= meta[col].to_numpy() > threshold
        print(f"[filter] {col} z > {threshold}: {mask.sum()}/{len(meta)} pass")

    sims_eff = np.where(mask, sims, -np.inf)
    pool_size = max(args.pool, args.k)
    pool_idx = [int(i) for i in np.argsort(-sims_eff)[:pool_size]
                if sims_eff[i] != -np.inf]
    if not pool_idx:
        print("no results after filters")
        return

    # ---- Stage 2b: hard cutoff against corpus pairwise cosine percentile ----
    if args.cutoff_percentile is not None:
        cal_path = idx / "calibration.json"
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            pct = cal.get("pairwise_cosine", {}).get("percentiles", {})
            cutoff = pct.get(int(args.cutoff_percentile)) or pct.get(str(int(args.cutoff_percentile)))
            if cutoff is not None:
                pool_idx = [i for i in pool_idx if sims[i] >= cutoff]
                print(f"[cutoff] keep cosine >= P{int(args.cutoff_percentile)}={cutoff:.3f}: {len(pool_idx)}")
            else:
                print(f"[cutoff] percentile {args.cutoff_percentile} not in calibration.json — skip")
        else:
            print(f"[cutoff] no calibration.json — skip")

    if not pool_idx:
        print("no results after cutoff")
        return

    # ---- Stage 2c: dedup ----
    if not args.no_dedup:
        # Pick a sensible threshold for the space we ranked in. BGE-M3 cosines
        # run higher than MuQ audio cosines, so we use 0.97 for text↔text and
        # 0.92 for audio↔audio unless the user overrode it.
        threshold = args.dedup_threshold
        if threshold == DEDUP_THRESHOLD and vecs is style:
            threshold = 0.97
        kept, clusters = _dedup(pool_idx, vecs, threshold)
        n_dropped = sum(len(c) for c in clusters)
        if n_dropped:
            print(f"[dedup] collapsed {n_dropped} near-duplicates (cos>={threshold})")
        pool_idx = kept

    # ---- Stage 2d: MMR rerank for diversity ----
    if not args.no_mmr and len(pool_idx) > args.k:
        top = _mmr(pool_idx, sims, vecs, args.k, args.mmr_lambda)
    else:
        top = pool_idx[:args.k]

    cols = [c for c in ("track_id", "title", "best::genre", "best::mood",
                        "best::instrument", "duration_sec",
                        "bpm_perceived", "key") if c in meta.columns]
    out = meta.iloc[top][cols].copy()
    out.insert(0, "sim", sims[top].round(3))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
