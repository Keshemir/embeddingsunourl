"""Bulk-embed Suno style descriptions for all tracks in the index.

Reads `data/index/tracks.parquet`, takes the `style` column (rich Suno text
that ingest/suno_url.py extracted from the share page), feeds it to BGE-M3
in batches, writes `data/index/style.npy` of shape (N, 1024).

This is the source of truth for /api/search going forward — a text query
("arabian", "drill", "арабский") is encoded with the same BGE-M3 and
cosined against this matrix. Way more reliable than audio cosine on a
single-author Suno corpus.

Usage:
    python -m scripts.embed_styles
    python -m scripts.embed_styles --index-dir data/index_real
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INDEX_DIR
from embed.style import StyleEncoder

PLACEHOLDERS = {
    "",
    "listen and make your own on suno.",
    "listen and make your own on suno",
}


def _useful(t: str) -> str:
    """Pick the best text for embedding: prefer style (rich Suno description),
    fall back to prompt or title for tracks where style is empty.
    """
    return t or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    idx = Path(args.index_dir)
    df = pd.read_parquet(idx / "tracks.parquet")
    print(f"loaded {len(df)} tracks from {idx}")

    # Build the text input for each track. Combine style + title so a track
    # whose Suno description is sparse still has something to encode.
    parts = []
    for _, r in df.iterrows():
        style = str(r.get("style") or "").strip()
        title = str(r.get("title") or "").strip()
        if style.lower() in PLACEHOLDERS:
            style = ""
        text = (style + "\n\n" + title).strip() if title else style
        parts.append(text or title or "music")
    print(f"  text length: median={int(np.median([len(p) for p in parts]))}, "
          f"max={max(len(p) for p in parts)}")

    print(f"loading encoder: {args.model}")
    enc = StyleEncoder(model_id=args.model)
    print(f"  device={enc.device}, dim={enc.dim}")

    print(f"encoding {len(parts)} descriptions...")
    M = enc.encode_batch(parts, batch_size=args.batch_size)
    out_path = idx / "style.npy"
    np.save(out_path, M.astype(np.float32))
    print(f"saved {M.shape} (dtype={M.dtype}) → {out_path}")
    print(f"L2 norms: mean={float(np.linalg.norm(M, axis=1).mean()):.3f}")


if __name__ == "__main__":
    main()
