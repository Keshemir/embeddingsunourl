"""Backfill `audio_url` (direct CDN mp3) for every URL-sourced track.

The ingest parser extracts the CDN URL today, but earlier extractions only
stored `source` (share-link). Wave mode plays inline through HTML5 <audio>
which needs the direct mp3, so we go back through every share-link and
fill in `audio_url`.

This is *only* metadata work — no embedding, no audio download. ~3 minutes
on 200 tracks.

Usage:
    python -m scripts.backfill_audio_url
    python -m scripts.backfill_audio_url --only-empty   # skip rows that
                                                        # already have audio_url
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INDEX_DIR
from ingest.suno_url import fetch_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--only-empty", action="store_true",
                    help="skip rows that already have a non-empty audio_url")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    idx = Path(args.index_dir)
    df = pd.read_parquet(idx / "tracks.parquet")
    print(f"loaded {len(df)} tracks from {idx}")

    if "audio_url" not in df.columns:
        df["audio_url"] = ""

    updated = 0
    skipped = 0
    errors: list[dict] = []

    for i in tqdm(df.index, desc="audio_url"):
        src = str(df.at[i, "source"])
        if not src.startswith("http"):
            skipped += 1
            continue
        if args.only_empty:
            existing = str(df.at[i, "audio_url"] or "").strip()
            if existing.startswith("http"):
                skipped += 1
                continue
        try:
            m = fetch_meta(src)
        except Exception as e:
            errors.append({"i": int(i), "source": src, "error": f"{type(e).__name__}: {e}"})
            continue
        if m.audio_url:
            df.at[i, "audio_url"] = m.audio_url
            updated += 1
        if args.sleep:
            time.sleep(args.sleep)

    df.to_parquet(idx / "tracks.parquet", index=False)
    n_filled = df["audio_url"].astype(str).str.startswith("http").sum()
    print(f"\nupdated: {updated}, skipped: {skipped}, errors: {len(errors)}")
    print(f"audio_url populated for {n_filled}/{len(df)} tracks")
    if errors:
        for e in errors[:5]:
            print(f"  {e['i']}: {e['error']}  ({e['source']})")
        if len(errors) > 5:
            print(f"  ... +{len(errors) - 5} more")


if __name__ == "__main__":
    main()
