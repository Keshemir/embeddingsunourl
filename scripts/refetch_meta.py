"""Re-pull Suno share pages for every URL-sourced track in the index, update
metadata fields (title / prompt / tags / lyrics) without re-embedding.

Use after upgrading `ingest/suno_url.py` to extract additional fields — this
gets the new fields into your existing parquet without burning ~30 min on
re-running MuQ on every track.

Audio.npy and all tag::*/raw::*/z::* columns are left alone.

    python -m scripts.refetch_meta
    python -m scripts.refetch_meta --index-dir data/index_real
    python -m scripts.refetch_meta --only-empty   # skip tracks that already
                                                  # have a non-placeholder prompt
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

PLACEHOLDER = "listen and make your own on suno"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--only-empty", action="store_true",
                    help="skip rows that already have rich metadata")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="seconds between requests; be polite")
    args = ap.parse_args()

    idx = Path(args.index_dir)
    df = pd.read_parquet(idx / "tracks.parquet")
    print(f"loaded {len(df)} tracks from {idx}")

    n_url = df["source"].astype(str).str.startswith("http").sum()
    print(f"  {n_url} have URL sources (will re-fetch)")

    updated = 0
    skipped = 0
    errors: list[dict] = []
    for i in tqdm(df.index, desc="refetch"):
        src = str(df.at[i, "source"])
        if not src.startswith("http"):
            skipped += 1
            continue
        existing = str(df.at[i, "prompt"] or "").strip().lower()
        if args.only_empty and existing and PLACEHOLDER not in existing and len(existing) > 50:
            skipped += 1
            continue
        try:
            m = fetch_meta(src)
        except Exception as e:
            errors.append({"i": int(i), "source": src, "error": f"{type(e).__name__}: {e}"})
            continue
        if m.title:    df.at[i, "title"] = m.title
        if m.prompt:   df.at[i, "prompt"] = m.prompt
        if m.tags:     df.at[i, "style"] = m.tags
        if m.lyrics:   df.at[i, "lyrics"] = m.lyrics
        updated += 1
        if args.sleep:
            time.sleep(args.sleep)

    df.to_parquet(idx / "tracks.parquet", index=False)
    print(f"\nupdated: {updated}, skipped: {skipped}, errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"  {e['i']}: {e['error']}  ({e['source']})")
        if len(errors) > 5:
            print(f"  ... +{len(errors) - 5} more")
    # Quick summary of what we got
    df_url = df[df["source"].astype(str).str.startswith("http")]
    rich = df_url["style"].astype(str).str.len() > 100
    print(f"\nstyle field length: median={int(df_url['style'].astype(str).str.len().median())}, "
          f"with rich text: {rich.sum()}/{len(df_url)}")


if __name__ == "__main__":
    main()
