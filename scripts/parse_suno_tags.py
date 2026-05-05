"""Pre-parse Suno style descriptions into structured tags for every track.

Reads `data/index/tracks.parquet`, runs `features.suno_tags.parse` on the
`style` column, and writes the result back as a JSON-encoded `suno_tags`
column. Cards in the UI then render these instead of zero-shot z-score
guesses — they're ground truth from Suno itself.

Usage:
    python -m scripts.parse_suno_tags
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INDEX_DIR
from features.suno_tags import parse, flat_tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    args = ap.parse_args()

    idx = Path(args.index_dir)
    df = pd.read_parquet(idx / "tracks.parquet")
    print(f"loaded {len(df)} tracks from {idx}")

    structured: list[str] = []
    flat: list[str] = []
    n_genre = 0
    n_inst = 0
    for s in df["style"].fillna("").astype(str).tolist():
        parsed = parse(s)
        structured.append(json.dumps(parsed, ensure_ascii=False))
        flat.append(json.dumps(flat_tags(parsed), ensure_ascii=False))
        if parsed.get("genre"): n_genre += 1
        if parsed.get("instrument"): n_inst += 1

    df["suno_tags"] = structured       # full per-group dict, JSON
    df["suno_tags_flat"] = flat        # flattened list for UI, JSON
    df.to_parquet(idx / "tracks.parquet", index=False)

    print(f"  tracks with genre tags:      {n_genre}/{len(df)}")
    print(f"  tracks with instrument tags: {n_inst}/{len(df)}")
    print(f"saved suno_tags + suno_tags_flat → tracks.parquet")
    print()
    print("=== Sample output (first 5 tracks) ===")
    for i in range(min(5, len(df))):
        title = df.iloc[i]["title"][:35]
        print(f"  {title}")
        print(f"    {df.iloc[i]['suno_tags']}")


if __name__ == "__main__":
    main()
