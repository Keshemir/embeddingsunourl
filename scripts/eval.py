"""Synthetic LOO evaluation: how well does the embedding's text encoder
recover the *original* track from its own Suno prompt?

This is the closest thing to ground truth we can build without users.
The recipe (NEWAVE / TalkPlay / standard text-to-music retrieval eval):

    for each track i with non-empty prompt p_i:
        q   = MuQ.embed_text(p_i)            # text query
        sims = audio_vecs @ q                # cosine to all tracks
        rank = (sims > sims[i]).sum() + 1    # rank of true track
    Recall@K = mean( rank <= K )
    MRR      = mean( 1 / rank )
    rank-1   = % of tracks recovered at top-1

Reports are unitless probabilities — use them as A/B numbers when changing
the embedding model, prompt ensembling, or any other component of the
pipeline. They're not absolute quality scores.

Tracks with empty prompt or prompt == placeholder ("Listen and make your own
on Suno.") are skipped because they carry no signal.

Usage:
    python -m scripts.eval                          # uses data/index/
    python -m scripts.eval --index-dir data/index_real
    python -m scripts.eval --paraphrase 3           # additionally test
                                                    # robustness on LLM-paraphrased prompts (NYI hook)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INDEX_DIR
from embed.audio import AudioEmbedder

PLACEHOLDERS = {
    "",
    "listen and make your own on suno.",
    "listen and make your own on suno",
}


def _is_useful_prompt(p: str) -> bool:
    if not isinstance(p, str):
        return False
    return p.strip().lower() not in PLACEHOLDERS and len(p.strip()) >= 5


def _ranks(vecs: np.ndarray, prompts: list[str], em: AudioEmbedder,
           group_ids: np.ndarray | None = None) -> np.ndarray:
    """Returns per-track rank when querying with its own prompt.

    If `group_ids` is provided, members of the same group (e.g. multiple
    takes of the same Suno prompt) are treated as the same item — rank is
    the position of the FIRST same-group hit. This is the right metric on
    AI-generated corpora where the same prompt produces many variants.

    -1 means the prompt was unusable.
    """
    ranks = np.full(len(vecs), -1, dtype=np.int32)
    for i, p in enumerate(prompts):
        if not _is_useful_prompt(p):
            continue
        q = em.embed_text(p)
        sims = vecs @ q
        if group_ids is not None:
            # First same-group hit (excluding self) — but self is allowed if
            # it's the highest in its group, that's fine.
            order = np.argsort(-sims)
            same = group_ids[order] == group_ids[i]
            same[order.tolist().index(i)] = True  # self counts
            rank = int(np.argmax(same)) + 1
        else:
            rank = int((sims > sims[i]).sum() + 1)
        ranks[i] = rank
    return ranks


def _report(ranks: np.ndarray, label: str) -> None:
    used = ranks[ranks > 0]
    total = len(ranks)
    if not len(used):
        print(f"\n=== {label} ===\n  no usable prompts (all empty/placeholder)")
        return
    print(f"\n=== {label} ===")
    print(f"  evaluated {len(used)}/{total} tracks")
    for k in (1, 5, 10, 25):
        if k > total:
            continue
        print(f"  Recall@{k:<3d}: {(used <= k).mean()*100:5.1f}%")
    print(f"  MRR       : {(1.0 / used).mean():.3f}")
    if len(used) >= 2:
        print(f"  rank dist : median={int(np.median(used))}, "
              f"P25={int(np.percentile(used, 25))}, P75={int(np.percentile(used, 75))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--prompt-col", default="prompt",
                    help="column to use as the text query (prompt | title | style)")
    args = ap.parse_args()

    idx = Path(args.index_dir)
    vecs = np.load(idx / "audio.npy")
    meta = pd.read_parquet(idx / "tracks.parquet")
    print(f"loaded {len(meta)} tracks from {idx}")

    if args.prompt_col not in meta.columns:
        print(f"no column {args.prompt_col!r} in parquet")
        sys.exit(1)

    em = AudioEmbedder.load()
    if not em.supports_text:
        print(f"[err] {type(em).__name__} has no text encoder — eval needs MuQ")
        sys.exit(1)

    # Group-aware: tracks with the same title are takes of the same prompt
    # in Suno. Treat them as one logical item — Recall@K = "did we surface
    # any track of the right group in the top K".
    titles = meta["title"].fillna("").tolist()
    title_to_id = {t: i for i, t in enumerate(sorted(set(titles)))}
    group_ids = np.array([title_to_id[t] for t in titles])
    n_groups = len(title_to_id)
    print(f"  {n_groups} unique title groups across {len(meta)} tracks "
          f"(avg {len(meta)/n_groups:.1f} takes/group)")

    prompts = meta[args.prompt_col].fillna("").tolist()
    ranks = _ranks(vecs, prompts, em, group_ids)
    _report(ranks, f"prompt → group recall (col={args.prompt_col})")

    if "title" in meta.columns and args.prompt_col != "title":
        ranks_title = _ranks(vecs, titles, em, group_ids)
        _report(ranks_title, "title → group recall (group-aware)")
        ranks_strict = _ranks(vecs, titles, em, None)
        _report(ranks_strict, "title → exact-track recall (strict, ignore dups)")


if __name__ == "__main__":
    main()
