"""End-to-end feature extraction pipeline.

Input:
    - a Suno share URL,
    - a path to a local mp3, or
    - a file with one URL/path per line (--batch).

Output (under data/index/):
    audio.npy        — (N, audio_dim) MuQ-MuLan vectors, L2-normalized
    tracks.parquet   — one row per track, with metadata + librosa features +
                       zero-shot tag scores per group, all in one flat table.

Designed for downstream recsys: a single parquet you can join, filter, train
on, or feed into a vector DB later.

Usage:
    python -m scripts.extract https://suno.com/song/<id>
    python -m scripts.extract /path/to/track.mp3
    python -m scripts.extract --batch urls.txt
    python -m scripts.extract --scan-audio-dir       # scan data/audio/*.mp3
    python -m scripts.extract --no-zeroshot          # skip tag scoring
    python -m scripts.extract --no-features          # skip librosa features
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import AUDIO_DIR, INDEX_DIR
from embed.audio import AudioEmbedder
from features import audio_features, zeroshot
from ingest.suno import load_track
from ingest.suno_url import resolve as resolve_suno_url


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _resolve_to_track(source: str) -> tuple:
    """source → (Track, error_str_or_None, was_downloaded_from_url)."""
    try:
        if _is_url(source):
            mp3, sm = resolve_suno_url(source)
            track = load_track(mp3)
            # URL-derived metadata wins
            if sm.suno_id:    track.track_id = sm.suno_id
            if sm.title:      track.title = sm.title
            if sm.prompt:     track.prompt = sm.prompt
            if sm.tags:       track.style = sm.tags
            if sm.lyrics:     track.lyrics = sm.lyrics
            track.path = str(mp3)
            return track, None, True
        track = load_track(source)
        return track, None, False
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", False


def _cleanup(mp3_path: str) -> None:
    """Delete a downloaded mp3 and its sidecar JSON. All metadata is already
    captured in the parquet row by the time we call this."""
    p = Path(mp3_path)
    p.unlink(missing_ok=True)
    p.with_suffix(".json").unlink(missing_ok=True)


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def _flatten_sigmoid_tags(tag_sigmoid: dict) -> dict:
    """`tag::group::name` carries the sigmoid score in [0, 1].

    Sigmoid is the right choice for multi-label tagging — same-group tags
    aren't mutually exclusive ("dark" + "energetic" are both possible).
    """
    out = {}
    for group, tag_to_s in tag_sigmoid.items():
        for tag, s in tag_to_s.items():
            out[f"tag::{group}::{_safe(tag)}"] = float(s)
    return out


def _flatten_raw_cosines(flat_pairs: list[tuple[str, str]], raw_cos: np.ndarray) -> dict:
    """`raw::group::name` carries the signed cosine — needed for re-calibration."""
    return {f"raw::{g}::{_safe(t)}": float(raw_cos[i])
            for i, (g, t) in enumerate(flat_pairs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="Suno URL or mp3 path")
    ap.add_argument("--batch", help="file with one URL/path per line")
    ap.add_argument("--scan-audio-dir", action="store_true",
                    help=f"scan {AUDIO_DIR} for *.mp3 and embed all of them")
    ap.add_argument("--out-dir", default=str(INDEX_DIR))
    ap.add_argument("--model", default=None, help="muq | mert (overrides env)")
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--keep-audio", action="store_true",
                    help="keep downloaded mp3s on disk (default: delete after embed, only the URL is kept in parquet)")
    args = ap.parse_args()

    # ------------------------- gather sources ------------------------------
    sources: list[str] = []
    if args.scan_audio_dir:
        sources = [str(p) for p in sorted(Path(AUDIO_DIR).rglob("*.mp3"))]
    elif args.batch:
        sources = [
            ln.strip() for ln in Path(args.batch).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    elif args.source:
        sources = [args.source]
    else:
        ap.error("provide a source, --batch, or --scan-audio-dir")
    if not sources:
        print("no sources")
        return
    print(f"sources: {len(sources)}")

    # ------------------------- load embedder -------------------------------
    em = AudioEmbedder.load(args.model)
    print(f"audio model: {type(em).__name__} dim={em.dim} device={em.device}")
    if not args.no_zeroshot and not em.supports_text:
        print("[warn] selected model has no text encoder → disabling zero-shot")
        args.no_zeroshot = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------- main loop -----------------------------------
    rows: list[dict] = []
    vecs: list[np.ndarray] = []
    failures: list[dict] = []

    for src in tqdm(sources, desc="extract"):
        track, err, was_downloaded = _resolve_to_track(src)
        if err:
            failures.append({"source": src, "error": err})
            continue

        try:
            audio_vec = em.embed_audio(track.path)
        except Exception as e:
            failures.append({"source": src, "error": f"audio_embed: {e}"})
            if was_downloaded and not args.keep_audio:
                _cleanup(track.path)
            continue

        row: dict = {
            "track_id": track.track_id,
            "source": src,
            # for URL inputs we drop the local mp3 below; keep `path` empty so
            # downstream code knows the audio is gone but the URL is preserved
            "path": "" if (was_downloaded and not args.keep_audio) else track.path,
            "title": track.title,
            "prompt": track.prompt,
            "style": track.style,
            "lyrics": track.lyrics,
            "bpm_meta": track.bpm,
            "key_meta": track.key,
        }

        if not args.no_features:
            try:
                feats = audio_features.extract(track.path)
                row.update(feats)
            except Exception as e:
                row["features_error"] = str(e)

        if not args.no_zeroshot:
            try:
                tags = zeroshot.score(audio_vec, em)
                row.update(_flatten_sigmoid_tags(tags["sigmoid"]))
                row.update(_flatten_raw_cosines(tags["flat"], tags["raw_cos"]))
                for g, t in tags["best_per_group"].items():
                    row[f"best::{g}"] = t
            except Exception as e:
                row["zeroshot_error"] = str(e)

        rows.append(row)
        vecs.append(audio_vec)

        # mp3 (and its sidecar) are no longer needed once features + embedding
        # are computed — drop them by default for URL-sourced tracks
        if was_downloaded and not args.keep_audio:
            _cleanup(track.path)

    # ------------------------- persist -------------------------------------
    if not rows:
        print("nothing extracted")
    else:
        df = pd.DataFrame(rows)
        audio_mat = np.stack(vecs).astype(np.float32)
        np.save(out_dir / "audio.npy", audio_mat)

        # ----- corpus calibration on raw cosines ----------------------------
        from features import calibrate as cal_mod  # late import keeps deps quiet
        raw_cols = sorted(c for c in df.columns if c.startswith("raw::"))
        if raw_cols and len(df) >= 2:
            raw_mat = df[raw_cols].to_numpy(dtype=np.float32)
            calibrator = cal_mod.fit(raw_mat)
            z = calibrator.zscore(raw_mat)  # (N, T) hubness-corrected
            z_df = pd.DataFrame(
                z, columns=[c.replace("raw::", "z::", 1) for c in raw_cols],
                index=df.index,
            )
            df = pd.concat([df, z_df], axis=1)

            # Pairwise audio cosine histogram → cutoff for "no-match" gating
            hist = cal_mod.cosine_histogram(audio_mat)
            (out_dir / "calibration.json").write_text(json.dumps({
                "n_tracks": int(len(df)),
                "n_tags": int(len(raw_cols)),
                "tag_means": calibrator.mean.tolist(),
                "tag_stds": calibrator.std.tolist(),
                "tag_columns": raw_cols,
                "pairwise_cosine": hist,
            }, indent=2))
            print(f"calibration: tag-mean={calibrator.mean.mean():.3f} "
                  f"pairwise-cosine p95={hist.get('percentiles',{}).get(95, 0):.3f}")

        df.to_parquet(out_dir / "tracks.parquet", index=False)
        print(f"saved {len(df)} rows  →  {out_dir/'tracks.parquet'}")
        print(f"saved {len(vecs)} vectors  →  {out_dir/'audio.npy'}")
        n_tag = sum(c.startswith('tag::') for c in df.columns)
        n_raw = sum(c.startswith('raw::') for c in df.columns)
        n_z = sum(c.startswith('z::') for c in df.columns)
        print(f"columns: {len(df.columns)}  (sigmoid={n_tag}, raw_cos={n_raw}, z-score={n_z})")

    if failures:
        fpath = out_dir / "failures.json"
        fpath.write_text(json.dumps(failures, ensure_ascii=False, indent=2))
        print(f"failures: {len(failures)}  →  {fpath}")


if __name__ == "__main__":
    main()
