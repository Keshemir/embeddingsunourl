"""Append-only event log.

One file, one DataFrame, one append per event. parquet because pandas already
loaded; if write contention shows up, switch to sqlite or a JSON-lines log.

Schema:
    event_id        sha1 hex (idempotency hint, but not enforced)
    user_id         opaque string (hardcoded today, cookie/auth tomorrow)
    track_id        from tracks.parquet
    action          play | complete | skip | like | dislike | save | share | unlike
    ts              unix seconds (float)
    session_id      stable per browser tab (frontend assigns)
    completion_pct  [0, 1] for play/complete/skip; NaN otherwise
    source          feed | similar | search | manual
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import DATA_DIR

EVENTS_PATH = DATA_DIR / "events.parquet"

VALID_ACTIONS = {"play", "complete", "skip", "like", "unlike",
                 "dislike", "save", "share"}


def _eid(user_id: str, track_id: str, action: str, ts: float) -> str:
    return hashlib.sha1(
        f"{user_id}|{track_id}|{action}|{ts:.3f}".encode("utf-8")
    ).hexdigest()


def append(
    user_id: str,
    track_id: str,
    action: str,
    *,
    ts: float | None = None,
    session_id: str = "",
    completion_pct: float | None = None,
    source: str = "manual",
) -> dict:
    if action not in VALID_ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    ts = ts if ts is not None else time.time()
    row = {
        "event_id": _eid(user_id, track_id, action, ts),
        "user_id": user_id,
        "track_id": track_id,
        "action": action,
        "ts": float(ts),
        "session_id": session_id,
        "completion_pct": float("nan") if completion_pct is None else float(completion_pct),
        "source": source,
    }
    df_new = pd.DataFrame([row])
    if EVENTS_PATH.exists():
        df = pd.read_parquet(EVENTS_PATH)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(EVENTS_PATH, index=False)
    return row


def load(user_id: str | None = None) -> pd.DataFrame:
    """All events, optionally filtered by user. Empty DF if no log yet."""
    if not EVENTS_PATH.exists():
        return pd.DataFrame(columns=[
            "event_id", "user_id", "track_id", "action", "ts",
            "session_id", "completion_pct", "source",
        ])
    df = pd.read_parquet(EVENTS_PATH)
    if user_id is not None:
        df = df[df["user_id"] == user_id]
    return df.reset_index(drop=True)


def recent_track_ids(user_id: str, within_seconds: float = 24 * 3600) -> set[str]:
    """Tracks the user touched recently — exclude from recs."""
    df = load(user_id)
    if df.empty:
        return set()
    cutoff = time.time() - within_seconds
    return set(df.loc[df["ts"] >= cutoff, "track_id"].tolist())
