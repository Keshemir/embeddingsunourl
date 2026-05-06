"""Append-only event log — JSONL primary, parquet snapshot on demand.

Why JSONL: read_parquet → concat → to_parquet is two full file rewrites for
every event, and worse, two concurrent POSTs both load → both append → both
write → one write wins → events lost (audit reproduced 28→8). Race-free for
single-process append: open(path, 'a') with `fcntl.flock(LOCK_EX)` over the
write block guarantees serialization across goroutines / threads / processes.

JSONL is also append-O(1), durable on partial writes (line-oriented), and
trivially diff-able. Parquet is rebuilt lazily by `load()` (and saved as a
side effect for tools that want columnar reads).

Schema (one JSON object per line):
    event_id        sha1 hex (idempotency hint, not enforced)
    user_id         opaque string (set by /api/event after auth check)
    track_id        from tracks.parquet
    action          play | complete | skip | like | dislike | save | share | unlike
    ts              unix seconds (float)
    session_id      stable per browser tab (frontend assigns)
    completion_pct  [0, 1] for play/complete/skip; null otherwise
    source          feed | similar | search | manual
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from config import DATA_DIR

# Primary durable store — append-only JSONL. flock guarantees no torn writes.
LOG_PATH = DATA_DIR / "events.jsonl"
# Optional columnar snapshot (rebuilt lazily). Kept for legacy parquet readers.
EVENTS_PATH = DATA_DIR / "events.parquet"

VALID_ACTIONS = {"play", "complete", "skip", "like", "unlike",
                 "dislike", "save", "share"}


def _eid(user_id: str, track_id: str, action: str, ts: float) -> str:
    return hashlib.sha1(
        f"{user_id}|{track_id}|{action}|{ts:.3f}".encode("utf-8")
    ).hexdigest()


@contextmanager
def _exclusive_append(path: Path) -> Iterator:
    """Open file in append+text mode under an exclusive flock. flock is
    advisory but every writer in this codebase goes through this context, so
    the contract is honoured."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield f
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def _migrate_legacy_parquet_once() -> None:
    """If only a legacy events.parquet exists, dump its rows into JSONL once.
    Idempotent: the JSONL marker file prevents re-migration."""
    if LOG_PATH.exists():
        return
    if not EVENTS_PATH.exists():
        return
    try:
        df = pd.read_parquet(EVENTS_PATH)
    except Exception:
        return
    if df.empty:
        LOG_PATH.touch()
        return
    with _exclusive_append(LOG_PATH) as f:
        for _, r in df.iterrows():
            row = {k: _json_safe(v) for k, v in r.to_dict().items()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _json_safe(v):
    # Pandas may give us numpy scalars + NaN — JSON wants None.
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


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
    """Append one event under an exclusive file lock. Returns the row dict.

    Concurrent callers will serialize on the lock; no events are lost.
    """
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
        "completion_pct": float(completion_pct) if completion_pct is not None else None,
        "source": source,
    }
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _exclusive_append(LOG_PATH) as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    return row


_COLUMNS = ("event_id", "user_id", "track_id", "action", "ts",
            "session_id", "completion_pct", "source")


def load(user_id: str | None = None) -> pd.DataFrame:
    """Read JSONL into a DataFrame, optionally filtered by user. Empty DF if
    no log yet. JSONL parsing is line-oriented so partial writes don't corrupt
    the rest of the file (we just skip unparseable trailing lines)."""
    _migrate_legacy_parquet_once()
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=list(_COLUMNS))

    rows: list[dict] = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        # Shared lock for readers — writers are serialized via LOCK_EX above.
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        except Exception:
            pass
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip a torn line — should not happen with flock, but be
                # defensive against pre-flock legacy data.
                continue
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=list(_COLUMNS))
    df = pd.DataFrame(rows)
    # Ensure stable column set even if old rows have a subset
    for c in _COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[list(_COLUMNS)]
    if user_id is not None:
        df = df[df["user_id"] == user_id]
    return df.reset_index(drop=True)


def snapshot_to_parquet() -> Path:
    """Materialize the JSONL log into a columnar parquet for analytics tools.
    Not used at request-time — call manually or as a cron."""
    df = load()
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EVENTS_PATH, index=False)
    return EVENTS_PATH


def recent_track_ids(user_id: str, within_seconds: float = 24 * 3600) -> set[str]:
    """Tracks the user touched recently — exclude from recs."""
    df = load(user_id)
    if df.empty:
        return set()
    cutoff = time.time() - within_seconds
    return set(df.loc[df["ts"] >= cutoff, "track_id"].tolist())
