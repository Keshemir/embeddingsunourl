"""Ingest Suno mp3s into a local catalog.

Suno-exported mp3s carry the prompt + style tags in ID3v2 frames (title, artist,
TIT2, COMM, TXXX). When a sidecar JSON sits next to the mp3 (e.g.
`<id>.mp3` + `<id>.json`), we prefer that — it has more reliable fields.

The prompt is not always clean: Suno often stuffs lyrics, gibberish, or "epic
banger fr fr" into the same field. We keep the full string but also extract
genre/style hints into a dedicated `style` field for the text-embedding side.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from mutagen import File as MutagenFile
from mutagen.id3 import ID3NoHeaderError

from config import AUDIO_DIR


@dataclass
class Track:
    track_id: str
    path: str
    title: str = ""
    prompt: str = ""
    style: str = ""
    lyrics: str = ""
    bpm: float | None = None
    key: str = ""

    def text_for_embedding(self) -> str:
        """Combined text view for the text-embedding side."""
        parts = [p for p in (self.style, self.prompt, self.title) if p]
        return " | ".join(parts)


_GENRE_HINT_RE = re.compile(
    r"\b(arabian|arabic|kpop|drill|trap|junky|junkyard|phonk|hyperpop|"
    r"jersey club|amapiano|reggaet[oó]n|afrobeat|"
    r"lo-?fi|boom-?bap|drum.?and.?bass|dnb|jungle|techno|house|hardstyle|"
    r"ambient|cinematic|orchestral|opera|metal|rock|punk|country|folk|jazz|"
    r"\d{2,3}\s*bpm)\b",
    re.IGNORECASE,
)


def _hash_id(p: Path) -> str:
    return hashlib.sha1(p.resolve().as_posix().encode("utf-8")).hexdigest()[:16]


def _read_id3(path: Path) -> dict:
    try:
        f = MutagenFile(path, easy=False)
    except (ID3NoHeaderError, Exception):
        return {}
    if f is None:
        return {}
    out = {}
    tags = getattr(f, "tags", None)
    if tags is None:
        return {}
    # Easy text frames
    for k, frame in tags.items():
        try:
            out[k] = str(frame)
        except Exception:
            pass
    return out


def _extract_style(text: str) -> str:
    if not text:
        return ""
    hits = _GENRE_HINT_RE.findall(text)
    return ", ".join(dict.fromkeys(h.lower() for h in hits))  # dedup, keep order


def _from_sidecar(json_path: Path, mp3_path: Path) -> Track | None:
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    title = meta.get("title") or meta.get("name") or ""
    prompt = (
        meta.get("prompt")
        or meta.get("gpt_description_prompt")
        or meta.get("description")
        or ""
    )
    style = meta.get("tags") or meta.get("style") or _extract_style(prompt)
    if isinstance(style, list):
        style = ", ".join(style)
    return Track(
        track_id=meta.get("id") or _hash_id(mp3_path),
        path=str(mp3_path),
        title=title,
        prompt=prompt,
        style=style,
        lyrics=meta.get("lyric") or meta.get("lyrics") or "",
        bpm=meta.get("bpm"),
        key=meta.get("key", ""),
    )


def _from_id3(path: Path) -> Track:
    tags = _read_id3(path)
    title = tags.get("TIT2") or tags.get("title") or path.stem
    # Suno commonly puts the prompt in COMM or TXXX:description
    prompt = ""
    for k, v in tags.items():
        kl = k.lower()
        if kl.startswith("comm") or "description" in kl or "prompt" in kl:
            if len(v) > len(prompt):
                prompt = v
    style = tags.get("TCON") or _extract_style(prompt) or _extract_style(title)
    return Track(
        track_id=_hash_id(path),
        path=str(path),
        title=title,
        prompt=prompt,
        style=style,
    )


def load_track(mp3_path: str | Path) -> Track:
    """Load a single mp3 + (optional) sidecar JSON into a Track."""
    p = Path(mp3_path)
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        t = _from_sidecar(sidecar, p)
        if t is not None:
            return t
    return _from_id3(p)


def discover_tracks(root: str | Path = AUDIO_DIR) -> list[Track]:
    root = Path(root)
    tracks = []
    for p in sorted(root.rglob("*.mp3")):
        tracks.append(load_track(p))
    return tracks


def to_records(tracks: Iterable[Track]) -> list[dict]:
    return [asdict(t) for t in tracks]
