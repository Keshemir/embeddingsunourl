"""Resolve a Suno share URL to a local mp3 + prompt + metadata.

Suno share pages embed the audio CDN URL and metadata as JSON inside the HTML
(Next.js __NEXT_DATA__, JSON-LD, or og:audio tags). Suno changes their markup
periodically, so we try a few strategies in order of robustness:

    1. JSON-LD <script type="application/ld+json"> — has contentUrl + name +
       description. Stable schema.
    2. __NEXT_DATA__ blob — props.pageProps.clip with the full clip object.
    3. og:audio meta tag — last-resort, gives the mp3 URL only.

If you hit a wall, fall back to manual export: download the mp3 yourself,
drop it in data/audio/, optionally save a sidecar <id>.json with {prompt,
title, tags}, and use ingest.suno.load_track() instead.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import AUDIO_DIR

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 ozenref/0.1"
TIMEOUT = 30
_ID_RE = re.compile(r"suno\.com/(?:song|s)/([a-zA-Z0-9_-]+)", re.IGNORECASE)
_UUID_RE = re.compile(r"suno\.com/song/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.IGNORECASE)
_CDN_MP3_RE = re.compile(r"https?://cdn\d?\.suno\S*?/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.mp3", re.IGNORECASE)


@dataclass
class SunoMeta:
    suno_id: str
    audio_url: str
    title: str = ""
    prompt: str = ""
    tags: str = ""
    lyrics: str = ""


def _suno_id_from_url(url: str) -> str:
    m = _ID_RE.search(url)
    if not m:
        raise ValueError(f"not a Suno share URL: {url}")
    return m.group(1)


def _try_jsonld(soup: BeautifulSoup) -> dict | None:
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            data = next((d for d in data if d.get("@type") in {"MusicRecording", "AudioObject"}), None)
        if data and data.get("contentUrl"):
            return data
    return None


def _try_next_data(soup: BeautifulSoup) -> dict | None:
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        data = json.loads(tag.string)
    except json.JSONDecodeError:
        return None
    # Walk the tree for a "clip"-shaped object
    def walk(o):
        if isinstance(o, dict):
            if "audio_url" in o and isinstance(o["audio_url"], str):
                return o
            for v in o.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = walk(v)
                if r:
                    return r
        return None
    return walk(data)


def _try_og_audio(soup: BeautifulSoup) -> str | None:
    tag = soup.find("meta", property="og:audio")
    return tag["content"] if tag and tag.get("content") else None


def _try_cdn_regex(html: str) -> tuple[str, str] | None:
    """Last-resort: scan raw HTML for any cdn{N}.suno*/<uuid>.mp3 URL.

    Suno's Next.js page embeds the audio URL in inline JSON state that's not
    structured as JSON-LD or __NEXT_DATA__ on share pages — but the CDN URL
    pattern is very stable. Skips the silent placeholder `sil-100.mp3`.

    Returns (audio_url, uuid) or None.
    """
    for m in _CDN_MP3_RE.finditer(html):
        url = m.group(0)
        if "sil-" in url:
            continue
        return url, m.group(1)
    return None


def fetch_meta(url: str) -> SunoMeta:
    """Fetch share page and extract audio URL + metadata. Raises if blocked."""
    suno_id = _suno_id_from_url(url)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    audio_url = title = prompt = tags = lyrics = ""

    if (clip := _try_next_data(soup)):
        audio_url = clip.get("audio_url", "")
        title = clip.get("title", "")
        meta = clip.get("metadata") or {}
        prompt = meta.get("prompt") or meta.get("gpt_description_prompt") or ""
        tags = meta.get("tags") or clip.get("tags") or ""
        lyrics = meta.get("lyric") or meta.get("lyrics") or ""

    if not audio_url and (jl := _try_jsonld(soup)):
        audio_url = jl.get("contentUrl", "")
        title = title or jl.get("name", "")
        prompt = prompt or jl.get("description", "")
        if "genre" in jl:
            tags = tags or (
                ", ".join(jl["genre"]) if isinstance(jl["genre"], list) else str(jl["genre"])
            )

    if not audio_url:
        audio_url = _try_og_audio(soup) or ""

    # Last-resort: regex on raw HTML. This is what works on share pages today
    # because Suno's Next.js client renders the audio URL via JS hydration.
    if not audio_url:
        if hit := _try_cdn_regex(html):
            audio_url, uuid = hit
            # Prefer the UUID from the CDN URL — it's the canonical track id.
            suno_id = uuid

    # Backup: use the UUID from the redirected URL if we still need an id
    if not _UUID_RE.search(suno_id) and (m := _UUID_RE.search(r.url)):
        suno_id = m.group(1)

    # Title fallback from <title>
    if not title:
        if t := soup.find("title"):
            title = t.get_text(strip=True).removesuffix(" | Suno")
    # Description fallback from meta tags
    if not prompt:
        for prop in ("og:description", "description", "twitter:description"):
            tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
            if tag and tag.get("content"):
                prompt = tag["content"]
                break

    if not audio_url:
        raise RuntimeError(
            f"could not extract audio URL from {url} — Suno may have changed their markup; "
            f"download the mp3 manually and use ingest.suno.load_track instead"
        )

    if isinstance(tags, list):
        tags = ", ".join(tags)

    return SunoMeta(
        suno_id=suno_id,
        audio_url=audio_url,
        title=title,
        prompt=prompt,
        tags=tags,
        lyrics=lyrics,
    )


def download_audio(audio_url: str, suno_id: str, dest_dir: Path = AUDIO_DIR) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{suno_id}.mp3"
    if out.exists() and out.stat().st_size > 0:
        return out
    with requests.get(audio_url, headers={"User-Agent": UA}, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    return out


def resolve(url: str, dest_dir: Path = AUDIO_DIR, sleep: float = 0.5) -> tuple[Path, SunoMeta]:
    """Full pipeline: URL → (mp3 path, metadata). Caches mp3 by Suno ID.

    Also writes a sidecar <id>.json so re-ingest from disk preserves prompt/tags.
    """
    meta = fetch_meta(url)
    mp3 = download_audio(meta.audio_url, meta.suno_id, dest_dir)
    sidecar = mp3.with_suffix(".json")
    if not sidecar.exists():
        sidecar.write_text(
            json.dumps(
                {
                    "id": meta.suno_id,
                    "title": meta.title,
                    "prompt": meta.prompt,
                    "tags": meta.tags,
                    "lyrics": meta.lyrics,
                    "source_url": url,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if sleep:
        time.sleep(sleep)  # be polite if iterating over many URLs
    return mp3, meta
