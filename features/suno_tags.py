"""Extract structured tags from Suno's free-form style descriptions.

Suno serves two formats interchangeably:

1. Structured ("Genre: X / Y / Z. Mood: A, B. Instruments: oud, ney flute,
   808 bass. Vocal: female soft singing."). When present, just split.

2. Free-form ("Ambient downtempo with slow breathing pads, soft granular
   bells, and a gently pulsing sub bed..."). No labels — extract by
   substring match against a hand-curated vocabulary.

Output: a dict of {group: [tag, ...]}, plus a flat list for UI cards.
The flat list uses Suno's own wording — these are ground-truth labels
authored by the generator, not zero-shot guesses.
"""
from __future__ import annotations

import re
from typing import Iterable

# ----- structured-format regexes --------------------------------------------
_FIELDS = {
    "genre":      re.compile(r"\b(?:Genre|Style)s?\s*:\s*([^.\n]+)", re.IGNORECASE),
    "mood":       re.compile(r"\bMood(?:s|/Tone)?\s*:\s*([^.\n]+)", re.IGNORECASE),
    "instrument": re.compile(r"\bInstruments?(?:ation)?\s*:\s*([^.\n]+)", re.IGNORECASE),
    "vocal":      re.compile(r"\b(?:Vocals?|Singer'?s?\s*Voice)\s*:\s*([^.\n]+)", re.IGNORECASE),
    "tempo":      re.compile(r"\bTempo\s*:\s*([^.\n]+)", re.IGNORECASE),
}

# Genres often have separators that aren't commas: "/" or " and "
_GENRE_SPLIT = re.compile(r"\s*(?:[,/]|\s+and\s+)\s*", re.IGNORECASE)
_LIST_SPLIT  = re.compile(r"\s*[,;/]\s*")

# ----- vocabulary for free-form fallback ------------------------------------
# Lowercase-only, matched as word-boundary substrings on the description.
# Order matters loosely: longer phrases first so "lo-fi hip hop" matches before
# bare "hip hop".
_VOCAB_GENRE = [
    "uk drill", "ny drill", "afro drill", "indian drill", "latin drill", "anime drill", "lofi drill",
    "drift phonk", "memphis phonk", "russian phonk", "brazilian phonk", "arabian phonk",
    "kpop drill", "arabian junky drill",
    "lo-fi hip hop", "lofi hip hop", "boom bap", "boom-bap",
    "cloud rap", "trap soul", "abstract hip-hop", "hip hop", "hip-hop",
    "russian hip hop", "russian rap", "russian pop", "russian pop-rock", "russian club",
    "drill", "trap", "phonk", "hyperpop", "rage", "plugg",
    "kpop", "k-pop", "j-pop", "anime pop",
    "arabic ambient", "arabic music", "arabian", "arabic", "middle eastern", "oriental",
    "turkish trap", "turkish pop", "balkan",
    "afrobeat", "afroswing", "amapiano",
    "reggaeton", "latin trap", "cumbia",
    "jersey club", "baltimore club",
    "drum and bass", "drum n bass", "dnb", "jungle", "breakcore",
    "deep house", "tech house", "slap house", "tribal house", "hardstyle", "trance",
    "future bass", "melodic dubstep", "dubstep",
    "techno", "house", "ambient", "dark ambient", "drone",
    "chillhop", "chill hop", "chillwave", "synthwave", "vaporwave",
    "cinematic", "orchestral", "epic trailer",
    "industrial metal", "alt rock", "alt-rock", "metal", "rock", "punk",
    "country", "folk", "indie folk",
    "jazz", "lo-fi jazz", "soul", "funk", "disco",
    "indian classical", "bollywood",
    "kazakh folk", "central asian",
    "pop r&b", "r&b",
    "pop-rock", "pop rock", "pop",
]

_VOCAB_INSTRUMENT = [
    "808 bass", "808", "sub bass", "synth bass",
    "trap hi-hats", "rolling hi-hats", "hi-hats", "hi-hat",
    "snare", "snares", "claps", "rim shot", "rim",
    "kick drum", "kick", "hand drums", "tabla", "djembe", "darbuka", "darbouka",
    "male vocal", "female vocal", "rap vocal", "auto-tuned vocal",
    "choir", "gospel choir", "operatic vocal",
    "violin", "strings", "cello", "viola",
    "rhodes", "felt piano", "piano", "organ",
    "acoustic guitar", "electric guitar", "distorted guitar", "spanish guitar", "guitar",
    "synth lead", "synth pad", "supersaw", "analog synth", "synth", "pads", "pad",
    "saxophone", "sax", "trumpet", "brass",
    "flute", "ney flute", "ney", "duduk", "shakuhachi",
    "oud", "sitar", "kalimba", "mbira", "kora",
    "accordion", "harmonica", "bagpipes",
    "vinyl crackle", "tape hiss",
]

_VOCAB_MOOD = [
    "dark", "bright", "moody", "happy", "sad", "melancholic", "melancholy",
    "energetic", "chill", "relaxed", "calm", "aggressive", "violent", "punchy",
    "romantic", "sensual", "mystical", "spiritual",
    "hypnotic", "dreamy", "nostalgic", "epic", "triumphant",
    "tense", "anxious", "playful", "angry", "uplifting",
    "tender", "intimate", "soft", "warm", "cold", "haunting",
    "saudade", "mono no aware",
]

_VOCAB_PRODUCTION = [
    "lo-fi", "lofi", "hi-fi", "polished", "raw",
    "distorted", "saturated", "clean",
    "reverb-heavy", "reverb", "dry", "spacious", "intimate",
    "compressed", "punchy",
    "vintage tape", "vintage", "modern digital",
    "minimal", "dense",
    "sidechain", "vinyl crackle", "tape hiss",
]

# Pre-compile word-boundary regexes for fast scan
def _compile_vocab(words: Iterable[str]) -> list[tuple[str, re.Pattern]]:
    # Sort by length (descending) so we match longer phrases first when overlap.
    sorted_words = sorted(set(words), key=lambda w: (-len(w), w))
    out = []
    for w in sorted_words:
        # word-boundary on alphanumeric tokens; allow internal hyphens/spaces
        pat = re.compile(r"(?<![\w\-])" + re.escape(w) + r"(?![\w\-])", re.IGNORECASE)
        out.append((w, pat))
    return out


_VOCAB_PATTERNS = {
    "genre":      _compile_vocab(_VOCAB_GENRE),
    "instrument": _compile_vocab(_VOCAB_INSTRUMENT),
    "mood":       _compile_vocab(_VOCAB_MOOD),
    "production": _compile_vocab(_VOCAB_PRODUCTION),
}


def _scan_vocabulary(text: str, group: str, max_hits: int = 8) -> list[str]:
    """Return word-boundary substring hits from the vocab of `group`."""
    hits: list[str] = []
    seen: set[str] = set()
    for word, pat in _VOCAB_PATTERNS[group]:
        if pat.search(text):
            # If a longer phrase already covered this short one, skip
            if any(word in h or h in word for h in seen):
                continue
            hits.append(word)
            seen.add(word)
            if len(hits) >= max_hits:
                break
    return hits


def _clean_tag(s: str) -> str:
    """Strip surrounding punctuation, parentheses, normalize whitespace."""
    s = s.strip().strip(".,;:()")
    # Drop everything from an opening paren that has no closing one
    if "(" in s and ")" not in s:
        s = s.split("(", 1)[0].strip()
    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _split_field(text: str) -> list[str]:
    """Split a structured field value like 'UK Drill / Arabic Ambient / Trap'."""
    parts = [_clean_tag(p) for p in _GENRE_SPLIT.split(text)]
    # Filter: must be 2-25 chars (drop noise like long descriptive sentences)
    return [p for p in parts if 2 <= len(p) <= 25]


def parse(style_text: str) -> dict[str, list[str]]:
    """Parse a Suno style description into structured tags.

    Returns: {"genre": [...], "mood": [...], "instrument": [...],
              "vocal": [...], "tempo": [...]}
    Empty lists if nothing found.
    """
    out: dict[str, list[str]] = {k: [] for k in ("genre", "mood", "instrument", "vocal", "tempo")}
    if not style_text or not isinstance(style_text, str):
        return out

    # Some descriptions came back with literal "\n" / "\t" (escape sequences)
    # because Suno's HTML serialization didn't fully unescape. Normalize so
    # field-extraction regexes (which split on real newlines) work correctly.
    text = (style_text
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\n")
            .strip())

    # Stage 1: structured fields if present
    for group, pat in _FIELDS.items():
        m = pat.search(text)
        if m:
            out[group] = _split_field(m.group(1))[:6]

    # Stage 2: free-form fallback for empty groups
    for group in ("genre", "instrument", "mood"):
        if not out[group]:
            hits = _scan_vocabulary(text, group, max_hits=6)
            if hits:
                out[group] = hits

    # Production tags as bonus signal (always scan)
    if "production" not in out:
        out["production"] = _scan_vocabulary(text, "production", max_hits=4)

    return out


def flat_tags(parsed: dict[str, list[str]], max_total: int = 6) -> list[dict]:
    """Flatten structured parse into a single list for UI cards.

    Order: genre → instrument → mood → production. Lowercase, deduped.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for group in ("genre", "instrument", "mood", "vocal", "production"):
        for tag in parsed.get(group, []):
            t = tag.lower().strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append({"group": group, "tag": t})
            if len(out) >= max_total:
                return out
    return out
