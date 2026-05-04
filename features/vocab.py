"""Tag vocabularies for zero-shot classification via MuQ-MuLan.

Scenario A target: chill / discovery — user gives a seed track or a free-text
mood, expects "more like this". No BPM filters, no genre filters in UI.

So tags are kept ONLY for:
  - interpretable best:: labels (one-line "what is this track")
  - explanations in the UI ("recommended because: drill, dark, 808")
  - optional fallback boolean filters

Removed from previous version:
  - tempo phrases with BPM numbers (MuQ wasn't trained to read numbers)
  - era (meaningless for AI-generated music)
  - language (zero-shot is unreliable for non-English)
  - energy (overlaps with mood)
  - production (kept — it's a real perceptual axis: lo-fi vs polished)

Edit freely. Adding a tag is one extra dot product per track.
"""
from __future__ import annotations

GENRES = [
    "drill", "uk drill", "ny drill", "jersey drill",
    "trap", "phonk", "drift phonk", "memphis phonk",
    "boom bap hip hop", "lo-fi hip hop", "cloud rap",
    "hyperpop", "rage", "plugg",
    "kpop", "j-pop", "anime pop",
    "arabic music", "middle eastern", "turkish pop", "balkan",
    "afrobeat", "afroswing", "amapiano",
    "reggaeton", "latin trap",
    "jersey club", "drum and bass", "jungle",
    "techno", "house", "deep house", "hardstyle",
    "ambient", "dark ambient", "drone",
    "cinematic", "orchestral", "epic trailer music",
    "metal", "rock", "punk",
    "country", "folk", "indie folk",
    "jazz", "soul", "funk",
    "indian classical", "bollywood",
    "kazakh folk", "central asian",
    "russian rap", "russian pop",
]

# Fusion / hybrid styles — your prompt-driven case
FUSION = [
    "arabian junky drill", "arabian phonk",
    "kpop drill beat",
    "afro drill", "latin drill", "indian drill",
    "brazilian phonk", "russian phonk", "central asian trap",
    "turkish trap", "anime drill",
    "orchestral trap", "lofi drill",
]

MOODS = [
    "dark", "bright", "moody", "happy", "sad", "melancholic",
    "energetic", "chill", "relaxed", "aggressive",
    "romantic", "sensual", "mystical",
    "hypnotic", "dreamy", "nostalgic", "epic", "uplifting",
]

INSTRUMENTS = [
    "808 bass", "sub bass", "synth bass",
    "trap hi-hats", "rolling hi-hats", "snare", "claps",
    "kick drum", "hand drums", "tabla", "djembe", "darbuka",
    "male vocal", "female vocal", "rap vocal", "auto-tuned vocal",
    "choir", "operatic vocal",
    "violin", "strings ensemble", "cello",
    "piano", "rhodes", "organ",
    "acoustic guitar", "electric guitar", "distorted guitar", "spanish guitar",
    "synth lead", "synth pad", "supersaw", "analog synth",
    "saxophone", "trumpet",
    "flute", "ney flute", "duduk",
    "oud", "sitar", "kalimba",
]

PRODUCTION = [
    "lo-fi production", "hi-fi production", "polished mix", "raw mix",
    "distorted", "saturated", "clean",
    "reverb-heavy", "dry", "spacious",
    "vintage tape", "modern digital",
    "minimal arrangement", "dense arrangement",
]

VOCAL = [
    "instrumental track without vocals",
    "track with prominent vocals",
    "track with chopped vocal samples",
    "rap performance",
    "singing performance",
]

# 5 groups, ~130 tags total. One dot product per tag per track.
TAG_GROUPS: dict[str, list[str]] = {
    "genre":      GENRES,
    "fusion":     FUSION,
    "mood":       MOODS,
    "instrument": INSTRUMENTS,
    "production": PRODUCTION,
    "vocal":      VOCAL,
}


def all_tags() -> list[tuple[str, str]]:
    """Flat list of (group, tag) pairs."""
    return [(g, t) for g, tags in TAG_GROUPS.items() for t in tags]
