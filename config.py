import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data")).resolve()
AUDIO_DIR = DATA_DIR / "audio"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"

for d in (AUDIO_DIR, INDEX_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

AUDIO_MODEL = os.getenv("AUDIO_MODEL", "muq").lower()
DEVICE = os.getenv("DEVICE", "mps").lower()

TARGET_SR = 24000
CHUNK_SECONDS = 30
