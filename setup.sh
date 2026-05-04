#!/usr/bin/env bash
# One-shot bootstrap for Apple Silicon (M1/M2/M3).
# Creates a venv, installs deps, verifies MPS, prepares data dirs.

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install via: brew install python@3.11"
  exit 1
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python $PYVER"
if [[ "$(printf '%s\n' "3.10" "$PYVER" | sort -V | head -1)" != "3.10" ]]; then
  echo "Python >= 3.10 required (you have $PYVER)"
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Creating .venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel

# Core deps. We split MuQ off so its potential failure doesn't kill the rest.
echo "Installing core dependencies ..."
grep -v '^muq' requirements.txt | pip install -r /dev/stdin

echo "Installing MuQ-MuLan (multimodal music encoder) ..."
if pip install 'muq>=0.1.0'; then
  echo "  → MuQ installed."
else
  echo "  → MuQ failed to install. Falling back to MERT (audio-only, no zero-shot)."
  if [[ ! -f .env ]]; then cp .env.example .env; fi
  if grep -q '^AUDIO_MODEL=' .env; then
    sed -i.bak 's/^AUDIO_MODEL=.*/AUDIO_MODEL=mert/' .env && rm -f .env.bak
  else
    echo 'AUDIO_MODEL=mert' >> .env
  fi
  echo "  → wrote AUDIO_MODEL=mert to .env"
fi

echo "Verifying PyTorch + MPS ..."
python -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  MPS available: {torch.backends.mps.is_available()}')
print(f'  MPS built:     {torch.backends.mps.is_built()}')
"

mkdir -p data/audio data/index data/cache
[[ -f .env ]] || cp .env.example .env

cat <<'EOF'

Setup complete. Next:

  source .venv/bin/activate

  # option A — drop mp3s into data/audio/ then:
  python -m scripts.extract --scan-audio-dir

  # option B — give Suno share URLs:
  echo "https://suno.com/song/<id>" > urls.txt
  python -m scripts.extract --batch urls.txt

Output → data/index/audio.npy  +  data/index/tracks.parquet
EOF
