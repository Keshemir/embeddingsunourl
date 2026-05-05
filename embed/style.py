"""Text embedder for Suno style descriptions, multilingual.

Uses BGE-M3 (BAAI/bge-m3, 568M params, ~2GB) — a multilingual retrieval-tuned
encoder. Handles Russian and English in the same space (so a query "арабский"
matches descriptions written in English, and vice versa).

Why this and not OpenAI / Voyage / e5: BGE-M3 is the practical SOTA for
multilingual retrieval as of 2026, runs locally on M2 in seconds for the
whole 200-track corpus, and is what NEWAVE ships for the same problem.

Output: 1024-dim float32, L2-normalized.

Usage:
    em = StyleEncoder()
    v = em.encode("Genre: UK Drill / Arabic Ambient...")   # (1024,)
    M = em.encode_batch(list_of_descriptions)               # (N, 1024)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from config import DEVICE


def _resolve_device() -> str:
    if DEVICE == "mps" and torch.backends.mps.is_available():
        return "mps"
    if DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class StyleEncoder:
    """Lazily-loaded BGE-M3 wrapper. One instance per process."""

    def __init__(self, model_id: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer

        self.device = _resolve_device()
        self.model = SentenceTransformer(model_id, device=self.device)
        self.dim = int(self.model.get_sentence_embedding_dimension() or 1024)

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    @torch.no_grad()
    def encode_batch(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self._encode(list(texts), batch_size=batch_size)

    def _encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        v = self.model.encode(
            texts, batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return v


@lru_cache(maxsize=1)
def get_default_encoder() -> StyleEncoder:
    """Singleton accessor."""
    return StyleEncoder()
