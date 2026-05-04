"""Audio embeddings via MuQ-MuLan (primary) or MERT (fallback).

MuQ-MuLan is multimodal (audio<->text in a shared space) — use embed_text() for
text-to-audio retrieval. MERT is audio-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch

from config import AUDIO_MODEL, CHUNK_SECONDS, DEVICE, TARGET_SR


def _resolve_device() -> torch.device:
    if DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_mono_24k(path: str | Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return y.astype(np.float32)


def _chunk(y: np.ndarray, sr: int, seconds: int) -> list[np.ndarray]:
    n = sr * seconds
    if len(y) <= n:
        return [y]
    return [y[i : i + n] for i in range(0, len(y), n) if len(y[i : i + n]) >= sr * 5]


class AudioEmbedder:
    """Common interface. Use AudioEmbedder.load() to construct."""

    dim: int
    supports_text: bool

    def embed_audio(self, path: str | Path) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} is audio-only")

    @staticmethod
    def load(model: str | None = None) -> "AudioEmbedder":
        choice = (model or AUDIO_MODEL).lower()
        if choice == "muq":
            try:
                return MuQEmbedder()
            except Exception as e:
                print(f"[audio] MuQ failed to load ({e}); falling back to MERT")
                return MertEmbedder()
        if choice == "mert":
            return MertEmbedder()
        raise ValueError(f"Unknown AUDIO_MODEL={choice!r}")


class MuQEmbedder(AudioEmbedder):
    """MuQ-MuLan large — joint audio/text space, 512-dim."""

    supports_text = True

    def __init__(self) -> None:
        from muq import MuQMuLan  # local import: optional dep

        self.device = _resolve_device()
        self.model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(self.device).eval()
        self.dim = 512

    @torch.no_grad()
    def embed_audio(self, path: str | Path) -> np.ndarray:
        # MuQ-MuLan does its own 10s internal chunking and averaging — pass the
        # full waveform with batch dim 1 instead of pre-chunking ourselves.
        y = _load_mono_24k(path)
        wavs = torch.from_numpy(y).unsqueeze(0).to(self.device)  # [1, T]
        emb = self.model(wavs=wavs)  # [1, 512]
        # MuQ normalizes per internal chunk but not the averaged output —
        # re-normalize so cosine search and softmax are well-behaved.
        return _l2(emb[0].cpu().numpy())

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        emb = self.model(texts=[text])  # [1, 512]
        return _l2(emb[0].cpu().numpy())


class MertEmbedder(AudioEmbedder):
    """MERT-v1-330M — audio-only, 1024-dim."""

    supports_text = False

    def __init__(self) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.device = _resolve_device()
        name = "m-a-p/MERT-v1-330M"
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).to(self.device).eval()
        self.dim = self.model.config.hidden_size  # 1024
        self.sr = self.fe.sampling_rate  # 24kHz
        self.chunk_secs = 10  # pretrain context is 5s; 5–10s yields the best embeddings

    @torch.no_grad()
    def embed_audio(self, path: str | Path) -> np.ndarray:
        y, _ = librosa.load(str(path), sr=self.sr, mono=True)
        chunks = _chunk(y.astype(np.float32), self.sr, self.chunk_secs)
        vecs = []
        for c in chunks:
            inp = self.fe(c, sampling_rate=self.sr, return_tensors="pt").to(self.device)
            out = self.model(**inp)
            h = out.last_hidden_state.mean(dim=1)[0]
            vecs.append(h.cpu().numpy())
        vec = np.mean(np.stack(vecs), axis=0)
        return _l2(vec)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


def embed_many(paths: Iterable[str | Path], model: AudioEmbedder) -> np.ndarray:
    out = []
    for p in paths:
        out.append(model.embed_audio(p))
    return np.stack(out).astype(np.float32)
