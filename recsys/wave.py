"""«Моя волна» — личная радиостанция с микшером направлений.

Логика:
  1. Для каждого ползунка храним пару BGE-M3 эмбеддингов («pos» и «neg»
     описание жанра). Direction = pos_vec - neg_vec — единичный вектор
     «куда двигаться при увеличении ползунка».
  2. Положение ползунка `x ∈ [0, 1]` мапится на push:
        push = (x - 0.5) * 2 * direction
     При x = 0.5 push = 0 (ползунок «отключён»).
  3. Сумма пушей всех ползунков = mixer_vec.
  4. Wave mode: query = α·taste_style + (1-α)·mixer_vec, α=0.6.
     Mixer mode: query = mixer_vec (нормированный).
     Если mixer_vec ≈ 0 в Mixer mode — fallback на mean(style_npy).
  5. Cosine на style.npy → top pool → MMR → dedup → top-k.

Слайдеры жанровые: Drill / Trap / Ambient / Pop / Rock. Семантика —
«хочу больше / меньше этого жанра в волне».
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from recsys import events, feed

# 5 жанровых ползунков — пары полярных фраз.
# Левая фраза описывает «не этот жанр», правая — «этот жанр».
# pos > 0.5 двигает запрос в сторону правой, pos < 0.5 — в сторону левой.
GENRE_DIRECTIONS: dict[str, tuple[str, str]] = {
    "drill": (
        "not drill, no UK drill, no trap drill",
        "UK drill, drill beat with rolling hi-hats, sliding 808s, dark melody",
    ),
    "trap": (
        "not trap, no 808 bass trap beat",
        "trap beat with 808 bass, rolling hi-hats, snappy snare, dark piano",
    ),
    "ambient": (
        "not ambient, no atmospheric pads, no drone",
        "ambient atmospheric pads, slow evolving texture, drone, downtempo",
    ),
    "pop": (
        "not pop, not chart-friendly, not mainstream hook",
        "pop song with chart-friendly hook, polished mainstream production",
    ),
    "rock": (
        "not rock, no electric guitar, no rock band instrumentation",
        "rock track with distorted electric guitar, drum kit, rock band",
    ),
}

ALPHA_TASTE = 0.6   # доля personalized taste в Wave-mode (1-ALPHA = микшер)
DEDUP_AUDIO_THRESHOLD = 0.92
MMR_LAMBDA = 0.65


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


@lru_cache(maxsize=1)
def _direction_bank(_encoder_id: int) -> dict[str, np.ndarray]:
    """Эмбеддит пары полярных фраз через BGE-M3 один раз.

    Возвращает {slider_name: direction_vec (unit)} — вектор от neg к pos.
    Cache привязан к id(encoder); пересоздаётся только при подмене модели.
    """
    from embed.style import StyleEncoder
    encoder = StyleEncoder()  # локальный — direction bank считается один раз
    out: dict[str, np.ndarray] = {}
    for name, (neg, pos) in GENRE_DIRECTIONS.items():
        v_neg = encoder.encode(neg)
        v_pos = encoder.encode(pos)
        # Direction = разница (не L2-нормированная — амплитуда несёт смысл).
        direction = (v_pos - v_neg).astype(np.float32)
        out[name] = direction
    return out


def _build_direction_bank(encoder) -> dict[str, np.ndarray]:
    """Строит direction bank используя уже-загруженный encoder.

    Это то что должно вызываться из serve.py — мы уже имеем StyleEncoder
    в памяти, не надо создавать второй.
    """
    out: dict[str, np.ndarray] = {}
    for name, (neg, pos) in GENRE_DIRECTIONS.items():
        v_neg = encoder.encode(neg)
        v_pos = encoder.encode(pos)
        out[name] = (v_pos - v_neg).astype(np.float32)
    return out


def mixer_vec(sliders: dict[str, float],
              direction_bank: dict[str, np.ndarray]) -> np.ndarray:
    """Сложить направления ползунков в один (1024,) вектор.

    pos = 0.5 → ползунок не вносит ничего.
    pos > 0.5 → пушим в сторону «pos» (положительная сторона).
    pos < 0.5 → пушим в сторону «neg» (отрицательная сторона).
    """
    if not direction_bank:
        raise RuntimeError("direction bank not built")
    dim = next(iter(direction_bank.values())).shape[0]
    acc = np.zeros(dim, dtype=np.float32)
    for name, direction in direction_bank.items():
        x = float(sliders.get(name, 0.5))
        x = max(0.0, min(1.0, x))
        push = (x - 0.5) * 2.0   # -1..+1
        acc += push * direction
    return acc


def recommend_wave(
    user_id: str,
    sliders: dict[str, float],
    *,
    mode: str,
    exclude: set[str],
    audio: np.ndarray,
    style: np.ndarray,
    meta: pd.DataFrame,
    direction_bank: dict[str, np.ndarray],
    k: int = 5,
    pool: int = 50,
    mmr_lambda: float = MMR_LAMBDA,
    dedup_threshold: float = DEDUP_AUDIO_THRESHOLD,
    exclude_recent_seconds: float = 6 * 3600,
) -> tuple[list[int], dict]:
    """Returns (track_indices, debug_info).

    mode = "wave" → α·taste_style + (1-α)·mixer_vec
    mode = "mixer" → чистый mixer_vec (или mean corpus если все ползунки=0.5)
    """
    debug: dict = {"mode_requested": mode}

    mix = mixer_vec(sliders, direction_bank)
    mix_norm = float(np.linalg.norm(mix))
    debug["mixer_norm"] = mix_norm

    # Wave mode: подмешиваем taste_style. Если у юзера 0 истории — fallback в mixer.
    taste = None
    if mode == "wave":
        taste = feed.taste_vector_style(user_id, style, meta)
        debug["taste_used"] = taste is not None
        if taste is None:
            mode = "mixer"  # downgrade to mixer
            debug["downgraded_to"] = "mixer"

    if mode == "wave":
        # При нейтральном микшере (mix_norm ≈ 0) запрос = чистый taste.
        if mix_norm < 1e-6:
            query = _l2(taste)
            debug["query_blend"] = "taste_only"
        else:
            query = ALPHA_TASTE * _l2(taste) + (1.0 - ALPHA_TASTE) * _l2(mix)
            query = _l2(query)
            debug["query_blend"] = f"taste*{ALPHA_TASTE} + mixer*{1 - ALPHA_TASTE}"
        debug["mode"] = "wave"
    else:
        # Mixer-only. Если все ползунки на 0.5 — берём центроид корпуса
        # (даёт разнообразный pool для exploration).
        if mix_norm < 1e-6:
            query = _l2(style.mean(axis=0))
            debug["query_blend"] = "corpus_centroid"
        else:
            query = _l2(mix)
            debug["query_blend"] = "mixer_only"
        debug["mode"] = "mixer"

    # Cosine over style space
    sims = (style @ query).astype(np.float32)

    # Exclude: client-supplied + recent listens (last 6h)
    excl = set(exclude or set())
    excl |= events.recent_track_ids(user_id, within_seconds=exclude_recent_seconds)
    if excl:
        idx_map = {t: i for i, t in enumerate(meta["track_id"].tolist())}
        for t in excl:
            i = idx_map.get(t)
            if i is not None:
                sims[i] = -np.inf
    debug["n_excluded"] = int(len(excl))

    pool_idx = [int(i) for i in np.argsort(-sims)[:pool] if sims[i] != -np.inf]
    if not pool_idx:
        debug["empty_pool"] = True
        return [], debug

    # MMR over the pool to ensure diversity, then dedup audio multi-takes.
    selected: list[int] = []
    remaining = list(pool_idx)
    target = k + 5  # over-pick to give dedup slack
    while len(selected) < target and remaining:
        if not selected:
            i = max(remaining, key=lambda c: sims[c])
        else:
            def mmr_score(c: int) -> float:
                rel = float(sims[c])
                div = max(float(audio[c] @ audio[s]) for s in selected)
                return mmr_lambda * rel - (1 - mmr_lambda) * div
            i = max(remaining, key=mmr_score)
        selected.append(i)
        remaining.remove(i)

    final: list[int] = []
    for i in selected:
        if all(float(audio[i] @ audio[k_]) < dedup_threshold for k_ in final):
            final.append(i)
        if len(final) >= k:
            break

    return final, debug
