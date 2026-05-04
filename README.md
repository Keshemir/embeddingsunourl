# ozenref

Audio embedding for AI-generated tracks (Suno), focused on
**discovery / chill recommendations** ("дай похожее на этот трек" /
"chill vibe для работы"). Built for fusion micro-genres
("arabian junky", "kpop drill") that generic classifiers can't handle.

Apple Silicon (M1/M2/M3), no GPU required.

## Design choice

The recommender is **content-based, embedding-first**. The audio embedding
does the heavy lifting; tags are kept only for interpretable labels and
optional UI filters; numerical features are kept only for display.

This matches what NEWAVE (Habr/989756) ship for the same problem at the
same scale.

## What you get per track

```
audio.npy        (N, 512)  MuQ-MuLan vector — primary signal, cosine = similarity
tracks.parquet   (N rows × ~165 cols)
                 ├── meta (~10):       track_id, source, title, prompt, style,
                 │                     lyrics, duration_sec, bpm, bpm_perceived,
                 │                     key, key_confidence, rms_mean
                 ├── tag::* (~140):    6 groups — genre, fusion, mood,
                 │                     instrument, production, vocal
                 │                     (per-group softmax probabilities)
                 └── best::* (6):      single best label per group, for the
                                       one-line summary
```

Two signal types — and that's enough:
1. **Dense embedding** for cosine similarity (the actual recsys).
2. **Sparse tag scores** for `--no-vocals` style boolean filters and "why this rec".

What's NOT here on purpose: explicit BPM matching, tempo phrases, language
detection, energy buckets, era buckets. They didn't pay off for the chill
discovery use case and added noise.

## Полная инструкция (Mac mini M1/M2/M3)

### 1. Открыть Terminal

Cmd+Space → набрать `Terminal` → Enter.

### 2. Получить репо

Если уже на GitHub:

```bash
git clone <repo-url> ~/ozenref
cd ~/ozenref
```

Если переносишь папку вручную (AirDrop / scp / iCloud) — просто положи её в `~/ozenref` и сделай `cd ~/ozenref`.

### 3. Установка (один раз, 5–10 минут)

```bash
./setup.sh
```

Что делает:
- создаёт `.venv` (изолированное окружение Python),
- ставит зависимости (`torch`, `transformers`, `librosa`, `muq`, ...),
- проверяет, что MPS (Apple Silicon GPU) работает,
- создаёт папки `data/audio/`, `data/index/`, `data/cache/`,
- копирует `.env.example` → `.env`.

Если `muq` не соберётся — скрипт сам пропишет `AUDIO_MODEL=mert` в `.env` и продолжит. В конце увидишь `Setup complete`.

### 4. Активировать окружение

Каждый раз, когда открываешь новый Terminal:

```bash
cd ~/ozenref
source .venv/bin/activate
```

### 5. Подготовить ссылки

Открой пустой файл `urls.txt` в TextEdit:

```bash
open -a TextEdit urls.txt
```

**Важно**: до того как начнёшь печатать, в меню TextEdit выбери `Format → Make Plain Text` (Cmd+Shift+T) — иначе сохранится `.rtf` и скрипт его не прочитает.

Вставляй ссылки **по одной на строку**:

```
https://suno.com/song/abc123
https://suno.com/song/def456
https://suno.com/song/ghi789
```

Сохрани (Cmd+S). Можно мешать ссылки и локальные пути в одном файле:

```
https://suno.com/song/abc123
/Users/alrakhymzhan/Music/my_local_track.mp3
```

### 6. Запустить эмбеддинг

```bash
python -m scripts.extract --batch urls.txt
```

Первый прогон скачает веса MuQ-MuLan (~1 GB, кешируется в `~/.cache/huggingface/`). Дальше — быстро. В Terminal будет прогрессбар:

```
extract: 100%|████████████| 50/50 [02:30<00:00,  3.0s/it]
saved 50 rows     →  data/index/tracks.parquet
saved 50 vectors  →  data/index/audio.npy
```

### 7. Готово — что у тебя на диске

```
~/ozenref/data/index/
├── audio.npy        ← (N, 512) векторы для cosine similarity
└── tracks.parquet   ← все фичи + ссылки на каждый трек
```

mp3 удаляются автоматически после эмбеддинга — в `tracks.parquet` остаётся URL. Если хочешь оставить mp3 локально:

```bash
python -m scripts.extract --batch urls.txt --keep-audio
```

### 8. Проверить, что всё ок

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/index/tracks.parquet')
print('rows:', len(df), 'cols:', len(df.columns))
cols = ['track_id','source','best::genre','best::mood','best::instrument','bpm_perceived','key']
print(df[cols].head())
"
```

### 9. Найти похожее (то, ради чего всё затевалось)

```bash
# по сид-треку
python -m scripts.search --track <track_id_из_parquet>

# по описанию вайба
python -m scripts.search --text "chill late-night drive"
python -m scripts.search --text "dark trap with arabic strings"

# только без вокала (для работы)
python -m scripts.search --text "study music" --no-vocals
```

---

## Шпаргалка на каждый день

```bash
cd ~/ozenref && source .venv/bin/activate

# докинуть новые ссылки и пере-эмбеддить
open -a TextEdit urls.txt
python -m scripts.extract --batch urls.txt

# скан локальной папки (если кидаешь mp3 в data/audio/)
python -m scripts.extract --scan-audio-dir

# поиск
python -m scripts.search --text "chill late-night drive"
python -m scripts.search --track <track_id>
python -m scripts.search --text "study music" --no-vocals
python -m scripts.search --text "drill" --min-tag genre::drill=0.2
```

Через Make:

```bash
make install                                # один раз
make extract BATCH=urls.txt
make scan                                   # data/audio/*.mp3
make search Q='chill late-night drive'
```

## Inspecting results

```python
import numpy as np, pandas as pd
v = np.load("data/index/audio.npy")
t = pd.read_parquet("data/index/tracks.parquet")

# 1. nearest neighbours of track 0 (the actual recsys)
sims = v @ v[0]
print(t.iloc[np.argsort(-sims)[:10]][['track_id','best::genre','best::mood']])

# 2. top tags of one track (the "why")
row = t.iloc[0]
top = row.filter(regex='^tag::').sort_values(ascending=False).head(10)
print(top)

# 3. boolean filters on tag scores
mask = (t['tag::genre::drill'] > 0.15) & (t['tag::mood::dark'] > 0.10)
print(t[mask][['track_id', 'title']])
```

## Architecture

```
ingest/
  suno_url.py      # Suno share URL → mp3 + prompt (4 parse strategies, last is regex on cdn URL)
  suno.py          # local mp3 + ID3 / sidecar JSON → Track

embed/
  audio.py         # MuQ-MuLan (primary, 512d) | MERT-v1-330M (fallback)

features/
  vocab.py         # ~140 tag phrases in 6 groups: genre / fusion / mood /
                   # instrument / production / vocal
  audio_features.py# librosa: duration, bpm, bpm_perceived, key, rms_mean
  zeroshot.py      # cosine(audio_vec, text_vec(tag)) → per-group softmax

scripts/
  extract.py       # ★ main pipeline: URL|mp3|batch → audio.npy + tracks.parquet
  search.py        # 2-stage: cosine retrieve → boolean tag filter

config.py          # AUDIO_MODEL, DEVICE, TARGET_SR, paths
```

## Configuration (`.env`)

```bash
AUDIO_MODEL=muq      # muq | mert
DEVICE=mps           # mps | cpu | cuda
DATA_DIR=./data
```

## Tag vocabulary

Edit [features/vocab.py](features/vocab.py) freely. Tags are short
natural-language phrases — cosine to `text_vec(phrase)` is the score.
Adding tags is free; one extra dot product per track.

Groups: `genre`, `fusion`, `mood`, `instrument`, `production`, `energy`,
`vocal`, `language`, `era`.

## Troubleshooting

**`muq` won't install.**  Set `AUDIO_MODEL=mert` in `.env` — uses MERT
(audio-only, no zero-shot tagging). Embeddings still work; you lose the
`tag::*` columns.

**Suno URL parse fails** (`could not extract audio URL`). Suno changes
their HTML periodically. Workaround: download the mp3 from Suno UI,
drop it under `data/audio/`, and run `--scan-audio-dir`. Pre-existing
ID3 + sidecar JSON metadata are honoured.

**MPS errors mid-run.** Some ops fall back to CPU on MPS. Set
`DEVICE=cpu` in `.env` if it's flaky — slower but stable.

**Empty `tag::*` columns.** Means zero-shot was disabled (MERT mode or
a runtime error). Check stdout for `[warn]` lines from `extract.py`.

## What's NOT in here yet

- `index/store.py` and `recsys/recommend.py` are stubs for the next
  stage (vector index + Rocchio + MMR rec). They don't run from the CLI
  yet — pipeline above is feature extraction only.
- No web UI. Inspect via pandas / your notebook of choice.
- No event logging. Add it before you start collecting user
  interactions, otherwise you'll have nothing to learn on later.

## License

MIT.
