# Beijing Multi-Site Air Quality

Inter-university datathon: forecast **next-hour PM2.5** (`PM2_5_next_hour`) at 12 Beijing stations. Metric: **RMSE**.

Train: 1 Mar 2013 – 31 Aug 2016. Test: 31 Aug 2016 23:00 – 28 Feb 2017. Chronological forecast — the test window is entirely after train.

**Protocol:** `current_PM2_5` is in **train only**. Test does not include it. Persistence using true current PM2.5 is not a valid submission.

The production approach is in **`temp.py`**: LightGBM predicts the one-hour **delta**, then each station is walked forward so the previous prediction becomes the next hour’s current PM2.5. **`fixed_model.py`** is the comparison hybrid (weather-only model + recursive-history model).

---

## README / reproduction instructions

The leaderboard file is **`submission.csv`** at the repo root, written by `temp.py`. Retraining can move predictions slightly (LightGBM).

### Required files

| Path | Role |
| --- | --- |
| `data/train.csv`, `data/test.csv` | Official competition tables |
| `Data Cleaning.ipynb` | Writes `data/train_cleaned.csv` and `data/test_cleaned.csv` |
| `data/train_cleaned.csv`, `data/test_cleaned.csv` | Tables used by EDA and modelling |
| `temp.py` | **Primary pipeline** — train, recursive inference, writes `submission.csv` |
| `fixed_model.py` | Optional comparison — writes `submission_fixed.csv` |
| `Modelling.ipynb` | Methodology report |
| `requirements.txt` | Python packages |

Optional: `EDA.ipynb` (uses the cleaned tables).

### Execution order

```bash
python -m pip install -r requirements.txt
```

1. Install packages (Python **3.11**).
2. Official `data/train.csv` and `data/test.csv` present.
3. Run **`Data Cleaning.ipynb`** (writes `data/train_cleaned.csv` and `data/test_cleaned.csv`). `current_PM2_5` is filled on train only.
4. From the repo root run **`python temp.py`** (requires the cleaned CSVs).
5. That script fits LightGBM on the PM2.5 delta, walks each station through test, restores official test `id` order, and writes `submission.csv`.

Optional comparison (does not overwrite `submission.csv`):

```bash
python fixed_model.py
```

That writes `submission_fixed.csv`.

To rebuild from the methodology notebook: open `Modelling.ipynb`, set `RETRAIN = True`, run all cells. The notebook also reads the cleaned CSVs.

CLI overrides for `temp.py`: `--data-dir` (default `data/`), `--output-dir` (default `recursive_lightgbm_output/`).

### Which script generates the prediction file

**`temp.py`** writes:

- `submission.csv` (repo root) — **leaderboard / finalist prediction file**
- `recursive_lightgbm_output/submission_recursive_lightgbm.csv` — copy
- `recursive_lightgbm_output/recursive_predictions_with_state.csv` — per-row recursive state
- `recursive_lightgbm_output/recursive_lightgbm_model.txt` — saved LightGBM booster
- `recursive_lightgbm_output/metrics.json` — row counts, seed, hyperparameters

**`fixed_model.py`** writes only `submission_fixed.csv` (comparison).

### Key dependencies / packages

From `requirements.txt`: pandas, numpy, lightgbm, scikit-learn (`fixed_model.py` RMSE), matplotlib, jupyter, nbformat.

### Random seeds and locked settings (`temp.py`)

| Setting | Value |
| --- | --- |
| seed / feature_fraction_seed / bagging_seed | **42** |
| objective / metric | regression / RMSE |
| learning_rate | **0.035** |
| num_leaves | **63** |
| min_data_in_leaf | **80** |
| feature_fraction / bagging_fraction | 0.85 / 0.85 |
| bagging_freq | 1 |
| lambda_l2 | 1.0 |
| num_boost_round | **900** |
| target | `PM2_5_next_hour - current_PM2_5` (delta), added back at inference |
| post-process | `max(0, current + delta)` |
| lags | 1, 2, 3, 6, 12, 24 hours (kept only if the timestamp gap is exact) |

`fixed_model.py` (comparison): seed **42**; Model A 1200 rounds, Model B 1500 rounds; `learning_rate=0.02`, `num_leaves=255`, `max_depth=12`.

---

## Pipeline (what produced `submission.csv`)

```
data/train.csv, data/test.csv
        ↓ Data Cleaning.ipynb  →  train_cleaned.csv, test_cleaned.csv
        ↓ canonicalize (timestamp, station_code, wind degrees)
        ↓ LightGBM on delta = next-hour PM2.5 − current PM2.5
        ↓ per-station recursive walk
        ↓ max(0, ·), restore test id order
submission.csv
```

True test `current_PM2_5` is never used. Recursive state is the model’s own previous prediction.

---

## Final model

**Recursive LightGBM delta** in `temp.py` (`lgb.train`, 900 rounds, seed 42). Full description: `Modelling.ipynb` §6–9.

`fixed_model.py` is the alternative, not the leaderboard file.

---

## Repository layout

```
data/train.csv, data/test.csv     competition files
Data Cleaning.ipynb               writes train_cleaned.csv / test_cleaned.csv
data/train_cleaned.csv            cleaned train (used by EDA and modelling)
data/test_cleaned.csv             cleaned test (no current_PM2_5)
temp.py                           primary train / recursive inference / submission.csv
fixed_model.py                    comparison hybrid A/B
Modelling.ipynb                   methodology report
EDA.ipynb                         exploratory analysis on the cleaned tables
requirements.txt
submission.csv                    leaderboard prediction file
```

---

## Required disclosure

- **External datasets:** none for fitting. Competition `data/train.csv` / `data/test.csv`, cleaned by `Data Cleaning.ipynb` to `train_cleaned.csv` / `test_cleaned.csv`. If `online.csv` is present, `temp.py` / `fixed_model.py` use it **after** inference to print a local RMSE; it is not a feature and is not written into recursive history.
- **External code / public solutions:** none copied. Heating-season dates, wind u/v, and lagged PM2.5 are standard air-quality features, implemented in `temp.py` and `fixed_model.py`.
- **Pretrained models:** none.
- **AI tools:** Cursor Grok 4.6 was used to draft and edit the methodology notebook and this README so they match `temp.py` and `fixed_model.py`.
- **Manual modification of predictions:** none. `temp.py` applies `max(0, ·)` only.
- **Additional information:** none beyond the competition files (and optional local `online.csv` labels).

---

## Data citation

[Inter-Uni Datathon Stream 2 — Beijing Multi-Site Air Quality](https://www.kaggle.com/competitions/inter-uni-datathon-stream-2-beijing-multi-site-air-quality/overview)
