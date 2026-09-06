# Beijing Multi-Site Air Quality

Inter-university datathon: forecast **next-hour PM2.5** (`PM2_5_next_hour`) at 12 Beijing stations. Metric: **RMSE**.

Train: 1 Mar 2013 – 31 Aug 2016 (360,954 rows). Test: 31 Aug 2016 23:00 – 28 Feb 2017 (51,063 rows). Chronological forecast — the test window is entirely after train.

**Protocol:** `current_PM2_5` is in **train only**. Test does not include it. Persistence using true current PM2.5 is not a valid submission.

---

## README / reproduction instructions

These steps regenerate the modelling pipeline. The file we uploaded to the leaderboard is the existing **`submission.csv`** at the repo root. Retraining can move predictions slightly (LightGBM + `n_jobs=-1`).

### Required files

| Path | Role |
| --- | --- |
| `data/train.csv` | Official train table |
| `data/test.csv` | Official test table (no `current_PM2_5`, no target) |
| `feature_engineering.py` | Gap-filling, stage-1 PM2.5 proxy, stage-2 features |
| `train_and_submit.py` | Validation, training, inference, writes `submission.csv` |
| `requirements.txt` | Python packages |
| `02_Modelling.ipynb` | Methodology report (required notebook) |

Optional (not required to rebuild predictions): `01_EDA.ipynb`, `Data Cleaning.ipynb`.

### Execution order

```bash
python -m pip install -r requirements.txt
python train_and_submit.py
```

1. Install packages (Python **3.11**).
2. Confirm `data/train.csv` and `data/test.csv` are present.
3. Run **`python train_and_submit.py`** from the repo root.
4. That script: loads the official CSVs → `interpolate_then_mean_fill` → stage-1/stage-2 features → winter validation → fits the locked ensemble on all train → writes the prediction file.

**Do not run `Data Cleaning.ipynb` as part of reproduction.** It is missing-value EDA. Production gap-filling is inside `feature_engineering.interpolate_then_mean_fill`.

To rebuild from the methodology notebook instead: open `02_Modelling.ipynb`, set `RETRAIN = True`, run all cells.

### Which script generates the prediction file

**`train_and_submit.py`** writes:

- `submission.csv` (repo root) — **leaderboard / finalist prediction file**
- `outputs/submission.csv` — copy
- `outputs/val_leaderboard.csv` — local validation scores
- `outputs/final_model_record.json` — model card
- `outputs/stage1_current_pm25.joblib`, `outputs/final_lgb_pass1.joblib`, `outputs/final_lgb_delta.joblib` — fitted models

### Key dependencies / packages

From `requirements.txt`:

- pandas, numpy, scikit-learn, lightgbm, matplotlib, joblib
- holidays (optional; if missing, the holiday flag is 0)
- jupyter, nbformat (to open the notebooks)

XGBoost and CatBoost are **not** required for the submitted model.

### Random seeds and locked settings

| Setting | Value |
| --- | --- |
| Primary seed | **42** |
| Seed-bag | **42** and **2024** |
| Ensemble | **0.75** seed-bag + **0.25** residual-on-proxy |
| Stage-2 trees (full-train fit) | **800** |
| Learning rate / leaves | 0.03 / 63 |
| Sample weights | Sep–Feb ×1.8, heating season ×1.2, times `(1 + clip(y,0,400)/180)` |
| Post-process | clip predictions to **[0, 999]** |
| Dropped column | `year` (test includes 2017) |
| Validation window | Sep 2015 – Feb 2016 (model selection / reporting only) |

`numpy.random.seed(42)` is also set in `02_Modelling.ipynb`.

---

## Pipeline (what produced the submitted file)

```
data/train.csv, data/test.csv
        ↓ interpolate_then_mean_fill (feature_engineering.py)
        ↓ stage-1 current-PM2.5 proxy + spatial / lag features
        ↓ LightGBM seed-bag + residual LightGBM
        ↓ clip [0, 999], restore test id order
submission.csv
```

True `current_PM2_5` is never a stage-2 feature and is never written onto test.

Processed tables are **not** stored. They are rebuilt in memory each run (item 4 of the expected materials: regenerate from code).

---

## Final model

**0.75 × LightGBM seed-bag (seeds 42, 2024) + 0.25 × LightGBM residual around the stage-1 current-PM2.5 proxy.**

Local winter RMSE (Sep 2015 – Feb 2016): **25.57**. Full hyperparameters: `outputs/final_model_record.json`. Methodology: `02_Modelling.ipynb`.

---

## Repository layout

```
data/train.csv, data/test.csv     competition files
feature_engineering.py            cleaning + features
train_and_submit.py               train / val / submission.csv
02_Modelling.ipynb                methodology report
01_EDA.ipynb                      exploratory analysis (optional)
Data Cleaning.ipynb               missing-value EDA (optional)
requirements.txt
submission.csv                    leaderboard prediction file
outputs/                          model card, val table, saved models
```

---

## Required disclosure

- **External datasets:** none. Only competition `train.csv` / `test.csv`.
- **External code / public solutions:** none copied. RH, wind u/v, heating-season dates, and pollutant lags are standard PM2.5 features, written here from scratch.
- **Pretrained models:** none.
- **AI tools:** Cursor Grok 4.6 was used to draft and edit code, notebooks, and this README.
- **Manual modification of predictions:** none. `submission.csv` is clipped model output.
- **Additional information:** optional `holidays` China calendar; PRSA Beijing station coordinates for inverse-distance features (not an extra concentration feed).

---

## Data citation

[Inter-Uni Datathon Stream 2 — Beijing Multi-Site Air Quality](https://www.kaggle.com/competitions/inter-uni-datathon-stream-2-beijing-multi-site-air-quality/overview)
