# Beijing Multi-Site Air Quality

Inter-university datathon: forecast **next-hour PM2.5** (`PM2_5_next_hour`) at 12 Beijing stations. Metric: **RMSE**.

Train: March 2013 – August 2016 (360,954 rows). Test: 31 Aug 2016 23:00 – 28 Feb 2017 (51,063 rows). Chronological forecast — the test window is entirely after train.

**Protocol:** `current_PM2_5` is present in **train only**. Test does not include it. Persistence using true current PM2.5 is not a valid submission.

## Reproduce the submission

Python 3.11+, seed **42**.

```bash
pip install -r requirements.txt
```

Data already in `data/`:

1. `data/train.csv`, `data/test.csv` — competition files.
2. Run `Data Cleaning.ipynb` if you need to regenerate `data/train_cleaned.csv` and `data/test_cleaned.csv` (training-only means; `wd` → `"Missing"`; **do not** impute `current_PM2_5` on test).
3. Train and write predictions (several minutes):

```bash
python train_and_submit.py
```

or open `02_Modelling.ipynb`, set `RETRAIN = True`, and run all cells.

**Prediction file:** `submission.csv` (also copied to `outputs/submission.csv`).  
**Model card:** `outputs/final_model_record.json`.  
**Validation table:** `outputs/val_leaderboard.csv`.

## Pipeline

| Step | File |
| --- | --- |
| Cleaning | `Data Cleaning.ipynb` |
| EDA | `01_EDA.ipynb` |
| Features (two-stage PM2.5 proxy + PM10 lags) | `feature_engineering.py` |
| Validation, training, submission | `train_and_submit.py` |
| Methodology report | `02_Modelling.ipynb` |

Stage 1 predicts current PM2.5 from test-available sensors (PM10, CO, NO2, weather, station, hour). Stage 2 predicts the next hour from that proxy plus PM10-centric lags, city-wide PM10, heating season and wind physics. True `current_PM2_5` is never a stage-2 feature.

Primary validation is **Sep 2015 – Feb 2016** (same season as test). A secondary last-6-month window (Mar–Aug 2016) is reported but not used for model selection.

Post-processing: clip predictions to `[0, 999]`.

## Features (raw)

| Column | Description |
| --- | --- |
| `id` | Observation ID |
| `observation_timestamp` | Hour of the reading |
| `station` | Monitoring site |
| `year`, `month`, `day`, `hour` | Calendar fields |
| `current_PM2_5` | PM2.5 at the current hour (train only) |
| `PM10`, `SO2`, `NO2`, `CO`, `O3` | Other pollutants |
| `TEMP`, `PRES`, `DEWP`, `RAIN`, `wd`, `WSPM` | Weather |
| `PM2_5_next_hour` | Target — PM2.5 one hour ahead (train only) |

Stations: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong.

## Data citation

[Inter-Uni Datathon Stream 2 — Beijing Multi-Site Air Quality](https://www.kaggle.com/competitions/inter-uni-datathon-stream-2-beijing-multi-site-air-quality/overview)
