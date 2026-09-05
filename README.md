# Beijing Multi-Site Air Quality

Inter-university datathon project: forecast **next-hour PM2.5** at 12 Beijing monitoring stations using hourly pollutant and weather readings.

## Task

Predict `PM2_5_next_hour` (μg/m³) for each test observation.

### Dataset overview

360,954 training observations
51,063 test observations
12 Beijing air-quality monitoring stations
Hourly measurements
Target: PM2.5 concentration one hour ahead

Train–test structure

Training: March 2013 – August 2016

Testing: September 2016 – February 2017

Test data occurs entirely after the training period

This is a time-series forecasting problem rather than a conventional random regression problem, the test set represents the future, randomly splitting observations into train and validation sets could leak future information into model training.

The split is chronological, so models should respect time order and avoid leaking later hours into earlier ones.

## Features

| Column | Description |
| --- | --- |
| `id` | Observation ID |
| `observation_timestamp` | Hour of the reading |
| `station` | Monitoring site (12 stations) |
| `year`, `month`, `day`, `hour` | Calendar fields from the timestamp |
| `current_PM2_5` | PM2.5 at the current hour (μg/m³) |
| `PM10`, `SO2`, `NO2`, `CO`, `O3` | Other pollutants (μg/m³; CO in μg/m³) |
| `TEMP` | Temperature (°C) |
| `PRES` | Pressure (hPa) |
| `DEWP` | Dew point (°C) |
| `RAIN` | Precipitation (mm) |
| `wd` | Wind direction |
| `WSPM` | Wind speed (m/s) |
| `PM2_5_next_hour` | **Target** — PM2.5 one hour ahead (train only) |

Stations: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong. Each site has roughly 30k training rows.

## Repository

```
train.csv              labelled hourly observations
test.csv               same features, no target
EDA.ipynb              exploratory analysis and persistence baseline
Data Cleaning.ipynb    placeholder for imputation / feature prep
```

## Setup

Python 3.11+ with Jupyter.

```bash
pip install pandas numpy matplotlib jupyter
jupyter notebook EDA.ipynb
```

Place `train.csv` and `test.csv` in the repo root. The notebooks load them with `pd.read_csv("train.csv")`.

## EDA highlights

From `EDA.ipynb`:

- **Missingness.** Weather fields are nearly complete. Pollutants have more gaps — CO is the sparsest in train (about 4.4% missing). Test has the same pattern at a smaller scale.
- **Target shape.** Next-hour PM2.5 is right-skewed (mean 78, median 55, max 999 μg/m³). About 63% of hours exceed 35 μg/m³ and 14% exceed 150 μg/m³.
- **Strong persistence.** Current PM2.5 correlates **0.97** with the next-hour target. PM10 (0.85), CO (0.76), and NO2 (0.64) are the next strongest numeric correlates. Wind speed is the strongest negative correlate (−0.28).
- **Baseline.** Predicting next-hour PM2.5 as equal to the current hour gives **MAE 10.4** and **RMSE 19.7**. Most hour-to-hour changes are small (median 0, IQR about −5 to +6), but the 95th percentile absolute jump is 35 μg/m³ — those spikes are where a model can beat persistence.


## Data citation

[Inter-Uni Datathon Stream 2 — Beijing Multi-Site Air Quality](https://www.kaggle.com/competitions/inter-uni-datathon-stream-2-beijing-multi-site-air-quality/overview)
