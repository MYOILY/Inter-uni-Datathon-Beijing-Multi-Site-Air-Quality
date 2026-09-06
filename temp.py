from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT_DEFAULT = # put path in
LAGS = (1, 2, 3, 6, 12, 24)
ONE_HOUR_NS = 3_600_000_000_000  # 1 hour in nanoseconds
WIND_DEGREES = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
    'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225,
    'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}
WEATHER_COLUMNS = ('PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM')


def add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'observation_timestamp' in out:
        out['timestamp'] = pd.to_datetime(out['observation_timestamp'], errors='raise')
    else:
        out['timestamp'] = pd.to_datetime(out[['year', 'month', 'day', 'hour']], errors='raise')
    return out


def canonicalize(df: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    out = add_timestamp(df).sort_values(['station', 'timestamp']).reset_index(drop=True)
    out['station_code'] = pd.Categorical(out['station'], categories=stations).codes.astype(float)
    out['wind_degrees'] = out['wd'].astype(str).map(WIND_DEGREES).astype(float)
    return out


def training_matrix(train: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Vectorised training features with identical definitions to inference."""
    d = train.sort_values(['station', 'timestamp']).reset_index(drop=True).copy()
    t = d.timestamp
    hour, doy = d.hour.astype(float), t.dt.dayofyear.astype(float)
    f = pd.DataFrame(index=d.index)
    f['station_code'] = d.station_code
    f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    f['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    f['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    f['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    f['dayofweek'] = t.dt.dayofweek.astype(float)
    f['is_weekend'] = (t.dt.dayofweek >= 5).astype(float)
    f['is_heating_season'] = (
        t.dt.month.isin([12, 1, 2])
        | ((t.dt.month == 11) & (t.dt.day >= 15))
        | ((t.dt.month == 3) & (t.dt.day <= 15))
    ).astype(float)
    f['current_pm25'] = d.current_PM2_5.astype(float)
    radians = np.deg2rad(d.wind_degrees)
    f['wind_u'] = -d.WSPM * np.sin(radians)
    f['wind_v'] = -d.WSPM * np.cos(radians)
    for col in WEATHER_COLUMNS:
        f[col] = d[col].astype(float)
        f[f'{col}_missing'] = d[col].isna().astype(float)
    f['dewpoint_spread'] = d.TEMP - d.DEWP
    f['pm25_pm10_ratio'] = d.current_PM2_5 / (d.PM10 + 1e-3)
    f['is_stagnant'] = (d.WSPM < 1.5).astype(float)
    grouped = d.groupby('station', observed=True, sort=False)
    lag_columns = []
    for lag in LAGS:
        values, prior_t = grouped.current_PM2_5.shift(lag), grouped.timestamp.shift(lag)
        values = values.where((d.timestamp - prior_t) == pd.Timedelta(hours=lag))
        name = f'pm25_lag_{lag}'
        f[name] = values
        f[f'{name}_missing'] = values.isna().astype(float)
        lag_columns.append(name)
    f['pm25_diff_1h'] = d.current_PM2_5 - f['pm25_lag_1']
    f['pm25_diff_3h'] = d.current_PM2_5 - f['pm25_lag_3']
    f['pm25_hist_mean'] = f[lag_columns].mean(axis=1)
    f['pm25_hist_std'] = f[lag_columns].std(axis=1, ddof=0)
    return f, (d.PM2_5_next_hour - d.current_PM2_5).to_numpy(dtype=float)


def train_model(train: pd.DataFrame) -> tuple[lgb.Booster, list[str]]:
    X, y = training_matrix(train)
    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.035,
        'num_leaves': 63, 'min_data_in_leaf': 80, 'feature_fraction': 0.85,
        'bagging_fraction': 0.85, 'bagging_freq': 1, 'lambda_l2': 1.0,
        'seed': 42, 'feature_fraction_seed': 42, 'bagging_seed': 42,
        'verbosity': -1, 'num_threads': -1,
    }
    model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=900)
    return model, list(X.columns)


def prepare_static_matrix(df: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Precompute all static features (weather, calendar, wind) into a NumPy matrix."""
    n = len(df)
    mat = np.full((n, len(feature_names)), np.nan, dtype=np.float64)
    feat_idx = {name: i for i, name in enumerate(feature_names)}

    t = df['timestamp']
    hour = df['hour'].astype(float).to_numpy()
    doy = t.dt.dayofyear.astype(float).to_numpy()
    dow = t.dt.dayofweek.astype(float).to_numpy()
    month = t.dt.month.to_numpy()
    day = t.dt.day.to_numpy()

    mat[:, feat_idx['station_code']] = df['station_code'].to_numpy(dtype=np.float64)
    mat[:, feat_idx['hour_sin']] = np.sin(2 * np.pi * hour / 24)
    mat[:, feat_idx['hour_cos']] = np.cos(2 * np.pi * hour / 24)
    mat[:, feat_idx['doy_sin']] = np.sin(2 * np.pi * doy / 365.25)
    mat[:, feat_idx['doy_cos']] = np.cos(2 * np.pi * doy / 365.25)
    mat[:, feat_idx['dayofweek']] = dow
    mat[:, feat_idx['is_weekend']] = (dow >= 5).astype(np.float64)

    is_heating = (
        np.isin(month, [12, 1, 2])
        | ((month == 11) & (day >= 15))
        | ((month == 3) & (day <= 15))
    ).astype(np.float64)
    mat[:, feat_idx['is_heating_season']] = is_heating

    radians = np.deg2rad(df['wind_degrees'].to_numpy(dtype=np.float64))
    wspm = df['WSPM'].to_numpy(dtype=np.float64)
    mat[:, feat_idx['wind_u']] = -wspm * np.sin(radians)
    mat[:, feat_idx['wind_v']] = -wspm * np.cos(radians)

    for col in WEATHER_COLUMNS:
        vals = df[col].to_numpy(dtype=np.float64) if col in df.columns else np.full(n, np.nan)
        mat[:, feat_idx[col]] = vals
        mat[:, feat_idx[f'{col}_missing']] = np.isnan(vals).astype(np.float64)

    temp = df['TEMP'].to_numpy(dtype=np.float64)
    dewp = df['DEWP'].to_numpy(dtype=np.float64)
    mat[:, feat_idx['dewpoint_spread']] = temp - dewp
    mat[:, feat_idx['is_stagnant']] = (wspm < 1.5).astype(np.float64)

    pm10 = df['PM10'].to_numpy(dtype=np.float64) if 'PM10' in df.columns else np.full(n, np.nan)
    return mat, pm10


def fill_dynamic_features(
    row_vec: np.ndarray,
    current: float,
    pm10_val: float,
    t_ns: int,
    history: dict[int, float],
    feat_idx: dict[str, int],
    lag_indices: list[tuple[int, int, int]],
) -> None:
    """In-place population of recursive state and historical lag features."""
    row_vec[feat_idx['current_pm25']] = current
    row_vec[feat_idx['pm25_pm10_ratio']] = (
        current / (pm10_val + 1e-3) if not np.isnan(pm10_val) else np.nan
    )

    lag_vals = []
    for lag, lag_col, lag_miss_col in lag_indices:
        val = history.get(t_ns - lag * ONE_HOUR_NS, np.nan)
        row_vec[lag_col] = val
        if np.isnan(val):
            row_vec[lag_miss_col] = 1.0
        else:
            row_vec[lag_miss_col] = 0.0
            lag_vals.append(val)

    lag_1 = row_vec[feat_idx['pm25_lag_1']]
    row_vec[feat_idx['pm25_diff_1h']] = (current - lag_1) if not np.isnan(lag_1) else np.nan

    lag_3 = row_vec[feat_idx['pm25_lag_3']]
    row_vec[feat_idx['pm25_diff_3h']] = (current - lag_3) if not np.isnan(lag_3) else np.nan

    if lag_vals:
        mean_val = sum(lag_vals) / len(lag_vals)
        row_vec[feat_idx['pm25_hist_mean']] = mean_val
        var_val = sum((v - mean_val) ** 2 for v in lag_vals) / len(lag_vals)
        row_vec[feat_idx['pm25_hist_std']] = np.sqrt(var_val)
    else:
        row_vec[feat_idx['pm25_hist_mean']] = np.nan
        row_vec[feat_idx['pm25_hist_std']] = np.nan


def recursive_predict(
    model: lgb.Booster,
    feature_names: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Optimized recursive inference executing in ~1-2 seconds with zero Pandas overhead."""
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    lag_indices = [
        (lag, feat_idx[f'pm25_lag_{lag}'], feat_idx[f'pm25_lag_{lag}_missing'])
        for lag in LAGS
    ]

    out_ids: list = []
    out_preds: list[float] = []
    out_stations: list[str] = []
    out_timestamps: list = []
    out_recursive_current: list[float] = []
    bridge_rows = 0

    train_by_station = {s: b.sort_values('timestamp') for s, b in train.groupby('station', observed=True)}

    for station, test_block in test.groupby('station', observed=True, sort=False):
        hist_train = train_by_station[station]

        # Fast dictionary initialization with integer nanosecond timestamps
        hist_ts = hist_train['timestamp'].values.astype('int64')
        hist_pm25 = hist_train['current_PM2_5'].to_numpy(dtype=np.float64)
        history = dict(zip(hist_ts, hist_pm25))

        # Warm-up forecast on the final training observation
        last_train_row = hist_train.iloc[[-1]]
        warm_mat, warm_pm10 = prepare_static_matrix(last_train_row, feature_names)
        last_current = float(hist_train['current_PM2_5'].iloc[-1])
        last_ts_ns = int(hist_train['timestamp'].iloc[-1].value)

        fill_dynamic_features(
            warm_mat[0], last_current, warm_pm10[0], last_ts_ns,
            history, feat_idx, lag_indices
        )
        warm_delta = float(model.predict(warm_mat, num_threads=1)[0])
        warm_next = max(0.0, last_current + warm_delta)
        history[last_ts_ns + ONE_HOUR_NS] = warm_next
        last_state = warm_next

        # Precompute static feature matrix for all rows of this station
        test_mat, test_pm10 = prepare_static_matrix(test_block, feature_names)
        test_ids = test_block['id'].to_numpy()
        test_timestamps = test_block['timestamp'].to_numpy()
        test_ts_ns = test_block['timestamp'].values.astype('int64')

        n_rows = len(test_block)
        for i in range(n_rows):
            t_ns = int(test_ts_ns[i])
            current = history.get(t_ns, np.nan)
            if np.isnan(current):
                current = last_state
                history[t_ns] = current
                bridge_rows += 1

            fill_dynamic_features(
                test_mat[i], current, test_pm10[i], t_ns,
                history, feat_idx, lag_indices
            )

            # Direct 2D numpy slice inference without thread contention
            delta = float(model.predict(test_mat[i : i + 1], num_threads=1)[0])
            next_pm25 = max(0.0, current + delta)

            out_ids.append(test_ids[i])
            out_preds.append(next_pm25)
            out_stations.append(station)
            out_timestamps.append(test_timestamps[i])
            out_recursive_current.append(current)

            history[t_ns + ONE_HOUR_NS] = next_pm25
            last_state = next_pm25

    preds_df = pd.DataFrame({
        'id': out_ids,
        'PM2_5_next_hour': out_preds,
        'station': out_stations,
        'timestamp': out_timestamps,
        'recursive_current_pm25': out_recursive_current,
    })
    return preds_df, bridge_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=Path('recursive_lightgbm_output'))
    args = parser.parse_args()
    data_dir, out = args.data_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_train = pd.read_csv(data_dir / 'train.csv')
    raw_test = pd.read_csv(data_dir / 'test(1).csv')
    print('Loaded input data', flush=True)
    if 'current_PM2_5' in raw_test.columns:
        raise ValueError('Expected the bare official schema test(1).csv without current_PM2_5.')
    stations = sorted(raw_train.station.unique())
    train, test = canonicalize(raw_train, stations), canonicalize(raw_test, stations)
    if train.duplicated(['station', 'timestamp']).any() or test.duplicated(['station', 'timestamp']).any():
        raise ValueError('Station/timestamp keys must be unique.')
    print('Training LightGBM', flush=True)
    model, features = train_model(train)
    print('Running recursive inference', flush=True)
    preds, bridge_rows = recursive_predict(model, features, train, test)
    print('Writing prediction artefacts', flush=True)
    original_order = raw_test[['id']].merge(preds[['id', 'PM2_5_next_hour']], on='id', how='left', validate='one_to_one')
    if len(original_order) != len(raw_test) or original_order.PM2_5_next_hour.isna().any() or (original_order.PM2_5_next_hour < 0).any():
        raise ValueError('Submission validation failed.')
    original_order.to_csv(out / 'submission_recursive_lightgbm.csv', index=False)
    preds.to_csv(out / 'recursive_predictions_with_state.csv', index=False)
    model.save_model(str(out / 'recursive_lightgbm_model.txt'))

    # Strictly post-inference evaluation: labels only, joined by official IDs.
    online = pd.read_csv(data_dir / 'online.csv', usecols=['id', 'PM2_5_next_hour'])
    print('Scoring against online labels', flush=True)
    truth = online.drop_duplicates('id').rename(columns={'PM2_5_next_hour': 'actual_next_hour'})
    scored = preds.merge(truth, on='id', how='left', validate='one_to_one')
    valid = scored.dropna(subset=['actual_next_hour'])
    rmse = float(np.sqrt(np.mean((valid.PM2_5_next_hour - valid.actual_next_hour) ** 2)))
    mae = float(np.mean(np.abs(valid.PM2_5_next_hour - valid.actual_next_hour)))
    scored['squared_error'] = (scored.PM2_5_next_hour - scored.actual_next_hour) ** 2
    scored.to_csv(out / 'recursive_online_validation_predictions.csv', index=False)
    metrics = {
        'metric': 'RMSE', 'rmse': rmse, 'mae_diagnostic': mae,
        'scored_rows': int(len(valid)), 'unscored_rows_missing_online_label': int(len(scored) - len(valid)),
        'submission_rows': int(len(original_order)), 'recursive_gap_bridge_rows': int(bridge_rows),
        'training_rows': int(len(train)), 'test_file': 'test(1).csv',
        'online_usage': 'labels only after recursive inference; excluded from fitting and state updates',
        'feature_count': len(features),
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()