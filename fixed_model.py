"""
Hybrid non-recursive / recursive-history model:
1. Model A: weather-only LightGBM (test-available sensors)
2. Model B: weather + PM2.5 history (train uses true current_PM2_5; test uses
   the last train hour then the model's own previous predictions)
3. Station-specific bias correction and a light distribution-shift term
4. Boost on predicted extremes (>150)
"""
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

print("="*80)
print("IMPROVED NON-RECURSIVE PM2.5 PREDICTION")
print("="*80)

# Load data
print("\n[1/7] Loading data...")
train_path = DATA / 'train_cleaned.csv'
test_path = DATA / 'test_cleaned.csv'
if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
        'Missing data/train_cleaned.csv or data/test_cleaned.csv. Run Data Cleaning.ipynb first.'
    )
print(f"  Files: {train_path.name}, {test_path.name}")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
if 'current_PM2_5' in test.columns:
    test = test.drop(columns=['current_PM2_5'])

print(f"  Train: {len(train):,} rows")
print(f"  Test: {len(test):,} rows")

# Sort chronologically
train['datetime'] = pd.to_datetime(train['observation_timestamp'])
test['datetime'] = pd.to_datetime(test['observation_timestamp'])
train = train.sort_values(['station', 'datetime']).reset_index(drop=True)
test = test.sort_values(['station', 'datetime']).reset_index(drop=True)

# Wind direction mapping
WIND_DEGREES = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
    'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225,
    'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}

def build_features(df, is_train=True):
    """Enhanced feature engineering with extreme value focus"""
    df = df.copy()
    
    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofyear'] = df['datetime'].dt.dayofyear
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # Temporal indicators
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_heating_season'] = (
        df['month'].isin([12, 1, 2]) | 
        ((df['month'] == 11) & (df['day'] >= 15)) |
        ((df['month'] == 3) & (df['day'] <= 15))
    ).astype(int)
    
    # Wind features
    df['wind_degrees'] = df['wd'].astype(str).map(WIND_DEGREES).fillna(180)
    radians = np.deg2rad(df['wind_degrees'])
    df['wind_u'] = -df['WSPM'].fillna(0) * np.sin(radians)
    df['wind_v'] = -df['WSPM'].fillna(0) * np.cos(radians)
    df['is_stagnant'] = (df['WSPM'].fillna(0) < 1.5).astype(int)
    
    # Weather interactions
    df['dewpoint_spread'] = df['TEMP'] - df['DEWP']
    
    # Humidity
    a, b = 17.625, 243.04
    alpha_dew = (a * df['DEWP']) / (b + df['DEWP'])
    alpha_temp = (a * df['TEMP']) / (b + df['TEMP'])
    df['humidity'] = np.clip(100.0 * np.exp(alpha_dew - alpha_temp), 0.0, 100.0)
    df['humidity'].fillna(50, inplace=True)
    
    # Station encoding
    station_map = {s: i for i, s in enumerate(sorted(df['station'].unique()))}
    df['station_code'] = df['station'].map(station_map)
    
    # Fill missing weather
    weather_cols = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    for col in weather_cols:
        if col in df.columns:
            df[col] = df.groupby('station')[col].ffill()
            df[col].fillna(df[col].median(), inplace=True)
    
    # ===== NEW: EXTREME VALUE FEATURES =====
    print(f"  Building extreme value features...")
    
    # Pollutant ratios (PM2.5 correlates with these)
    df['PM10_squared'] = df['PM10'] ** 2
    df['CO_squared'] = df['CO'] ** 2
    df['pm10_co_interaction'] = df['PM10'] * df['CO'] / 1000
    
    # Stagnation risk (low wind + high humidity = high PM2.5)
    df['stagnation_risk'] = (df['WSPM'] < 2) & (df['humidity'] > 70)
    df['stagnation_risk'] = df['stagnation_risk'].astype(int)
    
    # Extreme pollution indicators
    df['pm10_high'] = (df['PM10'] > 150).astype(int)
    df['co_high'] = (df['CO'] > 2000).astype(int)
    df['no2_high'] = (df['NO2'] > 100).astype(int)
    
    # Weather history with more lags
    for col in ['PM10', 'CO', 'NO2', 'TEMP', 'WSPM', 'PRES', 'humidity']:
        if col in df.columns:
            for lag in [1, 3, 6, 12, 24]:
                df[f'{col}_lag_{lag}'] = df.groupby('station')[col].shift(lag)
            # Rolling stats
            df[f'{col}_roll_mean_6h'] = df.groupby('station')[col].transform(
                lambda x: x.rolling(6, min_periods=1).mean()
            )
            df[f'{col}_roll_max_6h'] = df.groupby('station')[col].transform(
                lambda x: x.rolling(6, min_periods=1).max()
            )
    
    # PM2.5 features if training
    if is_train and 'current_PM2_5' in df.columns:
        df['pm25_current'] = df['current_PM2_5']
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'pm25_lag_{lag}'] = df.groupby('station')['current_PM2_5'].shift(lag)
        df['pm25_roll_mean_6h'] = df.groupby('station')['current_PM2_5'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df['pm25_roll_max_6h'] = df.groupby('station')['current_PM2_5'].transform(
            lambda x: x.rolling(6, min_periods=1).max()
        )
        # Extreme PM2.5 indicator
        df['pm25_extreme'] = (df['current_PM2_5'] > 150).astype(int)
    
    return df

print("\n[2/7] Engineering features...")
train_fe = build_features(train, is_train=True)
test_fe = build_features(test, is_train=False)

# Feature lists
base_features = [
    'station_code', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'doy_sin', 'doy_cos', 'dayofweek', 'is_weekend', 'is_heating_season',
    'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
    'wind_u', 'wind_v', 'is_stagnant', 'dewpoint_spread', 'humidity',
    'PM10_squared', 'CO_squared', 'pm10_co_interaction',
    'stagnation_risk', 'pm10_high', 'co_high', 'no2_high'
]

# Add historical weather lags
for col in ['PM10', 'CO', 'NO2', 'TEMP', 'WSPM', 'PRES', 'humidity']:
    for lag in [1, 3, 6, 12, 24]:
        base_features.append(f'{col}_lag_{lag}')
    base_features.append(f'{col}_roll_mean_6h')
    base_features.append(f'{col}_roll_max_6h')

pm25_features = ['pm25_current'] + \
                [f'pm25_lag_{lag}' for lag in [1, 2, 3, 6, 12, 24]] + \
                ['pm25_roll_mean_6h', 'pm25_roll_max_6h', 'pm25_extreme']

print(f"  Base features: {len(base_features)}")
print(f"  PM2.5 features: {len(pm25_features)}")

# Prepare training data
X_train_A = train_fe[base_features].fillna(0)
X_train_B = train_fe[base_features + pm25_features].fillna(0)
y_train = train_fe['PM2_5_next_hour']

valid_idx = y_train.notna()
X_train_A = X_train_A[valid_idx]
X_train_B = X_train_B[valid_idx]
y_train = y_train[valid_idx]

# ===== TRAIN MODEL A (Weather only) =====
print("\n[3/7] Training Model A (Weather → PM2.5)...")

params_A = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.02,
    'num_leaves': 255,
    'max_depth': 12,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'lambda_l1': 1.0,
    'lambda_l2': 2.0,
    'verbose': -1,
    'seed': 42
}

dtrain_A = lgb.Dataset(X_train_A, label=y_train)
model_A = lgb.train(params_A, dtrain_A, num_boost_round=1200)

# ===== TRAIN MODEL B (With PM2.5 history) =====
print("\n[4/7] Training Model B (Weather + PM2.5)...")

params_B = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.02,
    'num_leaves': 255,
    'max_depth': 12,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'lambda_l1': 1.0,
    'lambda_l2': 2.0,
    'verbose': -1,
    'seed': 42
}

dtrain_B = lgb.Dataset(X_train_B, label=y_train)
model_B = lgb.train(params_B, dtrain_B, num_boost_round=1500)

# ===== COMPUTE STATION-SPECIFIC BIAS CORRECTION =====
print("\n[5/7] Computing station-specific bias corrections...")

# Calibrate on validation split (last 10% of training)
val_split_idx = int(len(train_fe) * 0.9)
val_data = train_fe.iloc[val_split_idx:].copy()

station_bias = {}
station_shift = {}

for station in train['station'].unique():
    train_station = train_fe[train_fe['station'] == station]
    train_mean = train_station['current_PM2_5'].mean()
    
    # Measure distribution shift (test period is later)
    # Use recent training data as proxy
    recent_mean = train_station.tail(1000)['current_PM2_5'].mean()
    station_shift[station] = recent_mean - train_mean
    
    # Measure bias on validation
    val_station = val_data[val_data['station'] == station]
    if len(val_station) > 100:
        X_val = val_station[base_features + pm25_features].fillna(0)
        y_val = val_station['PM2_5_next_hour']
        preds = model_B.predict(X_val)
        bias = np.mean(preds - y_val)
        station_bias[station] = bias
    else:
        station_bias[station] = 0
    
    print(f"  {station:20s}: bias={station_bias[station]:+.2f}, shift={station_shift[station]:+.2f}")

# ===== INFERENCE WITH FIXES =====
print("\n[6/7] Running inference with fixes...")

predictions = []
test_ids = []

train_tail = {}
for station in train['station'].unique():
    station_data = train_fe[train_fe['station'] == station].tail(48)  # 48h history
    train_tail[station] = station_data

for station in test['station'].unique():
    station_test = test_fe[test_fe['station'] == station].copy().reset_index(drop=True)
    print(f"  {station}: {len(station_test)} predictions")
    
    tail_pm25 = train_tail[station]['current_PM2_5'].values
    station_preds = []
    
    # Get bias correction for this station
    bias_correction = station_bias.get(station, 0)
    dist_shift = station_shift.get(station, 0)
    
    for i, row in station_test.iterrows():
        # Model A prediction (weather only)
        weather_feats = row[base_features].fillna(0).values.reshape(1, -1)
        pred_A = model_A.predict(weather_feats)[0]
        
        # Build PM2.5 history
        available_history = np.concatenate([tail_pm25, station_preds]) if len(station_preds) > 0 else tail_pm25
        
        pm25_current = available_history[-1] if len(available_history) >= 1 else tail_pm25[-1]
        pm25_lags = {
            1: available_history[-2] if len(available_history) >= 2 else tail_pm25[-2 if len(tail_pm25) >= 2 else -1],
            2: available_history[-3] if len(available_history) >= 3 else tail_pm25[-3 if len(tail_pm25) >= 3 else -1],
            3: available_history[-4] if len(available_history) >= 4 else tail_pm25[-4 if len(tail_pm25) >= 4 else -1],
            6: available_history[-7] if len(available_history) >= 7 else tail_pm25[-7 if len(tail_pm25) >= 7 else 0],
            12: available_history[-13] if len(available_history) >= 13 else tail_pm25[-13 if len(tail_pm25) >= 13 else 0],
            24: available_history[-25] if len(available_history) >= 25 else tail_pm25[-25 if len(tail_pm25) >= 25 else 0],
        }
        
        recent_history = available_history[-6:] if len(available_history) >= 6 else tail_pm25[-6:]
        pm25_roll_mean = np.mean(recent_history)
        pm25_roll_max = np.max(recent_history)
        pm25_extreme = int(pm25_current > 150)
        
        pm25_feats = np.array([
            pm25_current, pm25_lags[1], pm25_lags[2], pm25_lags[3], 
            pm25_lags[6], pm25_lags[12], pm25_lags[24],
            pm25_roll_mean, pm25_roll_max, pm25_extreme
        ])
        
        full_feats = np.concatenate([weather_feats[0], pm25_feats]).reshape(1, -1)
        pred_B = model_B.predict(full_feats)[0]
        
        # Ensemble with adaptive weighting
        weight_A = max(0.15, 1.0 - i / 300)
        weight_B = 1.0 - weight_A
        ensemble_pred = weight_A * pred_A + weight_B * pred_B
        
        # Apply bias correction
        corrected_pred = ensemble_pred - bias_correction
        
        # Apply distribution shift adjustment
        corrected_pred = corrected_pred + (dist_shift * 0.3)  # 30% of observed shift
        
        # Boost extreme predictions (fix underprediction)
        if corrected_pred > 150:
            boost_factor = 1.15  # Boost high predictions by 15%
            corrected_pred = corrected_pred * boost_factor
        
        final_pred = max(0, corrected_pred)
        
        station_preds.append(final_pred)
        predictions.append(final_pred)
        test_ids.append(row['id'])

# ===== EVALUATION =====
print("\n[7/7] Evaluating...")

submission = pd.DataFrame({
    'id': test_ids,
    'PM2_5_next_hour': predictions
})
# Restore official test id order
submission = pd.read_csv(test_path, usecols=['id']).merge(submission, on='id', how='left')

online_path = DATA / 'online.csv' if (DATA / 'online.csv').exists() else ROOT / 'online.csv'
if online_path.exists():
    online = pd.read_csv(online_path)
    online_labels = online[['id', 'PM2_5_next_hour']].rename(columns={'PM2_5_next_hour': 'actual'})
    results = submission.merge(online_labels, on='id', how='left')
    valid_results = results.dropna(subset=['actual'])
    rmse = np.sqrt(mean_squared_error(valid_results['actual'], valid_results['PM2_5_next_hour']))
    mae = np.mean(np.abs(valid_results['actual'] - valid_results['PM2_5_next_hour']))
    print("\n" + "="*80)
    print("RESULTS (online.csv, post-inference labels only)")
    print("="*80)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("="*80)
    print("\nPer-station RMSE:")
    for station in test['station'].unique():
        station_results = results[results['id'].isin(
            test[test['station'] == station]['id']
        )].dropna(subset=['actual'])
        if len(station_results) > 0:
            station_rmse = np.sqrt(mean_squared_error(
                station_results['actual'],
                station_results['PM2_5_next_hour']
            ))
            print(f"  {station:20s}: {station_rmse:.2f}")
else:
    print("\nNo online.csv present — skipped local label scoring.")

submission.to_csv(ROOT / 'submission_fixed.csv', index=False)
print("\nSaved", ROOT / 'submission_fixed.csv')
