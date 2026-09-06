import numpy as np
import pandas as pd
import holidays

def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['observation_timestamp'] = pd.to_datetime(df['observation_timestamp'])
    df = df.sort_values(['station', 'observation_timestamp']).reset_index(drop=True)
    
    # 1. Atmospheric Physics: RH and Ventilation
    e_s = 6.112 * np.exp((17.67 * df['TEMP']) / (df['TEMP'] + 243.5))
    e = 6.112 * np.exp((17.67 * df['DEWP']) / (df['DEWP'] + 243.5))
    df['RH'] = np.clip(100.0 * (e / e_s), 0, 100)
    df['dew_spread'] = df['TEMP'] - df['DEWP']
    df['ventilation_idx'] = df['WSPM'] * np.maximum(df['dew_spread'], 0.1)

    # 2. Wind Vector Decomposition
    wd_angles = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    rad = np.deg2rad(df['wd'].map(wd_angles).fillna(0))
    df['wind_u'] = -df['WSPM'] * np.sin(rad)
    df['wind_v'] = -df['WSPM'] * np.cos(rad)

    # Southerly transport intensity (Positive wind_v pushes dirty air north)
    df['southerly_wind_flux'] = np.maximum(df['wind_v'], 0) * df['PM10']

    # 3. Co-Pollutant Chemical Interactions (Anchored on PM10 and Gases)
    df['secondary_aerosol_potential'] = (df['SO2'] * df['NO2']) / (df['TEMP'] + 40)
    df['combustion_index'] = df['CO'] * df['NO2']
    df['so2_to_no2'] = df['SO2'] / (df['NO2'] + 1e-4)
    df['pm10_to_co'] = df['PM10'] / (df['CO'] + 1e-4)
    df['no2_to_pm10'] = df['NO2'] / (df['PM10'] + 1e-4)

    # 4. Grouped Lag & Momentum Features
    grp = df.groupby('station')
    
    # Generate lags for key available predictors
    lagged_vars = ['PM10', 'CO', 'NO2', 'WSPM', 'PRES', 'TEMP']
    for var in lagged_vars:
        for lag in [1, 2, 3, 6]:
            df[f'{var}_lag_{lag}'] = grp[var].shift(lag)

    # Differences / Accelerations
    df['pm10_diff_1'] = df['PM10'] - df['PM10_lag_1']
    df['pm10_diff_2'] = df['PM10_lag_1'] - df['PM10_lag_2']
    df['pm10_accel'] = df['pm10_diff_1'] - df['pm10_diff_2']
    
    df['co_diff_1'] = df['CO'] - df['CO_lag_1']
    df['pres_diff_3'] = df['PRES'] - df['PRES_lag_3']
    df['temp_diff_3'] = df['TEMP'] - df['TEMP_lag_3']

    # Rolling windows for PM10 and CO
    for w in [3, 6, 24]:
        df[f'pm10_roll_mean_{w}'] = grp['PM10'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f'pm10_roll_std_{w}'] = grp['PM10'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        df[f'co_roll_mean_{w}'] = grp['CO'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    # 5. Cyclical Time Features
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # 6. Cumulative Rain & Wind Runs
    is_rain = (df['RAIN'] > 0).astype(int)
    rain_block = (is_rain == 0).cumsum()
    df['Ir'] = is_rain.groupby([df['station'], rain_block]).cumsum()

    direction_shifted = (df['wd'] != df.groupby('station')['wd'].shift(1)).astype(int)
    wind_block = direction_shifted.groupby(df['station']).cumsum()
    df['Iws'] = df.groupby(['station', wind_block])['WSPM'].cumsum()

    df['rain_roll_12h_sum'] = grp['RAIN'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).sum()
    )

    # 7. Heating Season & Calendar Context
    is_nov_heating = (df['month'] == 11) & (df['day'] >= 15)
    is_dec_jan_feb = df['month'].isin([12, 1, 2])
    is_mar_heating = (df['month'] == 3) & (df['day'] <= 15)
    df['is_heating_season'] = (is_nov_heating | is_dec_jan_feb | is_mar_heating).astype(int)

    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        return 'Autumn'
    df['season'] = df['month'].apply(get_season).astype('category')

    cn_holidays = holidays.China(years=range(2013, 2018))
    df['is_holiday'] = df['observation_timestamp'].dt.date.isin(cn_holidays).astype(int)

    return df

def build_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Geographic Micro-Regions
    station_regions = {
        'Dingling': 'North', 'Huairou': 'North', 'Changping': 'Northwest',
        'Dongsi': 'Core_East', 'Guanyuan': 'Core_West', 'Aotizhongxin': 'Core_North',
        'Tiantan': 'Core_South', 'Wanliu': 'Core_Northwest', 'Wanshouxigong': 'Core_Southwest',
        'Nongzhanguan': 'Core_East', 'Nansanhuan': 'South', 'Gucheng': 'Southwest_Industrial',
        'Shunyi': 'Northeast'
    }
    df['station_region'] = df['station'].map(station_regions).astype('category')

    # 2. Macro-Gradient Vulnerability Score
    north_south_score = {
        'North': -1.0, 'Northwest': -0.7, 'Northeast': -0.5,
        'Core_North': 0.0, 'Core_East': 0.1, 'Core_West': 0.0,
        'Core_Northwest': 0.0, 'Core_South': 0.4, 'Core_Southwest': 0.5,
        'Southwest_Industrial': 0.8, 'South': 1.0
    }
    df['station_regional_vulnerability'] = df['station_region'].map(north_south_score)
    df = df.sort_values(['station', 'observation_timestamp']).reset_index(drop=True)

    # 3. Spatial Aggregations
    time_grp = df.groupby('observation_timestamp')
    spatial_pm10 = time_grp['PM10'].agg(
        city_pm10_mean='mean',
        city_pm10_std='std',
        city_pm10_max='max',
        city_pm10_min='min'
    ).reset_index()
    
    df = df.merge(spatial_pm10, on='observation_timestamp', how='left')

    # Local vs. Regional Divergence
    df['station_dispersion_from_city'] = df['PM10'] - df['city_pm10_mean']
    df['pm10_to_city_ratio'] = df['PM10'] / (df['city_pm10_mean'] + 1e-4)

    return df

""" Execution """
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 1. Combine train and test into continuous timeline
all_data = pd.concat([train_df, test_df], ignore_index=True)
all_data['observation_timestamp'] = pd.to_datetime(all_data['observation_timestamp'])
all_data = all_data.sort_values(['station', 'observation_timestamp']).reset_index(drop=True)

# 2. Compute all features
all_data = build_advanced_features(all_data)
all_data = build_spatial_features(all_data)

# 3. Split back using original IDs
train_features = all_data[all_data['id'].isin(train_df['id'])].copy()
test_features = all_data[all_data['id'].isin(test_df['id'])].copy()

train_features.to_csv("train_featured.csv", index=False)
test_features.to_csv("test_featured.csv", index=False)
print("Done.")