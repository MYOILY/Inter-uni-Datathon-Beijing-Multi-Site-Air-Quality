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

    # 3. Co-Pollutant chemical interaction and ratios
    df['secondary_aerosol_potential'] = (df['SO2'] * df['NO2']) / (df['TEMP'] + 40)
    df['combustion_index'] = df['CO'] * df['NO2']
    df['coarse_fine_split'] = (df['PM10'] - df['current_PM2_5']).clip(lower=0)
    df['pm_ratio'] = df['current_PM2_5'] / (df['PM10'] + 1e-4)
    df['pm_to_co'] = df['current_PM2_5'] / (df['CO'] + 1e-4)
    df['so2_to_no2'] = df['SO2'] / (df['NO2'] + 1e-4)

    # 4. Grouped Lag and Momentum Features (Calculus works lol)
    grp = df.groupby('station')
    for lag in [1, 2, 3, 6]:
        df[f'pm25_lag_{lag}'] = grp['current_PM2_5'].shift(lag)
        df[f'wspm_lag_{lag}'] = grp['WSPM'].shift(lag)
        df[f'pres_lag_{lag}'] = grp['PRES'].shift(lag)
    # Differences
    df['pm25_diff_1'] = df['current_PM2_5'] - df['pm25_lag_1']
    df['pm25_diff_2'] = df['pm25_lag_1'] - df['pm25_lag_2']
    df['pm25_accel'] = df['pm25_diff_1'] - df['pm25_diff_2']
    df['pres_diff_3'] = df['PRES'] - df['pres_lag_3']
    # Rolling windows (shifted by 1 to prevent target leakage)
    for w in [3, 6, 24]:
        df[f'pm25_roll_mean_{w}'] = grp['current_PM2_5'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f'pm25_roll_std_{w}'] = grp['current_PM2_5'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())

    # 5. Cyclical Time Features
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # 6. Cumulative Context
    # a. Consecutive Rain Hours (Washout accumulation)
    is_rain = (df['RAIN'] > 0).astype(int)
    rain_block = (is_rain == 0).cumsum()
    df['Ir'] = is_rain.groupby([df['station'], rain_block]).cumsum()

    # b. Cumulative Wind Run along constant direction (Run-length dispersion)
    direction_shifted = (df['wd'] != df.groupby('station')['wd'].shift(1)).astype(int)
    wind_block = direction_shifted.groupby(df['station']).cumsum()
    df['Iws'] = df.groupby(['station', wind_block])['WSPM'].cumsum()

    # c. Rolling 12-hour total precipitation
    df['rain_roll_12h_sum'] = df.groupby('station')['RAIN'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).sum()
    )

    # 7. Additional Contextual Features
    # a. Central Heating Season Flag (Nov 15 - Mar 15)
    is_nov_heating = (df['month'] == 11) & (df['day'] >= 15)
    is_dec_jan_feb = df['month'].isin([12, 1, 2])
    is_mar_heating = (df['month'] == 3) & (df['day'] <= 15)
    df['is_heating_season'] = (is_nov_heating | is_dec_jan_feb | is_mar_heating).astype(int)

    # b. Season Category
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        return 'Autumn'
    df['season'] = df['month'].apply(get_season).astype('category')

    # c. Public Holiday Calendar (National Day Golden Week & New Year)
    cn_holidays = holidays.China(years=range(2013, 2017))
    df['is_holiday'] = df['observation_timestamp'].dt.date.isin(cn_holidays).astype(int)

    return df

def build_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Station directions to Beijing
    station_regions = {
        # Northern mountain border (clean baseline)
        'Dingling': 'North',
        'Huairou': 'North',
        'Changping': 'Northwest',
        
        # Urban Core / Traffic ring roads
        'Dongsi': 'Core_East',
        'Guanyuan': 'Core_West',
        'Aotizhongxin': 'Core_North',
        'Tiantan': 'Core_South',
        'Wanliu': 'Core_Northwest',
        'Wanshouxigong': 'Core_Southwest',
        'Nongzhanguan': 'Core_East',
        
        # Southern industrial gateway (most polluted)
        'Nansanhuan': 'South',
        'Gucheng': 'Southwest_Industrial',
        
        # Eastern suburbs
        'Shunyi': 'Northeast'
    }
    df['station_region'] = df['station'].map(station_regions).astype('category')

    # 2. Simplified macro-gradient: 1 = South/Industrial, 0 = Core, -1 = Mountain North
    north_south_score = {
        'North': -1.0, 'Northwest': -0.7, 'Northeast': -0.5,
        'Core_North': 0.0, 'Core_East': 0.1, 'Core_West': 0.0,
        'Core_Northwest': 0.0, 'Core_South': 0.4, 'Core_Southwest': 0.5,
        'Southwest_Industrial': 0.8, 'South': 1.0
    }
    df['station_regional_vulnerability'] = df['station_region'].map(north_south_score)
    df = df.sort_values(['station', 'observation_timestamp']).reset_index(drop=True)

    # 3. Spatial Network Aggregations (Timestamp Level)
    time_grp = df.groupby('observation_timestamp')
    spatial_pm = time_grp['current_PM2_5'].agg(city_pm25_mean='mean', city_pm25_std='std').reset_index()
    df = df.merge(spatial_pm, on='observation_timestamp', how='left')
    df['station_dispersion_from_city'] = df['current_PM2_5'] - df['city_pm25_mean']

    return df

df = pd.read_csv("train.csv")
df = build_advanced_features(df)
df = build_spatial_features(df)
df.to_csv("train_featured.csv", index=False)
print('Done')