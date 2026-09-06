"""Leakage-safe features for next-hour PM2.5 when test has no current_PM2_5.

Stage 1 predicts current PM2.5 from sensors that exist at test time (PM10, CO,
NO2, weather, station, calendar). Stage 2 uses that proxy plus PM10-centric
lags, spatial PM10, heating season and wind physics.

Fit stage 1 on labelled training rows only. Never use the next-hour target as
a stage-1 input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "PM2_5_next_hour"
TIME_COL = "observation_timestamp"

WD_ANGLES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

STATION_REGIONS = {
    "Dingling": "North", "Huairou": "North", "Changping": "Northwest",
    "Dongsi": "Core_East", "Guanyuan": "Core_West", "Aotizhongxin": "Core_North",
    "Tiantan": "Core_South", "Wanliu": "Core_Northwest",
    "Wanshouxigong": "Core_Southwest", "Nongzhanguan": "Core_East",
    "Nansanhuan": "South", "Gucheng": "Southwest_Industrial",
    "Shunyi": "Northeast",
}

NORTH_SOUTH_SCORE = {
    "North": -1.0, "Northwest": -0.7, "Northeast": -0.5,
    "Core_North": 0.0, "Core_East": 0.1, "Core_West": 0.0,
    "Core_Northwest": 0.0, "Core_South": 0.4, "Core_Southwest": 0.5,
    "Southwest_Industrial": 0.8, "South": 1.0,
}

STAGE1_NUM = [
    "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "hour", "month", "day",
    "RH", "dew_spread", "ventilation_idx", "wind_u", "wind_v",
    "is_heating_season", "city_pm10_mean", "city_co_mean", "city_no2_mean",
    "clim_pm25_from_pm10", "clim_pm25_level", "pm10_anomaly",
    "combustion_index", "pm10_x_heating", "pm10_over_wind", "local_haze", "city_haze",
    "log_pm10", "sqrt_co", "inversion", "pres_anom",
    "PM10_lag_1", "PM10_lag_3", "PM10_lag_6", "PM10_lag_12", "PM10_lag_24",
    "pm10_north_mean", "pm10_south_mean", "pm10_ns_contrast",
    "pm10_others_mean", "pm10_others_max", "co_others_mean",
    "pm10_idw_3", "co_idw_3", "city_pm10_lag_1",
]
STAGE1_CAT = ["station", "wd"]

DROP_FROM_FEATURES = {TARGET, "id", TIME_COL, "current_PM2_5", "year", "_bridge", "_split"}

NORTH_STATIONS = ["Dingling", "Huairou", "Changping", "Shunyi"]
SOUTH_STATIONS = ["Tiantan", "Wanshouxigong", "Gucheng", "Nansanhuan"]

# PRSA Beijing site coordinates (lat, lon) for inverse-distance spatial features.
STATION_LATLON = {
    "Aotizhongxin": (40.003, 116.397),
    "Changping": (40.217, 116.226),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}

INTERPOLATE_COLS = [
    "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
]


def interpolate_then_mean_fill(train: pd.DataFrame, test: pd.DataFrame):
    """Station-wise linear interpolation (up to 6h), city-hour median, then train means.

    Test sensors can ffill from the last train hours at the same station. True
    current_PM2_5 is interpolated on train only and never written onto test.
    """
    orig_test_ids = test["id"].to_numpy()
    orig_train_ids = train["id"].to_numpy()
    train = train.copy()
    test = test.copy()
    train[TIME_COL] = pd.to_datetime(train[TIME_COL])
    test[TIME_COL] = pd.to_datetime(test[TIME_COL])
    train["wd"] = train["wd"].fillna("Missing")
    test["wd"] = test["wd"].fillna("Missing")

    sensor_cols = [c for c in INTERPOLATE_COLS if c in train.columns]

    def _interp_cols(df, cols):
        df = df.sort_values(["station", TIME_COL])
        df[cols] = df.groupby("station")[cols].transform(
            lambda s: s.interpolate(method="linear", limit=6, limit_area="inside")
        )
        df[cols] = df.groupby("station")[cols].ffill(limit=6)
        return df

    train = _interp_cols(train, sensor_cols)
    if "current_PM2_5" in train.columns:
        train = _interp_cols(train, ["current_PM2_5"])

    # Carry the last 12 train hours so the first test gaps can ffill from history.
    tail = train.sort_values(["station", TIME_COL]).groupby("station", sort=False).tail(12)
    tail = tail[["id", "station", TIME_COL] + sensor_cols].copy()
    tail["_bridge"] = 1
    test["_bridge"] = 0
    bridged = pd.concat([tail, test], ignore_index=True, sort=False)
    bridged = _interp_cols(bridged, sensor_cols)
    test = bridged.loc[bridged["_bridge"] == 0].drop(columns=["_bridge"])
    train = train.drop(columns=["_bridge"], errors="ignore")
    test = test.drop(columns=["_bridge"], errors="ignore")

    def _city_hour_fill(df, cols):
        for c in cols:
            df[c] = df.groupby(TIME_COL)[c].transform(lambda s: s.fillna(s.median()))
        return df

    train = _city_hour_fill(train, sensor_cols)
    test = _city_hour_fill(test, sensor_cols)
    means = train[sensor_cols].mean()
    train[sensor_cols] = train[sensor_cols].fillna(means)
    test[sensor_cols] = test[sensor_cols].fillna(means)
    if "current_PM2_5" in train.columns:
        train["current_PM2_5"] = train["current_PM2_5"].fillna(train["current_PM2_5"].mean())

    train = train.set_index("id").loc[orig_train_ids].reset_index()
    test = test.set_index("id").loc[orig_test_ids].reset_index()
    return train.reset_index(drop=True), test


def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df.sort_values(["station", TIME_COL]).reset_index(drop=True)


def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_time(df)

    e_s = 6.112 * np.exp((17.67 * df["TEMP"]) / (df["TEMP"] + 243.5))
    e = 6.112 * np.exp((17.67 * df["DEWP"]) / (df["DEWP"] + 243.5))
    df["RH"] = np.clip(100.0 * (e / e_s), 0, 100)
    df["dew_spread"] = df["TEMP"] - df["DEWP"]
    df["ventilation_idx"] = df["WSPM"] * np.maximum(df["dew_spread"], 0.1)

    rad = np.deg2rad(df["wd"].map(WD_ANGLES).fillna(0))
    df["wind_u"] = -df["WSPM"] * np.sin(rad)
    df["wind_v"] = -df["WSPM"] * np.cos(rad)
    df["southerly_wind_flux"] = np.maximum(df["wind_v"], 0) * df["PM10"]

    df["secondary_aerosol_potential"] = (df["SO2"] * df["NO2"]) / (df["TEMP"] + 40)
    df["combustion_index"] = df["CO"] * df["NO2"]
    df["so2_to_no2"] = df["SO2"] / (df["NO2"] + 1e-4)
    df["pm10_to_co"] = df["PM10"] / (df["CO"] + 1e-4)
    df["no2_to_pm10"] = df["NO2"] / (df["PM10"] + 1e-4)

    grp = df.groupby("station", sort=False)
    for var in ["PM10", "CO", "NO2", "WSPM", "PRES", "TEMP"]:
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            df[f"{var}_lag_{lag}"] = grp[var].shift(lag)

    df["pm10_diff_1"] = df["PM10"] - df["PM10_lag_1"]
    df["pm10_diff_3"] = df["PM10"] - df["PM10_lag_3"]
    df["pm10_accel"] = (df["PM10"] - df["PM10_lag_1"]) - (df["PM10_lag_1"] - df["PM10_lag_2"])
    df["co_diff_1"] = df["CO"] - df["CO_lag_1"]
    df["pres_diff_3"] = df["PRES"] - df["PRES_lag_3"]
    df["temp_diff_3"] = df["TEMP"] - df["TEMP_lag_3"]

    station_key = df["station"]
    for w in [3, 6, 12, 24]:
        past = grp["PM10"].shift(1)
        df[f"pm10_roll_mean_{w}"] = past.groupby(station_key).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"pm10_roll_std_{w}"] = past.groupby(station_key).rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
        past_co = grp["CO"].shift(1)
        df[f"co_roll_mean_{w}"] = past_co.groupby(station_key).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow"] = df[TIME_COL].dt.dayofweek

    is_rain = (df["RAIN"].fillna(0) > 0).astype(int)
    rain_block = (is_rain == 0).cumsum()
    df["Ir"] = is_rain.groupby([df["station"], rain_block]).cumsum()
    direction_shifted = (df["wd"] != grp["wd"].shift(1)).astype(int)
    wind_block = direction_shifted.groupby(df["station"]).cumsum()
    df["Iws"] = df.groupby(["station", wind_block])["WSPM"].cumsum()
    df["rain_roll_12h_sum"] = grp["RAIN"].transform(lambda x: x.shift(1).rolling(12, min_periods=1).sum())

    is_nov = (df["month"] == 11) & (df["day"] >= 15)
    is_djf = df["month"].isin([12, 1, 2])
    is_mar = (df["month"] == 3) & (df["day"] <= 15)
    df["is_heating_season"] = (is_nov | is_djf | is_mar).astype(int)
    df["is_weekend"] = (df[TIME_COL].dt.dayofweek >= 5).astype(int)
    df["pm10_x_heating"] = df["PM10"] * df["is_heating_season"]
    df["co_x_heating"] = df["CO"] * df["is_heating_season"]
    df["pred_calm"] = ((df["WSPM"] < 1.0) & (df["dew_spread"] < 3)).astype(int)
    df["pm10_over_wind"] = df["PM10"] / (df["WSPM"] + 0.15)
    df["local_haze"] = ((df["PM10"] > 150) & (df["WSPM"] < 1.5)).astype(int)
    df["log_pm10"] = np.log1p(df["PM10"].clip(lower=0))
    df["sqrt_co"] = np.sqrt(df["CO"].clip(lower=0))
    past_pres = grp["PRES"].shift(1)
    df["pres_roll_mean_24"] = (
        past_pres.groupby(df["station"]).rolling(24, min_periods=6).mean().reset_index(level=0, drop=True)
    )
    df["pres_anom"] = df["PRES"] - df["pres_roll_mean_24"]
    night = df["hour"].isin([21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
    df["inversion"] = (night & (df["WSPM"] < 1.2) & (df["pres_anom"].fillna(0) > 0)).astype(int)

    season = np.where(df["month"].isin([12, 1, 2]), "Winter",
              np.where(df["month"].isin([3, 4, 5]), "Spring",
              np.where(df["month"].isin([6, 7, 8]), "Summer", "Autumn")))
    df["season"] = pd.Series(season, index=df.index).astype("category")

    try:
        import holidays
        cn = holidays.China(years=range(2013, 2018))
        df["is_holiday"] = df[TIME_COL].dt.date.isin(cn).astype(int)
    except Exception:
        df["is_holiday"] = 0

    return df


def build_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["station_region"] = df["station"].map(STATION_REGIONS).astype("category")
    df["station_regional_vulnerability"] = (
        df["station_region"].astype(object).map(NORTH_SOUTH_SCORE).astype(float)
    )
    df = df.sort_values(["station", TIME_COL]).reset_index(drop=True)

    spatial = df.groupby(TIME_COL).agg(
        city_pm10_mean=("PM10", "mean"),
        city_pm10_std=("PM10", "std"),
        city_pm10_max=("PM10", "max"),
        city_pm10_min=("PM10", "min"),
        city_co_mean=("CO", "mean"),
        city_no2_mean=("NO2", "mean"),
        city_wspm_mean=("WSPM", "mean"),
    ).reset_index()
    df = df.merge(spatial, on=TIME_COL, how="left")
    df["station_dispersion_from_city"] = df["PM10"] - df["city_pm10_mean"]
    df["pm10_to_city_ratio"] = df["PM10"] / (df["city_pm10_mean"] + 1e-4)

    for col in ["PM10", "CO", "NO2"]:
        wide = df.pivot_table(index=TIME_COL, columns="station", values=col, aggfunc="mean")
        wide = wide.rename(columns=lambda s: f"{col}_{s}")
        df = df.merge(wide, left_on=TIME_COL, right_index=True, how="left")

    north_cols = [c for c in df.columns if c.startswith("PM10_") and c.replace("PM10_", "") in NORTH_STATIONS]
    south_cols = [c for c in df.columns if c.startswith("PM10_") and c.replace("PM10_", "") in SOUTH_STATIONS]
    df["pm10_north_mean"] = df[north_cols].mean(axis=1) if north_cols else df["city_pm10_mean"]
    df["pm10_south_mean"] = df[south_cols].mean(axis=1) if south_cols else df["city_pm10_mean"]
    df["pm10_ns_contrast"] = df["pm10_north_mean"] - df["pm10_south_mean"]

    stations = sorted(df["station"].astype(str).unique())
    pm10_wide = [f"PM10_{s}" for s in stations if f"PM10_{s}" in df.columns]
    co_wide = [f"CO_{s}" for s in stations if f"CO_{s}" in df.columns]
    if pm10_wide:
        mat = df[pm10_wide].to_numpy(dtype=float, copy=True)
        col_of = {s: i for i, s in enumerate(stations) if f"PM10_{s}" in df.columns}
        for i, st in enumerate(df["station"].astype(str).to_numpy()):
            j = col_of.get(st)
            if j is not None:
                mat[i, j] = np.nan
        with np.errstate(all="ignore"):
            df["pm10_others_mean"] = np.nanmean(mat, axis=1)
        filled = np.where(np.isfinite(mat), mat, -1e30)
        others_max = filled.max(axis=1)
        others_max[others_max < -1e29] = np.nan
        df["pm10_others_max"] = pd.Series(others_max, index=df.index).fillna(df["city_pm10_max"])
    if co_wide:
        mat = df[co_wide].to_numpy(dtype=float, copy=True)
        col_of = {s: i for i, s in enumerate(stations) if f"CO_{s}" in df.columns}
        for i, st in enumerate(df["station"].astype(str).to_numpy()):
            j = col_of.get(st)
            if j is not None:
                mat[i, j] = np.nan
        with np.errstate(all="ignore"):
            df["co_others_mean"] = np.nanmean(mat, axis=1)

    df["city_haze"] = ((df["city_pm10_mean"] > 150) & (df["city_wspm_mean"] < 1.5)).astype(int)
    g = df.groupby("station", sort=False)
    df["city_pm10_lag_1"] = g["city_pm10_mean"].shift(1)
    df["city_pm10_lag_3"] = g["city_pm10_mean"].shift(3)
    df["city_co_lag_1"] = g["city_co_mean"].shift(1)
    df["city_haze_lag_1"] = g["city_haze"].shift(1)
    df["city_pm10_diff_1"] = df["city_pm10_mean"] - df["city_pm10_lag_1"]
    past_city = g["city_pm10_mean"].shift(1)
    df["city_pm10_roll_6"] = (
        past_city.groupby(df["station"]).rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df = _add_idw_features(df, "PM10", "pm10_idw_3", k=3)
    df = _add_idw_features(df, "CO", "co_idw_3", k=3)
    return df


def _add_idw_features(df: pd.DataFrame, prefix: str, out_name: str, k: int = 3) -> pd.DataFrame:
    stations = [s for s in STATION_LATLON if f"{prefix}_{s}" in df.columns]
    if len(stations) < 3:
        return df
    coords = np.array([STATION_LATLON[s] for s in stations], dtype=float)
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    dist_km = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    np.fill_diagonal(dist_km, np.inf)
    nn = np.argsort(dist_km, axis=1)[:, :k]
    weights = np.zeros_like(dist_km)
    for i in range(len(stations)):
        js = nn[i]
        w = 1.0 / np.maximum(dist_km[i, js], 0.1) ** 2
        weights[i, js] = w / w.sum()
    wide = df[[f"{prefix}_{s}" for s in stations]].to_numpy(dtype=float)
    finite = np.isfinite(wide).astype(float)
    num = np.nan_to_num(wide, nan=0.0) @ weights.T
    den = finite @ weights.T
    idw_all = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-12)
    st_to_j = {s: j for j, s in enumerate(stations)}
    idx = df["station"].astype(str).map(st_to_j)
    out = np.full(len(df), np.nan)
    known = idx.notna().to_numpy()
    out[known] = idw_all[np.flatnonzero(known), idx[known].astype(int).to_numpy()]
    df[out_name] = out
    return df


def add_station_climatology(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Train-only median PM2.5/PM10 ratio by station × month × hour, times current PM10."""
    df = df.copy()
    tmp = _ensure_time(ref).dropna(subset=["current_PM2_5", "PM10"]).copy()
    tmp = tmp[tmp["PM10"] > 0]
    tmp["ratio"] = tmp["current_PM2_5"] / tmp["PM10"]
    g1 = tmp.groupby(["station", "month", "hour"])["ratio"].median()
    g2 = tmp.groupby(["station", "month"])["ratio"].median()
    g3 = tmp.groupby("station")["ratio"].median()
    idx = df.index
    r = pd.Series(list(zip(df["station"], df["month"], df["hour"])), index=idx).map(g1)
    r = r.fillna(pd.Series(list(zip(df["station"], df["month"])), index=idx).map(g2))
    r = r.fillna(df["station"].map(g3))
    r = r.fillna(float(tmp["ratio"].median()))
    df["clim_pm25_from_pm10"] = r.to_numpy() * df["PM10"].to_numpy()
    lvl = tmp.groupby(["station", "month", "hour"])["current_PM2_5"].median()
    lvl2 = tmp.groupby(["station", "month"])["current_PM2_5"].median()
    lv = pd.Series(list(zip(df["station"], df["month"], df["hour"])), index=idx).map(lvl)
    lv = lv.fillna(pd.Series(list(zip(df["station"], df["month"])), index=idx).map(lvl2))
    df["clim_pm25_level"] = lv.fillna(float(tmp["current_PM2_5"].median())).to_numpy()
    med_pm10 = tmp.groupby(["station", "month", "hour"])["PM10"].median()
    m10 = pd.Series(list(zip(df["station"], df["month"], df["hour"])), index=idx).map(med_pm10)
    df["pm10_anomaly"] = df["PM10"].to_numpy() / (m10.fillna(df["PM10"]).to_numpy() + 1e-4)
    return df


def _stage1_frame(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in STAGE1_NUM if c in df.columns]
    cat = [c for c in STAGE1_CAT if c in df.columns]
    x = df[num + cat].copy()
    for c in cat:
        x[c] = x[c].astype("object").fillna("Missing").astype("category")
    return x


def _predict_current_pm25(model, df: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(_stage1_frame(df)), 0, None)


def fit_stage1_current_pm25(train_df: pd.DataFrame, random_state: int = 42):
    """LightGBM: current_PM2_5 from test-available sensors only."""
    import lightgbm as lgb

    mask = train_df["current_PM2_5"].notna()
    x = _stage1_frame(train_df.loc[mask])
    y = train_df.loc[mask, "current_PM2_5"].to_numpy(dtype=float)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.5,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x, y)
    return model


def add_predicted_pm25(df: pd.DataFrame, stage1_model) -> pd.DataFrame:
    df = df.copy()
    df["predicted_PM2_5"] = _predict_current_pm25(stage1_model, df)
    df["pred_pm25_to_pm10"] = df["predicted_PM2_5"] / (df["PM10"] + 1e-4)
    df["pm10_minus_pred_pm25"] = (df["PM10"] - df["predicted_PM2_5"]).clip(lower=0)
    df["pred_pm25_x_heating"] = df["predicted_PM2_5"] * df["is_heating_season"]
    return df


def add_predicted_pm25_oof(df: pd.DataFrame, ref: pd.DataFrame, n_blocks: int = 4, random_state: int = 42):
    """OOF proxy on labelled ``ref`` rows; full-ref model on holdout/test rows.

    Stage-2 then sees a proxy whose error matches test, instead of in-sample
    current-PM2.5 reconstructions that would not be available later.
    """
    df = df.reset_index(drop=True).copy()
    ref_fit = _ensure_time(ref).dropna(subset=["current_PM2_5"]).copy()

    pred = np.full(len(df), np.nan, dtype=float)
    id_to_pos = {i: k for k, i in enumerate(df["id"].to_numpy())}

    t = ref_fit[TIME_COL]
    blocks = pd.qcut(t.rank(method="first"), n_blocks, labels=False, duplicates="drop")
    for b in sorted(pd.unique(blocks)):
        va = ref_fit.loc[blocks == b]
        tr = ref_fit.loc[blocks != b]
        if len(tr) < 5000 or len(va) == 0:
            continue
        model_b = fit_stage1_current_pm25(tr, random_state)
        p = _predict_current_pm25(model_b, va)
        for i, value in zip(va["id"].to_numpy(), p):
            pos = id_to_pos.get(i)
            if pos is not None:
                pred[pos] = value

    full_model = fit_stage1_current_pm25(ref_fit, random_state)
    missing = np.isnan(pred)
    if missing.any():
        holdout = df.iloc[np.flatnonzero(missing)]
        pred[missing] = _predict_current_pm25(full_model, holdout)

    df["predicted_PM2_5"] = pred
    df["pred_pm25_to_pm10"] = df["predicted_PM2_5"] / (df["PM10"] + 1e-4)
    df["pm10_minus_pred_pm25"] = (df["PM10"] - df["predicted_PM2_5"]).clip(lower=0)
    df["pred_pm25_x_heating"] = df["predicted_PM2_5"] * df["is_heating_season"]
    return df, full_model


def apply_proxy_pipeline(df: pd.DataFrame, ref: pd.DataFrame):
    """Physics/spatial features + stage-1 proxy fit on ``ref`` (must have current_PM2_5)."""
    df = build_advanced_features(df)
    df = build_spatial_features(df)
    df = add_station_climatology(df, ref)
    ref_ids = set(_ensure_time(ref)["id"])
    ref_eng = df[df["id"].isin(ref_ids)].copy()
    df, model = add_predicted_pm25_oof(df, ref_eng)
    df = add_predicted_pm25_history(df)
    return df, model


def add_predicted_pm25_history(df: pd.DataFrame) -> pd.DataFrame:
    """Lags/rolling of the stage-1 proxy — available on every test hour."""
    df = df.sort_values(["station", TIME_COL]).copy()
    g = df.groupby("station", sort=False)
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f"pred_pm25_lag_{lag}"] = g["predicted_PM2_5"].shift(lag)
    df["pred_pm25_change_1"] = df["predicted_PM2_5"] - df["pred_pm25_lag_1"]
    df["pred_pm25_change_3"] = df["predicted_PM2_5"] - df["pred_pm25_lag_3"]
    past = g["predicted_PM2_5"].shift(1)
    key = df["station"]
    for w in [3, 6, 24]:
        df[f"pred_pm25_roll_mean_{w}"] = (
            past.groupby(key).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in DROP_FROM_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or str(df[c].dtype) in {"category", "object"}:
            cols.append(c)
    return cols


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, stage1_ref: pd.DataFrame | None = None):
    """Concat train/test, engineer features, fit stage 1 on stage1_ref (default: all train)."""
    orig_test_ids = test["id"].to_numpy()
    train = _ensure_time(train)
    test = _ensure_time(test)
    test = test.copy()
    if TARGET not in test.columns:
        test[TARGET] = np.nan
    if "current_PM2_5" not in test.columns:
        test["current_PM2_5"] = np.nan

    full = pd.concat([train, test], ignore_index=True, sort=False)
    ref = stage1_ref if stage1_ref is not None else train
    full, stage1 = apply_proxy_pipeline(full, _ensure_time(ref))

    train_feat = full[full["id"].isin(train["id"])].copy()
    test_feat = full.set_index("id").loc[orig_test_ids].reset_index()
    return train_feat, test_feat, stage1


if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    train_df, test_df = interpolate_then_mean_fill(train_df, test_df)
    tr, te, _ = prepare_features(train_df, test_df)
    print("train", tr.shape, "test", te.shape, "n_features", len(get_feature_columns(tr)))
