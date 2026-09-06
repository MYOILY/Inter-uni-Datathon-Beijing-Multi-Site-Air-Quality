"""Train the submitted two-stage model and write submission.csv.

Test has no current_PM2_5. Recursion is not used.

The leaderboard file is a fixed mix:
  0.75 * LightGBM seed-bag (seeds 42, 2024)
  0.25 * LightGBM residual around the stage-1 PM2.5 proxy

Run:  python train_and_submit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from feature_engineering import (
    TARGET,
    TIME_COL,
    apply_proxy_pipeline,
    get_feature_columns,
    interpolate_then_mean_fill,
    prepare_features,
)

SEED = 42
BAG_SEEDS = [42, 2024]
BLEND_BAG = 0.75
BLEND_RES = 0.25
FINAL_TREES = 800
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CAT_COLS = ["station", "wd", "season", "station_region"]

LGB_PARAMS = dict(
    objective="regression",
    n_estimators=2500,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)


def rmse(y, pred) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def postprocess(pred: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), 0.0, 999.0)


def row_weights(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df), dtype=float)
    w[df["month"].isin([9, 10, 11, 12, 1, 2]).to_numpy()] = 1.8
    if "is_heating_season" in df.columns:
        w[df["is_heating_season"].to_numpy() == 1] *= 1.2
    y = df[TARGET].to_numpy(dtype=float)
    w *= 1.0 + np.clip(y, 0, 400) / 180.0
    return w


def make_xy(df: pd.DataFrame, feature_cols: list[str]):
    x = df[feature_cols].copy().reset_index(drop=True)
    for c in CAT_COLS:
        if c in x.columns:
            x[c] = x[c].astype("object").fillna("Missing").astype(str)
    y = df[TARGET].to_numpy(dtype=float) if TARGET in df.columns else None
    return x, y


def align_categories(*frames: pd.DataFrame) -> None:
    for c in CAT_COLS:
        if not all(c in f.columns for f in frames):
            continue
        cats = sorted(set().union(*(set(f[c].astype(str)) for f in frames)))
        for f in frames:
            f[c] = pd.Categorical(f[c].astype(str), categories=cats)


def fit_lgb(x_tr, y_tr, x_va=None, y_va=None, sample_weight=None, seed=SEED, n_estimators=None):
    params = dict(LGB_PARAMS)
    params["random_state"] = seed
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    model = lgb.LGBMRegressor(**params)
    fit_kw = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    if x_va is not None:
        fit_kw["eval_set"] = [(x_va, y_va)]
        fit_kw["callbacks"] = [
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(250),
        ]
    model.fit(x_tr, y_tr, **fit_kw)
    return model


def evaluate_split(dev, val, feature_cols, tag):
    x_tr, y_tr = make_xy(dev, feature_cols)
    x_va, y_va = make_xy(val, feature_cols)
    align_categories(x_tr, x_va)
    w = row_weights(dev)
    proxy_va = val["predicted_PM2_5"].to_numpy(dtype=float)
    proxy_tr = dev["predicted_PM2_5"].to_numpy(dtype=float)
    rows = []

    print(f"\n[{tag}] LightGBM seed-bag {BAG_SEEDS}")
    preds = []
    trees = []
    for seed in BAG_SEEDS:
        print(f"  seed {seed}")
        m = fit_lgb(x_tr, y_tr, x_va, y_va, sample_weight=w, seed=seed)
        p = postprocess(m.predict(x_va))
        trees.append(int(getattr(m, "best_iteration_", m.n_estimators_)))
        preds.append(p)
        rows.append({"split": tag, "model": f"LGB seed {seed}", "rmse": rmse(y_va, p), "best_iteration": trees[-1]})

    bag = postprocess(np.mean(preds, axis=0))
    rows.append(
        {
            "split": tag,
            "model": "LightGBM seed-bag",
            "rmse": rmse(y_va, bag),
            "best_iteration": int(np.median(trees)),
        }
    )

    print(f"[{tag}] LightGBM residual around stage-1 proxy")
    m_res = fit_lgb(x_tr, y_tr - proxy_tr, x_va, y_va - proxy_va, sample_weight=w, seed=SEED)
    res_pred = postprocess(proxy_va + m_res.predict(x_va))
    mix = postprocess(BLEND_BAG * bag + BLEND_RES * res_pred)
    rows.append(
        {
            "split": tag,
            "model": "LightGBM residual-on-proxy",
            "rmse": rmse(y_va, res_pred),
            "best_iteration": int(getattr(m_res, "best_iteration_", m_res.n_estimators_)),
        }
    )
    rows.append(
        {
            "split": tag,
            "model": f"Blend {BLEND_BAG:g} bag + {BLEND_RES:g} residual",
            "rmse": rmse(y_va, mix),
        }
    )

    if "current_PM2_5" in val.columns:
        cur = val["current_PM2_5"].to_numpy(dtype=float)
        rows.append({"split": tag, "model": "Oracle persistence (ineligible)", "rmse": rmse(y_va, postprocess(cur))})
        rows.append({"split": tag, "model": "Stage-1 RMSE vs current (diagnostic)", "rmse": rmse(cur, postprocess(proxy_va))})
    rows.append({"split": tag, "model": "Stage-1 proxy as next-hour", "rmse": rmse(y_va, postprocess(proxy_va))})
    return pd.DataFrame(rows)


def load_cleaned():
    """Load competition CSVs and apply production gap-filling."""
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return interpolate_then_mean_fill(train, test)


def run_validation(train):
    tag = "season_matched_Sep2015_Feb2016"
    val_start, val_end = "2015-09-01", "2016-03-01"
    print("\n" + "=" * 64)
    print("Validation window:", tag)
    print("=" * 64)
    ref = train.loc[train[TIME_COL] < val_start].copy()
    feat, _ = apply_proxy_pipeline(train.copy(), ref)
    dev = feat.loc[feat[TIME_COL] < val_start].copy()
    val = feat.loc[(feat[TIME_COL] >= val_start) & (feat[TIME_COL] < val_end)].copy()
    feature_cols = get_feature_columns(feat)
    print("dev", dev.shape, "val", val.shape, "n_features", len(feature_cols))
    table = evaluate_split(dev, val, feature_cols, tag)
    print(table.to_string(index=False))
    return table, {"feature_cols": feature_cols}


def fit_submitted_model(train, test, n_trees: int = FINAL_TREES):
    """Fit the locked leaderboard recipe on all train rows; predict test."""
    orig_ids = test["id"].copy()
    train_feat, test_feat, stage1 = prepare_features(train, test)
    feature_cols = get_feature_columns(train_feat)
    x_tr, y_tr = make_xy(train_feat, feature_cols)
    x_te, _ = make_xy(test_feat, feature_cols)
    align_categories(x_tr, x_te)
    w = row_weights(train_feat)
    proxy_tr = train_feat["predicted_PM2_5"].to_numpy(dtype=float)
    proxy_te = test_feat["predicted_PM2_5"].to_numpy(dtype=float)

    bag_models = []
    for seed in BAG_SEEDS:
        print("Final seed", seed, "trees", n_trees)
        bag_models.append(fit_lgb(x_tr, y_tr, sample_weight=w, seed=seed, n_estimators=n_trees))
    bag = np.mean([m.predict(x_te) for m in bag_models], axis=0)

    print("Final residual-on-proxy, trees", n_trees)
    m_res = fit_lgb(x_tr, y_tr - proxy_tr, sample_weight=w, seed=SEED, n_estimators=n_trees)
    res = proxy_te + m_res.predict(x_te)
    y_hat = postprocess(BLEND_BAG * bag + BLEND_RES * res)

    sub_df = pd.DataFrame({"id": test_feat["id"].to_numpy(), "PM2_5_next_hour": y_hat})
    sub_df = orig_ids.to_frame(name="id").merge(sub_df, on="id", how="left")
    return sub_df, stage1, bag_models, m_res, feature_cols


def model_record(val_rmse, feature_cols, sub_df):
    return {
        "selected_label": f"Blend {BLEND_BAG:g} bag + {BLEND_RES:g} residual",
        "season_matched_val_rmse": None if val_rmse is None else float(val_rmse),
        "n_estimators": FINAL_TREES,
        "seeds": BAG_SEEDS,
        "blend_weights": {"seed_bag": BLEND_BAG, "residual_on_proxy": BLEND_RES},
        "learning_rate": LGB_PARAMS["learning_rate"],
        "seed": SEED,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "hyperparameters": {
            "lightgbm_stage2": {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"},
            "final_trees": FINAL_TREES,
            "early_stopping_rounds_on_val": 100,
            "sample_weights": "Sep–Feb ×1.8, heating ×1.2, times (1 + clip(y,0,400)/180)",
        },
        "postprocess": "clip predictions to [0, 999]",
        "notes": (
            "Stage-1 LightGBM reconstructs current PM2.5 from test-available sensors "
            "(including same-hour other stations and IDW neighbours). Stage-2 is a "
            "LightGBM seed-bag mixed with a residual model around that proxy. "
            "No recursive current-PM2.5. year dropped."
        ),
        "submission_rows": int(len(sub_df)),
        "submission_min": float(sub_df["PM2_5_next_hour"].min()),
        "submission_max": float(sub_df["PM2_5_next_hour"].max()),
        "submission_mean": float(sub_df["PM2_5_next_hour"].mean()),
        "submission_na": int(sub_df["PM2_5_next_hour"].isna().sum()),
    }


def select_and_fit_final(train, test, board):
    """Fit the locked submitted blend (ignores other val rows for selection)."""
    mix_name = f"Blend {BLEND_BAG:g} bag + {BLEND_RES:g} residual"
    hit = board.loc[board["model"] == mix_name, "rmse"]
    val_rmse = float(hit.iloc[0]) if len(hit) else None
    print("\nSubmitted recipe:", mix_name, "val RMSE", val_rmse)

    sub_df, stage1, bag_models, m_res, feature_cols = fit_submitted_model(train, test)
    record = model_record(val_rmse, feature_cols, sub_df)
    write_artifacts(sub_df, record, stage1, bag_models, m_res)
    return sub_df, record


def write_artifacts(sub_df, record, stage1, bag_models, m_res):
    OUT.mkdir(exist_ok=True)
    sub_df.to_csv(ROOT / "submission.csv", index=False)
    sub_df.to_csv(OUT / "submission.csv", index=False)
    joblib.dump(bag_models[0], OUT / "final_lgb_pass1.joblib")
    if len(bag_models) > 1:
        joblib.dump(bag_models[1], OUT / "final_lgb_seed2024.joblib")
    joblib.dump(m_res, OUT / "final_lgb_delta.joblib")
    joblib.dump(stage1, OUT / "stage1_current_pm25.joblib")
    (OUT / "final_model_record.json").write_text(json.dumps(record, indent=2))
    print("Wrote submission.csv", sub_df.shape, "na", record["submission_na"])


def main():
    train, test = load_cleaned()
    print(
        "train",
        train.shape,
        "test",
        test.shape,
        "test has current_PM2_5",
        "current_PM2_5" in test.columns,
    )
    board, _ = run_validation(train)
    OUT.mkdir(exist_ok=True)
    board.to_csv(OUT / "val_leaderboard.csv", index=False)
    return select_and_fit_final(train, test, board)


if __name__ == "__main__":
    main()
