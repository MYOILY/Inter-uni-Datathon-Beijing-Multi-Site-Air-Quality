import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


def train_ensemble(train_data, val_data, feature_cols, cat_cols=None):
    if cat_cols is None:
        cat_cols = ['station', 'wd', 'season', 'station_region']
    
    # 1. Clean missing targets ONLY (no current_PM2_5)
    train_valid = train_data.dropna(subset=['PM2_5_next_hour']).copy()
    val_valid = val_data.dropna(subset=['PM2_5_next_hour']).copy()

    # 2. Target is directly PM2_5_next_hour
    y_train = train_valid['PM2_5_next_hour'].values
    y_val = val_valid['PM2_5_next_hour'].values
    
    X_train = train_valid[feature_cols].copy()
    X_val = val_valid[feature_cols].copy()
    
    for c in cat_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype('category')
            X_val[c] = X_val[c].astype('category')
            
    # LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.02,
        'num_leaves': 45,
        'min_child_samples': 40,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 2,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    model_lgb = lgb.train(
        lgb_params, lgb_train, num_boost_round=2500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # CatBoost
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    cat_present = [c for c in cat_cols if c in X_train_cat.columns]
    for c in cat_present:
        X_train_cat[c] = X_train_cat[c].astype(str).fillna("missing")
        X_val_cat[c] = X_val_cat[c].astype(str).fillna("missing")
        
    train_pool = Pool(X_train_cat, y_train, cat_features=cat_present)
    val_pool = Pool(X_val_cat, y_val, cat_features=cat_present)
    
    model_cat = CatBoostRegressor(
        iterations=2500, learning_rate=0.03, depth=6,
        l2_leaf_reg=5.0, loss_function='RMSE', eval_metric='RMSE',
        random_seed=42, verbose=0
    )
    model_cat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    
    # XGBoost
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    for c in cat_present:
        X_train_xgb[c] = X_train_xgb[c].cat.codes
        X_val_xgb[c] = X_val_xgb[c].cat.codes
            
    model_xgb = xgb.XGBRegressor(
        n_estimators=2500, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror', eval_metric='rmse',
        random_state=42, n_jobs=-1
    )
    model_xgb.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
    
    # Optimize ensemble weights directly on predictions
    p_lgb = model_lgb.predict(X_val)
    p_cat = model_cat.predict(val_pool)
    p_xgb = model_xgb.predict(X_val_xgb)
    preds_matrix = np.column_stack([p_lgb, p_cat, p_xgb])

    def loss_func(weights):
        blend = preds_matrix @ weights
        return np.sqrt(mean_squared_error(y_val, blend))

    res = minimize(
        loss_func, x0=[0.4, 0.4, 0.2], bounds=[(0, 1), (0, 1), (0, 1)], 
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    )
    optimal_weights = res.x

    final_val_preds = preds_matrix @ optimal_weights
    final_val_preds = np.clip(final_val_preds, a_min=0, a_max=None)
    
    rmse = np.sqrt(mean_squared_error(y_val, final_val_preds))
    print(f"\n>> Validation Ensemble Direct RMSE: {rmse:.4f} µg/m³ <<")
    
    return {'lgb': model_lgb, 'cat': model_cat, 'xgb': model_xgb}, optimal_weights


def predict_test_and_submit(models_dict, test_data, feature_cols, weights, cat_cols=None, output_path='submission.csv'):
    if cat_cols is None:
        cat_cols = ['station', 'wd', 'season', 'station_region']
        
    X_test = test_data[feature_cols].copy()
    for c in cat_cols:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype('category')
            
    # Direct model inferences
    p_lgb = models_dict['lgb'].predict(X_test)
    
    X_test_cat = X_test.copy()
    for c in [c for c in cat_cols if c in X_test_cat.columns]:
        X_test_cat[c] = X_test_cat[c].astype(str).fillna("missing")
    p_cat = models_dict['cat'].predict(X_test_cat)
    
    X_test_xgb = X_test.copy()
    for c in [c for c in cat_cols if c in X_test_xgb.columns]:
        X_test_xgb[c] = X_test_xgb[c].cat.codes
    p_xgb = models_dict['xgb'].predict(X_test_xgb)
    
    # Weighted direct prediction
    final_pm25 = weights[0] * p_lgb + weights[1] * p_cat + weights[2] * p_xgb
    final_pm25 = np.clip(final_pm25, a_min=0, a_max=None)
    
    # Physical sanity boundary: PM2.5 cannot realistically exceed PM10 + instrument noise
    if 'PM10' in test_data.columns:
        pm10_cap = test_data['PM10'].fillna(pd.Series(final_pm25, index=test_data.index)).values * 1.25
        final_pm25 = np.minimum(final_pm25, pm10_cap)
    
    submission_df = pd.DataFrame({
        'id': test_data['id'],
        'PM2_5_next_hour': final_pm25
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"✓ Saved direct prediction submission to {output_path} (0 NaNs)")
    return submission_df


def find_best_damping_factor(val_valid, val_ensemble_delta):
    print("\n--- Tuning Delta Damping on Validation Split ---")
    best_rmse = float('inf')
    best_lambda = 1.0

    for lam in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
        candidate_preds = np.clip(
            val_valid['current_PM2_5'].values + lam * val_ensemble_delta, 
            a_min=0, 
            a_max=None
        )
        score = np.sqrt(mean_squared_error(val_valid['PM2_5_next_hour'], candidate_preds))
        print(f"  Lambda: {lam:.2f}  -->  Validation RMSE: {score:.4f} µg/m³")
        
        if score < best_rmse:
            best_rmse = score
            best_lambda = lam

    print(f"\n✓ Best Lambda selected: {best_lambda:.2f} (Val RMSE: {best_rmse:.4f})")
    return best_lambda


def compute_ensemble_feature_importance(models_dict, feature_names, weights=None, top_n=25):
    """
    Normalizes feature importance from LightGBM (gain/split), CatBoost (PredictionValuesChange),
    and XGBoost (gain) to sum to 1, then computes an ensemble-weighted importance score.
    """
    if weights is None:
        weights = [0.50, 0.30, 0.20]  # Fallback to default ensemble weights
        
    w_lgb, w_cat, w_xgb = weights

    # 1. LightGBM Importance (using 'gain' is strictly better than 'split' for trees)
    lgb_raw = models_dict['lgb'].feature_importance(importance_type='gain')
    lgb_norm = lgb_raw / (np.sum(lgb_raw) + 1e-9)

    # 2. CatBoost Importance
    cat_raw = models_dict['cat'].get_feature_importance()
    cat_norm = cat_raw / (np.sum(cat_raw) + 1e-9)

    # 3. XGBoost Importance (gain-based)
    xgb_raw = models_dict['xgb'].feature_importances_
    xgb_norm = xgb_raw / (np.sum(xgb_raw) + 1e-9)

    # 4. Ensemble Weighted Importance
    ensemble_importance = (
        w_lgb * lgb_norm +
        w_cat * cat_norm +
        w_xgb * xgb_norm
    )

    # Build Comparison DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Ensemble_Importance': ensemble_importance,
        'LGBM_Gain': lgb_norm,
        'CatBoost_Gain': cat_norm,
        'XGBoost_Gain': xgb_norm
    }).sort_values(by='Ensemble_Importance', ascending=False).reset_index(drop=True)

    # Format as percentages for display
    print(f"\n{'='*75}")
    print(f" 📊 TOP {top_n} ENSEMBLE FEATURE IMPORTANCES")
    print(f"{'='*75}")
    display_df = importance_df.head(top_n).copy()
    for col in ['Ensemble_Importance', 'LGBM_Gain', 'CatBoost_Gain', 'XGBoost_Gain']:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'
    print(display_df.to_string(index=False))

    return importance_df



""" Execution """
train_df = pd.read_csv("train_featured.csv")
test_df = pd.read_csv("test_featured.csv")

# 1. Cast string categories across train and test beforehand
categorical_features = ['station', 'wd', 'season', 'station_region']
for col in categorical_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype('category')
    if col in test_df.columns:
        test_df[col] = test_df[col].astype('category')

# 2. Time-Based Split: 85% Train, 15% Out-of-Time Validation from train_df
train_df['observation_timestamp'] = pd.to_datetime(train_df['observation_timestamp'])
split_time = train_df['observation_timestamp'].quantile(0.85)

train_split = train_df[train_df['observation_timestamp'] < split_time].copy()
val_split = train_df[train_df['observation_timestamp'] >= split_time].copy()

# 3. Identify Feature Columns
drop_cols = ['id', 'observation_timestamp', 'PM2_5_next_hour']
features = [c for c in train_df.columns if c not in drop_cols]

# 4. Fit Models
models, optimal_weights = train_ensemble(
    train_data=train_split,
    val_data=val_split,
    feature_cols=features,
    cat_cols=categorical_features
)

# 5. Compute and Print Ensemble Feature Importance
importance_df = compute_ensemble_feature_importance(
    models_dict=models,
    feature_names=features,
    weights=optimal_weights,
    top_n=20
)

# 6. Tune Damping and Predict
# (Use the optimal weights inside predict_test_and_submit as well)
submission = predict_test_and_submit(
    models_dict=models,
    test_data=test_df,
    feature_cols=features,
    weights=optimal_weights,
    cat_cols=categorical_features,
    output_path='submission.csv'
)