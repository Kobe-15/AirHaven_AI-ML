import os
import json
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, median_absolute_error)

from config import SENSOR_TARGETS


def _make_gbr() -> MultiOutputRegressor:
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        subsample=0.8,
        max_features=0.8,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4,
        random_state=42
    )
    return MultiOutputRegressor(gbr, n_jobs=-1)


def _time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = int(len(df) * (1 - test_ratio))
    return df.iloc[:n], df.iloc[n:]


def _mape(y_true, y_pred) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100)


def _evaluate(model, scaler, X_test: pd.DataFrame,
              y_test: pd.DataFrame):
    y_pred  = model.predict(scaler.transform(X_test.fillna(0)))
    results = {}

    present = [t for t in SENSOR_TARGETS if t in y_test.columns]
    for i, t in enumerate(present):
        actual    = y_test[t].values
        predicted = y_pred[:, i]

        results[t] = {
            'MAE':       round(float(mean_absolute_error(actual, predicted)),  4),
            'RMSE':      round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
            'R2':        round(float(r2_score(actual, predicted)),             4),
            'MAPE':      round(float(_mape(actual, predicted)),                4),
            'Median_AE': round(float(median_absolute_error(actual, predicted)),4),
        }

    return results, y_pred


def _cv_r2(featured_df: pd.DataFrame, feature_cols: list,
           scaler: StandardScaler, n_splits: int) -> dict:
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    present = [t for t in SENSOR_TARGETS if t in featured_df.columns]
    X_all   = featured_df[feature_cols].fillna(0)
    y_all   = featured_df[present].ffill().fillna(0).values
    X_all_s = scaler.transform(X_all)

    fold_scores = {t: [] for t in present}

    for train_idx, val_idx in tscv.split(X_all_s):
        X_fold_tr  = X_all_s[train_idx]
        X_fold_val = X_all_s[val_idx]
        y_fold_tr  = y_all[train_idx]
        y_fold_val = y_all[val_idx]

        if len(X_fold_tr) < 2 or len(X_fold_val) < 2:
            continue

        fold_model = _make_gbr()
        fold_model.fit(X_fold_tr, y_fold_tr)
        y_fold_pred = fold_model.predict(X_fold_val)

        for i, t in enumerate(present):
            try:
                score = r2_score(y_fold_val[:, i], y_fold_pred[:, i])
                fold_scores[t].append(score)
            except Exception:
                pass

    cv_results = {}
    for t in fold_scores:
        if fold_scores[t]:
            mean = np.mean(fold_scores[t])
            std  = np.std(fold_scores[t])
            cv_results[t] = f"{mean:.4f} ± {std:.4f}"
        else:
            cv_results[t] = "insufficient data"

    return cv_results


def train_model(featured_df: pd.DataFrame,
                model_path: str,
                scaler_path: str,
                meta_path: str,
                label: str) -> dict:

    feature_cols    = [c for c in featured_df.columns if c not in SENSOR_TARGETS]
    present_targets = [t for t in SENSOR_TARGETS if t in featured_df.columns]

    train_df, test_df = _time_split(featured_df)

    X_tr, y_tr = train_df[feature_cols], train_df[present_targets]
    X_te, y_te = test_df[feature_cols],  test_df[present_targets]

    X_tr = X_tr.fillna(0)
    X_te = X_te.fillna(0)
    y_tr = y_tr.ffill().fillna(0)
    y_te = y_te.ffill().fillna(0)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    print(f"\n[{label}] Training on {len(X_tr)} rows, "
          f"testing on {len(X_te)} rows...")
    model = _make_gbr()
    model.fit(X_tr_s, y_tr)

    metrics, _ = _evaluate(model, scaler, X_te, y_te)

    # Overfitting / Underfitting check
    y_pred_train = model.predict(X_tr_s)
    for i, t in enumerate(present_targets):
        r2_train    = r2_score(y_tr[t].values, y_pred_train[:, i])
        overfit_gap = r2_train - metrics[t]['R2']
        metrics[t]['Overfit_gap'] = round(float(overfit_gap), 4)
        metrics[t]['R2_train']       = round(float(r2_train), 4)
        if metrics[t]['R2'] < 0.5 and overfit_gap < 0.05:
            metrics[t]['Overfit_status'] = 'Underfitting suspected'
        elif overfit_gap < 0.05:
            metrics[t]['Overfit_status'] = 'Well generalized'
        elif overfit_gap < 0.10:
            metrics[t]['Overfit_status'] = 'Minor overfitting'
        elif overfit_gap < 0.20:
            metrics[t]['Overfit_status'] = 'Moderate overfitting'
        else:
            metrics[t]['Overfit_status'] = 'Severe overfitting'

    # Cross-validation
    n_splits   = 3 if len(featured_df) < 100 else 5
    cv_results = _cv_r2(featured_df, feature_cols, scaler, n_splits)
    for t in present_targets:
        metrics[t]['CV_R2'] = cv_results.get(t, 'insufficient data')

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        'label':           label,
        'feature_columns': feature_cols,
        'trained_at':      datetime.now().isoformat(),
        'n_train':         len(X_tr),
        'n_test':          len(X_te),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[{label}] Evaluation Results:")
    print(f"  {'Target':<12} {'MAE':>8} {'RMSE':>8} {'R2_train':>10} {'R2_test':>10} "
          f"{'MAPE%':>8} {'MedAE':>8} {'Overfit':>22}")
    print(f"  {'-'*100}")
    for t, m in metrics.items():
        print(f"  {t:<12} "
              f"{m['MAE']:>8.4f} "
              f"{m['RMSE']:>8.4f} "
              f"{m['R2_train']:>10.4f} "
              f"{m['R2']:>10.4f} "
              f"{m['MAPE']:>8.2f} "
              f"{m['Median_AE']:>8.4f} "
              f"{m['Overfit_status']:>22}")

    return metrics