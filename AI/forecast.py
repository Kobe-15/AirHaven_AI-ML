import json
import joblib
import pandas as pd

from features import build_hourly_features, build_daily_features
from config import (
    SENSOR_TARGETS,
    HOURLY_MODEL_PATH, HOURLY_SCALER_PATH, HOURLY_META_PATH,
    DAILY_MODEL_PATH,  DAILY_SCALER_PATH,  DAILY_META_PATH,
    HOURLY_STEPS, DAILY_STEPS
)


def _load_artifacts(model_path, scaler_path, meta_path):
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, scaler, meta


def forecast_hourly(area_df: pd.DataFrame) -> pd.DataFrame:
    model, scaler, meta = _load_artifacts(
        HOURLY_MODEL_PATH, HOURLY_SCALER_PATH, HOURLY_META_PATH
    )
    feature_cols = meta['feature_columns']
    window       = area_df[SENSOR_TARGETS].copy()
    predictions  = []

    # ✅ Anchor to the floored hour in Manila time
    last_ts = window.index[-1]
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_convert('Asia/Manila')
    base_ts = last_ts.floor('h')  # e.g. 10:00+08:00

    for step in range(HOURLY_STEPS):
        featured = build_hourly_features(window)
        if featured.empty:
            print(f"[forecast_hourly] Window too short at step {step} — stopping.")
            break

        last_row = featured.iloc[[-1]]
        X = last_row.reindex(columns=feature_cols, fill_value=0)
        X = X[[c for c in feature_cols if c not in SENSOR_TARGETS]]

        pred    = model.predict(scaler.transform(X.fillna(0)))[0]
        next_ts = base_ts + pd.Timedelta(hours=step + 1)  # ✅ 11:00, 12:00, ...

        window = pd.concat([
            window,
            pd.DataFrame([pred], columns=SENSOR_TARGETS, index=[next_ts])
        ])

        predictions.append({
            'timestamp': next_ts.isoformat(),
            **{t: round(float(pred[i]), 4)
               for i, t in enumerate(SENSOR_TARGETS)}
        })

    return pd.DataFrame(predictions)


def forecast_daily(area_df: pd.DataFrame) -> pd.DataFrame:
    model, scaler, meta = _load_artifacts(
        DAILY_MODEL_PATH, DAILY_SCALER_PATH, DAILY_META_PATH
    )
    feature_cols = meta['feature_columns']
    daily_window = area_df[SENSOR_TARGETS].resample('D').mean().dropna()
    predictions  = []

    for step in range(DAILY_STEPS):
        featured = build_daily_features(daily_window)
        if featured.empty:
            print(f"[forecast_daily] Window too short at step {step} — stopping.")
            break

        last_row = featured.iloc[[-1]]
        X = last_row.reindex(columns=feature_cols, fill_value=0)
        X = X[[c for c in feature_cols if c not in SENSOR_TARGETS]]

        pred     = model.predict(scaler.transform(X.fillna(0)))[0]
        next_day = daily_window.index[-1] + pd.Timedelta(days=1)

        daily_window = pd.concat([
            daily_window,
            pd.DataFrame([pred], columns=SENSOR_TARGETS, index=[next_day])
        ])

        predictions.append({
            'date':     next_day.strftime('%Y-%m-%d'),
            'day_name': next_day.strftime('%A'),
            **{t: round(float(pred[i]), 4)
               for i, t in enumerate(SENSOR_TARGETS)}
        })

    return pd.DataFrame(predictions)