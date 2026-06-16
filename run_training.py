from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from data_loader import init_firebase, fetch_and_average
from features import build_hourly_features, build_daily_features
from model import train_model
from config import (
    HOURLY_MODEL_PATH, HOURLY_SCALER_PATH, HOURLY_META_PATH,
    DAILY_MODEL_PATH,  DAILY_SCALER_PATH,  DAILY_META_PATH,
    HISTORY_PATH
)

MIN_HOURLY_ROWS = 30
MIN_DAILY_ROWS  = 14


def print_summary(label: str, metrics: dict):
    print(f"\n{'='*60}")
    print(f"{label.upper()} MODEL SUMMARY")
    print(f"{'='*60}")
    for target, m in metrics.items():
        print(f"\n  {target}")
        print(f"    MAE        : {m['MAE']}")
        print(f"    RMSE       : {m['RMSE']}")
        print(f"    R2         : {m['R2']}")
        print(f"    MAPE       : {m['MAPE']}%")
        print(f"    Median AE  : {m['Median_AE']}")
        print(f"    Overfit    : {m['Overfit_status']}")
        


def main():
    print("\n" + "="*60)
    print("INITIALIZING FIREBASE")
    print("="*60)
    init_firebase()

    print("\n" + "="*60)
    print("FETCHING AND CLEANING DATA")
    print("="*60)
    area_df = fetch_and_average()

    os.makedirs('data', exist_ok=True)
    area_df.to_csv(HISTORY_PATH)
    print(f"\nFirebase history saved: {len(area_df)} rows")
    print(f"Date range: {area_df.index.min()} → {area_df.index.max()}")

    print("\n" + "="*60)
    print("LOADING EXTERNAL DATA")
    print("="*60)
    combined_df = area_df
    print(f"\nDataset: {len(combined_df)} rows")

    print("\n" + "="*60)
    print("TRAINING HOURLY MODEL")
    print("="*60)
    hourly_featured = build_hourly_features(combined_df)
    print(f"Hourly feature rows: {len(hourly_featured)}")

    if len(hourly_featured) < MIN_HOURLY_ROWS:
        print(f"Not enough hourly rows ({len(hourly_featured)}) — "
              f"need at least {MIN_HOURLY_ROWS}. Skipping.")
    else:
        hourly_metrics = train_model(
            hourly_featured,
            HOURLY_MODEL_PATH,
            HOURLY_SCALER_PATH,
            HOURLY_META_PATH,
            label='hourly'
        )
        print_summary('hourly', hourly_metrics)

    print("\n" + "="*60)
    print("TRAINING DAILY MODEL")
    print("="*60)
    daily_featured = build_daily_features(combined_df)
    print(f"Daily feature rows: {len(daily_featured)}")

    if len(daily_featured) < MIN_DAILY_ROWS:
        print(f"Not enough daily rows ({len(daily_featured)}) — "
              f"need at least {MIN_DAILY_ROWS} (2 weeks of data). Skipping.")
        print("Daily model will become available once more data accumulates.")
    else:
        daily_metrics = train_model(
            daily_featured,
            DAILY_MODEL_PATH,
            DAILY_SCALER_PATH,
            DAILY_META_PATH,
            label='daily'
        )
        print_summary('daily', daily_metrics)

    print("\n" + "="*60)
    print("Training complete.")
    print("="*60)


if __name__ == "__main__":
    main()