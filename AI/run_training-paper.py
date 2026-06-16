"""
run_training_paper.py
─────────────────────
Trains the GBR models using ONLY the frozen area_history.csv.
No Firebase. No live data. Results will exactly match the paper
as long as area_history.csv is unchanged.

Usage:
    python run_training_paper.py
"""

import os
import logging
import pandas as pd

from features import build_hourly_features, build_daily_features
from model import train_model
from config import (
    HOURLY_MODEL_PATH, HOURLY_SCALER_PATH, HOURLY_META_PATH,
    DAILY_MODEL_PATH,  DAILY_SCALER_PATH,  DAILY_META_PATH,
    HISTORY_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

MIN_HOURLY_ROWS = 30
MIN_DAILY_ROWS  = 14


def load_frozen_history(path: str) -> pd.DataFrame:
    """Load the frozen CSV exactly as it was saved — no Firebase involved."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Frozen history not found at '{path}'.\n"
            f"Make sure area_history.csv is in the right place."
        )

    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)

    # Re-attach timezone if it was stripped on save
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Manila')
    else:
        df.index = df.index.tz_convert('Asia/Manila')

    df = df.sort_index()
    logger.info(f"Loaded frozen history: {len(df)} rows  "
                f"({df.index.min()} → {df.index.max()})")
    return df


def run():
    print("=" * 60)
    print("PAPER REPLICATION — FROZEN DATASET TRAINING")
    print("=" * 60)

    # ── Load frozen data ─────────────────────────────────────────
    area_df = load_frozen_history(HISTORY_PATH)

    # ── Hourly model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING HOURLY MODEL")
    print("=" * 60)

    hourly_featured = build_hourly_features(area_df)
    logger.info(f"Hourly feature rows: {len(hourly_featured)}")

    if len(hourly_featured) >= MIN_HOURLY_ROWS:
        os.makedirs('models', exist_ok=True)
        hourly_metrics = train_model(
            hourly_featured,
            HOURLY_MODEL_PATH,
            HOURLY_SCALER_PATH,
            HOURLY_META_PATH,
            label='hourly',
        )
        print_summary('HOURLY', hourly_metrics)
    else:
        logger.warning(
            f"Not enough hourly rows ({len(hourly_featured)}/{MIN_HOURLY_ROWS}) — skipping."
        )

    # ── Daily model ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING DAILY MODEL")
    print("=" * 60)

    daily_featured = build_daily_features(area_df)
    logger.info(f"Daily feature rows: {len(daily_featured)}")

    if len(daily_featured) >= MIN_DAILY_ROWS:
        os.makedirs('models', exist_ok=True)
        daily_metrics = train_model(
            daily_featured,
            DAILY_MODEL_PATH,
            DAILY_SCALER_PATH,
            DAILY_META_PATH,
            label='daily',
        )
        print_summary('DAILY', daily_metrics)
    else:
        logger.warning(
            f"Not enough daily rows ({len(daily_featured)}/{MIN_DAILY_ROWS}) — skipping.\n"
            f"Daily model needs at least 2 weeks of data."
        )

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)


def print_summary(label: str, metrics: dict):
    print(f"\n{'=' * 60}")
    print(f"{label} MODEL SUMMARY")
    print(f"{'=' * 60}")
    for target, m in metrics.items():
        print(f"  {target}")
        print(f"    MAE        : {m['MAE']}")
        print(f"    RMSE       : {m['RMSE']}")
        print(f"    R2 (test)  : {m['R2']}")
        print(f"    R2 (train) : {m['R2_train']}")
        print(f"    MAPE       : {m['MAPE']}%")
        print(f"    Median AE  : {m['Median_AE']}")
        print(f"    CV R2      : {m.get('CV_R2', 'n/a')}")
        print(f"    Overfit    : {m['Overfit_status']}")


if __name__ == '__main__':
    run()