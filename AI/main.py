import os
import logging
import pandas as pd
from datetime import datetime
from firebase_admin import db
from dotenv import load_dotenv
load_dotenv()

from data_loader import init_firebase, fetch_and_average
from features import build_hourly_features, build_daily_features
from model import train_model
from forecast import forecast_hourly, forecast_daily
from config import (
    HOURLY_MODEL_PATH, HOURLY_SCALER_PATH, HOURLY_META_PATH,
    DAILY_MODEL_PATH,  DAILY_SCALER_PATH,  DAILY_META_PATH,
    HISTORY_PATH
)

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
MIN_HOURLY_ROWS = 30
MIN_DAILY_ROWS  = 14


# ── Push forecasts to Firebase ───────────────────────────────────
def push_forecasts(hourly_df: pd.DataFrame, daily_df: pd.DataFrame):
    timestamp = datetime.now().isoformat()

    # Push hourly forecasts
    hourly_payload = {
        'generated_at': timestamp,
        'data': hourly_df.to_dict(orient='records')
    }
    db.reference('/forecasts/hourly').set(hourly_payload)
    logger.info(f"[Firebase] Hourly forecasts pushed: {len(hourly_df)} steps.")

    # Push daily forecasts
    daily_payload = {
        'generated_at': timestamp,
        'data': daily_df.to_dict(orient='records')
    }
    db.reference('/forecasts/daily').set(daily_payload)
    logger.info(f"[Firebase] Daily forecasts pushed: {len(daily_df)} steps.")


# ── Main pipeline ────────────────────────────────────────────────
def run():
    logger.info("=" * 60)
    logger.info("AirHaven Forecast Pipeline Started")
    logger.info("=" * 60)

    # Step 1: Connect to Firebase
    logger.info("[Step 1/5] Connecting to Firebase...")
    init_firebase()

    # Step 2: Fetch and clean sensor data
    logger.info("[Step 2/5] Fetching and cleaning sensor data...")
    area_df = fetch_and_average()

    # Step 3: Merge with local history if it exists
    logger.info("[Step 3/5] Merging with local history...")
    if os.path.exists(HISTORY_PATH):
        historical = pd.read_csv(
            HISTORY_PATH, index_col='timestamp', parse_dates=True
        )
        if historical.index.tz is None:
            historical.index = historical.index.tz_localize('Asia/Manila')
        else:
            historical.index = historical.index.tz_convert('Asia/Manila')
        area_df = pd.concat([historical, area_df])
        area_df = area_df[~area_df.index.duplicated(keep='last')]
        area_df = area_df.sort_index()
        logger.info(f"[History] Merged. Total rows: {len(area_df)}")
    else:
        logger.info("[History] No local history found — using Firebase data only.")

    # Save updated history locally
    os.makedirs('data', exist_ok=True)
    save_df = area_df.copy()
    save_df.index = save_df.index.tz_localize(None)
    save_df.to_csv(HISTORY_PATH)
    logger.info(f"[History] Saved locally: {len(area_df)} total rows.")

    # Step 4: Train models
    logger.info("[Step 4/5] Training models...")
    hourly_trained = False
    daily_trained  = False

    hourly_featured = build_hourly_features(area_df)
    if len(hourly_featured) >= MIN_HOURLY_ROWS:
        os.makedirs('models', exist_ok=True)
        train_model(
            hourly_featured,
            HOURLY_MODEL_PATH,
            HOURLY_SCALER_PATH,
            HOURLY_META_PATH,
            label='hourly'
        )
        hourly_trained = True
    else:
        logger.warning(
            f"[Training] Not enough hourly rows "
            f"({len(hourly_featured)}/{MIN_HOURLY_ROWS}) — skipping hourly model."
        )

    daily_featured = build_daily_features(area_df)
    if len(daily_featured) >= MIN_DAILY_ROWS:
        os.makedirs('models', exist_ok=True)
        train_model(
            daily_featured,
            DAILY_MODEL_PATH,
            DAILY_SCALER_PATH,
            DAILY_META_PATH,
            label='daily'
        )
        daily_trained = True
    else:
        logger.warning(
            f"[Training] Not enough daily rows "
            f"({len(daily_featured)}/{MIN_DAILY_ROWS}) — skipping daily model."
            f"\nNote: Daily model needs at least 2 weeks of data."
        )

    # Step 5: Generate forecasts and push to Firebase
    logger.info("[Step 5/5] Generating forecasts and pushing to Firebase...")

    if not hourly_trained and not daily_trained:
        logger.error("[Forecast] No models trained — cannot generate forecasts.")
        return

    hourly_df = pd.DataFrame()
    daily_df  = pd.DataFrame()

    if hourly_trained:
        hourly_df = forecast_hourly(area_df)
        logger.info(f"[Forecast] Hourly forecast generated: {len(hourly_df)} steps.")

    if daily_trained:
        daily_df = forecast_daily(area_df)
        logger.info(f"[Forecast] Daily forecast generated: {len(daily_df)} steps.")

    if not hourly_df.empty or not daily_df.empty:
        push_forecasts(hourly_df, daily_df)

    logger.info("=" * 60)
    logger.info("Pipeline Completed Successfully")
    logger.info(f"Hourly model: {'trained and pushed' if hourly_trained else 'skipped'}")
    logger.info(f"Daily model:  {'trained and pushed' if daily_trained else 'skipped'}")
    logger.info("=" * 60)


if __name__ == '__main__':
    run()