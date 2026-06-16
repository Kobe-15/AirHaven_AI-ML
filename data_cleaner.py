import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from config import TARGETS, SENSOR_TARGETS, FIELD_NODES

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
DEPLOY_DATE = datetime(2026, 3, 21, tzinfo=ZoneInfo('Asia/Manila'))
CUTOFF_DATE = datetime(2026, 4, 2, 23, 59, 59, tzinfo=ZoneInfo('Asia/Manila'))
USE_CUTOFF = True  # Set to True for Chapter 4, False for live demo

SENSOR_BOUNDS = {
    'humidity':    (30,    100),
    'temperature': (15,  50),
    'NO2':         (0,    500),
    'O3':          (0,    1),
    'PM10':        (0,    600),
    'PM2_5':       (0,    500),
    'co':          (0,    50),
}

SPIKE_THRESHOLDS = {
    'humidity':    20,
    'temperature': 5,
    'NO2':         100,
    'O3':          0.1,
    'PM10':        150,
    'PM2_5':       150,
    'co':          10,
}

MAX_INTERPOLATE_MINUTES = 60
MAX_GAP_MINUTES         = 360


# ── Node cleaner ─────────────────────────────────────────────────
def clean_node(df: pd.DataFrame, node_key: str) -> pd.DataFrame:
    df = df.copy().sort_index()

    # Fix 1: tz-aware DEPLOY_DATE comparison
    if USE_CUTOFF:
        df = df[(df.index >= DEPLOY_DATE) & (df.index <= CUTOFF_DATE)]
    else:
        df = df[df.index >= DEPLOY_DATE]
    if df.empty:
        logger.warning(f"[{node_key}] No data after deploy date — skipping.")
        return df

    # Out-of-range → NaN
    for col, (lo, hi) in SENSOR_BOUNDS.items():
        if col in df.columns:
            out_of_range = (df[col] < lo) | (df[col] > hi)
            if out_of_range.any():
                logger.warning(
                    f"[{node_key}] {out_of_range.sum()} "
                    f"out-of-range in {col} → NaN"
                )
            df.loc[out_of_range, col] = np.nan

    # Fix 2: spike detection skips first value after a NaN gap
    for col, threshold in SPIKE_THRESHOLDS.items():
        if col in df.columns:
            diff     = df[col].diff().abs()
            was_nan  = df[col].shift(1).isna()
            spikes   = (diff > threshold) & ~was_nan
            if spikes.any():
                logger.warning(
                    f"[{node_key}] {spikes.sum()} spikes in {col} → NaN"
                )
            df.loc[spikes, col] = np.nan

    # Resample to 1-minute grid
    df = df.resample('1min').mean()

    # Fix 3: recompute gap info AFTER resample
    is_nan    = df[SENSOR_TARGETS].isna().all(axis=1)
    gap_group = (is_nan != is_nan.shift()).cumsum()
    gap_sizes = is_nan.groupby(gap_group).transform('sum')

    # Fix 6: apply MAX_INTERPOLATE_MINUTES as interpolation limit
    df[SENSOR_TARGETS] = df[SENSOR_TARGETS].interpolate(
        method='time',
        limit=MAX_INTERPOLATE_MINUTES
    )

    # Re-NaN rows that belong to gaps longer than MAX_GAP_MINUTES
    long_gap_mask = is_nan & (gap_sizes > MAX_GAP_MINUTES)
    df.loc[long_gap_mask, SENSOR_TARGETS] = np.nan

    df = df.dropna(how='all')

    logger.info(
        f"[{node_key}] Clean rows: {len(df)} "
        f"({df.index.min()} → {df.index.max()})"
    )
    return df


# ── Averager ─────────────────────────────────────────────────────
def clean_and_average(raw_node_data: dict) -> pd.DataFrame:
    cleaned_frames = []

    for node_key, df in raw_node_data.items():
        if node_key not in FIELD_NODES:
            logger.info(f"Skipping {node_key} (collocation node)")
            continue

        logger.info(f"Cleaning {node_key}...")
        cleaned = clean_node(df, node_key)

        if cleaned.empty:
            logger.warning(
                f"{node_key} returned no usable data after cleaning."
            )
            continue

        cleaned_frames.append(cleaned)

    if not cleaned_frames:
        raise ValueError("No usable data after cleaning all field nodes.")

    # Fix 4: numeric_only=True to avoid silent column drops
    area_df = pd.concat(cleaned_frames).groupby(level=0).mean(numeric_only=True)
    area_df = area_df.sort_index()
    area_df.index.name = 'timestamp'

    before  = len(area_df)
    area_df = area_df.dropna(subset=SENSOR_TARGETS)
    dropped = before - len(area_df)
    if dropped > 0:
        logger.warning(
            f"Dropped {dropped} rows with remaining NaN after averaging."
        )

    logger.info(
        f"Final area dataset: {len(area_df)} rows "
        f"({area_df.index.min()} → {area_df.index.max()})"
    )
    return area_df