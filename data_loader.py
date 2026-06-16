import os
import json
import logging
import tempfile
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from config import SENSOR_TARGETS, PROTOTYPE_KEYS, FIELD_MAP
from data_cleaner import clean_and_average

# ── Logger setup ────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────
def _fetch_with_timeout(ref_path: str, timeout: int = 60):  # increased from 10
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(db.reference(ref_path).get)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"[Firebase] Fetch timed out: {ref_path}")


# ── Firebase init ────────────────────────────────────────────────
def init_firebase():
    if firebase_admin._apps:
        return

    cert_dict = json.loads(os.environ['FIREBASE_CERT'])
    cred = credentials.Certificate(cert_dict)
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.environ['FIREBASE_URL']
    })
    logger.info("[Firebase] Connected successfully.")


# ── Data fetching ────────────────────────────────────────────────
def parse_node(raw: dict) -> pd.DataFrame:
    records = list(raw.values())
    df = pd.DataFrame(records)

    if 'timestamp' not in df.columns:
        raise ValueError(f"[parse_node] Missing 'timestamp' column. Got: {list(df.columns)}")

    # Fix — two separate steps so tz_convert works on a DatetimeIndex
    df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.index = df.index.tz_convert('Asia/Manila')
    df.index.name = 'timestamp'

    df = df.drop(columns=['timestamp'])
    df = df.rename(columns=FIELD_MAP)

    available = [c for c in SENSOR_TARGETS if c in df.columns]

    if not available:
        raise ValueError(f"[parse_node] No expected sensor columns found. Got: {list(df.columns)}")

    df = df[available].astype(float)

    if 'O3' in df.columns:
        df['O3'] = df['O3'] / 1000

    df = df.sort_index()
    return df


def fetch_and_average() -> pd.DataFrame:
    raw = {}
    for key in PROTOTYPE_KEYS:
        try:
            node_data = _fetch_with_timeout(f'/sensor_data/{key}')
        except TimeoutError as e:
            logger.warning(f"{e} — skipping {key}.")
            continue

        if not node_data:
            logger.warning(f"[Firebase] No data found for {key} — skipping.")
            continue

        df = parse_node(node_data)
        raw[key] = df
        logger.info(f"[Firebase] {key}: {len(df)} records loaded.")

    if not raw:
        raise ValueError("No sensor data retrieved from Firebase — all nodes empty or timed out.")

    return clean_and_average(raw)