import pandas as pd
import numpy as np
from config import TARGETS, SENSOR_TARGETS


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_sin']  = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df.index.hour / 24)
    df['dow_sin']   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    return df


def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[SENSOR_TARGETS].resample('1h').mean()
    df = df.dropna(subset=SENSOR_TARGETS)
    df = _add_time_features(df)

    for col in SENSOR_TARGETS:
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'{col}_lag{lag}h'] = df[col].shift(lag)

    for col in SENSOR_TARGETS:
        df[f'{col}_roll3h_mean']  = df[col].rolling(3).mean()
        df[f'{col}_roll6h_mean']  = df[col].rolling(6).mean()
        df[f'{col}_roll12h_mean'] = df[col].rolling(12).mean()
        df[f'{col}_roll3h_std']   = df[col].rolling(3).std()
        df[f'{col}_roll6h_std']   = df[col].rolling(6).std()

    df = df.dropna()
    return df


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    daily = df[SENSOR_TARGETS].resample('D').mean()
    daily = daily.dropna(subset=SENSOR_TARGETS)
    daily = _add_time_features(daily)

    for col in SENSOR_TARGETS:
        for lag in [1, 2, 3, 7]:
            daily[f'{col}_lag{lag}d'] = daily[col].shift(lag)

    for col in SENSOR_TARGETS:
        daily[f'{col}_roll3d_mean'] = daily[col].rolling(3).mean()
        daily[f'{col}_roll7d_mean'] = daily[col].rolling(7).mean()
        daily[f'{col}_roll3d_std']  = daily[col].rolling(3).std()

    daily = daily.dropna()
    return daily