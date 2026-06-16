"""
visualize.py — AirHaven Thesis Graph Generator
================================================
Generates all evaluation graphs needed for Chapter 4:
  1. R² Comparison        — Hourly vs Daily per target (bar chart)
  2. Overfitting Gap       — Train R² vs Test R² per target (grouped bar)
  3. Metric Heatmap        — MAE, RMSE, MAPE, MedAE across all targets
  4. Actual vs Predicted   — Line plots per target (hourly & daily)

Usage:
    python visualize.py

Output:
    graphs/01_r2_comparison.png
    graphs/02_overfitting_gap.png
    graphs/03_metric_heatmap_hourly.png
    graphs/03_metric_heatmap_daily.png
    graphs/04_actual_vs_predicted_hourly.png
    graphs/04_actual_vs_predicted_daily.png

Requirements:
    pip install matplotlib seaborn pandas numpy scikit-learn joblib
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from config import SENSOR_TARGETS
from config import (
    HOURLY_MODEL_PATH, HOURLY_SCALER_PATH, HOURLY_META_PATH,
    DAILY_MODEL_PATH,  DAILY_SCALER_PATH,  DAILY_META_PATH,
    HISTORY_PATH,
)
from features import build_hourly_features, build_daily_features

# ── Output directory ─────────────────────────────────────────────
OUT_DIR = 'graphs'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi':   150,
})

COLORS = {
    'hourly':      '#2E86AB',
    'daily':       '#E84855',
    'train':       '#3BB273',
    'test':        '#F4A261',
    'actual':      '#264653',
    'predicted':   '#E76F51',
}

TARGET_LABELS = {
    'humidity':    'Humidity (%)',
    'temperature': 'Temperature (°C)',
    'NO2':         'NO₂ (ppb)',
    'O3':          'O₃ (ppm)',
    'PM10':        'PM10 (µg/m³)',
    'PM2_5':       'PM2.5 (µg/m³)',
    'co':          'CO (ppm)',
}


# ── Helpers ──────────────────────────────────────────────────────
def _mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100)


def _load_artifacts(model_path, scaler_path, meta_path):
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, scaler, meta


def _time_split(df, test_ratio=0.2):
    n = int(len(df) * (1 - test_ratio))
    return df.iloc[:n], df.iloc[n:]


def _compute_metrics(model, scaler, feature_cols, featured_df):
    """Returns metrics dict and (y_test_df, y_pred_array, test_index)."""
    present = [t for t in SENSOR_TARGETS if t in featured_df.columns]
    train_df, test_df = _time_split(featured_df)

    X_tr = train_df[feature_cols].fillna(0)
    y_tr = train_df[present].ffill().fillna(0)
    X_te = test_df[feature_cols].fillna(0)
    y_te = test_df[present].ffill().fillna(0)

    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    y_pred_test  = model.predict(X_te_s)
    y_pred_train = model.predict(X_tr_s)

    metrics = {}
    for i, t in enumerate(present):
        actual    = y_te[t].values
        predicted = y_pred_test[:, i]

        if len(actual) < 2:
            print(f"  [!] Skipping {t} — test set too small ({len(actual)} sample).")
            continue

        r2_test   = r2_score(actual, predicted)
        r2_train  = r2_score(y_tr[t].values, y_pred_train[:, i])
        gap       = r2_train - r2_test

        if r2_test < 0.5 and gap < 0.05:
            status = 'Underfitting'
        elif gap < 0.05:
            status = 'Well generalized'
        elif gap < 0.10:
            status = 'Minor overfitting'
        elif gap < 0.20:
            status = 'Moderate overfitting'
        else:
            status = 'Severe overfitting'

        metrics[t] = {
            'MAE':      round(float(mean_absolute_error(actual, predicted)), 4),
            'RMSE':     round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
            'R2_test':  round(float(r2_test), 4),
            'R2_train': round(float(r2_train), 4),
            'MAPE':     round(float(_mape(actual, predicted)), 4),
            'MedAE':    round(float(median_absolute_error(actual, predicted)), 4),
            'Overfit_gap':    round(float(gap), 4),
            'Overfit_status': status,
        }

    return metrics, y_te, y_pred_test, test_df.index


def _load_history():
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(
            f"History file not found at '{HISTORY_PATH}'.\n"
            "Run the main pipeline first to generate area_history.csv."
        )
    df = pd.read_csv(HISTORY_PATH, index_col='timestamp', parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Manila')
    else:
        df.index = df.index.tz_convert('Asia/Manila')
    return df


def _insert_gaps(test_index, values, max_gap_hours=2):
    """Insert NaN where time gaps exceed max_gap_hours to show breaks in the line."""
    idx = pd.DatetimeIndex(test_index)
    gaps = idx.to_series().diff() > pd.Timedelta(hours=max_gap_hours)
    gap_positions = np.where(gaps)[0]

    new_index  = list(test_index)
    new_values = list(values)

    for offset, pos in enumerate(gap_positions):
        insert_at = pos + offset
        mid_time  = test_index[pos - 1] + (test_index[pos] - test_index[pos - 1]) / 2
        new_index.insert(insert_at, mid_time)
        new_values.insert(insert_at, np.nan)

    return new_index, new_values


# ── Graph 1: R² Comparison (Hourly vs Daily) ─────────────────────
def plot_r2_comparison(h_metrics, d_metrics):
    targets  = list(h_metrics.keys())
    h_r2     = [h_metrics[t]['R2_test'] for t in targets]
    d_r2     = [d_metrics[t]['R2_test']  if t in d_metrics else 0 for t in targets]
    labels   = [TARGET_LABELS.get(t, t) for t in targets]

    x    = np.arange(len(targets))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    bars1 = ax.bar(x - w/2, h_r2, w, label='Hourly Model', color=COLORS['hourly'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, d_r2, w, label='Daily Model',  color=COLORS['daily'],  edgecolor='white', linewidth=0.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8.5)

    ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, label='R²=0.80 threshold')
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('R² Score (Test Set)')
    ax.set_title('Figure 1. R² Score Comparison — Hourly vs. Daily Gradient Boosting Model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylim(0.0, 1.15)
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, '01_r2_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {path}")


# ── Graph 2: Overfitting Gap ──────────────────────────────────────
def plot_overfitting_gap(h_metrics, d_metrics, label, metrics):
    targets  = list(metrics.keys())
    if not targets:
        print(f"[!] Skipping overfitting gap for {label} — no data.")
        return
    # Skip if all R2 values are non-finite
    if not any(np.isfinite(metrics[t]['R2_test']) for t in targets):
        print(f"[!] Skipping overfitting gap for {label} — all R² values undefined (too few test samples).")
        plt.close('all')
        return

    r2_train = [metrics[t]['R2_train'] for t in targets]
    r2_test  = [metrics[t]['R2_test']  for t in targets]
    statuses = [metrics[t]['Overfit_status'] for t in targets]
    labels   = [TARGET_LABELS.get(t, t) for t in targets]

    x = np.arange(len(targets))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_train = ax.bar(x - w/2, r2_train, w, label='R² Train', color=COLORS['train'], edgecolor='white')
    bars_test  = ax.bar(x + w/2, r2_test,  w, label='R² Test',  color=COLORS['test'],  edgecolor='white')

    # Value labels on each bar
    for bar in bars_train:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=7.5, color='#1a1a1a')

    for bar in bars_test:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=7.5, color='#1a1a1a')

    # Overfitting status label above each group
    for i, (tr, te, st) in enumerate(zip(r2_train, r2_test, statuses)):
        color = '#27ae60' if 'Well' in st else ('#e67e22' if 'Minor' in st else ('#f39c12' if 'Moderate' in st else ('#c0392b' if 'Severe' in st else '#8e44ad')))
        y_pos = max(tr, te)
        if np.isfinite(y_pos):
            ax.text(i, y_pos + 0.06, st, ha='center', va='bottom',
                    fontsize=7.5, color=color, fontweight='bold')

    cap = label.capitalize()
    ax.set_title(f'Figure 47. Train vs. Test R² — {cap} Model (Overfitting Analysis)')
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('R² Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylim(0.0, 1.15)
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    path = os.path.join(OUT_DIR, f'02_overfitting_gap_{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {path}")


# ── Graph 3: Actual vs Predicted ─────────────────────────────────
def plot_actual_vs_predicted(y_test_df, y_pred_array, test_index, label):
    present = [t for t in SENSOR_TARGETS if t in y_test_df.columns]
    n       = len(present)
    ncols   = 2
    nrows   = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = axes.flatten()

    for i, t in enumerate(present):
        ax     = axes[i]
        actual = y_test_df[t].values
        pred   = y_pred_array[:, i]
        r2     = r2_score(actual, pred)

        # ── Insert NaN at data gaps so the line breaks visually ──
        idx_actual, vals_actual = _insert_gaps(test_index, actual)
        idx_pred,   vals_pred   = _insert_gaps(test_index, pred)

        ax.plot(idx_actual, vals_actual, color=COLORS['actual'],
                linewidth=1.2, label='Actual', alpha=0.9)
        ax.plot(idx_pred,   vals_pred,   color=COLORS['predicted'],
                linewidth=1.2, label='Predicted', alpha=0.85, linestyle='--')

        ax.set_title(f'{TARGET_LABELS.get(t, t)}  (R²={r2:.4f})', fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=30)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    cap = label.capitalize()
    fig.suptitle(f'Figure 48. Actual vs. Predicted — {cap} Model (Test Set)', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'04_actual_vs_predicted_{label}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {path}")


# ── Graph 4: Actual vs Predicted Scatter ─────────────────────────
def plot_scatter(y_test_df, y_pred_array, label):
    present = [t for t in SENSOR_TARGETS if t in y_test_df.columns]
    n       = len(present)
    ncols   = 2
    nrows   = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, t in enumerate(present):
        ax     = axes[i]
        actual = y_test_df[t].values
        pred   = y_pred_array[:, i]
        r2     = r2_score(actual, pred)
        gap    = None

        # Determine status label
        if r2 < 0.5:
            status = 'Underfitting'
            color  = '#8e44ad'
        elif r2 >= 0.80:
            status = 'Good Fit'
            color  = '#27ae60'
        else:
            status = 'Moderate Fit'
            color  = '#e67e22'

        # Scatter plot
        ax.scatter(actual, pred, alpha=0.6, color=color,
                   edgecolors='white', linewidth=0.5, s=60)

        # Perfect prediction line (diagonal)
        all_vals = np.concatenate([actual, pred])
        mn, mx   = all_vals.min(), all_vals.max()
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.2,
                label='Perfect Prediction')

        ax.set_title(f'{TARGET_LABELS.get(t, t)}\nR²={r2:.4f} — {status}',
                     fontsize=10)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    cap = label.capitalize()
    fig.suptitle(f'Figure 5. Actual vs. Predicted Scatter Plot — {cap} Model\n'
                 f'(Points along diagonal = perfect prediction)',
                 fontsize=13)

    path = os.path.join(OUT_DIR, f'05_scatter_{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  AirHaven — Thesis Graph Generator")
    print("="*55)

    # Load history
    print("\n[1/3] Loading history data...")
    area_df = _load_history()
    print(f"      Loaded {len(area_df)} rows.")

    # ── Hourly ───────────────────────────────────────────────────
    print("\n[2/3] Processing HOURLY model...")
    h_model, h_scaler, h_meta = _load_artifacts(
        HOURLY_MODEL_PATH, HOURLY_SCALER_PATH, HOURLY_META_PATH
    )
    h_featured   = build_hourly_features(area_df)
    h_feat_cols  = h_meta['feature_columns']
    h_metrics, h_y_te, h_y_pred, h_idx = _compute_metrics(
        h_model, h_scaler, h_feat_cols, h_featured
    )

    # ── Daily ────────────────────────────────────────────────────
    print("\n[3/3] Processing DAILY model...")
    d_model, d_scaler, d_meta = _load_artifacts(
        DAILY_MODEL_PATH, DAILY_SCALER_PATH, DAILY_META_PATH
    )
    d_featured  = build_daily_features(area_df)
    d_feat_cols = d_meta['feature_columns']
    d_metrics, d_y_te, d_y_pred, d_idx = _compute_metrics(
        d_model, d_scaler, d_feat_cols, d_featured
    )

    # ── Generate graphs ──────────────────────────────────────────
    print("\n" + "-"*55)
    print("  Generating graphs...")
    print("-"*55)

    plot_r2_comparison(h_metrics, d_metrics)
    plot_overfitting_gap(h_metrics, d_metrics, 'hourly', h_metrics)
    plot_overfitting_gap(h_metrics, d_metrics, 'daily',  d_metrics)
    plot_actual_vs_predicted(h_y_te, h_y_pred, h_idx, 'hourly')
    plot_actual_vs_predicted(d_y_te, d_y_pred, d_idx, 'daily')
    plot_scatter(h_y_te, h_y_pred, 'hourly')
    plot_scatter(d_y_te, d_y_pred, 'daily')

    print("\n" + "="*55)
    print(f"  Done! All graphs saved to './{OUT_DIR}/'")
    print("="*55 + "\n")


if __name__ == '__main__':
    main()