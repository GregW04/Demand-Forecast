# features_block.py
import pandas as pd
import numpy as np
import time

from date_features import extract_comprehensive_date_features
from promo_features import promo_features_all
import Lags as lg
from seasonality import create_seasonality_features


def _ensure_df(ret):
    """Ensure a DataFrame is returned (unwrap (df, meta) tuples if necessary)."""
    return ret[0] if isinstance(ret, tuple) else ret


# [CHANGE] Resolve group columns to those that actually exist in df
def _resolve_group_cols(df: pd.DataFrame, group_cols):
    """Return only group columns present in df; warn if some are missing."""
    if not group_cols:
        return []
    available = [c for c in group_cols if c in df.columns]
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing group cols {missing}; using {available or 'no grouping'}.")
    return available


def MAKE_FEATURES(df: pd.DataFrame,
                  target_col: str,
                  lags=[7, 14, 28],
                  rollag=[1],
                  explag=[1],
                  group_cols=None,
                  windows=[7, 14, 28],
                  rolling_stats=['mean', 'std', 'max', 'min', 'sum'],
                  exp_stats=['mean', 'std', 'max', 'min', 'sum'],
                  pre_sorted: bool = False,
                  timers: bool = False
                  ) -> pd.DataFrame:
    """
    Build dataset features for a given slice.
    """
    t0 = time.perf_counter()

    # [CHANGE] Ensure group_cols exist in df
    gcols = _resolve_group_cols(df, group_cols)

    # Date features
    df = _ensure_df(extract_comprehensive_date_features(df))
    if timers:
        print(f"[FEATS] date features in {time.perf_counter()-t0:.2f}s"); t0 = time.perf_counter()

    # Seasonality features
    df = _ensure_df(create_seasonality_features(df))  # [CHANGE]
    if timers:
        print(f"[FEATS] seasonality features in {time.perf_counter()-t0:.2f}s"); t0 = time.perf_counter()

    # Promo features
    df = _ensure_df(promo_features_all(df, group_cols=gcols, pre_sorted=pre_sorted))  # [CHANGE]
    if timers:
        print(f"[FEATS] promo features in {time.perf_counter()-t0:.2f}s"); t0 = time.perf_counter()

    # Lag features
    df = lg.make_lag(df, lag=lags, group_cols=gcols, core_column=target_col)  # [CHANGE]
    if timers:
        print(f"[FEATS] lag features in {time.perf_counter()-t0:.2f}s"); t0 = time.perf_counter()

    # Rolling features
    df = lg.make_rolling(df,
                         rollag=rollag,
                         window=windows,
                         core_column=target_col,
                         group_cols=gcols,  # [CHANGE]
                         rolling_stats=rolling_stats)
    if timers:
        print(f"[FEATS] rolling features in {time.perf_counter()-t0:.2f}s"); t0 = time.perf_counter()

    # Expanding features
    df = lg.make_expanding(df,
                           core_column=target_col,
                           explag=explag,
                           exp_stats=exp_stats,
                           group_cols=gcols)  # [CHANGE]
    if timers:
        print(f"[FEATS] expanding features in {time.perf_counter()-t0:.2f}s")

    return df


def compute_frozen_days_from_max_lag(max_lag: int, max_window: int) -> int:
    """
    Compute embargo (frozen days) to prevent leakage when using rolling windows.
    Often equals max_lag; here we use max_lag + max_window - 1.
    """
    return int(max_lag + max_window - 1)


def unique_dates(df: pd.DataFrame) -> np.ndarray:
    """
    Return sorted unique dates for stable window construction.
    """
    return np.sort(pd.to_datetime(df['date']).unique())


def build_rolling_windows(full_dates: np.ndarray,
                          n_windows: int,
                          val_size: int,
                          stride: int,
                          embargo: int,
                          min_train_days: int):
    """
    Build rolling time-series CV windows:
    returns [((train_start_date, train_end_date), (val_start_date, val_end_date)), ...]
    """
    windows = []
    n_dates = len(full_dates)

    train_start = 0  # training always starts from the beginning of the dataset
    for i in range(n_windows):
        # Positions (end as exclusive)
        val_end = n_dates - i * stride
        val_start = val_end - val_size
        train_end = val_start - embargo

        # Minimal train length requirement
        if train_end < min_train_days:
            print(f'Training set too small at CV window {i + 1}')
            break

        # Convert positions to inclusive indices
        train_range = (full_dates[train_start], full_dates[train_end - 1])
        val_range = (full_dates[val_start], full_dates[val_end - 1])
        windows.append((train_range, val_range))

    return windows


def SLICE_BY_DATE(df: pd.DataFrame,
                  date_range,
                  copy: bool = True) -> pd.DataFrame:
    """
    Slice DataFrame by inclusive date range [start, end].
    - date_range: tuple/list of two elements (start, end); any string-like is parsed via pd.to_datetime
    - copy: if True, returns a copy to avoid chained assignment warnings
    """
    if "date" not in df.columns:
        raise ValueError("SLICE_BY_DATE: 'date' column is required in df.")

    if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        raise ValueError("SLICE_BY_DATE: date_range must be a list/tuple of (start, end).")

    start, end = date_range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if start > end:
        start, end = end, start  # ensure start <= end

    # Build mask using a local datetime-converted series (does not mutate original df)
    date_vals = pd.to_datetime(df["date"]) if not pd.api.types.is_datetime64_any_dtype(df["date"]) else df["date"]
    mask = (date_vals >= start) & (date_vals <= end)

    result = df.loc[mask]
    return result.copy() if copy else result


def DROP_TARGET_COLS(df: pd.DataFrame,
                     target_col: str,
                     copy: bool = False) -> pd.DataFrame:
    """
    Drop target column from feature frame (no error if the column is absent).
    - target_col: name of the target column to drop
    - copy: if True, returns a copy after dropping
    """
    result = df.drop(columns=[target_col], errors="ignore")
    return result.copy() if copy else result

