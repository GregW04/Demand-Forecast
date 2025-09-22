# features_block.py
import pandas as pd
import numpy as np

# External feature builders used inside MAKE_FEATURES
from date_features import extract_comprehensive_date_features
from promo_features import promo_features_all
import Lags as lg


def MAKE_FEATURES(df: pd.DataFrame,
                  target_col: str,
                  lags=[7, 14, 28],
                  rollag=[1],
                  explag=[1],
                  group_cols=None,
                  windows=[7, 14, 28],
                  rolling_stats=['mean', 'std', 'max', 'min', 'sum'],
                  exp_stats=['mean', 'std', 'max', 'min', 'sum']) -> pd.DataFrame:
    """
    Build dataset features for a given slice:
    - date-related features
    - promo-related features
    - lag/rolling/expanding statistics for the target column
    """
    # Date features
    df = extract_comprehensive_date_features(df)

    # Promo features (uses already joined promo/oil/holiday signals)
    df = promo_features_all(df)

    # Lag features (uses shift(+))
    df = lg.make_lag(df, lag=lags, group_cols=group_cols, core_column=target_col)

    # Moving window features (shift(1) then rolling)
    df = lg.make_rolling(df,
                         rollag=rollag,
                         window=windows,
                         core_column=target_col,
                         group_cols=group_cols,
                         rolling_stats=rolling_stats)

    # Expanding features (closed='left')
    df = lg.make_expanding(df,
                           core_column=target_col,
                           explag=explag,
                           exp_stats=exp_stats,
                           group_cols=group_cols)
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
