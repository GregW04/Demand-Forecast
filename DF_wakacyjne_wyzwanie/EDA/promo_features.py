import pandas as pd
import numpy as np

# Robust helper: convert an arbitrary promo-like Series to pandas BooleanDtype (True/False/NA), preserving NA.
def _promo_to_boolean_na(s: pd.Series) -> pd.Series:
    """
    Normalize a promo-like column into a pandas BooleanDtype (True/False/NA), preserving NA.
    Strategy:
    - Try numeric coercion (0 -> False, non-zero -> True, NaN -> <NA>)
    - Fallback: map common textual tokens to booleans and <NA>
    """
    # Try numeric route first
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        out = (num != 0)
        out = out.astype("boolean")  # convert to BooleanDtype
        out[num.isna()] = pd.NA      # preserve NA
        return out

    # Fallback to textual mapping
    s_str = s.astype("string").str.strip().str.lower()
    mapping = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
        "nan": pd.NA, "none": pd.NA, "": pd.NA
    }
    out = s_str.map(mapping)
    return pd.Series(out, index=s.index, dtype="boolean")


def add_days_since_last_promo_fast(df,
                                   group_cols=['store_nbr', 'item_nbr'],
                                   date_col='date',
                                   promo_col='onpromotion',
                                   pre_sorted: bool = False):
    """
    Compute days since last promo; preserves NA in onpromotion.
    """
    df_sorted = df if pre_sorted else df.sort_values(group_cols + [date_col])
    df_sorted = df_sorted.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])

    promo_flag = _promo_to_boolean_na(df_sorted[promo_col])
    true_mask = promo_flag.fillna(False)

    promo_dates = df_sorted[date_col].where(true_mask)
    # [CHANGE] Use Series groupers (list of arrays) for Series.groupby
    if group_cols:
        groupers = [df_sorted[c] for c in group_cols]
        last_promo_date = promo_dates.groupby(groupers, sort=False).ffill()
    else:
        last_promo_date = promo_dates.ffill()

    df_sorted['days_since_last_promo'] = (df_sorted[date_col] - last_promo_date).dt.days
    df_sorted['days_since_last_promo'] = df_sorted['days_since_last_promo'].where(last_promo_date.notna(), np.nan)
    return df_sorted


def add_days_until_next_promo(df,
                              group_cols=['store_nbr', 'item_nbr'],
                              date_col='date',
                              promo_col='onpromotion',
                              pre_sorted: bool = False):
    """
    Compute days until next promo; preserves NA in onpromotion.
    """
    df_sorted = df if pre_sorted else df.sort_values(group_cols + [date_col])
    df_sorted = df_sorted.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])

    promo_flag = _promo_to_boolean_na(df_sorted[promo_col])
    true_mask = promo_flag.fillna(False)

    promo_dates = df_sorted[date_col].where(true_mask)
    # [CHANGE] Series groupers for Series.groupby
    if group_cols:
        groupers = [df_sorted[c] for c in group_cols]
        next_promo_date = promo_dates.groupby(groupers, sort=False).bfill()
    else:
        next_promo_date = promo_dates.bfill()

    df_sorted['days_until_next_promo'] = (next_promo_date - df_sorted[date_col]).dt.days
    df_sorted['days_until_next_promo'] = df_sorted['days_until_next_promo'].where(next_promo_date.notna(), np.nan)
    return df_sorted


def add_promo_streak(df,
                     group_cols=['store_nbr', 'item_nbr'],
                     date_col='date',
                     promo_col='onpromotion',
                     pre_sorted: bool = False):
    """
    Compute current promo streak length; preserves NA where onpromotion is NA.
    """
    df_sorted = df if pre_sorted else df.sort_values(group_cols + [date_col])
    df_sorted = df_sorted.copy()

    promo_flag = _promo_to_boolean_na(df_sorted[promo_col])
    true_mask = promo_flag.fillna(False)

    def _streak_true(s: pd.Series) -> pd.Series:
        blocks = (s != s.shift()).cumsum()
        runlen = blocks.groupby(blocks).cumcount() + 1
        return runlen.where(s, 0)

    # [CHANGE] Use Series groupers if grouping; otherwise compute once
    if group_cols:
        groupers = [df_sorted[c] for c in group_cols]
        run = true_mask.groupby(groupers, sort=False).transform(_streak_true).astype('Int16')
    else:
        run = _streak_true(true_mask).astype('Int16')

    # Restore NA where original flag was NA
    run[promo_flag.isna()] = pd.NA
    df_sorted['promo_streak'] = run
    return df_sorted


def add_promo_next_7days_flag(df,
                              group_cols=['store_nbr', 'item_nbr'],
                              date_col='date',
                              promo_col='onpromotion',
                              window=7,
                              pre_sorted: bool = False):
    """
    Flag (Int8 with <NA>) if any promo occurs within next 'window' days per group.
    """
    df_sorted = df if pre_sorted else df.sort_values(group_cols + [date_col])
    df_sorted = df_sorted.copy()

    promo_flag = _promo_to_boolean_na(df_sorted[promo_col])

    future_true = pd.Series(0, index=df_sorted.index, dtype='Int16')
    future_na = pd.Series(0, index=df_sorted.index, dtype='Int16')

    # [CHANGE] Use Series groupers when grouping; else shift on entire series
    if group_cols:
        groupers = [df_sorted[c] for c in group_cols]
        for i in range(1, window + 1):
            shifted = promo_flag.groupby(groupers, sort=False).shift(-i)
            future_true += shifted.fillna(False).astype('Int8').astype('Int16')
            future_na += shifted.isna().astype('Int8').astype('Int16')
    else:
        for i in range(1, window + 1):
            shifted = promo_flag.shift(-i)
            future_true += shifted.fillna(False).astype('Int8').astype('Int16')
            future_na += shifted.isna().astype('Int8').astype('Int16')

    out = pd.Series(pd.NA, index=df_sorted.index, dtype="Int8")
    out[future_true > 0] = 1
    out[(future_true == 0) & (future_na == 0)] = 0

    df_sorted['promo_in_next_7days'] = out
    return df_sorted


def promo_features_all(df: pd.DataFrame,
                       group_cols=['store_nbr', 'item_nbr'],
                       date_col='date',
                       promo_col='onpromotion',
                       pre_sorted: bool = False,
                       window: int = 7):
    """
    Compose promo features (NA-preserving) for the given grouping.
    """
    df = add_days_since_last_promo_fast(df, group_cols=group_cols, date_col=date_col,
                                        promo_col=promo_col, pre_sorted=pre_sorted)
    df = add_days_until_next_promo(df, group_cols=group_cols, date_col=date_col,
                                   promo_col=promo_col, pre_sorted=pre_sorted)
    df = add_promo_streak(df, group_cols=group_cols, date_col=date_col,
                          promo_col=promo_col, pre_sorted=pre_sorted)
    df = add_promo_next_7days_flag(df, group_cols=group_cols, date_col=date_col,
                                   promo_col=promo_col, window=window, pre_sorted=pre_sorted)
    return df
