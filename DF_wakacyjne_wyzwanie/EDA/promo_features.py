"""
promo_features.py
Moduł z funkcjami do inżynierii cech promocyjnych na szeregach czasowych.
"""

import pandas as pd
import numpy as np

def add_days_since_last_promo_fast(df, group_cols=['store_nbr', 'item_nbr'], date_col='date', promo_col='onpromotion'):
    """
    Dodaje kolumnę 'days_since_last_promo' - liczba dni od ostatniej promocji dla danej grupy.
    """
    df_sorted = df.sort_values(group_cols + [date_col]).copy()
    promo_dates = df_sorted[date_col].where(df_sorted[promo_col] == True)
    last_promo_date = promo_dates.groupby([df_sorted[c] for c in group_cols]).ffill()
    df_sorted['days_since_last_promo'] = (df_sorted[date_col] - last_promo_date).dt.days
    df_sorted['days_since_last_promo'] = df_sorted['days_since_last_promo'].where(last_promo_date.notna(), np.nan)
    return df_sorted

def add_days_until_next_promo(df, group_cols=['store_nbr', 'item_nbr'], date_col='date', promo_col='onpromotion'):
    """
    Dodaje kolumnę 'days_until_next_promo' - liczba dni do najbliższej przyszłej promocji dla danej grupy.
    """
    df_sorted = df.sort_values(group_cols + [date_col]).copy()
    promo_dates = df_sorted[date_col].where(df_sorted[promo_col] == True)
    next_promo_date = promo_dates.groupby([df_sorted[c] for c in group_cols]).bfill()
    df_sorted['days_until_next_promo'] = (next_promo_date - df_sorted[date_col]).dt.days
    df_sorted['days_until_next_promo'] = df_sorted['days_until_next_promo'].where(next_promo_date.notna(), np.nan)
    return df_sorted

def add_promo_streak(df, group_cols=['store_nbr', 'item_nbr'], date_col='date', promo_col='onpromotion'):
    """
    Dodaje kolumnę 'promo_streak' - długość aktualnej serii promocji (ciąg kolejnych dni z promocją).
    """
    df_sorted = df.sort_values(group_cols + [date_col]).copy()
    def streak_func(x):
        streak = (x != x.shift()).cumsum()
        return x.groupby(streak).cumcount() + 1
    mask = df_sorted[promo_col] == True
    df_sorted['promo_streak'] = 0
    df_sorted.loc[mask, 'promo_streak'] = (
        df_sorted[mask].groupby(group_cols)[promo_col].apply(streak_func).values
    )
    return df_sorted

def add_promo_next_7days_flag(df, group_cols=['store_nbr', 'item_nbr'], date_col='date', promo_col='onpromotion', window=7):
    """
    Dodaje kolumnę 'promo_in_next_7days' - czy w ciągu najbliższych 7 dni będzie promocja (bool).
    """
    df_sorted = df.sort_values(group_cols + [date_col]).copy()
    future_promo = pd.DataFrame(0, index=df_sorted.index, columns=['future_promo'])
    for i in range(1, window+1):
        future_promo['future_promo'] += df_sorted.groupby(group_cols)[promo_col].shift(-i).fillna(False).astype(int)
    df_sorted['promo_in_next_7days'] = future_promo['future_promo'] > 0
    return df_sorted
