"""
promo_features.py
Moduł z funkcjami do inżynierii cech promocyjnych na szeregach czasowych.
"""

import pandas as pd
import numpy as np

def add_days_since_last_promo_fast(df, group_cols=['store_nbr', 'item_nbr'], date_col='date', promo_col='onpromotion'):
    """
    Oblicza liczbę dni od ostatniej promocji dla każdej grupy (np. sklep-produkt).

    Parameters
    ----------
    df : pd.DataFrame
        Dane wejściowe z kolumnami daty, promocji i kluczy grupujących.
    group_cols : list, default ['store_nbr', 'item_nbr']
        Lista kolumn określających poziom agregacji (np. sklep, produkt).
    date_col : str, default 'date'
        Nazwa kolumny z datą.
    promo_col : str, default 'onpromotion'
        Nazwa kolumny z informacją o promocji (bool lub NaN).

    Returns
    -------
    pd.DataFrame
        DataFrame z nową kolumną 'days_since_last_promo' (liczba dni od ostatniej promocji).
    """
    # Sortowanie i wyznaczanie dat ostatniej promocji
        # Sortujemy dane po kluczach grupujących i dacie, aby zachować chronologię
    df_sorted = df.sort_values(group_cols + [date_col]).copy()
        # Tworzymy serię z datami, gdzie wystąpiła promocja (True), reszta NaN
    promo_dates = df_sorted[date_col].where(df_sorted[promo_col] == True)
        # Przenosimy ostatnią znaną datę promocji w dół (ffill) w obrębie grupy
    last_promo_date = promo_dates.groupby([df_sorted[c] for c in group_cols]).ffill()
        # Obliczamy liczbę dni od ostatniej promocji
    df_sorted['days_since_last_promo'] = (df_sorted[date_col] - last_promo_date).dt.days
        # Jeśli nie było wcześniejszej promocji, zostawiamy NaN
    df_sorted['days_since_last_promo'] = df_sorted['days_since_last_promo'].where(last_promo_date.notna(), np.nan)
    return df_sorted

def add_days_until_next_promo(df, group_cols=['store_nbr', 'item_nbr'], date_col='date', promo_col='onpromotion'):
    """
    Oblicza liczbę dni do najbliższej przyszłej promocji dla każdej grupy (np. sklep-produkt).

    Parameters
    ----------
    df : pd.DataFrame
        Dane wejściowe z kolumnami daty, promocji i kluczy grupujących.
    group_cols : list, default ['store_nbr', 'item_nbr']
        Lista kolumn określających poziom agregacji (np. sklep, produkt).
    date_col : str, default 'date'
        Nazwa kolumny z datą.
    promo_col : str, default 'onpromotion'
        Nazwa kolumny z informacją o promocji (bool lub NaN).

    Returns
    -------
    pd.DataFrame
        DataFrame z nową kolumną 'days_until_next_promo' (liczba dni do najbliższej promocji).
    """
    # Sortowanie i wyznaczanie dat przyszłej promocji
        # Sortujemy dane po kluczach grupujących i dacie, aby zachować chronologię
    df_sorted = df.sort_values(group_cols + [date_col]).copy()
        # Tworzymy serię z datami, gdzie wystąpi promocja (True), reszta NaN
    promo_dates = df_sorted[date_col].where(df_sorted[promo_col] == True)
        # Przenosimy najbliższą przyszłą datę promocji w górę (bfill) w obrębie grupy
    next_promo_date = promo_dates.groupby([df_sorted[c] for c in group_cols]).bfill()
        # Obliczamy liczbę dni do najbliższej promocji
    df_sorted['days_until_next_promo'] = (next_promo_date - df_sorted[date_col]).dt.days
        # Jeśli nie ma przyszłej promocji, zostawiamy NaN
    df_sorted['days_until_next_promo'] = df_sorted['days_until_next_promo'].where(next_promo_date.notna(), np.nan)
    return df_sorted

def add_promo_streak(df, group_cols=['store_nbr', 'item_nbr'], date_col='date', promo_col='onpromotion'):
    """
    Wyznacza długość aktualnej serii dni z promocją (ciąg kolejnych dni z True).

    Parameters
    ----------
    df : pd.DataFrame
        Dane wejściowe z kolumnami daty, promocji i kluczy grupujących.
    group_cols : list, default ['store_nbr', 'item_nbr']
        Lista kolumn określających poziom agregacji (np. sklep, produkt).
    date_col : str, default 'date'
        Nazwa kolumny z datą.
    promo_col : str, default 'onpromotion'
        Nazwa kolumny z informacją o promocji (bool lub NaN).

    Returns
    -------
    pd.DataFrame
        DataFrame z nową kolumną 'promo_streak' (długość serii promocji).
    """
    # Obliczanie długości serii promocji
        # Sortujemy dane po kluczach grupujących i dacie
    df_sorted = df.sort_values(group_cols + [date_col]).copy()
        # Funkcja pomocnicza do liczenia długości serii (streak)
    def streak_func(x):
        streak = (x != x.shift()).cumsum()
        return x.groupby(streak).cumcount() + 1
        # Tworzymy maskę dla dni z promocją
    mask = df_sorted[promo_col] == True
        # Inicjalizujemy kolumnę streak zerami
    df_sorted['promo_streak'] = 0
        # Dla dni z promocją liczymy długość aktualnej serii (ciągu True)
    df_sorted.loc[mask, 'promo_streak'] = (
        df_sorted[mask].groupby(group_cols)[promo_col].apply(streak_func).values
    )
    return df_sorted

def add_promo_next_7days_flag(df, group_cols=['store_nbr', 'item_nbr'], date_col='date', promo_col='onpromotion', window=7):
    """
    Sprawdza, czy w najbliższym oknie (domyślnie 7 dni) wystąpi promocja dla danej grupy.

    Parameters
    ----------
    df : pd.DataFrame
        Dane wejściowe z kolumnami daty, promocji i kluczy grupujących.
    group_cols : list, default ['store_nbr', 'item_nbr']
        Lista kolumn określających poziom agregacji (np. sklep, produkt).
    date_col : str, default 'date'
        Nazwa kolumny z datą.
    promo_col : str, default 'onpromotion'
        Nazwa kolumny z informacją o promocji (bool lub NaN).
    window : int, default 7
        Liczba dni do przodu, w których sprawdzana jest obecność promocji.

    Returns
    -------
    pd.DataFrame
        DataFrame z nową kolumną 'promo_in_next_7days' (bool: czy będzie promocja w oknie).
    """
    # Przesuwanie okna i sumowanie flag promocji
        # Sortujemy dane po kluczach grupujących i dacie
    df_sorted = df.sort_values(group_cols + [date_col]).copy()
        # Tworzymy DataFrame do zliczania przyszłych promocji w oknie
    future_promo = pd.DataFrame(0, index=df_sorted.index, columns=['future_promo'])
        # Przesuwamy okno do przodu o 1..window dni i sumujemy flagi promocji
    for i in range(1, window+1):
        future_promo['future_promo'] += df_sorted.groupby(group_cols)[promo_col].shift(-i).fillna(False).astype(int)
        # Jeśli w oknie pojawi się choć jedna promocja, ustawiamy flagę True
    df_sorted['promo_in_next_7days'] = future_promo['future_promo'] > 0
    return df_sorted


def promo_features_all(df: pd.DataFrame):
    df = add_days_since_last_promo_fast(df)
    df = add_days_until_next_promo(df)
    df = add_promo_streak(df)
    df = add_promo_next_7days_flag(df)
    return df


