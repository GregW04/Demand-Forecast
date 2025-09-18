"""
date_features.py
Moduł do ekstrakcji cech daty dla szeregów czasowych.
"""

import pandas as pd

def extract_comprehensive_date_features(df, date_column='date'):
    """
    Tworzy zestaw cech kalendarzowych na podstawie kolumny daty.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame wejściowy z kolumną daty.
    date_column : str, default 'date'
        Nazwa kolumny z datą, na podstawie której generowane są cechy.

    Returns
    -------
    tuple (pd.DataFrame, list)
        DataFrame z nowymi cechami oraz lista nazw tych cech.
    """
    # Kopiowanie danych wejściowych, aby nie modyfikować oryginału
    df_processed = df.copy()
    # Konwersja kolumny daty do typu datetime (jeśli nie jest)
    df_processed[date_column] = pd.to_datetime(df_processed[date_column])
    # Dzień miesiąca (1-31)
    df_processed['day_in_the_month'] = df_processed[date_column].dt.day
    # Dzień tygodnia (0=poniedziałek, 6=niedziela)
    df_processed['day_of_the_week'] = df_processed[date_column].dt.dayofweek
    # Numer tygodnia w roku
    df_processed['week_of_the_year'] = df_processed[date_column].dt.isocalendar().week
    # Numer miesiąca
    df_processed['month'] = df_processed[date_column].dt.month
    # Rok
    df_processed['year'] = df_processed[date_column].dt.year
    # Numer kwartału
    df_processed['quarter'] = df_processed[date_column].dt.quarter
    # Flaga: czy weekend (sobota lub niedziela)
    df_processed['is_weekend'] = df_processed['day_of_the_week'].isin([5, 6])
    # Flagi początku i końca miesiąca
    df_processed['is_month_start'] = df_processed[date_column].dt.is_month_start
    df_processed['is_month_end'] = df_processed[date_column].dt.is_month_end
    # Flagi początku i końca kwartału
    df_processed['is_quarter_start'] = df_processed[date_column].dt.is_quarter_start
    df_processed['is_quarter_end'] = df_processed[date_column].dt.is_quarter_end
    # Liczba dni od początku zbioru (przydatne do trendów)
    data_start = df_processed[date_column].min()
    df_processed['days_since_start'] = (df_processed[date_column] - data_start).dt.days
    # Lista nazw wygenerowanych cech
    features = ['day_in_the_month', 'day_of_the_week', 'week_of_the_year',
                'month', 'year', 'quarter', 'is_weekend', 'is_month_start',
                'is_month_end', 'is_quarter_start', 'is_quarter_end', 'days_since_start']
    return df_processed, features
