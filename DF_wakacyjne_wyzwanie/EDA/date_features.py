"""
date_features.py
Moduł do ekstrakcji cech daty dla szeregów czasowych.
"""

import pandas as pd

def extract_comprehensive_date_features(df, date_column='date'):
    """
    Dodaje do DataFrame df bogaty zestaw cech daty na podstawie kolumny date_column.
    Zwraca (df z cechami, lista nazw cech).
    """
    df_processed = df.copy()
    df_processed[date_column] = pd.to_datetime(df_processed[date_column])
    df_processed['day_in_the_month'] = df_processed[date_column].dt.day
    df_processed['day_of_the_week'] = df_processed[date_column].dt.dayofweek
    df_processed['week_of_the_year'] = df_processed[date_column].dt.isocalendar().week
    df_processed['month'] = df_processed[date_column].dt.month
    df_processed['year'] = df_processed[date_column].dt.year
    df_processed['quarter'] = df_processed[date_column].dt.quarter
    df_processed['is_weekend'] = df_processed['day_of_the_week'].isin([5, 6])
    df_processed['is_month_start'] = df_processed[date_column].dt.is_month_start
    df_processed['is_month_end'] = df_processed[date_column].dt.is_month_end
    df_processed['is_quarter_start'] = df_processed[date_column].dt.is_quarter_start
    df_processed['is_quarter_end'] = df_processed[date_column].dt.is_quarter_end
    data_start = df_processed[date_column].min()
    df_processed['days_since_start'] = (df_processed[date_column] - data_start).dt.days
    features = ['day_in_the_month', 'day_of_the_week', 'week_of_the_year',
                'month', 'year', 'quarter', 'is_weekend', 'is_month_start',
                'is_month_end', 'is_quarter_start', 'is_quarter_end', 'days_since_start']
    return df_processed, features
