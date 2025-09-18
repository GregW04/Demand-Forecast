def make_lag(df, core_column, lag=[1], group_cols=None):
    """
    Tworzy kolumny opóźnień (lag) dla wskazanej tabeli i kolumny.

    Parametry:
    - df : DataFrame wejściowy
    - core_column : nazwa kolumny, dla której liczymy lag
    - lag : lista wartości opóźnienia (np. [1, 7])
    - group_cols : lista kolumn do grupowania (opcjonalnie)

    Zwraca:
    - DataFrame z nowymi kolumnami *_lag_{value}
    """
    df = df.copy()

    # Jeśli podano kolumny grupujące
    if group_cols:
        grouped = df.groupby(group_cols)[core_column]

    for value in lag:
        col_name = f'{core_column}_lag_{value}'  # nazwa nowej kolumny
        if group_cols:
            df[col_name] = grouped.shift(value)  # lag wewnątrz grup
        else:
            df[col_name] = df[core_column].shift(value)  # lag bez grupowania
    return df


def make_rolling(df, core_column, rollag=[1], window=[7],
                 rolling_stats=['mean', 'std', 'max', 'min', 'sum'], group_cols=None):
    """
    Tworzy statystyki kroczące (rolling window) dla wskazanej tabeli i kolumny.

    Parametry:
    - df : DataFrame wejściowy
    - core_column : kolumna bazowa
    - rollag : lista opóźnień przed rozpoczęciem rolling (np. [1, 7])
    - window : lista długości okna (np. [7, 14])
    - rolling_stats : lista statystyk do obliczenia ('mean', 'std', 'max', 'min', 'sum')
    - group_cols : lista kolumn do grupowania (opcjonalnie)

    Zwraca:
    - DataFrame z nowymi kolumnami *_roll{window}_{stat}_lag{value}
    """
    df = df.copy()
    # Jeśli podano kolumny grupujące
    if group_cols:
        grouped = df.groupby(group_cols)[core_column]

    for stat in rolling_stats:
        for value in rollag:
            for w in window:
                col_name = f'{core_column}_roll{w}_{stat}_lag{value}'
                if group_cols:
                    # obliczanie rolling w ramach każdej grupy po wcześniejszym przesunięciu
                    df[col_name] = grouped.transform(lambda x: x.shift(value).rolling(w).agg(stat))
                else:
                    df[col_name] = df[core_column].shift(value).rolling(window=w).agg(stat)

    return df


def make_expanding(df, core_column, explag=[1],
                   exp_stats=['mean', 'std', 'max', 'min', 'sum'], group_cols=None):
    """
    Tworzy statystyki narastające (expanding window) dla wskazanej kolumny.

    Parametry:
    - df : DataFrame wejściowy
    - core_column : kolumna bazowa
    - explag : lista opóźnień przed rozpoczęciem expanding (np. [1])
    - exp_stats : lista statystyk do obliczenia ('mean', 'std', 'max', 'min', 'sum')
    - group_cols : lista kolumn do grupowania (opcjonalnie)

    Zwraca:
    - DataFrame z nowymi kolumnami *_exp_{stat}_lag{value}
    """
    df = df.copy()

    if group_cols:
        grouped = df.groupby(group_cols)[core_column]

    for stat in exp_stats:
        for value in explag:
            col_name = f'{core_column}_exp_{stat}_lag{value}'
            if group_cols:
                # statystyki expanding w ramach każdej grupy po wcześniejszym przesunięciu
                df[col_name] = grouped.transform(lambda x: x.shift(value).expanding().agg(stat))
            else:
                df[col_name] = df[core_column].shift(value).expanding().agg(stat)

    return df
