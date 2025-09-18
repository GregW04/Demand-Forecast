def make_lag(df, core_column, lag=[1], group_cols=None):
    df = df.copy()
    if group_cols:
        grouped = df.groupby(group_cols)[core_column]

    for value in lag:
        col_name = f'{core_column}_lag_{value}'
        if group_cols:
            df[col_name] = grouped.shift(value)
        else:
            df[col_name] = df[core_column].shift(value)
    return df


def make_rolling(df, core_column, rollag=[1], window=[7], rolling_stats=['mean', 'std', 'max', 'min', 'sum'], group_cols=None):
    df = df.copy()
    if group_cols:
        grouped = df.groupby(group_cols)[core_column]

    for stat in rolling_stats:
        for value in rollag:
            for w in window:
                col_name = f'{core_column}_roll{w}_{stat}_lag{value}'
                if group_cols:
                    df[col_name] = grouped.transform(lambda x: x.shift(value).rolling(w).agg(stat))
                else:
                    df[col_name] = df[core_column].shift(value).rolling(window=w).agg(stat)

    return df


def make_expanding(df, core_column, explag=[1], exp_stats=['mean', 'std', 'max', 'min', 'sum'], group_cols=None):
    df = df.copy()
    if group_cols:
        grouped = df.groupby(group_cols)[core_column]
    for stat in exp_stats:
        for value in explag:
            col_name = f'{core_column}_exp_{stat}_lag{value}'
            if group_cols:
                df[col_name] = grouped.transform(lambda x: x.shift(value).expanding().agg(stat))
            else:
                df[col_name] = df[core_column].shift(value).expanding().agg(stat)

    return df