from pathlib import Path
import random
import pandas as pd
import numpy as np

def add_week_of_year(df: pd.DataFrame,
                     start_day: str | pd.Timestamp | None = None,
                     end_day: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Returns a copy of df with an additional column 'week_of_year'.
    start_day, end_day – optional; can be strings parsable by pd.to_datetime.
    """
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    if start_day is None:
        start_day = data['date'].min()
    else:
        start_day = pd.to_datetime(start_day)

    if end_day is None:
        end_day = data['date'].max()
    else:
        end_day = pd.to_datetime(end_day)

    data['week_of_year'] = pd.NA

    mask = (data['date'] >= start_day) & (data['date'] <= end_day)
    data.loc[mask, 'week_of_year'] = data.loc[mask, 'date'].dt.isocalendar().week

    return data


def add_month_of_year(df: pd.DataFrame,
                     start_day: str | pd.Timestamp | None = None,
                     end_day: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Returns a copy of df with an additional column 'month_of_year'.
    start_day, end_day – optional; can be strings parsable by pd.to_datetime.
    """
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    if start_day is None:
        start_day = data['date'].min()
    else:
        start_day = pd.to_datetime(start_day)

    if end_day is None:
        end_day = data['date'].max()
    else:
        end_day = pd.to_datetime(end_day)

    data['month_of_year'] = pd.NA

    mask = (data['date'] >= start_day) & (data['date'] <= end_day)
    data.loc[mask, 'month_of_year'] = data.loc[mask, 'date'].dt.month

    return data

def add_is_weekend(df: pd.DataFrame,
                     start_day: str | pd.Timestamp | None = None,
                     end_day: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Returns a copy of df with an additional column 'is_weekend'.
    start_day, end_day – optional; can be strings parsable by pd.to_datetime.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    if start_day is None:
        start_day = df['date'].min()
    else:
        start_day = pd.to_datetime(start_day)

    if end_day is None:
        end_day = df['date'].max()
    else:
        end_day = pd.to_datetime(end_day)

    df['is_weekend'] = pd.NA

    mask = (df['date'] >= start_day) & (df['date'] <= end_day)

    df.loc[mask, 'is_weekend'] = (
        df.loc[mask, 'date'].dt.dayofweek.isin([5, 6]).astype(int)
    )

    return df


def add_seasonal_idx_w(
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None
) -> pd.DataFrame:
    """
    Returns a copy of df with the column 'seasonal_idx_w' calculated as:
        (average sales in a given week of the year) /
        (average sales in the entire calendar year).

    Parameters:
    ----------
    start_year : int or None
        The first year for which the index is calculated. If None – starts from the smallest year in the data.
    end_year   : int or None
        The last year for which the index is calculated. If None – ends at the largest year in the data.

    Notes:
    ------
    * If the 'year' or 'week_of_year' columns already exist, the function does not overwrite them.
    * Rows outside the range of years will receive NaN in the 'seasonal_idx_w' column.
    """
    data = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    if 'year' not in data.columns:
        data['year'] = data['date'].dt.year
    if 'week_of_year' not in data.columns:
        data['week_of_year'] = data['date'].dt.isocalendar().week

    if start_year is None:
        start_year = data['year'].min()
    if end_year is None:
        end_year = data['year'].max()

    mask = (data['year'] >= start_year) & (data['year'] <= end_year)

    weekly_mean = (
        data.loc[mask]
        .groupby(['year', 'week_of_year'])['unit_sales']
        .mean()
        .rename('weekly_mean')
    )

    yearly_mean = (
        data.loc[mask]
        .groupby('year')['unit_sales']
        .mean()
        .rename('yearly_mean')
    )

    data = data.merge(weekly_mean, on=['year', 'week_of_year'], how='left')
    data = data.merge(yearly_mean, on='year', how='left')

    data['seasonal_idx_w'] = data['weekly_mean'] / data['yearly_mean']

    data.drop(columns=['weekly_mean', 'yearly_mean'], inplace=True)

    return data



def add_seasonal_idx_m(
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None
) -> pd.DataFrame:
    """
    Returns a copy of df with the column 'seasonal_idx_m' calculated as:
        (average sales in a given month of the year) /
        (average sales in the entire calendar year).

    Parameters:
    ----------
    start_year : int or None
        The first year for which the index is calculated. If None – starts from the smallest year in the data.
    end_year   : int or None
        The last year for which the index is calculated. If None – ends at the largest year in the data.

    Notes:
    ------
    * If the 'year' or 'month_of_year' columns already exist, the function does not overwrite them.
    * Rows outside the range of years will receive NaN in the 'seasonal_idx_w' column.
    """
    data = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    if 'year' not in data.columns:
        data['year'] = data['date'].dt.year
    if 'month_of_year' not in data.columns:
        data['month_of_year'] = data['date'].dt.month

    if start_year is None:
        start_year = data['year'].min()
    if end_year is None:
        end_year = data['year'].max()

    mask = (data['year'] >= start_year) & (data['year'] <= end_year)

    monthly_mean = (
        data.loc[mask]
        .groupby(['year', 'month_of_year'])['unit_sales']
        .mean()
        .rename('monthly_mean')
    )

    yearly_mean = (
        data.loc[mask]
        .groupby('year')['unit_sales']
        .mean()
        .rename('yearly_mean')
    )

    data = data.merge(monthly_mean, on=['year', 'month_of_year'], how='left')
    data = data.merge(yearly_mean, on='year', how='left')

    data['seasonal_idx_m'] = data['monthly_mean'] / data['yearly_mean']

    data.drop(columns=['monthly_mean', 'yearly_mean'], inplace=True)

    return data


def add_regular_coeff_w(df: pd.DataFrame,
                           start_year: int | None = None,
                           end_year: int | None = None) -> pd.DataFrame:
    """
    Adds a 'regular_coeff' column to the DataFrame, calculated based on the average
    values of 'seasonal_idx_w' from previous years for the same week (week_of_year).

    Weights:
        - most recent past year: 0.8
        - older years: 0.2 in total, distributed exponentially (the older the year, the smaller the weight)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the following columns:
        - 'date' (datetime or convertible to datetime)
        - 'unit_sales' (numeric)
        Optionally, it may contain 'year' and 'week_of_year' columns.

    start_year : int or None, default None
        The first year to include in the calculation. Rows from earlier years will be ignored.

    end_year : int or None, default None
        The last year to include in the calculation. Rows from later years will be ignored.

    Returns
    -------
    pd.DataFrame
        DataFrame extended with a 'regular_coeff' column containing the weighted average
        of historical 'seasonal_idx_w' values for the same week. 
        NaN appears when there is no historical data for that week.

    Notes
    -----
    - The function assumes a single time series and does not group by stores or products.
    - Computation is accelerated by pre-grouping by (year, week_of_year),
    avoiding apply(axis=1), which significantly reduces execution time.
    """

 
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    if 'year' not in data:
        data['year'] = data['date'].dt.year
    if 'week_of_year' not in data:
        data['week_of_year'] = data['date'].dt.isocalendar().week

    if 'seasonal_idx_w' not in data:
        data = add_seasonal_idx_w(data)

    if start_year is not None:
        data = data[data['year'] >= start_year]
    if end_year is not None:
        data = data[data['year'] <= end_year]

    weekly = (
        data.groupby(['year', 'week_of_year'])['seasonal_idx_w']
            .mean()
            .reset_index()
    )

    results = []
    for week, grp in weekly.groupby('week_of_year'):
        hist = grp.sort_values('year')
        for i, (curr_year, val) in enumerate(zip(hist['year'], hist['seasonal_idx_w'])):
            past = hist.iloc[:i]
            if past.empty:
                results.append((curr_year, week, np.nan))
                continue

            last = past.iloc[-1]
            older = past.iloc[:-1]

            weights = [0.8]
            values  = [last['seasonal_idx_w']]

            if not older.empty:
                base = np.exp(-np.arange(1, len(older)+1))
                base = base/base.sum()*0.2
                weights.extend(base)
                values.extend(older['seasonal_idx_w'])

            results.append((curr_year, week, np.average(values, weights=weights)))

    coeff_df = pd.DataFrame(results, columns=['year','week_of_year','regular_coeff'])
    return data.merge(coeff_df, on=['year','week_of_year'], how='left')

def add_regular_coeff_m(df: pd.DataFrame,
                           start_year: int | None = None,
                           end_year: int | None = None) -> pd.DataFrame:
    """
    Same as for regular_coeff_w but for months of the year.
    """

 
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    if 'year' not in data:
        data['year'] = data['date'].dt.year
    if 'month_of_year' not in data:
        data['month_of_year'] = data['date'].dt.month

    if 'seasonal_idx_m' not in data:
        data = add_seasonal_idx_m(data)

    if start_year is not None:
        data = data[data['year'] >= start_year]
    if end_year is not None:
        data = data[data['year'] <= end_year]

    weekly = (
        data.groupby(['year', 'month_of_year'])['seasonal_idx_m']
            .mean()
            .reset_index()
    )

    results = []
    for week, grp in weekly.groupby('month_of_year'):
        hist = grp.sort_values('year')
        for i, (curr_year, val) in enumerate(zip(hist['year'], hist['seasonal_idx_m'])):
            past = hist.iloc[:i]
            if past.empty:
                results.append((curr_year, week, np.nan))
                continue

            last = past.iloc[-1]
            older = past.iloc[:-1]

            weights = [0.8]
            values  = [last['seasonal_idx_m']]

            if not older.empty:
                base = np.exp(-np.arange(1, len(older)+1))
                base = base/base.sum()*0.2
                weights.extend(base)
                values.extend(older['seasonal_idx_m'])

            results.append((curr_year, week, np.average(values, weights=weights)))

    coeff_df = pd.DataFrame(results, columns=['year','month_of_year','regular_coeff'])
    return data.merge(coeff_df, on=['year','month_of_year'], how='left')


if __name__ == "__main__":
    pass
