"""
handling_nan_values.py
Moduł do obsługi braków danych (NaN) w projekcie Demand-Forecast.
Zawiera funkcje do:
- sprawdzania brakujących dat w oil i train,
- wyłapywania i uzupełniania NaN,
- raportowania braków.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def fill_oil_pchip(oil_df, value_col='dcoilwtico', date_col='date'):
	"""
	Uzupełnia braki w kolumnie value_col w DataFrame oil_df za pomocą interpolacji PCHIP.
	Zwraca nowy DataFrame z dodatkową kolumną '<value_col>_filled'.
	"""
	oil_processed = oil_df.copy()
	oil_processed[date_col] = pd.to_datetime(oil_processed[date_col])
	oil_processed['date_numeric'] = (oil_processed[date_col] - oil_processed[date_col].min()).dt.days
	non_null_mask = oil_processed[value_col].notnull()
	x_known = oil_processed.loc[non_null_mask, 'date_numeric'].values
	y_known = oil_processed.loc[non_null_mask, value_col].values
	x_all = oil_processed['date_numeric'].values
	print(f"Input data: {len(x_known)} known prices, {len(x_all)} total dates")
	if len(x_known) >= 4:
		pchip_func = interpolate.PchipInterpolator(x_known, y_known)
		oil_processed[f'{value_col}_filled'] = pchip_func(x_all)
		print("PCHIP interpolation applied successfully")
	else:
		oil_processed[f'{value_col}_filled'] = oil_processed[value_col].interpolate(method='linear')
		oil_processed[f'{value_col}_filled'] = oil_processed[f'{value_col}_filled'].fillna(method='bfill')
		print(" Insufficient points for PCHIP, used linear fallback")
	return oil_processed

def plot_oil_interpolation(oil_processed, value_col='dcoilwtico', filled_col='dcoilwtico_filled'):
	"""
	Rysuje wykres interpolacji cen ropy: oryginalne dane, wartości uzupełnione (PCHIP/linear), podgląd okresu 2016-2017.
	"""
	filled_mask = oil_processed[value_col].isnull()
	plt.figure(figsize=(14, 8))
	# Wykres 1: pełny zakres
	plt.subplot(2, 1, 1)
	plt.plot(oil_processed['date'], oil_processed[value_col], 'o', alpha=0.7, label='Original Data', markersize=4, color='blue')
	plt.plot(oil_processed['date'], oil_processed[filled_col], '-', alpha=0.8, label='PCHIP Interpolation', linewidth=2, color='green')
	if filled_mask.any():
		filled_dates = oil_processed.loc[filled_mask, 'date']
		plt.scatter(filled_dates, oil_processed.loc[filled_mask, filled_col], 
				   color='red', s=30, alpha=0.9, label=f'Filled Points ({filled_mask.sum()})', zorder=5)
	plt.title('Oil Price Interpolation - PCHIP Method', fontsize=14, fontweight='bold')
	plt.ylabel('Oil Price ($)')
	plt.legend()
	plt.grid(True, alpha=0.3)
	# Wykres 2: zoom na 2016-2017
	plt.subplot(2, 1, 2)
	zoom_start = pd.Timestamp('2016-10-01')
	zoom_end = pd.Timestamp('2017-12-31')
	zoom_mask = (oil_processed['date'] >= zoom_start) & (oil_processed['date'] <= zoom_end)
	plt.plot(oil_processed.loc[zoom_mask, 'date'], oil_processed.loc[zoom_mask, value_col], 
			 'o', alpha=0.7, label='Original Data', markersize=6, color='blue')
	plt.plot(oil_processed.loc[zoom_mask, 'date'], oil_processed.loc[zoom_mask, filled_col], 
			 '-', alpha=0.8, label='PCHIP Interpolation', linewidth=2, color='green')
	zoom_filled = zoom_mask & filled_mask
	if zoom_filled.any():
		plt.scatter(oil_processed.loc[zoom_filled, 'date'], oil_processed.loc[zoom_filled, filled_col], 
				   color='red', s=40, alpha=0.9, label='Filled Points', zorder=5)
	plt.title('Detail View: 2016-2017 Period', fontsize=12, fontweight='bold')
	plt.xlabel('Date')
	plt.ylabel('Oil Price ($)')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()
	



def fill_time_series_full_range(oil_df, value_col='dcoilwtico', date_col='date', filled_col='dcoilwtico_filled', complete_col='dcoilwtico_complete'):
	"""
	Uzupełnia wszystkie brakujące daty w szeregu czasowym (pełny zakres dat) i interpoluje wartości (PCHIP/linear fallback).
	Zwraca DataFrame z kolumnami: oryginał, filled, complete (pełny szereg).
	"""
	df = oil_df.copy()
	df[date_col] = pd.to_datetime(df[date_col])
	start = df[date_col].min()
	end = df[date_col].max()
	complete_dates = pd.date_range(start=start, end=end, freq='D')
	df_full = pd.DataFrame({date_col: complete_dates})
	df_full = df_full.merge(df, on=date_col, how='left')
	df_full['date_numeric'] = (df_full[date_col] - df_full[date_col].min()).dt.days
	non_null_mask = df_full[filled_col].notnull()
	x_known = df_full.loc[non_null_mask, 'date_numeric'].values
	y_known = df_full.loc[non_null_mask, filled_col].values
	x_all = df_full['date_numeric'].values
	print(f"   PCHIP input: {len(x_known)} known prices → {len(x_all)} complete dates")
	if len(x_known) >= 4:
		pchip_complete = interpolate.PchipInterpolator(x_known, y_known)
		df_full[complete_col] = pchip_complete(x_all)
	else:
		df_full[complete_col] = df_full[filled_col].interpolate(method='linear')
		df_full[complete_col] = df_full[complete_col].fillna(method='bfill')
		df_full[complete_col] = df_full[complete_col].fillna(method='ffill')
		print("   Used linear fallback (insufficient points for PCHIP)")
	df_full = df_full.drop('date_numeric', axis=1)
	return df_full

def check_missing_dates_full_range(oil_df, train_df, oil_date_col='date', train_date_col='date'):
	"""
	Zwraca DataFrame z pełnym zakresem dat oraz kolumnami: missing_in_oil, missing_in_train.
	"""
	oil_dates_raw = pd.to_datetime(oil_df[oil_date_col])
	train_dates_raw = pd.to_datetime(train_df[train_date_col])
	all_dates = pd.date_range(start=min(oil_dates_raw.min(), train_dates_raw.min()),
							 end=max(oil_dates_raw.max(), train_dates_raw.max()), freq='D')
	oil_dates_set = set(oil_dates_raw)
	train_dates_set = set(train_dates_raw)
	missing_in_oil = [d for d in all_dates if d not in oil_dates_set]
	missing_in_train = [d for d in all_dates if d not in train_dates_set]
	df_missing = pd.DataFrame({'date': all_dates})
	df_missing['missing_in_oil'] = df_missing['date'].isin(missing_in_oil)
	df_missing['missing_in_train'] = df_missing['date'].isin(missing_in_train)
	return df_missing


def process_promotions_flexible(df, dataset_name, strategy="nan"):
	"""
	Przetwarza kolumnę 'onpromotion' zgodnie z wybraną strategią:
	- 'nan': zostawia NaN jako osobną kategorię (idealne dla XGBoost/LightGBM)
	- 'new_category': zamienia NaN na nową kategorię tekstową
	Zwraca DataFrame z kolumną 'onpromotion_processed'.
	"""
	if 'onpromotion' not in df.columns:
		print(f"{dataset_name}: No onpromotion column found")
		return df
	df_processed = df.copy()
	original_nan = df_processed['onpromotion'].isnull().sum()
	print(f"\n🔧 {dataset_name} - Strategy: '{strategy}'")
	print(f"   Original NaN values: {original_nan:,}")
	if strategy == "nan":
		df_processed['onpromotion_processed'] = df_processed['onpromotion']
		remaining_nan = df_processed['onpromotion_processed'].isnull().sum()
		print(f"   Kept NaN values as separate category")
		print(f"   Final NaN count: {remaining_nan:,}")
		print(f"   Perfect for: XGBoost, LightGBM, CatBoost")
	elif strategy == "new_category":
		NEW_CATEGORY = "NO_PROMO_INFO"
		df_processed['onpromotion_processed'] = df_processed['onpromotion'].astype('object')
		df_processed['onpromotion_processed'] = df_processed['onpromotion_processed'].fillna(NEW_CATEGORY)
		remaining_nan = df_processed['onpromotion_processed'].isnull().sum()
		category_count = (df_processed['onpromotion_processed'] == NEW_CATEGORY).sum()
		print(f"   Replaced NaN with '{NEW_CATEGORY}' category")
		print(f"   Final NaN count: {remaining_nan:,}")
		print(f"   New category count: {category_count:,}")
		print(f"   Perfect for: All ML models, categorical encoding")
	else:
		raise ValueError(f"Unknown strategy: {strategy}. Use 'nan' or 'new_category'")
	return df_processed