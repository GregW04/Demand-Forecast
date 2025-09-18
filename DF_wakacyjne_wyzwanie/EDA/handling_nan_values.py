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

	Parameters
	----------
	oil_df : pd.DataFrame
		DataFrame z danymi wejściowymi, zawierający kolumnę z datą i wartościami do interpolacji.
	value_col : str, default 'dcoilwtico'
		Nazwa kolumny z wartościami do interpolacji.
	date_col : str, default 'date'
		Nazwa kolumny z datą.

	Returns
	-------
	pd.DataFrame
		Nowy DataFrame z dodatkową kolumną '<value_col>_filled' (uzupełnione wartości).
	"""
	# Tworzymy kopię danych wejściowych, aby nie nadpisać oryginału
	# Konwertujemy kolumnę daty do typu datetime
	# Przekształcamy daty na liczbę dni od początku (potrzebne do interpolacji)
	# Wybieramy tylko te wiersze, gdzie mamy znane wartości (nie NaN)
	# Przygotowujemy dane do interpolacji: x_known (dni), y_known (wartości)
	# x_all - wszystkie dni w szeregu czasowym
	oil_processed = oil_df.copy()
	oil_processed[date_col] = pd.to_datetime(oil_processed[date_col])
	oil_processed['date_numeric'] = (oil_processed[date_col] - oil_processed[date_col].min()).dt.days
	non_null_mask = oil_processed[value_col].notnull()
	x_known = oil_processed.loc[non_null_mask, 'date_numeric'].values
	y_known = oil_processed.loc[non_null_mask, value_col].values
	x_all = oil_processed['date_numeric'].values
	print(f"Input data: {len(x_known)} known prices, {len(x_all)} total dates")
	if len(x_known) >= 4:
		# Jeśli mamy wystarczająco dużo punktów, stosujemy interpolację PCHIP
		pchip_func = interpolate.PchipInterpolator(x_known, y_known)
		oil_processed[f'{value_col}_filled'] = pchip_func(x_all)
		print("PCHIP interpolation applied successfully")
	else:
		# Jeśli za mało punktów, stosujemy interpolację liniową i uzupełniamy braki
		oil_processed[f'{value_col}_filled'] = oil_processed[value_col].interpolate(method='linear')
		oil_processed[f'{value_col}_filled'] = oil_processed[f'{value_col}_filled'].fillna(method='bfill')
		print(" Insufficient points for PCHIP, used linear fallback")
	return oil_processed

def plot_oil_interpolation(oil_processed, value_col='dcoilwtico', filled_col='dcoilwtico_filled'):
	"""
	Rysuje wykres interpolacji cen ropy: oryginalne dane, wartości uzupełnione (PCHIP/linear), podgląd okresu 2016-2017.

	Parameters
	----------
	oil_processed : pd.DataFrame
		DataFrame z oryginalnymi i interpolowanymi wartościami cen ropy.
	value_col : str, default 'dcoilwtico'
		Nazwa kolumny z oryginalnymi wartościami.
	filled_col : str, default 'dcoilwtico_filled'
		Nazwa kolumny z wartościami po interpolacji.

	Returns
	-------
	None
		Funkcja wyświetla wykresy, nie zwraca wartości.
	"""
	# Zaznaczamy, które wartości były pierwotnie brakujące
	filled_mask = oil_processed[value_col].isnull()
	plt.figure(figsize=(14, 8))
	# Wykres 1: pełny zakres
	plt.subplot(2, 1, 1)
	# Rysujemy oryginalne dane i wartości po interpolacji dla całego okresu
	plt.plot(oil_processed['date'], oil_processed[value_col], 'o', alpha=0.7, label='Original Data', markersize=4, color='blue')
	plt.plot(oil_processed['date'], oil_processed[filled_col], '-', alpha=0.8, label='PCHIP Interpolation', linewidth=2, color='green')
	if filled_mask.any():
		# Zaznaczamy na czerwono punkty, które zostały uzupełnione
		filled_dates = oil_processed.loc[filled_mask, 'date']
		plt.scatter(filled_dates, oil_processed.loc[filled_mask, filled_col], 
				   color='red', s=30, alpha=0.9, label=f'Filled Points ({filled_mask.sum()})', zorder=5)
	plt.title('Oil Price Interpolation - PCHIP Method', fontsize=14, fontweight='bold')
	plt.ylabel('Oil Price ($)')
	plt.legend()
	plt.grid(True, alpha=0.3)
	# Wykres 2: zoom na 2016-2017
	plt.subplot(2, 1, 2)
	# Przybliżenie na wybrany okres (2016-2017)
	zoom_start = pd.Timestamp('2016-10-01')
	zoom_end = pd.Timestamp('2017-12-31')
	zoom_mask = (oil_processed['date'] >= zoom_start) & (oil_processed['date'] <= zoom_end)
	plt.plot(oil_processed.loc[zoom_mask, 'date'], oil_processed.loc[zoom_mask, value_col], 
			 'o', alpha=0.7, label='Original Data', markersize=6, color='blue')
	plt.plot(oil_processed.loc[zoom_mask, 'date'], oil_processed.loc[zoom_mask, filled_col], 
			 '-', alpha=0.8, label='PCHIP Interpolation', linewidth=2, color='green')
	zoom_filled = zoom_mask & filled_mask
	if zoom_filled.any():
		# Zaznaczamy na czerwono punkty uzupełnione w tym okresie
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

	Parameters
	----------
	oil_df : pd.DataFrame
		DataFrame z danymi wejściowymi, zawierający kolumnę z datą i wartościami.
	value_col : str, default 'dcoilwtico'
		Nazwa kolumny z oryginalnymi wartościami.
	date_col : str, default 'date'
		Nazwa kolumny z datą.
	filled_col : str, default 'dcoilwtico_filled'
		Nazwa kolumny z wartościami po pierwszej interpolacji.
	complete_col : str, default 'dcoilwtico_complete'
		Nazwa kolumny z wartościami po interpolacji na pełnym zakresie dat.

	Returns
	-------
	pd.DataFrame
		DataFrame z pełnym zakresem dat i kolumnami: oryginał, filled, complete.
	"""
	# Tworzymy kopię danych wejściowych
	df = oil_df.copy()
	# Konwertujemy kolumnę daty do typu datetime
	df[date_col] = pd.to_datetime(df[date_col])
	# Wyznaczamy pełny zakres dat (od min do max)
	start = df[date_col].min()
	end = df[date_col].max()
	complete_dates = pd.date_range(start=start, end=end, freq='D')
	# Tworzymy DataFrame z pełnym zakresem dat
	df_full = pd.DataFrame({date_col: complete_dates})
	# Dołączamy oryginalne dane do pełnego zakresu dat
	df_full = df_full.merge(df, on=date_col, how='left')
	# Przekształcamy daty na liczbę dni od początku (do interpolacji)
	df_full['date_numeric'] = (df_full[date_col] - df_full[date_col].min()).dt.days
	# Wybieramy tylko te wiersze, gdzie mamy znane wartości (nie NaN)
	non_null_mask = df_full[filled_col].notnull()
	x_known = df_full.loc[non_null_mask, 'date_numeric'].values
	y_known = df_full.loc[non_null_mask, filled_col].values
	x_all = df_full['date_numeric'].values
	print(f"   PCHIP input: {len(x_known)} known prices → {len(x_all)} complete dates")
	if len(x_known) >= 4:
		# Jeśli mamy wystarczająco dużo punktów, stosujemy interpolację PCHIP
		pchip_complete = interpolate.PchipInterpolator(x_known, y_known)
		df_full[complete_col] = pchip_complete(x_all)
	else:
		# Jeśli za mało punktów, stosujemy interpolację liniową i uzupełniamy braki
		df_full[complete_col] = df_full[filled_col].interpolate(method='linear')
		df_full[complete_col] = df_full[complete_col].fillna(method='bfill')
		df_full[complete_col] = df_full[complete_col].fillna(method='ffill')
		print("   Used linear fallback (insufficient points for PCHIP)")
	# Usuwamy pomocniczą kolumnę numeryczną
	df_full = df_full.drop('date_numeric', axis=1)
	return df_full

def check_missing_dates_full_range(oil_df, train_df, oil_date_col='date', train_date_col='date'):
	"""
	Sprawdza brakujące daty w szeregu czasowym oil, train oraz liczbę dat brakujących w oil, ale obecnych w train.

	Parameters
	----------
	oil_df : pd.DataFrame
		DataFrame z danymi oil, zawierający kolumnę z datą.
	train_df : pd.DataFrame
		DataFrame z danymi train, zawierający kolumnę z datą.
	oil_date_col : str, default 'date'
		Nazwa kolumny z datą w zbiorze oil.
	train_date_col : str, default 'date'
		Nazwa kolumny z datą w zbiorze train.

	Returns
	-------
	dict
		Słownik z DataFrame z informacją o brakach oraz statystykami liczbowymi.
	"""
	# Konwertujemy kolumny daty do typu datetime
	oil_dates_raw = pd.to_datetime(oil_df[oil_date_col])
	train_dates_raw = pd.to_datetime(train_df[train_date_col])
	# Wyznaczamy pełny zakres dat na podstawie obu zbiorów
	all_dates = pd.date_range(start=min(oil_dates_raw.min(), train_dates_raw.min()),
							 end=max(oil_dates_raw.max(), train_dates_raw.max()), freq='D')
	# Tworzymy zbiory dat występujących w oil i train
	oil_dates_set = set(oil_dates_raw)
	train_dates_set = set(train_dates_raw)
	# Szukamy dat, których brakuje w każdym zbiorze
	missing_in_oil = [d for d in all_dates if d not in oil_dates_set]
	missing_in_train = [d for d in all_dates if d not in train_dates_set]
	# Daty brakujące w oil, ale obecne w train
	missing_in_oil_not_in_train = [d for d in missing_in_oil if d in train_dates_set]
	# Tworzymy DataFrame z informacją o brakach
	df_missing = pd.DataFrame({'date': all_dates})
	df_missing['missing_in_oil'] = df_missing['date'].isin(missing_in_oil)
	df_missing['missing_in_train'] = df_missing['date'].isin(missing_in_train)

	stats = {
		'df_missing': df_missing,
		'liczba_brakujacych_w_oil': len(missing_in_oil),
		'liczba_brakujacych_w_train': len(missing_in_train),
		'liczba_brakujacych_w_oil_ale_obecnych_w_train': len(missing_in_oil_not_in_train),
		'przykladowe_brakujace_w_oil_ale_obecne_w_train': missing_in_oil_not_in_train[:10]
	}
	return stats


def process_promotions_flexible(df, dataset_name, strategy_for_missing_promo="nan"):
	"""
	Przetwarza kolumnę 'onpromotion' zgodnie z wybraną strategią.

	Parameters
	----------
	df : pd.DataFrame
		DataFrame z kolumną 'onpromotion'.
	dataset_name : str
		Nazwa zbioru danych (do logowania/raportowania).
	strategy_for_missing_promo : str, default 'nan'
		Sposób traktowania NaN: 'nan' (osobna kategoria), 'new_category' (nowa kategoria tekstowa).

	Returns
	-------
	pd.DataFrame
		DataFrame z nową kolumną 'onpromotion_processed' zgodnie z wybraną strategią.
	"""
	# Jeśli nie ma kolumny 'onpromotion', zwracamy oryginalny DataFrame
	if 'onpromotion' not in df.columns:
		print(f"{dataset_name}: No onpromotion column found")
		return df
	# Tworzymy kopię danych wejściowych
	df_processed = df.copy()
	# Liczymy ile było NaN na początku
	original_nan = df_processed['onpromotion'].isnull().sum()
	print(f"\n🔧 {dataset_name} - strategy_for_missing_promo: '{strategy_for_missing_promo}'")
	print(f"   Original NaN values: {original_nan:,}")
	if strategy_for_missing_promo == "nan":
		# Zostawiamy NaN jako osobną kategorię (np. dla XGBoost/LightGBM)
		df_processed['onpromotion_processed'] = df_processed['onpromotion']
		remaining_nan = df_processed['onpromotion_processed'].isnull().sum()
		print(f"   Kept NaN values as separate category")
		print(f"   Final NaN count: {remaining_nan:,}")
		print(f"   Perfect for: XGBoost, LightGBM, CatBoost")
	elif strategy_for_missing_promo == "new_category":
		# Zamieniamy NaN na nową kategorię tekstową (np. do kodowania kategorii)
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
		# Obsługa nieznanej strategii
		raise ValueError(f"Unknown strategy_for_missing_promo: {strategy_for_missing_promo}. Use 'nan' or 'new_category'")
	return df_processed
