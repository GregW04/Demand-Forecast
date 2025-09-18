import os
import pandas as pd

def create_extend_train(train, items, stores):
	"""
	Dołącza do DataFrame train cechy produktowe (family, class) oraz sklepowe (city, state).

	Parameters
	----------
	train : pd.DataFrame
		Dane sprzedażowe, zawierające co najmniej kolumny 'item_nbr' i 'store_nbr'.
	items : pd.DataFrame
		Dane o produktach, zawierające kolumny 'item_nbr', 'family', 'class'.
	stores : pd.DataFrame
		Dane o sklepach, zawierające kolumny 'store_nbr', 'city', 'state'.

	Returns
	-------
	pd.DataFrame
		Rozszerzony DataFrame train z cechami produktowymi i sklepowymi.
	"""
	# Wybieramy tylko potrzebne kolumny produktowe i usuwamy duplikaty po item_nbr
	items_cols = ['item_nbr', 'family', 'class']
	items_sel = items[items_cols].drop_duplicates('item_nbr')

	# Wybieramy tylko potrzebne kolumny sklepowe i usuwamy duplikaty po store_nbr
	stores_cols = ['store_nbr', 'city', 'state']
	stores_sel = stores[stores_cols].drop_duplicates('store_nbr')

	# Dołączamy cechy produktowe do train po item_nbr (lewy join)
	train_ext = train.merge(items_sel, on='item_nbr', how='left')

	# Dołączamy cechy sklepowe do train po store_nbr (lewy join)
	train_ext = train_ext.merge(stores_sel, on='store_nbr', how='left')

	# Zwracamy rozszerzony DataFrame
	return train_ext

# --- Funkcja do ładowania wszystkich zbiorów Favorita (pełne wczytanie) ---

def load_favorita_tables(csv_paths):
    """
    Uniwersalna funkcja do ładowania dowolnych plików CSV do słownika DataFrame'ów.
    Jeśli klucz/ścieżka zawiera 'train', ustawia optymalne dtypes.

    Parametry
    ---------
    csv_paths : dict lub list lub str
        - dict: {nazwa: sciezka}
        - list: lista ścieżek (nazwa = plik bez rozszerzenia)
        - str: folder lub pojedynczy plik csv

    Returns
    -------
    dict
        Słownik {nazwa: DataFrame}
    """

    train_dtypes = {
        'store_nbr': 'int8',
        'item_nbr': 'int32',
        'unit_sales': 'float32',
        'onpromotion': 'boolean'
    }

    def try_read(path, **kwargs):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            print(f"Nie udało się wczytać {path}: {e}")
            return None

    result = {}

    if isinstance(csv_paths, dict):
        items = csv_paths.items()
    elif isinstance(csv_paths, list):
        items = [(os.path.splitext(os.path.basename(p))[0], p) for p in csv_paths]
    elif isinstance(csv_paths, str):
        if os.path.isdir(csv_paths):
            files = [f for f in os.listdir(csv_paths) if f.endswith('.csv')]
            items = [(os.path.splitext(f)[0], os.path.join(csv_paths, f)) for f in files]
        elif csv_paths.endswith('.csv'):
            items = [(os.path.splitext(os.path.basename(csv_paths))[0], csv_paths)]
        else:
            raise ValueError('Podaj folder lub plik csv lub listę/ dict ścieżek')
    else:
        raise ValueError('csv_paths musi być dict, listą lub stringiem')

    for name, path in items:
        if 'train' in name:
            df = try_read(path, dtype=train_dtypes, parse_dates=['date'])
        else:
            df = try_read(path)
        result[name] = df

    return result


# dfs = load_favorita_tables(['train.csv', 'oil.csv', 'stores.csv'])
# # lub
# dfs = load_favorita_tables({'train': 'train.csv', 'oil': 'oil.csv'})
# # lub
# dfs = load_favorita_tables('C:/ścieżka/do/folderu_z_csv')
