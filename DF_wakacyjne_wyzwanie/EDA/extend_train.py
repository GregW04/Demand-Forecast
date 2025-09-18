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

# --- Funkcja do ładowania wszystkich zbiorów Favorita (bez chunków, pełne wczytanie) ---
import pandas as pd

def load_favorita_tables(paths):
    """
    Ładuje wszystkie podstawowe zbiory Favorita do DataFrame'ów.

    Parameters
    ----------
    paths : dict lub Namespace
        Słownik lub obiekt z atrybutami zawierającymi ścieżki do plików csv.

    Returns
    -------
    tuple
        (train, oil, stores, items, holidays, promos, test, sample_submission, transactions)
        Każdy element to pd.DataFrame lub None jeśli plik nie istnieje.
    """
    def try_read(path, **kwargs):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            print(f"Nie udało się wczytać {path}: {e}")
            return None

    train = try_read(paths.get('train', 'train.csv'), parse_dates=['date'])
    oil = try_read(paths.get('oil', 'oil.csv'))
    stores = try_read(paths.get('stores', 'stores.csv'))
    items = try_read(paths.get('items', 'items.csv'))
    holidays = try_read(paths.get('holidays', 'holidays_events.csv'))
    promos = try_read(paths.get('promos', 'promos.csv'))
    test = try_read(paths.get('test', 'test.csv'))
    sample_submission = try_read(paths.get('sample_submission', 'sample_submission.csv'))
    transactions = try_read(paths.get('transactions', 'transactions.csv'))

    return train, oil, stores, items, holidays, promos, test, sample_submission, transactions
