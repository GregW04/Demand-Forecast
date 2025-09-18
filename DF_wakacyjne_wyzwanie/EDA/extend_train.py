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
