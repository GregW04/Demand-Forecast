def create_extend_train(train, items, stores):
	"""
	Dołącza do train cechy produktowe (family, class) oraz sklepowe (city, state).
	Zwraca rozszerzony DataFrame.
	"""
	items_cols = ['item_nbr', 'family', 'class']
	items_sel = items[items_cols].drop_duplicates('item_nbr')
	stores_cols = ['store_nbr', 'city', 'state']
	stores_sel = stores[stores_cols].drop_duplicates('store_nbr')
	train_ext = train.merge(items_sel, on='item_nbr', how='left')
	train_ext = train_ext.merge(stores_sel, on='store_nbr', how='left')
	return train_ext