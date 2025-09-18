
import holidays_preproc as hp
import Lags as lg
from extend_train import create_extend_train, load_favorita_tables
from handling_nan_values import fill_time_series_full_range, process_promotions_flexible
from date_features import extract_comprehensive_date_features
from promo_features import promo_features_all

def run_modeling_pipeline(cfg):

        # 1) Load raw tables
    dfs = load_favorita_tables(['train.csv', 'oil.csv', 'stores.csv', 'items.csv', 'holidays_events.csv'])
    train = dfs['train']
    oil = dfs['oil']
    stores = dfs['stores']
    items = dfs['items']
    holidays = dfs['holidays_events']


    # 2) Preprocess (your functions)
    ext_train = create_extend_train(train, items, stores)    #KUBA   # City/State/FAMILY/Class #TODO
    oil = fill_time_series_full_range(oil)
    promos = process_promotions_flexible(promos, dataset_name="promos", strategy_for_missing_promo="nan")
    holidays = hp.preprocess(holidays)

    # 3) Join auxiliary tables into train
    train = hp.merge_train(train, holidays, stores)
    train = JOIN_OIL_PROMOS(ext_train, oil, promos) # Kuba - chyba tego nie robiłem

    # 4) Build features for a given dataset slice
    def MAKE_FEATURES(df, target_col, lags=[7, 14, 28], rollag=[1], explag=[1], group_cols=None, windows=[7, 14, 28],
                      rolling_stats=['mean', 'std', 'max', 'min', 'sum'],
                      exp_stats=['mean', 'std', 'max', 'min', 'sum']):
        df = extract_comprehensive_date_features(df) # Kuba
        # df = hp.merge_train(df, holidays, stores)
        df = promo_features_all(df) # Kuba
        df = lg.make_lag(df, lag = lags, group_cols=group_cols, core_column=target_col) # max_lag = lag.max())   # uses shift(+) # Tomek
        df = lg.make_rolling(df, rollag=rollag, window = windows, core_column=target_col, group_cols=group_cols, rolling_stats=rolling_stats)  # shift(1) then rolling # Tomek
        df = lg.make_expanding(df, core_column=target_col, explag=explag, exp_stats=exp_stats, group_cols=group_cols)  # closed='left' # Tomek

        return df


    def compute_frozen_days_from_max_lag(max_lag, max_window):
        return int(max_lag + max_window - 1)


    def unique_dates(df):
        return df['date'].unique()


    def build_rolling_windows(full_dates, n_windows, val_size, stride, embargo, min_tain_days):
        return


    # 5) Time-series cross-validation windows
    max_lag = 365
    validation_windows = 3
    size_of_validation_windows = 14
    frozen_days = compute_frozen_days_from_max_lag(max_lag)          # often == max_lag
    windows = build_rolling_windows(    # TODO
                 full_dates = unique_dates(train.date),
                 n_windows = validation_windows,
                 val_size = size_of_validation_windows,
                 stride = cfg.cv.stride_days,
                 embargo = frozen_days,
                 min_train_days = cfg.cv.min_train_days)

    fold_scores = []
    models_per_fold = []

    # 6) Cross-validation loop
    for (train_range, val_range) in windows:
        tr = SLICE_BY_DATE(train, train_range) # TODO
        va = SLICE_BY_DATE(train, val_range)

        Xtr = MAKE_FEATURES(tr, max_lag)
        Xva = MAKE_FEATURES(va, max_lag)

        y_tr = Xtr[cfg.cols.target];  Xtr = DROP_TARGET_COLS(Xtr, cfg.cols.target) # TODO
        y_va = Xva[cfg.cols.target];  Xva = DROP_TARGET_COLS(Xva, cfg.cols.target)

        model = TRAIN_MODEL(Xtr, y_tr, Xva, y_va, cfg.model)         # LightGBM / XGBoost + early stop
        preds = PREDICT(model, Xva, cfg.model)

        fold_scores.append( EVALUATE_METRICS(y_va, preds, metrics = ["RMSE","WMAPE"]) )
        models_per_fold.append(model)

    cv_summary = AGGREGATE_FOLD_SCORES(fold_scores)

    # HYPERTUNING

    # 7) (Optional) Refit final model on all history up to the end of training
    #    (optionally up to scoring_period.t_min - 1)
    full_hist = SLICE_BY_DATE(train, [MIN_DATE(train), MAX_DATE(train)])
    X_full = MAKE_FEATURES(full_hist, max_lag)
    y_full = X_full[cfg.cols.target];  X_full = DROP_TARGET_COLS(X_full, cfg.cols.target)

    final_model = TRAIN_MODEL_NO_VALIDATION(X_full, y_full, cfg.model)
