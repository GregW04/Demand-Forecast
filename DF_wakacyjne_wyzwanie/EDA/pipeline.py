import holidays_preproc as hp
import Lags as lg

def run_modeling_pipeline(cfg):

    # 1) Load raw tables
    train, oil, stores, items, holidays, promos = LOAD_FAVORITA_TABLES(cfg.paths) # TODO

    # 2) Preprocess (your functions)
    ext_train = EXTEND_TRAIN(train, stores, items)       # City/State/FAMILY/Class #TODO
    oil   = FILL_OIL_MISSING(oil)       # Kuba
    promos = FILL_PROMOS_MISSING(promos) # Kuba
    holidays = hp.preprocess(holidays)

    # 3) Join auxiliary tables into train
    train = hp.merge_train(train, holidays, stores)
    train = JOIN_OIL_PROMOS(ext_train, oil, promos) # Kuba

    # 4) Build features for a given dataset slice
    def MAKE_FEATURES(df, target_col, lags=[7, 14, 28], rollag=[1], explag=[1], group_cols=None, windows=[7, 14, 28],
                      rolling_stats=['mean', 'std', 'max', 'min', 'sum'], exp_stats=['mean', 'std', 'max', 'min', 'sum']):
        df = DATE_FEATURES(df) # Kuba
        # df = hp.merge_train(df, holidays, stores)
        df = PROMO_FEATURES(df, promos) # Kuba
        df = lg.make_lag(df, lag = lags, group_cols=group_cols, core_column=target_col) # max_lag = lag.max())   # uses shift(+) # Tomek
        df = lg.make_rolling(df, rollag=rollag, window = windows, core_column=target_col, group_cols=group_cols, rolling_stats=rolling_stats)  # shift(1) then rolling # Tomek
        df = lg.make_expanding(df, core_column=target_col, explag=explag, exp_stats=exp_stats, group_cols=group_cols)  # closed='left' # Tomek
        return df

    # 5) Time-series cross-validation windows
    max_lag = 365
    validation_windows = 3
    size_of_validation_windows = 14
    frozen_days = COMPUTE_FROZEN_DAYS_FROM_MAX_LAG(max_lag)          # often == max_lag # TODO
    windows = BUILD_ROLLING_WINDOWS(    # TODO
                 full_dates = UNIQUE_DATES(train.date),  # TODO
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
