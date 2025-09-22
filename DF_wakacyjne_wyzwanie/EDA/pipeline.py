
import holidays_preproc as hp
import Lags as lg
from extend_train import create_extend_train, load_favorita_tables
from handling_nan_values import fill_time_series_full_range, process_promotions_flexible, fill_oil_pchip
from JOIN_OIL_PROMOS import JOIN_OIL_PROMOS
from features_block import (
    MAKE_FEATURES,
    compute_frozen_days_from_max_lag,
    unique_dates,
    build_rolling_windows,
)

from types import SimpleNamespace


def run_modeling_pipeline(cfg):

    # 1) Load raw tables
    print('1) Load raw tables')
    dfs = load_favorita_tables(cfg.paths)
    train = dfs['train']
    train = train[:31000000] # sample - TO BE DELETED !
    oil = dfs['oil']
    stores = dfs['stores']
    items = dfs['items']
    holidays = dfs['holidays_events']


    # 2) Preprocess (your functions)
    print('2) Preprocess')
    train = create_extend_train(train, items, stores)    #KUBA   # City/State/FAMILY/Class #TODO
    oil = fill_oil_pchip(oil)
    oil = fill_time_series_full_range(oil)
    promos = process_promotions_flexible(train, dataset_name="promos", strategy_for_missing_promo="nan")
    holidays = hp.preprocess(holidays)

    # 3) Join auxiliary tables into train
    print('3) Join auxiliary tables into train')

    print('join holidays')
    train = hp.merge_train(train, holidays, stores)

    print('join oil promos')
    train = JOIN_OIL_PROMOS(train, oil, promos)

    # 4) Build features for a given dataset slice
    print('4) Build features for a given dataset slice')
    # Moved to features_block.py

    # 5) Time-series cross-validation windows
    print('5) Time-series cross-validation windows')
    # [CHANGE] Tune window parameters so at least one CV window can be built
    max_lag = 84        # was 365  # shorter max lag → smaller embargo
    max_window = 14     # was 30   # smaller rolling window → smaller embargo
    validation_windows = 3
    size_of_validation_windows = 14
    frozen_days = compute_frozen_days_from_max_lag(max_lag, max_window)          # often == max_lag

    # [DEBUG] Check window feasibility before building windows
    full_dates = unique_dates(train)
    n_dates = len(full_dates)
    val_size = size_of_validation_windows
    embargo = frozen_days
    min_train = cfg.cv.min_train_days

    print('Old debug info:')
    print(f"[DEBUG] n_dates={n_dates}, min_train_days={min_train}, embargo={embargo}, val_size={val_size}")
    print(f"[DEBUG] required_min_dates={min_train + embargo + val_size}")
    print(f"[DEBUG] feasibility={n_dates >= (min_train + embargo + val_size)}")

    print('New debug info:')
    print(f"[DEBUG] n_dates={len(full_dates)}, min_train_days={cfg.cv.min_train_days}, "
        f"embargo={compute_frozen_days_from_max_lag(max_lag, max_window)}, val_size={size_of_validation_windows}")
    print(f"[DEBUG] required_min_dates={cfg.cv.min_train_days + compute_frozen_days_from_max_lag(max_lag, max_window) + size_of_validation_windows}")

    windows = build_rolling_windows(    # TODO
                    full_dates = unique_dates(train),
                    n_windows = validation_windows,
                    val_size = size_of_validation_windows,
                    stride = cfg.cv.stride_days,
                    embargo = frozen_days,
                    min_train_days = cfg.cv.min_train_days)

    fold_scores = []
    models_per_fold = []

    print('Built windows:')
    print(windows)
    print(f"Total windows: {len(windows)}")

    # 6) Cross-validation loop
    print('6) Cross-validation loop')
    for (train_range, val_range) in windows:
        tr = SLICE_BY_DATE(train, train_range) # TODO
        va = SLICE_BY_DATE(train, val_range)

        Xtr = MAKE_FEATURES(tr, target_col=cfg.cols.target)
        Xva = MAKE_FEATURES(va, target_col=cfg.cols.target)

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
    print('7) (Optional) Refit final model on all history up to the end of training')
    full_hist = SLICE_BY_DATE(train, [MIN_DATE(train), MAX_DATE(train)])
    X_full = MAKE_FEATURES(full_hist, max_lag)
    y_full = X_full[cfg.cols.target];  X_full = DROP_TARGET_COLS(X_full, cfg.cols.target)

    final_model = TRAIN_MODEL_NO_VALIDATION(X_full, y_full, cfg.model)



cfg = SimpleNamespace(
    paths=['train.csv', 'oil.csv', 'stores.csv', 'items.csv', 'holidays_events.csv'],
    cv=SimpleNamespace(
        stride_days=14,      # validation window shift step (days)
        min_train_days=365   # was 730  # shorter minimal training length
    ),
    cols=SimpleNamespace(
        target="unit_sales"
    ),
    model=SimpleNamespace(
        framework="lightgbm",
        params=dict(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    )
)


run_modeling_pipeline(cfg)
