import numpy as np
import pandas as pd

import holidays_preproc as hp
import Lags as lg
from extend_train import create_extend_train, load_favorita_tables
from handling_nan_values import fill_time_series_full_range, process_promotions_flexible, fill_oil_pchip
from JOIN_OIL_PROMOS import JOIN_OIL_PROMOS
from features_block import *
from feature_checks import check_features_quick
from train_utils import *
from hypertune import HYPERTUNE_EMBEDDED_FS

from types import SimpleNamespace
import time
from pathlib import Path
import joblib

# Detect available Parquet engine at import time
try:
    import pyarrow  # noqa: F401
    PARQUET_ENGINE = "pyarrow"
except Exception:
    try:
        import fastparquet  # noqa: F401
        PARQUET_ENGINE = "fastparquet"
    except Exception:
        PARQUET_ENGINE = None


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
    # Tune window parameters so at least one CV window can be built
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
    for fold_idx, (train_range, val_range) in enumerate(windows):
        tr = SLICE_BY_DATE(train, train_range)
        va = SLICE_BY_DATE(train, val_range)

        # Sort once (stable) by group_cols + date to avoid re-sorting inside feature functions
        t0 = time.perf_counter()
        tr = tr.sort_values(cfg.cols.group_cols + ['date'], kind='mergesort')
        va = va.sort_values(cfg.cols.group_cols + ['date'], kind='mergesort')
        print(f"[FEATS] Fold {fold_idx}: sorted tr={len(tr):,}, va={len(va):,} in {time.perf_counter()-t0:.2f}s")

        # Xtr = MAKE_FEATURES(tr, target_col=cfg.cols.target, group_cols=cfg.cols.group_cols,
        #                     pre_sorted=True, timers=True)
        # Xva = MAKE_FEATURES(va, target_col=cfg.cols.target, group_cols=cfg.cols.group_cols,
        #                     pre_sorted=True, timers=True)
        # Limit rolling/expanding for first comparison; pass group_cols from cfg; keep timers
        print('Make features for Xtr')
        Xtr = MAKE_FEATURES(
            tr, target_col=cfg.cols.target, group_cols=cfg.cols.group_cols,
            lags=[7, 14],                 # keep only light lags
            windows=[7],                  # rolling 7 only
            rolling_stats=["mean"],       # mean only
            exp_stats=[],                 # disable expanding
            pre_sorted=True, timers=True
        )
        print('Make features for Xva')
        Xva = MAKE_FEATURES(
            va, target_col=cfg.cols.target, group_cols=cfg.cols.group_cols,
            lags=[7, 14],
            windows=[7],
            rolling_stats=["mean"],
            exp_stats=[],
            pre_sorted=True, timers=True
        )

        if fold_idx == 0:
            print("[INFO] Feature sanity check on fold 0 (train)")
            check_features_quick(Xtr, target=cfg.cols.target)

        y_tr = Xtr[cfg.cols.target];  Xtr = DROP_TARGET_COLS(Xtr, cfg.cols.target)
        y_va = Xva[cfg.cols.target];  Xva = DROP_TARGET_COLS(Xva, cfg.cols.target)

        # Save fold 0 features/targets to Parquet if engine is available, else fallback to pickle
        if fold_idx == 0 and getattr(cfg, "cache", None) is not None and getattr(cfg.cache, "save_fold0", True):
            cache_dir = Path(cfg.cache.dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            if PARQUET_ENGINE:
                # Use Parquet with the detected engine (optionally add compression="snappy"/"zstd")
                Xtr.to_parquet(cache_dir / "Xtr_fold0.parquet", index=False, engine=PARQUET_ENGINE)
                Xva.to_parquet(cache_dir / "Xva_fold0.parquet", index=False, engine=PARQUET_ENGINE)
                y_tr.to_frame(name=cfg.cols.target).to_parquet(cache_dir / "y_tr_fold0.parquet", engine=PARQUET_ENGINE)
                y_va.to_frame(name=cfg.cols.target).to_parquet(cache_dir / "y_va_fold0.parquet", engine=PARQUET_ENGINE)
                print(f"[CACHE] Saved fold 0 to {cache_dir} (engine={PARQUET_ENGINE})")
            else:
                # Fallback: pickle (larger files, but no extra deps)
                Xtr.to_pickle(cache_dir / "Xtr_fold0.pkl")
                Xva.to_pickle(cache_dir / "Xva_fold0.pkl")
                y_tr.to_frame(name=cfg.cols.target).to_pickle(cache_dir / "y_tr_fold0.pkl")
                y_va.to_frame(name=cfg.cols.target).to_pickle(cache_dir / "y_va_fold0.pkl")
                print(f"[CACHE] Parquet engine not available. Saved pickle files to {cache_dir}")

        model = TRAIN_MODEL(Xtr, y_tr, Xva, y_va, cfg.model)         # LightGBM / XGBoost + early stop
        preds = PREDICT(model, Xva, cfg.model)

        fold_scores.append( EVALUATE_METRICS(y_va, preds, metrics = ["RMSE","WMAPE"]) )
        models_per_fold.append(model)

        models_dir = Path("artifacts/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{cfg.model.framework}_fold{fold_idx}.pkl"
        joblib.dump(model, model_path)
        print(f"[MODEL] Saved {cfg.model.framework} fold {fold_idx} to {model_path}")

    cv_summary = AGGREGATE_FOLD_SCORES(fold_scores)
    print("[CV] fold_scores:", fold_scores)
    print("[CV] summary:", cv_summary)

    # HYPERTUNING

    # 7) (Optional) Refit final model on all history up to the end of training
    #    (optionally up to scoring_period.t_min - 1)
    print('7) (Optional) Refit final model on all history up to the end of training')

    print(f"[HPO] enabled? hpo in cfg: {hasattr(cfg,'hpo')}, enabled: {getattr(getattr(cfg,'hpo', None),'enabled', None)}")
    # Hyperparameter tuning (optional, separate step; does not affect what already works)
    if getattr(cfg, "hpo", None) and getattr(cfg.hpo, "enabled", False):
        print("[HPO] Starting hyperparameter search...")

        # Thin wrapper around MAKE_FEATURES with the same settings you used in CV
        def MAKE_FEATURES_WRAPPER(df, max_lag):
            # sort once to speed promo/rolling (same as in CV)
            df = df.sort_values(cfg.cols.group_cols + ['date'], kind='mergesort')
            return MAKE_FEATURES(
                df,
                target_col=cfg.cols.target,
                group_cols=cfg.cols.group_cols,
                lags=[7, 14],
                windows=[7],
                rolling_stats=["mean"],
                exp_stats=[],      # expanding disabled for initial comparison
                pre_sorted=True,
                timers=False
            )

        # Use current framework for HPO: "lgbm" if cfg.model.framework is LightGBM, otherwise "xgb"
        model_for_hpo = "xgb" if str(cfg.model.framework).lower() == "xgboost" else "lgbm"

        # Base params for the chosen framework (copied; HPO only merges FS-related knobs)
        base_params = dict(cfg.model.params)

        # IMPORTANT (only if framework == xgboost>=3.x):
        # make sure eval_metric and early_stopping_rounds are in base params
        if model_for_hpo == "xgb":
            base_params.setdefault("eval_metric", "rmse")
            base_params.setdefault("early_stopping_rounds", 50)
            base_params.setdefault("tree_method", "hist")
            base_params.setdefault("enable_categorical", True)

        # Run HPO on a limited number of windows/trials first (fast), can increase later
        hpo_result = HYPERTUNE_EMBEDDED_FS(
            train_df=train,
            windows=windows,
            make_features=MAKE_FEATURES_WRAPPER,
            base_model_params=base_params,
            max_lag=max_lag,                    # keep consistent with frozen_days logic
            target_col=cfg.cols.target,
            model_name=model_for_hpo,           # "lgbm" or "xgb" (derived from cfg.model.framework)
            metric="RMSE",
            n_trials=cfg.hpo.n_trials,
            pruner=cfg.hpo.pruner,
            cache_dir=cfg.hpo.cache_dir,        # cache fold features for reuse across trials
            windows_limit=cfg.hpo.windows_limit,# start with 1 window for speed
            SLICE_BY_DATE=SLICE_BY_DATE
        )

        print("[HPO] best_model_params:", hpo_result["best_model_params"])
        print("[HPO] best_tuned_only:  ", hpo_result["best_tuned_only"])
        print("[HPO] cv_mean:          ", hpo_result["cv_mean"])
        print("[HPO] cv_by_fold:       ", hpo_result["cv_by_fold"])



cfg = SimpleNamespace(
    paths=['train.csv', 'oil.csv', 'stores.csv', 'items.csv', 'holidays_events.csv'],
    cv=SimpleNamespace(
        stride_days=14,      # validation window shift step (days)
        min_train_days=365   # was 730  # shorter minimal training length
    ),
    cols=SimpleNamespace(
        target="unit_sales",
        group_cols=["store_nbr", "item_nbr"]  # group keys for lag/rolling/expanding
    ),
    cache=SimpleNamespace(
        dir="cache",          # directory to save parquet files
        save_fold0=True       # save features/targets for fold 0
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
    ),
    hpo = SimpleNamespace(
        enabled=True,        # set True to run HPO
        n_trials=3,           # start small; increase later (e.g., 50–80)
        windows_limit=2,      # use 2 CV window first for speed; set None to use all
        pruner="median",      # "median" or "none"
        cache_dir="cache/hpo" # where per-fold features are cached during HPO
    )
)


run_modeling_pipeline(cfg)
