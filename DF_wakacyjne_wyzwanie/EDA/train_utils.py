# train_utils.py
import numpy as np
import pandas as pd
import time

# [NOTE] Optional dependencies are imported lazily inside functions to avoid import errors if not installed.


def _mem_str(bytes_total: int) -> str:
    """Format bytes as human-readable string (approx)."""
    gb = bytes_total / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = bytes_total / (1024 ** 2)
    return f"{mb:.1f} MB"


def MIN_DATE(df: pd.DataFrame, date_col: str = "date") -> pd.Timestamp:
    """Return the minimum date in the specified date column."""
    return pd.to_datetime(df[date_col]).min()


def MAX_DATE(df: pd.DataFrame, date_col: str = "date") -> pd.Timestamp:
    """Return the maximum date in the specified date column."""
    return pd.to_datetime(df[date_col]).max()


def _is_pandas_nullable_int(dtype) -> bool:
    return pd.api.types.is_integer_dtype(dtype) and hasattr(dtype, "name") and dtype.name.startswith("Int")


def _prepare_features_for_framework(X: pd.DataFrame, framework: str):
    """
    Prepare a copy of X for the selected framework:
    - drop datetime-like columns (e.g., 'date')
    - object -> category
    - convert pandas nullable integer/boolean columns to float32 (preserve NaN)
    - keep category dtype for categorical features
    Returns (X_prepared, categorical_feature_names)
    """
    Xp = X.copy()
    # Drop datetimes
    dt_cols = [c for c in Xp.columns if pd.api.types.is_datetime64_any_dtype(Xp[c])]
    if dt_cols:
        Xp.drop(columns=dt_cols, inplace=True)
    # object -> category
    obj_cols = [c for c in Xp.columns if Xp[c].dtype == "object"]
    for c in obj_cols:
        Xp[c] = Xp[c].astype("category")
    # pandas BooleanDtype -> float32
    bool_na_cols = [c for c in Xp.columns if str(Xp[c].dtype) == "boolean"]
    for c in bool_na_cols:
        Xp[c] = Xp[c].astype("float32")
    # numpy bool -> uint8
    bool_cols = [c for c in Xp.columns if Xp[c].dtype == bool]
    for c in bool_cols:
        Xp[c] = Xp[c].astype("uint8")
    # pandas nullable ints -> float32
    pand_int_cols = [c for c in Xp.columns if _is_pandas_nullable_int(Xp[c].dtype)]
    for c in pand_int_cols:
        Xp[c] = Xp[c].astype("float32")
    # unsigned nullable ints -> float32 (jeśli występują)
    if hasattr(pd, "UInt8Dtype"):
        utypes = (pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, getattr(pd, "UInt64Dtype", type("D", (), {})))
        ucols = [c for c in Xp.columns if isinstance(Xp[c].dtype, utypes)]  # type: ignore
        for c in ucols:
            Xp[c] = Xp[c].astype("float32")

    cat_cols = [c for c in Xp.columns if pd.api.types.is_categorical_dtype(Xp[c])]
    return Xp, cat_cols


def TRAIN_MODEL(Xtr: pd.DataFrame, y_tr: pd.Series,
                Xva: pd.DataFrame, y_va: pd.Series,
                model_cfg) -> object:
    """
    Train a model with early stopping on the chosen framework ('lightgbm' or 'xgboost').
    - Automatically prepares features and handles categorical columns.
    - Uses 'rmse' as default eval metric; you can override via cfg.model.params.

    Returns
    -------
    Trained model object (LGBMRegressor or XGBRegressor).
    """
    framework = str(getattr(model_cfg, "framework", "lightgbm")).lower()
    params = dict(getattr(model_cfg, "params", {}))

    # basic logging before prep
    print(f"[TRAIN] Framework={framework} | Xtr={Xtr.shape}, Xva={Xva.shape}")

    # Prepare features
    t0 = time.perf_counter()
    Xtr_prep, cat_cols = _prepare_features_for_framework(Xtr, framework)
    Xva_prep, _ = _prepare_features_for_framework(Xva, framework)
    prep_time = time.perf_counter() - t0

    # memory and categories info
    mem_tr = _mem_str(Xtr_prep.memory_usage(deep=True).sum())
    mem_va = _mem_str(Xva_prep.memory_usage(deep=True).sum())
    print(f"[TRAIN] Prepared in {prep_time:.2f}s | cat_cols={len(cat_cols)} | mem(Xtr)={mem_tr}, mem(Xva)={mem_va}")

    t0 = time.perf_counter()
    if framework == "lightgbm":
        import lightgbm as lgb
        params.setdefault("objective", "regression")
        params.setdefault("metric", "rmse")
        params.setdefault("n_estimators", 2000)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("verbosity", -1)
        early_stopping_rounds = params.pop("early_stopping_rounds", 200)

        model = lgb.LGBMRegressor(**params)
        # add log_evaluation for periodic progress
        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        ]
        model.fit(
            Xtr_prep, y_tr,
            eval_set=[(Xva_prep, y_va)],
            eval_metric=params.get("metric", "rmse"),
            categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
            callbacks=callbacks
        )
        train_time = time.perf_counter() - t0
        # print best iteration/score
        best_iter = getattr(model, "best_iteration_", None)
        best_score = getattr(model, "best_score_", None)
        print(f"[TRAIN] Done in {train_time:.2f}s | best_iteration={best_iter} | best_score={best_score}")
        return model

    elif framework == "xgboost":
        import xgboost as xgb

        # Defaults for speed/stability
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("tree_method", "hist")
        params.setdefault("learning_rate", 0.05)
        params.setdefault("n_estimators", 2000)
        params.setdefault("enable_categorical", True)

        # [CHANGE] In XGBoost>=3.x pass eval_metric and early_stopping_rounds in the constructor
        params.setdefault("eval_metric", "rmse")
        params.setdefault("early_stopping_rounds", 200)
        params.setdefault("verbosity", 1)  # print training progress (optional)

        model = xgb.XGBRegressor(**params)

        # [CHANGE] fit without callbacks/early_stopping_rounds kwargs (not supported in 3.x fit)
        model.fit(
            Xtr_prep, y_tr,
            eval_set=[(Xva_prep, y_va)],
            verbose=True
        )

        train_time = time.perf_counter() - t0

        # [CHANGE] robust best iteration/score retrieval for 3.x
        best_iter = getattr(model, "best_iteration", None)
        best_score = getattr(model, "best_score", None)
        # Try evals_result() if needed
        if best_score is None and hasattr(model, "evals_result"):
            try:
                er = model.evals_result()
                metric = params.get("eval_metric", "rmse")
                if "validation_0" in er and metric in er["validation_0"]:
                    hist = er["validation_0"][metric]
                    best_idx = int(np.argmin(hist))
                    best_iter = best_idx
                    best_score = hist[best_idx]
            except Exception:
                pass

        print(f"[TRAIN] Done in {train_time:.2f}s | best_iteration={best_iter} | best_score={best_score}")
        return model

    else:
        raise ValueError(f"Unsupported framework: {framework}")


def PREDICT(model: object, X: pd.DataFrame, model_cfg) -> np.ndarray:
    """
    Predict using the trained model, preparing features consistently.
    """
    framework = str(getattr(model_cfg, "framework", "lightgbm")).lower()
    # log shape before predict
    print(f"[PRED] Predicting with {framework} on X={X.shape}")
    t0 = time.perf_counter()
    Xp, _ = _prepare_features_for_framework(X, framework)
    preds = model.predict(Xp)
    dt = time.perf_counter() - t0
    print(f"[PRED] Done in {dt:.2f}s")
    return np.asarray(preds, dtype="float64")


def EVALUATE_METRICS(y_true: pd.Series, y_pred: np.ndarray, metrics=("RMSE", "WMAPE")) -> dict:
    """
    Compute selected metrics. Supported: RMSE, WMAPE.
    - RMSE: sqrt(mean squared error)
    - WMAPE: sum(|y - yhat|) / sum(|y|)
    """
    out = {}
    yt = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    for m in metrics:
        m = m.upper()
        if m == "RMSE":
            out["RMSE"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        elif m == "WMAPE":
            denom = np.sum(np.abs(yt)) + 1e-8
            out["WMAPE"] = float(np.sum(np.abs(yt - yp)) / denom)
        else:
            # Unsupported metric name is ignored
            pass
    return out


def AGGREGATE_FOLD_SCORES(fold_scores: list[dict]) -> dict:
    """
    Aggregate fold-level metric dicts into mean and std:
    returns {metric: {"mean": x, "std": y}, ...}
    """
    if not fold_scores:
        return {}
    keys = set().union(*fold_scores)
    summary = {}
    for k in keys:
        vals = np.array([fs.get(k, np.nan) for fs in fold_scores], dtype="float64")
        vals = vals[np.isfinite(vals)]
        if vals.size:
            summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return summary

