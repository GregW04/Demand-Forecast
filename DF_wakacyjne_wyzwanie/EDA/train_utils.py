# train_utils.py
import numpy as np
import pandas as pd

# [NOTE] Optional dependencies are imported lazily inside functions to avoid import errors if not installed.

def _is_pandas_nullable_int(dtype) -> bool:
    """Return True if dtype is a pandas nullable integer (e.g., Int8/Int16/Int32/Int64)."""
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

    # Drop datetimes and unsupported columns
    dt_cols = [c for c in Xp.columns if pd.api.types.is_datetime64_any_dtype(Xp[c])]
    if dt_cols:
        Xp.drop(columns=dt_cols, inplace=True)

    # object -> category
    obj_cols = [c for c in Xp.columns if Xp[c].dtype == "object"]
    for c in obj_cols:
        Xp[c] = Xp[c].astype("category")

    # pandas BooleanDtype -> float32 (True=1.0, False=0.0, <NA>=NaN)
    bool_na_cols = [c for c in Xp.columns if str(Xp[c].dtype) == "boolean"]
    for c in bool_na_cols:
        Xp[c] = Xp[c].astype("float32")

    # numpy bool -> uint8 (0/1)
    bool_cols = [c for c in Xp.columns if Xp[c].dtype == bool]
    for c in bool_cols:
        Xp[c] = Xp[c].astype("uint8")

    # pandas nullable integers -> float32 (preserve NaN)
    pand_int_cols = [c for c in Xp.columns if _is_pandas_nullable_int(Xp[c].dtype)]
    for c in pand_int_cols:
        Xp[c] = Xp[c].astype("float32")

    # unsigned nullable ints (e.g., UInt8) -> float32 (preserve NaN if any)
    if hasattr(pd, "UInt8Dtype"):
        ucols = [c for c in Xp.columns if isinstance(Xp[c].dtype, (pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, getattr(pd, "UInt64Dtype", type("D", (), {}))))]  # type: ignore
        for c in ucols:
            Xp[c] = Xp[c].astype("float32")

    # numeric downcast where safe (optional)
    # Xp = Xp.apply(lambda s: pd.to_numeric(s, errors="ignore", downcast="float") if pd.api.types.is_float_dtype(s) else s)

    # categorical features: dtype == category
    cat_cols = [c for c in Xp.columns if pd.api.types.is_categorical_dtype(Xp[c])]

    # Framework-specific notes:
    # - LightGBM: accepts category dtype directly via categorical_feature list
    # - XGBoost: requires enable_categorical=True and pandas category dtype is OK

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

    # Prepare features per framework
    Xtr_prep, cat_cols = _prepare_features_for_framework(Xtr, framework)
    Xva_prep, _ = _prepare_features_for_framework(Xva, framework)

    if framework == "lightgbm":
        import lightgbm as lgb
        # Sensible defaults if not provided
        params.setdefault("objective", "regression")
        params.setdefault("metric", "rmse")
        params.setdefault("n_estimators", 2000)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("verbosity", -1)

        early_stopping_rounds = params.pop("early_stopping_rounds", 200)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            Xtr_prep, y_tr,
            eval_set=[(Xva_prep, y_va)],
            eval_metric=params.get("metric", "rmse"),
            categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)]
        )
        return model

    elif framework == "xgboost":
        import xgboost as xgb
        # Defaults for speed/stability
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("tree_method", "hist")
        params.setdefault("learning_rate", 0.05)
        params.setdefault("n_estimators", 2000)
        params.setdefault("enable_categorical", True)
        early_stopping_rounds = params.pop("early_stopping_rounds", 200)
        eval_metric = params.pop("eval_metric", "rmse")

        model = xgb.XGBRegressor(**params)
        model.fit(
            Xtr_prep, y_tr,
            eval_set=[(Xva_prep, y_va)],
            eval_metric=eval_metric,
            verbose=True,
            early_stopping_rounds=early_stopping_rounds
        )
        return model

    else:
        raise ValueError(f"Unsupported framework: {framework}")


def PREDICT(model: object, X: pd.DataFrame, model_cfg) -> np.ndarray:
    """
    Predict using the trained model, preparing features consistently.
    """
    framework = str(getattr(model_cfg, "framework", "lightgbm")).lower()
    Xp, _ = _prepare_features_for_framework(X, framework)
    preds = model.predict(Xp)
    # LightGBM returns shape (n,) float; XGBoost same
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


def TRAIN_MODEL_NO_VALIDATION(X: pd.DataFrame, y: pd.Series, model_cfg) -> object:
    """
    Train a final model on all data (no validation set).
    """
    framework = str(getattr(model_cfg, "framework", "lightgbm")).lower()
    params = dict(getattr(model_cfg, "params", {}))

    Xp, cat_cols = _prepare_features_for_framework(X, framework)

    if framework == "lightgbm":
        import lightgbm as lgb
        params.setdefault("objective", "regression")
        params.setdefault("metric", "rmse")
        params.setdefault("n_estimators", 2000)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("verbosity", -1)

        model = lgb.LGBMRegressor(**params)
        model.fit(Xp, y, categorical_feature=cat_cols if len(cat_cols) > 0 else "auto")
        return model

    elif framework == "xgboost":
        import xgboost as xgb
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("tree_method", "hist")
        params.setdefault("learning_rate", 0.05)
        params.setdefault("n_estimators", 2000)
        params.setdefault("enable_categorical", True)

        model = xgb.XGBRegressor(**params)
        model.fit(Xp, y, verbose=True)
        return model

    else:
        raise ValueError(f"Unsupported framework: {framework}")
