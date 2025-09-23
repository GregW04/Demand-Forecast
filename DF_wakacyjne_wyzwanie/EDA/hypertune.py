# hypertune.py
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

# Training + metrics from your utilities
from train_utils import TRAIN_MODEL, PREDICT, EVALUATE_METRICS
# Slicing + feature builder are passed in from your pipeline; SLICE_BY_DATE is already available there.

# Optional dependency for HPO
try:
    import optuna
except Exception as e:
    raise ImportError("Optuna is required for hypertuning. Install via: pip install optuna") from e

# Parquet engine detection
try:
    import pyarrow  # noqa
    _PARQUET_ENGINE = "pyarrow"
except Exception:
    try:
        import fastparquet  # noqa
        _PARQUET_ENGINE = "fastparquet"
    except Exception:
        _PARQUET_ENGINE = None


def _save_df(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to Parquet if engine available, else Pickle."""
    if _PARQUET_ENGINE:
        df.to_parquet(path.with_suffix(".parquet"), index=False, engine=_PARQUET_ENGINE)
    else:
        df.to_pickle(path.with_suffix(".pkl"))


def _load_df(path: Path) -> pd.DataFrame:
    """Load DataFrame from Parquet or Pickle (fallback)."""
    p_parquet = path.with_suffix(".parquet")
    p_pkl = path.with_suffix(".pkl")
    if p_parquet.exists() and _PARQUET_ENGINE:
        return pd.read_parquet(p_parquet)
    if p_pkl.exists():
        return pd.read_pickle(p_pkl)
    raise FileNotFoundError(f"Cached file not found: {p_parquet} or {p_pkl}")


def HYPERTUNE_EMBEDDED_FS(
    train_df: pd.DataFrame,
    windows: List[Tuple[Tuple[pd.Timestamp, pd.Timestamp], Tuple[pd.Timestamp, pd.Timestamp]]],
    make_features: Callable[[pd.DataFrame, int], pd.DataFrame],
    base_model_params: Dict[str, Any],
    max_lag: int,
    *,
    target_col: str = "unit_sales",
    model_name: str = "lgbm",          # "lgbm" | "xgb"
    metric: str = "RMSE",              # "RMSE" | "WMAPE"
    n_trials: int = 40,
    pruner: str = "median",            # "median" | "none"
    cache_dir: Optional[str] = "cache/hpo",
    windows_limit: Optional[int] = None,
    SLICE_BY_DATE: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Hyperparameter tuning with embedded 'FS' (feature subsampling/regularization).
    - Suggests only FS-related parameters; merges them with base_model_params.
    - Evaluates by time-series CV windows (mean metric).
    - Caches per-fold features on disk to avoid recomputing MAKE_FEATURES for each trial.

    Parameters
    ----------
    train_df : pd.DataFrame
        Full training dataset with 'date' and target_col.
    windows : list of ((train_start, train_end), (val_start, val_end))
        CV windows from your pipeline.
    make_features : callable(df, max_lag) -> DataFrame
        Function that builds features for a given slice (past-only).
        Pass a thin wrapper around MAKE_FEATURES with your cfg (group_cols, windows, etc.).
    base_model_params : dict
        Base params for the chosen model (learning rate, n_estimators, etc.).
        For XGBoost>=3.x include 'eval_metric' and 'early_stopping_rounds' here.
    max_lag : int
        Used by feature builder if needed (consistent with your pipeline).
    target_col : str
        Name of target column.
    model_name : str
        "lgbm" or "xgb".
    metric : str
        "RMSE" or "WMAPE" (minimize).
    n_trials : int
        Number of Optuna trials.
    pruner : str
        "median" or "none".
    cache_dir : str or None
        Where to cache per-fold features (Parquet/Pickle). Set None to disable caching.
    windows_limit : int or None
        Limit number of windows used for HPO (e.g., 1 on start).
    SLICE_BY_DATE : callable or None
        Your slicing function. If None, will define a simple inclusive slicer.

    Returns
    -------
    dict
        {
          "best_model_params": dict,
          "best_tuned_only": dict,
          "cv_mean": float,
          "cv_by_fold": list of dicts,
          "study_summary": list of dicts (trial history),
        }
    """
    metric_u = metric.upper()
    assert metric_u in {"RMSE", "WMAPE"}, f"Unsupported metric: {metric}"

    used_windows = windows[:windows_limit] if windows_limit else windows

    # Fallback slicer if not provided
    if SLICE_BY_DATE is None:
        def _slice(df, date_range):
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            d = pd.to_datetime(df["date"])
            return df[(d >= start) & (d <= end)].copy()
        slicer = _slice
    else:
        slicer = SLICE_BY_DATE

    # Cache init
    cache_root = Path(cache_dir) if cache_dir else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)

    def _fold_cache_paths(i: int):
        if cache_root is None:
            return None
        base = cache_root / f"hpo_fold{i}"
        return {
            "Xtr": base.with_name(base.name + "_Xtr"),
            "Xva": base.with_name(base.name + "_Xva"),
            "ytr": base.with_name(base.name + "_y_tr"),
            "yva": base.with_name(base.name + "_y_va"),
        }

    def SUGGEST_PARAMS(trial) -> Dict[str, Any]:
        """Suggest only FS-related params."""
        p = {}
        if model_name.lower() == "lgbm":
            p["feature_fraction"] = trial.suggest_float("feature_fraction", 0.40, 1.00)
            p["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 20, 200)
            p["lambda_l1"] = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True)
            p["lambda_l2"] = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)
        else:
            p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.40, 1.00)
            p["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
            p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            p["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 20.0, log=True)
        return p

    def BUILD_MODEL_PARAMS(base_params: Dict[str, Any], tuned: Dict[str, Any]) -> Dict[str, Any]:
        """Merge tuned FS params into a fresh copy of base params."""
        merged = dict(base_params)  # shallow copy
        merged.update(tuned)
        return merged

    def _load_or_build_fold(i: int, train_rng, val_rng):
        """Load cached features for a fold or build and cache them."""
        paths = _fold_cache_paths(i)
        if paths and (paths["Xtr"].with_suffix(".parquet").exists() or paths["Xtr"].with_suffix(".pkl").exists()):
            Xtr = _load_df(paths["Xtr"])
            Xva = _load_df(paths["Xva"])
            y_tr = _load_df(paths["ytr"])[target_col]
            y_va = _load_df(paths["yva"])[target_col]
            return Xtr, y_tr, Xva, y_va

        # Build features (one-time per fold)
        tr = slicer(train_df, train_rng)
        va = slicer(train_df, val_rng)

        # Sort once if your make_features expects pre_sorted=True (put it inside your wrapper)
        t0 = time.perf_counter()
        Xtr = make_features(tr, max_lag)
        Xva = make_features(va, max_lag)
        y_tr = Xtr[target_col]; Xtr = Xtr.drop(columns=[target_col], errors="ignore")
        y_va = Xva[target_col]; Xva = Xva.drop(columns=[target_col], errors="ignore")
        print(f"[HPO] Built features for fold {i} in {time.perf_counter()-t0:.1f}s | Xtr={Xtr.shape}, Xva={Xva.shape}")

        # Cache to disk
        if paths:
            _save_df(Xtr, paths["Xtr"])
            _save_df(Xva, paths["Xva"])
            _save_df(y_tr.to_frame(target_col), paths["ytr"])
            _save_df(y_va.to_frame(target_col), paths["yva"])
            print(f"[HPO] Cached fold {i} features to {cache_root}")
        return Xtr, y_tr, Xva, y_va

    def CV_EVAL(params_for_model: Dict[str, Any], trial=None):
        """Run CV across used_windows and return (mean_metric, fold_details)."""
        scores = []
        details = []
        for i, (train_rng, val_rng) in enumerate(used_windows):
            Xtr, y_tr, Xva, y_va = _load_or_build_fold(i, train_rng, val_rng)

            # Build model_cfg compatible with TRAIN_MODEL
            framework = "lightgbm" if model_name.lower() == "lgbm" else "xgboost"
            model_cfg = SimpleNamespace(framework=framework, params=params_for_model)

            # Train + predict
            model = TRAIN_MODEL(Xtr, y_tr, Xva, y_va, model_cfg)
            preds = PREDICT(model, Xva, model_cfg)

            sc = EVALUATE_METRICS(y_va, preds, metrics=(metric_u,)).get(metric_u, np.inf)
            scores.append(sc)
            details.append({
                "fold": i,
                "val_start": pd.to_datetime(val_rng[0]),
                "val_end": pd.to_datetime(val_rng[1]),
                "score": float(sc),
            })

            if trial is not None:
                trial.report(sc, step=i + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_sc = float(np.mean(scores)) if scores else float("inf")
        return mean_sc, details

    def OBJECTIVE(trial: "optuna.trial.Trial") -> float:
        tuned = SUGGEST_PARAMS(trial)
        params_for_model = BUILD_MODEL_PARAMS(base_model_params, tuned)
        print(f"[HPO] Trial {trial.number} | tuned={tuned}")
        cv_mean, _ = CV_EVAL(params_for_model, trial=trial)
        print(f"[HPO] Trial {trial.number} | mean {metric_u}={cv_mean:.5f}")
        return cv_mean

    # Pruner
    if pruner == "median":
        pruner_obj = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    else:
        pruner_obj = optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner_obj)
    print(f"[HPO] Start study | model={model_name} | metric={metric_u} | trials={n_trials} | pruner={pruner}")
    study.optimize(OBJECTIVE, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best_tuned = study.best_params
    best_params_for_model = BUILD_MODEL_PARAMS(base_model_params, best_tuned)
    best_cv_mean, best_cv_folds = CV_EVAL(best_params_for_model, trial=None)

    # Trial history
    history = []
    for t in study.trials:
        history.append({
            "number": t.number,
            "value": float(t.value) if t.value is not None else None,
            "state": str(t.state),
            "params": dict(t.params),
        })

    result = {
        "best_model_params": best_params_for_model,
        "best_tuned_only": best_tuned,
        "cv_mean": best_cv_mean,
        "cv_by_fold": best_cv_folds,
        "study_summary": history,
    }
    print(f"[HPO] Done | best mean {metric_u}={best_cv_mean:.5f} | tuned={best_tuned}")
    return result
