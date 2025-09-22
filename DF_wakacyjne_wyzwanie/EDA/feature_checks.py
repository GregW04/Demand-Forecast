# feature_checks.py
import pandas as pd
# import numpy as np

def check_features_quick(df: pd.DataFrame, target: str | None = None, topk: int = 15) -> None:
    """
    Quick sanity check for feature matrix:
    - shape and memory
    - dtype distribution
    - top NA ratios
    - constant columns
    - categorical cardinalities
    - optional target sanity (if target column present)
    """
    shape = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"[FEATS] shape={shape}, memory={mem_mb:.1f} MB")

    dtype_counts = df.dtypes.astype(str).value_counts()
    print("[FEATS] dtype counts:")
    print(dtype_counts.to_string())

    na_ratio = df.isna().mean().sort_values(ascending=False)
    print(f"[FEATS] top {topk} NA ratios:")
    print(na_ratio.head(topk).to_string())

    nunique = df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"[FEATS] constant columns ({len(const_cols)}): {const_cols[:topk]}{' ...' if len(const_cols) > topk else ''}")
    else:
        print("[FEATS] constant columns: none")

    cat_cols = [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c])]
    if cat_cols:
        card = pd.Series({c: df[c].nunique(dropna=False) for c in cat_cols}).sort_values(ascending=False)
        print(f"[FEATS] top {topk} categorical cardinalities:")
        print(card.head(topk).to_string())
    else:
        print("[FEATS] categorical columns: none")

    if target is not None and target in df.columns:
        t = pd.to_numeric(df[target], errors="coerce")
        print(f"[FEATS] target '{target}': count={t.notna().sum()}, NA={t.isna().sum()}, mean={t.mean():.4f}, std={t.std():.4f}")