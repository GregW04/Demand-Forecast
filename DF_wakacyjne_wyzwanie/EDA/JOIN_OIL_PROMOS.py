import gc
import pandas as pd
import numpy as np

# Enable copy-on-write to avoid unnecessary data copies (pandas 2.x)
pd.options.mode.copy_on_write = True


def JOIN_OIL_PROMOS(
    train: pd.DataFrame,
    oil: pd.DataFrame,
    promos: pd.DataFrame,
    *,
    oil_col: str | None = None,
    overwrite_onpromotion: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Attach oil prices ('oil') and processed promotions ('promos') to the 'train' DataFrame
    in a memory-aware way.

    - Oil: left-join on 'date'. The selected oil price column is cast to float32
           and exposed as 'oil_price'.
    - Promos: left-join on the best available key among
              ['date','store_nbr','item_nbr'] → ['date','store_nbr'] → ['date'].
              Protects against many-to-many merges via drop_duplicates on keys,
              resolves name collisions, and (by default) replaces train['onpromotion']
              with the version from 'promos'.

    Parameters:
    - train: base DataFrame to enrich.
    - oil: oil price time series (must contain a 'date' column or a DatetimeIndex).
    - promos: promotions dataset with date/store/item keys and promo feature columns.
    - oil_col: the name of the oil price column in 'oil'. If None, try 'dcoilwtico',
               else fall back to the first numeric column that's not 'date'.
    - overwrite_onpromotion: if True, drop 'onpromotion' from train and use the one from promos;
                             if False, promo columns that collide get a 'promo_' prefix.
    - verbose: if True, print some diagnostics (shapes, columns).

    Returns:
    - A copy of train with 'oil_price' and the promotion columns joined in.
    """
    # Work on a copy to avoid mutating the caller's DataFrame
    t = train.copy()

    # Ensure join keys in 'train' are correctly typed
    t["date"] = pd.to_datetime(t["date"])
    if "store_nbr" in t:
        t["store_nbr"] = pd.to_numeric(t["store_nbr"], downcast="integer")
    if "item_nbr" in t:
        t["item_nbr"] = pd.to_numeric(t["item_nbr"], downcast="integer")

    # ===== OIL =====
    if oil is not None and len(oil):
        o = oil.copy()

        # Ensure there is a 'date' column; if index is DatetimeIndex, convert it to a column
        if "date" not in o.columns:
            # sometimes 'oil' has the date as the index
            if isinstance(o.index, pd.DatetimeIndex):
                o = o.reset_index().rename(columns={"index": "date"})
            else:
                raise ValueError("Tabela 'oil' nie ma kolumny 'date' ani DatetimeIndex.")

        # Normalize date dtype
        o["date"] = pd.to_datetime(o["date"])

        # Choose the oil price column:
        # 1) prefer 'dcoilwtico', 2) else the first numeric column that's not 'date'
        if oil_col is None:
            if "dcoilwtico" in o.columns:
                oil_col = "dcoilwtico"
            else:
                candidates = [c for c in o.columns if c != "date" and pd.api.types.is_numeric_dtype(o[c])]
                if not candidates:
                    raise ValueError("Nie znaleziono kolumny z ceną ropy w 'oil'. Podaj oil_col.")
                oil_col = candidates[0]

        # Prepare the oil table for a many-to-one merge on 'date'
        o = o[["date", oil_col]].drop_duplicates(subset=["date"], keep="last")
        o = o.rename(columns={oil_col: "oil_price"})
        # Compact dtype to float32 to save memory
        o["oil_price"] = pd.to_numeric(o["oil_price"], downcast="float").astype("float32")

        if verbose:
            print(f"[JOIN_OIL_PROMOS] Oil before merge: {o.shape}, cols={list(o.columns)}")

        # Many-to-one merge: many train rows can map to one date in oil
        t = t.merge(o, on="date", how="left", sort=False, copy=False, validate="m:1")

        # Free temporary objects proactively
        del o
        gc.collect()

    # ===== PROMOS =====
    if promos is not None and len(promos):
        p = promos.copy()

        # Prefer the most specific key available: (date, store, item) → (date, store) → (date)
        candidate_keys = [["date", "store_nbr", "item_nbr"],
                          ["date", "store_nbr"],
                          ["date"]]
        keys = None
        for ks in candidate_keys:
            # Pick the first key set present in BOTH frames
            if all(k in p.columns and k in t.columns for k in ks):
                keys = ks
                break
        if keys is None:
            raise ValueError("Nie mogę dobrać kluczy do join z 'promos'. "
                             "Wymagane przynajmniej 'date' (+ opcjonalnie 'store_nbr','item_nbr').")

        # Align key dtypes between frames to ensure a clean merge
        p["date"] = pd.to_datetime(p["date"])
        if "store_nbr" in keys:
            p["store_nbr"] = pd.to_numeric(p["store_nbr"], downcast="integer")
        if "item_nbr" in keys:
            p["item_nbr"] = pd.to_numeric(p["item_nbr"], downcast="integer")

        # Determine which columns to bring from 'promos' (exclude join keys)
        feat_cols = [c for c in p.columns if c not in keys]
        if not feat_cols:
            # Nothing to attach if only keys are present
            if verbose:
                print("[JOIN_OIL_PROMOS] 'promos' nie zawiera kolumn do połączenia poza kluczami.")
        else:
            # Resolve name collisions with columns already in 'train'
            rename_map = {}
            collisions = set(feat_cols).intersection(t.columns)

            # Special handling for 'onpromotion'
            if "onpromotion" in collisions and overwrite_onpromotion:
                # Use the version from promos; drop the current one from train
                t.drop(columns=["onpromotion"], inplace=True, errors="ignore")
                collisions.remove("onpromotion")  # no longer collides

            # Prefix remaining colliding columns with 'promo_' to avoid overwriting
            for c in collisions:
                rename_map[c] = f"promo_{c}"

            # Apply renaming to the subset we will merge
            p_ren = p[keys + feat_cols].rename(columns=rename_map)

            # Keep 'onpromotion' compact in memory when present
            if "onpromotion" in p_ren.columns:
                # If numeric and strictly 0/1, use nullable Int8 (0/1/NA)
                if pd.api.types.is_numeric_dtype(p_ren["onpromotion"]):
                    uniq = set(pd.Series(p_ren["onpromotion"]).dropna().unique().tolist())
                    if uniq.issubset({0, 1}):
                        p_ren["onpromotion"] = p_ren["onpromotion"].astype("Int8")
                    else:
                        # Otherwise keep it numeric but downcast to float
                        p_ren["onpromotion"] = pd.to_numeric(p_ren["onpromotion"], downcast="float")
                elif p_ren["onpromotion"].dtype == "object":
                    # Strings/categories → use category dtype
                    p_ren["onpromotion"] = p_ren["onpromotion"].astype("category")

            # Drop duplicates on keys to ensure a valid many-to-one merge
            p_ren = p_ren.drop_duplicates(subset=keys, keep="last")

            if verbose:
                print(f"[JOIN_OIL_PROMOS] Promos before merge: {p_ren.shape}, keys={keys}, "
                      f"new_cols={[c for c in p_ren.columns if c not in keys]}")

            # Many-to-one merge on the selected keys
            t = t.merge(p_ren, on=keys, how="left", sort=False, copy=False, validate="m:1")

            # Cleanup
            del p_ren
            gc.collect()

    return t