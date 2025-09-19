import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from functools import reduce
import gc

# Enable pandas' copy-on-write mode to reduce unnecessary data copies during operations
pd.options.mode.copy_on_write = True


def clean(holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing of holiday_events - cleaning.

    Parameters:
    - holidays: DataFrame holiday_events.csv
    
    Result collumns:
    - date: date
    - scope: Local/Regional/National
    - scope_name: name of scope
    - anchor_key: clear name of holiday (ASCII, lower)
    - role: holiday/additional/transfer/bridge/event/work_day
    - offset_days: how many +/- days from anchor (ex. Navidad-1 = -1)
    """

    df = holidays.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Standardize column names for locale/scope
    df = df.rename(columns={"locale":"scope", "locale_name":"scope_name"})

    # # # Parse role / anchor / offset from description and type

    # Normalize description: ensure string, no NaN, strip whitespace
    desc = df["description"].fillna("").astype(str).str.strip()

    role = []
    anchor_label = []
    offset = []

    # Iterate row-wise over type and normalized description
    for holiday_type, description_text in zip(df["type"], desc):
        role_value = "other"            # default role if nothing matches
        anchor_value = description_text # default anchor is the full description
        offset_value = 0                # default: no offset

        if holiday_type == "Holiday":
            # Find pattern: "<anchor><+/-N>" e.g., "Navidad-1", "Navidad+2"
            match = re.match(r"^(.*?)([+-]\d+)$", description_text)
            if match:
                role_value = "additional"               # treat suffix as an additional day
                anchor_value = match.group(1).strip()   # base anchor without suffix
                offset_value = int(match.group(2))      # signed offset in days
            else:
                role_value = "holiday"

        elif holiday_type == "Transfer":
            role_value = "transfer"
            # Remove Spanish prefix "traslado" (case-insensitive) to recover the anchor name
            anchor_value = re.sub(r"(?i)^traslado\s+", "", description_text).strip()

        elif holiday_type == "Bridge":
            role_value = "bridge"
            # Remove Spanish prefix "puente" (case-insensitive)
            anchor_value = re.sub(r"(?i)^puente\s+", "", description_text).strip()

        elif holiday_type == "Work Day":
            role_value = "work_day"
            # Remove Spanish phrase "recupero puente" (case-insensitive)
            anchor_value = re.sub(r"(?i)^recupero\s+puente\s+", "", description_text).strip()

        elif holiday_type == "Additional":
            role_value = "additional"
            # Additional days mostly come with an explicit +/-N suffix
            match = re.match(r"^(.*?)([+-]\d+)$", description_text)
            if match:
                anchor_value = match.group(1).strip()   # base anchor without suffix
                offset_value = int(match.group(2))      # signed offset in days

        elif holiday_type == "Event":
            role_value = "event"
            # Events may also include an offset suffix
            match = re.match(r"^(.*?)([+-]\d+)$", description_text)
            if match:
                anchor_value = match.group(1).strip()   # base anchor without suffix
                offset_value = int(match.group(2))      # signed offset in days

        else:
            # Fallback: normalize unknown types by lowercasing and replacing spaces with underscores
            # (unlikely to happen)
            role_value = holiday_type.lower().replace(" ", "_")

        # Collect parsed values
        role.append(role_value)
        anchor_label.append(anchor_value)
        offset.append(offset_value)

    # Assign parsed columns
    df["role"] = role
    df["anchor_label"] = anchor_label
    df["offset_days"] = offset

    # Build a normalized anchor key:
    # - Fill missing with empty string
    # - Convert to ASCII, lowercase, strip
    # - Collapse multiple whitespaces to a single space
    df["anchor_key"] = (df["anchor_label"].fillna("")
                        .map(lambda s: re.sub(r"\s+"," ",unidecode(str(s)).lower().strip())))
    
    return df


def add_business(holidays: pd.DataFrame,
                 #stores: pd.DataFrame
                 ) -> pd.DataFrame:
    """
    Preprocessing of holiday_events - add business locations (city, state).

    Parameters:
    - holidays: DataFrame holiday_events.csv
    - stores: DataFrame stores.csv (for city:state mapping)

    Add collumns:
    - city, state: compatible with stores.csv
    """

    df = holidays.copy()

    # # # Map city/state based on scope and the stores reference

    # Prepare output columns compatible with stores.csv
    df["city"] = pd.NA
    df["state"] = pd.NA

    # Build a city:state mapping from stores
    # city_to_state = stores.drop_duplicates("city").set_index("city")["state"].to_dict()

    # Masks for scope types
    mask_L = df["scope"].eq("Local")
    mask_R = df["scope"].eq("Regional")

    # Local scope applies to city, derive state via mapping
    df.loc[mask_L, "city"] = df.loc[mask_L, "scope_name"]
    # df.loc[mask_L, "state"] = df.loc[mask_L, "city"].map(city_to_state)

    # Regional scope applies to state directly
    df.loc[mask_R, "state"] = df.loc[mask_R, "scope_name"]

    return df


def add_features(holidays: pd.DataFrame):
    """
    Preprocessing of holiday_events - add features.

    Parameters:
    - holidays: DataFrame holiday_events.csv

    Add collumns:
    - is_national: if scope == National
    - is_holiday: if role == holiday
    - is_event: if role == event
    - is_day_off: is it truly day off
    - holiday_duration: how long does the holiday last
    """

    df = holidays.copy()

    # # # Identify national
    df['is_national'] = df['scope'] == 'National'

    # # # Identify actual holidays (not bridges, events, etc.)
    df['is_holiday'] = df['role'] == 'holiday'

    # # # Identify events
    df['is_event'] = df['role'] == 'event'

    # # # Identify actual days off
    # Day off rules:
    # - Non-transferred holidays are days off
    # - Transfers are days off (holiday moved)
    # - Bridge days are days off
    df["is_day_off"] = (
        ((df["role"] == "holiday") & (~df["transferred"])) |
        (df["role"] == "transfer") |
        (df["role"] == "bridge")
    )

    # # # Compute current_holiday_len
    # Group by anchor_key
    grouped = df.groupby("anchor_key")["offset_days"]

    # Count the range for each anchor_key
    holiday_ranges = grouped.agg(["min", "max"])

    # Calculate duration (max - min + 1)
    holiday_ranges["holiday_duration"] = holiday_ranges["max"] - holiday_ranges["min"] + 1

    # Now merge back to df
    df = df.merge(
        holiday_ranges["holiday_duration"],
        left_on="anchor_key", right_index=True,
        how="left"
    )

    return df


def drop(holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing of holiday_events - drop unnecessary columns.

    Parameters:
    - holidays: DataFrame holiday_events.csv
    - stores: DataFrame stores.csv (for city:state mapping)

    Drop collumns:
    'type', 'description', 'transferred', 'anchor_label'
    """
    to_drop = ['type', 'scope_name', 'description', 'transferred', 'anchor_label']
    return holidays.drop(to_drop, axis=1)


def preprocess(holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing of holiday_events.

    Parameters:
    - holidays: DataFrame holiday_events.csv (must be preprocessesd)

    Result collumns:
    - date: date
    - scope: Local/Regional/National
    - scope_name: name of scope
    - anchor_key: clear name of holiday (ASCII, lower)
    - role: holiday/additional/transfer/bridge/event/work_day
    - offset_days: how many +/- days from anchor (ex. Navidad-1 = -1)
    - city, state: compatible with stores.csv
    - is_national: id holiday is national
    - is_holiday: if date is actual holiday
    - is_event: if date is an event
    - is_day_off: is it truly day off
    - holiday_duration: how long does the holiday last
    """
    df = holidays.copy()
    df = clean(df)
    df = add_business(df)
    df = add_features(df)
    df = drop(df)
    return df


def merge_train(train: pd.DataFrame,
                holidays: pd.DataFrame,
                stores: pd.DataFrame) -> pd.DataFrame:
    """
    Merge training data with holiday signals at three scopes (national / regional / local)
    in a memory-efficient way: small dtypes (UInt8), proper merge/groupby keys,
    minimal copies, and no float64 expansions.

    Returns the original training columns plus 4 holiday-related features:
      - is_holiday (UInt8: 0/1)
      - holiday_duration (UInt8: number of days, max across scopes)
      - is_day_off (UInt8: 0/1)
      - is_event (UInt8: 0/1)
    """
    # Work on a shallow copy so we don't mutate the caller's DataFrame
    train = train.copy()
    # Keep the original column list; we will return only these plus the 4 new features
    base_cols = list(train.columns)

    # Parse dates and downcast numeric dtypes to save memory in train
    train["date"] = pd.to_datetime(train["date"])
    train["store_nbr"] = pd.to_numeric(train["store_nbr"], downcast="integer")
    if "item_nbr" in train:
        train["item_nbr"] = pd.to_numeric(train["item_nbr"], downcast="integer")
    if "unit_sales" in train:
        train["unit_sales"] = pd.to_numeric(train["unit_sales"], downcast="float")
    if "onpromotion" in train:
        # float32 keeps memory low; NaN can be represented if present
        train["onpromotion"] = train["onpromotion"].astype("float32")

    # Ensure city/state are present in train; if missing, bring them from stores
    if not {"city", "state"}.issubset(train.columns):
        train = train.merge(
            stores[["store_nbr", "city", "state"]],
            on="store_nbr",
            how="left",
            sort=False,       # keep current row order
            copy=False,       # rely on copy-on-write to avoid allocating a new block
            validate="m:1",   # many train rows per one store row
        )

    # Convert object-typed categoricals to 'category' to reduce memory usage
    for c in ("family", "city", "state"):
        if c in train.columns and train[c].dtype == "object":
            train[c] = train[c].astype("category")

    # Prepare a lean holidays table with only needed columns and compact dtypes
    H = holidays[[
        "date", "scope", "city", "state",
        "is_holiday", "is_event", "is_day_off", "holiday_duration"
    ]].copy()
    H["date"] = pd.to_datetime(H["date"])
    for c in ("city", "state"):
        if c in H.columns and H[c].dtype == "object":
            H[c] = H[c].astype("category")

    # Use pandas' nullable Boolean for flags in H; duration fits in an unsigned byte
    H["is_holiday"] = H["is_holiday"].astype("boolean")
    H["is_event"] = H["is_event"].astype("boolean")
    H["is_day_off"] = H["is_day_off"].astype("boolean")
    H["holiday_duration"] = H["holiday_duration"].fillna(0).astype("UInt8")

    # Define merge keys per scope
    scopes = {
        "nat": ["date"],           # National scope: keyed by date only
        "reg": ["date", "state"],  # Regional scope: keyed by date and state
        "loc": ["date", "city"],   # Local scope: keyed by date and city
    }
    scope_map = {"nat": "national", "reg": "regional", "loc": "local"}

    # Collect columns added per scope so we can drop them later
    scoped_cols = []

    for sfx, keys in scopes.items():
        scope_name = scope_map[sfx]
        # Filter holidays by scope (case-insensitive)
        sub = H[H["scope"].str.lower() == scope_name]

        if sub.empty:
            continue

        # Aggregate per key; observed=True avoids generating unused category combinations
        sub_agg = (
            sub.groupby(keys, observed=True).agg(
                is_holiday=("is_holiday", "max"),        # logical OR for booleans via max
                is_event=("is_event", "max"),
                is_day_off=("is_day_off", "max"),
                holiday_duration=("holiday_duration", "max"),  # take max duration if overlaps
            )
            .reset_index()
        )

        # Cast flags to UInt8 (0/1) and ensure no NaNs
        for b in ("is_holiday", "is_event", "is_day_off"):
            sub_agg[b] = sub_agg[b].fillna(False).astype("UInt8")
        sub_agg["holiday_duration"] = sub_agg["holiday_duration"].fillna(0).astype("UInt8")

        # Many-to-one merge from train to aggregated holidays for this scope
        train = train.merge(
            sub_agg, on=keys, how="left", sort=False, copy=False, validate="m:1"
        )

        # Suffix new columns with the scope short code; fill NaNs to avoid float upcast
        rename = {c: f"{c}_{sfx}" for c in ("is_holiday", "is_event", "is_day_off", "holiday_duration")}
        train.rename(columns=rename, inplace=True)
        new_cols = list(rename.values())

        for c in new_cols:
            # All of these are small integers: binary flags or a small day count
            train[c] = train[c].fillna(0).astype("UInt8")

        scoped_cols.extend(new_cols)

        # Proactively free temporary objects to keep peak memory low
        del sub_agg, sub
        gc.collect()

    # Helper to safely fetch a column as UInt8; returns a zero-filled series if missing
    def get_series(name: str) -> pd.Series:
        if name in train.columns:
            # already UInt8 without NaN
            return train[name].astype("UInt8")
        return pd.Series(0, index=train.index, dtype="UInt8")

    # Combine flags across scopes via bitwise OR; result stays in UInt8 (0/1)
    train["is_event"] = (get_series("is_event_nat") | get_series("is_event_reg") | get_series("is_event_loc")).astype("UInt8")
    train["is_holiday"] = (get_series("is_holiday_nat") | get_series("is_holiday_reg") | get_series("is_holiday_loc")).astype("UInt8")
    train["is_day_off"] = (get_series("is_day_off_nat") | get_series("is_day_off_reg") | get_series("is_day_off_loc")).astype("UInt8")

    # Take the maximum holiday_duration across scopes
    hd = get_series("holiday_duration_nat")
    hd = np.maximum(hd, get_series("holiday_duration_reg")).astype("UInt8")
    hd = np.maximum(hd, get_series("holiday_duration_loc")).astype("UInt8")
    train["holiday_duration"] = hd

    # Drop per-scope intermediate columns; the combined features are now in place
    if scoped_cols:
        train.drop(columns=[c for c in scoped_cols if c in train.columns], inplace=True)

    # Return only the original columns plus the 4 final holiday features
    out_cols = base_cols + ["is_holiday", "holiday_duration", "is_day_off", "is_event"]
    return train[out_cols].reset_index(drop=True)

