import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from functools import reduce


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
    - holidays: DataFrame holiday_events.csv

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
    Merge training data with store metadata and holiday signals at multiple scopes
    (national, regional, local), and compute aggregated holiday features.

    Parameters
    ----------
    train : pd.DataFrame
        One row per store/date. Must include at least ['store_nbr', 'date', ...].
    holidays : pd.DataFrame
        Raw holiday calendar. After preprocess(holidays), it is expected to contain:
        - 'date' (datetime), 'scope' (e.g., 'national', 'regional', 'local'),
          'anchor_key' (identifier for a holiday event/window),
          'is_holiday' (bool), 
          'holiday_duration' (int),
          'is_day_off' (bool),
          'is_event' (bool),
          and optionally location fields used by scopes (e.g., 'state', 'city').
    stores : pd.DataFrame
        Store reference with at least ['store_nbr', 'city', 'state'].

    Returns
    -------
    pd.DataFrame
        The original train columns plus aggregated holiday features:
        ['is_holiday', 'holiday_duration', 'is_day_off', 'is_event'].
        Scope-specific columns are not retained in the final output.
    """
    train = train.copy()
    cols = list(train.columns)
    train["date"] = pd.to_datetime(train["date"])
    holidays = preprocess(holidays)

    # Attach store-level geography so we can join regional/local holidays
    train = train.merge(
        stores[["store_nbr","city","state"]], 
        on="store_nbr", how="left"
    )

    # Join keys per scope:
    # - national: only date matters
    # - regional: date + state
    # - local:    date + city
    scopes = {
        "nat":  ["date"],
        "reg":  ["date", "state"],
        "loc":  ["date", "city"]
    }

    # Holiday attributes we want to bring in from the calendar for each scope
    base_cols = [
        "date","anchor_key","is_holiday",
        "holiday_duration","is_day_off", "is_event"
    ]

    for suffix, merge_keys in scopes.items():
        # Keep only rows that belong to the current scope (e.g., 'national', 'regional', 'local')
        sub_holidays = holidays[holidays.scope.str.lower().str.startswith(suffix)]

        # Avoid duplicate columns
        merge_cols = base_cols + [c for c in merge_keys if c not in base_cols]

        # Merge scope-specific holiday features.
        # drop_duplicates guards against many-to-many merges due to duplicate calendar rows.
        train = train.merge(
            sub_holidays[merge_cols].drop_duplicates(),
            on=merge_keys, how="left", suffixes=("","_" + suffix)
        )

        # Rename the merged columns to include the scope suffix so they don't collide across scopes.
        train.rename(columns={
            "is_holiday": f"is_holiday_{suffix}",
            "holiday_duration": f"holiday_duration_{suffix}",
            "anchor_key": f"anchor_key_{suffix}",
            "is_day_off": f"is_day_off_{suffix}",
            "is_event": f"is_event_{suffix}",
        }, inplace=True)

    # Aggregate across scopes:
    # - is_event: true if any scope marks the date as event
    train["is_event"] = (
        train[[f"is_event_{s}" for s in scopes]]
        .fillna(False).any(axis=1)
    )

    # Aggregate across scopes:
    # - is_holiday: true if any scope marks the date as holiday
    train["is_holiday"] = (
        train[[f"is_holiday_{s}" for s in scopes]]
        .fillna(False).any(axis=1)
    )

    # - holiday_duration: take the maximum duration among scopes (0 if none)
    train["holiday_duration"] = (
        train[[f"holiday_duration_{s}" for s in scopes]]
        .fillna(0).max(axis=1)
    )

    # - is_day_off: true if any scope indicates a day off
    train["is_day_off"] = (
        train[[f"is_day_off_{s}" for s in scopes]]
        .fillna(False).any(axis=1)
    )

    # Return the original columns plus aggregated holiday features.
    # Scope-specific columns are intentionally dropped to keep the output tidy.
    cols.extend(["is_holiday", "holiday_duration", 
                 "is_day_off", "is_event"])
    return train[cols].reset_index(drop=True)
