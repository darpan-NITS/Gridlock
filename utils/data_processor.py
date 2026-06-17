import pandas as pd
import numpy as np


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Drop columns >60% null, but always keep junction
    null_pct   = df.isnull().mean()
    drop_cols  = null_pct[null_pct > 0.60].index
    drop_cols  = [c for c in drop_cols if c != "junction"]
    df         = df.drop(columns=drop_cols, errors="ignore")

    df["junction"] = df["junction"].fillna("Unknown")

    required = [
        "event_type", "start_datetime", "end_datetime",
        "latitude", "longitude", "corridor", "junction",
        "priority", "requires_road_closure", "status", "event_cause",
    ]
    existing = [c for c in required if c in df.columns]
    return df[existing].copy()


def categorize_cause(cause) -> str:
    if pd.isna(cause):
        return "Other"
    cause = str(cause).lower().strip()
    mapping = {
        "accident":                   "Accident",
        "vehicle_breakdown":          "Breakdown",
        "water_logging":              "Infrastructure",
        "tree_fall":                  "Infrastructure",
        "debris":                     "Infrastructure",
        "pot_holes":                  "Infrastructure",
        "road_conditions":            "Infrastructure",
        "fog / low visibility":       "Infrastructure",
        "vip_movement":               "Special Event",
        "procession":                 "Special Event",
        "protest":                    "Special Event",
        "public_event":               "Special Event",
        "construction":               "Construction",
        "congestion":                 "Traffic",
    }
    return mapping.get(cause, "Other")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["start_datetime"] = pd.to_datetime(df["start_datetime"], format="ISO8601", errors="coerce")

    median_duration = 60
    if "end_datetime" in df.columns:
        df["end_datetime"]      = pd.to_datetime(df["end_datetime"], format="ISO8601", errors="coerce")
        df["duration_minutes"]  = (df["end_datetime"] - df["start_datetime"]).dt.total_seconds() / 60
        df["duration_minutes"]  = df["duration_minutes"].clip(0, 1440).fillna(median_duration)
    else:
        df["duration_minutes"] = median_duration

    df["hour_of_day"]  = df["start_datetime"].dt.hour.fillna(0).astype(int)
    df["day_of_week"]  = df["start_datetime"].dt.day_name()
    df["is_weekend"]   = (df["start_datetime"].dt.weekday >= 5).astype(int)

    if "event_cause" in df.columns:
        df["event_cause_category"] = df["event_cause"].apply(categorize_cause)
    else:
        df["event_cause_category"] = "Other"

    return df


def compute_kpi_stats(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPI stats for the dashboard strip.
    Returns a dict consumed directly by app.py.
    """
    total_events = len(df)

    avg_resolution = (
        df["duration_minutes"].mean()
        if "duration_minutes" in df.columns
        else 60.0
    )

    closure_rate = 0.0
    if "requires_road_closure" in df.columns:
        closure_series = df["requires_road_closure"]
        # Handle both bool and string representations
        if closure_series.dtype == object:
            closure_series = closure_series.astype(str).str.lower().isin(["true", "yes", "1"])
        closure_rate = float(closure_series.mean() * 100)

    return {
        "total_events":        total_events,
        "avg_resolution_min":  float(avg_resolution),
        "closure_rate":        closure_rate,
    }
