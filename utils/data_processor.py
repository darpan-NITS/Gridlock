
import glob
from pathlib import Path

import numpy as np
import pandas as pd


IMPORTANT_COLUMNS = {
    "event_type",
    "start_datetime",
    "end_datetime",
    "closed_datetime",
    "resolved_datetime",
    "created_date",
    "modified_datetime",
    "latitude",
    "longitude",
    "corridor",
    "junction",
    "priority",
    "requires_road_closure",
    "status",
    "event_cause",
    "description",
    "veh_type",
    "reason_breakdown",
    "police_station",
}


def _resolve_file_path(file_path: str) -> str:
    """Try a few sensible locations before giving up."""
    candidate = Path(file_path)
    if candidate.exists():
        return str(candidate)

    search_roots = [
        Path.cwd(),
        Path.cwd() / "data",
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent / "data",
    ]
    for root in search_roots:
        direct = root / file_path
        if direct.exists():
            return str(direct)

    
    patterns = [
        str(Path.cwd() / "**" / "*.csv"),
        str(Path(__file__).resolve().parent.parent / "**" / "*.csv"),
    ]
    for pat in patterns:
        for match in glob.glob(pat, recursive=True):
            name = Path(match).name.lower()
            if "astram" in name or "event" in name or "incident" in name:
                return match

    raise FileNotFoundError(f"Could not locate dataset: {file_path}")


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    path = _resolve_file_path(file_path)
    df = pd.read_csv(path)

    
    null_pct = df.isnull().mean(numeric_only=False)
    drop_cols = [c for c in null_pct.index if null_pct[c] > 0.60 and c not in IMPORTANT_COLUMNS]
    df = df.drop(columns=drop_cols, errors="ignore").copy()

   
    for col in IMPORTANT_COLUMNS:
        if col not in df.columns:
            if col in {"requires_road_closure"}:
                df[col] = False
            else:
                df[col] = np.nan

    
    for col in ["start_datetime", "end_datetime", "closed_datetime", "resolved_datetime", "created_date", "modified_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="ISO8601")

    df["junction"] = df["junction"].fillna("Unknown")
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df["status"] = df["status"].fillna("unknown")
    df["priority"] = df["priority"].fillna("P3")

    if "requires_road_closure" in df.columns:
        if df["requires_road_closure"].dtype == object:
            df["requires_road_closure"] = (
                df["requires_road_closure"].astype(str).str.lower().isin(["true", "yes", "1", "y"])
            )
        else:
            df["requires_road_closure"] = df["requires_road_closure"].fillna(False).astype(bool)

    return df


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

    if "start_datetime" in df.columns:
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", format="ISO8601")
    else:
        df["start_datetime"] = pd.NaT

    
    end_source = None
    for candidate in ["end_datetime", "resolved_datetime", "closed_datetime"]:
        if candidate in df.columns:
            end_source = candidate
            break

    median_duration = 60.0
    if end_source:
        df[end_source] = pd.to_datetime(df[end_source], errors="coerce", format="ISO8601")
        df["duration_minutes"] = (df[end_source] - df["start_datetime"]).dt.total_seconds() / 60
        df["duration_minutes"] = df["duration_minutes"].clip(0, 1440).fillna(median_duration)
    else:
        df["duration_minutes"] = median_duration

    df["hour_of_day"] = df["start_datetime"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["start_datetime"].dt.day_name()
    df["is_weekend"] = (df["start_datetime"].dt.weekday >= 5).astype(int)

    if "event_cause" in df.columns:
        df["event_cause_category"] = df["event_cause"].apply(categorize_cause)
    else:
        df["event_cause_category"] = "Other"

    if "requires_road_closure" in df.columns:
        if df["requires_road_closure"].dtype == object:
            df["requires_road_closure"] = (
                df["requires_road_closure"].astype(str).str.lower().isin(["true", "yes", "1", "y"])
            )
        else:
            df["requires_road_closure"] = df["requires_road_closure"].fillna(False).astype(bool)
    else:
        df["requires_road_closure"] = False

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
        if closure_series.dtype == object:
            closure_series = closure_series.astype(str).str.lower().isin(["true", "yes", "1"])
        closure_rate = float(closure_series.mean() * 100)

    peak_rate = 0.0
    if "hour_of_day" in df.columns:
        peak_rate = float(df["hour_of_day"].isin([7, 8, 9, 10, 17, 18, 19, 20]).mean() * 100)

    active_rate = 0.0
    if "status" in df.columns:
        active_rate = float(df["status"].astype(str).str.lower().isin(["active", "pending"]).mean() * 100)

    return {
        "total_events":        total_events,
        "avg_resolution_min":  float(avg_resolution),
        "closure_rate":        closure_rate,
        "peak_rate":           peak_rate,
        "active_rate":         active_rate,
    }
