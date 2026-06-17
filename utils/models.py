import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────────────────────────────────────
# JUNCTION COORDINATE LOOKUP
# Real Bengaluru junction GPS coordinates for top known junctions.
# Fallback: dataset mean for unrecognised junctions.
# ─────────────────────────────────────────────────────────────────────────────
JUNCTION_COORDS = {
    "Silk Board Junction":     (12.9176, 77.6236),
    "Urvashi Junction":        (12.9184, 77.6048),
    "LalbaghMainGateJunc":     (12.9498, 77.5835),
    "Lalbagh Main Gate":       (12.9498, 77.5835),
    "Hebbal Flyover":          (13.0359, 77.5970),
    "Marathahalli Junction":   (12.9591, 77.6974),
    "KR Circle":               (12.9766, 77.5993),
    "MG Road Junction":        (12.9751, 77.6099),
    "Koramangala Junction":    (12.9279, 77.6271),
    "Yeshwanthpur Junction":   (13.0207, 77.5391),
    "Whitefield Signal":       (12.9698, 77.7499),
    "Electronic City Toll":    (12.8399, 77.6770),
    "Bannerghatta Road":       (12.9004, 77.5974),
    "Bellary Road Junction":   (13.0218, 77.5970),
    "JP Nagar Junction":       (12.9068, 77.5848),
    "Jayanagar 4th Block":     (12.9304, 77.5820),
    "Unknown":                 (12.9716, 77.5946),   # Bengaluru centre fallback
}

# ─────────────────────────────────────────────────────────────────────────────
# DIVERSION ROUTE LOOKUP
# Hardcoded realistic Bengaluru diversion routes per corridor.
# Each route has: name, primary route (polyline coords), alternate name,
# and barricade points.
# ─────────────────────────────────────────────────────────────────────────────
DIVERSION_ROUTES = {
    "Tumkur Road": {
        "name":          "Via Chord Road → Rajajinagar → Yeshwanthpur",
        "alternate_name": "Via Magadi Road → Kengeri",
        "primary_route": [
            [13.0207, 77.5391],
            [13.0100, 77.5500],
            [12.9980, 77.5600],
            [12.9850, 77.5680],
        ],
        "barricades": [
            [13.0207, 77.5391],
            [12.9980, 77.5600],
        ],
    },
    "ORR East 1": {
        "name":          "Via Marathahalli Bridge → Old Airport Road",
        "alternate_name": "Via Sarjapur Road → HSR Layout",
        "primary_route": [
            [12.9591, 77.6974],
            [12.9500, 77.6800],
            [12.9400, 77.6600],
            [12.9300, 77.6400],
        ],
        "barricades": [
            [12.9591, 77.6974],
            [12.9400, 77.6600],
        ],
    },
    "Hosur Road": {
        "name":          "Via Bannerghatta Road → JP Nagar → Ring Road",
        "alternate_name": "Via Sarjapur Road → HSR Layout",
        "primary_route": [
            [12.9176, 77.6236],
            [12.9100, 77.6100],
            [12.9000, 77.5970],
            [12.8900, 77.5900],
        ],
        "barricades": [
            [12.9176, 77.6236],
            [12.9000, 77.5970],
        ],
    },
    "Bannerghatta Road": {
        "name":          "Via Jayanagar → KR Road → City Market",
        "alternate_name": "Via Kanakapura Road → Uttarahalli",
        "primary_route": [
            [12.9068, 77.5848],
            [12.9200, 77.5800],
            [12.9350, 77.5770],
            [12.9498, 77.5835],
        ],
        "barricades": [
            [12.9068, 77.5848],
            [12.9350, 77.5770],
        ],
    },
    "Old Airport Road": {
        "name":          "Via Indiranagar → CMH Road → Old Madras Road",
        "alternate_name": "Via HAL → Domlur → Richmond Road",
        "primary_route": [
            [12.9591, 77.6974],
            [12.9700, 77.6700],
            [12.9751, 77.6350],
            [12.9751, 77.6099],
        ],
        "barricades": [
            [12.9700, 77.6700],
            [12.9751, 77.6350],
        ],
    },
    "Bellary Road": {
        "name":          "Via Hebbal → Outer Ring Road → Nagavara",
        "alternate_name": "Via MS Ramaiah → Sankey Road",
        "primary_route": [
            [13.0359, 77.5970],
            [13.0200, 77.5800],
            [13.0050, 77.5700],
            [12.9900, 77.5600],
        ],
        "barricades": [
            [13.0359, 77.5970],
            [13.0050, 77.5700],
        ],
    },
    "Non-corridor": {
        "name":          "Use alternate local routes as directed",
        "alternate_name": None,
        "primary_route": [],
        "barricades":    [],
    },
    # ── Additional corridors found in the real dataset ──────────────────────
    "Mysore Road": {
        "name":          "Via Kanakapura Road → JP Nagar → Ring Road",
        "alternate_name": "Via Magadi Road → Chord Road",
        "primary_route": [
            [12.9716, 77.5350],
            [12.9500, 77.5500],
            [12.9304, 77.5820],
            [12.9068, 77.5848],
        ],
        "barricades": [
            [12.9716, 77.5350],
            [12.9500, 77.5500],
        ],
    },
    "Sarjapur Road": {
        "name":          "Via HSR Layout → Agara → BTM Layout",
        "alternate_name": "Via ORR → Bellandur → Outer Ring Road",
        "primary_route": [
            [12.9176, 77.6236],
            [12.9100, 77.6400],
            [12.9000, 77.6600],
            [12.8900, 77.6800],
        ],
        "barricades": [
            [12.9176, 77.6236],
            [12.9000, 77.6600],
        ],
    },
    "Kanakapura Road": {
        "name":          "Via Bannerghatta Road → JP Nagar → Ring Road",
        "alternate_name": "Via Uttarahalli → Talaghattapura",
        "primary_route": [
            [12.9068, 77.5700],
            [12.8900, 77.5600],
            [12.8700, 77.5500],
            [12.8500, 77.5400],
        ],
        "barricades": [
            [12.9068, 77.5700],
            [12.8700, 77.5500],
        ],
    },
    "ORR West": {
        "name":          "Via Yeshwanthpur → Tumkur Road → Hebbal",
        "alternate_name": "Via Rajajinagar → Chord Road",
        "primary_route": [
            [13.0207, 77.5391],
            [13.0300, 77.5600],
            [13.0359, 77.5970],
            [13.0200, 77.6100],
        ],
        "barricades": [
            [13.0207, 77.5391],
            [13.0300, 77.5600],
        ],
    },
}

_DEFAULT_ROUTE = {
    "name":          "Use nearest alternate road — check map",
    "alternate_name": None,
    "primary_route": [],
    "barricades":    [],
}


def get_junction_coords(junction: str) -> tuple:
    """Return (lat, lon) for a given junction name, with fallback."""
    return JUNCTION_COORDS.get(junction, JUNCTION_COORDS["Unknown"])


def get_diversion_route(corridor: str) -> dict:
    """Return diversion route dict for a corridor."""
    return DIVERSION_ROUTES.get(corridor, _DEFAULT_ROUTE)


# ─────────────────────────────────────────────────────────────────────────────
# CORE SCORING
# ─────────────────────────────────────────────────────────────────────────────
CAUSE_WEIGHTS = {
    "vip_movement":      5,
    "procession":        4,
    "protest":           4,
    "accident":          4,
    "public_event":      3,
    "construction":      3,
    "water_logging":     3,
    "congestion":        3,
    "vehicle_breakdown": 2,
}

CROWD_MULTIPLIERS = {
    "Small":  0.8,
    "Medium": 1.0,
    "Large":  1.3,
    "Mega":   1.7,
}


def congestion_score(event_type, event_cause, hour, junction, junction_history) -> int:
    """Rule-based severity score 0–10."""
    score = 1
    score += CAUSE_WEIGHTS.get(str(event_cause).lower(), 1)

    if hour in [7, 8, 9, 10, 17, 18, 19, 20]:
        score += 2
    elif hour in [11, 12, 13, 16]:
        score += 1

    avg_duration = junction_history.get(junction, 30)
    if avg_duration > 180:
        score += 2
    elif avg_duration > 90:
        score += 1

    if str(event_type).lower() == "planned":
        score -= 1

    return max(0, min(10, int(score)))


def whatif_score(event_cause, event_type, hour, junction, junction_history,
                 crowd_size="Medium", requires_closure=False) -> int:
    """What-if variant of congestion_score with crowd scale modifier."""
    base = congestion_score(event_type, event_cause, hour, junction, junction_history)
    multiplier = CROWD_MULTIPLIERS.get(crowd_size, 1.0)
    adjusted = base * multiplier
    if requires_closure:
        adjusted += 1.5
    return max(0, min(10, round(adjusted)))


# ─────────────────────────────────────────────────────────────────────────────
# MANPOWER ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def manpower_engine(severity, event_type, junction, corridor) -> dict:
    """
    Returns personnel, barricades, and deployment positions.
    More granular tiers than Day 1.
    """
    if severity <= 2:
        personnel, barricades = 2, 0
    elif severity <= 4:
        personnel, barricades = 5, 2
    elif severity <= 6:
        personnel, barricades = 10, 5
    elif severity <= 8:
        personnel, barricades = 18, 9
    else:
        personnel, barricades = 30, 15

    deployment = []
    if pd.notna(junction) and junction != "Unknown":
        # Allocate more officers to primary junction for higher severity
        primary_count = max(2, personnel // 2)
        deployment.append(f"{primary_count} officers at {junction}")
        if severity > 5:
            deployment.append(f"{personnel // 4} officers — 500m approach (N/S)")

    if pd.notna(corridor) and str(corridor).strip() not in ("", "Non-corridor"):
        deployment.append(f"Corridor patrol on {corridor}")

    if severity > 7:
        deployment.append("Traffic Control Room — escalate to P1")

    return {
        "personnel":           personnel,
        "barricades":          barricades,
        "deployment_positions": deployment,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI REASONING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
CAUSE_LABELS = {
    "vip_movement":      "VIP movement events",
    "accident":          "Accident incidents",
    "procession":        "Procession events",
    "protest":           "Protest events",
    "public_event":      "Public events",
    "construction":      "Construction works",
    "vehicle_breakdown": "Vehicle breakdown incidents",
    "water_logging":     "Water logging events",
    "congestion":        "Congestion events",
}

PEAK_HOURS = {7, 8, 9, 10, 17, 18, 19, 20}


def generate_reasoning_text(event_cause, junction, hour, severity, junction_history) -> str:
    """
    Produce a 2–3 sentence plain-English explanation of the severity score.
    Computed entirely from real dataset statistics — no LLM needed.
    """
    avg_duration = junction_history.get(junction, 60)
    is_peak      = hour in PEAK_HOURS
    cause_label  = CAUSE_LABELS.get(str(event_cause).lower(), "Traffic events")

    if severity >= 7:
        level = "Critical"
    elif severity >= 4:
        level = "Moderate"
    else:
        level = "Low"

    parts = [f"Rated {level} ({severity}/10)."]

    if avg_duration > 180:
        parts.append(
            f"{junction} has a historical avg resolution of {avg_duration:.0f} min "
            f"— among the highest risk junctions in the dataset."
        )
    elif avg_duration > 90:
        parts.append(
            f"{junction} averages {avg_duration:.0f} min resolution time, "
            f"indicating moderate chronic congestion."
        )
    else:
        parts.append(
            f"{junction} has a relatively low historical avg of {avg_duration:.0f} min."
        )

    if is_peak:
        parts.append(
            f"Incident at {hour:02d}:00 falls within peak traffic hours — "
            f"severity increased by +2 points."
        )

    parts.append(
        f"{cause_label} in this corridor historically require "
        f"{'significant' if severity >= 7 else 'moderate'} resource deployment."
    )

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CORRIDOR RISK
# ─────────────────────────────────────────────────────────────────────────────
def compute_corridor_risk(df: pd.DataFrame) -> dict:
    """Return avg resolution time per corridor (higher = more risk)."""
    if "corridor" not in df.columns or "duration_minutes" not in df.columns:
        return {}
    return (
        df.groupby("corridor")["duration_minutes"]
        .mean()
        .dropna()
        .to_dict()
    )


# ─────────────────────────────────────────────────────────────────────────────
# JUNCTION HISTORY + KMEANS (unchanged from Day 1, kept here for completeness)
# ─────────────────────────────────────────────────────────────────────────────
def compute_junction_history(df: pd.DataFrame) -> dict:
    return df.groupby("junction")["duration_minutes"].mean().to_dict()


def process_kmeans_centers(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    cluster_df = df[["latitude", "longitude", "duration_minutes"]].dropna()
    k = max(1, min(k, len(cluster_df)))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    cluster_df = cluster_df.copy()
    cluster_df["cluster"] = km.fit_predict(cluster_df[["latitude", "longitude"]])
    centers = pd.DataFrame(km.cluster_centers_, columns=["latitude", "longitude"])
    return centers


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED OPERATIONS LAYER
# ─────────────────────────────────────────────────────────────────────────────
def _safe_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "yes", "1", "y"])
    return series.fillna(False).astype(bool)


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.empty:
        return s
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def compute_corridor_vulnerability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank corridors by operational vulnerability.
    Higher score means more fragile and higher-priority for control-room attention.
    """
    if df is None or df.empty or "corridor" not in df.columns:
        return pd.DataFrame(
            columns=[
                "corridor", "incidents", "avg_duration_min", "closure_rate_pct",
                "peak_hour_rate_pct", "cause_diversity", "vulnerability_score",
                "rank", "risk_label", "top_cause"
            ]
        )

    work = df.copy()
    if "hour_of_day" not in work.columns and "start_datetime" in work.columns:
        work["hour_of_day"] = pd.to_datetime(work["start_datetime"], errors="coerce").dt.hour.fillna(0).astype(int)
    if "duration_minutes" not in work.columns:
        work["duration_minutes"] = 60.0
    if "requires_road_closure" in work.columns:
        work["requires_road_closure"] = _safe_bool_series(work["requires_road_closure"])
    else:
        work["requires_road_closure"] = False

    peak_hours = {7, 8, 9, 10, 17, 18, 19, 20}
    grouped = work.groupby("corridor", dropna=False)

    rows = []
    for corridor, g in grouped:
        if pd.isna(corridor) or str(corridor).strip() == "":
            corridor = "Non-corridor"
        incidents = len(g)
        avg_duration = float(g["duration_minutes"].mean()) if incidents else 0.0
        closure_rate = float(g["requires_road_closure"].mean() * 100) if incidents else 0.0
        peak_rate = float(g["hour_of_day"].isin(peak_hours).mean() * 100) if incidents else 0.0
        cause_diversity = float(g["event_cause_category"].nunique() if "event_cause_category" in g.columns else g["event_cause"].nunique())
        top_cause = None
        if "event_cause_category" in g.columns and not g["event_cause_category"].dropna().empty:
            top_cause = g["event_cause_category"].value_counts().idxmax()
        elif "event_cause" in g.columns and not g["event_cause"].dropna().empty:
            top_cause = g["event_cause"].value_counts().idxmax()

        rows.append({
            "corridor": corridor,
            "incidents": incidents,
            "avg_duration_min": avg_duration,
            "closure_rate_pct": closure_rate,
            "peak_hour_rate_pct": peak_rate,
            "cause_diversity": cause_diversity,
            "top_cause": top_cause or "Unknown",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Normalize and score.
    out["incidents_norm"] = _minmax(out["incidents"])
    out["duration_norm"] = _minmax(out["avg_duration_min"])
    out["closure_norm"] = _minmax(out["closure_rate_pct"])
    out["peak_norm"] = _minmax(out["peak_hour_rate_pct"])
    out["cause_norm"] = _minmax(out["cause_diversity"])

    # Balanced control-room weighting: frequency and resolution dominate.
    out["vulnerability_score"] = (
        100
        * (
            0.25 * out["incidents_norm"]
            + 0.30 * out["duration_norm"]
            + 0.20 * out["closure_norm"]
            + 0.15 * out["peak_norm"]
            + 0.10 * out["cause_norm"]
        )
    ).round(1)

    out = out.sort_values(["vulnerability_score", "incidents"], ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["risk_label"] = pd.cut(
        out["vulnerability_score"],
        bins=[-0.1, 33.3, 66.6, 100.0],
        labels=["Low", "Moderate", "High"],
    )
    return out[[
        "rank", "corridor", "incidents", "avg_duration_min", "closure_rate_pct",
        "peak_hour_rate_pct", "cause_diversity", "top_cause", "vulnerability_score",
        "risk_label"
    ]]
def categorize_cause(event_cause: str) -> str:
    """
    Standardizes and groups raw event causes into general categories 
    for historical fallback matching.
    """
    if not event_cause:
        return "unknown"
        
    cause = str(event_cause).lower().strip()
    
    # Maps specific triggers to broader operational categories
    mapping = {
        "vip_movement": "infrastructure_event",
        "procession": "public_event",
        "protest": "public_event",
        "public_event": "public_event",
        "accident": "incident",
        "vehicle_breakdown": "incident",
        "construction": "road_works",
        "water_logging": "weather_hazard",
        "congestion": "heavy_traffic"
    }
    
    return mapping.get(cause, cause)

def _historical_support_score(df: pd.DataFrame, corridor: str, junction: str, event_cause: str, hour: int) -> int:
    if df is None or df.empty:
        return 0

    mask = pd.Series(True, index=df.index)
    if "corridor" in df.columns and pd.notna(corridor):
        mask &= df["corridor"].astype(str).str.lower().eq(str(corridor).lower())
    if "junction" in df.columns and pd.notna(junction):
        mask &= df["junction"].astype(str).str.lower().eq(str(junction).lower())
    if "event_cause" in df.columns and pd.notna(event_cause):
        mask &= df["event_cause"].astype(str).str.lower().eq(str(event_cause).lower())

    count = int(mask.sum())
    if count == 0 and "event_cause_category" in df.columns:
        # Fallback on category-level similarity.
        cat = categorize_cause(event_cause)
        mask = df["event_cause_category"].astype(str).str.lower().eq(str(cat).lower())
        count = int(mask.sum())

    hour_support = 0
    if "hour_of_day" in df.columns:
        hour_support = int(df["hour_of_day"].sub(hour).abs().le(2).sum())

    return count + hour_support


def forecast_traffic_impact(
    event_type: str,
    event_cause: str,
    hour: int,
    junction: str,
    corridor: str,
    crowd_scale: str,
    event_duration_min: float,
    requires_closure: bool,
    junction_history: dict,
    corridor_vulnerability: pd.DataFrame,
    dataset: pd.DataFrame | None = None,
) -> dict:
    """
    Lightweight operational forecast that behaves like a real control-room estimate.
    """
    base = congestion_score(event_type, event_cause, hour, junction, junction_history)
    crowd_multiplier = CROWD_MULTIPLIERS.get(crowd_scale, 1.0)
    corridor_row = None
    vuln_score = 45.0
    corridor_rank = None
    corridor_label = "Moderate"
    if corridor_vulnerability is not None and not corridor_vulnerability.empty:
        match = corridor_vulnerability[corridor_vulnerability["corridor"].astype(str).str.lower() == str(corridor).lower()]
        if not match.empty:
            corridor_row = match.iloc[0]
            vuln_score = float(corridor_row["vulnerability_score"])
            corridor_rank = int(corridor_row["rank"])
            corridor_label = str(corridor_row["risk_label"])

    duration_factor = float(np.clip(event_duration_min / 180.0, 0.25, 3.5))
    vulnerability_factor = vuln_score / 100.0

    severity = base
    severity += int(round((crowd_multiplier - 1.0) * 4))
    severity += int(round(duration_factor * 1.5))
    severity += int(round(vulnerability_factor * 3))
    if requires_closure:
        severity += 1
    if corridor_label == "High":
        severity += 1
    severity = int(np.clip(severity, 0, 10))

    similarity_support = _historical_support_score(dataset, corridor, junction, event_cause, hour)
    support_norm = min(1.0, similarity_support / 25.0)
    if "duration_minutes" in (dataset.columns if dataset is not None and not dataset.empty else []):
        hist_duration = float(dataset["duration_minutes"].mean())
    else:
        hist_duration = 60.0

    expected_delay = (
        10
        + severity * 9.5
        + duration_factor * 14
        + vulnerability_factor * 18
        + (crowd_multiplier - 1.0) * 12
        + (8 if requires_closure else 0)
    )
    expected_delay = float(np.clip(expected_delay, 5, 360))

    affected_radius_km = (
        0.35
        + severity * 0.16
        + vulnerability_factor * 0.7
        + (0.18 if requires_closure else 0.0)
        + (crowd_multiplier - 1.0) * 0.35
    )
    affected_radius_km = float(np.clip(affected_radius_km, 0.25, 6.0))

    estimated_recovery = (
        event_duration_min * 0.72
        + expected_delay * 0.55
        + hist_duration * 0.18
    )
    if requires_closure:
        estimated_recovery *= 1.12
    if corridor_label == "High":
        estimated_recovery *= 1.08
    estimated_recovery = float(np.clip(estimated_recovery, 20, 720))

    confidence = 54 + support_norm * 28 + (0.06 * (10 - severity) * 10 / 10)
    confidence += 4 if event_type.lower() == "planned" else 0
    confidence -= 6 if corridor_label == "High" and support_norm < 0.25 else 0
    confidence = float(np.clip(confidence, 38, 96))

    likely_driver = []
    if hour in PEAK_HOURS:
        likely_driver.append("Peak-hour load")
    if requires_closure:
        likely_driver.append("Road closure spillover")
    if corridor_label == "High":
        likely_driver.append("Corridor vulnerability")
    if crowd_scale in {"Large", "Mega"}:
        likely_driver.append("Crowd pressure")
    if not likely_driver:
        likely_driver.append("Historical traffic pattern")

    return {
        "severity": severity,
        "expected_delay_min": round(expected_delay, 1),
        "affected_radius_km": round(affected_radius_km, 2),
        "estimated_recovery_min": round(estimated_recovery, 1),
        "confidence": round(confidence, 0),
        "corridor_rank": corridor_rank,
        "corridor_vulnerability_score": round(vuln_score, 1),
        "corridor_risk_label": corridor_label,
        "supporting_event_count": int(similarity_support),
        "drivers": likely_driver,
        "crowd_multiplier": crowd_multiplier,
    }


def build_explainability_points(
    forecast: dict,
    event_type: str,
    event_cause: str,
    hour: int,
    junction: str,
    corridor: str,
    requires_closure: bool,
    corridor_vulnerability: pd.DataFrame,
    dataset: pd.DataFrame | None = None,
) -> list[str]:
    points = []
    support = int(forecast.get("supporting_event_count", 0))
    if support > 0:
        points.append(f"Found {support} similar historical events around this corridor/junction pattern.")
    else:
        points.append("Historical support is sparse, so the forecast leans more on rule-based operational patterns.")

    if hour in PEAK_HOURS:
        points.append(f"Hour {hour:02d}:00 sits inside the peak window, which pushes severity and delay upward.")
    else:
        points.append(f"Hour {hour:02d}:00 is outside the strongest peak window, so the forecast is slightly softer.")

    if requires_closure:
        points.append("Road closure is the main spillover risk; it increases queue length and recovery time.")
    else:
        points.append("No road closure requested, which keeps the recovery curve flatter.")

    if corridor_vulnerability is not None and not corridor_vulnerability.empty:
        match = corridor_vulnerability[corridor_vulnerability["corridor"].astype(str).str.lower() == str(corridor).lower()]
        if not match.empty:
            row = match.iloc[0]
            points.append(
                f"{corridor} ranks #{int(row['rank'])} on vulnerability with a {float(row['vulnerability_score']):.1f}/100 score."
            )
            points.append(f"The corridor is historically driven by {row['top_cause']} patterns.")
        else:
            points.append(f"{corridor} has limited historical volume, so the system uses nearby corridor behaviour as a proxy.")

    cause_label = CAUSE_LABELS.get(str(event_cause).lower(), "Traffic events")
    if str(event_cause).lower() in CAUSE_LABELS:
        points.append(f"{cause_label} usually create a repeatable resource pattern in the control room.")
    else:
        points.append("Cause-specific data is weak, so the model is relying on broader traffic signatures.")

    return points[:5]


def build_incident_timeline(forecast: dict, resources: dict) -> list[dict]:
    severity = int(forecast.get("severity", 5))
    clearance = float(forecast.get("estimated_recovery_min", 60))
    officers = int(resources.get("personnel", 2))
    barricades = int(resources.get("barricades", 0))

    dispatch = max(4, round(4 + severity * 0.7))
    barricade = dispatch + max(6, round(5 + barricades * 0.35))
    diversion = barricade + max(5, round(4 + severity * 0.35))
    clearance_min = max(diversion + 20, round(clearance))

    return [
        {
            "step": "Incident detected",
            "minute": 0,
            "detail": "Event intake confirmed in the command room.",
        },
        {
            "step": "Officer dispatch",
            "minute": dispatch,
            "detail": f"Deploy first wave of {officers} officers and verify junction control points.",
        },
        {
            "step": "Barricade setup",
            "minute": barricade,
            "detail": f"Place {barricades} barricades and secure diversion entry lanes.",
        },
        {
            "step": "Diversion active",
            "minute": diversion,
            "detail": "Traffic diversion goes live and route guidance is pushed to field teams.",
        },
        {
            "step": "Expected clearance",
            "minute": clearance_min,
            "detail": "Backlog should normalise if conditions stay within forecasted bounds.",
        },
    ]


def resource_optimization(forecast: dict, corridor_vulnerability: pd.DataFrame) -> dict:
    """
    Return minimum safe, recommended, and aggressive plans.
    """
    severity = int(forecast.get("severity", 5))
    delay = float(forecast.get("expected_delay_min", 60))
    radius = float(forecast.get("affected_radius_km", 1.0))
    corridor_score = float(forecast.get("corridor_vulnerability_score", 45.0))
    closure_risk = 1 if forecast.get("corridor_risk_label", "Moderate") == "High" else 0

    base_officers = max(2, round(2 + severity * 1.7 + radius * 1.5 + closure_risk))
    base_barricades = max(0, round(severity * 0.9 + radius * 1.2 + closure_risk * 2))

    plan_defs = {
        "Minimum safe": {
            "officers": max(2, round(base_officers * 0.75)),
            "barricades": max(0, round(base_barricades * 0.7)),
            "recovery_mult": 1.08,
            "severity_delta": +0.3,
            "delay_delta": +9,
            "tone": "Just enough to hold the line.",
        },
        "Recommended": {
            "officers": base_officers,
            "barricades": base_barricades,
            "recovery_mult": 0.92,
            "severity_delta": -0.2,
            "delay_delta": -8,
            "tone": "Balanced control-room default.",
        },
        "Aggressive": {
            "officers": max(base_officers + 3, round(base_officers * 1.35)),
            "barricades": max(base_barricades + 2, round(base_barricades * 1.45)),
            "recovery_mult": 0.78,
            "severity_delta": -0.6,
            "delay_delta": -18,
            "tone": "Heavy intervention to crush spillover fast.",
        },
    }

    plans = {}
    for name, cfg in plan_defs.items():
        plans[name] = {
            "officers": int(cfg["officers"]),
            "barricades": int(cfg["barricades"]),
            "expected_recovery_min": round(max(15, delay + (forecast.get("estimated_recovery_min", 60) - delay) * cfg["recovery_mult"]), 1),
            "severity_effect": round(np.clip(severity + cfg["severity_delta"], 0, 10), 1),
            "delay_effect": round(max(0, delay + cfg["delay_delta"]), 1),
            "tone": cfg["tone"],
            "corridor_score": corridor_score,
        }

    return plans


def scenario_delta(base_forecast: dict, sim_forecast: dict) -> dict:
    return {
        "severity": round(sim_forecast["severity"] - base_forecast["severity"], 1),
        "delay_min": round(sim_forecast["expected_delay_min"] - base_forecast["expected_delay_min"], 1),
        "radius_km": round(sim_forecast["affected_radius_km"] - base_forecast["affected_radius_km"], 2),
        "recovery_min": round(sim_forecast["estimated_recovery_min"] - base_forecast["estimated_recovery_min"], 1),
        "confidence": round(sim_forecast["confidence"] - base_forecast["confidence"], 1),
    }
