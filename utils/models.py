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
