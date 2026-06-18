from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium
# ── WORKAROUND FOR HTTPX 0.28+ COMPATIBILITY ISSUE ────────────────────────────
import groq
from groq._base_client import SyncHttpxClientWrapper

class CustomHttpxClientWrapper(SyncHttpxClientWrapper):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)  # Strip out the unsupported argument safely
        super().__init__(*args, **kwargs)

# Inject the patch into the groq base module before instantiation
groq._base_client.SyncHttpxClientWrapper = CustomHttpxClientWrapper
# ──────────────────────────────────────────────────────────────────────────────

# Now initialize your client as normal
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

from utils.data_processor import compute_kpi_stats, engineer_features, load_and_clean_data
from utils.models import (
    DIVERSION_ROUTES,
    JUNCTION_COORDS,
    build_explainability_points,
    build_incident_timeline,
    compute_corridor_vulnerability,
    compute_junction_history,
    forecast_traffic_impact,
    generate_reasoning_text,
    get_diversion_route,
    get_junction_coords,
    manpower_engine,
    process_kmeans_centers,
    resource_optimization,
    scenario_delta,
    whatif_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gridlock Hackathon— Traffic Command Center Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path("assets/custom.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)




# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEV_COLOR = {"Critical": "#f60000", "Moderate": "#f59e0b", "Low": "#08fc61"}
KNOWN_JUNCTIONS = [j for j in JUNCTION_COORDS if j != "Unknown"]
KNOWN_CORRIDORS = [c for c in DIVERSION_ROUTES if c != "Non-corridor"]
CAUSES = [
    "vehicle_breakdown", "accident", "vip_movement", "water_logging",
    "procession", "protest", "public_event", "construction", "congestion",
]
CROWD_OPTIONS = ["Small", "Medium", "Large", "Mega"]

def sev_label(s: int) -> str:
    return "Critical" if s >= 7 else "Moderate" if s >= 4 else "Low"

def fmt_min(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.0f} min"

def fmt_km(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.2f} km"

def delta_arrow(d: float) -> str:
    return f"▲ +{d:.1f}" if d > 0 else f"▼ {d:.1f}" if d < 0 else "— 0"

def delta_color(d: float, invert: bool = False) -> str:
    bad = "#ff0000"; good = "#03fe5f"
    if d == 0: return "#94a3b8"
    positive_is_bad = not invert
    return (bad if d > 0 else good) if positive_is_bad else (good if d > 0 else bad)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE  (cached — runs once, reuses on rerun)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading historical dataset…")
def load_pipeline():
    # Try to load the real CSV; fall back to synthetic demo data gracefully.
    try:
        df = load_and_clean_data("Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")
        if df is None or df.empty:
            raise ValueError("empty")
        df = engineer_features(df)
    except Exception:
        rng = np.random.default_rng(42)
        n = 400
        junctions  = KNOWN_JUNCTIONS[:8]
        corridors  = KNOWN_CORRIDORS[:6]
        causes     = CAUSES
        starts     = pd.date_range("2024-01-01", periods=n, freq="3h")
        durations  = rng.exponential(scale=80, size=n).clip(10, 600)
        df = pd.DataFrame({
            "event_type":            rng.choice(["unplanned", "planned"], n, p=[0.75, 0.25]),
            "start_datetime":        starts,
            "end_datetime":          starts + pd.to_timedelta(durations, unit="m"),
            "latitude":              rng.uniform(12.86, 13.10, n),
            "longitude":             rng.uniform(77.50, 77.75, n),
            "junction":              rng.choice(junctions, n),
            "corridor":              rng.choice(corridors, n),
            "event_cause":           rng.choice(causes, n),
            "priority":              rng.choice(["P1", "P2", "P3"], n),
            "requires_road_closure": rng.choice([True, False], n, p=[0.30, 0.70]),
            "status":                rng.choice(["resolved", "active", "pending"], n, p=[0.70, 0.15, 0.15]),
        })
        df = engineer_features(df)

    jh   = compute_junction_history(df)
    cv   = compute_corridor_vulnerability(df)
    kpi  = compute_kpi_stats(df)
    clus = process_kmeans_centers(df, k=10)
    return df, jh, cv, kpi, clus

df, junction_history, corridor_vuln, kpi_stats, cluster_centers = load_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — COMMAND TOWER OPERATIONS CONTROL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── HIGH-END COMMAND CENTER BRANDING BANNER ──
    st.markdown("""
        <div style="padding: 20px 16px; background: linear-gradient(135deg, rgba(15,23,42,0.6) 0%, rgba(30,41,59,0.9) 100%); border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 20px; border-left: 5px solid #E4580B; box-shadow: 0 4px 25px rgba(0,0,0,0.4);">
            <h2 style="margin:0; font-size:22px; color:#ffffff; font-weight:800; letter-spacing:-0.5px; display:flex; align-items:center; gap:10px;">
                 GRIDLOCK
            </h2>
            <div style="font-size:10.5px; color:#38bdf8; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-top:4px;">EventFlow Copilot • Traffic Command Center</div>
        </div>
    """, unsafe_allow_html=True)

    # ── GROUP 1: SPATIAL GRID COORDINATES ──
    st.markdown('<p style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px; margin-top:4px;"> Spatial Grid Coordinates</p>', unsafe_allow_html=True)
    
    sel_junction = st.selectbox("Target Node Junction", KNOWN_JUNCTIONS, index=0)
    sel_corridor = st.selectbox("Primary Impact Corridor", KNOWN_CORRIDORS, index=0)

    st.markdown("<div style='margin: 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)

    # ── GROUP 2: OPERATIONAL PROFILE ──
    st.markdown('<p style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;"> Operational Profile</p>', unsafe_allow_html=True)
    
    sel_cause = st.selectbox("Root Event Cause", CAUSES, index=0)
    sel_type = st.radio("Vector Class", ["unplanned", "planned"], horizontal=True)
    sel_duration = st.number_input("Target Window Horizon (min)", min_value=5, max_value=480, value=60, step=5)

    st.markdown("<div style='margin: 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)

    # ── GROUP 3: CONTEXTUAL CONSTRAINTS ──
    st.markdown('<p style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">Contextual Constraints</p>', unsafe_allow_html=True)
    
    sel_hour = st.slider("Timeline Operational Hour", 0, 23, datetime.now().hour)
    sel_crowd = st.select_slider("Simulated Scale Constraints", CROWD_OPTIONS, value="Medium")
    sel_closure = st.toggle("Enforce Structural Road Closure", value=False)

    # ── TACTICAL FOOTER DIAGNOSTIC BADGE ──
    st.markdown(f"""
        <div style="margin-top: 35px; padding: 12px; background: rgba(0, 0, 0, 0.2); border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); display: flex; flex-direction: column; gap: 4px;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; color: #94a3b8; font-weight: 500;">
                <span style="display:inline-block; width:6px; height:6px; background:#22c55e; border-radius:50%; box-shadow:0 0 6px #22c55e;"></span>
                Telemetry Database Status
            </div>
            <div style="font-size: 12px; color: #f1f5f9; font-weight: 600; padding-left: 12px;">
                 {len(df):,} Historical Logs Mounted
            </div>
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORE COMPUTATION — runs on every sidebar change
# ─────────────────────────────────────────────────────────────────────────────
forecast = forecast_traffic_impact(
    event_type=sel_type,
    event_cause=sel_cause,
    hour=sel_hour,
    junction=sel_junction,
    corridor=sel_corridor,
    crowd_scale=sel_crowd,
    event_duration_min=float(sel_duration),
    requires_closure=sel_closure,
    junction_history=junction_history,
    corridor_vulnerability=corridor_vuln,
    dataset=df,
)

sev_num   = forecast["severity"]
sev_lbl   = sev_label(sev_num)
sev_clr   = SEV_COLOR[sev_lbl]

resources = manpower_engine(sev_num, sel_type, sel_junction, sel_corridor)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="top-header">
        <div>
            <span class="header-title">🚦 Gridlock</span>
            <span class="header-sub">Predictive Incident Command — Bengaluru</span>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            <span class="live-badge"><span class="live-dot"></span>LIVE</span>
            <span class="sev-badge" style="background:{sev_clr}22;color:{sev_clr};border:1px solid {sev_clr}44;">
                {sev_lbl} — {sev_num}/10
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpi_defs = [
    (k1, "Severity Index",       f"{sev_num}/10",                          sev_clr),
    (k2, "Expected Delay",        fmt_min(forecast["expected_delay_min"]),   "#38bdf8"),
    (k3, "Impact Radius",         fmt_km(forecast["affected_radius_km"]),    "#a78bfa"),
    (k4, "Est. Recovery",         fmt_min(forecast["estimated_recovery_min"]),"#22c55e"),
    (k5, "Forecast Confidence",   f"{forecast['confidence']:.0f}%",          "#f59e0b"),
]
for col, label, value, color in kpi_defs:
    with col:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{color};">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin:1rem 0 0.5rem;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS — main navigation
# ─────────────────────────────────────────────────────────────────────────────
tab_map, tab_forecast, tab_whatif, tab_response, tab_corridors,tab_copilot = st.tabs([
    "  Live Map",
    "  Forecast & Intel",
    "  What-If Simulator",
    "  Response Plan",
    "  Corridor Intelligence",
    "   AI Copilot ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MAP
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MAP & RISK DIAGNOSTICS MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    # Lift lookup to the top so both map and analytics columns share the same route instance safely
    route = get_diversion_route(sel_corridor)
    
    col_map, col_feed = st.columns([6, 4], gap="medium")

    # ══════════════════════════════════════════════════════════════════════════
    # COLUMN 1: LIVE GEOSPATIAL INTELLIGENCE MAP (UPDATED V2)
    # ══════════════════════════════════════════════════════════════════════════
    with col_map:
        st.markdown('<div class="section-label" style="margin-bottom: 12px;">Live Geospatial Risk Map</div>', unsafe_allow_html=True)

        jlat, jlon = get_junction_coords(sel_junction)
        
        # Build map with constrained, precise relative dimensions
        m = folium.Map(
            location=[jlat, jlon], 
            zoom_start=14,  # Pulled in slightly closer for sharper local perspective
            tiles="CartoDB dark_matter",
            zoom_control=True,
            scrollWheelZoom=False
        )

        # ── 1. INJECT CUSTOM CSS FOR SYSTEM RADAR PULSE CRITICAL NODE ──
        pulse_animation_css = f"""
        <style>
            @keyframes tactical-pulse {{
                0% {{ transform: scale(0.6); opacity: 1; }}
                50% {{ opacity: 0.4; }}
                100% {{ transform: scale(2.8); opacity: 0; }}
            }}
            .epicenter-container {{
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .core-node {{
                width: 14px;
                height: 14px;
                background-color: {sev_clr};
                border: 2px solid #ffffff;
                border-radius: 50%;
                box-shadow: 0 0 14px {sev_clr};
                z-index: 10;
            }}
            .pulse-ring {{
                position: absolute;
                width: 32px;
                height: 32px;
                border: 2.5px solid {sev_clr};
                border-radius: 50%;
                animation: tactical-pulse 2s infinite cubic-bezier(0.215, 0.610, 0.355, 1);
                opacity: 0;
                z-index: 1;
            }}
        </style>
        """
        m.get_root().header.add_child(folium.Element(pulse_animation_css))

        # ── 2. HISTORICAL HEATMAP (REDUCED OPACITY FOR HIGHER CONTRAST) ──
        heat_pts = (
            df[["latitude", "longitude", "duration_minutes"]].dropna()
            .assign(w=lambda x: x["duration_minutes"].clip(0, 400) / 400)
            [["latitude", "longitude", "w"]].values.tolist()
        )
        if heat_pts:
            HeatMap(
                heat_pts, 
                radius=16, 
                blur=20,
                max_opacity=0.38,  # Dialed down to let vectors and secondary indicators pop
                gradient={"0.2": "#1d4ed8", "0.45": "#f59e0b", "0.8": "#dc2626"}
            ).add_to(m)

        # KMeans cluster points 
        for _, row in cluster_centers.iterrows():
            if pd.notna(row.latitude) and pd.notna(row.longitude):
                folium.CircleMarker(
                    location=[row.latitude, row.longitude],
                    radius=5, 
                    color="rgba(167, 139, 250, 0.4)",
                    fill=True, 
                    fill_color="#a78bfa", 
                    fill_opacity=0.3,
                    tooltip="Historical incident node proximity anchor",
                ).add_to(m)

        # ── 3. TRIPLE TRAFFIC SHOCKWAVE PROPAGATION RINGS ──
        base_radius_meters = max(forecast["affected_radius_km"] * 1000, 150.0)
        
        # Inner Shockwave (High Density Core)
        folium.Circle(
            location=[jlat, jlon],
            radius=base_radius_meters * 0.35,
            color=sev_clr,
            weight=1.0,
            fill=True,
            fill_color=sev_clr,
            fill_opacity=0.12,
        ).add_to(m)
        
        # Mid Shockwave (Spillover Buffer Zone)
        folium.Circle(
            location=[jlat, jlon],
            radius=base_radius_meters * 0.68,
            color=sev_clr,
            weight=1.5,
            fill=False,
            opacity=0.4
        ).add_to(m)
        
        # Outer Shockwave Edge (Maximum Kinetic Threshold Perimeter)
        folium.Circle(
            location=[jlat, jlon],
            radius=base_radius_meters,
            color=sev_clr,
            weight=2.0,
            dash_array="5 6",
            fill=False,
            opacity=0.22,
            tooltip=f"Terminal impact horizon: {forecast['affected_radius_km']:.2f} km",
        ).add_to(m)

        # ── 4. PULSING HARDWARE EPICENTER NODE MARKER ──
        folium.Marker(
            location=[jlat, jlon],
            icon=folium.DivIcon(
                html=f'<div class="epicenter-container"><div class="core-node"></div><div class="pulse-ring"></div></div>',
                icon_size=(32, 32),
                icon_anchor=(16, 16)
            ),
            popup=folium.Popup(
                f"<div style='font-family:sans-serif;font-size:12px;color:#1e293b;padding:2px;'>\n"
                f"<b> Active Node:</b> {sel_junction}<br>\n"
                f"<b>Cause profile:</b> {sel_cause.replace('_',' ').title()}<br>\n"
                f"<b>Computed Index:</b> {sev_num}/10 ({sev_lbl})\n"
                f"</div>", max_width=220
            ),
            tooltip=f"CRITICAL SYSTEM EPICENTER: {sel_junction}"
        ).add_to(m)

        # Diversion route handling vectors
        route = get_diversion_route(sel_corridor)
        if route.get("primary_route"):
            folium.PolyLine(
                locations=route["primary_route"],
                color="#10b981", 
                weight=4, 
                opacity=0.85,
                dash_array="6 6",
                tooltip=f"Active diversion channel: {route['name']}",
            ).add_to(m)
            
        for i, bp in enumerate(route.get("barricades", []), 1):
            folium.Marker(
                location=bp,
                icon=folium.DivIcon(
                    html='<div style="font-size:16px; text-shadow:0 0 6px rgba(245,158,11,0.5); transform:translate(-2px,-4px);">🚧</div>'
                ),
                tooltip=f"Barricade Intercept Node {i}",
            ).add_to(m)

        # ── 5. CLEAN WRAPPER TO COMPRESS VERTICAL BLANK SPACE ──
        map_html = m._repr_html_()
        styled_iframe_content = f"""
        <style>
            html, body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; height: 100%; }}
            .folium-map {{ border-radius: 12px !important; overflow: hidden !important; box-shadow: 0 12px 24px rgba(0,0,0,0.4); }}
            .leaflet-control-attribution {{ background: rgba(15,23,42,0.85) !important; color:#475569 !important; font-size:9px !important; }}
            .leaflet-control-attribution a {{ color: #38bdf8 !important; }}
        </style>
        {map_html}
        """
        
        # Reduced height profile down from 500/535 to 445 to perfectly align columns 
        # without overflowing into unseemly empty bottom regions
        import streamlit.components.v1 as components
        components.html(styled_iframe_content, height=445, scrolling=False)


    
    # ── COLUMN 2: AUTOMATED DIAGNOSTICS & ANALYTICS CARDS ─────────────────────
    with col_feed:
        
        # ── CARD 1: AI Engine Diagnostics ──
        reasoning = generate_reasoning_text(
            sel_cause, sel_junction, sel_hour, sev_num, junction_history
        )
        st.markdown(f"""
<div class="ui-card">
<h4 style="margin-top:0; margin-bottom:12px; color:#38bdf8; display:flex; align-items:center; gap:8px; font-size:16px;">
 AI Engine Diagnostics
</h4>
<p style="font-size:14px; line-height:1.5; color:#f1f5f9; margin-bottom:0;">
{reasoning}
</p>
</div>
        """, unsafe_allow_html=True)

        # ── CARD 2: Forecast Drivers ──
        drivers_html = "".join([f'<span class="driver-tag" style="display:inline-block; margin-right:6px; margin-bottom:6px; background:rgba(251,146,60,0.15); color:#fb923c; border:1px solid rgba(251,146,60,0.3); padding:4px 10px; border-radius:8px; font-size:12px; font-weight:500;">{driver}</span>' for driver in forecast.get("drivers", [])])
        if not drivers_html:
            drivers_html = '<span style="color:#94a3b8; font-size:13px; font-style:italic;">No anomalous drivers triggered.</span>'
            
        st.markdown(f"""
<div class="ui-card">
<h4 style="margin-top:0; margin-bottom:12px; color:#fb923c; display:flex; align-items:center; gap:8px; font-size:16px;">
 Primary Forecast Drivers
</h4>
<div style="display:flex; flex-wrap:wrap; gap:4px;">
{drivers_html}
</div>
</div>
        """, unsafe_allow_html=True)

        # ── CARD 3: Explainability & Core Logic ──
        expl_points = build_explainability_points(
            forecast=forecast,
            event_type=sel_type,
            event_cause=sel_cause,
            hour=sel_hour,
            junction=sel_junction,
            corridor=sel_corridor,
            requires_closure=sel_closure,
            corridor_vulnerability=corridor_vuln,
            dataset=df,
        )
        
        # Built as an explicit flat one-liner string string to block markdown code conversions
        pts_html = ""
        for i, pt in enumerate(expl_points, 1):
            pts_html += f'<div style="display:flex; gap:12px; margin-bottom:10px; align-items:start;"><span style="background:#34d399; color:#0f172a; font-weight:700; font-size:11px; width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px;">{i}</span><span style="font-size:13.5px; color:#e2e8f0; line-height:1.4;">{pt}</span></div>'
            
        if not pts_html:
            pts_html = '<div style="color:#94a3b8; font-size:13px; font-style:italic;">No telemetry log explainability items computed.</div>'

        st.markdown(f"""
<div class="ui-card">
<h4 style="margin-top:0; margin-bottom:14px; color:#34d399; display:flex; align-items:center; gap:8px; font-size:16px;">
 Explainability Logs & Logic
</h4>
{pts_html}
</div>
        """, unsafe_allow_html=True)

        # ── CARD 4: Tactical Diversion Protocol ──
        alt_route_html = f'<div style="font-size:12px; color:#cbd5e1; margin-top:4px;"><b>Alternative:</b> {route["alternate_name"]}</div>' if route.get("alternate_name") else ""
        st.markdown(f"""
<div class="ui-card" style="margin-bottom:0px;">
<h4 style="margin-top:0; margin-bottom:12px; color:#a78bfa; display:flex; align-items:center; gap:8px; font-size:16px;">
 Tactical Diversion Protocol
</h4>
<div style="background:rgba(0,0,0,0.25); padding:12px 14px; border-radius:12px; border:1px solid rgba(255,255,255,0.05); margin-bottom:10px;">
<span style="font-size:11px; text-transform:uppercase; color:#c084fc; font-weight:600; display:block; letter-spacing:0.05em; margin-bottom:2px;">Bypass Vector Direction</span>
<span style="font-size:15px; font-weight:600; color:#ffffff;">{route.get("name","—")}</span>
{alt_route_html}
</div>
<div style="display:flex; align-items:center; gap:8px; color:#e2e8f0; font-size:13.5px; background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.02);">
<span style="font-size:16px;"></span> <span>Active deployment contains <b>{len(route.get("barricades",[]))}</b> perimeter checkpoints.</span>
</div>
</div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECAST & INTEL (VISUALLY APPEALING METRIC CARDS)
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="section-label"> Traffic Impact Forecast Matrix</div>', unsafe_allow_html=True)
    st.caption(
        f"Based on {forecast['supporting_event_count']} similar historical events · "
        f"Corridor vulnerability: {forecast['corridor_vulnerability_score']:.1f}/100 "
        f"({forecast['corridor_risk_label']})"
    )

    # ── METRIC CORE: 5 DISTINCT HORIZONTAL CARDS ─────────────────────────────
    fm1, fm2, fm3, fm4, fm5 = st.columns(5)

    # Card 1: Severity Score
    with fm1:
        st.markdown(f"""
<div class="ui-card" style="text-align: center; padding: 20px 10px !important; height: 100%;">
<div style="font-size: 26px; margin-bottom: 4px;">🚨</div>
<div style="text-transform: uppercase; font-size: 11px; color: #94a3b8; letter-spacing: 0.05em; font-weight: 600;">Severity Score</div>
<div style="font-size: 24px; font-weight: 700; color: {sev_clr}; margin-top: 4px;">{sev_num} <span style="font-size:14px; color:#94a3b8;">/ 10</span></div>
<div style="font-size: 11px; color: #94a3b8; margin-top: 4px; line-height: 1.2;">Rule engine + corridor history</div>
</div>
        """, unsafe_allow_html=True)

    # Card 2: Expected Delay
    with fm2:
        st.markdown(f"""
<div class="ui-card" style="text-align: center; padding: 20px 10px !important; height: 100%;">
<div style="font-size: 26px; margin-bottom: 4px;">⏱️</div>
<div style="text-transform: uppercase; font-size: 11px; color: #94a3b8; letter-spacing: 0.05em; font-weight: 600;">Expected Delay</div>
<div style="font-size: 24px; font-weight: 700; color: #38bdf8; margin-top: 4px;">{fmt_min(forecast["expected_delay_min"])}</div>
<div style="font-size: 11px; color: #94a3b8; margin-top: 4px; line-height: 1.2;">Queue propagation model</div>
</div>
        """, unsafe_allow_html=True)

    # Card 3: Impact Radius
    with fm3:
        st.markdown(f"""
<div class="ui-card" style="text-align: center; padding: 20px 10px !important; height: 100%;">
<div style="font-size: 26px; margin-bottom: 4px;">📍</div>
<div style="text-transform: uppercase; font-size: 11px; color: #94a3b8; letter-spacing: 0.05em; font-weight: 600;">Impact Radius</div>
<div style="font-size: 24px; font-weight: 700; color: #fb923c; margin-top: 4px;">{fmt_km(forecast["affected_radius_km"])}</div>
<div style="font-size: 11px; color: #94a3b8; margin-top: 4px; line-height: 1.2;">Spatial spillover horizon</div>
</div>
        """, unsafe_allow_html=True)

    # Card 4: Recovery Time
    with fm4:
        st.markdown(f"""
<div class="ui-card" style="text-align: center; padding: 20px 10px !important; height: 100%;">
<div style="font-size: 26px; margin-bottom: 4px;">🔄</div>
<div style="text-transform: uppercase; font-size: 11px; color: #94a3b8; letter-spacing: 0.05em; font-weight: 600;">Recovery Time</div>
<div style="font-size: 24px; font-weight: 700; color: #34d399; margin-top: 4px;">{fmt_min(forecast["estimated_recovery_min"])}</div>
<div style="font-size: 11px; color: #94a3b8; margin-top: 4px; line-height: 1.2;">Historical resolution patterns</div>
</div>
        """, unsafe_allow_html=True)

    # Card 5: Model Confidence
    with fm5:
        st.markdown(f"""
<div class="ui-card" style="text-align: center; padding: 20px 10px !important; height: 100%;">
<div style="font-size: 26px; margin-bottom: 4px;">🛡️</div>
<div style="text-transform: uppercase; font-size: 11px; color: #94a3b8; letter-spacing: 0.05em; font-weight: 600;">Confidence</div>
<div style="font-size: 24px; font-weight: 700; color: #a78bfa; margin-top: 4px;">{forecast['confidence']:.0f}%</div>
<div style="font-size: 11px; color: #c084fc; margin-top: 4px; line-height: 1.2;">n={forecast['supporting_event_count']} matching cases</div>
</div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1.2rem 0;'></div>", unsafe_allow_html=True)

    # ── GAUGES MATRIX ────────────────────────────────────────────────────────
    g_col, c_col = st.columns([1, 1])
    with g_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sev_num,
            title={"text": "Severity Score", "font": {"size": 14, "color": "#c5c7c9"}},
            gauge={
                "axis": {"range": [0, 10], "tickcolor": "#b5b8bb"},
                "bar":  {"color": sev_clr},
                "steps": [
                    {"range": [0, 4],  "color": "#052e16"},
                    {"range": [4, 7],  "color": "#422006"},
                    {"range": [7, 10], "color": "#450a0a"},
                ],
            },
            number={"suffix": "/10", "font": {"size": 28, "color": "#f1f5f9"}},
        ))
        fig_gauge.update_layout(
            height=240,
            margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with c_col:
        conf = forecast["confidence"]
        fig_conf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf,
            title={"text": "Forecast Confidence", "font": {"size": 14, "color": "#c3c4c7"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#c1c1c1"},
                "bar":  {"color": "#f59e0b"},
                "steps": [
                    {"range": [0,  40], "color": "#1c0a00"},
                    {"range": [40, 70], "color": "#1c1400"},
                    {"range": [70,100], "color": "#0a1c00"},
                ],
            },
            number={"suffix": "%", "font": {"size": 28, "color": "#f1f5f9"}},
        ))
        fig_conf.update_layout(
            height=240,
            margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cccdcd",
        )
        st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})

    # ── EXPLAINABILITY ENGINE FULL LOG PANEL ─────────────────────────────────
    st.markdown('<div class="section-label"> Why This Forecast</div>', unsafe_allow_html=True)
    ep_cols = st.columns(2)
    
    expl_pts_list = expl_points if 'expl_points' in dir() else build_explainability_points(
        forecast=forecast, event_type=sel_type, event_cause=sel_cause, hour=sel_hour,
        junction=sel_junction, corridor=sel_corridor, requires_closure=sel_closure,
        corridor_vulnerability=corridor_vuln, dataset=df,
    )
    
    for i, pt in enumerate(expl_pts_list):
        with ep_cols[i % 2]:
            st.markdown(f"""
<div class="expl-card" style="margin-bottom:12px;">
<span class="expl-num-lg">{i+1}</span>{pt}
</div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF SCENARIO SIMULATOR & VECTOR DELTAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_whatif:
    st.markdown('<div class="section-label"> What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.caption("Manipulate ambient vectors below to model real-time queue shifts and recovery path disruptions against baseline operations.")

    # ── CONTROL PANEL CONTROLS MATRIX ────────────────────────────────────────
    wi_c1, wi_c2 = st.columns(2, gap="medium")
    with wi_c1:
        wi_cause = st.selectbox("Simulated Incident Cause", CAUSES, index=CAUSES.index(sel_cause), key="wi_cause")
        wi_type = st.radio("Simulated Vector Category", ["unplanned", "planned"], horizontal=True, key="wi_type", index=0 if sel_type == "unplanned" else 1)
        wi_hour = st.slider("Simulated Operational Hour", 0, 23, sel_hour, key="wi_hour")
    with wi_c2:
        wi_crowd = st.select_slider("Simulated Scale Constraints", CROWD_OPTIONS, value=sel_crowd, key="wi_crowd")
        wi_duration = st.number_input("Simulated Core Duration (min)", 5, 480, int(sel_duration), step=5, key="wi_dur")
        wi_closure = st.toggle("Simulated Structural Road Closure", value=sel_closure, key="wi_closure")

    # Compute simulated telemetry models
    sim_forecast = forecast_traffic_impact(
        event_type=wi_type,
        event_cause=wi_cause,
        hour=wi_hour,
        junction=sel_junction,
        corridor=sel_corridor,
        crowd_scale=wi_crowd,
        event_duration_min=float(wi_duration),
        requires_closure=wi_closure,
        junction_history=junction_history,
        corridor_vulnerability=corridor_vuln,
        dataset=df,
    )

    delta = scenario_delta(forecast, sim_forecast)
    sim_sev_lbl = sev_label(sim_forecast["severity"])
    sim_sev_clr = SEV_COLOR[sim_sev_lbl]

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label"> Tactical Plan Comparison Matrix</div>', unsafe_allow_html=True)

    # Assemble normalized metrics structures
    metrics_compare = [
        ("Severity Index", f"{sev_num}/10 ({sev_lbl})", f"{sim_forecast['severity']}/10 ({sim_sev_lbl})", delta["severity"], False, "🚨"),
        ("Expected Queue Delay", fmt_min(forecast["expected_delay_min"]), fmt_min(sim_forecast["expected_delay_min"]), delta["delay_min"], False, "⏱️"),
        ("Spatial Impact Radius", fmt_km(forecast["affected_radius_km"]), fmt_km(sim_forecast["affected_radius_km"]), delta["radius_km"], False, "📍"),
        ("System Recovery Window", fmt_min(forecast["estimated_recovery_min"]), fmt_min(sim_forecast["estimated_recovery_min"]), delta["recovery_min"], False, "🔄"),
        ("Model Forecast Confidence", f"{forecast['confidence']:.0f}%", f"{sim_forecast['confidence']:.0f}%", delta["confidence"], True, "🛡️"),
    ]

    # ── COMPARISON LEDGER GENERATOR ──
    # Formatted strictly flat as an inline flex structure to protect layout conversions
    ledger_html = ""
    for metric, base_v, sim_v, d, higher_is_good, icon in metrics_compare:
        arr_icon = delta_arrow(d)
        arr_color = delta_color(d, invert=higher_is_good)
        sim_val_style = f"color:{sim_sev_clr}; font-weight:700;" if metric == "Severity Index" else "color:#f1f5f9; font-weight:600;"
        
        ledger_html += f'<div style="display:flex; justify-content:between; align-items:center; padding:12px 16px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:10px; margin-bottom:8px;"><div style="width:28%; display:flex; align-items:center; gap:8px; font-size:14px; color:#cbd5e1; font-weight:500;"><span>{icon}</span>{metric}</div><div style="width:28%; font-size:14px; color:#94a3b8;">{base_v}</div><div style="width:28%; font-size:14px; {sim_val_style}">{sim_v}</div><div style="width:16%; text-align:right; font-size:14px; color:{arr_color}; font-weight:700; letter-spacing:0.02em;">{arr_icon}</div></div>'

    st.markdown(f"""
<div class="ui-card" style="padding:16px !important; margin-bottom:20px;">
<div style="display:flex; justify-content:between; padding:0px 16px 12px 16px; border-bottom:1px solid rgba(255,255,255,0.1); margin-bottom:10px; font-size:12px; font-weight:600; text-transform:uppercase; color:#94a3b8; letter-spacing:0.05em;">
<div style="width:28%;">Telemetry Vector</div>
<div style="width:28%;">Baseline Benchmark</div>
<div style="width:28%;">Simulated Curve</div>
<div style="width:16%; text-align:right;">Net Shift</div>
</div>
{ledger_html}
</div>
    """, unsafe_allow_html=True)

    # ── RESOURCE PLANNING DEPLOYMENT CAPACITIES ──────────────────────────────
    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label"> Resource Reallocation Profiler</div>', unsafe_allow_html=True)
    
    sim_resources = manpower_engine(sim_forecast["severity"], wi_type, sel_junction, sel_corridor)
    base_p = resources["personnel"]; sim_p = sim_resources["personnel"]
    base_b = resources["barricades"]; sim_b = sim_resources["barricades"]

    mr1, mr2 = st.columns(2, gap="medium")
    
    with mr1:
        st.markdown(f"""
<div class="ui-card" style="border-left: 5px solid #3b82f6 !important; height: 100%;">
<div style="text-transform: uppercase; font-size: 11px; color: #3b82f6; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 8px;">Baseline Response Footprint</div>
<div style="font-size: 20px; font-weight: 600; color: #ffffff; margin-bottom: 4px;">👮 {base_p} Dispatch Officers</div>
<div style="font-size: 20px; font-weight: 600; color: #ffffff;">🚧 {base_b} Containment Barricades</div>
<p style="font-size:12px; color:#94a3b8; margin-top:10px; margin-bottom:0; font-style:italic;">Calculated from active real-world checkpoint blueprints.</p>
</div>
        """, unsafe_allow_html=True)

    with mr2:
        p_color = "#ef4444" if sim_p > base_p else "#22c55e" if sim_p < base_p else "#94a3b8"
        b_color = "#ef4444" if sim_b > base_b else "#22c55e" if sim_b < base_b else "#94a3b8"
        
        p_delta_str = f"({sim_p-base_p:+d} units needed)" if sim_p != base_p else "(No allocation change)"
        b_delta_str = f"({sim_b-base_b:+d} units needed)" if sim_b != base_b else "(No allocation change)"
        
        st.markdown(f"""
<div class="ui-card" style="border-left: 5px solid {sim_sev_clr} !important; height: 100%;">
<div style="text-transform: uppercase; font-size: 11px; color: {sim_sev_clr}; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 8px;">Simulated Operational Requirement</div>
<div style="font-size: 20px; font-weight: 600; color: {p_color}; margin-bottom: 4px;">👮 {sim_p} Tactical Officers <span style="font-size:12px; font-weight:400; opacity:0.85; margin-left:4px;">{p_delta_str}</span></div>
<div style="font-size: 20px; font-weight: 600; color: {b_color};">🚧 {sim_b} Target Barricades <span style="font-size:12px; font-weight:400; opacity:0.85; margin-left:4px;">{b_delta_str}</span></div>
<p style="font-size:12px; color:#94a3b8; margin-top:10px; margin-bottom:0; font-style:italic;">Dynamic reallocation response matched to simulated risk depth.</p>
</div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESPONSE PLAN (CLEAN CHROMATIC TIMELINE & PLAN STRIPS)
# ══════════════════════════════════════════════════════════════════════════════
with tab_response:
    rp_left, rp_right = st.columns([5, 5], gap="medium")

    # ── TIMELINE DISPATCH LEDGER ──────────────────────────────────────────────
    with rp_left:
        st.markdown('<div class="section-label"> Incident Response Timeline</div>', unsafe_allow_html=True)
        timeline = build_incident_timeline(forecast, resources)

        for i, step in enumerate(timeline):
            is_last   = (i == len(timeline) - 1)
            clr       = "#22c55e" if i == 0 else "#3b82f6" if i == 1 else "#94a3b8"
            dot_style = f"background:{clr};"
            
            # Formatted flat to the absolute margin to block markdown code conversions
            st.markdown(f"""
<div class="tl-row" style="display:flex; gap:16px; margin-bottom:0px;">
<div class="tl-left" style="display:flex; flex-direction:column; align-items:center; width:20px; flex-shrink:0;">
<div class="tl-dot" style="{dot_style} width:12px; height:12px; border-radius:50%; margin-top:4px; box-shadow: 0 0 8px {clr};"></div>
{"" if is_last else f'<div class="tl-line" style="width:2px; background:rgba(255,255,255,0.1); flex-grow:1; margin-top:6px; margin-bottom:4px; min-height:50px;"></div>'}
</div>
<div class="tl-content" style="padding-bottom:18px; flex-grow:1;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
<span class="tl-step" style="font-size:14px; font-weight:600; color:#f1f5f9;">{step['step']}</span>
<span class="tl-time" style="font-size:11px; color:#38bdf8; font-weight:600; background:rgba(56,189,248,0.1); padding:2px 8px; border-radius:12px; border:1px solid rgba(56,189,248,0.15);">T + {step['minute']} MIN</span>
</div>
<div class="tl-detail" style="font-size:13px; color:#94a3b8; line-height:1.4;">{step['detail']}</div>
</div>
</div>
            """, unsafe_allow_html=True)

    # ── RESOURCE OPTIMIZATION ENGINE CARDS ────────────────────────────────────
    with rp_right:
        st.markdown('<div class="section-label"> Resource Optimization Engine</div>', unsafe_allow_html=True)
        plans = resource_optimization(forecast, corridor_vuln)

        plan_colors = {
            "Minimum safe": "#f59e0b",
            "Recommended":  "#38bdf8",
            "Aggressive":   "#ec4899",
        }
        plan_icons = {
            "Minimum safe": "🟡",
            "Recommended":  "🔵",
            "Aggressive":   "🔴",
        }

        for plan_name, plan in plans.items():
            clr  = plan_colors.get(plan_name, "#94a3b8")
            icon = plan_icons.get(plan_name, "⚪")
            rec_min = plan.get("expected_recovery_min", "—")
            sev_eff = plan.get("severity_effect", "—")
            delay_eff = plan.get("delay_effect", "—")

            # Upgraded container using unified layout system architecture to blend seamlessly
            st.markdown(f"""
<div class="ui-card" style="border-left: 5px solid {clr} !important; padding:16px 20px !important; margin-bottom:14px;">
<div class="plan-header" style="color:{clr}; font-size:15px; font-weight:700; display:flex; align-items:center; gap:8px; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.02em;">
<span>{icon}</span> {plan_name} Deployment Strategy
</div>
<div class="plan-row" style="display:flex; gap:20px; margin-bottom:10px;">
<span class="plan-stat" style="font-size:14px; color:#e2e8f0; font-weight:500;">👮 <b>{plan['officers']}</b> Tactical Officers</span>
<span class="plan-stat" style="font-size:14px; color:#e2e8f0; font-weight:500;">🚧 <b>{plan['barricades']}</b> Perimeter Checkpoints</span>
</div>
<div class="plan-row" style="display:flex; gap:20px; font-size:12.5px; color:#94a3b8; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px; margin-bottom:8px;">
<span class="plan-meta"> Target Horizon: <b>{fmt_min(rec_min)}</b></span>
<span class="plan-meta"> Mitigation Curve: <b>{sev_eff} Severity</b></span>
</div>
<div class="plan-tone" style="font-size:12px; font-style:italic; color:#cbd5e1; background:rgba(0,0,0,0.15); padding:6px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.02);">
💡 {plan.get('tone','')}
</div>
</div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CORRIDOR INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_corridors:
    st.markdown('<div class="section-label"> Corridor Vulnerability Intelligence</div>', unsafe_allow_html=True)

    if corridor_vuln is not None and not corridor_vuln.empty:
        # Bar chart
        top_n = corridor_vuln.head(10)
        bar_colors = [
            "#ef4444" if str(r) == "High" else "#f59e0b" if str(r) == "Moderate" else "#22c55e"
            for r in top_n["risk_label"]
        ]
        fig_bar = go.Figure(go.Bar(
            x=top_n["vulnerability_score"],
            y=top_n["corridor"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}" for v in top_n["vulnerability_score"]],
            textposition="outside",
            textfont=dict(color="#bec0c4", size=11),
        ))
        fig_bar.update_layout(
            title=dict(text="Corridor Vulnerability Scores (0–100)", font=dict(size=13, color="#bec0c4")),
            height=340,
            xaxis=dict(title="Vulnerability Score", gridcolor="#1e293b", color="#64748b"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", color="#bec0c4"),
            margin=dict(t=40, b=20, l=10, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # Cause donut chart
        ch1, ch2 = st.columns([1, 1])
        with ch1:
            if "event_cause_category" in df.columns or "event_cause" in df.columns:
                cause_col = "event_cause_category" if "event_cause_category" in df.columns else "event_cause"
                cc = df[cause_col].value_counts().head(8)
                fig_pie = go.Figure(go.Pie(
                    labels=cc.index,
                    values=cc.values,
                    hole=0.5,
                    marker_colors=["#3b82f6","#f59e0b","#ef4444","#22c55e","#8b5cf6","#ec4899","#06b6d4","#a3e635"],
                    textinfo="label+percent",
                    textfont_size=10,
                ))
                fig_pie.update_layout(
                    title=dict(text="Event Cause Distribution", font=dict(size=13, color="#bec0c4")),
                    height=300,
                    margin=dict(t=40, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with ch2:
            # Peak hour heatmap by hour of day
            if "hour_of_day" in df.columns:
                hourly = df.groupby("hour_of_day").size().reindex(range(24), fill_value=0)
                fig_hr = go.Figure(go.Bar(
                    x=hourly.index,
                    y=hourly.values,
                    marker_color=[
                        "#ef4444" if h in [7,8,9,10,17,18,19,20] else "#3b82f6"
                        for h in hourly.index
                    ],
                ))
                fig_hr.update_layout(
                    title=dict(text="Incident Volume by Hour", font=dict(size=13, color="#bec0c4")),
                    height=300,
                    xaxis=dict(title="Hour of Day", gridcolor="#1e293b", color="#afb0b0",
                               tickvals=list(range(0,24,2))),
                    yaxis=dict(title="Event Count", gridcolor="#1e293b", color="#b3b5b8"),
                    margin=dict(t=40, b=20, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_hr, use_container_width=True, config={"displayModeBar": False})

        # Vulnerability table
        st.markdown('<div class="section-label"> Full Corridor Ranking</div>', unsafe_allow_html=True)
        display_cols = ["rank","corridor","incidents","avg_duration_min","closure_rate_pct",
                        "peak_hour_rate_pct","vulnerability_score","risk_label","top_cause"]
        display_df = corridor_vuln[
            [c for c in display_cols if c in corridor_vuln.columns]
        ].copy()

        # Rename for readability
        display_df = display_df.rename(columns={
            "rank": "Rank", "corridor": "Corridor", "incidents": "Events",
            "avg_duration_min": "Avg Duration (min)",
            "closure_rate_pct": "Closure Rate %",
            "peak_hour_rate_pct": "Peak Hour %",
            "vulnerability_score": "Vuln Score",
            "risk_label": "Risk Tier",
            "top_cause": "Top Cause",
        })
        display_df["Avg Duration (min)"] = display_df["Avg Duration (min)"].round(1)
        display_df["Closure Rate %"]     = display_df["Closure Rate %"].round(1)
        display_df["Peak Hour %"]         = display_df["Peak Hour %"].round(1)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Vuln Score": st.column_config.ProgressColumn(
                    "Vuln Score", min_value=0, max_value=100, format="%.1f",
                ),
            },
        )
    else:
        st.info("Corridor vulnerability data not available — ensure the dataset is loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI COPILOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_copilot:
    st.markdown('<div class="section-label"> Command Center AI Copilot</div>', unsafe_allow_html=True)
    st.caption("Ask your Groq-powered advisor for resource adjustments, mitigation protocols, or scenario breakdowns.")

    # Initialize Groq Client safely using Streamlit Secrets
    if "GROQ_API_KEY" not in st.secrets:
        st.info(" To activate the AI Copilot, please add your `GROQ_API_KEY` to your Streamlit Secrets.")
    else:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        # TRACK CONTEXT SWITCHES
        # If the operator changes the junction in the sidebar, reset the chat state automatically
        if "current_tracked_junction" not in st.session_state:
            st.session_state.current_tracked_junction = sel_junction

        if st.session_state.current_tracked_junction != sel_junction:
            st.session_state.current_tracked_junction = sel_junction
            # Wiping the array forces it to re-initialize with the fresh junction name below
            if "chat_messages" in st.session_state:
                del st.session_state.chat_messages

        # Initialize chat messages dynamically with the active junction context
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant", 
                    "content": f"Hello Commander. I have analyzed the situation at **{sel_junction}**. How can I assist you with traffic mitigation plans right now?"
                }
            ]

        # Display history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle new user input
        if user_prompt := st.chat_input("Ask about active routing or deployment strategy..."):
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_prompt)
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})

            # System instruction injecting realtime dashboard context
            system_context = f"""
            You are 'Gridlock Copilot', an expert tactical AI Assistant for the Bengaluru Metropolitan Traffic Operations.
            You help the user coordinate solutions for the currently active incident dashboard configuration:
            - Target Junction: {sel_junction}
            - Impact Corridor: {sel_corridor}
            - Event Cause: {sel_cause} (Type: {sel_type})
            - Dashboard Computed Metrics: Severity is {sev_num}/10 ({sev_lbl}), expected delay is {fmt_min(forecast['expected_delay_min'])}, impact radius is {fmt_km(forecast['affected_radius_km'])}, estimated recovery time is {fmt_min(forecast['estimated_recovery_min'])}.
            - Allocated Resources: {resources['personnel']} officers and {resources['barricades']} barricades.
            
            Be concise, highly professional, deeply knowledgeable about urban traffic management, and reference specific local parameters provided where applicable.
            """

            # Construct the complete payload for Groq
            api_messages = [{"role": "system", "content": system_context}]
            for msg in st.session_state.chat_messages:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

            # Call the Groq API using a fast, active stable model
            try:
                with st.spinner("Analyzing operational directives..."):
                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=api_messages,
                        temperature=0.6,
                        max_tokens=512,
                    )
                    response_text = completion.choices[0].message.content

                # Render response
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                st.rerun()

            except Exception as e:
                st.error(f"Failed to fetch insight from Groq: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="footer">'
    f'Flipkart Gridlock Hackathon 2.0 · Final Build · {len(df):,} historical events · '
    f'Bengaluru Metropolitan Traffic Operations · Built for Hackathon Purposes'
    f'</div>',
    unsafe_allow_html=True,
)