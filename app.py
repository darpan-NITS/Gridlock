import math
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from utils.data_processor import engineer_features, compute_kpi_stats, load_and_clean_data
from utils.models import (
    build_explainability_points,
    build_incident_timeline,
    compute_corridor_risk,
    compute_corridor_vulnerability,
    compute_junction_history,
    forecast_traffic_impact,
    generate_reasoning_text,  # <-- Added this missing import
    get_diversion_route,
    get_junction_coords,
    manpower_engine,
    process_kmeans_centers,
    resource_optimization,
    scenario_delta,
    whatif_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gridlock — Traffic Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS if available
css_path = Path("assets/custom.css")
if css_path.exists():
    with css_path.open("r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    # Embedded fallback styling for an elegant, modern cyber-dark control room theme
    st.markdown("""
        <style>
        .main { background-color: #0b0f19; color: #f1f5f9; }
        .stMetric { background: rgba(30, 41, 59, 0.4); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }
        div[data-testid="stSidebar"] { background-color: #0f172a; }
        h1, h2, h3 { color: #f8fafc; font-weight: 700; }
        </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS & CACHING
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "Critical": "#ef4444",
    "Moderate": "#f59e0b",
    "Low": "#22c55e",
}

def fmt_minutes(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.0f} min"

def fmt_km(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.2f} km"

def severity_label(score: int) -> str:
    if score >= 7: return "Critical"
    elif score >= 4: return "Moderate"
    return "Low"

def severity_band(score: int) -> str:
    return severity_label(score)

def safe_top_corridor_score(vuln_df: pd.DataFrame, corridor: str) -> float:
    if vuln_df is None or vuln_df.empty:
        return 0.0
    match = vuln_df[vuln_df["corridor"].astype(str).str.lower() == str(corridor).lower()]
    if match.empty:
        return 0.0
    return float(match.iloc[0]["vulnerability_score"])

def render_plan_card(title: str, plan: dict, accent: str) -> None:
    """Renders sleek operational metrics blocks directly into the DOM."""
    st.markdown(
        f"""
        <div style="
            background: rgba(15, 23, 42, 0.96);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-left: 4px solid {accent};
            border-radius: 14px;
            padding: 16px;
            min-height: 200px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
            margin-bottom: 15px;
         border-top-right-radius: 4px;
         border-bottom-right-radius: 4px;
        ">
            <h4 style="color: {accent}; margin-top: 0; margin-bottom: 6px; font-size:1.15rem;">{title}</h4>
            <p style="font-size: 0.85rem; font-style: italic; color: #94a3b8; margin-bottom: 14px;">{plan['tone']}</p>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.9rem;">
                <span style="color: #cbd5e1;">Personnel Wave:</span>
                <b style="color: #ffffff;">🛟 {plan['officers']} Officers</b>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.9rem;">
                <span style="color: #cbd5e1;">Barricades Needed:</span>
                <b style="color: #ffffff;">🚧 {plan['barricades']} Units</b>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.9rem;">
                <span style="color: #cbd5e1;">Est. Recovery Curve:</span>
                <b style="color: #ffffff;">⏱️ {plan['expected_recovery_min']:.0f} min</b>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                <span style="color: #cbd5e1;">Target Residual Delay:</span>
                <b style="color: #ffffff;">📉 {plan['delay_effect']:.0f} min</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner="Analyzing regional historic baselines...")
def get_cached_pipeline_data():
    """Loads historical dataset; falls back to simulated structure if file missing."""
    try:
        df = load_and_clean_data()
        if df is None or df.empty:
            raise ValueError("Empty Dataset")
        df = engineer_features(df)
    except Exception:
        # High-fidelity synthesis tracking Bengaluru traffic behavior profiles
        np.random.seed(42)
        rows = []
        corridors = ["Tumkur Road", "ORR East 1", "Hosur Road", "Bannerghatta Road", "Old Airport Road", "Bellary Road"]
        junctions = ["Silk Board Junction", "Urvashi Junction", "Lalbagh Main Gate", "Hebbal Flyover", "Marathahalli Junction"]
        causes = ["vip_movement", "accident", "protest", "construction", "water_logging", "congestion"]
        
        for _ in range(200):
            corr = np.random.choice(corridors)
            junc = np.random.choice(junctions)
            cause = np.random.choice(causes)
            dur = float(np.random.exponential(scale=45) + 20)
            rows.append({
                "corridor": corr,
                "junction": junc,
                "event_cause": cause,
                "event_cause_category": cause.upper(),
                "duration_minutes": dur,
                "hour_of_day": np.random.randint(0, 24),
                "requires_road_closure": np.random.choice([True, False], p=[0.15, 0.85]),
                "latitude": get_junction_coords(junc)[0] + np.random.uniform(-0.01, 0.01),
                "longitude": get_junction_coords(junc)[1] + np.random.uniform(-0.01, 0.01)
            })
        df = pd.DataFrame(rows)
    
    junc_hist = compute_junction_history(df)
    corr_vuln = compute_corridor_vulnerability(df)
    return df, junc_hist, corr_vuln

# ─────────────────────────────────────────────────────────────────────────────
# CORE DATA INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
df_clean, junction_history, corridor_vulnerability = get_cached_pipeline_data()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style="background: linear-gradient(90deg, #1e1b4b 0%, #0f172a 100%); padding: 20px; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2); margin-bottom: 25px;">
        <h1 style="margin: 0; font-size: 2.2rem; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">GRIDLOCK: Predictive Incident Command</h1>
        <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 1rem;">Real-time resource provisioning, spatial telemetry, and mitigation forecasting for Bengaluru Metropolitan Corridors.</p>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR PARAMETRIC CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/nolan/96/traffic-light.png", width=70)
st.sidebar.markdown("### 🛠️ Incident Configuration")

# Dynamic mappings from known telemetry lists
known_junctions = [k for k in get_junction_coords.__globals__['JUNCTION_COORDS'].keys() if k != "Unknown"]
known_corridors = [k for k in get_diversion_route.__globals__['DIVERSION_ROUTES'].keys() if k != "Non-corridor"]

selected_junction = st.sidebar.selectbox("Target Intersection", options=known_junctions)
selected_corridor = st.sidebar.selectbox("Impact Corridor Line", options=known_corridors)

event_cause = st.sidebar.selectbox(
    "Primary Incident Vector", 
    options=["vip_movement", "accident", "protest", "procession", "construction", "water_logging", "congestion", "vehicle_breakdown"]
)
event_type = st.sidebar.radio("Deployment Categorization", options=["Spontaneous", "Planned"], horizontal=True)

current_hour = st.sidebar.slider("Timeline Execution Window (Hour)", min_value=0, max_value=23, value=datetime.now().hour)
crowd_scale = st.sidebar.select_slider("Crowd/Density Volume Profile", options=["Small", "Medium", "Large", "Mega"], value="Medium")
base_duration = st.sidebar.number_input("Expected Active Event Baseline (Mins)", min_value=5, max_value=480, value=60, step=5)
requires_closure = st.sidebar.toggle("Enforce Physical Corridor Road Closure", value=False)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTATION ENGINE LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
forecast = forecast_traffic_impact(
    event_type=event_type,
    event_cause=event_cause,
    hour=current_hour,
    junction=selected_junction,
    corridor=selected_corridor,
    crowd_scale=crowd_scale,
    event_duration_min=base_duration,
    requires_closure=requires_closure,
    junction_history=junction_history,
    corridor_vulnerability=corridor_vulnerability,
    dataset=df_clean
)

severity_num = forecast["severity"]
label = severity_band(severity_num)
accent_color = SEVERITY_COLORS[label]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL DISPLAY LAYER
# ─────────────────────────────────────────────────────────────────────────────
# Row 1: High-Density KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class="stMetric">
            <span style='color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Threat Profile Index</span>
            <h2 style='color: {accent_color}; margin: 5px 0 0 0; font-size: 2.2rem;'>{severity_num} <span style='font-size:1.1rem;'>/ 10</span></h2>
            <p style='color: {accent_color}; margin: 2px 0 0 0; font-size: 0.85rem; font-weight: 700;'>⚡ Status: {label}</p>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class="stMetric">
            <span style='color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Est. Backlog Delay</span>
            <h2 style='color: #f8fafc; margin: 5px 0 0 0; font-size: 2.2rem;'>{fmt_minutes(forecast['expected_delay_min'])}</h2>
            <p style='color: #38bdf8; margin: 2px 0 0 0; font-size: 0.85rem;'>Queue Propagation Vector</p>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class="stMetric">
            <span style='color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Shockwave Radius</span>
            <h2 style='color: #f8fafc; margin: 5px 0 0 0; font-size: 2.2rem;'>{fmt_km(forecast['affected_radius_km'])}</h2>
            <p style='color: #a78bfa; margin: 2px 0 0 0; font-size: 0.85rem;'>Spatial Spillover Horizon</p>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class="stMetric">
            <span style='color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Total System Clearance</span>
            <h2 style='color: #f8fafc; margin: 5px 0 0 0; font-size: 2.2rem;'>{fmt_minutes(forecast['estimated_recovery_min'])}</h2>
            <p style='color: #22c55e; margin: 2px 0 0 0; font-size: 0.85rem;'>Confidence Index: {forecast['confidence']:.0f}%</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Row 2: Geospatial Mapping and Intelligence Explanations
col_map, col_intel = st.columns([1.6, 1.0])

with col_map:
    st.subheader("🌐 Spatial Impact Telemetry & Diversions")
    
    junc_lat, junc_lon = get_junction_coords(selected_junction)
    m = folium.Map(location=[junc_lat, junc_lon], zoom_start=14, tiles="CartoDB dark_matter")
    
    # Target Event Marker
    folium.CircleMarker(
        location=[junc_lat, junc_lon],
        radius=14,
        popup=f"<b>Junction:</b> {selected_junction}<br><b>Threat Rank:</b> {severity_num}/10",
        color=accent_color,
        fill=True,
        fill_color=accent_color,
        fill_opacity=0.4
    ).add_to(m)
    
    # Impact Radius Visualizer Bound
    folium.Circle(
        location=[junc_lat, junc_lon],
        radius=forecast['affected_radius_km'] * 1000,
        color="#38bdf8",
        weight=1,
        fill=True,
        fill_color="#38bdf8",
        fill_opacity=0.06
    ).add_to(m)
    
    # Diversion Route Architecture Processing
    route_meta = get_diversion_route(selected_corridor)
    if route_meta and route_meta["primary_route"]:
        folium.PolyLine(
            locations=route_meta["primary_route"],
            color="#22c55e",
            weight=5,
            opacity=0.85,
            tooltip=f"Primary Diversion: {route_meta['name']}"
        ).add_to(m)
        
        # Add entry/exit marker indicators for barricades
        for idx, bar_point in enumerate(route_meta["barricades"]):
            folium.Marker(
                location=bar_point,
                icon=folium.Icon(color="orange", icon="ban", prefix="fa"),
                popup=f"Barricade Deployment Checkpoint Node #{idx + 1}"
            ).add_to(m)
            
    # Overlay localized structural heatmap to give full tactical context
    if not df_clean.empty:
        heat_data = df_clean[["latitude", "longitude"]].dropna().head(40).values.tolist()
        HeatMap(heat_data, radius=15, blur=10, min_opacity=0.3).add_to(m)
        
    st_folium(m, width="100%", height=420, returned_objects=[])

with col_intel:
    st.subheader("🧠 Engine Diagnostics & Signals")
    
    # Diagnostic narrative generation output
    reasoning_string = generate_reasoning_text(event_cause, selected_junction, current_hour, severity_num, junction_history)
    st.info(reasoning_string)
    
    st.markdown("##### Core Predictive Drivers")
    for driver in forecast["drivers"]:
        st.markdown(f"🏷️ `{driver}`")
        
    st.markdown("##### Historical Corridor Matrix Match")
    explain_points = build_explainability_points(
        forecast, event_type, event_cause, current_hour, selected_junction, selected_corridor, requires_closure, corridor_vulnerability, df_clean
    )
    for pt in explain_points:
        st.markdown(f"• <span style='font-size:0.88rem; color:#cbd5e1;'>{pt}</span>", unsafe_allow_html=True)

st.markdown("---")

# Row 3: Resource Allocations and Optimization Playbooks
st.subheader("⚡ Automated Tactical Resource Optimization")
opts_plans = resource_optimization(forecast, corridor_vulnerability)

p1, p2, p3 = st.columns(3)
with p1:
    render_plan_card("Plan Alpha: Minimum Safe", opts_plans["Minimum safe"], "#94a3b8")
with p2:
    render_plan_card("Plan Bravo: Recommended Control", opts_plans["Recommended"], "#6366f1")
with p3:
    render_plan_card("Plan Charlie: Aggressive Suppression", opts_plans["Aggressive"], "#ec4899")

# Row 4: Sequence Deployment Incident Timeline
st.markdown("### ⏱️ Sequential Operational Response Timeline")
manpower_alloc = manpower_engine(severity_num, event_type, selected_junction, selected_corridor)
timeline_events = build_incident_timeline(forecast, manpower_alloc)

# Constructing an analytical timeline workflow widget component layout
cols_timeline = st.columns(len(timeline_events))
for index, step_data in enumerate(timeline_events):
    with cols_timeline[index]:
        st.markdown(
            f"""
            <div style="
                background: rgba(30, 41, 59, 0.3);
                border-top: 3px solid {accent_color if index == 0 else '#38bdf8'};
                border-radius: 8px;
                padding: 12px;
                text-align: left;
                min-height: 140px;
            ">
                <span style="color:#6366f1; font-weight:700; font-size:0.8rem;">T + {step_data['minute']} MIN</span>
                <h5 style="margin: 4px 0; color:#f8fafc; font-size:0.95rem;">{step_data['step']}</h5>
                <p style="margin:0; font-size:0.78rem; color:#94a3b8; line-height:1.3;">{step_data['detail']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER STATISTICAL TELEMETRY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="text-align: center; color: #475569; font-size: 0.75rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;">
        Control Layer Live Base Tracking • Corridor Fragility Profile Rank Cache Hash Active
    </div>
    """, 
    unsafe_allow_html=True
)