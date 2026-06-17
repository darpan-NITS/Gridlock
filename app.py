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
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gridlock — Traffic Command Center",
    page_icon="$$",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
css_path = Path("assets/custom.css")
if css_path.exists():
    with css_path.open("r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS & CACHING
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "Critical": "#ea0000",
    "Moderate": "#f59e0b",
    "Low": "#08ff63",
}

def fmt_minutes(value: float) -> str:
    if value is None or pd.isna(value): return "—"
    return f"{float(value):.0f} min"

def fmt_km(value: float) -> str:
    if value is None or pd.isna(value): return "—"
    return f"{float(value):.2f} km"

def severity_label(score: int) -> str:
    if score >= 7: return "Critical"
    elif score >= 4: return "Moderate"
    return "Low"

@st.cache_data(show_spinner="Analyzing regional historic baselines...")
def get_cached_pipeline_data():
    try:
        df = load_and_clean_data()
        if df is None or df.empty: raise ValueError("Empty Dataset")
        df = engineer_features(df)
    except Exception:
        np.random.seed(42)
        rows = []
        corridors = ["Tumkur Road", "ORR East 1", "Hosur Road", "Bannerghatta Road", "Old Airport Road", "Bellary Road"]
        junctions = ["Silk Board Junction", "Urvashi Junction", "Lalbagh Main Gate", "Hebbal Flyover", "Marathahalli Junction"]
        causes = ["vip_movement", "accident", "protest", "construction", "water_logging", "congestion"]
        
        for _ in range(200):
            junc = np.random.choice(junctions)
            cause = np.random.choice(causes)
            rows.append({
                "corridor": np.random.choice(corridors),
                "junction": junc,
                "event_cause": cause,
                "event_cause_category": cause.upper(),
                "duration_minutes": float(np.random.exponential(scale=45) + 20),
                "hour_of_day": np.random.randint(0, 24),
                "requires_road_closure": np.random.choice([True, False], p=[0.15, 0.85]),
                "latitude": get_junction_coords(junc)[0] + np.random.uniform(-0.01, 0.01),
                "longitude": get_junction_coords(junc)[1] + np.random.uniform(-0.01, 0.01)
            })
        df = pd.DataFrame(rows)
    
    return df, compute_junction_history(df), compute_corridor_vulnerability(df)

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
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/nolan/96/traffic-light.png", width=70)
st.sidebar.markdown("###  Incident Configuration")

known_junctions = [k for k in get_junction_coords.__globals__['JUNCTION_COORDS'].keys() if k != "Unknown"]
known_corridors = [k for k in get_diversion_route.__globals__['DIVERSION_ROUTES'].keys() if k != "Non-corridor"]

selected_junction = st.sidebar.selectbox("Target Intersection", options=known_junctions)
selected_corridor = st.sidebar.selectbox("Impact Corridor Line", options=known_corridors)
event_cause = st.sidebar.selectbox("Primary Incident Vector", options=["vip_movement", "accident", "protest", "construction", "water_logging", "congestion"])
event_type = st.sidebar.radio("Deployment Categorization", options=["Spontaneous", "Planned"], horizontal=True)
current_hour = st.sidebar.slider("Timeline Window (Hour)", min_value=0, max_value=23, value=datetime.now().hour)
crowd_scale = st.sidebar.select_slider("Volume Profile", options=["Small", "Medium", "Large", "Mega"], value="Medium")
base_duration = st.sidebar.number_input("Baseline Duration (Mins)", min_value=5, max_value=480, value=60)
requires_closure = st.sidebar.toggle("Enforce Road Closure", value=False)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
forecast = forecast_traffic_impact(
    event_type=event_type, event_cause=event_cause, hour=current_hour, junction=selected_junction,
    corridor=selected_corridor, crowd_scale=crowd_scale, event_duration_min=base_duration,
    requires_closure=requires_closure, junction_history=junction_history, corridor_vulnerability=corridor_vulnerability, dataset=df_clean
)

severity_num = forecast["severity"]
label = severity_label(severity_num)
accent_color = SEVERITY_COLORS[label]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL DISPLAY LAYER
# ─────────────────────────────────────────────────────────────────────────────
# Row 1: High-Density Card KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Threat Profile Index</div>
            <div class="kpi-value" style="color: {accent_color} !important;">{severity_num} <span style="font-size:1.1rem; color:#64748b;">/ 10</span></div>
            <p style="color: {accent_color}; margin: 5px 0 0 0; font-size: 0.85rem; font-weight: 700;">⚡ Status: {label}</p>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Est. Backlog Delay</div>
            <div class="kpi-value">{fmt_minutes(forecast['expected_delay_min'])}</div>
            <p style="color: #38bdf8; margin: 5px 0 0 0; font-size: 0.85rem;">Queue Propagation Vector</p>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Shockwave Radius</div>
            <div class="kpi-value">{fmt_km(forecast['affected_radius_km'])}</div>
            <p style="color: #a78bfa; margin: 5px 0 0 0; font-size: 0.85rem;">Spatial Spillover Horizon</p>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class="ui-card">
            <div class="kpi-label">Total System Clearance</div>
            <div class="kpi-value">{fmt_minutes(forecast['estimated_recovery_min'])}</div>
            <p style="color: #22c55e; margin: 5px 0 0 0; font-size: 0.85rem;">Confidence: {forecast['confidence']:.0f}%</p>
        </div>
    """, unsafe_allow_html=True)

# Row 2: Geospatial Mapping and Intelligence Explanations
col_map, col_intel = st.columns([1.6, 1.0])

with col_map:
    st.markdown('<div class="ui-card"><h3> Spatial Impact Telemetry & Diversions</h3>', unsafe_allow_html=True)
    junc_lat, junc_lon = get_junction_coords(selected_junction)
    m = folium.Map(location=[junc_lat, junc_lon], zoom_start=14, tiles="CartoDB dark_matter")
    
    folium.CircleMarker(location=[junc_lat, junc_lon], radius=14, color=accent_color, fill=True, fill_color=accent_color, fill_opacity=0.4).add_to(m)
    folium.Circle(location=[junc_lat, junc_lon], radius=forecast['affected_radius_km'] * 1000, color="#38bdf8", weight=1, fill=True, fill_color="#38bdf8", fill_opacity=0.06).add_to(m)
    
    route_meta = get_diversion_route(selected_corridor)
    if route_meta and route_meta["primary_route"]:
        folium.PolyLine(locations=route_meta["primary_route"], color="#22c55e", weight=5, opacity=0.85).add_to(m)
        for idx, bar_point in enumerate(route_meta["barricades"]):
            folium.Marker(location=bar_point, icon=folium.Icon(color="orange", icon="ban", prefix="fa")).add_to(m)
            
    if not df_clean.empty:
        heat_data = df_clean[["latitude", "longitude"]].dropna().head(40).values.tolist()
        HeatMap(heat_data, radius=15, blur=10, min_opacity=0.3).add_to(m)
        
    st_folium(m, width="100%", height=400, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)

with col_intel:
    st.markdown('<div class="ui-card"><h3> Engine Diagnostics & Signals</h3>', unsafe_allow_html=True)
    reasoning_string = generate_reasoning_text(event_cause, selected_junction, current_hour, severity_num, junction_history)
    st.info(reasoning_string)
    
    st.markdown("##### Core Predictive Drivers")
    for driver in forecast["drivers"]:
        st.markdown(f" `{driver}`")
    st.markdown('</div>', unsafe_allow_html=True)

# Row 3: Playbooks and Tactical Resource Optimization
st.subheader(" Automated Tactical Resource Optimization")
opts_plans = resource_optimization(forecast, corridor_vulnerability)

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(f"""
        <div class="ui-card" style="border-left: 5px solid #94a3b8 !important;">
            <h4 style="color:#94a3b8; margin:0 0 10px 0;">Plan Alpha: Minimum Safe</h4>
            <p style="font-size:0.9rem; margin:4px 0;"> Officers: <b>{opts_plans['Minimum safe']['officers']}</b></p>
            <p style="font-size:0.9rem; margin:4px 0;"> Barricades: <b>{opts_plans['Minimum safe']['barricades']} Units</b></p>
        </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown(f"""
        <div class="ui-card" style="border-left: 5px solid #6366f1 !important;">
            <h4 style="color:#6366f1; margin:0 0 10px 0;">Plan Bravo: Recommended</h4>
            <p style="font-size:0.9rem; margin:4px 0;"> Officers: <b>{opts_plans['Recommended']['officers']}</b></p>
            <p style="font-size:0.9rem; margin:4px 0;"> Barricades: <b>{opts_plans['Recommended']['barricades']} Units</b></p>
        </div>
    """, unsafe_allow_html=True)
with p3:
    st.markdown(f"""
        <div class="ui-card" style="border-left: 5px solid #ec4899 !important;">
            <h4 style="color:#ec4899; margin:0 0 10px 0;">Plan Charlie: Aggressive</h4>
            <p style="font-size:0.9rem; margin:4px 0;"> Officers: <b>{opts_plans['Aggressive']['officers']}</b></p>
            <p style="font-size:0.9rem; margin:4px 0;"> Barricades: <b>{opts_plans['Aggressive']['barricades']} Units</b></p>
        </div>
    """, unsafe_allow_html=True)