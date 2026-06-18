from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

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
# SIDEBAR — inputs that drive EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Gridlock")
    st.caption("Bengaluru Traffic Command Center")
    st.markdown("---")
    st.markdown("###  Incident Configuration")

    sel_junction  = st.selectbox("Target Junction",    KNOWN_JUNCTIONS, index=0)
    sel_corridor  = st.selectbox("Impact Corridor",    KNOWN_CORRIDORS, index=0)
    sel_cause     = st.selectbox("Event Cause",        CAUSES,          index=0)
    sel_type      = st.radio("Event Type", ["unplanned", "planned"], horizontal=True)
    sel_hour      = st.slider("Hour of Day", 0, 23, datetime.now().hour)
    sel_crowd     = st.select_slider("Crowd Scale", CROWD_OPTIONS, value="Medium")
    sel_duration  = st.number_input("Est. Duration (min)", min_value=5, max_value=480, value=60, step=5)
    sel_closure   = st.toggle("Requires Road Closure", value=False)

    st.markdown("---")
    st.caption(f" Dataset: {len(df):,} historical events loaded")

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
tab_map, tab_forecast, tab_whatif, tab_response, tab_corridors = st.tabs([
    "  Live Map",
    "  Forecast & Intel",
    "  What-If Simulator",
    "  Response Plan",
    "  Corridor Intelligence",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    col_map, col_feed = st.columns([6, 4], gap="medium")

    with col_map:
        st.markdown('<div class="section-label"> Spatial Risk Intelligence</div>', unsafe_allow_html=True)

        jlat, jlon = get_junction_coords(sel_junction)
        m = folium.Map(location=[jlat, jlon], zoom_start=13, tiles="CartoDB dark_matter")

        # Historical heatmap
        heat_pts = (
            df[["latitude", "longitude", "duration_minutes"]].dropna()
            .assign(w=lambda x: x["duration_minutes"].clip(0, 400) / 400)
            [["latitude", "longitude", "w"]].values.tolist()
        )
        if heat_pts:
            HeatMap(heat_pts, radius=18, blur=22,
                    gradient={"0.2": "#3b82f6", "0.5": "#f59e0b", "0.8": "#ef4444"}).add_to(m)

        # KMeans cluster markers
        for _, row in cluster_centers.iterrows():
            if pd.notna(row.latitude) and pd.notna(row.longitude):
                folium.CircleMarker(
                    location=[row.latitude, row.longitude],
                    radius=7, color="#cabde9",
                    fill=True, fill_color="#c5b8e1", fill_opacity=0.55,
                    tooltip="Historical risk cluster",
                ).add_to(m)

        # Current incident marker + impact ring
        folium.CircleMarker(
            location=[jlat, jlon], radius=13,
            color=sev_clr, fill=True, fill_color=sev_clr, fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{sel_cause.replace('_',' ').title()}</b><br>"
                f"Severity: {sev_num}/10 ({sev_lbl})<br>"
                f"Junction: {sel_junction}",
                max_width=220,
            ),
            tooltip=f"{sel_junction} — {sev_lbl} ({sev_num}/10)",
        ).add_to(m)

        folium.Circle(
            location=[jlat, jlon],
            radius=forecast["affected_radius_km"] * 1000,
            color=sev_clr, weight=2,
            fill=True, fill_color=sev_clr, fill_opacity=0.07,
            tooltip=f"Impact radius: {forecast['affected_radius_km']:.2f} km",
        ).add_to(m)

        # Diversion route + barricades
        route = get_diversion_route(sel_corridor)
        if route.get("primary_route"):
            folium.PolyLine(
                locations=route["primary_route"],
                color="#22c55e", weight=4, opacity=0.85,
                dash_array="8 4",
                tooltip=f"Diversion: {route['name']}",
            ).add_to(m)
        for i, bp in enumerate(route.get("barricades", []), 1):
            folium.Marker(
                location=bp,
                icon=folium.DivIcon(html='<div style="font-size:18px;margin-top:-9px">🚧</div>',
                                    icon_size=(24, 24), icon_anchor=(12, 12)),
                tooltip=f"Barricade Point {i}",
            ).add_to(m)

        import streamlit.components.v1 as components
        map_html = m._repr_html_()
        components.html(map_html, height=500, scrolling=False)

    with col_feed:
        st.markdown('<div class="section-label"> AI Engine Diagnostics</div>', unsafe_allow_html=True)

        # Reasoning text
        reasoning = generate_reasoning_text(
            sel_cause, sel_junction, sel_hour, sev_num, junction_history
        )
        st.markdown(
            f'<div class="reasoning-box">{reasoning}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Forecast Drivers**")
        for driver in forecast.get("drivers", []):
            st.markdown(f'<span class="driver-tag">{driver}</span>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label"> Explainability Points</div>', unsafe_allow_html=True)

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
        for i, pt in enumerate(expl_points, 1):
            st.markdown(
                f'<div class="expl-point"><span class="expl-num">{i}</span>{pt}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label"> Diversion Route</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="route-card">'
            f'<div class="route-primary"> {route.get("name","—")}</div>'
            f'{"<div class=route-alt> Alt: " + route["alternate_name"] + "</div>" if route.get("alternate_name") else ""}'
            f'<div class="route-bar"> {len(route.get("barricades",[]))} barricade points</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECAST & INTEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="section-label"> Traffic Impact Forecast</div>', unsafe_allow_html=True)
    st.caption(
        f"Based on {forecast['supporting_event_count']} similar historical events · "
        f"Corridor vulnerability: {forecast['corridor_vulnerability_score']:.1f}/100 "
        f"({forecast['corridor_risk_label']})"
    )

    # Forecast metric cards
    fm1, fm2, fm3, fm4, fm5 = st.columns(5)
    fmetrics = [
        (fm1, "Severity Score",    f"{sev_num}/10",                                sev_clr,   "Rule engine + corridor history"),
        (fm2, "Expected Delay",     fmt_min(forecast["expected_delay_min"]),         "#38bdf8", "Queue propagation model"),
        (fm3, "Impact Radius",      fmt_km(forecast["affected_radius_km"]),          "#dfdce5", "Spatial spillover horizon"),
        (fm4, "Recovery Time",      fmt_min(forecast["estimated_recovery_min"]),     "#22c55e", "Historical resolution patterns"),
        (fm5, "Confidence",         f"{forecast['confidence']:.0f}%",               "#f59e0b", f"n={forecast['supporting_event_count']} similar events"),
    ]
    for col, label, value, color, hint in fmetrics:
        with col:
            st.markdown(
                f'<div class="forecast-card">'
                f'<div class="fc-label">{label}</div>'
                f'<div class="fc-value" style="color:{color};">{value}</div>'
                f'<div class="fc-hint">{hint}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin:1.2rem 0;'></div>", unsafe_allow_html=True)

    # Plotly gauge + confidence bar
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
        # Confidence radial chart
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

    # Explainability full panel
    st.markdown('<div class="section-label"> Why This Forecast</div>', unsafe_allow_html=True)
    ep_cols = st.columns(2)
    for i, pt in enumerate(expl_points if 'expl_points' in dir() else build_explainability_points(
        forecast=forecast, event_type=sel_type, event_cause=sel_cause, hour=sel_hour,
        junction=sel_junction, corridor=sel_corridor, requires_closure=sel_closure,
        corridor_vulnerability=corridor_vuln, dataset=df,
    )):
        with ep_cols[i % 2]:
            st.markdown(
                f'<div class="expl-card"><span class="expl-num-lg">{i+1}</span>{pt}</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_whatif:
    st.markdown('<div class="section-label"> What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.caption("Change parameters below and instantly see how the impact forecast shifts vs. the current plan.")

    wi_c1, wi_c2 = st.columns(2)
    with wi_c1:
        wi_cause    = st.selectbox("Simulated Cause",    CAUSES,        index=CAUSES.index(sel_cause), key="wi_cause")
        wi_type     = st.radio("Simulated Type", ["unplanned","planned"], horizontal=True, key="wi_type",
                               index=0 if sel_type=="unplanned" else 1)
        wi_hour     = st.slider("Simulated Hour", 0, 23, sel_hour, key="wi_hour")
    with wi_c2:
        wi_crowd    = st.select_slider("Simulated Crowd", CROWD_OPTIONS, value=sel_crowd, key="wi_crowd")
        wi_duration = st.number_input("Simulated Duration (min)", 5, 480, int(sel_duration), step=5, key="wi_dur")
        wi_closure  = st.toggle("Simulated Road Closure", value=sel_closure, key="wi_closure")

    # Compute simulated forecast
    sim_forecast = forecast_traffic_impact(
        event_type=wi_type,
        event_cause=wi_cause,
        hour=wi_hour,
        junction=sel_junction,       # junction stays fixed — same location
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

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label"> Plan Comparison</div>', unsafe_allow_html=True)

    # Comparison table
    metrics_compare = [
        ("Severity",        f"{sev_num}/10 ({sev_lbl})",
                            f"{sim_forecast['severity']}/10 ({sim_sev_lbl})",
                            delta["severity"], False),
        ("Delay",           fmt_min(forecast["expected_delay_min"]),
                            fmt_min(sim_forecast["expected_delay_min"]),
                            delta["delay_min"], False),
        ("Impact Radius",   fmt_km(forecast["affected_radius_km"]),
                            fmt_km(sim_forecast["affected_radius_km"]),
                            delta["radius_km"], False),
        ("Recovery Time",   fmt_min(forecast["estimated_recovery_min"]),
                            fmt_min(sim_forecast["estimated_recovery_min"]),
                            delta["recovery_min"], False),
        ("Confidence",      f"{forecast['confidence']:.0f}%",
                            f"{sim_forecast['confidence']:.0f}%",
                            delta["confidence"], True),
    ]

    hdr, base_col, sim_col, chg_col = st.columns([2, 2, 2, 1])
    hdr.markdown("**Metric**")
    base_col.markdown("**Current Plan**")
    sim_col.markdown("**Simulated Scenario**")
    chg_col.markdown("**Change**")

    for metric, base_v, sim_v, d, higher_is_good in metrics_compare:
        hdr, base_col, sim_col, chg_col = st.columns([2, 2, 2, 1])
        hdr.markdown(f"`{metric}`")
        base_col.markdown(base_v)
        sim_col.markdown(f'<span style="color:{sim_sev_clr if metric == "Severity" else "#f1f5f9"};">{sim_v}</span>', unsafe_allow_html=True)
        chg_col.markdown(
            f'<span style="color:{delta_color(d, invert=higher_is_good)};font-weight:700;">{delta_arrow(d)}</span>',
            unsafe_allow_html=True,
        )

    # Simulated manpower
    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    sim_resources = manpower_engine(sim_forecast["severity"], wi_type, sel_junction, sel_corridor)
    base_p = resources["personnel"]; sim_p = sim_resources["personnel"]
    base_b = resources["barricades"]; sim_b = sim_resources["barricades"]

    mr1, mr2 = st.columns(2)
    with mr1:
        st.markdown(
            f'<div class="compare-resource-card" style="border-color:#3b82f6;">'
            f'<div class="cr-head" style="color:#3b82f6;">Current Plan</div>'
            f'<div class="cr-stat"> {base_p} officers</div>'
            f'<div class="cr-stat"> {base_b} barricades</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with mr2:
        p_color = "#ef4444" if sim_p > base_p else "#22c55e" if sim_p < base_p else "#b6b7b8"
        b_color = "#ef4444" if sim_b > base_b else "#22c55e" if sim_b < base_b else "#b4b6b8"
        st.markdown(
            f'<div class="compare-resource-card" style="border-color:{sim_sev_clr};">'
            f'<div class="cr-head" style="color:{sim_sev_clr};">Simulated Scenario</div>'
            f'<div class="cr-stat" style="color:{p_color};"> {sim_p} officers ({("+" if sim_p>=base_p else "")}{sim_p-base_p:+d})</div>'
            f'<div class="cr-stat" style="color:{b_color};"> {sim_b} barricades ({("+" if sim_b>=base_b else "")}{sim_b-base_b:+d})</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESPONSE PLAN
# ══════════════════════════════════════════════════════════════════════════════
with tab_response:
    rp_left, rp_right = st.columns([5, 5], gap="medium")

    # ── Timeline ──────────────────────────────────────────────────────────────
    with rp_left:
        st.markdown('<div class="section-label"> Incident Response Timeline</div>', unsafe_allow_html=True)
        timeline = build_incident_timeline(forecast, resources)

        for i, step in enumerate(timeline):
            is_last   = (i == len(timeline) - 1)
            clr       = "#22c55e" if i == 0 else "#3b82f6" if i == 1 else "#c0c0c0"
            dot_style = f"background:{clr};"
            st.markdown(
                f"""
                <div class="tl-row">
                    <div class="tl-left">
                        <div class="tl-dot" style="{dot_style}"></div>
                        {"" if is_last else '<div class="tl-line"></div>'}
                    </div>
                    <div class="tl-content">
                        <div class="tl-step">{step['step']}</div>
                        <div class="tl-time">T + {step['minute']} min</div>
                        <div class="tl-detail">{step['detail']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Resource Optimization ─────────────────────────────────────────────────
    with rp_right:
        st.markdown('<div class="section-label"> Resource Optimization Plans</div>', unsafe_allow_html=True)
        plans = resource_optimization(forecast, corridor_vuln)

        plan_colors = {
            "Minimum safe": "#94a3b8",
            "Recommended":  "#6366f1",
            "Aggressive":   "#ec4899",
        }
        plan_icons = {
            "Minimum safe": "🟡",
            "Recommended":  "🔵",
            "Aggressive":   "🔴",
        }

        for plan_name, plan in plans.items():
            clr  = plan_colors.get(plan_name, "#b3b5b8")
            icon = plan_icons.get(plan_name, "⚪")
            rec_min = plan.get("expected_recovery_min", "—")
            sev_eff = plan.get("severity_effect", "—")
            delay_eff = plan.get("delay_effect", "—")

            st.markdown(
                f"""
                <div class="plan-card" style="border-left:4px solid {clr};">
                    <div class="plan-header" style="color:{clr};">{icon} {plan_name}</div>
                    <div class="plan-row">
                        <span class="plan-stat"> <b>{plan['officers']}</b> officers</span>
                        <span class="plan-stat"> <b>{plan['barricades']}</b> barricades</span>
                    </div>
                    <div class="plan-row">
                        <span class="plan-meta">Recovery: {fmt_min(rec_min)}</span>
                        <span class="plan-meta">Severity effect: {sev_eff}</span>
                    </div>
                    <div class="plan-tone">{plan.get('tone','')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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