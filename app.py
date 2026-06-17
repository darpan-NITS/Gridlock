import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime
import plotly.graph_objects as go

from utils.data_processor import load_and_clean_data, engineer_features, compute_kpi_stats
from utils.models import (
    compute_junction_history, congestion_score, manpower_engine,
    process_kmeans_centers, generate_reasoning_text,
    get_junction_coords, get_diversion_route, whatif_score,
    compute_corridor_risk
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gridlock — Traffic Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_backend_pipeline():
    try:
        df = load_and_clean_data(
            "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
        )
    except Exception:
        rng = np.random.default_rng(42)
        n = 300
        junctions = [
            "Silk Board Junction", "Urvashi Junction", "LalbaghMainGateJunc",
            "Hebbal Flyover", "Marathahalli Junction", "KR Circle",
            "MG Road Junction", "Koramangala Junction", "Yeshwanthpur Junction",
            "Whitefield Signal", "Electronic City Toll", "Unknown"
        ]
        corridors = [
            "Tumkur Road", "ORR East 1", "Hosur Road", "Bannerghatta Road",
            "Old Airport Road", "Bellary Road", "Non-corridor"
        ]
        causes = [
            "vehicle_breakdown", "accident", "vip_movement", "water_logging",
            "procession", "protest", "public_event", "construction", "congestion"
        ]
        durations = rng.exponential(scale=90, size=n).clip(10, 600)
        start = pd.date_range("2024-01-01", periods=n, freq="3h")
        df = pd.DataFrame({
            "event_type":    rng.choice(["unplanned", "planned"], n, p=[0.75, 0.25]),
            "start_datetime": start,
            "end_datetime":  start + pd.to_timedelta(durations, unit="m"),
            "latitude":      rng.uniform(12.85, 13.10, n),
            "longitude":     rng.uniform(77.50, 77.75, n),
            "junction":      rng.choice(junctions, n),
            "corridor":      rng.choice(corridors, n),
            "event_cause":   rng.choice(causes, n),
            "priority":      rng.choice(["P1", "P2", "P3"], n),
            "requires_road_closure": rng.choice([True, False], n, p=[0.35, 0.65]),
            "status":        rng.choice(["resolved", "active", "pending"], n, p=[0.70, 0.15, 0.15]),
        })

    df = engineer_features(df)
    j_history  = compute_junction_history(df)
    kpi        = compute_kpi_stats(df)
    centers    = process_kmeans_centers(df, k=10)
    corr_risk  = compute_corridor_risk(df)
    return df, j_history, kpi, centers, corr_risk

df, junction_history, kpi_stats, cluster_centers, corridor_risk = run_backend_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "incidents" not in st.session_state:
    st.session_state.incidents = []
if "last_incident" not in st.session_state:
    st.session_state.last_incident = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INCIDENT FORM
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚨 Incident Entry")
st.sidebar.markdown("---")

with st.sidebar.form("incident_form"):
    event_cause_input = st.selectbox(
        "Event Cause",
        ["vehicle_breakdown", "accident", "vip_movement", "water_logging",
         "procession", "protest", "public_event", "construction", "congestion"],
        help="Primary cause of the traffic event"
    )
    event_type_input = st.selectbox(
        "Event Type",
        ["unplanned", "planned"],
    )
    junction_input = st.selectbox(
        "Junction",
        sorted(df["junction"].unique()),
    )
    corridor_input = st.selectbox(
        "Corridor",
        sorted(df["corridor"].dropna().unique()),
    )
    input_hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
    requires_closure = st.checkbox("Requires Road Closure")
    submit_btn = st.form_submit_button("⚡ Analyse & Deploy Plan", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔬 What-If Simulator")
st.sidebar.caption("Adjust parameters to preview impact without submitting")

wi_crowd   = st.sidebar.select_slider(
    "Crowd Scale", options=["Small", "Medium", "Large", "Mega"], value="Medium"
)
wi_hour    = st.sidebar.slider("Simulated Hour", 0, 23, 12)
wi_closure = st.sidebar.checkbox("Assume Road Closure", value=False)

# Live what-if recompute (uses last submitted incident's cause+junction as base)
wi_base_cause    = st.session_state.last_incident["cause"] if st.session_state.last_incident else "vehicle_breakdown"
wi_base_junction = st.session_state.last_incident["junction"] if st.session_state.last_incident else junction_input
wi_base_type     = st.session_state.last_incident["event_type"] if st.session_state.last_incident else "unplanned"

wi_severity = whatif_score(
    wi_base_cause, wi_base_type, wi_hour,
    wi_base_junction, junction_history, wi_crowd, wi_closure
)
wi_resources = manpower_engine(wi_severity, wi_base_type, wi_base_junction, corridor_input)

SEVERITY_COLORS = {
    "Critical":  "#ef4444",
    "Moderate":  "#f59e0b",
    "Low":       "#22c55e",
}

def severity_label(s):
    if s >= 7: return "Critical"
    if s >= 4: return "Moderate"
    return "Low"

wi_label = severity_label(wi_severity)
wi_color = SEVERITY_COLORS[wi_label]
st.sidebar.markdown(
    f"""
    <div class="whatif-card">
        <div class="whatif-header">Simulated Impact</div>
        <div class="whatif-score" style="color:{wi_color};">
            {wi_severity}/10 — {wi_label}
        </div>
        <div class="whatif-detail">
            👮 {wi_resources['personnel']} officers &nbsp;|&nbsp;
            🚧 {wi_resources['barricades']} barricades
        </div>
        <div class="whatif-detail">
            📍 Crowd: {wi_crowd} &nbsp;|&nbsp;
            ⏰ Hour: {wi_hour:02d}:00
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS FORM SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
if submit_btn:
    severity = congestion_score(
        event_type_input, event_cause_input,
        input_hour, junction_input, junction_history
    )
    resources = manpower_engine(severity, event_type_input, junction_input, corridor_input)
    reasoning = generate_reasoning_text(
        event_cause_input, junction_input, input_hour, severity, junction_history
    )
    coords = get_junction_coords(junction_input)
    route  = get_diversion_route(corridor_input)

    incident = {
        "id":           len(st.session_state.incidents) + 1,
        "time":         datetime.now().strftime("%H:%M"),
        "cause":        event_cause_input,
        "event_type":   event_type_input,
        "junction":     junction_input,
        "corridor":     corridor_input,
        "hour":         input_hour,
        "severity":     severity,
        "label":        severity_label(severity),
        "resources":    resources,
        "reasoning":    reasoning,
        "coords":       coords,
        "route":        route,
        "requires_closure": requires_closure,
        "status":       "Active",
    }
    st.session_state.incidents.insert(0, incident)
    st.session_state.last_incident = incident

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="top-header">
        <span class="header-title">🚦 Gridlock</span>
        <span class="header-sub">Bengaluru Traffic Command Center</span>
        <span class="header-live"><span class="live-dot"></span> LIVE</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
active_now      = len([i for i in st.session_state.incidents if i["status"] == "Active"])
high_risk_corrs = sum(1 for v in corridor_risk.values() if v > 120)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Active Incidents</div>'
        f'<div class="kpi-value" style="color:#ef4444;">{active_now}</div></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Avg Resolution</div>'
        f'<div class="kpi-value">{kpi_stats["avg_resolution_min"]:.0f} min</div></div>',
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">High-Risk Corridors</div>'
        f'<div class="kpi-value" style="color:#f59e0b;">{high_risk_corrs}</div></div>',
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Historical Events</div>'
        f'<div class="kpi-value">{kpi_stats["total_events"]:,}</div></div>',
        unsafe_allow_html=True,
    )
with k5:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Road Closure Rate</div>'
        f'<div class="kpi-value">{kpi_stats["closure_rate"]:.0f}%</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT — MAP  |  INCIDENT FEED
# ─────────────────────────────────────────────────────────────────────────────
col_map, col_feed = st.columns([6, 4], gap="medium")

# ── MAP ──────────────────────────────────────────────────────────────────────
with col_map:
    st.markdown('<div class="section-label">📍 Live Risk Intelligence Map</div>', unsafe_allow_html=True)

    m = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # Historical event density heatmap
    # NOTE: assign() adds a column — must explicitly select 3 cols for HeatMap
    heat_data = (
        df[["latitude", "longitude", "duration_minutes"]]
        .dropna()
        .assign(weight=lambda x: x["duration_minutes"].clip(0, 400) / 400)
        [["latitude", "longitude", "weight"]]
        .values.tolist()
    )
    if heat_data:
        HeatMap(
            heat_data,
            radius=18,
            blur=22,
            max_zoom=13,
            gradient={"0.2": "#3b82f6", "0.5": "#f59e0b", "0.8": "#ef4444"},
        ).add_to(m)

    # KMeans cluster centers — sized and colored by risk
    for idx, row in cluster_centers.iterrows():
        if pd.notna(row.latitude) and pd.notna(row.longitude):
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=9,
                color="#7c3aed",
                fill=True,
                fill_color="#7c3aed",
                fill_opacity=0.6,
                popup=folium.Popup(f"Risk Cluster {idx + 1}", max_width=120),
                tooltip=f"Cluster {idx + 1} — historical hotspot",
            ).add_to(m)

    # Overlay submitted incidents
    severity_color_map = {
        "Critical": "#ef4444",
        "Moderate": "#f59e0b",
        "Low":      "#22c55e",
    }
    for inc in st.session_state.incidents:
        lat, lon = inc["coords"]
        color    = severity_color_map[inc["label"]]

        # Pulse ring
        folium.Circle(
            location=[lat, lon],
            radius=800 + inc["severity"] * 150,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.10,
            weight=2,
        ).add_to(m)

        # Event marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>#{inc['id']} {inc['cause'].replace('_',' ').title()}</b><br>"
                f"Severity: {inc['severity']}/10 ({inc['label']})<br>"
                f"Junction: {inc['junction']}<br>"
                f"Time: {inc['time']}",
                max_width=220,
            ),
            tooltip=f"#{inc['id']} — {inc['label']} ({inc['severity']}/10)",
        ).add_to(m)

        # Barricade markers
        for i, (blat, blon) in enumerate(inc["route"].get("barricades", []), 1):
            folium.Marker(
                location=[blat, blon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:16px;margin-top:-8px;">🚧</div>',
                    icon_size=(24, 24),
                    icon_anchor=(12, 12),
                ),
                tooltip=f"Barricade Point {i}",
            ).add_to(m)

        # Diversion route polyline
        route_coords = inc["route"].get("primary_route", [])
        if route_coords:
            folium.PolyLine(
                locations=route_coords,
                color="#2563eb",
                weight=4,
                opacity=0.75,
                dash_array="8 4",
                tooltip=f"Diversion: {inc['route'].get('name','Alt Route')}",
            ).add_to(m)

    st_folium(m, use_container_width=True, height=520, returned_objects=[])

# ── INCIDENT FEED ─────────────────────────────────────────────────────────────
with col_feed:
    st.markdown('<div class="section-label">📋 Incident Feed</div>', unsafe_allow_html=True)

    if not st.session_state.incidents:
        st.markdown(
            '<div class="empty-feed">Submit an incident from the sidebar to see it appear here.</div>',
            unsafe_allow_html=True,
        )
    else:
        for inc in st.session_state.incidents[:6]:   # show latest 6
            color  = severity_color_map[inc["label"]]
            badge_cls = inc["label"].lower()

            deploy_html = "".join(
                f'<span class="deploy-tag">{p}</span>'
                for p in inc["resources"]["deployment_positions"]
            )

            st.markdown(
                f"""
                <div class="incident-card" style="border-left: 4px solid {color};">
                    <div class="inc-header">
                        <span class="inc-id">#{inc['id']}</span>
                        <span class="inc-cause">{inc['cause'].replace('_',' ').title()}</span>
                        <span class="badge badge-{badge_cls}">{inc['label']}</span>
                        <span class="inc-time">{inc['time']}</span>
                    </div>
                    <div class="inc-junction">📍 {inc['junction']} — {inc['corridor']}</div>
                    <div class="inc-reasoning">🤖 {inc['reasoning']}</div>
                    <div class="inc-resources">
                        👮 <b>{inc['resources']['personnel']}</b> officers &nbsp;
                        🚧 <b>{inc['resources']['barricades']}</b> barricades
                    </div>
                    <div class="deploy-row">{deploy_html}</div>
                    <div class="diversion-note">
                        🔀 <b>Diversion:</b> {inc['route'].get('name', 'See map')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# ACTION PLAN PANEL (shows only when an incident exists)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.last_incident:
    inc = st.session_state.last_incident
    st.markdown("---")
    st.markdown('<div class="section-label">⚡ Active Response Plan — Most Recent Incident</div>', unsafe_allow_html=True)

    color = severity_color_map[inc["label"]]
    p1, p2, p3 = st.columns([1, 2, 2])

    with p1:
        # Severity gauge using plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=inc["severity"],
            title={"text": "Severity Score", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 10], "tickwidth": 1},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0, 4],  "color": "#dcfce7"},
                    {"range": [4, 7],  "color": "#fef9c3"},
                    {"range": [7, 10], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "value": inc["severity"],
                },
            },
            number={"suffix": "/10", "font": {"size": 24}},
        ))
        fig.update_layout(
            height=220,
            margin=dict(t=30, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#374151",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})  # noqa: deprecated but stable on 1.35

    with p2:
        st.markdown("**Deployment Breakdown**")
        deploy = inc["resources"]["deployment_positions"]
        if deploy:
            for pos in deploy:
                st.markdown(f"- {pos}")
        else:
            st.markdown("_No specific positions mapped._")

        if inc["requires_closure"]:
            st.markdown(
                '<span class="badge badge-critical" style="font-size:12px;">⚠️ Road Closure Required</span>',
                unsafe_allow_html=True,
            )

        st.markdown("**What-If Comparison**")
        base_s  = inc["severity"]
        wi_s    = wi_severity
        delta   = wi_s - base_s
        delta_s = f"+{delta}" if delta > 0 else str(delta)
        delta_c = "#ef4444" if delta > 0 else "#22c55e"
        st.markdown(
            f'<div class="whatif-compare">'
            f'Submitted plan: <b>{base_s}/10</b> &nbsp;→&nbsp; '
            f'Simulated scenario: <b style="color:{delta_c};">{wi_s}/10 ({delta_s})</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown("**Diversion Route**")
        route = inc["route"]
        st.markdown(f"🔵 **Primary:** {route.get('name', 'Not specified')}")
        if route.get("alternate_name"):
            st.markdown(f"🟡 **Alternate:** {route['alternate_name']}")
        st.markdown(f"🚧 **Barricade points:** {len(route.get('barricades', []))}")

        st.markdown("**Timeline Estimate**")
        avg_dur = junction_history.get(inc["junction"], 60)
        clear_time = int(avg_dur * (0.7 + inc["severity"] * 0.06))
        st.markdown(f"⏱ Expected clear time: **{clear_time} min**")
        st.markdown(f"📊 Based on {inc['junction']} historical avg: {avg_dur:.0f} min")

# ─────────────────────────────────────────────────────────────────────────────
# CORRIDOR RISK CHART (always visible, from dataset analysis)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">📊 Corridor Risk Intelligence (Historical)</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    sorted_corr = sorted(corridor_risk.items(), key=lambda x: x[1], reverse=True)[:8]
    corr_names  = [c[0][:18] for c in sorted_corr]
    corr_vals   = [c[1] for c in sorted_corr]
    bar_colors  = ["#ef4444" if v > 120 else "#f59e0b" if v > 60 else "#22c55e" for v in corr_vals]

    fig2 = go.Figure(go.Bar(
        x=corr_vals,
        y=corr_names,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.0f} min" for v in corr_vals],
        textposition="outside",
    ))
    fig2.update_layout(
        title="Avg Resolution Time by Corridor",
        title_font_size=13,
        xaxis_title="Minutes",
        height=300,
        margin=dict(t=40, b=20, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        xaxis=dict(gridcolor="#e5e7eb", gridwidth=0.5),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})  # noqa

with chart_col2:
    cause_counts = (
        df["event_cause_category"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "cause", "event_cause_category": "count"})
    )
    if "cause" not in cause_counts.columns:
        cause_counts.columns = ["cause", "count"]

    fig3 = go.Figure(go.Pie(
        labels=cause_counts["cause"],
        values=cause_counts["count"],
        hole=0.48,
        marker_colors=["#3b82f6","#f59e0b","#ef4444","#22c55e","#8b5cf6","#ec4899","#06b6d4"],
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig3.update_layout(
        title="Event Cause Distribution",
        title_font_size=13,
        height=300,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})  # noqa

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Gridlock v0.2 — Day 2 Build &nbsp;|&nbsp; '
    'Powered by Bengaluru historical event data (8,173 incidents) &nbsp;|&nbsp; '
    'Built for Smart City Hackathon</div>',
    unsafe_allow_html=True,
)
