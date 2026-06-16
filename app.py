import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime

# Import identical logic structures extracted directly from your notebook
from utils.data_processor import load_and_clean_data, engineer_features
from utils.models import compute_junction_history, congestion_score, manpower_engine, process_kmeans_centers

st.set_page_config(page_title="Gridlock Analytics Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# CORE DATA PIPELINE WIRING
# -----------------------------------------------------------------------------
@st.cache_data
def run_backend_pipeline():
    try:
        # Tries loading your real dataset if placed in working root directory
        df = load_and_clean_data("Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")
    except Exception:
        # Mock fallback data mimicking your real notebook values if file is absent
        import numpy as np
        df = pd.DataFrame({
            'event_type': np.random.choice(['unplanned', 'planned'], size=100),
            'start_datetime': pd.date_range(start='2024-01-01', periods=100, freq='h'),
            'latitude': np.random.uniform(12.85, 13.15, size=100),
            'longitude': np.random.uniform(77.50, 77.75, size=100),
            'junction': np.random.choice(['UrvashiJunction', 'LalbaghMainGateJunc', 'Unknown'], size=100),
            'corridor': np.random.choice(['Tumkur Road', 'ORR East 1', 'Non-corridor'], size=100),
            'event_cause': np.random.choice(['vehicle_breakdown', 'accident', 'vip_movement', 'water_logging'], size=100)
        })
    
    df = engineer_features(df)
    j_history = compute_junction_history(df)
    kmeans_centers = process_kmeans_centers(df, k=10)
    
    return df, j_history, kmeans_centers

df, junction_history, centers = run_backend_pipeline()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL ENTRY CAPTURE
# -----------------------------------------------------------------------------
st.sidebar.header("🚨 Live Traffic Incident Entry")

with st.sidebar.form("incident_submission_form"):
    event_type_input = st.selectbox("Event Structural Type", ["unplanned", "planned"])
    
    event_cause_input = st.selectbox(
        "Event Cause Parameter",
        ["vehicle_breakdown", "accident", "vip_movement", "water_logging", "procession", "protest", "public_event", "construction", "congestion"]
    )
    
    input_time = st.time_input("Reporting Outbreak Time", datetime.now().time())
    
    junction_input = st.selectbox("Assigned Targeted Junction", sorted(df['junction'].unique()))
    corridor_input = st.selectbox("Assigned Targeted Corridor", sorted(df['corridor'].dropna().unique()))
    
    submit_btn = st.form_submit_button("Run Resource Allocator Models")

# -----------------------------------------------------------------------------
# MAIN INTERACTIVE DISPLAY LAYOUT
# -----------------------------------------------------------------------------
st.title("Gridlock Incident Response Tactical Dashboard")
st.markdown("---")

col_map, col_metrics = st.columns([3, 2])

with col_map:
    st.subheader("📍 Historical Risk Clusters Map (Step 5 KMeans Centers)")
    
    # Force coordinates to your dataset's location (Bangalore) if data mean is invalid
    try:
        if not centers.dropna(subset=['latitude', 'longitude']).empty:
            map_lat = centers['latitude'].dropna().iloc[0]
            map_lon = centers['longitude'].dropna().iloc[0]
        else:
            map_lat, map_lon = 12.9716, 77.5946
    except Exception:
        map_lat, map_lon = 12.9716, 77.5946 # Bangalore standard coordinates fallback
        
    # Build Folium map - explicitly using an integer height/width for compatibility
    risk_map = folium.Map(location=[map_lat, map_lon], zoom_start=11, tiles="CartoDB positron")
    
    # Plot markers
    for idx, row in centers.iterrows():
        if pd.notna(row.latitude) and pd.notna(row.longitude):
            folium.Marker(
                location=[row.latitude, row.longitude],
                popup=f"Risk Cluster {idx}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(risk_map)
        
    # Render with numeric pixel definitions to prevent container collapse
    st_folium(risk_map, width=600, height=500)

with col_metrics:
    st.subheader("📊 Live Assessment Matrix Outputs")
    
    if submit_btn:
        # Run calculated scoring algorithms from Cell 6
        severity = congestion_score(
            event_type_input,
            event_cause_input,
            input_time.hour,
            junction_input,
            junction_history
        )
        
        # Pull required resource counts from Cell 8
        resources = manpower_engine(severity, event_type_input, junction_input, corridor_input)
        
        # Display Results
        st.markdown("### **Congestion Assessment Metrics**")
        if severity >= 7:
            st.error(f"🔴 Critical Impact Level: {severity}/10")
        elif 4 <= severity < 7:
            st.warning(f"🟡 Moderate Impact Level: {severity}/10")
        else:
            st.success(f"🟢 Minor Impact Level: {severity}/10")
            
        st.progress(severity / 10.0)
        
        st.markdown("---")
        st.markdown("### **Deployment Allocations**")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Personnel Target Count", f"{resources['personnel']} Officers")
        with c2:
            st.metric("Barricades Target Count", f"{resources['barricades']} Units")
            
        st.markdown("**Field Positions Deployment Strategy:**")
        if resources['deployment_positions']:
            for pos in resources['deployment_positions']:
                st.write(f"- `{pos}`")
        else:
            st.write("_No specific junction or corridor deployments mapped._")
            
    else:
        st.info("👈 Fill out the Incident Entry form on the sidebar and click submit to trigger the predictive rule models.")