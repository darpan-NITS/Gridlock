import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def compute_junction_history(df):
    # Cell 5 calculation
    return df.groupby('junction')['duration_minutes'].mean().to_dict()

def congestion_score(event_type, event_cause, hour, junction, junction_history):
    # Your exact mathematical mapping logic from Cell 6
    score = 1
    
    cause_weights = {
        "vip_movement": 5,
        "accident": 4,
        "procession": 4,
        "protest": 4,
        "public_event": 3,
        "construction": 3,
        "vehicle_breakdown": 2,
        "water_logging": 3,
        "congestion": 3
    }
    
    score += cause_weights.get(str(event_cause).lower(), 1)
    
    # Peak hour effect mapping
    if hour in [8, 9, 10, 17, 18, 19]:
        score += 2
        
    # Junction historical risk lookup
    avg_duration = junction_history.get(junction, 30)
    if avg_duration > 180:
        score += 2
    elif avg_duration > 90:
        score += 1
        
    # Planned offsets
    if str(event_type).lower() == "planned":
        score -= 1
        
    return max(0, min(10, score))

def manpower_engine(severity, event_type, junction, corridor):
    # Your exact threshold rules from Cell 8
    if severity <= 3:
        personnel = 2
        barricades = 0
    elif severity <= 6:
        personnel = 5
        barricades = 4
    elif severity <= 8:
        personnel = 10
        barricades = 8
    else:
        personnel = 20
        barricades = 15

    deployment = []
    if pd.notna(junction) and junction != 'Unknown':
        deployment.append(f"Control at {junction}")
    if pd.notna(corridor) and str(corridor).strip() != "":
        deployment.append(f"Patrol along {corridor}")
        
    return {
        "personnel": personnel,
        "barricades": barricades,
        "deployment_positions": deployment
    }

def process_kmeans_centers(df, k=10):
    # Logic extracted from cells 9, 10 & 11
    cluster_df = df[['latitude', 'longitude', 'duration_minutes']].dropna()
    if len(cluster_df) < k:
        k = max(1, len(cluster_df))
        
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_df['cluster'] = kmeans.fit_predict(cluster_df[['latitude', 'longitude']])
    
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude'])
    return centers