import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    # Load raw data from your local file
    df = pd.read_csv(file_path)
    
    # Drop columns with >60% null values, keeping 'junction'
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > 0.60].index
    cols_to_drop = cols_to_drop.drop('junction', errors='ignore')
    df = df.drop(columns=cols_to_drop)
    
    df['junction'] = df['junction'].fillna('Unknown')
    
    # Track the exact columns specified in your notebook step 1
    required_cols = [
        'event_type', 'start_datetime', 'end_datetime', 'latitude', 'longitude',
        'corridor', 'junction', 'priority', 'requires_road_closure', 'status', 'event_cause'
    ]
    existing_cols = [c for c in required_cols if c in df.columns]
    return df[existing_cols].copy()

def categorize_cause(cause):
    if pd.isna(cause):
        return "Other"
    cause = str(cause).lower()
    if cause in ["accident"]:
        return "Accident"
    elif cause in ["vehicle_breakdown"]:
        return "Breakdown"
    elif cause in ["water_logging", "tree_fall", "debris", "pot_holes", "road_conditions", "fog / low visibility"]:
        return "Infrastructure"
    elif cause in ["vip_movement", "procession", "protest", "public_event"]:
        return "Special Event"
    elif cause in ["construction"]:
        return "Construction"
    elif cause in ["congestion"]:
        return "Traffic"
    return "Other"

def engineer_features(df):
    # Datetime parse
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], format='ISO8601', errors='coerce')
    
    # Duration Calculation with your notebook's exact fallback logic
    median_duration = 60
    if 'end_datetime' in df.columns and 'start_datetime' in df.columns:
        df['end_datetime'] = pd.to_datetime(df['end_datetime'], format='ISO8601', errors='coerce')
        df['duration_minutes'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 60
        df['duration_minutes'] = df['duration_minutes'].fillna(median_duration)
    else:
        df['duration_minutes'] = median_duration
        
    # Feature Engineering columns from cell 3 & 4
    df['hour_of_day'] = df['start_datetime'].dt.hour.fillna(0).astype(int)
    df['day_of_week'] = df['start_datetime'].dt.day_name()
    df['is_weekend'] = (df['start_datetime'].dt.weekday >= 5).astype(int)
    
    if 'event_cause' in df.columns:
        df['event_cause_category'] = df['event_cause'].apply(categorize_cause)
    else:
        df['event_cause_category'] = "Other"
        
    return df