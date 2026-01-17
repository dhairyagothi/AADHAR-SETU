"""
AADHAAR SETU - Data Loading Module
Handles CSV loading and future API integration
"""
import pandas as pd
import streamlit as st
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import MASTER_DATA, RISK_DATA, ALERTS_DATA, STATE_DATA, CLUSTER_DATA


@st.cache_data(ttl=3600)
def load_master_data():
    """Load master cleaned data"""
    try:
        df = pd.read_csv(MASTER_DATA)
        df['year_month'] = pd.to_datetime(df['year_month'])
        df['state'] = df['state'].str.title()
        df['district'] = df['district'].str.title()
        
        # Fill NaN with 0 for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading master data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_risk_data():
    """Load high risk regions data"""
    try:
        df = pd.read_csv(RISK_DATA)
        df['state'] = df['state'].str.title()
        df['district'] = df['district'].str.title()
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_alerts_data():
    """Load monthly alerts data"""
    try:
        df = pd.read_csv(ALERTS_DATA)
        df['state'] = df['state'].str.title()
        df['district'] = df['district'].str.title()
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_state_summary():
    """Load state summary data"""
    try:
        df = pd.read_csv(STATE_DATA)
        df['state'] = df['state'].str.title()
        return df
    except Exception as e:
        return pd.DataFrame()


def get_states(df):
    """Get unique states"""
    return sorted(df['state'].unique().tolist())


def get_districts(df, state):
    """Get districts for a state"""
    return sorted(df[df['state'] == state]['district'].unique().tolist())


def get_pincodes(df, state, district):
    """Get pincodes for a district"""
    mask = (df['state'] == state) & (df['district'] == district)
    return sorted(df[mask]['pincode'].unique().tolist())


def filter_data(df, state=None, district=None, pincode=None):
    """Filter data by location"""
    filtered = df.copy()
    
    if state and state != "All States":
        filtered = filtered[filtered['state'] == state]
    
    if district and district != "All Districts":
        filtered = filtered[filtered['district'] == district]
    
    if pincode and pincode != "All Pincodes":
        filtered = filtered[filtered['pincode'] == pincode]
    
    return filtered


# ============================================================================
# FUTURE API INTEGRATION PLACEHOLDER
# ============================================================================

class APIHandler:
    """
    Placeholder for future ministry API integration.
    Currently returns None - falls back to CSV data.
    """
    
    def __init__(self, endpoint=None, api_key=None):
        self.endpoint = endpoint
        self.api_key = api_key
        self.is_connected = False
    
    def test_connection(self):
        """Test API connection - to be implemented"""
        # TODO: Implement when ministry provides API
        return False
    
    def fetch_data(self, params=None):
        """Fetch data from API - to be implemented"""
        # TODO: Implement when ministry provides API
        return None
    
    def get_realtime_stats(self):
        """Get real-time statistics - to be implemented"""
        # TODO: Implement when ministry provides API
        return None
