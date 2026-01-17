"""
AADHAAR SETU - Settings Page
Configuration and API integration settings
"""
import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APP_NAME, VERSION, DATA_DIR, ML_DIR

st.set_page_config(page_title=f"{APP_NAME} - Settings", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%); padding: 1rem 1.5rem; margin: -1rem -1rem 1.5rem -1rem; color: white;">
    <h2 style="margin: 0; color: white;">Settings</h2>
    <p style="margin: 0.2rem 0 0 0; color: #ecf0f1; font-size: 0.9rem;">Configuration and data source management</p>
</div>
""", unsafe_allow_html=True)

# Application Info
st.markdown("### Application Information")

col1, col2, col3 = st.columns(3)

col1.metric("Application", APP_NAME)
col2.metric("Version", VERSION)
col3.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))

st.markdown("---")

# Data Source Configuration
st.markdown("### Data Source Configuration")

data_source = st.radio(
    "Current Data Source",
    ["CSV Files (Current)", "Live API (Future)"],
    index=0,
    disabled=True  # Only CSV is currently supported
)

if data_source == "CSV Files (Current)":
    st.success("Using local CSV files from cleaned_data/ and ml_outputs/ folders")
    
    st.markdown("**Data Files Status:**")
    
    # Check file status
    files_status = []
    
    master_file = DATA_DIR / "master_cleaned.csv"
    if master_file.exists():
        size = master_file.stat().st_size / (1024 * 1024)
        files_status.append(("master_cleaned.csv", "Available", f"{size:.2f} MB"))
    else:
        files_status.append(("master_cleaned.csv", "Missing", "-"))
    
    risk_file = ML_DIR / "high_risk_regions.csv"
    if risk_file.exists():
        size = risk_file.stat().st_size / (1024 * 1024)
        files_status.append(("high_risk_regions.csv", "Available", f"{size:.2f} MB"))
    else:
        files_status.append(("high_risk_regions.csv", "Missing", "-"))
    
    alerts_file = ML_DIR / "monthly_alerts.csv"
    if alerts_file.exists():
        size = alerts_file.stat().st_size / (1024 * 1024)
        files_status.append(("monthly_alerts.csv", "Available", f"{size:.2f} MB"))
    else:
        files_status.append(("monthly_alerts.csv", "Missing", "-"))
    
    state_file = ML_DIR / "state_summary.csv"
    if state_file.exists():
        size = state_file.stat().st_size / (1024 * 1024)
        files_status.append(("state_summary.csv", "Available", f"{size:.2f} MB"))
    else:
        files_status.append(("state_summary.csv", "Missing", "-"))
    
    import pandas as pd
    status_df = pd.DataFrame(files_status, columns=["File", "Status", "Size"])
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    if st.button("Refresh Data Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Data will be reloaded on next page visit.")

st.markdown("---")

# Future API Configuration
st.markdown("### API Integration (Future)")

st.info("""
**Note:** API integration is planned for future releases when UIDAI/Ministry provides live data endpoints.

The application is designed to seamlessly switch from CSV to API data source without any UI changes.
""")

with st.expander("API Configuration (For Future Use)"):
    api_endpoint = st.text_input(
        "API Endpoint URL",
        placeholder="https://api.uidai.gov.in/v1/data",
        disabled=True
    )
    
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Enter API key when available",
        disabled=True
    )
    
    refresh_interval = st.selectbox(
        "Data Refresh Interval",
        ["Every hour", "Every 6 hours", "Daily", "Weekly"],
        disabled=True
    )
    
    st.button("Test Connection", disabled=True)
    st.button("Save API Configuration", disabled=True)

st.markdown("---")

# Display Settings
st.markdown("### Display Settings")

col1, col2 = st.columns(2)

with col1:
    show_powerbi = st.checkbox("Show Power BI Page", value=True)
    show_alerts = st.checkbox("Show Alerts Page", value=True)

with col2:
    default_state = st.selectbox(
        "Default State Filter",
        ["All States", "Uttar Pradesh", "Maharashtra", "Bihar", "West Bengal"],
        index=0
    )

st.markdown("---")

# Data Period
st.markdown("### Data Period")

st.markdown("""
| Parameter | Value |
|-----------|-------|
| Start Date | March 2025 |
| End Date | December 2025 |
| Total Months | 10 |
| Data Type | Historical |
""")

st.markdown("---")

# About
st.markdown("### About AADHAAR SETU")

st.markdown("""
**AADHAAR SETU** is a District Administration Dashboard designed to provide insights into Aadhaar service delivery across India.

**Features:**
- District and Pincode level analysis
- Monthly trend tracking
- Risk assessment and alerts
- Downloadable reports
- Power BI integration

**Data Sources:**
- Biometric update records
- Demographic update records  
- New enrollment records

**Target Users:**
- District Collectors
- State Administration
- UIDAI Officials
- Policy Makers

**Technical Stack:**
- Python / Streamlit
- Pandas / Plotly
- Power BI (optional)

**Contact:**
For technical support or feature requests, contact the development team.
""")

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>{APP_NAME} v{VERSION} | Settings</p>", unsafe_allow_html=True)
