"""
AADHAAR SETU - Power BI Integration Page
Embed Power BI dashboards
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APP_NAME

st.set_page_config(page_title=f"{APP_NAME} - Power BI Dashboard", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%); padding: 1rem 1.5rem; margin: -1rem -1rem 1.5rem -1rem; color: white;">
    <h2 style="margin: 0; color: white;">Power BI Dashboard</h2>
    <p style="margin: 0.2rem 0 0 0; color: #ecf0f1; font-size: 0.9rem;">Interactive analytics dashboard</p>
</div>
""", unsafe_allow_html=True)

# Power BI Embed URL (to be configured)
# This can be updated by administrator
POWERBI_EMBED_URL = st.session_state.get('powerbi_url', '')

if not POWERBI_EMBED_URL:
    st.markdown("### Power BI Dashboard Integration")
    
    st.markdown("""
    <div style="background: #f8f9fa; border: 2px dashed #bdc3c7; border-radius: 8px; padding: 3rem; text-align: center; margin: 2rem 0;">
        <h3 style="color: #7f8c8d; margin-bottom: 1rem;">Power BI Dashboard Placeholder</h3>
        <p style="color: #95a5a6;">Power BI dashboard will be displayed here after configuration.</p>
        <p style="color: #95a5a6; font-size: 0.9rem;">Dashboard size: Full width, 800px height</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Configure Power BI Embed")
    
    st.markdown("""
    To embed a Power BI dashboard:
    
    1. Open Power BI Service (app.powerbi.com)
    2. Navigate to your report
    3. Click File > Embed Report > Website or Portal
    4. Copy the embed URL
    5. Paste it below
    """)
    
    embed_url = st.text_input(
        "Power BI Embed URL",
        placeholder="https://app.powerbi.com/reportEmbed?reportId=..."
    )
    
    if st.button("Save and Preview"):
        if embed_url:
            st.session_state['powerbi_url'] = embed_url
            st.rerun()
        else:
            st.error("Please enter a valid Power BI embed URL")
    
    st.markdown("---")
    
    st.markdown("### Alternative: Upload Power BI File")
    
    st.info("""
    **Note:** For full Power BI functionality, use Power BI Service (online).
    
    Local .pbix files cannot be embedded directly. You need to:
    1. Publish your .pbix to Power BI Service
    2. Get the embed URL from Power BI Service
    3. Enter the URL above
    """)

else:
    # Display embedded Power BI
    st.markdown(f"""
    <iframe 
        src="{POWERBI_EMBED_URL}"
        width="100%" 
        height="800" 
        frameborder="0" 
        allowFullScreen="true"
        style="border-radius: 8px; border: 1px solid #e0e0e0;">
    </iframe>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Change Dashboard URL"):
            st.session_state['powerbi_url'] = ''
            st.rerun()

# Instructions
with st.expander("How to create Power BI dashboard for AADHAAR SETU"):
    st.markdown("""
    ### Steps to Create Power BI Dashboard
    
    **1. Open Power BI Desktop**
    - Download from: https://powerbi.microsoft.com/desktop/
    
    **2. Load Data Files**
    - Get Data > Text/CSV
    - Load these files from `ml_outputs/` folder:
        - `cluster_summary.csv`
        - `high_risk_regions.csv`
        - `monthly_alerts.csv`
        - `state_summary.csv`
    
    **3. Create Visualizations**
    
    Recommended visuals:
    
    | Data Source | Visual Type |
    |-------------|-------------|
    | state_summary | Map (states colored by risk) |
    | high_risk_regions | Table with conditional formatting |
    | monthly_alerts | Line chart (trends over time) |
    | cluster_summary | Pie chart (cluster distribution) |
    
    **4. Publish to Power BI Service**
    - File > Publish > Publish to Power BI
    - Select your workspace
    
    **5. Get Embed URL**
    - Open report in Power BI Service
    - File > Embed Report > Website or Portal
    - Copy the URL
    
    **6. Paste URL Above**
    - Enter the embed URL in the configuration field
    - Click Save and Preview
    """)

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>{APP_NAME} | Power BI Integration</p>", unsafe_allow_html=True)
