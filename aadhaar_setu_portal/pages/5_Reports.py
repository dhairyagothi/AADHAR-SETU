"""
AADHAAR SETU - Reports Page
Downloadable reports and summaries
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APP_NAME, COLORS
from data_loader import load_master_data, load_risk_data, load_state_summary, get_states, get_districts

st.set_page_config(page_title=f"{APP_NAME} - Reports", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%); padding: 1rem 1.5rem; margin: -1rem -1rem 1.5rem -1rem; color: white;">
    <h2 style="margin: 0; color: white;">Reports</h2>
    <p style="margin: 0.2rem 0 0 0; color: #ecf0f1; font-size: 0.9rem;">Generate and download summary reports</p>
</div>
""", unsafe_allow_html=True)

# Load data
df = load_master_data()
risk_df = load_risk_data()
state_df = load_state_summary()

if df.empty:
    st.error("Unable to load data")
    st.stop()

st.markdown("### Select Report Type")

report_type = st.selectbox(
    "Report Type",
    ["State Summary Report", "District Summary Report", "Risk Assessment Report", "Full Data Export"]
)

st.markdown("---")

if report_type == "State Summary Report":
    st.markdown("### State Summary Report")
    
    state_summary = df.groupby('state').agg({
        'pincode': 'nunique',
        'district': 'nunique',
        'total_activity_bio': 'sum',
        'total_activity_demo': 'sum',
        'enrollments_age_0_5': 'sum',
        'enrollments_age_5_17': 'sum',
        'enrollments_age_18_plus': 'sum'
    }).reset_index()
    
    state_summary['Total Enrollments'] = state_summary['enrollments_age_0_5'] + state_summary['enrollments_age_5_17'] + state_summary['enrollments_age_18_plus']
    state_summary['Total Activity'] = state_summary['total_activity_bio'] + state_summary['total_activity_demo'] + state_summary['Total Enrollments']
    
    state_summary = state_summary.rename(columns={
        'state': 'State',
        'pincode': 'Pincodes',
        'district': 'Districts',
        'total_activity_bio': 'Biometric Updates',
        'total_activity_demo': 'Demographic Updates'
    })
    
    display_cols = ['State', 'Districts', 'Pincodes', 'Biometric Updates', 'Demographic Updates', 'Total Enrollments', 'Total Activity']
    state_summary = state_summary[display_cols].sort_values('Total Activity', ascending=False)
    
    st.dataframe(state_summary, use_container_width=True, hide_index=True)
    
    # Download button
    csv = state_summary.to_csv(index=False)
    st.download_button(
        label="Download State Summary (CSV)",
        data=csv,
        file_name=f"state_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

elif report_type == "District Summary Report":
    st.markdown("### District Summary Report")
    
    # State filter
    states = get_states(df)
    selected_state = st.selectbox("Select State", states)
    
    filtered_df = df[df['state'] == selected_state]
    
    district_summary = filtered_df.groupby('district').agg({
        'pincode': 'nunique',
        'total_activity_bio': 'sum',
        'total_activity_demo': 'sum',
        'enrollments_age_0_5': 'sum',
        'enrollments_age_5_17': 'sum',
        'enrollments_age_18_plus': 'sum',
        'updates_age_5_17_bio': 'sum',
        'updates_age_18_plus_bio': 'sum',
        'updates_age_5_17_demo': 'sum',
        'updates_age_18_plus_demo': 'sum'
    }).reset_index()
    
    district_summary['Total Enrollments'] = district_summary['enrollments_age_0_5'] + district_summary['enrollments_age_5_17'] + district_summary['enrollments_age_18_plus']
    district_summary['Total Activity'] = district_summary['total_activity_bio'] + district_summary['total_activity_demo'] + district_summary['Total Enrollments']
    
    district_summary = district_summary.rename(columns={
        'district': 'District',
        'pincode': 'Pincodes',
        'total_activity_bio': 'Biometric Updates',
        'total_activity_demo': 'Demographic Updates',
        'updates_age_5_17_bio': 'Bio (5-17 yrs)',
        'updates_age_18_plus_bio': 'Bio (18+ yrs)',
        'updates_age_5_17_demo': 'Demo (5-17 yrs)',
        'updates_age_18_plus_demo': 'Demo (18+ yrs)'
    })
    
    display_cols = ['District', 'Pincodes', 'Biometric Updates', 'Demographic Updates', 'Total Enrollments', 'Total Activity']
    district_summary = district_summary[display_cols].sort_values('Total Activity', ascending=False)
    
    st.dataframe(district_summary, use_container_width=True, hide_index=True)
    
    # Download button
    csv = district_summary.to_csv(index=False)
    st.download_button(
        label="Download District Summary (CSV)",
        data=csv,
        file_name=f"district_summary_{selected_state.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

elif report_type == "Risk Assessment Report":
    st.markdown("### Risk Assessment Report")
    
    if risk_df.empty:
        st.warning("Risk data not available")
    else:
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            states = ["All States"] + get_states(risk_df)
            selected_state = st.selectbox("Filter by State", states)
        
        with col2:
            risk_levels = ["All Levels", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
            selected_risk = st.selectbox("Filter by Risk Level", risk_levels)
        
        filtered_risk = risk_df.copy()
        
        if selected_state != "All States":
            filtered_risk = filtered_risk[filtered_risk['state'] == selected_state]
        
        if selected_risk != "All Levels":
            filtered_risk = filtered_risk[filtered_risk['risk_category'] == selected_risk]
        
        display_risk = filtered_risk[['state', 'district', 'pincode', 'risk_category', 
                                       'risk_reason', 'total_activity', 'trend_status']].copy()
        
        display_risk = display_risk.rename(columns={
            'state': 'State',
            'district': 'District',
            'pincode': 'Pincode',
            'risk_category': 'Risk Level',
            'risk_reason': 'Reason',
            'total_activity': 'Activity',
            'trend_status': 'Trend'
        })
        
        st.markdown(f"**Total Records:** {len(display_risk):,}")
        
        st.dataframe(display_risk.head(100), use_container_width=True, hide_index=True)
        
        if len(display_risk) > 100:
            st.info("Showing first 100 records. Download full report for all data.")
        
        # Download button
        csv = display_risk.to_csv(index=False)
        st.download_button(
            label="Download Risk Report (CSV)",
            data=csv,
            file_name=f"risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

elif report_type == "Full Data Export":
    st.markdown("### Full Data Export")
    
    st.markdown("""
    Export complete cleaned dataset for external analysis.
    
    **Available Columns:**
    - Location: State, District, Pincode
    - Time: Year-Month
    - Biometric Updates: Age 5-17, Age 18+, Total
    - Demographic Updates: Age 5-17, Age 18+, Total  
    - Enrollments: Age 0-5, Age 5-17, Age 18+, Total
    """)
    
    # State filter (optional)
    states = ["All India"] + get_states(df)
    selected_state = st.selectbox("Filter by State (Optional)", states)
    
    if selected_state == "All India":
        export_df = df.copy()
    else:
        export_df = df[df['state'] == selected_state].copy()
    
    export_df['year_month'] = export_df['year_month'].dt.strftime('%Y-%m')
    
    st.markdown(f"**Total Records:** {len(export_df):,}")
    st.markdown(f"**States:** {export_df['state'].nunique()}")
    st.markdown(f"**Districts:** {export_df['district'].nunique()}")
    st.markdown(f"**Pincodes:** {export_df['pincode'].nunique()}")
    
    # Preview
    st.markdown("**Preview (First 10 rows):**")
    st.dataframe(export_df.head(10), use_container_width=True, hide_index=True)
    
    # Download button
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Download Full Data (CSV)",
        data=csv,
        file_name=f"aadhaar_setu_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>{APP_NAME} | Reports</p>", unsafe_allow_html=True)
