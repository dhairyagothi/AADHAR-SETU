"""
AADHAAR SETU - District View Page
Detailed district-level analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APP_NAME, COLORS
from data_loader import load_master_data, get_states, get_districts, filter_data

st.set_page_config(page_title=f"{APP_NAME} - District View", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%); padding: 1rem 1.5rem; margin: -1rem -1rem 1.5rem -1rem; color: white;">
    <h2 style="margin: 0; color: white;">District Analysis</h2>
    <p style="margin: 0.2rem 0 0 0; color: #ecf0f1; font-size: 0.9rem;">Detailed view of Aadhaar activity at district level</p>
</div>
""", unsafe_allow_html=True)

# Load data
df = load_master_data()

if df.empty:
    st.error("Unable to load data")
    st.stop()

# Filters
col1, col2 = st.columns(2)

with col1:
    states = get_states(df)
    selected_state = st.selectbox("Select State", states, key="district_state")

with col2:
    districts = get_districts(df, selected_state)
    selected_district = st.selectbox("Select District", districts, key="district_district")

# Filter data
filtered_df = filter_data(df, selected_state, selected_district)

st.markdown("---")

# District Summary Metrics
col1, col2, col3, col4, col5 = st.columns(5)

total_bio = filtered_df['total_activity_bio'].sum()
total_demo = filtered_df['total_activity_demo'].sum()
total_enroll = filtered_df['enrollments_age_0_5'].sum() + filtered_df['enrollments_age_5_17'].sum() + filtered_df['enrollments_age_18_plus'].sum()
total_pincodes = filtered_df['pincode'].nunique()
total_months = filtered_df['year_month'].nunique()

col1.metric("Total Biometric Updates", f"{total_bio:,.0f}")
col2.metric("Total Demographic Updates", f"{total_demo:,.0f}")
col3.metric("Total Enrollments", f"{total_enroll:,.0f}")
col4.metric("Pincodes", f"{total_pincodes:,}")
col5.metric("Data Points (Months)", f"{total_months}")

st.markdown("<br>", unsafe_allow_html=True)

# Monthly Trend for this District
st.markdown("### Monthly Activity Trend")

monthly = filtered_df.groupby('year_month').agg({
    'total_activity_bio': 'sum',
    'total_activity_demo': 'sum',
    'enrollments_age_0_5': 'sum',
    'enrollments_age_5_17': 'sum',
    'enrollments_age_18_plus': 'sum'
}).reset_index()

monthly['Enrollments'] = monthly['enrollments_age_0_5'] + monthly['enrollments_age_5_17'] + monthly['enrollments_age_18_plus']

fig = go.Figure()

fig.add_trace(go.Bar(
    x=monthly['year_month'],
    y=monthly['total_activity_bio'],
    name='Biometric Updates',
    marker_color=COLORS['primary']
))

fig.add_trace(go.Bar(
    x=monthly['year_month'],
    y=monthly['total_activity_demo'],
    name='Demographic Updates',
    marker_color=COLORS['secondary']
))

fig.add_trace(go.Bar(
    x=monthly['year_month'],
    y=monthly['Enrollments'],
    name='Enrollments',
    marker_color=COLORS['success']
))

fig.update_layout(
    barmode='group',
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Month",
    yaxis_title="Count",
    plot_bgcolor='white',
    paper_bgcolor='white'
)

st.plotly_chart(fig, use_container_width=True)

# Two columns: Age distribution and Update types
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Update Types Distribution")
    
    bio_5_17 = filtered_df['updates_age_5_17_bio'].sum()
    bio_18_plus = filtered_df['updates_age_18_plus_bio'].sum()
    demo_5_17 = filtered_df['updates_age_5_17_demo'].sum()
    demo_18_plus = filtered_df['updates_age_18_plus_demo'].sum()
    
    update_data = pd.DataFrame({
        'Category': ['Biometric (5-17 yrs)', 'Biometric (18+ yrs)', 
                    'Demographic (5-17 yrs)', 'Demographic (18+ yrs)'],
        'Count': [bio_5_17, bio_18_plus, demo_5_17, demo_18_plus]
    })
    
    fig_bar = px.bar(
        update_data,
        x='Category',
        y='Count',
        color='Category',
        color_discrete_sequence=[COLORS['info'], COLORS['primary'], 
                                COLORS['warning'], COLORS['secondary']]
    )
    
    fig_bar.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="",
        yaxis_title="Count",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### Enrollment by Age Group")
    
    enroll_0_5 = filtered_df['enrollments_age_0_5'].sum()
    enroll_5_17 = filtered_df['enrollments_age_5_17'].sum()
    enroll_18_plus = filtered_df['enrollments_age_18_plus'].sum()
    
    enroll_data = pd.DataFrame({
        'Age Group': ['0-5 Years', '5-17 Years', '18+ Years'],
        'Count': [enroll_0_5, enroll_5_17, enroll_18_plus]
    })
    
    fig_pie = px.pie(
        enroll_data,
        values='Count',
        names='Age Group',
        color_discrete_sequence=[COLORS['success'], COLORS['info'], COLORS['primary']],
        hole=0.4
    )
    
    fig_pie.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

# Pincode-wise breakdown
st.markdown("### Pincode-wise Activity")

pincode_summary = filtered_df.groupby('pincode').agg({
    'total_activity_bio': 'sum',
    'total_activity_demo': 'sum',
    'enrollments_age_0_5': 'sum',
    'enrollments_age_5_17': 'sum',
    'enrollments_age_18_plus': 'sum'
}).reset_index()

pincode_summary['Total Enrollments'] = pincode_summary['enrollments_age_0_5'] + pincode_summary['enrollments_age_5_17'] + pincode_summary['enrollments_age_18_plus']
pincode_summary['Total Activity'] = pincode_summary['total_activity_bio'] + pincode_summary['total_activity_demo'] + pincode_summary['Total Enrollments']

pincode_summary = pincode_summary.rename(columns={
    'pincode': 'Pincode',
    'total_activity_bio': 'Biometric Updates',
    'total_activity_demo': 'Demographic Updates'
})

display_cols = ['Pincode', 'Biometric Updates', 'Demographic Updates', 'Total Enrollments', 'Total Activity']
pincode_summary = pincode_summary[display_cols].sort_values('Total Activity', ascending=False)

st.dataframe(pincode_summary, use_container_width=True, hide_index=True)

# Key Insights
st.markdown("### Key Observations")

avg_activity = pincode_summary['Total Activity'].mean()
max_pincode = pincode_summary.loc[pincode_summary['Total Activity'].idxmax(), 'Pincode']
min_pincode = pincode_summary.loc[pincode_summary['Total Activity'].idxmin(), 'Pincode']

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.info(f"**Highest Activity Pincode:** {int(max_pincode)} with {pincode_summary['Total Activity'].max():,.0f} total transactions")
    
with insights_col2:
    st.warning(f"**Lowest Activity Pincode:** {int(min_pincode)} with {pincode_summary['Total Activity'].min():,.0f} total transactions")

if total_bio > total_demo:
    st.success(f"This district has more biometric updates ({total_bio:,.0f}) than demographic updates ({total_demo:,.0f}), indicating active identity verification.")
else:
    st.info(f"This district has more demographic updates ({total_demo:,.0f}) than biometric updates ({total_bio:,.0f}), indicating address/mobile updates are common.")

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>{APP_NAME} | District Analysis</p>", unsafe_allow_html=True)
