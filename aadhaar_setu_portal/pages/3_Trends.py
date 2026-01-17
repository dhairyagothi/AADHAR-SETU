"""
AADHAAR SETU - Trends Page
Monthly trends and patterns
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

st.set_page_config(page_title=f"{APP_NAME} - Trends", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%); padding: 1rem 1.5rem; margin: -1rem -1rem 1.5rem -1rem; color: white;">
    <h2 style="margin: 0; color: white;">Monthly Trends</h2>
    <p style="margin: 0.2rem 0 0 0; color: #ecf0f1; font-size: 0.9rem;">Track activity patterns over time</p>
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
    states = ["All India"] + get_states(df)
    selected_state = st.selectbox("Select State", states, key="trends_state")

with col2:
    if selected_state != "All India":
        districts = ["All Districts"] + get_districts(df, selected_state)
    else:
        districts = ["All Districts"]
    selected_district = st.selectbox("Select District", districts, key="trends_district")

# Filter data
if selected_state == "All India":
    filtered_df = df.copy()
elif selected_district == "All Districts":
    filtered_df = filter_data(df, selected_state)
else:
    filtered_df = filter_data(df, selected_state, selected_district)

st.markdown("---")

# Aggregate by month
monthly = filtered_df.groupby('year_month').agg({
    'total_activity_bio': 'sum',
    'total_activity_demo': 'sum',
    'enrollments_age_0_5': 'sum',
    'enrollments_age_5_17': 'sum',
    'enrollments_age_18_plus': 'sum',
    'updates_age_5_17_bio': 'sum',
    'updates_age_18_plus_bio': 'sum',
    'updates_age_5_17_demo': 'sum',
    'updates_age_18_plus_demo': 'sum',
    'pincode': 'nunique'
}).reset_index()

monthly['Total Enrollments'] = monthly['enrollments_age_0_5'] + monthly['enrollments_age_5_17'] + monthly['enrollments_age_18_plus']
monthly['Total Activity'] = monthly['total_activity_bio'] + monthly['total_activity_demo'] + monthly['Total Enrollments']
monthly = monthly.sort_values('year_month')

# Overview metrics
st.markdown("### Period Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Activity", f"{monthly['Total Activity'].sum():,.0f}")
col2.metric("Avg Monthly Activity", f"{monthly['Total Activity'].mean():,.0f}")
col3.metric("Peak Month", monthly.loc[monthly['Total Activity'].idxmax(), 'year_month'].strftime('%B %Y'))
col4.metric("Active Pincodes", f"{filtered_df['pincode'].nunique():,}")

st.markdown("<br>", unsafe_allow_html=True)

# Main Trend Chart
st.markdown("### Activity Trend (March 2025 - December 2025)")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=monthly['year_month'],
    y=monthly['total_activity_bio'],
    name='Biometric Updates',
    fill='tozeroy',
    line=dict(color=COLORS['primary'], width=2),
    mode='lines'
))

fig.add_trace(go.Scatter(
    x=monthly['year_month'],
    y=monthly['total_activity_demo'],
    name='Demographic Updates',
    fill='tozeroy',
    line=dict(color=COLORS['secondary'], width=2),
    mode='lines'
))

fig.add_trace(go.Scatter(
    x=monthly['year_month'],
    y=monthly['Total Enrollments'],
    name='Enrollments',
    fill='tozeroy',
    line=dict(color=COLORS['success'], width=2),
    mode='lines'
))

fig.update_layout(
    height=450,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Month",
    yaxis_title="Count",
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Month-over-Month Changes
st.markdown("### Month-over-Month Change")

monthly['Bio_Change'] = monthly['total_activity_bio'].pct_change() * 100
monthly['Demo_Change'] = monthly['total_activity_demo'].pct_change() * 100
monthly['Enroll_Change'] = monthly['Total Enrollments'].pct_change() * 100

change_df = monthly[['year_month', 'Bio_Change', 'Demo_Change', 'Enroll_Change']].dropna()

fig_change = go.Figure()

fig_change.add_trace(go.Bar(
    x=change_df['year_month'],
    y=change_df['Bio_Change'],
    name='Biometric',
    marker_color=COLORS['primary']
))

fig_change.add_trace(go.Bar(
    x=change_df['year_month'],
    y=change_df['Demo_Change'],
    name='Demographic',
    marker_color=COLORS['secondary']
))

fig_change.add_trace(go.Bar(
    x=change_df['year_month'],
    y=change_df['Enroll_Change'],
    name='Enrollments',
    marker_color=COLORS['success']
))

fig_change.add_hline(y=0, line_dash="dash", line_color="gray")

fig_change.update_layout(
    barmode='group',
    height=350,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Month",
    yaxis_title="% Change",
    plot_bgcolor='white',
    paper_bgcolor='white'
)

st.plotly_chart(fig_change, use_container_width=True)

# Age-wise trends
st.markdown("### Age-wise Trends")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Biometric Updates by Age Group**")
    
    fig_bio_age = go.Figure()
    
    fig_bio_age.add_trace(go.Scatter(
        x=monthly['year_month'],
        y=monthly['updates_age_5_17_bio'],
        name='5-17 Years',
        line=dict(color=COLORS['info'], width=2),
        mode='lines+markers'
    ))
    
    fig_bio_age.add_trace(go.Scatter(
        x=monthly['year_month'],
        y=monthly['updates_age_18_plus_bio'],
        name='18+ Years',
        line=dict(color=COLORS['primary'], width=2),
        mode='lines+markers'
    ))
    
    fig_bio_age.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_bio_age, use_container_width=True)

with col2:
    st.markdown("**Enrollment by Age Group**")
    
    fig_enroll_age = go.Figure()
    
    fig_enroll_age.add_trace(go.Scatter(
        x=monthly['year_month'],
        y=monthly['enrollments_age_0_5'],
        name='0-5 Years',
        line=dict(color=COLORS['success'], width=2),
        mode='lines+markers'
    ))
    
    fig_enroll_age.add_trace(go.Scatter(
        x=monthly['year_month'],
        y=monthly['enrollments_age_5_17'],
        name='5-17 Years',
        line=dict(color=COLORS['info'], width=2),
        mode='lines+markers'
    ))
    
    fig_enroll_age.add_trace(go.Scatter(
        x=monthly['year_month'],
        y=monthly['enrollments_age_18_plus'],
        name='18+ Years',
        line=dict(color=COLORS['primary'], width=2),
        mode='lines+markers'
    ))
    
    fig_enroll_age.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_enroll_age, use_container_width=True)

# Monthly data table
st.markdown("### Monthly Summary Table")

display_monthly = monthly[['year_month', 'total_activity_bio', 'total_activity_demo', 
                           'Total Enrollments', 'Total Activity', 'pincode']].copy()

display_monthly['year_month'] = display_monthly['year_month'].dt.strftime('%B %Y')
display_monthly = display_monthly.rename(columns={
    'year_month': 'Month',
    'total_activity_bio': 'Biometric Updates',
    'total_activity_demo': 'Demographic Updates',
    'pincode': 'Active Pincodes'
})

display_monthly = display_monthly.sort_values('Month', ascending=False)

st.dataframe(display_monthly, use_container_width=True, hide_index=True)

# Trend Insights
st.markdown("### Key Observations")

# Calculate insights
peak_month = monthly.loc[monthly['Total Activity'].idxmax()]
low_month = monthly.loc[monthly['Total Activity'].idxmin()]

col1, col2 = st.columns(2)

with col1:
    st.success(f"**Peak Activity:** {peak_month['year_month'].strftime('%B %Y')} with {peak_month['Total Activity']:,.0f} total transactions")

with col2:
    st.warning(f"**Lowest Activity:** {low_month['year_month'].strftime('%B %Y')} with {low_month['Total Activity']:,.0f} total transactions")

# Overall trend
first_total = monthly.iloc[0]['Total Activity']
last_total = monthly.iloc[-1]['Total Activity']
change_pct = ((last_total - first_total) / first_total) * 100 if first_total > 0 else 0

if change_pct > 10:
    st.info(f"Overall activity has **increased by {change_pct:.1f}%** from March to December 2025.")
elif change_pct < -10:
    st.warning(f"Overall activity has **decreased by {abs(change_pct):.1f}%** from March to December 2025.")
else:
    st.info(f"Overall activity has remained **stable** (change: {change_pct:.1f}%) from March to December 2025.")

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>{APP_NAME} | Trends Analysis</p>", unsafe_allow_html=True)
