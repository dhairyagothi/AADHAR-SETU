"""
AADHAAR SETU - Pincode View Page
Detailed pincode-level analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APP_NAME, COLORS
from data_loader import load_master_data, get_states, get_districts, get_pincodes, filter_data

st.set_page_config(page_title=f"{APP_NAME} - Pincode View", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%); padding: 1rem 1.5rem; margin: -1rem -1rem 1.5rem -1rem; color: white;">
    <h2 style="margin: 0; color: white;">Pincode Analysis</h2>
    <p style="margin: 0.2rem 0 0 0; color: #ecf0f1; font-size: 0.9rem;">Granular view of Aadhaar activity at pincode level</p>
</div>
""", unsafe_allow_html=True)

# Load data
df = load_master_data()

if df.empty:
    st.error("Unable to load data")
    st.stop()

# Filters
col1, col2, col3 = st.columns(3)

with col1:
    states = get_states(df)
    selected_state = st.selectbox("Select State", states, key="pincode_state")

with col2:
    districts = get_districts(df, selected_state)
    selected_district = st.selectbox("Select District", districts, key="pincode_district")

with col3:
    pincodes = get_pincodes(df, selected_state, selected_district)
    selected_pincode = st.selectbox("Select Pincode", pincodes, key="pincode_pincode")

# Filter data for selected pincode
filtered_df = df[(df['state'] == selected_state) & 
                 (df['district'] == selected_district) & 
                 (df['pincode'] == selected_pincode)]

st.markdown("---")

if filtered_df.empty:
    st.warning("No data available for selected pincode")
    st.stop()

# Pincode Summary Metrics
st.markdown(f"### Pincode: {selected_pincode}")

col1, col2, col3, col4 = st.columns(4)

total_bio = filtered_df['total_activity_bio'].sum()
total_demo = filtered_df['total_activity_demo'].sum()
total_enroll = filtered_df['enrollments_age_0_5'].sum() + filtered_df['enrollments_age_5_17'].sum() + filtered_df['enrollments_age_18_plus'].sum()
total_months = filtered_df['year_month'].nunique()

col1.metric("Biometric Updates", f"{total_bio:,.0f}")
col2.metric("Demographic Updates", f"{total_demo:,.0f}")
col3.metric("Enrollments", f"{total_enroll:,.0f}")
col4.metric("Months of Data", f"{total_months}")

st.markdown("<br>", unsafe_allow_html=True)

# Monthly Trend for this Pincode
st.markdown("### Monthly Activity Timeline")

monthly = filtered_df.sort_values('year_month')

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=monthly['year_month'],
    y=monthly['total_activity_bio'],
    name='Biometric Updates',
    line=dict(color=COLORS['primary'], width=3),
    mode='lines+markers',
    marker=dict(size=8)
))

fig.add_trace(go.Scatter(
    x=monthly['year_month'],
    y=monthly['total_activity_demo'],
    name='Demographic Updates',
    line=dict(color=COLORS['secondary'], width=3),
    mode='lines+markers',
    marker=dict(size=8)
))

enrollments = monthly['enrollments_age_0_5'] + monthly['enrollments_age_5_17'] + monthly['enrollments_age_18_plus']
fig.add_trace(go.Scatter(
    x=monthly['year_month'],
    y=enrollments,
    name='Enrollments',
    line=dict(color=COLORS['success'], width=3),
    mode='lines+markers',
    marker=dict(size=8)
))

fig.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Month",
    yaxis_title="Count",
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Detailed breakdown
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Biometric Updates Breakdown")
    
    bio_5_17 = filtered_df['updates_age_5_17_bio'].sum()
    bio_18_plus = filtered_df['updates_age_18_plus_bio'].sum()
    
    bio_data = pd.DataFrame({
        'Age Group': ['5-17 Years', '18+ Years'],
        'Count': [bio_5_17, bio_18_plus]
    })
    
    fig_bio = px.bar(
        bio_data,
        x='Age Group',
        y='Count',
        color='Age Group',
        color_discrete_sequence=[COLORS['info'], COLORS['primary']]
    )
    
    fig_bio.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_bio, use_container_width=True)
    
    # Stats
    st.markdown(f"**Total Biometric Updates:** {total_bio:,.0f}")
    if bio_5_17 + bio_18_plus > 0:
        st.markdown(f"- Children (5-17): {bio_5_17:,.0f} ({bio_5_17/(bio_5_17+bio_18_plus)*100:.1f}%)")
        st.markdown(f"- Adults (18+): {bio_18_plus:,.0f} ({bio_18_plus/(bio_5_17+bio_18_plus)*100:.1f}%)")

with col2:
    st.markdown("### Demographic Updates Breakdown")
    
    demo_5_17 = filtered_df['updates_age_5_17_demo'].sum()
    demo_18_plus = filtered_df['updates_age_18_plus_demo'].sum()
    
    demo_data = pd.DataFrame({
        'Age Group': ['5-17 Years', '18+ Years'],
        'Count': [demo_5_17, demo_18_plus]
    })
    
    fig_demo = px.bar(
        demo_data,
        x='Age Group',
        y='Count',
        color='Age Group',
        color_discrete_sequence=[COLORS['warning'], COLORS['secondary']]
    )
    
    fig_demo.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_demo, use_container_width=True)
    
    # Stats
    st.markdown(f"**Total Demographic Updates:** {total_demo:,.0f}")
    if demo_5_17 + demo_18_plus > 0:
        st.markdown(f"- Children (5-17): {demo_5_17:,.0f} ({demo_5_17/(demo_5_17+demo_18_plus)*100:.1f}%)")
        st.markdown(f"- Adults (18+): {demo_18_plus:,.0f} ({demo_18_plus/(demo_5_17+demo_18_plus)*100:.1f}%)")

# Enrollment breakdown
st.markdown("### Enrollment Details")

enroll_0_5 = filtered_df['enrollments_age_0_5'].sum()
enroll_5_17 = filtered_df['enrollments_age_5_17'].sum()
enroll_18_plus = filtered_df['enrollments_age_18_plus'].sum()

col1, col2, col3 = st.columns(3)

col1.metric("Infants (0-5 Years)", f"{enroll_0_5:,.0f}")
col2.metric("Children (5-17 Years)", f"{enroll_5_17:,.0f}")
col3.metric("Adults (18+ Years)", f"{enroll_18_plus:,.0f}")

# Monthly data table
st.markdown("### Monthly Data")

display_df = filtered_df[['year_month', 'updates_age_5_17_bio', 'updates_age_18_plus_bio',
                          'updates_age_5_17_demo', 'updates_age_18_plus_demo',
                          'enrollments_age_0_5', 'enrollments_age_5_17', 'enrollments_age_18_plus']].copy()

display_df = display_df.rename(columns={
    'year_month': 'Month',
    'updates_age_5_17_bio': 'Bio (5-17)',
    'updates_age_18_plus_bio': 'Bio (18+)',
    'updates_age_5_17_demo': 'Demo (5-17)',
    'updates_age_18_plus_demo': 'Demo (18+)',
    'enrollments_age_0_5': 'Enroll (0-5)',
    'enrollments_age_5_17': 'Enroll (5-17)',
    'enrollments_age_18_plus': 'Enroll (18+)'
})

display_df['Month'] = display_df['Month'].dt.strftime('%Y-%m')
display_df = display_df.sort_values('Month', ascending=False)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Insights
st.markdown("### Observations")

# Calculate trends
if len(monthly) >= 2:
    first_month_bio = monthly.iloc[0]['total_activity_bio']
    last_month_bio = monthly.iloc[-1]['total_activity_bio']
    
    if last_month_bio > first_month_bio * 1.2:
        st.success("Biometric updates have increased over the period. Activity is growing.")
    elif last_month_bio < first_month_bio * 0.8:
        st.warning("Biometric updates have decreased. This pincode may need attention.")
    else:
        st.info("Biometric update activity is stable across the period.")

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>{APP_NAME} | Pincode Analysis</p>", unsafe_allow_html=True)
