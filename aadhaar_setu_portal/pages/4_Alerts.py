"""
AADHAAR SETU - Alerts Page
Risk areas and attention regions
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APP_NAME, COLORS, RISK_COLORS
from data_loader import load_master_data, load_risk_data, load_alerts_data, get_states

st.set_page_config(page_title=f"{APP_NAME} - Alerts", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%); padding: 1rem 1.5rem; margin: -1rem -1rem 1.5rem -1rem; color: white;">
    <h2 style="margin: 0; color: white;">Alerts and Risk Areas</h2>
    <p style="margin: 0.2rem 0 0 0; color: #ecf0f1; font-size: 0.9rem;">Regions requiring attention based on activity patterns</p>
</div>
""", unsafe_allow_html=True)

# Load data
risk_df = load_risk_data()
alerts_df = load_alerts_data()
master_df = load_master_data()

if risk_df.empty:
    st.warning("Risk data not available")
    st.stop()

# Overview metrics
st.markdown("### Alert Summary")

col1, col2, col3, col4 = st.columns(4)

critical_count = len(risk_df[risk_df['risk_category'] == 'CRITICAL'])
high_count = len(risk_df[risk_df['risk_category'] == 'HIGH'])
medium_count = len(risk_df[risk_df['risk_category'] == 'MEDIUM'])
low_count = len(risk_df[risk_df['risk_category'] == 'LOW'])

col1.metric("Critical", f"{critical_count:,}", delta=None)
col2.metric("High", f"{high_count:,}", delta=None)
col3.metric("Medium", f"{medium_count:,}", delta=None)
col4.metric("Low/Normal", f"{low_count:,}", delta=None)

st.markdown("---")

# Filter
states = ["All States"] + get_states(risk_df)
selected_state = st.selectbox("Filter by State", states)

if selected_state != "All States":
    filtered_risk = risk_df[risk_df['state'] == selected_state]
else:
    filtered_risk = risk_df.copy()

# Risk distribution
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Risk Distribution by State")
    
    state_risk = risk_df.groupby(['state', 'risk_category']).size().reset_index(name='count')
    state_risk = state_risk.pivot(index='state', columns='risk_category', values='count').fillna(0)
    
    # Sort by critical count
    if 'CRITICAL' in state_risk.columns:
        state_risk = state_risk.sort_values('CRITICAL', ascending=True).tail(15)
    
    fig = go.Figure()
    
    for risk_cat in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if risk_cat in state_risk.columns:
            fig.add_trace(go.Bar(
                y=state_risk.index,
                x=state_risk[risk_cat],
                name=risk_cat,
                orientation='h',
                marker_color=RISK_COLORS.get(risk_cat, COLORS['info'])
            ))
    
    fig.update_layout(
        barmode='stack',
        height=450,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Number of Pincodes",
        yaxis_title="",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Risk Categories")
    
    risk_counts = risk_df['risk_category'].value_counts()
    
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map=RISK_COLORS,
        hole=0.4
    )
    
    fig_pie.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("**What do risk levels mean?**")
    st.markdown("""
    - **CRITICAL:** Unusual activity spike (>50% increase)
    - **HIGH:** Significant change in activity pattern
    - **MEDIUM:** Minor anomaly detected
    - **LOW:** Normal activity, no concerns
    """)

# Critical Regions Table
st.markdown("### Critical Regions Requiring Attention")

critical_regions = filtered_risk[filtered_risk['risk_category'] == 'CRITICAL'].head(20)

if not critical_regions.empty:
    display_critical = critical_regions[['state', 'district', 'pincode', 'risk_reason', 
                                          'total_activity', 'trend_status']].copy()
    
    display_critical = display_critical.rename(columns={
        'state': 'State',
        'district': 'District',
        'pincode': 'Pincode',
        'risk_reason': 'Reason',
        'total_activity': 'Activity',
        'trend_status': 'Trend'
    })
    
    st.dataframe(display_critical, use_container_width=True, hide_index=True)
else:
    st.info("No critical regions in selected filter")

# High Risk Regions
st.markdown("### High Risk Regions")

high_regions = filtered_risk[filtered_risk['risk_category'] == 'HIGH'].head(20)

if not high_regions.empty:
    display_high = high_regions[['state', 'district', 'pincode', 'risk_reason', 
                                  'total_activity', 'trend_status']].copy()
    
    display_high = display_high.rename(columns={
        'state': 'State',
        'district': 'District',
        'pincode': 'Pincode',
        'risk_reason': 'Reason',
        'total_activity': 'Activity',
        'trend_status': 'Trend'
    })
    
    st.dataframe(display_high, use_container_width=True, hide_index=True)
else:
    st.info("No high risk regions in selected filter")

# Trend Status breakdown
if not alerts_df.empty:
    st.markdown("### Activity Trend Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trend_counts = alerts_df['trend_status'].value_counts()
        
        fig_trend = px.bar(
            x=trend_counts.index,
            y=trend_counts.values,
            color=trend_counts.index,
            color_discrete_map={
                'SURGING': COLORS['danger'],
                'DECLINING': COLORS['warning'],
                'STABLE': COLORS['success'],
                'GROWING': COLORS['info']
            }
        )
        
        fig_trend.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="Trend Status",
            yaxis_title="Count",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.markdown("**Trend Status Explained:**")
        st.markdown("""
        - **SURGING:** Activity increasing rapidly (may need investigation)
        - **GROWING:** Healthy growth in activity
        - **STABLE:** Consistent activity levels
        - **DECLINING:** Activity dropping (may need intervention)
        """)
        
        # Alert counts
        if 'drop_alert' in alerts_df.columns and 'spike_alert' in alerts_df.columns:
            drop_alerts = alerts_df['drop_alert'].sum()
            spike_alerts = alerts_df['spike_alert'].sum()
            
            st.metric("Drop Alerts", f"{drop_alerts:,.0f}")
            st.metric("Spike Alerts", f"{spike_alerts:,.0f}")

# Key Observations
st.markdown("### Key Observations")

top_critical_state = risk_df[risk_df['risk_category'] == 'CRITICAL']['state'].value_counts().head(1)
if not top_critical_state.empty:
    st.warning(f"**{top_critical_state.index[0]}** has the highest number of critical alerts ({top_critical_state.values[0]} pincodes)")

surging_count = len(risk_df[risk_df['trend_status'] == 'SURGING'])
declining_count = len(risk_df[risk_df['trend_status'] == 'DECLINING'])

if surging_count > declining_count:
    st.info(f"{surging_count:,} pincodes show surging activity patterns. Review these for unusual spikes.")
else:
    st.info(f"{declining_count:,} pincodes show declining activity. These may need service improvement.")

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>{APP_NAME} | Alerts & Risk Analysis</p>", unsafe_allow_html=True)
