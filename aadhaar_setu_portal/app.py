"""
AADHAAR SETU - Main Application
District Administration Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_NAME, APP_SUBTITLE, COLORS, RISK_COLORS, COLUMNS
from data_loader import (
    load_master_data, load_risk_data, load_alerts_data, load_state_summary,
    get_states, get_districts, get_pincodes, filter_data
)

# Page Configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        padding: 1.5rem 2rem;
        border-radius: 0;
        margin: -1rem -1rem 1.5rem -1rem;
        color: white;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .main-header p {
        color: #ecf0f1;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1a5276;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 0.8rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a5276;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #7f8c8d;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #1a5276;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Insight box */
    .insight-box {
        background: #eef6fc;
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Alert boxes */
    .alert-critical {
        background: #fdf2f2;
        border-left: 4px solid #e74c3c;
        padding: 0.8rem;
        border-radius: 4px;
    }
    
    .alert-warning {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        padding: 0.8rem;
        border-radius: 4px;
    }
    
    .alert-success {
        background: #eafaf1;
        border-left: 4px solid #27ae60;
        padding: 0.8rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render main header"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{APP_NAME}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(value, label, prefix="", suffix=""):
    """Render a metric card"""
    formatted_value = f"{prefix}{value:,.0f}{suffix}" if isinstance(value, (int, float)) else f"{prefix}{value}{suffix}"
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{formatted_value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(text):
    """Render section header"""
    st.markdown(f'<p class="section-header">{text}</p>', unsafe_allow_html=True)


def render_insight(text, type="info"):
    """Render insight box"""
    colors = {
        "info": ("#eef6fc", "#3498db"),
        "success": ("#eafaf1", "#27ae60"),
        "warning": ("#fef9e7", "#f39c12"),
        "danger": ("#fdf2f2", "#e74c3c")
    }
    bg, border = colors.get(type, colors["info"])
    st.markdown(f"""
    <div style="background: {bg}; border-left: 4px solid {border}; padding: 1rem; border-radius: 4px; margin: 1rem 0;">
        {text}
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Load data
    df = load_master_data()
    risk_df = load_risk_data()
    state_df = load_state_summary()
    
    if df.empty:
        st.error("Unable to load data. Please check data files.")
        return
    
    # Header
    render_header()
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        
        # State filter
        states = ["All States"] + get_states(df)
        selected_state = st.selectbox("State", states, index=0)
        
        # District filter
        if selected_state != "All States":
            districts = ["All Districts"] + get_districts(df, selected_state)
        else:
            districts = ["All Districts"]
        selected_district = st.selectbox("District", districts, index=0)
        
        # Pincode filter
        if selected_district != "All Districts":
            pincodes = ["All Pincodes"] + get_pincodes(df, selected_state, selected_district)
        else:
            pincodes = ["All Pincodes"]
        selected_pincode = st.selectbox("Pincode", pincodes, index=0)
        
        st.markdown("---")
        st.markdown("### Data Period")
        st.info("March 2025 - December 2025")
        
        st.markdown("---")
        st.markdown("### Quick Links")
        st.page_link("pages/1_District_View.py", label="District Analysis")
        st.page_link("pages/2_Pincode_View.py", label="Pincode Details")
        st.page_link("pages/3_Trends.py", label="Monthly Trends")
        st.page_link("pages/4_Alerts.py", label="Alerts & Risks")
        st.page_link("pages/5_Reports.py", label="Reports")
    
    # Filter data
    filtered_df = filter_data(df, selected_state, selected_district, 
                              selected_pincode if selected_pincode != "All Pincodes" else None)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_bio = filtered_df['total_activity_bio'].sum()
    total_demo = filtered_df['total_activity_demo'].sum()
    total_enroll = filtered_df['enrollments_age_0_5'].sum() + filtered_df['enrollments_age_5_17'].sum() + filtered_df['enrollments_age_18_plus'].sum()
    total_pincodes = filtered_df['pincode'].nunique()
    
    with col1:
        render_metric_card(total_bio, "Biometric Updates")
    with col2:
        render_metric_card(total_demo, "Demographic Updates")
    with col3:
        render_metric_card(total_enroll, "New Enrollments")
    with col4:
        render_metric_card(total_pincodes, "Pincodes Covered")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two column layout
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        render_section_header("Monthly Activity Trend")
        
        # Aggregate by month
        monthly = filtered_df.groupby('year_month').agg({
            'total_activity_bio': 'sum',
            'total_activity_demo': 'sum',
            'enrollments_age_0_5': 'sum',
            'enrollments_age_5_17': 'sum',
            'enrollments_age_18_plus': 'sum'
        }).reset_index()
        
        monthly['Total Enrollments'] = monthly['enrollments_age_0_5'] + monthly['enrollments_age_5_17'] + monthly['enrollments_age_18_plus']
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly['year_month'],
            y=monthly['total_activity_bio'],
            name='Biometric Updates',
            line=dict(color=COLORS['primary'], width=2),
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly['year_month'],
            y=monthly['total_activity_demo'],
            name='Demographic Updates',
            line=dict(color=COLORS['secondary'], width=2),
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly['year_month'],
            y=monthly['Total Enrollments'],
            name='Enrollments',
            line=dict(color=COLORS['success'], width=2),
            mode='lines+markers'
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="",
            yaxis_title="Count",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        render_section_header("Activity Breakdown")
        
        # Pie chart for activity types
        activity_data = {
            'Type': ['Biometric Updates', 'Demographic Updates', 'Enrollments'],
            'Count': [total_bio, total_demo, total_enroll]
        }
        
        fig_pie = px.pie(
            activity_data,
            values='Count',
            names='Type',
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['success']],
            hole=0.4
        )
        
        fig_pie.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Age-wise breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Age-wise Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Biometric Updates by Age**")
        bio_5_17 = filtered_df['updates_age_5_17_bio'].sum()
        bio_18_plus = filtered_df['updates_age_18_plus_bio'].sum()
        
        fig_bio = px.bar(
            x=['5-17 Years', '18+ Years'],
            y=[bio_5_17, bio_18_plus],
            color=['5-17 Years', '18+ Years'],
            color_discrete_map={'5-17 Years': COLORS['info'], '18+ Years': COLORS['primary']}
        )
        fig_bio.update_layout(
            height=250,
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="",
            yaxis_title="Count",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_bio, use_container_width=True)
    
    with col2:
        st.markdown("**Demographic Updates by Age**")
        demo_5_17 = filtered_df['updates_age_5_17_demo'].sum()
        demo_18_plus = filtered_df['updates_age_18_plus_demo'].sum()
        
        fig_demo = px.bar(
            x=['5-17 Years', '18+ Years'],
            y=[demo_5_17, demo_18_plus],
            color=['5-17 Years', '18+ Years'],
            color_discrete_map={'5-17 Years': COLORS['warning'], '18+ Years': COLORS['secondary']}
        )
        fig_demo.update_layout(
            height=250,
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="",
            yaxis_title="Count",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_demo, use_container_width=True)
    
    with col3:
        st.markdown("**Enrollments by Age**")
        enroll_0_5 = filtered_df['enrollments_age_0_5'].sum()
        enroll_5_17 = filtered_df['enrollments_age_5_17'].sum()
        enroll_18_plus = filtered_df['enrollments_age_18_plus'].sum()
        
        fig_enroll = px.bar(
            x=['0-5 Yrs', '5-17 Yrs', '18+ Yrs'],
            y=[enroll_0_5, enroll_5_17, enroll_18_plus],
            color=['0-5 Yrs', '5-17 Yrs', '18+ Yrs'],
            color_discrete_map={'0-5 Yrs': COLORS['success'], '5-17 Yrs': COLORS['info'], '18+ Yrs': COLORS['primary']}
        )
        fig_enroll.update_layout(
            height=250,
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="",
            yaxis_title="Count",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_enroll, use_container_width=True)
    
    # Insights section
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Key Insights")
    
    # Generate dynamic insights
    insights = []
    
    if total_bio > total_demo:
        insights.append(f"Biometric updates ({total_bio:,.0f}) exceed demographic updates ({total_demo:,.0f}). This indicates active identity verification activity.")
    else:
        insights.append(f"Demographic updates ({total_demo:,.0f}) exceed biometric updates ({total_bio:,.0f}). Address/mobile updates are more common.")
    
    if total_enroll > 0:
        child_enroll = enroll_0_5 + enroll_5_17
        adult_enroll = enroll_18_plus
        if child_enroll > adult_enroll:
            insights.append(f"Child enrollments ({child_enroll:,.0f}) are higher than adult enrollments ({adult_enroll:,.0f}).")
        else:
            insights.append(f"Adult enrollments ({adult_enroll:,.0f}) are higher than child enrollments ({child_enroll:,.0f}).")
    
    if not risk_df.empty and selected_state != "All States":
        state_risk = risk_df[risk_df['state'] == selected_state]
        critical_count = len(state_risk[state_risk['risk_category'] == 'CRITICAL'])
        if critical_count > 0:
            insights.append(f"Alert: {critical_count} pincodes in this state require immediate attention due to unusual activity patterns.")
    
    for insight in insights:
        render_insight(insight)
    
    # Top Districts Table (if viewing state level)
    if selected_state != "All States" and selected_district == "All Districts":
        st.markdown("<br>", unsafe_allow_html=True)
        render_section_header("District-wise Summary")
        
        district_summary = filtered_df.groupby('district').agg({
            'pincode': 'nunique',
            'total_activity_bio': 'sum',
            'total_activity_demo': 'sum',
            'enrollments_age_0_5': 'sum',
            'enrollments_age_5_17': 'sum',
            'enrollments_age_18_plus': 'sum'
        }).reset_index()
        
        district_summary['Total Enrollments'] = district_summary['enrollments_age_0_5'] + district_summary['enrollments_age_5_17'] + district_summary['enrollments_age_18_plus']
        district_summary['Total Activity'] = district_summary['total_activity_bio'] + district_summary['total_activity_demo'] + district_summary['Total Enrollments']
        
        district_summary = district_summary.rename(columns={
            'district': 'District',
            'pincode': 'Pincodes',
            'total_activity_bio': 'Biometric',
            'total_activity_demo': 'Demographic'
        })
        
        display_cols = ['District', 'Pincodes', 'Biometric', 'Demographic', 'Total Enrollments', 'Total Activity']
        district_summary = district_summary[display_cols].sort_values('Total Activity', ascending=False)
        
        st.dataframe(
            district_summary.head(15),
            use_container_width=True,
            hide_index=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>"
        f"{APP_NAME} | Data Period: March 2025 - December 2025 | Last Updated: January 2026"
        f"</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
