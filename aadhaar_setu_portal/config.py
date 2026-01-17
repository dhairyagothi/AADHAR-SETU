"""
AADHAAR SETU - Configuration
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "cleaned_data"
ML_DIR = BASE_DIR.parent / "ml_outputs"

# Data Files
MASTER_DATA = DATA_DIR / "master_cleaned.csv"
CLUSTER_DATA = ML_DIR / "cluster_summary.csv"
RISK_DATA = ML_DIR / "high_risk_regions.csv"
ALERTS_DATA = ML_DIR / "monthly_alerts.csv"
STATE_DATA = ML_DIR / "state_summary.csv"

# App Settings
APP_NAME = "AADHAAR SETU"
APP_SUBTITLE = "District Administration Dashboard"
VERSION = "1.0.0"

# Colors (Professional)
COLORS = {
    "primary": "#1a5276",
    "secondary": "#2980b9",
    "success": "#27ae60",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
    "light": "#ecf0f1",
    "dark": "#2c3e50",
    "white": "#ffffff"
}

# Risk Level Colors
RISK_COLORS = {
    "CRITICAL": "#c0392b",
    "HIGH": "#e74c3c",
    "MEDIUM": "#f39c12",
    "LOW": "#27ae60",
    "NORMAL": "#3498db"
}

# Column Mappings (for future API compatibility)
COLUMNS = {
    "year_month": "Period",
    "state": "State",
    "district": "District",
    "pincode": "Pincode",
    "updates_age_5_17_bio": "Biometric Updates (5-17 yrs)",
    "updates_age_18_plus_bio": "Biometric Updates (18+ yrs)",
    "total_activity_bio": "Total Biometric Updates",
    "updates_age_5_17_demo": "Demographic Updates (5-17 yrs)",
    "updates_age_18_plus_demo": "Demographic Updates (18+ yrs)",
    "total_activity_demo": "Total Demographic Updates",
    "enrollments_age_0_5": "Enrollments (0-5 yrs)",
    "enrollments_age_5_17": "Enrollments (5-17 yrs)",
    "enrollments_age_18_plus": "Enrollments (18+ yrs)",
    "total_activity": "Total Enrollments"
}
