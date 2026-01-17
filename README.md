# AADHAAR SETU

District-level dashboard for tracking Aadhaar service delivery across India. Built for administrators who need clear insights without technical jargon.

## What This Does

Shows Aadhaar activity (enrollments, biometric updates, demographic updates) at state, district, and pincode levels. Helps identify service gaps and unusual patterns that need attention.

## Quick Start

```bash
cd aadhaar_setu_portal
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

## Data Sources

The project uses three types of data:
- **Biometric updates** - fingerprint/iris scans
- **Demographic updates** - address/phone changes  
- **Enrollments** - new Aadhaar registrations

Data period: March 2025 - December 2025

## Project Structure

```
aadhar-dataset/
│
├── aadhaar_setu_portal/          # Main web dashboard
│   ├── app.py                    # Home page
│   ├── config.py                 # Settings
│   ├── data_loader.py            # Data handling
│   └── pages/                    # Dashboard pages
│
├── cleaned_data/                 # Processed datasets
│   ├── master_cleaned.csv        # Primary data
│   ├── biometric_cleaned.csv
│   ├── demographic_cleaned.csv
│   └── enrolment_cleaned.csv
│
├── ml_outputs/                   # Analysis results
│   ├── cluster_summary.csv       # Region groupings
│   ├── high_risk_regions.csv    # Areas needing attention
│   ├── monthly_alerts.csv       # Trend flags
│   └── state_summary.csv        # State-level stats
│
├── data_cleaning_pipeline.py    # Data processing
├── ml_pipeline.py               # Pattern detection
└── analysis.py                  # Summary stats
```

## Dashboard Pages

**Home** - Overview with key metrics and trends  
**District View** - Detailed breakdown by district  
**Pincode View** - Granular data for specific pincodes  
**Trends** - Month-over-month patterns  
**Alerts** - Regions with unusual activity  
**Reports** - Download CSV summaries  
**Power BI** - Embed advanced analytics (optional)  
**Settings** - Configuration and future API setup

## Running the Pipeline

### Clean raw data
```bash
python data_cleaning_pipeline.py
```
Output: `cleaned_data/master_cleaned.csv`

### Generate insights
```bash
python ml_pipeline.py
```
Output: Risk scores, clusters, alerts in `ml_outputs/`

### View summary stats
```bash
python analysis.py
```

## For Administrators

The dashboard is designed for non-technical users. Everything is in plain language with clear recommendations.

**Filter by location:** Use sidebar to select state/district/pincode  
**Understand colors:**
- Green = Normal activity
- Yellow = Minor concern
- Red = Needs attention

**Download reports:** Reports page has CSV exports for offline analysis

## Power BI Integration

Dashboard has a placeholder for Power BI reports. To embed:

1. Create Power BI dashboard using CSVs from `ml_outputs/`
2. Publish to Power BI Service
3. Get embed URL
4. Paste in Settings page

See `POWER_BI_GUIDE.md` for step-by-step instructions (if you're new to Power BI).

## Future: Live API Integration

The system is ready to switch from CSV files to live API when available. No UI changes needed - just configure endpoint in Settings page.

API handler code is in `aadhaar_setu_portal/data_loader.py` (currently returns CSV data).

## Data Quality

All datasets go through validation:
- Remove duplicates
- Fill missing values  
- Standardize state/district names
- Flag anomalies

Quality scores are included in the cleaned data.

## Technical Stack

- **Python 3.8+**
- **Streamlit** - Web interface
- **Pandas** - Data processing
- **Plotly** - Interactive charts
- **Scikit-learn** - Pattern detection

## Installation

```bash
# Clone repo
git clone <repo-url>
cd aadhar-dataset

# Install dependencies
pip install -r aadhaar_setu_portal/requirements.txt

# Run dashboard
cd aadhaar_setu_portal
streamlit run app.py
```

## Common Issues

**"Unable to load data"** - Check that `cleaned_data/` and `ml_outputs/` folders exist with CSV files

**"Module not found"** - Run `pip install -r requirements.txt` again

**Dashboard loads but shows empty charts** - Run `python ml_pipeline.py` first to generate outputs

## Contact

For questions about the dashboard or data issues, reach out to the development team.

## License

Government of India project for administrative use.
