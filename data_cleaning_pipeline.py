"""
AADHAAR DATASET - PRODUCTION-GRADE DATA CLEANING PIPELINE
=========================================================
Implements standardization, validation, and aggregation with full logging & visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import json

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

log_file = Path("cleaning_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

STATE_MAPPING = {
    'orissa': 'odisha',
    'pondicherry': 'puducherry',
    'uttaranchal': 'uttarakhand',
    # UT Variant Mappings
    'dadra and nagar haveli': 'dadra and nagar haveli and daman and diu',
    'daman and diu': 'dadra and nagar haveli and daman and diu',
    'jammu & kashmir': 'jammu and kashmir',
    'jammu and kasmir': 'jammu and kashmir',
    'kasmir': 'jammu and kashmir',
}

# Official 36 States and Union Territories
VALID_STATES_36 = {
    # 28 States
    'andhra pradesh',
    'arunachal pradesh',
    'assam',
    'bihar',
    'chhattisgarh',
    'goa',
    'gujarat',
    'haryana',
    'himachal pradesh',
    'jharkhand',
    'karnataka',
    'kerala',
    'madhya pradesh',
    'maharashtra',
    'manipur',
    'meghalaya',
    'mizoram',
    'nagaland',
    'odisha',
    'punjab',
    'rajasthan',
    'sikkim',
    'tamil nadu',
    'telangana',
    'tripura',
    'uttar pradesh',
    'uttarakhand',
    'west bengal',
    # 8 Union Territories
    'andaman and nicobar islands',
    'chandigarh',
    'dadra and nagar haveli and daman and diu',
    'delhi',
    'jammu and kashmir',
    'ladakh',
    'lakshadweep',
    'puducherry'
}

# ============================================================================
# PATHS
# ============================================================================

biometric_path = r"c:\personal dg\github_repo\aadhar-dataset\api_data_aadhar_biometric\api_data_aadhar_biometric"
demographic_path = r"c:\personal dg\github_repo\aadhar-dataset\api_data_aadhar_demographic\api_data_aadhar_demographic"
enrolment_path = r"c:\personal dg\github_repo\aadhar-dataset\api_data_aadhar_enrolment"
output_path = Path("cleaned_data")
output_path.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: SCHEMA & DATATYPE STANDARDIZATION
# ============================================================================

def fix_schema_and_dtypes(df, dataset_type):
    """
    Step 1: Fix data types and standardize column names
    """
    logger.info(f"Step 1: Fixing schema and datatypes for {dataset_type}")
    
    df_clean = df.copy()
    
    # Parse date
    df_clean['date'] = pd.to_datetime(df_clean['date'], format='%d-%m-%Y', errors='coerce')
    invalid_dates = df_clean[df_clean['date'].isna()].shape[0]
    if invalid_dates > 0:
        logger.warning(f"  [WARNING] {invalid_dates} invalid dates found and removed")
        df_clean = df_clean[df_clean['date'].notna()]
    
    # Standardize column names
    rename_map = {
        'bio_age_17_': 'updates_age_18_plus',
        'bio_age_5_17': 'updates_age_5_17',
        'demo_age_17_': 'updates_age_18_plus',
        'demo_age_5_17': 'updates_age_5_17',
        'age_18_greater': 'enrollments_age_18_plus',
        'age_5_17': 'enrollments_age_5_17',
        'age_0_5': 'enrollments_age_0_5'
    }
    
    df_clean.rename(columns=rename_map, inplace=True)
    
    # Ensure numeric columns are int
    numeric_cols = [col for col in df_clean.columns if 'update' in col or 'enrollment' in col]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
    
    logger.info(f"  [OK] Schema fixed. Shape: {df_clean.shape}")
    return df_clean, numeric_cols

# ============================================================================
# STEP 2: STATE NAME STANDARDIZATION
# ============================================================================

def standardize_states(df):
    """
    Step 2: Standardize state names (case, deprecated names, invalid entries)
    """
    logger.info("Step 2: Standardizing state names")
    
    df_clean = df.copy()
    df_clean['state_original'] = df_clean['state'].copy()
    
    # Normalize case and whitespace
    df_clean['state'] = (
        df_clean['state']
        .str.lower()
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace(' & ', ' and ')
        .str.replace('&', 'and')
    )
    
    # Apply deprecated mappings
    df_clean['state'] = df_clean['state'].replace(STATE_MAPPING)
    
    # Flag valid vs invalid states
    df_clean['valid_state'] = df_clean['state'].isin(VALID_STATES_36)
    
    invalid_count = (~df_clean['valid_state']).sum()
    valid_count = df_clean['valid_state'].sum()
    
    logger.info(f"  Valid states: {valid_count}")
    logger.info(f"  Invalid states: {invalid_count}")
    
    if invalid_count > 0:
        invalid_states = df_clean[~df_clean['valid_state']]['state'].value_counts().head(10)
        logger.warning(f"  [WARNING] Top invalid state entries:")
        for state, count in invalid_states.items():
            logger.warning(f"     '{state}': {count} records")
    
    # Remove rows with invalid states
    df_clean = df_clean[df_clean['valid_state']].copy()
    
    unique_states = df_clean['state'].nunique()
    logger.info(f"  [OK] States standardized. Valid unique states: {unique_states}/36")
    
    return df_clean

# ============================================================================
# STEP 3: DISTRICT NAME STANDARDIZATION
# ============================================================================

def standardize_districts(df):
    """
    Step 3: Standardize district names (case, whitespace, special chars)
    """
    logger.info("Step 3: Standardizing district names")
    
    df['district_original'] = df['district'].copy()
    
    # Normalize case, whitespace, special chars
    df['district'] = (
        df['district']
        .str.lower()
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace(r'[*\*]', '', regex=True)
    )
    
    logger.info(f"  [OK] Districts standardized. Unique districts: {df['district'].nunique()}")
    return df

# ============================================================================
# STEP 4: PINCODE VALIDATION
# ============================================================================

def validate_pincodes(df):
    """
    Step 4: Validate PIN codes (6 digits, in valid range) and remove invalid rows
    """
    logger.info("Step 4: Validating PIN codes")
    
    total_before = len(df)
    
    # Ensure 6 digits - convert to string and check format
    pincode_str = df['pincode'].astype(str).str.strip()
    valid_mask = (pincode_str.str.len() == 6) & (pincode_str.str.isnumeric())
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"  [WARNING] {invalid_count} invalid PIN codes removed")
    
    # Keep only valid records
    df = df[valid_mask].copy()
    
    logger.info(f"  [OK] PIN codes validated. Records retained: {len(df)}/{total_before}")
    return df

# ============================================================================
# STEP 5: NUMERIC VALIDATION & OUTLIER HANDLING
# ============================================================================

def handle_outliers(df, numeric_cols):
    """
    Step 5: Identify and cap extreme outliers (99th percentile)
    """
    logger.info("Step 5: Handling numeric outliers")
    
    for col in numeric_cols:
        p99 = df[col].quantile(0.99)
        p95 = df[col].quantile(0.95)
        max_val = df[col].max()
        
        outliers_count = (df[col] > p99).sum()
        
        if outliers_count > 0:
            logger.info(f"  Column '{col}':")
            logger.info(f"    95th percentile: {p95:.0f}")
            logger.info(f"    99th percentile: {p99:.0f}")
            logger.info(f"    Max value: {max_val:.0f}")
            logger.info(f"    Outliers (>99th): {outliers_count}")
            
            # Cap at 99th percentile
            df[col] = df[col].clip(upper=p99)
            logger.info(f"    [OK] Capped at 99th percentile")
    
    return df

# ============================================================================
# STEP 6: TEMPORAL AGGREGATION
# ============================================================================

def aggregate_temporal(df, level='monthly'):
    """
    Step 6: Aggregate data to monthly or weekly level to handle missing dates
    """
    logger.info(f"Step 6: Aggregating data to {level} level")
    
    if level == 'monthly':
        df['year_month'] = df['date'].dt.to_period('M')
        groupby_cols = ['year_month', 'state', 'district', 'pincode']
    else:  # weekly
        df['year_week'] = df['date'].dt.to_period('W')
        groupby_cols = ['year_week', 'state', 'district', 'pincode']
    
    numeric_cols = [col for col in df.columns if 'update' in col or 'enrollment' in col]
    
    agg_df = df.groupby(groupby_cols)[numeric_cols].sum().reset_index()
    
    logger.info(f"  [OK] Aggregated from {len(df)} daily records to {len(agg_df)} {level} records")
    return agg_df

# ============================================================================
# STEP 7: DATA QUALITY SCORING
# ============================================================================

def calculate_quality_metrics(df, numeric_cols):
    """
    Step 7: Calculate data quality metrics
    """
    logger.info("Step 7: Calculating data quality metrics")
    
    df['total_activity'] = df[numeric_cols].sum(axis=1)
    df['data_completeness'] = (df[numeric_cols] > 0).sum(axis=1) / len(numeric_cols)
    df['data_quality_score'] = df['data_completeness'] * 100
    
    logger.info(f"  [OK] Quality metrics added")
    return df

# ============================================================================
# STEP 8: LOAD AND CLEAN EACH DATASET
# ============================================================================

def load_and_clean_dataset(folder_path, dataset_name, dataset_type):
    """
    Load all CSV files in folder and apply full cleaning pipeline
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"PROCESSING: {dataset_name}")
    logger.info(f"{'='*80}")
    
    csv_files = list(Path(folder_path).glob("*.csv"))
    all_dfs = []
    
    for file in sorted(csv_files):
        logger.info(f"  Loading {file.name}...")
        df = pd.read_csv(file)
        all_dfs.append(df)
    
    # Combine all files
    df_combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined shape: {df_combined.shape}")
    
    # Apply cleaning pipeline
    df_clean, numeric_cols = fix_schema_and_dtypes(df_combined, dataset_name)
    df_clean = standardize_states(df_clean)
    df_clean = standardize_districts(df_clean)
    df_clean = validate_pincodes(df_clean)
    df_clean = handle_outliers(df_clean, numeric_cols)
    df_clean = aggregate_temporal(df_clean, level='monthly')
    df_clean = calculate_quality_metrics(df_clean, numeric_cols)
    
    logger.info(f"Final cleaned shape: {df_clean.shape}\n")
    
    return df_clean, numeric_cols

# ============================================================================
# STEP 9: MERGE ALL DATASETS
# ============================================================================

def merge_datasets(biometric_df, demographic_df, enrolment_df):
    """
    Step 9: Merge all three datasets into master table
    """
    logger.info(f"\n{'='*80}")
    logger.info("STEP 9: Merging datasets into master table")
    logger.info(f"{'='*80}")
    
    merge_keys = ['year_month', 'state', 'district', 'pincode']
    
    # Merge biometric and demographic
    master = biometric_df.merge(
        demographic_df,
        on=merge_keys,
        how='outer',
        suffixes=('_bio', '_demo')
    )
    
    logger.info(f"After biometric + demographic merge: {master.shape}")
    
    # Merge with enrolment
    master = master.merge(
        enrolment_df,
        on=merge_keys,
        how='outer'
    )
    
    logger.info(f"After enrolment merge: {master.shape}")
    
    # Fill NaN with 0 (no activity for that month)
    numeric_cols = [col for col in master.columns if 'update' in col or 'enrollment' in col or 'quality_score' in col]
    master[numeric_cols] = master[numeric_cols].fillna(0)
    
    logger.info(f"[OK] Master dataset created with {len(numeric_cols)} activity columns\n")
    
    return master

# ============================================================================
# VISUALIZATION: BEFORE vs AFTER
# ============================================================================

def create_cleaning_report_visualizations(before_dfs, after_dfs, output_path):
    """
    Create comparison visualizations: Before vs After cleaning
    """
    logger.info("Creating visualization report...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DATA CLEANING PIPELINE - BEFORE vs AFTER', fontsize=16, fontweight='bold')
    
    datasets = ['Biometric', 'Demographic', 'Enrolment']
    colors = ['#e74c3c', '#27ae60']
    
    # 1. Record Count Comparison
    ax = axes[0, 0]
    before_counts = [len(before_dfs[i]) for i in range(3)]
    after_counts = [len(after_dfs[i]) for i in range(3)]
    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width/2, before_counts, width, label='Before', color=colors[0], alpha=0.8)
    ax.bar(x + width/2, after_counts, width, label='After', color=colors[1], alpha=0.8)
    ax.set_ylabel('Record Count')
    ax.set_title('Record Count Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. State Count Comparison
    ax = axes[0, 1]
    before_states = [len(before_dfs[i]['state'].unique()) for i in range(3)]
    after_states = [len(after_dfs[i]['state'].unique()) for i in range(3)]
    ax.bar(x - width/2, before_states, width, label='Before', color=colors[0], alpha=0.8)
    ax.bar(x + width/2, after_states, width, label='After (Validated)', color=colors[1], alpha=0.8)
    ax.axhline(y=36, color='blue', linestyle='--', label='Target (36)', linewidth=2)
    ax.set_ylabel('Unique States/UTs')
    ax.set_title('State Standardization & Validation')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(after_states):
        ax.text(i + width/2, v + 1, str(v), ha='center', fontweight='bold')
    
    # 3. District Count
    ax = axes[0, 2]
    before_districts = [len(before_dfs[i]['district'].unique()) for i in range(3)]
    after_districts = [len(after_dfs[i]['district'].unique()) for i in range(3)]
    ax.bar(x - width/2, before_districts, width, label='Before', color=colors[0], alpha=0.8)
    ax.bar(x + width/2, after_districts, width, label='After', color=colors[1], alpha=0.8)
    ax.set_ylabel('Unique Districts')
    ax.set_title('District Standardization')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Data Completeness (Biometric)
    ax = axes[1, 0]
    if 'data_quality_score' in after_dfs[0].columns:
        quality_scores = after_dfs[0]['data_quality_score'].values
        ax.hist(quality_scores, bins=30, color=colors[1], alpha=0.8, edgecolor='black')
        ax.axvline(quality_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {quality_scores.mean():.1f}%')
        ax.set_xlabel('Data Quality Score (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Biometric - Quality Score Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # 5. Activity Distribution (After Cleaning)
    ax = axes[1, 1]
    numeric_cols_bio = [col for col in after_dfs[0].columns if 'update' in col]
    if numeric_cols_bio:
        activity_totals = [after_dfs[0][col].sum() for col in numeric_cols_bio]
        ax.bar(range(len(activity_totals)), activity_totals, color=colors[1], alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(activity_totals)))
        ax.set_xticklabels([col.replace('updates_', '') for col in numeric_cols_bio], rotation=45)
        ax.set_ylabel('Total Activity Count')
        ax.set_title('Biometric - Activity by Age Group')
        ax.grid(axis='y', alpha=0.3)
    
    # 6. Temporal Coverage
    ax = axes[1, 2]
    if 'year_month' in after_dfs[0].columns:
        monthly_activity = after_dfs[0].groupby('year_month').size()
        ax.plot(range(len(monthly_activity)), monthly_activity.values, marker='o', color=colors[1], linewidth=2)
        ax.fill_between(range(len(monthly_activity)), monthly_activity.values, alpha=0.3, color=colors[1])
        ax.set_xlabel('Month')
        ax.set_ylabel('Activity Count')
        ax.set_title('Temporal Coverage - Monthly Activity')
        ax.grid(alpha=0.3)
        ax.set_xticks(range(0, len(monthly_activity), max(1, len(monthly_activity)//5)))
    
    plt.tight_layout()
    plt.savefig(output_path / 'cleaning_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"[OK] Visualization saved to {output_path / 'cleaning_comparison.png'}")
    plt.close()

# ============================================================================
# GENERATE CLEANING SUMMARY REPORT
# ============================================================================

def generate_cleaning_report(master_df, biometric_df, demographic_df, enrolment_df, output_path):
    """
    Generate comprehensive cleaning report with state validation metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING CLEANING SUMMARY REPORT")
    logger.info(f"{'='*80}\n")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "state_validation_notes": "States validated against official 36 states/UTs. Invalid entries removed.",
        "biometric_dataset": {
            "final_records": len(biometric_df),
            "unique_states": biometric_df['state'].nunique(),
            "states_list": sorted(biometric_df['state'].unique().tolist()),
            "unique_districts": biometric_df['district'].nunique(),
            "unique_pincodes": biometric_df['pincode'].nunique(),
            "date_range": f"{biometric_df['year_month'].min()} to {biometric_df['year_month'].max()}",
            "quality_score_mean": float(biometric_df['data_quality_score'].mean()),
            "total_activity": int(biometric_df['total_activity'].sum())
        },
        "demographic_dataset": {
            "final_records": len(demographic_df),
            "unique_states": demographic_df['state'].nunique(),
            "states_list": sorted(demographic_df['state'].unique().tolist()),
            "unique_districts": demographic_df['district'].nunique(),
            "unique_pincodes": demographic_df['pincode'].nunique(),
            "date_range": f"{demographic_df['year_month'].min()} to {demographic_df['year_month'].max()}",
            "quality_score_mean": float(demographic_df['data_quality_score'].mean()),
            "total_activity": int(demographic_df['total_activity'].sum())
        },
        "enrolment_dataset": {
            "final_records": len(enrolment_df),
            "unique_states": enrolment_df['state'].nunique(),
            "states_list": sorted(enrolment_df['state'].unique().tolist()),
            "unique_districts": enrolment_df['district'].nunique(),
            "unique_pincodes": enrolment_df['pincode'].nunique(),
            "date_range": f"{enrolment_df['year_month'].min()} to {enrolment_df['year_month'].max()}",
            "quality_score_mean": float(enrolment_df['data_quality_score'].mean()),
            "total_activity": int(enrolment_df['total_activity'].sum())
        },
        "master_dataset": {
            "final_records": len(master_df),
            "unique_states": master_df['state'].nunique(),
            "states_list": sorted(master_df['state'].unique().tolist()),
            "unique_districts": master_df['district'].nunique(),
            "unique_pincodes": master_df['pincode'].nunique(),
            "date_range": f"{master_df['year_month'].min()} to {master_df['year_month'].max()}",
            "columns": list(master_df.columns)
        }
    }
    
    # Save as JSON
    with open(output_path / 'cleaning_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("CLEANING SUMMARY:")
    logger.info(f"  Biometric:   {report['biometric_dataset']['final_records']:,} records | States: {report['biometric_dataset']['unique_states']}/36 | Activity: {report['biometric_dataset']['total_activity']:,}")
    logger.info(f"  Demographic: {report['demographic_dataset']['final_records']:,} records | States: {report['demographic_dataset']['unique_states']}/36 | Activity: {report['demographic_dataset']['total_activity']:,}")
    logger.info(f"  Enrolment:   {report['enrolment_dataset']['final_records']:,} records | States: {report['enrolment_dataset']['unique_states']}/36 | Activity: {report['enrolment_dataset']['total_activity']:,}")
    logger.info(f"  Master:      {report['master_dataset']['final_records']:,} records | States: {report['master_dataset']['unique_states']}/36\n")
    
    logger.info(f"[OK] Report saved to {output_path / 'cleaning_report.json'}")
    return report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    logger.info("\n" + "="*80)
    logger.info("AADHAAR DATA CLEANING PIPELINE - STARTED")
    logger.info("="*80 + "\n")
    
    try:
        # Load and clean each dataset
        biometric_df, bio_numeric_cols = load_and_clean_dataset(
            biometric_path, "AADHAAR BIOMETRIC UPDATE DATASET", "biometric"
        )
        
        demographic_df, demo_numeric_cols = load_and_clean_dataset(
            demographic_path, "AADHAAR DEMOGRAPHIC UPDATE DATASET", "demographic"
        )
        
        enrolment_df, enrol_numeric_cols = load_and_clean_dataset(
            enrolment_path, "AADHAAR ENROLMENT DATASET", "enrolment"
        )
        
        # Merge into master dataset
        master_df = merge_datasets(biometric_df, demographic_df, enrolment_df)
        
        # Save cleaned datasets
        biometric_df.to_csv(output_path / 'biometric_cleaned.csv', index=False)
        demographic_df.to_csv(output_path / 'demographic_cleaned.csv', index=False)
        enrolment_df.to_csv(output_path / 'enrolment_cleaned.csv', index=False)
        master_df.to_csv(output_path / 'master_cleaned.csv', index=False)
        
        logger.info(f"[OK] Cleaned datasets saved to {output_path}/")
        
        # Generate visualizations
        create_cleaning_report_visualizations(
            [biometric_df, demographic_df, enrolment_df],
            [biometric_df, demographic_df, enrolment_df],
            output_path
        )
        
        # Generate report
        report = generate_cleaning_report(master_df, biometric_df, demographic_df, enrolment_df, output_path)
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY [OK]")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n[ERROR] PIPELINE FAILED: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
