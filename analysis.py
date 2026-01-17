import pandas as pd
import os
from pathlib import Path
import numpy as np
from datetime import datetime

# Define paths
biometric_path = r"c:\personal dg\github_repo\aadhar-dataset\api_data_aadhar_biometric\api_data_aadhar_biometric"
demographic_path = r"c:\personal dg\github_repo\aadhar-dataset\api_data_aadhar_demographic\api_data_aadhar_demographic"
enrolment_path = r"c:\personal dg\github_repo\aadhar-dataset\api_data_aadhar_enrolment"

def analyze_dates(df, dataset_name):
    """Deep analysis of date column"""
    print(f"\n{'*'*80}")
    print(f"DATE COLUMN ANALYSIS - {dataset_name}")
    print(f"{'*'*80}")
    
    if 'date' not in df.columns:
        print("No date column found")
        return
    
    try:
        df['date_parsed'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        print(f"Date Range: {df['date_parsed'].min()} to {df['date_parsed'].max()}")
        print(f"Total days covered: {(df['date_parsed'].max() - df['date_parsed'].min()).days} days")
        
        # Check for invalid dates
        invalid_dates = df[df['date_parsed'].isna()]
        if len(invalid_dates) > 0:
            print(f"\n⚠️  INVALID DATES FOUND: {len(invalid_dates):,}")
            print(f"Samples: {invalid_dates['date'].head(10).tolist()}")
        
        # Date distribution
        print(f"\nDate Distribution:")
        date_counts = df['date_parsed'].dt.date.value_counts().sort_index()
        print(f"  Earliest date records: {date_counts.head(3).to_dict()}")
        print(f"  Latest date records: {date_counts.tail(3).to_dict()}")
        
        # Missing dates
        all_dates = pd.date_range(df['date_parsed'].min(), df['date_parsed'].max(), freq='D')
        missing_dates = set(all_dates.date) - set(df['date_parsed'].dt.date.unique())
        if missing_dates:
            print(f"\n⚠️  MISSING DATES: {len(missing_dates)} dates have no records")
            print(f"Sample missing dates: {sorted(list(missing_dates))[:10]}")
        
        # Date duplicates per location
        date_state_district_pincode = df.groupby(['date', 'state', 'district', 'pincode']).size()
        duplicates_per_location = date_state_district_pincode[date_state_district_pincode > 1]
        if len(duplicates_per_location) > 0:
            print(f"\n⚠️  SAME DATE-STATE-DISTRICT-PINCODE appears multiple times: {len(duplicates_per_location):,}")
            print(f"Max occurrences: {duplicates_per_location.max()}")
            
    except Exception as e:
        print(f"ERROR parsing dates: {e}")

def analyze_states_detailed(df, dataset_name):
    """Deep analysis of state column"""
    print(f"\n{'*'*80}")
    print(f"STATE COLUMN ANALYSIS - {dataset_name}")
    print(f"{'*'*80}")
    
    if 'state' not in df.columns:
        print("No state column found")
        return
    
    unique_states = df['state'].unique()
    print(f"Total unique 'states': {len(unique_states)}")
    
    # Check for case variations
    state_lower_map = {}
    for state in unique_states:
        lower = state.lower().strip()
        if lower not in state_lower_map:
            state_lower_map[lower] = []
        state_lower_map[lower].append(state)
    
    case_variations = {k: v for k, v in state_lower_map.items() if len(v) > 1}
    if case_variations:
        print(f"\n⚠️  CASE VARIATIONS FOUND: {len(case_variations)}")
        for canonical, variations in sorted(case_variations.items()):
            print(f"  {canonical}: {variations}")
    
    # Check for invalid entries (cities, areas, etc)
    cities_keywords = ['Jaipur', 'Nagpur', 'Delhi', 'Chandigarh', 'Darbhanga', 'Madanapalle', 
                       'Puttenahalli', 'Raja', 'Balanagar']
    invalid_states = [s for s in unique_states if any(kw in s for kw in cities_keywords)]
    if invalid_states:
        print(f"\n⚠️  POSSIBLY INVALID STATE ENTRIES (cities/areas): {len(invalid_states)}")
        for state in sorted(invalid_states):
            count = (df['state'] == state).sum()
            print(f"  '{state}': {count:,} records")
    
    # Check for extra spaces or special characters
    whitespace_issues = [s for s in unique_states if '  ' in s or s != s.strip()]
    if whitespace_issues:
        print(f"\n⚠️  WHITESPACE ISSUES: {len(whitespace_issues)}")
        for state in whitespace_issues:
            count = (df['state'] == state).sum()
            print(f"  '{state}': {count:,} records (visible: '{state.replace(' ', '·')}')")
    
    # Deprecated state names
    deprecated = {'Orissa': 'Odisha', 'Pondicherry': 'Puducherry', 'Uttaranchal': 'Uttarakhand'}
    deprecated_found = {old: (df['state'] == old).sum() for old, new in deprecated.items() if (df['state'] == old).sum() > 0}
    if deprecated_found:
        print(f"\n⚠️  DEPRECATED STATE NAMES FOUND:")
        for old, count in deprecated_found.items():
            new = deprecated[old]
            print(f"  '{old}' → should be '{new}': {count:,} records")

def analyze_districts_detailed(df, dataset_name):
    """Deep analysis of district column"""
    print(f"\n{'*'*80}")
    print(f"DISTRICT COLUMN ANALYSIS - {dataset_name}")
    print(f"{'*'*80}")
    
    if 'district' not in df.columns:
        print("No district column found")
        return
    
    print(f"Total unique districts: {df['district'].nunique():,}")
    
    # Check for NULL/empty districts
    empty_districts = df[df['district'].isna() | (df['district'] == '')]
    if len(empty_districts) > 0:
        print(f"\n⚠️  EMPTY/NULL DISTRICTS: {len(empty_districts):,}")
    
    # Check for case variations in districts
    district_lower_map = {}
    for district in df['district'].unique():
        lower = str(district).lower().strip()
        if lower not in district_lower_map:
            district_lower_map[lower] = []
        district_lower_map[lower].append(district)
    
    case_variations = {k: v for k, v in district_lower_map.items() if len(v) > 1}
    if case_variations:
        print(f"\n⚠️  CASE VARIATIONS IN DISTRICTS: {len(case_variations)}")
        for canonical, variations in sorted(list(case_variations.items())[:10]):
            print(f"  {canonical}: {variations}")
        if len(case_variations) > 10:
            print(f"  ... and {len(case_variations) - 10} more")
    
    # Check for extra spaces
    whitespace_issues = [d for d in df['district'].unique() if '  ' in str(d)]
    if whitespace_issues:
        print(f"\n⚠️  DISTRICTS WITH EXTRA SPACES: {len(whitespace_issues)}")
        for district in whitespace_issues[:5]:
            print(f"  '{district}'")
    
    # Check state-district combinations validity
    print(f"\nSample State-District combinations:")
    state_district = df.groupby(['state', 'district']).size().reset_index(name='count')
    print(f"  Total unique state-district pairs: {len(state_district):,}")
    print(f"  Sample:\n{state_district.head(10).to_string()}")

def analyze_pincodes_detailed(df, dataset_name):
    """Deep analysis of pincode column"""
    print(f"\n{'*'*80}")
    print(f"PINCODE COLUMN ANALYSIS - {dataset_name}")
    print(f"{'*'*80}")
    
    if 'pincode' not in df.columns:
        print("No pincode column found")
        return
    
    print(f"Total unique PINcodes: {df['pincode'].nunique():,}")
    
    # Check for NULL pincodes
    null_pincodes = df[df['pincode'].isna()].shape[0]
    if null_pincodes > 0:
        print(f"\n⚠️  NULL PINCODES: {null_pincodes:,}")
    
    # Check pincode format (should be 6 digits)
    df_pincode = df.copy()
    df_pincode['pincode_str'] = df_pincode['pincode'].astype(str)
    invalid_format = df_pincode[df_pincode['pincode_str'].str.len() != 6]
    if len(invalid_format) > 0:
        print(f"\n⚠️  INVALID PINCODE FORMAT (not 6 digits): {len(invalid_format):,}")
        print(f"Examples: {invalid_format['pincode'].unique()[:10].tolist()}")
    
    # Check pincode range (should be 100000-999999)
    invalid_range = df[(df['pincode'] < 100000) | (df['pincode'] > 999999)]
    if len(invalid_range) > 0:
        print(f"\n⚠️  PINCODE OUT OF RANGE (< 100000 or > 999999): {len(invalid_range):,}")
        print(f"Examples: {invalid_range['pincode'].unique()[:10].tolist()}")
    
    # Check for PINcodes that should be 5 digits (Eastern India)
    pincode_min = df['pincode'].min()
    pincode_max = df['pincode'].max()
    print(f"\nPINcode Range: {pincode_min:,} to {pincode_max:,}")
    
    # PIN code distribution by first digit (geographic regions)
    df_pincode['pincode_region'] = (df_pincode['pincode'] // 100000).astype(int)
    print(f"\nPINcode Distribution by Region:")
    region_counts = df_pincode['pincode_region'].value_counts().sort_index()
    for region, count in region_counts.items():
        print(f"  {region}00000-{region}99999: {count:,}")

def analyze_numeric_columns_detailed(df, dataset_name):
    """Deep analysis of numeric columns"""
    print(f"\n{'*'*80}")
    print(f"NUMERIC COLUMNS ANALYSIS - {dataset_name}")
    print(f"{'*'*80}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'pincode']  # Exclude pincode
    
    if len(numeric_cols) == 0:
        print("No numeric columns found")
        return
    
    for col in numeric_cols:
        print(f"\n📊 Column: {col}")
        print(f"  Data type: {df[col].dtype}")
        print(f"  Null count: {df[col].isna().sum():,}")
        
        # Basic stats
        print(f"  Min: {df[col].min()}")
        print(f"  Max: {df[col].max()}")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std Dev: {df[col].std():.2f}")
        
        # Check for negative values (shouldn't exist for counts)
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"  ⚠️  NEGATIVE VALUES: {negative_count:,}")
            print(f"     Min negative: {df[df[col] < 0][col].min()}")
        
        # Check for zero values
        zero_count = (df[col] == 0).sum()
        zero_percentage = (zero_count / len(df)) * 100
        print(f"  Zero values: {zero_count:,} ({zero_percentage:.2f}%)")
        
        # Check for extreme outliers (> 95th percentile)
        p95 = df[col].quantile(0.95)
        p99 = df[col].quantile(0.99)
        outliers_95 = (df[col] > p95).sum()
        outliers_99 = (df[col] > p99).sum()
        print(f"  95th percentile: {p95:.2f} ({outliers_95:,} records above)")
        print(f"  99th percentile: {p99:.2f} ({outliers_99:,} records above)")
        
        # Show distribution
        print(f"  Percentile Distribution:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = df[col].quantile(p/100)
            print(f"    {p}th: {val:.2f}")
        
        # Detect potential errors (very high values)
        if df[col].max() > 1000:
            extreme_records = df[df[col] > 1000].shape[0]
            print(f"  ⚠️  EXTREME VALUES (> 1000): {extreme_records:,}")
            print(f"     Highest values: {sorted(df[col].unique(), reverse=True)[:5]}")

def analyze_duplicates_detailed(df, dataset_name):
    """Deep analysis of duplicate records"""
    print(f"\n{'*'*80}")
    print(f"DUPLICATE ANALYSIS - {dataset_name}")
    print(f"{'*'*80}")
    
    total_dups = df.duplicated().sum()
    print(f"Total duplicate rows (all columns): {total_dups:,} ({(total_dups/len(df)*100):.2f}%)")
    
    # Check duplicates on key columns
    key_cols = [col for col in ['date', 'state', 'district', 'pincode'] if col in df.columns]
    if key_cols:
        key_dups = df.duplicated(subset=key_cols, keep=False).sum()
        print(f"Duplicate by {key_cols}: {key_dups:,} ({(key_dups/len(df)*100):.2f}%)")
    
    # Show which columns have duplicates
    for col in df.columns:
        col_dups = df.duplicated(subset=[col], keep=False).sum()
        if col_dups > 0:
            print(f"  Duplicates on '{col}': {col_dups:,}")

def analyze_dataset(folder_path, dataset_name):
    """Analyze a dataset folder"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {dataset_name}")
    print(f"{'='*80}")
    
    csv_files = list(Path(folder_path).glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    total_rows = 0
    all_data = []
    
    for file in sorted(csv_files):
        print(f"\nFile: {file.name}")
        try:
            df = pd.read_csv(file)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  File Size: {file.stat().st_size / (1024*1024):.2f} MB")
            total_rows += len(df)
            all_data.append(df)
            
            # Check for nulls
            nulls = df.isnull().sum()
            if nulls.sum() > 0:
                print(f"  NULL values: {dict(nulls[nulls > 0])}")
            else:
                print(f"  NULL values: NONE")
                
            # Check duplicates
            dup_count = df.duplicated().sum()
            print(f"  Duplicate rows: {dup_count}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'-'*80}")
    print(f"DATASET SUMMARY: {dataset_name}")
    print(f"{'-'*80}")
    print(f"Total rows across all files: {total_rows:,}")
    print(f"Number of files: {len(csv_files)}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\nCombined Dataset Shape: {combined_df.shape}")
        print(f"Columns: {list(combined_df.columns)}")
        
        # Run detailed analyses
        analyze_dates(combined_df, dataset_name)
        analyze_states_detailed(combined_df, dataset_name)
        analyze_districts_detailed(combined_df, dataset_name)
        analyze_pincodes_detailed(combined_df, dataset_name)
        analyze_numeric_columns_detailed(combined_df, dataset_name)
        analyze_duplicates_detailed(combined_df, dataset_name)

# Analyze each dataset
print("\n" + "="*80)
print("AADHAAR DATASET - DEEP COMPREHENSIVE ANALYSIS")
print("="*80)

analyze_dataset(biometric_path, "AADHAAR BIOMETRIC UPDATE DATASET")
analyze_dataset(demographic_path, "AADHAAR DEMOGRAPHIC UPDATE DATASET")
analyze_dataset(enrolment_path, "AADHAAR ENROLMENT DATASET")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
