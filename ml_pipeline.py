"""
AADHAAR SETU - MACHINE LEARNING PIPELINE
=========================================
Transforms cleaned Aadhaar service data into regional profiles, service gap 
indicators, and early warning signals using clustering, anomaly detection, 
and trend analysis models.

Author: AADHAAR SETU Project
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
import pickle
import warnings

# ML Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy import stats

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION & METADATA BLOCK
# ============================================================================

class MLConfig:
    """Configuration for reproducibility and governance"""
    
    # Run Metadata
    RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATA_VERSION = "v1.0_cleaned"
    MODEL_VERSION = "v1.0.0"
    
    # Paths
    INPUT_PATH = Path("cleaned_data")
    OUTPUT_PATH = Path("ml_outputs")
    MODEL_PATH = Path("ml_models")
    
    # Input file
    MASTER_DATASET = "master_cleaned.csv"
    
    # Clustering Configuration
    N_CLUSTERS = 5  # Will be optimized
    MIN_CLUSTERS = 3
    MAX_CLUSTERS = 8
    
    # Anomaly Detection Configuration
    CONTAMINATION_RATE = 0.10  # Expected proportion of anomalies
    ISOLATION_FOREST_ESTIMATORS = 100
    
    # Trend Analysis Configuration
    ROLLING_WINDOW = 3  # months
    ZSCORE_THRESHOLD = 2.0  # Standard deviations for anomaly
    DROP_THRESHOLD = -0.30  # 30% drop triggers alert
    SPIKE_THRESHOLD = 0.50  # 50% spike triggers alert
    
    # Quality Thresholds
    MIN_QUALITY_SCORE = 50.0  # Minimum data quality to include
    
    # Feature Columns (from cleaned data)
    ACTIVITY_COLUMNS = [
        'updates_age_5_17_bio', 'updates_age_18_plus_bio',
        'updates_age_5_17_demo', 'updates_age_18_plus_demo',
        'enrollments_age_0_5', 'enrollments_age_5_17', 'enrollments_age_18_plus'
    ]
    
    QUALITY_COLUMNS = [
        'data_quality_score_bio', 'data_quality_score_demo', 'data_quality_score'
    ]

# Create output directories
MLConfig.OUTPUT_PATH.mkdir(exist_ok=True)
MLConfig.MODEL_PATH.mkdir(exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

log_file = MLConfig.OUTPUT_PATH / f"ml_pipeline_{MLConfig.RUN_TIMESTAMP}.log"
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
# 2. LOAD CLEANED MASTER DATASET
# ============================================================================

def load_and_validate_data():
    """
    Load cleaned master dataset and validate required columns
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: LOADING AND VALIDATING DATA")
    logger.info("=" * 80)
    
    input_file = MLConfig.INPUT_PATH / MLConfig.MASTER_DATASET
    
    if not input_file.exists():
        raise FileNotFoundError(f"Master dataset not found: {input_file}")
    
    logger.info(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    
    logger.info(f"  Raw shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['year_month', 'state', 'district', 'pincode']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for activity columns (at least some should exist)
    activity_cols_present = [col for col in df.columns if 'update' in col or 'enrollment' in col]
    logger.info(f"  Activity columns found: {len(activity_cols_present)}")
    
    # Basic validation
    initial_rows = len(df)
    
    # Remove rows with negative activity values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if 'update' in col or 'enrollment' in col:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                logger.warning(f"  [WARNING] {negative_count} negative values in {col} - setting to 0")
                df[col] = df[col].clip(lower=0)
    
    # Filter by quality score if available
    quality_cols = [col for col in df.columns if 'quality_score' in col]
    if quality_cols:
        # Use average quality across available quality columns
        df['avg_quality'] = df[quality_cols].mean(axis=1)
        low_quality = (df['avg_quality'] < MLConfig.MIN_QUALITY_SCORE).sum()
        if low_quality > 0:
            logger.info(f"  Rows with low quality (<{MLConfig.MIN_QUALITY_SCORE}): {low_quality}")
            # Don't remove, just flag for now
            df['low_quality_flag'] = df['avg_quality'] < MLConfig.MIN_QUALITY_SCORE
    
    logger.info(f"  [OK] Data validated. Final shape: {df.shape}")
    
    return df

# ============================================================================
# 3. FEATURE ENGINEERING MODULE
# ============================================================================

def build_features(df):
    """
    Generate ML-ready features from cleaned data
    THIS IS THE CORE OF THE ML PIPELINE
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    df_features = df.copy()
    
    # Identify available activity columns
    bio_cols = [col for col in df.columns if 'bio' in col and ('update' in col or 'age' in col)]
    demo_cols = [col for col in df.columns if 'demo' in col and ('update' in col or 'age' in col)]
    enrol_cols = [col for col in df.columns if 'enrollment' in col]
    
    logger.info(f"  Biometric columns: {bio_cols}")
    logger.info(f"  Demographic columns: {demo_cols}")
    logger.info(f"  Enrolment columns: {enrol_cols}")
    
    # -------------------------------------------------------------------------
    # FEATURE 1: Total Activity Aggregates
    # -------------------------------------------------------------------------
    logger.info("  Creating total activity features...")
    
    if bio_cols:
        df_features['total_bio_updates'] = df_features[bio_cols].sum(axis=1)
    else:
        df_features['total_bio_updates'] = 0
        
    if demo_cols:
        df_features['total_demo_updates'] = df_features[demo_cols].sum(axis=1)
    else:
        df_features['total_demo_updates'] = 0
        
    if enrol_cols:
        df_features['total_enrollments'] = df_features[enrol_cols].sum(axis=1)
    else:
        df_features['total_enrollments'] = 0
    
    df_features['total_activity'] = (
        df_features['total_bio_updates'] + 
        df_features['total_demo_updates'] + 
        df_features['total_enrollments']
    )
    
    # -------------------------------------------------------------------------
    # FEATURE 2: Activity Ratios
    # -------------------------------------------------------------------------
    logger.info("  Creating activity ratio features...")
    
    # Avoid division by zero
    total_safe = df_features['total_activity'].replace(0, 1)
    
    df_features['bio_ratio'] = df_features['total_bio_updates'] / total_safe
    df_features['demo_ratio'] = df_features['total_demo_updates'] / total_safe
    df_features['enrol_ratio'] = df_features['total_enrollments'] / total_safe
    
    # Update vs Enrolment ratio (key policy metric)
    enrol_safe = df_features['total_enrollments'].replace(0, 1)
    df_features['update_to_enrol_ratio'] = (
        (df_features['total_bio_updates'] + df_features['total_demo_updates']) / enrol_safe
    )
    
    # -------------------------------------------------------------------------
    # FEATURE 3: Age-Group Proportions
    # -------------------------------------------------------------------------
    logger.info("  Creating age-group proportion features...")
    
    # Child vs Adult proportions (important for policy)
    if 'enrollments_age_0_5' in df.columns and 'enrollments_age_5_17' in df.columns:
        df_features['child_enrol_total'] = (
            df_features.get('enrollments_age_0_5', 0) + 
            df_features.get('enrollments_age_5_17', 0)
        )
        df_features['child_enrol_proportion'] = df_features['child_enrol_total'] / enrol_safe
    
    # Bio updates age proportion
    bio_total_safe = df_features['total_bio_updates'].replace(0, 1)
    if 'updates_age_5_17_bio' in df.columns:
        df_features['bio_child_proportion'] = df_features['updates_age_5_17_bio'] / bio_total_safe
    
    # -------------------------------------------------------------------------
    # FEATURE 4: Location-Level Statistics
    # -------------------------------------------------------------------------
    logger.info("  Creating location-level statistics...")
    
    # District-level aggregates
    district_stats = df_features.groupby('district').agg({
        'total_activity': ['mean', 'std', 'sum'],
        'total_bio_updates': 'mean',
        'total_demo_updates': 'mean',
        'total_enrollments': 'mean'
    })
    district_stats.columns = ['_'.join(col) for col in district_stats.columns]
    district_stats = district_stats.reset_index()
    district_stats.columns = ['district'] + ['district_' + col for col in district_stats.columns[1:]]
    
    df_features = df_features.merge(district_stats, on='district', how='left')
    
    # Deviation from district mean
    district_mean_safe = df_features['district_total_activity_mean'].replace(0, 1)
    df_features['activity_vs_district_mean'] = (
        df_features['total_activity'] / district_mean_safe
    )
    
    # -------------------------------------------------------------------------
    # FEATURE 5: Temporal Features (Month-on-Month)
    # -------------------------------------------------------------------------
    logger.info("  Creating temporal features...")
    
    # Sort by location and time
    df_features = df_features.sort_values(['state', 'district', 'pincode', 'year_month'])
    
    # Group by pincode for temporal calculations
    for col in ['total_activity', 'total_bio_updates', 'total_demo_updates', 'total_enrollments']:
        # Previous month value
        df_features[f'{col}_prev'] = df_features.groupby(['state', 'district', 'pincode'])[col].shift(1)
        
        # Month-on-Month growth
        prev_safe = df_features[f'{col}_prev'].replace(0, 1)
        df_features[f'{col}_mom_growth'] = (
            (df_features[col] - df_features[f'{col}_prev']) / prev_safe
        )
        
        # Rolling average (3-month)
        df_features[f'{col}_rolling_avg'] = df_features.groupby(
            ['state', 'district', 'pincode']
        )[col].transform(lambda x: x.rolling(MLConfig.ROLLING_WINDOW, min_periods=1).mean())
        
        # Rolling std (volatility)
        df_features[f'{col}_volatility'] = df_features.groupby(
            ['state', 'district', 'pincode']
        )[col].transform(lambda x: x.rolling(MLConfig.ROLLING_WINDOW, min_periods=1).std())
    
    # Fill NaN from rolling calculations
    df_features = df_features.fillna(0)
    
    # -------------------------------------------------------------------------
    # FEATURE 6: Data Quality Weight
    # -------------------------------------------------------------------------
    logger.info("  Creating data quality weight...")
    
    quality_cols = [col for col in df_features.columns if 'quality_score' in col and col != 'avg_quality']
    if quality_cols:
        df_features['quality_weight'] = df_features[quality_cols].mean(axis=1) / 100
    else:
        df_features['quality_weight'] = 1.0
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    feature_cols = [
        'total_bio_updates', 'total_demo_updates', 'total_enrollments', 'total_activity',
        'bio_ratio', 'demo_ratio', 'enrol_ratio', 'update_to_enrol_ratio',
        'activity_vs_district_mean',
        'total_activity_mom_growth', 'total_activity_rolling_avg', 'total_activity_volatility',
        'quality_weight'
    ]
    
    # Add age-group features if available
    if 'child_enrol_proportion' in df_features.columns:
        feature_cols.append('child_enrol_proportion')
    if 'bio_child_proportion' in df_features.columns:
        feature_cols.append('bio_child_proportion')
    
    logger.info(f"  [OK] Created {len(feature_cols)} ML features")
    
    return df_features, feature_cols

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================

def scale_features(df, feature_cols):
    """
    Scale features for ML models
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: FEATURE SCALING")
    logger.info("=" * 80)
    
    # Extract feature matrix
    X = df[feature_cols].copy()
    
    # Replace infinities with 0
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    
    logger.info(f"  Feature matrix shape: {X.shape}")
    logger.info(f"  Features: {list(X.columns)}")
    
    # StandardScaler for clustering (preserves distribution)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create scaled DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=[f"{col}_scaled" for col in feature_cols])
    
    logger.info(f"  [OK] Features scaled using StandardScaler")
    
    return X_scaled, scaler, feature_cols

# ============================================================================
# 5. REGIONAL PROFILING MODEL (K-MEANS CLUSTERING)
# ============================================================================

def find_optimal_clusters(X_scaled, min_k=3, max_k=8):
    """
    Find optimal number of clusters using silhouette score
    """
    logger.info("  Finding optimal K...")
    
    silhouette_scores = []
    
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append((k, score))
        logger.info(f"    K={k}: Silhouette Score = {score:.4f}")
    
    # Select best K
    best_k = max(silhouette_scores, key=lambda x: x[1])[0]
    logger.info(f"  [OK] Optimal K = {best_k}")
    
    return best_k

def train_clustering_model(X_scaled, df):
    """
    Train K-Means clustering model for regional profiling
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 4: REGIONAL PROFILING (K-MEANS CLUSTERING)")
    logger.info("=" * 80)
    
    # Find optimal K
    optimal_k = find_optimal_clusters(
        X_scaled, 
        MLConfig.MIN_CLUSTERS, 
        MLConfig.MAX_CLUSTERS
    )
    
    # Train final model
    logger.info(f"  Training K-Means with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add to dataframe
    df['cluster_id'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = df.groupby('cluster_id').agg({
        'total_activity': ['mean', 'sum', 'count'],
        'total_bio_updates': 'mean',
        'total_demo_updates': 'mean',
        'total_enrollments': 'mean',
        'update_to_enrol_ratio': 'mean'
    }).round(2)
    
    cluster_stats.columns = ['_'.join(col) for col in cluster_stats.columns]
    cluster_stats = cluster_stats.reset_index()
    
    logger.info("\n  Cluster Statistics:")
    for _, row in cluster_stats.iterrows():
        logger.info(f"    Cluster {int(row['cluster_id'])}: "
                   f"{int(row['total_activity_count'])} locations, "
                   f"Avg Activity: {row['total_activity_mean']:.0f}")
    
    # Final silhouette score
    final_score = silhouette_score(X_scaled, cluster_labels)
    logger.info(f"\n  [OK] Clustering complete. Silhouette Score: {final_score:.4f}")
    
    return df, kmeans, cluster_stats

# ============================================================================
# 6. SERVICE GAP DETECTION (ISOLATION FOREST)
# ============================================================================

def train_anomaly_model(X_scaled, df):
    """
    Train Isolation Forest for service gap detection
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 5: SERVICE GAP DETECTION (ISOLATION FOREST)")
    logger.info("=" * 80)
    
    # Train Isolation Forest
    logger.info(f"  Training Isolation Forest (contamination={MLConfig.CONTAMINATION_RATE})...")
    
    iso_forest = IsolationForest(
        n_estimators=MLConfig.ISOLATION_FOREST_ESTIMATORS,
        contamination=MLConfig.CONTAMINATION_RATE,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit and predict
    anomaly_predictions = iso_forest.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    anomaly_scores = iso_forest.decision_function(X_scaled)  # Lower = more anomalous
    
    # Add to dataframe
    df['anomaly_prediction'] = anomaly_predictions
    df['anomaly_score'] = anomaly_scores
    
    # Convert to risk levels
    df['is_anomaly'] = (df['anomaly_prediction'] == -1).astype(int)
    
    # Risk categorization based on anomaly score percentiles
    df['risk_percentile'] = df['anomaly_score'].rank(pct=True)
    
    def categorize_risk(row):
        if row['is_anomaly'] == 1:
            if row['risk_percentile'] < 0.05:
                return 'CRITICAL'
            else:
                return 'HIGH'
        elif row['risk_percentile'] < 0.15:
            return 'MEDIUM'
        elif row['risk_percentile'] < 0.30:
            return 'LOW'
        else:
            return 'NORMAL'
    
    df['risk_category'] = df.apply(categorize_risk, axis=1)
    
    # Statistics
    risk_counts = df['risk_category'].value_counts()
    logger.info("\n  Risk Distribution:")
    for risk, count in risk_counts.items():
        pct = count / len(df) * 100
        logger.info(f"    {risk}: {count} locations ({pct:.1f}%)")
    
    anomaly_count = df['is_anomaly'].sum()
    logger.info(f"\n  [OK] Detected {anomaly_count} anomalous locations ({anomaly_count/len(df)*100:.1f}%)")
    
    return df, iso_forest

# ============================================================================
# 7. TREND DEVIATION LOGIC (Z-SCORE BASED)
# ============================================================================

def analyze_trends(df):
    """
    Detect trend deviations using statistical methods
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 6: TREND ANALYSIS & EARLY WARNING")
    logger.info("=" * 80)
    
    # Z-score based deviation detection
    logger.info("  Calculating Z-scores for trend deviation...")
    
    # Calculate Z-scores for key metrics
    for col in ['total_activity', 'total_bio_updates', 'total_demo_updates', 'total_enrollments']:
        if col in df.columns:
            # Group by district for local context
            df[f'{col}_zscore'] = df.groupby('district')[col].transform(
                lambda x: stats.zscore(x, nan_policy='omit')
            )
            df[f'{col}_zscore'] = df[f'{col}_zscore'].fillna(0)
    
    # Detect drops and spikes
    logger.info("  Detecting activity drops and spikes...")
    
    # Drop detection (significant decrease from rolling average)
    if 'total_activity_mom_growth' in df.columns:
        df['drop_alert'] = (df['total_activity_mom_growth'] < MLConfig.DROP_THRESHOLD).astype(int)
        df['spike_alert'] = (df['total_activity_mom_growth'] > MLConfig.SPIKE_THRESHOLD).astype(int)
    else:
        df['drop_alert'] = 0
        df['spike_alert'] = 0
    
    # Z-score based anomaly detection
    if 'total_activity_zscore' in df.columns:
        df['zscore_anomaly'] = (
            (df['total_activity_zscore'].abs() > MLConfig.ZSCORE_THRESHOLD)
        ).astype(int)
    else:
        df['zscore_anomaly'] = 0
    
    # Combined trend status
    def determine_trend_status(row):
        if row.get('drop_alert', 0) == 1:
            return 'DECLINING'
        elif row.get('spike_alert', 0) == 1:
            return 'SURGING'
        elif row.get('zscore_anomaly', 0) == 1:
            return 'UNUSUAL'
        else:
            return 'STABLE'
    
    df['trend_status'] = df.apply(determine_trend_status, axis=1)
    
    # Statistics
    trend_counts = df['trend_status'].value_counts()
    logger.info("\n  Trend Distribution:")
    for trend, count in trend_counts.items():
        pct = count / len(df) * 100
        logger.info(f"    {trend}: {count} ({pct:.1f}%)")
    
    drop_count = df['drop_alert'].sum()
    spike_count = df['spike_alert'].sum()
    logger.info(f"\n  [OK] Trend analysis complete. Drops: {drop_count}, Spikes: {spike_count}")
    
    return df

# ============================================================================
# 8. INSIGHT MAPPING LAYER (EXPLAINABILITY)
# ============================================================================

# Cluster Profile Mappings
CLUSTER_PROFILES = {
    0: {
        'name': 'High Activity - Well Maintained',
        'description': 'Regions with consistently high Aadhaar service activity across all categories',
        'policy_action': 'Maintain current service levels; can serve as model regions'
    },
    1: {
        'name': 'Update-Heavy - Established',
        'description': 'Regions with high update activity relative to new enrollments',
        'policy_action': 'Focus on maintaining biometric update infrastructure'
    },
    2: {
        'name': 'Enrollment-Focused - Growing',
        'description': 'Regions with high new enrollment activity, lower updates',
        'policy_action': 'Plan for increased update demand in 3-5 years'
    },
    3: {
        'name': 'Low Activity - At Risk',
        'description': 'Regions with consistently low Aadhaar service activity',
        'policy_action': 'Investigate barriers; consider mobile enrollment camps'
    },
    4: {
        'name': 'Volatile - Unstable',
        'description': 'Regions with high month-to-month variability in service usage',
        'policy_action': 'Stabilize service delivery; investigate seasonal patterns'
    }
}

# Risk Explanations
RISK_EXPLANATIONS = {
    'CRITICAL': 'Severe service gap detected. Immediate attention required.',
    'HIGH': 'Significant deviation from expected service patterns.',
    'MEDIUM': 'Moderate service concerns. Monitor closely.',
    'LOW': 'Minor variations from normal patterns.',
    'NORMAL': 'Service activity within expected range.'
}

def generate_insights(df, cluster_stats):
    """
    Map ML outputs to human-readable insights
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 7: INSIGHT GENERATION (EXPLAINABILITY)")
    logger.info("=" * 80)
    
    # -------------------------------------------------------------------------
    # Cluster Profiling Insights
    # -------------------------------------------------------------------------
    logger.info("  Generating cluster profile insights...")
    
    # Map clusters to profiles based on characteristics
    # Analyze each cluster's behavior to assign meaningful labels
    cluster_means = df.groupby('cluster_id').agg({
        'total_activity': 'mean',
        'update_to_enrol_ratio': 'mean',
        'total_activity_volatility': 'mean'
    }).to_dict('index')
    
    # Sort clusters by activity level
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1]['total_activity'], reverse=True)
    
    # Assign profiles based on relative ranking
    cluster_mapping = {}
    n_clusters = len(sorted_clusters)
    
    for i, (cluster_id, stats) in enumerate(sorted_clusters):
        if i < n_clusters * 0.2:  # Top 20%
            cluster_mapping[cluster_id] = 0  # High Activity
        elif stats['update_to_enrol_ratio'] > df['update_to_enrol_ratio'].median():
            cluster_mapping[cluster_id] = 1  # Update-Heavy
        elif stats['total_activity_volatility'] > df['total_activity_volatility'].quantile(0.75):
            cluster_mapping[cluster_id] = 4  # Volatile
        elif stats['total_activity'] < df['total_activity'].quantile(0.25):
            cluster_mapping[cluster_id] = 3  # Low Activity
        else:
            cluster_mapping[cluster_id] = 2  # Enrollment-Focused
    
    # Apply profile mapping
    df['cluster_profile_id'] = df['cluster_id'].map(cluster_mapping)
    df['cluster_profile'] = df['cluster_profile_id'].map(
        lambda x: CLUSTER_PROFILES.get(x, CLUSTER_PROFILES[2])['name']
    )
    df['cluster_description'] = df['cluster_profile_id'].map(
        lambda x: CLUSTER_PROFILES.get(x, CLUSTER_PROFILES[2])['description']
    )
    df['policy_recommendation'] = df['cluster_profile_id'].map(
        lambda x: CLUSTER_PROFILES.get(x, CLUSTER_PROFILES[2])['policy_action']
    )
    
    # -------------------------------------------------------------------------
    # Risk Explanations
    # -------------------------------------------------------------------------
    logger.info("  Generating risk explanations...")
    
    df['risk_explanation'] = df['risk_category'].map(RISK_EXPLANATIONS)
    
    # Detailed risk reasons based on actual data
    def generate_risk_reason(row):
        reasons = []
        
        if row.get('is_anomaly', 0) == 1:
            if row.get('total_activity', 0) < row.get('district_total_activity_mean', 1) * 0.5:
                reasons.append("Activity significantly below district average")
            if row.get('total_bio_updates', 0) == 0:
                reasons.append("No biometric updates recorded")
            if row.get('total_enrollments', 0) == 0:
                reasons.append("No new enrollments recorded")
        
        if row.get('drop_alert', 0) == 1:
            reasons.append(f"Sharp decline in activity (>{abs(MLConfig.DROP_THRESHOLD)*100:.0f}% drop)")
        
        if row.get('spike_alert', 0) == 1:
            reasons.append(f"Unusual activity spike (>{MLConfig.SPIKE_THRESHOLD*100:.0f}% increase)")
        
        return "; ".join(reasons) if reasons else "No specific concerns"
    
    df['risk_reason'] = df.apply(generate_risk_reason, axis=1)
    
    # -------------------------------------------------------------------------
    # Generate text insights for each location
    # -------------------------------------------------------------------------
    logger.info("  Generating location-specific insights...")
    
    def generate_insight_text(row):
        insights = []
        
        # Cluster insight
        insights.append(f"Profile: {row.get('cluster_profile', 'Unknown')}")
        
        # Risk insight
        if row.get('risk_category', 'NORMAL') != 'NORMAL':
            insights.append(f"Risk: {row.get('risk_category', 'Unknown')} - {row.get('risk_explanation', '')}")
        
        # Trend insight
        if row.get('trend_status', 'STABLE') != 'STABLE':
            insights.append(f"Trend: {row.get('trend_status', 'Unknown')}")
        
        # Activity insight
        if row.get('activity_vs_district_mean', 1) < 0.5:
            insights.append("Below district average activity")
        elif row.get('activity_vs_district_mean', 1) > 1.5:
            insights.append("Above district average activity")
        
        return " | ".join(insights)
    
    df['insight_summary'] = df.apply(generate_insight_text, axis=1)
    
    # Profile distribution
    profile_counts = df['cluster_profile'].value_counts()
    logger.info("\n  Cluster Profile Distribution:")
    for profile, count in profile_counts.items():
        pct = count / len(df) * 100
        logger.info(f"    {profile}: {count} ({pct:.1f}%)")
    
    logger.info(f"\n  [OK] Generated insights for {len(df)} locations")
    
    return df

# ============================================================================
# 9. OUTPUT GENERATION
# ============================================================================

def generate_outputs(df, cluster_stats, scaler, kmeans, iso_forest, feature_cols):
    """
    Export ML outputs for consumption by dashboards
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 8: OUTPUT GENERATION")
    logger.info("=" * 80)
    
    output_path = MLConfig.OUTPUT_PATH
    
    # -------------------------------------------------------------------------
    # 1. Full ML Output Table
    # -------------------------------------------------------------------------
    logger.info("  Exporting full ML output table...")
    
    # Select relevant columns for output
    output_cols = [
        'year_month', 'state', 'district', 'pincode',
        'total_activity', 'total_bio_updates', 'total_demo_updates', 'total_enrollments',
        'bio_ratio', 'demo_ratio', 'enrol_ratio', 'update_to_enrol_ratio',
        'activity_vs_district_mean',
        'cluster_id', 'cluster_profile', 'cluster_description', 'policy_recommendation',
        'is_anomaly', 'anomaly_score', 'risk_category', 'risk_explanation', 'risk_reason',
        'trend_status', 'drop_alert', 'spike_alert',
        'insight_summary'
    ]
    
    # Add available columns
    output_cols = [col for col in output_cols if col in df.columns]
    
    df_output = df[output_cols].copy()
    df_output.to_csv(output_path / 'ml_full_output.csv', index=False)
    logger.info(f"    Saved: ml_full_output.csv ({len(df_output)} rows)")
    
    # -------------------------------------------------------------------------
    # 2. Cluster Summary Table
    # -------------------------------------------------------------------------
    logger.info("  Exporting cluster summary table...")
    
    cluster_summary = df.groupby(['cluster_id', 'cluster_profile']).agg({
        'pincode': 'count',
        'state': 'nunique',
        'district': 'nunique',
        'total_activity': ['mean', 'sum'],
        'total_bio_updates': 'mean',
        'total_demo_updates': 'mean',
        'total_enrollments': 'mean'
    }).round(2)
    
    cluster_summary.columns = ['_'.join(col) for col in cluster_summary.columns]
    cluster_summary = cluster_summary.reset_index()
    cluster_summary.to_csv(output_path / 'cluster_summary.csv', index=False)
    logger.info(f"    Saved: cluster_summary.csv")
    
    # -------------------------------------------------------------------------
    # 3. High-Risk Region Table
    # -------------------------------------------------------------------------
    logger.info("  Exporting high-risk region table...")
    
    high_risk = df[df['risk_category'].isin(['CRITICAL', 'HIGH'])].copy()
    high_risk_cols = [
        'year_month', 'state', 'district', 'pincode',
        'risk_category', 'risk_reason', 'anomaly_score',
        'total_activity', 'trend_status', 'policy_recommendation'
    ]
    high_risk_cols = [col for col in high_risk_cols if col in high_risk.columns]
    high_risk = high_risk[high_risk_cols].sort_values('anomaly_score')
    high_risk.to_csv(output_path / 'high_risk_regions.csv', index=False)
    logger.info(f"    Saved: high_risk_regions.csv ({len(high_risk)} regions)")
    
    # -------------------------------------------------------------------------
    # 4. Monthly Alert Table
    # -------------------------------------------------------------------------
    logger.info("  Exporting monthly alert table...")
    
    alerts = df[(df['drop_alert'] == 1) | (df['spike_alert'] == 1) | (df['is_anomaly'] == 1)].copy()
    alert_cols = [
        'year_month', 'state', 'district', 'pincode',
        'drop_alert', 'spike_alert', 'is_anomaly',
        'trend_status', 'risk_category', 'risk_reason'
    ]
    alert_cols = [col for col in alert_cols if col in alerts.columns]
    alerts = alerts[alert_cols]
    alerts.to_csv(output_path / 'monthly_alerts.csv', index=False)
    logger.info(f"    Saved: monthly_alerts.csv ({len(alerts)} alerts)")
    
    # -------------------------------------------------------------------------
    # 5. State-Level Summary
    # -------------------------------------------------------------------------
    logger.info("  Exporting state-level summary...")
    
    state_summary = df.groupby('state').agg({
        'pincode': 'nunique',
        'district': 'nunique',
        'total_activity': 'sum',
        'is_anomaly': 'sum',
        'drop_alert': 'sum',
        'spike_alert': 'sum'
    }).reset_index()
    state_summary.columns = [
        'state', 'unique_pincodes', 'unique_districts', 
        'total_activity', 'anomaly_count', 'drop_alerts', 'spike_alerts'
    ]
    state_summary['risk_score'] = (
        state_summary['anomaly_count'] + 
        state_summary['drop_alerts'] * 2 + 
        state_summary['spike_alerts']
    )
    state_summary = state_summary.sort_values('risk_score', ascending=False)
    state_summary.to_csv(output_path / 'state_summary.csv', index=False)
    logger.info(f"    Saved: state_summary.csv")
    
    # -------------------------------------------------------------------------
    # 6. JSON Metadata
    # -------------------------------------------------------------------------
    logger.info("  Exporting metadata...")
    
    metadata = {
        "run_timestamp": MLConfig.RUN_TIMESTAMP,
        "data_version": MLConfig.DATA_VERSION,
        "model_version": MLConfig.MODEL_VERSION,
        "configuration": {
            "n_clusters": int(df['cluster_id'].nunique()),
            "contamination_rate": MLConfig.CONTAMINATION_RATE,
            "zscore_threshold": MLConfig.ZSCORE_THRESHOLD,
            "drop_threshold": MLConfig.DROP_THRESHOLD,
            "spike_threshold": MLConfig.SPIKE_THRESHOLD
        },
        "statistics": {
            "total_locations": int(len(df)),
            "unique_states": int(df['state'].nunique()),
            "unique_districts": int(df['district'].nunique()),
            "unique_pincodes": int(df['pincode'].nunique()),
            "total_activity": int(df['total_activity'].sum()),
            "anomaly_count": int(df['is_anomaly'].sum()),
            "high_risk_count": int((df['risk_category'].isin(['CRITICAL', 'HIGH'])).sum()),
            "drop_alert_count": int(df['drop_alert'].sum()),
            "spike_alert_count": int(df['spike_alert'].sum())
        },
        "cluster_distribution": df['cluster_profile'].value_counts().to_dict(),
        "risk_distribution": df['risk_category'].value_counts().to_dict(),
        "trend_distribution": df['trend_status'].value_counts().to_dict(),
        "feature_columns": feature_cols,
        "output_files": [
            "ml_full_output.csv",
            "cluster_summary.csv",
            "high_risk_regions.csv",
            "monthly_alerts.csv",
            "state_summary.csv"
        ]
    }
    
    with open(output_path / 'ml_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"    Saved: ml_metadata.json")
    
    logger.info(f"\n  [OK] All outputs saved to {output_path}/")
    
    return metadata

# ============================================================================
# 10. ML VISUALIZATION & GRAPHICAL RESULTS
# ============================================================================

def create_ml_visualizations(df, X_scaled, kmeans, iso_forest, feature_cols, metadata):
    """
    Create comprehensive visualizations of ML training results
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 10: GENERATING ML VISUALIZATIONS")
    logger.info("=" * 80)
    
    output_path = MLConfig.OUTPUT_PATH
    
    # Create figure directory
    fig_path = output_path / 'figures'
    fig_path.mkdir(exist_ok=True)
    
    # =========================================================================
    # FIGURE 1: CLUSTER ANALYSIS DASHBOARD (2x2 grid)
    # =========================================================================
    logger.info("  Creating Cluster Analysis Dashboard...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('AADHAAR SETU - Regional Clustering Analysis', fontsize=18, fontweight='bold', y=1.02)
    
    # 1A: Cluster Distribution (Pie Chart)
    ax = axes[0, 0]
    cluster_counts = df['cluster_profile'].value_counts()
    colors = sns.color_palette("husl", len(cluster_counts))
    wedges, texts, autotexts = ax.pie(
        cluster_counts.values, 
        labels=None,
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.02] * len(cluster_counts),
        shadow=True,
        startangle=90
    )
    ax.legend(wedges, cluster_counts.index, title="Cluster Profiles", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title('Regional Cluster Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # 1B: Cluster Characteristics (Radar/Bar Chart)
    ax = axes[0, 1]
    cluster_means = df.groupby('cluster_profile')[['total_bio_updates', 'total_demo_updates', 'total_enrollments']].mean()
    cluster_means_normalized = cluster_means / cluster_means.max()
    cluster_means_normalized.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title('Cluster Activity Profiles (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster Profile', fontsize=12)
    ax.set_ylabel('Normalized Activity Level', fontsize=12)
    ax.legend(title='Activity Type', labels=['Biometric', 'Demographic', 'Enrolment'])
    ax.tick_params(axis='x', rotation=45)
    
    # 1C: Silhouette Score by K
    ax = axes[1, 0]
    k_range = range(MLConfig.MIN_CLUSTERS, MLConfig.MAX_CLUSTERS + 1)
    silhouette_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    bars = ax.bar(k_range, silhouette_scores, color=sns.color_palette("viridis", len(k_range)), edgecolor='black')
    ax.axhline(y=max(silhouette_scores), color='red', linestyle='--', linewidth=2, label=f'Best Score: {max(silhouette_scores):.4f}')
    
    # Highlight best K
    best_idx = silhouette_scores.index(max(silhouette_scores))
    bars[best_idx].set_color('red')
    bars[best_idx].set_edgecolor('darkred')
    bars[best_idx].set_linewidth(2)
    
    ax.set_title('Optimal K Selection (Silhouette Analysis)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.legend()
    
    # 1D: Cluster by State Heatmap
    ax = axes[1, 1]
    cluster_state_matrix = pd.crosstab(df['state'], df['cluster_profile'], normalize='index') * 100
    top_states = df.groupby('state')['total_activity'].sum().nlargest(15).index
    cluster_state_matrix = cluster_state_matrix.loc[cluster_state_matrix.index.isin(top_states)]
    
    sns.heatmap(cluster_state_matrix, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5)
    ax.set_title('Cluster Distribution by State (Top 15)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster Profile', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(fig_path / 'cluster_analysis_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    Saved: cluster_analysis_dashboard.png")
    
    # =========================================================================
    # FIGURE 2: ANOMALY DETECTION RESULTS
    # =========================================================================
    logger.info("  Creating Anomaly Detection Dashboard...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('AADHAAR SETU - Service Gap Detection (Anomaly Analysis)', fontsize=18, fontweight='bold', y=1.02)
    
    # 2A: Risk Category Distribution
    ax = axes[0, 0]
    risk_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']
    risk_colors = {'CRITICAL': '#d62728', 'HIGH': '#ff7f0e', 'MEDIUM': '#ffbb78', 'LOW': '#98df8a', 'NORMAL': '#2ca02c'}
    risk_counts = df['risk_category'].value_counts().reindex(risk_order).fillna(0)
    
    bars = ax.bar(risk_counts.index, risk_counts.values, 
                  color=[risk_colors[r] for r in risk_counts.index],
                  edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, risk_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title('Risk Category Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Risk Category', fontsize=12)
    ax.set_ylabel('Number of Locations', fontsize=12)
    ax.set_ylim(0, risk_counts.max() * 1.15)
    
    # 2B: Anomaly Score Distribution
    ax = axes[0, 1]
    normal_scores = df[df['is_anomaly'] == 0]['anomaly_score']
    anomaly_scores = df[df['is_anomaly'] == 1]['anomaly_score']
    
    ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='#2ca02c', edgecolor='black')
    ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='#d62728', edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    
    ax.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Anomaly Score (Lower = More Anomalous)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    
    # 2C: Anomalies by State (Top 15)
    ax = axes[1, 0]
    state_anomalies = df[df['is_anomaly'] == 1].groupby('state').size().nlargest(15)
    state_anomalies.plot(kind='barh', ax=ax, color=sns.color_palette("Reds_r", len(state_anomalies)),
                         edgecolor='black', linewidth=0.5)
    ax.set_title('High-Risk Locations by State (Top 15)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Anomalous Locations', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    
    # Add count labels
    for i, v in enumerate(state_anomalies.values):
        ax.text(v + 0.5, i, str(v), va='center', fontsize=10)
    
    # 2D: Anomaly vs Cluster Relationship
    ax = axes[1, 1]
    anomaly_cluster = pd.crosstab(df['cluster_profile'], df['risk_category'], normalize='index') * 100
    anomaly_cluster = anomaly_cluster.reindex(columns=risk_order)
    
    anomaly_cluster.plot(kind='bar', stacked=True, ax=ax, 
                         color=[risk_colors[r] for r in risk_order],
                         edgecolor='black', linewidth=0.5, width=0.8)
    ax.set_title('Risk Distribution by Cluster Profile', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster Profile', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.legend(title='Risk Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(fig_path / 'anomaly_detection_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    Saved: anomaly_detection_dashboard.png")
    
    # =========================================================================
    # FIGURE 3: TREND ANALYSIS RESULTS
    # =========================================================================
    logger.info("  Creating Trend Analysis Dashboard...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('AADHAAR SETU - Trend Analysis & Early Warning System', fontsize=18, fontweight='bold', y=1.02)
    
    # 3A: Trend Status Distribution
    ax = axes[0, 0]
    trend_colors = {'STABLE': '#2ca02c', 'DECLINING': '#d62728', 'SURGING': '#ff7f0e', 'UNUSUAL': '#9467bd'}
    trend_counts = df['trend_status'].value_counts()
    
    wedges, texts, autotexts = ax.pie(
        trend_counts.values,
        labels=trend_counts.index,
        autopct='%1.1f%%',
        colors=[trend_colors.get(t, '#7f7f7f') for t in trend_counts.index],
        explode=[0.05 if t != 'STABLE' else 0 for t in trend_counts.index],
        shadow=True,
        startangle=90
    )
    ax.set_title('Trend Status Distribution', fontsize=14, fontweight='bold')
    
    # 3B: Monthly Activity Trend
    ax = axes[0, 1]
    if 'year_month' in df.columns:
        monthly_activity = df.groupby('year_month')['total_activity'].sum()
        x_labels = [str(m) for m in monthly_activity.index]
        ax.plot(range(len(monthly_activity)), monthly_activity.values, 
                marker='o', linewidth=2, markersize=8, color='#1f77b4')
        ax.fill_between(range(len(monthly_activity)), monthly_activity.values, alpha=0.3, color='#1f77b4')
        
        # Add trend line
        z = np.polyfit(range(len(monthly_activity)), monthly_activity.values, 1)
        p = np.poly1d(z)
        ax.plot(range(len(monthly_activity)), p(range(len(monthly_activity))), 
                '--', color='red', linewidth=2, label=f'Trend Line')
        
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_title('Monthly Total Activity Trend', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Total Activity', fontsize=12)
        ax.legend()
    
    # 3C: Drop & Spike Alert Summary
    ax = axes[1, 0]
    alert_summary = pd.DataFrame({
        'Alert Type': ['Drop Alerts', 'Spike Alerts', 'Z-Score Anomalies'],
        'Count': [df['drop_alert'].sum(), df['spike_alert'].sum(), df.get('zscore_anomaly', pd.Series([0])).sum()]
    })
    bars = ax.bar(alert_summary['Alert Type'], alert_summary['Count'], 
                  color=['#d62728', '#ff7f0e', '#9467bd'], edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, alert_summary['Count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Alert Summary', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Alerts', fontsize=12)
    ax.set_ylim(0, alert_summary['Count'].max() * 1.2)
    
    # 3D: Activity Volatility by Cluster
    ax = axes[1, 1]
    if 'total_activity_volatility' in df.columns:
        volatility_by_cluster = df.groupby('cluster_profile')['total_activity_volatility'].mean().sort_values(ascending=False)
        volatility_by_cluster.plot(kind='bar', ax=ax, color=sns.color_palette("coolwarm", len(volatility_by_cluster)),
                                   edgecolor='black', linewidth=0.5)
        ax.axhline(y=df['total_activity_volatility'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Overall Mean: {df['total_activity_volatility'].mean():.2f}")
        ax.set_title('Activity Volatility by Cluster Profile', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster Profile', fontsize=12)
        ax.set_ylabel('Average Volatility', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(fig_path / 'trend_analysis_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    Saved: trend_analysis_dashboard.png")
    
    # =========================================================================
    # FIGURE 4: FEATURE IMPORTANCE & MODEL INSIGHTS
    # =========================================================================
    logger.info("  Creating Model Insights Dashboard...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('AADHAAR SETU - ML Model Insights', fontsize=18, fontweight='bold', y=1.02)
    
    # 4A: Feature Correlation Heatmap
    ax = axes[0, 0]
    feature_subset = [col for col in feature_cols if col in df.columns][:10]  # Top 10 features
    if len(feature_subset) > 3:
        corr_matrix = df[feature_subset].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                    cbar_kws={'label': 'Correlation'}, linewidths=0.5, square=True)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    
    # 4B: Feature Distributions by Cluster (Box Plot)
    ax = axes[0, 1]
    if 'total_activity' in df.columns:
        df_sample = df.sample(min(5000, len(df)), random_state=42)  # Sample for performance
        sns.boxplot(data=df_sample, x='cluster_profile', y='total_activity', ax=ax, palette='husl')
        ax.set_title('Activity Distribution by Cluster', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster Profile', fontsize=12)
        ax.set_ylabel('Total Activity', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
    
    # 4C: Cluster Centroids Visualization
    ax = axes[1, 0]
    cluster_centroids = df.groupby('cluster_profile')[['bio_ratio', 'demo_ratio', 'enrol_ratio']].mean()
    cluster_centroids.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title('Cluster Centroids (Activity Ratios)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster Profile', fontsize=12)
    ax.set_ylabel('Ratio', fontsize=12)
    ax.legend(title='Activity Type', labels=['Biometric', 'Demographic', 'Enrolment'])
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1)
    
    # 4D: Model Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    ML PIPELINE SUMMARY                       ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Run Timestamp:      {MLConfig.RUN_TIMESTAMP}               
    ║  Model Version:      {MLConfig.MODEL_VERSION}                     
    ╠══════════════════════════════════════════════════════════════╣
    ║  DATA STATISTICS                                             ║
    ║  ─────────────────────────────────────────────────────────── ║
    ║  Total Locations:    {metadata['statistics']['total_locations']:,}                            
    ║  Unique States:      {metadata['statistics']['unique_states']}                              
    ║  Unique Districts:   {metadata['statistics']['unique_districts']}                           
    ║  Unique Pincodes:    {metadata['statistics']['unique_pincodes']:,}                        
    ║  Total Activity:     {metadata['statistics']['total_activity']:,}                   
    ╠══════════════════════════════════════════════════════════════╣
    ║  MODEL RESULTS                                               ║
    ║  ─────────────────────────────────────────────────────────── ║
    ║  Clusters Created:   {len(metadata['cluster_distribution'])}                              
    ║  Anomalies Detected: {metadata['statistics']['anomaly_count']:,}                         
    ║  High Risk Regions:  {metadata['statistics']['high_risk_count']:,}                          
    ║  Drop Alerts:        {metadata['statistics']['drop_alert_count']:,}                          
    ║  Spike Alerts:       {metadata['statistics']['spike_alert_count']:,}                           
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.set_title('Model Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_path / 'model_insights_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    Saved: model_insights_dashboard.png")
    
    # =========================================================================
    # FIGURE 5: COMBINED EXECUTIVE SUMMARY (Single Page)
    # =========================================================================
    logger.info("  Creating Executive Summary Dashboard...")
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('AADHAAR SETU - ML Analytics Executive Summary', fontsize=22, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # 5A: Cluster Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_counts = df['cluster_profile'].value_counts()
    ax1.pie(cluster_counts.values, autopct='%1.0f%%', colors=sns.color_palette("husl", len(cluster_counts)),
            startangle=90, textprops={'fontsize': 9})
    ax1.set_title('Cluster Distribution', fontsize=12, fontweight='bold')
    
    # 5B: Risk Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    risk_counts = df['risk_category'].value_counts().reindex(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']).fillna(0)
    ax2.barh(risk_counts.index, risk_counts.values, 
             color=[risk_colors.get(r, '#7f7f7f') for r in risk_counts.index])
    ax2.set_title('Risk Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Count', fontsize=10)
    
    # 5C: Trend Status
    ax3 = fig.add_subplot(gs[0, 2])
    trend_counts = df['trend_status'].value_counts()
    ax3.pie(trend_counts.values, autopct='%1.0f%%', 
            colors=[trend_colors.get(t, '#7f7f7f') for t in trend_counts.index],
            startangle=90, textprops={'fontsize': 9})
    ax3.set_title('Trend Status', fontsize=12, fontweight='bold')
    
    # 5D: Key Metrics
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    metrics_text = f"""
    KEY METRICS
    ───────────────
    Total Locations: {metadata['statistics']['total_locations']:,}
    States: {metadata['statistics']['unique_states']}
    Districts: {metadata['statistics']['unique_districts']}
    
    High Risk: {metadata['statistics']['high_risk_count']:,}
    Alerts: {metadata['statistics']['drop_alert_count'] + metadata['statistics']['spike_alert_count']:,}
    """
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 5E: Monthly Trend (Full Width)
    ax5 = fig.add_subplot(gs[1, :2])
    if 'year_month' in df.columns:
        monthly_activity = df.groupby('year_month')['total_activity'].sum()
        ax5.plot(range(len(monthly_activity)), monthly_activity.values, 
                 marker='o', linewidth=2, markersize=6, color='#1f77b4')
        ax5.fill_between(range(len(monthly_activity)), monthly_activity.values, alpha=0.3)
        ax5.set_title('Monthly Activity Trend', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Total Activity', fontsize=10)
    
    # 5F: State-wise Risk (Heat-style)
    ax6 = fig.add_subplot(gs[1, 2:])
    state_risk = df.groupby('state').agg({
        'is_anomaly': 'sum',
        'total_activity': 'sum'
    }).nlargest(10, 'is_anomaly')
    ax6.barh(state_risk.index, state_risk['is_anomaly'], color='#d62728', alpha=0.8)
    ax6.set_title('Top 10 States by Risk Count', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Anomaly Count', fontsize=10)
    
    # 5G: Cluster Activity Comparison
    ax7 = fig.add_subplot(gs[2, :2])
    cluster_activity = df.groupby('cluster_profile')[['total_bio_updates', 'total_demo_updates', 'total_enrollments']].sum()
    cluster_activity.plot(kind='bar', ax=ax7, width=0.8)
    ax7.set_title('Total Activity by Cluster', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Cluster Profile', fontsize=10)
    ax7.set_ylabel('Total Activity', fontsize=10)
    ax7.tick_params(axis='x', rotation=45)
    ax7.legend(title='Type', fontsize=8)
    
    # 5H: Alert Timeline
    ax8 = fig.add_subplot(gs[2, 2:])
    if 'year_month' in df.columns:
        monthly_alerts = df.groupby('year_month').agg({
            'drop_alert': 'sum',
            'spike_alert': 'sum',
            'is_anomaly': 'sum'
        })
        monthly_alerts.plot(kind='bar', ax=ax8, width=0.8, color=['#d62728', '#ff7f0e', '#9467bd'])
        ax8.set_title('Monthly Alerts', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Month', fontsize=10)
        ax8.set_ylabel('Alert Count', fontsize=10)
        ax8.tick_params(axis='x', rotation=45)
        ax8.legend(['Drops', 'Spikes', 'Anomalies'], fontsize=8)
    
    plt.savefig(fig_path / 'executive_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    Saved: executive_summary.png")
    
    logger.info(f"\n  [OK] All visualizations saved to {fig_path}/")
    
    return fig_path


# ============================================================================
# 11. MODEL PERSISTENCE
# ============================================================================

def save_models(scaler, kmeans, iso_forest, feature_cols):
    """
    Save trained models for future reuse
    """
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 9: MODEL PERSISTENCE")
    logger.info("=" * 80)
    
    model_path = MLConfig.MODEL_PATH
    
    # Save scaler
    with open(model_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("  Saved: scaler.pkl")
    
    # Save K-Means model
    with open(model_path / 'kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    logger.info("  Saved: kmeans_model.pkl")
    
    # Save Isolation Forest model
    with open(model_path / 'isolation_forest_model.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)
    logger.info("  Saved: isolation_forest_model.pkl")
    
    # Save feature list
    with open(model_path / 'feature_columns.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    logger.info("  Saved: feature_columns.json")
    
    logger.info(f"\n  [OK] All models saved to {model_path}/")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main ML pipeline execution
    """
    logger.info("\n" + "=" * 80)
    logger.info("AADHAAR SETU - ML PIPELINE STARTED")
    logger.info(f"Run Timestamp: {MLConfig.RUN_TIMESTAMP}")
    logger.info(f"Model Version: {MLConfig.MODEL_VERSION}")
    logger.info("=" * 80 + "\n")
    
    try:
        # Stage 1: Load and validate data
        df = load_and_validate_data()
        
        # Stage 2: Feature engineering
        df, feature_cols = build_features(df)
        
        # Stage 3: Feature scaling
        X_scaled, scaler, feature_cols = scale_features(df, feature_cols)
        
        # Stage 4: Clustering
        df, kmeans, cluster_stats = train_clustering_model(X_scaled, df)
        
        # Stage 5: Anomaly detection
        df, iso_forest = train_anomaly_model(X_scaled, df)
        
        # Stage 6: Trend analysis
        df = analyze_trends(df)
        
        # Stage 7: Insight generation
        df = generate_insights(df, cluster_stats)
        
        # Stage 8: Output generation
        metadata = generate_outputs(df, cluster_stats, scaler, kmeans, iso_forest, feature_cols)
        
        # Stage 9: Visualizations
        fig_path = create_ml_visualizations(df, X_scaled, kmeans, iso_forest, feature_cols, metadata)
        
        # Stage 10: Model persistence
        save_models(scaler, kmeans, iso_forest, feature_cols)
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("ML PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nSUMMARY:")
        logger.info(f"  Total Locations Analyzed: {metadata['statistics']['total_locations']:,}")
        logger.info(f"  Clusters Created: {len(metadata['cluster_distribution'])}")
        logger.info(f"  High Risk Regions: {metadata['statistics']['high_risk_count']:,}")
        logger.info(f"  Drop Alerts: {metadata['statistics']['drop_alert_count']:,}")
        logger.info(f"  Spike Alerts: {metadata['statistics']['spike_alert_count']:,}")
        logger.info(f"\nOutputs saved to: {MLConfig.OUTPUT_PATH}/")
        logger.info(f"Figures saved to: {fig_path}/")
        logger.info(f"Models saved to: {MLConfig.MODEL_PATH}/")
        logger.info("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"\n[ERROR] ML PIPELINE FAILED: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
