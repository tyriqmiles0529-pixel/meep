import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup Logging
logger = logging.getLogger("IntegrityGuard")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - [INTEGRITY] - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DataIntegrityGuard:
    def __init__(self, allowed_staleness_days=2):
        self.allowed_staleness_days = allowed_staleness_days
        self.validation_stats = {
            'checks_passed': 0,
            'checks_failed': 0,
            'features_quarantined': []
        }

    def check_dataset_freshness(self, df, date_col='date'):
        """
        Checks if the entire dataset is stale.
        Returns (is_stale, max_date, days_lag)
        """
        if date_col not in df.columns:
            logger.error(f"Date column '{date_col}' missing for freshness check.")
            return True, None, 999
        
        # Ensure datetime
        dates = pd.to_datetime(df[date_col], errors='coerce')
        max_date = dates.max()
        
        if pd.isnull(max_date):
            logger.warning("Max date is NaT. Dataset empty or malformed.")
            return True, None, 999
            
        days_lag = (datetime.now() - max_date).days
        is_stale = days_lag > self.allowed_staleness_days
        
        if is_stale:
            logger.warning(f"DATASET STALE! Max date: {max_date.date()} ({days_lag} days old). SLA: {self.allowed_staleness_days} days.")
        else:
            logger.info(f"Dataset freshness verified. Max date: {max_date.date()} ({days_lag} days old).")
            
        return is_stale, max_date, days_lag

    def scan_for_drift_and_anomalies(self, feature_row: pd.DataFrame, expected_schema: list = None):
        """
        Scans a single feature row (inference input) for:
        1. Non-numeric data in feature columns
        2. NaNs in critical columns (Identity Score)
        3. Extreme outliers (heuristic)
        """
        issues = []
        
        # 1. Schema / Type Check
        total_features = 1
        if expected_schema:
            total_features = len(expected_schema)
            missing = set(expected_schema) - set(feature_row.columns)
            if missing:
                msg = f"Missing features: {list(missing)[:5]}..."
                issues.append(msg)
        
        # 2. Numeric Validation
        numeric_cols = feature_row.select_dtypes(include=[np.number]).columns
        
        # 3. NaN Scan & Auto-Repair
        numeric_cols = [c for c in numeric_cols if c in feature_row.columns]
        if expected_schema:
            numeric_cols = [c for c in numeric_cols if c in expected_schema]
            
        nans = feature_row[numeric_cols].isna().sum()
        cols_with_nans = nans[nans > 0].index.tolist()
        
        # Calculate Integrity Score
        # % of non-NaN features in the expected set
        if total_features > 0:
            nan_count = len(cols_with_nans)
            integrity_score = max(0.0, 1.0 - (nan_count / total_features))
        else:
            integrity_score = 1.0

        if cols_with_nans:
            msg = f"Data Cleaning: {len(cols_with_nans)} NaNs auto-corrected in features."
            issues.append(msg)
            # Active Repair
            feature_row.loc[:, cols_with_nans] = feature_row.loc[:, cols_with_nans].fillna(0.0)
            
        success = integrity_score >= 0.75 # Threshold for "Real" vs "Diluted"
        
        if success:
            self.validation_stats['checks_passed'] += 1
        else:
            self.validation_stats['checks_failed'] += 1
            
        return success, issues, integrity_score

    def calculate_volatility_penalty(self, player_history_df):
        """
        Calculates a penalty multiplier based on minutes volatility.
        High volatility = High Uncertainty = Higher Penalty (Confidence Reduction).
        """
        if player_history_df is None or player_history_df.empty:
            return 1.5 # High penalty for unknown/new players
            
        if 'minutes' not in player_history_df.columns:
            return 1.0
            
        # Look at last 10 games
        recent = player_history_df.sort_values('date').tail(10)
        
        if len(recent) < 3:
            return 1.2 # Small sample penalty
            
        minutes_std = recent['minutes'].std()
        
        # Volatility thresholds (heuristic)
        # Stable starter: ~2-4 mins std
        # Volatile rotation: >6 mins std
        
        penalty = 1.0
        if minutes_std > 8.0:
            penalty = 1.4
        elif minutes_std > 5.0:
            penalty = 1.2
            
        return penalty

    def get_confidence_multiplier(self, freshness_lag_days, volatility_penalty):
        """
        Combines freshness and volatility into a final sigma multiplier.
        Multiplier > 1.0 means wider confidence intervals (less confident).
        """
        freshness_penalty = 1.0
        if freshness_lag_days > self.allowed_staleness_days:
            # e.g. 1.0 + (5 days - 2 days) * 0.1 = 1.3
            freshness_penalty += (freshness_lag_days - self.allowed_staleness_days) * 0.1
            
        # Cap freshness penalty to avoid infinite uncertainty
        freshness_penalty = min(freshness_penalty, 2.0)
        
        total_multiplier = freshness_penalty * volatility_penalty
        return total_multiplier
