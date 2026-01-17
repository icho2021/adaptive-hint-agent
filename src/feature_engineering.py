"""
Behavioral Feature Engineering

Extract features from learner interaction logs:
- Consecutive failures
- Recent activity patterns
- Success rate trends
- Time-based features
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class FeatureEngineer:
    """Extract behavioral features from learner logs"""
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Number of recent attempts to consider for rolling features
        """
        self.window_size = window_size
    
    def extract_features(self, logs: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features for each log entry
        
        Args:
            logs: DataFrame with columns: learner_id, problem_id, attempt_num, 
                  timestamp, is_correct, time_gap_seconds, consecutive_failures
        """
        features_df = logs.copy()
        
        # Group by learner and problem to compute session-level features
        grouped = features_df.groupby(['learner_id', 'problem_id'])
        
        # Rolling window features
        features_df['recent_success_rate'] = grouped['is_correct'].transform(
            lambda x: x.rolling(window=min(self.window_size, len(x)), min_periods=1).mean()
        )
        
        features_df['recent_avg_time_gap'] = grouped['time_gap_seconds'].transform(
            lambda x: x.rolling(window=min(self.window_size, len(x)), min_periods=1).mean()
        )
        
        # Cumulative features
        features_df['cumulative_success_rate'] = grouped['is_correct'].transform(
            lambda x: x.expanding().mean()
        )
        
        features_df['total_attempts'] = grouped['attempt_num'].transform('max')
        features_df['attempt_progress'] = features_df['attempt_num'] / features_df['total_attempts']
        
        # Time-based features
        features_df['time_since_start'] = grouped['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds()
        )
        features_df['avg_time_per_attempt'] = (
            features_df['time_since_start'] / features_df['attempt_num']
        )
        
        # Trend features (is performance improving or declining?)
        features_df['success_trend'] = grouped['is_correct'].transform(
            lambda x: x.rolling(window=3, min_periods=2).mean().diff().fillna(0)
        )
        
        # Activity intensity (attempts per minute)
        features_df['activity_intensity'] = (
            features_df['attempt_num'] / (features_df['time_since_start'] / 60 + 1)
        )
        
        # Failure patterns
        features_df['max_consecutive_failures'] = grouped['consecutive_failures'].transform('max')
        features_df['failure_rate'] = 1 - features_df['cumulative_success_rate']
        
        # Recent activity flag (active in last N attempts)
        features_df['recent_activity'] = (
            features_df['time_gap_seconds'] < features_df['recent_avg_time_gap'] * 1.5
        ).astype(int)
        
        return features_df
    
    def get_feature_vector(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract feature vector for a single log entry
        
        Returns dict of feature names to values
        """
        return {
            'consecutive_failures': float(row['consecutive_failures']),
            'recent_success_rate': float(row['recent_success_rate']),
            'recent_avg_time_gap': float(row['recent_avg_time_gap']),
            'cumulative_success_rate': float(row['cumulative_success_rate']),
            'attempt_progress': float(row['attempt_progress']),
            'time_since_start': float(row['time_since_start']),
            'success_trend': float(row['success_trend']),
            'activity_intensity': float(row['activity_intensity']),
            'max_consecutive_failures': float(row['max_consecutive_failures']),
            'failure_rate': float(row['failure_rate']),
            'recent_activity': float(row['recent_activity'])
        }


if __name__ == "__main__":
    from .data_generator import LearnerLogGenerator
    
    # Generate sample data
    generator = LearnerLogGenerator(seed=42)
    logs = generator.generate_multiple_sessions(n_learners=2, n_problems=1, attempts_per_session=10)
    
    # Extract features
    engineer = FeatureEngineer(window_size=5)
    features_df = engineer.extract_features(logs)
    
    print("Feature Engineering Results:")
    print(f"\nTotal features: {len([c for c in features_df.columns if c not in ['learner_id', 'problem_id', 'timestamp']])}")
    print("\nSample feature vectors:")
    print(features_df[['learner_id', 'attempt_num', 'consecutive_failures', 
                       'recent_success_rate', 'cumulative_success_rate', 
                       'success_trend', 'activity_intensity']].head(10))
