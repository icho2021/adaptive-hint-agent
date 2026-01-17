"""
Learner State Inference

Infer simple learner states such as "confused" or "progressing"
based on behavioral features.
"""

import pandas as pd
import numpy as np
from typing import Dict
from .feature_engineering import FeatureEngineer


class StateInference:
    """Infer learner state from behavioral features"""
    
    def __init__(self, feature_engineer: FeatureEngineer = None):
        self.feature_engineer = feature_engineer or FeatureEngineer()
        
        # Thresholds for state classification (can be tuned)
        self.confusion_thresholds = {
            'consecutive_failures': 3,  # 3+ consecutive failures
            'recent_success_rate': 0.2,  # <20% success in recent attempts
            'failure_rate': 0.7,  # >70% overall failure rate
            'time_gap_multiplier': 1.5  # Time gaps 1.5x above average
        }
        
        self.progressing_thresholds = {
            'recent_success_rate': 0.5,  # >50% success in recent attempts
            'success_trend': 0.1,  # Improving trend
            'cumulative_success_rate': 0.4  # >40% overall success
        }
    
    def infer_state(
        self, 
        features: Dict[str, float]
    ) -> str:  # Returns 'confused', 'progressing', or 'neutral'
        """
        Infer learner state from feature vector
        
        Args:
            features: Dictionary of feature names to values
            
        Returns:
            'confused', 'progressing', or 'neutral'
        """
        confusion_score = 0
        progressing_score = 0
        
        # Confusion indicators
        if features['consecutive_failures'] >= self.confusion_thresholds['consecutive_failures']:
            confusion_score += 2
        if features['recent_success_rate'] < self.confusion_thresholds['recent_success_rate']:
            confusion_score += 2
        if features['failure_rate'] > self.confusion_thresholds['failure_rate']:
            confusion_score += 1
        if features['recent_avg_time_gap'] > features.get('avg_time_per_attempt', 30) * self.confusion_thresholds['time_gap_multiplier']:
            confusion_score += 1
        
        # Progressing indicators
        if features['recent_success_rate'] > self.progressing_thresholds['recent_success_rate']:
            progressing_score += 2
        if features['success_trend'] > self.progressing_thresholds['success_trend']:
            progressing_score += 1
        if features['cumulative_success_rate'] > self.progressing_thresholds['cumulative_success_rate']:
            progressing_score += 1
        
        # Classify based on scores
        if confusion_score >= 3:
            return 'confused'
        elif progressing_score >= 2:
            return 'progressing'
        else:
            return 'neutral'
    
    def infer_states_batch(self, logs_with_features: pd.DataFrame) -> pd.DataFrame:
        """
        Infer states for all log entries
        
        Args:
            logs_with_features: DataFrame with extracted features
            
        Returns:
            DataFrame with added 'inferred_state' column
        """
        result_df = logs_with_features.copy()
        
        # Extract feature vectors and infer states
        states = []
        for idx, row in result_df.iterrows():
            feature_vector = self.feature_engineer.get_feature_vector(row)
            state = self.infer_state(feature_vector)
            states.append(state)
        
        result_df['inferred_state'] = states
        return result_df
    
    def get_state_confidence(
        self, 
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get confidence scores for each state
        
        Returns:
            Dict with 'confused', 'progressing', 'neutral' confidence scores (0-1)
        """
        state = self.infer_state(features)
        
        # Simple confidence based on how clear the indicators are
        confusion_indicators = sum([
            features['consecutive_failures'] >= self.confusion_thresholds['consecutive_failures'],
            features['recent_success_rate'] < self.confusion_thresholds['recent_success_rate'],
            features['failure_rate'] > self.confusion_thresholds['failure_rate']
        ])
        
        progressing_indicators = sum([
            features['recent_success_rate'] > self.progressing_thresholds['recent_success_rate'],
            features['success_trend'] > self.progressing_thresholds['success_trend'],
            features['cumulative_success_rate'] > self.progressing_thresholds['cumulative_success_rate']
        ])
        
        total_indicators = confusion_indicators + progressing_indicators
        
        if state == 'confused':
            confidence = min(0.9, 0.5 + confusion_indicators * 0.15)
        elif state == 'progressing':
            confidence = min(0.9, 0.5 + progressing_indicators * 0.15)
        else:
            confidence = 0.6  # Neutral is default
        
        return {
            'confused': confidence if state == 'confused' else (1 - confidence) / 2,
            'progressing': confidence if state == 'progressing' else (1 - confidence) / 2,
            'neutral': confidence if state == 'neutral' else (1 - confidence) / 2,
            'predicted_state': state
        }


if __name__ == "__main__":
    from .data_generator import LearnerLogGenerator
    
    # Generate and process sample data
    generator = LearnerLogGenerator(seed=42)
    logs = generator.generate_multiple_sessions(n_learners=2, n_problems=1, attempts_per_session=10)
    
    engineer = FeatureEngineer()
    features_df = engineer.extract_features(logs)
    
    # Infer states
    state_inference = StateInference(engineer)
    result_df = state_inference.infer_states_batch(features_df)
    
    print("State Inference Results:")
    print(f"\nState distribution:")
    print(result_df['inferred_state'].value_counts())
    print("\nSample results:")
    print(result_df[['learner_id', 'attempt_num', 'is_correct', 
                     'consecutive_failures', 'inferred_state']].head(15))
