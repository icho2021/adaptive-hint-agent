"""
Feedback Policy System

Rule-based and data-driven policies to decide when to trigger hints.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class FeedbackPolicy(ABC):
    """Base class for feedback policies"""
    
    @abstractmethod
    def should_trigger_hint(
        self, 
        features: Dict[str, float],
        state: str,
        attempt_num: int,
        history: List[Dict]
    ) -> bool:
        """Decide whether to trigger a hint"""
        pass
    
    @abstractmethod
    def get_policy_name(self) -> str:
        """Return policy name"""
        pass


class RuleBasedPolicy(FeedbackPolicy):
    """Rule-based hint triggering policy"""
    
    def __init__(
        self,
        trigger_on_confused: bool = True,
        trigger_on_consecutive_failures: int = 3,
        trigger_on_low_success_rate: float = 0.2,
        min_attempts_before_hint: int = 2,
        cooldown_after_hint: int = 2
    ):
        """
        Args:
            trigger_on_confused: Trigger hint when learner is confused
            trigger_on_consecutive_failures: Trigger after N consecutive failures
            trigger_on_low_success_rate: Trigger if recent success rate < threshold
            min_attempts_before_hint: Minimum attempts before first hint
            cooldown_after_hint: Don't trigger again for N attempts after hint
        """
        self.trigger_on_confused = trigger_on_confused
        self.trigger_on_consecutive_failures = trigger_on_consecutive_failures
        self.trigger_on_low_success_rate = trigger_on_low_success_rate
        self.min_attempts_before_hint = min_attempts_before_hint
        self.cooldown_after_hint = cooldown_after_hint
    
    def should_trigger_hint(
        self,
        features: Dict[str, float],
        state: str,
        attempt_num: int,
        history: List[Dict]
    ) -> bool:
        # Check minimum attempts
        if attempt_num < self.min_attempts_before_hint:
            return False
        
        # Check cooldown period
        if history:
            last_hint_attempt = max(
                [h.get('attempt_num', 0) for h in history if h.get('hint_triggered', False)],
                default=0
            )
            if attempt_num - last_hint_attempt < self.cooldown_after_hint:
                return False
        
        # Rule 1: Trigger on confused state
        if self.trigger_on_confused and state == 'confused':
            return True
        
        # Rule 2: Trigger on consecutive failures
        if features['consecutive_failures'] >= self.trigger_on_consecutive_failures:
            return True
        
        # Rule 3: Trigger on low recent success rate
        if features['recent_success_rate'] < self.trigger_on_low_success_rate:
            return True
        
        return False
    
    def get_policy_name(self) -> str:
        return "RuleBasedPolicy"


class DataDrivenPolicy(FeedbackPolicy):
    """Data-driven policy using learned thresholds"""
    
    def __init__(
        self,
        confusion_weight: float = 0.4,
        failure_weight: float = 0.3,
        success_rate_weight: float = 0.3,
        threshold: float = 0.6,
        min_attempts_before_hint: int = 2
    ):
        """
        Args:
            confusion_weight: Weight for confusion state indicator
            failure_weight: Weight for failure patterns
            success_rate_weight: Weight for success rate indicators
            threshold: Score threshold to trigger hint (0-1)
            min_attempts_before_hint: Minimum attempts before first hint
        """
        self.confusion_weight = confusion_weight
        self.failure_weight = failure_weight
        self.success_rate_weight = success_rate_weight
        self.threshold = threshold
        self.min_attempts_before_hint = min_attempts_before_hint
        
        # Normalize weights
        total = confusion_weight + failure_weight + success_rate_weight
        self.confusion_weight /= total
        self.failure_weight /= total
        self.success_rate_weight /= total
    
    def should_trigger_hint(
        self,
        features: Dict[str, float],
        state: str,
        attempt_num: int,
        history: List[Dict]
    ) -> bool:
        if attempt_num < self.min_attempts_before_hint:
            return False
        
        # Compute weighted score
        confusion_score = 1.0 if state == 'confused' else (0.5 if state == 'neutral' else 0.0)
        
        # Normalize consecutive failures (0-1 scale, assuming max ~10)
        failure_score = min(1.0, features['consecutive_failures'] / 5.0)
        
        # Invert success rate (low success = high need for hint)
        success_rate_score = 1.0 - features['recent_success_rate']
        
        # Weighted combination
        total_score = (
            self.confusion_weight * confusion_score +
            self.failure_weight * failure_score +
            self.success_rate_weight * success_rate_score
        )
        
        return total_score >= self.threshold
    
    def get_policy_name(self) -> str:
        return "DataDrivenPolicy"


class AdaptivePolicy(FeedbackPolicy):
    """Adaptive policy that adjusts based on learner response"""
    
    def __init__(
        self,
        base_policy: FeedbackPolicy,
        responsiveness_threshold: float = 0.3
    ):
        """
        Args:
            base_policy: Base policy to use
            responsiveness_threshold: Minimum improvement after hint to consider it effective
        """
        self.base_policy = base_policy
        self.responsiveness_threshold = responsiveness_threshold
        self.hint_effectiveness = {}  # Track hint effectiveness per learner
    
    def should_trigger_hint(
        self,
        features: Dict[str, float],
        state: str,
        attempt_num: int,
        history: List[Dict]
    ) -> bool:
        # Use base policy decision
        if not self.base_policy.should_trigger_hint(features, state, attempt_num, history):
            return False
        
        # Check if previous hints were effective
        if history:
            recent_hints = [h for h in history[-5:] if h.get('hint_triggered', False)]
            if recent_hints:
                # Check if learner improved after hints
                improvements = []
                for hint_event in recent_hints:
                    hint_attempt = hint_event['attempt_num']
                    # Check next 2 attempts after hint
                    post_hint = [h for h in history 
                                if h['attempt_num'] > hint_attempt and h['attempt_num'] <= hint_attempt + 2]
                    if post_hint:
                        pre_success = features.get('cumulative_success_rate', 0)
                        post_success = sum(h.get('is_correct', 0) for h in post_hint) / len(post_hint)
                        improvements.append(post_success - pre_success)
                
                if improvements:
                    avg_improvement = np.mean(improvements)
                    # If hints not effective, be more conservative
                    if avg_improvement < self.responsiveness_threshold:
                        return False
        
        return True
    
    def get_policy_name(self) -> str:
        return f"AdaptivePolicy({self.base_policy.get_policy_name()})"


class PolicyManager:
    """Manage and apply different feedback policies"""
    
    def __init__(self, policy: FeedbackPolicy):
        self.policy = policy
        self.hint_history = []  # Track hint triggers
    
    def evaluate_attempt(
        self,
        features: Dict[str, float],
        state: str,
        attempt_num: int,
        learner_id: str,
        problem_id: str
    ) -> Dict:
        """
        Evaluate an attempt and decide on hint
        
        Returns:
            Dict with 'hint_triggered' boolean and other metadata
        """
        # Get relevant history for this learner-problem pair
        relevant_history = [
            h for h in self.hint_history
            if h['learner_id'] == learner_id and h['problem_id'] == problem_id
        ]
        
        should_trigger = self.policy.should_trigger_hint(
            features, state, attempt_num, relevant_history
        )
        
        result = {
            'learner_id': learner_id,
            'problem_id': problem_id,
            'attempt_num': attempt_num,
            'hint_triggered': should_trigger,
            'state': state,
            'policy_name': self.policy.get_policy_name()
        }
        
        if should_trigger:
            self.hint_history.append(result)
        
        return result
    
    def reset_history(self):
        """Clear hint history"""
        self.hint_history = []


if __name__ == "__main__":
    from .data_generator import LearnerLogGenerator
    from .feature_engineering import FeatureEngineer
    from .state_inference import StateInference
    
    # Generate sample data
    generator = LearnerLogGenerator(seed=42)
    logs = generator.generate_multiple_sessions(n_learners=1, n_problems=1, attempts_per_session=10)
    
    engineer = FeatureEngineer()
    features_df = engineer.extract_features(logs)
    
    state_inference = StateInference(engineer)
    result_df = state_inference.infer_states_batch(features_df)
    
    # Test different policies
    print("Testing Feedback Policies:\n")
    
    policies = [
        RuleBasedPolicy(),
        DataDrivenPolicy(),
        AdaptivePolicy(RuleBasedPolicy())
    ]
    
    for policy in policies:
        manager = PolicyManager(policy)
        hints_triggered = []
        
        for idx, row in result_df.iterrows():
            features = engineer.get_feature_vector(row)
            result = manager.evaluate_attempt(
                features,
                row['inferred_state'],
                row['attempt_num'],
                row['learner_id'],
                row['problem_id']
            )
            hints_triggered.append(result['hint_triggered'])
        
        result_df[f'hints_{policy.get_policy_name()}'] = hints_triggered
        hint_count = sum(hints_triggered)
        print(f"{policy.get_policy_name()}: {hint_count} hints triggered out of {len(result_df)} attempts")
    
    print("\nSample results:")
    print(result_df[['attempt_num', 'is_correct', 'inferred_state', 
                     'hints_RuleBasedPolicy', 'hints_DataDrivenPolicy']].head(10))
