"""
Trace-Replay Script for Strategy Comparison

Compare different pacing strategies in terms of hint frequency and responsiveness.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from .feedback_policy import FeedbackPolicy, PolicyManager
from .hint_generator import HintGenerator
from .feature_engineering import FeatureEngineer
from .state_inference import StateInference


@dataclass
class StrategyMetrics:
    """Metrics for evaluating a strategy"""
    policy_name: str
    total_hints: int
    hint_frequency: float  # hints per attempt
    avg_hints_per_learner: float
    hints_to_confused: int
    hints_to_progressing: int
    hints_to_neutral: int
    responsiveness_score: float  # improvement after hints
    avg_time_to_first_hint: float
    hint_effectiveness: float  # % of hints followed by improvement


class TraceReplay:
    """Replay interaction traces with different strategies"""
    
    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        state_inference: StateInference,
        hint_generator: HintGenerator
    ):
        self.feature_engineer = feature_engineer
        self.state_inference = state_inference
        self.hint_generator = hint_generator
    
    def replay_with_policy(
        self,
        logs: pd.DataFrame,
        policy: FeedbackPolicy,
        problem_descriptions: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Replay interaction logs with a specific policy
        
        Args:
            logs: Interaction logs with features and states
            policy: Feedback policy to use
            problem_descriptions: Optional dict mapping problem_id to description
        
        Returns:
            DataFrame with added hint trigger information
        """
        if problem_descriptions is None:
            problem_descriptions = {
                pid: f"Problem {pid}" for pid in logs['problem_id'].unique()
            }
        
        # Extract features if not already present
        if 'recent_success_rate' not in logs.columns:
            logs = self.feature_engineer.extract_features(logs)
        
        if 'inferred_state' not in logs.columns:
            logs = self.state_inference.infer_states_batch(logs)
        
        # Initialize policy manager
        manager = PolicyManager(policy)
        manager.reset_history()
        
        results = []
        
        # Process each attempt
        for idx, row in logs.iterrows():
            features = self.feature_engineer.get_feature_vector(row)
            
            # Evaluate hint trigger
            hint_result = manager.evaluate_attempt(
                features,
                row['inferred_state'],
                row['attempt_num'],
                row['learner_id'],
                row['problem_id']
            )
            
            # Generate hint if triggered
            hint_info = None
            if hint_result['hint_triggered']:
                # Get recent attempts for context
                learner_logs = logs[
                    (logs['learner_id'] == row['learner_id']) &
                    (logs['problem_id'] == row['problem_id']) &
                    (logs['attempt_num'] <= row['attempt_num'])
                ]
                recent_attempts = learner_logs.tail(3).to_dict('records')
                
                hint_info = self.hint_generator.generate_hint(
                    problem_id=row['problem_id'],
                    problem_description=problem_descriptions.get(
                        row['problem_id'], 
                        f"Problem {row['problem_id']}"
                    ),
                    state=row['inferred_state'],
                    features=features,
                    attempt_num=row['attempt_num'],
                    recent_attempts=recent_attempts
                )
            
            result_row = row.to_dict()
            result_row['hint_triggered'] = hint_result['hint_triggered']
            result_row['policy_name'] = policy.get_policy_name()
            
            if hint_info:
                result_row.update({
                    'hint_text': hint_info['hint_text'],
                    'hint_detail_level': hint_info['detail_level'],
                    'hint_tone': hint_info['tone']
                })
            else:
                result_row.update({
                    'hint_text': None,
                    'hint_detail_level': None,
                    'hint_tone': None
                })
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def compute_metrics(
        self,
        replay_results: pd.DataFrame
    ) -> StrategyMetrics:
        """
        Compute evaluation metrics for a strategy
        
        Args:
            replay_results: Results from replay_with_policy
        
        Returns:
            StrategyMetrics object
        """
        policy_name = replay_results['policy_name'].iloc[0]
        total_attempts = len(replay_results)
        total_hints = replay_results['hint_triggered'].sum()
        
        hint_frequency = total_hints / total_attempts if total_attempts > 0 else 0
        
        # Hints per learner
        hints_per_learner = replay_results.groupby('learner_id')['hint_triggered'].sum()
        avg_hints_per_learner = hints_per_learner.mean()
        
        # Hints by state
        hint_rows = replay_results[replay_results['hint_triggered']]
        hints_to_confused = (hint_rows['inferred_state'] == 'confused').sum()
        hints_to_progressing = (hint_rows['inferred_state'] == 'progressing').sum()
        hints_to_neutral = (hint_rows['inferred_state'] == 'neutral').sum()
        
        # Responsiveness: improvement after hints
        responsiveness_scores = []
        for learner_id in replay_results['learner_id'].unique():
            learner_data = replay_results[
                replay_results['learner_id'] == learner_id
            ].sort_values('attempt_num')
            
            hint_indices = learner_data[learner_data['hint_triggered']].index.tolist()
            
            for hint_idx in hint_indices:
                hint_attempt = learner_data.loc[hint_idx, 'attempt_num']
                
                # Get success rate before hint
                before = learner_data[learner_data['attempt_num'] < hint_attempt]
                before_rate = before['is_correct'].mean() if len(before) > 0 else 0
                
                # Get success rate after hint (next 2-3 attempts)
                after = learner_data[
                    (learner_data['attempt_num'] > hint_attempt) &
                    (learner_data['attempt_num'] <= hint_attempt + 3)
                ]
                after_rate = after['is_correct'].mean() if len(after) > 0 else before_rate
                
                improvement = after_rate - before_rate
                responsiveness_scores.append(improvement)
        
        responsiveness_score = np.mean(responsiveness_scores) if responsiveness_scores else 0
        
        # Time to first hint
        first_hint_times = []
        for learner_id in replay_results['learner_id'].unique():
            learner_data = replay_results[
                replay_results['learner_id'] == learner_id
            ].sort_values('attempt_num')
            
            first_hint = learner_data[learner_data['hint_triggered']]
            if len(first_hint) > 0:
                first_hint_time = first_hint.iloc[0]['time_since_start']
                first_hint_times.append(first_hint_time)
        
        avg_time_to_first_hint = np.mean(first_hint_times) if first_hint_times else 0
        
        # Hint effectiveness (hints followed by improvement)
        effective_hints = sum(1 for score in responsiveness_scores if score > 0)
        hint_effectiveness = effective_hints / len(responsiveness_scores) if responsiveness_scores else 0
        
        return StrategyMetrics(
            policy_name=policy_name,
            total_hints=total_hints,
            hint_frequency=hint_frequency,
            avg_hints_per_learner=avg_hints_per_learner,
            hints_to_confused=hints_to_confused,
            hints_to_progressing=hints_to_progressing,
            hints_to_neutral=hints_to_neutral,
            responsiveness_score=responsiveness_score,
            avg_time_to_first_hint=avg_time_to_first_hint,
            hint_effectiveness=hint_effectiveness
        )
    
    def compare_strategies(
        self,
        logs: pd.DataFrame,
        policies: List[FeedbackPolicy],
        problem_descriptions: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            logs: Interaction logs
            policies: List of policies to compare
            problem_descriptions: Optional problem descriptions
        
        Returns:
            DataFrame with metrics for each strategy
        """
        all_metrics = []
        
        for policy in policies:
            print(f"Replaying with {policy.get_policy_name()}...")
            replay_results = self.replay_with_policy(logs, policy, problem_descriptions)
            metrics = self.compute_metrics(replay_results)
            all_metrics.append(metrics)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame([
            {
                'Policy': m.policy_name,
                'Total Hints': m.total_hints,
                'Hint Frequency': f"{m.hint_frequency:.3f}",
                'Avg Hints/Learner': f"{m.avg_hints_per_learner:.2f}",
                'Hints to Confused': m.hints_to_confused,
                'Hints to Progressing': m.hints_to_progressing,
                'Hints to Neutral': m.hints_to_neutral,
                'Responsiveness Score': f"{m.responsiveness_score:.3f}",
                'Avg Time to First Hint (s)': f"{m.avg_time_to_first_hint:.1f}",
                'Hint Effectiveness': f"{m.hint_effectiveness:.2%}"
            }
            for m in all_metrics
        ])
        
        return metrics_df


if __name__ == "__main__":
    from .data_generator import LearnerLogGenerator
    from .feedback_policy import RuleBasedPolicy, DataDrivenPolicy, AdaptivePolicy
    
    # Generate sample data
    print("Generating sample interaction logs...")
    generator = LearnerLogGenerator(seed=42)
    logs = generator.generate_multiple_sessions(
        n_learners=5,
        n_problems=2,
        attempts_per_session=10,
        confusion_distribution='mixed'
    )
    
    # Set up components
    engineer = FeatureEngineer()
    state_inference = StateInference(engineer)
    hint_generator = HintGenerator(use_llm=False)  # Use fallback
    
    # Set up trace replay
    replay = TraceReplay(engineer, state_inference, hint_generator)
    
    # Compare strategies
    policies = [
        RuleBasedPolicy(),
        DataDrivenPolicy(),
        AdaptivePolicy(RuleBasedPolicy())
    ]
    
    print("\nComparing strategies...")
    comparison = replay.compare_strategies(logs, policies)
    
    print("\n" + "="*80)
    print("Strategy Comparison Results:")
    print("="*80)
    print(comparison.to_string(index=False))
