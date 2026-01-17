"""
Simulated Learner Interaction Log Generator

Generates realistic learner interaction logs with:
- Attempts (problem-solving attempts)
- Correctness (success/failure)
- Timing gaps (time between interactions)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random


class LearnerLogGenerator:
    """Generate simulated learner interaction logs"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
    
    def generate_learner_session(
        self,
        learner_id: str,
        problem_id: str,
        n_attempts: int = 10,
        base_success_rate: float = 0.3,
        confusion_level: float = 0.0,  # 0.0 = progressing, 1.0 = very confused
        start_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate a session of learner interactions
        
        Args:
            learner_id: Unique identifier for the learner
            problem_id: Problem being attempted
            n_attempts: Number of attempts in this session
            base_success_rate: Base probability of success (0-1)
            confusion_level: How confused the learner is (0-1)
            start_time: Starting timestamp (defaults to now)
        """
        if start_time is None:
            start_time = datetime.now()
        
        logs = []
        current_time = start_time
        consecutive_failures = 0
        
        # Adjust success rate based on confusion
        # Confused learners have lower success rates
        adjusted_success_rate = base_success_rate * (1 - confusion_level * 0.5)
        
        for attempt_num in range(1, n_attempts + 1):
            # Time gap between attempts (confused learners take longer)
            base_gap = self.rng.exponential(30)  # Base 30 seconds
            confusion_multiplier = 1 + confusion_level * 2
            time_gap = base_gap * confusion_multiplier
            
            current_time += timedelta(seconds=time_gap)
            
            # Determine correctness
            # More consecutive failures -> lower success chance
            failure_penalty = min(consecutive_failures * 0.1, 0.4)
            attempt_success_rate = max(0.1, adjusted_success_rate - failure_penalty)
            
            is_correct = self.rng.rand() < attempt_success_rate
            
            if is_correct:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            
            log_entry = {
                'learner_id': learner_id,
                'problem_id': problem_id,
                'attempt_num': attempt_num,
                'timestamp': current_time,
                'is_correct': is_correct,
                'time_gap_seconds': time_gap,
                'consecutive_failures': consecutive_failures,
                'session_duration_seconds': (current_time - start_time).total_seconds()
            }
            
            logs.append(log_entry)
        
        return pd.DataFrame(logs)
    
    def generate_multiple_sessions(
        self,
        n_learners: int = 5,
        n_problems: int = 3,
        attempts_per_session: int = 10,
        confusion_distribution: str = 'mixed'  # 'mixed', 'confused', 'progressing'
    ) -> pd.DataFrame:
        """
        Generate logs for multiple learners and problems
        
        Args:
            n_learners: Number of different learners
            n_problems: Number of problems
            attempts_per_session: Attempts per learner-problem session
            confusion_distribution: Distribution of confusion levels
        """
        all_logs = []
        
        for learner_idx in range(n_learners):
            learner_id = f"learner_{learner_idx:03d}"
            
            for problem_idx in range(n_problems):
                problem_id = f"problem_{problem_idx:03d}"
                
                # Assign confusion level based on distribution
                if confusion_distribution == 'mixed':
                    confusion = self.rng.beta(2, 3)  # Skewed toward lower confusion
                elif confusion_distribution == 'confused':
                    confusion = self.rng.uniform(0.6, 1.0)
                else:  # progressing
                    confusion = self.rng.uniform(0.0, 0.4)
                
                # Vary base success rate
                base_success_rate = self.rng.uniform(0.2, 0.5)
                
                session_logs = self.generate_learner_session(
                    learner_id=learner_id,
                    problem_id=problem_id,
                    n_attempts=attempts_per_session,
                    base_success_rate=base_success_rate,
                    confusion_level=confusion
                )
                
                all_logs.append(session_logs)
        
        combined_logs = pd.concat(all_logs, ignore_index=True)
        return combined_logs.sort_values('timestamp').reset_index(drop=True)


if __name__ == "__main__":
    # Demo: Generate sample logs
    generator = LearnerLogGenerator(seed=42)
    
    # Generate logs for 3 learners, 2 problems each
    logs = generator.generate_multiple_sessions(
        n_learners=3,
        n_problems=2,
        attempts_per_session=8,
        confusion_distribution='mixed'
    )
    
    print("Generated Learner Interaction Logs:")
    print(f"Total entries: {len(logs)}")
    print("\nSample entries:")
    print(logs.head(10))
    print("\nStatistics:")
    print(logs.groupby('learner_id').agg({
        'is_correct': 'mean',
        'time_gap_seconds': 'mean',
        'consecutive_failures': 'max'
    }))
