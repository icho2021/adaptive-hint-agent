"""
Adaptive Hint Agent - Source Package

A system for processing learner interaction logs, inferring learner states,
and providing adaptive hints using rule-based and LLM-powered strategies.
"""

__version__ = "1.0.0"

from .data_generator import LearnerLogGenerator
from .feature_engineering import FeatureEngineer
from .state_inference import StateInference
from .feedback_policy import (
    FeedbackPolicy,
    RuleBasedPolicy,
    DataDrivenPolicy,
    AdaptivePolicy,
    PolicyManager
)
from .hint_generator import HintGenerator
from .trace_replay import TraceReplay, StrategyMetrics

__all__ = [
    'LearnerLogGenerator',
    'FeatureEngineer',
    'StateInference',
    'FeedbackPolicy',
    'RuleBasedPolicy',
    'DataDrivenPolicy',
    'AdaptivePolicy',
    'PolicyManager',
    'HintGenerator',
    'TraceReplay',
    'StrategyMetrics'
]
