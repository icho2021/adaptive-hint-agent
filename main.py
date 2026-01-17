"""
Main Demo Script for Adaptive Hint Agent

Demonstrates the complete pipeline:
1. Generate simulated learner interaction logs
2. Extract behavioral features
3. Infer learner states
4. Apply feedback policies
5. Generate adaptive hints
6. Compare strategies via trace replay
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import LearnerLogGenerator
from src.feature_engineering import FeatureEngineer
from src.state_inference import StateInference
from src.feedback_policy import RuleBasedPolicy, DataDrivenPolicy, AdaptivePolicy, PolicyManager
from src.hint_generator import HintGenerator
from src.trace_replay import TraceReplay
import pandas as pd


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_basic_pipeline(provider="auto", model=None):
    """Demonstrate the basic pipeline for a single learner"""
    print_section("Demo: Basic Pipeline - Single Learner Session")
    
    # Step 1: Generate logs
    print("Step 1: Generating simulated learner interaction logs...")
    generator = LearnerLogGenerator(seed=42)
    logs = generator.generate_learner_session(
        learner_id="learner_001",
        problem_id="problem_001",
        n_attempts=10,
        base_success_rate=0.3,
        confusion_level=0.7  # Confused learner
    )
    print(f"Generated {len(logs)} interaction records")
    print("\nSample logs:")
    print(logs[['attempt_num', 'is_correct', 'time_gap_seconds', 'consecutive_failures']].head())
    
    # Step 2: Extract features
    print("\nStep 2: Extracting behavioral features...")
    engineer = FeatureEngineer()
    features_df = engineer.extract_features(logs)
    print("Extracted features:")
    feature_cols = ['recent_success_rate', 'cumulative_success_rate', 
                    'success_trend', 'activity_intensity', 'failure_rate']
    print(features_df[['attempt_num'] + feature_cols].head())
    
    # Step 3: Infer states
    print("\nStep 3: Inferring learner states...")
    state_inference = StateInference(engineer)
    result_df = state_inference.infer_states_batch(features_df)
    print("State distribution:")
    print(result_df['inferred_state'].value_counts())
    print("\nState inference results:")
    print(result_df[['attempt_num', 'is_correct', 'consecutive_failures', 'inferred_state']].head(10))
    
    # Step 4: Apply feedback policy
    print("\nStep 4: Applying feedback policy...")
    policy = RuleBasedPolicy()
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
    
    result_df['hint_triggered'] = hints_triggered
    print(f"Total hints triggered: {sum(hints_triggered)} out of {len(result_df)} attempts")
    print("\nHint triggers:")
    hint_rows = result_df[result_df['hint_triggered']]
    print(hint_rows[['attempt_num', 'inferred_state', 'consecutive_failures']])
    
    # Step 5: Generate hints
    print("\nStep 5: Generating adaptive hints...")
    # Try to use LLM if available, fallback to templates
    hint_generator = HintGenerator(use_llm=True, provider=provider, model=model)
    
    for idx, row in hint_rows.iterrows():
        features = engineer.get_feature_vector(row)
        hint = hint_generator.generate_hint(
            problem_id=row['problem_id'],
            problem_description="Solve a quadratic equation",
            state=row['inferred_state'],
            features=features,
            attempt_num=row['attempt_num']
        )
        print(f"\nAttempt {row['attempt_num']} (State: {row['inferred_state']}):")
        print(f"  Detail: {hint['detail_level']}, Tone: {hint['tone']}")
        print(f"  Hint: {hint['hint_text']}")


def demo_strategy_comparison(provider="auto", model=None):
    """Demonstrate strategy comparison via trace replay"""
    print_section("Demo: Strategy Comparison via Trace Replay")
    
    # Generate logs for multiple learners
    print("Generating interaction logs for multiple learners...")
    generator = LearnerLogGenerator(seed=42)
    logs = generator.generate_multiple_sessions(
        n_learners=5,
        n_problems=2,
        attempts_per_session=10,
        confusion_distribution='mixed'
    )
    print(f"Generated {len(logs)} interaction records for {logs['learner_id'].nunique()} learners")
    
    # Set up components
    engineer = FeatureEngineer()
    state_inference = StateInference(engineer)
    # Try to use LLM if available, fallback to templates
    hint_generator = HintGenerator(use_llm=True, provider=provider, model=model)
    replay = TraceReplay(engineer, state_inference, hint_generator)
    
    # Define policies to compare
    policies = [
        RuleBasedPolicy(trigger_on_consecutive_failures=3),
        RuleBasedPolicy(trigger_on_consecutive_failures=2),  # More aggressive
        DataDrivenPolicy(threshold=0.6),
        DataDrivenPolicy(threshold=0.5),  # More sensitive
        AdaptivePolicy(RuleBasedPolicy())
    ]
    
    print(f"\nComparing {len(policies)} strategies...")
    comparison = replay.compare_strategies(logs, policies)
    
    print("\nStrategy Comparison Results:")
    print(comparison.to_string(index=False))
    
    # Additional analysis
    print("\n" + "-"*80)
    print("Key Insights:")
    print("-"*80)
    
    # Find most responsive strategy
    comparison_numeric = comparison.copy()
    comparison_numeric['Responsiveness'] = comparison_numeric['Responsiveness Score'].str.replace('−', '-').astype(float)
    best_responsive = comparison_numeric.loc[comparison_numeric['Responsiveness'].idxmax()]
    print(f"Most responsive strategy: {best_responsive['Policy']} "
          f"(score: {best_responsive['Responsiveness']:.3f})")
    
    # Find most efficient (hints per improvement)
    comparison_numeric['Hint Freq'] = comparison_numeric['Hint Frequency'].astype(float)
    most_efficient = comparison_numeric.loc[
        (comparison_numeric['Hint Freq'] > 0) &
        (comparison_numeric['Responsiveness'] > 0)
    ]
    if len(most_efficient) > 0:
        most_efficient['Efficiency'] = most_efficient['Responsiveness'] / most_efficient['Hint Freq']
        best_efficient = most_efficient.loc[most_efficient['Efficiency'].idxmax()]
        print(f"Most efficient strategy: {best_efficient['Policy']} "
              f"(responsiveness/hint: {best_efficient['Efficiency']:.3f})")


def demo_full_system(provider="auto", model=None):
    """Run the complete system end-to-end"""
    print_section("Full System Demo")
    
    print("This demo shows the complete Adaptive Hint Agent system:")
    print("1. Simulated learner interaction logs")
    print("2. Behavioral feature engineering")
    print("3. Learner state inference")
    print("4. Feedback policy application")
    print("5. Adaptive hint generation")
    print("6. Strategy comparison and evaluation")
    
    # Run basic pipeline
    demo_basic_pipeline(provider=provider, model=model)
    
    # Run strategy comparison
    demo_strategy_comparison(provider=provider, model=model)
    
    print_section("Demo Complete")
    print("✅ Demo completed successfully!")
    print("✅ The system is ready for use!")
    print("\nOptional next steps (for enhanced functionality):")
    print("- Local LLM: Install Ollama and run 'ollama pull llama3:8b-instruct'")
    print("- OpenAI API: Set OPENAI_API_KEY environment variable")
    print("- Adjust policy parameters in feedback_policy.py")
    print("- Customize hint templates in hint_generator.py")
    print("- Add your own problem descriptions and real interaction logs")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Hint Agent Demo")
    parser.add_argument(
        '--demo',
        choices=['basic', 'comparison', 'full'],
        default='full',
        help='Which demo to run (default: full)'
    )
    parser.add_argument(
        '--llm-provider',
        choices=['auto', 'ollama', 'openai', 'template'],
        default='auto',
        help='LLM provider to use: auto (try Ollama→OpenAI→template), ollama, openai, or template (default: auto)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default=None,
        help='Model name (e.g., "llama3:8b-instruct" for Ollama, "gpt-3.5-turbo" for OpenAI)'
    )
    
    args = parser.parse_args()
    
    if args.demo == 'basic':
        demo_basic_pipeline(provider=args.llm_provider, model=args.llm_model)
    elif args.demo == 'comparison':
        demo_strategy_comparison(provider=args.llm_provider, model=args.llm_model)
    else:
        demo_full_system(provider=args.llm_provider, model=args.llm_model)
