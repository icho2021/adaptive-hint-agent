# Adaptive Hint Agent for Problem-Solving Tasks

An intelligent system that processes learner interaction logs, infers learner states, and provides adaptive hints using rule-based and LLM-powered strategies.

## Overview

An intelligent adaptive hint generation system for problem-solving tasks. Processes learner interaction logs, infers learner states using behavioral features, and provides context-aware hints using LLM-powered generation.

**Key Features**:
- 6 core modules with modular architecture
- 3 feedback policies (rule-based, data-driven, adaptive)
- LLM provider abstraction (Ollama/OpenAI/Template)
- Trace-replay evaluation framework
- Zero API costs with local LLM inference

## Project Structure

```
adaptive_hint_agent/
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_generator.py      # Simulated learner interaction logs
│   ├── feature_engineering.py # Behavioral feature extraction
│   ├── state_inference.py     # Learner state inference (confused/progressing)
│   ├── feedback_policy.py     # Rule-based and data-driven hint triggering
│   ├── hint_generator.py      # LLM-powered adaptive hint generation
│   └── trace_replay.py        # Strategy comparison and evaluation
├── data/
│   └── logs/                  # Generated interaction logs
├── requirements.txt
├── main.py                    # Main demo script
├── test_llm_providers.py      # LLM provider testing utility
├── PROJECT_REPORT_COMPLETE.md # Comprehensive project report
├── .gitignore
└── README.md
```

## Quick Start

1. **Install dependencies:**
```bash
cd adaptive_hint_agent
pip install -r requirements.txt
```

2. **Set up LLM (optional, for LLM-powered hints):** 

   **Option A: Local LLM with Ollama (Recommended for development)** 
   ```bash
   # Install Ollama (if not already installed)
   # Visit https://ollama.ai or run: curl -fsSL https://ollama.ai/install.sh | sh
   
   # Download Llama3 model
   ollama pull llama3:8b-instruct
   
   # Verify installation
   ollama list
   ```

   **Option B: OpenAI API (Cloud-based)**
   ```bash
   export OPENAI_API_KEY=your_key_here
   # Or create a .env file with: OPENAI_API_KEY=your_key_here
   ```

3. **Run the demo:**
```bash
python main.py
```

4. **Run specific demos:**
```bash
python main.py --demo basic        # Basic pipeline demo
python main.py --demo comparison   # Strategy comparison only
python main.py --demo full         # Full system demo (default)
```

## Features

- **Data Processing**: Simulated learner interaction logs with behavioral patterns
- **Feature Engineering**: Extracts consecutive failures, recent activity, success trends
- **State Inference**: Classifies learners as "confused", "progressing", or "neutral"
- **Feedback Policies**: Rule-based, data-driven, and adaptive hint triggering
- **LLM Integration**: Local (Ollama) or cloud (OpenAI) hint generation with automatic fallback
- **Strategy Comparison**: Trace-replay framework for systematic policy evaluation

## Usage Example

```python
from src.data_generator import LearnerLogGenerator
from src.feature_engineering import FeatureEngineer
from src.state_inference import StateInference
from src.feedback_policy import RuleBasedPolicy, PolicyManager
from src.hint_generator import HintGenerator

# Generate data
generator = LearnerLogGenerator(seed=42)
logs = generator.generate_multiple_sessions(n_learners=5, n_problems=2, attempts_per_session=10)

# Extract features and infer states
engineer = FeatureEngineer()
features_df = engineer.extract_features(logs)
state_inference = StateInference(engineer)
result_df = state_inference.infer_states_batch(features_df)

# Apply policy and generate hints
policy = RuleBasedPolicy(trigger_on_consecutive_failures=3)
manager = PolicyManager(policy)
hint_generator = HintGenerator(use_llm=True, provider="auto")

for idx, row in result_df.iterrows():
    features = engineer.get_feature_vector(row)
    result = manager.evaluate_attempt(features, row['inferred_state'], row['attempt_num'], 
                                      row['learner_id'], row['problem_id'])
    if result['hint_triggered']:
        hint = hint_generator.generate_hint(
            problem_id=row['problem_id'],
            problem_description="Solve a quadratic equation",
            state=row['inferred_state'],
            features=features,
            attempt_num=row['attempt_num']
        )
```

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for detailed usage examples and API documentation.

## Requirements

- Python 3.7+
- Dependencies: `numpy`, `pandas`, `scikit-learn`, `requests` (see `requirements.txt`)
- Optional: `openai` (for cloud API), Ollama (for local LLM)

## Results

**Strategy Comparison** (100 interaction records, 5 learners, 2 problems):

| Strategy | Hints | Effectiveness | Responsiveness | Efficiency |
|----------|-------|---------------|----------------|------------|
| **AdaptivePolicy** | **21** | **61.90%** | **0.041** | **0.195** ⭐ |
| RuleBased (t=3) | 32 | 56.25% | 0.035 | 0.109 |
| RuleBased (t=2) | 35 | 51.43% | 0.006 | 0.002 |
| DataDriven (t=0.6) | 55 | 50.91% | 0.035 | 0.064 |
| DataDriven (t=0.5) | 67 | 47.76% | 0.007 | 0.001 |

**Key Finding**: AdaptivePolicy achieves highest effectiveness (61.90%) with fewest hints (21).

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for detailed experimental results, analysis, and methodology.

## Documentation

For detailed documentation, technical explanations, cost analysis, and research findings, see:
- **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Comprehensive project report

## License

This project is for demonstration and portfolio purposes.
