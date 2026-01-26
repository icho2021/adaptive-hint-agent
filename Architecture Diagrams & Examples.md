# Architecture Diagrams & Examples
## AI-Driven Adaptive Game Feedback Using LLMs and Offline RL

## 📊 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│           Learner Interaction Event Stream                        │
│  [attempt@10s] → [fail] → [edit@20s] → [attempt@50s] → [fail]   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                  Phase 1: Problem Intake & Data Collection
                       ↓
              Task Definition + Learner Interaction Data
                   │
        ┌──────────┴──────────┐
        │                     │
    Phase 2              Phase 2
  Learner State         Policy Selection
  Inference             & NPC Decision
        │                     │
   From events:        Choose NPC intervention:
   → confusion?        - NO_INTERVENTION?
   → progressing?      - SMALL_HINT?
   → disengaged?       - DETAILED_HINT?
   → persistence?      - FULL_SOLUTION?
        │                     │
        └──────────┬──────────┘
                   │
        Phase 3: LLM-Based NPC Feedback Generation
                   ↓
         LLM generates formatted hints/guidance
         (constrained by prompt templates)
                   │
        ┌──────────┴──────────┐
        │                     │
    Send to player         Record decision +
        │                 LLM response
        │                     │
        └──────────┬──────────┘
                   │
      Phase 4: Evaluation & Feedback Effectiveness
                   ↓
         Player continues interacting → New events
         ↓
    Record outcome (success/fail/abandon/engagement_change)
    ↓
  Offline Analysis: Trace Replay
    Compare different feedback policies:
    - Early intervention vs Late intervention
    - High-detail hints vs Minimal hints
    - Different pacing strategies
```

---

## 💡 Concrete Example: Player Solving "Quadratic Equation" Problem

### Raw Interaction Trace

```json
{
  "task_id": "quadratic_puzzle_1",
  "task_description": "Solve the equation: x² - 3x + 2 = 0",
  "player_id": "P101",
  "difficulty_level": "intermediate",
  "game_context": "Math Adventure Game - Chapter 3",
  "events": [
    {"time": "00:00", "type": "task_start", "status": "viewing_problem"},
    
    {"time": "00:15", "type": "attempt", "attempt_id": 1, "answer": "x=3", "result": "FAIL"},
    {"time": "00:45", "type": "edit_action", "edit_count": 1, "action": "changed_coefficient"},
    
    {"time": "01:10", "type": "attempt", "attempt_id": 2, "answer": "x=1,2", "result": "SUCCESS"},
    
    {"time": "01:30", "type": "task_end", "completion_status": "solved"}
  ]
}
```

---

### Learner State Inference Process

```
t=00:15 (first attempt fails):
  ├─ consecutive_failures = 1
  ├─ time_since_task_start = 15s (normal)
  ├─ inferred_state = "EXPLORING" 
  └─ npc_action = NO_INTERVENTION (let player explore)

t=00:45 (editing observed):
  ├─ player_is_editing = true (active recovery)
  ├─ signal = "player is thinking and adjusting strategy"
  └─ npc_action = NO_INTERVENTION (positive signal, continue)

t=01:10 (second attempt succeeds):
  ├─ consecutive_failures = 0 (reset)
  ├─ recovery_time = 55 seconds
  ├─ inferred_state = "MASTERED"
  └─ outcome = SUCCESS ✓
     (player solved via own exploration - high learning quality)
```

---

### Policy Comparison: Two Different Feedback Strategies

```
STRATEGY A: EARLY INTERVENTION (Supportive NPC)
────────────────────────────────────────────────
  Decision Rule:
    IF consecutive_failures ≥ 1 AND time_gap ≥ 30s → NPC sends SMALL_HINT
  
  Feedback at t=00:15 (after first failure):
    NPC: "Have you considered what the quadratic formula tells you?
          What does 'a', 'b', and 'c' represent in your equation?"
  
  Outcome:
    - Player receives hint at 00:15
    - Player succeeds at 00:35 (35 seconds total)
    - Note: Clear hint support, but unclear if player learned deep understanding
    - Engagement: Player follows NPC guidance

STRATEGY B: LATE INTERVENTION (Discovery-Focused NPC)
────────────────────────────────────────────────────
  Decision Rule:
    IF consecutive_failures ≥ 2 AND inactivity > 60s → NPC sends hint
  
  Feedback timing:
    At t=00:15: No intervention (first failure normal)
    At t=00:45: No intervention (player actively editing - good sign)
    At t=01:10: No intervention (player succeeded)
  
  Outcome:
    - Player succeeds via self-discovery at 01:10 (70 seconds total)
    - High learning quality (player discovered solution independently)
    - Engagement: Player feels accomplished

POLICY COMPARISON METRICS:
┌──────────────────────────────────────────────────────┐
│ Metric                    │ Strategy A │ Strategy B │
├──────────────────────────────────────────────────────┤
│ Time to Complete          │ 35s        │ 70s       │
│ NPC Interventions         │ 1          │ 0         │
│ Estimated Learning Depth  │ Medium     │ High      │
│ Player Autonomy           │ Medium     │ High      │
│ Engagement Signal         │ Good       │ Excellent │
│ Post-hint Success Rate    │ 100%       │ N/A       │
│ Time in Active Problem    │ 20s        │ 55s       │
│ Solving (vs waiting)      │            │           │
└──────────────────────────────────────────────────────┘

STRATEGIC INSIGHT:
  Strategy A: Faster completion, clear player support (better for retention)
  Strategy B: Deeper learning, player autonomy, intrinsic motivation
  
  RECOMMENDATION: Use adaptive blend based on player profile:
    - Struggling players → Strategy A (more NPC support)
    - Advanced players → Strategy B (encourage discovery)
    - First-time players → Blend (guided discovery)
```

---

## 🎯 Learner State Decision Tree - When Should the NPC Intervene?

```
                      ┌─ New interaction event
                      │
        ┌─────────────┴──────────────────────┐
        │                                    │
      Player                            Player
     FAILED?                           SUCCEEDED?
        │                                    │
        ↓                                    ↓
   consecutive_failures++          consecutive_failures = 0
        │                                    │
        ├─ Check threshold                   └─→ [PROGRESSING STATE]
        │   (≥ 2?)                                    │
        │                                    Problem solved?
        ├─ NO (only 1 failure)                       │
        │   └─→ [PROGRESSING]                        └─ YES
        │         Check response time                 └→ [MASTERED]
        │         (gap > 45s?)                          Update player profile
        │         NO
        │         └─→ NO_INTERVENTION
        │
        ├─ YES (≥2 consecutive failures)
        │   Check inactivity level
        │   (gap > 90s?)
        │   │
        │   ├─ NO (player still engaged)
        │   │  └─→ Send SMALL_HINT
        │   │      (guiding question, not solution)
        │   │      Start cooldown (300s - prevent hint spam)
        │   │
        │   └─ YES (≥2 failures + long inactivity)
        │      Check failure severity
        │      (consecutive_failures ≥ 3?)
        │      │
        │      ├─ NO
        │      │  └─→ Send DETAILED_HINT (escalate support)
        │      │      Step-by-step guidance
        │      │      Longer cooldown (600s)
        │      │
        │      └─ YES (≥3 failures, disengagement risk)
        │         └─→ Send FULL_SOLUTION + encouragement
        │            Flag: "emergency_intervention"
        │            Goal: Re-engage player
        │
        └─→ [LOG all decisions to trace]
            Record: state, action, npc_response, player_reaction
```

---

## 📈 Trace Replay: Comparing Feedback Strategies at Scale

### Setup: 100 Player Sessions, One Puzzle Problem

```
ORIGINAL DATA: 100 players × 1 puzzle task with interaction logs

POLICY A (EARLY INTERVENTION):
  ├─ confusion_threshold = 1 failure
  ├─ max_inactivity = 120 seconds
  └─ NPC_intervention_level = AGGRESSIVE
  
  Results after trace replay:
    ├─ total_hints_sent = 187
    ├─ avg_hints_per_player = 1.87
    ├─ puzzle_completion_rate = 78%
    ├─ avg_attempts_needed = 3.1
    ├─ player_abandonment_rate = 12%
    ├─ post_hint_success_rate = 52%
    ├─ avg_time_to_completion = 45s
    └─ player_reported_frustration = "Low"

POLICY B (LATE INTERVENTION):
  ├─ confusion_threshold = 2 failures
  ├─ max_inactivity = 90 seconds
  └─ NPC_intervention_level = CONSERVATIVE
  
  Results after trace replay:
    ├─ total_hints_sent = 103
    ├─ avg_hints_per_player = 1.03
    ├─ puzzle_completion_rate = 72%
    ├─ avg_attempts_needed = 4.2
    ├─ player_abandonment_rate = 18%
    ├─ post_hint_success_rate = 45%
    ├─ avg_time_to_completion = 70s
    └─ player_reported_frustration = "Medium"

POLICY C (ADAPTIVE BLEND):
  ├─ confusion_threshold = 1.5 failures (dynamic)
  ├─ difficulty_weighting = task_base_time × 1.2
  ├─ NPC_intervention_level = BALANCED
  └─ Escalates based on player profile
  
  Results after trace replay:
    ├─ total_hints_sent = 135
    ├─ avg_hints_per_player = 1.35
    ├─ puzzle_completion_rate = 76%
    ├─ avg_attempts_needed = 3.4
    ├─ player_abandonment_rate = 13%
    ├─ post_hint_success_rate = 54%
    ├─ avg_time_to_completion = 55s
    └─ player_reported_frustration = "Low-Medium"

STRATEGIC DECISION:
  ✓ Policy A: Highest completion (78%), lowest frustration
  ✗ Policy A: Hint overload (1.87/player), possible dependency
  
  ✓ Policy B: Encourages autonomy, good learning outcomes
  ✗ Policy B: Abandonment rate too high (18%)
  
  ✓ Policy C: BEST BALANCE - High completion (76%), reasonable autonomy
     Good for diverse player populations
  
  RECOMMENDATION: Deploy Policy C with monitoring
```

---

## 🎯 NPC Feedback Generation Using LLMs - Concrete Example

### Player Context

```
player_state = {
  consecutive_failures: 2,
  recent_attempts: ["x=3", "x=1,2 (incorrect formula application)"],
  time_since_last_action: 45s,
  game_level: "Chapter 3: Quadratic Equations",
  player_level: "intermediate",
  npc_personality: "mentor", 
  feedback_policy_decision: "SMALL_HINT",
  difficulty_estimation: 0.7
}
```

---

### Feedback Generation Process

#### Version 1.0 (Initial NPC Prompt Template)

```
SYSTEM ROLE:
You are Sage, a wise mentor NPC in a math adventure game.
Guide the player toward problem-solving without giving away the answer.
Be encouraging and age-appropriate.

PLAYER CONTEXT:
The player has tried twice to solve x² - 3x + 2 = 0.
First attempt: x=3 (incorrect)
Second attempt: x=1,2 but wrong formula application

HINT TEMPLATE (SMALL_HINT):
Ask clarifying questions to help player reflect:
- What formula are you using?
- What do each part mean (a, b, c)?

GENERATED RESPONSE:
Sage: "Hmm, your approach is close! Let me ask you something...
      What special formula helps us solve quadratic equations?
      What do 'a', 'b', and 'c' represent in YOUR equation?"
```

#### Version 1.1 (Improved - More Concise, Better Pacing)

```
SYSTEM ROLE:
You are Sage, a brief but supportive mentor.
One guiding question per hint. Be concise.

PLAYER CONTEXT:
Second failure, player is thinking actively (time: 45s).
Needs reinforcement without complete guidance.

HINT TEMPLATE (SMALL_HINT):
Ask ONE question about discriminant/formula application.

GENERATED RESPONSE:
Sage: "Great effort! Quick question: When you use the quadratic formula,
      what value do you calculate first (the discriminant, b²-4ac)?"
```

### JSON Output Schema & Validation

```json
{
  "npc_name": "Sage",
  "intervention_type": "small_hint",
  "hint_content": "Great effort! Quick question: When you use the quadratic formula, what value do you calculate first (the discriminant, b²-4ac)?",
  "token_count": 32,
  "tone": "encouraging",
  "contains_spoiler": false,
  "is_solution_given": false,
  "validation_status": "PASS",
  "prompt_template_version": "v1.1",
  "model_used": "gpt-4",
  "timestamp": "2026-01-25T14:30:00Z",
  "estimated_difficulty_relief": 0.6,
  "player_engagement_signal": "high"
}

VALIDATION CHECKLIST:
  ✓ Length: 32 tokens (< 100 for SMALL_HINT)
  ✓ No direct answer keywords ("x=1" "x=2")
  ✓ Question-based not directive
  ✓ Matches JSON schema
  ✓ Tone appropriate for game context
  ✓ Consistent with NPC personality
```

---

## 📝 Complete Interaction Decision Log

```
trace_id: TRACE_SESSION_P101_quadratic_puzzle
player_id: P101
puzzle_id: quadratic_puzzle_1
game_session: Chapter3_Session_2
session_date: 2026-01-25

═════════════════════════════════════════════════════════

EVENT 1: [00:00] TASK_START
  player_state = {
    consecutive_failures: 0,
    historical_success_rate: 0.65,
    engagement_level: "high",
    player_skill_estimate: "intermediate"
  }
  npc_decision = NO_INTERVENTION
  reasoning = "task just started, let player read and think"
  log = {
    action: "observe",
    timestamp: "00:00",
    npc_status: "watching"
  }

─────────────────────────────────────────────────────────

EVENT 2: [00:15] ATTEMPT_1_FAILED
  attempt = "x=3"
  player_state = {
    consecutive_failures: 1,
    time_since_start: 15s,
    last_action: "solve_attempt",
    edits_made: 0
  }
  policy_evaluation:
    - confusion_threshold = 2 (current_policy: BALANCED)
    - current_failures (1) < threshold (2) ✓
    - time_gap (15s) < concern_threshold (45s) ✓
  npc_decision = NO_INTERVENTION
  reasoning = "first failure within normal time, player still in exploration phase"
  log = {
    action: "observe",
    attempt_id: 1,
    result: "FAIL",
    timestamp: "00:15",
    failure_reason: "wrong_answer"
  }

─────────────────────────────────────────────────────────

EVENT 3: [00:45] PLAYER_EDITING
  edit_action = {
    type: "coefficient_adjustment",
    edit_count: 1,
    time_since_last_attempt: 30s
  }
  player_state = {
    consecutive_failures: 1,
    is_actively_problem_solving: true,
    time_since_last_attempt: 30s
  }
  signal = "POSITIVE - Player is actively adjusting and re-attempting"
  npc_decision = NO_INTERVENTION
  reasoning = "Player showing active recovery behavior, don't interrupt"
  log = {
    action: "observe",
    edit_detail: "adjusted_equation_coefficients",
    timestamp: "00:45",
    player_engagement_signal: "high"
  }

─────────────────────────────────────────────────────────

EVENT 4: [01:10] ATTEMPT_2_SUCCESS
  attempt = "x=1,2"
  verification = "CORRECT - Quadratic factorization verified"
  player_state = {
    consecutive_failures: 0,
    recovery_path: "SELF_DIRECTED",
    time_to_recovery: 55s,
    autonomy_signal: "high"
  }
  npc_decision = NO_INTERVENTION + ENCOURAGEMENT
  reasoning = "Player solved independently, celebrate this!"
  npc_feedback = "Sage: Fantastic work! You solved it perfectly! 
                   That's the power of persistence. 
                   Ready for the next challenge?"
  log = {
    action: "celebrate_success",
    attempt_id: 2,
    result: "SUCCESS",
    timestamp: "01:10",
    total_session_time: 70s,
    self_recovery_rate: 100% (solved without hint),
    learning_quality_estimate: "HIGH"
  }

═════════════════════════════════════════════════════════

SESSION SUMMARY:
  Total NPC interventions: 0
  Task completion: SUCCESS
  Player recovery: AUTONOMOUS
  Estimated learning depth: HIGH (self-driven discovery)
  Player engagement: HIGH
  Session outcome: EXCELLENT
  
FOLLOW-UP RECOMMENDATIONS:
  → Increase difficulty of next puzzle slightly
  → Continue monitoring for this player's optimal difficulty level
  → Player shows good problem-solving persistence
```

---

## 🔍 Quantum-Inspired State Representation (Advanced Feature)

```
TRADITIONAL BINARY STATE:
  State = "confused" OR "progressing" OR "disengaged"
  Problem: Real players exist in overlapping states
  
QUANTUM-INSPIRED STATE (Superposition Model):
  Instead of one state, represent player as probability distribution:
  
  state_vector = [
    P(confused),      # probability of confusion signals
    P(progressing),   # probability of forward progress
    P(persistent),    # probability of persisting despite difficulty
    P(disengaged)     # probability of giving up
  ]
  
  Example after 2 failures, 45s pause, but still actively editing:
    state_vector = [0.6, 0.3, 0.1, 0.0]
    Interpretation: Likely confused (60%), some progress (30%), 
                   slight persistence signal (10%), not disengaged (0%)
  
  ADVANTAGE:
    - Captures uncertainty in real-time
    - Enables nuanced intervention decisions
    - Can detect state transitions (confusion → persistence)
    - Better reflects human learning reality

**Prepared by:** Zimin (Icho) Cai  
**For:** RA Application - AI-Driven Adaptive Game Feedback  
**With:** Dr. Hazra Imran, Northeastern University  
**Date:** January 25, 2026
