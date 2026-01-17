adaptive_hint_agent icho$ python main.py

================================================================================
  Full System Demo
================================================================================

This demo shows the complete Adaptive Hint Agent system:
1. Simulated learner interaction logs
2. Behavioral feature engineering
3. Learner state inference
4. Feedback policy application
5. Adaptive hint generation
6. Strategy comparison and evaluation

================================================================================
  Demo: Basic Pipeline - Single Learner Session
================================================================================

Step 1: Generating simulated learner interaction logs...
Generated 10 interaction records

Sample logs:
   attempt_num  is_correct  time_gap_seconds  consecutive_failures
0            1       False         33.787302                     1
1            2       False         94.805690                     2
2            3       False         12.212991                     3
3            4       False          4.308391                     4
4            5       False         66.173915                     5

Step 2: Extracting behavioral features...
Extracted features:
   attempt_num  recent_success_rate  cumulative_success_rate  success_trend  activity_intensity  failure_rate
0            1                False                    False          False            1.000000             1
1            2                False                    False          False            0.775165             1
2            3                False                    False          False            1.077724             1
3            4                False                    False          False            1.400829             1
4            5                False                    False          False            1.263153             1

Step 3: Inferring learner states...
State distribution:
confused    10
Name: inferred_state, dtype: int64

State inference results:
   attempt_num  is_correct  consecutive_failures inferred_state
0            1       False                     1       confused
1            2       False                     2       confused
2            3       False                     3       confused
3            4       False                     4       confused
4            5       False                     5       confused
5            6       False                     6       confused
6            7       False                     7       confused
7            8       False                     8       confused
8            9       False                     9       confused
9           10       False                    10       confused

Step 4: Applying feedback policy...
Total hints triggered: 5 out of 10 attempts

Hint triggers:
   attempt_num inferred_state  consecutive_failures
1            2       confused                     2
3            4       confused                     4
5            6       confused                     6
7            8       confused                     8
9           10       confused                    10

Step 5: Generating adaptive hints...
Using Ollama provider with model: llama3:latest

Attempt 2 (State: confused):
  Detail: detailed, Tone: supportive and encouraging
  Hint: I'm here to help you tackle this quadratic equation! You're feeling a bit stuck, but that's totally okay. Let's break it down together.

Here's a detailed hint to get you started:

When solving a quadratic equation in the form of ax^2 + bx + c = 0, remember that there are two main approaches: factoring and using the quadratic formula.

For now, let's focus on factoring. Can you recall the steps for factoring a quadratic expression?

1. Look for two numbers whose product is the constant term (c) and whose sum is the coefficient of the linear term (b).
2. Write these two numbers as the factors of the middle term.
3. Combine the factors to get your final

Attempt 4 (State: confused):
  Detail: detailed, Tone: supportive and encouraging
  Hint: I'm here to help you tackle this quadratic equation! Don't worry, it's normal to feel confused at first - we'll work through it together.

Here's a detailed hint to get you started:

"Think of the quadratic equation as a balance scale. On one side, you have the 'x' term, which is like the weight on one pan. On the other side, you have the constant terms (the number and the squared x term), which are like the weights on the other pan.

Notice that the coefficients of the 'x' term (the numbers in front of the 'x') are like the scales' balance points. If they're equal, the scale is balanced! In this case, you can set up an equation

Attempt 6 (State: confused):
  Detail: detailed, Tone: supportive and encouraging
  Hint: I'm here to help you navigate this quadratic equation!

Let's take it one step at a time, okay? It seems like you're feeling a bit stuck right now, and that's totally normal! Quadratic equations can be tricky, but with some guidance, I know we can get through this together.

Here's a detailed hint to help you get started:

When solving a quadratic equation in the form of ax^2 + bx + c = 0, one common technique is to start by trying to factor the left-hand side. If you're not sure where to begin, try identifying any perfect squares or differences of perfect squares that might be hiding within the equation.

For example, if the equation looks like x^2 + 5x

Attempt 8 (State: confused):
  Detail: detailed, Tone: supportive and encouraging
  Hint: I'm here to help you untangle this quadratic equation! Don't worry if it seems overwhelming - we'll take it one step at a time.

Here's a detailed hint to get us started:

**Step 1:** Let's break down the quadratic equation into its most basic components. Can you identify the general form of a quadratic equation? It should look something like this: ax^2 + bx + c = 0, where a, b, and c are constants.

Think about it for a moment... What do these coefficients (a, b, and c) tell us about the equation?

**Next step:** Once you're comfortable with the general form, let's explore how each coefficient affects the shape of the parabola.

Attempt 10 (State: confused):
  Detail: detailed, Tone: supportive and encouraging
  Hint: I'm here to help you crack this quadratic equation!

Let's take a closer look at the equation together. It seems like you're feeling a bit stuck, and that's totally normal! Quadratics can be tricky, but don't worry, we'll get through this step by step.

Here's a hint that might help you get started:

Think about the fact that quadratic equations always have two solutions (or roots). When we're trying to find these solutions, it's helpful to consider what makes a perfect square. A perfect square is when a number multiplied by itself gives us our original value. For example, 4 is a perfect square because 2 x 2 = 4.

Now, let's apply this idea to your quadratic equation

================================================================================
  Demo: Strategy Comparison via Trace Replay
================================================================================

Generating interaction logs for multiple learners...
Generated 100 interaction records for 5 learners
Using Ollama provider with model: llama3:latest

Comparing 5 strategies...
Replaying with RuleBasedPolicy...
Replaying with RuleBasedPolicy...
Replaying with DataDrivenPolicy...
Replaying with DataDrivenPolicy...
Replaying with AdaptivePolicy(RuleBasedPolicy)...

Strategy Comparison Results:
Avg Hints/Learner Avg Time to First Hint (s) Hint Effectiveness Hint Frequency  Hints to Confused  Hints to Neutral  Hints to Progressing                           Policy Responsiveness Score  Total Hints
            6.40                      137.8             56.25%          0.320                 32                 0                     0                  RuleBasedPolicy                0.035           32
            7.00                       77.5             51.43%          0.350                 27                 8                     0                  RuleBasedPolicy                0.006           35
           11.00                      137.8             50.91%          0.550                 55                 0                     0                 DataDrivenPolicy                0.035           55
           13.40                       77.5             47.76%          0.670                 55                12                     0                 DataDrivenPolicy                0.007           67
            4.20                      137.8             61.90%          0.210                 21                 0                     0  AdaptivePolicy(RuleBasedPolicy)                0.041           21

--------------------------------------------------------------------------------
Key Insights:
--------------------------------------------------------------------------------
Most responsive strategy: AdaptivePolicy(RuleBasedPolicy) (score: 0.041)
Most efficient strategy: AdaptivePolicy(RuleBasedPolicy) (responsiveness/hint: 0.195)

================================================================================
  Demo Complete
================================================================================

The system is ready for use!

Next steps:
- Local LLM: Install Ollama and run 'ollama pull llama3:8b-instruct'
- OpenAI API: Set OPENAI_API_KEY environment variable
- Adjust policy parameters in feedback_policy.py
- Customize hint templates in hint_generator.py
- Add your own problem descriptions and real interaction logs