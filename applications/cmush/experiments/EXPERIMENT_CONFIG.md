# Experiment Configuration Log

## Current Test Run: November 23, 2025

### Models Used

**Both Systems Use Same Base Model:**
- **Model**: qwen/qwen3-4b-2507
- **Backend**: LMStudio (localhost:1234)
- **Hardware**: M3 Ultra
- **Temperature**: 0.7
- **Max Tokens**: 200 per response

**Key Insight:** We're testing architectural differences, not model differences. Same LLM, two different ways of using it.

### System Configurations

**Noodling (Simulated Cognitive Stack):**
- ResponseTypeDecider: 1 LLM call
- 7 Cognitive Transistors: 7 LLM calls
  - Cultural (beliefs)
  - Personality (traits)
  - Mood (emotional state)
  - Intuition (present awareness)
  - Memory (past recall)
  - Social (rules/expectations)
  - Deception (secret management)
- Manifold Blend: 1 LLM call
- Voice Translation: 1 LLM call
- Social Filter: 1 LLM call
- **Total: 11 LLM calls per turn**

**Baseline (Standard Approach):**
- Single LLM call with:
  - Character description in system prompt
  - Full conversation history in context
  - New user message
- **Total: 1 LLM call per turn**

### Test Parameters

**Experiment 1: Temporal Scaling**
- Turns tested: 100, 500, 1000
- Method: Mathematical simulation
- Result: Crossover at turn 206

**Experiment 2: Personality Consistency**
- Turns tested: 10 (initial), 100 (current)
- Method: Real LLM calls
- Metrics: Keyword frequency, memory references
- **Current run**: 100 turns = 200 LLM calls total

### Future Model Tests

**Planned Variations:**
1. **Larger model**: qwen/qwen3-14b or deepseek-v3
   - Hypothesis: Both systems improve, but does gap widen/narrow?

2. **Smaller model**: qwen/qwen3-1.8b
   - Hypothesis: Does Noodling structure help weaker models?

3. **Different family**: llama3.3 or mistral
   - Hypothesis: Is advantage model-specific?

4. **Frontier model**: GPT-4 or Claude (API)
   - Hypothesis: Does a smarter base model eliminate the need for architecture?

### Hardware Environment

**Primary Machine**: M3 Ultra
- RAM: 512GB
- Metal acceleration via MLX
- Handles multiple concurrent LLM instances

**LLM Serving**: LMStudio
- Supports OpenAI-compatible API
- Local inference (no API costs)
- Model: qwen3-4b-2507 (4-bit quantized)

### Expected Timeline

**100-turn test (current):**
- Start: 8:51 PM
- Estimated completion: 9:15 PM (~25 minutes)
- LLM calls: 200
- Estimated tokens: ~40,000

### Results Location

- JSON: `experiment_results/experiment2_consistency_TIMESTAMP.json`
- Log: `experiment_100turn.log`

---

**Note for Future Tests:**
When changing models, update this file with new model name and re-run both experiments to compare architectural advantage across different base models.
