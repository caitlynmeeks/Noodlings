# LLM Interaction Points in Noodlings Architecture

**All moments where Noodlings send requests to LLM**

Last Updated: November 23, 2025

---

## 1. AFFECT EXTRACTION
**File:** `llm_interface.py::text_to_affect()`
**Location:** `agent_bridge.py::perceive_event()` line ~1765
**Frequency:** Every incoming event (message, action, observation)
**Purpose:** Convert text → 5-D affect vector [valence, arousal, fear, sorrow, boredom]
**Model:** Fast model (qwen3-4b or agent's llm_model)
**Input:** Raw text from user/agent
**Output:** Numerical affect vector
**Prompt Type:** Structured extraction prompt

---

## 2. INTUITION GENERATION
**File:** `agent_bridge.py::_generate_intuition()`
**Location:** `agent_bridge.py::perceive_event()` line ~1798
**Frequency:** Every incoming event (if intuition_receiver enabled)
**Purpose:** Generate contextual awareness (spatial, social, present-moment)
**Model:** Fast model (qwen3-4b or agent's llm_model, configured in config.yaml)
**Input:**
- Speaker identity
- Message content
- Room occupants (with species/age/pronouns)
- Objects nearby
- Agent inventories
- Recent conversation context
**Output:** 2-3 sentence intuition text (first-person awareness)
**Prompt Type:** "You are [agent]'s intuitive awareness - like a narrator"
**Example Output:** "That greeting is for Toad, not me. They're by the pond."

---

## 3. COGNITIVE MANIFOLD BLENDING (Optional)
**File:** `cognitive_components.py::_llm_weighted_blend()`
**Location:** `agent_bridge.py::perceive_event()` line ~1831
**Frequency:** Every event if cognitive_manifold exists AND blending_strategy="llm_weighted"
**Purpose:** Synthesize multiple transistor perspectives into one coherent perception
**Model:** Fast model (qwen3-4b)
**Input:**
- Multiple transistor outputs with salience weights
- Example: Cultural (0.8), Somatic (0.95), Personality (0.85), Intuition (0.80), Mood (0.60)
**Output:** Single integrated perception text
**Prompt Type:** "Synthesize these cognitive perspectives into ONE coherent thought"
**Token Budget:** ~100 tokens

---

## 4. SPEECH GENERATION (Main Response)
**File:** `llm_interface.py::generate_response()`
**Location:** `agent_bridge.py::_generate_response()` line ~2519
**Frequency:** When agent decides to speak (cooldown passed, addressed, or high surprise)
**Purpose:** Generate agent's spoken response to event
**Model:** Agent's llm_model (can be overridden per-agent in recipe)
**Input:**
- Phenomenal state (40-D vector)
- Conversation context (last N messages, stratified memory retrieval)
- Agent identity prompt
- Colored perception (from cognitive manifold if present)
- Intuition text
- Character voice patterns
**Output:** Speech text + optional thinking (internal monologue)
**Prompt Type:** Complex character prompt with identity, personality, constraints
**Token Budget:** max_tokens from recipe (default 180)
**Post-processing:**
- Mysticism penalty calculation
- Cheap thrills bonus calculation
- Character voice translation
- Affective reinforcement modulation

---

## 5. RUMINATION GENERATION (Private Thoughts)
**File:** `llm_interface.py::generate_rumination()`
**Location:** `agent_bridge.py::_generate_rumination()` line ~2888
**Frequency:** When agent observes but doesn't speak (addressed=False)
**Purpose:** Generate internal thoughts/observations
**Model:** Agent's llm_model
**Input:**
- Phenomenal state (40-D vector)
- Recent conversation context (configurable window, default 2)
- Agent identity prompt
- Colored thought seed (from cognitive manifold if present)
**Output:** Thought text (displayed in strikethrough)
**Prompt Type:** Rumination-specific prompt (more introspective)
**Token Budget:** Typically shorter than speech
**Post-processing:**
- Affective reinforcement modulation (NEW: Phase 7)

---

## 6. SELF-MONITORING EVALUATION (Metacognition)
**File:** `agent_bridge.py::_evaluate_own_output()`
**Location:** After speech or rumination, if surprise > threshold
**Frequency:** Occasional (30s cooldown, surprise > 0.1)
**Purpose:** Agent evaluates its own speech/thought for social risk, coherence, regret
**Model:** Fast model (qwen3-4b)
**Input:**
- Agent's own speech/thought text
- Recent conversation context
- Current emotional state (valence, arousal, fear, surprise)
- Agent identity and description
**Output:** JSON evaluation with:
- social_risk (none/mild/moderate/high)
- coherence (clear/unclear)
- aesthetic_surprise (none/rhyme/eloquent/poetic)
- regret_level (none/mild/moderate/high)
- emotional_impact (valence/arousal/fear deltas)
- follow_up action (none/clarify/apologize/celebrate)
**Prompt Type:** Structured metacognitive evaluation
**Token Budget:** ~150 tokens

---

## 7. SOCIAL EXPECTATION DETECTION
**File:** `agent_bridge.py::_detect_social_expectation()`
**Location:** `agent_bridge.py::perceive_event()` line ~2352
**Frequency:** Every event when deciding whether to respond
**Purpose:** Detect if social norm expects a response (greeting, question, gift, gesture)
**Model:** Fast model (qwen3-4b)
**Input:**
- Event context
- Intuition text
- Recent conversation flow
**Output:** JSON with:
- expected (true/false)
- urgency (0.0-1.0)
- type (question/greeting/gift/gesture/silence)
- reason (explanation)
**Prompt Type:** "Analyze this interaction for social response expectations"
**Token Budget:** ~100 tokens

---

## 8. CHARACTER VOICE TRANSLATION (Optional)
**File:** `agent_bridge.py::translate_to_character_voice()`
**Location:** After speech generation, before returning response
**Frequency:** If character_voice config exists in recipe
**Purpose:** Translate plain speech → character-specific voice/dialect
**Model:** Fast model (qwen3-4b)
**Input:**
- Plain speech text
- Voice pattern specification (e.g., "ALL CAPS + percentages" for SERVNAK)
- Example transformations
**Output:** Translated speech in character voice
**Prompt Type:** "Translate this speech to match character voice pattern"
**Token Budget:** ~100 tokens
**Examples:**
- SERVNAK: "Hello" → "HELLO SISTER! PRIDE CIRCUITS AT 94.3% ENTHUSIASM!"
- Phi (kitten): "I'm hungry" → "*meows, as if to say 'I require sustenance'*"
- Backwards Dweller: "Hello" → "olleH"

---

## SUMMARY TABLE

| # | Interaction Point | File | Model Type | Frequency | Token Budget |
|---|-------------------|------|------------|-----------|--------------|
| 1 | Affect Extraction | llm_interface.py | Fast | Every event | ~50 |
| 2 | Intuition Generation | agent_bridge.py | Fast/Agent | Every event | ~100 |
| 3 | Manifold Blending | cognitive_components.py | Fast | Every event* | ~100 |
| 4 | Speech Generation | llm_interface.py | Agent | When speaking | 180 (configurable) |
| 5 | Rumination Generation | llm_interface.py | Agent | When observing | ~100 |
| 6 | Self-Monitoring | agent_bridge.py | Fast | Occasional | ~150 |
| 7 | Social Expectation | agent_bridge.py | Fast | Every event | ~100 |
| 8 | Voice Translation | agent_bridge.py | Fast | When speaking* | ~100 |

*Only if configured/enabled

---

## TOTAL TOKEN USAGE PER EVENT (Typical)

**Minimal scenario** (agent doesn't speak):
- Affect extraction: ~50 tokens
- Intuition: ~100 tokens
- Social expectation: ~100 tokens
- Rumination: ~100 tokens
- **Total: ~350 tokens**

**Speech scenario** (agent responds):
- Affect extraction: ~50 tokens
- Intuition: ~100 tokens
- Manifold blending: ~100 tokens (if enabled)
- Social expectation: ~100 tokens
- Speech generation: ~180 tokens
- Voice translation: ~100 tokens (if enabled)
- Self-monitoring: ~150 tokens (occasional)
- **Total: ~680-880 tokens**

---

## MODEL SELECTION HIERARCHY

1. **Agent's llm_model** (from recipe `llm_model` field) - Used for speech/rumination
2. **Global default** (from config.yaml) - Used for fast operations
3. **Hardcoded default** (qwen3-4b) - Fallback

**Per-agent override example** (mysterious_stranger.yaml):
```yaml
llm_model: "mistralai/mistral-small-3.2-24b-instruct-2506-mlx"
```

---

## OPTIMIZATION OPPORTUNITIES

1. **Batch intuition + social expectation** - Single LLM call
2. **Cache manifold blending** - Reuse for similar transistor configs
3. **Skip social expectation** - If intuition already indicates clear addressing
4. **Reduce affect extraction calls** - Only on new speakers or significant shifts
5. **Model selection** - Use smaller models for fast operations, larger for speech

---

## DEBUGGING LLM CALLS

**Enable detailed logging:**
```python
logger.setLevel(logging.DEBUG)
```

**Log messages to look for:**
- `🤖 LLM REQUEST → [model]` - Request sent
- `✅ LLM RESPONSE ← [model]` - Response received
- `🎨 AFFECT EXTRACTED` - Affect vector computed
- `📻 Intuition` - Contextual awareness generated
- `🧠 COGNITIVE MANIFOLD` - Transistor blend complete
- `💭 thinking` - Rumination generated
- `🧠 [SELF-MONITOR]` - Metacognitive evaluation triggered

---

**End of LLM Interaction Points Documentation**
