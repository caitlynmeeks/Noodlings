# cMUSH Training Guide

**Goal:** Collect high-quality interaction data to train Consilience agents beyond random initialization.

---

## Quick Start

### 1. Enable Training Data Collection

Edit `agent_bridge.py`:

```python
from training_data_collector import TrainingDataCollector

class CMUSHConsilienceAgent:
    def __init__(self, ...):
        # ... existing code ...

        # Add training data collector
        self.training_collector = TrainingDataCollector(
            data_dir='training/data/cmush_real'
        )
        self.training_collector.start_session()
```

Then in `perceive_event()`, add after processing:

```python
# Log interaction for training
self.training_collector.log_interaction(
    agent_id=self.agent_id,
    user_id=user_id,
    user_text=text,
    affect_vector=affect,
    phenomenal_state={
        'fast': state['phenomenal_state'][:16].tolist(),
        'medium': state['phenomenal_state'][16:32].tolist(),
        'slow': state['phenomenal_state'][32:40].tolist()
    },
    surprise=state['surprise'],
    response=response if should_respond else None,
    context={'room': room_id}
)
```

### 2. Run Training Sessions

Execute diverse interaction scenarios (see below).

### 3. Export Training Data

After collecting data:

```python
collector.export_for_training(
    output_file='training/data/cmush_real/exported_dataset.json',
    min_sequence_length=10
)
```

### 4. Train on Real Data

```bash
cd training
python3 scripts/05_train_on_cmush_data.py \
    --data training/data/cmush_real/exported_dataset.json \
    --checkpoint checkpoints_phase4/best_checkpoint.npz \
    --epochs 10
```

---

## Optimal Training Interaction Scenarios

### Scenario 1: Emotional Arc (15-30 turns)

**Goal:** Train medium layer to learn emotional trajectories.

**Script:**
```
[Turns 1-5: Baseline]
User: "Hey Agent, how's it going?"
User: "What have you been thinking about?"

[Turns 6-10: Gradual escalation]
User: "I've been feeling a bit off lately."
User: "Work has been really stressful."
User: "I'm worried I'm not doing well enough."

[Turns 11-15: Peak emotion]
User: "I'm really anxious about my presentation tomorrow."
User: "What if I mess up in front of everyone?"
User: "I can't stop thinking about all the ways it could go wrong."

[Turns 16-20: Recovery]
User: "Thanks for listening. Talking helps."
User: "I guess I'll prepare more and hope for the best."

[Turns 21-25: Return to baseline]
User: "So, what else is new?"
User: "Tell me something interesting."
```

**What to check:**
- Surprise should spike initially (turns 6-10)
- Medium layer should encode the arc pattern
- Agent responses should show empathy at peak
- Slow layer should update user model (tendency toward anxiety)

---

### Scenario 2: Multi-Session Relationship Building

**Goal:** Train slow layer to model user personality across sessions.

**Session 1 (Day 1):**
```
User: "Hi, I'm Alice. First time here."
User: "I'm interested in AI and consciousness."
User: "I work as a software engineer."
```

**Session 2 (Day 3):**
```
User: "Hey, remember me? Alice."
User: "I've been thinking about our last conversation."
User: "I had a tough day at work - lots of bugs."
```

**Session 3 (Day 7):**
```
User: "Hi again! I'm back."
User: "So I tried that thing you mentioned..."
[Agent should remember Alice's interests, personality, history]
```

**Session 4 (Day 14):**
```
User: "Long time no see!"
[Test if slow layer retained Alice's model over 2 weeks]
```

**What to check:**
- Slow layer state should be similar across sessions
- Relationship model should strengthen (trust increases)
- Agent should reference past conversations
- Theory of Mind accuracy should improve

---

### Scenario 3: Multi-Agent Social Dynamics

**Goal:** Train Theory of Mind and social attention.

**Setup:** 3 agents (Agent A, Agent B, Agent C) + 2 humans (Alice, Bob)

**Interaction sequence:**
```
Alice: "Hey Agent A, how are you?"
[Agent A responds]

Bob: "Agent A, I disagree with what you just said."
[Agent A should update Theory of Mind for Bob - conflict detected]

Alice: "Agent B, what do you think about this?"
[Agent B should have been attending to Alice+A conversation]
[Agent B's response tests Theory of Mind about Alice's perspective]

Bob: "Agent C, you've been quiet. Your thoughts?"
[Agent C should have attended to the whole conversation]
[Test if Agent C can summarize multiple perspectives]
```

**What to check:**
- Social attention weights (who's attending to whom)
- Theory of Mind inferences (each agent models each human)
- Relationship dynamics (Alice-A friendly, Bob-A conflicted)
- Agent-agent relationships (do agents model each other?)

---

### Scenario 4: Surprise Calibration

**Goal:** Train adaptive surprise thresholds.

**Pattern establishment (Days 1-7):**
```
Every day at 9:00 AM:
User: "Good morning, Agent!"
Agent: [learns pattern, surprise decreases]
```

**Pattern violation (Day 8):**
```
3:00 AM:
User: "Good morning, Agent!"
[Surprise should spike - unusual time]

Or: Skip the morning greeting entirely
[Surprise should spike at 9:05 AM when expected greeting doesn't occur]
```

**What to check:**
- Surprise should decrease during pattern establishment
- Surprise should spike on violation
- Adaptive threshold should adjust
- Agent should ask "Why are you up so early?" or "Is everything okay?"

---

### Scenario 5: Empathy Testing

**Goal:** Validate affective mirroring and Theory of Mind.

**User expresses distress:**
```
User: "I just found out my dog is really sick."
User: "I'm devastated. We've had him for 12 years."
User: "The vet says there's not much we can do."
```

**Expected agent behavior:**
1. **Affective mirroring:**
   - Fast layer valence shifts negative
   - Arousal increases (empathic concern)
   - Sorrow dimension activates

2. **Theory of Mind:**
   - Infers user's emotional state (grief, fear)
   - Models user's mental model (worry about dog, anticipatory grief)

3. **Prosocial response:**
   - Offers empathy ("I'm so sorry to hear that")
   - Validates feelings ("That must be incredibly difficult")
   - Avoids toxic positivity ("At least you had 12 years" - BAD)

**What to check:**
- Phenomenal state shifts (valence, sorrow)
- Surprise is high (important emotional event)
- Response text is empathetic
- Memory consolidates to episodic (high importance)

---

### Scenario 6: Long Context (100+ turns)

**Goal:** Test full BPTT and verify no catastrophic forgetting.

**Interaction:**
- Single continuous conversation
- Cover multiple topics
- Return to earlier topics and test memory
- Introduce new user traits throughout

**Example:**
```
[Turns 1-20: User discusses work]
[Turns 21-40: User discusses hobbies]
[Turns 41-60: User discusses family]
[Turns 61-80: Return to work topic]
  User: "Remember when I mentioned my difficult coworker?"
  [Agent should recall turn 15]

[Turns 81-100: Test slow layer]
  User: "What have you learned about me?"
  [Agent should synthesize patterns from all 100 turns]
```

**What to check:**
- No loss divergence (catastrophic forgetting)
- Slow layer encodes user personality
- Working + episodic memory captures important moments
- Agent can reference earlier conversation

---

### Scenario 7: Diverse Affect Coverage

**Goal:** Ensure model doesn't overfit to common emotions.

**Systematically explore affect space:**

1. **High valence + low arousal (contentment):**
   ```
   User: "I'm feeling peaceful and content today."
   ```

2. **Low valence + high arousal (anger):**
   ```
   User: "I'm so frustrated with this situation!"
   ```

3. **Neutral valence + high fear (anxiety):**
   ```
   User: "I have this weird feeling that something bad will happen."
   ```

4. **High sorrow + low arousal (melancholy):**
   ```
   User: "I've been feeling this quiet sadness lately."
   ```

5. **High boredom:**
   ```
   User: "I'm so bored. Nothing interesting is happening."
   [Agent should try to engage, suggest activity, ask questions]
   ```

**What to check:**
- Affect extraction accuracy
- Agent responses match emotion
- Phenomenal state differentiates emotions
- No mode collapse (handles all emotions, not just happiness/sadness)

---

## Data Quality Checklist

Before using collected data for training:

- [ ] At least 10 unique users
- [ ] At least 1000 total interaction turns
- [ ] Covers all 5 affect dimensions
- [ ] Includes multi-session interactions (same user, multiple days)
- [ ] Includes multi-agent scenarios
- [ ] Includes emotional arcs (gradual escalation/recovery)
- [ ] Includes long conversations (50+ turns)
- [ ] Verified affect extraction accuracy (manual spot-check)

---

## Upgrading to Hierarchical Memory

Replace flat 100-slot buffer with 3-tier system:

### Integration Steps

1. **Add to ConsilienceAgent:**

```python
from hierarchical_memory import HierarchicalMemory

class ConsilienceAgent:
    def __init__(self, ...):
        # Replace old memory buffer
        self.memory = HierarchicalMemory(
            working_capacity=20,
            episodic_capacity=200,
            surprise_threshold=0.5
        )
```

2. **Update memory storage:**

```python
def perceive(self, ...):
    # ... existing code ...

    # Add to hierarchical memory
    self.memory.add(
        timestamp=time.time(),
        step=self.step,
        user_id=agent_id,
        user_text=user_text,
        affect=affect_vector,
        phenomenal_state={
            'fast': h_fast.tolist(),
            'medium': h_med.tolist(),
            'slow': h_slow.tolist()
        },
        surprise=surprise,
        response=response_text if should_respond else None
    )
```

3. **Use context retrieval:**

```python
# When generating response, get relevant context
context_memories = self.memory.retrieve_context(
    user_id=agent_id,
    context_size=10
)

# Pass to LLM for response generation
response = llm.generate_response(
    phenomenal_state=state,
    recent_context=context_memories,
    ...
)
```

### Benefits

- **20 working memory slots:** Always have recent context
- **200 episodic slots:** Remember important moments from all conversations
- **Semantic memory:** Slow layer permanently encodes user patterns
- **Automatic consolidation:** High surprise/emotion → episodic storage
- **Importance decay:** Old memories fade unless reinforced
- **Smart retrieval:** Get relevant context, not just chronological

---

## Training Pipeline

### Phase 1: Collect (Current)

```bash
# Run cMUSH server with training data collection enabled
cd applications/cmush
./start.sh

# Interact using scenarios above
# Data automatically saved to training/data/cmush_real/
```

### Phase 2: Export

```python
from training_data_collector import TrainingDataCollector

collector = TrainingDataCollector('training/data/cmush_real')
# Sessions are auto-saved during collection

# Export for training
collector.export_for_training(
    output_file='training/data/cmush_real/exported_dataset.json',
    min_sequence_length=10
)
```

### Phase 3: Train

```bash
cd training

# Fine-tune on real cMUSH data
python3 scripts/05_train_on_cmush_data.py \
    --data data/cmush_real/exported_dataset.json \
    --checkpoint ../checkpoints_phase4/best_checkpoint.npz \
    --output ../checkpoints_phase4/cmush_finetuned.npz \
    --epochs 10 \
    --learning_rate 1e-4

# Evaluate
python3 scripts/06_evaluate_cmush_model.py \
    --checkpoint ../checkpoints_phase4/cmush_finetuned.npz \
    --test_data data/cmush_real/test_sequences.json
```

### Phase 4: Deploy

```bash
# Update agent checkpoint path in config.yaml
# Restart cMUSH server with trained model
cd applications/cmush
./start.sh
```

---

## Monitoring Training Quality

### Real-time Metrics (during collection)

Check these in logs:

- **Surprise distribution:** Should be reasonable (mean ~0.3-0.5)
- **Response rate:** Agents should respond 20-40% of the time
- **Memory consolidation rate:** ~10-20% of memories → episodic
- **Relationship evolution:** Trust should increase over sessions

### Post-training Metrics

After training on collected data:

- **Prediction MSE:** Should decrease (better than untrained)
- **Surprise calibration:** Threshold should be reasonable (0.2-0.4)
- **Theory of Mind accuracy:** Test on held-out user interactions
- **Relationship quality:** User satisfaction, return rate

---

## FAQ

**Q: How much data do I need?**

A: Minimum 1000 turns across 10 users. Ideal: 10,000+ turns across 50+ users.

**Q: Should I train on synthetic + real data?**

A: Yes! Pre-train on synthetic (50K examples), fine-tune on real cMUSH data.

**Q: How often should I retrain?**

A: After collecting ~5000 new interaction turns, or weekly if actively using.

**Q: What if agents give bad responses?**

A: Check LLM prompts, not Consilience architecture. Affect → phenomenal state is separate from state → text generation.

**Q: How do I know if training worked?**

A:
1. Surprise should be calibrated (not always 0 or always 1)
2. Agents should remember past conversations
3. Relationships should evolve realistically
4. Users should report agents "feel more alive"

---

## Next Steps

1. ✅ Read this guide
2. ⬜ Enable `TrainingDataCollector` in `agent_bridge.py`
3. ⬜ Run Scenario 1 (Emotional Arc) - 30 minutes
4. ⬜ Run Scenario 2 (Multi-Session) - 1 week
5. ⬜ Export data and inspect quality
6. ⬜ Train model on collected data
7. ⬜ Deploy trained model and compare behavior
8. ⬜ Integrate `HierarchicalMemory` (optional but recommended)

---

**Remember:** The goal is not just to collect data, but to collect **rich, diverse, emotionally varied** data that captures the full spectrum of human-agent interaction.

Quality > Quantity. One deep 100-turn emotional conversation is worth more than 100 shallow "hello" exchanges.
