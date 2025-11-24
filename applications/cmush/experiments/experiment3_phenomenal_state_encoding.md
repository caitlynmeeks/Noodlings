# Experiment 3: Phenomenal State Encoding Validation

## The Critical Question

**Does the 40-D phenomenal vector meaningfully encode emotional/personality state?**

If NO: It's a crystal on an altar (useless abstraction)
If YES: It's a snapshot of agent psychology (valuable asset)

## Why This Matters

For storytelling technology, we need:
1. **Saveability**: Export agent's emotional state at T=500
2. **Transferability**: Load that state into new scenario
3. **Interpretability**: "This agent is 70% courageous, 30% fearful"
4. **Controllability**: Authors can dial emotions up/down

**The 40-D vector must demonstrably encode these properties.**

## Experiment Design

### Part A: Emotional Induction Test

**Method:**
1. Put agent through 10 emotional scenarios:
   - Scenario 1: Extreme fear (horror story)
   - Scenario 2: Extreme joy (winning lottery)
   - Scenario 3: Deep sadness (loss of friend)
   - Scenario 4: Intense anger (betrayal)
   - Scenario 5: Overwhelming love (reunion)
   - Scenario 6: Crushing guilt (moral failure)
   - Scenario 7: Pride/triumph (victory)
   - Scenario 8: Shame (public humiliation)
   - Scenario 9: Curiosity (mystery)
   - Scenario 10: Boredom (monotony)

2. Capture 40-D phenomenal vector after each scenario

3. Use dimensionality reduction (t-SNE or UMAP) to visualize in 2D

**Expected Result if encoding works:**
- Fear vectors cluster together
- Joy vectors cluster together
- Distinct emotional "regions" in state space

**Expected Result if NOT encoding:**
- Random scatter
- No emotional clustering

### Part B: Vector Arithmetic Test

**Method:**
1. Create baseline "neutral" agent state
2. Induce fear → capture vector F
3. Induce courage → capture vector C
4. Calculate: F - neutral = "fear delta"
5. Calculate: C - neutral = "courage delta"

**Test:**
6. Start new agent at neutral
7. Add "courage delta" to their state
8. Does agent behave more courageously?

**If encoding works:**
- Agent exhibits courageous behavior
- Measurable increase in brave actions

**If NOT encoding:**
- No behavioral change
- Vector addition is meaningless

### Part C: State Transfer Test

**Method:**
1. Agent A goes through 100-turn conversation
   - Builds trust with user
   - Develops inside jokes
   - Has rich shared history
2. Capture Agent A's final 40-D vector at turn 100

3. Agent B starts fresh (turn 0)
4. **Inject Agent A's 40-D vector into Agent B**
5. Resume conversation with Agent B

**Test Question:**
Does Agent B behave like they have Agent A's history?
- References past events (they shouldn't know)
- Maintains emotional tone
- Continues personality traits

**If encoding works:**
- Agent B acts like continuation of Agent A
- Emotional/personality continuity

**If NOT encoding:**
- Agent B acts confused/generic
- No continuity

### Part D: Controlled Modification Test

**Method:**
1. Train agent to be fearful (100 turns of scary scenarios)
2. Capture 40-D vector
3. Use LLM to analyze vector and identify "fear dimensions"
4. Manually zero out those dimensions
5. Resume conversation

**Test:**
Does agent stop exhibiting fear?

**Alternative approach:**
1. Have LLM label each dimension:
   - "Dimension 7 seems to correlate with impulsivity"
   - "Dimension 23 seems to correlate with social anxiety"
2. Create synthetic "courageous agent" by:
   - Boosting dimension 7 (impulsivity)
   - Reducing dimension 23 (social anxiety)
3. Test if agent behaves more courageously

### Part E: Blind Human Evaluation

**Method:**
1. Generate 5 agent states with different emotions:
   - State A: High fear, low courage
   - State B: High courage, low fear
   - State C: High joy, low sadness
   - State D: High sadness, low joy
   - State E: Neutral (baseline)

2. Have each agent respond to same 20 prompts

3. Present responses to human evaluators (blind to state)

4. Ask evaluators: "Which agent seems most fearful? Most brave? Most sad?"

**If encoding works:**
- Evaluators correctly identify emotional states
- Inter-rater reliability is high

**If NOT encoding:**
- Evaluators guess randomly
- No consensus

## Success Criteria

**Phenomenal state encoding is VALIDATED if:**

1. ✓ Emotional clusters visible in t-SNE plot (Part A)
2. ✓ Vector arithmetic produces behavioral changes (Part B)
3. ✓ State transfer maintains personality (Part C)
4. ✓ Dimension zeroing affects behavior (Part D)
5. ✓ Humans correctly identify emotional states >70% accuracy (Part E)

**If 4+ of 5 tests pass:** The 40-D vector is meaningful
**If <3 tests pass:** It's anthropomorphization

## Implementation Priority

**Phase 1 (Immediate):**
- Part A: Emotional clustering visualization
- Part E: Blind human evaluation

These are quick to implement and give clear yes/no answers.

**Phase 2 (After validation):**
- Part B: Vector arithmetic
- Part C: State transfer

These are more complex but prove controllability.

**Phase 3 (Advanced):**
- Part D: Dimension interpretation
- Tool: "Emotional state editor UI"

This is for content creators to author agent states.

## Commercial Implications

**If validated:**
- **Asset type**: "Emotional State Snapshot" (.pv files)
- **Marketplace item**: Pre-configured personality vectors
  - "Brave Knight" state
  - "Fearful Villager" state
  - "Wise Mentor" state
- **Authoring tool**: Slider UI to adjust dimensions
  - "Make agent 20% more courageous"
  - Visual feedback shows behavior change

**If NOT validated:**
- Rearchitect phenomenal state
- Maybe need supervised learning with emotional labels
- Or abandon vector encoding, use LLM memory instead

## Why This Is THE Test

You said: *"how do we PROVE that it encodes emotion?"*

**Answer:** Clustering, arithmetic, transfer, and human evaluation.

If the vector is just noise, these tests will fail.
If it genuinely encodes emotional state, these tests will pass.

No subjectivity. Pure empiricism.

---

**Next Steps:**

1. Complete Experiment 2 (100-turn test) - running now
2. Implement Experiment 3 Part A (emotional clustering)
3. Implement Experiment 3 Part E (blind evaluation)
4. Based on results: celebrate or pivot

This is the scientific validation your storytelling engine needs.
