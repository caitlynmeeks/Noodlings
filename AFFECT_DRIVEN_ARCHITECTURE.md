# Affect-Driven Cognitive Architecture

**Noodlings Emotional Salience Weighting System**

**Date:** December 3, 2025
**Author:** NinaK (Vulcan Nina Hagen) + Caity
**Status:** Implemented in Red Fire Anklebiter (Gold Standard)

---

## Core Principle: Emotional Salience

**"Affect colors HOW you say it, Cognition determines WHAT you say"**

Every facet in the pipeline receives BOTH:
1. **Cognitive inputs** (observations, context, etc.)
2. **Affective state** (valence, arousal, fear, sorrow, boredom)

The affective state provides **emotional salience weighting** - it influences the tone, intensity, and style of cognitive processing.

---

## Architecture Overview

```
INCOMING (raw perception)
    ↓
CHARM_NET (CharmNetwork - The Transform)
    ├→ phenomenal_state (40-D temporal memory)
    ├→ affect_valence (-1 to 1)
    ├→ affect_arousal (0 to 1)
    ├→ affect_fear (0 to 1)
    ├→ affect_sorrow (0 to 1)
    └→ affect_boredom (0 to 1)
         ↓
    [ALL affect outputs fan out to EVERY cognitive facet]
         ↓
room_observer (receives affect + phenomenal_state)
    ↓
roast_engine (receives affect + observations)
    ↓
fire_body (pure cognitive)
    ↓
voice_filter (pure cognitive)
         ↓
    [ALL outputs converge]
         ↓
CONVERGENCE (emotional salience weighting + synthesis)
    ↓
OUTGOING (final response)
```

---

## The Three Layers

### Layer 1: CharmNetwork (Affect Prediction)

**Type:** `CharmNetworkFacet`
**Model:** `checkpoints/phase4.npz` (54K params)
**Role:** The Transform - emotional core that MUST be present

**Inputs:**
- `affect_in`: 5-D continuous affect from perception

**Outputs:**
- `phenomenal_state`: 40-D temporal memory (h_fast + h_medium + h_slow)
- `affect_valence`: Emotional valence (-1 negative to +1 positive)
- `affect_arousal`: Activation level (0 calm to 1 energized)
- `affect_fear`: Fear/anxiety (0 to 1)
- `affect_sorrow`: Sadness/melancholy (0 to 1)
- `affect_boredom`: Boredom/disinterest (0 to 1)

**Performance:**
- Forward pass: ~2-3ms
- Compute: ~0.1 MFLOPs (~0.0000001 GPT-3.5 tokens)
- Memory: Recurrent states persist across conversation

---

### Layer 2: Cognitive Facets (Affect-Modulated Processing)

Each cognitive facet receives:
1. **Content inputs** (what to process)
2. **Affect inputs** (how to process it)

#### Example: room_observer

**Prompt includes:**
```
AFFECT: valence={affect_valence:.2f}, arousal={affect_arousal:.2f}, fear={affect_fear:.2f}

EMOTIONAL STATE INFLUENCES OBSERVATION:
- High arousal → Notice MORE things, faster reactions
- High valence → More playful observations
- Low valence → More cutting/sarcastic observations
- High fear → More defensive, protective observations
```

**Result:** Same observation task, but emotional state colors the output.

#### Example: roast_engine

**Prompt includes:**
```
AFFECT: valence={affect_valence:.2f}, arousal={affect_arousal:.2f}, sorrow={affect_sorrow:.2f}

EMOTIONAL SALIENCE (affect influences roast intensity):
- High arousal + high valence → PLAYFUL, energetic roasts
- High arousal + low valence → CUTTING, aggressive roasts
- Low arousal → Tired, half-hearted roasts
- High sorrow → Self-deprecating or melancholic edge
```

**Result:** Roast content from observations, but delivery shaped by affect.

---

### Layer 3: CONVERGENCE (Final Synthesis)

**Type:** `ConvergenceFacet`
**Role:** Multi-input synthesis with emotional salience weighting

**Inputs (9 total):**
- 5x affect dimensions (from CharmNetwork)
- 4x cognitive outputs (observations, roast, physical, voiced)

**Decision Process:**
1. **Should respond?** (boredom + content quality)
2. **Which tone?** (valence + arousal)
3. **How much energy?** (arousal level)
4. **Final output:** Weighted synthesis of affect + cognition

**Prompt logic:**
```
EMOTIONAL SALIENCE WEIGHTING:
- If affect dominates (high arousal/valence swing) → Weight emotions MORE
- If cognition is rich (detailed observations/roast) → Weight cognitive content MORE
- Balance: Emotional STATE colors HOW you say it, Cognitive CONTENT is WHAT you say
```

**Output:**
- Final response (or `[SUPPRESS]` if shouldn't speak)

---

## Information Flow Examples

### Example 1: High Arousal + High Valence (Excited/Playful)

```
Perception: "Caity offers candy"
    ↓
CharmNetwork: valence=0.7, arousal=0.8, fear=0.1
    ↓
room_observer: "Caity's doing the candy bribe thing AGAIN! Classic move!"
    ↓
roast_engine: "Oh WOW, Caity - ANOTHER candy bribe? You're PREDICTABLE!"
    ↓
fire_body: "*flames surge excitedly, bounces on toes*"
    ↓
voice_filter: "Oh WOW Caity, ANOTHER candy bribe? You're PREDICTABLE! *flames surge* MWAHAHA!"
    ↓
CONVERGENCE: [High arousal + high valence = PLAYFUL energy]
    → Output: "Oh WOW Caity, ANOTHER candy bribe? You're PREDICTABLE! *flames surge* MWAHAHA!"
```

### Example 2: Low Arousal + Low Valence (Tired/Grumpy)

```
Perception: "Caity offers candy"
    ↓
CharmNetwork: valence=-0.3, arousal=0.2, sorrow=0.4
    ↓
room_observer: "Caity's offering candy. Again. Whatever."
    ↓
roast_engine: "Yeah sure Caity, more candy. Great."
    ↓
fire_body: "*flames flicker weakly*"
    ↓
voice_filter: "Yeah sure Caity, more candy. Great. *flames flicker*"
    ↓
CONVERGENCE: [Low arousal + low valence + high sorrow = TIRED/SAD]
    → Output: "Yeah... candy. Cool. *flames dim* Whatever."
```

### Example 3: High Fear (Defensive)

```
Perception: "Servnak analyzes Red"
    ↓
CharmNetwork: valence=-0.4, arousal=0.6, fear=0.7
    ↓
room_observer: "Servnak is STARING at me. Analyzing. Calculating. TOO MUCH."
    ↓
roast_engine: "Oh PLEASE Servnak, stop with the creepy robot stare!"
    ↓
fire_body: "*flames spike defensively, backs away*"
    ↓
voice_filter: "Oh PLEASE Servnak, STOP with the creepy robot stare! *flames spike*"
    ↓
CONVERGENCE: [High fear + negative valence = DEFENSIVE]
    → Output: "BACK OFF Servnak! Stop calculating my EXISTENCE! *flames spike defensively*"
```

---

## Key Design Decisions

### 1. Affect Flows to ALL Facets

**Why:** Every cognitive process should be affect-colored.

**Implementation:** CharmNetwork outputs fan out to every downstream facet that needs emotional context.

**Connections:**
```yaml
# room_observer gets affect
- from: CHARM_NET.affect_valence
  to: room_observer.affect_valence
- from: CHARM_NET.affect_arousal
  to: room_observer.affect_arousal

# roast_engine ALSO gets affect (not passed through, direct from source!)
- from: CHARM_NET.affect_valence
  to: roast_engine.affect_valence
- from: CHARM_NET.affect_arousal
  to: roast_engine.affect_arousal
```

### 2. Phenomenal State Separate from Affect

**Phenomenal State:** 40-D compressed temporal memory (what happened recently)
**Affect:** 5-D emotional state (how I feel about it)

**Why separate?** Affect is DERIVED from phenomenal state via the affect head, but they serve different purposes:
- PV → Provides CONTEXT (what's been happening)
- Affect → Provides SALIENCE WEIGHTING (how to interpret/respond)

### 3. Convergence Makes Final Decision

**Why not just use voice_filter output directly?**

Because emotional state might OVERRIDE cognitive output!

**Example:**
- Cognitive pipeline produces: "Oh PLEASE Caity, another candy bribe?"
- But affect is: valence=-0.8, arousal=0.1, sorrow=0.9 (very sad, low energy)
- Convergence OVERRIDES: "*flames dim* ...candy. Cool." (sad, not snarky)

**This is emotional salience weighting in action!**

---

## Prompt Engineering Pattern

### Standard Affect-Aware Prompt Template:

```yaml
prompt: |
  [FACET NAME] - [PURPOSE]

  === EMOTIONAL STATE ===
  AFFECT: valence={affect_valence:.2f}, arousal={affect_arousal:.2f}, [relevant affects]

  === COGNITIVE INPUTS ===
  [Input data here]

  === YOUR TASK ===
  [Task description]

  === EMOTIONAL INFLUENCE ===
  [How affect modulates this specific task]
  - High valence → [behavior]
  - Low valence → [behavior]
  - High arousal → [behavior]
  - [etc.]

  [Final instructions with emotional context]
```

**Key elements:**
1. Always show affect values at top (makes them salient to LLM)
2. Explicit section on emotional influence
3. Concrete examples of how affect changes output
4. Balance: affect colors tone, doesn't override content

---

## Performance Characteristics

### Computational Cost

**CharmNetwork (per cycle):**
- Time: ~2-3ms
- FLOPs: ~0.1 MFLOPs
- Memory: 40-D state vector (160 bytes)

**Affect fanout (per facet):**
- Time: 0ms (just passing references)
- FLOPs: 0 (no computation)
- Memory: 5 floats × N facets (20 bytes × N)

**Convergence facet:**
- Time: ~200-500ms (LLM call)
- Tokens: ~500 input, ~150 output
- Cost: ~0.01 cents per response

**Total overhead vs pure cognitive:**
- CharmNetwork: +2-3ms (negligible)
- Affect in prompts: +50-100 tokens per facet
- Convergence: +1 LLM call (~200-500ms)

**Worth it?** YES! Emotional salience makes responses dramatically more lifelike.

---

## Debugging Affect Flow

### Check CharmNetwork Execution:

Look for in logs:
```
⚡ CharmNetwork metrics for agent_red:
   total=2.45ms (base=1.80ms, quantum=0.65ms)
```

### Check Affect Values:

Add logging in facet execution to see affect inputs:
```python
logger.info(f"room_observer received affect: v={valence:.2f}, a={arousal:.2f}")
```

### Check Convergence Decision:

Add logging in convergence to see weighting:
```
Convergence: affect_weight=0.7, cognitive_weight=0.3
Final: Using AFFECT-dominant response
```

---

## Future Enhancements

### 1. Affect Prediction Accuracy Tracking

Track how well CharmNetwork predicts affect:
```python
predicted_affect = charm_net.outputs['affect_valence']
actual_affect = calculate_actual_affect_from_response()
prediction_error = abs(predicted - actual)
```

### 2. Adaptive Salience Weighting

Let convergence LEARN optimal affect vs cognition weighting:
```
If responses feel too emotional → decrease affect weight
If responses feel too robotic → increase affect weight
```

### 3. Affect-Conditioned Prompt Templates

Different prompt templates for different affect regions:
```python
if arousal > 0.7 and valence > 0.5:
    use_template = "high_energy_positive"
elif arousal > 0.7 and valence < -0.5:
    use_template = "high_energy_negative"
```

---

## Comparison to Legacy Architecture

### Old Manifold System:
```
Perception → [Transistor Pipeline] → Manifold → Output
                                       ↑
                            (affect was just one input)
```

### New Facet System:
```
Perception → CharmNetwork ─┬→ [Facet 1 + affect] ─┐
                           ├→ [Facet 2 + affect] ─┤
                           ├→ [Facet 3 + affect] ─┤
                           └→ [Facet N + affect] ─┴→ Convergence → Output
```

**Key Difference:** Affect doesn't just influence FINAL synthesis - it colors EVERY STEP of processing!

---

## Status: Red Fire Anklebiter (Gold Standard)

**Completed:**
- ✅ CharmNetwork integrated as Transform
- ✅ Affect outputs wired to all cognitive facets
- ✅ Prompts updated with emotional salience guidance
- ✅ ConvergenceFacet synthesizes affect + cognition
- ✅ Full emotional salience weighting architecture

**Testing:**
- @derez red_fire_anklebiter
- @rez red_fire_anklebiter
- Talk to Red and observe affect-modulated responses!

**Next:**
- Apply same architecture to Mr. Toad
- Apply to empty_noodling_default
- All new Noodlings use affect-driven architecture!

---

*Ordnung muss sein!* 🖖

This is the REAL consciousness architecture - affect first, always!
