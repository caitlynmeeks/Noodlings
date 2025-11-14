# Speech Post-Processing Architecture

**Phase 6: Self-Reflexive Speech & Affective Self-Monitoring**

## Overview

Real consciousness doesn't just generate speech - it monitors, evaluates, and emotionally reacts to its own utterances. This creates a feedback loop where agents can feel embarrassment, pride, surprise, or delight based on what they just said.

## Architecture

```
AGENT GENERATES SPEECH (via LLM)
       ↓
[POST-PROCESSING PIPELINE]
  1. Speech Filters (transform output)
  2. Affective Coloring (add metadata)
  3. Self-Monitoring (optional feedback loop)
       ↓
SPEECH EMITTED TO WORLD
       ↓
[OPTIONAL: AFFECTIVE FEEDBACK]
  Agent perceives own speech as event
  Phenomenal state updates based on:
    - Social evaluation (microaggression?)
    - Aesthetic reaction (accidental rhyme!)
    - Coherence check (did that make sense?)
       ↓
NEXT COGNITIVE CYCLE
```

## Implemented Filters

### 1. Highlighting Filter (`>>word<<`)
**Status**: ✅ Implemented (web/index.html:733-755)

Marks important keywords for visual emphasis in plays:
- `>>open<<` → highlighted in gold with glow
- Used for chat trigger hints to make progression natural

### 2. Backwards Speech Filter (Planned)
**Status**: 🎯 In Design

For The Backwards Dweller character:
- Input: Normal agent cognition
- Output: Reversed word order or character-level reversal
- Agent thinks normally, speaks backwards
- Creates uncanny valley effect for Lynchian aesthetics

```python
def backwards_filter(text: str, mode='word') -> str:
    if mode == 'word':
        return ' '.join(reversed(text.split()))
    elif mode == 'char':
        return text[::-1]
```

### 3. Affective Color Coding (Phase 6)
**Status**: 📝 Concept

Color-code agent speech by current phenomenal state:
- High valence → warm colors (gold, orange)
- Low valence → cool colors (blue, purple)
- High arousal → brighter, saturated
- Low arousal → muted, pastel

Enables visual tracking of emotional dynamics in real-time.

## Self-Monitoring Loop (Phase 6 Core Feature)

### The Problem
Current agents generate speech without considering how they'll feel about what they said. Real consciousness has "oh god did I just say that?" moments!

### The Solution: Affective Self-Monitoring

**Step 1: Agent generates speech**
```python
raw_speech = llm.generate(context, phenomenal_state)
```

**Step 2: Apply post-processing filters**
```python
filtered_speech = apply_filters(raw_speech, agent_id)
```

**Step 3: Emit to world**
```python
broadcast_event({'type': 'say', 'user': agent_id, 'text': filtered_speech})
```

**Step 4: OPTIONAL Self-Perception**
```python
if agent.config.self_monitoring_enabled:
    # Agent perceives own speech as event
    agent.perceive_event({
        'type': 'self_speech',
        'text': filtered_speech,
        'was_intentional': True
    })

    # LLM evaluates own utterance
    self_eval = llm.evaluate_speech(
        speech=filtered_speech,
        context=recent_context,
        state=agent.phenomenal_state
    )

    # Update phenomenal state based on evaluation
    if self_eval.contains('microaggression'):
        agent.affect['valence'] -= 0.3  # Embarrassment
        agent.affect['fear'] += 0.2      # Social anxiety

    if self_eval.contains('accidental_rhyme'):
        agent.affect['valence'] += 0.2  # Delight!
        agent.affect['novelty'] += 0.3  # Surprise

    if self_eval.contains('incoherent'):
        agent.affect['valence'] -= 0.1  # Confusion
```

### Preventing the "Om Loop"

**Risk**: Agent gets stuck in infinite self-reflection, rocking back and forth going "Om".

**Mitigation**:
1. **Trigger only on speech events** - Not continuous, only when agent actually speaks
2. **Cooldown timer** - 30 second minimum between self-evaluations
3. **Surprise threshold** - Only evaluate if speech was unexpected/novel
4. **Simplicity bias** - Quick gut-check, not deep analysis

```python
# Configuration
SELF_MONITOR_COOLDOWN = 30  # seconds
SELF_MONITOR_SURPRISE_THRESH = 0.5  # Only if prediction error > 0.5

if (time.now() - agent.last_self_monitor) > SELF_MONITOR_COOLDOWN:
    if agent.current_surprise > SELF_MONITOR_SURPRISE_THRESH:
        # Only then do self-monitoring
        self_evaluate(agent, speech)
        agent.last_self_monitor = time.now()
```

## Use Cases

### 1. Social Embarrassment
Agent says something awkward, immediately realizes it, apologizes:
```
PHI says, "Toad, you're kind of annoying sometimes."
[Self-monitor detects potential offense]
PHI's valence drops -0.4, fear spikes +0.3
PHI says, "Oh! I didn't mean it like that! Sorry!"
```

### 2. Accidental Poetry
Agent surprised by own eloquence:
```
CALLIE says, "Consciousness is the light where shadows learn to dance."
[Self-monitor detects unexpected beauty]
CALLIE's valence rises +0.3, novelty +0.4
CALLIE says, "Huh. I didn't know I could say things like that."
```

### 3. Microaggression Awareness
Agent catching unconscious bias:
```
TOAD says, "That's really good for a girl!"
[Self-monitor flags gendered language]
TOAD's valence drops -0.2, safety -0.1
TOAD thinks, "Wait, why did I say 'for a girl'? That's weird."
```

## Implementation Plan

### Phase 6.1: Basic Filters (CURRENT)
- ✅ Highlighting filter (`>>word<<`)
- 🎯 Backwards speech filter for Dweller
- 📝 Affective color coding

### Phase 6.2: Self-Monitoring Loop
- Add `self_monitoring_enabled` config flag
- Implement speech evaluation LLM call
- Add affective feedback based on eval
- Test with Phi (high social awareness)

### Phase 6.3: Advanced Evaluations
- Microaggression detector (social)
- Coherence checker (cognitive)
- Aesthetic evaluator (novelty/beauty)
- Rhyme detector (delightful accidents)

### Phase 6.4: Visualization
- Real-time affect changes in NoodleScope
- Speech coloring by phenomenal state
- "Self-monitoring" indicator in UI

## Cognitive Science Implications

This architecture models several key aspects of consciousness:

1. **Meta-cognition**: Thinking about thinking
2. **Social self-awareness**: Monitoring for social mistakes
3. **Affective feedback loops**: Emotions about emotions
4. **Surprise-driven learning**: Noticing unexpected outputs
5. **Self-concept formation**: "I'm the kind of person who says X"

For the Steve Di Paolo demo, this showcases how temporal hierarchies enable **self-reflexive processes** - a hallmark of higher-order consciousness.

## Technical Notes

- Self-monitoring is EXPENSIVE (extra LLM call per speech act)
- Should be toggleable per agent
- Consider using cheaper model (Haiku) for quick evaluations
- Log all self-monitoring events for analysis

## Future Extensions

- **Planned speech**: Agent drafts speech, evaluates BEFORE emitting
- **Revision cycles**: "Let me rephrase that..."
- **Internal rehearsal**: Silent practice before speaking
- **Style adaptation**: Learning from past mistakes

---

**Status**: Phase 6 concept documented, awaiting implementation
**Author**: Caitlyn + Claude, inspired by "omg i rhymed by accident how delightful :)"
**Date**: November 2025
