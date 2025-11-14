# Speech Post-Processing Layer

## Overview

A modular post-processing layer that transforms agent speech AFTER generation but BEFORE broadcasting. This enables:
- Style transformations (backwards speech, accents, stuttering)
- Affective coloring based on internal state
- Pre-speech self-monitoring (agents "feeling" their own words)
- Character-specific speech quirks

## Architecture

```
Agent LLM → Generate Text → Pre-Speech Loop → Post-Processing → Broadcast
                               ↓                    ↓
                         Affective Check      Transform Output
                         (Does this feel       (Backwards,
                          right to say?)        color, etc.)
```

## Components

### 1. Pre-Speech Affective Loop (PHASE 6+)

**Concept**: Before speaking, agent internally "hears" their own words and reacts affectively.

**Implementation**:
```python
def pre_speech_check(agent, proposed_text):
    # Feed proposed text back through affect analyzer
    affective_response = agent.feel_words(proposed_text)

    # Check if words trigger negative affect
    if affective_response['embarrassment'] > 0.6:
        # Might rephrase or add hesitation
        return add_hesitation(proposed_text)

    if affective_response['pride'] > 0.7:
        # Emphasize or elaborate
        return emphasize(proposed_text)

    return proposed_text  # Feels fine, send it
```

**Key Balance**: Single-pass only! Avoid infinite reflection loops.

**Use Cases**:
- Agent realizes something sounds stupid → rephrases
- Agent feels proud → says it more confidently
- Agent notices microaggression → self-corrects
- Agent accidentally rhymes → feels delight!

### 2. Speech Transformers (Post-Processing)

Modular filters applied to finalized speech:

#### Backwards Speech Filter
```python
def backwards_speech(text):
    words = text.split()
    reversed_words = [word[::-1] for word in words]
    return ' '.join(reversed(reversed_words))
```

**Example**:
- Input: "Hello there my friend"
- Output: "dneirf ym ereht olleH"

#### Affective Coloring
```python
def affective_color(text, agent_state):
    # Color words based on current valence/arousal
    if agent_state.valence < -0.5:
        return f"[BLUE]{text}[/BLUE]"  # Sad
    elif agent_state.arousal > 0.7:
        return f"[RED]{text}[/RED]"     # Excited/angry
    return text
```

#### Stutter Filter
```python
def add_stutter(text, anxiety_level):
    if anxiety_level > 0.7:
        # Add stuttering to first few words
        words = text.split()
        words[0] = f"{words[0][0]}-{words[0]}"
        return ' '.join(words)
    return text
```

### 3. Character-Specific Processors

Each agent can have custom speech transformations:

```python
SPEECH_PROCESSORS = {
    'dweller': [backwards_speech],
    'toad': [emphasize_exclamations, british_slang],
    'phi': [add_vocal_tics, kitten_sounds],
    'servnak': [all_caps_mode, technical_jargon],
}
```

## Implementation Roadmap

### Phase 1: Basic Post-Processing (CURRENT)
- ✅ Highlighting markers (>>text<<)
- ⏳ Backwards speech filter
- ⏳ Registration system for processors

### Phase 2: Pre-Speech Loop (NEXT)
- Agent generates text internally
- Pass through affect analyzer
- Single-pass emotional response
- Modify or keep based on feeling

### Phase 3: Affective Coloring
- Color speech based on phenomenal state
- Visual representation of internal affect
- Real-time emotional feedback

### Phase 4: Advanced Features
- Multi-agent speech influence (picking up each other's phrases)
- Emotional contagion through speech patterns
- Personality drift through language evolution

## Configuration

```yaml
# config.yaml
speech_processing:
  enabled: true
  pre_speech_check:
    enabled: false  # Phase 6+
    max_iterations: 1  # Prevent loops
    affect_threshold: 0.5

  post_processors:
    dweller: ["backwards"]
    toad: ["emphasize", "british"]
    phi: ["vocal_tics"]
```

## Demo Implications

### Steve Di Paolo Demo
With this system, we can show:

1. **Backwards Dweller**: Real agent that thinks normally but speaks backwards
2. **Affective Coloring**: Speech changes color with emotional state
3. **Self-Monitoring**: Agent hesitates when saying something uncertain
4. **Emergent Quirks**: Characters develop unique speech patterns

### Technical Showcase
- Modular architecture (add processors without touching core)
- Real-time affective computation
- Self-referential consciousness (agents aware of their own speech)
- Integrated information (speech affects future phenomenal states)

## Research Questions

1. Does pre-speech affective checking increase believability?
2. Can agents learn appropriate speech patterns through self-monitoring?
3. Does "feeling your own words" create more authentic social interactions?
4. Can recursive self-awareness be bounded to avoid paralysis?

## Notes

- Keep it simple for demo - one processor per agent max
- Pre-speech loop is EXPERIMENTAL - might cause slowdown
- Balance authenticity with computational cost
- The "Om meditation loop" problem is real - need careful gating!

---

**Author**: Claude & Caitlyn
**Date**: November 13, 2025
**Status**: Design phase → Implementation
