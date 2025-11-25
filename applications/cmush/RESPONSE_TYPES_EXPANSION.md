# Response Types Expansion & Component Architecture

**Date**: November 25, 2025
**Status**: Design specification
**Priority**: HIGH - Affects transistor prompt design and output channels

---

## Current Response Types

**Existing**:
1. **SAY** - Verbal speech (directed to others)
2. **DO** - Physical action (movement, manipulation)
3. **THINK** - Internal rumination (not visible to others)
4. **NONE** - No response (silent observation)

---

## Proposed Expansion

### Core Response Types

1. **SAY** - Verbal speech
   - Output: Text dialogue
   - Example: "Hello there! How are you?"

2. **EMOTE** - Emotional expression with action
   - Output: Third-person action description
   - Example: "*jumps excitedly* *tail wagging*"

3. **DO** - Physical action (no emotion)
   - Output: Third-person action
   - Example: "*picks up the stone* *examines it*"

4. **THINK** - Internal rumination
   - Output: Internal monologue (not broadcast)
   - Example: "I wonder what that means..."

5. **FEEL** - Somatic/bodily sensation (NEW)
   - Output: Body state description
   - Example: "Heart racing... butterflies in stomach..."

6. **NONE** - Silent observation
   - Output: Nothing

### Specialized Output Channels (Components)

7. **FACS** - Facial Action Coding System
   - Component: `FacialExpressionComponent`
   - Output: Facial muscle movements
   - Example: AU6 (cheek raiser) + AU12 (lip corner puller) = smile
   - Only generated if component present

8. **LABAN** - Laban Movement Analysis
   - Component: `BodyLanguageComponent`
   - Output: Movement quality descriptors
   - Example: "Light, direct, sudden" (dabbing motion)
   - Only generated if component present

---

## Transistor Prompt Redesign

### Current Problem

Transistor prompts are ANALYTICAL:
```
"Transform this 5D emotional texture into RICH language."
```

Output: "Heaviness... like everything's slightly gray" (description)

### Proposed Solution

Transistor prompts are ACTION-ORIENTED and FIRST-PERSON:
```
"Based on your emotional state, what do you WANT to {SAY/DO/EMOTE}?"
```

Output: "I wanna jump for joy with this great news!" (desire/intention)

### New AffectTransistor Prompt Template

```
You are experiencing this emotional state:
- Valence: {valence:.3f} (how you feel overall)
- Arousal: {arousal:.3f} (your energy level)
- Dominance: {dominance:.3f} (your sense of power/control)
- Sorrow: {sorrow:.3f} (your sadness level)
- Boredom: {boredom:.3f} (your engagement level)

Something just happened: "{input_text}"

TASK: Write what you WANT to {response_type} in first-person, RAW emotional response.

Examples:

WANT TO SAY (valence=0.7, arousal=0.8):
"HEY! Wow I wasn't expecting that! I feel GREAT! This is awesome!"

WANT TO SAY (valence=-0.4, dominance=0.1, sorrow=0.7):
"I... I should probably keep quiet right now. I don't want to upset anyone."

WANT TO DO (valence=0.6, arousal=0.8):
"I wanna jump! Dance! Spin around! This energy needs OUT!"

WANT TO DO (valence=-0.3, arousal=0.2, dominance=0.1):
"I should probably just... hang my head down low. Curl up small."

WANT TO EMOTE (valence=0.5, arousal=0.3):
"Soft smile spreading across my face... quiet contentment settling in."

Write your raw emotional impulse - what you WANT to {response_type}. 1-2 sentences, first-person.
```

---

## FACS Component Design

### FacialExpressionComponent

**Purpose**: Generate FACS codes from affect vector

**Input**: Predicted affect (5D continuous)

**Output**: FACS action units
```json
{
  "facs": {
    "AU6": 0.8,   // Cheek raiser (joy)
    "AU12": 0.9,  // Lip corner puller (smile)
    "AU1": 0.3,   // Inner brow raiser (surprise)
    "AU4": 0.2    // Brow lowerer (concern)
  },
  "description": "Broad genuine smile with slight surprise"
}
```

**Integration**:
```yaml
cognitive_components:
  facial:
    type: "FacialExpressionComponent"
    system: "FACS"  # Facial Action Coding System
    enabled: true
```

**Only generates output if component present on Noodling.**

### Implementation

```python
class FacialExpressionComponent(CognitiveTransistor):
    """
    Generates FACS (Facial Action Coding System) codes from affect.

    Maps continuous affect vector → facial muscle activations.
    Only active if added to Noodling's components.
    """

    DEFAULT_PROMPT = """
    Based on your emotional state, what facial expression would naturally appear?

    AFFECT STATE:
    - Valence: {valence:.3f}
    - Arousal: {arousal:.3f}
    - Dominance: {dominance:.3f}
    - Sorrow: {sorrow:.3f}
    - Boredom: {boredom:.3f}

    Generate FACS action units (0.0 to 1.0 intensity):
    Available AUs:
    - AU1: Inner brow raiser (surprise, concern)
    - AU2: Outer brow raiser (surprise)
    - AU4: Brow lowerer (anger, concern)
    - AU6: Cheek raiser (joy, genuine smile)
    - AU12: Lip corner puller (smile)
    - AU15: Lip corner depressor (sadness)
    - AU20: Lip stretcher (fear)
    - AU26: Jaw drop (surprise)

    Output JSON only:
    {
      "AU6": 0.8,
      "AU12": 0.9
    }
    """

    async def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        # Generate FACS codes from affect
        # Return as structured data for renderer
        pass
```

---

## LABAN Component Design

### BodyLanguageComponent

**Purpose**: Generate Laban movement descriptors from affect

**Input**: Predicted affect (5D continuous)

**Output**: Laban effort qualities
```json
{
  "laban": {
    "weight": "light",      // light vs strong
    "time": "sustained",    // sustained vs sudden
    "space": "indirect",    // direct vs indirect
    "flow": "free"          // bound vs free
  },
  "description": "Gentle, flowing movements with hesitant quality"
}
```

**Integration**:
```yaml
cognitive_components:
  body_language:
    type: "BodyLanguageComponent"
    system: "LABAN"  # Laban Movement Analysis
    enabled: true
```

### Implementation

```python
class BodyLanguageComponent(CognitiveTransistor):
    """
    Generates Laban movement descriptors from affect.

    Maps continuous affect vector → movement qualities.
    Only active if added to Noodling's components.
    """

    DEFAULT_PROMPT = """
    Based on your emotional state and intended action, describe your movement quality using Laban effort dimensions.

    AFFECT STATE:
    - Valence: {valence:.3f}
    - Arousal: {arousal:.3f}
    - Dominance: {dominance:.3f}

    ACTION: {input_text}

    Laban Effort Qualities:
    - Weight: light (gentle, delicate) vs strong (forceful, powerful)
    - Time: sustained (slow, leisurely) vs sudden (fast, abrupt)
    - Space: indirect (meandering, unfocused) vs direct (aimed, focused)
    - Flow: free (flowing, continuous) vs bound (controlled, restrained)

    Output JSON with your movement quality:
    {
      "weight": "light|strong",
      "time": "sustained|sudden",
      "space": "indirect|direct",
      "flow": "free|bound"
    }
    """
```

---

## Response Type Usage

### When to use each type:

**SAY**:
- Greeting someone
- Answering questions
- Making comments
- Dialogue

**EMOTE**:
- Expressing emotion physically
- Reactive gestures
- Body language + emotion
- Example: "*sighs heavily*", "*bounces excitedly*"

**DO**:
- Instrumental actions (picking up objects)
- Movement (walking, turning)
- Manipulation (opening doors)
- Physical tasks

**THINK**:
- Internal monologue
- Observations
- Self-reflection
- Not broadcast to others

**FEEL** (proposed):
- Somatic sensations
- Body awareness
- Physical responses to emotion
- Example: "Stomach drops... cold rush..."

---

## Proposed Response Type Expansion

Add these for flexibility:

1. **WHISPER** - Quiet speech (only nearby hear)
2. **SHOUT** - Loud speech (heard in adjacent rooms)
3. **GESTURE** - Non-verbal communication (wave, nod, point)
4. **EMOTE** - Emotional expression with action
5. **FEEL** - Somatic/body sensations
6. **FACS** - Facial expression (if FacialExpressionComponent present)
7. **LABAN** - Body movement quality (if BodyLanguageComponent present)

---

## Updated Transistor Architecture

### Input to Transistor

```python
context = {
    'response_decision': {
        'response_type': 'SAY',  # or DO, EMOTE, THINK, etc.
        'guidance': 'greet the newcomer warmly'
    },
    'predicted_affect': {
        'valence': 0.6,
        'arousal': 0.7,
        'dominance': 0.5,
        'sorrow': 0.1,
        'boredom': 0.0
    }
}
```

### Transistor Processing

Each transistor generates FIRST-PERSON INTENTION:

**AffectTransistor** → "I wanna jump for joy!"
**PersonalityTransistor** → "My competitive nature says CHALLENGE THEM!"
**CulturalTransistor** → "My beliefs say I should welcome them warmly"
**IntuitionTransistor** → "I sense they're nervous - I should be gentle"

### Manifold Integration

Manifold blends all perspectives:
```
Input: "Someone just arrived"
Response type: SAY

AffectTransistor (0.85):  "I feel PUMPED! I wanna shout HI!"
PersonalityTransistor (0.80): "My competitive side says assert dominance!"
CulturalTransistor (0.75): "My beliefs say greet them warmly"

Manifold output: "HEY THERE! Welcome! I'm Red Fire - the BEST anklebiter here!"
```

---

## Implementation Tasks

1. **Update AffectTransistor prompt** - First-person action-oriented
2. **Update other transistor prompts** - Same pattern
3. **Add EMOTE response type** - Separate from DO
4. **Add FEEL response type** - Somatic sensations
5. **Create FacialExpressionComponent** - FACS generation
6. **Create BodyLanguageComponent** - Laban generation
7. **Update ResponseTypeDecider** - Support new types
8. **Update manifold** - Handle new output formats

---

Shall I proceed with:
A) Updating AffectTransistor prompt to action-oriented first-person
B) Creating FACS/Laban components
C) Adding new response types (EMOTE, FEEL)
D) All of the above

Your preference?
