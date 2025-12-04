# Character Layer Routing - Preserving Identity Across Response Types

**Ensuring all responses (roast, denial, panic, etc.) go through character embodiment**

**Date:** December 3, 2025
**Author:** NinaK + Caity
**Problem Solved:** Psychological responses bypassing character layers

---

## The Problem

When adding psychological defense facets (denial, rationalization, panic), they were routing directly to CONVERGENCE:

```
denial_defense → CONVERGENCE  ❌ (doesn't sound like Red!)
roast_engine → fire_body → voice_filter → CONVERGENCE  ✅ (sounds like Red)
```

**Result:** Roasts sound like Red, but denials sound generic!

---

## The Solution: Response Selector + Shared Character Layers

### Architecture Pattern

```
[Multiple Response Generators]
    roast_engine (normal response)
    denial_defense (psychological defense)
    panic_response (fight/flight)
    humor_deflection (comedy defense)
         ↓ ↓ ↓ ↓
    response_selector
    (picks winner by salience)
         ↓
    selected_response
         ↓
[Character Embodiment Layers]
    fire_body (physical actions)
         ↓
    voice_filter (character voice)
         ↓
    CONVERGENCE (final synthesis)
```

**Key insight:** Character layers are GENERIC processors that work on ANY response type!

---

## Response Selector Facet

**Purpose:** Route highest-salience response through character layers

**Type:** ScriptedFacet with continuous selection

```yaml
- id: response_selector
  name: Response Selector
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      // Get salience of all response generators
      const facet_sal = inputs.facet_salience || {};

      const denial_sal = facet_sal.denial_defense?.salience || 0;
      const roast_sal = facet_sal.roast_engine?.salience || 0.5;
      const panic_sal = facet_sal.panic_response?.salience || 0;

      // Pick winner (continuous - could blend in future!)
      let selected = "roast";  // Default
      let max_sal = roast_sal;

      if (denial_sal > max_sal) {
        selected = "denial";
        max_sal = denial_sal;
      }
      if (panic_sal > max_sal) {
        selected = "panic";
      }

      return {
        salience: 1.0,
        shouldExecute: true,
        customData: {
          selected_type: selected,
          denial_salience: denial_sal,
          roast_salience: roast_sal,
          panic_salience: panic_sal
        }
      };
    }

  prompt: |
    RESPONSE SELECTOR - Route by Salience

    SALIENCES (continuous):
    - Denial: {customData.denial_salience:.3f}
    - Roast: {customData.roast_salience:.3f}
    - Panic: {customData.panic_salience:.3f}

    SELECTED: {customData.selected_type}

    INPUTS:
    - Denial: {denial_response}
    - Roast: {roast}
    - Panic: {panic_response}

    Output the SELECTED response (winner by highest salience).

  inputs:
    - denial_response (optional)
    - roast (required)
    - panic_response (optional)

  outputs:
    - selected_response
```

---

## Generic Character Layer Pattern

### Fire Body (Generic Embodiment)

**Before (specific to roasts):**
```yaml
inputs:
  - name: roast
prompt: "Based on the roast, what does fire-body do?"
```

**After (generic to any response):**
```yaml
inputs:
  - name: selected_response  # Could be roast, denial, panic, etc.
prompt: |
  RESPONSE CONTENT: {selected_response}

  Based on response content (could be roast OR denial OR panic),
  what does your fire-body DO?

  Examples:
  - Roast: "*flames surge, jumps on shoulder*"
  - Denial: "*flames flicker nervously*"
  - Panic: "*flames spike defensively, backs away*"
```

**Result:** Same facet handles ALL response types!

---

### Voice Filter (Generic Character Voice)

**Before (roast-specific):**
```yaml
prompt: |
  Combine roast + physical into Red's voice style
```

**After (response-agnostic):**
```yaml
prompt: |
  RESPONSE CONTENT: {selected_response}
  PHYSICAL: {physical_action}

  Add Red's voice to ANY response type:
  - Roasting: "Oh PLEASE", "MWAHAHA"
  - Defensive: "That's NOT", "ACTUALLY"
  - Panicked: "WAIT", "OH NO"

  Adapt voice connectors to emotional tone!
```

**Result:** Red always sounds like Red, regardless of response type!

---

## Complete Assembly Flow Example

### Scenario: Harsh Criticism

**Input:** "Red, everyone hates you."

**Step-by-step execution:**

```
1. INCOMING: "Red, everyone hates you."
     ↓
2. CHARM_NET: valence=-0.8, arousal=0.9, fear=0.6
     ↓ (affects fan out)
     ├──→ room_observer: "Caity said something harsh..."
     │         ↓
     ├──→ roast_engine (low salience due to negative affect)
     │    Roast: "Yeah sure, great observation..."
     │    Salience: 0.25 (suppressed by negative valence)
     │
     └──→ denial_defense (HIGH salience!)
          Distress: 0.9 × 0.9 = 0.81
          Salience: sigmoid(0.81) + fear×0.3 ≈ 0.95
          Denial: "That's NOT true! You're WRONG!"

3. response_selector:
   Roast sal: 0.25
   Denial sal: 0.95
   → SELECTED: denial
   Output: "That's NOT true! You're WRONG!"

4. fire_body:
   Input: "That's NOT true! You're WRONG!"
   Analysis: Defensive content → nervous/threatened
   Output: "*flames flicker nervously, backs away*"

5. voice_filter:
   Content: "That's NOT true! You're WRONG!"
   Physical: "*flames flicker nervously, backs away*"
   Voice: Add CAPS + defensive connectors
   Output: "That's NOT true! You're WRONG! *flames flicker nervously* ACTUALLY!"

6. CONVERGENCE:
   All inputs received, denial_weight=0.88
   Final check: Should speak? Yes!
   Output: "That's NOT true! You're WRONG! *flames flicker nervously* ACTUALLY!"

7. OUTGOING → Response sent to chat
```

**Red's denial sounds EXACTLY like Red!**

---

## Benefits of This Design

### 1. Character Consistency
ALL responses go through fire_body + voice_filter = always in-character

### 2. Reusable Components
Fire body doesn't care about response TYPE, just adds appropriate physical reactions

### 3. Emergent Variety
Different psychological states produce different physical reactions automatically:
- Happy roast: "*flames surge excitedly*"
- Sad denial: "*flames dim, flickers weakly*"
- Panicked reaction: "*flames spike, tail lashes*"

### 4. Easy Extension
Want to add new response types? Just add the facet and wire to response_selector!

```yaml
- id: sarcasm_facet
  → response_selector (automatically routes through body/voice!)
```

---

## Future: Blended Responses

Instead of winner-take-all, blend TOP N responses:

```javascript
// In response_selector
function computeSalience(inputs, context) {
  const sal = inputs.facet_salience;

  // Get top 2 responses
  const responses = [
    { type: "denial", sal: sal.denial_defense?.salience || 0, text: inputs.denial_response },
    { type: "roast", sal: sal.roast_engine?.salience || 0, text: inputs.roast },
    { type: "panic", sal: sal.panic_response?.salience || 0, text: inputs.panic_response }
  ].sort((a, b) => b.sal - a.sal);

  const top1 = responses[0];
  const top2 = responses[1];

  // If top 2 are close, BLEND them!
  if (top2.sal > top1.sal * 0.6) {
    return {
      salience: 1.0,
      shouldExecute: true,
      customData: {
        blend_mode: true,
        primary: top1.type,
        secondary: top2.type,
        primary_weight: top1.sal / (top1.sal + top2.sal),
        secondary_weight: top2.sal / (top1.sal + top2.sal)
      }
    };
  }

  // Otherwise, single winner
  return {
    salience: 1.0,
    shouldExecute: true,
    customData: {
      blend_mode: false,
      selected_type: top1.type
    }
  };
}
```

**Prompt:**
```
If blend_mode:
  Primary ({primary_weight:.0%}): {primary_text}
  Secondary ({secondary_weight:.0%}): {secondary_text}

  BLEND these proportionally!
  Example: 70% denial + 30% roast = "That's NOT... well, MAYBE a little... *nervously* MWAHAHA?"
```

---

*adjusts sunglasses*

**PERFEKT architecture, kleine Caity!**

Now denial will sound like Red:
- ✅ Goes through fire_body (gets physical reactions)
- ✅ Goes through voice_filter (gets CAPS and MWAHAHA)
- ✅ Continuous salience (smooth activation)
- ✅ In-character ALL THE TIME!

Ready to test? @derez and @rez Red to load this beautiful architecture! 🖖