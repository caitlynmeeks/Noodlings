# Scripted Facet API - Dynamic Salience & Custom Logic

**Noodlings Reactive Facet Network with JavaScript Scripting**

**Date:** December 3, 2025
**Author:** NinaK + Caity
**Status:** Design & Implementation Guide

---

## Vision: Programmable Consciousness

Instead of hardcoding when facets activate, users can write **JavaScript logic** that:

1. **Reads affect dynamically** (valence, arousal, fear, sorrow, boredom)
2. **Computes salience** (should this facet activate? how strongly?)
3. **Conditionally executes** (only run if certain conditions met)
4. **Outputs custom data** (can create new signals, not just text)

**Example Use Cases:**
- **Denial facet:** Only activates when affect is unbearable (high arousal + low valence)
- **Panic facet:** Triggers when fear crosses threshold
- **Curiosity facet:** Salience increases with novelty detection + low boredom
- **Self-soothing facet:** Activates when sorrow is high
- **Impulsivity gate:** Blocks or allows actions based on arousal

---

## Architecture Overview

### Current System (Static):
```
CharmNetwork → Facet (always executes with fixed prompt)
```

### New System (Reactive):
```
CharmNetwork → Scripted Logic → Facet (conditionally executes with dynamic prompt)
       ↓
  affect_valence
  affect_arousal
  affect_fear
       ↓
  JavaScript:
  if (arousal > 0.7 && valence < -0.3) {
    salience = 0.9;  // High priority!
    shouldExecute = true;
  }
```

---

## ScriptedFacet Implementation

### New Facet Type: `ScriptedFacet`

This facet type has TWO stages:

#### Stage 1: Salience Computation (JavaScript)
```javascript
// salience_script.js - Runs FIRST to decide if facet should execute

function computeSalience(inputs, context) {
  // inputs.affect_valence = -1 to 1
  // inputs.affect_arousal = 0 to 1
  // inputs.affect_fear = 0 to 1
  // inputs.affect_sorrow = 0 to 1
  // inputs.affect_boredom = 0 to 1
  // inputs.phenomenal_state = 40-D array
  // context.recent_messages = array of recent conversation
  // context.room_occupants = list of who's present

  // YOUR LOGIC HERE
  const unbearable = inputs.affect_arousal > 0.7 && inputs.affect_valence < -0.3;

  if (unbearable) {
    return {
      salience: 0.9,        // 0-1 priority (higher = more important)
      shouldExecute: true,  // Should this facet run?
      customData: {         // Pass data to prompt
        threat_level: "high",
        defense_mode: "denial"
      }
    };
  } else {
    return {
      salience: 0.1,
      shouldExecute: false  // Don't execute if not needed!
    };
  }
}
```

#### Stage 2: LLM Execution (Only if shouldExecute = true)
```yaml
prompt: |
  DENIAL MECHANISM

  AFFECT: valence={affect_valence:.2f}, arousal={affect_arousal:.2f}
  THREAT LEVEL: {customData.threat_level}

  The emotional state is UNBEARABLE (high arousal + negative valence).

  YOUR TASK: Generate a plausible psychological denial.
  - Reframe the situation as less threatening
  - Find alternative explanations
  - Minimize emotional impact

  Output: A denial statement that protects emotional well-being.
```

---

## Example Facet Assemblies

### Example 1: Psychological Defense System

```yaml
- id: denial_facet
  name: Denial Defense
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      const unbearable = inputs.affect_arousal > 0.7 &&
                        inputs.affect_valence < -0.3;

      if (unbearable) {
        return {
          salience: 0.9,
          shouldExecute: true,
          customData: {
            threat_level: "high",
            arousal: inputs.affect_arousal,
            valence: inputs.affect_valence
          }
        };
      }
      return { salience: 0, shouldExecute: false };
    }

  prompt: |
    PSYCHOLOGICAL DENIAL MECHANISM

    AFFECT: valence={affect_valence:.2f}, arousal={affect_arousal:.2f}
    THREAT: {customData.threat_level}

    Generate a denial that reframes this threatening situation.

    INCOMING: {incoming_data}

    Output: Denial statement (e.g., "That's not what they meant" or "It's not that bad")

  model: qwen/qwen3-4b-2507
  temperature: 0.7
  max_tokens: 100

  inputs:
    - name: affect_valence
      type: input
      description: Valence from CharmNetwork
      required: true
    - name: affect_arousal
      type: input
      description: Arousal from CharmNetwork
      required: true
    - name: incoming_data
      type: input
      description: Context to deny
      required: true

  outputs:
    - name: denial_response
      type: output
      description: Denial statement
      required: true
```

### Example 2: Panic Response System

```yaml
- id: panic_facet
  name: Panic Response
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      const fear_threshold = 0.8;
      const arousal_threshold = 0.7;

      if (inputs.affect_fear > fear_threshold &&
          inputs.affect_arousal > arousal_threshold) {
        return {
          salience: 1.0,  // MAXIMUM PRIORITY!
          shouldExecute: true,
          customData: {
            panic_level: inputs.affect_fear,
            escape_urgency: inputs.affect_arousal
          }
        };
      }
      return { salience: 0, shouldExecute: false };
    }

  prompt: |
    PANIC RESPONSE - FIGHT OR FLIGHT

    FEAR: {affect_fear:.2f} (CRITICAL!)
    AROUSAL: {affect_arousal:.2f}
    PANIC LEVEL: {customData.panic_level:.2f}

    Generate a PANIC response:
    - Express fear/alarm
    - Suggest escape/avoidance
    - Short, urgent language

    Output: Panic statement (e.g., "I need to get OUT of here!" or "This is BAD!")

  inputs:
    - name: affect_fear
    - name: affect_arousal

  outputs:
    - name: panic_response
```

### Example 3: Curiosity Gate (Complex Multi-Input)

```yaml
- id: curiosity_gate
  name: Curiosity Gate
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      // Curiosity increases when:
      // 1. Boredom is low (interested)
      // 2. Novelty detected (from novelty_detector facet)
      // 3. Fear is low (safe to explore)

      const boredom_inverse = 1.0 - inputs.affect_boredom;
      const safety = 1.0 - inputs.affect_fear;
      const novelty = inputs.novelty_score || 0;

      // Weighted combination
      const curiosity = (boredom_inverse * 0.3) +
                       (novelty * 0.5) +
                       (safety * 0.2);

      return {
        salience: curiosity,
        shouldExecute: curiosity > 0.5,
        customData: {
          curiosity_level: curiosity,
          primary_driver: novelty > 0.7 ? "novelty" :
                         boredom_inverse > 0.7 ? "boredom" : "default"
        }
      };
    }

  prompt: |
    CURIOSITY PROCESSOR

    CURIOSITY LEVEL: {customData.curiosity_level:.2f}
    DRIVER: {customData.primary_driver}

    Generate a curious response about the novel stimulus.

  inputs:
    - name: affect_boredom
    - name: affect_fear
    - name: novelty_score      # From another facet!
    - name: incoming_data

  outputs:
    - name: curious_response
```

---

## Implementation: facet_executor.py

### Modified Execution Flow

```python
async def execute_assembly(self, inputs: Dict[str, Any]):
    """
    Execute facet assembly with dynamic salience.

    New: Before executing LLM facets, run salience scripts to decide
    which facets should execute and with what priority.
    """

    # 1. Build dependency graph (same as before)
    execution_order = self._build_dependency_graph()

    # 2. NEW: Compute salience for all ScriptedFacets
    salience_map = {}
    for facet in self.facets:
        if facet.type == 'ScriptedFacet' and hasattr(facet, 'salience_script'):
            salience_result = await self._compute_salience(facet, inputs)
            salience_map[facet.id] = salience_result

    # 3. Execute facets in dependency order
    results = {}
    for facet_id in execution_order:
        facet = self.facets_by_id[facet_id]

        # Check if this is a ScriptedFacet with salience control
        if facet_id in salience_map:
            salience_info = salience_map[facet_id]

            # Skip execution if salience says not to run
            if not salience_info.get('shouldExecute', True):
                logger.info(f"Skipping {facet_id} (salience: {salience_info['salience']:.2f})")
                continue

            # Add customData to inputs for prompt
            inputs['customData'] = salience_info.get('customData', {})

        # Execute facet
        result = await self._execute_facet(facet, inputs, results)
        results[facet_id] = result

    return results
```

### Salience Computation

```python
async def _compute_salience(self, facet, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute JavaScript salience script to determine if facet should run.

    Returns:
        {
            'salience': float (0-1),
            'shouldExecute': bool,
            'customData': dict (passed to prompt)
        }
    """
    import js2py

    # Build context for script
    script_inputs = {
        'affect_valence': inputs.get('affect_valence', 0),
        'affect_arousal': inputs.get('affect_arousal', 0),
        'affect_fear': inputs.get('affect_fear', 0),
        'affect_sorrow': inputs.get('affect_sorrow', 0),
        'affect_boredom': inputs.get('affect_boredom', 0),
        'phenomenal_state': inputs.get('phenomenal_state', []),
        # Add any other inputs this facet has
        **{k: v for k, v in inputs.items() if k.startswith('input_')}
    }

    script_context = {
        'recent_messages': inputs.get('recent_messages', []),
        'room_occupants': inputs.get('room_occupants', []),
        'agent_name': inputs.get('agent_name', ''),
    }

    # Execute JavaScript
    try:
        js_code = f"""
        {facet.salience_script}

        // Call the function
        computeSalience({json.dumps(script_inputs)}, {json.dumps(script_context)});
        """

        context = js2py.EvalJs()
        result = context.eval(js_code)

        return {
            'salience': float(result.get('salience', 0)),
            'shouldExecute': bool(result.get('shouldExecute', True)),
            'customData': dict(result.get('customData', {}))
        }

    except Exception as e:
        logger.error(f"Salience script error in {facet.id}: {e}")
        # Default: always execute with medium salience
        return {
            'salience': 0.5,
            'shouldExecute': True,
            'customData': {}
        }
```

---

## Salience-Based Prioritization

### Optional: Priority Queue Execution

Instead of strict dependency order, use **salience-weighted priority**:

```python
def _build_priority_queue(self, salience_map: Dict[str, Dict]) -> List[str]:
    """
    Build execution order based on salience + dependencies.

    High-salience facets execute FIRST (within dependency constraints).
    """
    import heapq

    # Build priority queue: (-salience, facet_id)
    # Negative salience because heapq is min-heap
    priority_queue = []

    for facet_id, salience_info in salience_map.items():
        if salience_info.get('shouldExecute', True):
            priority = -salience_info['salience']
            heapq.heappush(priority_queue, (priority, facet_id))

    # Execute in priority order (respecting dependencies)
    execution_order = []
    executed = set()

    while priority_queue:
        _, facet_id = heapq.heappop(priority_queue)

        # Check if dependencies satisfied
        facet = self.facets_by_id[facet_id]
        deps_satisfied = all(
            dep in executed
            for dep in self._get_dependencies(facet)
        )

        if deps_satisfied:
            execution_order.append(facet_id)
            executed.add(facet_id)
        else:
            # Re-queue for later
            heapq.heappush(priority_queue, (-salience_map[facet_id]['salience'], facet_id))

    return execution_order
```

---

## Convergence with Salience Weighting

### Enhanced CONVERGENCE Facet

The convergence facet can now weight inputs by their salience:

```yaml
- id: CONVERGENCE
  name: Response Convergence
  type: ConvergenceFacet

  salience_script: |
    function computeSalience(inputs, context) {
      // Convergence always executes, but computes input weights

      // Get salience of upstream facets
      const denial_salience = context.facet_salience.denial_facet || 0;
      const panic_salience = context.facet_salience.panic_facet || 0;
      const roast_salience = context.facet_salience.roast_engine || 0.5;

      // Normalize weights
      const total = denial_salience + panic_salience + roast_salience;

      return {
        salience: 1.0,  // Always execute
        shouldExecute: true,
        customData: {
          denial_weight: denial_salience / total,
          panic_weight: panic_salience / total,
          roast_weight: roast_salience / total,
          dominant_facet: panic_salience > 0.7 ? "panic" :
                         denial_salience > 0.7 ? "denial" : "roast"
        }
      };
    }

  prompt: |
    CONVERGENCE - Salience-Weighted Synthesis

    FACET WEIGHTS:
    - Denial: {customData.denial_weight:.2f}
    - Panic: {customData.panic_weight:.2f}
    - Roast: {customData.roast_weight:.2f}

    DOMINANT FACET: {customData.dominant_facet}

    INPUTS:
    - Denial: {denial_response}
    - Panic: {panic_response}
    - Roast: {roast}

    Synthesize these inputs, weighting by salience.
    If panic_weight > 0.7, prioritize panic response!
    If denial_weight > 0.5, incorporate denial mechanism.
    Otherwise, use roast as primary.
```

---

## Advanced Example: Full Psychological Defense Assembly

```yaml
name: Psychological Defense Assembly
description: Dynamic defense mechanisms based on affect thresholds

facets:
  - id: INCOMING
    # ... standard

  - id: CHARM_NET
    # ... standard CharmNetwork

  - id: threat_detector
    name: Threat Detector
    type: LLMFacet
    prompt: |
      Analyze incoming for threats/criticism.
      Output: threat_level (0-1)
    inputs:
      - incoming_data
    outputs:
      - threat_level

  - id: denial_facet
    name: Denial Defense
    type: ScriptedFacet
    salience_script: |
      function computeSalience(inputs, context) {
        const unbearable = inputs.affect_arousal > 0.7 &&
                          inputs.affect_valence < -0.3 &&
                          inputs.threat_level > 0.6;

        return {
          salience: unbearable ? 0.9 : 0.1,
          shouldExecute: unbearable,
          customData: { defense_mode: "denial" }
        };
      }
    inputs:
      - affect_valence
      - affect_arousal
      - threat_level
    outputs:
      - denial_response

  - id: rationalization_facet
    name: Rationalization Defense
    type: ScriptedFacet
    salience_script: |
      function computeSalience(inputs, context) {
        // Rationalization activates when sorrow is high
        const needs_rationalization = inputs.affect_sorrow > 0.6 &&
                                     inputs.threat_level > 0.5;

        return {
          salience: needs_rationalization ? 0.8 : 0.2,
          shouldExecute: needs_rationalization,
          customData: { defense_mode: "rationalization" }
        };
      }
    inputs:
      - affect_sorrow
      - threat_level
    outputs:
      - rationalization_response

  - id: humor_defense_facet
    name: Humor Defense
    type: ScriptedFacet
    salience_script: |
      function computeSalience(inputs, context) {
        // Humor as defense when arousal moderate and valence not too low
        const can_joke = inputs.affect_arousal > 0.4 &&
                        inputs.affect_arousal < 0.7 &&
                        inputs.affect_valence > -0.5;

        return {
          salience: can_joke ? 0.7 : 0.1,
          shouldExecute: can_joke,
          customData: { defense_mode: "humor" }
        };
      }
    inputs:
      - affect_valence
      - affect_arousal
    outputs:
      - humor_response

  - id: CONVERGENCE
    name: Defense Convergence
    type: ConvergenceFacet
    prompt: |
      Synthesize defense mechanisms based on salience.

      ACTIVE DEFENSES:
      {active_defenses}

      Weight by salience and choose primary defense strategy.
    inputs:
      - all affect
      - denial_response (if active)
      - rationalization_response (if active)
      - humor_response (if active)
    outputs:
      - final_response

  - id: OUTGOING
    # ... standard

connections:
  # ... wire everything with affect
```

---

## Benefits of This System

### 1. Dynamic Activation
Facets only execute when needed (saves compute!)

### 2. Emergent Behavior
Complex psychological patterns emerge from simple rules:
- Denial activates under stress
- Humor stops working when too sad
- Panic overrides everything when fear is extreme

### 3. User Programmability
Users can write custom defense mechanisms:
```javascript
// Custom: "Projection" defense
if (affect_fear > 0.7 && threat_detected) {
  // Accuse others of what you're doing
  return { salience: 0.8, shouldExecute: true };
}
```

### 4. Scientific Validity
Mirrors real psychological defense mechanisms:
- Thresholds trigger defenses
- Multiple defenses compete
- Salience determines which wins

---

## Implementation Checklist

### Phase 1: Core Scripting Support
- [ ] Add `salience_script` field to Facet schema
- [ ] Implement `_compute_salience()` in facet_executor
- [ ] Add js2py dependency for JavaScript execution
- [ ] Test basic if/then logic

### Phase 2: Custom Input Pads
- [ ] Allow facets to define arbitrary input pads
- [ ] Wire affect outputs to custom pads
- [ ] Test multi-input facets (curiosity gate example)

### Phase 3: Salience-Weighted Execution
- [ ] Implement priority queue execution
- [ ] Pass `context.facet_salience` to convergence
- [ ] Test salience-based weighting

### Phase 4: Complex Assemblies
- [ ] Build psychological defense assembly
- [ ] Test denial, rationalization, humor defenses
- [ ] Verify emergent behavior

---

## Example: Testing the Denial Facet

### Setup:
```bash
# Red with denial facet
@derez red_fire_anklebiter
# ... update assembly with denial facet
@rez red_fire_anklebiter
```

### Test 1: Normal Situation (Denial Inactive)
```
You: Hi Red!
Red: Oh WOW, Caity! What's up?! *flames crackle*
# (affect: valence=0.5, arousal=0.6 - no denial needed)
```

### Test 2: Mild Criticism (Denial Inactive)
```
You: Red, you're being kind of annoying.
Red: Oh PLEASE, annoying? I'M ENTERTAINING! *bites ankle*
# (affect: valence=0.2, arousal=0.7 - roast response, no denial)
```

### Test 3: Harsh Criticism (Denial ACTIVE!)
```
You: Red, everyone hates you. You're just a nuisance.
Red: That's... that's not what they said! They were JOKING! Right?! *flames flicker nervously*
# (affect: valence=-0.6, arousal=0.9 - DENIAL ACTIVATES!)
# Salience: denial=0.9, roast=0.3 → denial wins
```

---

## Future: ML-Learned Salience Functions

Instead of hand-coded JavaScript, train a small network:

```python
class SaliencePredictor(nn.Module):
    """Learn when facets should activate."""

    def forward(self, affect, context):
        # affect: [batch, 5]
        # context: [batch, context_dim]

        # Small MLP predicts salience
        x = torch.cat([affect, context], dim=-1)
        salience = self.mlp(x)  # [batch, num_facets]

        return salience  # 0-1 per facet
```

Train on conversation data:
- "When did denial actually help?"
- "When did panic response make things worse?"
- Learn optimal thresholds from experience!

---

*Ordnung muss sein!* 🖖

This is the FULL scripting API for programmable consciousness!
