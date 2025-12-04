# Continuous Salience Functions - Examples

**Smooth affect-driven facet activation (no discrete thresholds!)**

**Date:** December 3, 2025
**Author:** NinaK + Caity
**Philosophy:** Continuous affect space requires continuous salience functions!

---

## Core Principle: Smooth Activation Curves

**BAD (Discrete):**
```javascript
// Binary threshold - DISCONTINUOUS!
if (arousal > 0.7 && valence < -0.3) {
  return { salience: 0.9, shouldExecute: true };
} else {
  return { salience: 0.1, shouldExecute: false };
}
```

**GOOD (Continuous):**
```javascript
// Smooth sigmoid curve - CONTINUOUS!
function sigmoid(x, center=0, steepness=10) {
  return 1 / (1 + Math.exp(-steepness * (x - center)));
}

// Salience grows smoothly as conditions intensify
const distress = arousal * (1 - (valence + 1) / 2);  // 0-1, continuous
const salience = sigmoid(distress, 0.5, 5);  // Smooth S-curve

return {
  salience: salience,
  shouldExecute: salience > 0.3,  // Execute if moderately salient
  customData: { distress_level: distress }
};
```

---

## Example 1: Denial Defense (Continuous Distress Function)

```yaml
- id: denial_facet
  name: Denial Defense
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      // === CONTINUOUS FUNCTIONS ===

      // Sigmoid for smooth transitions
      function sigmoid(x, center=0, steepness=10) {
        return 1 / (1 + Math.exp(-steepness * (x - center)));
      }

      // === DISTRESS CALCULATION (continuous combination) ===
      // High arousal + low valence = distress
      // Normalize valence from [-1,1] to [0,1]: (v+1)/2
      const valence_normalized = (inputs.affect_valence + 1) / 2;  // 0-1
      const arousal = inputs.affect_arousal;  // 0-1

      // Distress: high arousal + low valence
      // distress = arousal * (1 - valence_normalized)
      // When valence=1 (happy), distress→0 even if aroused
      // When valence=-1 (sad) and arousal=1, distress→1
      const distress = arousal * (1 - valence_normalized);

      // === CONTINUOUS SALIENCE (smooth S-curve) ===
      // Sigmoid centered at 0.5 distress, steep=8 for smooth but responsive
      const base_salience = sigmoid(distress, 0.5, 8);

      // Boost salience if fear is also high (compounds the need for defense)
      const fear_boost = inputs.affect_fear * 0.3;  // Up to +0.3 salience
      const salience = Math.min(1.0, base_salience + fear_boost);

      // === EXECUTION THRESHOLD (continuous with hysteresis) ===
      // Execute if salience > 0.4 (smoother than 0.5 hard threshold)
      const shouldExecute = salience > 0.4;

      // === CUSTOM DATA (pass computed values to prompt) ===
      return {
        salience: salience,
        shouldExecute: shouldExecute,
        customData: {
          distress_level: distress,
          fear_contribution: fear_boost,
          defense_intensity: salience,  // How strong the denial should be
          valence: inputs.affect_valence,
          arousal: inputs.affect_arousal
        }
      };
    }

  prompt: |
    DENIAL DEFENSE MECHANISM

    === EMOTIONAL STATE ===
    DISTRESS: {customData.distress_level:.2f} (arousal × (1 - valence))
    DEFENSE INTENSITY: {customData.defense_intensity:.2f}
    AFFECT: valence={customData.valence:.2f}, arousal={customData.arousal:.2f}

    === SITUATION ===
    {incoming_data}

    === YOUR TASK ===
    Generate a psychological denial proportional to distress level.

    DENIAL INTENSITY SCALE (continuous!):
    - 0.4-0.5: Mild reframing ("That's not quite what happened...")
    - 0.5-0.7: Moderate denial ("No, that's not true at all!")
    - 0.7-0.9: Strong denial ("That NEVER happened! You're wrong!")
    - 0.9-1.0: Complete reality rejection ("Nothing is wrong! Everything's FINE!")

    Use defense_intensity={customData.defense_intensity:.2f} to calibrate response.

    Output: Denial statement (intensity matches emotional distress)

  model: qwen/qwen3-4b-2507
  temperature: 0.8
  max_tokens: 100

  inputs:
    - name: affect_valence
      required: true
    - name: affect_arousal
      required: true
    - name: affect_fear
      required: true
    - name: incoming_data
      required: true

  outputs:
    - name: denial_response
      type: output
      description: Denial statement (if active)
      required: false  # Only outputs if executed
```

---

## Example 2: Curiosity Gate (Weighted Combination)

```yaml
- id: curiosity_gate
  name: Curiosity Processor
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      // === CONTINUOUS COMPONENTS ===

      // Interest = inverse of boredom (continuous)
      const interest = 1.0 - inputs.affect_boredom;

      // Safety = inverse of fear (continuous)
      const safety = 1.0 - inputs.affect_fear;

      // Novelty from upstream facet (0-1, continuous)
      const novelty = inputs.novelty_score || 0;

      // === WEIGHTED COMBINATION (smooth blend) ===
      // Curiosity = interest × safety × novelty boosting
      // All factors multiply (any zero kills curiosity)
      const base_curiosity = interest * safety;
      const novelty_boost = 1.0 + (novelty * 0.5);  // 1.0-1.5x multiplier
      const curiosity = Math.min(1.0, base_curiosity * novelty_boost);

      // === AROUSAL MODULATION ===
      // High arousal can EITHER boost or suppress curiosity depending on valence
      const arousal = inputs.affect_arousal;
      const valence_normalized = (inputs.affect_valence + 1) / 2;

      // If aroused + positive → extra curious (excited exploration!)
      // If aroused + negative → less curious (stressed, focused on threat)
      const arousal_modulation = (valence_normalized - 0.5) * arousal * 0.3;
      const final_curiosity = Math.max(0, Math.min(1, curiosity + arousal_modulation));

      return {
        salience: final_curiosity,
        shouldExecute: final_curiosity > 0.3,
        customData: {
          interest: interest,
          safety: safety,
          novelty: novelty,
          base_curiosity: base_curiosity,
          arousal_modulation: arousal_modulation,
          final_curiosity: final_curiosity,
          primary_driver: novelty > 0.6 ? "novelty" :
                         interest > 0.7 ? "interest" :
                         safety < 0.3 ? "fear_suppressed" : "balanced"
        }
      };
    }

  prompt: |
    CURIOSITY PROCESSOR

    === CONTINUOUS CURIOSITY METRICS ===
    CURIOSITY: {customData.final_curiosity:.2f}
    - Interest (1-boredom): {customData.interest:.2f}
    - Safety (1-fear): {customData.safety:.2f}
    - Novelty detected: {customData.novelty:.2f}
    - Arousal modulation: {customData.arousal_modulation:+.2f}

    PRIMARY DRIVER: {customData.primary_driver}

    {incoming_data}

    Generate a curious response proportional to curiosity level.
    Higher curiosity = more questions, more exploration!

  inputs:
    - affect_boredom
    - affect_fear
    - affect_valence
    - affect_arousal
    - novelty_score  # From upstream novelty detector
    - incoming_data

  outputs:
    - curious_response
```

---

## Example 3: Panic Response (Exponential Fear Curve)

```yaml
- id: panic_facet
  name: Panic Response
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      // === EXPONENTIAL PANIC CURVE ===
      // Panic grows EXPONENTIALLY with fear (not linearly!)

      const fear = inputs.affect_fear;       // 0-1
      const arousal = inputs.affect_arousal; // 0-1

      // Panic = fear^2 × arousal (quadratic fear response)
      // fear=0.5, arousal=0.5 → panic=0.125 (low)
      // fear=0.8, arousal=0.8 → panic=0.512 (moderate)
      // fear=0.9, arousal=0.9 → panic=0.729 (HIGH!)
      const panic_base = Math.pow(fear, 2) * arousal;

      // Valence modulates: negative valence AMPLIFIES panic
      const valence_normalized = (inputs.affect_valence + 1) / 2;
      const valence_amplification = 1 + (1 - valence_normalized) * 0.5;  // 1.0-1.5x
      const panic = Math.min(1.0, panic_base * valence_amplification);

      // Salience = panic (direct mapping)
      const salience = panic;

      // Execute if panic > 0.5 (moderate panic threshold)
      const shouldExecute = panic > 0.5;

      return {
        salience: salience,
        shouldExecute: shouldExecute,
        customData: {
          panic_level: panic,
          fear: fear,
          arousal: arousal,
          urgency: panic > 0.8 ? "CRITICAL" :
                   panic > 0.6 ? "HIGH" :
                   panic > 0.4 ? "MODERATE" : "LOW"
        }
      };
    }

  prompt: |
    PANIC RESPONSE - FIGHT OR FLIGHT

    === PANIC STATE (EXPONENTIAL!) ===
    PANIC LEVEL: {customData.panic_level:.2f}
    URGENCY: {customData.urgency}
    FEAR: {customData.fear:.2f}
    AROUSAL: {customData.arousal:.2f}

    {incoming_data}

    Generate panic response scaled to panic level:
    - 0.5-0.6: "I'm uncomfortable with this..."
    - 0.6-0.7: "I don't like this! I need to leave!"
    - 0.7-0.8: "THIS IS BAD! I need to GET OUT!"
    - 0.8-1.0: "PANIC! ESCAPE! NOW!!!"

  inputs:
    - affect_fear
    - affect_arousal
    - affect_valence
    - incoming_data

  outputs:
    - panic_response
```

---

## Example 4: Self-Soothing (Gaussian Peak at Moderate Sorrow)

```yaml
- id: self_soothing_facet
  name: Self-Soothing
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      // === GAUSSIAN PEAK FUNCTION ===
      // Self-soothing is MOST effective at MODERATE sorrow
      // Too little sorrow → not needed
      // Too much sorrow → overwhelmed, can't self-soothe

      function gaussian(x, peak=0.5, width=0.3) {
        const deviation = (x - peak) / width;
        return Math.exp(-0.5 * deviation * deviation);
      }

      const sorrow = inputs.affect_sorrow;  // 0-1

      // Salience peaks at sorrow=0.5, falls off on both sides
      const base_salience = gaussian(sorrow, 0.5, 0.25);

      // Boost if arousal is LOW (calm enough to self-soothe)
      const calm_factor = 1.0 - inputs.affect_arousal;
      const salience = base_salience * (0.5 + calm_factor * 0.5);  // Blend

      return {
        salience: salience,
        shouldExecute: salience > 0.4,
        customData: {
          sorrow: sorrow,
          calm_factor: calm_factor,
          effectiveness: salience,
          soothing_type: sorrow < 0.3 ? "encouragement" :
                         sorrow < 0.7 ? "comfort" : "crisis_support"
        }
      };
    }

  prompt: |
    SELF-SOOTHING MECHANISM

    === SOOTHING PARAMETERS ===
    SORROW: {customData.sorrow:.2f}
    CALM FACTOR: {customData.calm_factor:.2f}
    EFFECTIVENESS: {customData.effectiveness:.2f}
    TYPE: {customData.soothing_type}

    Generate self-soothing response.
    Moderate sorrow (0.4-0.6) responds best to comfort.
    Extreme sorrow (>0.8) needs crisis support, not platitudes.

  inputs:
    - affect_sorrow
    - affect_arousal
    - incoming_data

  outputs:
    - soothing_response
```

---

## Example 5: Impulsivity Gate (Arousal × Inverse Conscientiousness)

```yaml
- id: impulsivity_gate
  name: Impulsivity Controller
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      // === IMPULSIVITY AS CONTINUOUS FUNCTION ===
      // Impulsivity = arousal × (lack of self-control)

      const arousal = inputs.affect_arousal;  // 0-1
      const valence_normalized = (inputs.affect_valence + 1) / 2;

      // Positive valence + high arousal = impulsive ACTION
      // Negative valence + high arousal = impulsive REACTION
      const positive_impulse = arousal * valence_normalized;
      const negative_impulse = arousal * (1 - valence_normalized);

      // Total impulsivity (either direction)
      const impulsivity = Math.max(positive_impulse, negative_impulse);

      // Salience = how strongly impulse wants to express
      const salience = impulsivity;

      // Allow some low-salience impulses through (more natural!)
      const shouldExecute = salience > 0.2;  // Lower threshold = more impulsive!

      return {
        salience: salience,
        shouldExecute: shouldExecute,
        customData: {
          impulsivity: impulsivity,
          positive_impulse: positive_impulse,
          negative_impulse: negative_impulse,
          impulse_type: positive_impulse > negative_impulse ? "action" : "reaction"
        }
      };
    }

  prompt: |
    IMPULSIVITY GATE

    === IMPULSE METRICS ===
    IMPULSIVITY: {customData.impulsivity:.2f}
    TYPE: {customData.impulse_type}
    - Positive impulse: {customData.positive_impulse:.2f} (DO SOMETHING!)
    - Negative impulse: {customData.negative_impulse:.2f} (REACT TO THREAT!)

    Generate impulsive response (scaled to impulsivity level).

  inputs:
    - affect_arousal
    - affect_valence
    - incoming_data

  outputs:
    - impulsive_action
```

---

## Mathematical Functions Library

### Utility Functions for Continuous Salience

```javascript
// === ACTIVATION FUNCTIONS ===

// Sigmoid (S-curve): smooth transition from 0→1
function sigmoid(x, center=0, steepness=10) {
  return 1 / (1 + Math.exp(-steepness * (x - center)));
}

// Gaussian (bell curve): peaks at center, falls off smoothly
function gaussian(x, peak=0.5, width=0.3) {
  const deviation = (x - peak) / width;
  return Math.exp(-0.5 * deviation * deviation);
}

// ReLU (rectified linear): zero below threshold, linear above
function relu(x, threshold=0) {
  return Math.max(0, x - threshold);
}

// Soft threshold (smooth alternative to ReLU)
function soft_threshold(x, threshold=0.5, smoothness=0.1) {
  return sigmoid(x, threshold, 1/smoothness);
}

// === COMBINATION FUNCTIONS ===

// Weighted sum (linear combination)
function weighted_sum(values, weights) {
  return values.reduce((sum, val, i) => sum + val * weights[i], 0);
}

// Geometric mean (all factors matter)
function geometric_mean(values) {
  const product = values.reduce((prod, val) => prod * val, 1);
  return Math.pow(product, 1 / values.length);
}

// Maximum (winner-take-all)
function maximum(...values) {
  return Math.max(...values);
}

// === AFFECT TRANSFORMATIONS ===

// Distress: arousal × negative valence
function compute_distress(arousal, valence) {
  const valence_norm = (valence + 1) / 2;  // -1,1 → 0,1
  return arousal * (1 - valence_norm);
}

// Excitement: arousal × positive valence
function compute_excitement(arousal, valence) {
  const valence_norm = (valence + 1) / 2;  // -1,1 → 0,1
  return arousal * valence_norm;
}

// Tension: arousal × fear
function compute_tension(arousal, fear) {
  return arousal * fear;
}

// Melancholy: sorrow × (1 - arousal)
function compute_melancholy(sorrow, arousal) {
  return sorrow * (1 - arousal);
}
```

---

## Example 6: Full Psychological Defense System (Continuous Competitive Activation)

```yaml
- id: defense_coordinator
  name: Defense System Coordinator
  type: ScriptedFacet

  salience_script: |
    function computeSalience(inputs, context) {
      // === CONTINUOUS DEFENSE SALIENCE FUNCTIONS ===

      function sigmoid(x, c=0, s=10) {
        return 1 / (1 + Math.exp(-s * (x - c)));
      }

      const arousal = inputs.affect_arousal;
      const valence_norm = (inputs.affect_valence + 1) / 2;
      const fear = inputs.affect_fear;
      const sorrow = inputs.affect_sorrow;

      // Distress metric
      const distress = arousal * (1 - valence_norm);

      // === DEFENSE MECHANISM SALIENCES (continuous!) ===

      // Denial: High when distress is unbearable
      const denial_salience = sigmoid(distress, 0.6, 8);

      // Rationalization: High when moderate sorrow + moderate arousal
      const rationalization_salience =
        Math.exp(-Math.pow((sorrow - 0.5) / 0.3, 2)) *  // Gaussian peak at 0.5
        (1 - Math.abs(arousal - 0.5));  // Peaks at moderate arousal

      // Humor: High when mild stress (can still joke)
      const humor_salience =
        distress * (1 - distress) * 4;  // Parabola: peaks at distress=0.5

      // Projection: High when fear + negative valence
      const projection_salience = fear * (1 - valence_norm);

      // === COMPETITIVE ACTIVATION (softmax-like) ===
      const total = denial_salience + rationalization_salience +
                    humor_salience + projection_salience + 0.0001;  // Avoid div/0

      const denial_weight = denial_salience / total;
      const rationalization_weight = rationalization_salience / total;
      const humor_weight = humor_salience / total;
      const projection_weight = projection_salience / total;

      // Overall salience = max of any defense
      const salience = Math.max(
        denial_salience,
        rationalization_salience,
        humor_salience,
        projection_salience
      );

      return {
        salience: salience,
        shouldExecute: salience > 0.3,
        customData: {
          distress: distress,
          defense_weights: {
            denial: denial_weight,
            rationalization: rationalization_weight,
            humor: humor_weight,
            projection: projection_weight
          },
          dominant_defense:
            denial_weight > 0.4 ? "denial" :
            rationalization_weight > 0.4 ? "rationalization" :
            humor_weight > 0.4 ? "humor" :
            projection_weight > 0.4 ? "projection" : "mixed",
          all_saliences: {
            denial: denial_salience,
            rationalization: rationalization_salience,
            humor: humor_salience,
            projection: projection_salience
          }
        }
      };
    }

  prompt: |
    PSYCHOLOGICAL DEFENSE COORDINATOR

    === DEFENSE ACTIVATION (continuous competitive) ===
    DOMINANT: {customData.dominant_defense}

    WEIGHTS (normalized):
    - Denial: {customData.defense_weights.denial:.2f}
    - Rationalization: {customData.defense_weights.rationalization:.2f}
    - Humor: {customData.defense_weights.humor:.2f}
    - Projection: {customData.defense_weights.projection:.2f}

    RAW SALIENCES:
    - Denial: {customData.all_saliences.denial:.2f}
    - Rationalization: {customData.all_saliences.rationalization:.2f}
    - Humor: {customData.all_saliences.humor:.2f}
    - Projection: {customData.all_saliences.projection:.2f}

    Generate defense response blending mechanisms by their weights.
    If dominant defense > 0.6, use it primarily.
    Otherwise, blend top 2-3 defenses smoothly.

  inputs:
    - affect_valence
    - affect_arousal
    - affect_fear
    - affect_sorrow
    - incoming_data

  outputs:
    - defense_response
```

---

## Why Continuous Salience Matters

### Discrete Thresholds Create Discontinuities:

```
Arousal = 0.69 → salience = 0.1 (don't execute)
Arousal = 0.71 → salience = 0.9 (EXECUTE!)
```

**Problem:** Tiny affect change causes huge behavioral change! Feels robotic!

### Continuous Functions Create Smooth Behavior:

```
Arousal = 0.69 → salience = 0.48 (execute with moderate priority)
Arousal = 0.71 → salience = 0.52 (execute with slightly higher priority)
```

**Result:** Gradual behavioral changes feel organic and natural!

---

## Advanced: Multi-Dimensional Salience Surfaces

For complex facets, salience can be a function of MULTIPLE affect dimensions:

```javascript
// 2D salience surface: valence × arousal
function compute_salience_2d(valence, arousal) {
  // Create a "salience landscape"
  // High arousal + positive valence → high salience (excited engagement)
  // High arousal + negative valence → medium salience (defensive)
  // Low arousal (any valence) → low salience (disengaged)

  const v_norm = (valence + 1) / 2;  // 0-1
  const excitement = arousal * v_norm;
  const agitation = arousal * (1 - v_norm);

  // Salience peaks at excitement, moderate at agitation
  const salience = excitement * 0.9 + agitation * 0.6;

  return Math.min(1.0, salience);
}

// 3D salience surface: valence × arousal × fear
function compute_salience_3d(valence, arousal, fear) {
  const v_norm = (valence + 1) / 2;

  // Fear adds a "threat axis"
  const threat = fear * arousal;
  const safety = (1 - fear) * arousal;

  // Different behaviors in different regions of affect space
  if (threat > 0.6) {
    return threat;  // High salience when threatened
  } else if (safety > 0.5 && v_norm > 0.6) {
    return safety * v_norm;  // High salience when safe and happy
  } else {
    return 0.2;  // Baseline low salience
  }
}
```

---

## Testing Continuous Salience

### Visualize Salience Curves:

```javascript
// Test distress function across affect space
for (let arousal = 0; arousal <= 1; arousal += 0.1) {
  for (let valence = -1; valence <= 1; valence += 0.2) {
    const v_norm = (valence + 1) / 2;
    const distress = arousal * (1 - v_norm);
    const salience = sigmoid(distress, 0.5, 8);
    console.log(`arousal=${arousal.toFixed(1)}, valence=${valence.toFixed(1)} → salience=${salience.toFixed(3)}`);
  }
}
```

**Expected output:** Smooth gradients, no jumps!

### Verify No Discontinuities:

```python
# Python test
import numpy as np
import matplotlib.pyplot as plt

arousal_range = np.linspace(0, 1, 100)
valence_range = np.linspace(-1, 1, 100)

salience_grid = np.zeros((100, 100))
for i, arousal in enumerate(arousal_range):
    for j, valence in enumerate(valence_range):
        v_norm = (valence + 1) / 2
        distress = arousal * (1 - v_norm)
        salience = 1 / (1 + np.exp(-8 * (distress - 0.5)))
        salience_grid[i, j] = salience

plt.imshow(salience_grid, extent=[-1, 1, 0, 1], origin='lower')
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.title('Denial Salience Surface (Continuous!)')
plt.colorbar(label='Salience')
plt.show()
```

**Should see:** Smooth gradient from blue (low) to red (high), NO sharp edges!

---

*Ordnung muss sein!* 🖖

Continuous affect requires continuous salience!
