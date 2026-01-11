# Facial Expression System

> FACS → VRM: Making Noodlings Expressive

## Overview

The Facial Expression System bridges the cognitive affect model to visible facial expressions on VRM avatars. When a noodling feels something, their face shows it.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Affect Output  │────▶│   FACS Mapper   │────▶│  VRM Blendshape │
│  (PAD + S + B)  │     │  (Action Units) │     │     Driver      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
   From facets           Psychological            Avatar mesh
   (CharmNetwork,        to muscular              deformation
   NeuralCanvas)         mapping
```

## The Affect Model

NoodleStudio's affect system outputs a 5-dimensional vector:

| Dimension | Range | Description |
|-----------|-------|-------------|
| **Valence** | -1 to +1 | Pleasure/displeasure (PAD) |
| **Arousal** | 0 to 1 | Activation/energy level (PAD) |
| **Dominance** | 0 to 1 | Control/agency (PAD) |
| **Sorrow** | 0 to 1 | Melancholy, grief, loss |
| **Boredom** | 0 to 1 | Disengagement, tedium |

This maps to the Mehrabian-Russell PAD model plus Caity's extensions for richer emotional texture.

## FACS: Facial Action Coding System

Paul Ekman's FACS describes facial expressions as combinations of **Action Units (AUs)** - individual muscle movements.

### Core Action Units for Expression

| AU | Name | Muscles | Expression Role |
|----|------|---------|-----------------|
| AU1 | Inner Brow Raiser | Frontalis (medial) | Sadness, worry |
| AU2 | Outer Brow Raiser | Frontalis (lateral) | Surprise |
| AU4 | Brow Lowerer | Corrugator, Depressor supercilii | Anger, concentration |
| AU5 | Upper Lid Raiser | Levator palpebrae | Surprise, fear |
| AU6 | Cheek Raiser | Orbicularis oculi (orbital) | Genuine smile (Duchenne) |
| AU7 | Lid Tightener | Orbicularis oculi (palpebral) | Anger, concentration |
| AU9 | Nose Wrinkler | Levator labii superioris | Disgust |
| AU10 | Upper Lip Raiser | Levator labii superioris | Disgust |
| AU12 | Lip Corner Puller | Zygomaticus major | Smile, happiness |
| AU15 | Lip Corner Depressor | Depressor anguli oris | Sadness, frown |
| AU17 | Chin Raiser | Mentalis | Doubt, sadness |
| AU20 | Lip Stretcher | Risorius | Fear, tension |
| AU23 | Lip Tightener | Orbicularis oris | Anger |
| AU24 | Lip Pressor | Orbicularis oris | Tension, suppression |
| AU25 | Lips Part | Depressor labii | Surprise, speech |
| AU26 | Jaw Drop | Masseter (relaxed) | Surprise, shock |
| AU43 | Eyes Closed | Orbicularis oculi | Sleep, bliss, pain |

### Emotion → AU Recipes

Classic emotional expressions as AU combinations:

```python
EMOTION_AU_RECIPES = {
    'happiness': {
        'AU6': 0.8,   # Cheek raiser (Duchenne marker)
        'AU12': 1.0,  # Lip corner puller
    },
    'sadness': {
        'AU1': 0.8,   # Inner brow raiser
        'AU4': 0.3,   # Slight brow lowerer
        'AU15': 0.7,  # Lip corner depressor
        'AU17': 0.4,  # Chin raiser
    },
    'anger': {
        'AU4': 1.0,   # Brow lowerer
        'AU5': 0.3,   # Upper lid raiser (glare)
        'AU7': 0.8,   # Lid tightener
        'AU23': 0.6,  # Lip tightener
    },
    'fear': {
        'AU1': 0.9,   # Inner brow raiser
        'AU2': 0.7,   # Outer brow raiser
        'AU4': 0.3,   # Slight brow lowerer
        'AU5': 0.8,   # Upper lid raiser (wide eyes)
        'AU20': 0.6,  # Lip stretcher
        'AU25': 0.4,  # Lips part
    },
    'surprise': {
        'AU1': 0.7,   # Inner brow raiser
        'AU2': 0.9,   # Outer brow raiser
        'AU5': 0.9,   # Upper lid raiser
        'AU25': 0.6,  # Lips part
        'AU26': 0.7,  # Jaw drop
    },
    'disgust': {
        'AU9': 0.8,   # Nose wrinkler
        'AU10': 0.6,  # Upper lip raiser
        'AU4': 0.3,   # Slight brow lowerer
    },
    'contempt': {
        'AU12': 0.4,  # Asymmetric lip corner (one side)
        'AU14': 0.5,  # Dimpler (one side)
    },
    'concentration': {
        'AU4': 0.5,   # Brow lowerer
        'AU7': 0.4,   # Lid tightener
        'AU24': 0.3,  # Lip pressor
    },
    'boredom': {
        'AU43': 0.3,  # Partial eye close (heavy lids)
        'AU15': 0.2,  # Slight lip corner depressor
        'AU4': 0.1,   # Minimal brow lowerer
    },
}
```

## VRM Blendshapes

VRM (Virtual Reality Model) is the standard avatar format. It defines preset blendshapes:

### VRM Standard Expression Blendshapes

| Blendshape | Description |
|------------|-------------|
| `Fcl_ALL_Neutral` | Default/rest face |
| `Fcl_ALL_Joy` | Happy, smiling |
| `Fcl_ALL_Angry` | Angry, frowning |
| `Fcl_ALL_Sorrow` | Sad, downcast |
| `Fcl_ALL_Fun` | Playful, amused |
| `Fcl_ALL_Surprised` | Surprised, shocked |

### VRM Phoneme/Viseme Blendshapes

| Blendshape | Phoneme |
|------------|---------|
| `Fcl_MTH_A` | "ah" |
| `Fcl_MTH_I` | "ee" |
| `Fcl_MTH_U` | "oo" |
| `Fcl_MTH_E` | "eh" |
| `Fcl_MTH_O` | "oh" |

### VRM Eye Blendshapes

| Blendshape | Description |
|------------|-------------|
| `Fcl_EYE_Close` | Eyes closed |
| `Fcl_EYE_Close_L` | Left eye closed (wink) |
| `Fcl_EYE_Close_R` | Right eye closed |
| `Fcl_EYE_Joy` | Happy eyes (squint) |
| `Fcl_EYE_Angry` | Angry eyes |
| `Fcl_EYE_Sorrow` | Sad eyes |
| `Fcl_EYE_Surprised` | Wide eyes |

### VRM Brow Blendshapes

| Blendshape | Description |
|------------|-------------|
| `Fcl_BRW_Joy` | Relaxed brows |
| `Fcl_BRW_Angry` | Furrowed brows |
| `Fcl_BRW_Sorrow` | Raised inner brows |
| `Fcl_BRW_Surprised` | Raised brows |
| `Fcl_BRW_Fun` | Playful brows |

## The Mapping Pipeline

### Stage 1: Affect → Emotion Weights

Convert PAD + sorrow + boredom to emotion weights:

```python
def affect_to_emotions(valence, arousal, dominance, sorrow, boredom):
    """
    Map 5D affect to emotion weights.

    Based on Mehrabian-Russell mappings with extensions.
    """
    emotions = {}

    # Happiness: positive valence + moderate-high arousal
    emotions['happiness'] = max(0, valence) * (0.5 + 0.5 * arousal)

    # Sadness: negative valence + low arousal + sorrow
    emotions['sadness'] = max(0, -valence) * (1 - arousal) * 0.5 + sorrow * 0.5

    # Anger: negative valence + high arousal + high dominance
    emotions['anger'] = max(0, -valence) * arousal * dominance

    # Fear: negative valence + high arousal + low dominance
    emotions['fear'] = max(0, -valence) * arousal * (1 - dominance)

    # Surprise: high arousal (valence-neutral)
    emotions['surprise'] = arousal * (1 - abs(valence)) * 0.5

    # Disgust: negative valence + low arousal
    emotions['disgust'] = max(0, -valence) * (1 - arousal) * 0.3

    # Boredom: direct mapping
    emotions['boredom'] = boredom

    # Concentration: moderate arousal + high dominance + neutral valence
    emotions['concentration'] = dominance * (1 - abs(valence)) * 0.5

    return emotions
```

### Stage 2: Emotion Weights → Action Units

Blend AU recipes by emotion weights:

```python
def emotions_to_aus(emotions):
    """
    Blend emotion AU recipes by weights.
    """
    aus = defaultdict(float)

    for emotion, weight in emotions.items():
        if emotion in EMOTION_AU_RECIPES and weight > 0.01:
            recipe = EMOTION_AU_RECIPES[emotion]
            for au, intensity in recipe.items():
                # Additive blending with max cap
                aus[au] = min(1.0, aus[au] + intensity * weight)

    return dict(aus)
```

### Stage 3: Action Units → VRM Blendshapes

Map AUs to VRM blendshapes:

```python
AU_TO_VRM = {
    # Eyes
    'AU5': [('Fcl_EYE_Surprised', 1.0)],  # Wide eyes
    'AU6': [('Fcl_EYE_Joy', 0.8)],        # Happy squint
    'AU7': [('Fcl_EYE_Angry', 0.7)],      # Lid tightener
    'AU43': [('Fcl_EYE_Close', 1.0)],     # Eyes closed

    # Brows
    'AU1': [('Fcl_BRW_Sorrow', 0.8)],     # Inner brow raise
    'AU2': [('Fcl_BRW_Surprised', 0.9)],  # Outer brow raise
    'AU4': [('Fcl_BRW_Angry', 0.8)],      # Brow lowerer

    # Mouth - map to expression blendshapes
    'AU12': [('Fcl_ALL_Joy', 0.7)],       # Smile
    'AU15': [('Fcl_ALL_Sorrow', 0.6)],    # Frown

    # Composite expressions
    'AU9': [('Fcl_ALL_Angry', 0.3)],      # Nose wrinkle → partial anger
    'AU20': [('Fcl_ALL_Surprised', 0.4)], # Lip stretch → partial surprise
}

def aus_to_vrm_blendshapes(aus):
    """
    Map Action Units to VRM blendshapes.
    """
    blendshapes = defaultdict(float)

    for au, intensity in aus.items():
        if au in AU_TO_VRM:
            for vrm_shape, scale in AU_TO_VRM[au]:
                blendshapes[vrm_shape] = min(1.0,
                    blendshapes[vrm_shape] + intensity * scale)

    return dict(blendshapes)
```

## FacialExpressionComponent

A new component type that drives avatar expressions:

```python
@dataclass
class FacialExpressionComponent(ComponentBase):
    """
    Drives VRM avatar facial expressions from affect channels.

    Subscribes to affect output channels and maps them through
    FACS to VRM blendshapes.
    """

    # Channel subscriptions
    affect_channel: str = "affect"  # Channel to listen on

    # Smoothing (prevent jittery expressions)
    smoothing_factor: float = 0.3  # 0 = instant, 1 = very smooth

    # Expression intensity multiplier
    intensity: float = 1.0

    # Micro-expression settings
    enable_micro_expressions: bool = True
    micro_expression_probability: float = 0.05
    micro_expression_duration: float = 0.1  # seconds

    # Blink settings
    enable_auto_blink: bool = True
    blink_interval_mean: float = 4.0  # seconds
    blink_interval_variance: float = 1.0
    blink_duration: float = 0.15

    # State
    _current_blendshapes: Dict[str, float] = field(default_factory=dict)
    _target_blendshapes: Dict[str, float] = field(default_factory=dict)
    _last_blink_time: float = 0
    _next_blink_time: float = 0
```

### Component Configuration in Inspector

```yaml
# In a noodling's component list
components:
  - type: FacialExpressionComponent
    affect_channel: "amelia/affect"
    smoothing_factor: 0.25
    intensity: 1.2
    enable_micro_expressions: true
    enable_auto_blink: true
    blink_interval_mean: 3.5
```

## Integration with Cognition Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cognition Cycle                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INCOMING ──▶ [Facets...] ──▶ CharmNetwork ──▶ OUTGOING         │
│                                     │                            │
│                                     ▼                            │
│                              affect_channel                      │
│                                     │                            │
│                                     ▼                            │
│                        FacialExpressionComponent                 │
│                                     │                            │
│                                     ▼                            │
│                           VRM Avatar Mesh                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

The FacialExpressionComponent:
1. Subscribes to the noodling's affect channel
2. On each affect update, runs the mapping pipeline
3. Smoothly interpolates blendshapes toward targets
4. Applies to the VRM avatar mesh

## Advanced Features

### Micro-Expressions

Brief, involuntary expressions that leak true feelings:

```python
def maybe_trigger_micro_expression(self, true_emotion):
    """
    Occasionally show a micro-expression of suppressed emotion.

    Micro-expressions last ~100ms and reveal what the noodling
    is really feeling, even if they're trying to hide it.
    """
    if random.random() < self.micro_expression_probability:
        # Flash the true emotion briefly
        self._micro_expression_active = True
        self._micro_expression_emotion = true_emotion
        self._micro_expression_end_time = time.time() + self.micro_expression_duration
```

### Emotional Inertia

Expressions don't snap instantly - they have momentum:

```python
def update_blendshapes(self, target, dt):
    """
    Smoothly interpolate toward target blendshapes.

    Uses exponential smoothing for natural movement.
    """
    alpha = 1 - math.exp(-dt / self.smoothing_factor)

    for shape, target_value in target.items():
        current = self._current_blendshapes.get(shape, 0)
        self._current_blendshapes[shape] = current + alpha * (target_value - current)
```

### Auto-Blink

Natural blinking at random intervals:

```python
def update_blink(self, current_time):
    """
    Handle automatic blinking.
    """
    if current_time >= self._next_blink_time:
        # Trigger blink
        self._blinking = True
        self._blink_end_time = current_time + self.blink_duration

        # Schedule next blink
        interval = random.gauss(self.blink_interval_mean,
                               self.blink_interval_variance)
        self._next_blink_time = current_time + max(1.0, interval)

    if self._blinking:
        # Blend in eye close
        self._current_blendshapes['Fcl_EYE_Close'] = 1.0

        if current_time >= self._blink_end_time:
            self._blinking = False
```

### Gaze Direction

Eye movement driven by attention:

```python
def update_gaze(self, attention_target):
    """
    Point eyes toward attention target.

    Uses VRM look-at blendshapes or bone rotation.
    """
    if attention_target:
        direction = normalize(attention_target - self.eye_position)
        # Map to VRM eye bone rotation or blendshapes
        self._eye_rotation = direction_to_rotation(direction)
```

## File Locations

```
noodlestudio/
├── core/
│   └── facial_expression_component.py  # Main component
├── runtime/
│   └── facs_mapper.py                  # Affect → FACS → VRM pipeline
└── resources/
    └── facs/
        ├── au_recipes.yaml             # Emotion → AU mappings
        └── vrm_mappings.yaml           # AU → VRM blendshape mappings
```

## Testing

```python
def test_happiness_produces_smile():
    """Positive valence + arousal should activate AU12 (smile)."""
    affect = Affect(valence=0.8, arousal=0.6, dominance=0.5, sorrow=0, boredom=0)
    aus = affect_to_aus(affect)
    assert aus['AU12'] > 0.5  # Lip corner puller active
    assert aus['AU6'] > 0.3   # Cheek raiser (Duchenne marker)

def test_sadness_produces_frown():
    """Negative valence + sorrow should activate AU1, AU15."""
    affect = Affect(valence=-0.6, arousal=0.2, dominance=0.3, sorrow=0.7, boredom=0)
    aus = affect_to_aus(affect)
    assert aus['AU1'] > 0.5   # Inner brow raiser
    assert aus['AU15'] > 0.4  # Lip corner depressor

def test_smoothing_prevents_jitter():
    """Expression changes should be smooth, not instant."""
    component = FacialExpressionComponent(smoothing_factor=0.3)

    # Sudden affect change
    component.set_target({'Fcl_ALL_Joy': 1.0})
    component.update(dt=0.016)  # One frame

    # Should not instantly reach target
    assert component.current['Fcl_ALL_Joy'] < 0.5
```

## Kimii-Sensei: The FACS Teacher

**Kimii-Sensei** is an axolotl animation teacher in Let's Consciousness! She teaches kids how facial expressions work using FACS as her curriculum.

*Named after Kim Tempest, Caity's animation teacher.*

### Kimii-Sensei's Lessons

Kimii-Sensei demonstrates expressions on her own face (axolotls have surprisingly expressive faces with their external gills and wide mouths), then asks kids to identify the Action Units:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     🦎 KIMII-SENSEI                                        │
│                                                             │
│     "Okay class! Watch my face carefully..."                │
│                                                             │
│     *demonstrates surprise*                                 │
│                                                             │
│     "What muscles did I use? Let's break it down!"          │
│                                                             │
│     [ ] Eyebrows went UP (AU1 + AU2)                       │
│     [ ] Eyes went WIDE (AU5)                               │
│     [ ] Mouth opened (AU25 + AU26)                         │
│                                                             │
│     "Now YOU try making a surprised face!"                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Lesson Structure

1. **Mirror Mode** - Kimii shows an expression, kid's webcam detects if they match
2. **Build-an-Expression** - Drag AU sliders to create target emotions
3. **Emotion Detective** - Watch video clips, identify which AUs were used
4. **Expression Blender** - Mix emotions (70% happy + 30% surprised = ?)

### Kimii's Teaching Moments

```yaml
lessons:
  - name: "The Duchenne Smile"
    kimii_says: |
      "Did you know there's a REAL smile and a FAKE smile?
      A real smile uses your CHEEK muscles too - that's AU6!
      Watch - can you see my gills move when I smile for real?"
    demonstrates:
      - fake_smile: {AU12: 0.8}  # Just mouth
      - real_smile: {AU12: 0.8, AU6: 0.7}  # Mouth + eyes

  - name: "Sad vs Angry"
    kimii_says: |
      "Both feel bad, but look different!
      Sad eyebrows go UP in the middle... *demonstrates*
      Angry eyebrows go DOWN and together... *demonstrates*
      The eyebrows tell the whole story!"
    demonstrates:
      - sadness: {AU1: 0.8, AU15: 0.6}
      - anger: {AU4: 0.9, AU7: 0.7}

  - name: "Micro-Expressions"
    kimii_says: |
      "Sometimes feelings are SO fast, they flash across your face
      in less than a second! These are called micro-expressions.
      Let's practice spotting them..."
    activity: "micro_expression_game"
```

### Integration with FacialExpressionComponent

Kimii-Sensei herself uses the FacialExpressionComponent, but with special "teaching mode" that can:
- Isolate individual AUs for demonstration
- Slow down expression transitions for clarity
- Highlight which muscles are active (glow effect on face regions)
- Mirror the student's detected expressions back to them

```yaml
# Kimii-Sensei's noodling configuration
components:
  - type: FacialExpressionComponent
    teaching_mode: true
    au_isolation_enabled: true
    transition_speed: 0.5  # Slower for teaching
    highlight_active_aus: true
```

### Why an Axolotl?

1. **Non-threatening** - Axolotls are cute, not intimidating
2. **Expressive gills** - External gills add visible expression channels
3. **Regeneration metaphor** - "Emotions can heal, just like I can regrow my gills!"
4. **Memorable** - Kids remember the unusual animal teacher
5. **Animation heritage** - Honors Kim Tempest's teaching legacy

## Future Enhancements

1. **Asymmetric Expressions** - Contempt, skepticism (one-sided)
2. **Speech-Driven Visemes** - Lip sync from TTS output
3. **Cultural Variations** - Different expression intensities by culture
4. **Age/Personality Modifiers** - Children emote bigger, stoics emote less
5. **Fatigue Effects** - Tired expressions when processing is heavy
6. **Mirror Neurons** - Noodlings subtly mirror user expressions (if webcam available)

## References

- Ekman, P. & Friesen, W.V. (1978). *Facial Action Coding System*
- Mehrabian, A. & Russell, J.A. (1974). *An Approach to Environmental Psychology*
- VRM Consortium. *VRM Specification* - https://vrm.dev/en/
- Noodlings Affect Model - `/docs/noodlestudio/affect-model.md`

---

*"The face is the mirror of the mind, and eyes without speaking confess the secrets of the heart."* — St. Jerome

Made with love by Caity & Claude
