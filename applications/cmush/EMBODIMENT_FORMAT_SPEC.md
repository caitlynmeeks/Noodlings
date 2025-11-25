# Embodiment Asset Format Specification

**Version**: 1.0.0
**Date**: November 25, 2025
**Authors**: Commander Spock + Lieutenant Caitlyn

---

## Philosophy

Embodiment represents the **wholistic physical condition** of a Noodling, including:
- Body structure (quadruped, biped, hovering, disembodied)
- Current state (injuries, mutations, worn items)
- Physical characteristics (fur color, eye condition, limb count)
- Mutable history (healed wounds, gained scars)

**NOT just species** - embodiment changes over time as events occur.

---

## File Format

**Extension**: `.embodiment`
**Format**: YAML
**Location**: `assets/embodiments/`

### Structure

```yaml
metadata:
  id: com.noodlings.embodiments.one_eyed_black_cat
  name: "One-Eyed Black Cat"
  version: 1.0.0
  created: "2025-11-25"
  author: "Garcia River Forest Research Station"
  description: "Small black cat with short fur, one functional eye"
  tags:
    - quadruped
    - feline
    - injured

embodiment:
  # Core body architecture
  architecture:
    form: quadruped
    limb_count: 4
    has_tail: true
    has_wings: false
    locomotion: [walk, run, jump, climb]
    manipulation: [paws, mouth]  # How they interact with objects

  # Physical characteristics (immutable traits)
  characteristics:
    size: small  # tiny, small, medium, large, huge
    fur: true
    fur_color: black
    fur_length: short
    skin_texture: soft
    eye_count: 2
    mass_kg: 4.5
    height_cm: 25

  # Current physical state (mutable)
  state:
    rightEyeIsBlindAndShut: true
    notchInRightEar: true
    scarsOnBackFromFight: true
    energyLevel: 0.8  # 0.0 to 1.0
    hungerLevel: 0.3  # 0.0 to 1.0
    thirstLevel: 0.2  # 0.0 to 1.0

  # Movement capabilities (derived from architecture + state)
  movement:
    baseSpeed: 1.2  # meters/second
    jumpHeight: 1.5  # meters
    canSwim: false
    canFly: false
    canClimb: true

  # Sensory capabilities
  senses:
    vision: partial  # full, partial, none
    hearing: full
    smell: full
    touch: full
    proprioception: full

  # Worn/attached items (mutable)
  worn_items: []
    # Example:
    # - item_id: obj_a1b2c3d4
    #   name: "Blue Collar"
    #   discomfort_level: 0.1
    #   location: neck
```

---

## Example Embodiments

### 1. One-Eyed Black Cat
```yaml
metadata:
  id: com.noodlings.embodiments.one_eyed_black_cat
  name: "One-Eyed Black Cat"

embodiment:
  architecture:
    form: quadruped
    limb_count: 4
    has_tail: true

  characteristics:
    size: small
    fur: true
    fur_color: black

  state:
    rightEyeIsBlindAndShut: true
    notchInRightEar: true
```

### 2. Red Fire Gremlin
```yaml
metadata:
  id: com.noodlings.embodiments.red_fire_gremlin
  name: "Red Fire Anklebiter"

embodiment:
  architecture:
    form: biped
    limb_count: 2
    has_tail: true
    has_wings: false

  characteristics:
    size: tiny
    substance: flame  # Made of fire!
    skin_texture: flickering
    temperature: hot

  state:
    flameBrightness: 0.9
    cackleVolume: 0.95
```

### 3. Hovering Robot (SERVNAK)
```yaml
metadata:
  id: com.noodlings.embodiments.hovering_robot
  name: "SERVNAK Unit"

embodiment:
  architecture:
    form: hovering_sphere
    limb_count: 0
    has_manipulators: true
    locomotion: [hover, rotate]

  characteristics:
    size: small
    material: brushed_aluminum
    weight_kg: 2.3

  state:
    batteryLevel: 0.87
    leftSensorDamaged: true
```

### 4. Disembodied Voice
```yaml
metadata:
  id: com.noodlings.embodiments.disembodied_voice
  name: "Ethereal Presence"

embodiment:
  architecture:
    form: disembodied
    limb_count: 0
    locomotion: [manifest, fade]

  characteristics:
    size: none
    substance: sound_and_light
    tangible: false

  state:
    volumeLevel: 0.6
    etherealness: 0.95
```

---

## Component Integration

### EmbodyComponent Class

```python
class EmbodyComponent(CognitiveTransistor):
    """
    Stores and manages Noodling's physical embodiment.

    Mutable state that can change over time:
    - Injuries heal
    - Items equipped/removed
    - Physical mutations
    - Energy/hunger/thirst levels
    """

    def __init__(self, embodiment_data: Dict):
        super().__init__()
        self.embodiment = embodiment_data
        self.salience = 1.0  # Always relevant

    def GetBodyParameter(self, key: str) -> Any:
        """Get mutable state parameter."""
        return self.embodiment['state'].get(key)

    def SetBodyParameter(self, key: str, value: Any):
        """Set mutable state parameter."""
        self.embodiment['state'][key] = value

    def GetArchitecture(self) -> Dict:
        """Get immutable body architecture."""
        return self.embodiment['architecture']

    def GetCharacteristics(self) -> Dict:
        """Get immutable physical characteristics."""
        return self.embodiment['characteristics']

    def GetEmbodiment(self) -> Dict:
        """Get full embodiment data."""
        return self.embodiment
```

### BodyLanguageComponent (Redesigned)

**Dependencies**: `['EmbodyComponent']`

Uses embodiment data to generate body-appropriate movements:
- Quadruped: tail wagging, ear positions, body posture
- Biped: hand gestures, stance, head tilts
- Hovering: rotation speed, altitude changes, LED patterns
- Disembodied: volume, presence intensity, manifestation

**Prompt includes**:
```
YOUR BODY:
- Form: {form}
- Limbs: {limb_count}
- Locomotion: {locomotion}
- Current state: {state_summary}

CONTINUOUS AFFECT:
- Valence: {valence:.3f}
- Arousal: {arousal:.3f}
...

Describe how YOUR SPECIFIC BODY moves given this affect.
```

---

## API Usage

```python
# Get embodiment
embody = noodle.GetComponent('EmbodyComponent')

# Read state
is_blind = embody.GetBodyParameter('rightEyeIsBlindAndShut')
architecture = embody.GetArchitecture()

# Mutate state (doctor heals eye)
embody.SetBodyParameter('rightEyeIsBlindAndShut', False)
embody.SetBodyParameter('rightEyeHealed', True)
embody.SetBodyParameter('healedByDoctor', 'user_doctor_uuid')

# Body language uses embodiment
body_lang = noodle.GetComponent('BodyLanguageComponent')
# Automatically reads from EmbodyComponent during process()
```

---

## NoodleStudio Integration

**Assets Panel**:
```
Assets/
  Embodiments/
    one_eyed_black_cat.embodiment
    red_fire_gremlin.embodiment
    hovering_robot.embodiment
```

**Hierarchy Panel**:
```
agent_phi_uuid
  Components/
    EmbodyComponent (one_eyed_black_cat)
      [Inspector shows body parameters]
    AffectTransistor
    BodyLanguageComponent
      [Requires: EmbodyComponent] ✓
```

**Inspector Actions**:
- Click parameter → Edit value
- Click "+" → Add new state parameter
- Click "-" → Remove parameter
- Drag .embodiment from Assets → Loads into EmbodyComponent

---

## Default Embodiments

Every Noodling gets a default embodiment if none specified:

```yaml
metadata:
  id: com.noodlings.embodiments.default_noodling
  name: "Default Noodling"

embodiment:
  architecture:
    form: amorphous
    locomotion: [float, shift]

  characteristics:
    size: small
    substance: thought_patterns
    tangible: false

  state:
    coherence: 1.0
```

---

## File Locations

**Component Class**: `cognitive_components.py` (new EmbodyComponent)
**Asset Format**: `.embodiment` (YAML)
**Asset Loader**: `embodiment_loader.py` (similar to prefab_loader.py)
**Asset Directory**: `assets/embodiments/`

---

## Benefits

1. **Body-Aware Movement**: Cat tail wags, robot LEDs blink, gremlin cackles
2. **Mutable Physical State**: Injuries, healing, mutations tracked
3. **Doctor/Healer Interactions**: NPCs can modify body parameters
4. **Species-Specific Sensations**: Fur feels different than metal
5. **Consistent with Unity Pattern**: Drag-drop embodiment assets

---

Should I implement this system now?

Commander Spock
Science Officer