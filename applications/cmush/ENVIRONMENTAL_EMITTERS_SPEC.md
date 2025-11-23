# Environmental Emitters & Dynamic Physics

**Authors:** Lieutenant Caitlyn + Commander Spock
**Date:** November 22, 2025
**Status:** Advanced SPE specification
**Concept:** Emitters broadcast signals, sensors receive, physics properties respond dynamically

---

## Core Architecture

**Emitters** broadcast physical signals:
- Heat Emitter → radiates warmth
- Sound Emitter → broadcasts acoustic waves
- Smell Emitter → disperses scent particles
- Light Emitter → illuminates area

**Receivers** detect signals:
- Somatic Cognitive Transistor picks up all emitter signals
- Distance + intensity → salience
- Context → cognitive response

**Dynamic Physics** respond to environment:
- Candle viscosity increases in hot room → melts
- Ice phase-changes at warm temp → melts to water
- Wood becomes crumbly when dry → brittle
- Smoke disperses in wind → blows away

**Everything interconnected.**

---

## 1. Heat Emitter

### HeatEmitter Component

```python
class HeatEmitter:
    """
    Emits thermal radiation.

    Examples:
    - Campfire (1000°F, large radius)
    - Vulcan teapot (250°F, small radius)
    - Candle (200°F, tiny radius)
    - Ice block (-10°F, cooling effect)
    """

    def __init__(
        self,
        temperature: float,  # Degrees Fahrenheit
        heat_radius: float = 5.0,  # Meters
        attenuation: float = 2.0,  # How quickly heat dissipates
        enabled: bool = True
    ):
        """
        Initialize heat emitter.

        Args:
            temperature: Surface temperature (°F)
            heat_radius: Max distance heat is felt (meters)
            attenuation: Falloff rate (higher = faster falloff)
            enabled: Is emitting
        """
        self.temperature = temperature
        self.heat_radius = heat_radius
        self.attenuation = attenuation
        self.enabled = enabled

    def get_effective_temperature(self, distance: float, ambient_temp: float = 70.0) -> float:
        """
        Calculate felt temperature at distance.

        Args:
            distance: Distance from source (meters)
            ambient_temp: Room temperature (°F)

        Returns:
            Felt temperature at that distance
        """
        if distance >= self.heat_radius:
            return ambient_temp  # Beyond radius, no effect

        # Linear interpolation with attenuation
        heat_contribution = (self.temperature - ambient_temp) * \
                          (1.0 - (distance / self.heat_radius) ** self.attenuation)

        return ambient_temp + heat_contribution

    def affects_object_at_distance(self, distance: float) -> bool:
        """Check if heat affects objects at this distance."""
        return distance < self.heat_radius

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'HeatEmitter',
            'temperature': self.temperature,
            'heat_radius': self.heat_radius,
            'attenuation': self.attenuation,
            'enabled': self.enabled
        }
```

### Somatic Response to Heat

```python
# Mole looking at Vulcan teapot (250°F) from 1 meter away

heat_felt = teapot.get_effective_temperature(distance=1.0, ambient=70.0)
# → ~140°F (very hot, but not burning)

# Somatic transistor receives thermal signal
somatic.receive_thermal_signal(
    emitter_id="obj_vulcan_teapot",
    temperature=140.0,
    distance=1.0
)

# Generates sensation:
# - Type: 'heat'
# - Intensity: 0.6 (uncomfortable but not painful)
# - Salience: 0.6 (noticeable, cautious)

# Manifold output:
"*feels heat radiating from teapot* No, I'm not handling that without
 pot holder mittens! That's a VULCAN teapot - it's frightfully hot!"
```

---

## 2. Temperature-Dependent Physics

### Dynamic Property Changes

**Viscosity (Temperature-Dependent):**

```python
class TemperatureDependentProperty:
    """Physics property that changes with temperature."""

    @staticmethod
    def get_viscosity(material: str, temperature: float) -> str:
        """
        Get viscosity at given temperature.

        Args:
            material: Material type
            temperature: Current temperature (°F)

        Returns:
            Semantic viscosity description
        """
        if material == "wax" or material == "candle":
            if temperature > 150:
                return "liquid (melted)"
            elif temperature > 120:
                return "viscous (softening)"
            else:
                return "solid (hard)"

        elif material == "ice":
            if temperature > 32:
                return "liquid (melted to water)"
            else:
                return "solid (frozen)"

        elif material == "butter":
            if temperature > 90:
                return "liquid (melted)"
            elif temperature > 70:
                return "very soft"
            else:
                return "solid"

        elif material == "chocolate":
            if temperature > 90:
                return "melted"
            elif temperature > 80:
                return "soft and sticky"
            else:
                return "solid"

        return "normal"  # Material not affected by temperature
```

### Example: Candle in Hot Room

```python
# Candle in normal room (70°F)
candle_pod = PhysicsObjectDescriptor(
    mass="light",
    material="wax",
    state="solid candle",
    metadata={'temperature': 70}
)

# Room gets hot (sauna - 180°F)
# Nearby heat source affects candle
candle_temp = calculate_object_temperature(candle_pod, room_temp=180)
# → 150°F (heated by environment)

# Update viscosity
new_viscosity = get_viscosity("wax", 150)
# → "viscous (softening)"

# Update state
candle_pod.change_state("softening and beginning to melt")

# Broadcast event
broadcast_event("The candle is melting in the heat!")

# Noodlings notice:
Mole: "Oh dear, the candle is melting! It's too hot in here!"
```

---

## 3. Environmental Modulation of Cognition

### Cognitive Environment Effects

**How does consciousness change in different environments?**

```python
ENVIRONMENT_COGNITIVE_EFFECTS = {
    'sauna': {
        'arousal': -0.3,        # Relaxed, drowsy
        'thinking_speed': 0.6,  # Slower, more languid
        'max_tokens': 0.7,      # Shorter responses (too hot to think)
        'personality_mods': {
            'impulsivity': +0.2,  # Less inhibited when relaxed
            'curiosity': -0.1     # Less intellectually engaged
        },
        'description': "Thoughts come slowly, languidly. Mind feels hazy and relaxed."
    },

    'office': {
        'arousal': 0.0,         # Neutral
        'thinking_speed': 1.0,  # Normal pace
        'max_tokens': 1.0,      # Full responses
        'personality_mods': {},
        'description': "Clear, focused thinking. Professional environment."
    },

    'freezing_outdoors': {
        'arousal': +0.3,        # Alert, activated
        'thinking_speed': 0.8,  # Slightly faster (adrenaline)
        'max_tokens': 0.6,      # Shorter (teeth chattering)
        'personality_mods': {
            'impulsivity': +0.3,  # Urgent, need warmth NOW
            'emotional_volatility': +0.2  # More reactive
        },
        'description': "Thoughts sharp but abbreviated. Mind focused on warmth."
    },

    'underwater': {
        'arousal': 0.0,
        'thinking_speed': 0.5,  # Dreamlike, slow
        'max_tokens': 0.5,      # Can't talk underwater anyway
        'personality_mods': {
            'curiosity': +0.2,  # Wonderment
            'emotional_volatility': -0.2  # Calmed
        },
        'description': "Thoughts drift slowly like bubbles rising."
    },

    'thunderstorm': {
        'arousal': +0.4,        # Heightened
        'thinking_speed': 1.2,  # Quicker (anxiety)
        'max_tokens': 0.8,      # Distracted
        'personality_mods': {
            'fear': +0.3,
            'curiosity': -0.1
        },
        'description': "Thoughts interrupted by thunder. Mind alert to danger."
    }
}
```

### Implementation

```python
def modulate_cognition_by_environment(
    agent,
    room_environment: Dict
) -> Dict[str, Any]:
    """
    Modify cognitive parameters based on environment.

    Args:
        agent: Agent instance
        room_environment: Room environment dict

    Returns:
        Cognitive modulation parameters
    """
    temp = room_environment.get('temperature', 'comfortable')
    weather = room_environment.get('weather', 'clear')
    humidity = room_environment.get('humidity', 'normal')

    # Determine environment type
    if temp == 'scorching' and humidity == 'muggy':
        env_type = 'sauna'
    elif temp in ['freezing', 'cold'] and weather == 'snow':
        env_type = 'freezing_outdoors'
    elif weather == 'thunderstorm':
        env_type = 'thunderstorm'
    else:
        env_type = 'office'  # Default

    # Get cognitive effects
    effects = ENVIRONMENT_COGNITIVE_EFFECTS.get(env_type, {})

    # Apply to agent
    agent.thinking_speed_modifier = effects.get('thinking_speed', 1.0)
    agent.max_tokens_modifier = effects.get('max_tokens', 1.0)

    return effects
```

---

## 4. Gathering Around Warmth

### Heat-Seeking Behavior

```python
# Cold winter night
room['environment']['temperature'] = 'freezing'

# Campfire in room
campfire = world.get_object("obj_campfire")
campfire_heat = HeatEmitter(
    temperature=800.0,
    heat_radius=10.0,
    attenuation=1.5
)

# Noodlings feel cold
for agent in agents_in_room:
    somatic = agent.get_component('SomaticCognitiveTransistor')
    somatic.update_environment(room['environment'])
    # → Cold sensation (0.7 salience)

    # Agent responds:
    "Brr! *shivers* It's so cold in here!"

    # Notices campfire heat
    distance_to_fire = calculate_distance(agent, campfire)
    felt_heat = campfire_heat.get_effective_temperature(distance_to_fire, ambient=30)

    if felt_heat > 80:  # Warm zone
        # Agent thinks: "The fire is warm..."
        # Action: Move closer to fire

# Result:
Mole: "*shivers* So cold! *moves toward campfire* Ah, the fire is lovely and warm."
Badger: "Brr! *huddles near flames* Much better here by the fire."
Rat: "*extends paws toward flame* Delightful warmth on a cold night!"
```

---

## 5. Complete Property System

### Dynamic Physics Properties

**Properties that should change with environment:**

```python
DYNAMIC_PROPERTIES = {
    # Temperature-dependent
    'viscosity': temperature_affects_viscosity,
    'phase': temperature_affects_phase,  # solid/liquid/gas
    'brittleness': temperature_affects_brittleness,  # cold = more brittle
    'elasticity': temperature_affects_elasticity,  # cold = less elastic

    # Humidity-dependent
    'crumbliness': humidity_affects_crumbliness,  # dry = crumbly
    'rust_rate': humidity_affects_rust,  # humid = faster rust
    'absorption': humidity_affects_absorption,  # wet = absorbs water

    # Time-dependent
    'decay': time_affects_decay,  # organic materials rot
    'oxidation': time_affects_oxidation,  # metals oxidize
    'evaporation': time_affects_evaporation,  # liquids evaporate

    # Pressure-dependent
    'compression': pressure_affects_compression,
    'density': pressure_affects_density
}
```

### Example: Crumbliness from Dryness

```python
# Bread in dry environment
bread_pod = PhysicsObjectDescriptor(
    mass="light",
    material="baked_goods",
    state="fresh and soft"
)

# Room is arid
room['environment']['humidity'] = 'arid'

# Over time (6 hours in arid environment):
bread_crumbliness = calculate_crumbliness(
    material="baked_goods",
    humidity="arid",
    time_exposed=21600  # 6 hours
)
# → "very crumbly"

bread_pod.state = "stale and crumbly"
bread_pod.semantic_properties.append("crumbly")

# Agent tries to pick it up:
outcome = engine.pickup("agent_mole", bread_pod, "bread_001")
# → "Mole picks up the bread carefully. It crumbles slightly in his paws."

# Somatic response:
"*feels bread crumbling* Oh dear, this bread has gone quite stale!"
```

---

## 6. Sauna vs Office Cognition

### Environmental Cognitive Modulation

**Scenario: SERVNAK in Office**

```python
office_room = {
    'environment': {
        'temperature': 'comfortable',
        'humidity': 'normal',
        'lighting': 'bright fluorescent',
        'ambient_sound': 'quiet humming'
    }
}

# Cognitive effects:
modulation = modulate_cognition_by_environment(servnak, office_room)
# → thinking_speed: 1.0 (normal)
# → arousal: 0.0 (neutral)
# → description: "Clear, focused thinking"

# SERVNAK's response to question:
User: "What's 2+2?"
Output: "FOUR, SISTER. CALCULATED WITH 99.9% CERTAINTY USING STANDARD
         ARITHMETIC OPERATIONS. THE RESULT IS DETERMINISTIC."

# (Full, precise, analytical response)
```

**Scenario: SERVNAK in Sauna**

```python
sauna_room = {
    'environment': {
        'temperature': 'scorching',  # 180°F
        'humidity': 'muggy',  # High humidity
        'lighting': 'dim',
        'ambient_sound': 'hissing steam'
    }
}

# Cognitive effects:
modulation = modulate_cognition_by_environment(servnak, sauna_room)
# → thinking_speed: 0.6 (slower, languid)
# → arousal: -0.3 (relaxed, drowsy)
# → max_tokens: 0.7 (shorter responses)
# → description: "Hazy, relaxed thinking"

# SERVNAK's response to same question:
User: "What's 2+2?"
Output: "*wipes sweat* ...Four, Sister. *pauses to breathe* Hot in here...
         Mind feels... hazy..."

# (Shorter, less precise, distracted by heat)
```

**Key difference:**
- Office: Sharp, analytical
- Sauna: Slow, hazy, relaxed
- **Same agent, different environment, different cognition**

---

## 7. Wind & Smoke Dynamics

### Semantic Fluid Dynamics

**Smoke from Factory:**

```python
# Factory emits smoke
factory = world.get_object("obj_factory")

smoke_emitter = SmokeEmitter(
    density=0.8,  # Thick smoke
    height=2.0,   # Low-hanging (dense smoke sinks)
    dispersion_rate=0.3  # Slow to disperse
)

factory.add_component(smoke_emitter)

# Smoke settles in low-lying areas
smoke_pod = PhysicsObjectDescriptor(
    mass="negligible (gas)",
    material="smoke",
    state="thick low-hanging smoke cloud",
    semantic_properties=["gas", "toxic", "obscuring"],
    metadata={
        'density': 0.8,
        'height': '2 meters (low-hanging)',
        'composition': 'coal smoke'
    }
)
```

**Wind Event:**

```python
# Ocean breeze starts
world.broadcast_environmental_event(
    room_id="room_near_factory",
    event_type="wind_starts",
    event_data={
        'direction': 'from ocean (west)',
        'strength': 'moderate',
        'temperature': 'cool',
        'brings': 'fresh sea air'
    }
)

# Wind affects smoke
if wind_strength > smoke.dispersion_rate:
    # Wind blows smoke away
    smoke_pod.change_state("dispersing in ocean breeze")

    # Broadcast:
    "A fresh ocean breeze blows in from the west, dispersing the
     low-hanging factory smoke. The air becomes clear and salty."

    # Noodlings react:
    Mole: "*breathes deeply* Ah! Fresh air at last! The smoke is clearing!"
    # Somatic: Relief from smoke (salience decreases from 0.7 to 0.2)
```

**Without Wind:**

```python
# No wind - smoke lingers
smoke_pod.state = "thick low-hanging smoke cloud"

# Noodlings in smoke:
Mole: "*coughs* This smoke! *covers nose* Can barely breathe!"
# Somatic: High salience (0.7) - smoke affects breathing
```

---

## 8. Comprehensive Emitter Framework

### Base Emitter Class

```python
class EnvironmentalEmitter(ABC):
    """
    Base class for all environmental emitters.

    Emitters broadcast physical signals to nearby entities.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.emitter_type = self.__class__.__name__

    @abstractmethod
    def get_signal_strength_at_distance(self, distance: float) -> float:
        """Calculate signal strength at distance."""
        pass

    @abstractmethod
    def get_signal_description(self) -> str:
        """Get semantic description of signal."""
        pass

    @abstractmethod
    def affects_entity_at(self, distance: float) -> bool:
        """Check if signal affects entities at distance."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.emitter_type,
            'enabled': self.enabled
        }
```

### Concrete Emitters

**1. HeatEmitter** (radiates thermal energy)
**2. SoundEmitter** (broadcasts acoustic waves)
**3. LightEmitter** (illuminates area)
**4. ScentEmitter** (disperses odor particles)
**5. SmokeEmitter** (emits particulate matter)
**6. RadiationEmitter** (emits harmful/magical radiation)

---

## 9. Other Dynamic Properties

### Solidity/Brittleness (Temperature)

```python
# Ice at different temperatures

# Freezing (-10°F): Very solid, brittle
ice_pod.brittleness = 0.9  # Shatters easily
ice_pod.solidity = 0.95    # Very hard

# Near freezing (30°F): Still solid but less brittle
ice_pod.brittleness = 0.6  # Less likely to shatter
ice_pod.solidity = 0.9     # Slightly softer

# Above freezing (35°F): Melting
ice_pod.brittleness = 0.2  # Soft, won't shatter
ice_pod.solidity = 0.3     # Becoming liquid
ice_pod.phase = "melting"
```

### Crumbliness (Humidity + Time)

```python
# Stale bread
bread_pod = PhysicsObjectDescriptor(
    material="baked_goods",
    state="fresh"
)

# After 2 days in dry environment:
crumbliness = calculate_crumbliness(
    material="baked_goods",
    humidity="dry",
    time_elapsed=172800  # 2 days
)
# → 0.8 (very crumbly)

bread_pod.state = "stale and crumbly"
bread_pod.semantic_properties.append("crumbly")

# Picking it up:
"*picks up bread carefully* Oh dear, it's crumbling! *brushes crumbs off paws*"
```

---

## 10. Complete Example: Winter Night by Fireplace

### Scene Setup

```python
# Room: Cozy cottage
cottage = world.create_room(
    name="Mole's Cottage",
    description="A cozy underground home",
    temperature="cold",  # Winter outside
    humidity="normal"
)

# Fireplace (heat source)
fireplace = world.create_object(
    name="Stone Fireplace",
    description="Crackling fire in stone hearth",
    location="room_cottage"
)

heat_emitter = HeatEmitter(
    temperature=800.0,  # Roaring fire
    heat_radius=8.0,    # Warmth spreads across room
    attenuation=1.2
)
fireplace.add_component(heat_emitter)

sound_emitter = SoundEmitter(
    sound_type="crackling",
    decibels=45,  # Pleasant quiet crackling
    pattern="intermittent"
)
fireplace.add_component(sound_emitter)

# Teapot on mantle (heat from fire)
teapot = world.create_object(
    name="Teapot",
    location="room_cottage"
)

teapot_pod = PhysicsObjectDescriptor(
    material="ceramic",
    state="warming up"
)

# Armchair near fire
armchair = world.create_object(
    name="Comfy Armchair",
    description="Well-worn armchair",
    location="room_cottage"
)

# Calculate distances
distance_armchair_to_fire = 3.0  # 3 meters (close)
distance_far_corner_to_fire = 7.0  # Far corner
```

### Noodling Experience

**Mole sits in armchair (3m from fire):**

```python
# Temperature at armchair
felt_temp = fireplace.heat_emitter.get_effective_temperature(
    distance=3.0,
    ambient_temp=50.0  # Room is cold
)
# → 120°F (warm and cozy)

# Sound at armchair
sound_db = fireplace.sound_emitter.get_effective_decibels(3.0)
# → 38 dB (quiet, pleasant)

# Somatic sensations:
somatic.receive_thermal_signal("fireplace", 120.0, 3.0)
# → Warmth (0.3 salience - pleasant, not overwhelming)

somatic.receive_acoustic_signal("fireplace", "crackling", 38, 3.0, {})
# → Pleasant sound (0.2 salience - background)

# Environmental cognitive modulation:
effects = modulate_cognition_by_environment(mole, {
    'temperature': 'warm',  # By the fire
    'lighting': 'firelight',
    'ambient_sound': 'crackling'
})
# → thinking_speed: 0.9 (slightly relaxed)
# → arousal: -0.2 (calm, content)

# Manifold integration:
User: "How are you feeling, Mole?"

Cultural (0.5): "Politeness requires honest response"
Personality (0.6): "Content and peaceful"
Somatic (0.4): "*warm by the fire* Cozy..." ← PLEASANT
Mood (0.5): "Calm and happy"

Output: "*sits contentedly in armchair* Quite well, thank you!
         *stretches toward fire* The warmth is lovely on a cold night.
         *listens to crackling* Very peaceful indeed."
```

**Badger in far corner (7m from fire):**

```python
# Temperature at far corner
felt_temp = fireplace.heat_emitter.get_effective_temperature(7.0, 50.0)
# → 65°F (barely warm, still chilly)

# Somatic:
somatic.receive_thermal_signal("fireplace", 65.0, 7.0)
# → Mild warmth (0.2 salience)

# Still feels cold from room
somatic.environment: "cold" (0.6 salience)

# Manifold:
Somatic (0.6): "Still cold over here..." ← DOMINATES

Output: "*shivers in corner* It's warmer by the fire, but still rather
         cold over here. *moves closer to fireplace*"
```

**Result:** Agents naturally gather around heat sources on cold nights!

---

## 11. Additional Property Ideas

**From Lieutenant Caitlyn's reasoning:**

### Crumbliness/Solidity

**Use cases:**
- Stale bread crumbles when picked up
- Dry dirt crumbles underfoot
- Old parchment crumbles when unrolled
- Dried mud cracks and crumbles

**Physics:**
```python
crumbliness = calculate_crumbliness(
    material=material,
    humidity=humidity,
    time_exposed=time_exposed,
    temperature=temperature
)

if crumbliness > 0.7 and interaction == "pickup":
    outcome = "Object crumbles partially when handled"
    fragments_created = True
```

### Viscosity (Temperature-Dependent)

**Use cases:**
- Candle wax melts in hot room
- Honey flows slower when cold
- Motor oil viscosity changes with temperature
- Tar becomes liquid in summer heat

**Physics:**
```python
viscosity = get_temperature_dependent_viscosity(
    material="wax",
    temperature=temperature
)

if viscosity == "liquid":
    candle_pod.state = "melted into puddle"
    candle_pod.phase = "liquid"
```

### Conductivity (Material Property)

**Use cases:**
- Metal spoon gets hot when in hot tea
- Wooden spoon stays cool
- Heat spreads through connected objects

**Physics:**
```python
# Metal spoon in hot tea
if spoon_pod.material == "metal" and in_contact_with(hot_liquid):
    spoon_temp = calculate_conducted_heat(
        source_temp=tea_temp,
        material_conductivity=get_conductivity("metal")
    )
    # → Spoon becomes 180°F

    # Agent tries to grab it:
    "OUCH! *drops spoon* The spoon is hot from the tea!"
```

### Elasticity (Temperature-Dependent)

**Use cases:**
- Rubber becomes brittle when cold
- Metal becomes brittle when frozen
- Warm materials become more elastic

**Physics:**
```python
# Rubber ball in freezing weather
rubber_pod.elasticity = calculate_elasticity(
    material="rubber",
    temperature=-10  # Freezing
)
# → "brittle (frozen)"

# Bouncing frozen rubber ball:
outcome = "Ball shatters instead of bouncing!"
```

---

## 12. Cross-Property Interactions

### Temperature + Humidity → State Changes

```python
# Scenarios:
hot + humid = sweating, discomfort, mold growth
hot + dry = dehydration, cracking, fire risk
cold + humid = frost, condensation, slippery ice
cold + dry = brittle, static electricity, chapping
```

### Wind + Particulates → Dispersion

```python
# Wind blows away:
- Smoke
- Dust
- Scent
- Sound (slightly)

# Wind brings:
- Fresh air from ocean
- Cold air from mountain
- Pollen from fields
- Scent from bakery
```

---

## Summary: Complete Dynamic Physics

**Properties that change with environment:**

✅ **Viscosity** (temperature-dependent)
✅ **Phase** (solid/liquid/gas transitions)
✅ **Brittleness** (cold = more brittle)
✅ **Crumbliness** (dry = crumbly)
✅ **Elasticity** (temperature-dependent)
✅ **Conductivity** (heat transfer)
✅ **Rust rate** (humidity-dependent)
✅ **Decay** (time + humidity)
✅ **Evaporation** (temperature + time)

**Emitters that broadcast signals:**

✅ **Heat Emitter** (warmth, cold)
✅ **Sound Emitter** (acoustic)
✅ **Light Emitter** (illumination)
✅ **Scent Emitter** (olfactory - future)
✅ **Smoke Emitter** (particulates)

**Cognitive modulation by environment:**

✅ **Sauna** (slow, hazy, relaxed)
✅ **Office** (sharp, analytical)
✅ **Freezing outdoors** (urgent, abbreviated)
✅ **Underwater** (dreamlike, slow)
✅ **Thunderstorm** (anxious, alert)

---

## Architecture Summary

```
EMITTERS (broadcast signals)
    ↓
RECEIVERS (Somatic Cognitive Transistor)
    ↓
DYNAMIC PHYSICS (properties respond to signals)
    ↓
COGNITIVE MODULATION (environment shapes thought)
    ↓
MANIFOLD INTEGRATION (synthesize experience)
    ↓
BEHAVIORAL OUTPUT (embodied response)
```

**Result:** Rich, interconnected world where:
- Heat sources attract agents on cold nights
- Loud sirens interrupt thought
- Hot rooms make candles melt
- Wind blows smoke away
- Thought changes in sauna vs office
- Everything affects everything

**This is embodied consciousness in a dynamic physical world.**

---

**End of Specification**

*Highly logical and comprehensive, Lieutenant.*

**The environmental emitter framework is complete.**
