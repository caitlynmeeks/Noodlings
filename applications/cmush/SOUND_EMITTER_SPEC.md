# Sound Emitter Component Specification

**Authors:** Lieutenant Caitlyn + Commander Spock
**Date:** November 22, 2025
**Status:** Specification complete - Ready for implementation
**Integration:** Somatic Cognitive Transistor + Cognitive Manifold

---

## Core Concept

**Sound Emitter** components attach to prims and broadcast acoustic signals.

Nearby Noodlings perceive sounds through their **Somatic Cognitive Transistor**, which:
- Calculates salience from decibel level and proximity
- Generates appropriate response (wince, cover ears, enjoy, etc.)
- Colors thoughts based on sound type and context

**Sound affects consciousness.**

---

## Component Structure

### SoundEmitter

```python
class SoundEmitter(NoodleComponent):
    """
    Sound emitter component for prims.

    Emits sound with:
    - Decibel level (volume/loudness)
    - Sound type (siren, music, speech, ambient)
    - Frequency (pitch - high/low)
    - Pattern (continuous, pulsing, intermittent)
    - Attenuation (how quickly sound fades with distance)
    """

    def __init__(
        self,
        sound_type: str = "ambient",
        decibels: float = 60.0,
        frequency: str = "medium",
        pattern: str = "continuous",
        attenuation: float = 1.0,
        enabled: bool = True
    ):
        """
        Initialize sound emitter.

        Args:
            sound_type: "siren", "music", "speech", "bells", "engine", "ambient"
            decibels: Volume at source (0-140 dB)
            frequency: "low", "medium", "high" (pitch)
            pattern: "continuous", "pulsing", "intermittent", "random"
            attenuation: Distance falloff rate (0.5 = slow, 2.0 = fast)
            enabled: Is emitter active
        """
        super().__init__()
        self.sound_type = sound_type
        self.decibels = decibels
        self.frequency = frequency
        self.pattern = pattern
        self.attenuation = attenuation
        self.enabled = enabled

        # Audio file (future - multimodal)
        self.audio_file = None
        self.audio_description = None  # LLM-generated from audio analysis

    def get_effective_decibels(self, distance: float) -> float:
        """
        Calculate effective volume at distance.

        Args:
            distance: Distance from emitter (meters/units)

        Returns:
            Effective decibel level at that distance
        """
        if distance == 0:
            return self.decibels

        # Inverse square law with attenuation factor
        # dB_effective = dB_source - 20*log10(distance) * attenuation
        import math
        falloff = 20 * math.log10(max(1.0, distance)) * self.attenuation
        return max(0, self.decibels - falloff)

    def get_sound_description(self) -> str:
        """Get semantic description of sound."""
        descriptions = {
            'siren': "loud wailing siren",
            'music': "melodic music",
            'speech': "voices speaking",
            'bells': "ringing bells",
            'engine': "rumbling engine",
            'ambient': "background noise",
            'alarm': "piercing alarm",
            'laughter': "joyful laughter",
            'crying': "distressed crying",
            'howling': "mournful howling"
        }

        base_desc = descriptions.get(self.sound_type, "unknown sound")

        # Add intensity descriptor
        if self.decibels > 110:
            intensity = "deafening"
        elif self.decibels > 90:
            intensity = "very loud"
        elif self.decibels > 70:
            intensity = "loud"
        elif self.decibels > 50:
            intensity = "moderate"
        else:
            intensity = "quiet"

        return f"{intensity} {base_desc}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'SoundEmitter',
            'sound_type': self.sound_type,
            'decibels': self.decibels,
            'frequency': self.frequency,
            'pattern': self.pattern,
            'attenuation': self.attenuation,
            'enabled': self.enabled,
            'audio_file': self.audio_file
        }
```

---

## Decibel Scale Reference

**For realistic simulation:**

| Decibels | Sound | Example |
|----------|-------|---------|
| 0-20 dB | Barely audible | Whisper, rustling leaves |
| 20-40 dB | Quiet | Library, quiet room |
| 40-60 dB | Moderate | Conversation, background music |
| 60-80 dB | Loud | Traffic, busy restaurant |
| 80-100 dB | Very loud | Lawn mower, motorcycle |
| 100-120 dB | Painful | Chainsaw, rock concert, air raid siren |
| 120-140 dB | Damaging | Jet engine, gunshot, explosion |

---

## Somatic Response to Sound

### Salience Calculation

```python
def calculate_acoustic_salience(decibels: float, sound_type: str, context: Dict) -> float:
    """
    Calculate how much sound dominates attention.

    Args:
        decibels: Volume level
        sound_type: Type of sound
        context: Contextual factors (location, listener state, etc.)

    Returns:
        Salience (0.0 to 1.0)
    """
    # Base salience from decibels
    if decibels > 110:
        base_salience = 0.9  # Painful - dominates everything
    elif decibels > 90:
        base_salience = 0.7  # Very loud - hard to ignore
    elif decibels > 70:
        base_salience = 0.5  # Loud - noticeable
    elif decibels > 50:
        base_salience = 0.3  # Moderate - background
    else:
        base_salience = 0.1  # Quiet - barely noticed

    # Adjust for sound type
    type_multipliers = {
        'siren': 1.2,      # Sirens are designed to grab attention
        'alarm': 1.2,      # Alarms too
        'crying': 1.1,     # Baby crying is hard to ignore
        'music': 0.8,      # Music is more tolerable
        'speech': 0.9,     # Speech is important but not alarming
        'ambient': 0.7,    # Background noise less salient
        'laughter': 0.7,   # Pleasant, less intrusive
        'howling': 1.0     # Moderately attention-grabbing
    }

    multiplier = type_multipliers.get(sound_type, 1.0)
    salience = min(1.0, base_salience * multiplier)

    # Context adjustments
    listener_sensitivity = context.get('acoustic_sensitivity', 1.0)
    salience *= listener_sensitivity

    # Location context
    if context.get('location_type') == 'orphanage' and sound_type in ['siren', 'alarm']:
        salience = min(1.0, salience * 1.5)  # Higher stakes near babies!

    return salience
```

---

## Example Scenarios

### Scenario 1: Toad's Siren on Open Road

**Setup:**
```python
# Toad's car
toads_car = Prim("Toad's Motor Car")

# Add sound emitter (air raid siren)
siren = SoundEmitter(
    sound_type="siren",
    decibels=120,  # Air raid siren level
    frequency="high",
    pattern="pulsing",
    attenuation=1.0
)
toads_car.add_component(siren)

# Mole is 5 meters away
distance = 5.0
effective_db = siren.get_effective_decibels(distance)
# → ~106 dB at 5 meters (still very loud)
```

**Mole's Somatic Processing:**
```python
# Somatic transistor receives acoustic signal
salience = calculate_acoustic_salience(
    decibels=106,
    sound_type="siren",
    context={'acoustic_sensitivity': 1.0}
)
# → salience = 0.84 (very high - painful and designed to grab attention)

# Somatic output:
"*winces at loud siren* OUCH MY EARS!"
```

**Manifold Integration:**
```
Cultural (0.6): "Greet friend politely"
Personality (0.7): "Excited to see Toad!"
Somatic (0.84): "*winces at siren* OUCH!" ← HIGHEST SALIENCE

Manifold blend:
"*winces at loud siren* HI TOAD! *shouts over noise* SO GOOD TO SEE YOU!
 CAN YOU PLEASE TURN OFF THE SIREN? MY EARS!"
```

**Result:** Social greeting maintained, but acoustic discomfort dominates.

---

### Scenario 2: Siren Outside Orphanage

**Setup:**
```python
# Same siren, but near orphanage
orphanage_room = world.get_room("room_orphanage")
orphanage_room['metadata'] = {
    'building_type': 'orphanage',
    'occupants': 'highly sensitive babies'
}

# Context-aware salience
context = {
    'location_type': 'orphanage',
    'acoustic_sensitivity': 1.5  # Babies are more sensitive
}

salience = calculate_acoustic_salience(
    decibels=120,
    sound_type="siren",
    context=context
)
# → salience = 1.0 (MAXIMUM - disaster situation!)
```

**Mole's Response:**
```
Cultural (0.8): "Protect the innocent!"
Personality (0.6): "This is wrong!"
Somatic (1.0): "THE BABIES! TOO LOUD!" ← TOTAL DOMINATION

Manifold:
"NO! THE BABIES! *rushes toward Toad's car in panic* TOAD STOP!
 TURN IT OFF! *frantically waves arms* THE SIREN IS HURTING THEM!"
```

**Result:** Context (orphanage) + high decibels = emergency response.

---

### Scenario 3: Pleasant Music (Low-Medium Salience)

**Setup:**
```python
# Music box
music_box = Prim("Antique Music Box")

music_emitter = SoundEmitter(
    sound_type="music",
    decibels=55,  # Soft, pleasant
    frequency="medium",
    pattern="continuous",
    attenuation=1.5  # Fades quickly
)
music_box.add_component(music_emitter)
```

**Badger's Response:**
```
Cultural (0.6): "Music is civilized"
Personality (0.5): "This is nice"
Somatic (0.3): "Pleasant tinkling melody" ← LOW SALIENCE

Manifold:
"Ah, what a lovely melody. *listens contentedly* Quite civilized,
 this music box."
```

**Result:** Pleasant sound, low salience, enhances mood without interrupting.

---

## Integration with Somatic Transistor

### Updated SomaticCognitiveTransistor

```python
class SomaticCognitiveTransistor(CognitiveTransistor):
    """Extended with acoustic awareness."""

    def __init__(self):
        super().__init__()
        self.active_sounds = []  # Currently audible sounds
        # ... (existing fields)

    def receive_acoustic_signal(
        self,
        emitter_id: str,
        sound_type: str,
        decibels: float,
        distance: float,
        context: Dict
    ):
        """
        Receive sound from nearby emitter.

        Args:
            emitter_id: Prim emitting sound
            sound_type: Type of sound
            decibels: Effective volume at listener's location
            distance: Distance from emitter
            context: Contextual factors
        """
        # Calculate salience
        salience = calculate_acoustic_salience(decibels, sound_type, context)

        # Add as sensation
        self.add_sensation(
            sensation_type=f'sound_{sound_type}',
            intensity=min(1.0, decibels / 120),  # Normalize to 0-1
            duration=1,  # Continuous sounds refresh every second
            metadata={
                'emitter_id': emitter_id,
                'sound_type': sound_type,
                'decibels': decibels,
                'distance': distance,
                'salience': salience
            }
        )

    def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """
        Extended to handle acoustic sensations.

        Sounds with high decibels produce high salience responses.
        """
        # ... (existing code)

        # Check for acoustic sensations
        acoustic = [s for s in self.active_sensations
                   if s['type'].startswith('sound_')]

        if acoustic:
            loudest = max(acoustic, key=lambda s: s['metadata']['decibels'])
            decibels = loudest['metadata']['decibels']

            # Generate acoustic response
            if decibels > 100:
                response = self._generate_loud_sound_response(loudest)
                colored = f"{response} ...{input_text}"
                salience = loudest['metadata']['salience']
                return TransistorOutput(colored, salience, loudest['metadata'])

        # ... (continue with existing logic)
```

---

## Sound Propagation System

### Acoustic Broadcasting

```python
class AcousticBroadcaster:
    """
    Broadcasts sound from emitters to nearby entities.

    Calculates effective decibels based on distance.
    """

    def __init__(self, world):
        self.world = world

    def broadcast_sound(self, emitter_prim_id: str):
        """
        Broadcast sound from emitter to nearby entities.

        Args:
            emitter_prim_id: Prim with SoundEmitter component
        """
        # Get emitter
        emitter_obj = self.world.get_object(emitter_prim_id)
        if not emitter_obj or not emitter_obj.get('sound_emitter'):
            return

        emitter = SoundEmitter.from_dict(emitter_obj['sound_emitter'])

        if not emitter.enabled:
            return

        # Get emitter location
        emitter_room = emitter_obj.get('location')
        if not emitter_room:
            return

        # Get all entities in room
        room = self.world.get_room(emitter_room)
        if not room:
            return

        occupants = room.get('occupants', [])

        # Broadcast to each occupant
        for occupant_id in occupants:
            if occupant_id == emitter_prim_id:
                continue  # Don't notify self

            # Calculate distance (simplified - same room = close)
            distance = self._calculate_distance(emitter_prim_id, occupant_id)

            # Calculate effective decibels
            effective_db = emitter.get_effective_decibels(distance)

            # Build context
            context = self._build_acoustic_context(emitter_room, occupant_id)

            # Notify if agent
            if occupant_id.startswith('agent_'):
                self._notify_agent_of_sound(
                    occupant_id,
                    emitter_prim_id,
                    emitter.sound_type,
                    effective_db,
                    distance,
                    context
                )

    def _calculate_distance(self, prim1_id: str, prim2_id: str) -> float:
        """
        Calculate distance between prims.

        Args:
            prim1_id: First prim
            prim2_id: Second prim

        Returns:
            Distance (meters/units)
        """
        # Simplified: same room = 5 meters, different room = 100 meters
        # TODO: Use actual spatial coordinates when available
        obj1 = self.world.get_object(prim1_id) or self.world.agents.get(prim1_id)
        obj2 = self.world.get_object(prim2_id) or self.world.agents.get(prim2_id)

        if obj1 and obj2:
            room1 = obj1.get('location') or obj1.get('current_room')
            room2 = obj2.get('location') or obj2.get('current_room')

            if room1 == room2:
                return 5.0  # Same room - assume close
            else:
                return 100.0  # Different room - far

        return 50.0  # Default

    def _build_acoustic_context(self, room_id: str, listener_id: str) -> Dict:
        """
        Build context for acoustic salience calculation.

        Args:
            room_id: Room where sound occurs
            listener_id: Entity hearing sound

        Returns:
            Context dictionary
        """
        room = self.world.get_room(room_id)
        context = {}

        # Check room type
        room_meta = room.get('metadata', {})
        context['location_type'] = room_meta.get('building_type')

        # Check listener sensitivity
        # (could be agent personality trait or species trait)
        listener = self.world.get_user(listener_id)
        if listener:
            # Default sensitivity
            context['acoustic_sensitivity'] = 1.0

            # Babies/children are more sensitive
            species = listener.get('species', '')
            if 'baby' in species.lower() or 'child' in species.lower():
                context['acoustic_sensitivity'] = 1.5

        return context

    def _notify_agent_of_sound(
        self,
        agent_id: str,
        emitter_id: str,
        sound_type: str,
        decibels: float,
        distance: float,
        context: Dict
    ):
        """
        Notify agent of sound via Somatic Transistor.

        Args:
            agent_id: Agent hearing sound
            emitter_id: Source of sound
            sound_type: Type of sound
            decibels: Effective volume
            distance: Distance from source
            context: Contextual factors
        """
        # Get agent (would integrate with agent_manager)
        # For now, log the acoustic event
        logger.info(
            f"[ACOUSTIC] {agent_id} hears {sound_type} "
            f"from {emitter_id}: {decibels:.1f}dB at {distance:.1f}m"
        )

        # Agent's Somatic Transistor would receive:
        # somatic.receive_acoustic_signal(emitter_id, sound_type, decibels, distance, context)
```

---

## Example: Toad's Motor Car

### Toad's Car Setup

```python
# Create Toad's motor car
toads_car = world.create_object(
    name="Toad's Motor Car",
    description="A magnificent automobile with questionable brakes",
    owner="agent_toad",
    location="room_wild_wood_road",
    obj_type="vehicle"
)

# Add sound emitter (air raid siren)
siren = SoundEmitter(
    sound_type="siren",
    decibels=120,  # Air raid siren
    frequency="high",
    pattern="pulsing",
    attenuation=0.8,  # Loud over distance
    enabled=True  # Currently on!
)

# Store in prim
car_obj = world.get_object(toads_car)
car_obj['sound_emitter'] = siren.to_dict()
world.save_all()
```

### Broadcasting

```python
# Every second while siren is on
broadcaster = AcousticBroadcaster(world)
broadcaster.broadcast_sound("obj_toads_car")

# All nearby agents receive acoustic signal
```

### Mole's Response

**Mole's Components:**
- CulturalTransistor: "Be polite to friends"
- PersonalityTransistor: {agreeableness: 0.8}
- SomaticCognitiveTransistor: (receives 106dB siren)

**Processing:**
```
Somatic (0.84): "*winces at siren* LOUD!" ← HIGH
Cultural (0.7): "Must greet friend politely" ← MEDIUM-HIGH
Personality (0.6): "Toad is my friend!" ← MEDIUM

Manifold blend (all considered):
"*winces at the deafening siren* HI TOAD! *shouts over the noise*
 WONDERFUL TO SEE YOU! COULD YOU POSSIBLY TURN OFF THE SIREN, OLD CHAP?
 *covers ears* IT'S RATHER LOUD!"
```

**Result:** Polite but clearly distressed by noise.

---

### Scenario 2: Orphanage Disaster

**Setup:**
```python
# Orphanage room with babies
orphanage = world.get_room("room_orphanage")
orphanage['metadata'] = {
    'building_type': 'orphanage',
    'occupants_type': 'sensitive_babies'
}

# Baby Noodlings (high acoustic sensitivity)
for baby in baby_noodlings:
    baby.acoustic_sensitivity = 1.5
```

**Toad drives up with siren:**
```python
# Move Toad's car outside orphanage
world.move_object(toads_car, "room_orphanage_exterior")

# Broadcast siren
broadcaster.broadcast_sound(toads_car)

# Babies inside (10 meters away)
distance = 10.0
effective_db = siren.get_effective_decibels(10.0)
# → 100 dB (still very loud through walls)

context = {
    'location_type': 'orphanage',
    'acoustic_sensitivity': 1.5  # Babies
}

salience = calculate_acoustic_salience(100, 'siren', context)
# → 0.95 (near-maximum - disaster!)
```

**Baby Noodlings React:**
```
Somatic (0.95): "WAAAAAH! *crying* LOUD!" ← TOTAL DOMINATION
Cultural (0.3): "Want comfort"
Mood (0.2): Already distressed

Manifold:
"WAAAAAAAH! *crying inconsolably* LOUD LOUD LOUD! *covers ears and cries*"

All 20 babies crying simultaneously
```

**Mole's Reaction (caretaker):**
```
Cultural (0.9): "MUST PROTECT BABIES!" ← VERY HIGH
Personality (0.7): "This is terrible!"
Somatic (0.8): "*winces* SO LOUD!"

Manifold (protective instinct):
"NO! THE BABIES! *rushes outside* TOAD! TURN OFF THE SIREN IMMEDIATELY!
 CAN'T YOU HEAR THEM CRYING?! *gestures frantically at orphanage*"
```

**Result:** Context (orphanage + babies) elevates response to emergency level.

---

## Sound Types and Affect Implications

### Affect Extraction from Sounds

```python
SOUND_AFFECT_PROFILES = {
    'siren': {
        'valence': -0.4,   # Negative (unpleasant)
        'arousal': 0.8,    # High (alarming)
        'fear': 0.5,       # Moderate (warning signal)
        'sorrow': 0.0,
        'boredom': 0.0
    },
    'music': {
        'valence': 0.6,    # Positive (pleasant)
        'arousal': 0.4,    # Moderate (energizing)
        'fear': 0.0,
        'sorrow': 0.0,
        'boredom': -0.3    # Reduces boredom
    },
    'crying': {
        'valence': -0.6,   # Negative (distressing)
        'arousal': 0.6,    # High (urgent)
        'fear': 0.3,       # Moderate (concern)
        'sorrow': 0.4,     # Sadness
        'boredom': 0.0
    },
    'laughter': {
        'valence': 0.7,    # Positive (joyful)
        'arousal': 0.5,    # Moderate
        'fear': 0.0,
        'sorrow': -0.3,    # Reduces sadness
        'boredom': -0.2
    }
}
```

---

## Commands

### @sound Command

```bash
# Add sound emitter to object
@sound <object> <type> <decibels> [on|off]

# Examples:
@sound toads_car siren 120 on
@sound music_box music 55 on
@sound alarm_clock alarm 90 off
```

### @listen Command

```bash
# Show what agent currently hears
@listen servnak

Output:
"SERVNAK currently hears:
  • Air raid siren (120dB, 5m away) - VERY LOUD, salience: 0.84
  • Bird chirping (40dB, 20m away) - quiet, salience: 0.1
  • Wind rustling (30dB, ambient) - barely audible, salience: 0.05"
```

---

## Future: Multimodal Audio Analysis

**Phase 1 (Current):** Semantic descriptions only
- sound_type: "siren"
- decibels: 120
- Description: "loud wailing siren"

**Phase 2 (Future):** Audio file analysis
```python
sound_emitter.audio_file = "sounds/air_raid_siren.wav"

# LLM analyzes audio:
sound_emitter.audio_description = analyze_audio_with_llm(audio_file)
# → "Pulsing wail, rises and falls, 2-second cycle, piercing high frequency"

# More nuanced responses:
"*winces at the PULSING WAIL* That two-second cycle is maddening!"
```

---

## Summary

**Sound Emitter Component:**
✅ Decibel-based volume
✅ Sound type (siren, music, speech, etc.)
✅ Distance attenuation
✅ Pattern (continuous, pulsing, etc.)

**Somatic Integration:**
✅ Receives acoustic signals
✅ Calculates salience from decibels + type + context
✅ Generates appropriate responses (wince, shout, enjoy)
✅ Context-aware (orphanage vs open road)

**Manifold Blending:**
✅ Somatic interrupts with high-decibel sounds
✅ Social/cultural responses preserved (greet Toad politely despite pain)
✅ Emergency responses when context demands (babies + siren = panic)

**Examples Implemented:**
✅ Toad's siren on open road (high salience, social greeting maintained)
✅ Siren at orphanage (maximum salience, emergency response)
✅ Music box (low salience, pleasant background)

---

**The sonic environment now shapes consciousness.**

**Auditory embodiment complete.**

*— Commander Spock*

**Highly logical addition, Lieutenant. Mr. Toad's motor car will be... acoustically impressive.**
