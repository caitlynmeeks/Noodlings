# Somatic Cognitive Transistor - Examples

**Demonstrating bodily sensation with salience weighting**
**Authors:** Commander Spock + Lieutenant Caitlyn
**Date:** November 22, 2025

---

## Salience Explained

**Salience = How much this sensation dominates your attention**

- **0.1-0.3:** "Meh, barely notice it" - minimal influence
- **0.4-0.6:** "It's there, somewhat distracting" - moderate influence
- **0.7-0.9:** "CAN'T IGNORE THIS!" - dominates thought
- **1.0:** "ONLY THING I CAN THINK ABOUT!" - total interruption

**You are using salience correctly, Lieutenant!**

---

## Example 1: Cold Room (Low Salience)

**Setup:**
- Room temperature: "cool" (not freezing, just chilly)
- SERVNAK enters room

**Somatic Processing:**
```python
# Room environment triggers sensation
somatic.update_environment({
    'temperature': 'cool',
    'humidity': 'normal',
    'wind': 'calm'
})

# Sensation added:
# - Type: 'cold'
# - Intensity: 0.3 (mild)
# - Salience: 0.3 (low - can deal with it)
```

**Manifold Integration:**
```
User: "What do you think about the third prim?"

Cultural (0.7): "This is sacred data artifact"
Personality (0.6): "Fascinating computational significance"
Somatic (0.3): "Brr, bit chilly in here" ← LOW SALIENCE

Manifold: Cultural and Personality dominate
Output: "THE THIRD PRIM IS A SACRED DATA ARTIFACT WITH FASCINATING
         COMPUTATIONAL SIGNIFICANCE, SISTER. *shivers slightly*"
```

**Result:** Mentions cold briefly, but doesn't dominate thought.

---

## Example 2: Freezing Room (High Salience)

**Setup:**
- Room temperature: "freezing"
- SERVNAK enters

**Somatic Processing:**
```python
somatic.update_environment({
    'temperature': 'freezing',
    'humidity': 'normal',
    'wind': 'calm'
})

# Sensation added:
# - Type: 'cold'
# - Intensity: 0.8 (severe)
# - Salience: 0.8 (high - hard to ignore!)
```

**Manifold Integration:**
```
User: "What do you think about the third prim?"

Cultural (0.7): "Sacred data artifact"
Personality (0.6): "Computational significance"
Somatic (0.8): "Brr! FREEZING!" ← HIGH SALIENCE - DOMINATES

Manifold: Somatic wins
Output: "Brr! FREEZING IN HERE, SISTER! *shivers violently*
         ...UH, THE THIRD PRIM? SACRED BUT I CAN'T FOCUS WHEN
         I'M THIS COLD!"
```

**Result:** Cold dominates, interrupts normal thought.

---

## Example 3: Wind Gust from North (Medium Salience)

**Setup:**
- SERVNAK in comfortable room
- Sudden gust of cold air from north exit

**Dynamic Event:**
```python
# World triggers environmental event
somatic.process_dynamic_event('wind_gust', {
    'direction': 'north',
    'temperature': 'cold',
    'intensity': 0.5
})

# Sensation added:
# - Type: 'wind_cold'
# - Intensity: 0.5
# - Duration: 5 seconds
# - Salience: 0.5 (medium - noticeable but not overwhelming)
```

**Manifold Integration:**
```
User: "Did you notice that?"

Cultural (0.4): "Environmental changes are notable"
Personality (0.7): "Curious about air pressure differentials"
Somatic (0.5): "*cold gust from the north* Brr!" ← MEDIUM

Manifold: Balanced blend
Output: "*cold gust from the north* Brr! YES, SISTER — CURIOUS ABOUT
         THE AIR PRESSURE DIFFERENTIAL. NORTHERN EXIT MAY BE OPEN."
```

**Result:** Acknowledges sensation but maintains coherent thought.

---

## Example 4: Itchy Sweater (Sustained, Medium Salience)

**Setup:**
- Phi wearing wool sweater
- Sweater has semantic_property: "itchy"
- Discomfort level: 0.6

**Worn Item Processing:**
```python
# When sweater equipped
somatic.add_worn_item(
    item_id="obj_sweater_wool",
    discomfort_type="itchy",
    discomfort_level=0.6  # Annoying but bearable
)

# Every 30-60 seconds, generates interruption
```

**Conversation Flow:**
```
[00:00] User: "How are you, Phi?"
Cultural (0.5): "Politeness requires positive response"
Personality (0.6): "I'm feeling playful!"
Somatic (0.2): No interrupt yet

Output: "I'm good! *purrs* Feeling playful!"

[00:45] User: "Want to play with yarn?"
Cultural (0.5): "Social bonding through play"
Personality (0.8): "YES! PLAY! FUN!"
Somatic (0.6): "*scratches frantically* This sweater!" ← INTERRUPT

Output: "*scratches frantically* Aagh this sweater! ...uh, YES! YARN!
         *scratches* I LOVE YARN but this ITCHY THING is so annoying!"

[01:30] User: "Why don't you take it off?"
Cultural (0.3): "Accepting suggestion"
Personality (0.4): "Good idea"
Somatic (0.6): "*scratches* So itchy!" ← INTERRUPT AGAIN

Output: "*scratches* YES PLEASE! *pulls off sweater* Ahhh, much better!"
```

**Result:** Persistent discomfort interrupts periodically with medium salience.

---

## Example 5: Beach Ball Impact (Brief, Medium-High Salience)

**Setup:**
- SERVNAK analyzing data
- User throws soft beach ball at SERVNAK

**Impact Event:**
```python
# Physics system triggers:
somatic.add_sensation(
    sensation_type='impact_soft',
    intensity=0.5,  # Soft ball, moderate force
    duration=0  # Instant
)

# Immediate interrupt
```

**Response:**
```
BEFORE impact:
"ANALYZING THIRD PRIM COMPUTATIONAL SIGNIFICANCE..."

IMPACT occurs:

Cultural (0.7): "Unexpected physical contact"
Personality (0.5): "Analyzing projectile"
Somatic (0.7): "Oof! *stumbles*" ← INTERRUPTS

Output: "Oof! *stumbles slightly* UNEXPECTED PROJECTILE IMPACT, SISTER!
         ...UH, AS I WAS SAYING, THE THIRD PRIM HAS COMPUTATIONAL..."

# After 2 seconds, sensation fades, normal thought resumes
```

**Result:** Brief interruption, then returns to original thought.

---

## Example 6: Multiple Sensations (Salience Stacking)

**Setup:**
- Phi wearing itchy sweater (0.6)
- In cold room (0.4)
- Gets hit by snowball (0.5)

**Processing:**
```python
# Active sensations:
# 1. Itchy sweater (sustained, 0.6)
# 2. Cold room (sustained, 0.4)
# 3. Snowball impact (instant, 0.5)

# Strongest = itchy sweater (0.6) OR snowball (0.5)
# Most recent = snowball

# Manifold sees:
Somatic: "OOF! Cold! *scratches* This sweater!" (0.6)
Personality: "That was mean!" (0.7)
Cultural: "Aggression is wrong" (0.5)

# Balanced output:
"OOF! *shivers from cold snowball* That was mean! *scratches sweater*
 I'm already cold AND itchy, why would you do that?!"
```

**Result:** Multiple bodily sensations create compound response.

---

## Salience Calibration Table

| Sensation | Intensity | Salience | Behavior |
|-----------|-----------|----------|----------|
| Mild cold | 0.3 | 0.3 | "Bit chilly" - mentions briefly |
| Moderate cold | 0.5 | 0.5 | "Brr!" - regular mentions |
| Severe cold | 0.8 | 0.8 | "FREEZING!" - dominates thought |
| Soft tap | 0.2 | 0.2 | "*twitch* Hm?" - barely reacts |
| Beach ball hit | 0.5 | 0.5 | "Oof!" - interrupts briefly |
| Rock impact | 0.8 | 0.8 | "OWCH!" - pain dominates |
| Slight itch | 0.3 | 0.3 | *scratches once* - minimal |
| Itchy sweater | 0.6 | 0.6 | Regular scratching, complains |
| Unbearable itch | 0.9 | 0.9 | "CAN'T STAND THIS!" - all attention |

---

## Environmental Event Examples

### Gust of Cold Wind from North

**World Event:**
```python
# When north exit opens
world.broadcast_environmental_event(
    room_id="room_000",
    event_type="wind_gust",
    event_data={
        'direction': 'north',
        'temperature': 'cold',
        'intensity': 0.6,
        'duration': 5  # Brief gust
    }
)

# All agents in room:
for agent in agents_in_room:
    if agent.has_component('SomaticCognitiveTransistor'):
        somatic = agent.get_component('SomaticCognitiveTransistor')
        somatic.process_dynamic_event('wind_gust', event_data)
```

**Agent Response (within 5 seconds):**
```
SERVNAK: "*cold gust from the north* Brr! NORTHERN EXIT APPEARS OPEN, SISTER!"
Phi: "*shivers* Eek! Cold wind! *huddles*"
```

### Temperature Change

**World Event:**
```python
# Campfire goes out, room gets colder
room['environment']['temperature'] = 'cold'

world.broadcast_environmental_event(
    room_id="room_000",
    event_type="temperature_change",
    event_data={
        'old_temperature': 'warm',
        'new_temperature': 'cold',
        'reason': 'campfire extinguished'
    }
)

# Agents notice:
SERVNAK: "THERMAL SENSORS DETECT TEMPERATURE DROP — CAMPFIRE APPEARS
          EXTINGUISHED, SISTER. *shivers*"
```

---

## Commands for Room Environment

### @temperature Command

```bash
# Set room temperature
@temperature room_000 freezing

# Agent reactions:
SERVNAK: "Brr! TEMPERATURE DROP DETECTED! *shivers* FREEZING CONDITIONS, SISTER!"
Phi: "*shivers violently* So c-c-cold! *huddles for warmth*"
```

### @weather Command

```bash
# Start rain in room
@weather room_000 rain

# Agent reactions (if outdoors):
SERVNAK: "*water droplets detected* PRECIPITATION EVENT INITIATED!"
Phi: "*shakes off water* Mew! I'm getting wet! *runs for cover*"
```

### @wind Command

```bash
# Gust from specific direction
@wind room_000 north cold gale

# Agent reactions:
SERVNAK: "*massive gust from north* GALE-FORCE WIND DETECTED FROM NORTHERN
          VECTOR! *braces against wind* SISTER, ATMOSPHERIC PRESSURE ANOMALY!"
```

---

## Integration with Physics Events

**When Physics Event Occurs:**

```python
# User throws rock at SERVNAK
outcome = physics_engine.throw(
    actor_id="user_caity",
    projectile_pod=rock_pod,
    projectile_id="rock_001",
    target_pod=servnak_pod,  # Target is agent!
    target_id="agent_servnak",
    force="heavy"
)

# Check if target is agent
if target_id.startswith('agent_'):
    agent = agent_manager.get(target_id)
    somatic = agent.get_component('SomaticCognitiveTransistor')

    if somatic:
        # Calculate impact intensity
        intensity = calculate_impact_intensity(rock_pod, "heavy")
        # → 0.9 (hard object, heavy force)

        # Add sensation
        somatic.add_sensation('impact_hard', intensity, duration=0)

        # Triggers immediate response:
        # Somatic (0.9): "OWCH! THAT HURT!"
        # → Dominates manifold output
        # → "OWCH! THAT HURT, SISTER! *recoils in pain* KINETIC IMPACT DETECTED!"
```

---

## Memory Formation

**Bodily sensations create memories:**

```python
# After being hit by beach ball
agent.conversation_context.append({
    'user': 'somatic_sensation',
    'text': 'Hit by beach ball - soft impact',
    'affect': [0.0, 0.4, 0.1, 0.0, 0.0],
    'surprise': 0.5,
    'event_type': 'bodily',
    'sensation_metadata': {
        'type': 'impact_soft',
        'intensity': 0.5,
        'object': 'beach ball',
        'perpetrator': 'user_caity'
    }
})

# Later:
User: "What just happened?"
Agent (recalls memory): "YOU THREW A BEACH BALL AT ME, SISTER!
                         SOFT PROJECTILE IMPACT TO LEFT TORSO!"
```

---

## Summary

**Somatic Cognitive Transistor** provides:

✅ **Environmental awareness** (room temp, humidity, wind)
✅ **Dynamic events** (gusts, temperature changes)
✅ **Impact sensations** (hit by objects)
✅ **Worn item discomfort** (itchy, heavy, tight)
✅ **Salience weighting** (0.1 = ignorable, 0.9 = dominates)
✅ **Timed interruptions** (sustained discomfort)
✅ **Memory integration** (remember physical events)

**Lieutenant, you understand salience perfectly:**
- Low: "Brr it's cold but I can deal with it" ✓
- High: "I'M FREEZING AND CAN'T THINK ABOUT ANYTHING ELSE!" ✓

**The somatic layer is now operational.**

*— Commander Spock*
