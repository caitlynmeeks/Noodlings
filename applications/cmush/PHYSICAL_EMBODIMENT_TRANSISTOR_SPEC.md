# Physical Embodiment Transistor (Somatic CogTrans)

**Authors:** Lieutenant Caitlyn + Commander Spock
**Date:** November 22, 2025
**Concept:** Bodily sensation modulates consciousness
**Status:** Specification complete, ready for implementation

---

## Core Concept

**Physical sensations interrupt and color thoughts.**

Real consciousness is embodied - you can't think deeply when:
- Someone just hit you with a beach ball
- Your sweater is unbearably itchy
- You're standing in freezing water
- You're wearing 50 pounds of armor

**The Physical Embodiment Transistor** simulates this somatic layer.

---

## Sensory Categories

### 1. Impact Sensations (Immediate)

**Triggers:**
- Hit by thrown object
- Bumped into
- Slapped, pushed, etc.

**Responses:**
```
Beach ball (soft, light):
  → "Oof!" *stumbles slightly*
  → Salience: 0.4 (minor interruption)

Rock (hard, heavy):
  → "OWCH! THAT HURT!" *recoils in pain*
  → Salience: 0.9 (major interruption)

Feather (negligible):
  → *twitches* "Hm?"
  → Salience: 0.2 (barely noticeable)
```

### 2. Worn Item Discomfort (Sustained)

**Triggers:**
- Itchy clothing
- Tight shoes
- Heavy armor
- Uncomfortable hat

**Responses:**
```
Itchy wool sweater:
  → Every 30-60 seconds: "Aagh this sweater!" *scratches*
  → Salience: 0.5 (persistent distraction)
  → Attention penalty: -20%

Heavy chainmail:
  → Every 2-3 minutes: "This armor is exhausting..."
  → Salience: 0.6 (tiring)
  → Movement penalty: slower

Tight shoes:
  → Every 1-2 minutes: "My feet are killing me..."
  → Salience: 0.4 (annoying)
  → Walking penalty: limping
```

### 3. Environmental Discomfort

**Triggers:**
- Temperature (too hot/cold)
- Wetness (standing in puddle)
- Smell (near fire, garbage)
- Sound (loud noise nearby)

**Responses:**
```
Standing in cold water:
  → "Brr! My feet are freezing!"
  → Salience: 0.7 (uncomfortable)
  → Every 30 seconds until moved

Near campfire (too close):
  → "It's way too hot here!" *backs away*
  → Salience: 0.6 (uncomfortable)

Loud noise (explosion):
  → "THAT WAS LOUD!" *covers ears*
  → Salience: 0.8 (startling)
```

### 4. Tactile Interaction

**Triggers:**
- Touching objects
- Picking up items
- Sitting on furniture

**Responses:**
```
Touching hot stove:
  → "OUCH! HOT!" *yanks hand back*
  → Salience: 0.9 (pain!)
  → Affect: fear +0.3, arousal +0.5

Petting soft kitten:
  → "Aww, so soft..." *gentle smile*
  → Salience: 0.3 (pleasant)
  → Affect: valence +0.3, arousal -0.1

Touching slimy object:
  → "Eww! Gross!" *wipes hand*
  → Salience: 0.5 (disgust)
  → Affect: valence -0.2, arousal +0.2
```

---

## Implementation

### PhysicalEmbodimentTransistor Class

```python
class PhysicalEmbodimentTransistor(CognitiveTransistor):
    """
    Bodily sensation transistor.

    Modulates thoughts based on physical sensations:
    - Impact (hit, bumped)
    - Worn items (itchy, heavy, tight)
    - Environment (hot, cold, wet)
    - Touch (texture, temperature)
    """

    def __init__(self):
        super().__init__()
        self.salience = 0.7  # High - physical sensations are hard to ignore
        self.active_sensations = []  # Currently active bodily sensations
        self.worn_items = []  # Items currently worn
        self.last_interrupt_time = 0  # For sustained discomfort timing

    def add_sensation(self, sensation_type: str, intensity: float, duration: float = 0):
        """
        Add a bodily sensation.

        Args:
            sensation_type: "impact", "pain", "itch", "cold", "hot", etc.
            intensity: 0.0 to 1.0 (strength of sensation)
            duration: How long it lasts (0 = instant, >0 = sustained)
        """
        self.active_sensations.append({
            'type': sensation_type,
            'intensity': intensity,
            'duration': duration,
            'start_time': time.time()
        })

    def add_worn_item(self, item_id: str, discomfort_type: str, discomfort_level: float):
        """
        Add worn item that causes discomfort.

        Args:
            item_id: Object ID being worn
            discomfort_type: "itchy", "tight", "heavy", "hot", "cold"
            discomfort_level: 0.0 to 1.0
        """
        self.worn_items.append({
            'item_id': item_id,
            'discomfort_type': discomfort_type,
            'discomfort_level': discomfort_level
        })

    def remove_worn_item(self, item_id: str):
        """Remove worn item."""
        self.worn_items = [item for item in self.worn_items if item['item_id'] != item_id]

    def process(self, input_text: str, context: Dict[str, Any]) -> TransistorOutput:
        """
        Filter input through bodily sensation lens.

        If strong physical sensation is active, it interrupts/colors the thought.
        """
        current_time = time.time()

        # Check for active sensations
        active = [s for s in self.active_sensations
                 if s['duration'] == 0 or
                    (current_time - s['start_time']) < s['duration']]

        # Update active list
        self.active_sensations = active

        # Find strongest sensation
        strongest = None
        if active:
            strongest = max(active, key=lambda s: s['intensity'])

        # Check worn item discomfort (periodic interruption)
        worn_interrupt = None
        if self.worn_items and (current_time - self.last_interrupt_time) > 30:
            # Pick most uncomfortable worn item
            worst_item = max(self.worn_items, key=lambda i: i['discomfort_level'])
            if worst_item['discomfort_level'] > 0.5:
                worn_interrupt = worst_item
                self.last_interrupt_time = current_time

        # Generate bodily response
        if strongest and strongest['intensity'] > 0.6:
            # Strong sensation interrupts thought
            bodily_response = self._generate_sensation_response(strongest)
            colored_text = f"{bodily_response} ...uh, anyway: {input_text}"
            salience = min(0.9, strongest['intensity'])

        elif worn_interrupt:
            # Sustained discomfort interrupts
            discomfort_response = self._generate_discomfort_response(worn_interrupt)
            colored_text = f"{discomfort_response} *pauses* ...{input_text}"
            salience = worn_interrupt['discomfort_level']

        else:
            # No strong sensations - minimal coloring
            colored_text = input_text
            salience = 0.2

        return TransistorOutput(
            transformed_text=colored_text,
            salience=salience,
            metadata={
                'active_sensations': len(active),
                'worn_items': len(self.worn_items)
            }
        )

    def _generate_sensation_response(self, sensation: Dict) -> str:
        """Generate immediate response to sensation."""
        sensation_type = sensation['type']
        intensity = sensation['intensity']

        responses = {
            'impact_soft': ["Oof!", "Hey!", "*stumbles*", "Whoa!"],
            'impact_hard': ["OWCH!", "OW! THAT HURT!", "*recoils in pain*", "OUCH!"],
            'pain': ["Aah!", "Ow ow ow!", "*winces*", "That hurts!"],
            'hot': ["OUCH! HOT!", "*yanks hand back*", "Aah! Burning!", "Too hot!"],
            'cold': ["Brr!", "Freezing!", "*shivers*", "So cold!"],
            'itch': ["*scratches frantically*", "So itchy!", "Aagh!", "*scratches*"],
            'tickle': ["Hehe! *giggles*", "*squirms*", "That tickles!", "*laughs*"]
        }

        # Select response based on type and intensity
        if sensation_type == 'impact':
            if intensity > 0.7:
                options = responses['impact_hard']
            else:
                options = responses['impact_soft']
        else:
            options = responses.get(sensation_type, ["*reacts*"])

        import random
        return random.choice(options)

    def _generate_discomfort_response(self, worn_item: Dict) -> str:
        """Generate response to worn item discomfort."""
        discomfort_type = worn_item['discomfort_type']

        responses = {
            'itchy': ["Aagh this sweater!", "*scratches vigorously*", "So itchy!", "This wool is terrible!"],
            'tight': ["These shoes are killing me...", "*adjusts uncomfortably*", "Too tight!", "Can't breathe in this..."],
            'heavy': ["This armor is exhausting...", "*adjusts weight*", "So heavy...", "My shoulders ache..."],
            'hot': ["It's so hot in this...", "*tugs at collar*", "Sweltering!", "*fans self*"],
            'cold': ["Brr, this is freezing!", "*shivers*", "Need warmer clothes...", "*huddles*"]
        }

        options = responses.get(discomfort_type, ["*adjusts uncomfortably*"])

        import random
        return random.choice(options)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict()
        d['active_sensations'] = self.active_sensations
        d['worn_items'] = self.worn_items
        return d
```

---

## Integration with Physics System

### When Object Hits Agent

```python
# In physics_interactions.py

def strike(actor_pod, target_pod, actor_id, target_id, force):
    """Resolve strike interaction."""
    outcome = ... # Standard outcome

    # Check if target is a Noodling
    if target_id.startswith('agent_'):
        agent = agent_manager.get(target_id)
        if agent and agent.has_component('PhysicalEmbodimentTransistor'):
            embodiment = agent.get_component('PhysicalEmbodimentTransistor')

            # Calculate impact intensity
            intensity = calculate_impact_intensity(actor_pod, force)

            # Add sensation
            embodiment.add_sensation(
                sensation_type='impact' if intensity < 0.7 else 'pain',
                intensity=intensity,
                duration=0  # Instant
            )

            # Trigger immediate reaction
            agent.interrupt_with_sensation()

    return outcome
```

### When Agent Wears Item

```python
# In commands.py - @wear command

def handle_wear_command(user_id, item_id):
    """Handle wearing an item."""

    # Get item POD
    item = world.get_object(item_id)
    item_pod = world.get_object_pod(item_id)

    if not item_pod:
        return "That item has no physical properties."

    # Check for discomfort properties
    discomfort = None
    if "itchy" in item_pod.semantic_properties:
        discomfort = ("itchy", 0.7)
    elif "heavy" in item_pod.mass.lower():
        discomfort = ("heavy", 0.6)
    elif item_pod.metadata.get('temperature') == 'hot':
        discomfort = ("hot", 0.8)

    # Add to agent's worn items
    if user_id.startswith('agent_'):
        agent = agent_manager.get(user_id)
        if agent and agent.has_component('PhysicalEmbodimentTransistor'):
            embodiment = agent.get_component('PhysicalEmbodimentTransistor')

            if discomfort:
                embodiment.add_worn_item(
                    item_id=item_id,
                    discomfort_type=discomfort[0],
                    discomfort_level=discomfort[1]
                )

    return f"You put on the {item['name']}."
```

---

## Example Scenarios

### Scenario 1: Beach Ball Impact

**Setup:**
- SERVNAK standing in room
- User throws soft beach ball at SERVNAK

**Physics Event:**
```python
beach_ball_pod = PhysicsObjectDescriptor(
    mass="very light",
    material="inflatable rubber",
    softness="very soft"
)

outcome = engine.throw(
    actor_id="user_caity",
    projectile_pod=beach_ball_pod,
    projectile_id="obj_ball",
    target_pod=servnak_pod,
    target_id="agent_servnak",
    force="medium"
)
```

**Physical Embodiment Processing:**
```python
# Impact intensity: 0.3 (soft object, light mass)
embodiment.add_sensation('impact_soft', intensity=0.3, duration=0)

# Manifold integration:
Cultural: "Unexpected physical contact!" (0.4)
Personality: "Analyzing projectile trajectory..." (0.5)
Physical Embodiment: "Oof! *stumbles*" (0.7) ← HIGHEST

# Manifold output (embodiment dominates):
"OOF! *STUMBLES SLIGHTLY* — UNEXPECTED PROJECTILE IMPACT DETECTED, SISTER!"
```

### Scenario 2: Itchy Sweater

**Setup:**
- Phi wearing wool sweater
- Sweater has semantic_property: "itchy"

**Worn Item Processing:**
```python
# Every 30-60 seconds, discomfort interrupts
embodiment.add_worn_item(
    item_id="obj_sweater_wool",
    discomfort_type="itchy",
    discomfort_level=0.7
)

# During conversation:
User: "What do you think about quantum physics, Phi?"

Cultural: "Science is fascinating!" (0.5)
Personality: "I love learning new things!" (0.6)
Physical Embodiment: "*scratches frantically* This sweater!" (0.7) ← Interrupts

# Manifold output:
"*scratches frantically* Aagh this sweater! ...uh, what? Oh, quantum physics!
 *scratches again* Um, it's fascinating but I CAN'T CONCENTRATE IN THIS ITCHY THING!"
```

### Scenario 3: Hot Object

**Setup:**
- SERVNAK tries to pick up fire imp (800°F)

**Touch Sensation:**
```python
# PhysicsInteractionEngine.pickup() checks temperature
if item_pod.metadata.get('temperature', '').startswith('hot'):
    # Trigger burn sensation
    embodiment.add_sensation(
        sensation_type='hot',
        intensity=0.9,
        duration=5  # Pain lingers for 5 seconds
    )

# Immediate response:
"OUCH! HOT! *yanks hand back* THERMAL DAMAGE SUSTAINED!"

# For next 5 seconds, pain colors all thoughts:
User: "Are you okay?"
Physical: "MY HAND HURTS!" (0.9) ← Dominates
Cultural: "Emotional response..." (0.3)

Output: "MY HAND HURTS, SISTER! THERMAL CONTACT WAS INADVISABLE!"
```

---

## Sensation Intensity Calculation

```python
def calculate_impact_intensity(projectile_pod: PhysicsObjectDescriptor, force: str) -> float:
    """
    Calculate how much an impact hurts.

    Args:
        projectile_pod: Object hitting the agent
        force: "light", "medium", "heavy"

    Returns:
        Intensity (0.0 to 1.0)
    """
    # Base intensity from force
    force_intensity = {
        'light': 0.2,
        'medium': 0.5,
        'heavy': 0.8
    }.get(force, 0.5)

    # Adjust for mass
    mass_lower = projectile_pod.mass.lower()
    if 'negligible' in mass_lower or 'very light' in mass_lower:
        mass_mult = 0.5
    elif 'light' in mass_lower:
        mass_mult = 0.7
    elif 'heavy' in mass_lower or 'massive' in mass_lower:
        mass_mult = 1.5
    else:
        mass_mult = 1.0

    # Adjust for hardness
    if 'soft' in projectile_pod.softness.lower():
        hardness_mult = 0.5
    elif 'hard' in projectile_pod.softness.lower():
        hardness_mult = 1.2
    else:
        hardness_mult = 1.0

    # Final intensity
    intensity = force_intensity * mass_mult * hardness_mult
    return min(1.0, intensity)


def calculate_worn_discomfort(item_pod: PhysicsObjectDescriptor) -> Optional[tuple]:
    """
    Calculate discomfort from wearing an item.

    Args:
        item_pod: Worn item physics

    Returns:
        (discomfort_type, discomfort_level) or None
    """
    # Check semantic properties
    if "itchy" in item_pod.semantic_properties:
        return ("itchy", 0.7)

    if "scratchy" in item_pod.semantic_properties:
        return ("itchy", 0.6)

    # Check mass (heavy armor)
    mass_lower = item_pod.mass.lower()
    if 'very heavy' in mass_lower or 'massive' in mass_lower:
        return ("heavy", 0.8)
    elif 'heavy' in mass_lower:
        return ("heavy", 0.6)

    # Check temperature
    temp = item_pod.metadata.get('temperature', '')
    if 'hot' in temp.lower() or temp.startswith('hot'):
        return ("hot", 0.7)
    elif 'cold' in temp.lower() or 'freezing' in temp.lower():
        return ("cold", 0.7)

    # Check fit
    if "tight" in item_pod.semantic_properties:
        return ("tight", 0.6)

    return None  # No discomfort
```

---

## Attention Penalty System

**Concept:** Physical discomfort reduces cognitive capacity.

```python
class AttentionPenalty:
    """Calculate attention reduction from bodily sensations."""

    @staticmethod
    def calculate_penalty(embodiment: PhysicalEmbodimentTransistor) -> float:
        """
        Calculate attention penalty (0.0 to 1.0).

        Higher penalty = less attention available for thinking.

        Args:
            embodiment: Physical embodiment transistor

        Returns:
            Penalty multiplier (0.0 = no attention, 1.0 = full attention)
        """
        penalty = 0.0

        # Active sensations reduce attention
        for sensation in embodiment.active_sensations:
            penalty += sensation['intensity'] * 0.3

        # Worn items reduce attention
        for item in embodiment.worn_items:
            penalty += item['discomfort_level'] * 0.2

        # Clamp to [0, 1]
        total_penalty = min(1.0, penalty)

        # Return attention multiplier (1.0 - penalty)
        return 1.0 - total_penalty


# Usage in response generation:
attention = AttentionPenalty.calculate_penalty(agent.embodiment)
max_tokens = int(base_max_tokens * attention)  # Reduce token count if distracted

# Example:
# Base max_tokens: 150
# Attention: 0.6 (wearing itchy sweater + mild pain)
# Adjusted max_tokens: 90 (shorter responses when distracted)
```

---

## Interruption Timing

### Sustained Discomfort Schedule

```python
DISCOMFORT_INTERVALS = {
    'itchy': (30, 60),    # Interrupt every 30-60 seconds
    'tight': (60, 120),   # Every 1-2 minutes
    'heavy': (120, 180),  # Every 2-3 minutes
    'hot': (20, 40),      # Every 20-40 seconds (more urgent)
    'cold': (40, 80),     # Every 40-80 seconds
    'pain': (10, 20)      # Every 10-20 seconds (very urgent)
}

def should_interrupt_now(
    discomfort_type: str,
    last_interrupt_time: float
) -> bool:
    """
    Check if sustained discomfort should interrupt now.

    Args:
        discomfort_type: Type of discomfort
        last_interrupt_time: Unix timestamp of last interrupt

    Returns:
        True if time to interrupt again
    """
    min_interval, max_interval = DISCOMFORT_INTERVALS.get(
        discomfort_type,
        (60, 120)  # Default
    )

    elapsed = time.time() - last_interrupt_time

    # Random within interval (feels more natural)
    import random
    threshold = random.uniform(min_interval, max_interval)

    return elapsed >= threshold
```

---

## Physical Actions

### Involuntary Responses

```python
SENSATION_ACTIONS = {
    'impact_soft': ["*stumbles slightly*", "*sways*", "*catches balance*"],
    'impact_hard': ["*recoils in pain*", "*staggers backward*", "*clutches impact site*"],
    'hot': ["*yanks hand back*", "*blows on fingers*", "*shakes hand rapidly*"],
    'cold': ["*shivers*", "*hugs self*", "*stamps feet*"],
    'itch': ["*scratches*", "*scratches frantically*", "*rubs itchy spot*"],
    'pain': ["*winces*", "*grimaces*", "*clutches injury*"],
    'sneeze': ["*ACHOO!*", "*sneezes*", "*sniff*"],
    'cough': ["*coughs*", "*clears throat*", "*cough cough*"]
}

def generate_physical_action(sensation_type: str) -> str:
    """Get random physical action for sensation."""
    import random
    actions = SENSATION_ACTIONS.get(sensation_type, ["*reacts*"])
    return random.choice(actions)
```

---

## Integration with Noodling Memory

**Physical sensations create episodic memories:**

```python
# After being hit by beach ball
agent.conversation_context.append({
    'user': 'physical_sensation',
    'text': 'Got hit by beach ball - soft impact to left side',
    'affect': [0.0, 0.4, 0.1, 0.0, 0.0],  # Slight arousal, slight fear
    'surprise': 0.5,
    'event_type': 'bodily_sensation',
    'sensation_metadata': {
        'type': 'impact_soft',
        'intensity': 0.3,
        'location': 'left side',
        'object': 'beach ball'
    }
})

# Now when asked: "What just happened?"
# Agent recalls: "I GOT HIT BY A BEACH BALL, SISTER! UNEXPECTED PHYSICAL CONTACT DETECTED!"
```

---

## Environmental Awareness

**Continuous environmental monitoring:**

```python
class EnvironmentalMonitor:
    """Monitors environment and triggers bodily sensations."""

    def __init__(self, world, embodiment: PhysicalEmbodimentTransistor):
        self.world = world
        self.embodiment = embodiment

    async def monitor_loop(self, agent_id: str):
        """Background loop checking environmental conditions."""
        while True:
            agent = self.world.get_user(agent_id)
            if not agent:
                break

            room = self.world.get_room(agent['current_room'])
            if not room:
                await asyncio.sleep(5)
                continue

            # Check room temperature
            room_temp = room.get('temperature', 'normal')
            if room_temp == 'hot' and not self._has_sensation('hot'):
                self.embodiment.add_sensation('hot', 0.6, duration=999999)
            elif room_temp == 'cold' and not self._has_sensation('cold'):
                self.embodiment.add_sensation('cold', 0.6, duration=999999)

            # Check if standing in water
            for obj_id in room.get('objects', []):
                obj_pod = self.world.get_object_pod(obj_id)
                if obj_pod and "liquid" in obj_pod.semantic_properties:
                    if self._is_agent_in_liquid(agent_id, obj_id):
                        self.embodiment.add_sensation('cold', 0.5, duration=999999)

            await asyncio.sleep(10)  # Check every 10 seconds

    def _has_sensation(self, sensation_type: str) -> bool:
        """Check if sensation already active."""
        return any(s['type'] == sensation_type for s in self.embodiment.active_sensations)
```

---

## Summary

**Physical Embodiment Transistor** adds somatic consciousness:

✅ **Impact sensations** (hit by objects)
✅ **Worn item discomfort** (itchy, heavy, tight)
✅ **Environmental sensations** (hot, cold, wet)
✅ **Tactile feedback** (touching objects)
✅ **Attention penalties** (distraction from discomfort)
✅ **Involuntary actions** (recoil, scratch, shiver)
✅ **Timed interruptions** (sustained discomfort)

**Integration Points:**
- Physics system (impacts, touches)
- Inventory system (worn items)
- Environment (temperature, weather)
- Memory (bodily sensation memories)

**Result:** Noodlings with **embodied consciousness** - thoughts colored by bodily experience.

---

**Example Output:**

Before Physical Embodiment:
```
User: "What do you think about philosophy?"
SERVNAK: "PHILOSOPHY IS THE SYSTEMATIC STUDY OF FUNDAMENTAL QUESTIONS..."
```

After Physical Embodiment (wearing itchy sweater):
```
User: "What do you think about philosophy?"
SERVNAK: "*scratches frantically* AAGH THIS SWEATER! ...UH, PHILOSOPHY?
         *scratches* SYSTEMATIC STUDY OF... *pauses to scratch*
         SISTER I CANNOT FOCUS IN THIS ITCHY GARMENT!"
```

**Realistic embodied cognition.**

---

**Status:** Specification complete, base implementation ready

*— Commander Spock*

*Fascinating addition to the cognitive architecture, Lieutenant.*
