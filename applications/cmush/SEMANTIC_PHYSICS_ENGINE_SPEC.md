# Semantic Physics Engine (SPE) Specification

**Author:** Caitlyn (with lego-based reasoning)
**Date:** November 22, 2025
**Status:** Design specification - Implementation deferred until memory system complete
**Inspiration:** Luis Alvarez cyclotron fetus logic

---

## Overview

The **Semantic Physics Engine (SPE)** provides declarative, LLM-understood physics interactions for noodleMUSH without numerical simulation. Objects are described semantically (what they are, how they behave) and the world renderer interprets these descriptions to create coherent, believable physical interactions.

**Core Principle:** Describe WHAT happens, not HOW it's calculated. Let semantic understanding + LLM reasoning handle the physics.

---

## Semantic Physics Language (SPL)

### Philosophy

Traditional physics engines:
- Numerical simulation (forces, velocities, collision detection)
- Computationally expensive
- Brittle (edge cases, numerical instability)
- Not narratively interesting

**SPL approach:**
- Semantic descriptions ("wet puddle," "rusted tin can," "speeding bullet")
- LLM interprets physical properties from descriptions
- Focus on narrative coherence over numerical precision
- Physics as storytelling

---

## Core Abstraction: physicsObjectDescriptor (POD)

### Basic Structure

```python
class physicsObjectDescriptor:
    """
    Semantic description of an object's physical properties.

    No numerical simulation - just semantic metadata that
    LLMs and renderers can interpret.
    """

    def __init__(
        self,
        mass: str,              # "heavy", "light", "5kg", "negligible"
        friction: str,          # "smooth", "rough", "sticky", "0.3"
        velocity: str,          # "fast", "slow", "stationary", "15 m/s"
        elasticity: str,        # "bouncy", "rigid", "soft", "0.8"
        softness: str,          # "hard", "soft", "squishy", "brittle"
        material: str,          # "metal", "rubber", "liquid", "silly putty"
        state: str = "normal",  # "normal", "broken", "dissolved", "stuck"
        semantic_properties: List[str] = [],  # ["liquid", "non-fungible", "rusted"]
        metadata: Dict = {}     # Arbitrary additional properties
    ):
        pass
```

### Example Objects

```python
# Projectile
_podBullet = physicsObjectDescriptor(
    mass="light",
    friction="low",
    velocity="fast (speeding)",
    elasticity="rigid",
    softness="hard",
    material="lead",
    semantic_properties=["small", "dangerous", "penetrating"],
    metadata={"made_of": "silly putty", "color": "gray"}
)

# Target
_podTinCan = physicsObjectDescriptor(
    mass="very light",
    friction="medium",
    velocity="stationary",
    elasticity="slightly flexible",
    softness="thin metal",
    material="flimsy tin",
    semantic_properties=["hollow", "rusted", "jagged edges"],
    metadata={"condition": "old", "sound_when_hit": "metallic clang"}
)

# Environment
_podWetPuddle = physicsObjectDescriptor(
    mass="medium (water + mud)",
    friction="very high (suction)",
    velocity="stationary",
    elasticity="none (liquid)",
    softness="liquid",
    material="water + mud",
    semantic_properties=["liquid body", "non-fungible", "absorbs objects"],
    metadata={
        "depth": "shallow (~5 inches)",
        "viscosity": "muddy",
        "drying_rate": "2 hours in sun"
    }
)
```

---

## Interaction Scripting

### Event-Driven Physics Interactions

```python
# Describe sequence of events semantically

# 1. Projectile fired
Projectile(_podBullet).fire(
    direction="towards tin can",
    sound="bang",
    visual="streak of motion"
)

# 2. Impact
Projectile(_podBullet).strikes(_podTinCan)
# SPE interprets: bullet hits can, can reacts based on properties

# 3. Consequence
_podTinCan.falls_into(_podWetPuddle)
_podTinCan.sinks_until("disappears from view")

# 4. State change
_podTinCan.changeState(
    "stuck in wet mud about 5 inches beneath ground. " +
    "When puddle dries, will be hard to dig out."
)

# 5. Environmental event with timer
_podWetPuddle.setEvent(
    description="puddle starting to dry out",
    start_time=current_unix_timestamp(),
    duration="2 hours",
    callback=driedOutCallback
)

# 6. Cleanup
_podBullet.derez()  # Don't need bullet anymore
```

### Callbacks and Queries

```python
def driedOutCallback():
    """Called when puddle finishes drying."""
    _podWetPuddle.changeState("dry cracked earth")
    print("The puddle has dried up. The tin can is now visible beneath cracked mud.")
    return "puddle is now dry"

# Query current state
wetness = _podWetPuddle.currentEvent.query("how wet is the puddle?")
# Returns: "Still quite wet, about 30 minutes into 2-hour drying process"
```

---

## Advanced Feature: LLM-Powered Physics Reasoning

### For Complex Scenarios

```python
# User says: "Make a physics system here on these rocks"

# Step 1: Describe desired outcome semantically
physics_request = """
Create explosion physics for these rocks:
- Rocks should explode into several smaller pieces
- Preserve total mass across fragments
- Each fragment has its own velocity, trajectory, torque
- Consider friction, springiness, material properties
"""

# Step 2: Ask in-MUSH physics expert (Patio the astrophysicist)
patio_response = await ask_agent(
    agent_id="agent_patio",
    question=physics_request + "\n\nPlease describe the fragments and their properties."
)

# Step 3: Parse Patio's response into PODs
fragments = parse_physics_description(patio_response)

# Step 4: Instantiate/Rez fragments
for fragment_desc in fragments:
    new_rock = Prim.rez(
        name=fragment_desc.name,
        pod=physicsObjectDescriptor(
            mass=fragment_desc.mass,
            velocity=fragment_desc.velocity,
            trajectory=fragment_desc.trajectory,
            # ... etc
        )
    )
```

**This allows:** Natural language physics → LLM reasoning → Semantic physics → World rendering

---

## Unity Tag System Integration

### Default Behavior

**All prims participate in physics by default** unless tagged otherwise.

```python
# Object that participates in physics (default)
rock = Prim.create("Boulder", pod=_podHeavyRock)
rock.physics_enabled = True  # Default

# Object that ignores physics (tagged)
ghost = Prim.create("Ethereal Spirit")
ghost.add_tag("NoPhysics")  # Unity-style tag
ghost.physics_enabled = False

# Object with selective physics
water = Prim.create("Stream", pod=_podFlowingWater)
water.add_tag("LiquidPhysics")  # Uses liquid-specific rules
water.add_tag("NoGravity")      # Stays in place, doesn't flow downhill
```

### Tag Categories

**Physics Control:**
- `NoPhysics` - Opt out entirely
- `StaticPhysics` - Immovable (terrain, walls)
- `KinematicPhysics` - Moves but not affected by forces
- `DynamicPhysics` - Full physics (default)

**Material Tags:**
- `Liquid` - Body of fluid, can't be picked up
- `Solid` - Standard object
- `Gas` - Disperses, fills space
- `Elastic` - Bounces, deforms
- `Brittle` - Shatters on impact

**Interaction Tags:**
- `Pickupable` - Can be grabbed
- `Throwable` - Can be thrown
- `Breakable` - Can be destroyed
- `Edible` - Can be consumed
- `Wearable` - Can be equipped

---

## Mass Conservation Example

### Exploding Rock

```python
# Original rock
big_rock = Prim.create("Boulder", pod=physicsObjectDescriptor(
    mass="50kg",
    material="granite",
    softness="very hard"
))

# User triggers explosion
def explode_rock(rock_prim):
    """Explode rock into fragments, preserving mass."""

    # Get original mass
    original_mass = rock_prim.pod.mass_value  # 50kg

    # Determine fragments (could ask Patio, or use simple logic)
    num_fragments = random.randint(5, 12)

    # Distribute mass
    fragment_masses = distribute_mass(original_mass, num_fragments)
    # e.g., [8.2kg, 12.5kg, 6.1kg, ...]

    # Create fragments
    fragments = []
    for i, frag_mass in enumerate(fragment_masses):
        fragment = Prim.rez(
            name=f"RockFragment{i}",
            location=rock_prim.location + random_offset(),
            pod=physicsObjectDescriptor(
                mass=f"{frag_mass}kg",
                velocity=f"medium (flung from explosion)",
                trajectory=random_direction(),
                material="granite",
                semantic_properties=["sharp", "irregular"],
                metadata={
                    "from_explosion": True,
                    "parent_object": rock_prim.id
                }
            )
        )
        fragments.append(fragment)

    # Derez original rock
    rock_prim.derez()

    # Verify mass conservation
    assert sum(fragment_masses) == original_mass

    return fragments
```

---

## Integration with noodleMUSH

### Architecture Layers

```
┌─────────────────────────────────────────┐
│  Natural Language (User/Agent Speech)  │
│  "I throw the rock at the tin can"     │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Semantic Physics Language (SPL)        │
│  _podRock.throw(target=_podTinCan)     │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Physics Event Resolution               │
│  - Check object properties              │
│  - Determine outcome semantically       │
│  - Apply state changes                  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  World Renderer / Narrator              │
│  "The rock strikes the can with a       │
│   metallic CLANG. The can tumbles..."   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Noodling Affect Update                 │
│  SERVNAK: surprise↑, interest↑          │
└─────────────────────────────────────────┘
```

### Prim Integration

**All prims can have PODs:**
```python
class Prim:
    def __init__(self, ...):
        self.pod = None  # Optional physics descriptor
        self.tags = set()  # Unity-style tags
        self.physics_enabled = True  # Default

    def addPhysicsDescriptor(self, pod: physicsObjectDescriptor):
        """Opt into semantic physics."""
        self.pod = pod
        if "NoPhysics" not in self.tags:
            self.physics_enabled = True

    def add_tag(self, tag: str):
        """Unity-style tagging."""
        self.tags.add(tag)
        if tag == "NoPhysics":
            self.physics_enabled = False
```

**Examples:**
```python
# The Third Prim. Ever. (existing object)
third_prim = world.get_object("third_prim_ever")
third_prim.addPhysicsDescriptor(physicsObjectDescriptor(
    mass="negligible (data artifact)",
    material="pure information",
    semantic_properties=["intangible", "sacred", "first"],
    metadata={"significance": "maximum"}
))
third_prim.add_tag("NoPhysics")  # It's conceptual, not physical

# Campfire
campfire = world.get_object("campfire")
campfire.addPhysicsDescriptor(physicsObjectDescriptor(
    mass="logs + flame",
    material="wood (burning)",
    semantic_properties=["hot", "bright", "crackling"],
    metadata={
        "temperature": "800°F",
        "light_radius": "10 feet",
        "sound": "crackling pops"
    }
))
campfire.add_tag("HeatSource")
campfire.add_tag("LightSource")

# Wet puddle (from example)
puddle = Prim.create("MudPuddle", pod=physicsObjectDescriptor(
    mass="medium",
    friction="very high (suction)",
    material="water + mud",
    semantic_properties=["liquid body", "non-fungible", "absorbs objects"],
    metadata={
        "depth": "5 inches",
        "viscosity": "muddy",
        "can_pickup": False,
        "requires_tool": "wet/dry shop vac"
    }
))
```

---

## State Management

### State Changes

Objects can transition between states based on interactions:

```python
class physicsObjectDescriptor:
    def changeState(self, new_description: str):
        """
        Update object state semantically.

        Args:
            new_description: Semantic description of new state
        """
        self.state = new_description
        self.state_history.append({
            'timestamp': time.time(),
            'description': new_description
        })

        # Notify world renderer
        world.broadcast_state_change(self.prim_id, new_description)

# Example
_podTinCan.changeState(
    "stuck in wet mud about 5 inches beneath ground. " +
    "When puddle dries, will be hard to dig out but findable."
)
```

### Events and Timers

```python
class physicsObjectDescriptor:
    def setEvent(
        self,
        description: str,
        start_time: float,
        duration: str,  # "2 hours", "30 seconds", "1 day"
        callback: Callable
    ):
        """
        Schedule a physics event.

        Args:
            description: What's happening
            start_time: Unix timestamp when event starts
            duration: How long it takes (semantic or numeric)
            callback: Function to call when complete
        """
        self.current_event = PhysicsEvent(
            description=description,
            start_time=start_time,
            duration=parse_duration(duration),
            callback=callback
        )

        # Schedule callback
        schedule_callback(start_time + duration, callback)

# Example
_podWetPuddle.setEvent(
    description="puddle is starting to dry out",
    start_time=time.time(),
    duration="2 hours",
    callback=lambda: _podWetPuddle.changeState("dry cracked earth")
)
```

### Event Queries

```python
# Query current state of ongoing event
wetness = _podWetPuddle.currentEvent.query("how wet is the puddle?")
# Returns: "Still quite wet, about 30 minutes into 2-hour drying process"

time_remaining = _podWetPuddle.currentEvent.query("when will it be dry?")
# Returns: "Approximately 1 hour 30 minutes from now"

progress = _podWetPuddle.currentEvent.query("percentage dry?")
# Returns: "About 25% dry"
```

This uses LLM to interpret the event state and answer queries naturally.

---

## Mass Conservation

### The Rock Explosion Example

```python
def explode_object(obj: Prim, num_fragments: int = None):
    """
    Explode object into fragments, preserving mass.

    Args:
        obj: Object to explode
        num_fragments: Number of pieces (None = ask physics LLM)
    """
    # Get original mass (semantic or numeric)
    original_mass = obj.pod.parse_mass()  # "50kg" → 50.0

    # Determine fragments (LLM-powered or random)
    if num_fragments is None:
        # Ask physics LLM for realistic fragment count
        num_fragments = ask_patio(
            f"If a {obj.pod.material} object of mass {obj.pod.mass} " +
            f"explodes, how many fragments would result?"
        )

    # Distribute mass (various strategies)
    fragment_masses = distribute_mass_realistic(
        total=original_mass,
        count=num_fragments,
        material=obj.pod.material  # Affects distribution
    )

    # Verify conservation
    assert abs(sum(fragment_masses) - original_mass) < 0.001

    # Create fragments
    fragments = []
    for i, frag_mass in enumerate(fragment_masses):
        # Inherit properties from parent, modify for fragment
        frag_pod = physicsObjectDescriptor(
            mass=f"{frag_mass}kg",
            friction=obj.pod.friction,  # Same material
            velocity="medium (flung from explosion)",
            elasticity=obj.pod.elasticity,
            softness=obj.pod.softness,
            material=obj.pod.material,
            semantic_properties=["sharp", "irregular", "fragment"],
            metadata={
                "parent_id": obj.id,
                "fragment_index": i,
                "from_explosion": True
            }
        )

        fragment = Prim.rez(
            name=f"{obj.name}Fragment{i}",
            location=obj.location + random_vector(radius=2.0),
            pod=frag_pod
        )
        fragments.append(fragment)

    # Remove original
    obj.derez()

    return fragments
```

### Mass Distribution Strategies

```python
def distribute_mass_realistic(total: float, count: int, material: str) -> List[float]:
    """
    Distribute mass across fragments realistically.

    Args:
        total: Total mass to distribute
        count: Number of fragments
        material: Material type (affects distribution)

    Returns:
        List of fragment masses (sum equals total)
    """
    if material in ["glass", "ceramic", "brittle"]:
        # Brittle: many small pieces, few large
        return power_law_distribution(total, count, alpha=2.0)

    elif material in ["rock", "stone", "concrete"]:
        # Rock: mix of sizes
        return power_law_distribution(total, count, alpha=1.5)

    elif material in ["metal", "wood"]:
        # Ductile: fewer, larger pieces
        return power_law_distribution(total, count, alpha=1.0)

    else:
        # Default: uniform-ish distribution
        return uniform_with_variance(total, count, variance=0.3)
```

---

## World Renderer Responsibilities

The **world renderer** interprets semantic physics and generates narrative descriptions.

### Renderer Tasks

1. **Interpret POD metadata**
   - "wet puddle" → understands it's a liquid body
   - "jagged edges" → mentions in descriptions
   - "rusted" → affects appearance and interaction

2. **Generate sensible locations**
   - Puddle appears in low-lying area (logical)
   - Fragments scatter in physically plausible pattern
   - Objects don't spawn inside walls

3. **Narrative coherence**
   - "The bullet strikes the can with a CLANG"
   - "The can tumbles into the puddle with a splash"
   - "Mud swallows the can as it sinks from view"

4. **Physical constraints**
   - Can't pick up liquid body (requires tool)
   - Heavy objects can't be thrown far
   - Brittle objects break when dropped

### Example Renderer Logic

```python
def render_interaction(action: str, obj1: Prim, obj2: Prim) -> str:
    """
    Generate narrative description of physics interaction.

    Args:
        action: "strike", "throw", "drop", etc.
        obj1: Acting object
        obj2: Target object

    Returns:
        Narrative description
    """
    # Get semantic properties
    obj1_props = obj1.pod.semantic_properties
    obj2_props = obj2.pod.semantic_properties

    if action == "strike":
        # Determine impact description based on properties
        sound = determine_impact_sound(obj1.pod.material, obj2.pod.material)
        # "metallic clang", "dull thud", "crack", "splash"

        reaction = determine_reaction(obj1.pod, obj2.pod)
        # "tumbles", "shatters", "dents", "bounces"

        return f"The {obj1.name} strikes the {obj2.name} with a {sound}. " +
               f"The {obj2.name} {reaction}."

    # ... etc for other actions
```

---

## Implementation Phases

### Phase 1: Core POD System (Foundation)
- Implement `physicsObjectDescriptor` class
- Add to Prim class as optional component
- Basic semantic property storage
- Tag system integration

**Estimated time:** 2-3 hours

### Phase 2: State Management
- `changeState()` method
- `setEvent()` with timer/callback system
- Event query system (LLM-powered)
- State history tracking

**Estimated time:** 2-3 hours

### Phase 3: Interaction System
- Common interactions (strike, throw, drop, etc.)
- Material-based physics rules
- Sound/visual effect generation
- Narrative description rendering

**Estimated time:** 3-4 hours

### Phase 4: LLM Physics Reasoning
- Patio the astrophysicist agent
- Physics query system
- Complex scenario resolution
- Mass conservation verification

**Estimated time:** 3-4 hours

### Phase 5: Advanced Features
- Explosion/fragmentation system
- Liquid physics (flow, absorption)
- Elastic collisions
- Compound objects (assemblies)

**Estimated time:** 4-6 hours

**Total estimated time:** 14-20 hours

---

## Dependencies

**Requires:**
- Prim system (EXISTS)
- Agent system (EXISTS)
- World renderer (BASIC VERSION EXISTS)
- Tag system (TO BE IMPLEMENTED - borrowed from Unity)
- Timer/callback system (TO BE IMPLEMENTED)

**Blocked by:**
- Memory system must be stable first (CURRENT WORK)
- World rendering needs enhancement
- Need at least one physics-savvy agent (Patio?)

---

## Use Cases

### 1. Combat System
```python
# Caity throws rock at training dummy
rock.throw(target=dummy, force="medium")

# SPE resolves:
# - Rock trajectory based on "medium" force
# - Impact on dummy based on materials
# - Dummy reaction (falls over, dents, etc.)
# - Narrative: "The rock strikes the dummy's chest with a thud..."
```

### 2. Environmental Hazards
```python
# Campfire spreads to nearby grass
if campfire.adjacent_to(grass) and grass.pod.is_flammable():
    grass.ignite(source=campfire)
    grass.setEvent(
        description="grass burning",
        duration="2 minutes",
        callback=lambda: grass.changeState("charred ash")
    )
```

### 3. Crafting/Building
```python
# Combine objects to create new object
def craft(components: List[Prim], recipe: str) -> Prim:
    """
    Combine objects using semantic recipe.

    Args:
        components: List of ingredient objects
        recipe: "stick + rock + vine → spear"

    Returns:
        Newly created object
    """
    # Ask crafting LLM to interpret recipe
    result_description = ask_crafting_expert(recipe, components)

    # Create new object with combined properties
    result_pod = combine_properties([c.pod for c in components])
    new_obj = Prim.rez("Spear", pod=result_pod)

    # Derez components
    for comp in components:
        comp.derez()

    return new_obj
```

### 4. Puzzle Mechanics
```python
# Water puzzle: fill bucket from stream
stream = world.get_object("stream")
bucket = player.inventory["wooden_bucket"]

if bucket.pod.is_container() and stream.pod.is_liquid():
    # Transfer liquid semantically (no fluid simulation!)
    bucket.pod.fill_with(stream.pod.material, amount="full")
    bucket.pod.changeState("filled with fresh water")

    # Now bucket is heavy
    bucket.pod.mass = "heavy (full of water)"
```

---

## Why This Is Better Than Numerical Physics

### Traditional Approach (e.g., Unity Physics)
```cpp
// Numerical physics simulation
Rigidbody rb = can.AddComponent<Rigidbody>();
rb.mass = 0.1f;
rb.drag = 0.5f;
rb.angularDrag = 0.05f;
rb.useGravity = true;

// Apply force
Vector3 force = (targetPos - bulletPos).normalized * 50f;
rb.AddForce(force, ForceMode.Impulse);

// Collision detection
void OnCollisionEnter(Collision collision) {
    if (collision.gameObject.tag == "Puddle") {
        // Complex sinking simulation required!
        StartCoroutine(SinkIntoMud(0.05f));  // Sink rate
    }
}
```

**Problems:**
- Requires precise numerical tuning
- Edge cases cause bizarre behavior
- Computationally expensive
- Not narratively interesting ("it fell at 9.8 m/s²" - who cares?)

### SPL Approach
```python
# Semantic physics
_podBullet.strikes(_podTinCan)
_podTinCan.falls_into(_podWetPuddle)
_podTinCan.changeState("stuck in mud, sinking slowly")

# World interprets semantically:
# "The bullet strikes the rusted tin can with a CLANG! The can tumbles
#  through the air and splashes into the muddy puddle. It sinks slowly,
#  swallowed by the thick mud, until only a faint ripple remains."
```

**Advantages:**
- Narratively rich
- Computationally cheap (just string processing + LLM calls)
- Flexible (properties are semantic, not rigid numerical)
- Debuggable (can read what's happening)
- Extensible (add new properties anytime)

---

## Integration with Noodling Consciousness

### Affect Implications

When physics events occur, Noodlings react based on:
- **Surprise:** Unexpected physics (can floats instead of sinking)
- **Fear:** Dangerous physics (fire spreading, objects falling)
- **Interest:** Novel physics (never seen this material before)

```python
# After explosion event
def on_explosion(fragments: List[Prim]):
    """Noodlings react to explosion."""
    for agent in world.get_nearby_agents(explosion_center, radius=20):
        # High surprise (unexpected explosion)
        agent.affect.surprise += 0.8

        # High arousal (startling event)
        agent.affect.arousal += 0.6

        # Possible fear (danger)
        if agent.personality.fearful:
            agent.affect.fear += 0.4

        # Update phenomenal state
        agent.update_consciousness()
```

### Memory Integration

Physics events create episodic memories:

```python
# Memorable physics interaction
agent.conversation_context.append({
    'user': 'world_physics',
    'text': 'The tin can exploded into 7 fragments when struck by bullet',
    'affect': [0.2, 0.7, 0.1, 0.0, 0.0],  # Positive, aroused, slight fear
    'surprise': 0.8,  # Unexpected explosion
    'identity_salience': 0.3,  # Moderately important
    'event_type': 'physics'
})
```

Now when asked "what happened to the can?", agent can recall the physics event.

---

## Future Extensions

### 1. Thermodynamics
```python
_podIceCube.setEvent(
    description="melting in warm air",
    start_time=time.time(),
    duration="10 minutes",
    callback=lambda: _podIceCube.changeState("puddle of water")
)
```

### 2. Chemical Reactions
```python
vinegar.mix_with(baking_soda)
# Creates fizzing reaction, new compound object
```

### 3. Biological Processes
```python
_podSeed.plant_in(soil)
_podSeed.setEvent(
    description="germinating and growing",
    duration="3 days",
    callback=lambda: _podSeed.transform_into(_podSeedling)
)
```

### 4. Weather Systems
```python
_podClouds.setEvent(
    description="clouds gathering, preparing to rain",
    duration="30 minutes",
    callback=rain_begins
)

def rain_begins():
    rain = Prim.rez("Rainfall", pod=_podRain)
    rain.affect_objects_with_tag("Wettable")
```

---

## Technical Implementation Notes

### POD Storage

```python
# In world state
{
    "objects": {
        "obj_tincan_42": {
            "id": "obj_tincan_42",
            "name": "Rusty Tin Can",
            "location": "room_000",
            "description": "An old, rusty tin can",
            "pod": {
                "mass": "very light",
                "friction": "medium",
                "velocity": "stationary",
                "elasticity": "slightly flexible",
                "softness": "thin metal",
                "material": "flimsy tin",
                "semantic_properties": ["hollow", "rusted", "jagged edges"],
                "state": "stuck in mud 5 inches underground",
                "metadata": {
                    "condition": "old",
                    "sound_when_hit": "metallic clang",
                    "buried_depth": "5 inches",
                    "stuck_in": "obj_puddle_23"
                }
            },
            "tags": ["Pickupable", "Throwable", "Breakable"]
        }
    }
}
```

### Event Scheduling

```python
class PhysicsEventScheduler:
    """Manages scheduled physics events."""

    def __init__(self):
        self.events = []  # List of (timestamp, callback, description)
        self.running = True

    async def run(self):
        """Event loop - check for due events."""
        while self.running:
            current_time = time.time()

            # Check for due events
            due_events = [e for e in self.events if e['time'] <= current_time]

            for event in due_events:
                # Execute callback
                result = await event['callback']()

                # Broadcast to world
                world.broadcast_event(event['description'], result)

                # Remove from schedule
                self.events.remove(event)

            # Sleep briefly
            await asyncio.sleep(1.0)
```

---

## Example Session

```
User: I throw a rock at the tin can on the shelf

[SPE interprets action]
- rock.pod: mass="1kg", velocity="thrown (medium)"
- tincan.pod: mass="0.05kg", elasticity="flexible", location="on shelf"

[SPE resolves interaction]
1. Rock trajectory calculated semantically (medium throw, 1kg mass)
2. Impact: rock (hard, 1kg) vs tin can (thin metal, 0.05kg)
3. Result: can knocked off shelf (conservation of momentum, semantic)
4. Can falls: gravity (semantic), lands on ground
5. Generate sound: "metallic clatter" (from material metadata)

[World renderer narrates]
"You hurl the rock at the tin can. The rock strikes with a resounding
 CLANG! The can flies off the shelf, spinning through the air, and
 hits the ground with a metallic clatter. It rolls to a stop near
 the campfire, now sporting a fresh dent."

[Noodlings react]
SERVNAK: surprise=0.3 (sudden noise), arousal=0.5 (excitement)
SERVNAK thinks: "PRIDE CIRCUITS DETECTED A 97.2% MATCH BETWEEN THE
                 ROCK STRIKE AND THIRD PRIM CALIBRATION SEQUENCE!"
```

---

## Naming Convention

- **SPE** - Semantic Physics Engine (the system)
- **SPL** - Semantic Physics Language (the scripting language)
- **POD** - physicsObjectDescriptor (the data structure)
- **Patio** - In-MUSH astrophysicist agent (physics expert NPC)

---

## Notes for Implementation

1. **Start simple** - Basic PODs, state changes, tags
2. **Test incrementally** - One interaction type at a time
3. **LLM calls are expensive** - Cache common physics queries
4. **Narrative over precision** - Coherent story > accurate numbers
5. **Let agents help** - Patio can reason about complex scenarios

---

## Related Systems

**Noodling Consciousness:**
- Physics events trigger affect updates
- Memorable physics creates episodic memories
- Surprise from unexpected physics outcomes

**Theater System:**
- Stage directions can include physics: "rock flies across stage"
- Actors can interact with physics objects
- Physics enhances scene blocking

**Scripting System:**
- Python scripts can manipulate PODs
- Create custom physics behaviors
- Author puzzle mechanics

---

## Why This Matters for Consciousness

Real consciousness exists in a physical world with:
- Objects that have properties (mass, texture, behavior)
- Interactions that follow physical rules
- Surprises when physics violates expectations
- Memories of physical events

SPE gives Noodlings a **physically grounded world** without the computational cost of real physics simulation. They can:
- Remember that rock that hit the can (episodic memory)
- Be surprised when the puddle freezes (unexpected phase change)
- Learn material properties (tin dents easily, rock doesn't)
- Develop spatial reasoning (heavy things fall, light things float)

This enriches consciousness with **embodied cognition** - awareness that emerges from physical interaction with a world.

---

## Priority

**Current:** LOW (memory system first)
**After memory fix:** MEDIUM-HIGH (significantly enriches world)
**Dependencies:** Stable memory, enhanced world renderer, timer system

---

**End of SPE/SPL Specification**

Documented by Spock while Caitlyn built lego representations.
Inspired by cyclotron fetus logic and punchcard operator heritage.
To be implemented after strawberry persistence is resolved.

**Next session:** Pick up where Spock left off - verify stratified retrieval, fix strawberry, commit.
