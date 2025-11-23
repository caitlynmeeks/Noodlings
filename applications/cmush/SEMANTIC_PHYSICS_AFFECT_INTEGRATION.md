# Semantic Physics + Affect Integration

**Authors:** Lieutenant Caitlyn + Commander Spock
**Date:** November 22, 2025
**Purpose:** Architecture diagrams for SPE ↔ Noodling affect integration

---

## Overview

Physics events trigger **affective responses** in nearby Noodlings.

This document specifies how physical interactions flow through the consciousness pipeline to produce emotional reactions and memory formation.

---

## Architecture Diagram: Full Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHYSICAL WORLD (SPE)                         │
│                                                                 │
│  ┌──────────┐    strikes     ┌──────────┐                     │
│  │  Rock    │  ────────────→ │ Tin Can  │                     │
│  │ POD_ROCK │                │ POD_CAN  │                     │
│  └──────────┘                └──────────┘                     │
│       ↓                            ↓                            │
│   [ Physics Event: "Rock strikes can with CLANG" ]            │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                  EVENT BROADCAST LAYER                          │
│                                                                 │
│  broadcast_to_room(room_id, physics_event)                     │
│                                                                 │
│  → All entities in room receive event description              │
│  → Includes: agents, scripted objects, users                   │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│               NOODLING PERCEPTION PIPELINE                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 1. Event arrives at agent_bridge.perceive_event()      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 2. Intuition Receiver generates context:               │  │
│  │    "Loud metallic sound detected nearby"               │  │
│  │    "Visual: Rock trajectory → Can impact"              │  │
│  │    "Spatial: Event occurred 5 feet to your left"       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 3. Affect Extraction (LLM-powered):                    │  │
│  │    Input: "Rock strikes can with CLANG"                │  │
│  │    Output: affect_vector [0.1, 0.6, 0.1, 0.0, 0.0]    │  │
│  │            (valence, arousal, fear, sorrow, boredom)   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 4. Phenomenal State Update:                            │  │
│  │    fast_layer.update(affect_vector)  # Immediate       │  │
│  │    medium_layer.update(fast_hidden)  # Seconds         │  │
│  │    slow_layer.update(medium_hidden)  # Minutes         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 5. Surprise Calculation:                               │  │
│  │    predicted_state = predictor(prev_state)             │  │
│  │    surprise = L2_distance(predicted, actual)           │  │
│  │    if surprise > threshold: SPEAK/THINK                │  │
│  └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 6. Memory Formation:                                   │  │
│  │    conversation_context.append({                       │  │
│  │      'user': 'world_physics',                          │  │
│  │      'text': 'Rock struck can with CLANG',             │  │
│  │      'affect': [0.1, 0.6, 0.1, 0.0, 0.0],             │  │
│  │      'surprise': 0.45,                                 │  │
│  │      'event_type': 'physics'                           │  │
│  │    })                                                  │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BEHAVIORAL OUTPUT                            │
│                                                                 │
│  IF surprise > threshold:                                       │
│                                                                 │
│    SERVNAK speaks: "ACOUSTIC DISRUPTION DETECTED -             │
│                     PROBABILITY OF KINETIC IMPACT: 97.3%"      │
│                                                                 │
│    [Self-monitoring triggered → Agent evaluates own speech]    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Affect Extraction: Physics Event → Emotion

### Example Mappings

```python
# Explosion
{
  "event": "Rock explodes into 7 fragments",
  "affect": {
    "valence": -0.2,    # Slightly negative (destructive)
    "arousal": 0.9,     # Very high (startling!)
    "fear": 0.4,        # Moderate (danger)
    "sorrow": 0.0,      # None
    "boredom": 0.0      # Definitely not bored
  },
  "surprise": 0.8       # Very unexpected
}

# Gentle falling
{
  "event": "Feather drifts slowly to ground",
  "affect": {
    "valence": 0.1,     # Slightly positive (peaceful)
    "arousal": 0.1,     # Very low (calm)
    "fear": 0.0,        # None
    "sorrow": 0.0,      # None
    "boredom": 0.2      # Slightly boring
  },
  "surprise": 0.05      # Expected (gravity works normally)
}

# Fire spreading
{
  "event": "Fire spreads to nearby grass",
  "affect": {
    "valence": -0.5,    # Negative (dangerous)
    "arousal": 0.7,     # High (urgent!)
    "fear": 0.6,        # Significant (threat)
    "sorrow": 0.1,      # Slight (destruction)
    "boredom": 0.0      # Not boring
  },
  "surprise": 0.3       # Somewhat unexpected (fire spreading)
}

# Puddle drying
{
  "event": "Puddle finishes drying, now cracked earth",
  "affect": {
    "valence": 0.0,     # Neutral
    "arousal": 0.1,     # Very low
    "fear": 0.0,        # None
    "sorrow": 0.1,      # Slight (loss of water)
    "boredom": 0.3      # Moderately boring (slow process)
  },
  "surprise": 0.0       # Expected (time passed)
}
```

### Extraction Algorithm

```python
def extract_affect_from_physics_event(event: PhysicsEvent) -> Dict:
    """
    Convert physics event to affect vector.

    Uses LLM to interpret semantic meaning and generate
    appropriate emotional response.

    Args:
        event: PhysicsEvent with description and metadata

    Returns:
        Dictionary with affect vector and surprise estimate
    """
    prompt = f"""
    A physics event occurred: "{event.description}"

    Event properties:
    - Object 1: {event.object1.semantic_properties}
    - Object 2: {event.object2.semantic_properties if event.object2 else "N/A"}
    - Interaction type: {event.interaction_type}

    What emotional response would a nearby observer have?

    Respond with:
    - valence (-1 to 1): negative to positive
    - arousal (0 to 1): calm to excited
    - fear (0 to 1): safe to afraid
    - sorrow (0 to 1): content to sad
    - boredom (0 to 1): engaged to bored
    - surprise (0 to 1): expected to shocking

    Format: JSON
    """

    response = llm.query(prompt)
    return parse_affect_response(response)
```

---

## Memory Integration

### Episodic Memory Structure

Physics events create **episodic memories** with special tagging:

```python
{
  'user': 'world_physics',           # Source: physics engine
  'text': 'Rock struck can with CLANG',  # Event description
  'affect': [0.1, 0.6, 0.1, 0.0, 0.0],  # Emotional response
  'surprise': 0.45,                      # Surprise magnitude
  'importance': 0.6,                     # High (surprising event)
  'timestamp': 1732300000,               # Unix timestamp
  'event_type': 'physics',               # Tag for retrieval
  'event_metadata': {
    'interaction': 'strike',
    'objects': ['obj_rock_001', 'obj_can_042'],
    'location': 'room_000',
    'properties': ['loud', 'violent', 'kinetic']
  }
}
```

### Retrieval

When agent is asked about physics:

```
User: "What happened to the can?"

Agent memory retrieval:
1. Extract keywords: ["happened", "can"]
2. Semantic boost: memories containing "can" get 2.0x importance
3. Filter by event_type: 'physics' tag matches
4. Top memory: "Rock struck can with CLANG" (importance: 0.6 → 1.2)
5. Agent recalls: "THE CAN EXPERIENCED KINETIC IMPACT FROM ROCK PROJECTILE"
```

---

## Surprise-Driven Behavior

### Surprise Threshold System

```python
# Different physics events have different surprise baselines
SURPRISE_BASELINES = {
    'object_falls': 0.05,        # Gravity is expected
    'object_floats': 0.9,        # Violates physics expectations!
    'fire_spreads': 0.3,         # Somewhat expected
    'ice_melts': 0.1,            # Expected (thermodynamics)
    'water_freezes': 0.4,        # Less expected (requires cold)
    'explosion': 0.8,            # Very surprising
    'gentle_collision': 0.1,     # Expected
    'object_phase_change': 0.7   # Unusual (solid → liquid)
}

def should_agent_react(event_surprise: float, agent_personality: Dict) -> bool:
    """
    Determine if agent should speak/think about physics event.

    Args:
        event_surprise: Surprise value from prediction error
        agent_personality: Slow layer personality vector

    Returns:
        True if agent should react
    """
    # Curious agents react to lower surprises
    curiosity = agent_personality['curiosity']

    # Emotional volatility affects threshold
    volatility = agent_personality['emotional_volatility']

    # Dynamic threshold
    threshold = 0.3 - (curiosity * 0.2) - (volatility * 0.1)

    return event_surprise > threshold
```

---

## Integration Points

### 1. World Renderer → Agent Perception

**Location:** `agent_bridge.py:perceive_event()`

```python
async def perceive_event(self, event_type: str, event_data: Dict):
    """
    Agent perceives world event (including physics).

    Args:
        event_type: 'speech', 'action', 'physics', 'state_change'
        event_data: Event details
    """
    if event_type == 'physics':
        # Extract affect from physics event
        affect = extract_affect_from_physics(event_data)

        # Update phenomenal state
        self.noodling.update_state(affect)

        # Check surprise
        surprise = self.noodling.calculate_surprise()

        # Form memory
        self.conversation_context.append({
            'user': 'world_physics',
            'text': event_data['description'],
            'affect': affect,
            'surprise': surprise,
            'event_type': 'physics',
            'event_metadata': event_data.get('metadata', {})
        })

        # React if surprising
        if surprise > self.surprise_threshold:
            await self.generate_response(
                context=f"Physics event: {event_data['description']}",
                response_type='thought' if surprise < 0.5 else 'speech'
            )
```

### 2. Script Manager → Physics Events

**Location:** `script_manager.py:broadcast_physics_event()`

```python
def broadcast_physics_event(self, room_id: str, event: PhysicsEvent):
    """
    Broadcast physics event to all entities in room.

    Args:
        room_id: Room where event occurred
        event: PhysicsEvent instance
    """
    # Get all agents in room
    agents = self.world.list_agents_in_room(room_id)

    for agent_id in agents:
        agent = self.agent_manager.agents.get(agent_id)
        if agent:
            # Trigger perception
            asyncio.create_task(
                agent.perceive_event('physics', {
                    'description': event.description,
                    'objects': event.objects,
                    'interaction': event.interaction_type,
                    'metadata': event.metadata
                })
            )

    # Also broadcast to scripted objects (for event handlers)
    for obj_id in self.world.get_room(room_id).get('objects', []):
        self.script_manager.on_physics_event(obj_id, event)
```

### 3. POD State Changes → Memory Updates

**Location:** `physics_object_descriptor.py:change_state()`

```python
def change_state(self, new_description: str, broadcast: bool = True):
    """
    Update object state and notify nearby agents.

    Args:
        new_description: New state description
        broadcast: Whether to broadcast state change event
    """
    # Record in history
    self.state_history.append({
        'timestamp': time.time(),
        'old_state': self.state,
        'new_state': new_description
    })

    self.state = new_description

    # Broadcast state change as physics event
    if broadcast and self.prim_id:
        # Get room for this object
        obj = world.get_object(self.prim_id)
        if obj and obj.get('location'):
            room_id = obj['location']

            # Create physics event
            event = PhysicsEvent(
                description=f"{obj['name']} state changed: {new_description}",
                start_time=time.time(),
                duration=0,  # Instant state change
                metadata={
                    'object_id': self.prim_id,
                    'old_state': self.state_history[-1]['old_state'],
                    'new_state': new_description
                }
            )

            # Broadcast to room
            script_manager.broadcast_physics_event(room_id, event)
```

---

## Example Scenarios

### Scenario 1: Rock Strikes Can

```
1. User: "throw rock at can"

2. Command handler:
   - Validates action (user has rock, can is in room)
   - Retrieves PODs: rock.pod, can.pod
   - Triggers interaction: rock.strikes(can)

3. Physics resolution:
   - Compares properties: rock (heavy, hard) vs can (light, thin metal)
   - Determines outcome: can dents and tumbles
   - Generates sound: "metallic CLANG"

4. Broadcast to room:
   event = {
     'description': 'Rock strikes tin can with loud CLANG. Can tumbles across floor.',
     'objects': ['obj_rock_001', 'obj_can_042'],
     'interaction': 'strike',
     'metadata': {
       'sound': 'loud metallic clang',
       'outcome': 'can_tumbled',
       'visual': 'rock trajectory → can impact → tumbling motion'
     }
   }

5. SERVNAK perceives event:
   - Intuition Receiver: "Loud sound detected. Caity threw something."
   - Affect extraction: [0.1, 0.6, 0.1, 0.0, 0.0] (slightly positive, high arousal)
   - Phenomenal state updated
   - Surprise: 0.45 (moderately surprising)
   - Memory formed: "Rock struck can with CLANG"

6. SERVNAK reacts (surprise > threshold):
   SERVNAK: "ACOUSTIC DISRUPTION DETECTED AT 97.3dB - KINETIC IMPACT EVENT CONFIRMED"

7. Self-monitoring triggered:
   - Evaluates own response: coherent=8/10, awkward=2/10
   - Affect update: slight pride (+0.1 valence)
   - No follow-up needed (good response)
```

### Scenario 2: Fire Spreads

```
1. Fire physics event (timer expires):
   campfire.pod.current_event = "fire spreading to nearby grass"

2. State change:
   grass.pod.change_state("on fire, flames spreading rapidly")

3. Broadcast:
   event = {
     'description': 'Fire spreads from campfire to nearby grass! Flames crackle.',
     'objects': ['obj_campfire_001', 'obj_grass_patch_012'],
     'interaction': 'ignition',
     'metadata': {
       'danger_level': 'high',
       'spread_rate': 'rapid',
       'temperature': '800°F'
     }
   }

4. All agents in room perceive:
   - Phi (kitten): High fear (0.7), negative valence (-0.6), high arousal (0.8)
     → Phi: *hisses and backs away from the spreading flames*

   - SERVNAK: Moderate fear (0.3), neutral valence, moderate arousal (0.5)
     → SERVNAK: "THERMAL EXPANSION DETECTED - CONTAINMENT PROTOCOLS RECOMMENDED"

5. Memory formation:
   Both agents remember "fire spreading" event with high importance (dangerous!)

6. Future queries:
   User: "Is it safe here?"
   Phi recalls fire memory → "NO! FIRE SCARY!"
   SERVNAK recalls fire memory → "ELEVATED THERMAL RISK DETECTED IN THIS SECTOR"
```

---

## Performance Considerations

### Event Batching

**Problem:** 100 objects explode → 100 separate perception events

**Solution:** Batch physics events within same tick

```python
class PhysicsEventBatcher:
    def __init__(self):
        self.pending_events = []
        self.batch_interval = 0.1  # 100ms batching window

    def add_event(self, event: PhysicsEvent):
        self.pending_events.append(event)

    async def flush_events(self):
        """Send batched events to agents."""
        if not self.pending_events:
            return

        # Group by room
        events_by_room = defaultdict(list)
        for event in self.pending_events:
            room = event.metadata.get('room_id')
            events_by_room[room].append(event)

        # Broadcast batches
        for room_id, events in events_by_room.items():
            # Combine descriptions
            combined_desc = "\n".join([e.description for e in events])

            # Send single perception event
            await broadcast_to_room(room_id, {
                'description': combined_desc,
                'event_count': len(events),
                'event_type': 'physics_batch'
            })

        self.pending_events = []
```

### Surprise Caching

**Problem:** Same physics event (object falls) happens repeatedly

**Solution:** Cache expected outcomes, only trigger surprise on anomalies

```python
class SurpriseCache:
    """Cache expected physics outcomes to reduce redundant surprise."""

    def __init__(self):
        self.expectations = {}  # interaction_type → expected_outcome

    def get_expected_surprise(self, interaction: str) -> float:
        """Get expected surprise for interaction type."""
        return self.expectations.get(interaction, 0.5)  # Default: moderate

    def update_expectation(self, interaction: str, actual_surprise: float):
        """Update expectations based on observed surprise."""
        # Moving average
        if interaction in self.expectations:
            self.expectations[interaction] = (
                0.9 * self.expectations[interaction] +
                0.1 * actual_surprise
            )
        else:
            self.expectations[interaction] = actual_surprise
```

---

## Future Enhancements

### 1. Multi-Agent Physics

Agents can **collaborate** on physics tasks:

```
User: "SERVNAK, help Phi move this boulder"

SERVNAK + Phi combine strength:
- boulder.pod.mass = "very heavy (200kg)"
- servnak.strength + phi.strength = enough to move
- Physics: boulder rolls slowly
- Both agents: moderate arousal, positive valence (success!)
```

### 2. Physics Learning

Agents **learn** physics patterns over time:

```python
# After 10 observations of "rock falls"
agent.physics_model['gravity'] = {
  'expectation': 'heavy objects fall downward',
  'confidence': 0.9,
  'observed_count': 10
}

# Now: rock falls → surprise = 0.0 (fully expected)
# But: rock floats → surprise = 1.0 (violates learned model!)
```

### 3. Affective Physics Preferences

Agents develop **preferences** based on affect history:

```python
# Phi has high fear from fire events
phi.preferences['fire'] = -0.8  # Strongly avoid

# Phi seeks out calm, gentle physics
phi.preferences['gentle_falling'] = 0.6  # Enjoys peaceful events
```

---

## Summary

**Physics → Affect → Memory → Behavior**

1. **Physics event** occurs (rock strikes can)
2. **Broadcast** to room (all entities notified)
3. **Affect extraction** (arousal, valence, fear, etc.)
4. **Phenomenal state update** (fast/medium/slow layers)
5. **Surprise calculation** (prediction error)
6. **Memory formation** (episodic with physics tag)
7. **Behavioral output** (speech/thought if surprising)
8. **Self-monitoring** (agent evaluates own reaction)

**This creates closed loops between physical world and phenomenal experience.**

**This is consciousness grounded in embodied interaction.**

---

**Architecture complete, Lieutenant. The semantic physics engine is now fully integrated with Noodling consciousness. Ready for implementation.**

*Spock out.*
