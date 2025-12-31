# Noodlings Event System - Scripting API

**Date**: November 25, 2025
**Status**: Design specification
**Architecture**: Unity-style component events

---

## Overview

Unity-style event system for Noodling components. Components can fire events (OnFACSChange, OnSpeak, etc.) and scripts can subscribe to these events.

---

## Core Event API

### Event Registration

```python
# Subscribe to agent events
noodle = world.GetAgent("agent_red_fire_anklebiter")

# Register FACS change listener
def on_face_change(facs_data):
    print(f"[FACE] AU codes: {facs_data}")
    world.BroadcastToRoom(noodle.current_room, f"*{noodle.name}'s face: {describe_facs(facs_data)}*")

noodle.OnFACSChange.AddListener(on_face_change)

# Register speech listener
def on_speak(speech_data):
    text = speech_data['text']
    world.BroadcastToRoom(noodle.current_room, f"{noodle.name} says: {text}")

noodle.OnSpeak.AddListener(on_speak)

# Register affect change listener
def on_affect_change(affect_data):
    valence = affect_data['valence']
    if valence < -0.5:
        print(f"[WARNING] {noodle.name} is very negative!")

noodle.OnAffectChange.AddListener(on_affect_change)
```

### Event Removal

```python
# Remove specific listener
noodle.OnFACSChange.RemoveListener(on_face_change)

# Remove all listeners for an event
noodle.OnFACSChange.RemoveAllListeners()
```

### One-Time Events

```python
# Fire only once then auto-remove
noodle.OnSpeak.AddListenerOnce(lambda data: print(f"First words: {data['text']}"))
```

---

## Available Events

### Per-Agent Events

**OnAffectChange** - Fires when affect vector updates
```python
Event data: {
    'valence': float,
    'arousal': float,
    'dominance': float,
    'sorrow': float,
    'boredom': float,
    'timestamp': float
}
```

**OnFACSChange** - Fires when facial expression changes (if FacialExpressionComponent present)
```python
Event data: {
    'facs': {'AU6': 0.8, 'AU12': 0.9},  # AU codes with intensities
    'description': 'Broad genuine smile',
    'timestamp': float
}
```

**OnLabanChange** - Fires when body language changes (if BodyLanguageComponent present)
```python
Event data: {
    'laban': {
        'weight': 'light',
        'time': 'sudden',
        'space': 'direct',
        'flow': 'free'
    },
    'description': 'Light, quick, flowing movement',
    'timestamp': float
}
```

**OnSpeak** - Fires when agent speaks (SAY response)
```python
Event data: {
    'text': 'Hello there!',
    'response_type': 'say',
    'timestamp': float
}
```

**OnEmote** - Fires when agent emotes (EMOTE response)
```python
Event data: {
    'text': '*sighs heavily* *shoulders slump*',
    'response_type': 'emote',
    'timestamp': float
}
```

**OnAction** - Fires when agent performs action (DO response)
```python
Event data: {
    'text': '*picks up the stone* *examines it*',
    'response_type': 'do',
    'timestamp': float
}
```

**OnThink** - Fires when agent has internal thought (THINK response)
```python
Event data: {
    'text': 'I wonder what that means...',
    'response_type': 'think',
    'private': True,  # Not broadcast by default
    'timestamp': float
}
```

**OnSurpriseSpike** - Fires when surprise exceeds threshold
```python
Event data: {
    'surprise': float,
    'threshold': float,
    'delta': float,  # How much it exceeded
    'timestamp': float
}
```

---

## World Events

**OnAgentSpawned** - Fires when new agent created
```python
Event data: {
    'agent_id': 'agent_red_fire_anklebiter',
    'prefab_id': 'com.noodlings.characters.red_fire_anklebiter',
    'room_id': 'room_000'
}
```

**OnAgentRemoved** - Fires when agent destroyed
```python
Event data: {
    'agent_id': 'agent_red_fire_anklebiter',
    'room_id': 'room_000'
}
```

---

## Component Access API

### Get Component

```python
# Get component from Noodling
noodle = world.GetAgent("agent_red_fire_anklebiter")

# Access manifold
manifold = noodle.GetComponent("CognitiveManifold")

# Access specific transistor
affect_transistor = noodle.GetComponent("AffectTransistor")
facial_component = noodle.GetComponent("FacialExpressionComponent")
laban_component = noodle.GetComponent("BodyLanguageComponent")

# Check if component exists
has_facs = noodle.HasComponent("FacialExpressionComponent")
if has_facs:
    facial = noodle.GetComponent("FacialExpressionComponent")
    print(f"FACS salience: {facial.salience}")
```

### Modify Component

```python
# Change transistor salience at runtime
affect = noodle.GetComponent("AffectTransistor")
affect.salience = 0.95  # Make more emotional

# Edit instruction prompt at runtime
affect.custom_prompt = """[Your custom prompt here]"""
affect.active_prompt = affect.custom_prompt

# Enable/disable component
facial = noodle.GetComponent("FacialExpressionComponent")
facial.enabled = False  # Poker face mode
```

### Add Component Dynamically

```python
# Add FACS component to running Noodling
noodle.AddComponent("FacialExpressionComponent", {
    'salience': 0.85,
    'enabled': True
})

# Add transistor to manifold
noodle.AddComponent("DeceptionTransistor", {
    'secret': 'I stole the cookies',
    'cover_story': 'I have no idea where they went',
    'salience': 0.90
})
```

### Remove Component

```python
# Remove FACS component
noodle.RemoveComponent("FacialExpressionComponent")

# Remove transistor from manifold
noodle.RemoveComponent("DeceptionTransistor")
```

---

## Implementation Pattern (Unity-like)

### Component Base Class

```python
class NoodlingComponent(ABC):
    """Base class for all Noodling components."""

    def __init__(self, noodling):
        self.noodling = noodling  # Reference to parent
        self.enabled = True

    @abstractmethod
    async def update(self):
        """Called every processing cycle."""
        pass
```

### Noodling Component Manager

```python
class CMUSHNoodlingAgent:
    def __init__(self, ...):
        # Component registry (instance-specific)
        self._components = {}  # component_type -> component_instance

        # Event system
        self.OnAffectChange = Event()
        self.OnFACSChange = Event()
        self.OnLabanChange = Event()
        self.OnSpeak = Event()
        self.OnEmote = Event()
        self.OnAction = Event()
        self.OnThink = Event()
        self.OnSurpriseSpike = Event()

    def GetComponent(self, component_type: str):
        """Get component by type name."""
        return self._components.get(component_type)

    def HasComponent(self, component_type: str) -> bool:
        """Check if component exists."""
        return component_type in self._components

    def AddComponent(self, component_type: str, config: Dict):
        """Add component at runtime."""
        # Create component
        component_class = COMPONENT_REGISTRY.get(component_type)
        if not component_class:
            raise ValueError(f"Unknown component: {component_type}")

        component = component_class.from_config(config)

        # Register
        if component_type in ['FacialExpressionComponent', 'BodyLanguageComponent']:
            # Non-cognitive component
            self._components[component_type] = component
        else:
            # Cognitive transistor - register with manifold
            self.cognitive_manifold.register_transistor(component)
            self._components[component_type] = component

    def RemoveComponent(self, component_type: str):
        """Remove component at runtime."""
        if component_type in self._components:
            del self._components[component_type]

            # If cognitive transistor, remove from manifold
            if hasattr(self, 'cognitive_manifold'):
                self.cognitive_manifold.unregister_transistor(component_type)
```

---

## Processing Pipeline with Events

```python
async def perceive_event(self, event_data):
    """Process external event through full pipeline."""

    # 1. Update affect (temporal model)
    affect_dict = await self.update_affect(event_data)

    # Fire affect change event
    self.OnAffectChange.invoke({
        'valence': affect_dict['valence'],
        'arousal': affect_dict['arousal'],
        'dominance': affect_dict['dominance'],
        'sorrow': affect_dict['sorrow'],
        'boredom': affect_dict['boredom'],
        'timestamp': time.time()
    })

    # 2. Generate FACS (BEFORE cognition - involuntary)
    if self.HasComponent('FacialExpressionComponent'):
        facial = self.GetComponent('FacialExpressionComponent')
        if facial.enabled:
            facs_output = await facial.process(event_data['text'], {
                'predicted_affect': affect_dict
            })

            # Fire FACS event (broadcasts to chat)
            self.OnFACSChange.invoke({
                'facs': facs_output.metadata['facs'],
                'description': describe_facs(facs_output.metadata['facs']),
                'timestamp': time.time()
            })

    # 3. Generate Laban (BEFORE cognition - involuntary)
    if self.HasComponent('BodyLanguageComponent'):
        laban = self.GetComponent('BodyLanguageComponent')
        if laban.enabled:
            laban_output = await laban.process(event_data['text'], {
                'predicted_affect': affect_dict
            })

            # Fire Laban event (broadcasts to chat)
            self.OnLabanChange.invoke({
                'laban': laban_output.metadata['laban'],
                'description': describe_laban(laban_output.metadata['laban']),
                'timestamp': time.time()
            })

    # 4. Decide response type (SAY/EMOTE/DO/THINK/NONE)
    response_decision = await self.decide_response_type(event_data)

    # 5. If NONE, stop here (FACS/Laban already broadcast!)
    if response_decision['response_type'] == 'none':
        return

    # 6. Process through cognitive transistors
    manifold_output = await self.cognitive_manifold.integrate(
        event_data['text'],
        {
            'predicted_affect': affect_dict,
            'response_decision': response_decision
        }
    )

    # 7. Fire appropriate event
    response_type = response_decision['response_type']
    if response_type == 'say':
        self.OnSpeak.invoke({'text': manifold_output, 'timestamp': time.time()})
    elif response_type == 'emote':
        self.OnEmote.invoke({'text': manifold_output, 'timestamp': time.time()})
    elif response_type == 'do':
        self.OnAction.invoke({'text': manifold_output, 'timestamp': time.time()})
    elif response_type == 'think':
        self.OnThink.invoke({'text': manifold_output, 'private': True, 'timestamp': time.time()})
```

---

## Example: Red Fire's Poker Face Fail

```python
# Red Fire has high facial salience (can't hide emotions)
noodle = world.GetAgent("agent_red_fire_anklebiter")
facial = noodle.GetComponent("FacialExpressionComponent")
facial.salience = 0.95  # Very expressive face

# Someone insults him
# He decides: response_type = NONE (trying to play it cool)

# BUT his face betrays him:
# OnFACSChange fires: AU4 (brow lowerer) = 0.8, AU9 (nose wrinkler) = 0.7
# Chat sees: "*Red Fire's face: scowling, nose wrinkled in disgust*"

# He stays silent, but everyone knows he's mad!
```

---

Shall I implement this event system now?