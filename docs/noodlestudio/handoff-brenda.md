# Handoff: Brenda Stage Director Implementation

**From**: Architecture Claude
**To**: Coding Claude
**Date**: 2026-01-09
**Priority**: High (orchestrates Let's Consciousness)

---

## Context

Brenda is an **invisible noodling** who directs performances. She reads `.play.yaml` scripts, tracks state, evaluates triggers, and sends cues to actors via `#directors.cues`. She's the conductor - you never see her, but she runs the show.

Full design spec: `/docs/noodlestudio/brenda-assembly.md`
Play format spec: `/docs/noodlestudio/play-format.md`
Example play: `/docs/noodlestudio/plays/lets_consciousness_intro.play.yaml`

---

## What Brenda Does

1. **Loads a play** - Parses `.play.yaml` file
2. **Tracks state** - Current beat, completed beats, character emotional states
3. **Evaluates triggers** - Checks if conditions are met to advance
4. **Classifies user input** - Figures out if user made a choice, asked a question, etc.
5. **Makes decisions** - Advance beat? Send cue? Wait? End improv zone?
6. **Sends cues** - Publishes to `#directors.cues` for actors
7. **Listens to feedback** - Subscribes to `#directors.feedback` from actors
8. **Controls world** - Can set ambiance, trigger events

---

## Implementation Approach

Rather than implementing Brenda as a full facet assembly initially, I recommend a **staged approach**:

### Stage 1: BrendaDirector Class (Python)

Create a Python class that encapsulates Brenda's logic. This can later be exposed as facets.

```python
# runtime/brenda.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml
import time
import re

from .channels import ChannelBus, ChannelMessage


class BeatTriggerType(Enum):
    SEQUENCE = "sequence"      # After previous beat
    AFTER = "after"            # After specific beat
    THRESHOLD = "threshold"    # Emotional condition
    DELAY = "delay"            # Time delay
    USER_CHOICE = "user_choice"
    IMPROV_COMPLETE = "improv_complete"


@dataclass
class PlayState:
    """Tracks current state of the performance."""
    current_beat_id: Optional[str] = None
    current_beat_index: int = 0
    completed_beats: List[str] = field(default_factory=list)
    active_improv_zone: Optional[Dict] = None
    waiting_for: Optional[Dict] = None
    mode: str = "active"  # active, passive, paused

    # Character emotional states (PAD)
    character_states: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Conversation tracking
    last_user_message: Optional[str] = None
    last_user_choice: Optional[str] = None
    exchange_count: int = 0

    # Timing
    beat_start_times: Dict[str, float] = field(default_factory=dict)


class BrendaDirector:
    """
    Stage director that orchestrates performances from .play.yaml scripts.

    Invisible noodling - no rendering, just logic.
    """

    def __init__(self, channel_bus: ChannelBus):
        self.channel_bus = channel_bus
        self.play_data: Optional[Dict] = None
        self.state = PlayState()
        self._running = False

        # Subscribe to channels
        self.channel_bus.subscribe("#directors.feedback", self._on_feedback)
        self.channel_bus.subscribe("#user.input", self._on_user_input)

    def load_play(self, play_path: str) -> bool:
        """Load and parse a .play.yaml file."""
        try:
            with open(play_path, 'r') as f:
                self.play_data = yaml.safe_load(f)

            # Initialize character states from play
            characters = self.play_data.get('characters', {})
            for char_id, char_def in characters.items():
                initial_pad = char_def.get('initial_pad', {})
                self.state.character_states[char_id] = {
                    'pleasure': initial_pad.get('pleasure', 0.5),
                    'arousal': initial_pad.get('arousal', 0.5),
                    'dominance': initial_pad.get('dominance', 0.5),
                }

            # Find first beat
            beats = self.play_data.get('beats', [])
            if beats:
                self.state.current_beat_id = beats[0].get('id')
                self.state.current_beat_index = 0

            print(f"[Brenda] Loaded play: {self.play_data.get('title')}")
            return True

        except Exception as e:
            print(f"[Brenda] Failed to load play: {e}")
            return False

    def start(self):
        """Start directing the play."""
        if not self.play_data:
            print("[Brenda] No play loaded")
            return

        self._running = True
        self.state.beat_start_times[self.state.current_beat_id] = time.time()

        # Send initial cue
        self._send_cue_for_current_beat()
        print(f"[Brenda] Started directing: {self.play_data.get('title')}")

    def stop(self):
        """Stop directing."""
        self._running = False
        print("[Brenda] Stopped directing")

    def tick(self):
        """
        Called periodically to check triggers and advance state.
        Should be called every ~500ms or so.
        """
        if not self._running or not self.play_data:
            return

        # Check if we should advance to next beat
        self._check_triggers()

    # =========================================================================
    # TRIGGER EVALUATION
    # =========================================================================

    def _check_triggers(self):
        """Check if any pending beat should fire."""
        beats = self.play_data.get('beats', [])
        current_index = self.state.current_beat_index

        # Look at next beats to see if any should trigger
        for i, beat in enumerate(beats):
            if beat['id'] in self.state.completed_beats:
                continue
            if beat['id'] == self.state.current_beat_id:
                continue

            if self._evaluate_trigger(beat):
                self._advance_to_beat(beat['id'], i)
                break

    def _evaluate_trigger(self, beat: Dict) -> bool:
        """Check if a beat's trigger condition is satisfied."""
        trigger = beat.get('trigger', {'type': 'sequence'})
        trigger_type = trigger.get('type', 'sequence')

        if trigger_type == 'sequence':
            # Fire after previous beat in list
            beats = self.play_data.get('beats', [])
            beat_index = next(
                (i for i, b in enumerate(beats) if b['id'] == beat['id']),
                -1
            )
            if beat_index <= 0:
                return True  # First beat
            prev_beat = beats[beat_index - 1]
            return prev_beat['id'] in self.state.completed_beats

        elif trigger_type == 'after':
            target = trigger.get('beat')
            return target in self.state.completed_beats

        elif trigger_type == 'threshold':
            condition = trigger.get('condition', '')
            return self._evaluate_condition(condition)

        elif trigger_type == 'delay':
            seconds = trigger.get('seconds', 0)
            beat_start = self.state.beat_start_times.get(
                self.state.current_beat_id, time.time()
            )
            return (time.time() - beat_start) >= seconds

        elif trigger_type == 'user_choice':
            expected = trigger.get('choice')
            return self.state.last_user_choice == expected

        elif trigger_type == 'improv_complete':
            return self.state.active_improv_zone is None

        return False

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition like 'toad.arousal > 0.85'."""
        match = re.match(r'(\w+)\.(\w+)\s*(>|<|>=|<=|==)\s*([\d.]+)', condition)
        if not match:
            return False

        char_id, attr, op, value = match.groups()
        char_state = self.state.character_states.get(char_id, {})
        actual = char_state.get(attr, 0.5)
        target = float(value)

        if op == '>': return actual > target
        if op == '<': return actual < target
        if op == '>=': return actual >= target
        if op == '<=': return actual <= target
        if op == '==': return abs(actual - target) < 0.01

        return False

    # =========================================================================
    # BEAT MANAGEMENT
    # =========================================================================

    def _advance_to_beat(self, beat_id: str, beat_index: int):
        """Advance to a new beat."""
        # Mark current as completed
        if self.state.current_beat_id:
            self.state.completed_beats.append(self.state.current_beat_id)

        # Update state
        self.state.current_beat_id = beat_id
        self.state.current_beat_index = beat_index
        self.state.beat_start_times[beat_id] = time.time()
        self.state.exchange_count = 0
        self.state.active_improv_zone = None
        self.state.last_user_choice = None

        print(f"[Brenda] Advancing to beat: {beat_id}")

        # Send cue for new beat
        self._send_cue_for_current_beat()

    def _get_current_beat(self) -> Optional[Dict]:
        """Get the current beat definition."""
        if not self.state.current_beat_id:
            return None
        beats = self.play_data.get('beats', [])
        return next(
            (b for b in beats if b['id'] == self.state.current_beat_id),
            None
        )

    # =========================================================================
    # CUE SENDING
    # =========================================================================

    def _send_cue_for_current_beat(self):
        """Send cues to actors for the current beat."""
        beat = self._get_current_beat()
        if not beat:
            return

        characters = self.play_data.get('characters', {})
        on_stage = beat.get('on_stage', [])

        for actor_id in on_stage:
            actor_beat = beat.get(actor_id, {})
            character = characters.get(actor_id, {})

            if not actor_beat and not beat.get('direction'):
                continue  # Nothing for this actor

            cue = self._build_cue(beat, actor_id, actor_beat, character)
            self._send_cue(cue)

        # Check for improv zone
        if beat.get('improv_zone'):
            self.state.active_improv_zone = beat['improv_zone']

        # Check for wait_for
        if beat.get('wait_for'):
            self.state.waiting_for = beat['wait_for']

        # Check for set_mode
        if beat.get('set_mode'):
            self.state.mode = beat['set_mode']

    def _build_cue(self, beat: Dict, actor_id: str,
                   actor_beat: Dict, character: Dict) -> Dict:
        """Build a cue message for an actor."""
        # Calculate emotional target
        pad_drift = actor_beat.get('pad_drift', {})
        emotional_target = self._calculate_emotional_target(
            actor_id, character, pad_drift
        )

        return {
            'type': 'cue',
            'beat_id': beat.get('id'),
            'beat_name': beat.get('name'),
            'direction': beat.get('direction', ''),
            'target_actor': actor_id,
            'your_action': {
                'blocking': actor_beat.get('blocking'),
                'speaks': actor_beat.get('speaks'),
                'speaks_continued': actor_beat.get('speaks_continued'),
                'reaction': actor_beat.get('reaction'),
                'computer_use': actor_beat.get('computer_use'),
            },
            'motivation': character.get('motivation'),
            'emotional_target': emotional_target,
            'improv_zone': beat.get('improv_zone'),
        }

    def _calculate_emotional_target(self, actor_id: str,
                                     character: Dict,
                                     pad_drift: Dict) -> Dict[str, float]:
        """Apply PAD drift to get target emotional state."""
        current = self.state.character_states.get(actor_id, {})
        initial = character.get('initial_pad', {})

        target = {}
        for dim in ['pleasure', 'arousal', 'dominance']:
            base = current.get(dim, initial.get(dim, 0.5))
            drift = pad_drift.get(dim, 0)

            if isinstance(drift, str):
                if drift.startswith('+'):
                    target[dim] = base + float(drift[1:])
                elif drift.startswith('-'):
                    target[dim] = base - float(drift[1:])
                else:
                    target[dim] = float(drift)
            elif drift != 0:
                target[dim] = base + float(drift) if abs(drift) < 1 else float(drift)
            else:
                target[dim] = base

            # Clamp
            if dim == 'pleasure':
                target[dim] = max(-1.0, min(1.0, target[dim]))
            else:
                target[dim] = max(0.0, min(1.0, target[dim]))

        # Update state
        self.state.character_states[actor_id] = target

        return target

    def _send_cue(self, cue: Dict):
        """Publish a cue to #directors.cues."""
        self.channel_bus.publish(
            "#directors.cues",
            ChannelMessage(
                channel="#directors.cues",
                from_noodling="brenda",
                timestamp=time.time(),
                payload=cue
            )
        )
        print(f"[Brenda] Sent cue to {cue.get('target_actor')}: {cue.get('beat_name')}")

    # =========================================================================
    # INPUT HANDLING
    # =========================================================================

    def _on_user_input(self, message: ChannelMessage):
        """Handle user input from #user.input."""
        if not self._running:
            return

        user_text = message.payload.get('text', '')
        self.state.last_user_message = user_text
        self.state.exchange_count += 1

        # Check if this matches a wait_for choice
        if self.state.waiting_for:
            choice = self._classify_user_choice(user_text)
            if choice:
                self.state.last_user_choice = choice
                self.state.waiting_for = None

        # Check improv zone exit conditions
        if self.state.active_improv_zone:
            if self._should_exit_improv():
                self.state.active_improv_zone = None

    def _classify_user_choice(self, text: str) -> Optional[str]:
        """Check if user text matches any expected choice patterns."""
        if not self.state.waiting_for:
            return None

        options = self.state.waiting_for.get('options', [])
        text_lower = text.lower()

        for option in options:
            patterns = option.get('patterns', [])
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    return option.get('id')

        return None

    def _should_exit_improv(self) -> bool:
        """Check if improv zone should end."""
        zone = self.state.active_improv_zone
        if not zone:
            return True

        duration = zone.get('duration', {})

        # Check max exchanges
        max_exchanges = duration.get('max_exchanges')
        if max_exchanges and self.state.exchange_count >= max_exchanges:
            return True

        # Could add time-based checks, condition checks, etc.

        return False

    def _on_feedback(self, message: ChannelMessage):
        """Handle feedback from actors via #directors.feedback."""
        if not self._running:
            return

        payload = message.payload
        actor_id = payload.get('actor_id')

        # Update character state from feedback
        if actor_id and payload.get('emotional_state'):
            self.state.character_states[actor_id] = payload['emotional_state']

        # Check if beat completed
        if payload.get('status') == 'completed':
            beat_id = payload.get('beat_id')
            if beat_id == self.state.current_beat_id:
                # Current beat completed, check what's next
                self._check_triggers()

        print(f"[Brenda] Received feedback from {actor_id}: {payload.get('status')}")

    # =========================================================================
    # WORLD CONTROL
    # =========================================================================

    def set_ambiance(self, mood: str, energy: float = None):
        """Set world ambiance (delegates to world channels)."""
        payload = {'mood': mood}
        if energy is not None:
            payload['energy'] = energy

        self.channel_bus.publish(
            "#world.ambiance",
            ChannelMessage(
                channel="#world.ambiance",
                from_noodling="brenda",
                timestamp=time.time(),
                payload=payload
            )
        )

    def trigger_event(self, event_type: str, source: str, description: str):
        """Trigger a world event."""
        self.channel_bus.publish(
            "#world.events",
            ChannelMessage(
                channel="#world.events",
                from_noodling="brenda",
                timestamp=time.time(),
                payload={
                    'type': 'event',
                    'event_type': event_type,
                    'source': source,
                    'description': description,
                }
            )
        )
```

### Stage 2: Integration with NoodleApp

```python
# In runtime/app.py

class NoodleApp:
    def __init__(self, ...):
        self.channel_bus = ChannelBus()
        self.world_channels = WorldChannelService(self.channel_bus)
        self.director: Optional[BrendaDirector] = None

    def load_director(self, play_path: str):
        """Load Brenda with a play."""
        self.director = BrendaDirector(self.channel_bus)
        self.director.load_play(play_path)

    def start_performance(self):
        """Start the directed performance."""
        if self.director:
            self.director.start()

    def tick(self):
        """Main loop tick - call director tick."""
        if self.director:
            self.director.tick()
```

### Stage 3: User Input Publishing

Ensure user messages get published to `#user.input`:

```python
# Wherever user input is received (chat panel, etc.)
def on_user_message(text: str):
    app.channel_bus.publish(
        "#user.input",
        ChannelMessage(
            channel="#user.input",
            from_noodling="user",
            timestamp=time.time(),
            payload={'text': text}
        )
    )
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `runtime/brenda.py` | CREATE - BrendaDirector class |
| `runtime/app.py` | MODIFY - Add director integration |
| `tests/test_brenda.py` | CREATE - Unit tests |

---

## Testing Strategy

### Unit Tests

```python
def test_load_play():
    """Test loading a .play.yaml file."""
    bus = ChannelBus()
    brenda = BrendaDirector(bus)
    assert brenda.load_play("docs/noodlestudio/plays/lets_consciousness_intro.play.yaml")
    assert brenda.play_data['title'] == "Let's Consciousness"

def test_trigger_sequence():
    """Test sequence triggers fire after previous beat."""
    # ...

def test_trigger_threshold():
    """Test emotional threshold triggers."""
    # ...

def test_user_choice_matching():
    """Test user input matches choice patterns."""
    # ...

def test_cue_sending():
    """Test cues are published to #directors.cues."""
    bus = ChannelBus()
    received = []
    bus.subscribe("#directors.cues", lambda m: received.append(m))

    brenda = BrendaDirector(bus)
    brenda.load_play(test_play_path)
    brenda.start()

    assert len(received) > 0
    assert received[0].payload['type'] == 'cue'

def test_feedback_handling():
    """Test feedback updates character state."""
    # ...

def test_improv_zone_exit():
    """Test improv zone exits after max exchanges."""
    # ...
```

### Integration Test

```python
def test_full_performance():
    """Test a complete performance flow."""
    app = NoodleApp()
    app.load_director("lets_consciousness_intro.play.yaml")
    app.start_performance()

    # Simulate user choosing tour
    app.channel_bus.publish("#user.input", ChannelMessage(
        channel="#user.input",
        from_noodling="user",
        timestamp=time.time(),
        payload={'text': "Show me around"}
    ))

    app.tick()

    # Verify we advanced to tour_start beat
    assert app.director.state.current_beat_id == "tour_start"
```

---

## Cue Format Reference

What Brenda sends via `#directors.cues`:

```yaml
type: cue
beat_id: "show_file_menu"
beat_name: "Showing the File Menu"
target_actor: guide
direction: |
  Guide demonstrates opening the File menu.
your_action:
  blocking: null
  speaks: |
    Let me show you where to find your project settings.
  computer_use:
    - action: move
      target: "Tab: File"
motivation: |
  Genuinely excited to help someone discover NoodleStudio.
emotional_target:
  pleasure: 0.7
  arousal: 0.6
  dominance: 0.5
improv_zone: null
```

---

## Next After Brenda

Once Brenda is working:

1. **Guide's channel integration** - Subscribe to `#directors.cues`, handle cues, report to `#directors.feedback`
2. **Unfold UX** - The panel animation for View Project
3. **Let's Consciousness assembly** - Wire Guide + Brenda + the play together

---

## Notes

- Start with the Python class approach - it's testable and debuggable
- Later we can expose Brenda's components as facets for the dogfooding principle
- The LLM-based user classification can come in Stage 2; for now, pattern matching works
- Focus on the happy path first: load play → start → send cues → handle choices → advance

---

*"A good director is invisible. You only see the performance."*
