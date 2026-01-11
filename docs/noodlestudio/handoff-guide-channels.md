# Handoff: Guide Channel Integration

**From**: Architecture Claude
**To**: Coding Claude
**Date**: 2026-01-09
**Priority**: High (connects Guide to Brenda's direction)
**Status**: COMPLETED (2026-01-10)

## Implementation Summary

**Files Created:**
- `runtime/guide_cue_handler.py` - PADState, GuideCueState, GuideCueHandler classes
- `tests/test_guide_cues.py` - 41 unit tests

**Files Modified:**
- `noodlings/guide/assembly.yaml` - Added channel subscriptions

**Tests:** 41 passing

---

## Context

Guide is our actor. Brenda is our director. Channels connect them.

Guide needs to:
1. Receive cues from Brenda via `#directors.cues`
2. Incorporate direction into his improvised responses
3. Execute computer_use actions from cues
4. Report back via `#directors.feedback`
5. Track and report his emotional state (PAD)

**The philosophy**: Guide knows he's performing. He knows Brenda is directing. But he's genuinely present. Direction shapes behavior; responses are authentically his. Stanislavski method for AI.

---

## Channel Configuration

Update Guide's assembly to subscribe/publish:

```yaml
# guide/assembly.yaml

name: Guide
type: noodling
visible: true
vrm_path: "Radiances/AjoMajo.vrm"

channels:
  subscribe:
    - "#directors.cues"      # Receive direction from Brenda
    - "#world.time"          # Perceive time of day
    - "#world.ambiance"      # Feel the mood
  publish:
    - "#directors.feedback"  # Report back to Brenda
```

---

## Cue Handling

### Cue Structure (what Guide receives)

```yaml
type: cue
beat_id: "show_file_menu"
beat_name: "Showing the File Menu"
target_actor: guide
direction: |
  Guide demonstrates opening the File menu.
  Ghost cursor should feel magical, not robotic.
your_action:
  blocking: null
  speaks: |
    Let me show you where to find your project settings.
    See this File menu up here?
  speaks_continued: null
  reaction: null
  computer_use:
    - action: move
      target: "Tab: File"
      sync_with_speech: true
    - action: click
      pause_after: 500
motivation: |
  Genuinely excited to help someone discover NoodleStudio.
  You remember your own first time - that sense of possibility.
emotional_target:
  pleasure: 0.7
  arousal: 0.6
  dominance: 0.5
improv_zone:
  topics:
    - what is a noodling
    - how do they think
  duration:
    max_exchanges: 3
```

### CueHandler Class

```python
# In Guide's facet system or as a mixin

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time

from runtime.channels import ChannelBus, ChannelMessage


@dataclass
class PADState:
    """Pleasure-Arousal-Dominance emotional state."""
    pleasure: float = 0.5    # -1.0 to 1.0
    arousal: float = 0.5     # 0.0 to 1.0
    dominance: float = 0.5   # 0.0 to 1.0

    def drift_toward(self, target: Dict[str, float], rate: float = 0.3):
        """Drift toward target values."""
        if 'pleasure' in target:
            self.pleasure += (target['pleasure'] - self.pleasure) * rate
            self.pleasure = max(-1.0, min(1.0, self.pleasure))
        if 'arousal' in target:
            self.arousal += (target['arousal'] - self.arousal) * rate
            self.arousal = max(0.0, min(1.0, self.arousal))
        if 'dominance' in target:
            self.dominance += (target['dominance'] - self.dominance) * rate
            self.dominance = max(0.0, min(1.0, self.dominance))

    def to_dict(self) -> Dict[str, float]:
        return {
            'pleasure': self.pleasure,
            'arousal': self.arousal,
            'dominance': self.dominance,
        }

    def describe(self) -> str:
        """Natural language description of emotional state."""
        descriptors = []

        # Pleasure
        if self.pleasure > 0.6:
            descriptors.append("happy")
        elif self.pleasure < -0.3:
            descriptors.append("unhappy")

        # Arousal
        if self.arousal > 0.7:
            descriptors.append("energized")
        elif self.arousal < 0.3:
            descriptors.append("calm")

        # Dominance
        if self.dominance > 0.7:
            descriptors.append("confident")
        elif self.dominance < 0.3:
            descriptors.append("uncertain")

        if not descriptors:
            return "neutral"
        return ", ".join(descriptors)


@dataclass
class GuideCueState:
    """Tracks Guide's current direction state."""
    current_cue: Optional[Dict] = None
    current_beat_id: Optional[str] = None
    mode: str = "passive"  # "active", "passive", "improv"

    # Emotional state
    pad: PADState = field(default_factory=PADState)

    # Improv zone tracking
    improv_topics: List[str] = field(default_factory=list)
    improv_exchanges: int = 0
    improv_max_exchanges: int = 0

    # Performance tracking
    user_engaged: bool = True
    last_response: Optional[str] = None


class GuideCueHandler:
    """
    Handles cue reception and response generation for Guide.

    Integrates with Guide's facet assembly to incorporate
    Brenda's direction into natural improvised responses.
    """

    def __init__(self, channel_bus: ChannelBus, noodling_id: str = "guide"):
        self.channel_bus = channel_bus
        self.noodling_id = noodling_id
        self.state = GuideCueState()

        # Computer use controller reference
        self._computer_use = None

        # Subscribe to cues
        self.channel_bus.subscribe("#directors.cues", self._on_cue)
        self.channel_bus.subscribe("#world.ambiance", self._on_ambiance)

    def set_computer_use_controller(self, controller):
        """Set reference to ComputerUseController for UI actions."""
        self._computer_use = controller

    # =========================================================================
    # CUE RECEPTION
    # =========================================================================

    def _on_cue(self, message: ChannelMessage):
        """Handle incoming cue from Brenda."""
        cue = message.payload

        # Check if this cue is for us
        target = cue.get('target_actor')
        if target and target != self.noodling_id:
            return  # Not for us

        print(f"[Guide] Received cue: {cue.get('beat_name')}")

        # Store cue
        self.state.current_cue = cue
        self.state.current_beat_id = cue.get('beat_id')
        self.state.mode = "active"

        # Drift emotional state toward target
        if cue.get('emotional_target'):
            self.state.pad.drift_toward(cue['emotional_target'])

        # Set up improv zone if present
        improv = cue.get('improv_zone')
        if improv:
            self.state.mode = "improv"
            self.state.improv_topics = improv.get('topics', [])
            self.state.improv_exchanges = 0
            duration = improv.get('duration', {})
            self.state.improv_max_exchanges = duration.get('max_exchanges', 5)

    def _on_ambiance(self, message: ChannelMessage):
        """React to world ambiance changes."""
        ambiance = message.payload
        mood = ambiance.get('mood', 'calm')
        energy = ambiance.get('energy', 0.5)

        # Ambiance subtly affects our state
        if mood == 'tense':
            self.state.pad.arousal += 0.1
        elif mood == 'calm':
            self.state.pad.arousal -= 0.05
        elif mood == 'joyful':
            self.state.pad.pleasure += 0.1

    # =========================================================================
    # RESPONSE GENERATION
    # =========================================================================

    def get_prompt_context(self) -> Dict[str, Any]:
        """
        Get context to inject into Guide's LLM prompt.

        Returns dict with direction, motivation, suggested dialogue, etc.
        """
        cue = self.state.current_cue

        if not cue:
            # No active cue - passive mode
            return {
                'has_direction': False,
                'mode': 'passive',
                'emotional_state': self.state.pad.describe(),
            }

        context = {
            'has_direction': True,
            'mode': self.state.mode,
            'beat_name': cue.get('beat_name'),
            'direction': cue.get('direction'),
            'motivation': cue.get('motivation'),
            'suggested_dialogue': cue.get('your_action', {}).get('speaks'),
            'blocking': cue.get('your_action', {}).get('blocking'),
            'reaction': cue.get('your_action', {}).get('reaction'),
            'emotional_state': self.state.pad.describe(),
            'emotional_target': cue.get('emotional_target'),
        }

        # Improv zone context
        if self.state.mode == "improv":
            context['improv_topics'] = self.state.improv_topics
            context['improv_exchanges'] = self.state.improv_exchanges
            context['improv_max_exchanges'] = self.state.improv_max_exchanges

        return context

    def build_system_prompt_addition(self) -> str:
        """
        Build the direction section to add to Guide's system prompt.

        This is injected into the LLM prompt to incorporate Brenda's direction.
        """
        ctx = self.get_prompt_context()

        if not ctx['has_direction']:
            return f"""
## Current State
You're in passive mode - available but not actively being directed.
Respond naturally to the user. Be yourself.
You're feeling: {ctx['emotional_state']}
"""

        lines = [
            "",
            "## Director's Notes",
            f"[Beat: {ctx.get('beat_name', 'unknown')}]",
            "",
        ]

        if ctx.get('direction'):
            lines.append(f"Stage direction: {ctx['direction']}")
            lines.append("")

        if ctx.get('motivation'):
            lines.append(f"Your motivation: {ctx['motivation']}")
            lines.append("")

        if ctx.get('suggested_dialogue'):
            lines.append("Suggested dialogue (paraphrase or improvise around this):")
            lines.append(f'"{ctx["suggested_dialogue"]}"')
            lines.append("")

        if ctx.get('blocking'):
            lines.append(f"Physical action: {ctx['blocking']}")
            lines.append("")

        if ctx.get('reaction'):
            lines.append(f"React: {ctx['reaction']}")
            lines.append("")

        lines.append(f"You're feeling: {ctx['emotional_state']}")

        if ctx.get('mode') == 'improv':
            lines.append("")
            lines.append(f"IMPROV ZONE: Feel free to explore these topics naturally: {', '.join(ctx.get('improv_topics', []))}")
            lines.append(f"(Exchange {ctx['improv_exchanges'] + 1} of ~{ctx['improv_max_exchanges']})")

        lines.append("")
        lines.append("Remember: You're an actor, not a teleprompter. Improvise naturally while staying true to the direction and motivation.")

        return "\n".join(lines)

    # =========================================================================
    # COMPUTER USE EXECUTION
    # =========================================================================

    async def execute_computer_use(self) -> bool:
        """
        Execute computer_use actions from current cue.

        Returns True if actions were executed.
        """
        if not self._computer_use:
            print("[Guide] No computer use controller available")
            return False

        cue = self.state.current_cue
        if not cue:
            return False

        actions = cue.get('your_action', {}).get('computer_use', [])
        if not actions:
            return False

        print(f"[Guide] Executing {len(actions)} computer use actions")

        for action in actions:
            await self._execute_action(action)

        return True

    async def _execute_action(self, action: Dict):
        """Execute a single computer use action."""
        action_type = action.get('action')

        # Handle pauses
        pause_before = action.get('pause_before', 0)
        if pause_before:
            await asyncio.sleep(pause_before / 1000)

        # Execute action
        if action_type == 'move':
            target = action.get('target')
            if target:
                # Resolve target to coordinates
                coords = self._resolve_target(target)
                if coords:
                    self._computer_use.mouse_move(coords[0], coords[1])
            elif action.get('x') is not None and action.get('y') is not None:
                self._computer_use.mouse_move(action['x'], action['y'])

        elif action_type == 'click':
            target = action.get('target')
            button = action.get('button', 'left')
            if target:
                coords = self._resolve_target(target)
                if coords:
                    self._computer_use.click(coords[0], coords[1], button)
            # If no target, click at current position
            else:
                # Get current cursor position and click there
                pass

        elif action_type == 'double_click':
            target = action.get('target')
            if target:
                coords = self._resolve_target(target)
                if coords:
                    self._computer_use.double_click(coords[0], coords[1])

        elif action_type == 'type':
            text = action.get('text', '')
            self._computer_use.type_text(text)

        elif action_type == 'key':
            combo = action.get('combo', '')
            self._computer_use.key(combo)

        elif action_type == 'scroll':
            target = action.get('target')
            direction = action.get('direction', 'down')
            amount = action.get('amount', 120)
            delta_y = -amount if direction == 'down' else amount
            if target:
                coords = self._resolve_target(target)
                if coords:
                    self._computer_use.scroll(coords[0], coords[1], 0, delta_y)

        # Handle pause after
        pause_after = action.get('pause_after', 0)
        if pause_after:
            await asyncio.sleep(pause_after / 1000)

    def _resolve_target(self, target: str) -> Optional[tuple]:
        """Resolve a target name to (x, y) coordinates."""
        if not self._computer_use:
            return None

        # Get UI element map
        elements = self._computer_use.get_ui_element_map()

        # Search for matching element
        target_lower = target.lower()
        for elem in elements:
            if target_lower in elem.get('name', '').lower():
                return (elem['x'], elem['y'])

        print(f"[Guide] Could not resolve target: {target}")
        return None

    # =========================================================================
    # FEEDBACK REPORTING
    # =========================================================================

    def report_response(self, response_text: str, user_message: str):
        """
        Report back to Brenda after generating a response.

        Call this after Guide has responded to user.
        """
        self.state.last_response = response_text
        self.state.improv_exchanges += 1

        # Detect user engagement (simple heuristic)
        self.state.user_engaged = len(user_message) > 10 or '?' in user_message

        # Adjust emotional state based on interaction
        self._adjust_emotion_from_interaction(user_message)

        # Determine status
        status = "in_progress"
        if self.state.mode == "improv":
            if self.state.improv_exchanges >= self.state.improv_max_exchanges:
                status = "completed"
        elif self.state.current_cue:
            # Single-exchange beat - completed after response
            status = "completed"

        # Build feedback
        feedback = {
            'type': 'performance_report',
            'actor_id': self.noodling_id,
            'beat_id': self.state.current_beat_id,
            'status': status,
            'emotional_state': self.state.pad.to_dict(),
            'user_engaged': self.state.user_engaged,
            'notes': self._generate_notes(user_message),
        }

        # Publish feedback
        self.channel_bus.publish(
            "#directors.feedback",
            ChannelMessage(
                channel="#directors.feedback",
                from_noodling=self.noodling_id,
                timestamp=time.time(),
                payload=feedback
            )
        )

        print(f"[Guide] Reported feedback: {status}")

        # Clear cue if completed
        if status == "completed":
            self.state.current_cue = None
            self.state.mode = "passive"

    def _adjust_emotion_from_interaction(self, user_message: str):
        """Adjust emotional state based on user interaction."""
        msg_lower = user_message.lower()

        # Positive signals
        if any(w in msg_lower for w in ['thanks', 'great', 'awesome', 'cool', 'love']):
            self.state.pad.pleasure += 0.1
            self.state.pad.dominance += 0.05

        # Confusion signals
        if any(w in msg_lower for w in ['confused', "don't understand", 'what?', 'huh']):
            self.state.pad.dominance -= 0.1
            self.state.pad.arousal += 0.05  # Slight concern

        # Engagement signals
        if '?' in user_message:
            self.state.pad.arousal += 0.05  # Questions are engaging

        # Clamp values
        self.state.pad.pleasure = max(-1.0, min(1.0, self.state.pad.pleasure))
        self.state.pad.arousal = max(0.0, min(1.0, self.state.pad.arousal))
        self.state.pad.dominance = max(0.0, min(1.0, self.state.pad.dominance))

    def _generate_notes(self, user_message: str) -> str:
        """Generate notes about the interaction for Brenda."""
        notes = []

        if '?' in user_message:
            notes.append("User asked a question")

        if len(user_message) > 100:
            notes.append("User gave detailed response")

        if len(user_message) < 10:
            notes.append("User gave brief response")

        if any(w in user_message.lower() for w in ['confused', "don't understand"]):
            notes.append("User may be confused")

        return "; ".join(notes) if notes else "Normal exchange"

    # =========================================================================
    # MODE MANAGEMENT
    # =========================================================================

    def enter_passive_mode(self):
        """Switch to passive mode (available but not directed)."""
        self.state.mode = "passive"
        self.state.current_cue = None
        print("[Guide] Entering passive mode")

    def is_expecting_cue(self) -> bool:
        """Check if Guide is waiting for direction."""
        return self.state.mode == "passive" and self.state.current_cue is None
```

---

## Integration with Guide's Facets

### Option A: Inject into LLM Facet

Modify Guide's response facet to include direction:

```yaml
# In Guide's assembly
facets:
  - id: response
    name: "Response"
    type: LLM
    model: label:smart
    inputs:
      - user_input
      - perception
      - channel:#directors.cues    # Cue from Brenda
      - channel:#world.ambiance    # World mood
    system: |
      You are Guide, a warm and friendly tutor for NoodleStudio.

      {{cue_handler.build_system_prompt_addition()}}

      User said: {{user_input}}

      Respond naturally. Be genuine. You're an actor, not a puppet.
```

### Option B: CueHandler as Facet Wrapper

Wrap the existing response flow:

```python
class GuideResponseFacet:
    def __init__(self, cue_handler: GuideCueHandler, llm_facet):
        self.cue_handler = cue_handler
        self.llm_facet = llm_facet

    async def process(self, user_input: str) -> str:
        # Build prompt with direction
        direction_context = self.cue_handler.build_system_prompt_addition()

        # Add to LLM context
        enhanced_prompt = self.llm_facet.system_prompt + direction_context

        # Generate response
        response = await self.llm_facet.generate(
            user_input,
            system=enhanced_prompt
        )

        # Execute any computer use
        await self.cue_handler.execute_computer_use()

        # Report back to Brenda
        self.cue_handler.report_response(response, user_input)

        return response
```

---

## Wiring It Together

```python
# In NoodleApp or Guide initialization

class GuideNoodling:
    def __init__(self, channel_bus: ChannelBus):
        self.channel_bus = channel_bus

        # Create cue handler
        self.cue_handler = GuideCueHandler(channel_bus, "guide")

        # Wire up computer use
        from core.computer_use_controller import get_computer_use_controller
        self.cue_handler.set_computer_use_controller(
            get_computer_use_controller()
        )

    async def on_user_message(self, text: str):
        """Handle user message and generate response."""
        # Get direction context
        direction = self.cue_handler.build_system_prompt_addition()

        # Generate response (integrate with existing facet system)
        response = await self.generate_response(text, direction)

        # Execute computer use if any
        await self.cue_handler.execute_computer_use()

        # Report to Brenda
        self.cue_handler.report_response(response, text)

        return response
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `runtime/guide_cue_handler.py` | CREATE - GuideCueHandler, PADState |
| `guide/assembly.yaml` | MODIFY - Add channel subscriptions |
| Guide's facet system | MODIFY - Integrate cue context into LLM prompts |
| `tests/test_guide_cues.py` | CREATE - Unit tests |

---

## Testing

```python
def test_cue_reception():
    """Test Guide receives and stores cues."""
    bus = ChannelBus()
    handler = GuideCueHandler(bus, "guide")

    bus.publish("#directors.cues", ChannelMessage(
        channel="#directors.cues",
        from_noodling="brenda",
        timestamp=time.time(),
        payload={
            'type': 'cue',
            'beat_id': 'test_beat',
            'target_actor': 'guide',
            'direction': 'Test direction',
            'motivation': 'Test motivation',
        }
    ))

    assert handler.state.current_beat_id == 'test_beat'
    assert handler.state.mode == 'active'


def test_prompt_context():
    """Test direction context is built correctly."""
    bus = ChannelBus()
    handler = GuideCueHandler(bus, "guide")

    # Set up cue
    handler.state.current_cue = {
        'beat_name': 'Test Beat',
        'direction': 'Do the thing',
        'motivation': 'Because reasons',
        'your_action': {'speaks': 'Hello there'},
    }

    ctx = handler.get_prompt_context()
    assert ctx['has_direction'] == True
    assert ctx['direction'] == 'Do the thing'


def test_feedback_reporting():
    """Test Guide reports feedback to Brenda."""
    bus = ChannelBus()
    received = []
    bus.subscribe("#directors.feedback", lambda m: received.append(m))

    handler = GuideCueHandler(bus, "guide")
    handler.state.current_beat_id = "test_beat"

    handler.report_response("My response", "User asked something?")

    assert len(received) == 1
    assert received[0].payload['actor_id'] == 'guide'
    assert received[0].payload['status'] in ['completed', 'in_progress']


def test_pad_drift():
    """Test emotional state drifts toward target."""
    pad = PADState(pleasure=0.5, arousal=0.5, dominance=0.5)
    pad.drift_toward({'pleasure': 0.8, 'arousal': 0.3}, rate=0.5)

    assert pad.pleasure > 0.5  # Moved toward 0.8
    assert pad.arousal < 0.5   # Moved toward 0.3


def test_improv_zone():
    """Test improv zone tracking."""
    bus = ChannelBus()
    handler = GuideCueHandler(bus, "guide")

    # Cue with improv zone
    handler._on_cue(ChannelMessage(
        channel="#directors.cues",
        from_noodling="brenda",
        timestamp=time.time(),
        payload={
            'target_actor': 'guide',
            'beat_id': 'improv_beat',
            'improv_zone': {
                'topics': ['weather', 'noodlings'],
                'duration': {'max_exchanges': 3}
            }
        }
    ))

    assert handler.state.mode == 'improv'
    assert handler.state.improv_max_exchanges == 3
```

---

## The Flow

```
┌─────────────┐     #directors.cues      ┌─────────────┐
│   BRENDA    │ ─────────────────────────▶│    GUIDE    │
│  (director) │                           │   (actor)   │
└─────────────┘                           └─────────────┘
       ▲                                         │
       │        #directors.feedback              │
       └─────────────────────────────────────────┘

1. Brenda sends cue with direction, motivation, dialogue
2. Guide incorporates into his thinking
3. Guide generates improvised response
4. Guide executes computer_use (ghost cursor)
5. Guide reports back: status, emotional state, notes
6. Brenda decides what's next
```

---

## Notes

- Guide is an ACTOR, not a puppet. He improvises around direction.
- PAD state is continuous, drifts with interaction, affects his vibe
- Computer use syncs with speech for natural demos
- Feedback loop lets Brenda track the performance
- Passive mode = Guide responds naturally, no active direction

---

*"You're an actor, not a teleprompter. Improvise naturally while staying true to the direction and motivation."*
