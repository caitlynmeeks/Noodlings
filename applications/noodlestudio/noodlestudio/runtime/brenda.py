# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Brenda - Stage Director
#
#   An invisible noodling who directs performances. She reads
#   .play.yaml scripts, tracks state, evaluates triggers, and
#   sends cues to actors via #directors.cues. She's the conductor -
#   you never see her, but she runs the show.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.brenda
# PURPOSE:  Stage Director
# LAYER:    Studio / Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   BrendaDirector, PlayState
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import re
import time
import yaml
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .channels import ChannelBus, ChannelMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CHANNEL_CUES = "#directors.cues"
CHANNEL_FEEDBACK = "#directors.feedback"
CHANNEL_USER_INPUT = "#user.input"
CHANNEL_AMBIANCE = "#world.ambiance"
CHANNEL_EVENTS = "#world.events"


# =============================================================================
# Enums
# =============================================================================

class TriggerType(Enum):
    """Types of beat triggers."""
    SEQUENCE = "sequence"          # After previous beat completes
    AFTER = "after"                # After specific beat
    THRESHOLD = "threshold"        # Emotional condition met
    DELAY = "delay"                # Time delay
    USER_CHOICE = "user_choice"    # User made a specific choice
    USER_RESPONSE = "user_response"  # Any user response
    IMPROV_COMPLETE = "improv_complete"  # Improv zone ended
    ALL = "all"                    # Multiple conditions (AND)


class DirectorMode(Enum):
    """Brenda's operating modes."""
    ACTIVE = "active"              # Actively directing
    PASSIVE = "passive"            # Waiting, responding when addressed
    PASSIVE_AVAILABLE = "passive_available"  # Available but not pushing
    PAUSED = "paused"              # Not running


# =============================================================================
# Play State
# =============================================================================

@dataclass
class PlayState:
    """
    Tracks current state of the performance.

    This is Brenda's "script memory" - everything she knows about
    how the play is progressing.
    """
    # Beat tracking
    current_beat_id: Optional[str] = None
    current_beat_index: int = 0
    completed_beats: List[str] = field(default_factory=list)

    # Mode and zones
    mode: DirectorMode = DirectorMode.ACTIVE
    active_improv_zone: Optional[Dict] = None
    waiting_for: Optional[Dict] = None

    # Character emotional states (PAD model)
    character_states: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Conversation tracking
    last_user_message: Optional[str] = None
    last_user_choice: Optional[str] = None
    exchange_count: int = 0
    last_user_input_time: float = 0.0

    # Timing
    beat_start_times: Dict[str, float] = field(default_factory=dict)
    wait_start_time: float = 0.0

    # Actor response gate -- blocks trigger evaluation until actors respond
    awaiting_actor_response: bool = False
    awaiting_actor_response_since: float = 0.0

    def reset(self):
        """Reset state for a new performance."""
        self.current_beat_id = None
        self.current_beat_index = 0
        self.completed_beats = []
        self.mode = DirectorMode.ACTIVE
        self.active_improv_zone = None
        self.waiting_for = None
        self.last_user_message = None
        self.last_user_choice = None
        self.exchange_count = 0
        self.beat_start_times = {}
        self.wait_start_time = 0.0
        self.awaiting_actor_response = False
        self.awaiting_actor_response_since = 0.0


# =============================================================================
# Brenda Director
# =============================================================================

class BrendaDirector:
    """
    Stage director that orchestrates performances from .play.yaml scripts.

    Brenda is an invisible noodling - no rendering, just directing logic.
    She reads plays, tracks state, evaluates triggers, and sends cues
    to actors via the channel system.

    Usage:
        bus = ChannelBus()
        brenda = BrendaDirector(bus)
        brenda.load_play("path/to/play.play.yaml")
        brenda.start()

        # In main loop:
        brenda.tick()

        # Later:
        brenda.stop()
    """

    def __init__(self, channel_bus: ChannelBus):
        """
        Initialize Brenda.

        Args:
            channel_bus: The channel bus for pub/sub communication
        """
        self.channel_bus = channel_bus
        self.play_data: Optional[Dict] = None
        self.play_path: Optional[Path] = None
        self.state = PlayState()
        self._running = False

        # Callbacks for external events
        self._on_beat_change: Optional[Callable[[str], None]] = None
        self._on_mode_change: Optional[Callable[[DirectorMode], None]] = None

        # Subscribe to channels
        self.channel_bus.subscribe(CHANNEL_FEEDBACK, self._on_feedback)
        self.channel_bus.subscribe(CHANNEL_USER_INPUT, self._on_user_input)

        logger.debug("BrendaDirector initialized")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def load_play(self, play_path: str) -> bool:
        """
        Load and parse a .play.yaml file.

        Args:
            play_path: Path to the play file

        Returns:
            True if loaded successfully
        """
        try:
            path = Path(play_path)
            if not path.exists():
                logger.error(f"[Brenda] Play file not found: {play_path}")
                return False

            with open(path, 'r') as f:
                self.play_data = yaml.safe_load(f)

            self.play_path = path
            self.state.reset()

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

            logger.info(f"[Brenda] Loaded play: {self.play_data.get('title')}")
            return True

        except Exception as e:
            logger.error(f"[Brenda] Failed to load play: {e}")
            return False

    def start(self):
        """Start directing the play."""
        if not self.play_data:
            logger.warning("[Brenda] No play loaded")
            return

        self._running = True
        self.state.mode = DirectorMode.ACTIVE

        if self.state.current_beat_id:
            self.state.beat_start_times[self.state.current_beat_id] = time.time()

        # Send initial cue
        self._send_cue_for_current_beat()

        logger.info(f"[Brenda] Started directing: {self.play_data.get('title')}")

    def stop(self):
        """Stop directing."""
        self._running = False
        self.state.mode = DirectorMode.PAUSED
        logger.info("[Brenda] Stopped directing")

    def tick(self):
        """
        Called periodically to check triggers and advance state.

        Should be called every ~100-500ms from the app's main loop.
        """
        if not self._running or not self.play_data:
            return

        # Gate: don't advance while waiting for actor to respond
        if self.state.awaiting_actor_response:
            # Safety timeout: 120 seconds max wait
            elapsed = time.time() - self.state.awaiting_actor_response_since
            if elapsed < 120.0:
                return
            # Timeout -- force advance
            print(f"[Brenda] Actor response timeout after {elapsed:.0f}s, advancing", flush=True)
            self.state.awaiting_actor_response = False

        # Check timeout on wait_for
        if self.state.waiting_for:
            self._check_wait_timeout()

        # Check if we should advance to next beat
        self._check_triggers()

        # In passive mode, check for idle user
        if self.state.mode in (DirectorMode.PASSIVE, DirectorMode.PASSIVE_AVAILABLE):
            self._check_passive_triggers()

    # =========================================================================
    # Trigger Evaluation
    # =========================================================================

    def _check_triggers(self):
        """Check if any pending beat should fire."""
        beats = self.play_data.get('beats', [])

        # Look at all beats to see if any should trigger
        for i, beat in enumerate(beats):
            beat_id = beat.get('id')

            if beat_id in self.state.completed_beats:
                continue
            if beat_id == self.state.current_beat_id:
                continue

            if self._evaluate_trigger(beat):
                self._advance_to_beat(beat_id, i)
                break

    def _evaluate_trigger(self, beat: Dict) -> bool:
        """
        Check if a beat's trigger condition is satisfied.

        Args:
            beat: The beat definition

        Returns:
            True if trigger fires
        """
        trigger = beat.get('trigger', {'type': 'sequence'})
        trigger_type = trigger.get('type', 'sequence')

        if trigger_type == 'sequence':
            return self._eval_sequence_trigger(beat)

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

        elif trigger_type == 'user_response':
            # Any user response triggers
            return self.state.last_user_message is not None

        elif trigger_type == 'improv_complete':
            return self.state.active_improv_zone is None

        elif trigger_type == 'all':
            # All conditions must be true
            conditions = trigger.get('conditions', [])
            for cond in conditions:
                if not self._evaluate_trigger({'trigger': cond}):
                    return False
            return True

        return False

    def _eval_sequence_trigger(self, beat: Dict) -> bool:
        """Evaluate sequence trigger - fires after previous beat in list."""
        beats = self.play_data.get('beats', [])
        beat_id = beat.get('id')

        beat_index = next(
            (i for i, b in enumerate(beats) if b.get('id') == beat_id),
            -1
        )

        if beat_index <= 0:
            return True  # First beat

        prev_beat = beats[beat_index - 1]
        return prev_beat.get('id') in self.state.completed_beats

    def _evaluate_condition(self, condition: str) -> bool:
        """
        Evaluate a condition like 'toad.arousal > 0.85'.

        Args:
            condition: Condition string

        Returns:
            True if condition met
        """
        # Handle user_idle condition
        idle_match = re.match(r'user_idle\s*>\s*(\d+)', condition)
        if idle_match:
            idle_seconds = int(idle_match.group(1))
            time_since_input = time.time() - self.state.last_user_input_time
            return time_since_input > idle_seconds

        # Handle character.attribute comparison
        match = re.match(r'(\w+)\.(\w+)\s*(>|<|>=|<=|==)\s*([\d.]+)', condition)
        if not match:
            return False

        char_id, attr, op, value = match.groups()
        char_state = self.state.character_states.get(char_id, {})
        actual = char_state.get(attr, 0.5)
        target = float(value)

        if op == '>':
            return actual > target
        if op == '<':
            return actual < target
        if op == '>=':
            return actual >= target
        if op == '<=':
            return actual <= target
        if op == '==':
            return abs(actual - target) < 0.01

        return False

    def _check_wait_timeout(self):
        """Check if wait_for has timed out."""
        if not self.state.waiting_for:
            return

        timeout = self.state.waiting_for.get('timeout', 0)
        if timeout <= 0:
            return

        elapsed = time.time() - self.state.wait_start_time
        if elapsed >= timeout:
            # Timeout - use default
            default_choice = self.state.waiting_for.get('default')
            default_action = self.state.waiting_for.get('default_action')

            if default_choice:
                self.state.last_user_choice = default_choice
                self.state.waiting_for = None
                logger.debug(f"[Brenda] Wait timeout, defaulting to: {default_choice}")
            elif default_action == 'continue':
                self.state.waiting_for = None
                logger.debug("[Brenda] Wait timeout, continuing")

    def _check_passive_triggers(self):
        """Check triggers that fire from passive mode."""
        current_beat = self._get_current_beat()
        if not current_beat:
            return

        triggers = current_beat.get('triggers_from_passive', [])
        for trigger in triggers:
            pattern = trigger.get('pattern', '').lower()
            condition = trigger.get('condition')

            # Check pattern match against last message
            if pattern and self.state.last_user_message:
                if pattern in self.state.last_user_message.lower():
                    response = trigger.get('response')
                    next_beat = trigger.get('next_beat')

                    if response:
                        # Send quick response (simple cue)
                        self._send_quick_response(response)
                    if next_beat:
                        beat_index = self._get_beat_index(next_beat)
                        if beat_index >= 0:
                            self._advance_to_beat(next_beat, beat_index)

                    # Clear the message so we don't re-trigger
                    self.state.last_user_message = None
                    return

            # Check condition
            if condition and self._evaluate_condition(condition):
                next_beat = trigger.get('next_beat')
                if next_beat:
                    beat_index = self._get_beat_index(next_beat)
                    if beat_index >= 0:
                        self._advance_to_beat(next_beat, beat_index)
                return

    # =========================================================================
    # Beat Management
    # =========================================================================

    def _advance_to_beat(self, beat_id: str, beat_index: int):
        """
        Advance to a new beat.

        Args:
            beat_id: ID of the beat to advance to
            beat_index: Index in the beats list
        """
        # Mark current as completed
        if self.state.current_beat_id:
            if self.state.current_beat_id not in self.state.completed_beats:
                self.state.completed_beats.append(self.state.current_beat_id)

        # Update state
        self.state.current_beat_id = beat_id
        self.state.current_beat_index = beat_index
        self.state.beat_start_times[beat_id] = time.time()
        self.state.exchange_count = 0
        self.state.active_improv_zone = None
        self.state.waiting_for = None
        self.state.last_user_choice = None

        logger.info(f"[Brenda] Advancing to beat: {beat_id}")

        # Callback
        if self._on_beat_change:
            self._on_beat_change(beat_id)

        # Send cue for new beat
        self._send_cue_for_current_beat()

    def _get_current_beat(self) -> Optional[Dict]:
        """Get the current beat definition."""
        if not self.state.current_beat_id:
            return None
        beats = self.play_data.get('beats', [])
        return next(
            (b for b in beats if b.get('id') == self.state.current_beat_id),
            None
        )

    def _get_beat_index(self, beat_id: str) -> int:
        """Get the index of a beat by ID."""
        beats = self.play_data.get('beats', [])
        for i, beat in enumerate(beats):
            if beat.get('id') == beat_id:
                return i
        return -1

    # =========================================================================
    # Cue Sending
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

            # Check if actor should enter on this beat
            enters = character.get('enters')
            if enters and enters != beat.get('id'):
                if enters not in self.state.completed_beats:
                    continue  # Not their entrance yet

            # Check if actor has exited
            exits = character.get('exits')
            if exits and exits in self.state.completed_beats:
                continue  # Already exited

            if not actor_beat and not beat.get('direction'):
                continue  # Nothing for this actor

            cue = self._build_cue(beat, actor_id, actor_beat, character)
            self._send_cue(cue)

        # Handle beat-level state changes
        self._process_beat_state_changes(beat)

    def _build_cue(self, beat: Dict, actor_id: str,
                   actor_beat: Dict, character: Dict) -> Dict:
        """
        Build a cue message for an actor.

        Args:
            beat: The beat definition
            actor_id: The actor's ID
            actor_beat: Actor-specific beat actions
            character: Character definition

        Returns:
            Cue payload dict
        """
        # Calculate emotional target
        pad_drift = actor_beat.get('pad_drift', {})
        emotional_target = self._calculate_emotional_target(
            actor_id, character, pad_drift
        )

        # Build your_action from actor_beat
        your_action = {}
        for key in ['blocking', 'speaks', 'speaks_continued', 'reaction',
                    'computer_use']:
            if key in actor_beat:
                your_action[key] = actor_beat[key]

        return {
            'type': 'cue',
            'beat_id': beat.get('id'),
            'beat_name': beat.get('name'),
            'direction': beat.get('direction', ''),
            'target_actor': actor_id,
            'your_action': your_action,
            'motivation': character.get('motivation'),
            'emotional_target': emotional_target,
            'improv_zone': beat.get('improv_zone'),
        }

    def _calculate_emotional_target(self, actor_id: str,
                                     character: Dict,
                                     pad_drift: Dict) -> Dict[str, float]:
        """
        Apply PAD drift to get target emotional state.

        Args:
            actor_id: Actor ID
            character: Character definition
            pad_drift: Drift to apply

        Returns:
            Target PAD state
        """
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
                # Small values are relative, large values are absolute
                if abs(drift) < 1:
                    target[dim] = base + float(drift)
                else:
                    target[dim] = float(drift)
            else:
                target[dim] = base

            # Clamp to valid ranges
            if dim == 'pleasure':
                target[dim] = max(-1.0, min(1.0, target[dim]))
            else:
                target[dim] = max(0.0, min(1.0, target[dim]))

        # Update state
        self.state.character_states[actor_id] = target

        return target

    def _process_beat_state_changes(self, beat: Dict):
        """Process state changes defined in the beat."""
        # Check for improv zone
        if beat.get('improv_zone'):
            self.state.active_improv_zone = beat['improv_zone']
            logger.debug(f"[Brenda] Entering improv zone: {beat['improv_zone'].get('topics', [])}")

        # Check for wait_for
        if beat.get('wait_for'):
            self.state.waiting_for = beat['wait_for']
            self.state.wait_start_time = time.time()
            logger.debug(f"[Brenda] Waiting for: {beat['wait_for'].get('type')}")

        # Check for set_mode
        if beat.get('set_mode'):
            mode_str = beat['set_mode']
            try:
                new_mode = DirectorMode(mode_str)
            except ValueError:
                # Try mapping common variations
                mode_map = {
                    'passive_available': DirectorMode.PASSIVE_AVAILABLE,
                    'passive': DirectorMode.PASSIVE,
                    'active': DirectorMode.ACTIVE,
                }
                new_mode = mode_map.get(mode_str, DirectorMode.ACTIVE)

            self.state.mode = new_mode
            logger.debug(f"[Brenda] Mode changed to: {new_mode.value}")

            if self._on_mode_change:
                self._on_mode_change(new_mode)

    def _send_cue(self, cue: Dict):
        """Publish a cue to #directors.cues and wait for actor response."""
        print(f"[Brenda] Sending cue -> {cue.get('target_actor')}: "
              f"beat={cue.get('beat_name')}, direction={cue.get('direction', '')[:60]}...", flush=True)
        self.channel_bus.publish(
            CHANNEL_CUES,
            ChannelMessage(
                channel=CHANNEL_CUES,
                from_noodling="brenda",
                timestamp=time.time(),
                payload=cue
            )
        )

        # Gate: wait for actor feedback before advancing to next beat
        self.state.awaiting_actor_response = True
        self.state.awaiting_actor_response_since = time.time()

        logger.debug(f"[Brenda] Sent cue to {cue.get('target_actor')}: {cue.get('beat_name')}")

    def _send_quick_response(self, text: str):
        """Send a quick response cue (for passive mode triggers)."""
        cue = {
            'type': 'quick_response',
            'text': text,
            'target_actor': 'guide',  # Default to guide
        }
        self._send_cue(cue)

    # =========================================================================
    # Input Handling
    # =========================================================================

    def _on_user_input(self, message: ChannelMessage):
        """Handle user input from #user.input."""
        if not self._running:
            return

        user_text = message.payload.get('text', '')
        self.state.last_user_message = user_text
        self.state.last_user_input_time = time.time()
        self.state.exchange_count += 1

        logger.debug(f"[Brenda] User input: {user_text[:50]}...")

        # Check if this matches a wait_for choice
        if self.state.waiting_for:
            choice = self._classify_user_choice(user_text)
            if choice:
                self.state.last_user_choice = choice
                self.state.waiting_for = None
                logger.debug(f"[Brenda] User choice classified as: {choice}")

        # Check improv zone exit conditions
        if self.state.active_improv_zone:
            if self._should_exit_improv():
                logger.debug("[Brenda] Exiting improv zone")
                self.state.active_improv_zone = None

    def _classify_user_choice(self, text: str) -> Optional[str]:
        """
        Check if user text matches any expected choice patterns.

        Args:
            text: User input text

        Returns:
            Choice ID if matched, None otherwise
        """
        if not self.state.waiting_for:
            return None

        options = self.state.waiting_for.get('options', [])
        text_lower = text.lower().strip()

        for option in options:
            patterns = option.get('patterns', [])
            for pattern in patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in text_lower:
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

        # Check max time
        max_seconds = duration.get('max_seconds')
        if max_seconds:
            beat_start = self.state.beat_start_times.get(
                self.state.current_beat_id, time.time()
            )
            if (time.time() - beat_start) >= max_seconds:
                return True

        # Check exit conditions
        exit_conditions = zone.get('exit_conditions', [])
        for cond in exit_conditions:
            cond_type = cond.get('type')

            if cond_type == 'topic_exhausted':
                # Would need topic tracking - skip for now
                pass

            elif cond_type == 'threshold':
                condition = cond.get('condition', '')
                if self._evaluate_condition(condition):
                    return True

            elif cond_type == 'user_interrupt':
                # Check if user is trying to move on
                interrupt_patterns = ['anyway', 'moving on', 'next', 'continue']
                if self.state.last_user_message:
                    msg_lower = self.state.last_user_message.lower()
                    if any(p in msg_lower for p in interrupt_patterns):
                        return True

        return False

    def _on_feedback(self, message: ChannelMessage):
        """Handle feedback from actors via #directors.feedback."""
        if not self._running:
            return

        payload = message.payload
        actor_id = payload.get('actor_id')
        status = payload.get('status')

        print(f"[Brenda] Feedback from {actor_id}: status={status}, "
              f"beat={payload.get('beat_id')}", flush=True)

        # Update character state from feedback
        if actor_id and payload.get('emotional_state'):
            self.state.character_states[actor_id] = payload['emotional_state']

        # Actor responded -- clear the gate
        self.state.awaiting_actor_response = False

        # Check if beat completed
        if status == 'completed':
            beat_id = payload.get('beat_id')
            if beat_id == self.state.current_beat_id:
                # Current beat completed, check what's next
                self._check_triggers()

    # =========================================================================
    # World Control
    # =========================================================================

    def set_ambiance(self, mood: str, energy: Optional[float] = None):
        """
        Set world ambiance (publishes to #world.ambiance).

        Args:
            mood: Mood string (calm, tense, joyful, etc.)
            energy: Optional energy level (0.0 to 1.0)
        """
        payload = {'type': 'ambiance', 'mood': mood}
        if energy is not None:
            payload['energy'] = energy

        self.channel_bus.publish(
            CHANNEL_AMBIANCE,
            ChannelMessage(
                channel=CHANNEL_AMBIANCE,
                from_noodling="brenda",
                timestamp=time.time(),
                payload=payload
            )
        )

    def trigger_event(self, event_type: str, source: str, description: str,
                      **kwargs):
        """
        Trigger a world event (publishes to #world.events).

        Args:
            event_type: Type of event (sound, visual, physical, social)
            source: What caused the event
            description: Human-readable description
            **kwargs: Additional event data
        """
        self.channel_bus.publish(
            CHANNEL_EVENTS,
            ChannelMessage(
                channel=CHANNEL_EVENTS,
                from_noodling="brenda",
                timestamp=time.time(),
                payload={
                    'type': 'event',
                    'event_type': event_type,
                    'source': source,
                    'description': description,
                    **kwargs
                }
            )
        )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_beat_change(self, callback: Callable[[str], None]):
        """Register callback for beat changes."""
        self._on_beat_change = callback

    def on_mode_change(self, callback: Callable[[DirectorMode], None]):
        """Register callback for mode changes."""
        self._on_mode_change = callback

    # =========================================================================
    # Introspection
    # =========================================================================

    def get_play_info(self) -> Dict[str, Any]:
        """Get info about the loaded play."""
        if not self.play_data:
            return {}

        return {
            'title': self.play_data.get('title'),
            'author': self.play_data.get('author'),
            'setting': self.play_data.get('setting'),
            'beat_count': len(self.play_data.get('beats', [])),
            'characters': list(self.play_data.get('characters', {}).keys()),
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current performance state."""
        return {
            'running': self._running,
            'mode': self.state.mode.value,
            'current_beat_id': self.state.current_beat_id,
            'completed_beats': self.state.completed_beats,
            'character_states': self.state.character_states,
            'exchange_count': self.state.exchange_count,
            'in_improv_zone': self.state.active_improv_zone is not None,
            'waiting_for_choice': self.state.waiting_for is not None,
        }


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
