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
#   Guide Cue Handler
#
#   Handles cue reception and response generation for Guide.
#   Guide is an actor, not a puppet. He receives direction from Brenda
#   via #directors.cues and incorporates it into improvised responses.
#   He reports back via #directors.feedback.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.guide_cue_handler
# PURPOSE:  Guide Channel Integration
# LAYER:    Studio / Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PADState, GuideCueState, GuideCueHandler
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .channels import ChannelBus, ChannelMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CHANNEL_CUES = "#directors.cues"
CHANNEL_FEEDBACK = "#directors.feedback"
CHANNEL_AMBIANCE = "#world.ambiance"


# =============================================================================
# PAD State (Pleasure-Arousal-Dominance)
# =============================================================================

@dataclass
class PADState:
    """
    Pleasure-Arousal-Dominance emotional state.

    Guide's continuous emotional state that drifts over time
    and with direction from Brenda.

    Ranges:
        pleasure: -1.0 to +1.0 (unhappy to happy)
        arousal: 0.0 to 1.0 (calm to energized)
        dominance: 0.0 to 1.0 (submissive to dominant)
    """
    pleasure: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5

    def drift_toward(self, target: Dict[str, float], rate: float = 0.3):
        """
        Drift emotional state toward target values.

        Args:
            target: Dict with 'pleasure', 'arousal', 'dominance' keys
            rate: How fast to drift (0.0-1.0, higher = faster)
        """
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
        """Serialize to dictionary."""
        return {
            'pleasure': round(self.pleasure, 3),
            'arousal': round(self.arousal, 3),
            'dominance': round(self.dominance, 3),
        }

    def describe(self) -> str:
        """
        Natural language description of emotional state.

        Returns a comma-separated list of emotional descriptors,
        or "neutral" if no strong signals.
        """
        descriptors = []

        # Pleasure dimension
        if self.pleasure > 0.6:
            descriptors.append("happy")
        elif self.pleasure > 0.3:
            descriptors.append("content")
        elif self.pleasure < -0.3:
            descriptors.append("unhappy")
        elif self.pleasure < 0.0:
            descriptors.append("slightly uneasy")

        # Arousal dimension
        if self.arousal > 0.7:
            descriptors.append("energized")
        elif self.arousal > 0.5:
            descriptors.append("engaged")
        elif self.arousal < 0.3:
            descriptors.append("calm")

        # Dominance dimension
        if self.dominance > 0.7:
            descriptors.append("confident")
        elif self.dominance > 0.5:
            descriptors.append("assured")
        elif self.dominance < 0.3:
            descriptors.append("uncertain")

        if not descriptors:
            return "neutral"
        return ", ".join(descriptors)

    @staticmethod
    def from_dict(data: Dict[str, float]) -> 'PADState':
        """Create PADState from dictionary."""
        return PADState(
            pleasure=data.get('pleasure', 0.5),
            arousal=data.get('arousal', 0.5),
            dominance=data.get('dominance', 0.5),
        )


# =============================================================================
# Guide Cue State
# =============================================================================

@dataclass
class GuideCueState:
    """
    Tracks Guide's current direction state.

    Contains the current cue, emotional state, improv zone tracking,
    and performance metrics.
    """
    # Current direction
    current_cue: Optional[Dict] = None
    current_beat_id: Optional[str] = None
    mode: str = "passive"  # "active", "passive", "improv"

    # Emotional state (PAD model)
    pad: PADState = field(default_factory=PADState)

    # Improv zone tracking
    improv_topics: List[str] = field(default_factory=list)
    improv_exchanges: int = 0
    improv_max_exchanges: int = 0

    # Performance tracking
    user_engaged: bool = True
    last_response: Optional[str] = None
    last_user_message: Optional[str] = None


# =============================================================================
# Guide Cue Handler
# =============================================================================

class GuideCueHandler:
    """
    Handles cue reception and response generation for Guide.

    Integrates with Guide's facet assembly to incorporate
    Brenda's direction into natural improvised responses.

    Philosophy: Guide knows he's performing. He knows Brenda is directing.
    But he's genuinely present. Direction shapes behavior; responses are
    authentically his. Stanislavski method for AI.

    Usage:
        bus = ChannelBus()
        handler = GuideCueHandler(bus, "guide")

        # Set computer use controller for UI actions
        handler.set_computer_use_controller(controller)

        # In response generation:
        direction = handler.build_system_prompt_addition()
        response = await llm.generate(user_input, system=base_prompt + direction)

        # After responding:
        handler.report_response(response, user_message)
    """

    def __init__(self, channel_bus: ChannelBus, noodling_id: str = "guide"):
        """
        Initialize the cue handler.

        Args:
            channel_bus: The channel bus for pub/sub
            noodling_id: This noodling's ID (for filtering cues)
        """
        self.channel_bus = channel_bus
        self.noodling_id = noodling_id
        self.state = GuideCueState()

        # Computer use controller reference
        self._computer_use: Optional[Any] = None

        # Callbacks
        self._on_cue_received: Optional[Callable[[Dict], None]] = None
        self._on_mode_change: Optional[Callable[[str], None]] = None

        # Subscribe to channels
        self.channel_bus.subscribe(CHANNEL_CUES, self._on_cue)
        self.channel_bus.subscribe(CHANNEL_AMBIANCE, self._on_ambiance)

        logger.debug(f"[{self.noodling_id}] GuideCueHandler initialized")

    def set_computer_use_controller(self, controller):
        """
        Set reference to ComputerUseController for UI actions.

        Args:
            controller: ComputerUseController instance
        """
        self._computer_use = controller
        logger.debug(f"[{self.noodling_id}] Computer use controller set")

    # =========================================================================
    # CUE RECEPTION
    # =========================================================================

    def _on_cue(self, message: ChannelMessage):
        """
        Handle incoming cue from Brenda.

        Filters cues by target_actor, stores state, drifts emotion.
        """
        cue = message.payload

        # Check if this cue is for us
        target = cue.get('target_actor')
        if target and target != self.noodling_id:
            return  # Not for us

        logger.info(f"[{self.noodling_id}] Received cue: {cue.get('beat_name')}")

        # Store cue
        self.state.current_cue = cue
        self.state.current_beat_id = cue.get('beat_id')
        self.state.mode = "active"

        # Drift emotional state toward target
        emotional_target = cue.get('emotional_target')
        if emotional_target:
            self.state.pad.drift_toward(emotional_target)

        # Set up improv zone if present
        improv = cue.get('improv_zone')
        if improv:
            self.state.mode = "improv"
            self.state.improv_topics = improv.get('topics', [])
            self.state.improv_exchanges = 0
            duration = improv.get('duration', {})
            self.state.improv_max_exchanges = duration.get('max_exchanges', 5)

        # Callback
        if self._on_cue_received:
            self._on_cue_received(cue)

    def _on_ambiance(self, message: ChannelMessage):
        """
        React to world ambiance changes.

        Ambiance subtly influences emotional state.
        """
        ambiance = message.payload
        mood = ambiance.get('mood', 'calm')
        energy = ambiance.get('energy', 0.5)

        # Ambiance subtly affects our state
        if mood == 'tense':
            self.state.pad.arousal = min(1.0, self.state.pad.arousal + 0.1)
        elif mood == 'calm':
            self.state.pad.arousal = max(0.0, self.state.pad.arousal - 0.05)
        elif mood == 'joyful':
            self.state.pad.pleasure = min(1.0, self.state.pad.pleasure + 0.1)
        elif mood == 'somber':
            self.state.pad.pleasure = max(-1.0, self.state.pad.pleasure - 0.1)

        logger.debug(f"[{self.noodling_id}] Ambiance update: {mood}, PAD now {self.state.pad.describe()}")

    # =========================================================================
    # RESPONSE GENERATION
    # =========================================================================

    def get_prompt_context(self) -> Dict[str, Any]:
        """
        Get context to inject into Guide's LLM prompt.

        Returns:
            Dict with direction, motivation, suggested dialogue, etc.
        """
        cue = self.state.current_cue

        if not cue:
            # No active cue - passive mode
            return {
                'has_direction': False,
                'mode': 'passive',
                'emotional_state': self.state.pad.describe(),
                'pad': self.state.pad.to_dict(),
            }

        context = {
            'has_direction': True,
            'mode': self.state.mode,
            'beat_id': cue.get('beat_id'),
            'beat_name': cue.get('beat_name'),
            'direction': cue.get('direction'),
            'motivation': cue.get('motivation'),
            'suggested_dialogue': cue.get('your_action', {}).get('speaks'),
            'blocking': cue.get('your_action', {}).get('blocking'),
            'reaction': cue.get('your_action', {}).get('reaction'),
            'emotional_state': self.state.pad.describe(),
            'emotional_target': cue.get('emotional_target'),
            'pad': self.state.pad.to_dict(),
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

        Returns:
            Multi-line string to append to system prompt
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
            topics = ', '.join(ctx.get('improv_topics', []))
            lines.append(f"IMPROV ZONE: Feel free to explore these topics naturally: {topics}")
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

        Returns:
            True if actions were executed
        """
        if not self._computer_use:
            logger.debug(f"[{self.noodling_id}] No computer use controller available")
            return False

        cue = self.state.current_cue
        if not cue:
            return False

        actions = cue.get('your_action', {}).get('computer_use', [])
        if not actions:
            return False

        logger.info(f"[{self.noodling_id}] Executing {len(actions)} computer use actions")

        for action in actions:
            await self._execute_action(action)

        return True

    async def _execute_action(self, action: Dict):
        """
        Execute a single computer use action.

        Args:
            action: Action dict with type, target, etc.
        """
        action_type = action.get('action')

        # Handle pause before
        pause_before = action.get('pause_before', 0)
        if pause_before:
            await asyncio.sleep(pause_before / 1000)

        # Execute action
        if action_type == 'move':
            target = action.get('target')
            if target:
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
        """
        Resolve a target name to (x, y) coordinates.

        Args:
            target: Target identifier (e.g., "Tab: File")

        Returns:
            (x, y) tuple or None if not found
        """
        if not self._computer_use:
            return None

        # Get UI element map from controller
        try:
            elements = self._computer_use.get_ui_element_map()
        except Exception as e:
            logger.warning(f"[{self.noodling_id}] Failed to get UI elements: {e}")
            return None

        # Search for matching element
        target_lower = target.lower()
        for elem in elements:
            elem_name = elem.get('name', '').lower()
            if target_lower in elem_name:
                return (elem['x'], elem['y'])

        logger.warning(f"[{self.noodling_id}] Could not resolve target: {target}")
        return None

    # =========================================================================
    # FEEDBACK REPORTING
    # =========================================================================

    def report_response(self, response_text: str, user_message: str):
        """
        Report back to Brenda after generating a response.

        Call this after Guide has responded to user. Updates internal state
        and publishes feedback to #directors.feedback.

        Args:
            response_text: Guide's response to the user
            user_message: The user's message that prompted this response
        """
        self.state.last_response = response_text
        self.state.last_user_message = user_message
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
            CHANNEL_FEEDBACK,
            ChannelMessage(
                channel=CHANNEL_FEEDBACK,
                from_noodling=self.noodling_id,
                timestamp=time.time(),
                payload=feedback
            )
        )

        logger.info(f"[{self.noodling_id}] Reported feedback: {status}")

        # Clear cue if completed
        if status == "completed":
            old_mode = self.state.mode
            self.state.current_cue = None
            self.state.mode = "passive"
            if self._on_mode_change and old_mode != "passive":
                self._on_mode_change("passive")

    def _adjust_emotion_from_interaction(self, user_message: str):
        """
        Adjust emotional state based on user interaction.

        Args:
            user_message: The user's message
        """
        msg_lower = user_message.lower()

        # Positive signals
        if any(w in msg_lower for w in ['thanks', 'great', 'awesome', 'cool', 'love', 'amazing']):
            self.state.pad.pleasure = min(1.0, self.state.pad.pleasure + 0.1)
            self.state.pad.dominance = min(1.0, self.state.pad.dominance + 0.05)

        # Negative signals
        if any(w in msg_lower for w in ['frustrated', 'annoying', 'hate', 'terrible']):
            self.state.pad.pleasure = max(-1.0, self.state.pad.pleasure - 0.15)
            self.state.pad.dominance = max(0.0, self.state.pad.dominance - 0.1)

        # Confusion signals
        if any(w in msg_lower for w in ['confused', "don't understand", 'what?', 'huh', "i don't get"]):
            self.state.pad.dominance = max(0.0, self.state.pad.dominance - 0.1)
            self.state.pad.arousal = min(1.0, self.state.pad.arousal + 0.05)  # Slight concern

        # Engagement signals
        if '?' in user_message:
            self.state.pad.arousal = min(1.0, self.state.pad.arousal + 0.05)  # Questions are engaging

    def _generate_notes(self, user_message: str) -> str:
        """
        Generate notes about the interaction for Brenda.

        Args:
            user_message: The user's message

        Returns:
            Semi-colon separated notes string
        """
        notes = []

        if '?' in user_message:
            notes.append("User asked a question")

        if len(user_message) > 100:
            notes.append("User gave detailed response")

        if len(user_message) < 10:
            notes.append("User gave brief response")

        msg_lower = user_message.lower()

        if any(w in msg_lower for w in ['confused', "don't understand", "i don't get"]):
            notes.append("User may be confused")

        if any(w in msg_lower for w in ['thanks', 'great', 'awesome']):
            notes.append("Positive feedback")

        if any(w in msg_lower for w in ['frustrated', 'annoying']):
            notes.append("User may be frustrated")

        return "; ".join(notes) if notes else "Normal exchange"

    # =========================================================================
    # MODE MANAGEMENT
    # =========================================================================

    def enter_passive_mode(self):
        """Switch to passive mode (available but not directed)."""
        old_mode = self.state.mode
        self.state.mode = "passive"
        self.state.current_cue = None
        logger.info(f"[{self.noodling_id}] Entering passive mode")

        if self._on_mode_change and old_mode != "passive":
            self._on_mode_change("passive")

    def is_expecting_cue(self) -> bool:
        """
        Check if Guide is waiting for direction.

        Returns:
            True if in passive mode with no active cue
        """
        return self.state.mode == "passive" and self.state.current_cue is None

    def has_active_direction(self) -> bool:
        """
        Check if Guide has active direction from Brenda.

        Returns:
            True if currently directed
        """
        return self.state.current_cue is not None

    def get_mode(self) -> str:
        """
        Get current mode.

        Returns:
            One of: "passive", "active", "improv"
        """
        return self.state.mode

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_cue_received(self, callback: Callable[[Dict], None]):
        """
        Register callback for when a cue is received.

        Args:
            callback: Function taking cue dict
        """
        self._on_cue_received = callback

    def on_mode_change(self, callback: Callable[[str], None]):
        """
        Register callback for mode changes.

        Args:
            callback: Function taking mode string
        """
        self._on_mode_change = callback

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get current state as a dictionary.

        Returns:
            State dict for debugging/display
        """
        return {
            'noodling_id': self.noodling_id,
            'mode': self.state.mode,
            'has_active_cue': self.state.current_cue is not None,
            'current_beat_id': self.state.current_beat_id,
            'emotional_state': self.state.pad.describe(),
            'pad': self.state.pad.to_dict(),
            'improv_exchanges': self.state.improv_exchanges,
            'improv_max_exchanges': self.state.improv_max_exchanges,
            'user_engaged': self.state.user_engaged,
        }


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
