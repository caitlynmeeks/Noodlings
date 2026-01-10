# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   State Transitions - Object Transformation System
#
#   Objects change over time: ice melts, fire burns wood to ash,
#   metal rusts in rain. This module manages those transitions
#   with proper timing. Start burning a log, and over 2 minutes
#   it progresses from "catching fire" to "burning brightly" to
#   "burnt to ash". Agents can observe and react to these
#   gradual changes, creating emergent narrative possibilities.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.state_transitions
# PURPOSE:  Manage timed object state changes
# LAYER:    Backend / Physics
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   StateTransitionType     Enum of transition types
#   StateTransition         Single transition in progress
#   StateTransitionManager  Background manager for all transitions
#
# KEY FUNCTIONS:
#   break_object()          Shatter, crack, or dent an object
#   ignite_object()         Set object on fire
#   freeze_object()         Freeze an object solid
#   melt_object()           Thaw a frozen object
#   rust_object()           Corrode metal over time
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Phase 2: State Transition System

Handles semantic state changes for objects with physics:
- Breaking/shattering
- Burning/ignition/extinguishing
- Freezing/melting
- Rusting/decay
- Phase changes (solid/liquid/gas)

Uses event-driven state machine with affect integration.
"""

import time
import asyncio
from typing import Optional, List, Callable, Dict, Any
from enum import Enum
import logging

from physics_object_descriptor import PhysicsObjectDescriptor, PhysicsEvent

logger = logging.getLogger(__name__)


class StateTransitionType(Enum):
    """Types of state transitions."""
    BREAKING = "breaking"
    BURNING = "burning"
    FREEZING = "freezing"
    MELTING = "melting"
    RUSTING = "rusting"
    DISSOLVING = "dissolving"
    EVAPORATING = "evaporating"
    CONDENSING = "condensing"
    ROTTING = "rotting"
    HEALING = "healing"


class StateTransition:
    """
    Represents a state transition in progress.

    Example: Ice cube melting over 10 minutes
    """

    def __init__(
        self,
        pod: PhysicsObjectDescriptor,
        transition_type: StateTransitionType,
        from_state: str,
        to_state: str,
        duration: float,  # Seconds
        callback: Optional[Callable] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize state transition.

        Args:
            pod: Physics object descriptor
            transition_type: Type of transition
            from_state: Starting state description
            to_state: Ending state description
            duration: How long transition takes (seconds)
            callback: Function to call when complete
            metadata: Additional transition data
        """
        self.pod = pod
        self.transition_type = transition_type
        self.from_state = from_state
        self.to_state = to_state
        self.duration = duration
        self.callback = callback
        self.metadata = metadata or {}

        self.start_time = time.time()
        self.completed = False

    def progress(self) -> float:
        """Get completion percentage (0.0 to 1.0)."""
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.duration)

    def is_complete(self) -> bool:
        """Check if transition is finished."""
        return time.time() >= (self.start_time + self.duration)

    def get_current_state(self) -> str:
        """Get interpolated state description."""
        prog = self.progress()

        if prog < 0.25:
            return f"{self.from_state} (starting to {self.transition_type.value})"
        elif prog < 0.50:
            return f"partially {self.transition_type.value}"
        elif prog < 0.75:
            return f"mostly {self.transition_type.value}"
        elif prog < 1.0:
            return f"almost {self.to_state}"
        else:
            return self.to_state

    async def execute_callback(self):
        """Execute completion callback if present."""
        if self.callback:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback()
            else:
                self.callback()


class StateTransitionManager:
    """
    Manages all active state transitions in the world.

    Runs background task to update transitions and trigger callbacks.
    """

    def __init__(self):
        """Initialize transition manager."""
        self.active_transitions: Dict[str, StateTransition] = {}  # prim_id -> transition
        self.running = False
        self.update_task = None

    def start(self):
        """Start background update loop."""
        if not self.running:
            self.running = True
            self.update_task = asyncio.create_task(self._update_loop())
            logger.info("State transition manager started")

    def stop(self):
        """Stop background update loop."""
        self.running = False
        if self.update_task:
            self.update_task.cancel()
        logger.info("State transition manager stopped")

    def add_transition(self, prim_id: str, transition: StateTransition):
        """
        Add a state transition.

        Args:
            prim_id: Object ID
            transition: StateTransition instance
        """
        self.active_transitions[prim_id] = transition
        logger.info(f"State transition started: {prim_id} → {transition.transition_type.value}")

    def get_transition(self, prim_id: str) -> Optional[StateTransition]:
        """Get active transition for object."""
        return self.active_transitions.get(prim_id)

    def cancel_transition(self, prim_id: str):
        """Cancel active transition."""
        if prim_id in self.active_transitions:
            del self.active_transitions[prim_id]
            logger.info(f"State transition cancelled: {prim_id}")

    async def _update_loop(self):
        """Background loop to update transitions."""
        while self.running:
            try:
                # Check all transitions
                completed = []

                for prim_id, transition in self.active_transitions.items():
                    if transition.is_complete():
                        # Apply final state
                        transition.pod.change_state(transition.to_state)
                        transition.completed = True

                        # Execute callback
                        await transition.execute_callback()

                        completed.append(prim_id)
                        logger.info(f"State transition completed: {prim_id} → {transition.to_state}")
                    else:
                        # Update intermediate state
                        transition.pod.state = transition.get_current_state()

                # Remove completed transitions
                for prim_id in completed:
                    del self.active_transitions[prim_id]

                # Sleep briefly
                await asyncio.sleep(1.0)  # Check every second

            except Exception as e:
                logger.error(f"State transition update error: {e}")
                await asyncio.sleep(5.0)  # Longer sleep on error


# ===== Common State Transitions =====

def break_object(
    pod: PhysicsObjectDescriptor,
    prim_id: str,
    transition_mgr: StateTransitionManager,
    severity: str = "moderate"
) -> StateTransition:
    """
    Break an object (crack, shatter, fragment).

    Args:
        pod: Physics object descriptor
        prim_id: Object ID
        transition_mgr: Transition manager
        severity: "minor", "moderate", "severe"

    Returns:
        StateTransition instance
    """
    # Determine breaking behavior based on material
    material_lower = pod.material.lower()

    if any(x in material_lower for x in ['glass', 'ceramic', 'crystal']):
        # Brittle materials shatter
        from_state = pod.state
        to_state = "shattered into sharp fragments"
        duration = 0.1  # Instant
    elif any(x in material_lower for x in ['metal', 'steel', 'iron']):
        # Ductile materials dent/bend
        from_state = pod.state
        to_state = "dented and bent"
        duration = 0.5
    elif any(x in material_lower for x in ['wood', 'plastic']):
        # Splinters/cracks
        from_state = pod.state
        to_state = "cracked with splinters"
        duration = 0.2
    else:
        # Default breaking
        from_state = pod.state
        to_state = "broken"
        duration = 0.3

    transition = StateTransition(
        pod=pod,
        transition_type=StateTransitionType.BREAKING,
        from_state=from_state,
        to_state=to_state,
        duration=duration,
        metadata={'severity': severity}
    )

    transition_mgr.add_transition(prim_id, transition)
    return transition


def ignite_object(
    pod: PhysicsObjectDescriptor,
    prim_id: str,
    transition_mgr: StateTransitionManager,
    burn_duration: float = 120.0  # 2 minutes
) -> StateTransition:
    """
    Set object on fire.

    Args:
        pod: Physics object descriptor
        prim_id: Object ID
        transition_mgr: Transition manager
        burn_duration: How long it burns (seconds)

    Returns:
        StateTransition instance
    """
    from_state = pod.state
    to_state = "burnt to ash"

    # Check if flammable
    if "non-flammable" in pod.semantic_properties or "fireproof" in pod.semantic_properties:
        logger.warning(f"Object {prim_id} is not flammable")
        to_state = "scorched but intact"
        burn_duration = 5.0

    transition = StateTransition(
        pod=pod,
        transition_type=StateTransitionType.BURNING,
        from_state=from_state,
        to_state=to_state,
        duration=burn_duration,
        metadata={
            'temperature': '800°F',
            'light_emitted': True,
            'smoke': True
        }
    )

    # Add fire properties
    pod.semantic_properties.append("on_fire")
    pod.metadata['temperature'] = '800°F'

    transition_mgr.add_transition(prim_id, transition)
    return transition


def freeze_object(
    pod: PhysicsObjectDescriptor,
    prim_id: str,
    transition_mgr: StateTransitionManager,
    freeze_duration: float = 30.0
) -> StateTransition:
    """
    Freeze an object.

    Args:
        pod: Physics object descriptor
        prim_id: Object ID
        transition_mgr: Transition manager
        freeze_duration: How long freezing takes (seconds)

    Returns:
        StateTransition instance
    """
    from_state = pod.state
    to_state = "frozen solid, covered in frost"

    transition = StateTransition(
        pod=pod,
        transition_type=StateTransitionType.FREEZING,
        from_state=from_state,
        to_state=to_state,
        duration=freeze_duration,
        metadata={'temperature': '0°F'}
    )

    # Add frozen properties
    pod.semantic_properties.append("frozen")
    pod.metadata['temperature'] = '0°F'

    transition_mgr.add_transition(prim_id, transition)
    return transition


def melt_object(
    pod: PhysicsObjectDescriptor,
    prim_id: str,
    transition_mgr: StateTransitionManager,
    melt_duration: float = 60.0
) -> StateTransition:
    """
    Melt a frozen object.

    Args:
        pod: Physics object descriptor
        prim_id: Object ID
        transition_mgr: Transition manager
        melt_duration: How long melting takes (seconds)

    Returns:
        StateTransition instance
    """
    from_state = pod.state
    to_state = "thawed, dripping wet"

    transition = StateTransition(
        pod=pod,
        transition_type=StateTransitionType.MELTING,
        from_state=from_state,
        to_state=to_state,
        duration=melt_duration,
        metadata={'temperature': 'room temperature'}
    )

    # Remove frozen properties
    if "frozen" in pod.semantic_properties:
        pod.semantic_properties.remove("frozen")
    pod.metadata['temperature'] = 'room temperature'

    transition_mgr.add_transition(prim_id, transition)
    return transition


def rust_object(
    pod: PhysicsObjectDescriptor,
    prim_id: str,
    transition_mgr: StateTransitionManager,
    rust_duration: float = 3600.0  # 1 hour
) -> StateTransition:
    """
    Rust a metal object.

    Args:
        pod: Physics object descriptor
        prim_id: Object ID
        transition_mgr: Transition manager
        rust_duration: How long rusting takes (seconds)

    Returns:
        StateTransition instance
    """
    # Check if metal
    material_lower = pod.material.lower()
    if not any(x in material_lower for x in ['metal', 'iron', 'steel']):
        logger.warning(f"Object {prim_id} is not rustable ({pod.material})")
        return None

    from_state = pod.state
    to_state = "heavily rusted, structurally weakened"

    transition = StateTransition(
        pod=pod,
        transition_type=StateTransitionType.RUSTING,
        from_state=from_state,
        to_state=to_state,
        duration=rust_duration,
        metadata={'color': 'reddish-brown'}
    )

    pod.semantic_properties.append("rusting")

    transition_mgr.add_transition(prim_id, transition)
    return transition


# ===== Conditional State Transitions =====

def apply_environmental_effects(
    pod: PhysicsObjectDescriptor,
    prim_id: str,
    transition_mgr: StateTransitionManager,
    environment: Dict[str, Any]
):
    """
    Apply environmental effects to object.

    Args:
        pod: Physics object descriptor
        prim_id: Object ID
        transition_mgr: Transition manager
        environment: Environment conditions
            - temperature: "hot", "cold", "freezing"
            - weather: "rain", "snow", "dry"
            - time: seconds elapsed
    """
    temp = environment.get('temperature', 'normal')
    weather = environment.get('weather', 'dry')

    # Freezing conditions
    if temp == 'freezing' and "frozen" not in pod.semantic_properties:
        freeze_object(pod, prim_id, transition_mgr, freeze_duration=60.0)

    # Melting conditions
    if temp == 'hot' and "frozen" in pod.semantic_properties:
        melt_object(pod, prim_id, transition_mgr, melt_duration=30.0)

    # Rusting from rain
    if weather == 'rain' and 'metal' in pod.material.lower():
        if "rusted" not in pod.semantic_properties:
            rust_object(pod, prim_id, transition_mgr, rust_duration=1800.0)


# ===== Example Usage =====

if __name__ == '__main__':
    # Test state transitions
    import asyncio

    async def test_transitions():
        # Create transition manager
        mgr = StateTransitionManager()
        mgr.start()

        # Create test POD
        glass_pod = PhysicsObjectDescriptor(
            mass="light",
            material="glass",
            state="pristine"
        )

        # Break it
        print("Breaking glass...")
        transition = break_object(glass_pod, "obj_glass", mgr, severity="severe")

        # Wait for completion
        while not transition.is_complete():
            print(f"State: {glass_pod.state} ({transition.progress()*100:.0f}%)")
            await asyncio.sleep(0.5)

        print(f"Final state: {glass_pod.state}")

        mgr.stop()

    asyncio.run(test_transitions())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
