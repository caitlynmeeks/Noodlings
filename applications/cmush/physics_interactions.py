"""
Phase 3: Physics Interaction System

Implements semantic physics verbs:
- Strike (hit, slam, bash)
- Throw (toss, hurl, lob)
- Drop (release, let fall)
- Pick up (grab, take)
- Give (hand to, transfer)
- Push (shove, nudge)
- Pull (tug, drag)

Uses POD properties to determine outcomes semantically.

Author: Commander Spock + Lieutenant Caitlyn
Date: November 22, 2025
"""

from typing import Optional, Dict, Any, Tuple
from enum import Enum
import random
import logging

from physics_object_descriptor import PhysicsObjectDescriptor
from state_transitions import (
    StateTransitionManager,
    break_object,
    StateTransitionType
)

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of physical interactions."""
    STRIKE = "strike"
    THROW = "throw"
    DROP = "drop"
    PICKUP = "pickup"
    GIVE = "give"
    PUSH = "push"
    PULL = "pull"


class InteractionOutcome:
    """
    Result of a physics interaction.

    Contains:
    - Narrative description
    - Sound effects
    - State changes
    - Secondary effects (breaking, etc.)
    """

    def __init__(
        self,
        description: str,
        sound: Optional[str] = None,
        visual: Optional[str] = None,
        actor_state_change: Optional[str] = None,
        target_state_change: Optional[str] = None,
        secondary_effects: Optional[list] = None
    ):
        """
        Initialize interaction outcome.

        Args:
            description: Narrative description of what happened
            sound: Sound effect description
            visual: Visual effect description
            actor_state_change: How actor is affected
            target_state_change: How target is affected
            secondary_effects: List of additional effects
        """
        self.description = description
        self.sound = sound
        self.visual = visual
        self.actor_state_change = actor_state_change
        self.target_state_change = target_state_change
        self.secondary_effects = secondary_effects or []


class PhysicsInteractionEngine:
    """
    Resolves physics interactions between objects semantically.

    No numerical simulation - uses POD properties to determine
    narratively coherent outcomes.
    """

    def __init__(self, transition_mgr: Optional[StateTransitionManager] = None):
        """
        Initialize interaction engine.

        Args:
            transition_mgr: State transition manager for secondary effects
        """
        self.transition_mgr = transition_mgr

    def strike(
        self,
        actor_pod: PhysicsObjectDescriptor,
        target_pod: PhysicsObjectDescriptor,
        actor_id: str,
        target_id: str,
        force: str = "medium"
    ) -> InteractionOutcome:
        """
        Resolve strike interaction.

        Args:
            actor_pod: Striking object physics
            target_pod: Target object physics
            actor_id: Actor prim ID
            target_id: Target prim ID
            force: "light", "medium", "heavy"

        Returns:
            InteractionOutcome
        """
        # Determine impact sound based on materials
        sound = self._determine_impact_sound(actor_pod.material, target_pod.material)

        # Determine target reaction based on mass comparison
        reaction = self._determine_strike_reaction(actor_pod, target_pod, force)

        # Check if target breaks
        breaks = self._should_object_break(target_pod, force)

        # Build description
        desc = f"The {actor_pod.material} strikes the {target_pod.material} "
        desc += f"with a {sound}. "
        desc += f"The target {reaction}."

        outcome = InteractionOutcome(
            description=desc,
            sound=sound,
            visual=f"{actor_id} → {target_id} impact",
            target_state_change=reaction
        )

        # Apply breaking if necessary
        if breaks and self.transition_mgr:
            break_object(target_pod, target_id, self.transition_mgr, severity=force)
            outcome.secondary_effects.append("target_breaks")

        return outcome

    def throw(
        self,
        actor_id: str,
        projectile_pod: PhysicsObjectDescriptor,
        projectile_id: str,
        target_pod: Optional[PhysicsObjectDescriptor] = None,
        target_id: Optional[str] = None,
        force: str = "medium"
    ) -> InteractionOutcome:
        """
        Resolve throw interaction.

        Args:
            actor_id: Thrower
            projectile_pod: Thrown object physics
            projectile_id: Projectile prim ID
            target_pod: Optional target physics
            target_id: Optional target prim ID
            force: "light", "medium", "heavy"

        Returns:
            InteractionOutcome
        """
        # Determine throw trajectory
        trajectory = self._determine_throw_trajectory(projectile_pod, force)

        if target_pod and target_id:
            # Throw at target (like strike)
            return self.strike(projectile_pod, target_pod, projectile_id, target_id, force)
        else:
            # Throw into space
            desc = f"{actor_id} throws {projectile_id} {force}ly. "
            desc += f"It {trajectory}."

            # Determine landing
            landing_sound = self._determine_landing_sound(projectile_pod)

            return InteractionOutcome(
                description=desc,
                sound=landing_sound,
                visual=f"arc trajectory from {actor_id}",
                target_state_change=f"landed on ground"
            )

    def drop(
        self,
        actor_id: str,
        object_pod: PhysicsObjectDescriptor,
        object_id: str
    ) -> InteractionOutcome:
        """
        Resolve drop interaction.

        Args:
            actor_id: Dropper
            object_pod: Dropped object physics
            object_id: Object prim ID

        Returns:
            InteractionOutcome
        """
        # Determine fall behavior
        fall_desc = self._determine_fall_behavior(object_pod)

        # Determine landing sound
        landing_sound = self._determine_landing_sound(object_pod)

        # Check if breaks on landing
        breaks = self._should_object_break_on_drop(object_pod)

        desc = f"{actor_id} drops {object_id}. "
        desc += f"It {fall_desc} "
        desc += f"and lands with a {landing_sound}."

        outcome = InteractionOutcome(
            description=desc,
            sound=landing_sound,
            visual="downward motion",
            target_state_change="on ground"
        )

        # Apply breaking if necessary
        if breaks and self.transition_mgr:
            break_object(object_pod, object_id, self.transition_mgr, severity="moderate")
            outcome.secondary_effects.append("breaks_on_landing")

        return outcome

    def pickup(
        self,
        actor_id: str,
        object_pod: PhysicsObjectDescriptor,
        object_id: str
    ) -> InteractionOutcome:
        """
        Resolve pickup interaction.

        Args:
            actor_id: Picker
            object_pod: Object physics
            object_id: Object prim ID

        Returns:
            InteractionOutcome
        """
        # Check if too heavy
        too_heavy = self._is_too_heavy(object_pod)

        if too_heavy:
            return InteractionOutcome(
                description=f"{actor_id} tries to lift {object_id} but it's too heavy!",
                sound="grunt of effort",
                actor_state_change="strained"
            )

        # Check if too hot
        if "hot" in object_pod.semantic_properties or object_pod.metadata.get('temperature', '').startswith('hot'):
            return InteractionOutcome(
                description=f"{actor_id} reaches for {object_id} but recoils from the heat!",
                sound="sizzle",
                actor_state_change="burned hand"
            )

        # Normal pickup
        return InteractionOutcome(
            description=f"{actor_id} picks up {object_id}.",
            sound="rustle" if object_pod.mass == "light" else "hefty lift",
            target_state_change="being held"
        )

    def give(
        self,
        giver_id: str,
        receiver_id: str,
        object_pod: PhysicsObjectDescriptor,
        object_id: str
    ) -> InteractionOutcome:
        """
        Resolve give interaction.

        Args:
            giver_id: Giver
            receiver_id: Receiver
            object_pod: Object physics
            object_id: Object prim ID

        Returns:
            InteractionOutcome
        """
        return InteractionOutcome(
            description=f"{giver_id} hands {object_id} to {receiver_id}. {receiver_id} takes it.",
            sound="exchange",
            target_state_change="held by receiver"
        )

    def push(
        self,
        actor_id: str,
        target_pod: PhysicsObjectDescriptor,
        target_id: str,
        force: str = "medium"
    ) -> InteractionOutcome:
        """
        Resolve push interaction.

        Args:
            actor_id: Pusher
            target_pod: Target physics
            target_id: Target prim ID
            force: "light", "medium", "heavy"

        Returns:
            InteractionOutcome
        """
        # Determine movement based on mass and force
        movement = self._determine_push_movement(target_pod, force)

        return InteractionOutcome(
            description=f"{actor_id} pushes {target_id} {force}ly. It {movement}.",
            sound="scrape" if "rough" in target_pod.friction else "slide",
            visual="lateral motion",
            target_state_change=f"moved by push"
        )

    # ===== Internal Resolution Methods =====

    def _determine_impact_sound(self, material1: str, material2: str) -> str:
        """Determine sound from material collision."""
        m1_lower = material1.lower()
        m2_lower = material2.lower()

        # Metal on metal
        if 'metal' in m1_lower and 'metal' in m2_lower:
            return random.choice(['CLANG', 'CLANK', 'DING'])

        # Glass/ceramic
        if any(x in m1_lower + m2_lower for x in ['glass', 'ceramic']):
            return random.choice(['CLINK', 'tinkle', 'shatter'])

        # Wood
        if 'wood' in m1_lower or 'wood' in m2_lower:
            return random.choice(['THUNK', 'CRACK', 'thud'])

        # Stone/rock
        if any(x in m1_lower + m2_lower for x in ['stone', 'rock']):
            return random.choice(['CRACK', 'thud', 'crunch'])

        # Default
        return random.choice(['thud', 'impact', 'collision'])

    def _determine_strike_reaction(
        self,
        actor_pod: PhysicsObjectDescriptor,
        target_pod: PhysicsObjectDescriptor,
        force: str
    ) -> str:
        """Determine how target reacts to strike."""
        # Compare masses semantically
        actor_mass_heavy = any(x in actor_pod.mass.lower() for x in ['heavy', 'massive'])
        target_mass_light = any(x in target_pod.mass.lower() for x in ['light', 'negligible'])

        if actor_mass_heavy and target_mass_light:
            return random.choice(['flies across the room', 'tumbles end over end', 'goes flying'])
        elif target_mass_light:
            return random.choice(['tumbles', 'rolls away', 'bounces'])
        else:
            return random.choice(['wobbles', 'shifts slightly', 'barely moves'])

    def _should_object_break(self, pod: PhysicsObjectDescriptor, force: str) -> bool:
        """Determine if object breaks from strike."""
        # Brittle materials break easily
        if any(x in pod.material.lower() for x in ['glass', 'ceramic', 'brittle']):
            return force in ['medium', 'heavy']

        # Fragile property
        if "fragile" in pod.semantic_properties:
            return force in ['medium', 'heavy']

        # Already damaged
        if "cracked" in pod.state.lower() or "damaged" in pod.state.lower():
            return force in ['medium', 'heavy']

        # Heavy force breaks most things
        return force == 'heavy' and pod.softness != "hard"

    def _should_object_break_on_drop(self, pod: PhysicsObjectDescriptor) -> bool:
        """Determine if object breaks when dropped."""
        # Brittle materials break on drop
        if any(x in pod.material.lower() for x in ['glass', 'ceramic']):
            return True

        # Fragile property
        if "fragile" in pod.semantic_properties:
            return True

        return False

    def _determine_throw_trajectory(self, pod: PhysicsObjectDescriptor, force: str) -> str:
        """Determine throw trajectory description."""
        mass_light = any(x in pod.mass.lower() for x in ['light', 'negligible'])

        if mass_light:
            if force == 'heavy':
                return "sails through the air in a high arc"
            elif force == 'medium':
                return "flies in a graceful arc"
            else:
                return "drifts gently forward"
        else:
            if force == 'heavy':
                return "hurtles forward in a fast arc"
            elif force == 'medium':
                return "arcs through the air"
            else:
                return "drops quickly after a short flight"

    def _determine_landing_sound(self, pod: PhysicsObjectDescriptor) -> str:
        """Determine sound when object lands."""
        material_lower = pod.material.lower()

        if 'glass' in material_lower or 'ceramic' in material_lower:
            return random.choice(['CRASH', 'SHATTER', 'tinkle'])
        elif 'metal' in material_lower:
            return random.choice(['CLANG', 'CLATTER', 'clank'])
        elif 'wood' in material_lower:
            return random.choice(['thunk', 'clatter', 'thud'])
        else:
            return 'thud'

    def _determine_fall_behavior(self, pod: PhysicsObjectDescriptor) -> str:
        """Determine how object falls."""
        mass_light = any(x in pod.mass.lower() for x in ['light', 'negligible'])

        if mass_light:
            return "drifts downward gently"
        else:
            return "drops quickly"

    def _is_too_heavy(self, pod: PhysicsObjectDescriptor) -> bool:
        """Check if object is too heavy to pick up."""
        mass_lower = pod.mass.lower()
        return any(x in mass_lower for x in ['very heavy', 'massive', 'immense'])

    def _determine_push_movement(self, pod: PhysicsObjectDescriptor, force: str) -> str:
        """Determine movement from push."""
        mass_light = any(x in pod.mass.lower() for x in ['light', 'negligible'])
        friction_high = any(x in pod.friction.lower() for x in ['high', 'sticky', 'rough'])

        if mass_light and not friction_high:
            if force == 'heavy':
                return "slides rapidly across the floor"
            elif force == 'medium':
                return "slides a few feet"
            else:
                return "nudges forward slightly"
        elif friction_high:
            return "barely budges" if force == 'light' else "scrapes forward with effort"
        else:
            if force == 'heavy':
                return "slides steadily forward"
            else:
                return "shifts position"


# ===== Example Usage =====

if __name__ == '__main__':
    # Test interactions
    from state_transitions import StateTransitionManager

    # Setup
    transition_mgr = StateTransitionManager()
    engine = PhysicsInteractionEngine(transition_mgr)

    # Create test objects
    rock_pod = PhysicsObjectDescriptor(
        mass="medium",
        material="granite",
        softness="very hard"
    )

    glass_pod = PhysicsObjectDescriptor(
        mass="very light",
        material="glass",
        softness="brittle",
        semantic_properties=["fragile", "transparent"]
    )

    # Test strike
    print("=== STRIKE TEST ===")
    outcome = engine.strike(rock_pod, glass_pod, "rock_001", "glass_001", force="medium")
    print(f"Description: {outcome.description}")
    print(f"Sound: {outcome.sound}")
    print(f"Secondary effects: {outcome.secondary_effects}")
    print()

    # Test throw
    print("=== THROW TEST ===")
    outcome = engine.throw("user_caity", rock_pod, "rock_001", force="heavy")
    print(f"Description: {outcome.description}")
    print(f"Sound: {outcome.sound}")
    print()

    # Test drop
    print("=== DROP TEST ===")
    outcome = engine.drop("user_caity", glass_pod, "glass_001")
    print(f"Description: {outcome.description}")
    print(f"Sound: {outcome.sound}")
    print(f"Secondary effects: {outcome.secondary_effects}")
