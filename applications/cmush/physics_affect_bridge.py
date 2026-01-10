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
#   Physics-Affect Integration Bridge
#
#   When something happens in the physical world (a glass breaks,
#   something burns), this module broadcasts the event to nearby
#   AI agents. Each agent "perceives" the event - extracting an
#   emotional response, updating their internal state, calculating
#   how surprised they are, and possibly reacting aloud if it was
#   startling enough. Connects physics simulation to AI feelings.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.physics_affect_bridge
# PURPOSE:  Connect physics events to agent phenomenal states
# LAYER:    Backend / Semantic Physics
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PhysicsAffectEvent       Physics event with affect implications
#   PhysicsAffectExtractor   Extract emotional meaning from events
#   PhysicsAffectBroadcaster Notify agents of physics events
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Phase 4: Physics to Affect Integration Bridge

Connects semantic physics events to Noodling consciousness system.
Physics events trigger affective responses in nearby agents.

Pipeline:
1. Physics event occurs (strike, break, burn, etc.)
2. Broadcast to room
3. Each agent perceives event
4. Affect extraction (LLM-powered)
5. Phenomenal state update
6. Surprise calculation
7. Memory formation
8. Behavioral response (if surprising)

Author: Caitlyn + Claude
Date: November 22, 2025
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from physics_interactions import InteractionOutcome, InteractionType
from physics_object_descriptor import PhysicsEvent

logger = logging.getLogger(__name__)


@dataclass
class PhysicsAffectEvent:
    """
    Physics event with affect implications for Noodlings.

    Contains both physical description and affective interpretation.
    """
    # Physical event data
    description: str
    interaction_type: str
    objects_involved: List[str]
    room_id: str

    # Affect extraction (5-D vector)
    valence: float  # -1.0 to 1.0 (negative to positive)
    arousal: float  # 0.0 to 1.0 (calm to excited)
    fear: float     # 0.0 to 1.0 (safe to afraid)
    sorrow: float   # 0.0 to 1.0 (content to sad)
    boredom: float  # 0.0 to 1.0 (engaged to bored)

    # Surprise estimate
    surprise: float  # 0.0 to 1.0 (expected to shocking)

    # Additional context
    sound: Optional[str] = None
    visual: Optional[str] = None
    metadata: Optional[Dict] = None

    def get_affect_vector(self) -> List[float]:
        """Get affect as 5-D vector."""
        return [self.valence, self.arousal, self.fear, self.sorrow, self.boredom]


class PhysicsAffectExtractor:
    """
    Extracts affective meaning from physics events.

    Uses rule-based heuristics (fast) or LLM reasoning (accurate).
    """

    def __init__(self, use_llm: bool = False):
        """
        Initialize affect extractor.

        Args:
            use_llm: Use LLM for extraction (slower but more accurate)
        """
        self.use_llm = use_llm

    def extract_affect(
        self,
        outcome: InteractionOutcome,
        interaction_type: InteractionType
    ) -> PhysicsAffectEvent:
        """
        Extract affect from interaction outcome.

        Args:
            outcome: Interaction outcome
            interaction_type: Type of interaction

        Returns:
            PhysicsAffectEvent with affect vector
        """
        if self.use_llm:
            return self._extract_affect_llm(outcome, interaction_type)
        else:
            return self._extract_affect_heuristic(outcome, interaction_type)

    def _extract_affect_heuristic(
        self,
        outcome: InteractionOutcome,
        interaction_type: InteractionType
    ) -> PhysicsAffectEvent:
        """Extract affect using rule-based heuristics (fast)."""

        # Default affect
        valence = 0.0
        arousal = 0.3
        fear = 0.0
        sorrow = 0.0
        boredom = 0.0
        surprise = 0.2

        # Adjust based on interaction type
        if interaction_type == InteractionType.STRIKE:
            arousal = 0.6
            surprise = 0.4
            if "breaks" in outcome.secondary_effects:
                valence = -0.3
                surprise = 0.7
                sorrow = 0.2

        elif interaction_type == InteractionType.THROW:
            arousal = 0.5
            surprise = 0.3

        elif interaction_type == InteractionType.DROP:
            arousal = 0.2
            if "breaks" in outcome.secondary_effects:
                valence = -0.4
                surprise = 0.6
                sorrow = 0.3

        # Adjust based on sounds
        if outcome.sound:
            sound_lower = outcome.sound.lower()
            if any(x in sound_lower for x in ['crash', 'shatter', 'clang', 'bang']):
                arousal += 0.3
                surprise += 0.2
            if 'shatter' in sound_lower:
                valence -= 0.2
                sorrow += 0.2

        # Clamp values
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        fear = max(0.0, min(1.0, fear))
        sorrow = max(0.0, min(1.0, sorrow))
        boredom = max(0.0, min(1.0, boredom))
        surprise = max(0.0, min(1.0, surprise))

        return PhysicsAffectEvent(
            description=outcome.description,
            interaction_type=interaction_type.value,
            objects_involved=[],  # TODO: extract from outcome
            room_id="",  # TODO: get from context
            valence=valence,
            arousal=arousal,
            fear=fear,
            sorrow=sorrow,
            boredom=boredom,
            surprise=surprise,
            sound=outcome.sound,
            visual=outcome.visual,
            metadata={}
        )

    def _extract_affect_llm(
        self,
        outcome: InteractionOutcome,
        interaction_type: InteractionType
    ) -> PhysicsAffectEvent:
        """Extract affect using LLM reasoning (accurate but slower)."""
        # TODO: Implement LLM-based affect extraction
        # For now, fall back to heuristic
        return self._extract_affect_heuristic(outcome, interaction_type)


class PhysicsAffectBroadcaster:
    """
    Broadcasts physics events to agents in room.

    Integrates with agent_manager to trigger perception events.
    """

    def __init__(self, world, agent_manager):
        """
        Initialize broadcaster.

        Args:
            world: World instance
            agent_manager: AgentManager instance
        """
        self.world = world
        self.agent_manager = agent_manager
        self.extractor = PhysicsAffectExtractor(use_llm=False)

    async def broadcast_physics_event(
        self,
        room_id: str,
        outcome: InteractionOutcome,
        interaction_type: InteractionType
    ):
        """
        Broadcast physics event to all agents in room.

        Args:
            room_id: Room where event occurred
            outcome: Interaction outcome
            interaction_type: Type of interaction
        """
        # Extract affect
        affect_event = self.extractor.extract_affect(outcome, interaction_type)
        affect_event.room_id = room_id

        # Get all agents in room
        agent_ids = self.world.list_agents_in_room(room_id)

        logger.info(
            f"[PHYSICS BROADCAST] {room_id}: {outcome.description} "
            f"→ {len(agent_ids)} agents"
        )

        # Broadcast to each agent
        tasks = []
        for agent_id in agent_ids:
            task = self._notify_agent(agent_id, affect_event)
            tasks.append(task)

        # Run concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _notify_agent(self, agent_id: str, event: PhysicsAffectEvent):
        """
        Notify individual agent of physics event.

        Args:
            agent_id: Agent to notify
            event: Physics affect event
        """
        agent = self.agent_manager.agents.get(agent_id)
        if not agent:
            return

        try:
            # Trigger perception event
            await agent.perceive_event('physics', {
                'description': event.description,
                'affect': event.get_affect_vector(),
                'surprise': event.surprise,
                'sound': event.sound,
                'visual': event.visual,
                'interaction_type': event.interaction_type,
                'metadata': event.metadata or {}
            })

            logger.debug(f"[PHYSICS → {agent_id}] {event.description}")

        except Exception as e:
            logger.error(f"Error notifying agent {agent_id}: {e}")


# ===== Integration with agent_bridge.py =====

async def handle_physics_event_perception(
    agent_bridge,
    event_data: Dict[str, Any]
):
    """
    Handle physics event perception in agent_bridge.

    This function should be called from agent_bridge.py's perceive_event()
    when event_type == 'physics'.

    Args:
        agent_bridge: AgentBridge instance
        event_data: Physics event data from broadcaster
    """
    # Extract affect vector
    affect_vector = event_data['affect']

    # Update phenomenal state
    # (agent_bridge has noodling model with update_state() method)
    agent_bridge.noodling.update_state(affect_vector)

    # Calculate surprise
    surprise = agent_bridge.noodling.calculate_surprise()

    # Form episodic memory
    agent_bridge.conversation_context.append({
        'user': 'world_physics',
        'text': event_data['description'],
        'affect': affect_vector,
        'surprise': surprise,
        'event_type': 'physics',
        'event_metadata': event_data.get('metadata', {}),
        'sound': event_data.get('sound'),
        'visual': event_data.get('visual')
    })

    logger.info(
        f"[{agent_bridge.agent_id}] Physics perception: "
        f"surprise={surprise:.2f}, affect={affect_vector}"
    )

    # React if surprising
    if surprise > agent_bridge.surprise_threshold:
        # Generate response (speech or thought)
        response_type = 'speech' if surprise > 0.5 else 'thought'

        await agent_bridge.generate_response(
            context=f"Physics event: {event_data['description']}",
            response_type=response_type
        )


# ===== Example Usage =====

if __name__ == '__main__':
    # Test affect extraction
    from physics_interactions import PhysicsInteractionEngine, InteractionType
    from physics_object_descriptor import PhysicsObjectDescriptor

    # Setup
    engine = PhysicsInteractionEngine()
    extractor = PhysicsAffectExtractor()

    # Create test objects
    rock_pod = PhysicsObjectDescriptor(
        mass="medium",
        material="granite"
    )

    glass_pod = PhysicsObjectDescriptor(
        mass="very light",
        material="glass",
        semantic_properties=["fragile"]
    )

    # Simulate strike
    outcome = engine.strike(rock_pod, glass_pod, "rock_001", "glass_001", force="heavy")

    # Extract affect
    affect_event = extractor.extract_affect(outcome, InteractionType.STRIKE)

    print("=== PHYSICS AFFECT EXTRACTION ===")
    print(f"Description: {affect_event.description}")
    print(f"Sound: {affect_event.sound}")
    print(f"Affect vector: {affect_event.get_affect_vector()}")
    print(f"  Valence: {affect_event.valence:.2f}")
    print(f"  Arousal: {affect_event.arousal:.2f}")
    print(f"  Fear: {affect_event.fear:.2f}")
    print(f"  Sorrow: {affect_event.sorrow:.2f}")
    print(f"  Boredom: {affect_event.boredom:.2f}")
    print(f"Surprise: {affect_event.surprise:.2f}")
    print(f"Secondary effects: {outcome.secondary_effects}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
