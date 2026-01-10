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
#   Physics → Affect Bridge
#
#   Connects the Gaussian collision system to the CharmNetwor...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.physics_affect_bridge
# PURPOSE:  Physics → Affect Bridge
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PhysicsAffectConfig, PhysicsAffectBridge, PhysicsAffectFacet, init_physics_affect_bridge(), get_physics_affect_bridge()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PhysicsAffectConfig:
    """Configuration for physics → affect mapping."""

    # Enable/disable the bridge
    enabled: bool = True

    # Minimum overlap to trigger affect change
    touch_threshold: float = 0.05

    # Maximum affect change per tick (prevents sudden spikes)
    max_valence_delta: float = 0.3
    max_arousal_delta: float = 0.5
    max_dominance_delta: float = 0.2

    # Decay rate for physics-induced affect (how fast it fades)
    impulse_decay_rate: float = 0.5

    # Whether to use relationship modifiers from social memory
    use_relationships: bool = True

    # Spring bone physics → arousal mapping
    spring_energy_arousal_scale: float = 0.1
    spring_energy_threshold: float = 0.1

    # Impact startle parameters
    impact_startle_duration: float = 0.5
    impact_startle_decay: float = 2.0


class PhysicsAffectBridge:
    """
    Bridge between physics system and affect system.

    Listens for:
    - Touch events (from GaussianCollisionDetector)
    - Spring bone energy (from SpringBoneSimulator)
    - Collision impacts

    Emits:
    - Affect state changes to CharmNetworkFacet
    """

    def __init__(self, config: Optional[PhysicsAffectConfig] = None):
        self.config = config or PhysicsAffectConfig()

        # Entity → CharmNetworkFacet mapping
        # We need to know which facet to inject into for each entity
        self._entity_facets: Dict[str, Any] = {}

        # Social relationship provider (optional)
        self._relationship_provider: Optional[Callable[[str, str], float]] = None

        # Pending impulses per entity (accumulated between ticks)
        self._pending_impulses: Dict[str, List[Dict[str, float]]] = {}

        # Active startle states
        self._startle_states: Dict[str, float] = {}  # entity_id → remaining duration

        # Stats
        self.touch_count = 0
        self.impulse_count = 0

    def register_entity_facet(self, entity_id: str, charm_facet: Any):
        """
        Register a CharmNetworkFacet for an entity.

        Args:
            entity_id: Entity identifier
            charm_facet: CharmNetworkFacet instance with inject_state() method
        """
        self._entity_facets[entity_id] = charm_facet
        self._pending_impulses[entity_id] = []
        logger.debug(f"Registered entity facet: {entity_id}")

    def unregister_entity(self, entity_id: str):
        """Unregister an entity."""
        self._entity_facets.pop(entity_id, None)
        self._pending_impulses.pop(entity_id, None)
        self._startle_states.pop(entity_id, None)

    def set_relationship_provider(self, provider: Callable[[str, str], float]):
        """
        Set function to get relationship valence between entities.

        Args:
            provider: Function(entity_a, entity_b) -> float (-1 to 1)
        """
        self._relationship_provider = provider

    def on_touch_event(self, touch_event: 'TouchEvent'):
        """
        Handle a touch event from the collision system.

        Called by PhysicsEventBus when touch is detected.
        """
        if not self.config.enabled:
            return

        self.touch_count += 1

        # Get relationship valence
        relationship = 0.0
        if self.config.use_relationships and self._relationship_provider:
            relationship = self._relationship_provider(
                touch_event.entity_a,
                touch_event.entity_b
            )

        # Generate affect impulse for the touched entity (entity_b is receiver)
        from .semantic_world.gaussian_collision import TouchAffectMapper

        mapper = TouchAffectMapper()
        impulse = mapper.generate_impulse(touch_event, relationship)

        # Queue impulse for entity_b (the one being touched)
        if touch_event.entity_b in self._pending_impulses:
            self._pending_impulses[touch_event.entity_b].append({
                'valence': impulse.valence_delta,
                'arousal': impulse.arousal_delta,
                'dominance': impulse.dominance_delta,
                'source': 'touch',
                'decay': impulse.decay_rate,
            })

            # Handle startle from impact
            if impulse.startle > 0.3:
                self._startle_states[touch_event.entity_b] = self.config.impact_startle_duration

        # Also generate (smaller) impulse for toucher (entity_a)
        if touch_event.entity_a in self._pending_impulses:
            # Toucher gets smaller arousal bump, neutral valence
            self._pending_impulses[touch_event.entity_a].append({
                'valence': 0.0,
                'arousal': impulse.arousal_delta * 0.3,
                'dominance': 0.1,  # Slight dominance boost for initiator
                'source': 'touch_initiated',
                'decay': impulse.decay_rate,
            })

        logger.debug(f"Touch event: {touch_event.description()}")

    def on_spring_energy(self, entity_id: str, kinetic_energy: float):
        """
        Handle spring bone energy update.

        High spring bone movement (e.g., tail wagging, hair bouncing)
        translates to arousal.
        """
        if not self.config.enabled:
            return

        if entity_id not in self._pending_impulses:
            return

        if kinetic_energy > self.config.spring_energy_threshold:
            arousal_delta = min(
                self.config.max_arousal_delta,
                kinetic_energy * self.config.spring_energy_arousal_scale
            )
            self._pending_impulses[entity_id].append({
                'valence': 0.0,
                'arousal': arousal_delta,
                'dominance': 0.0,
                'source': 'spring_physics',
                'decay': 0.8,
            })

    def on_impact(self, entity_id: str, force: float, source: str = "collision"):
        """
        Handle impact/collision event.

        Strong impacts cause startle and negative valence.
        """
        if not self.config.enabled:
            return

        if entity_id not in self._pending_impulses:
            return

        # Normalize force to 0-1 range (assume max force ~100)
        intensity = min(1.0, force / 100.0)

        self._pending_impulses[entity_id].append({
            'valence': -intensity * 0.5,  # Impacts feel bad
            'arousal': intensity * 0.8,   # High arousal
            'dominance': -intensity * 0.3, # Loss of control
            'source': f'impact:{source}',
            'decay': 0.3,  # Fast decay
        })

        # Trigger startle
        if intensity > 0.3:
            self._startle_states[entity_id] = self.config.impact_startle_duration

        self.impulse_count += 1

    async def tick(self, delta_time: float):
        """
        Process pending impulses and update affect states.

        Called each frame/tick to apply accumulated physics influences.
        """
        if not self.config.enabled:
            return

        for entity_id, impulses in self._pending_impulses.items():
            if not impulses:
                continue

            facet = self._entity_facets.get(entity_id)
            if not facet:
                continue

            # Aggregate impulses
            total_valence = 0.0
            total_arousal = 0.0
            total_dominance = 0.0

            for imp in impulses:
                total_valence += imp['valence']
                total_arousal += imp['arousal']
                total_dominance += imp['dominance']

            # Clamp to max deltas
            total_valence = max(-self.config.max_valence_delta,
                               min(self.config.max_valence_delta, total_valence))
            total_arousal = max(0, min(self.config.max_arousal_delta, total_arousal))
            total_dominance = max(-self.config.max_dominance_delta,
                                 min(self.config.max_dominance_delta, total_dominance))

            # Inject into CharmNetworkFacet
            if hasattr(facet, 'inject_state'):
                try:
                    facet.inject_state({
                        'affect_valence_delta': total_valence,
                        'affect_arousal_delta': total_arousal,
                        'affect_dominance_delta': total_dominance,
                        'source': 'physics',
                    })
                    self.impulse_count += 1
                except Exception as e:
                    logger.error(f"Failed to inject affect state: {e}")

            # Clear processed impulses
            impulses.clear()

        # Process startle decay
        entities_to_clear = []
        for entity_id, remaining in self._startle_states.items():
            remaining -= delta_time
            if remaining <= 0:
                entities_to_clear.append(entity_id)
            else:
                self._startle_states[entity_id] = remaining

                # Inject startle arousal
                facet = self._entity_facets.get(entity_id)
                if facet and hasattr(facet, 'inject_state'):
                    startle_intensity = remaining / self.config.impact_startle_duration
                    facet.inject_state({
                        'affect_arousal_delta': startle_intensity * 0.3,
                        'source': 'startle',
                    })

        for entity_id in entities_to_clear:
            del self._startle_states[entity_id]

    def connect_to_event_bus(self, event_bus: 'PhysicsEventBus'):
        """
        Connect to a PhysicsEventBus to receive events.

        Args:
            event_bus: PhysicsEventBus instance
        """
        event_bus.subscribe('touch_start', self.on_touch_event)
        event_bus.subscribe('impulse', self._on_impulse_event)

        logger.info("Connected to physics event bus")

    def _on_impulse_event(self, impulse: 'AffectImpulse'):
        """Handle raw affect impulse event."""
        if not impulse.source_event:
            return

        # Already processed in on_touch_event
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            'enabled': self.config.enabled,
            'registered_entities': list(self._entity_facets.keys()),
            'touch_count': self.touch_count,
            'impulse_count': self.impulse_count,
            'active_startles': list(self._startle_states.keys()),
            'pending_impulse_counts': {
                k: len(v) for k, v in self._pending_impulses.items()
            },
        }


# =============================================================================
# Integration with Facet System
# =============================================================================

class PhysicsAffectFacet:
    """
    Facet that manages physics → affect integration for an entity.

    Add this to a facet assembly to enable physics-driven emotions.
    """

    facet_type = "PHYSICS_AFFECT"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_id = config.get('entity_id', '')

        # Bridge instance (shared)
        self._bridge: Optional[PhysicsAffectBridge] = None

        # Reference to CharmNetworkFacet (set during wiring)
        self._charm_facet = None

        # Stats
        self.execution_count = 0

    def set_bridge(self, bridge: PhysicsAffectBridge):
        """Set the shared physics-affect bridge."""
        self._bridge = bridge

    def set_charm_facet(self, charm_facet):
        """Wire to CharmNetworkFacet for state injection."""
        self._charm_facet = charm_facet

        if self._bridge and self.entity_id:
            self._bridge.register_entity_facet(self.entity_id, charm_facet)

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process physics state and return affect modifiers.

        This facet doesn't produce output directly - it modifies
        CharmNetworkFacet state via injection.
        """
        self.execution_count += 1

        # The actual work happens via the bridge's tick()
        # This facet is mostly for configuration and wiring

        return {
            'physics_affect_active': self._bridge is not None and self._bridge.config.enabled,
            'touch_count': self._bridge.touch_count if self._bridge else 0,
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        return {
            'execution_count': self.execution_count,
            'total_tokens': 0,
            'avg_tokens': 0,
            'total_time': 0,
            'avg_time': 0,
            'last_tokens': 0,
            'last_time': 0,
        }

    def get_token_usage(self) -> Dict[str, Any]:
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.execution_count,
            'avg_tokens': 0,
        }


# =============================================================================
# Global Bridge Instance
# =============================================================================

_bridge: Optional[PhysicsAffectBridge] = None


def init_physics_affect_bridge(config: Optional[PhysicsAffectConfig] = None) -> PhysicsAffectBridge:
    """Initialize the global physics-affect bridge."""
    global _bridge
    _bridge = PhysicsAffectBridge(config)

    # Connect to collision system's event bus
    try:
        from .semantic_world.gaussian_collision import get_physics_event_bus
        event_bus = get_physics_event_bus()
        if event_bus:
            _bridge.connect_to_event_bus(event_bus)
    except ImportError:
        logger.warning("Collision system not available, physics events won't be processed")

    logger.info("Physics-affect bridge initialized")
    return _bridge


def get_physics_affect_bridge() -> Optional[PhysicsAffectBridge]:
    """Get the global physics-affect bridge."""
    return _bridge


async def tick_physics_affect(delta_time: float):
    """Tick the physics-affect bridge."""
    if _bridge:
        await _bridge.tick(delta_time)


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example of setting up physics → affect integration."""

    # 1. Initialize collision system
    from .semantic_world.gaussian_collision import init_collision_system, get_detector
    init_collision_system(overlap_threshold=0.05)

    # 2. Initialize physics-affect bridge
    bridge = init_physics_affect_bridge(PhysicsAffectConfig(
        enabled=True,
        touch_threshold=0.05,
        use_relationships=True,
    ))

    # 3. Register entities with their CharmNetworkFacets
    # (In real code, this happens during agent setup)
    # bridge.register_entity_facet("red", red_charm_facet)
    # bridge.register_entity_facet("yuki", yuki_charm_facet)

    # 4. Add entities to collision detector
    detector = get_detector()
    # detector.add_entity("red", red_radiance_asset)
    # detector.add_entity("yuki", yuki_radiance_asset)

    # 5. In game loop:
    delta_time = 1.0 / 60.0  # 60 FPS

    # Detect touches (updates entity positions first)
    from .semantic_world.gaussian_collision import detect_and_emit_touches
    touches = detect_and_emit_touches()

    # Process physics → affect
    await tick_physics_affect(delta_time)

    # The CharmNetworkFacets now have modified affect states!


__all__ = [
    'PhysicsAffectConfig',
    'PhysicsAffectBridge',
    'PhysicsAffectFacet',
    'init_physics_affect_bridge',
    'get_physics_affect_bridge',
    'tick_physics_affect',
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
