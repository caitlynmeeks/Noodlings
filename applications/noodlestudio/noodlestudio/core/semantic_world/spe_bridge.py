"""
SPE Bridge - Connects Semantic Physics Engine to Action Stream

Routes ActionStream interactions through the Semantic Physics Engine (SPE)
for narrative-first physics resolution.

Flow:
    ActionStream action (interact, throw, etc.)
        ↓
    SPE Bridge resolves via PhysicsInteractionEngine
        ↓
    InteractionOutcome (description, sound, state_change)
        ↓
    SceneStateManager updates
        ↓
    PhysicsAffectBroadcaster notifies noodlings
        ↓
    ScenePacketEmitter emits delta

Author: Commander Spock + Cadet Caity
Date: December 18, 2025
"""

import logging
import sys
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Add cmush to path for SPE imports
_cmush_path = os.path.join(os.path.dirname(__file__), '../../../../cmush')
if _cmush_path not in sys.path:
    sys.path.insert(0, _cmush_path)

# Import SPE components
try:
    from physics_object_descriptor import PhysicsObjectDescriptor, PhysicsEvent
    from physics_interactions import (
        PhysicsInteractionEngine,
        InteractionOutcome,
        InteractionType,
    )
    from state_transitions import StateTransitionManager
    from physics_affect_bridge import (
        PhysicsAffectBroadcaster,
        PhysicsAffectExtractor,
        PhysicsAffectEvent,
    )
    SPE_AVAILABLE = True
    logger.info("[SPE Bridge] Semantic Physics Engine loaded successfully")
except ImportError as e:
    logger.warning(f"[SPE Bridge] SPE not available: {e}")
    SPE_AVAILABLE = False
    PhysicsObjectDescriptor = None
    PhysicsInteractionEngine = None
    InteractionOutcome = None


# =============================================================================
# Spatial Resolver
# =============================================================================

@dataclass
class SpatialContext:
    """
    Spatial context for a physics interaction.

    Provides distance, direction, and trajectory information
    that the SPE uses to generate appropriate descriptions.
    """
    # Distance between actor and target
    distance: float = 0.0

    # Direction labels (relative to actor)
    direction_from_actor: str = "ahead"  # "ahead", "left", "behind", etc.
    direction_to_actor: str = "ahead"    # From target's perspective

    # Is actor facing target?
    actor_facing_target: bool = True
    target_facing_actor: bool = False

    # Zone context
    same_zone: bool = True
    actor_zone: str = ""
    target_zone: str = ""

    # Trajectory hints (for thrown objects)
    trajectory_description: str = ""  # "arcs over the campfire", "rolls along the ground"

    # Nearby entities that might be affected
    entities_in_path: List[str] = field(default_factory=list)
    entities_nearby: List[str] = field(default_factory=list)


class SpatialResolver:
    """
    Resolves spatial relationships for SPE physics.

    Uses Transform data from entities to calculate:
    - Distance and direction between actor and target
    - Whether entities are facing each other
    - Trajectory descriptions for thrown objects
    - Which other entities might witness or be affected
    """

    def __init__(self, scene_state_manager=None):
        self.scene_state_manager = scene_state_manager

    def set_scene_state_manager(self, manager):
        self.scene_state_manager = manager

    def resolve(
        self,
        actor_id: str,
        target_id: str,
        verb: str = "interact"
    ) -> SpatialContext:
        """
        Resolve spatial context for an interaction.

        Args:
            actor_id: ID of acting entity (player, noodling)
            target_id: ID of target entity (prim, noodling)
            verb: Interaction verb for trajectory hints

        Returns:
            SpatialContext with distance, direction, trajectory info
        """
        ctx = SpatialContext()

        if not self.scene_state_manager:
            return ctx

        # Get transforms
        actor_transform = self._get_transform(actor_id)
        target_transform = self._get_transform(target_id)

        if not actor_transform or not target_transform:
            return ctx

        # Calculate distance
        ctx.distance = actor_transform.distance_to(target_transform)

        # Calculate directions
        ctx.direction_from_actor = actor_transform.relative_direction_label(target_transform)
        ctx.direction_to_actor = target_transform.relative_direction_label(actor_transform)

        # Check facing
        ctx.actor_facing_target = actor_transform.is_facing(target_transform)
        ctx.target_facing_actor = target_transform.is_facing(actor_transform)

        # Zone context
        actor_zone = self._get_zone(actor_id)
        target_zone = self._get_zone(target_id)
        ctx.actor_zone = actor_zone
        ctx.target_zone = target_zone
        ctx.same_zone = (actor_zone == target_zone) if actor_zone and target_zone else True

        # Trajectory description for thrown objects
        if verb in ["throw", "toss", "hurl", "fling"]:
            ctx.trajectory_description = self._describe_trajectory(
                actor_transform, target_transform, ctx.distance
            )

        # Find entities in path and nearby
        ctx.entities_in_path = self._find_entities_in_path(
            actor_transform.position, target_transform.position
        )
        ctx.entities_nearby = self._find_nearby_entities(
            target_transform.position, radius=2.0, exclude=[actor_id, target_id]
        )

        return ctx

    def _get_transform(self, entity_id: str):
        """Get transform for an entity."""
        if not self.scene_state_manager:
            return None

        # Check noodlings
        if entity_id in self.scene_state_manager.noodlings:
            return self.scene_state_manager.noodlings[entity_id].transform

        # Check players
        if entity_id in self.scene_state_manager.players:
            return self.scene_state_manager.players[entity_id].transform

        # Check prims
        if entity_id in self.scene_state_manager.prims:
            return self.scene_state_manager.prims[entity_id].transform

        return None

    def _get_zone(self, entity_id: str) -> str:
        """Get zone ID for an entity."""
        if not self.scene_state_manager:
            return ""

        # Check noodlings
        if entity_id in self.scene_state_manager.noodlings:
            return self.scene_state_manager.noodlings[entity_id].zone

        # Check players
        if entity_id in self.scene_state_manager.players:
            return self.scene_state_manager.players[entity_id].zone

        # Check prims
        if entity_id in self.scene_state_manager.prims:
            return self.scene_state_manager.prims[entity_id].zone

        return ""

    def _describe_trajectory(
        self,
        actor_transform,
        target_transform,
        distance: float
    ) -> str:
        """Generate a trajectory description for thrown objects."""
        from .scene_packet import Vector3

        # Height difference
        height_diff = target_transform.position.y - actor_transform.position.y

        # Distance categories
        if distance < 2:
            dist_desc = "short"
        elif distance < 5:
            dist_desc = "moderate"
        else:
            dist_desc = "long"

        # Trajectory based on height diff and distance
        if height_diff > 1:
            return f"arcs upward over the {dist_desc} distance"
        elif height_diff < -1:
            return f"descends along the {dist_desc} trajectory"
        elif distance > 5:
            return f"sails through the air across the {dist_desc} distance"
        else:
            return f"travels the {dist_desc} distance"

    def _find_entities_in_path(
        self,
        start: 'Vector3',
        end: 'Vector3',
        radius: float = 0.5
    ) -> List[str]:
        """Find entities that might be in the path between two points."""
        in_path = []

        if not self.scene_state_manager:
            return in_path

        # Simple line-sphere intersection check
        # For each entity, check if they're close to the line from start to end

        def point_to_line_distance(point, line_start, line_end):
            """Distance from point to line segment."""
            from .scene_packet import Vector3

            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = line_vec.magnitude()

            if line_len == 0:
                return point_vec.magnitude()

            # Project point onto line
            t = max(0, min(1, point_vec.dot(line_vec) / (line_len * line_len)))
            projection = line_start + line_vec * t

            return point.distance_to(projection)

        # Check noodlings
        for nid, noodling in self.scene_state_manager.noodlings.items():
            dist = point_to_line_distance(noodling.transform.position, start, end)
            if dist < radius:
                in_path.append(nid)

        # Check players
        for pid, player in self.scene_state_manager.players.items():
            dist = point_to_line_distance(player.transform.position, start, end)
            if dist < radius:
                in_path.append(pid)

        return in_path

    def _find_nearby_entities(
        self,
        position: 'Vector3',
        radius: float = 2.0,
        exclude: List[str] = None
    ) -> List[str]:
        """Find entities near a position."""
        nearby = []
        exclude = exclude or []

        if not self.scene_state_manager:
            return nearby

        # Check noodlings
        for nid, noodling in self.scene_state_manager.noodlings.items():
            if nid in exclude:
                continue
            dist = position.distance_to(noodling.transform.position)
            if dist <= radius:
                nearby.append(nid)

        # Check players
        for pid, player in self.scene_state_manager.players.items():
            if pid in exclude:
                continue
            dist = position.distance_to(player.transform.position)
            if dist <= radius:
                nearby.append(pid)

        return nearby


# =============================================================================
# Interaction Verb Mapping
# =============================================================================

# Map action verbs to SPE InteractionTypes
VERB_TO_INTERACTION = {
    # Strike verbs
    "strike": "STRIKE",
    "hit": "STRIKE",
    "bash": "STRIKE",
    "slam": "STRIKE",
    "punch": "STRIKE",
    "kick": "STRIKE",

    # Throw verbs
    "throw": "THROW",
    "toss": "THROW",
    "hurl": "THROW",
    "lob": "THROW",
    "fling": "THROW",

    # Drop verbs
    "drop": "DROP",
    "release": "DROP",
    "let_go": "DROP",

    # Pickup verbs
    "pickup": "PICKUP",
    "pick_up": "PICKUP",
    "grab": "PICKUP",
    "take": "PICKUP",

    # Give verbs
    "give": "GIVE",
    "hand": "GIVE",
    "transfer": "GIVE",
    "offer": "GIVE",

    # Push verbs
    "push": "PUSH",
    "shove": "PUSH",
    "nudge": "PUSH",

    # Pull verbs
    "pull": "PULL",
    "tug": "PULL",
    "drag": "PULL",

    # Generic
    "use": "STRIKE",  # Default to strike for generic "use"
    "toggle": "STRIKE",  # Toggle is like pressing
    "activate": "STRIKE",
}


# =============================================================================
# POD Cache
# =============================================================================

@dataclass
class PODCache:
    """
    Cache of PhysicsObjectDescriptors for entities.

    PODs are created from entity metadata or defaults.
    """
    pods: Dict[str, 'PhysicsObjectDescriptor'] = field(default_factory=dict)

    def get_or_create(
        self,
        entity_id: str,
        entity_type: str = "prim",
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'PhysicsObjectDescriptor':
        """
        Get or create POD for an entity.

        Args:
            entity_id: Entity identifier
            entity_type: "prim", "noodling", "player"
            metadata: Entity metadata for POD creation

        Returns:
            PhysicsObjectDescriptor for entity
        """
        if not SPE_AVAILABLE:
            return None

        if entity_id in self.pods:
            return self.pods[entity_id]

        # Create from metadata or defaults
        meta = metadata or {}
        pod = PhysicsObjectDescriptor(
            mass=meta.get('mass', 'medium'),
            friction=meta.get('friction', 'medium'),
            velocity=meta.get('velocity', 'stationary'),
            elasticity=meta.get('elasticity', 'normal'),
            softness=meta.get('softness', 'normal'),
            material=meta.get('material', _guess_material(entity_id, entity_type)),
            semantic_properties=meta.get('semantic_properties', []),
            state=meta.get('state', 'normal'),
            metadata=meta.get('pod_metadata', {}),
            tags=meta.get('tags', [])
        )

        self.pods[entity_id] = pod
        return pod

    def update_from_prim(self, prim) -> 'PhysicsObjectDescriptor':
        """
        Update POD from a Prim entity.

        Pulls all physics properties (mass, friction, elasticity, softness, material)
        from the prim to create an accurate POD for the SPE.

        Boulder vs feather: A boulder has mass='immovable', friction='high',
        while a feather has mass='negligible', friction='low'.
        """
        if not SPE_AVAILABLE:
            return None

        # If prim already has a POD, use it
        if hasattr(prim, 'pod') and prim.pod:
            self.pods[prim.id] = prim.pod
            return prim.pod

        # Create from prim's physics properties
        return self.get_or_create(
            prim.id,
            "prim",
            {
                # Core physics properties from Prim
                'mass': getattr(prim, 'mass', 'medium'),
                'friction': getattr(prim, 'friction', 'medium'),
                'elasticity': getattr(prim, 'elasticity', 'normal'),
                'softness': getattr(prim, 'softness', 'normal'),
                'material': getattr(prim, 'material', None) or getattr(prim, 'prim_type', 'object'),
                'state': getattr(prim, 'state', {}).get('condition', 'normal') if isinstance(getattr(prim, 'state', {}), dict) else 'normal',
            }
        )


def _guess_material(entity_id: str, entity_type: str) -> str:
    """Guess material from entity ID/type."""
    id_lower = entity_id.lower()

    # Common material hints in names
    if 'rock' in id_lower or 'stone' in id_lower:
        return 'stone'
    elif 'metal' in id_lower or 'iron' in id_lower:
        return 'metal'
    elif 'wood' in id_lower or 'log' in id_lower:
        return 'wood'
    elif 'glass' in id_lower:
        return 'glass'
    elif 'water' in id_lower or 'puddle' in id_lower:
        return 'water'
    elif 'fire' in id_lower or 'flame' in id_lower:
        return 'flame'
    elif 'radio' in id_lower:
        return 'plastic and metal'
    elif 'can' in id_lower:
        return 'thin metal'

    # Entity type defaults
    if entity_type == 'noodling':
        return 'organic'
    elif entity_type == 'player':
        return 'organic'

    return 'unknown material'


# =============================================================================
# SPE Bridge
# =============================================================================

class SPEBridge:
    """
    Bridge between ActionStream and Semantic Physics Engine.

    Resolves interaction actions through SPE and applies outcomes
    to SceneStateManager.
    """

    def __init__(self, scene_state_manager=None):
        """
        Initialize SPE Bridge.

        Args:
            scene_state_manager: SceneStateManager for state updates
        """
        self.scene_state_manager = scene_state_manager
        self.pod_cache = PODCache()

        # Spatial resolver for transform-based calculations
        self.spatial_resolver = SpatialResolver(scene_state_manager)

        # SPE components (lazy init)
        self._interaction_engine: Optional['PhysicsInteractionEngine'] = None
        self._transition_manager: Optional['StateTransitionManager'] = None
        self._affect_broadcaster: Optional['PhysicsAffectBroadcaster'] = None
        self._affect_extractor: Optional['PhysicsAffectExtractor'] = None

        # Callbacks for events
        self._on_interaction: List[callable] = []
        self._on_state_change: List[callable] = []

    @property
    def interaction_engine(self) -> Optional['PhysicsInteractionEngine']:
        """Get or create PhysicsInteractionEngine."""
        if not SPE_AVAILABLE:
            return None

        if self._interaction_engine is None:
            self._interaction_engine = PhysicsInteractionEngine(
                transition_mgr=self.transition_manager
            )
        return self._interaction_engine

    @property
    def transition_manager(self) -> Optional['StateTransitionManager']:
        """Get or create StateTransitionManager."""
        if not SPE_AVAILABLE:
            return None

        if self._transition_manager is None:
            self._transition_manager = StateTransitionManager()
        return self._transition_manager

    @property
    def affect_extractor(self) -> Optional['PhysicsAffectExtractor']:
        """Get or create PhysicsAffectExtractor."""
        if not SPE_AVAILABLE:
            return None

        if self._affect_extractor is None:
            self._affect_extractor = PhysicsAffectExtractor()
        return self._affect_extractor

    def set_scene_state_manager(self, manager):
        """Set the scene state manager."""
        self.scene_state_manager = manager
        self.spatial_resolver.set_scene_state_manager(manager)

    # =========================================================================
    # Interaction Resolution
    # =========================================================================

    def resolve_interaction(
        self,
        actor_id: str,
        target_id: str,
        verb: str,
        force: str = "medium",
        actor_metadata: Optional[Dict] = None,
        target_metadata: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a physics interaction through SPE.

        Uses Transform data to calculate spatial context (distance, direction,
        trajectory) and incorporates it into the physics description.

        Args:
            actor_id: Acting entity (player, noodling, or projectile)
            target_id: Target entity (prim, noodling)
            verb: Interaction verb ("throw", "strike", "push", etc.)
            force: Force level ("light", "medium", "heavy")
            actor_metadata: Optional metadata for actor POD
            target_metadata: Optional metadata for target POD

        Returns:
            Dict with outcome details including spatial context, or None if SPE unavailable
        """
        if not SPE_AVAILABLE or not self.interaction_engine:
            logger.debug("[SPE Bridge] SPE not available, skipping interaction")
            return None

        # Resolve spatial context from transforms
        spatial_ctx = self.spatial_resolver.resolve(actor_id, target_id, verb)

        # Adjust force based on distance (throwing from far = less impact)
        effective_force = force
        if verb in ["throw", "toss", "hurl"] and spatial_ctx.distance > 5:
            if force == "heavy":
                effective_force = "medium"
            elif force == "medium":
                effective_force = "light"

        # Get interaction type from verb
        interaction_type = VERB_TO_INTERACTION.get(verb.lower(), "STRIKE")

        # Get or create PODs
        actor_pod = self.pod_cache.get_or_create(
            actor_id, "actor", actor_metadata
        )
        target_pod = self.pod_cache.get_or_create(
            target_id, "prim", target_metadata
        )

        # Resolve through SPE
        try:
            if interaction_type == "STRIKE":
                outcome = self.interaction_engine.strike(
                    actor_pod, target_pod,
                    actor_id, target_id,
                    force=effective_force
                )
            elif interaction_type == "THROW":
                outcome = self.interaction_engine.throw(
                    actor_id, actor_pod,
                    actor_id,  # projectile is actor in this case
                    target_pod, target_id,
                    force=effective_force
                )
            elif interaction_type == "DROP":
                outcome = self.interaction_engine.drop(
                    target_pod, target_id
                )
            elif interaction_type == "PICKUP":
                outcome = self.interaction_engine.pick_up(
                    actor_id, target_pod, target_id
                )
            elif interaction_type == "PUSH":
                outcome = self.interaction_engine.push(
                    actor_id, target_pod, target_id,
                    force=effective_force
                )
            elif interaction_type == "PULL":
                outcome = self.interaction_engine.pull(
                    actor_id, target_pod, target_id,
                    force=effective_force
                )
            else:
                # Default to strike
                outcome = self.interaction_engine.strike(
                    actor_pod, target_pod,
                    actor_id, target_id,
                    force=effective_force
                )

            logger.info(f"[SPE Bridge] Resolved {verb}: {outcome.description[:80]}...")

            # Apply outcome to SceneStateManager
            self._apply_outcome(actor_id, target_id, outcome, interaction_type)

            # Extract affect for noodlings
            affect_event = self._extract_affect(outcome, interaction_type)

            # Notify callbacks
            for callback in self._on_interaction:
                try:
                    callback(actor_id, target_id, verb, outcome)
                except Exception as e:
                    logger.error(f"[SPE Bridge] Callback error: {e}")

            return {
                'description': outcome.description,
                'sound': outcome.sound,
                'visual': outcome.visual,
                'actor_state_change': outcome.actor_state_change,
                'target_state_change': outcome.target_state_change,
                'secondary_effects': outcome.secondary_effects,
                'affect': asdict(affect_event) if affect_event else None,
                # Spatial context
                'spatial': {
                    'distance': spatial_ctx.distance,
                    'direction': spatial_ctx.direction_from_actor,
                    'trajectory': spatial_ctx.trajectory_description,
                    'actor_facing_target': spatial_ctx.actor_facing_target,
                    'target_facing_actor': spatial_ctx.target_facing_actor,
                    'same_zone': spatial_ctx.same_zone,
                    'entities_in_path': spatial_ctx.entities_in_path,
                    'entities_nearby': spatial_ctx.entities_nearby,
                },
            }

        except Exception as e:
            logger.error(f"[SPE Bridge] Interaction resolution failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_outcome(
        self,
        actor_id: str,
        target_id: str,
        outcome: 'InteractionOutcome',
        interaction_type: str
    ):
        """Apply interaction outcome to SceneStateManager."""
        if not self.scene_state_manager:
            return

        # Update target prim state
        if outcome.target_state_change and target_id in self.scene_state_manager.prims:
            prim = self.scene_state_manager.prims[target_id]
            if hasattr(prim, 'state') and isinstance(prim.state, dict):
                prim.state['condition'] = outcome.target_state_change
            # Update POD state too
            if target_id in self.pod_cache.pods:
                self.pod_cache.pods[target_id].state = outcome.target_state_change

        # Record event in narrative context (if available)
        if hasattr(self.scene_state_manager, 'narrative_context') and self.scene_state_manager.narrative_context:
            from .scene_packet import EventEntry
            import time
            event = EventEntry(
                actor=actor_id,
                action=f"{interaction_type.lower()} {target_id}",
                timestamp=time.time(),
                seconds_ago=0.0,
                target=target_id,
                detail=outcome.description[:100],
            )
            self.scene_state_manager.narrative_context.recent_events.append(event)

            # Handle secondary effects
            for effect in outcome.secondary_effects:
                if effect == "target_breaks":
                    self.scene_state_manager.narrative_context.scene_state.tension += 0.1
                elif effect == "target_ignites":
                    self.scene_state_manager.narrative_context.scene_state.tension += 0.2

        # Add sound to ambient if significant (if spatial_truth available)
        if outcome.sound and hasattr(self.scene_state_manager, 'spatial_truth') and self.scene_state_manager.spatial_truth:
            if hasattr(self.scene_state_manager.spatial_truth, 'ambient'):
                if outcome.sound not in self.scene_state_manager.spatial_truth.ambient.soundscape:
                    self.scene_state_manager.spatial_truth.ambient.soundscape.append(
                        f"recent:{outcome.sound}"
                    )

        logger.debug(f"[SPE Bridge] Applied outcome to SceneStateManager")

    def _extract_affect(
        self,
        outcome: 'InteractionOutcome',
        interaction_type: str
    ) -> Optional['PhysicsAffectEvent']:
        """Extract affect event from interaction outcome."""
        if not self.affect_extractor:
            return None

        try:
            # Map interaction type string to enum
            from physics_interactions import InteractionType
            int_type = InteractionType[interaction_type]

            affect_event = self.affect_extractor.extract_affect(outcome, int_type)
            return affect_event
        except Exception as e:
            logger.debug(f"[SPE Bridge] Affect extraction failed: {e}")
            return None

    # =========================================================================
    # POD Management
    # =========================================================================

    def register_prim_pod(
        self,
        prim_id: str,
        pod: 'PhysicsObjectDescriptor'
    ):
        """Register a POD for a prim."""
        if SPE_AVAILABLE:
            self.pod_cache.pods[prim_id] = pod

    def get_prim_pod(self, prim_id: str) -> Optional['PhysicsObjectDescriptor']:
        """Get POD for a prim."""
        return self.pod_cache.pods.get(prim_id)

    def sync_pods_from_scene(self):
        """Sync POD cache from SceneStateManager prims."""
        if not self.scene_state_manager:
            return

        for prim_id, prim in self.scene_state_manager.prims.items():
            self.pod_cache.update_from_prim(prim)

    # =========================================================================
    # Event Callbacks
    # =========================================================================

    def on_interaction(self, callback: callable):
        """Register callback for interaction events."""
        self._on_interaction.append(callback)

    def on_state_change(self, callback: callable):
        """Register callback for state changes."""
        self._on_state_change.append(callback)


# =============================================================================
# Global Instance
# =============================================================================

_spe_bridge: Optional[SPEBridge] = None


def get_spe_bridge() -> SPEBridge:
    """Get or create global SPE Bridge."""
    global _spe_bridge
    if _spe_bridge is None:
        _spe_bridge = SPEBridge()
    return _spe_bridge


def init_spe_bridge(scene_state_manager=None) -> SPEBridge:
    """Initialize global SPE Bridge with SceneStateManager."""
    global _spe_bridge
    _spe_bridge = SPEBridge(scene_state_manager)
    return _spe_bridge


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SPE_AVAILABLE",
    "SPEBridge",
    "PODCache",
    "VERB_TO_INTERACTION",
    "SpatialContext",
    "SpatialResolver",
    "get_spe_bridge",
    "init_spe_bridge",
]
