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
#   Perception System - Filters Scene Packets by Entity Perception
#
#   A Perception Slice is a filtered Scene Packet representin...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.semantic_world.perception
# PURPOSE:  Perception
# LAYER:    Studio / Semantic World
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PerceivedEntity, PerceivedEvent, SpatialAwareness, PerceptionSlice, PerceptionCalculator
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import math

from .scene_packet import (
    ScenePacket,
    Vector3,
    PerceptionCone,
    Affect,
    Zone,
    Noodling,
    Player,
    Prim,
    DialogueEntry,
    EventEntry,
    NarrativeContext,
)


# =============================================================================
# Perceived Entity (external observables only)
# =============================================================================

@dataclass
class PerceivedEntity:
    """
    What we can observe about another entity.

    Note: NO internal state access (affect, memories, true intentions).
    Only observable externals.
    """
    id: str
    display_name: str
    entity_type: str  # "noodling", "player", "prim"

    # Spatial (relative to perceiver)
    position: Vector3 = field(default_factory=Vector3)
    distance: float = 0.0
    direction: str = ""  # "in_front", "left", "behind", etc.
    visibility: float = 1.0  # 0-1, how clearly visible

    # Observable state (what we can SEE)
    posture: str = "standing"
    current_action: str = "idle"
    expression: str = "neutral"
    gaze_target: Optional[str] = None  # Who/what they're looking at

    # For prims: observable state
    prim_state: Dict[str, Any] = field(default_factory=dict)

    # Visual form (for noodlings with multiple forms)
    visual_form: str = "default"

    # NOT included:
    # - affect (internal)
    # - memories (internal)
    # - attention_focus details (internal)
    # - true intentions (internal)


@dataclass
class PerceivedEvent:
    """An event as perceived by an entity."""
    event_type: str  # "heard_speech", "observed_action", "noticed_change"
    actor: str
    action: str
    timestamp: float
    seconds_ago: float = 0.0

    # For speech
    text: Optional[str] = None
    tone: Optional[str] = None

    # For actions
    target: Optional[str] = None
    detail: Optional[str] = None

    # How well did we perceive it?
    clarity: float = 1.0  # 0-1


@dataclass
class SpatialAwareness:
    """What the perceiver knows about their spatial context."""
    current_zone: str = ""
    zone_name: str = ""
    known_exits: Dict[str, str] = field(default_factory=dict)

    ambient_lighting: str = "ambient"
    ambient_sounds: List[str] = field(default_factory=list)
    ambient_mood: str = "neutral"


# =============================================================================
# Perception Slice
# =============================================================================

@dataclass
class PerceptionSlice:
    """
    A filtered view of the world from one entity's perspective.

    This is what gets fed to a noodling's facet assembly as context.
    """

    # Who is perceiving
    perceiver_id: str
    timestamp: float

    # SELF - full internal access
    self_position: Vector3 = field(default_factory=Vector3)
    self_facing: Vector3 = field(default_factory=Vector3)
    self_zone: str = ""
    self_affect: Affect = field(default_factory=Affect)
    self_posture: str = "standing"
    self_action: str = "idle"

    # PERCEIVED ENTITIES - observable externals only
    perceived_entities: Dict[str, PerceivedEntity] = field(default_factory=dict)

    # PERCEIVED EVENTS - only witnessed
    perceived_events: List[PerceivedEvent] = field(default_factory=list)

    # SPATIAL AWARENESS
    spatial_context: SpatialAwareness = field(default_factory=SpatialAwareness)

    # Conversation context (if in conversation)
    conversation_partner: Optional[str] = None
    last_input_to_self: Optional[str] = None

    def to_context_dict(self) -> Dict[str, Any]:
        """
        Convert to a dict suitable for facet assembly context.
        """
        return {
            "self": {
                "id": self.perceiver_id,
                "position": self.self_position.to_list(),
                "facing": self.self_facing.to_list(),
                "zone": self.self_zone,
                "posture": self.self_posture,
                "action": self.self_action,
            },
            "affect": self.self_affect.to_dict(),
            "perceived_entities": {
                eid: {
                    "id": e.id,
                    "display_name": e.display_name,
                    "type": e.entity_type,
                    "distance": e.distance,
                    "direction": e.direction,
                    "visibility": e.visibility,
                    "posture": e.posture,
                    "action": e.current_action,
                    "expression": e.expression,
                    "looking_at": e.gaze_target,
                }
                for eid, e in self.perceived_entities.items()
            },
            "perceived_events": [
                {
                    "type": e.event_type,
                    "actor": e.actor,
                    "action": e.action,
                    "text": e.text,
                    "tone": e.tone,
                    "seconds_ago": e.seconds_ago,
                }
                for e in self.perceived_events
            ],
            "spatial": {
                "zone": self.spatial_context.current_zone,
                "zone_name": self.spatial_context.zone_name,
                "exits": self.spatial_context.known_exits,
                "lighting": self.spatial_context.ambient_lighting,
                "sounds": self.spatial_context.ambient_sounds,
            },
            "conversation_partner": self.conversation_partner,
            "last_input": self.last_input_to_self,
        }

    def to_narrative_text(self) -> str:
        """
        Convert to narrative text suitable for LLM context.
        """
        lines = []

        # Location
        if self.spatial_context.zone_name:
            lines.append(f"You are in {self.spatial_context.zone_name}.")
        if self.spatial_context.ambient_sounds:
            lines.append(f"You can hear: {', '.join(self.spatial_context.ambient_sounds)}.")

        # Others present
        if self.perceived_entities:
            lines.append("")
            for eid, entity in self.perceived_entities.items():
                if entity.entity_type == "noodling" or entity.entity_type == "player":
                    loc = f"{entity.direction}, {entity.distance:.1f}m away" if entity.direction else "nearby"
                    lines.append(
                        f"{entity.display_name} is {loc}, "
                        f"appearing {entity.expression}, {entity.posture}."
                    )
                    if entity.gaze_target == self.perceiver_id:
                        lines.append(f"  (They are looking at you.)")
                elif entity.entity_type == "prim":
                    lines.append(f"You notice: {entity.display_name}.")

        # Recent events
        if self.perceived_events:
            lines.append("")
            lines.append("Recently:")
            for event in self.perceived_events[-5:]:
                if event.event_type == "heard_speech":
                    lines.append(f'- {event.actor} said "{event.text}"')
                elif event.event_type == "observed_action":
                    lines.append(f"- {event.actor} {event.action}")

        return "\n".join(lines)


# =============================================================================
# Perception Calculator
# =============================================================================

class PerceptionCalculator:
    """
    Calculates what an entity can perceive from the full scene.
    """

    def compute_visibility(
        self,
        perceiver_pos: Vector3,
        perceiver_facing: Vector3,
        perceiver_cone: PerceptionCone,
        target_pos: Vector3,
        check_occlusion: bool = False
    ) -> float:
        """
        Calculate visibility of a target from perceiver's perspective.

        Returns 0-1 visibility score (0 = can't see, 1 = clearly visible).
        """
        # Distance check
        distance = perceiver_pos.distance_to(target_pos)
        if distance > perceiver_cone.range:
            return 0.0

        # Angle check (is target in FOV?)
        to_target = Vector3(
            target_pos.x - perceiver_pos.x,
            target_pos.y - perceiver_pos.y,
            target_pos.z - perceiver_pos.z
        ).normalized()

        facing_norm = perceiver_facing.normalized()

        # Dot product gives cosine of angle
        dot = facing_norm.dot(to_target)
        angle_deg = math.degrees(math.acos(max(-1, min(1, dot))))

        # Check against FOV (half-angle)
        half_fov = perceiver_cone.fov_horizontal / 2
        if angle_deg > half_fov:
            # Outside FOV... unless special abilities
            if perceiver_cone.occlusion_ignore:
                # Ghost form: 360 awareness
                pass
            elif perceiver_cone.heat_sense and distance < 10:
                # Heat sense: can detect nearby warm bodies
                return 0.3  # Partial awareness
            else:
                return 0.0

        # Calculate visibility score
        # Full visibility in center, drops toward edges
        angle_factor = 1.0 - (angle_deg / half_fov) * 0.3

        # Distance falloff
        distance_factor = 1.0 - (distance / perceiver_cone.range) * 0.5

        visibility = angle_factor * distance_factor

        # Motion bonus
        # (would need velocity data to implement properly)

        return max(0.0, min(1.0, visibility))

    def compute_direction_label(
        self,
        perceiver_pos: Vector3,
        perceiver_facing: Vector3,
        target_pos: Vector3
    ) -> str:
        """
        Get a human-readable direction label (in_front, left, behind, etc.)
        """
        to_target = Vector3(
            target_pos.x - perceiver_pos.x,
            0,  # Ignore vertical for direction
            target_pos.z - perceiver_pos.z
        ).normalized()

        facing_norm = Vector3(
            perceiver_facing.x,
            0,
            perceiver_facing.z
        ).normalized()

        # Dot product for front/back
        forward_dot = facing_norm.dot(to_target)

        # Cross product Y component for left/right
        cross_y = facing_norm.x * to_target.z - facing_norm.z * to_target.x

        if forward_dot > 0.7:
            if abs(cross_y) < 0.3:
                return "in_front"
            elif cross_y > 0:
                return "front_right"
            else:
                return "front_left"
        elif forward_dot < -0.7:
            if abs(cross_y) < 0.3:
                return "behind"
            elif cross_y > 0:
                return "behind_right"
            else:
                return "behind_left"
        else:
            if cross_y > 0:
                return "right"
            else:
                return "left"

    def can_hear(
        self,
        perceiver_pos: Vector3,
        source_pos: Vector3,
        volume: float = 1.0,  # 0-1, whisper to shout
        max_distance: float = 30.0
    ) -> Tuple[bool, float]:
        """
        Check if perceiver can hear something.

        Returns (can_hear, clarity) where clarity is 0-1.
        """
        distance = perceiver_pos.distance_to(source_pos)
        effective_range = max_distance * volume

        if distance > effective_range:
            return False, 0.0

        clarity = 1.0 - (distance / effective_range)

        # Whispers are harder to make out
        if volume < 0.3:
            clarity *= 0.5

        return True, clarity


# =============================================================================
# Perception Slice Generator
# =============================================================================

class PerceptionSliceGenerator:
    """
    Generates perception slices from full scene packets.

    Usage:
        generator = PerceptionSliceGenerator()
        slice = generator.generate(full_packet, "red")
        context = slice.to_context_dict()
    """

    def __init__(self):
        self.calculator = PerceptionCalculator()

    def generate(
        self,
        packet: ScenePacket,
        perceiver_id: str
    ) -> PerceptionSlice:
        """
        Generate a perception slice for the given entity.

        Args:
            packet: The full scene packet
            perceiver_id: ID of the entity to generate slice for

        Returns:
            PerceptionSlice filtered to what this entity perceives
        """
        # Find the perceiver in the packet
        perceiver = None
        perceiver_type = None

        if perceiver_id in packet.noodlings:
            perceiver = packet.noodlings[perceiver_id]
            perceiver_type = "noodling"
        elif perceiver_id in packet.players:
            perceiver = packet.players[perceiver_id]
            perceiver_type = "player"

        if not perceiver:
            # Perceiver not in scene, return empty slice
            return PerceptionSlice(
                perceiver_id=perceiver_id,
                timestamp=packet.header.timestamp
            )

        # Create the slice
        slice = PerceptionSlice(
            perceiver_id=perceiver_id,
            timestamp=packet.header.timestamp,
            self_position=perceiver.position,
            self_facing=perceiver.facing,
            self_zone=perceiver.zone,
            self_posture=perceiver.posture,
            self_action=perceiver.current_action,
        )

        # Add self affect (noodlings only)
        if perceiver_type == "noodling" and hasattr(perceiver, 'affect'):
            slice.self_affect = perceiver.affect

        # Get perception cone
        cone = perceiver.perception

        # Filter entities by perception
        self._filter_noodlings(packet, perceiver, cone, slice)
        self._filter_players(packet, perceiver, cone, slice)
        self._filter_prims(packet, perceiver, cone, slice)

        # Filter events by witness
        self._filter_events(packet, perceiver, cone, slice)

        # Build spatial awareness
        self._build_spatial_awareness(packet, perceiver, slice)

        # Determine conversation partner
        self._determine_conversation_partner(packet, perceiver, slice)

        return slice

    def _filter_noodlings(
        self,
        packet: ScenePacket,
        perceiver: Any,
        cone: PerceptionCone,
        slice: PerceptionSlice
    ):
        """Filter noodlings by perception."""
        for nid, noodling in packet.noodlings.items():
            if nid == slice.perceiver_id:
                continue

            visibility = self.calculator.compute_visibility(
                perceiver.position,
                perceiver.facing,
                cone,
                noodling.position
            )

            if visibility > 0.1:
                direction = self.calculator.compute_direction_label(
                    perceiver.position,
                    perceiver.facing,
                    noodling.position
                )

                slice.perceived_entities[nid] = PerceivedEntity(
                    id=nid,
                    display_name=noodling.display_name,
                    entity_type="noodling",
                    position=noodling.position,
                    distance=perceiver.position.distance_to(noodling.position),
                    direction=direction,
                    visibility=visibility,
                    posture=noodling.posture,
                    current_action=noodling.current_action,
                    expression=noodling.expression,
                    gaze_target=noodling.gaze_target,
                    visual_form=noodling.visual_state,
                )

    def _filter_players(
        self,
        packet: ScenePacket,
        perceiver: Any,
        cone: PerceptionCone,
        slice: PerceptionSlice
    ):
        """Filter players by perception."""
        for pid, player in packet.players.items():
            if pid == slice.perceiver_id:
                continue

            visibility = self.calculator.compute_visibility(
                perceiver.position,
                perceiver.facing,
                cone,
                player.position
            )

            if visibility > 0.1:
                direction = self.calculator.compute_direction_label(
                    perceiver.position,
                    perceiver.facing,
                    player.position
                )

                slice.perceived_entities[pid] = PerceivedEntity(
                    id=pid,
                    display_name=player.display_name,
                    entity_type="player",
                    position=player.position,
                    distance=perceiver.position.distance_to(player.position),
                    direction=direction,
                    visibility=visibility,
                    posture=player.posture,
                    current_action=player.current_action,
                    gaze_target=player.gaze_target,
                )

    def _filter_prims(
        self,
        packet: ScenePacket,
        perceiver: Any,
        cone: PerceptionCone,
        slice: PerceptionSlice
    ):
        """Filter prims by perception."""
        for prim_id, prim in packet.prims.items():
            # Create a cone without heat_sense for prims (they're not warm bodies)
            prim_cone = PerceptionCone(
                fov_horizontal=cone.fov_horizontal,
                fov_vertical=cone.fov_vertical,
                range=cone.range,
                night_vision=cone.night_vision,
                heat_sense=False,  # Prims don't trigger heat sense
                motion_sensitivity=cone.motion_sensitivity,
                occlusion_ignore=cone.occlusion_ignore,
            )
            visibility = self.calculator.compute_visibility(
                perceiver.position,
                perceiver.facing,
                prim_cone,
                prim.position
            )

            if visibility > 0.1:
                direction = self.calculator.compute_direction_label(
                    perceiver.position,
                    perceiver.facing,
                    prim.position
                )

                slice.perceived_entities[prim_id] = PerceivedEntity(
                    id=prim_id,
                    display_name=prim.description or prim.prim_type,
                    entity_type="prim",
                    position=prim.position,
                    distance=perceiver.position.distance_to(prim.position),
                    direction=direction,
                    visibility=visibility,
                    prim_state=prim.state,
                )

    def _filter_events(
        self,
        packet: ScenePacket,
        perceiver: Any,
        cone: PerceptionCone,
        slice: PerceptionSlice
    ):
        """Filter events by what perceiver witnessed."""
        now = packet.header.timestamp

        # Filter dialogue
        for dialogue in packet.narrative_context.recent_dialogue:
            # Can we hear it?
            speaker_pos = self._get_entity_position(packet, dialogue.speaker)
            if speaker_pos:
                can_hear, clarity = self.calculator.can_hear(
                    perceiver.position,
                    speaker_pos,
                    volume=0.7 if dialogue.entry_type == "whisper" else 1.0
                )

                if can_hear:
                    slice.perceived_events.append(PerceivedEvent(
                        event_type="heard_speech",
                        actor=dialogue.speaker,
                        action="said",
                        text=dialogue.text,
                        tone=dialogue.tone,
                        timestamp=dialogue.timestamp,
                        seconds_ago=now - dialogue.timestamp,
                        clarity=clarity,
                    ))

        # Filter actions/events
        for event in packet.narrative_context.recent_events:
            actor_pos = self._get_entity_position(packet, event.actor)
            if actor_pos:
                visibility = self.calculator.compute_visibility(
                    perceiver.position,
                    perceiver.facing,
                    cone,
                    actor_pos
                )

                if visibility > 0.2:
                    slice.perceived_events.append(PerceivedEvent(
                        event_type="observed_action",
                        actor=event.actor,
                        action=event.action,
                        target=event.target,
                        detail=event.detail,
                        timestamp=event.timestamp,
                        seconds_ago=now - event.timestamp,
                        clarity=visibility,
                    ))

        # Sort by recency
        slice.perceived_events.sort(key=lambda e: e.seconds_ago)

    def _build_spatial_awareness(
        self,
        packet: ScenePacket,
        perceiver: Any,
        slice: PerceptionSlice
    ):
        """Build spatial context awareness."""
        # Find current zone
        current_zone = None
        max_strength = 0.0

        for zone in packet.spatial_truth.zones:
            strength = zone.perception_strength(perceiver.position)
            if strength > max_strength:
                max_strength = strength
                current_zone = zone

        if current_zone:
            slice.spatial_context = SpatialAwareness(
                current_zone=current_zone.id,
                zone_name=current_zone.name,
                known_exits=current_zone.exits.copy(),
                ambient_lighting=current_zone.lighting,
                ambient_mood=current_zone.mood,
                ambient_sounds=packet.spatial_truth.ambient.soundscape.copy(),
            )

    def _determine_conversation_partner(
        self,
        packet: ScenePacket,
        perceiver: Any,
        slice: PerceptionSlice
    ):
        """Determine who (if anyone) the perceiver is in conversation with."""
        # Simple heuristic: closest entity looking at us who we're looking at
        if not hasattr(perceiver, 'gaze_target') or not perceiver.gaze_target:
            return

        target_id = perceiver.gaze_target

        if target_id in slice.perceived_entities:
            target = slice.perceived_entities[target_id]
            # Are they looking back at us?
            if target.gaze_target == slice.perceiver_id:
                slice.conversation_partner = target_id

    def _get_entity_position(
        self,
        packet: ScenePacket,
        entity_id: str
    ) -> Optional[Vector3]:
        """Get an entity's position from the packet."""
        if entity_id in packet.noodlings:
            return packet.noodlings[entity_id].position
        elif entity_id in packet.players:
            return packet.players[entity_id].position
        elif entity_id in packet.prims:
            return packet.prims[entity_id].position
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

_global_generator: Optional[PerceptionSliceGenerator] = None


def get_perception_generator() -> PerceptionSliceGenerator:
    """Get or create the global perception slice generator."""
    global _global_generator
    if _global_generator is None:
        _global_generator = PerceptionSliceGenerator()
    return _global_generator


def generate_perception_slice(
    packet: ScenePacket,
    perceiver_id: str
) -> PerceptionSlice:
    """
    Convenience function to generate a perception slice.

    Args:
        packet: The full scene packet
        perceiver_id: ID of the entity to generate slice for

    Returns:
        PerceptionSlice filtered to what this entity perceives
    """
    return get_perception_generator().generate(packet, perceiver_id)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "PerceivedEntity",
    "PerceivedEvent",
    "SpatialAwareness",
    "PerceptionSlice",

    # Calculator
    "PerceptionCalculator",

    # Generator
    "PerceptionSliceGenerator",
    "get_perception_generator",
    "generate_perception_slice",
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
