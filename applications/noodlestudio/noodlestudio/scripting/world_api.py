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
#   World API - Scripting interface for scene perception and world interaction.
#
#   Provides context.noodle.world in ScriptedFacets with perc...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.scripting.world_api
# PURPOSE:  World Api
# LAYER:    Studio / Scripting API
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PerceivedEntityJS, PerceivedEventJS, WorldAPIState, WorldAPI, get_world_api()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerceivedEntityJS:
    """
    Perceived entity data for JavaScript access.

    Only observable externals - NO internal state access.
    """
    id: str
    displayName: str
    entityType: str  # "noodling", "player", "prim"

    # Spatial (relative to perceiver)
    position: List[float] = field(default_factory=lambda: [0, 0, 0])
    distance: float = 0.0
    direction: str = ""  # "in_front", "left", "behind", etc.
    visibility: float = 1.0  # 0-1, how clearly visible

    # Observable state
    posture: str = "standing"
    action: str = "idle"
    expression: str = "neutral"
    lookingAt: Optional[str] = None  # gaze target

    # Visual form (for multi-state characters like Yuki)
    visualForm: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JavaScript-compatible dict."""
        return {
            'id': self.id,
            'displayName': self.displayName,
            'entityType': self.entityType,
            'position': self.position,
            'distance': self.distance,
            'direction': self.direction,
            'visibility': self.visibility,
            'posture': self.posture,
            'action': self.action,
            'expression': self.expression,
            'lookingAt': self.lookingAt,
            'visualForm': self.visualForm,
        }


@dataclass
class PerceivedEventJS:
    """Perceived event for JavaScript access."""
    eventType: str  # "heard_speech", "observed_action"
    actor: str
    action: str
    text: Optional[str] = None
    tone: Optional[str] = None
    target: Optional[str] = None
    secondsAgo: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'eventType': self.eventType,
            'actor': self.actor,
            'action': self.action,
            'text': self.text,
            'tone': self.tone,
            'target': self.target,
            'secondsAgo': self.secondsAgo,
        }


@dataclass
class WorldAPIState:
    """
    Snapshot of world state for JavaScript access.

    Updated from PerceptionSlice at each cognitive cycle.
    """
    # Self state
    my_position: List[float] = field(default_factory=lambda: [0, 0, 0])
    my_facing: List[float] = field(default_factory=lambda: [0, 0, 1])
    my_zone: str = ""
    my_zone_name: str = ""
    my_posture: str = "standing"
    my_action: str = "idle"
    my_expression: str = "neutral"
    my_gaze: Optional[str] = None

    # Affect (own internal state - accessible to self)
    affect_valence: float = 0.0
    affect_arousal: float = 0.5
    affect_dominance: float = 0.5
    affect_boredom: float = 0.0
    affect_sorrow: float = 0.0

    # Perceived entities
    perceived_entities: List[PerceivedEntityJS] = field(default_factory=list)

    # Perceived events
    perceived_events: List[PerceivedEventJS] = field(default_factory=list)

    # Spatial context
    zone_exits: Dict[str, str] = field(default_factory=dict)
    ambient_sounds: List[str] = field(default_factory=list)
    ambient_mood: str = "neutral"
    ambient_lighting: str = "ambient"

    # Conversation
    conversation_partner: Optional[str] = None
    last_input_to_self: Optional[str] = None

    # Pending commands (from JS to scene state manager)
    pending_expression: Optional[str] = None
    pending_posture: Optional[str] = None
    pending_gaze: Optional[str] = None
    pending_action: Optional[str] = None
    pending_move_to: Optional[List[float]] = None
    pending_speak: Optional[Dict[str, str]] = None  # {text, tone}
    pending_emote: Optional[str] = None

    # Camera commands (for director-enabled noodlings)
    pending_camera_focus: Optional[Dict[str, Any]] = None
    pending_camera_two_shot: Optional[Dict[str, Any]] = None
    pending_camera_pov: Optional[str] = None
    pending_camera_establish: Optional[str] = None


class WorldAPI:
    """
    World scripting API for context.noodle.world.

    Provides perception-filtered access to scene state and
    commands for modifying own state.

    Each noodling gets their own WorldAPI instance with their
    own filtered view of reality.
    """

    def __init__(self, noodling_id: str = "unknown"):
        """
        Initialize World API.

        Args:
            noodling_id: ID of the noodling this API serves
        """
        self.noodling_id = noodling_id
        self._state = WorldAPIState()
        self._scene_state_manager = None  # Set via set_scene_state_manager()
        self._camera_enabled = False  # Set for director-enabled noodlings
        self._spe_bridge = None  # SPE Bridge for physics (set via set_spe_bridge())

    def set_scene_state_manager(self, manager):
        """Set the scene state manager reference."""
        self._scene_state_manager = manager

    def set_spe_bridge(self, bridge):
        """Set the SPE Bridge for physics interactions."""
        self._spe_bridge = bridge

    def set_noodling_id(self, noodling_id: str):
        """Set which noodling this API serves."""
        self.noodling_id = noodling_id

    def enable_camera_control(self, enabled: bool = True):
        """Enable/disable camera control for this noodling."""
        self._camera_enabled = enabled

    def update_from_perception_slice(self, slice) -> None:
        """
        Update state from a PerceptionSlice.

        Called at the start of each cognitive cycle.

        Args:
            slice: PerceptionSlice for this noodling
        """
        # Self state
        self._state.my_position = slice.self_position.to_list() if hasattr(slice.self_position, 'to_list') else list(slice.self_position)
        self._state.my_facing = slice.self_facing.to_list() if hasattr(slice.self_facing, 'to_list') else list(slice.self_facing)
        self._state.my_zone = slice.self_zone
        self._state.my_zone_name = slice.spatial_context.zone_name
        self._state.my_posture = slice.self_posture
        self._state.my_action = slice.self_action

        # Affect
        if hasattr(slice, 'self_affect'):
            self._state.affect_valence = slice.self_affect.valence
            self._state.affect_arousal = slice.self_affect.arousal
            self._state.affect_dominance = slice.self_affect.dominance
            self._state.affect_boredom = slice.self_affect.boredom
            self._state.affect_sorrow = slice.self_affect.sorrow

        # Perceived entities
        self._state.perceived_entities = []
        for eid, entity in slice.perceived_entities.items():
            self._state.perceived_entities.append(PerceivedEntityJS(
                id=entity.id,
                displayName=entity.display_name,
                entityType=entity.entity_type,
                position=entity.position.to_list() if hasattr(entity.position, 'to_list') else [0, 0, 0],
                distance=entity.distance,
                direction=entity.direction,
                visibility=entity.visibility,
                posture=entity.posture,
                action=entity.current_action,
                expression=entity.expression,
                lookingAt=entity.gaze_target,
                visualForm=entity.visual_form,
            ))

        # Perceived events
        self._state.perceived_events = []
        for event in slice.perceived_events:
            self._state.perceived_events.append(PerceivedEventJS(
                eventType=event.event_type,
                actor=event.actor,
                action=event.action,
                text=event.text,
                tone=event.tone,
                target=event.target,
                secondsAgo=event.seconds_ago,
            ))

        # Spatial context
        self._state.zone_exits = slice.spatial_context.known_exits.copy()
        self._state.ambient_sounds = slice.spatial_context.ambient_sounds.copy()
        self._state.ambient_mood = slice.spatial_context.ambient_mood
        self._state.ambient_lighting = slice.spatial_context.ambient_lighting

        # Conversation
        self._state.conversation_partner = slice.conversation_partner
        self._state.last_input_to_self = slice.last_input_to_self

    def get_pending_commands(self) -> Dict[str, Any]:
        """
        Get and clear pending commands.

        Called by facet executor after cognitive cycle.
        """
        commands = {}

        if self._state.pending_expression:
            commands['expression'] = self._state.pending_expression
            self._state.pending_expression = None

        if self._state.pending_posture:
            commands['posture'] = self._state.pending_posture
            self._state.pending_posture = None

        if self._state.pending_gaze:
            commands['gaze'] = self._state.pending_gaze
            self._state.pending_gaze = None

        if self._state.pending_action:
            commands['action'] = self._state.pending_action
            self._state.pending_action = None

        if self._state.pending_move_to:
            commands['move_to'] = self._state.pending_move_to
            self._state.pending_move_to = None

        if self._state.pending_speak:
            commands['speak'] = self._state.pending_speak
            self._state.pending_speak = None

        if self._state.pending_emote:
            commands['emote'] = self._state.pending_emote
            self._state.pending_emote = None

        # Camera commands
        if self._camera_enabled:
            if self._state.pending_camera_focus:
                commands['camera_focus'] = self._state.pending_camera_focus
                self._state.pending_camera_focus = None

            if self._state.pending_camera_two_shot:
                commands['camera_two_shot'] = self._state.pending_camera_two_shot
                self._state.pending_camera_two_shot = None

            if self._state.pending_camera_pov:
                commands['camera_pov'] = self._state.pending_camera_pov
                self._state.pending_camera_pov = None

            if self._state.pending_camera_establish:
                commands['camera_establish'] = self._state.pending_camera_establish
                self._state.pending_camera_establish = None

        return commands

    # =========================================================================
    # JavaScript-accessible Properties
    # =========================================================================

    @property
    def perceivedEntities(self) -> List[Dict[str, Any]]:
        """List of entities I can perceive."""
        return [e.to_dict() for e in self._state.perceived_entities]

    @property
    def perceivedEvents(self) -> List[Dict[str, Any]]:
        """List of events I witnessed."""
        return [e.to_dict() for e in self._state.perceived_events]

    @property
    def myPosition(self) -> List[float]:
        """My current position [x, y, z]."""
        return self._state.my_position

    @property
    def myFacing(self) -> List[float]:
        """Direction I'm facing [x, y, z]."""
        return self._state.my_facing

    @property
    def myZone(self) -> str:
        """Current zone ID."""
        return self._state.my_zone

    @property
    def myZoneName(self) -> str:
        """Current zone display name."""
        return self._state.my_zone_name

    @property
    def myPosture(self) -> str:
        """My current posture (sitting, standing, etc.)."""
        return self._state.my_posture

    @property
    def myAction(self) -> str:
        """What I'm currently doing."""
        return self._state.my_action

    @property
    def myExpression(self) -> str:
        """My current facial expression."""
        return self._state.my_expression

    @property
    def myGaze(self) -> Optional[str]:
        """What/who I'm looking at."""
        return self._state.my_gaze

    @property
    def affect(self) -> Dict[str, float]:
        """My current affect state."""
        return {
            'valence': self._state.affect_valence,
            'arousal': self._state.affect_arousal,
            'dominance': self._state.affect_dominance,
            'boredom': self._state.affect_boredom,
            'sorrow': self._state.affect_sorrow,
        }

    @property
    def zoneExits(self) -> Dict[str, str]:
        """Available exits from current zone."""
        return self._state.zone_exits

    @property
    def ambientSounds(self) -> List[str]:
        """Sounds I can hear in current zone."""
        return self._state.ambient_sounds

    @property
    def ambientMood(self) -> str:
        """Mood of current zone."""
        return self._state.ambient_mood

    @property
    def conversationPartner(self) -> Optional[str]:
        """Who I'm in conversation with (mutual gaze)."""
        return self._state.conversation_partner

    @property
    def lastInput(self) -> Optional[str]:
        """Last input directed at me."""
        return self._state.last_input_to_self

    # =========================================================================
    # JavaScript-accessible Query Methods
    # =========================================================================

    def canSee(self, entity_id: str) -> bool:
        """Check if I can see a specific entity."""
        return any(e.id == entity_id for e in self._state.perceived_entities)

    def canHear(self, entity_id: str) -> bool:
        """Check if I can hear a specific entity (based on recent speech events)."""
        return any(
            e.eventType == "heard_speech" and e.actor == entity_id
            for e in self._state.perceived_events
        )

    def getEntity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get observable info about a perceived entity."""
        for e in self._state.perceived_entities:
            if e.id == entity_id:
                return e.to_dict()
        return None

    def getDistanceTo(self, entity_id: str) -> float:
        """Get distance to an entity (or -1 if not perceived)."""
        for e in self._state.perceived_entities:
            if e.id == entity_id:
                return e.distance
        return -1.0

    def getDirectionTo(self, entity_id: str) -> str:
        """Get direction label to entity ("in_front", "left", etc.)."""
        for e in self._state.perceived_entities:
            if e.id == entity_id:
                return e.direction
        return "unknown"

    def isLookingAtMe(self, entity_id: str) -> bool:
        """Check if entity is looking at me."""
        for e in self._state.perceived_entities:
            if e.id == entity_id:
                return e.lookingAt == self.noodling_id
        return False

    def getEntitiesInDirection(self, direction: str) -> List[Dict[str, Any]]:
        """Get all entities in a specific direction."""
        return [
            e.to_dict() for e in self._state.perceived_entities
            if e.direction == direction
        ]

    def getEntitiesWithinRange(self, max_distance: float) -> List[Dict[str, Any]]:
        """Get all entities within a certain distance."""
        return [
            e.to_dict() for e in self._state.perceived_entities
            if e.distance <= max_distance
        ]

    def getRecentSpeech(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent speech events."""
        speech = [
            e.to_dict() for e in self._state.perceived_events
            if e.eventType == "heard_speech"
        ]
        return speech[:limit]

    # =========================================================================
    # JavaScript-accessible Command Methods
    # =========================================================================

    def setExpression(self, expression: str) -> None:
        """Change my expression."""
        self._state.pending_expression = expression
        logger.debug(f"[WorldAPI] {self.noodling_id} setting expression: {expression}")

    def setPosture(self, posture: str) -> None:
        """Change my posture (sitting, standing, crouching, etc.)."""
        self._state.pending_posture = posture
        logger.debug(f"[WorldAPI] {self.noodling_id} setting posture: {posture}")

    def setGaze(self, target_id: str) -> None:
        """Look at something/someone."""
        self._state.pending_gaze = target_id
        logger.debug(f"[WorldAPI] {self.noodling_id} setting gaze: {target_id}")

    def setAction(self, action: str) -> None:
        """Set what I'm doing (speaking, listening, thinking, etc.)."""
        self._state.pending_action = action
        logger.debug(f"[WorldAPI] {self.noodling_id} setting action: {action}")

    def moveTo(self, x: float, y: float, z: float) -> None:
        """Move to a position."""
        self._state.pending_move_to = [x, y, z]
        logger.debug(f"[WorldAPI] {self.noodling_id} moving to: [{x}, {y}, {z}]")

    def speak(self, text: str, tone: str = "neutral") -> None:
        """
        Say something (records to dialogue history).

        Args:
            text: What to say
            tone: Tone of voice ("friendly", "teasing", "angry", etc.)
        """
        self._state.pending_speak = {'text': text, 'tone': tone}
        logger.debug(f"[WorldAPI] {self.noodling_id} speaking: '{text}' ({tone})")

    def emote(self, text: str) -> None:
        """
        Do an emote/action.

        Args:
            text: The emote text (e.g., "waves dismissively")
        """
        self._state.pending_emote = text
        logger.debug(f"[WorldAPI] {self.noodling_id} emoting: {text}")

    # =========================================================================
    # Camera Control Methods (if enabled)
    # =========================================================================

    def focusCamera(self, entity_id: str, framing: str = "medium") -> bool:
        """
        Focus camera on an entity.

        Args:
            entity_id: Entity to focus on
            framing: "closeup", "medium", "wide", etc.

        Returns:
            True if command accepted (camera control enabled)
        """
        if not self._camera_enabled:
            logger.warning(f"[WorldAPI] {self.noodling_id} camera control not enabled")
            return False

        self._state.pending_camera_focus = {
            'subject': entity_id,
            'framing': framing
        }
        return True

    def twoShot(self, entity_a: str, entity_b: str, framing: str = "medium") -> bool:
        """Frame two entities together."""
        if not self._camera_enabled:
            return False

        self._state.pending_camera_two_shot = {
            'subjects': [entity_a, entity_b],
            'framing': framing
        }
        return True

    def povCamera(self, entity_id: str) -> bool:
        """Switch to entity's point of view."""
        if not self._camera_enabled:
            return False

        self._state.pending_camera_pov = entity_id
        return True

    def establishShot(self, zone_id: str) -> bool:
        """Wide establishing shot of a zone."""
        if not self._camera_enabled:
            return False

        self._state.pending_camera_establish = zone_id
        return True

    # =========================================================================
    # Physics Methods (SPE - Semantic Physics Engine)
    # =========================================================================

    def throw(self, target_id: str, force: str = "medium") -> Optional[Dict[str, Any]]:
        """
        Throw something at a target.

        Args:
            target_id: Entity to throw at
            force: "light", "medium", or "heavy"

        Returns:
            Outcome dict with description, sound, state_change, or None if SPE unavailable

        Example (JavaScript in ScriptedFacet):
            var result = context.noodle.world.throw("radio", "heavy");
            if (result) {
                context.log("Threw it: " + result.description);
            }
        """
        return self._resolve_physics("throw", target_id, force)

    def strike(self, target_id: str, force: str = "medium") -> Optional[Dict[str, Any]]:
        """
        Strike/hit something.

        Args:
            target_id: Entity to strike
            force: "light", "medium", or "heavy"

        Returns:
            Outcome dict with description, sound, state_change
        """
        return self._resolve_physics("strike", target_id, force)

    def push(self, target_id: str, force: str = "medium") -> Optional[Dict[str, Any]]:
        """
        Push something.

        Args:
            target_id: Entity to push
            force: "light", "medium", or "heavy"

        Returns:
            Outcome dict with description, sound, state_change
        """
        return self._resolve_physics("push", target_id, force)

    def pull(self, target_id: str, force: str = "medium") -> Optional[Dict[str, Any]]:
        """
        Pull something.

        Args:
            target_id: Entity to pull
            force: "light", "medium", or "heavy"

        Returns:
            Outcome dict with description, sound, state_change
        """
        return self._resolve_physics("pull", target_id, force)

    def pickup(self, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Pick up something.

        Args:
            target_id: Entity to pick up

        Returns:
            Outcome dict with description, sound, state_change
        """
        return self._resolve_physics("pickup", target_id, "medium")

    def drop(self, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Drop something.

        Args:
            target_id: Entity to drop

        Returns:
            Outcome dict with description, sound, state_change
        """
        return self._resolve_physics("drop", target_id, "medium")

    def give(self, item_id: str, recipient_id: str) -> Optional[Dict[str, Any]]:
        """
        Give something to someone.

        Args:
            item_id: Item to give
            recipient_id: Entity to give it to

        Returns:
            Outcome dict with description, sound, state_change
        """
        return self._resolve_physics("give", item_id, "medium", recipient_id)

    def interact(self, target_id: str, verb: str = "use") -> Optional[Dict[str, Any]]:
        """
        Generic interaction with semantic physics resolution.

        Args:
            target_id: Entity to interact with
            verb: Interaction verb ("use", "toggle", "press", etc.)

        Returns:
            Outcome dict with description, sound, state_change
        """
        return self._resolve_physics(verb, target_id, "medium")

    def _resolve_physics(
        self,
        verb: str,
        target_id: str,
        force: str,
        secondary_target: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a physics interaction through SPE.

        Internal method used by all physics actions.
        """
        if not self._spe_bridge:
            logger.debug(f"[WorldAPI] SPE not available for {verb}")
            return None

        try:
            outcome = self._spe_bridge.resolve_interaction(
                actor_id=self.noodling_id,
                target_id=target_id,
                verb=verb,
                force=force
            )
            if outcome:
                logger.debug(f"[WorldAPI] {self.noodling_id} {verb} {target_id}: {outcome.get('description', '')[:40]}...")
            return outcome
        except Exception as e:
            logger.error(f"[WorldAPI] Physics resolution failed: {e}")
            return None

    @property
    def physicsEnabled(self) -> bool:
        """Check if SPE physics is available."""
        return self._spe_bridge is not None

    # =========================================================================
    # Entity Manipulation Methods (for manipulating OTHER entities)
    # =========================================================================

    def setPosition(self, entity_id: str, x: float, y: float, z: float) -> bool:
        """
        Set position of ANY entity (not just self).

        Args:
            entity_id: ID of the entity to move
            x, y, z: New position coordinates

        Returns:
            True if successful

        Example (JavaScript in ScriptedFacet):
            context.noodle.world.setPosition("radio", 10, 0, 5);
        """
        return self._update_entity_transform(entity_id, position=[x, y, z])

    def setRotation(self, entity_id: str, pitch: float, yaw: float, roll: float) -> bool:
        """
        Set rotation of ANY entity.

        Args:
            entity_id: ID of the entity to rotate
            pitch, yaw, roll: Euler angles in degrees

        Returns:
            True if successful

        Example (JavaScript in ScriptedFacet):
            context.noodle.world.setRotation("radio", 0, 90, 0);
        """
        return self._update_entity_transform(entity_id, rotation=[pitch, yaw, roll])

    def setScale(self, entity_id: str, x: float, y: float, z: float) -> bool:
        """
        Set scale of ANY entity.

        Args:
            entity_id: ID of the entity to scale
            x, y, z: Scale factors

        Returns:
            True if successful

        Example (JavaScript in ScriptedFacet):
            context.noodle.world.setScale("radio", 2, 2, 2);
        """
        return self._update_entity_transform(entity_id, scale=[x, y, z])

    def setMaterial(self, entity_id: str, material: str) -> bool:
        """
        Set material preset of ANY entity.

        Args:
            entity_id: ID of the entity
            material: Material preset name ("wood", "metal", "glass", etc.)

        Returns:
            True if successful

        Example (JavaScript in ScriptedFacet):
            context.noodle.world.setMaterial("radio", "metal");
        """
        return self._update_entity_property(entity_id, 'material', material)

    def setPhysics(self, entity_id: str, physics: Dict[str, str]) -> bool:
        """
        Set physics properties of ANY entity.

        Args:
            entity_id: ID of the entity
            physics: Dict with mass, friction, elasticity, softness

        Returns:
            True if successful

        Example (JavaScript in ScriptedFacet):
            context.noodle.world.setPhysics("radio", {mass: "heavy", friction: "high"});
        """
        try:
            for key, value in physics.items():
                if key in ('mass', 'friction', 'elasticity', 'softness'):
                    self._update_entity_property(entity_id, key, value)
            return True
        except Exception as e:
            logger.error(f"[WorldAPI] setPhysics failed: {e}")
            return False

    def setProperty(self, entity_id: str, key: str, value: Any) -> bool:
        """
        Set a custom property on ANY entity.

        Args:
            entity_id: ID of the entity
            key: Property key
            value: Property value (any JSON-serializable type)

        Returns:
            True if successful

        Example (JavaScript in ScriptedFacet):
            context.noodle.world.setProperty("radio", "is_playing", true);
        """
        return self._set_entity_custom_property(entity_id, key, value)

    def getProperty(self, entity_id: str, key: str) -> Any:
        """
        Get a custom property from ANY entity.

        Args:
            entity_id: ID of the entity
            key: Property key

        Returns:
            Property value or None if not found

        Example (JavaScript in ScriptedFacet):
            var isPlaying = context.noodle.world.getProperty("radio", "is_playing");
        """
        return self._get_entity_custom_property(entity_id, key)

    def getTransform(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get transform of ANY entity.

        Args:
            entity_id: ID of the entity

        Returns:
            Dict with position, rotation, scale or None if not found

        Example (JavaScript in ScriptedFacet):
            var transform = context.noodle.world.getTransform("radio");
            if (transform) {
                context.log("Radio is at " + transform.position.x);
            }
        """
        return self._get_entity_transform(entity_id)

    def _update_entity_transform(
        self,
        entity_id: str,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None
    ) -> bool:
        """Internal: Update entity transform via REST API or scene state manager."""
        if self._scene_state_manager:
            try:
                # Use scene state manager if available
                if hasattr(self._scene_state_manager, 'update_entity_transform'):
                    return self._scene_state_manager.update_entity_transform(
                        entity_id, position=position, rotation=rotation, scale=scale
                    )
            except Exception as e:
                logger.error(f"[WorldAPI] Scene state manager update failed: {e}")

        # Fall back to REST API call
        try:
            import urllib.request
            import json

            data = {}
            if position:
                data['position'] = {'x': position[0], 'y': position[1], 'z': position[2]}
            if rotation:
                data['rotation'] = {'x': rotation[0], 'y': rotation[1], 'z': rotation[2]}
            if scale:
                data['scale'] = {'x': scale[0], 'y': scale[1], 'z': scale[2]}

            url = f"http://localhost:8081/api/entities/{entity_id}/transform"
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='PATCH'
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('success', False)

        except Exception as e:
            logger.error(f"[WorldAPI] Transform update failed for {entity_id}: {e}")
            return False

    def _update_entity_property(self, entity_id: str, key: str, value: Any) -> bool:
        """Internal: Update single entity property via REST API."""
        try:
            import urllib.request
            import json

            url = f"http://localhost:8081/api/entities/{entity_id}/properties/{key}"
            req = urllib.request.Request(
                url,
                data=json.dumps({'value': value}).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('success', False)

        except Exception as e:
            logger.error(f"[WorldAPI] Property update failed for {entity_id}.{key}: {e}")
            return False

    def _set_entity_custom_property(self, entity_id: str, key: str, value: Any) -> bool:
        """Internal: Set custom property on entity."""
        return self._update_entity_property(entity_id, key, value)

    def _get_entity_custom_property(self, entity_id: str, key: str) -> Any:
        """Internal: Get custom property from entity via REST API."""
        try:
            import urllib.request
            import json

            url = f"http://localhost:8081/api/entities/{entity_id}/properties"
            req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                properties = result.get('properties', {})
                return properties.get(key)

        except Exception as e:
            logger.error(f"[WorldAPI] Property get failed for {entity_id}.{key}: {e}")
            return None

    def _get_entity_transform(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Internal: Get entity transform via REST API."""
        try:
            import urllib.request
            import json

            url = f"http://localhost:8081/api/entities/{entity_id}/transform"
            req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                if 'error' in result:
                    return None
                return {
                    'position': result.get('position'),
                    'rotation': result.get('rotation'),
                    'scale': result.get('scale')
                }

        except Exception as e:
            logger.error(f"[WorldAPI] Transform get failed for {entity_id}: {e}")
            return None

    # =========================================================================
    # JavaScript-compatible dict for full state access
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert full state to JavaScript-compatible dict."""
        return {
            'myPosition': self.myPosition,
            'myFacing': self.myFacing,
            'myZone': self.myZone,
            'myZoneName': self.myZoneName,
            'myPosture': self.myPosture,
            'myAction': self.myAction,
            'myExpression': self.myExpression,
            'myGaze': self.myGaze,
            'affect': self.affect,
            'zoneExits': self.zoneExits,
            'ambientSounds': self.ambientSounds,
            'ambientMood': self.ambientMood,
            'conversationPartner': self.conversationPartner,
            'lastInput': self.lastInput,
            'perceivedEntities': self.perceivedEntities,
            'perceivedEvents': self.perceivedEvents,
            'cameraEnabled': self._camera_enabled,
            'physicsEnabled': self.physicsEnabled,
        }


# =============================================================================
# Global instance management
# =============================================================================

_world_apis: Dict[str, WorldAPI] = {}


def get_world_api(noodling_id: str = "default") -> WorldAPI:
    """
    Get or create WorldAPI for a noodling.

    Each noodling gets their own API instance with their own
    perception-filtered view.
    """
    global _world_apis
    if noodling_id not in _world_apis:
        _world_apis[noodling_id] = WorldAPI(noodling_id)
    return _world_apis[noodling_id]


def clear_world_apis():
    """Clear all WorldAPI instances."""
    global _world_apis
    _world_apis.clear()


__all__ = [
    "WorldAPI",
    "WorldAPIState",
    "PerceivedEntityJS",
    "PerceivedEventJS",
    "get_world_api",
    "clear_world_apis",
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
