"""
Scene State Manager - The Canonical Truth Holder

The SceneStateManager maintains the authoritative state of the world.
It is the single source of truth from which:
    - Perception slices are generated (for noodling cognition)
    - Scene packets are emitted (for Genie/Mirage rendering)
    - MUD text is rendered (for text interface)

Events flow IN (from user actions, noodling outputs, world simulation).
State is maintained.
Packets flow OUT (to renderers, to facet assemblies).

          ┌─────────────────────────────────────────────┐
          │           SCENE STATE MANAGER               │
          │        (canonical truth holder)             │
          └─────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
    Red's Slice         Yuki's Slice         Full Packet
    (cognition)         (cognition)          (rendering)

Author: Caitlyn + Claude
Date: December 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
import yaml
import json
import time

from .scene_packet import (
    ScenePacket,
    PacketHeader,
    PacketType,
    SpatialTruth,
    AmbientState,
    Zone,
    Vector3,
    Noodling,
    Player,
    Prim,
    PerceptionCone,
    Affect,
    VisualForm,
    Affordance,
    NarrativeContext,
    DialogueEntry,
    EventEntry,
    SceneState,
    CameraDirective,
    CameraMode,
    Framing,
    ReferenceBundle,
    CharacterReference,
)

from .perception import (
    PerceptionSlice,
    PerceptionSliceGenerator,
    generate_perception_slice,
)

from .event import Event, EventType
from .event_store import EventStore


# =============================================================================
# Scene State Manager
# =============================================================================

class SceneStateManager:
    """
    The canonical truth holder for scene state.

    Maintains:
        - Current positions of all entities
        - Current states (affect, expression, action)
        - Spatial truth (zones, ambient)
        - Recent narrative events

    Provides:
        - Full scene packets (for Genie)
        - Perception slices (for each noodling)
        - MUD text rendering
    """

    def __init__(
        self,
        stage_id: str = "default",
        stage_name: str = "The Stage",
        event_store: Optional[EventStore] = None
    ):
        """
        Initialize the scene state manager.

        Args:
            stage_id: Unique identifier for this stage
            stage_name: Display name
            event_store: Optional event store for historical events
        """
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.event_store = event_store

        # Spatial truth
        self.spatial_truth = SpatialTruth()
        self.zones: Dict[str, Zone] = {}

        # Entities
        self.noodlings: Dict[str, Noodling] = {}
        self.players: Dict[str, Player] = {}
        self.prims: Dict[str, Prim] = {}

        # Narrative context (recent events for packet)
        self.recent_dialogue: List[DialogueEntry] = []
        self.recent_events: List[EventEntry] = []
        self.max_recent_items = 20

        # Scene state
        self.scene_state = SceneState()

        # Camera
        self.camera_directive = CameraDirective()

        # Reference assets (loaded from noodling definitions)
        self.character_refs: Dict[str, CharacterReference] = {}

        # Perception slice generator
        self.perception_generator = PerceptionSliceGenerator()

        # Event callbacks
        self._on_state_change: List[Callable[['SceneStateManager'], None]] = []

        # Packet tracking
        self._last_packet_id: Optional[str] = None

    # =========================================================================
    # Stage/Zone Management
    # =========================================================================

    def load_stage_from_yaml(self, stage_path: str):
        """Load stage definition from YAML file."""
        stage_yaml = Path(stage_path) / "stage.yaml"
        if stage_yaml.exists():
            with open(stage_yaml, 'r') as f:
                data = yaml.safe_load(f) or {}

            self.stage_id = data.get('id', self.stage_id)
            self.stage_name = data.get('name', self.stage_name)

            # Load world properties
            world = data.get('world', {})
            bounds = world.get('bounds', {})
            if bounds:
                self.spatial_truth.bounds_min = Vector3.from_list(bounds.get('min', [-100, 0, -100]))
                self.spatial_truth.bounds_max = Vector3.from_list(bounds.get('max', [100, 50, 100]))

            ambient = world.get('ambient', {})
            self.spatial_truth.ambient = AmbientState(
                time_of_day=ambient.get('time_of_day', 'day'),
                weather=ambient.get('weather', 'clear'),
                soundscape=ambient.get('soundscape', []),
            )

        # Load zones
        zones_dir = Path(stage_path) / "Zones"
        if zones_dir.exists():
            for zone_file in zones_dir.glob("*.zone.yaml"):
                self.load_zone_from_yaml(str(zone_file))

    def load_zone_from_yaml(self, zone_path: str):
        """Load a zone definition from YAML."""
        with open(zone_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        zone_id = data.get('id', Path(zone_path).stem.replace('.zone', ''))

        spatial = data.get('spatial', {})
        text = data.get('text', {})

        zone = Zone(
            id=zone_id,
            name=data.get('name', zone_id),
            center=Vector3.from_list(spatial.get('center', [0, 0, 0])),
            radius=spatial.get('radius', 15.0),
            falloff=spatial.get('falloff', 10.0),
            shape=spatial.get('shape', 'sphere'),
            description=text.get('description', ''),
            features=text.get('features', []),
            exits=text.get('exits', {}),
            mood=data.get('perception', {}).get('mood', 'neutral'),
            lighting=data.get('perception', {}).get('lighting', 'ambient'),
        )

        self.zones[zone_id] = zone
        self.spatial_truth.zones.append(zone)

    def add_zone(self, zone: Zone):
        """Add a zone to the stage."""
        self.zones[zone.id] = zone
        if zone not in self.spatial_truth.zones:
            self.spatial_truth.zones.append(zone)
        self._notify_state_change()

    # =========================================================================
    # Entity Management
    # =========================================================================

    def add_noodling(
        self,
        noodling_id: str,
        display_name: str,
        position: List[float] = None,
        rotation: List[float] = None,
        **kwargs
    ) -> Noodling:
        """Add a noodling to the scene."""
        from .scene_packet import Transform
        transform = Transform(
            position=Vector3.from_list(position or [0, 0, 0]),
            rotation=Vector3.from_list(rotation or [0, 0, 0]),
        )
        noodling = Noodling(
            id=noodling_id,
            display_name=display_name,
            transform=transform,
            **kwargs
        )
        self.noodlings[noodling_id] = noodling
        self._notify_state_change()
        return noodling

    def add_player(
        self,
        player_id: str,
        display_name: str,
        position: List[float] = None,
        rotation: List[float] = None,
        **kwargs
    ) -> Player:
        """Add a player to the scene."""
        from .scene_packet import Transform
        transform = Transform(
            position=Vector3.from_list(position or [0, 0, 0]),
            rotation=Vector3.from_list(rotation or [0, 0, 0]),
        )
        player = Player(
            id=player_id,
            display_name=display_name,
            transform=transform,
            **kwargs
        )
        self.players[player_id] = player
        self._notify_state_change()
        return player

    def add_prim(
        self,
        prim_id: str,
        prim_type: str,
        position: List[float] = None,
        rotation: List[float] = None,
        scale: List[float] = None,
        **kwargs
    ) -> Prim:
        """Add a prim to the scene."""
        from .scene_packet import Transform
        transform = Transform(
            position=Vector3.from_list(position or [0, 0, 0]),
            rotation=Vector3.from_list(rotation or [0, 0, 0]),
            scale=Vector3.from_list(scale or [1, 1, 1]),
        )
        prim = Prim(
            id=prim_id,
            prim_type=prim_type,
            transform=transform,
            **kwargs
        )
        self.prims[prim_id] = prim
        self._notify_state_change()
        return prim

    def remove_entity(self, entity_id: str):
        """Remove an entity from the scene."""
        if entity_id in self.noodlings:
            del self.noodlings[entity_id]
        elif entity_id in self.players:
            del self.players[entity_id]
        elif entity_id in self.prims:
            del self.prims[entity_id]
        self._notify_state_change()

    def get_entity(self, entity_id: str) -> Optional[Any]:
        """Get an entity by ID."""
        return (
            self.noodlings.get(entity_id) or
            self.players.get(entity_id) or
            self.prims.get(entity_id)
        )

    # =========================================================================
    # State Updates
    # =========================================================================

    def update_noodling_position(
        self,
        noodling_id: str,
        position: List[float],
        facing: List[float] = None
    ):
        """Update a noodling's position."""
        if noodling_id in self.noodlings:
            self.noodlings[noodling_id].position = Vector3.from_list(position)
            if facing:
                self.noodlings[noodling_id].facing = Vector3.from_list(facing)
            # Update zone
            self.noodlings[noodling_id].zone = self._determine_zone(position)
            self._notify_state_change()

    def update_noodling_affect(
        self,
        noodling_id: str,
        affect: Dict[str, float]
    ):
        """Update a noodling's affect state."""
        if noodling_id in self.noodlings:
            self.noodlings[noodling_id].affect = Affect.from_dict(affect)
            self._notify_state_change()

    def update_noodling_expression(
        self,
        noodling_id: str,
        expression: str,
        posture: str = None,
        action: str = None
    ):
        """Update a noodling's visible state."""
        if noodling_id in self.noodlings:
            self.noodlings[noodling_id].expression = expression
            if posture:
                self.noodlings[noodling_id].posture = posture
            if action:
                self.noodlings[noodling_id].current_action = action
            self._notify_state_change()

    def update_noodling_gaze(
        self,
        noodling_id: str,
        gaze_target: Optional[str]
    ):
        """Update what a noodling is looking at."""
        if noodling_id in self.noodlings:
            self.noodlings[noodling_id].gaze_target = gaze_target
            self._notify_state_change()

    def update_noodling_visual_state(
        self,
        noodling_id: str,
        visual_state: str
    ):
        """Change a noodling's visual form (e.g., Yuki: ghost -> humanoid)."""
        if noodling_id in self.noodlings:
            self.noodlings[noodling_id].visual_state = visual_state
            self._notify_state_change()

    def update_prim_state(
        self,
        prim_id: str,
        state_updates: Dict[str, Any]
    ):
        """Update a prim's state."""
        if prim_id in self.prims:
            self.prims[prim_id].state.update(state_updates)
            self._notify_state_change()

    def _determine_zone(self, position: List[float]) -> str:
        """Determine which zone a position is in."""
        pos = Vector3.from_list(position)
        best_zone = ""
        best_strength = 0.0

        for zone in self.spatial_truth.zones:
            strength = zone.perception_strength(pos)
            if strength > best_strength:
                best_strength = strength
                best_zone = zone.id

        return best_zone

    # =========================================================================
    # Narrative Events
    # =========================================================================

    def record_dialogue(
        self,
        speaker: str,
        text: str,
        tone: str = "neutral",
        entry_type: str = "speech"
    ):
        """Record a piece of dialogue."""
        now = time.time()
        entry = DialogueEntry(
            speaker=speaker,
            text=text,
            timestamp=now,
            seconds_ago=0.0,
            tone=tone,
            entry_type=entry_type,
        )
        self.recent_dialogue.insert(0, entry)
        self._trim_recent_items()
        self._notify_state_change()

        # Also record to event store if available
        if self.event_store:
            from .event import speech_event
            event = speech_event(
                speaker=speaker,
                content=text,
                stage_id=self.stage_id,
                manner=tone,
            )
            self.event_store.append(event)

    def record_action(
        self,
        actor: str,
        action: str,
        target: Optional[str] = None,
        detail: Optional[str] = None
    ):
        """Record an action event."""
        now = time.time()
        entry = EventEntry(
            actor=actor,
            action=action,
            timestamp=now,
            target=target,
            detail=detail,
        )
        self.recent_events.insert(0, entry)
        self._trim_recent_items()
        self._notify_state_change()

    def _trim_recent_items(self):
        """Trim recent items to max size."""
        if len(self.recent_dialogue) > self.max_recent_items:
            self.recent_dialogue = self.recent_dialogue[:self.max_recent_items]
        if len(self.recent_events) > self.max_recent_items:
            self.recent_events = self.recent_events[:self.max_recent_items]

    def update_scene_state(
        self,
        tension: float = None,
        energy: float = None,
        intimacy: float = None,
        humor: float = None,
        mystery: float = None,
        current_beat: str = None
    ):
        """Update the high-level scene state."""
        if tension is not None:
            self.scene_state.tension = tension
        if energy is not None:
            self.scene_state.energy = energy
        if intimacy is not None:
            self.scene_state.intimacy = intimacy
        if humor is not None:
            self.scene_state.humor = humor
        if mystery is not None:
            self.scene_state.mystery = mystery
        if current_beat is not None:
            self.scene_state.current_beat = current_beat
        self._notify_state_change()

    # =========================================================================
    # Camera Control
    # =========================================================================

    def set_camera_focus(
        self,
        subject: str,
        framing: str = "medium",
        mode: str = "FOCUS_ON"
    ):
        """Set camera to focus on a subject."""
        self.camera_directive.mode = CameraMode[mode]
        self.camera_directive.subject = subject
        self.camera_directive.framing = Framing[framing.upper()]
        self._notify_state_change()

    def set_camera_two_shot(
        self,
        subject_a: str,
        subject_b: str,
        framing: str = "medium"
    ):
        """Set camera for a two-shot."""
        self.camera_directive.mode = CameraMode.TWO_SHOT
        self.camera_directive.subjects = [subject_a, subject_b]
        self.camera_directive.framing = Framing[framing.upper()]
        self._notify_state_change()

    def set_camera_pov(self, subject: str):
        """Set camera to subject's POV."""
        self.camera_directive.mode = CameraMode.POV
        self.camera_directive.subject = subject
        self._notify_state_change()

    def set_camera_establish(self, zone: str):
        """Set camera for establishing shot of a zone."""
        self.camera_directive.mode = CameraMode.ESTABLISH
        self.camera_directive.zone = zone
        self.camera_directive.framing = Framing.WIDE
        self._notify_state_change()

    # =========================================================================
    # Packet Generation
    # =========================================================================

    def generate_scene_packet(
        self,
        packet_type: PacketType = PacketType.FULL
    ) -> ScenePacket:
        """
        Generate a complete scene packet.

        This is what gets sent to Genie/Mirage for rendering.
        """
        now = time.time()

        # Update seconds_ago for recent items
        for d in self.recent_dialogue:
            d.seconds_ago = now - d.timestamp
        for e in self.recent_events:
            e.seconds_ago = now - e.timestamp

        # Build narrative context
        narrative_context = NarrativeContext(
            recent_dialogue=self.recent_dialogue.copy(),
            recent_events=self.recent_events.copy(),
            scene_state=self.scene_state,
            context_summary=self._generate_context_summary(),
        )

        # Build reference bundle
        reference_bundle = self._build_reference_bundle()

        # Create packet
        packet = ScenePacket(
            header=PacketHeader(
                stage_id=self.stage_id,
                stage_name=self.stage_name,
                packet_type=packet_type,
                previous_packet_id=self._last_packet_id,
            ),
            spatial_truth=self.spatial_truth,
            noodlings=self.noodlings.copy(),
            players=self.players.copy(),
            prims=self.prims.copy(),
            reference_bundle=reference_bundle,
            narrative_context=narrative_context,
            camera_directive=self.camera_directive,
        )

        self._last_packet_id = packet.header.packet_id
        return packet

    def generate_perception_slice(self, perceiver_id: str) -> PerceptionSlice:
        """
        Generate a perception slice for a specific entity.

        This is what gets fed to a noodling's facet assembly.
        """
        packet = self.generate_scene_packet()
        return self.perception_generator.generate(packet, perceiver_id)

    def _generate_context_summary(self) -> str:
        """Generate an LLM-readable context summary."""
        lines = []

        # Who's here
        names = [n.display_name for n in self.noodlings.values()]
        names.extend([p.display_name for p in self.players.values()])
        if names:
            lines.append(f"Present: {', '.join(names)}")

        # Recent activity
        if self.recent_dialogue:
            last = self.recent_dialogue[0]
            lines.append(f"Last spoke: {last.speaker} said \"{last.text[:50]}...\"")

        # Mood
        lines.append(f"Scene: {self.scene_state.current_beat.replace('_', ' ')}")

        return " ".join(lines)

    def _build_reference_bundle(self) -> ReferenceBundle:
        """Build the reference bundle for visual assets."""
        bundle = ReferenceBundle()

        for nid, noodling in self.noodlings.items():
            form = noodling.get_current_form()
            bundle.characters[nid] = CharacterReference(
                form=noodling.visual_state,
                primary_ref=noodling.get_reference_image(),
                expression_ref=noodling.get_reference_image(noodling.expression),
                description=form.description if form else "",
            )

        return bundle

    # =========================================================================
    # Event Callbacks
    # =========================================================================

    def on_state_change(self, callback: Callable[['SceneStateManager'], None]):
        """Register a callback for state changes."""
        self._on_state_change.append(callback)

    def _notify_state_change(self):
        """Notify all registered callbacks of state change."""
        for callback in self._on_state_change:
            try:
                callback(self)
            except Exception as e:
                print(f"Error in state change callback: {e}")

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current state to dictionary."""
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "noodlings": {
                nid: {
                    "id": n.id,
                    "display_name": n.display_name,
                    "position": n.position.to_list(),
                    "zone": n.zone,
                    "visual_state": n.visual_state,
                    "expression": n.expression,
                    "posture": n.posture,
                    "affect": n.affect.to_dict(),
                }
                for nid, n in self.noodlings.items()
            },
            "players": {
                pid: {
                    "id": p.id,
                    "display_name": p.display_name,
                    "position": p.position.to_list(),
                    "zone": p.zone,
                }
                for pid, p in self.players.items()
            },
            "scene_state": {
                "tension": self.scene_state.tension,
                "energy": self.scene_state.energy,
                "current_beat": self.scene_state.current_beat,
            },
        }

    def save_state(self, path: str):
        """Save current state to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================

_global_manager: Optional[SceneStateManager] = None


def get_scene_state_manager() -> Optional[SceneStateManager]:
    """Get the global scene state manager if initialized."""
    return _global_manager


def init_scene_state_manager(
    stage_id: str = "default",
    stage_name: str = "The Stage",
    event_store: Optional[EventStore] = None
) -> SceneStateManager:
    """Initialize the global scene state manager."""
    global _global_manager
    _global_manager = SceneStateManager(stage_id, stage_name, event_store)
    return _global_manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SceneStateManager",
    "get_scene_state_manager",
    "init_scene_state_manager",
]
