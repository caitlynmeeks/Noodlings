"""
Semantic World System

Events are the atomic unit of reality. The universe is not made of things,
it is made of happenings.

This package provides:
    - Event: The fundamental primitive of existence
    - EventStore: The append-only log of all happenings
    - Projections: Stage, Situation, Experience derived from events
    - ContextBuilder: Generates agent context from events
    - ScenePacket: Complete scene state for renderers (Genie/Mirage)
    - PerceptionSlice: Filtered view for each entity's cognition
    - SceneStateManager: Canonical truth holder

The event log IS the world. Everything else is a view.
Genie is stateless. Noodlings is stateful.

Usage:
    from semantic_world import Event, EventType, speech_event, movement_event
    from semantic_world import EventStore
    from semantic_world import project_situation, build_agent_context
    from semantic_world import ScenePacket, SceneStateManager
    from semantic_world import generate_perception_slice

Author: Caitlyn + Claude
Date: December 2025
"""

from .event import (
    Event,
    EventType,
    Witness,
    Effect,
    SpatialContext,
    speech_event,
    movement_event,
    arrival_event,
    departure_event,
    action_event,
    perception_event,
    internal_event,
    environmental_event,
    social_event,
)

from .event_store import (
    EventStore,
    get_event_store,
    init_event_store,
)

from .projections import (
    Presence,
    SpatialRelation,
    Tension,
    Situation,
    Experience,
    project_situation,
    project_experience,
    project_narrative,
)

from .context_builder import (
    StageDefinition,
    AgentContext,
    ContextBuilder,
    get_context_builder,
    init_context_builder,
    build_agent_context,
)

# Scene Protocol - Noodlings Scene Protocol (NSP)
from .scene_packet import (
    # Enums
    PacketType,
    CameraMode,
    Framing,
    CameraAngle,
    CameraMovement,
    # Core types
    Vector3,
    Transform,
    PerceptionCone,
    Affect,
    VisualForm,
    ZoneBounds,
    Zone,
    ZoneConnection,
    Affordance,
    # Entities
    Noodling,
    Player,
    Prim,
    # Physics presets
    MATERIAL_PRESETS,
    # Narrative
    DialogueEntry,
    EventEntry,
    SceneState,
    NarrativeContext,
    # Camera
    CameraDirective,
    CameraStyle,
    # References
    ReferenceBundle,
    CharacterReference,
    # Packet
    PacketHeader,
    ScenePacket,
)

from .perception import (
    PerceivedEntity,
    PerceivedEvent,
    SpatialAwareness,
    PerceptionSlice,
    PerceptionCalculator,
    PerceptionSliceGenerator,
    get_perception_generator,
    generate_perception_slice,
)

from .scene_state_manager import (
    SceneStateManager,
    get_scene_state_manager,
    init_scene_state_manager,
)

from .scene_emitter import (
    EmitterConfig,
    RendererType,
    RendererConnection,
    ScenePacketEmitter,
    WebSocketPacketAdapter,
    GenieAdapter,
)

# Action Stream - High-frequency lightweight actions
from .action_stream import (
    ActionType,
    Action,
    ActionAck,
    ActionSession,
    ActionStreamHandler,
    WebSocketActionStream,
    get_action_handler,
    init_action_handler,
)

# SPE Bridge - Semantic Physics Engine integration
from .spe_bridge import (
    SPE_AVAILABLE,
    SPEBridge,
    PODCache,
    VERB_TO_INTERACTION,
    SpatialContext,
    SpatialResolver,
    get_spe_bridge,
    init_spe_bridge,
)

__all__ = [
    # Core classes
    "Event",
    "EventType",
    "Witness",
    "Effect",
    "SpatialContext",

    # Event Store
    "EventStore",
    "get_event_store",
    "init_event_store",

    # Factory functions
    "speech_event",
    "movement_event",
    "arrival_event",
    "departure_event",
    "action_event",
    "perception_event",
    "internal_event",
    "environmental_event",
    "social_event",

    # Projections
    "Presence",
    "SpatialRelation",
    "Tension",
    "Situation",
    "Experience",
    "project_situation",
    "project_experience",
    "project_narrative",

    # Context Builder
    "StageDefinition",
    "AgentContext",
    "ContextBuilder",
    "get_context_builder",
    "init_context_builder",
    "build_agent_context",

    # Scene Protocol - Enums
    "PacketType",
    "CameraMode",
    "Framing",
    "CameraAngle",
    "CameraMovement",

    # Scene Protocol - Core Types
    "Vector3",
    "Transform",
    "PerceptionCone",
    "Affect",
    "VisualForm",
    "ZoneBounds",
    "Zone",
    "ZoneConnection",
    "Affordance",

    # Scene Protocol - Entities
    "Noodling",
    "Player",
    "Prim",

    # Physics Presets
    "MATERIAL_PRESETS",

    # Scene Protocol - Narrative
    "DialogueEntry",
    "EventEntry",
    "SceneState",
    "NarrativeContext",

    # Scene Protocol - Camera
    "CameraDirective",
    "CameraStyle",

    # Scene Protocol - References
    "ReferenceBundle",
    "CharacterReference",

    # Scene Protocol - Packet
    "PacketHeader",
    "ScenePacket",

    # Perception System
    "PerceivedEntity",
    "PerceivedEvent",
    "SpatialAwareness",
    "PerceptionSlice",
    "PerceptionCalculator",
    "PerceptionSliceGenerator",
    "get_perception_generator",
    "generate_perception_slice",

    # Scene State Manager
    "SceneStateManager",
    "get_scene_state_manager",
    "init_scene_state_manager",

    # Scene Emitter
    "EmitterConfig",
    "RendererType",
    "RendererConnection",
    "ScenePacketEmitter",
    "WebSocketPacketAdapter",
    "GenieAdapter",

    # Action Stream
    "ActionType",
    "Action",
    "ActionAck",
    "ActionSession",
    "ActionStreamHandler",
    "WebSocketActionStream",
    "get_action_handler",
    "init_action_handler",

    # SPE Bridge
    "SPE_AVAILABLE",
    "SPEBridge",
    "PODCache",
    "VERB_TO_INTERACTION",
    "SpatialContext",
    "SpatialResolver",
    "get_spe_bridge",
    "init_spe_bridge",
]
