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

# Gaussian Adapter - Bridge to 3D Gaussian Splatting
from .gaussian_adapter import (
    GaussianAsset,
    GaussianInstance,
    GaussianScene,
    GaussianAssetManager,
    GaussianSceneCompositor,
    GaussianGenerator,
    init_gaussian_adapter,
    get_asset_manager,
    get_compositor,
    get_generator,
    compose_scene_from_packet,
)

# VRM Parser - Parse VRM avatars for Gaussian conversion
from .vrm_parser import (
    VRMAvatar,
    VRMMetadata,
    VRMParser,
    Skeleton,
    Bone,
    BlendShape,
    SpringBoneChain,
    SpringBoneCollider,
    SpringBoneSystem,
    MToonMaterial,
    Mesh,
    parse_vrm,
    vrm_to_gaussian_package,
    export_skeleton_json,
    export_skinning_json,
    export_spring_bones_json,
    export_blend_shapes_json,
)

# Spring Bone Simulation - Physics for hair/cloth
from .spring_bone_simulation import (
    SpringJoint,
    SpringChainState,
    ColliderState,
    SpringBoneSimulator,
    GaussianSpringDeformer,
    create_spring_simulation,
    create_gaussian_deformer,
)

# Mesh Import - Import arbitrary 3D meshes for Gaussian conversion
from .mesh_import import (
    MeshMaterial,
    MeshPrimitive,
    ImportedMesh,
    MeshImporter,
    GaussianConversionConfig,
    MeshToGaussianPipeline,
    import_mesh,
    mesh_to_gaussians,
)

# Network Bridge - Connect scene state to network layer
from .network_bridge import (
    noodling_to_network,
    player_to_network,
    prim_to_network,
    NetworkBridge,
    get_network_bridge,
    init_network_bridge,
)

# Radiance Format - Semantic Gaussian Splat file format
from .radiance_format import (
    RadianceAsset,
    RadianceBone,
    RadianceSkeleton,
    RadianceMetadata,
    SpringChain,
    SpringCollider,
    BodyRegion,
    BODY_REGION_NAMES,
    load_radiance,
    save_radiance,
    ply_to_radiance,
)

# Gaussian Collision Detection - Touch detection between entities
from .gaussian_collision import (
    TouchEvent,
    TouchType,
    TouchRegion,
    AffectImpulse,
    GaussianCollisionDetector,
    TouchAffectMapper,
    PhysicsEventBus,
    gaussian_overlap_integral,
    sphere_approximation_touch,
    build_covariance_matrix,
    init_collision_system,
    get_detector,
    get_affect_mapper,
    get_physics_event_bus,
    detect_and_emit_touches,
)

# Semantic Query - Click-to-inspect and CLIP search
from .semantic_query import (
    SplatHitInfo,
    SemanticSearchResult,
    SemanticMatch,
    CLIPEmbeddingIndex,
    SemanticQueryEngine,
    init_semantic_query_engine,
    get_semantic_query_engine,
    click_to_inspect,
    query_scene,
    ray_gaussian_intersection,
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

    # Gaussian Adapter
    "GaussianAsset",
    "GaussianInstance",
    "GaussianScene",
    "GaussianAssetManager",
    "GaussianSceneCompositor",
    "GaussianGenerator",
    "init_gaussian_adapter",
    "get_asset_manager",
    "get_compositor",
    "get_generator",
    "compose_scene_from_packet",

    # VRM Parser
    "VRMAvatar",
    "VRMMetadata",
    "VRMParser",
    "Skeleton",
    "Bone",
    "BlendShape",
    "SpringBoneChain",
    "SpringBoneCollider",
    "SpringBoneSystem",
    "MToonMaterial",
    "Mesh",
    "parse_vrm",
    "vrm_to_gaussian_package",
    "export_skeleton_json",
    "export_skinning_json",
    "export_spring_bones_json",
    "export_blend_shapes_json",

    # Spring Bone Simulation
    "SpringJoint",
    "SpringChainState",
    "ColliderState",
    "SpringBoneSimulator",
    "GaussianSpringDeformer",
    "create_spring_simulation",
    "create_gaussian_deformer",

    # Mesh Import
    "MeshMaterial",
    "MeshPrimitive",
    "ImportedMesh",
    "MeshImporter",
    "GaussianConversionConfig",
    "MeshToGaussianPipeline",
    "import_mesh",
    "mesh_to_gaussians",

    # Network Bridge
    "noodling_to_network",
    "player_to_network",
    "prim_to_network",
    "NetworkBridge",
    "get_network_bridge",
    "init_network_bridge",

    # Radiance Format
    "RadianceAsset",
    "RadianceBone",
    "RadianceSkeleton",
    "RadianceMetadata",
    "SpringChain",
    "SpringCollider",
    "BodyRegion",
    "BODY_REGION_NAMES",
    "load_radiance",
    "save_radiance",
    "ply_to_radiance",

    # Gaussian Collision Detection
    "TouchEvent",
    "TouchType",
    "TouchRegion",
    "AffectImpulse",
    "GaussianCollisionDetector",
    "TouchAffectMapper",
    "PhysicsEventBus",
    "gaussian_overlap_integral",
    "sphere_approximation_touch",
    "build_covariance_matrix",
    "init_collision_system",
    "get_detector",
    "get_affect_mapper",
    "get_physics_event_bus",
    "detect_and_emit_touches",

    # Semantic Query
    "SplatHitInfo",
    "SemanticSearchResult",
    "SemanticMatch",
    "CLIPEmbeddingIndex",
    "SemanticQueryEngine",
    "init_semantic_query_engine",
    "get_semantic_query_engine",
    "click_to_inspect",
    "query_scene",
    "ray_gaussian_intersection",
]
