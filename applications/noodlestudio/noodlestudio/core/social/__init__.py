"""
Social Features Module

VRChat-killer features for the Gaussian World Engine:
- Mirrors (the VRChat obsession)
- Portals (Portal-game style teleportation)
- Spatial Audio (3D positioned sound + voice chat)
- Gaussian Particles (fire, smoke, sparkles, snow)

Key Insight: Gaussians make mirrors and portals trivial.
Just render from a different camera - no stencil buffers needed!

Author: Caitlyn + Claude
Date: December 2025
"""

# Mirror and Portal System
from .mirror_portal_system import (
    # Math utilities
    normalize,
    reflect_vector,
    reflect_point,
    quaternion_from_axis_angle,
    quaternion_multiply,
    rotate_vector,

    # Camera
    Camera,

    # Mirrors
    MirrorType,
    MirrorSurface,
    create_mirror,

    # Portals
    Portal,
    create_portal_pair,

    # Manager
    MirrorPortalManager,
)

# Spatial Audio
from .spatial_audio import (
    # Types
    AudioSourceType,
    DistanceModel,
    PanningModel,

    # Sources
    AudioSource,
    AudioListener,
    AmbientZone,
    VoiceChannel,

    # Manager
    SpatialAudioManager,

    # Factory
    create_point_source,
    create_ambient_zone,
)

# Gaussian Particles
from .gaussian_particles import (
    # Types
    EmitterShape,
    BlendMode,
    ColorGradient,
    Curve,

    # Particle
    GaussianParticle,
    ParticleEmitter,
    ParticleSystem,

    # Presets
    create_fire_emitter,
    create_smoke_emitter,
    create_sparkle_emitter,
    create_snow_emitter,
)

# Network Synchronization
from .network_sync import (
    # Math utilities
    lerp,
    lerp_vec3,
    slerp,

    # Snapshots and Interpolation
    EntitySnapshot,
    InterpolationBuffer,

    # Messages
    MessageType,
    NetworkMessage,

    # Lobby System
    LobbyPlayer,
    Lobby,

    # Client
    NetworkClient,

    # Voice Chat
    VoiceState,
    VoiceChannel,
    VoiceManager,

    # Interest Management
    InterestArea,
    InterestManager,

    # Delta Compression
    DeltaCompressor,

    # Server
    NetworkServer,
    get_network_server,
    init_network_server,
)

__all__ = [
    # Mirror/Portal
    "Camera",
    "MirrorType",
    "MirrorSurface",
    "Portal",
    "MirrorPortalManager",
    "create_mirror",
    "create_portal_pair",

    # Audio
    "AudioSourceType",
    "DistanceModel",
    "PanningModel",
    "AudioSource",
    "AudioListener",
    "AmbientZone",
    "VoiceChannel",
    "SpatialAudioManager",
    "create_point_source",
    "create_ambient_zone",

    # Particles
    "EmitterShape",
    "BlendMode",
    "ColorGradient",
    "Curve",
    "GaussianParticle",
    "ParticleEmitter",
    "ParticleSystem",
    "create_fire_emitter",
    "create_smoke_emitter",
    "create_sparkle_emitter",
    "create_snow_emitter",

    # Network Sync
    "lerp",
    "lerp_vec3",
    "slerp",
    "EntitySnapshot",
    "InterpolationBuffer",
    "MessageType",
    "NetworkMessage",
    "LobbyPlayer",
    "Lobby",
    "NetworkClient",
    "VoiceState",
    "VoiceChannel",
    "VoiceManager",
    "InterestArea",
    "InterestManager",
    "DeltaCompressor",
    "NetworkServer",
    "get_network_server",
    "init_network_server",
]
