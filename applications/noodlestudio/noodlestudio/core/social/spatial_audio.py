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
#   Spatial Audio System
#
#   3D positional audio for immersive social presence. Suppor...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.social.spatial_audio
# PURPOSE:  Spatial Audio System
# LAYER:    Studio / Social
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AudioSourceType, DistanceModel, PanningModel, AudioSource, AudioListener
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AudioSourceType(Enum):
    POINT = "point"          # 3D positioned sound
    AMBIENT = "ambient"      # Background audio for a zone
    VOICE = "voice"          # User voice chat
    UI = "ui"                # Non-spatial UI sounds


class DistanceModel(Enum):
    LINEAR = "linear"        # Linear falloff
    INVERSE = "inverse"      # 1/distance
    EXPONENTIAL = "exponential"  # Exponential decay


class PanningModel(Enum):
    HRTF = "HRTF"           # Head-related transfer function (realistic)
    EQUAL_POWER = "equalpower"  # Simple stereo panning


# =============================================================================
# Audio Source
# =============================================================================

@dataclass
class AudioSource:
    """
    A sound source in 3D space.

    Can be a point source (footsteps, ambient objects),
    voice chat from another user, or zone-based ambient audio.
    """
    id: str
    source_type: AudioSourceType = AudioSourceType.POINT
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # For doppler

    # Audio clip reference
    clip_url: Optional[str] = None
    is_looping: bool = False
    is_playing: bool = False

    # Volume and spatial
    volume: float = 1.0
    pitch: float = 1.0

    # Distance attenuation
    distance_model: DistanceModel = DistanceModel.INVERSE
    ref_distance: float = 1.0      # Distance at which volume is 1.0
    max_distance: float = 100.0    # Beyond this, no further attenuation
    rolloff_factor: float = 1.0    # How quickly sound fades with distance

    # Cone (directional audio)
    cone_inner_angle: float = 360.0  # Full volume within this angle
    cone_outer_angle: float = 360.0  # Attenuated outside inner, silent outside outer
    cone_outer_gain: float = 0.0     # Volume at outer angle

    # Occlusion
    is_occluded: bool = False
    occlusion_factor: float = 0.0   # 0 = clear, 1 = fully blocked

    def calculate_distance_gain(self, listener_position: np.ndarray) -> float:
        """Calculate volume based on distance from listener."""
        distance = np.linalg.norm(self.position - listener_position)

        if distance <= self.ref_distance:
            return 1.0

        if distance >= self.max_distance:
            if self.distance_model == DistanceModel.LINEAR:
                return 0.0
            # For inverse/exponential, use max_distance value
            distance = self.max_distance

        if self.distance_model == DistanceModel.LINEAR:
            return 1.0 - self.rolloff_factor * (distance - self.ref_distance) / (self.max_distance - self.ref_distance)
        elif self.distance_model == DistanceModel.INVERSE:
            return self.ref_distance / (self.ref_distance + self.rolloff_factor * (distance - self.ref_distance))
        else:  # EXPONENTIAL
            return (distance / self.ref_distance) ** (-self.rolloff_factor)

    def calculate_direction(self, listener_position: np.ndarray) -> np.ndarray:
        """Get direction from listener to source."""
        direction = self.position - listener_position
        length = np.linalg.norm(direction)
        if length < 0.001:
            return np.array([0, 0, 1])
        return direction / length


# =============================================================================
# Audio Listener
# =============================================================================

@dataclass
class AudioListener:
    """
    The listener (player's ears) in 3D space.

    Position and orientation determine how sounds are spatialized.
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    forward: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1]))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # For doppler

    @property
    def right(self) -> np.ndarray:
        """Get right direction."""
        return np.cross(self.forward, self.up)

    def world_to_local_direction(self, world_dir: np.ndarray) -> np.ndarray:
        """Convert world direction to listener's local space."""
        return np.array([
            np.dot(world_dir, self.right),
            np.dot(world_dir, self.up),
            np.dot(world_dir, self.forward),
        ])


# =============================================================================
# Ambient Zone
# =============================================================================

@dataclass
class AmbientZone:
    """
    A region with ambient audio (forest sounds, city noise, etc.).

    Fades in as you enter, fades out as you leave.
    """
    id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 10.0
    fade_distance: float = 2.0  # Distance over which audio fades in/out

    clip_url: Optional[str] = None
    base_volume: float = 0.5
    is_looping: bool = True

    def get_volume(self, listener_position: np.ndarray) -> float:
        """Get volume based on listener position in zone."""
        distance = np.linalg.norm(listener_position - self.position)

        if distance >= self.radius + self.fade_distance:
            return 0.0
        elif distance <= self.radius - self.fade_distance:
            return self.base_volume
        else:
            # Fade region
            if distance > self.radius:
                # Fading out
                fade = 1.0 - (distance - self.radius) / self.fade_distance
            else:
                # Fading in
                fade = (distance - (self.radius - self.fade_distance)) / self.fade_distance
            return self.base_volume * max(0.0, min(1.0, fade))


# =============================================================================
# Voice Channel
# =============================================================================

@dataclass
class VoiceChannel:
    """
    A voice chat participant.

    Voice audio is spatially positioned at the speaker's avatar location.
    """
    user_id: str
    display_name: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Voice state
    is_speaking: bool = False
    voice_volume: float = 1.0
    is_muted: bool = False

    # Spatial settings
    voice_range: float = 15.0  # Hearing distance in meters
    falloff_start: float = 5.0  # Distance at which falloff begins

    def get_voice_gain(self, listener_position: np.ndarray) -> float:
        """Get voice volume based on distance."""
        if self.is_muted:
            return 0.0

        distance = np.linalg.norm(self.position - listener_position)

        if distance <= self.falloff_start:
            return self.voice_volume
        elif distance >= self.voice_range:
            return 0.0
        else:
            # Linear falloff
            return self.voice_volume * (1.0 - (distance - self.falloff_start) / (self.voice_range - self.falloff_start))


# =============================================================================
# Spatial Audio Manager
# =============================================================================

class SpatialAudioManager:
    """
    Manages all spatial audio in a scene.

    Handles source positioning, distance attenuation, occlusion,
    and voice chat spatialization.
    """

    def __init__(self):
        self.listener = AudioListener()
        self.sources: Dict[str, AudioSource] = {}
        self.zones: Dict[str, AmbientZone] = {}
        self.voice_channels: Dict[str, VoiceChannel] = {}

        # Callbacks for audio engine integration
        self.on_source_update: Optional[Callable[[str, Dict], None]] = None
        self.on_voice_update: Optional[Callable[[str, Dict], None]] = None

        # Speed of sound for doppler (meters/second)
        self.speed_of_sound: float = 343.0

        # Occlusion raycast callback
        self.raycast_callback: Optional[Callable[[np.ndarray, np.ndarray], float]] = None

    def add_source(self, source: AudioSource):
        """Add an audio source to the scene."""
        self.sources[source.id] = source
        logger.debug(f"Added audio source: {source.id}")

    def remove_source(self, source_id: str):
        """Remove an audio source."""
        if source_id in self.sources:
            del self.sources[source_id]
            logger.debug(f"Removed audio source: {source_id}")

    def add_zone(self, zone: AmbientZone):
        """Add an ambient zone."""
        self.zones[zone.id] = zone

    def add_voice_channel(self, channel: VoiceChannel):
        """Add a voice chat participant."""
        self.voice_channels[channel.user_id] = channel

    def update_listener(self, position: np.ndarray, forward: np.ndarray, up: np.ndarray):
        """Update listener position and orientation."""
        self.listener.position = position
        self.listener.forward = forward
        self.listener.up = up

    def update_source_position(self, source_id: str, position: np.ndarray, velocity: np.ndarray = None):
        """Update a source's position."""
        if source_id in self.sources:
            self.sources[source_id].position = position
            if velocity is not None:
                self.sources[source_id].velocity = velocity

    def update_voice_position(self, user_id: str, position: np.ndarray):
        """Update a voice channel's position."""
        if user_id in self.voice_channels:
            self.voice_channels[user_id].position = position

    def update(self):
        """
        Update all audio sources with current spatial parameters.

        Call this each frame to update spatialization.
        """
        # Update point sources
        for source_id, source in self.sources.items():
            if source.source_type == AudioSourceType.UI:
                continue  # UI sounds are non-spatial

            params = self._calculate_source_params(source)
            if self.on_source_update:
                self.on_source_update(source_id, params)

        # Update voice channels
        for user_id, channel in self.voice_channels.items():
            params = self._calculate_voice_params(channel)
            if self.on_voice_update:
                self.on_voice_update(user_id, params)

    def _calculate_source_params(self, source: AudioSource) -> Dict:
        """Calculate spatial audio parameters for a source."""
        # Distance-based gain
        distance_gain = source.calculate_distance_gain(self.listener.position)

        # Occlusion
        occlusion_gain = 1.0
        if self.raycast_callback:
            occlusion = self.raycast_callback(self.listener.position, source.position)
            occlusion_gain = 1.0 - occlusion * 0.8  # Max 80% reduction

        # Direction (for panning)
        world_direction = source.calculate_direction(self.listener.position)
        local_direction = self.listener.world_to_local_direction(world_direction)

        # Doppler effect
        doppler = self._calculate_doppler(source)

        return {
            "gain": source.volume * distance_gain * occlusion_gain,
            "direction": local_direction.tolist(),
            "doppler_pitch": doppler,
            "distance": np.linalg.norm(source.position - self.listener.position),
        }

    def _calculate_voice_params(self, channel: VoiceChannel) -> Dict:
        """Calculate spatial parameters for voice chat."""
        gain = channel.get_voice_gain(self.listener.position)

        world_direction = channel.position - self.listener.position
        length = np.linalg.norm(world_direction)
        if length > 0.001:
            world_direction = world_direction / length
        else:
            world_direction = np.array([0, 0, 1])

        local_direction = self.listener.world_to_local_direction(world_direction)

        return {
            "gain": gain,
            "direction": local_direction.tolist(),
            "distance": length,
            "is_speaking": channel.is_speaking,
        }

    def _calculate_doppler(self, source: AudioSource) -> float:
        """Calculate doppler pitch shift."""
        if np.linalg.norm(source.velocity) < 0.01 and np.linalg.norm(self.listener.velocity) < 0.01:
            return 1.0  # No doppler if stationary

        # Direction from listener to source
        direction = source.position - self.listener.position
        distance = np.linalg.norm(direction)
        if distance < 0.001:
            return 1.0
        direction = direction / distance

        # Relative velocity along the line between them
        source_velocity = np.dot(source.velocity, direction)
        listener_velocity = np.dot(self.listener.velocity, direction)

        # Doppler formula
        denominator = self.speed_of_sound + source_velocity
        if abs(denominator) < 0.01:
            return 1.0

        return (self.speed_of_sound + listener_velocity) / denominator

    def get_active_zones(self) -> List[Tuple[str, float]]:
        """Get all ambient zones with non-zero volume at listener position."""
        active = []
        for zone_id, zone in self.zones.items():
            volume = zone.get_volume(self.listener.position)
            if volume > 0.01:
                active.append((zone_id, volume))
        return active


# =============================================================================
# Factory Functions
# =============================================================================

def create_point_source(
    id: str,
    position: Tuple[float, float, float],
    clip_url: str,
    volume: float = 1.0,
    looping: bool = False,
    max_distance: float = 50.0,
) -> AudioSource:
    """Create a point audio source."""
    return AudioSource(
        id=id,
        source_type=AudioSourceType.POINT,
        position=np.array(position, dtype=np.float32),
        clip_url=clip_url,
        volume=volume,
        is_looping=looping,
        max_distance=max_distance,
    )


def create_ambient_zone(
    id: str,
    position: Tuple[float, float, float],
    radius: float,
    clip_url: str,
    volume: float = 0.5,
) -> AmbientZone:
    """Create an ambient audio zone."""
    return AmbientZone(
        id=id,
        position=np.array(position, dtype=np.float32),
        radius=radius,
        clip_url=clip_url,
        base_volume=volume,
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    print("Spatial Audio Test")
    print("=" * 40)

    manager = SpatialAudioManager()

    # Add a point source
    campfire = create_point_source(
        id="campfire",
        position=(5, 0, 5),
        clip_url="sounds/fire.ogg",
        volume=0.8,
        looping=True,
        max_distance=20.0,
    )
    manager.add_source(campfire)

    # Add ambient zone
    forest = create_ambient_zone(
        id="forest_ambient",
        position=(0, 0, 0),
        radius=50.0,
        clip_url="sounds/forest.ogg",
        volume=0.3,
    )
    manager.add_zone(forest)

    # Add voice channel
    other_player = VoiceChannel(
        user_id="player_123",
        display_name="Alice",
        position=np.array([3, 0, 0]),
        is_speaking=True,
    )
    manager.add_voice_channel(other_player)

    # Simulate listener movement
    print("\nListener moving toward campfire...")
    for z in [0, 2, 4, 6, 8, 10]:
        manager.update_listener(
            position=np.array([0, 0, float(z)]),
            forward=np.array([0, 0, 1]),
            up=np.array([0, 1, 0]),
        )

        params = manager._calculate_source_params(campfire)
        voice_params = manager._calculate_voice_params(other_player)
        zones = manager.get_active_zones()

        print(f"  Z={z}: campfire_gain={params['gain']:.2f}, voice_gain={voice_params['gain']:.2f}, zones={len(zones)}")

    print("\nTest complete!")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
