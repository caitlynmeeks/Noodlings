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
#   Network Synchronization System
#
#   Multi-user state synchronization for social presence. Han...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.social.network_sync
# PURPOSE:  Network Synchronization System
# LAYER:    Studio / Social
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   EntitySnapshot, InterpolationBuffer, MessageType, NetworkMessage, LobbyPlayer
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple, Any
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Math Utilities
# =============================================================================

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def lerp_vec3(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation for vectors."""
    return a + (b - a) * t


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation for quaternions."""
    dot = np.dot(q1, q2)

    # If dot < 0, negate one quaternion to take shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q2_perp = q2 - q1 * dot
    q2_perp = q2_perp / np.linalg.norm(q2_perp)

    return q1 * np.cos(theta) + q2_perp * np.sin(theta)


# =============================================================================
# Snapshot & Interpolation
# =============================================================================

@dataclass
class EntitySnapshot:
    """A single state snapshot for an entity."""
    timestamp: float
    position: np.ndarray
    rotation: np.ndarray  # Quaternion (x, y, z, w)
    velocity: np.ndarray

    # Optional additional state
    animation_state: Optional[str] = None
    blend_shapes: Optional[Dict[str, float]] = None
    audio_emitters: Optional[List[Dict]] = None


@dataclass
class InterpolationBuffer:
    """
    Buffer for interpolating remote entity state.

    Uses a delay buffer to smooth network jitter.
    """
    entity_id: str
    snapshots: deque = field(default_factory=lambda: deque(maxlen=30))

    # Interpolation settings
    interp_delay: float = 0.1  # 100ms buffer
    max_extrapolation: float = 0.2  # Max 200ms prediction

    # Current interpolated state
    current_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    current_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def add_snapshot(self, snapshot: EntitySnapshot):
        """Add a new snapshot from the network."""
        self.snapshots.append(snapshot)

    def get_interpolated(self, render_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get interpolated position and rotation at render_time.

        render_time should be current_time - interp_delay
        """
        if len(self.snapshots) < 2:
            if self.snapshots:
                s = self.snapshots[-1]
                return s.position, s.rotation
            return self.current_position, self.current_rotation

        target_time = render_time - self.interp_delay

        # Find surrounding snapshots
        before: Optional[EntitySnapshot] = None
        after: Optional[EntitySnapshot] = None

        for snapshot in self.snapshots:
            if snapshot.timestamp <= target_time:
                before = snapshot
            elif after is None:
                after = snapshot
                break

        # Only have past data - extrapolate
        if after is None and before is not None:
            dt = target_time - before.timestamp
            if dt > self.max_extrapolation:
                dt = self.max_extrapolation

            # Simple linear extrapolation
            position = before.position + before.velocity * dt
            rotation = before.rotation  # Don't extrapolate rotation

            self.current_position = position
            self.current_rotation = rotation
            return position, rotation

        # Only have future data - snap to earliest
        if before is None and after is not None:
            self.current_position = after.position
            self.current_rotation = after.rotation
            return after.position, after.rotation

        # Have both - interpolate
        if before and after:
            t = (target_time - before.timestamp) / (after.timestamp - before.timestamp)
            t = max(0, min(1, t))

            position = lerp_vec3(before.position, after.position, t)
            rotation = slerp(before.rotation, after.rotation, t)

            self.current_position = position
            self.current_rotation = rotation
            return position, rotation

        return self.current_position, self.current_rotation

    def get_animation_state(self) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """Get latest animation state (not interpolated)."""
        if self.snapshots:
            latest = self.snapshots[-1]
            return latest.animation_state, latest.blend_shapes
        return None, None


# =============================================================================
# Network Messages
# =============================================================================

class MessageType(Enum):
    # Client -> Server
    JOIN_LOBBY = "join_lobby"
    LEAVE_LOBBY = "leave_lobby"
    PLAYER_INPUT = "player_input"
    VOICE_READY = "voice_ready"
    CHAT_MESSAGE = "chat_message"

    # Server -> Client
    LOBBY_STATE = "lobby_state"
    ENTITY_UPDATE = "entity_update"
    ENTITY_SPAWN = "entity_spawn"
    ENTITY_DESPAWN = "entity_despawn"
    PLAYER_JOINED = "player_joined"
    PLAYER_LEFT = "player_left"
    VOICE_OFFER = "voice_offer"
    CHAT_BROADCAST = "chat_broadcast"

    # Bidirectional
    PING = "ping"
    PONG = "pong"


@dataclass
class NetworkMessage:
    """A network message."""
    type: MessageType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, data: str) -> 'NetworkMessage':
        obj = json.loads(data)
        return cls(
            type=MessageType(obj["type"]),
            payload=obj["payload"],
            timestamp=obj.get("timestamp", time.time()),
        )


# =============================================================================
# Lobby System
# =============================================================================

@dataclass
class LobbyPlayer:
    """A player in a lobby."""
    user_id: str
    display_name: str
    avatar_id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    is_speaking: bool = False
    is_muted: bool = False

    # Network state
    last_update: float = 0.0
    latency_ms: float = 0.0


@dataclass
class Lobby:
    """A multiplayer lobby/room."""
    lobby_id: str
    name: str
    stage_id: str  # Which stage this lobby is for

    # Players
    players: Dict[str, LobbyPlayer] = field(default_factory=dict)
    max_players: int = 32

    # State
    is_public: bool = True
    created_at: float = field(default_factory=time.time)

    @property
    def player_count(self) -> int:
        return len(self.players)

    @property
    def is_full(self) -> bool:
        return self.player_count >= self.max_players


# =============================================================================
# Network Client
# =============================================================================

class NetworkClient:
    """
    Client-side networking for multiplayer.

    Handles:
    - WebSocket connection to server
    - Entity state interpolation
    - Player input sending
    - Lobby management
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.websocket = None
        self.connected = False

        # Local player
        self.local_player_id: Optional[str] = None
        self.local_position = np.zeros(3)
        self.local_rotation = np.array([0, 0, 0, 1])

        # Remote entities (interpolated)
        self.remote_entities: Dict[str, InterpolationBuffer] = {}

        # Lobby state
        self.current_lobby: Optional[Lobby] = None

        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_player_joined: Optional[Callable[[LobbyPlayer], None]] = None
        self.on_player_left: Optional[Callable[[str], None]] = None
        self.on_entity_spawned: Optional[Callable[[str, Dict], None]] = None
        self.on_entity_despawned: Optional[Callable[[str], None]] = None
        self.on_chat_message: Optional[Callable[[str, str, str], None]] = None

        # Network stats
        self.latency_ms: float = 0.0
        self.last_ping_time: float = 0.0
        self._ping_interval: float = 1.0

        # Update rate
        self.input_send_rate: float = 20.0  # 20 Hz
        self._last_input_send: float = 0.0

    async def connect(self):
        """Connect to the server."""
        try:
            import websockets
            self.websocket = await websockets.connect(self.server_url)
            self.connected = True
            logger.info(f"Connected to {self.server_url}")

            if self.on_connected:
                self.on_connected()

            # Start receive loop
            asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            raise

    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
        self.connected = False
        self.current_lobby = None

        if self.on_disconnected:
            self.on_disconnected()

    async def join_lobby(self, lobby_id: str, player_name: str, avatar_id: str):
        """Join a lobby."""
        await self._send(NetworkMessage(
            type=MessageType.JOIN_LOBBY,
            payload={
                "lobby_id": lobby_id,
                "player_name": player_name,
                "avatar_id": avatar_id,
            }
        ))

    async def leave_lobby(self):
        """Leave current lobby."""
        if self.current_lobby:
            await self._send(NetworkMessage(
                type=MessageType.LEAVE_LOBBY,
                payload={"lobby_id": self.current_lobby.lobby_id}
            ))
            self.current_lobby = None
            self.remote_entities.clear()

    async def send_chat(self, message: str):
        """Send a chat message."""
        await self._send(NetworkMessage(
            type=MessageType.CHAT_MESSAGE,
            payload={"message": message}
        ))

    def set_local_transform(self, position: np.ndarray, rotation: np.ndarray):
        """Update local player transform (call every frame)."""
        self.local_position = position
        self.local_rotation = rotation

    async def update(self, current_time: float):
        """
        Update network state.

        Call this every frame.
        """
        if not self.connected:
            return

        # Send player input at fixed rate
        if current_time - self._last_input_send >= 1.0 / self.input_send_rate:
            await self._send_player_input()
            self._last_input_send = current_time

        # Ping for latency
        if current_time - self.last_ping_time >= self._ping_interval:
            await self._send_ping()
            self.last_ping_time = current_time

        # Update interpolation for all remote entities
        for entity_id, buffer in self.remote_entities.items():
            buffer.get_interpolated(current_time)

    def get_remote_entity_transform(self, entity_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get interpolated transform for a remote entity."""
        if entity_id in self.remote_entities:
            buffer = self.remote_entities[entity_id]
            return buffer.current_position, buffer.current_rotation
        return None

    def get_all_remote_transforms(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get all remote entity transforms."""
        return {
            entity_id: (buffer.current_position, buffer.current_rotation)
            for entity_id, buffer in self.remote_entities.items()
        }

    async def _send(self, message: NetworkMessage):
        """Send a message to the server."""
        if self.websocket:
            await self.websocket.send(message.to_json())

    async def _send_player_input(self):
        """Send current player state to server."""
        await self._send(NetworkMessage(
            type=MessageType.PLAYER_INPUT,
            payload={
                "position": self.local_position.tolist(),
                "rotation": self.local_rotation.tolist(),
                "velocity": [0, 0, 0],  # TODO: Calculate from position delta
            }
        ))

    async def _send_ping(self):
        """Send ping for latency measurement."""
        await self._send(NetworkMessage(
            type=MessageType.PING,
            payload={"client_time": time.time()}
        ))

    async def _receive_loop(self):
        """Receive and process messages from server."""
        try:
            async for message in self.websocket:
                await self._handle_message(NetworkMessage.from_json(message))
        except Exception as e:
            logger.error(f"Receive error: {e}")
            self.connected = False
            if self.on_disconnected:
                self.on_disconnected()

    async def _handle_message(self, message: NetworkMessage):
        """Handle an incoming message."""
        if message.type == MessageType.PONG:
            client_time = message.payload.get("client_time", 0)
            self.latency_ms = (time.time() - client_time) * 1000 / 2

        elif message.type == MessageType.LOBBY_STATE:
            self._handle_lobby_state(message.payload)

        elif message.type == MessageType.ENTITY_UPDATE:
            self._handle_entity_update(message.payload)

        elif message.type == MessageType.ENTITY_SPAWN:
            self._handle_entity_spawn(message.payload)

        elif message.type == MessageType.ENTITY_DESPAWN:
            self._handle_entity_despawn(message.payload)

        elif message.type == MessageType.PLAYER_JOINED:
            self._handle_player_joined(message.payload)

        elif message.type == MessageType.PLAYER_LEFT:
            self._handle_player_left(message.payload)

        elif message.type == MessageType.CHAT_BROADCAST:
            if self.on_chat_message:
                self.on_chat_message(
                    message.payload["user_id"],
                    message.payload["user_name"],
                    message.payload["message"]
                )

    def _handle_lobby_state(self, payload: Dict):
        """Handle full lobby state update."""
        self.current_lobby = Lobby(
            lobby_id=payload["lobby_id"],
            name=payload["name"],
            stage_id=payload["stage_id"],
            max_players=payload.get("max_players", 32),
        )

        self.local_player_id = payload.get("your_player_id")

        # Initialize remote entities
        for entity_data in payload.get("entities", []):
            entity_id = entity_data["id"]
            if entity_id != self.local_player_id:
                self.remote_entities[entity_id] = InterpolationBuffer(entity_id=entity_id)

    def _handle_entity_update(self, payload: Dict):
        """Handle entity state updates."""
        current_time = time.time()

        for entity_data in payload.get("entities", []):
            entity_id = entity_data["id"]

            # Skip local player
            if entity_id == self.local_player_id:
                continue

            # Create buffer if needed
            if entity_id not in self.remote_entities:
                self.remote_entities[entity_id] = InterpolationBuffer(entity_id=entity_id)

            # Add snapshot
            snapshot = EntitySnapshot(
                timestamp=entity_data.get("timestamp", current_time),
                position=np.array(entity_data["position"]),
                rotation=np.array(entity_data["rotation"]),
                velocity=np.array(entity_data.get("velocity", [0, 0, 0])),
                animation_state=entity_data.get("animation_state"),
                blend_shapes=entity_data.get("blend_shapes"),
                audio_emitters=entity_data.get("audio_emitters"),
            )
            self.remote_entities[entity_id].add_snapshot(snapshot)

    def _handle_entity_spawn(self, payload: Dict):
        """Handle entity spawn."""
        entity_id = payload["id"]
        self.remote_entities[entity_id] = InterpolationBuffer(entity_id=entity_id)

        if self.on_entity_spawned:
            self.on_entity_spawned(entity_id, payload)

    def _handle_entity_despawn(self, payload: Dict):
        """Handle entity despawn."""
        entity_id = payload["id"]
        if entity_id in self.remote_entities:
            del self.remote_entities[entity_id]

        if self.on_entity_despawned:
            self.on_entity_despawned(entity_id)

    def _handle_player_joined(self, payload: Dict):
        """Handle player join."""
        player = LobbyPlayer(
            user_id=payload["user_id"],
            display_name=payload["display_name"],
            avatar_id=payload["avatar_id"],
        )

        if self.current_lobby:
            self.current_lobby.players[player.user_id] = player

        if self.on_player_joined:
            self.on_player_joined(player)

    def _handle_player_left(self, payload: Dict):
        """Handle player leave."""
        user_id = payload["user_id"]

        if self.current_lobby and user_id in self.current_lobby.players:
            del self.current_lobby.players[user_id]

        if user_id in self.remote_entities:
            del self.remote_entities[user_id]

        if self.on_player_left:
            self.on_player_left(user_id)


# =============================================================================
# Voice Chat Integration
# =============================================================================

class VoiceState(Enum):
    """Voice chat state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SPEAKING = "speaking"


@dataclass
class VoiceChannel:
    """
    Voice chat channel management.

    Integrates with SFU (LiveKit, mediasoup, etc.) for WebRTC.
    """
    channel_id: str
    room_id: str

    # Connection state
    state: VoiceState = VoiceState.DISCONNECTED
    sfu_token: Optional[str] = None
    sfu_url: Optional[str] = None

    # Audio settings
    is_muted: bool = False
    is_deafened: bool = False
    input_gain: float = 1.0
    output_gain: float = 1.0

    # Spatial audio
    spatial_enabled: bool = True
    listener_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    listener_rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))

    # Voice activity
    speaking_users: Dict[str, float] = field(default_factory=dict)  # user_id -> last_activity


class VoiceManager:
    """
    Manages voice chat connections and spatial audio routing.

    Integrates with LiveKit or similar SFU.
    """

    def __init__(self, sfu_url: Optional[str] = None, api_key: Optional[str] = None):
        self.sfu_url = sfu_url
        self.api_key = api_key

        self.channels: Dict[str, VoiceChannel] = {}
        self.local_user_id: Optional[str] = None

        # Audio processing callbacks
        self.on_voice_activity: Optional[Callable[[str, bool], None]] = None
        self.on_audio_level: Optional[Callable[[str, float], None]] = None

    async def create_room_token(self, room_id: str, user_id: str) -> Optional[str]:
        """
        Create a token for joining a voice room.

        In production, this calls the SFU API (LiveKit, etc.)
        """
        # Placeholder - would call LiveKit API
        # token = livekit.AccessToken(api_key, api_secret)
        # token.add_grant(VideoGrant(room_join=True, room=room_id))
        # return token.to_jwt()

        logger.info(f"[VoiceManager] Would create token for room={room_id}, user={user_id}")
        return f"placeholder_token_{room_id}_{user_id}"

    async def join_channel(self, channel_id: str, room_id: str, token: str) -> VoiceChannel:
        """Join a voice channel."""
        channel = VoiceChannel(
            channel_id=channel_id,
            room_id=room_id,
            sfu_token=token,
            sfu_url=self.sfu_url,
            state=VoiceState.CONNECTING,
        )
        self.channels[channel_id] = channel

        # In production: connect to SFU with WebRTC
        # await self._connect_webrtc(channel)

        channel.state = VoiceState.CONNECTED
        logger.info(f"[VoiceManager] Joined channel: {channel_id}")
        return channel

    async def leave_channel(self, channel_id: str):
        """Leave a voice channel."""
        if channel_id in self.channels:
            channel = self.channels[channel_id]
            channel.state = VoiceState.DISCONNECTED
            del self.channels[channel_id]
            logger.info(f"[VoiceManager] Left channel: {channel_id}")

    def set_mute(self, channel_id: str, muted: bool):
        """Mute/unmute local audio."""
        if channel_id in self.channels:
            self.channels[channel_id].is_muted = muted

    def set_deafen(self, channel_id: str, deafened: bool):
        """Deafen/undeafen (stop receiving audio)."""
        if channel_id in self.channels:
            self.channels[channel_id].is_deafened = deafened

    def update_listener_transform(self, channel_id: str, position: np.ndarray, rotation: np.ndarray):
        """Update local listener position for spatial audio."""
        if channel_id in self.channels:
            self.channels[channel_id].listener_position = position
            self.channels[channel_id].listener_rotation = rotation

    def get_speaker_positions(self, channel_id: str) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Get positions of currently speaking users for spatial audio.

        Returns: Dict of user_id -> (position, audio_level)
        """
        # In production: get from SFU + entity positions
        return {}


# =============================================================================
# Interest Management
# =============================================================================

@dataclass
class InterestArea:
    """
    Area of interest for a client.

    Entities within range get full updates.
    Entities outside get reduced updates or none.
    """
    center: np.ndarray
    full_detail_radius: float = 50.0  # Full update rate
    reduced_detail_radius: float = 100.0  # Reduced update rate
    max_radius: float = 200.0  # Beyond this, no updates

    # Voice culling
    voice_radius: float = 50.0  # Voice only from users within this range

    def get_update_priority(self, entity_position: np.ndarray) -> float:
        """
        Get update priority for an entity (0-1, 0 = no updates).

        Higher priority = more frequent updates.
        """
        distance = np.linalg.norm(entity_position - self.center)

        if distance <= self.full_detail_radius:
            return 1.0
        elif distance <= self.reduced_detail_radius:
            # Linear falloff
            t = (distance - self.full_detail_radius) / (self.reduced_detail_radius - self.full_detail_radius)
            return 1.0 - t * 0.5  # 50-100% priority
        elif distance <= self.max_radius:
            # Steeper falloff
            t = (distance - self.reduced_detail_radius) / (self.max_radius - self.reduced_detail_radius)
            return 0.5 - t * 0.5  # 0-50% priority
        else:
            return 0.0  # No updates

    def should_receive_voice(self, speaker_position: np.ndarray) -> bool:
        """Check if voice from speaker should be forwarded."""
        distance = np.linalg.norm(speaker_position - self.center)
        return distance <= self.voice_radius


class InterestManager:
    """
    Manages areas of interest for all connected clients.

    Filters updates to reduce bandwidth.
    """

    def __init__(self):
        self.client_interests: Dict[str, InterestArea] = {}

    def update_client_position(self, client_id: str, position: np.ndarray):
        """Update a client's area of interest center."""
        if client_id not in self.client_interests:
            self.client_interests[client_id] = InterestArea(center=position)
        else:
            self.client_interests[client_id].center = position

    def remove_client(self, client_id: str):
        """Remove a client's interest area."""
        if client_id in self.client_interests:
            del self.client_interests[client_id]

    def filter_entities_for_client(
        self,
        client_id: str,
        all_entities: List[Dict],
        current_time: float
    ) -> List[Dict]:
        """
        Filter entities for a specific client based on interest.

        Also applies LOD (level of detail) based on distance.
        """
        if client_id not in self.client_interests:
            return all_entities

        interest = self.client_interests[client_id]
        filtered = []

        for entity in all_entities:
            pos = np.array(entity.get("position", [0, 0, 0]))
            priority = interest.get_update_priority(pos)

            if priority <= 0:
                continue

            # For low priority, randomly skip some updates
            if priority < 0.5:
                # 50% chance to skip at 0% priority, linear
                if np.random.random() > priority * 2:
                    continue

            # Apply LOD - remove blend shapes at distance
            if priority < 0.75:
                entity_copy = entity.copy()
                entity_copy.pop("blend_shapes", None)
                filtered.append(entity_copy)
            else:
                filtered.append(entity)

        return filtered


# =============================================================================
# Delta Compression
# =============================================================================

class DeltaCompressor:
    """
    Compresses entity updates by sending only changed fields.

    Maintains baseline state per client.
    """

    def __init__(self):
        # baseline[client_id][entity_id] = last sent state
        self.baselines: Dict[str, Dict[str, Dict]] = {}

        # Threshold for position delta (don't send tiny movements)
        self.position_threshold: float = 0.01  # 1cm
        self.rotation_threshold: float = 0.001  # Very small angle

    def get_baseline(self, client_id: str, entity_id: str) -> Optional[Dict]:
        """Get baseline state for an entity."""
        if client_id in self.baselines:
            return self.baselines[client_id].get(entity_id)
        return None

    def set_baseline(self, client_id: str, entity_id: str, state: Dict):
        """Update baseline state."""
        if client_id not in self.baselines:
            self.baselines[client_id] = {}
        self.baselines[client_id][entity_id] = state.copy()

    def clear_client(self, client_id: str):
        """Clear baselines for a disconnected client."""
        if client_id in self.baselines:
            del self.baselines[client_id]

    def compute_delta(self, client_id: str, entity_id: str, current_state: Dict) -> Optional[Dict]:
        """
        Compute delta from baseline.

        Returns None if no significant change.
        Returns dict with only changed fields if there are changes.
        """
        baseline = self.get_baseline(client_id, entity_id)

        if baseline is None:
            # First time - send full state
            self.set_baseline(client_id, entity_id, current_state)
            return current_state

        delta = {"id": entity_id}
        has_changes = False

        # Check position
        if "position" in current_state:
            old_pos = np.array(baseline.get("position", [0, 0, 0]))
            new_pos = np.array(current_state["position"])
            if np.linalg.norm(new_pos - old_pos) > self.position_threshold:
                delta["position"] = current_state["position"]
                has_changes = True

        # Check rotation
        if "rotation" in current_state:
            old_rot = np.array(baseline.get("rotation", [0, 0, 0, 1]))
            new_rot = np.array(current_state["rotation"])
            if np.linalg.norm(new_rot - old_rot) > self.rotation_threshold:
                delta["rotation"] = current_state["rotation"]
                has_changes = True

        # Check animation state (always send if changed)
        if current_state.get("animation_state") != baseline.get("animation_state"):
            delta["animation_state"] = current_state.get("animation_state")
            has_changes = True

        # Check blend shapes
        old_blend = baseline.get("blend_shapes", {})
        new_blend = current_state.get("blend_shapes", {})
        if old_blend != new_blend:
            delta["blend_shapes"] = new_blend
            has_changes = True

        # Check audio emitters
        old_audio = baseline.get("audio_emitters", [])
        new_audio = current_state.get("audio_emitters", [])
        if old_audio != new_audio:
            delta["audio_emitters"] = new_audio
            has_changes = True

        if has_changes:
            self.set_baseline(client_id, entity_id, current_state)
            return delta

        return None


# =============================================================================
# Server-Side Sync (for cMUSH integration)
# =============================================================================

class NetworkServer:
    """
    Server-side networking integration for cMUSH.

    Broadcasts entity updates to connected clients with:
    - Interest management (distance-based filtering)
    - Delta compression (only send changed fields)
    - Voice chat routing
    """

    def __init__(self):
        self.clients: Dict[str, Any] = {}  # user_id -> websocket
        self.lobbies: Dict[str, Lobby] = {}

        # Optimization systems
        self.interest_manager = InterestManager()
        self.delta_compressor = DeltaCompressor()
        self.voice_manager = VoiceManager()

        # Update rate
        self.broadcast_rate: float = 20.0  # 20 Hz
        self._last_broadcast: float = 0.0

        # Stats
        self.bytes_sent: int = 0
        self.bytes_saved: int = 0  # By delta compression

    def add_client(self, user_id: str, websocket):
        """Register a connected client."""
        self.clients[user_id] = websocket
        self.interest_manager.update_client_position(user_id, np.zeros(3))
        logger.info(f"Client connected: {user_id}")

    def remove_client(self, user_id: str):
        """Unregister a disconnected client."""
        if user_id in self.clients:
            del self.clients[user_id]
            logger.info(f"Client disconnected: {user_id}")

        # Cleanup optimization state
        self.interest_manager.remove_client(user_id)
        self.delta_compressor.clear_client(user_id)

        # Remove from any lobby
        for lobby in self.lobbies.values():
            if user_id in lobby.players:
                del lobby.players[user_id]

    def update_client_position(self, user_id: str, position: np.ndarray):
        """Update a client's position for interest management."""
        self.interest_manager.update_client_position(user_id, position)

    def create_lobby(self, lobby_id: str, name: str, stage_id: str) -> Lobby:
        """Create a new lobby."""
        lobby = Lobby(
            lobby_id=lobby_id,
            name=name,
            stage_id=stage_id,
        )
        self.lobbies[lobby_id] = lobby
        return lobby

    def get_lobby(self, lobby_id: str) -> Optional[Lobby]:
        """Get a lobby by ID."""
        return self.lobbies.get(lobby_id)

    def list_lobbies(self, stage_id: Optional[str] = None) -> List[Lobby]:
        """List available lobbies, optionally filtered by stage."""
        if stage_id:
            return [l for l in self.lobbies.values() if l.stage_id == stage_id]
        return list(self.lobbies.values())

    async def broadcast_entity_updates(self, entities: List[Dict]):
        """
        Broadcast entity updates to all clients.

        Applies interest management and delta compression.
        Called by cMUSH game loop.
        """
        current_time = time.time()

        for user_id, ws in list(self.clients.items()):
            try:
                # Filter by interest
                filtered_entities = self.interest_manager.filter_entities_for_client(
                    user_id, entities, current_time
                )

                # Apply delta compression
                compressed_entities = []
                for entity in filtered_entities:
                    entity_id = entity.get("id", "")
                    delta = self.delta_compressor.compute_delta(user_id, entity_id, entity)
                    if delta:
                        compressed_entities.append(delta)

                if not compressed_entities:
                    continue

                message = NetworkMessage(
                    type=MessageType.ENTITY_UPDATE,
                    payload={"entities": compressed_entities}
                )

                msg_json = message.to_json()
                self.bytes_sent += len(msg_json)
                await ws.send(msg_json)

            except Exception as e:
                logger.warning(f"Failed to send to {user_id}: {e}")
                self.remove_client(user_id)

    async def broadcast_to_lobby(self, lobby_id: str, message_type: MessageType, payload: Dict):
        """Broadcast a message to all clients in a lobby."""
        if lobby_id not in self.lobbies:
            return

        message = NetworkMessage(type=message_type, payload=payload)
        msg_json = message.to_json()

        lobby = self.lobbies[lobby_id]
        for user_id in lobby.players:
            if user_id in self.clients:
                try:
                    await self.clients[user_id].send(msg_json)
                except Exception as e:
                    logger.warning(f"Failed to send to {user_id}: {e}")

    async def handle_player_input(self, user_id: str, payload: Dict):
        """Handle player input from client."""
        position = np.array(payload.get("position", [0, 0, 0]))

        # Update interest management
        self.update_client_position(user_id, position)

        # Update lobby player state
        for lobby in self.lobbies.values():
            if user_id in lobby.players:
                player = lobby.players[user_id]
                player.position = position
                player.rotation = np.array(payload.get("rotation", [0, 0, 0, 1]))
                player.last_update = time.time()

    async def _broadcast(self, message: str):
        """Broadcast to all connected clients."""
        for user_id, ws in list(self.clients.items()):
            try:
                await ws.send(message)
                self.bytes_sent += len(message)
            except Exception as e:
                logger.warning(f"Failed to send to {user_id}: {e}")
                self.remove_client(user_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get networking statistics."""
        return {
            "connected_clients": len(self.clients),
            "active_lobbies": len(self.lobbies),
            "total_players": sum(l.player_count for l in self.lobbies.values()),
            "bytes_sent": self.bytes_sent,
            "bytes_saved": self.bytes_saved,
        }


# =============================================================================
# Global Singletons
# =============================================================================

_network_server: Optional[NetworkServer] = None


def get_network_server() -> NetworkServer:
    """Get the global NetworkServer singleton."""
    global _network_server
    if _network_server is None:
        _network_server = NetworkServer()
    return _network_server


def init_network_server() -> NetworkServer:
    """Initialize the global NetworkServer singleton."""
    global _network_server
    _network_server = NetworkServer()
    return _network_server


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Network Sync Test")
    print("=" * 60)

    # ==================== Interpolation Buffer ====================
    print("\n1. Interpolation Buffer Test")
    print("-" * 40)

    buffer = InterpolationBuffer(entity_id="test")

    # Add some snapshots
    for i in range(10):
        snapshot = EntitySnapshot(
            timestamp=i * 0.05,  # 20 Hz
            position=np.array([float(i) * 0.1, 0, 0]),
            rotation=np.array([0, 0, 0, 1]),
            velocity=np.array([2.0, 0, 0]),
        )
        buffer.add_snapshot(snapshot)

    print(f"Added {len(buffer.snapshots)} snapshots")

    # Test interpolation at various times
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pos, rot = buffer.get_interpolated(t)
        print(f"  t={t:.1f}: position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # Test extrapolation
    pos, rot = buffer.get_interpolated(0.8)
    print(f"  t=0.8 (extrapolated): position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # ==================== Interest Management ====================
    print("\n2. Interest Management Test")
    print("-" * 40)

    interest_mgr = InterestManager()
    interest_mgr.update_client_position("player1", np.array([0, 0, 0]))

    # Test entities at various distances
    test_entities = [
        {"id": "close", "position": [10, 0, 0]},      # Within full detail
        {"id": "medium", "position": [75, 0, 0]},     # Reduced detail
        {"id": "far", "position": [150, 0, 0]},       # Low priority
        {"id": "very_far", "position": [250, 0, 0]},  # Beyond max radius
    ]

    filtered = interest_mgr.filter_entities_for_client("player1", test_entities, time.time())
    print(f"  Input entities: {len(test_entities)}")
    print(f"  Filtered entities: {len(filtered)}")
    for e in filtered:
        print(f"    - {e['id']}")

    # ==================== Delta Compression ====================
    print("\n3. Delta Compression Test")
    print("-" * 40)

    compressor = DeltaCompressor()

    # First update - full state
    state1 = {"id": "entity1", "position": [1.0, 0, 0], "rotation": [0, 0, 0, 1]}
    delta1 = compressor.compute_delta("client1", "entity1", state1)
    print(f"  First update: {len(str(delta1))} bytes (full)")

    # Second update - no change
    delta2 = compressor.compute_delta("client1", "entity1", state1)
    print(f"  No change: {delta2} (None = no update needed)")

    # Third update - position changed
    state2 = {"id": "entity1", "position": [1.1, 0.05, 0], "rotation": [0, 0, 0, 1]}
    delta3 = compressor.compute_delta("client1", "entity1", state2)
    print(f"  Position changed: {delta3}")

    # ==================== Voice Manager ====================
    print("\n4. Voice Manager Test")
    print("-" * 40)

    voice_mgr = VoiceManager(sfu_url="wss://voice.example.com")
    print(f"  SFU URL: {voice_mgr.sfu_url}")

    # Simulate token creation
    import asyncio

    async def test_voice():
        token = await voice_mgr.create_room_token("lobby1", "player1")
        print(f"  Token created: {token[:30]}...")

        channel = await voice_mgr.join_channel("channel1", "lobby1", token)
        print(f"  Joined channel: {channel.channel_id}, state={channel.state.value}")

        voice_mgr.set_mute("channel1", True)
        print(f"  Muted: {voice_mgr.channels['channel1'].is_muted}")

        await voice_mgr.leave_channel("channel1")
        print(f"  Left channel, remaining: {len(voice_mgr.channels)}")

    asyncio.run(test_voice())

    # ==================== Network Server ====================
    print("\n5. Network Server Test")
    print("-" * 40)

    server = NetworkServer()
    print(f"  Server created")

    # Create a lobby
    lobby = server.create_lobby("lobby1", "Test Lobby", "stage_nexus")
    print(f"  Lobby created: {lobby.name} (stage: {lobby.stage_id})")

    # Add a mock client
    class MockWebSocket:
        def __init__(self):
            self.messages = []
        async def send(self, msg):
            self.messages.append(msg)

    mock_ws = MockWebSocket()
    server.add_client("player1", mock_ws)
    print(f"  Client added: player1")

    # Update position
    server.update_client_position("player1", np.array([10, 0, 5]))

    # Test broadcast
    async def test_broadcast():
        entities = [
            {"id": "npc1", "position": [15, 0, 0], "rotation": [0, 0, 0, 1]},
            {"id": "npc2", "position": [100, 0, 0], "rotation": [0, 0, 0, 1]},
        ]
        await server.broadcast_entity_updates(entities)
        print(f"  Broadcast sent, messages: {len(mock_ws.messages)}")
        print(f"  Stats: {server.get_stats()}")

    asyncio.run(test_broadcast())

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
