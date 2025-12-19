"""
Scene Packet Emitter - Streams Scene Packets to Renderers

The ScenePacketEmitter handles:
    - Emitting full packets on significant state changes
    - Delta packets for incremental updates
    - Camera-only packets for smooth camera moves
    - WebSocket streaming to connected renderers
    - Text flattening for LLM-based renderers

This is the outbound interface to Genie, Mirage, or any renderer.

Author: Caitlyn + Claude
Date: December 2025
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set
from enum import Enum
import weakref

from .scene_packet import ScenePacket, PacketType
from .scene_state_manager import SceneStateManager


# =============================================================================
# Emitter Configuration
# =============================================================================

@dataclass
class EmitterConfig:
    """Configuration for the scene packet emitter."""

    # Packet emission rates
    full_packet_interval: float = 5.0    # Seconds between full packets
    delta_packet_interval: float = 0.1   # Seconds between delta packets
    camera_packet_interval: float = 0.033  # ~30fps for smooth camera

    # Optimization
    emit_deltas: bool = True             # Emit delta packets
    emit_camera_only: bool = True        # Emit camera-only packets
    include_references_always: bool = False  # Include refs in every packet

    # Output
    flatten_to_text: bool = False        # Also emit text flattening
    json_indent: Optional[int] = None    # JSON formatting (None = compact)


# =============================================================================
# Renderer Connection
# =============================================================================

class RendererType(Enum):
    """Types of renderers we can emit to."""
    GENIE = "genie"
    MIRAGE = "mirage"
    CUSTOM = "custom"
    DEBUG = "debug"


@dataclass
class RendererConnection:
    """A connected renderer receiving packets."""
    id: str
    renderer_type: RendererType
    callback: Callable[[ScenePacket], None]

    # What this renderer wants
    wants_full: bool = True
    wants_delta: bool = True
    wants_camera_only: bool = True
    wants_text: bool = False

    # Stats
    packets_sent: int = 0
    last_packet_time: float = 0.0
    errors: int = 0


# =============================================================================
# Scene Packet Emitter
# =============================================================================

class ScenePacketEmitter:
    """
    Emits scene packets to connected renderers.

    Usage:
        emitter = ScenePacketEmitter(scene_state_manager)

        # Connect a renderer
        emitter.connect_renderer(
            "genie_1",
            RendererType.GENIE,
            my_genie_callback
        )

        # Start emitting (async)
        await emitter.start()

        # Or emit manually
        packet = emitter.emit_now()
    """

    def __init__(
        self,
        state_manager: SceneStateManager,
        config: EmitterConfig = None
    ):
        """
        Initialize the emitter.

        Args:
            state_manager: The scene state manager to emit from
            config: Emitter configuration
        """
        self.state_manager = state_manager
        self.config = config or EmitterConfig()

        # Connected renderers
        self.renderers: Dict[str, RendererConnection] = {}

        # State tracking for deltas
        self._last_full_packet: Optional[ScenePacket] = None
        self._last_full_time: float = 0.0
        self._last_delta_time: float = 0.0
        self._last_camera_time: float = 0.0

        # Change tracking
        self._changed_noodlings: Set[str] = set()
        self._changed_players: Set[str] = set()
        self._changed_prims: Set[str] = set()
        self._camera_changed: bool = False
        self._narrative_changed: bool = False

        # Running state
        self._running: bool = False
        self._emit_task: Optional[asyncio.Task] = None

        # Subscribe to state changes
        state_manager.on_state_change(self._on_state_change)

        # Event callbacks
        self._on_emit: List[Callable[[ScenePacket], None]] = []

    # =========================================================================
    # Renderer Management
    # =========================================================================

    def connect_renderer(
        self,
        renderer_id: str,
        renderer_type: RendererType,
        callback: Callable[[ScenePacket], None],
        wants_full: bool = True,
        wants_delta: bool = True,
        wants_camera_only: bool = True,
        wants_text: bool = False
    ) -> RendererConnection:
        """
        Connect a renderer to receive packets.

        Args:
            renderer_id: Unique ID for this connection
            renderer_type: Type of renderer
            callback: Function to call with packets
            wants_*: What packet types this renderer wants

        Returns:
            The renderer connection object
        """
        connection = RendererConnection(
            id=renderer_id,
            renderer_type=renderer_type,
            callback=callback,
            wants_full=wants_full,
            wants_delta=wants_delta,
            wants_camera_only=wants_camera_only,
            wants_text=wants_text,
        )
        self.renderers[renderer_id] = connection
        return connection

    def disconnect_renderer(self, renderer_id: str):
        """Disconnect a renderer."""
        if renderer_id in self.renderers:
            del self.renderers[renderer_id]

    def get_renderer_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all connected renderers."""
        return {
            rid: {
                "type": r.renderer_type.value,
                "packets_sent": r.packets_sent,
                "last_packet_time": r.last_packet_time,
                "errors": r.errors,
            }
            for rid, r in self.renderers.items()
        }

    # =========================================================================
    # State Change Tracking
    # =========================================================================

    def _on_state_change(self, manager: SceneStateManager):
        """Called when scene state changes."""
        # For now, mark everything as changed
        # Future: track specific changes for delta packets
        self._narrative_changed = True

    def mark_noodling_changed(self, noodling_id: str):
        """Mark a noodling as changed for delta tracking."""
        self._changed_noodlings.add(noodling_id)

    def mark_camera_changed(self):
        """Mark camera as changed."""
        self._camera_changed = True

    def _clear_change_tracking(self):
        """Clear change tracking after delta emission."""
        self._changed_noodlings.clear()
        self._changed_players.clear()
        self._changed_prims.clear()
        self._camera_changed = False
        self._narrative_changed = False

    # =========================================================================
    # Packet Emission
    # =========================================================================

    def emit_full_packet(self) -> ScenePacket:
        """Emit a full scene packet to all renderers."""
        packet = self.state_manager.generate_scene_packet(PacketType.FULL)
        self._last_full_packet = packet
        self._last_full_time = time.time()

        self._emit_to_renderers(packet, "full")
        self._clear_change_tracking()

        return packet

    def emit_delta_packet(self) -> Optional[ScenePacket]:
        """Emit a delta packet with only changes."""
        if not self._has_changes():
            return None

        # For now, emit full packet as delta
        # Future: build actual delta with only changed fields
        packet = self.state_manager.generate_scene_packet(PacketType.DELTA)
        self._last_delta_time = time.time()

        self._emit_to_renderers(packet, "delta")
        self._clear_change_tracking()

        return packet

    def emit_camera_only(self) -> Optional[ScenePacket]:
        """Emit a camera-only packet."""
        if not self._camera_changed:
            return None

        # Build minimal packet with just camera
        packet = ScenePacket(
            header=self.state_manager.generate_scene_packet().header,
            camera_directive=self.state_manager.camera_directive,
        )
        packet.header.packet_type = PacketType.CAMERA_ONLY
        self._last_camera_time = time.time()

        self._emit_to_renderers(packet, "camera_only")
        self._camera_changed = False

        return packet

    def emit_now(self, packet_type: PacketType = PacketType.FULL) -> ScenePacket:
        """Immediately emit a packet."""
        if packet_type == PacketType.FULL:
            return self.emit_full_packet()
        elif packet_type == PacketType.DELTA:
            return self.emit_delta_packet() or self.emit_full_packet()
        elif packet_type == PacketType.CAMERA_ONLY:
            return self.emit_camera_only() or self.emit_full_packet()

    def _has_changes(self) -> bool:
        """Check if there are any tracked changes."""
        return (
            bool(self._changed_noodlings) or
            bool(self._changed_players) or
            bool(self._changed_prims) or
            self._narrative_changed
        )

    def _emit_to_renderers(self, packet: ScenePacket, packet_type: str):
        """Emit packet to all interested renderers."""
        now = time.time()

        for renderer in self.renderers.values():
            # Check if renderer wants this type
            if packet_type == "full" and not renderer.wants_full:
                continue
            if packet_type == "delta" and not renderer.wants_delta:
                continue
            if packet_type == "camera_only" and not renderer.wants_camera_only:
                continue

            try:
                # Emit the packet
                renderer.callback(packet)

                # Also emit text if wanted
                if renderer.wants_text and self.config.flatten_to_text:
                    text = packet.flatten_to_text()
                    # Could have a separate text callback

                renderer.packets_sent += 1
                renderer.last_packet_time = now

            except Exception as e:
                renderer.errors += 1
                print(f"Error emitting to renderer {renderer.id}: {e}")

        # Call general emit callbacks
        for callback in self._on_emit:
            try:
                callback(packet)
            except Exception as e:
                print(f"Error in emit callback: {e}")

    # =========================================================================
    # Continuous Emission
    # =========================================================================

    async def start(self):
        """Start continuous packet emission."""
        if self._running:
            return

        self._running = True
        self._emit_task = asyncio.create_task(self._emit_loop())

    async def stop(self):
        """Stop continuous packet emission."""
        self._running = False
        if self._emit_task:
            self._emit_task.cancel()
            try:
                await self._emit_task
            except asyncio.CancelledError:
                pass
            self._emit_task = None

    async def _emit_loop(self):
        """Main emission loop."""
        while self._running:
            now = time.time()

            try:
                # Check if we need a full packet
                if now - self._last_full_time >= self.config.full_packet_interval:
                    self.emit_full_packet()

                # Check if we need a delta packet
                elif (
                    self.config.emit_deltas and
                    self._has_changes() and
                    now - self._last_delta_time >= self.config.delta_packet_interval
                ):
                    self.emit_delta_packet()

                # Check if we need a camera-only packet
                elif (
                    self.config.emit_camera_only and
                    self._camera_changed and
                    now - self._last_camera_time >= self.config.camera_packet_interval
                ):
                    self.emit_camera_only()

            except Exception as e:
                print(f"Error in emit loop: {e}")

            # Sleep until next check
            await asyncio.sleep(self.config.camera_packet_interval)

    # =========================================================================
    # Event Callbacks
    # =========================================================================

    def on_emit(self, callback: Callable[[ScenePacket], None]):
        """Register a callback for when packets are emitted."""
        self._on_emit.append(callback)

    def connect_action_handler(self, action_handler):
        """
        Connect an ActionStreamHandler to this emitter.

        When the action handler detects semantic changes, it will
        trigger appropriate packet emissions.

        Args:
            action_handler: ActionStreamHandler instance
        """
        def on_semantic_change(change_type: str):
            """Handle semantic change from action stream."""
            if change_type == "sync_request":
                # Full sync requested - emit full packet
                self.emit_full_packet()
            elif change_type.startswith("interact:") or change_type.startswith("gesture:"):
                # Interactions and gestures are semantic - mark for delta
                self._narrative_changed = True
            else:
                # Camera or minor change
                self.mark_camera_changed()

        action_handler.on_semantic_change(on_semantic_change)

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_current_packet_json(self, indent: int = 2) -> str:
        """Get current state as JSON string."""
        packet = self.state_manager.generate_scene_packet()
        return packet.to_json(indent=indent)

    def get_current_packet_text(self) -> str:
        """Get current state as flattened text."""
        packet = self.state_manager.generate_scene_packet()
        return packet.flatten_to_text()


# =============================================================================
# WebSocket Adapter
# =============================================================================

class WebSocketPacketAdapter:
    """
    Adapter for sending packets over WebSocket.

    Usage:
        adapter = WebSocketPacketAdapter(emitter)

        # In your WebSocket handler:
        async def handle_ws(websocket):
            await adapter.handle_connection(websocket, "renderer_1")
    """

    def __init__(self, emitter: ScenePacketEmitter):
        """Initialize with an emitter."""
        self.emitter = emitter
        self._connections: Dict[str, Any] = {}  # websocket connections

    async def handle_connection(
        self,
        websocket,
        client_id: str,
        renderer_type: RendererType = RendererType.CUSTOM
    ):
        """
        Handle a WebSocket connection from a renderer.

        Args:
            websocket: The WebSocket connection
            client_id: Unique ID for this client
            renderer_type: Type of renderer
        """
        self._connections[client_id] = websocket

        def send_packet(packet: ScenePacket):
            """Send packet over websocket."""
            try:
                # This needs to be async in real implementation
                data = packet.to_json()
                # websocket.send(data) - would need async
            except Exception as e:
                print(f"Error sending to {client_id}: {e}")

        # Connect the renderer
        self.emitter.connect_renderer(
            client_id,
            renderer_type,
            send_packet,
        )

        try:
            # Send initial full packet
            packet = self.emitter.emit_full_packet()
            await websocket.send(packet.to_json())

            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(client_id, message)

        finally:
            # Disconnect on close
            self.emitter.disconnect_renderer(client_id)
            del self._connections[client_id]

    async def _handle_message(self, client_id: str, message: str):
        """Handle incoming message from renderer."""
        try:
            data = json.loads(message)

            # Handle different message types
            if data.get("type") == "camera_request":
                # Renderer requesting camera change
                pass
            elif data.get("type") == "ping":
                # Keepalive
                pass

        except Exception as e:
            print(f"Error handling message from {client_id}: {e}")


# =============================================================================
# Genie Adapter
# =============================================================================

class GenieAdapter:
    """
    Adapter for Google Genie integration.

    Transforms Noodlings Scene Packets to Genie's expected format.
    """

    def __init__(self, emitter: ScenePacketEmitter):
        """Initialize with an emitter."""
        self.emitter = emitter
        self._connected = False

    def transform_packet(self, packet: ScenePacket) -> Dict[str, Any]:
        """
        Transform a Noodlings packet to Genie format.

        This is where we adapt our rich semantic data to whatever
        Genie expects as input.
        """
        # This will need to be updated based on Genie's actual API
        return {
            "scene_description": packet.flatten_to_text(),
            "reference_images": self._extract_references(packet),
            "camera": self._transform_camera(packet.camera_directive),
            "style_hints": self._extract_style_hints(packet),
        }

    def _extract_references(self, packet: ScenePacket) -> List[Dict[str, Any]]:
        """Extract reference images for Genie."""
        refs = []
        for char_id, char_ref in packet.reference_bundle.characters.items():
            if char_ref.primary_ref:
                refs.append({
                    "entity_id": char_id,
                    "type": "character",
                    "uri": char_ref.primary_ref,
                    "description": char_ref.description,
                })
        return refs

    def _transform_camera(self, camera: Any) -> Dict[str, Any]:
        """Transform camera directive to Genie format."""
        return {
            "mode": camera.mode.value,
            "subject": camera.subject,
            "framing": camera.framing.value,
            "style": camera.style.color_grade,
        }

    def _extract_style_hints(self, packet: ScenePacket) -> Dict[str, Any]:
        """Extract style hints for Genie."""
        return {
            "time_of_day": packet.spatial_truth.ambient.time_of_day,
            "weather": packet.spatial_truth.ambient.weather,
            "mood": packet.narrative_context.scene_state.current_beat,
            "tension": packet.narrative_context.scene_state.tension,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "EmitterConfig",
    "RendererType",
    "RendererConnection",
    "ScenePacketEmitter",
    "WebSocketPacketAdapter",
    "GenieAdapter",
]
