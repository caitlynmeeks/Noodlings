"""
Network Bridge - Connects Scene State Manager to Network Server

Bridges the semantic world state to the network layer:
    - Subscribes to scene state changes
    - Converts entities to network message format
    - Broadcasts updates to connected clients
    - Handles incoming player inputs

Data Flow:
    SceneStateManager → NetworkBridge → NetworkServer → WebSocket → Clients
    Clients → WebSocket → NetworkServer → NetworkBridge → SceneStateManager

Author: Caitlyn + Claude
Date: December 2025
"""

import asyncio
import logging
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
import time

from .scene_state_manager import SceneStateManager, get_scene_state_manager
from .scene_packet import Noodling, Player, Prim, Vector3

# Import from parent's social module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from social.network_sync import (
    NetworkServer,
    NetworkClient,
    NetworkMessage,
    MessageType,
    get_network_server,
    init_network_server,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Entity Converters
# =============================================================================

def noodling_to_network(noodling: Noodling) -> Dict[str, Any]:
    """Convert a Noodling to network message format."""
    # Handle position from transform
    position = [0, 0, 0]
    rotation = [0, 0, 0, 1]

    if noodling.transform:
        if noodling.transform.position:
            position = noodling.transform.position.to_list()
        if noodling.transform.rotation:
            rotation = noodling.transform.rotation.to_list()

    return {
        "id": noodling.id,
        "type": "noodling",
        "display_name": noodling.display_name,
        "position": position,
        "rotation": rotation,
        "velocity": [0, 0, 0],
        "zone": noodling.zone,
        "animation_state": f"{noodling.posture}_{noodling.current_action}",
        "blend_shapes": {
            "happy": max(0, noodling.affect.valence) if noodling.affect else 0,
            "sad": max(0, -noodling.affect.valence) if noodling.affect else 0,
            "aroused": noodling.affect.arousal if noodling.affect else 0.5,
        },
        "expression": noodling.expression,
        "visual_state": noodling.visual_state,
        "audio_emitters": [],  # TODO: Wire audio emitters
    }


def player_to_network(player: Player) -> Dict[str, Any]:
    """Convert a Player to network message format."""
    # Handle position from transform
    position = [0, 0, 0]
    rotation = [0, 0, 0, 1]

    if player.transform:
        if player.transform.position:
            position = player.transform.position.to_list()
        if player.transform.rotation:
            rotation = player.transform.rotation.to_list()

    return {
        "id": player.id,
        "type": "player",
        "display_name": player.display_name,
        "position": position,
        "rotation": rotation,
        "velocity": [0, 0, 0],
        "zone": player.zone,
        "animation_state": f"{player.posture}_{player.current_action}",
        "blend_shapes": {},
        "audio_emitters": [],
    }


def prim_to_network(prim: Prim) -> Dict[str, Any]:
    """Convert a Prim to network message format."""
    return {
        "id": prim.id,
        "type": "prim",
        "name": prim.name,
        "position": prim.transform.position.to_list() if prim.transform else [0, 0, 0],
        "rotation": prim.transform.rotation.to_list() if prim.transform else [0, 0, 0, 1],
        "scale": prim.transform.scale.to_list() if prim.transform else [1, 1, 1],
        "animation_state": None,
        "blend_shapes": {},
        "audio_emitters": [
            {
                "id": e.id,
                "clip": e.clip,
                "playing": e.playing,
                "volume": e.volume,
            }
            for e in (prim.audio_emitters or [])
        ],
    }


# =============================================================================
# Network Bridge
# =============================================================================

class NetworkBridge:
    """
    Bridges scene state to network layer.

    Subscribes to SceneStateManager changes and broadcasts
    to all connected clients via NetworkServer.
    """

    def __init__(
        self,
        scene_manager: Optional[SceneStateManager] = None,
        network_server: Optional[NetworkServer] = None,
        broadcast_rate: float = 20.0,  # Hz
    ):
        """
        Initialize the network bridge.

        Args:
            scene_manager: Scene state manager (uses global if None)
            network_server: Network server (uses global if None)
            broadcast_rate: Updates per second to broadcast
        """
        self.scene_manager = scene_manager
        self.network_server = network_server

        self.broadcast_rate = broadcast_rate
        self._broadcast_interval = 1.0 / broadcast_rate
        self._last_broadcast = 0.0

        # Track last known state for delta detection
        self._last_entity_states: Dict[str, Dict] = {}

        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Stats
        self.updates_sent = 0
        self.bytes_sent = 0

    def connect(self):
        """
        Connect bridge to scene manager and network server.

        Uses global instances if not provided in constructor.
        """
        if self.scene_manager is None:
            self.scene_manager = get_scene_state_manager()

        if self.network_server is None:
            self.network_server = get_network_server()

        # Subscribe to state changes for immediate event handling
        self.scene_manager.on_state_change(self._on_scene_change)

        logger.info("[NetworkBridge] Connected to SceneStateManager and NetworkServer")

    async def start(self):
        """Start the broadcast loop."""
        if self._running:
            return

        self.connect()
        self._running = True
        self._task = asyncio.create_task(self._broadcast_loop())
        logger.info(f"[NetworkBridge] Started broadcast loop at {self.broadcast_rate} Hz")

    async def stop(self):
        """Stop the broadcast loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[NetworkBridge] Stopped broadcast loop")

    async def _broadcast_loop(self):
        """Main broadcast loop - sends entity updates at fixed rate."""
        while self._running:
            try:
                current_time = time.time()

                if current_time - self._last_broadcast >= self._broadcast_interval:
                    await self._broadcast_entities()
                    self._last_broadcast = current_time

                # Sleep until next update
                await asyncio.sleep(self._broadcast_interval / 2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[NetworkBridge] Broadcast error: {e}")

    async def _broadcast_entities(self):
        """Broadcast all entity updates to connected clients."""
        if not self.scene_manager or not self.network_server:
            return

        entities = self._collect_entities()

        if entities:
            await self.network_server.broadcast_entity_updates(entities)
            self.updates_sent += 1

    def _collect_entities(self) -> List[Dict[str, Any]]:
        """Collect all entities from scene manager in network format."""
        entities = []

        # Add noodlings
        for nid, noodling in self.scene_manager.noodlings.items():
            entities.append(noodling_to_network(noodling))

        # Add players
        for pid, player in self.scene_manager.players.items():
            entities.append(player_to_network(player))

        # Add prims (selective - only prims with audio or interaction)
        for prim_id, prim in self.scene_manager.prims.items():
            # Include prims that have audio emitters or affordances
            if prim.audio_emitters or prim.affordances:
                entities.append(prim_to_network(prim))

        return entities

    def _on_scene_change(self, scene_manager: SceneStateManager):
        """
        Handle scene state changes.

        Called when entities are added/removed/modified.
        For important events, we may want to broadcast immediately.
        """
        # Most updates will be handled by the broadcast loop.
        # This callback is for event-driven updates like spawns/despawns.
        pass

    # =========================================================================
    # Player Input Handling
    # =========================================================================

    async def handle_player_input(self, user_id: str, payload: Dict[str, Any]):
        """
        Handle incoming player input from network.

        Updates player position in scene state manager.
        """
        if not self.scene_manager:
            return

        position = payload.get("position", [0, 0, 0])
        rotation = payload.get("rotation", [0, 0, 0, 1])

        # Update player in scene manager
        if user_id in self.scene_manager.players:
            player = self.scene_manager.players[user_id]
            player.position = Vector3(position[0], position[1], position[2])
            # Rotation would be stored as quaternion if we add that field

        # Also update in network server for interest management
        if self.network_server:
            await self.network_server.handle_player_input(user_id, payload)

    async def handle_player_action(self, user_id: str, action: str, target: Optional[str] = None):
        """
        Handle player actions (interact, use, etc.)

        Dispatches to scene manager for processing.
        """
        if not self.scene_manager:
            return

        # Record action as event
        self.scene_manager.record_event(
            event_type="action",
            actor=user_id,
            description=f"{action} {target}" if target else action,
        )

        logger.info(f"[NetworkBridge] Player {user_id} action: {action} target={target}")

    # =========================================================================
    # Entity Spawning
    # =========================================================================

    async def notify_entity_spawn(self, entity_id: str, entity_type: str, data: Dict):
        """Notify clients of a new entity spawning."""
        if not self.network_server:
            return

        message = NetworkMessage(
            type=MessageType.ENTITY_SPAWN,
            payload={
                "id": entity_id,
                "type": entity_type,
                **data,
            }
        )

        await self.network_server._broadcast(message.to_json())
        logger.info(f"[NetworkBridge] Entity spawned: {entity_id} ({entity_type})")

    async def notify_entity_despawn(self, entity_id: str):
        """Notify clients of an entity despawning."""
        if not self.network_server:
            return

        message = NetworkMessage(
            type=MessageType.ENTITY_DESPAWN,
            payload={"id": entity_id}
        )

        await self.network_server._broadcast(message.to_json())
        logger.info(f"[NetworkBridge] Entity despawned: {entity_id}")

    # =========================================================================
    # Chat/Dialogue
    # =========================================================================

    async def broadcast_chat(self, user_id: str, user_name: str, message: str):
        """Broadcast a chat message to all clients."""
        if not self.network_server:
            return

        net_message = NetworkMessage(
            type=MessageType.CHAT_BROADCAST,
            payload={
                "user_id": user_id,
                "user_name": user_name,
                "message": message,
            }
        )

        await self.network_server._broadcast(net_message.to_json())

    async def broadcast_dialogue(self, speaker_id: str, text: str, emotion: Optional[str] = None):
        """Broadcast noodling dialogue to all clients."""
        # Record in scene state
        if self.scene_manager:
            self.scene_manager.record_dialogue(speaker_id, text)

        # Broadcast as chat
        speaker_name = speaker_id
        if self.scene_manager and speaker_id in self.scene_manager.noodlings:
            speaker_name = self.scene_manager.noodlings[speaker_id].display_name

        await self.broadcast_chat(speaker_id, speaker_name, text)

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        server_stats = self.network_server.get_stats() if self.network_server else {}

        return {
            "running": self._running,
            "broadcast_rate": self.broadcast_rate,
            "updates_sent": self.updates_sent,
            "connected_clients": server_stats.get("connected_clients", 0),
            "total_players": server_stats.get("total_players", 0),
        }


# =============================================================================
# Global Singleton
# =============================================================================

_network_bridge: Optional[NetworkBridge] = None


def get_network_bridge() -> NetworkBridge:
    """Get the global NetworkBridge singleton."""
    global _network_bridge
    if _network_bridge is None:
        _network_bridge = NetworkBridge()
    return _network_bridge


def init_network_bridge(
    scene_manager: Optional[SceneStateManager] = None,
    network_server: Optional[NetworkServer] = None,
) -> NetworkBridge:
    """Initialize the global NetworkBridge singleton."""
    global _network_bridge
    _network_bridge = NetworkBridge(scene_manager, network_server)
    return _network_bridge


# =============================================================================
# Test
# =============================================================================

def _run_tests():
    """Run network bridge tests."""
    # Import for direct test execution
    from noodlestudio.core.semantic_world.scene_state_manager import SceneStateManager
    from noodlestudio.core.semantic_world.scene_packet import Noodling, Transform, Vector3, Affect

    print("Network Bridge Test")
    print("=" * 60)

    # Create mock scene manager
    scene_mgr = SceneStateManager(stage_id="test_stage", stage_name="Test Stage")

    # Add a test noodling
    noodling = Noodling(
        id="noodling_test",
        display_name="Test Noodling",
        species="fire_imp",
        transform=Transform(position=Vector3(1, 0, 2)),
        zone="main",
        affect=Affect(valence=0.5, arousal=0.7, dominance=0.3),
        expression="happy",
        posture="standing",
    )
    scene_mgr.noodlings["noodling_test"] = noodling

    # Create bridge
    bridge = NetworkBridge(scene_manager=scene_mgr)

    # Test entity collection
    entities = bridge._collect_entities()
    print(f"\nCollected {len(entities)} entities:")
    for e in entities:
        print(f"  - {e['id']} ({e['type']}): pos={e['position']}")

    # Test conversion
    net_entity = noodling_to_network(noodling)
    print(f"\nNoodling network format:")
    print(f"  id: {net_entity['id']}")
    print(f"  display_name: {net_entity['display_name']}")
    print(f"  position: {net_entity['position']}")
    print(f"  animation_state: {net_entity['animation_state']}")
    print(f"  blend_shapes: {net_entity['blend_shapes']}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _run_tests()
