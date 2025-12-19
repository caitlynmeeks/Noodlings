"""
Action Stream - Lightweight high-frequency action API for world models

Complements ScenePacketEmitter with a high-frequency action stream for:
- Player movement vectors
- Camera deltas
- Simple interaction triggers
- Entity micro-actions

Design Philosophy (per Gemini analysis):
    - Full Scene Packets: Low frequency (~5s), when semantic truth changes
    - Delta Packets: Medium frequency (~100ms), on state changes
    - Action Stream: High frequency (~30fps), tiny JSON payloads

The world model (Genie/Mirage) interpolates smooth rendering between
semantic updates. Actions are the lightweight "steering" inputs.

Example action payloads:
    {"action": "player_move", "direction": [0, 0, 1], "speed": 1.0}
    {"action": "camera_look", "target": "red"}
    {"action": "camera_orbit", "delta": [5, 0]}
    {"action": "interact", "entity": "radio", "verb": "toggle"}
    {"action": "entity_gaze", "entity": "red", "target": "player"}

Author: Commander Spock + Cadet Caity
Date: December 18, 2025
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Set
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Action Types
# =============================================================================

class ActionType(Enum):
    """Types of lightweight actions."""

    # Player actions
    PLAYER_MOVE = "player_move"
    PLAYER_LOOK = "player_look"
    PLAYER_JUMP = "player_jump"
    PLAYER_CROUCH = "player_crouch"

    # Camera actions (highest frequency)
    CAMERA_LOOK = "camera_look"
    CAMERA_ORBIT = "camera_orbit"
    CAMERA_ZOOM = "camera_zoom"
    CAMERA_TRACK = "camera_track"
    CAMERA_SHAKE = "camera_shake"

    # Interaction actions
    INTERACT = "interact"
    USE = "use"
    PICKUP = "pickup"
    DROP = "drop"

    # Physics actions (SPE-resolved)
    THROW = "throw"
    STRIKE = "strike"
    PUSH = "push"
    PULL = "pull"
    GIVE = "give"

    # Entity micro-actions (noodling steering)
    ENTITY_GAZE = "entity_gaze"
    ENTITY_GESTURE = "entity_gesture"
    ENTITY_EMOTE = "entity_emote"

    # Environment
    ENV_TRIGGER = "env_trigger"

    # System
    PING = "ping"
    SYNC = "sync"


# =============================================================================
# Action Data Classes
# =============================================================================

@dataclass
class Action:
    """
    A single lightweight action.

    Designed for minimal payload size - typically < 100 bytes.
    """
    action_type: ActionType
    timestamp: float = field(default_factory=time.time)

    # Common fields (all optional, depends on action type)
    entity_id: Optional[str] = None
    target_id: Optional[str] = None
    direction: Optional[List[float]] = None  # [x, y, z] normalized
    delta: Optional[List[float]] = None      # [dx, dy, dz] or [yaw, pitch]
    value: Optional[float] = None            # speed, intensity, etc.
    verb: Optional[str] = None               # for interactions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to minimal JSON-serializable dict."""
        d = {"action": self.action_type.value, "t": self.timestamp}
        if self.entity_id:
            d["entity"] = self.entity_id
        if self.target_id:
            d["target"] = self.target_id
        if self.direction:
            d["dir"] = self.direction
        if self.delta:
            d["delta"] = self.delta
        if self.value is not None:
            d["val"] = self.value
        if self.verb:
            d["verb"] = self.verb
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """Parse from minimal JSON dict."""
        action_type = ActionType(data.get("action", "ping"))
        return cls(
            action_type=action_type,
            timestamp=data.get("t", time.time()),
            entity_id=data.get("entity"),
            target_id=data.get("target"),
            direction=data.get("dir"),
            delta=data.get("delta"),
            value=data.get("val"),
            verb=data.get("verb"),
        )

    def to_json(self) -> str:
        """Serialize to compact JSON."""
        return json.dumps(self.to_dict(), separators=(',', ':'))


@dataclass
class ActionAck:
    """
    Acknowledgment of an action.

    Sent back to confirm action was processed.
    """
    action_type: ActionType
    timestamp: float
    accepted: bool = True
    message: Optional[str] = None

    # Optional state hints (minimal)
    new_position: Optional[List[float]] = None
    new_camera: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"ack": self.action_type.value, "t": self.timestamp, "ok": self.accepted}
        if self.message:
            d["msg"] = self.message
        if self.new_position:
            d["pos"] = self.new_position
        if self.new_camera:
            d["cam"] = self.new_camera
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(',', ':'))


# =============================================================================
# Action Session
# =============================================================================

@dataclass
class ActionSession:
    """
    A streaming action session with a renderer.

    Created when renderer connects and receives initial ScenePacket.
    Actions are processed in the context of this session.
    """
    session_id: str
    created_at: float = field(default_factory=time.time)

    # Session state
    player_id: Optional[str] = None
    current_stage: Optional[str] = None

    # Stats
    actions_received: int = 0
    actions_processed: int = 0
    last_action_time: float = 0.0

    # Rate limiting
    actions_per_second: float = 0.0
    _action_times: List[float] = field(default_factory=list)

    def record_action(self):
        """Record an action for rate tracking."""
        now = time.time()
        self.actions_received += 1
        self.last_action_time = now

        # Track last second of actions for rate calculation
        self._action_times.append(now)
        cutoff = now - 1.0
        self._action_times = [t for t in self._action_times if t > cutoff]
        self.actions_per_second = len(self._action_times)


# =============================================================================
# Action Stream Handler
# =============================================================================

class ActionStreamHandler:
    """
    Processes incoming action stream and updates world state.

    Works alongside ScenePacketEmitter:
    - ActionStream handles high-frequency steering inputs
    - ScenePacketEmitter handles semantic truth broadcasting

    Usage:
        handler = ActionStreamHandler(scene_state_manager)

        # Process incoming action
        ack = await handler.process_action(session_id, action_json)

        # Send ack back to renderer
        websocket.send(ack.to_json())
    """

    def __init__(self, scene_state_manager=None):
        """
        Initialize action stream handler.

        Args:
            scene_state_manager: SceneStateManager for world state updates
        """
        self.scene_state_manager = scene_state_manager

        # SPE Bridge for physics resolution (lazy init)
        self._spe_bridge = None
        self._init_spe_bridge()

        # Active sessions
        self.sessions: Dict[str, ActionSession] = {}

        # Action handlers by type
        self._handlers: Dict[ActionType, Callable] = {
            ActionType.PLAYER_MOVE: self._handle_player_move,
            ActionType.PLAYER_LOOK: self._handle_player_look,
            ActionType.CAMERA_LOOK: self._handle_camera_look,
            ActionType.CAMERA_ORBIT: self._handle_camera_orbit,
            ActionType.CAMERA_ZOOM: self._handle_camera_zoom,
            ActionType.CAMERA_TRACK: self._handle_camera_track,
            ActionType.INTERACT: self._handle_interact,
            ActionType.ENTITY_GAZE: self._handle_entity_gaze,
            ActionType.ENTITY_GESTURE: self._handle_entity_gesture,
            ActionType.PING: self._handle_ping,
            ActionType.SYNC: self._handle_sync,
            # Physics actions (SPE-resolved)
            ActionType.THROW: self._handle_physics_action,
            ActionType.STRIKE: self._handle_physics_action,
            ActionType.PUSH: self._handle_physics_action,
            ActionType.PULL: self._handle_physics_action,
            ActionType.GIVE: self._handle_physics_action,
            ActionType.PICKUP: self._handle_physics_action,
            ActionType.DROP: self._handle_physics_action,
        }

        # Callbacks for semantic changes (trigger delta packets)
        self._on_semantic_change: List[Callable[[str], None]] = []

        # Rate limiting
        self.max_actions_per_second = 60  # Per session

    def set_scene_state_manager(self, manager):
        """Set the scene state manager."""
        self.scene_state_manager = manager
        # Reconnect SPE bridge to manager
        if self._spe_bridge:
            self._spe_bridge.set_scene_state_manager(manager)

    def _init_spe_bridge(self):
        """Initialize SPE Bridge for physics resolution."""
        try:
            from .spe_bridge import SPEBridge, SPE_AVAILABLE
            if SPE_AVAILABLE:
                self._spe_bridge = SPEBridge(self.scene_state_manager)
                logger.info("[ActionStream] SPE Bridge connected")
            else:
                logger.info("[ActionStream] SPE not available, physics disabled")
        except ImportError as e:
            logger.debug(f"[ActionStream] SPE Bridge import failed: {e}")
            self._spe_bridge = None

    def get_spe_bridge(self):
        """Get the SPE Bridge for direct physics access."""
        return self._spe_bridge

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(
        self,
        player_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ) -> ActionSession:
        """
        Create a new action session.

        Called when renderer connects and receives initial ScenePacket.

        Args:
            player_id: Player entity ID for this session
            stage_id: Current stage ID

        Returns:
            New ActionSession with unique session_id
        """
        session = ActionSession(
            session_id=str(uuid.uuid4())[:8],
            player_id=player_id,
            current_stage=stage_id,
        )
        self.sessions[session.session_id] = session
        logger.info(f"[ActionStream] Session created: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ActionSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def close_session(self, session_id: str):
        """Close and remove a session."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            logger.info(f"[ActionStream] Session closed: {session_id} "
                       f"({session.actions_processed} actions)")

    # =========================================================================
    # Action Processing
    # =========================================================================

    async def process_action(
        self,
        session_id: str,
        action_data: Dict[str, Any]
    ) -> ActionAck:
        """
        Process an incoming action.

        Args:
            session_id: Session ID from initial connection
            action_data: Action dict (from JSON)

        Returns:
            ActionAck to send back to renderer
        """
        session = self.sessions.get(session_id)
        if not session:
            return ActionAck(
                action_type=ActionType.PING,
                timestamp=time.time(),
                accepted=False,
                message="Invalid session"
            )

        # Rate limiting
        session.record_action()
        if session.actions_per_second > self.max_actions_per_second:
            return ActionAck(
                action_type=ActionType.PING,
                timestamp=time.time(),
                accepted=False,
                message="Rate limited"
            )

        # Parse action
        try:
            action = Action.from_dict(action_data)
        except Exception as e:
            return ActionAck(
                action_type=ActionType.PING,
                timestamp=time.time(),
                accepted=False,
                message=f"Parse error: {e}"
            )

        # Get handler
        handler = self._handlers.get(action.action_type)
        if not handler:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False,
                message="Unknown action type"
            )

        # Process action
        try:
            ack = await handler(session, action)
            session.actions_processed += 1
            return ack
        except Exception as e:
            logger.error(f"[ActionStream] Error processing {action.action_type}: {e}")
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False,
                message=str(e)
            )

    async def process_action_json(
        self,
        session_id: str,
        json_str: str
    ) -> str:
        """
        Process action from JSON string, return JSON ack.

        Convenience method for WebSocket handlers.
        """
        try:
            action_data = json.loads(json_str)
            ack = await self.process_action(session_id, action_data)
            return ack.to_json()
        except json.JSONDecodeError as e:
            return ActionAck(
                action_type=ActionType.PING,
                timestamp=time.time(),
                accepted=False,
                message=f"JSON error: {e}"
            ).to_json()

    # =========================================================================
    # Semantic Change Callbacks
    # =========================================================================

    def on_semantic_change(self, callback: Callable[[str], None]):
        """
        Register callback for semantic changes.

        Called when an action causes a semantic change that should
        trigger a delta packet from ScenePacketEmitter.

        Args:
            callback: Function(change_type) to call
        """
        self._on_semantic_change.append(callback)

    def _notify_semantic_change(self, change_type: str):
        """Notify listeners of semantic change."""
        for callback in self._on_semantic_change:
            try:
                callback(change_type)
            except Exception as e:
                logger.error(f"[ActionStream] Semantic change callback error: {e}")

    # =========================================================================
    # Action Handlers
    # =========================================================================

    async def _handle_player_move(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle player movement."""
        if not self.scene_state_manager or not session.player_id:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False,
                message="No player"
            )

        player = self.scene_state_manager.players.get(session.player_id)
        if not player:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False,
                message="Player not found"
            )

        # Apply movement
        if action.direction:
            speed = action.value or 1.0
            from .scene_packet import Vector3
            delta = Vector3(
                action.direction[0] * speed * 0.1,
                action.direction[1] * speed * 0.1,
                action.direction[2] * speed * 0.1
            )
            player.position = Vector3(
                player.position.x + delta.x,
                player.position.y + delta.y,
                player.position.z + delta.z
            )

        return ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True,
            new_position=player.position.to_list()
        )

    async def _handle_player_look(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle player look direction."""
        if not self.scene_state_manager or not session.player_id:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False
            )

        player = self.scene_state_manager.players.get(session.player_id)
        if player and action.target_id:
            player.gaze_target = action.target_id

        return ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True
        )

    async def _handle_camera_look(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle camera look at target."""
        if not self.scene_state_manager:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=True  # Accept even without manager (renderer handles it)
            )

        if action.target_id:
            from .scene_packet import CameraMode
            self.scene_state_manager.camera_directive.subject = action.target_id
            self.scene_state_manager.camera_directive.mode = CameraMode.FOCUS_ON

        return ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True,
            new_camera={"target": action.target_id}
        )

    async def _handle_camera_orbit(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle camera orbit (yaw/pitch delta)."""
        # Orbit is typically handled client-side
        # We just acknowledge and optionally store hint
        return ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True,
            new_camera={"orbit_delta": action.delta}
        )

    async def _handle_camera_zoom(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle camera zoom."""
        return ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True,
            new_camera={"zoom": action.value}
        )

    async def _handle_camera_track(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle camera track entity."""
        if self.scene_state_manager and action.entity_id:
            from .scene_packet import CameraMode
            self.scene_state_manager.camera_directive.subject = action.entity_id
            self.scene_state_manager.camera_directive.mode = CameraMode.FOLLOW

        return ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True
        )

    async def _handle_interact(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """
        Handle interaction with entity/prim.

        Routes through Semantic Physics Engine (SPE) if available
        for narrative-first physics resolution.
        """
        if not action.entity_id:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False,
                message="No entity specified"
            )

        # Try to resolve through SPE Bridge
        outcome_data = None
        if self._spe_bridge:
            actor_id = session.player_id or "unknown_actor"
            target_id = action.entity_id
            verb = action.verb or "use"
            force = "medium"
            if action.value:
                if action.value > 0.7:
                    force = "heavy"
                elif action.value < 0.3:
                    force = "light"

            outcome_data = self._spe_bridge.resolve_interaction(
                actor_id=actor_id,
                target_id=target_id,
                verb=verb,
                force=force
            )

            if outcome_data:
                logger.info(f"[ActionStream] SPE resolved: {outcome_data.get('description', '')[:60]}...")

        # Interactions are semantic changes - notify emitter
        self._notify_semantic_change(f"interact:{action.entity_id}:{action.verb}")

        # Build ack with outcome data if available
        ack = ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True
        )

        if outcome_data:
            ack.message = outcome_data.get('description')

        return ack

    async def _handle_entity_gaze(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle entity gaze change."""
        if not self.scene_state_manager or not action.entity_id:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False
            )

        # Update noodling gaze
        noodling = self.scene_state_manager.noodlings.get(action.entity_id)
        if noodling and action.target_id:
            noodling.gaze_target = action.target_id

        return ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True
        )

    async def _handle_entity_gesture(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle entity gesture/emote."""
        if not self.scene_state_manager or not action.entity_id:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False
            )

        noodling = self.scene_state_manager.noodlings.get(action.entity_id)
        if noodling and action.verb:
            noodling.current_action = action.verb
            # Gestures are semantic - notify emitter
            self._notify_semantic_change(f"gesture:{action.entity_id}:{action.verb}")

        return ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True
        )

    async def _handle_ping(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle keepalive ping."""
        return ActionAck(
            action_type=ActionType.PING,
            timestamp=action.timestamp,
            accepted=True
        )

    async def _handle_sync(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """Handle sync request (request full packet)."""
        # Notify that a full packet should be sent
        self._notify_semantic_change("sync_request")

        return ActionAck(
            action_type=ActionType.SYNC,
            timestamp=action.timestamp,
            accepted=True,
            message="Full packet will be sent"
        )

    async def _handle_physics_action(
        self,
        session: ActionSession,
        action: Action
    ) -> ActionAck:
        """
        Handle physics action through SPE (throw, strike, push, pull, etc.).

        Routes through Semantic Physics Engine for narrative-first resolution.
        Returns descriptive outcome text suitable for both MUD and generative rendering.
        """
        if not self._spe_bridge:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False,
                message="Physics engine not available"
            )

        # Get actor and target
        actor_id = session.player_id or "unknown_actor"
        target_id = action.entity_id or action.target_id
        if not target_id:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=False,
                message="No target specified"
            )

        # Map action type to verb
        verb = action.action_type.value  # "throw", "strike", etc.

        # Determine force from action value
        force = "medium"
        if action.value is not None:
            if action.value > 0.7:
                force = "heavy"
            elif action.value < 0.3:
                force = "light"

        # Resolve through SPE
        outcome_data = self._spe_bridge.resolve_interaction(
            actor_id=actor_id,
            target_id=target_id,
            verb=verb,
            force=force
        )

        if not outcome_data:
            return ActionAck(
                action_type=action.action_type,
                timestamp=action.timestamp,
                accepted=True,
                message=f"You {verb} the {target_id}"
            )

        # Physics actions are semantic changes
        self._notify_semantic_change(f"physics:{verb}:{target_id}")

        logger.info(f"[ActionStream] Physics {verb}: {outcome_data.get('description', '')[:60]}...")

        return ActionAck(
            action_type=action.action_type,
            timestamp=action.timestamp,
            accepted=True,
            message=outcome_data.get('description')
        )

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get handler stats."""
        total_actions = sum(s.actions_processed for s in self.sessions.values())
        total_rate = sum(s.actions_per_second for s in self.sessions.values())

        return {
            "active_sessions": len(self.sessions),
            "total_actions_processed": total_actions,
            "aggregate_actions_per_second": total_rate,
            "sessions": {
                sid: {
                    "actions": s.actions_processed,
                    "rate": s.actions_per_second,
                    "player": s.player_id,
                }
                for sid, s in self.sessions.items()
            }
        }


# =============================================================================
# WebSocket Action Stream Adapter
# =============================================================================

class WebSocketActionStream:
    """
    WebSocket adapter for action streaming.

    Handles the high-frequency bidirectional communication
    between renderer and Noodlings.

    Usage:
        stream = WebSocketActionStream(action_handler, packet_emitter)

        async def websocket_handler(websocket):
            await stream.handle_connection(websocket, player_id="caity")
    """

    def __init__(
        self,
        action_handler: ActionStreamHandler,
        packet_emitter=None
    ):
        """
        Initialize WebSocket action stream.

        Args:
            action_handler: ActionStreamHandler for processing actions
            packet_emitter: ScenePacketEmitter for sending packets
        """
        self.action_handler = action_handler
        self.packet_emitter = packet_emitter

        # Connect semantic change notifications to packet emitter
        if packet_emitter:
            action_handler.on_semantic_change(self._on_semantic_change)

    def _on_semantic_change(self, change_type: str):
        """Handle semantic change - trigger delta packet."""
        if self.packet_emitter:
            if change_type == "sync_request":
                self.packet_emitter.emit_full_packet()
            else:
                self.packet_emitter.mark_camera_changed()
                # Could call emit_delta_packet() but let the emitter's
                # rate limiting handle when to actually send

    async def handle_connection(
        self,
        websocket,
        player_id: Optional[str] = None,
        stage_id: Optional[str] = None
    ):
        """
        Handle a WebSocket connection for action streaming.

        Args:
            websocket: WebSocket connection
            player_id: Player entity ID
            stage_id: Current stage ID
        """
        # Create session
        session = self.action_handler.create_session(
            player_id=player_id,
            stage_id=stage_id
        )

        try:
            # Send session info
            await websocket.send(json.dumps({
                "type": "session",
                "session_id": session.session_id,
                "player_id": player_id
            }))

            # Send initial full packet if emitter available
            if self.packet_emitter:
                packet = self.packet_emitter.emit_full_packet()
                await websocket.send(json.dumps({
                    "type": "scene",
                    "packet": json.loads(packet.to_json())
                }))

            # Process incoming actions
            async for message in websocket:
                try:
                    data = json.loads(message)

                    # Handle different message types
                    if data.get("type") == "action":
                        ack = await self.action_handler.process_action(
                            session.session_id,
                            data.get("data", data)
                        )
                        await websocket.send(ack.to_json())

                    elif "action" in data:
                        # Direct action format
                        ack = await self.action_handler.process_action(
                            session.session_id,
                            data
                        )
                        await websocket.send(ack.to_json())

                except json.JSONDecodeError:
                    # Try processing as raw action JSON
                    ack_json = await self.action_handler.process_action_json(
                        session.session_id,
                        message
                    )
                    await websocket.send(ack_json)
                except Exception as e:
                    logger.error(f"[ActionStream] Message error: {e}")

        finally:
            # Cleanup session
            self.action_handler.close_session(session.session_id)


# =============================================================================
# Global Instance
# =============================================================================

_action_handler: Optional[ActionStreamHandler] = None


def get_action_handler() -> ActionStreamHandler:
    """Get or create global ActionStreamHandler."""
    global _action_handler
    if _action_handler is None:
        _action_handler = ActionStreamHandler()
    return _action_handler


def init_action_handler(scene_state_manager=None) -> ActionStreamHandler:
    """Initialize global ActionStreamHandler with SceneStateManager."""
    global _action_handler
    _action_handler = ActionStreamHandler(scene_state_manager)
    return _action_handler


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "ActionType",
    "Action",
    "ActionAck",
    "ActionSession",

    # Handler
    "ActionStreamHandler",

    # WebSocket adapter
    "WebSocketActionStream",

    # Global access
    "get_action_handler",
    "init_action_handler",
]
