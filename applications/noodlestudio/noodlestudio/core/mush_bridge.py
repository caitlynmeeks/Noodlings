"""
noodleMUSH Bridge

WebSocket client for NoodleStudio to communicate with noodleMUSH server.
Sends rez/derez commands and receives updates.

Supports two authentication modes:
1. Studio mode: Invisible admin connection for NoodleStudio operations
2. User mode: Token-based auth with avatar selection for in-world presence
"""

import asyncio
import websockets
import json
from typing import Optional, Dict, Callable, Any
from PyQt6.QtCore import QObject, pyqtSignal, QThread


class MUSHBridge(QObject):
    """
    Bridge between NoodleStudio and noodleMUSH.

    Handles:
    - Rezzing Noodlings/Prims from Assets
    - Derezzing entities from Scene
    - Receiving world updates
    - Executing admin commands
    - User authentication with avatar selection
    """

    connected = pyqtSignal()
    disconnected = pyqtSignal()
    world_updated = pyqtSignal(dict)  # Emitted when world state changes
    user_authenticated = pyqtSignal(dict)  # Emitted on token auth success
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.server_url = "ws://localhost:8765"
        self.current_avatar_name: Optional[str] = None  # Display name in world

    async def connect(self, use_token_auth: bool = False, token: str = None,
                      avatar_id: str = None, avatar: Dict = None):
        """
        Connect to noodleMUSH server.

        Args:
            use_token_auth: If True, use cloud token authentication
            token: NoodleStudio session token (required if use_token_auth=True)
            avatar_id: Optional avatar ID to use
            avatar: Optional avatar metadata dict
        """
        try:
            self.ws = await websockets.connect(self.server_url)

            if use_token_auth and token:
                # Cloud token authentication with avatar
                return await self._authenticate_with_token(token, avatar_id, avatar)
            else:
                # Fallback: Studio admin mode (invisible)
                return await self._authenticate_studio_mode()

        except Exception as e:
            self.error.emit(f"Connection failed: {e}")
            return False

    async def _authenticate_studio_mode(self) -> bool:
        """Authenticate as invisible studio admin."""
        login_msg = {
            'type': 'login',
            'username': 'studio',
            'password': 'studio',
            'invisible': True
        }

        await self.ws.send(json.dumps(login_msg))

        response = await self.ws.recv()
        data = json.loads(response)

        if data.get('type') == 'login_response' and data.get('success'):
            self.is_connected = True
            self.connected.emit()
            print("NoodleStudio connected to noodleMUSH (studio mode)")
            return True
        else:
            self.error.emit(f"Login failed: {data.get('message', 'Unknown error')}")
            return False

    async def _authenticate_with_token(self, token: str, avatar_id: str = None,
                                        avatar: Dict = None) -> bool:
        """
        Authenticate using NoodleStudio cloud token.

        Args:
            token: NoodleStudio session token
            avatar_id: Optional avatar ID
            avatar: Optional avatar metadata dict

        Returns:
            True if authentication succeeded
        """
        auth_msg = {
            'type': 'token_auth',
            'token': token,
            'avatar_id': avatar_id,
            'avatar': avatar or {}
        }

        await self.ws.send(json.dumps(auth_msg))

        # Consume responses until we get the auth response
        # (server sends history, welcome banner, etc. after auth)
        while True:
            response = await self.ws.recv()
            data = json.loads(response)

            if data.get('type') == 'token_auth_response':
                if data.get('success'):
                    self.is_connected = True
                    self.current_avatar_name = data.get('avatar_name', 'Traveler')
                    self.connected.emit()
                    self.user_authenticated.emit(data)
                    print(f"NoodleStudio connected to noodleMUSH as {self.current_avatar_name}")
                    return True
                else:
                    self.error.emit(f"Token auth failed: {data.get('message', 'Unknown error')}")
                    return False

            # Skip other message types (history, welcome, etc.)
            # TODO: Could emit these to a chat panel

    async def disconnect(self):
        """Disconnect from server."""
        if self.ws:
            await self.ws.close()
            self.ws = None
            self.is_connected = False
            self.disconnected.emit()

    async def send_command(self, command: str) -> Dict:
        """
        Send a command to noodleMUSH.

        Args:
            command: Command string (e.g., "@rez callie", "@derez callie")

        Returns:
            Response dict from server
        """
        if not self.is_connected or not self.ws:
            return {'success': False, 'error': 'Not connected'}

        try:
            msg = {
                'type': 'command',
                'command': command
            }

            await self.ws.send(json.dumps(msg))

            # Wait for response
            response = await self.ws.recv()
            data = json.loads(response)

            return data

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def rez_noodling(self, noodling_name: str, room_id: str = "room_000") -> bool:
        """
        Rez a Noodling in the world.

        Args:
            noodling_name: Name of the Noodling (from recipe)
            room_id: Room to rez in

        Returns:
            True if successful
        """
        command = f"@rez {noodling_name}"
        result = await self.send_command(command)
        return result.get('success', False)

    async def derez_entity(self, entity_id: str, silent: bool = True) -> bool:
        """
        De-rez an entity (Noodling, prim, etc.).

        Args:
            entity_id: ID of entity to derez
            silent: If True, use -s flag for silent removal

        Returns:
            True if successful
        """
        # Extract name from ID (e.g., agent_callie -> callie)
        entity_name = entity_id.replace('agent_', '').replace('prim_', '')

        flag = " -s" if silent else ""
        command = f"@derez {entity_name}{flag}"

        result = await self.send_command(command)
        return result.get('success', False)

    async def get_world_state(self) -> Dict:
        """
        Get current world state (rooms, agents, prims).

        Returns:
            Dict with world state
        """
        # For now, read from files
        # TODO: Add @worldstate command to server
        import os
        import json

        base_path = os.path.join(
            os.path.dirname(__file__),
            "../../../cmush/world"
        )

        world_state = {}

        try:
            with open(os.path.join(base_path, "rooms.json")) as f:
                world_state['rooms'] = json.load(f)

            with open(os.path.join(base_path, "agents.json")) as f:
                world_state['agents'] = json.load(f)

            with open(os.path.join(base_path, "objects.json")) as f:
                world_state['prims'] = json.load(f)

            return world_state

        except Exception as e:
            print(f"Error loading world state: {e}")
            return {}


# Singleton instance
_bridge = None


def get_bridge() -> MUSHBridge:
    """Get the global MUSH bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = MUSHBridge()
    return _bridge
