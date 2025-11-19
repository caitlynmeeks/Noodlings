"""
World State & Persistence for cMUSH

Manages:
- Rooms (spatial structure)
- Objects (items, props)
- Users (human players)
- Agents (Consilience consciousness agents)

All state stored as human-readable JSON for git-friendly collaboration.

Author: cMUSH Project
Date: October 2025
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class World:
    """
    World state manager with JSON persistence.

    Handles all persistent state for cMUSH including rooms, objects,
    users, and agents. Provides methods for creation, modification,
    and querying of world entities.
    """

    def __init__(self, world_dir: str = "world"):
        """
        Initialize world state manager.

        Args:
            world_dir: Directory for JSON storage (default "world")
        """
        self.world_dir = world_dir
        os.makedirs(world_dir, exist_ok=True)
        os.makedirs(os.path.join(world_dir, "agents"), exist_ok=True)

        # Load all state from disk
        self.rooms = self._load_json("rooms.json", {})
        self.objects = self._load_json("objects.json", {})
        self.users = self._load_json("users.json", {})
        self.agents = self._load_json("agents.json", {})

        logger.info(f"World loaded: {len(self.rooms)} rooms, {len(self.objects)} objects, "
                   f"{len(self.users)} users, {len(self.agents)} agents")

    def _load_json(self, filename: str, default: dict) -> dict:
        """
        Load JSON file or return default.

        Args:
            filename: File to load
            default: Default value if file doesn't exist

        Returns:
            Loaded data or default
        """
        path = os.path.join(self.world_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                return default
        return default

    def _save_json(self, filename: str, data: dict):
        """
        Save data to JSON file.

        Args:
            filename: File to save
            data: Data to write
        """
        path = os.path.join(self.world_dir, filename)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")

    def save_all(self):
        """Persist all world state to disk."""
        self._save_json("rooms.json", self.rooms)
        self._save_json("objects.json", self.objects)
        self._save_json("users.json", self.users)
        self._save_json("agents.json", self.agents)
        logger.debug("World state saved")

    # ===== Room Methods =====

    def get_room(self, room_id: str) -> Optional[Dict]:
        """Get room by ID."""
        return self.rooms.get(room_id)

    def create_room(
        self,
        name: str,
        description: str,
        owner: str,
        room_id: Optional[str] = None
    ) -> str:
        """
        Create a new room.

        Args:
            name: Room name
            description: Room description
            owner: User ID of owner
            room_id: Optional specific room ID

        Returns:
            Room ID
        """
        if room_id is None:
            room_id = f"room_{len(self.rooms):03d}"

        self.rooms[room_id] = {
            'uid': room_id,
            'name': name,
            'description': description,
            'exits': {},
            'objects': [],
            'occupants': [],
            'owner': owner,
            'created': datetime.now().isoformat()
        }

        self.save_all()
        logger.info(f"Room created: {room_id} '{name}' by {owner}")
        return room_id

    def set_exit(self, room_id: str, direction: str, target_room_id: str):
        """
        Create an exit from one room to another.

        Args:
            room_id: Source room
            direction: Direction (north, south, east, west, up, down)
            target_room_id: Destination room
        """
        room = self.get_room(room_id)
        if room:
            room['exits'][direction] = target_room_id
            self.save_all()
            logger.info(f"Exit created: {room_id} -> {direction} -> {target_room_id}")

    def get_room_occupants(self, room_id: str) -> List[str]:
        """
        Get all occupants (users + agents) in a room.

        Args:
            room_id: Room to check

        Returns:
            List of user/agent IDs
        """
        room = self.get_room(room_id)
        return room['occupants'] if room else []

    def get_visible_occupants(self, room_id: str) -> List[str]:
        """
        Get visible occupants (users + agents) in a room.
        Excludes invisible admin users.

        Args:
            room_id: Room to check

        Returns:
            List of visible user/agent IDs
        """
        all_occupants = self.get_room_occupants(room_id)
        visible = []

        for occupant_id in all_occupants:
            # Check if this is an invisible user
            if occupant_id.startswith('user_'):
                user = self.get_user(occupant_id)
                if user and user.get('invisible', False):
                    # Skip invisible users
                    continue
            visible.append(occupant_id)

        return visible

    # ===== Object Methods =====

    def get_object(self, obj_id: str) -> Optional[Dict]:
        """Get object by ID."""
        return self.objects.get(obj_id)

    def create_object(
        self,
        name: str,
        description: str,
        owner: str,
        location: Optional[str] = None,
        portable: bool = True,
        takeable: bool = True,
        obj_type: str = "prop",
        script: Optional[str] = None
    ) -> str:
        """
        Create a new object (prim).

        Args:
            name: Object name
            description: Object description
            owner: User ID of creator
            location: Initial location (room_id or user_id)
            portable: Can be moved
            takeable: Can be picked up
            obj_type: Prim type (prop, furniture, container, vending_machine, etc.)
            script: Optional script name attached to this prim

        Returns:
            Object ID
        """
        obj_id = f"obj_{len(self.objects):03d}"

        self.objects[obj_id] = {
            'uid': obj_id,
            'name': name,
            'description': description,
            'type': obj_type,
            'location': location,
            'owner': owner,
            'created': datetime.now().isoformat(),
            'script': {
                'name': script,  # Script class name (e.g., "AnklebiterVendingMachine")
                'code': None,  # Python source code (stored when uploaded)
                'state': {},  # Persistent instance variables
                'version': 1,  # Script version for migrations
                'compiled': False  # Whether backend has successfully compiled
            } if script else None,
            'properties': {
                'portable': portable,
                'takeable': takeable
            }
        }

        # Add to room if location specified
        if location and location.startswith('room_'):
            room = self.get_room(location)
            if room and obj_id not in room['objects']:
                room['objects'].append(obj_id)

        self.save_all()
        logger.info(f"Object created: {obj_id} '{name}' by {owner}")
        return obj_id

    # ===== User Methods =====

    def get_user(self, uid: str) -> Optional[Dict]:
        """
        Get user or agent by ID.

        Args:
            uid: User or agent ID

        Returns:
            User/agent data or None
        """
        return self.users.get(uid) or self.agents.get(uid)

    def create_user(
        self,
        username: str,
        password_hash: str,
        spawn_room: str = "room_000"
    ) -> str:
        """
        Create a new user account.

        Args:
            username: Username
            password_hash: Hashed password
            spawn_room: Initial room

        Returns:
            User ID
        """
        user_id = f"user_{username}"

        self.users[user_id] = {
            'uid': user_id,
            'username': username,
            'password_hash': password_hash,
            'current_room': spawn_room,
            'inventory': [],
            'created': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat()
        }

        # Add to spawn room
        room = self.get_room(spawn_room)
        if room and user_id not in room['occupants']:
            room['occupants'].append(user_id)

        self.save_all()
        logger.info(f"User created: {user_id}")
        return user_id

    def get_user_room(self, uid: str) -> Optional[Dict]:
        """
        Get the room a user/agent is currently in.

        Args:
            uid: User or agent ID

        Returns:
            Room data or None
        """
        user = self.get_user(uid)
        if user:
            return self.get_room(user['current_room'])
        return None

    def move_user(self, uid: str, new_room_id: str) -> bool:
        """
        Move user/agent to a new room.

        Args:
            uid: User or agent ID
            new_room_id: Destination room

        Returns:
            True if successful
        """
        user = self.get_user(uid)
        new_room = self.get_room(new_room_id)

        if not user or not new_room:
            return False

        old_room = self.get_room(user['current_room'])

        # Remove from old room
        if old_room and uid in old_room['occupants']:
            old_room['occupants'].remove(uid)

        # Add to new room
        if uid not in new_room['occupants']:
            new_room['occupants'].append(uid)

        user['current_room'] = new_room_id
        user['last_seen'] = datetime.now().isoformat()

        self.save_all()
        logger.info(f"User moved: {uid} -> {new_room_id}")
        return True

    def user_exists(self, username: str) -> bool:
        """Check if username exists."""
        user_id = f"user_{username}"
        return user_id in self.users

    # ===== Agent Methods =====

    def create_agent(
        self,
        name: str,
        checkpoint_path: str,
        spawn_room: str = "room_000",
        config: Optional[Dict] = None
    ) -> str:
        """
        Create a new Consilience agent.

        Args:
            name: Agent name
            checkpoint_path: Path to Phase 4 checkpoint
            spawn_room: Initial room
            config: Agent configuration

        Returns:
            Agent ID
        """
        agent_id = f"agent_{name}"

        self.agents[agent_id] = {
            'uid': agent_id,
            'name': name,
            'checkpoint_path': checkpoint_path,
            'current_room': spawn_room,
            'inventory': [],  # Agents can have inventory too!
            'config': config or {},
            'created': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat()
        }

        # Add to spawn room
        room = self.get_room(spawn_room)
        if room and agent_id not in room['occupants']:
            room['occupants'].append(agent_id)

        # Create agent state directory
        agent_state_dir = os.path.join(self.world_dir, "agents", agent_id)
        os.makedirs(agent_state_dir, exist_ok=True)

        self.save_all()
        logger.info(f"Agent created: {agent_id} in {spawn_room}")
        return agent_id

    def get_agent_state_path(self, agent_id: str) -> str:
        """
        Get path to agent's state directory.

        Args:
            agent_id: Agent ID

        Returns:
            Path to agent state directory
        """
        return os.path.join(self.world_dir, "agents", agent_id)

    def list_agents_in_room(self, room_id: str) -> List[str]:
        """
        Get all agents in a room.

        Args:
            room_id: Room to check

        Returns:
            List of agent IDs
        """
        occupants = self.get_room_occupants(room_id)
        return [uid for uid in occupants if uid.startswith('agent_')]

    def list_users_in_room(self, room_id: str) -> List[str]:
        """
        Get all human users in a room.

        Args:
            room_id: Room to check

        Returns:
            List of user IDs
        """
        occupants = self.get_room_occupants(room_id)
        return [uid for uid in occupants if uid.startswith('user_')]

    # ===== Query Methods =====

    def get_all_rooms(self) -> Dict[str, Dict]:
        """Get all rooms."""
        return self.rooms

    def get_all_agents(self) -> Dict[str, Dict]:
        """Get all agents."""
        return self.agents

    def get_all_users(self) -> Dict[str, Dict]:
        """Get all users."""
        return self.users

    def search_rooms(self, query: str) -> List[str]:
        """
        Search for rooms by name.

        Args:
            query: Search string

        Returns:
            List of matching room IDs
        """
        query_lower = query.lower()
        return [
            room_id for room_id, room in self.rooms.items()
            if query_lower in room['name'].lower() or query_lower in room['description'].lower()
        ]

    def get_stats(self) -> Dict:
        """
        Get world statistics.

        Returns:
            Dictionary with counts and info
        """
        return {
            'rooms': len(self.rooms),
            'objects': len(self.objects),
            'users': len(self.users),
            'agents': len(self.agents),
            'world_dir': self.world_dir
        }
