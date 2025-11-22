"""
Noodlings Scripting API - Unity-like Global Functions

Global functions and classes available to all scripts:

- Noodlings.Rez() - Spawn a Noodling from recipe
- Noodlings.RezPrim() - Create a generic prim
- Noodlings.Find() - Find entity by ID
- Noodlings.SendMessage() - Send message to Noodling
- Debug.Log() - Console logging
- Vector3, Transform, Prim - Spatial utilities

Server-authoritative model:
- All API calls execute in cmush backend
- No network round-trips during script execution
- Backend injects proper implementation via dependency injection

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from typing import Optional, List, Callable, Any, Dict
import logging

logger = logging.getLogger(__name__)


class Debug:
    """Debug utilities (like Unity.Debug)."""

    _log_callback: Optional[Callable] = None

    @staticmethod
    def SetLogCallback(callback: Callable[[str, str], None]):
        """
        Set callback for log messages (used by backend).

        Args:
            callback: Function(level, message) to handle logs
        """
        Debug._log_callback = callback

    @staticmethod
    def Log(message: str):
        """Log message to console (Unity-style)."""
        if Debug._log_callback:
            Debug._log_callback('INFO', message)
        else:
            logger.info(f"[Script] {message}")

    @staticmethod
    def LogWarning(message: str):
        """Log warning."""
        if Debug._log_callback:
            Debug._log_callback('WARNING', message)
        else:
            logger.warning(f"[Script] {message}")

    @staticmethod
    def LogError(message: str):
        """Log error."""
        if Debug._log_callback:
            Debug._log_callback('ERROR', message)
        else:
            logger.error(f"[Script] {message}")


class Vector3:
    """3D vector (like Unity.Vector3)."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def zero():
        """Return zero vector (0, 0, 0)."""
        return Vector3(0, 0, 0)

    @staticmethod
    def one():
        """Return unit vector (1, 1, 1)."""
        return Vector3(1, 1, 1)

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"


class Transform:
    """Transform component (position, rotation, scale)."""

    def __init__(self, prim_id: str):
        self.prim_id = prim_id
        self.position = Vector3.zero()
        self.rotation = Vector3.zero()
        self.scale = Vector3.one()

    def Translate(self, direction: Vector3):
        """
        Move prim by direction vector.

        Args:
            direction: Vector3 offset
        """
        self.position.x += direction.x
        self.position.y += direction.y
        self.position.z += direction.z
        # Backend will sync this to world state


class Prim:
    """
    Prim instance (like Unity GameObject).

    Represents any entity in the world (prop, furniture, agent, etc.)
    """

    def __init__(self, prim_id: str, prim_type: str = "prim", room: str = None):
        self.id = prim_id
        self.type = prim_type
        self.name = prim_id
        self.room = room
        self.active = True
        self.transform = Transform(prim_id)
        self._components: Dict[str, Any] = {}  # Component cache

    def GetComponent(self, component_name: str) -> Optional[Any]:
        """
        Get component by name (Unity-style).

        Args:
            component_name: Component type name ("Noodle", "Transform", etc.)

        Returns:
            Component instance or None if not found

        Example:
            noodle = prim.GetComponent("Noodle")
            affect = noodle.GetCurrentAffect()
        """
        # Check cache first
        if component_name in self._components:
            return self._components[component_name]

        # Create component on demand
        if component_name == "Transform":
            return self.transform

        elif component_name == "Noodle":
            # Only Noodling agents have Noodle component
            if self.type == "noodling" and self.id.startswith("agent_"):
                try:
                    from noodlings_scripting.noodle_component import NoodleComponent
                    comp = NoodleComponent(self.id)
                    self._components["Noodle"] = comp
                    return comp
                except Exception as e:
                    Debug.LogError(f"Error creating Noodle component: {e}")
                    return None
            else:
                Debug.LogWarning(f"{self.id} is not a Noodling - no Noodle component")
                return None

        else:
            Debug.LogWarning(f"Unknown component type: {component_name}")
            return None

    def SetActive(self, active: bool):
        """Enable/disable prim."""
        self.active = active
        # Backend will update world state

    def Destroy(self, delay: float = 0.0):
        """
        Destroy this prim.

        Args:
            delay: Delay in seconds before destruction
        """
        Debug.Log(f"Destroying {self.name} in {delay}s")
        # Backend will handle actual destruction

    def __repr__(self):
        return f"Prim({self.id}, type={self.type}, room={self.room})"


class Noodlings:
    """
    Global Noodlings API (like Unity's GameObject static methods).

    Server-authoritative:
    - All methods call backend implementation
    - No direct world state manipulation from scripts
    - Backend injects implementation via dependency injection

    Usage:
        anklebiter = Noodlings.Rez("blue_fire_anklebiter", room="room_000")
        phi = Noodlings.Find("agent_phi")
        Noodlings.SendMessage(phi.id, "Hello!")
    """

    # Backend implementations (injected by script_manager)
    _rez_impl: Optional[Callable] = None
    _rez_prim_impl: Optional[Callable] = None
    _find_impl: Optional[Callable] = None
    _send_message_impl: Optional[Callable] = None
    _broadcast_impl: Optional[Callable] = None

    @staticmethod
    def SetBackend(
        rez_impl: Callable,
        rez_prim_impl: Callable,
        find_impl: Callable,
        send_message_impl: Callable,
        broadcast_impl: Callable
    ):
        """
        Inject backend implementations (called by script_manager).

        Args:
            rez_impl: Function(recipe, room) -> agent_id
            rez_prim_impl: Function(type, name, room) -> prim_id
            find_impl: Function(prim_id) -> Prim
            send_message_impl: Function(target_id, message)
            broadcast_impl: Function(room_id, message)
        """
        Noodlings._rez_impl = rez_impl
        Noodlings._rez_prim_impl = rez_prim_impl
        Noodlings._find_impl = find_impl
        Noodlings._send_message_impl = send_message_impl
        Noodlings._broadcast_impl = broadcast_impl

    @staticmethod
    def Rez(recipe: str, room: str = "room_000") -> Optional[Prim]:
        """
        Rez a Noodling from recipe file (Second Life terminology).

        Args:
            recipe: Recipe filename (e.g., "blue_fire_anklebiter" or "blue_fire_anklebiter.yaml")
            room: Room ID to rez into

        Returns:
            Prim instance of rezzed Noodling, or None if failed
        """
        if not Noodlings._rez_impl:
            Debug.LogError("Noodlings.Rez() - Backend not initialized!")
            return None

        # Clean recipe name
        recipe_clean = recipe.replace('.nood', '').replace('.yaml', '')
        Debug.Log(f"Rezzing Noodling: {recipe_clean} in {room}")

        try:
            agent_id = Noodlings._rez_impl(recipe_clean, room)
            if agent_id:
                return Prim(agent_id, "noodling", room)
            else:
                Debug.LogError(f"Failed to rez {recipe_clean}")
                return None
        except Exception as e:
            Debug.LogError(f"Error rezzing {recipe_clean}: {e}")
            return None

    @staticmethod
    def RezPrim(prim_type: str, name: str, room: str = "room_000") -> Optional[Prim]:
        """
        Rez a generic prim (prop, furniture, container, etc.).

        Args:
            prim_type: Type of prim (prop, furniture, container, vending_machine)
            name: Prim name (display name)
            room: Room ID to rez into

        Returns:
            Prim instance, or None if failed
        """
        if not Noodlings._rez_prim_impl:
            Debug.LogError("Noodlings.RezPrim() - Backend not initialized!")
            return None

        Debug.Log(f"Rezzing prim: {name} ({prim_type}) in {room}")

        try:
            prim_id = Noodlings._rez_prim_impl(prim_type, name, room)
            if prim_id:
                return Prim(prim_id, prim_type, room)
            else:
                Debug.LogError(f"Failed to rez prim {name}")
                return None
        except Exception as e:
            Debug.LogError(f"Error rezzing prim {name}: {e}")
            return None

    @staticmethod
    def Find(prim_id: str) -> Optional[Prim]:
        """
        Find an entity by ID (Unity-style GameObject.Find).

        Args:
            prim_id: Entity ID (user_foo, agent_bar, obj_baz)

        Returns:
            Prim instance or None
        """
        if not Noodlings._find_impl:
            Debug.LogError("Noodlings.Find() - Backend not initialized!")
            return None

        try:
            return Noodlings._find_impl(prim_id)
        except Exception as e:
            Debug.LogError(f"Error finding {prim_id}: {e}")
            return None

    @staticmethod
    def SendMessage(target_id: str, message: str):
        """
        Send message to a Noodling (inject into their perception).

        Args:
            target_id: Agent ID (agent_foo)
            message: Message text
        """
        if not Noodlings._send_message_impl:
            Debug.LogError("Noodlings.SendMessage() - Backend not initialized!")
            return

        Debug.Log(f"Sending to {target_id}: {message}")

        try:
            Noodlings._send_message_impl(target_id, message)
        except Exception as e:
            Debug.LogError(f"Error sending message: {e}")

    @staticmethod
    def Broadcast(room_id: str, message: str):
        """
        Broadcast message to everyone in a room.

        Args:
            room_id: Room ID
            message: Message text
        """
        if not Noodlings._broadcast_impl:
            Debug.LogError("Noodlings.Broadcast() - Backend not initialized!")
            return

        Debug.Log(f"Broadcasting to {room_id}: {message}")

        try:
            Noodlings._broadcast_impl(room_id, message)
        except Exception as e:
            Debug.LogError(f"Error broadcasting: {e}")


class Time:
    """Time utilities (like Unity.Time)."""

    deltaTime: float = 0.0  # Time since last frame
    time: float = 0.0  # Time since session start

    @staticmethod
    def Wait(seconds: float):
        """
        Wait for seconds (coroutine-like).

        Note: Not implemented yet - requires async coroutine system.
        """
        Debug.LogWarning("Time.Wait() not yet implemented")
