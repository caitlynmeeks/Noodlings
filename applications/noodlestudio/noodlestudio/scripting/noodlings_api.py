"""
Noodlings Scripting API - Unity-like commands

Global functions and classes available to all scripts:

- Noodlings.Spawn()
- Noodlings.Find()
- Noodlings.Destroy()
- Debug.Log()
- Transform (position, rotation)
- Prim (properties)

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from typing import Optional, List
import requests


class Debug:
    """Debug utilities (like Unity.Debug)."""

    API_BASE = "http://localhost:8081/api"

    @staticmethod
    def Log(message: str):
        """Log message to console (Unity-style)."""
        print(f"[Script] {message}")
        # TODO: Send to noodleMUSH console

    @staticmethod
    def LogWarning(message: str):
        """Log warning."""
        print(f"[Script WARNING] {message}")

    @staticmethod
    def LogError(message: str):
        """Log error."""
        print(f"[Script ERROR] {message}")


class Vector3:
    """3D vector (like Unity.Vector3)."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def zero():
        return Vector3(0, 0, 0)

    @staticmethod
    def one():
        return Vector3(1, 1, 1)


class Transform:
    """Transform component (position, rotation, scale)."""

    def __init__(self, prim_id: str):
        self.prim_id = prim_id
        self.position = Vector3.zero()
        self.rotation = Vector3.zero()
        self.scale = Vector3.one()

    def Translate(self, direction: Vector3):
        """Move prim by direction vector."""
        self.position.x += direction.x
        self.position.y += direction.y
        self.position.z += direction.z
        # TODO: Update in noodleMUSH


class Prim:
    """
    Prim instance (like Unity GameObject).

    Represents any prim in the stage.
    """

    def __init__(self, prim_id: str, prim_type: str = "prim"):
        self.id = prim_id
        self.type = prim_type
        self.name = prim_id
        self.active = True
        self.transform = Transform(prim_id)

    def SetActive(self, active: bool):
        """Enable/disable prim."""
        self.active = active
        # TODO: Update in noodleMUSH

    def Destroy(self, delay: float = 0.0):
        """Destroy this prim."""
        Debug.Log(f"Destroying {self.name} in {delay}s")
        # TODO: Send destroy command to noodleMUSH


class Noodlings:
    """
    Global Noodlings API (like Unity's GameObject static methods).

    Usage:
        Noodlings.Rez("anklebiter.noodling", position)
        phi = Noodlings.Find("agent_phi")
        Noodlings.Destroy(phi)
    """

    API_BASE = "http://localhost:8081/api"

    @staticmethod
    def Rez(noodling_file: str, position: Vector3 = None, room: str = "room_000") -> Optional[Prim]:
        """
        Rez a Noodling from .noodling file (Second Life terminology!).

        Args:
            noodling_file: Path to .noodling file (e.g., "anklebiter.noodling")
            position: Rez position (optional, for spatial games)
            room: Room ID to rez into

        Returns:
            Prim instance of rezzed Noodling
        """
        Debug.Log(f"Rezzing Noodling: {noodling_file} in {room}")

        # TODO: Load .noodling file
        # TODO: POST to /api/agents/rez
        # TODO: Return Prim instance

        # For now, simulate
        noodling_id = f"agent_{noodling_file.replace('.noo', '').replace('.noodling', '')}"
        return Prim(noodling_id, "noodling")

    @staticmethod
    def RezPrim(prim_type: str, name: str, room: str = "room_000") -> Optional[Prim]:
        """
        Rez a generic prim (prop, furniture, container, etc.).

        Args:
            prim_type: Type of prim (prop, furniture, container, etc.)
            name: Prim name
            room: Room ID to rez into

        Returns:
            Prim instance
        """
        Debug.Log(f"Rezzing prim: {name} ({prim_type}) in {room}")

        # TODO: POST to /api/prims/create
        prim_id = f"prim_{name.lower().replace(' ', '_')}"
        return Prim(prim_id, prim_type)

    @staticmethod
    def Find(prim_id: str) -> Optional[Prim]:
        """
        Find a prim by ID (Unity-style GameObject.Find).

        Args:
            prim_id: Prim ID to find

        Returns:
            Prim instance or None
        """
        try:
            resp = requests.get(f"{Noodlings.API_BASE}/prims/{prim_id}", timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                return Prim(prim_id, data.get('type', 'prim'))
        except:
            pass

        return None

    @staticmethod
    def FindAll(prim_type: str) -> List[Prim]:
        """
        Find all prims of a certain type.

        Args:
            prim_type: Type to search for (noodling, prim, room, etc.)

        Returns:
            List of Prim instances
        """
        # TODO: GET /api/prims?type={prim_type}
        return []

    @staticmethod
    def Destroy(prim: Prim, delay: float = 0.0):
        """
        Destroy a prim (Unity-style).

        Args:
            prim: Prim to destroy
            delay: Delay in seconds
        """
        prim.Destroy(delay)

    @staticmethod
    def SendMessage(prim_id: str, message: str):
        """
        Send message to a Noodling (make them hear it).

        Args:
            prim_id: Target Noodling ID
            message: Message to send
        """
        Debug.Log(f"Sending to {prim_id}: {message}")
        # TODO: POST to /api/agents/{prim_id}/hear


class Time:
    """Time utilities (like Unity.Time)."""

    deltaTime: float = 0.0  # Time since last frame
    time: float = 0.0  # Time since session start

    @staticmethod
    def Wait(seconds: float):
        """Wait for seconds (coroutine-like)."""
        # TODO: Implement coroutine system
        pass
