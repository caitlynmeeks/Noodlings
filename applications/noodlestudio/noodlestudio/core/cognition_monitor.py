"""
Cognition Monitor - Central registry for assembly status reporting

This singleton allows running assemblies to report their status to a central
location that can be polled by the Cognitive Cycles Panel.

Architecture-agnostic: The monitor doesn't know what KIND of cognition is
happening - assemblies publish their own status_text strings.

Usage:
    # From within an assembly or facet executor:
    from noodlestudio.core.cognition_monitor import get_cognition_monitor, CyclePhase

    monitor = get_cognition_monitor()
    monitor.report_status(
        thing_id="chester",
        assembly_id="emotional-processing",
        phase=CyclePhase.FACET,
        current_facet="LLMFacet",
        status_text="valence: 0.7, arousal: 0.4",
        activity=0.8
    )

Author: Caitlyn + Claude
Date: January 2026
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
import time
import threading
import logging

logger = logging.getLogger(__name__)


class CyclePhase(Enum):
    """Cognitive cycle phases."""
    IDLE = 0
    INCOMING = 1   # Message received, starting cycle
    PRECOG = 2     # Pre-facet processing (context intelligence)
    FACET = 3      # Facet assembly execution (LLM calls)
    NEURAL = 4     # CharmNetwork / MLX
    POSTCOG = 5    # Post-facet processing (convergence)
    OUTGOING = 6   # Final emission


@dataclass
class AssemblyStatus:
    """
    Status report from a running assembly.

    Architecture-agnostic - the platform doesn't interpret the data,
    just displays it. Assemblies publish their own status strings.
    """
    assembly_id: str
    thing_id: str

    # Phase (required)
    phase: CyclePhase = CyclePhase.IDLE

    # Current facet being executed (optional)
    current_facet: str = ""

    # Free-form status string (optional, assembly-defined)
    status_text: str = ""

    # Activity level 0.0-1.0 (optional, for sparklines)
    activity: float = 0.0

    # Is this assembly paused?
    is_paused: bool = False

    # Timestamp of last update
    last_update: float = field(default_factory=time.time)

    # Custom data blob (optional, for assembly-specific inspectors)
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'assembly_id': self.assembly_id,
            'thing_id': self.thing_id,
            'phase': self.phase.name,
            'current_facet': self.current_facet,
            'status_text': self.status_text,
            'activity': self.activity,
            'is_paused': self.is_paused,
            'last_update': self.last_update,
            'custom_data': self.custom_data
        }


@dataclass
class ThingStatus:
    """Aggregate status for a Thing with multiple assemblies."""
    thing_id: str
    thing_name: str
    assemblies: Dict[str, AssemblyStatus] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'thing_id': self.thing_id,
            'name': self.thing_name,
            'assemblies': {
                aid: status.to_dict()
                for aid, status in self.assemblies.items()
            }
        }


class CognitionMonitor:
    """
    Singleton monitor for assembly status reporting.

    Thread-safe registry that assemblies report their status to.
    The Cognitive Cycles Panel polls this for display.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._things: Dict[str, ThingStatus] = {}
        self._listeners: List[Callable[[str, str, AssemblyStatus], None]] = []
        self._data_lock = threading.RLock()
        self._initialized = True

        logger.info("[CognitionMonitor] Initialized")

    def register_thing(self, thing_id: str, thing_name: str) -> None:
        """
        Register a Thing that will have assemblies.

        Args:
            thing_id: Unique identifier for the thing
            thing_name: Display name
        """
        with self._data_lock:
            if thing_id not in self._things:
                self._things[thing_id] = ThingStatus(
                    thing_id=thing_id,
                    thing_name=thing_name
                )
                logger.debug(f"[CognitionMonitor] Registered thing: {thing_name} ({thing_id})")

    def unregister_thing(self, thing_id: str) -> None:
        """
        Unregister a Thing and all its assemblies.

        Args:
            thing_id: Thing to remove
        """
        with self._data_lock:
            if thing_id in self._things:
                del self._things[thing_id]
                logger.debug(f"[CognitionMonitor] Unregistered thing: {thing_id}")

    def report_status(
        self,
        thing_id: str,
        assembly_id: str,
        phase: CyclePhase = CyclePhase.IDLE,
        current_facet: str = "",
        status_text: str = "",
        activity: float = 0.0,
        is_paused: bool = False,
        custom_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report status for an assembly.

        This is the main API for assemblies to publish their current state.
        Call this frequently during execution to keep the UI updated.

        Args:
            thing_id: ID of the owning Thing
            assembly_id: ID of the assembly
            phase: Current cognitive phase
            current_facet: Name of facet being executed
            status_text: Free-form status string (assembly-defined)
            activity: Activity level 0.0-1.0 (for sparklines)
            is_paused: Whether this assembly is paused
            custom_data: Optional custom data blob
        """
        with self._data_lock:
            # Auto-register thing if needed
            if thing_id not in self._things:
                self.register_thing(thing_id, thing_id)

            thing = self._things[thing_id]

            # Create or update assembly status
            if assembly_id not in thing.assemblies:
                thing.assemblies[assembly_id] = AssemblyStatus(
                    assembly_id=assembly_id,
                    thing_id=thing_id
                )

            status = thing.assemblies[assembly_id]
            status.phase = phase
            status.current_facet = current_facet
            status.status_text = status_text
            status.activity = activity
            status.is_paused = is_paused
            status.last_update = time.time()
            if custom_data:
                status.custom_data.update(custom_data)

        # Notify listeners (outside lock)
        for listener in self._listeners:
            try:
                listener(thing_id, assembly_id, status)
            except Exception as e:
                logger.error(f"[CognitionMonitor] Listener error: {e}")

    def clear_assembly(self, thing_id: str, assembly_id: str) -> None:
        """
        Clear status for an assembly (set to IDLE).

        Args:
            thing_id: ID of the owning Thing
            assembly_id: ID of the assembly
        """
        self.report_status(
            thing_id=thing_id,
            assembly_id=assembly_id,
            phase=CyclePhase.IDLE,
            activity=0.0
        )

    def get_thing_status(self, thing_id: str) -> Optional[ThingStatus]:
        """
        Get status for a specific Thing.

        Args:
            thing_id: Thing to get status for

        Returns:
            ThingStatus or None if not found
        """
        with self._data_lock:
            return self._things.get(thing_id)

    def get_assembly_status(self, thing_id: str, assembly_id: str) -> Optional[AssemblyStatus]:
        """
        Get status for a specific assembly.

        Args:
            thing_id: ID of the owning Thing
            assembly_id: ID of the assembly

        Returns:
            AssemblyStatus or None if not found
        """
        with self._data_lock:
            thing = self._things.get(thing_id)
            if thing:
                return thing.assemblies.get(assembly_id)
            return None

    def get_all_statuses(self) -> Dict[str, ThingStatus]:
        """
        Get all thing statuses.

        Returns:
            Dictionary of thing_id -> ThingStatus
        """
        with self._data_lock:
            return dict(self._things)

    def get_hierarchical_data(self) -> Dict[str, Any]:
        """
        Get data in hierarchical format for API response.

        Returns:
            Dictionary suitable for JSON response:
            {
                "things": {
                    "chester-uuid": {
                        "name": "chester",
                        "assemblies": {
                            "emotional-processing": {...},
                            ...
                        }
                    }
                }
            }
        """
        with self._data_lock:
            return {
                "things": {
                    thing_id: status.to_dict()
                    for thing_id, status in self._things.items()
                }
            }

    def add_listener(self, callback: Callable[[str, str, AssemblyStatus], None]) -> None:
        """
        Add listener for status updates.

        Args:
            callback: Function(thing_id, assembly_id, status) called on updates
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, str, AssemblyStatus], None]) -> None:
        """
        Remove listener.

        Args:
            callback: Previously registered callback
        """
        if callback in self._listeners:
            self._listeners.remove(callback)

    def pause_assembly(self, thing_id: str, assembly_id: str, paused: bool = True) -> bool:
        """
        Set pause state for an assembly.

        Args:
            thing_id: ID of the owning Thing
            assembly_id: ID of the assembly
            paused: Whether to pause

        Returns:
            True if successful, False if assembly not found
        """
        with self._data_lock:
            thing = self._things.get(thing_id)
            if thing and assembly_id in thing.assemblies:
                thing.assemblies[assembly_id].is_paused = paused
                return True
            return False

    def pause_thing(self, thing_id: str, paused: bool = True) -> bool:
        """
        Set pause state for all assemblies of a Thing.

        Args:
            thing_id: ID of the Thing
            paused: Whether to pause

        Returns:
            True if successful, False if thing not found
        """
        with self._data_lock:
            thing = self._things.get(thing_id)
            if thing:
                for assembly in thing.assemblies.values():
                    assembly.is_paused = paused
                return True
            return False

    def pause_all(self, paused: bool = True) -> None:
        """
        Set pause state for all assemblies globally.

        Args:
            paused: Whether to pause
        """
        with self._data_lock:
            for thing in self._things.values():
                for assembly in thing.assemblies.values():
                    assembly.is_paused = paused

    def is_paused(self, thing_id: str, assembly_id: Optional[str] = None) -> bool:
        """
        Check if an assembly or thing is paused.

        Args:
            thing_id: ID of the Thing
            assembly_id: ID of assembly (if None, checks any assembly in thing)

        Returns:
            True if paused
        """
        with self._data_lock:
            thing = self._things.get(thing_id)
            if not thing:
                return False

            if assembly_id:
                assembly = thing.assemblies.get(assembly_id)
                return assembly.is_paused if assembly else False
            else:
                return any(a.is_paused for a in thing.assemblies.values())

    def reset(self) -> None:
        """Clear all registered things and assemblies."""
        with self._data_lock:
            self._things.clear()
        logger.info("[CognitionMonitor] Reset")


# Singleton accessor
_monitor_instance: Optional[CognitionMonitor] = None


def get_cognition_monitor() -> CognitionMonitor:
    """
    Get the singleton CognitionMonitor instance.

    Returns:
        The global CognitionMonitor
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = CognitionMonitor()
    return _monitor_instance


__all__ = [
    'CognitionMonitor',
    'get_cognition_monitor',
    'CyclePhase',
    'AssemblyStatus',
    'ThingStatus',
]
