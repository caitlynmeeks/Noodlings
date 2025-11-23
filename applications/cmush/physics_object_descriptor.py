"""
Semantic Physics Engine (SPE) - Physics Object Descriptor (POD)

Provides semantic (not numerical) physics for noodleMUSH objects.
Objects describe WHAT they are and HOW they behave using natural language
rather than numerical simulation.

Core Principle: "Describe what happens, not how it's calculated."

Author: Lieutenant Caitlyn + Commander Spock
Date: November 22, 2025
Inspiration: Cyclotron fetus logic
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import json


@dataclass
class PhysicsEvent:
    """
    An ongoing or scheduled physics event (e.g., puddle drying, fire burning).

    Events have duration and can be queried for current state.
    """
    description: str
    start_time: float  # Unix timestamp
    duration: float    # Seconds
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if event has finished."""
        return time.time() >= (self.start_time + self.duration)

    def progress(self) -> float:
        """Get completion percentage (0.0 to 1.0)."""
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.duration)

    def query(self, question: str) -> str:
        """
        Query current state of event using semantic understanding.

        This would ideally call an LLM to interpret the event state,
        but for now provides simple template responses.

        Args:
            question: Natural language question about event state

        Returns:
            Natural language answer
        """
        progress_pct = self.progress() * 100
        remaining = (self.start_time + self.duration) - time.time()

        question_lower = question.lower()

        if 'progress' in question_lower or 'percent' in question_lower or 'how far' in question_lower:
            return f"About {progress_pct:.0f}% complete"

        elif 'when' in question_lower or 'time' in question_lower or 'long' in question_lower:
            if remaining > 0:
                return f"Approximately {remaining/60:.1f} minutes remaining"
            else:
                return "Event has completed"

        elif 'complete' in question_lower or 'done' in question_lower or 'finished' in question_lower:
            return "Yes, completed" if self.is_complete() else "No, still in progress"

        else:
            # Generic response
            return f"{self.description} - {progress_pct:.0f}% complete, {remaining/60:.1f} minutes remaining"


class PhysicsObjectDescriptor:
    """
    Semantic description of an object's physical properties.

    Instead of numerical physics (mass=50.0kg, velocity=Vector3(1,0,0)),
    we use semantic descriptions that LLMs can understand and interpret:
    mass="heavy", velocity="fast (speeding)", material="lead".

    This enables narrative-first physics without expensive simulation.

    Example:
        bullet = PhysicsObjectDescriptor(
            mass="light",
            friction="low",
            velocity="fast (speeding)",
            elasticity="rigid",
            softness="hard",
            material="lead",
            semantic_properties=["small", "dangerous", "penetrating"]
        )
    """

    def __init__(
        self,
        # Core physical properties (semantic descriptions)
        mass: str = "medium",
        friction: str = "medium",
        velocity: str = "stationary",
        elasticity: str = "normal",
        softness: str = "normal",
        material: str = "unknown",

        # Semantic properties (list of descriptive tags)
        semantic_properties: Optional[List[str]] = None,

        # Current state description
        state: str = "normal",

        # Arbitrary metadata
        metadata: Optional[Dict[str, Any]] = None,

        # Tags (Unity-style)
        tags: Optional[List[str]] = None
    ):
        """
        Initialize physics object descriptor.

        Args:
            mass: Semantic mass description ("heavy", "light", "5kg", "negligible")
            friction: Surface friction ("smooth", "rough", "sticky", "0.3")
            velocity: Current velocity ("fast", "slow", "stationary", "15 m/s")
            elasticity: Bounciness ("bouncy", "rigid", "soft", "0.8")
            softness: Material hardness ("hard", "soft", "squishy", "brittle")
            material: Material type ("metal", "rubber", "liquid", "silly putty")
            semantic_properties: List of descriptive tags (["liquid", "fragile", "hot"])
            state: Current state description ("normal", "broken", "on fire", "frozen")
            metadata: Arbitrary additional properties (temperature, sound, etc.)
            tags: Unity-style tags for interaction filtering
        """
        self.mass = mass
        self.friction = friction
        self.velocity = velocity
        self.elasticity = elasticity
        self.softness = softness
        self.material = material

        self.semantic_properties = semantic_properties or []
        self.state = state
        self.metadata = metadata or {}
        self.tags = set(tags or [])

        # State history for debugging/memory
        self.state_history: List[Dict[str, Any]] = []

        # Current ongoing event (e.g., "puddle drying", "fire burning")
        self.current_event: Optional[PhysicsEvent] = None

        # Reference to owning prim (set by world when attached)
        self.prim_id: Optional[str] = None

        # Physics enabled flag (respects NoPhysics tag)
        self.physics_enabled = "NoPhysics" not in self.tags

    def add_tag(self, tag: str):
        """
        Add a Unity-style tag.

        Args:
            tag: Tag to add (e.g., "NoPhysics", "Pickupable", "Flammable")
        """
        self.tags.add(tag)
        if tag == "NoPhysics":
            self.physics_enabled = False

    def remove_tag(self, tag: str):
        """Remove a tag."""
        self.tags.discard(tag)
        if tag == "NoPhysics":
            self.physics_enabled = True

    def has_tag(self, tag: str) -> bool:
        """Check if object has a specific tag."""
        return tag in self.tags

    def change_state(self, new_description: str):
        """
        Update object state semantically.

        Args:
            new_description: New state description
        """
        # Record in history
        self.state_history.append({
            'timestamp': time.time(),
            'old_state': self.state,
            'new_state': new_description
        })

        self.state = new_description

    def set_event(
        self,
        description: str,
        start_time: float,
        duration: float,  # Seconds
        callback: Optional[Callable] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Schedule a physics event.

        Args:
            description: What's happening ("puddle drying", "fire spreading")
            start_time: Unix timestamp when event starts
            duration: Duration in seconds
            callback: Function to call when complete
            metadata: Additional event data
        """
        self.current_event = PhysicsEvent(
            description=description,
            start_time=start_time,
            duration=duration,
            callback=callback,
            metadata=metadata or {}
        )

    def parse_mass(self) -> float:
        """
        Attempt to extract numerical mass from semantic description.

        Returns:
            Estimated mass in kg (best guess if not specified)
        """
        mass_lower = self.mass.lower()

        # Try to find number with "kg"
        if 'kg' in mass_lower:
            try:
                # Extract number before "kg"
                num_str = mass_lower.split('kg')[0].strip()
                return float(num_str)
            except ValueError:
                pass

        # Semantic approximations
        if 'negligible' in mass_lower or 'tiny' in mass_lower:
            return 0.01
        elif 'very light' in mass_lower:
            return 0.1
        elif 'light' in mass_lower:
            return 1.0
        elif 'medium' in mass_lower:
            return 10.0
        elif 'heavy' in mass_lower:
            return 50.0
        elif 'very heavy' in mass_lower:
            return 200.0
        else:
            return 10.0  # Default medium

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation
        """
        return {
            'mass': self.mass,
            'friction': self.friction,
            'velocity': self.velocity,
            'elasticity': self.elasticity,
            'softness': self.softness,
            'material': self.material,
            'semantic_properties': self.semantic_properties,
            'state': self.state,
            'metadata': self.metadata,
            'tags': list(self.tags),
            'state_history': self.state_history,
            'current_event': {
                'description': self.current_event.description,
                'start_time': self.current_event.start_time,
                'duration': self.current_event.duration,
                'metadata': self.current_event.metadata
            } if self.current_event else None,
            'prim_id': self.prim_id,
            'physics_enabled': self.physics_enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicsObjectDescriptor':
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            PhysicsObjectDescriptor instance
        """
        pod = cls(
            mass=data.get('mass', 'medium'),
            friction=data.get('friction', 'medium'),
            velocity=data.get('velocity', 'stationary'),
            elasticity=data.get('elasticity', 'normal'),
            softness=data.get('softness', 'normal'),
            material=data.get('material', 'unknown'),
            semantic_properties=data.get('semantic_properties', []),
            state=data.get('state', 'normal'),
            metadata=data.get('metadata', {}),
            tags=data.get('tags', [])
        )

        # Restore state history
        pod.state_history = data.get('state_history', [])

        # Restore event (but not callback - can't serialize functions)
        if data.get('current_event'):
            event_data = data['current_event']
            pod.current_event = PhysicsEvent(
                description=event_data['description'],
                start_time=event_data['start_time'],
                duration=event_data['duration'],
                metadata=event_data.get('metadata', {})
            )

        pod.prim_id = data.get('prim_id')
        pod.physics_enabled = data.get('physics_enabled', True)

        return pod

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"POD(mass={self.mass}, material={self.material}, "
            f"state={self.state}, tags={self.tags})"
        )


# ===== Helper Functions =====

def distribute_mass_uniform(total_mass: float, num_fragments: int) -> List[float]:
    """
    Distribute mass uniformly across fragments.

    Args:
        total_mass: Total mass to distribute
        num_fragments: Number of fragments

    Returns:
        List of fragment masses (sum equals total_mass)
    """
    base_mass = total_mass / num_fragments
    return [base_mass] * num_fragments


def distribute_mass_power_law(
    total_mass: float,
    num_fragments: int,
    alpha: float = 1.5
) -> List[float]:
    """
    Distribute mass following power law (realistic fragmentation).

    Args:
        total_mass: Total mass to distribute
        num_fragments: Number of fragments
        alpha: Power law exponent (higher = more uneven distribution)

    Returns:
        List of fragment masses (sum equals total_mass)
    """
    # Generate power law distribution
    weights = [(i + 1) ** (-alpha) for i in range(num_fragments)]
    total_weight = sum(weights)

    # Normalize to total mass
    masses = [(w / total_weight) * total_mass for w in weights]

    # Ensure exact conservation (fix rounding errors)
    mass_sum = sum(masses)
    if mass_sum != total_mass:
        masses[0] += (total_mass - mass_sum)

    return masses


def parse_duration(duration_str: str) -> float:
    """
    Parse semantic duration to seconds.

    Args:
        duration_str: Semantic duration ("2 hours", "30 seconds", "1 day")

    Returns:
        Duration in seconds
    """
    duration_lower = duration_str.lower()

    # Extract number
    parts = duration_lower.split()
    if len(parts) < 2:
        return 60.0  # Default 1 minute

    try:
        value = float(parts[0])
    except ValueError:
        return 60.0

    unit = parts[1]

    # Convert to seconds
    if 'second' in unit:
        return value
    elif 'minute' in unit:
        return value * 60
    elif 'hour' in unit:
        return value * 3600
    elif 'day' in unit:
        return value * 86400
    else:
        return 60.0  # Default


# ===== Example PODs =====

# Projectile
POD_BULLET = PhysicsObjectDescriptor(
    mass="light",
    friction="low",
    velocity="fast (speeding)",
    elasticity="rigid",
    softness="hard",
    material="lead",
    semantic_properties=["small", "dangerous", "penetrating"],
    metadata={"made_of": "silly putty", "color": "gray"}
)

# Target
POD_TIN_CAN = PhysicsObjectDescriptor(
    mass="very light",
    friction="medium",
    velocity="stationary",
    elasticity="slightly flexible",
    softness="thin metal",
    material="flimsy tin",
    semantic_properties=["hollow", "rusted", "jagged edges"],
    metadata={"condition": "old", "sound_when_hit": "metallic clang"}
)

# Environment
POD_WET_PUDDLE = PhysicsObjectDescriptor(
    mass="medium (water + mud)",
    friction="very high (suction)",
    velocity="stationary",
    elasticity="none (liquid)",
    softness="liquid",
    material="water + mud",
    semantic_properties=["liquid body", "non-fungible", "absorbs objects"],
    metadata={
        "depth": "shallow (~5 inches)",
        "viscosity": "muddy",
        "drying_rate": "2 hours in sun"
    },
    tags=["Liquid", "NoPickup"]
)

# Fire Imp (for vending machine example)
POD_FIRE_IMP = PhysicsObjectDescriptor(
    mass="negligible (pure energy)",
    friction="none (floats)",
    velocity="hovering",
    elasticity="none (incorporeal)",
    softness="intangible",
    material="living flame",
    semantic_properties=["hot", "bright", "flickering", "alive", "mischievous"],
    metadata={
        "temperature": "800°F",
        "light_radius": "5 feet",
        "burn_damage": "moderate",
        "personality": "snarky and obnoxious"
    },
    tags=["HeatSource", "LightSource", "Alive", "NoPhysics"]
)
