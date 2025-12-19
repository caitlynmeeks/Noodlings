"""
Scene Packet - The Noodlings Scene Protocol Data Structures

A Scene Packet is a complete snapshot of semantic truth that can be
projected to any output modality:
    - Text (MUD descriptions)
    - 2D illustrated maps (NoodleStudio editor)
    - 3D generative video (Genie/Mirage)
    - Traditional 3D render (USD pipeline)

The Scene Packet provides everything a stateless renderer needs:
    - Spatial truth (zones, positions, connections)
    - Entity state (noodlings, players, prims)
    - Visual references (per-form reference images)
    - Narrative context (recent events, mood, tension)
    - Camera directive (cinematography instructions)

Genie is stateless. Noodlings is stateful.
We are the persistent brain and heart.

Author: Caitlyn + Claude
Date: December 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from uuid import uuid4
import json
import math


# =============================================================================
# Enums
# =============================================================================

class PacketType(Enum):
    """Type of scene packet for renderer optimization."""
    FULL = "full"              # Complete scene state
    DELTA = "delta"            # Only changes since last packet
    CAMERA_ONLY = "camera_only"  # Only camera directive changed


class CameraMode(Enum):
    """High-level camera directive modes."""
    POV = "POV"                    # First person through subject's eyes
    SHOW_WHAT_SEES = "SHOW_WHAT_SEES"  # Subject's POV looking at attention target
    FOCUS_ON = "FOCUS_ON"          # Camera on subject
    TWO_SHOT = "TWO_SHOT"          # Frame two subjects
    GROUP_SHOT = "GROUP_SHOT"      # Frame multiple subjects
    ESTABLISH = "ESTABLISH"        # Wide shot establishing location
    FOLLOW = "FOLLOW"              # Third person following
    FREE = "FREE"                  # Specific camera placement
    CINEMATIC = "CINEMATIC"        # Let renderer choose dramatic framing


class Framing(Enum):
    """Shot framing options."""
    EXTREME_CLOSEUP = "extreme_closeup"
    CLOSEUP = "closeup"
    MEDIUM_CLOSEUP = "medium_closeup"
    MEDIUM = "medium"
    MEDIUM_WIDE = "medium_wide"
    WIDE = "wide"
    EXTREME_WIDE = "extreme_wide"


class CameraAngle(Enum):
    """Camera angle options."""
    WORMS_EYE = "worms_eye"
    LOW = "low"
    EYE_LEVEL = "eye_level"
    HIGH = "high"
    BIRDS_EYE = "birds_eye"
    DUTCH = "dutch"


class CameraMovement(Enum):
    """Camera movement styles."""
    STATIC = "static"
    GENTLE_DRIFT = "gentle_drift"
    SLOW_PUSH = "slow_push"
    SLOW_PULL = "slow_pull"
    TRACKING = "tracking"
    HANDHELD = "handheld"
    CRANE = "crane"
    ORBIT = "orbit"


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class Vector3:
    """3D vector for positions, directions, rotations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, lst: List[float]) -> 'Vector3':
        return cls(lst[0], lst[1], lst[2]) if lst else cls()

    def distance_to(self, other: 'Vector3') -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def normalized(self) -> 'Vector3':
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag == 0:
            return Vector3(0, 0, 1)
        return Vector3(self.x/mag, self.y/mag, self.z/mag)

    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def direction_to(self, other: 'Vector3') -> 'Vector3':
        """Get normalized direction vector from self to other."""
        delta = other - self
        return delta.normalized()

    def angle_to_2d(self, other: 'Vector3') -> float:
        """Get angle in degrees to other point on XZ plane (Y is up)."""
        dx = other.x - self.x
        dz = other.z - self.z
        return math.degrees(math.atan2(dx, dz))

    @classmethod
    def from_euler_y(cls, degrees: float) -> 'Vector3':
        """Create forward direction from Y-axis rotation (yaw)."""
        rad = math.radians(degrees)
        return cls(math.sin(rad), 0, math.cos(rad))


@dataclass
class Transform:
    """
    Standard transform component for all entities.

    All positions are relative to zone center (zone-local coordinates).
    Y-axis is up. Rotation is euler angles in degrees.

    This is the fundamental spatial primitive for:
    - 2D top-down editor (XZ plane)
    - SPE physics calculations (distance/direction)
    - Perception calculations (visibility)
    - Scene rendering (any modality)
    """
    position: Vector3 = field(default_factory=Vector3)
    rotation: Vector3 = field(default_factory=Vector3)  # Euler degrees (pitch, yaw, roll)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))

    @property
    def forward(self) -> Vector3:
        """Get forward direction based on Y rotation (yaw)."""
        return Vector3.from_euler_y(self.rotation.y)

    @property
    def right(self) -> Vector3:
        """Get right direction (perpendicular to forward on XZ plane)."""
        fwd = self.forward
        return Vector3(fwd.z, 0, -fwd.x)

    def distance_to(self, other: 'Transform') -> float:
        """Distance to another transform."""
        return self.position.distance_to(other.position)

    def direction_to(self, other: 'Transform') -> Vector3:
        """Normalized direction vector to another transform."""
        return self.position.direction_to(other.position)

    def relative_angle_to(self, other: 'Transform') -> float:
        """
        Angle in degrees to other transform, relative to our facing direction.

        Returns: -180 to 180, where 0 is directly ahead, 90 is right, -90 is left
        """
        # World angle to target
        world_angle = self.position.angle_to_2d(other.position)
        # Our facing angle
        my_angle = self.rotation.y
        # Relative angle
        rel = world_angle - my_angle
        # Normalize to -180..180
        while rel > 180:
            rel -= 360
        while rel < -180:
            rel += 360
        return rel

    def relative_direction_label(self, other: 'Transform') -> str:
        """
        Human-readable direction label to other transform.

        Returns: "ahead", "left", "right", "behind", "ahead_left", etc.
        """
        angle = self.relative_angle_to(other)

        if -22.5 <= angle < 22.5:
            return "ahead"
        elif 22.5 <= angle < 67.5:
            return "ahead_right"
        elif 67.5 <= angle < 112.5:
            return "right"
        elif 112.5 <= angle < 157.5:
            return "behind_right"
        elif angle >= 157.5 or angle < -157.5:
            return "behind"
        elif -157.5 <= angle < -112.5:
            return "behind_left"
        elif -112.5 <= angle < -67.5:
            return "left"
        else:  # -67.5 to -22.5
            return "ahead_left"

    def is_facing(self, other: 'Transform', fov: float = 90.0) -> bool:
        """Check if we're facing another transform within FOV."""
        angle = abs(self.relative_angle_to(other))
        return angle <= fov / 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.to_list(),
            "rotation": self.rotation.to_list(),
            "scale": self.scale.to_list(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Transform':
        return cls(
            position=Vector3.from_list(d.get("position", [0, 0, 0])),
            rotation=Vector3.from_list(d.get("rotation", [0, 0, 0])),
            scale=Vector3.from_list(d.get("scale", [1, 1, 1])),
        )


@dataclass
class PerceptionCone:
    """Defines what an entity can perceive."""
    fov_horizontal: float = 120.0    # degrees
    fov_vertical: float = 60.0       # degrees
    range: float = 15.0              # meters

    # Special perception abilities
    night_vision: bool = False
    heat_sense: bool = False         # See warm bodies through occlusion
    motion_sensitivity: float = 0.5  # 0-1, how easily notices movement
    occlusion_ignore: bool = False   # Can see through walls (ghost form)

    # Computed: what's currently perceived
    sees: List[str] = field(default_factory=list)
    attention_focus: Optional[str] = None
    attention_strength: float = 0.5


@dataclass
class Affect:
    """5-dimensional continuous affect state."""
    valence: float = 0.0      # -1 to 1 (pleasure)
    arousal: float = 0.5      # 0 to 1 (energy)
    dominance: float = 0.5    # 0 to 1 (control)
    boredom: float = 0.0      # 0 to 1
    sorrow: float = 0.0       # 0 to 1

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "boredom": self.boredom,
            "sorrow": self.sorrow
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'Affect':
        return cls(
            valence=d.get("valence", 0.0),
            arousal=d.get("arousal", 0.5),
            dominance=d.get("dominance", 0.5),
            boredom=d.get("boredom", 0.0),
            sorrow=d.get("sorrow", 0.0)
        )


@dataclass
class VisualForm:
    """A visual form/state a character can take."""
    id: str
    description: str
    reference_images: Dict[str, str] = field(default_factory=dict)
    # Keys like: neutral, happy, angry, action, etc.
    # Values are URIs: "noodlings://red/portrait.png"

    style_hints: Dict[str, Any] = field(default_factory=dict)
    # Things like: opacity, glow_color, fur_quality, etc.


@dataclass
class Affordance:
    """Something you can do with a prim."""
    verb: str
    description: str
    aliases: List[str] = field(default_factory=list)
    requires: Optional[str] = None  # Permission or item required


# =============================================================================
# Zone
# =============================================================================

@dataclass
class ZoneBounds:
    """
    Spatial bounds for a zone.

    Zones are soft attention regions - entities can be in multiple zones
    with varying perception strengths. The bounds define:
    - Where entities can be positioned (for the 2D editor)
    - Physics simulation boundaries (SPE throws stop at zone edge)
    - Perception falloff (how strongly the zone affects entities)
    """
    # Primary bounds (for 2D editor layout)
    shape: str = "circle"  # circle, rectangle, polygon
    radius: float = 10.0   # For circle shape
    width: float = 20.0    # For rectangle shape (X dimension)
    depth: float = 20.0    # For rectangle shape (Z dimension)
    vertices: List[Vector3] = field(default_factory=list)  # For polygon

    # Soft perception bounds
    perception_radius: float = 15.0   # How far perception extends
    perception_falloff: float = 5.0   # Meters of gradual falloff

    # Height bounds (Y dimension)
    floor_height: float = 0.0
    ceiling_height: float = 4.0

    def contains_point(self, point: Vector3) -> bool:
        """Check if a point is within the hard bounds."""
        if self.shape == "circle":
            dist_2d = math.sqrt(point.x**2 + point.z**2)
            return dist_2d <= self.radius
        elif self.shape == "rectangle":
            return (abs(point.x) <= self.width / 2 and
                    abs(point.z) <= self.depth / 2)
        return True  # Default to contained

    def clamp_point(self, point: Vector3) -> Vector3:
        """Clamp a point to within bounds."""
        if self.shape == "circle":
            dist_2d = math.sqrt(point.x**2 + point.z**2)
            if dist_2d > self.radius:
                scale = self.radius / dist_2d
                return Vector3(point.x * scale, point.y, point.z * scale)
        elif self.shape == "rectangle":
            return Vector3(
                max(-self.width/2, min(self.width/2, point.x)),
                point.y,
                max(-self.depth/2, min(self.depth/2, point.z))
            )
        return point


@dataclass
class Zone:
    """
    A soft attention region within a stage.

    Zones define areas of interest where entities gather and interact.
    All entity positions within a zone are relative to the zone's origin
    (the center point). This enables:
    - Easy 2D editor layout (place furniture relative to room center)
    - Portable zone definitions (copy a cafe layout to another stage)
    - SPE physics that work consistently within each zone
    """
    id: str
    name: str

    # Zone transform (position in world/stage space)
    # The origin (0,0,0) of the zone in stage coordinates
    world_position: Vector3 = field(default_factory=Vector3)
    world_rotation: float = 0.0  # Y-axis rotation in degrees

    # Spatial bounds (all relative to zone origin)
    bounds: ZoneBounds = field(default_factory=ZoneBounds)

    # Description
    description: str = ""
    features: List[str] = field(default_factory=list)
    mood: str = "neutral"
    lighting: str = "ambient"

    # Exits to other zones
    exits: Dict[str, str] = field(default_factory=dict)
    # direction label -> zone_id

    # Zone connections (more detailed than exits)
    connections: List['ZoneConnection'] = field(default_factory=list)

    @property
    def center(self) -> Vector3:
        """Zone center (alias for world_position for backwards compat)."""
        return self.world_position

    @property
    def radius(self) -> float:
        """Zone radius (alias for bounds.radius for backwards compat)."""
        return self.bounds.radius

    def perception_strength(self, local_position: Vector3) -> float:
        """
        Calculate how strongly this zone affects a position (0-1).

        Args:
            local_position: Position relative to zone origin
        """
        dist = local_position.magnitude()

        if dist <= self.bounds.perception_radius:
            return 1.0
        elif dist <= self.bounds.perception_radius + self.bounds.perception_falloff:
            return 1.0 - (dist - self.bounds.perception_radius) / self.bounds.perception_falloff
        else:
            return 0.0

    def local_to_world(self, local_pos: Vector3) -> Vector3:
        """Convert zone-local position to world/stage position."""
        # Apply zone rotation then translation
        rad = math.radians(self.world_rotation)
        rotated_x = local_pos.x * math.cos(rad) - local_pos.z * math.sin(rad)
        rotated_z = local_pos.x * math.sin(rad) + local_pos.z * math.cos(rad)
        return Vector3(
            self.world_position.x + rotated_x,
            self.world_position.y + local_pos.y,
            self.world_position.z + rotated_z
        )

    def world_to_local(self, world_pos: Vector3) -> Vector3:
        """Convert world/stage position to zone-local position."""
        # Translate then inverse rotate
        dx = world_pos.x - self.world_position.x
        dy = world_pos.y - self.world_position.y
        dz = world_pos.z - self.world_position.z
        rad = math.radians(-self.world_rotation)
        return Vector3(
            dx * math.cos(rad) - dz * math.sin(rad),
            dy,
            dx * math.sin(rad) + dz * math.cos(rad)
        )


@dataclass
class ZoneConnection:
    """Connection between two zones."""
    target_zone: str
    direction: str  # "north", "east", "through_door", etc.
    description: str = ""
    # Position of the connection point in zone-local coords
    local_position: Vector3 = field(default_factory=Vector3)
    # Is this a door, path, portal, etc.
    connection_type: str = "path"


# =============================================================================
# Entities
# =============================================================================

@dataclass
class Noodling:
    """A noodling (AI character) in the scene."""
    id: str
    display_name: str
    species: str = ""

    # Spatial - zone-local coordinates
    transform: Transform = field(default_factory=Transform)
    zone: str = ""
    height: float = 1.0  # Character height in meters

    # Visual state
    visual_state: str = "default"
    visual_forms: Dict[str, VisualForm] = field(default_factory=dict)
    posture: str = "standing"
    current_action: str = "idle"
    expression: str = "neutral"
    gaze_target: Optional[str] = None

    # Perception
    perception: PerceptionCone = field(default_factory=PerceptionCone)

    # Internal state
    affect: Affect = field(default_factory=Affect)
    mood_hint: str = ""  # Derived hint for renderer

    # Voice (for audio)
    voice_reference: Optional[str] = None
    voice_description: str = ""

    # Backwards compatibility properties
    @property
    def position(self) -> Vector3:
        return self.transform.position

    @position.setter
    def position(self, value: Vector3):
        self.transform.position = value

    @property
    def rotation(self) -> Vector3:
        return self.transform.rotation

    @rotation.setter
    def rotation(self, value: Vector3):
        self.transform.rotation = value

    @property
    def facing(self) -> Vector3:
        return self.transform.forward

    def get_current_form(self) -> Optional[VisualForm]:
        """Get the currently active visual form."""
        return self.visual_forms.get(self.visual_state)

    def get_reference_image(self, variant: str = "neutral") -> Optional[str]:
        """Get reference image URI for current form."""
        form = self.get_current_form()
        if form and form.reference_images:
            return form.reference_images.get(variant) or \
                   form.reference_images.get("neutral") or \
                   next(iter(form.reference_images.values()), None)
        return None


@dataclass
class Player:
    """A player (human) in the scene."""
    id: str
    display_name: str

    # Spatial - zone-local coordinates
    transform: Transform = field(default_factory=Transform)
    zone: str = ""
    height: float = 1.65  # Default human height

    # Avatar (optional)
    avatar_description: Optional[str] = None
    avatar_reference: Optional[str] = None

    posture: str = "standing"
    current_action: str = "idle"
    gaze_target: Optional[str] = None

    perception: PerceptionCone = field(default_factory=PerceptionCone)

    # Backwards compatibility properties
    @property
    def position(self) -> Vector3:
        return self.transform.position

    @position.setter
    def position(self, value: Vector3):
        self.transform.position = value

    @property
    def rotation(self) -> Vector3:
        return self.transform.rotation

    @rotation.setter
    def rotation(self, value: Vector3):
        self.transform.rotation = value

    @property
    def facing(self) -> Vector3:
        return self.transform.forward


@dataclass
class Prim:
    """
    An interactive object in the scene.

    Prims have transforms with full position/rotation/scale.
    Scale is important for prims (a chair vs a building).
    """
    id: str
    prim_type: str  # "fire", "radio", "poster", etc.

    # Spatial - zone-local coordinates
    transform: Transform = field(default_factory=Transform)
    zone: str = ""
    parent: Optional[str] = None  # ID of containing prim

    description: str = ""
    reference_image: Optional[str] = None

    state: Dict[str, Any] = field(default_factory=dict)
    visual_dynamics: Dict[str, Any] = field(default_factory=dict)
    affordances: List[Affordance] = field(default_factory=list)

    # Physics properties for SPE (semantic, not numerical)
    mass: str = "medium"  # negligible, very_light, light, medium, heavy, very_heavy, immovable
    material: str = "unknown"  # wood, metal, glass, fabric, stone, organic, etc.
    friction: str = "medium"  # slippery, low, medium, high, sticky
    elasticity: str = "normal"  # none, low, normal, high, bouncy
    softness: str = "normal"  # rigid, hard, normal, soft, squishy

    # Backwards compatibility properties
    @property
    def position(self) -> Vector3:
        return self.transform.position

    @position.setter
    def position(self, value: Vector3):
        self.transform.position = value

    @property
    def rotation(self) -> Vector3:
        return self.transform.rotation

    @rotation.setter
    def rotation(self, value: Vector3):
        self.transform.rotation = value

    @property
    def scale(self) -> Vector3:
        return self.transform.scale

    @scale.setter
    def scale(self, value: Vector3):
        self.transform.scale = value

    def apply_material_preset(self, material: str):
        """Apply physics properties from a material preset."""
        preset = MATERIAL_PRESETS.get(material.lower())
        if preset:
            self.material = material
            self.mass = preset.get('mass', self.mass)
            self.friction = preset.get('friction', self.friction)
            self.elasticity = preset.get('elasticity', self.elasticity)
            self.softness = preset.get('softness', self.softness)


# =============================================================================
# Material Physics Presets
# =============================================================================

# Default physics properties for common materials
# Used to quickly configure prims with sensible defaults
# Boulder vs feather: stone/very_heavy vs feather/negligible
MATERIAL_PRESETS: Dict[str, Dict[str, str]] = {
    # Heavy/solid materials
    'stone': {'mass': 'very_heavy', 'friction': 'high', 'elasticity': 'none', 'softness': 'rigid'},
    'boulder': {'mass': 'immovable', 'friction': 'high', 'elasticity': 'none', 'softness': 'rigid'},
    'metal': {'mass': 'heavy', 'friction': 'medium', 'elasticity': 'low', 'softness': 'rigid'},
    'iron': {'mass': 'very_heavy', 'friction': 'medium', 'elasticity': 'low', 'softness': 'rigid'},
    'steel': {'mass': 'very_heavy', 'friction': 'low', 'elasticity': 'low', 'softness': 'rigid'},
    'brick': {'mass': 'heavy', 'friction': 'high', 'elasticity': 'none', 'softness': 'rigid'},
    'concrete': {'mass': 'immovable', 'friction': 'high', 'elasticity': 'none', 'softness': 'rigid'},

    # Medium weight materials
    'wood': {'mass': 'medium', 'friction': 'medium', 'elasticity': 'low', 'softness': 'hard'},
    'glass': {'mass': 'medium', 'friction': 'slippery', 'elasticity': 'none', 'softness': 'rigid'},
    'ceramic': {'mass': 'medium', 'friction': 'medium', 'elasticity': 'none', 'softness': 'rigid'},
    'plastic': {'mass': 'light', 'friction': 'low', 'elasticity': 'normal', 'softness': 'normal'},
    'leather': {'mass': 'light', 'friction': 'high', 'elasticity': 'low', 'softness': 'normal'},

    # Light materials
    'fabric': {'mass': 'very_light', 'friction': 'medium', 'elasticity': 'none', 'softness': 'soft'},
    'paper': {'mass': 'very_light', 'friction': 'low', 'elasticity': 'none', 'softness': 'normal'},
    'cardboard': {'mass': 'light', 'friction': 'medium', 'elasticity': 'none', 'softness': 'normal'},
    'foam': {'mass': 'very_light', 'friction': 'medium', 'elasticity': 'high', 'softness': 'squishy'},
    'rubber': {'mass': 'light', 'friction': 'sticky', 'elasticity': 'bouncy', 'softness': 'soft'},
    'balloon': {'mass': 'negligible', 'friction': 'low', 'elasticity': 'bouncy', 'softness': 'soft'},

    # Ultra-light materials (feather scenario)
    'feather': {'mass': 'negligible', 'friction': 'low', 'elasticity': 'none', 'softness': 'soft'},
    'leaf': {'mass': 'negligible', 'friction': 'low', 'elasticity': 'none', 'softness': 'soft'},
    'dust': {'mass': 'negligible', 'friction': 'low', 'elasticity': 'none', 'softness': 'squishy'},

    # Liquids and special
    'water': {'mass': 'medium', 'friction': 'slippery', 'elasticity': 'none', 'softness': 'squishy'},
    'ice': {'mass': 'medium', 'friction': 'slippery', 'elasticity': 'low', 'softness': 'rigid'},
    'mud': {'mass': 'heavy', 'friction': 'sticky', 'elasticity': 'none', 'softness': 'squishy'},
    'sand': {'mass': 'heavy', 'friction': 'high', 'elasticity': 'none', 'softness': 'soft'},

    # Organic
    'organic': {'mass': 'medium', 'friction': 'medium', 'elasticity': 'low', 'softness': 'soft'},
    'flesh': {'mass': 'medium', 'friction': 'medium', 'elasticity': 'low', 'softness': 'soft'},
    'bone': {'mass': 'medium', 'friction': 'medium', 'elasticity': 'none', 'softness': 'hard'},
    'fur': {'mass': 'very_light', 'friction': 'medium', 'elasticity': 'none', 'softness': 'soft'},
}


# =============================================================================
# Narrative Context
# =============================================================================

@dataclass
class DialogueEntry:
    """A recent piece of dialogue."""
    speaker: str
    text: str
    timestamp: float
    seconds_ago: float = 0.0
    tone: str = "neutral"
    entry_type: str = "speech"  # speech, emote, whisper


@dataclass
class EventEntry:
    """A recent event."""
    actor: str
    action: str
    timestamp: float
    seconds_ago: float = 0.0
    target: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class SceneState:
    """High-level narrative state of the scene."""
    tension: float = 0.0          # 0-1
    energy: float = 0.5           # 0-1
    intimacy: float = 0.5         # 0-1
    humor: float = 0.0            # 0-1
    mystery: float = 0.0          # 0-1

    current_beat: str = "neutral"
    # Examples: "playful_banter", "tense_standoff", "quiet_moment"


@dataclass
class NarrativeContext:
    """The narrative context of what's happening."""
    recent_dialogue: List[DialogueEntry] = field(default_factory=list)
    recent_events: List[EventEntry] = field(default_factory=list)
    scene_state: SceneState = field(default_factory=SceneState)
    context_summary: str = ""  # LLM-readable summary


# =============================================================================
# Camera Directive
# =============================================================================

@dataclass
class FrameSubject:
    """A subject to include in frame."""
    entity: str
    importance: float = 0.5  # 0-1, how important to keep visible


@dataclass
class CameraStyle:
    """Camera style parameters."""
    focal_length: int = 50       # mm equivalent
    dof_mode: str = "medium"     # none, shallow, medium, deep
    dof_focus: str = "subject"   # subject, position, auto

    color_temperature: str = "neutral"  # cool, neutral, warm
    color_grade: str = "natural"        # natural, cinematic, firelight, etc.

    film_grain: float = 0.0      # 0-1
    vignette: float = 0.0        # 0-1
    bloom: float = 0.0           # 0-1


@dataclass
class CameraTransition:
    """Transition from previous shot."""
    transition_type: str = "cut"  # cut, dissolve, fade, wipe
    duration: float = 0.0         # seconds


@dataclass
class CameraDirective:
    """High-level cinematography instructions."""
    mode: CameraMode = CameraMode.FOCUS_ON

    # Primary subject(s)
    subject: Optional[str] = None
    subjects: List[str] = field(default_factory=list)

    # For ESTABLISH mode
    zone: Optional[str] = None

    # For FREE mode
    position: Optional[Vector3] = None
    look_at: Optional[Vector3] = None

    # Framing
    framing: Framing = Framing.MEDIUM
    angle: CameraAngle = CameraAngle.EYE_LEVEL

    # Other subjects to keep visible
    include_in_frame: List[FrameSubject] = field(default_factory=list)

    # Movement
    movement: CameraMovement = CameraMovement.GENTLE_DRIFT

    # Style
    style: CameraStyle = field(default_factory=CameraStyle)

    # Transition
    transition: CameraTransition = field(default_factory=CameraTransition)

    # Hints for cinematic mode
    cinematic_hints: List[str] = field(default_factory=list)


# =============================================================================
# Ambient / World State
# =============================================================================

@dataclass
class AmbientState:
    """World ambient state."""
    time_of_day: str = "day"     # dawn, morning, midday, afternoon, dusk, night
    time_precise: str = ""       # HH:MM
    weather: str = "clear"       # clear, cloudy, fog, rain, storm, snow
    season: str = "summer"
    temperature: str = "mild"    # cold, cool, mild, warm, hot
    lighting_mood: str = "natural"
    soundscape: List[str] = field(default_factory=list)


@dataclass
class SpatialTruth:
    """The canonical spatial representation."""
    coordinate_system: str = "meters"
    up_axis: str = "Y"
    bounds_min: Vector3 = field(default_factory=lambda: Vector3(-100, 0, -100))
    bounds_max: Vector3 = field(default_factory=lambda: Vector3(100, 50, 100))

    ambient: AmbientState = field(default_factory=AmbientState)
    zones: List[Zone] = field(default_factory=list)


# =============================================================================
# Reference Bundle
# =============================================================================

@dataclass
class CharacterReference:
    """Visual reference for a character."""
    form: str
    primary_ref: Optional[str] = None
    primary_ref_base64: Optional[str] = None
    expression_ref: Optional[str] = None
    description: str = ""


@dataclass
class PrimReference:
    """Visual reference for a prim."""
    ref: Optional[str] = None
    description: str = ""


@dataclass
class ReferenceBundle:
    """Pre-packaged visual references for the current frame."""
    characters: Dict[str, CharacterReference] = field(default_factory=dict)
    prims: Dict[str, PrimReference] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Scene Packet Header
# =============================================================================

@dataclass
class PacketHeader:
    """Scene packet header metadata."""
    protocol_version: str = "0.1.0"
    packet_id: str = field(default_factory=lambda: f"pkt_{uuid4().hex[:16]}")
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    timestamp_iso: str = ""

    stage_id: str = ""
    stage_name: str = ""

    packet_type: PacketType = PacketType.FULL
    previous_packet_id: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp_iso:
            self.timestamp_iso = datetime.fromtimestamp(self.timestamp).isoformat()


# =============================================================================
# Scene Packet (Complete)
# =============================================================================

@dataclass
class ScenePacket:
    """
    Complete scene packet - a snapshot of semantic truth.

    This is what gets sent to Genie/Mirage or any renderer.
    """

    header: PacketHeader = field(default_factory=PacketHeader)
    spatial_truth: SpatialTruth = field(default_factory=SpatialTruth)

    # Entities
    noodlings: Dict[str, Noodling] = field(default_factory=dict)
    players: Dict[str, Player] = field(default_factory=dict)
    prims: Dict[str, Prim] = field(default_factory=dict)

    # References
    reference_bundle: ReferenceBundle = field(default_factory=ReferenceBundle)

    # Narrative
    narrative_context: NarrativeContext = field(default_factory=NarrativeContext)

    # Camera
    camera_directive: CameraDirective = field(default_factory=CameraDirective)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return _serialize_dataclass(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenePacket':
        """Deserialize from dictionary."""
        # This would need proper deserialization logic
        # For now, basic implementation
        return cls()

    def flatten_to_text(self) -> str:
        """
        Flatten the scene packet to LLM-readable text.

        This is what Genie's underlying LLM sees for text-to-3D.
        """
        lines = []

        # Header
        lines.append(f"SCENE: {self.header.stage_name}")
        lines.append(f"TIME: {self.spatial_truth.ambient.time_of_day.title()}, "
                     f"{self.spatial_truth.ambient.weather}")
        lines.append("")

        # Location
        if self.spatial_truth.zones:
            primary_zone = self.spatial_truth.zones[0]
            lines.append(f"LOCATION: {primary_zone.name}")
            if primary_zone.description:
                lines.append(primary_zone.description)
            lines.append("")

        # Characters
        if self.noodlings or self.players:
            lines.append("CHARACTERS PRESENT:")
            for nid, noodling in self.noodlings.items():
                form = noodling.get_current_form()
                desc = form.description if form else noodling.species
                lines.append(f"- {noodling.display_name} ({desc[:50]}...): "
                             f"{noodling.posture}, {noodling.expression}")
            for pid, player in self.players.items():
                lines.append(f"- {player.display_name} (player): "
                             f"{player.posture}")
            lines.append("")

        # Recent events
        if self.narrative_context.recent_dialogue:
            lines.append("JUST HAPPENED:")
            for d in self.narrative_context.recent_dialogue[:3]:
                lines.append(f"- {d.speaker} said: \"{d.text}\"")
            lines.append("")

        # Mood
        state = self.narrative_context.scene_state
        lines.append(f"MOOD: {state.current_beat.replace('_', ' ').title()}, "
                     f"tension={state.tension:.1f}, energy={state.energy:.1f}")

        # Camera
        cam = self.camera_directive
        lines.append("")
        lines.append(f"CAMERA: {cam.mode.value} on {cam.subject or 'scene'}, "
                     f"{cam.framing.value}, {cam.style.color_grade}")

        return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================

def _serialize_dataclass(obj: Any) -> Any:
    """Recursively serialize a dataclass to dict."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _serialize_dataclass(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [_serialize_dataclass(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: _serialize_dataclass(v) for k, v in obj.items()}
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "PacketType",
    "CameraMode",
    "Framing",
    "CameraAngle",
    "CameraMovement",

    # Core types
    "Vector3",
    "PerceptionCone",
    "Affect",
    "VisualForm",
    "Affordance",

    # Spatial
    "Zone",
    "AmbientState",
    "SpatialTruth",

    # Entities
    "Noodling",
    "Player",
    "Prim",

    # Narrative
    "DialogueEntry",
    "EventEntry",
    "SceneState",
    "NarrativeContext",

    # Camera
    "FrameSubject",
    "CameraStyle",
    "CameraTransition",
    "CameraDirective",

    # References
    "CharacterReference",
    "PrimReference",
    "ReferenceBundle",

    # Packet
    "PacketHeader",
    "ScenePacket",
]
