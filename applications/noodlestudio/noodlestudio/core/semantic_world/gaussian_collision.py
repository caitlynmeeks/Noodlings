# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Gaussian Collision Detection - Detect when semantic Gaussians touch.
#
#   Every Gaussian knows its position in space. This module p...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.semantic_world.gaussian_collision
# PURPOSE:  Gaussian Collision
# LAYER:    Studio / Semantic World
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TouchType, TouchRegion, TouchEvent, AffectImpulse, GaussianCollisionDetector
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Touch Event Types
# =============================================================================

class TouchType(Enum):
    """Classification of touch interactions."""
    CONTACT = "contact"           # Generic touch
    GENTLE = "gentle"             # Low intensity, soft touch
    FIRM = "firm"                 # Medium intensity
    IMPACT = "impact"             # High intensity, collision
    SUSTAINED = "sustained"       # Ongoing contact


class TouchRegion(Enum):
    """Body region involved in touch (for affect mapping)."""
    HEAD = "head"
    FACE = "face"
    TORSO = "torso"
    ARM = "arm"
    HAND = "hand"
    LEG = "leg"
    FOOT = "foot"
    TAIL = "tail"
    OTHER = "other"


@dataclass
class TouchEvent:
    """
    A detected touch between two entities.

    Contains full semantic information about what touched what,
    enabling rich affect responses.
    """
    # Entities involved
    entity_a: str                              # Entity ID
    entity_b: str                              # Entity ID (or "environment")

    # Body parts
    body_part_a: str                           # e.g., "right_hand"
    body_part_b: str                           # e.g., "left_shoulder"
    region_a: TouchRegion = TouchRegion.OTHER
    region_b: TouchRegion = TouchRegion.OTHER

    # Spatial
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Contact point
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)    # Contact normal

    # Intensity
    overlap_integral: float = 0.0              # Raw overlap value
    intensity: float = 0.0                     # Normalized 0-1
    touch_type: TouchType = TouchType.CONTACT

    # Timing
    timestamp: float = 0.0
    duration: float = 0.0                      # For sustained touches

    # Gaussian indices (for debugging/visualization)
    gaussian_a: int = -1
    gaussian_b: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'entity_a': self.entity_a,
            'entity_b': self.entity_b,
            'body_part_a': self.body_part_a,
            'body_part_b': self.body_part_b,
            'region_a': self.region_a.value,
            'region_b': self.region_b.value,
            'position': list(self.position),
            'normal': list(self.normal),
            'intensity': self.intensity,
            'touch_type': self.touch_type.value,
            'timestamp': self.timestamp,
            'duration': self.duration,
        }

    def description(self) -> str:
        """Human-readable description."""
        return (f"{self.entity_a}'s {self.body_part_a} touched "
                f"{self.entity_b}'s {self.body_part_b} "
                f"({self.touch_type.value}, intensity={self.intensity:.2f})")


@dataclass
class AffectImpulse:
    """
    An affect change triggered by a touch event.

    Maps physical interaction to emotional response.
    """
    source: str = "physics"
    source_event: Optional[TouchEvent] = None

    # PAD changes (deltas, not absolute values)
    valence_delta: float = 0.0      # Pleasure change
    arousal_delta: float = 0.0      # Energy change
    dominance_delta: float = 0.0    # Control change

    # Additional affect dimensions
    startle: float = 0.0            # Surprise/startle response
    comfort: float = 0.0            # Comfort/soothing

    # Decay
    decay_rate: float = 0.5         # How fast this impulse fades

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'valence_delta': self.valence_delta,
            'arousal_delta': self.arousal_delta,
            'dominance_delta': self.dominance_delta,
            'startle': self.startle,
            'comfort': self.comfort,
            'decay_rate': self.decay_rate,
        }


# =============================================================================
# Gaussian Math
# =============================================================================

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)


def build_covariance_matrix(scale: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Build 3x3 covariance matrix from Gaussian scale and rotation.

    The covariance defines the shape/orientation of the Gaussian ellipsoid.
    Sigma = R @ S @ S @ R.T  where S is diagonal scale matrix
    """
    # Scale matrix (squared because covariance uses variance)
    S = np.diag(scale.astype(np.float64) ** 2)

    # Rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(rotation)

    # Covariance: R @ S @ R.T
    return R @ S @ R.T


def gaussian_overlap_integral(
    pos1: np.ndarray, cov1: np.ndarray,
    pos2: np.ndarray, cov2: np.ndarray
) -> float:
    """
    Compute the overlap integral of two 3D Gaussians.

    This is the closed-form solution for:
        integral of G1(x) * G2(x) over all R^3

    For two normalized Gaussians, the result is:
        sqrt(det(Σ1) * det(Σ2) / det(Σ1 + Σ2)) * exp(-0.5 * d_mahal^2)

    Where d_mahal is the Mahalanobis distance between centers
    using the combined covariance.

    Returns:
        Overlap value in [0, 1] range (1 = identical Gaussians)
    """
    # Combined covariance
    cov_sum = cov1 + cov2

    # Check for singularity
    det_sum = np.linalg.det(cov_sum)
    if det_sum < 1e-10:
        return 0.0

    # Determinants
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)

    if det1 < 1e-10 or det2 < 1e-10:
        return 0.0

    # Determinant ratio
    det_ratio = (det1 * det2) / det_sum

    # Mahalanobis distance squared
    mu_diff = pos1 - pos2
    try:
        cov_sum_inv = np.linalg.inv(cov_sum)
        d_sq = mu_diff @ cov_sum_inv @ mu_diff
    except np.linalg.LinAlgError:
        return 0.0

    # Overlap integral
    # The (2*pi)^(3/2) factors cancel out for normalized Gaussians
    overlap = np.sqrt(det_ratio) * np.exp(-0.5 * d_sq)

    # Normalize to [0, 1] range
    # Maximum overlap is 1 when Gaussians are identical
    return float(np.clip(overlap, 0.0, 1.0))


def sphere_approximation_touch(
    pos1: np.ndarray, scale1: np.ndarray,
    pos2: np.ndarray, scale2: np.ndarray,
    margin: float = 0.0
) -> Tuple[bool, float]:
    """
    Fast sphere approximation for Gaussian touch detection.

    Treats each Gaussian as a sphere with radius = max(scale).
    Much faster than full overlap integral.

    Returns:
        (touching, distance) tuple
    """
    r1 = float(np.max(scale1))
    r2 = float(np.max(scale2))
    dist = float(np.linalg.norm(pos1 - pos2))

    touching = dist < (r1 + r2 + margin)
    return touching, dist


# =============================================================================
# Collision Detector
# =============================================================================

class GaussianCollisionDetector:
    """
    Detects collisions/touches between semantic Gaussian entities.

    Maintains a registry of entities and their Gaussian data,
    then provides efficient collision queries.
    """

    def __init__(self,
                 overlap_threshold: float = 0.05,
                 use_spatial_hash: bool = True,
                 hash_cell_size: float = 0.5):
        """
        Initialize detector.

        Args:
            overlap_threshold: Minimum overlap integral to count as touch
            use_spatial_hash: Use spatial hashing for broad phase
            hash_cell_size: Size of spatial hash cells
        """
        self.overlap_threshold = overlap_threshold
        self.use_spatial_hash = use_spatial_hash
        self.hash_cell_size = hash_cell_size

        # Entity registry
        # entity_id -> {positions, scales, rotations, covariances, labels, regions}
        self.entities: Dict[str, Dict[str, Any]] = {}

        # Spatial hash for broad phase
        self._spatial_hash: Dict[Tuple[int, int, int], List[Tuple[str, int]]] = {}

        # Touch state tracking (for sustained touches)
        self._active_touches: Dict[str, TouchEvent] = {}

        # Callbacks
        self.on_touch_start: Optional[Callable[[TouchEvent], None]] = None
        self.on_touch_end: Optional[Callable[[TouchEvent], None]] = None

    def add_entity(self, entity_id: str, asset: 'RadianceAsset'):
        """
        Add an entity to the collision system.

        Args:
            entity_id: Unique identifier
            asset: RadianceAsset with Gaussian data
        """
        if asset.positions is None:
            logger.warning(f"Cannot add entity {entity_id}: no position data")
            return

        n = asset.gaussian_count

        # Precompute covariance matrices
        covariances = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            scale = asset.scales[i] if asset.scales is not None else np.array([0.01, 0.01, 0.01])
            rotation = asset.rotations[i] if asset.rotations is not None else np.array([0, 0, 0, 1])
            covariances[i] = build_covariance_matrix(scale, rotation)

        self.entities[entity_id] = {
            'positions': asset.positions.astype(np.float64),
            'scales': asset.scales.astype(np.float64) if asset.scales is not None else np.ones((n, 3)) * 0.01,
            'rotations': asset.rotations.astype(np.float64) if asset.rotations is not None else np.tile([0, 0, 0, 1], (n, 1)),
            'covariances': covariances,
            'labels': asset.semantic_labels if asset.semantic_labels else [''] * n,
            'regions': [asset.get_body_region(i) for i in range(n)],
            'display_name': asset.metadata.display_name,
        }

        # Update spatial hash
        if self.use_spatial_hash:
            self._update_spatial_hash(entity_id)

        logger.debug(f"Added entity {entity_id} with {n} Gaussians")

    def update_entity_positions(self, entity_id: str, positions: np.ndarray):
        """
        Update positions for an entity (after animation/physics).

        Args:
            entity_id: Entity to update
            positions: New positions (N, 3)
        """
        if entity_id not in self.entities:
            return

        self.entities[entity_id]['positions'] = positions.astype(np.float64)

        if self.use_spatial_hash:
            self._update_spatial_hash(entity_id)

    def remove_entity(self, entity_id: str):
        """Remove an entity from collision system."""
        if entity_id in self.entities:
            del self.entities[entity_id]
            self._rebuild_spatial_hash()

    def _update_spatial_hash(self, entity_id: str):
        """Update spatial hash for an entity."""
        # Remove old entries
        to_remove = []
        for cell, entries in self._spatial_hash.items():
            self._spatial_hash[cell] = [(eid, idx) for eid, idx in entries if eid != entity_id]
            if not self._spatial_hash[cell]:
                to_remove.append(cell)
        for cell in to_remove:
            del self._spatial_hash[cell]

        # Add new entries
        entity = self.entities[entity_id]
        for i, pos in enumerate(entity['positions']):
            cell = self._position_to_cell(pos)
            if cell not in self._spatial_hash:
                self._spatial_hash[cell] = []
            self._spatial_hash[cell].append((entity_id, i))

    def _rebuild_spatial_hash(self):
        """Rebuild entire spatial hash."""
        self._spatial_hash.clear()
        for entity_id in self.entities:
            self._update_spatial_hash(entity_id)

    def _position_to_cell(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert position to spatial hash cell."""
        return (
            int(pos[0] / self.hash_cell_size),
            int(pos[1] / self.hash_cell_size),
            int(pos[2] / self.hash_cell_size),
        )

    def _get_nearby_cells(self, cell: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get cell and all 26 neighbors."""
        cx, cy, cz = cell
        cells = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    cells.append((cx + dx, cy + dy, cz + dz))
        return cells

    def detect_touches(self,
                       entity_a: Optional[str] = None,
                       entity_b: Optional[str] = None,
                       threshold: Optional[float] = None,
                       max_touches: int = 100) -> List[TouchEvent]:
        """
        Detect touch events between entities.

        Args:
            entity_a: Specific entity to check (None = all)
            entity_b: Specific entity to check against (None = all)
            threshold: Override overlap threshold
            max_touches: Maximum touches to return

        Returns:
            List of TouchEvent objects
        """
        threshold = threshold if threshold is not None else self.overlap_threshold
        touches = []
        timestamp = time.time()

        # Determine which entity pairs to check
        if entity_a and entity_b:
            pairs = [(entity_a, entity_b)]
        elif entity_a:
            pairs = [(entity_a, eid) for eid in self.entities if eid != entity_a]
        elif entity_b:
            pairs = [(eid, entity_b) for eid in self.entities if eid != entity_b]
        else:
            # All pairs
            entity_ids = list(self.entities.keys())
            pairs = [(entity_ids[i], entity_ids[j])
                    for i in range(len(entity_ids))
                    for j in range(i+1, len(entity_ids))]

        for eid_a, eid_b in pairs:
            if eid_a not in self.entities or eid_b not in self.entities:
                continue

            pair_touches = self._detect_pair_touches(eid_a, eid_b, threshold, timestamp)
            touches.extend(pair_touches)

            if len(touches) >= max_touches:
                break

        # Sort by intensity
        touches.sort(key=lambda t: t.intensity, reverse=True)
        return touches[:max_touches]

    def _detect_pair_touches(self, eid_a: str, eid_b: str,
                             threshold: float, timestamp: float) -> List[TouchEvent]:
        """Detect touches between two specific entities."""
        touches = []

        entity_a = self.entities[eid_a]
        entity_b = self.entities[eid_b]

        pos_a = entity_a['positions']
        pos_b = entity_b['positions']
        scales_a = entity_a['scales']
        scales_b = entity_b['scales']
        cov_a = entity_a['covariances']
        cov_b = entity_b['covariances']

        # Broad phase: use spatial hash or brute force
        if self.use_spatial_hash:
            candidate_pairs = self._get_candidate_pairs_spatial(eid_a, eid_b)
        else:
            candidate_pairs = [(i, j) for i in range(len(pos_a)) for j in range(len(pos_b))]

        # Narrow phase: compute actual overlaps
        for i, j in candidate_pairs:
            # Quick sphere check first
            touching, dist = sphere_approximation_touch(
                pos_a[i], scales_a[i],
                pos_b[j], scales_b[j],
                margin=0.05
            )

            if not touching:
                continue

            # Full overlap computation
            overlap = gaussian_overlap_integral(
                pos_a[i], cov_a[i],
                pos_b[j], cov_b[j]
            )

            if overlap < threshold:
                continue

            # Classify touch type
            touch_type = self._classify_touch(overlap)

            # Get labels and regions
            label_a = entity_a['labels'][i] if i < len(entity_a['labels']) else ""
            label_b = entity_b['labels'][j] if j < len(entity_b['labels']) else ""
            region_a = self._string_to_touch_region(entity_a['regions'][i])
            region_b = self._string_to_touch_region(entity_b['regions'][j])

            # Contact point (midpoint weighted by inverse scale)
            contact_point = (pos_a[i] + pos_b[j]) / 2

            # Contact normal (direction from a to b)
            direction = pos_b[j] - pos_a[i]
            dist_norm = np.linalg.norm(direction)
            normal = direction / dist_norm if dist_norm > 1e-6 else np.array([0, 1, 0])

            touch = TouchEvent(
                entity_a=eid_a,
                entity_b=eid_b,
                body_part_a=label_a,
                body_part_b=label_b,
                region_a=region_a,
                region_b=region_b,
                position=tuple(contact_point),
                normal=tuple(normal),
                overlap_integral=overlap,
                intensity=min(1.0, overlap / 0.5),  # Normalize to 0-1
                touch_type=touch_type,
                timestamp=timestamp,
                gaussian_a=i,
                gaussian_b=j,
            )

            touches.append(touch)

        return touches

    def _get_candidate_pairs_spatial(self, eid_a: str, eid_b: str) -> List[Tuple[int, int]]:
        """Get candidate Gaussian pairs using spatial hash."""
        candidates = set()

        entity_a = self.entities[eid_a]

        for i, pos in enumerate(entity_a['positions']):
            cell = self._position_to_cell(pos)
            nearby_cells = self._get_nearby_cells(cell)

            for nearby_cell in nearby_cells:
                if nearby_cell not in self._spatial_hash:
                    continue

                for other_eid, j in self._spatial_hash[nearby_cell]:
                    if other_eid == eid_b:
                        candidates.add((i, j))

        return list(candidates)

    def _classify_touch(self, overlap: float) -> TouchType:
        """Classify touch type based on overlap intensity."""
        if overlap > 0.4:
            return TouchType.IMPACT
        elif overlap > 0.2:
            return TouchType.FIRM
        elif overlap > 0.1:
            return TouchType.CONTACT
        else:
            return TouchType.GENTLE

    def _string_to_touch_region(self, region_str: str) -> TouchRegion:
        """Convert region string to TouchRegion enum."""
        region_map = {
            'head': TouchRegion.HEAD,
            'face': TouchRegion.FACE,
            'torso': TouchRegion.TORSO,
            'left_arm': TouchRegion.ARM,
            'right_arm': TouchRegion.ARM,
            'left_hand': TouchRegion.HAND,
            'right_hand': TouchRegion.HAND,
            'left_leg': TouchRegion.LEG,
            'right_leg': TouchRegion.LEG,
            'left_foot': TouchRegion.FOOT,
            'right_foot': TouchRegion.FOOT,
            'tail': TouchRegion.TAIL,
        }
        return region_map.get(region_str, TouchRegion.OTHER)


# =============================================================================
# Affect Impulse Generator
# =============================================================================

class TouchAffectMapper:
    """
    Maps touch events to affect impulses.

    Different body regions and touch types produce different
    emotional responses.
    """

    def __init__(self):
        # Base affect responses by touch region
        # (valence, arousal, dominance, startle, comfort)
        self.region_responses = {
            TouchRegion.HEAD: (0.1, 0.2, -0.1, 0.3, 0.2),    # Head touches: intimate, startling
            TouchRegion.FACE: (0.2, 0.3, -0.2, 0.4, 0.1),    # Face: very intimate
            TouchRegion.HAND: (0.3, 0.1, 0.0, 0.1, 0.4),     # Hand: social, comforting
            TouchRegion.ARM: (0.1, 0.1, 0.0, 0.1, 0.2),      # Arm: neutral
            TouchRegion.TORSO: (0.0, 0.2, -0.1, 0.2, 0.1),   # Torso: vulnerable
            TouchRegion.LEG: (-0.1, 0.1, -0.1, 0.2, 0.0),    # Leg: unexpected
            TouchRegion.FOOT: (-0.1, 0.1, -0.1, 0.2, 0.0),   # Foot: unexpected
            TouchRegion.TAIL: (0.2, 0.3, 0.0, 0.2, 0.3),     # Tail: playful (for tailed creatures)
            TouchRegion.OTHER: (0.0, 0.1, 0.0, 0.1, 0.0),    # Other: minimal
        }

        # Multipliers by touch type
        self.type_multipliers = {
            TouchType.GENTLE: {'intensity': 0.5, 'startle': 0.3, 'comfort': 1.5},
            TouchType.CONTACT: {'intensity': 1.0, 'startle': 0.7, 'comfort': 1.0},
            TouchType.FIRM: {'intensity': 1.5, 'startle': 1.0, 'comfort': 0.5},
            TouchType.IMPACT: {'intensity': 2.0, 'startle': 2.0, 'comfort': 0.0},
            TouchType.SUSTAINED: {'intensity': 0.8, 'startle': 0.2, 'comfort': 1.2},
        }

        # Relationship modifier (would be populated from social memory)
        self.relationship_modifiers: Dict[Tuple[str, str], float] = {}

    def generate_impulse(self, touch: TouchEvent,
                         relationship_valence: float = 0.0) -> AffectImpulse:
        """
        Generate an affect impulse from a touch event.

        Args:
            touch: The touch event
            relationship_valence: How positively the toucher is viewed (-1 to 1)

        Returns:
            AffectImpulse to inject into affect system
        """
        # Get base response for touched region
        base = self.region_responses.get(touch.region_b, self.region_responses[TouchRegion.OTHER])
        base_valence, base_arousal, base_dominance, base_startle, base_comfort = base

        # Get type multipliers
        mults = self.type_multipliers.get(touch.touch_type, self.type_multipliers[TouchType.CONTACT])

        # Calculate final values
        intensity = touch.intensity * mults['intensity']

        # Valence is affected by relationship
        # Positive relationship = touch feels good, negative = touch feels bad
        valence_mod = relationship_valence * 0.5
        valence_delta = base_valence * intensity + valence_mod * intensity

        # Impact type flips valence negative (collision = bad)
        if touch.touch_type == TouchType.IMPACT:
            valence_delta = -abs(valence_delta) - 0.2 * intensity

        arousal_delta = base_arousal * intensity
        dominance_delta = base_dominance * intensity

        startle = base_startle * intensity * mults['startle']
        comfort = base_comfort * intensity * mults['comfort']

        # Strangers touching sensitive areas = more startle, less comfort
        if relationship_valence < 0:
            startle *= 1.5
            comfort *= 0.5

        return AffectImpulse(
            source="touch",
            source_event=touch,
            valence_delta=float(np.clip(valence_delta, -1, 1)),
            arousal_delta=float(np.clip(arousal_delta, 0, 1)),
            dominance_delta=float(np.clip(dominance_delta, -1, 1)),
            startle=float(np.clip(startle, 0, 1)),
            comfort=float(np.clip(comfort, 0, 1)),
            decay_rate=0.3 if touch.touch_type == TouchType.IMPACT else 0.5,
        )

    def set_relationship(self, entity_a: str, entity_b: str, valence: float):
        """Set relationship valence between two entities."""
        self.relationship_modifiers[(entity_a, entity_b)] = valence
        self.relationship_modifiers[(entity_b, entity_a)] = valence

    def get_relationship(self, entity_a: str, entity_b: str) -> float:
        """Get relationship valence (default 0 = neutral)."""
        return self.relationship_modifiers.get((entity_a, entity_b), 0.0)


# =============================================================================
# Physics Event Bus
# =============================================================================

class PhysicsEventBus:
    """
    Event bus for physics-related events (collisions, touches, forces).

    Allows decoupled communication between physics system and affect system.
    """

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {
            'touch_start': [],
            'touch_end': [],
            'touch_update': [],
            'collision': [],
            'impulse': [],
        }

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type."""
        if event_type in self._listeners:
            self._listeners[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type."""
        if event_type in self._listeners and callback in self._listeners[event_type]:
            self._listeners[event_type].remove(callback)

    def emit(self, event_type: str, data: Any):
        """Emit an event."""
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in physics event handler: {e}")

    def emit_touch(self, touch: TouchEvent, is_start: bool = True):
        """Emit a touch event."""
        self.emit('touch_start' if is_start else 'touch_end', touch)

    def emit_impulse(self, impulse: AffectImpulse):
        """Emit an affect impulse."""
        self.emit('impulse', impulse)


# =============================================================================
# Module Interface
# =============================================================================

# Global instances
_detector: Optional[GaussianCollisionDetector] = None
_affect_mapper: Optional[TouchAffectMapper] = None
_event_bus: Optional[PhysicsEventBus] = None


def init_collision_system(overlap_threshold: float = 0.05):
    """Initialize the collision detection system."""
    global _detector, _affect_mapper, _event_bus

    _detector = GaussianCollisionDetector(overlap_threshold=overlap_threshold)
    _affect_mapper = TouchAffectMapper()
    _event_bus = PhysicsEventBus()

    logger.info("Collision system initialized")


def get_detector() -> Optional[GaussianCollisionDetector]:
    """Get the collision detector."""
    return _detector


def get_affect_mapper() -> Optional[TouchAffectMapper]:
    """Get the affect mapper."""
    return _affect_mapper


def get_physics_event_bus() -> Optional[PhysicsEventBus]:
    """Get the physics event bus."""
    return _event_bus


def detect_and_emit_touches(entity_a: Optional[str] = None,
                            entity_b: Optional[str] = None) -> List[TouchEvent]:
    """
    Detect touches and emit events + affect impulses.

    Convenience function that:
    1. Detects touches
    2. Emits touch events
    3. Generates affect impulses
    4. Emits impulse events

    Returns list of detected touches.
    """
    if not _detector or not _affect_mapper or not _event_bus:
        return []

    touches = _detector.detect_touches(entity_a, entity_b)

    for touch in touches:
        _event_bus.emit_touch(touch, is_start=True)

        # Generate and emit affect impulse
        relationship = _affect_mapper.get_relationship(touch.entity_a, touch.entity_b)
        impulse = _affect_mapper.generate_impulse(touch, relationship)
        _event_bus.emit_impulse(impulse)

    return touches


__all__ = [
    # Events
    'TouchEvent',
    'TouchType',
    'TouchRegion',
    'AffectImpulse',

    # Detection
    'GaussianCollisionDetector',
    'gaussian_overlap_integral',
    'sphere_approximation_touch',
    'build_covariance_matrix',

    # Affect mapping
    'TouchAffectMapper',

    # Event bus
    'PhysicsEventBus',

    # Module interface
    'init_collision_system',
    'get_detector',
    'get_affect_mapper',
    'get_physics_event_bus',
    'detect_and_emit_touches',
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
