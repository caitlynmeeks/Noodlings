"""
Auto-Rigger - Mixamo-style automatic rigging for arbitrary meshes.

Takes an unskinned mesh and produces a rigged Gaussian avatar by:
1. User places markers (or auto-detect extremities)
2. Fit humanoid skeleton template to markers
3. Sample Gaussians from mesh surface
4. Weight Gaussians directly to bones (skip mesh skinning)

This is cleaner than traditional rigging because we never create
intermediate mesh skinning - Gaussians get bone weights directly.

Usage:
    rigger = AutoRigger()
    rigger.load_mesh('/path/to/conker.obj')

    # Option A: Auto-detect markers
    markers = rigger.auto_detect_markers()

    # Option B: Manual markers (from UI)
    markers = MarkerSet()
    markers.hips = (0, 1.0, 0)
    markers.head = (0, 1.8, 0)
    # ... etc

    # Fit and rig
    result = rigger.rig(markers)
    # -> Writes .radiance with skeleton

Author: Caitlyn + Claude
Date: December 24, 2025
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Marker System
# =============================================================================

@dataclass
class MarkerSet:
    """
    Anatomical markers for skeleton fitting.

    Minimum required: hips, head, left_hand, right_hand, left_foot, right_foot
    Optional markers improve fit quality.
    """
    # Required (6 markers minimum)
    hips: Optional[Tuple[float, float, float]] = None
    head: Optional[Tuple[float, float, float]] = None
    left_hand: Optional[Tuple[float, float, float]] = None
    right_hand: Optional[Tuple[float, float, float]] = None
    left_foot: Optional[Tuple[float, float, float]] = None
    right_foot: Optional[Tuple[float, float, float]] = None

    # Optional (improve fit)
    neck: Optional[Tuple[float, float, float]] = None
    chest: Optional[Tuple[float, float, float]] = None
    left_shoulder: Optional[Tuple[float, float, float]] = None
    right_shoulder: Optional[Tuple[float, float, float]] = None
    left_elbow: Optional[Tuple[float, float, float]] = None
    right_elbow: Optional[Tuple[float, float, float]] = None
    left_knee: Optional[Tuple[float, float, float]] = None
    right_knee: Optional[Tuple[float, float, float]] = None

    # Tail (for non-humanoid characters like Conker)
    tail_base: Optional[Tuple[float, float, float]] = None
    tail_tip: Optional[Tuple[float, float, float]] = None

    def validate(self) -> Tuple[bool, str]:
        """Check if minimum markers are present."""
        required = ['hips', 'head', 'left_hand', 'right_hand', 'left_foot', 'right_foot']
        missing = [m for m in required if getattr(self, m) is None]
        if missing:
            return False, f"Missing required markers: {', '.join(missing)}"
        return True, "OK"

    def to_dict(self) -> Dict[str, Tuple[float, float, float]]:
        """Export non-None markers as dict."""
        result = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                if value is not None and isinstance(value, tuple):
                    result[attr] = value
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MarkerSet':
        """Create from dict."""
        markers = cls()
        for key, value in d.items():
            if hasattr(markers, key):
                if isinstance(value, (list, tuple)) and len(value) == 3:
                    setattr(markers, key, tuple(value))
        return markers


# =============================================================================
# Humanoid Skeleton Template
# =============================================================================

@dataclass
class BoneTemplate:
    """Template for a single bone."""
    name: str
    humanoid_name: str  # Standard name like 'leftUpperArm'
    parent: str  # Parent bone name, '' for root
    local_offset: Tuple[float, float, float]  # Offset from parent
    length_ratio: float  # Length relative to total height


# Default humanoid skeleton proportions (normalized to height=1.0)
# Based on standard anatomical proportions
HUMANOID_SKELETON_TEMPLATE = [
    # Core
    BoneTemplate('Hips', 'hips', '', (0, 0.52, 0), 0.0),
    BoneTemplate('Spine', 'spine', 'Hips', (0, 0.06, 0), 0.06),
    BoneTemplate('Chest', 'chest', 'Spine', (0, 0.08, 0), 0.08),
    BoneTemplate('UpperChest', 'upperChest', 'Chest', (0, 0.06, 0), 0.06),
    BoneTemplate('Neck', 'neck', 'UpperChest', (0, 0.04, 0), 0.04),
    BoneTemplate('Head', 'head', 'Neck', (0, 0.04, 0), 0.14),

    # Left Arm
    BoneTemplate('LeftShoulder', 'leftShoulder', 'UpperChest', (0.04, 0.02, 0), 0.06),
    BoneTemplate('LeftUpperArm', 'leftUpperArm', 'LeftShoulder', (0.06, -0.02, 0), 0.15),
    BoneTemplate('LeftLowerArm', 'leftLowerArm', 'LeftUpperArm', (0.14, 0, 0), 0.14),
    BoneTemplate('LeftHand', 'leftHand', 'LeftLowerArm', (0.12, 0, 0), 0.08),

    # Right Arm
    BoneTemplate('RightShoulder', 'rightShoulder', 'UpperChest', (-0.04, 0.02, 0), 0.06),
    BoneTemplate('RightUpperArm', 'rightUpperArm', 'RightShoulder', (-0.06, -0.02, 0), 0.15),
    BoneTemplate('RightLowerArm', 'rightLowerArm', 'RightUpperArm', (-0.14, 0, 0), 0.14),
    BoneTemplate('RightHand', 'rightHand', 'RightLowerArm', (-0.12, 0, 0), 0.08),

    # Left Leg
    BoneTemplate('LeftUpperLeg', 'leftUpperLeg', 'Hips', (0.08, -0.04, 0), 0.24),
    BoneTemplate('LeftLowerLeg', 'leftLowerLeg', 'LeftUpperLeg', (0, -0.24, 0), 0.24),
    BoneTemplate('LeftFoot', 'leftFoot', 'LeftLowerLeg', (0, -0.24, 0.04), 0.08),
    BoneTemplate('LeftToes', 'leftToes', 'LeftFoot', (0, 0, 0.08), 0.04),

    # Right Leg
    BoneTemplate('RightUpperLeg', 'rightUpperLeg', 'Hips', (-0.08, -0.04, 0), 0.24),
    BoneTemplate('RightLowerLeg', 'rightLowerLeg', 'RightUpperLeg', (0, -0.24, 0), 0.24),
    BoneTemplate('RightFoot', 'rightFoot', 'RightLowerLeg', (0, -0.24, 0.04), 0.08),
    BoneTemplate('RightToes', 'rightToes', 'RightFoot', (0, 0, 0.08), 0.04),
]


@dataclass
class FittedBone:
    """A bone positioned in world space after fitting."""
    name: str
    humanoid_name: str
    parent_index: int
    head_position: np.ndarray  # (3,) world position of bone head
    tail_position: np.ndarray  # (3,) world position of bone tail
    local_rotation: np.ndarray  # (4,) quaternion xyzw

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.tail_position - self.head_position))

    @property
    def direction(self) -> np.ndarray:
        d = self.tail_position - self.head_position
        length = np.linalg.norm(d)
        return d / length if length > 1e-6 else np.array([0, 1, 0])


# =============================================================================
# Skeleton Fitter
# =============================================================================

class SkeletonFitter:
    """
    Fit humanoid skeleton template to marker positions.

    Uses a combination of:
    - Direct positioning from markers
    - Proportional interpolation for unmarked joints
    - Simple IK for arm/leg chains
    """

    def __init__(self, template: List[BoneTemplate] = None):
        self.template = template or HUMANOID_SKELETON_TEMPLATE
        self.bone_name_to_template = {b.name: b for b in self.template}

    def fit(self, markers: MarkerSet, mesh_bounds: Tuple[np.ndarray, np.ndarray] = None) -> List[FittedBone]:
        """
        Fit skeleton to markers.

        Args:
            markers: User-placed or auto-detected markers
            mesh_bounds: Optional (min, max) bounds for scale reference

        Returns:
            List of FittedBone with world positions
        """
        valid, msg = markers.validate()
        if not valid:
            raise ValueError(msg)

        # Compute scale from markers
        hips = np.array(markers.hips)
        head = np.array(markers.head)
        height = abs(head[1] - markers.left_foot[1]) if markers.left_foot else abs(head[1] - hips[1]) * 2

        # Build fitted skeleton
        fitted_bones: List[FittedBone] = []
        bone_name_to_index: Dict[str, int] = {}
        bone_positions: Dict[str, np.ndarray] = {}

        # First pass: position bones from markers and proportions
        for i, template in enumerate(self.template):
            parent_idx = bone_name_to_index.get(template.parent, -1)

            # Get position from marker or compute from parent
            position = self._get_bone_position(template, markers, bone_positions, height)
            bone_positions[template.name] = position
            bone_name_to_index[template.name] = i

            # Compute tail position (will be refined in second pass)
            tail = position + np.array([0, template.length_ratio * height, 0])

            fitted_bones.append(FittedBone(
                name=template.name,
                humanoid_name=template.humanoid_name,
                parent_index=parent_idx,
                head_position=position.copy(),
                tail_position=tail,
                local_rotation=np.array([0, 0, 0, 1], dtype=np.float32),
            ))

        # Second pass: refine tail positions to connect hierarchy
        for i, bone in enumerate(fitted_bones):
            # Find children and set tail to average child position
            children = [b for b in fitted_bones if b.parent_index == i]
            if children:
                avg_child_pos = np.mean([c.head_position for c in children], axis=0)
                bone.tail_position = avg_child_pos

        # Third pass: IK for limbs using hand/foot markers
        self._apply_limb_ik(fitted_bones, bone_name_to_index, markers)

        # Compute local rotations
        self._compute_local_rotations(fitted_bones)

        logger.info(f"Fitted {len(fitted_bones)} bones to markers (height={height:.3f})")
        return fitted_bones

    def _get_bone_position(
        self,
        template: BoneTemplate,
        markers: MarkerSet,
        bone_positions: Dict[str, np.ndarray],
        height: float
    ) -> np.ndarray:
        """Get bone position from marker or compute from template."""

        # Direct marker mapping
        marker_map = {
            'Hips': 'hips',
            'Head': 'head',
            'Neck': 'neck',
            'Chest': 'chest',
            'LeftShoulder': 'left_shoulder',
            'RightShoulder': 'right_shoulder',
            'LeftHand': 'left_hand',
            'RightHand': 'right_hand',
            'LeftUpperArm': 'left_shoulder',  # Approximate
            'RightUpperArm': 'right_shoulder',
            'LeftLowerArm': 'left_elbow',
            'RightLowerArm': 'right_elbow',
            'LeftUpperLeg': 'hips',  # Will be offset
            'RightUpperLeg': 'hips',
            'LeftLowerLeg': 'left_knee',
            'RightLowerLeg': 'right_knee',
            'LeftFoot': 'left_foot',
            'RightFoot': 'right_foot',
        }

        marker_name = marker_map.get(template.name)
        if marker_name:
            marker_pos = getattr(markers, marker_name, None)
            if marker_pos is not None:
                return np.array(marker_pos, dtype=np.float32)

        # Compute from parent + template offset
        if template.parent and template.parent in bone_positions:
            parent_pos = bone_positions[template.parent]
            offset = np.array(template.local_offset) * height
            return parent_pos + offset

        # Root bone - use hips marker
        if template.name == 'Hips' and markers.hips:
            return np.array(markers.hips, dtype=np.float32)

        # Fallback
        return np.array([0, 0, 0], dtype=np.float32)

    def _apply_limb_ik(
        self,
        bones: List[FittedBone],
        name_to_idx: Dict[str, int],
        markers: MarkerSet
    ):
        """Apply simple IK to position arm/leg chains."""

        # Left arm IK
        if markers.left_hand:
            self._solve_two_bone_ik(
                bones,
                name_to_idx.get('LeftUpperArm', -1),
                name_to_idx.get('LeftLowerArm', -1),
                name_to_idx.get('LeftHand', -1),
                np.array(markers.left_hand),
                np.array([0, 0, -1])  # Elbow hint
            )

        # Right arm IK
        if markers.right_hand:
            self._solve_two_bone_ik(
                bones,
                name_to_idx.get('RightUpperArm', -1),
                name_to_idx.get('RightLowerArm', -1),
                name_to_idx.get('RightHand', -1),
                np.array(markers.right_hand),
                np.array([0, 0, -1])
            )

        # Left leg IK
        if markers.left_foot:
            self._solve_two_bone_ik(
                bones,
                name_to_idx.get('LeftUpperLeg', -1),
                name_to_idx.get('LeftLowerLeg', -1),
                name_to_idx.get('LeftFoot', -1),
                np.array(markers.left_foot),
                np.array([0, 0, 1])  # Knee hint
            )

        # Right leg IK
        if markers.right_foot:
            self._solve_two_bone_ik(
                bones,
                name_to_idx.get('RightUpperLeg', -1),
                name_to_idx.get('RightLowerLeg', -1),
                name_to_idx.get('RightFoot', -1),
                np.array(markers.right_foot),
                np.array([0, 0, 1])
            )

    def _solve_two_bone_ik(
        self,
        bones: List[FittedBone],
        upper_idx: int,
        lower_idx: int,
        end_idx: int,
        target: np.ndarray,
        pole_hint: np.ndarray
    ):
        """Simple two-bone IK solver."""
        if upper_idx < 0 or lower_idx < 0 or end_idx < 0:
            return

        upper = bones[upper_idx]
        lower = bones[lower_idx]
        end = bones[end_idx]

        # Bone lengths
        upper_len = upper.length
        lower_len = lower.length

        root = upper.head_position.copy()
        to_target = target - root
        dist = np.linalg.norm(to_target)

        # Clamp to reachable distance
        max_reach = upper_len + lower_len - 0.001
        min_reach = abs(upper_len - lower_len) + 0.001
        dist = np.clip(dist, min_reach, max_reach)

        # Direction to target
        direction = to_target / (np.linalg.norm(to_target) + 1e-6)

        # Law of cosines for elbow angle
        cos_angle = (upper_len**2 + dist**2 - lower_len**2) / (2 * upper_len * dist + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        # Rotate around pole to get elbow position
        # Simplified: project pole hint onto plane perpendicular to direction
        pole_proj = pole_hint - np.dot(pole_hint, direction) * direction
        pole_proj = pole_proj / (np.linalg.norm(pole_proj) + 1e-6)

        # Elbow position
        elbow_offset = direction * np.cos(angle) + pole_proj * np.sin(angle)
        elbow_pos = root + elbow_offset * upper_len

        # Update bone positions
        upper.tail_position = elbow_pos.copy()
        lower.head_position = elbow_pos.copy()
        lower.tail_position = target.copy()
        end.head_position = target.copy()

    def _compute_local_rotations(self, bones: List[FittedBone]):
        """Compute local rotation quaternions for each bone."""
        for bone in bones:
            # Compute rotation from default up (0,1,0) to bone direction
            direction = bone.direction
            up = np.array([0, 1, 0])

            # Handle edge case where direction is parallel to up
            if abs(np.dot(direction, up)) > 0.999:
                bone.local_rotation = np.array([0, 0, 0, 1], dtype=np.float32)
                continue

            # Rotation axis and angle
            axis = np.cross(up, direction)
            axis = axis / (np.linalg.norm(axis) + 1e-6)
            angle = np.arccos(np.clip(np.dot(up, direction), -1, 1))

            # Quaternion from axis-angle
            half_angle = angle / 2
            bone.local_rotation = np.array([
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
                np.cos(half_angle)
            ], dtype=np.float32)


# =============================================================================
# Direct Gaussian Skinning
# =============================================================================

class DirectGaussianSkinner:
    """
    Weight Gaussians directly to bones without intermediate mesh skinning.

    Uses distance-based weights:
    - Each Gaussian gets weights to nearest N bones
    - Weight falls off smoothly with distance
    - Optional heat diffusion for smoother weights
    """

    def __init__(self, max_influences: int = 4, falloff: str = 'linear'):
        """
        Args:
            max_influences: Max bones per Gaussian (typically 4)
            falloff: 'linear', 'quadratic', or 'smooth' weight falloff
        """
        self.max_influences = max_influences
        self.falloff = falloff

    def compute_weights(
        self,
        gaussian_positions: np.ndarray,
        bones: List[FittedBone],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute skinning weights for Gaussians.

        Args:
            gaussian_positions: (N, 3) Gaussian positions
            bones: List of fitted bones with head/tail positions

        Returns:
            bone_indices: (N, 4) int32 bone indices
            bone_weights: (N, 4) float32 weights (sum to 1.0)
        """
        n_gaussians = len(gaussian_positions)
        n_bones = len(bones)

        logger.info(f"Computing skinning weights for {n_gaussians:,} Gaussians, {n_bones} bones")

        # Compute distance from each Gaussian to each bone segment
        distances = np.zeros((n_gaussians, n_bones), dtype=np.float32)

        for i, bone in enumerate(bones):
            # Distance to bone segment (head to tail)
            distances[:, i] = self._point_to_segment_distance(
                gaussian_positions,
                bone.head_position,
                bone.tail_position
            )

        # Apply falloff
        if self.falloff == 'linear':
            weights = 1.0 / (distances + 0.001)
        elif self.falloff == 'quadratic':
            weights = 1.0 / (distances**2 + 0.001)
        else:  # smooth
            weights = np.exp(-distances * 10)

        # For each Gaussian, keep top K bones
        bone_indices = np.zeros((n_gaussians, self.max_influences), dtype=np.int32)
        bone_weights = np.zeros((n_gaussians, self.max_influences), dtype=np.float32)

        for i in range(n_gaussians):
            # Get indices sorted by weight (descending)
            sorted_indices = np.argsort(-weights[i])[:self.max_influences]

            bone_indices[i] = sorted_indices
            raw_weights = weights[i, sorted_indices]

            # Normalize to sum to 1
            total = raw_weights.sum()
            if total > 0:
                bone_weights[i] = raw_weights / total
            else:
                bone_weights[i, 0] = 1.0  # Fallback

        logger.info(f"  Computed weights (falloff={self.falloff})")
        return bone_indices, bone_weights

    def _point_to_segment_distance(
        self,
        points: np.ndarray,  # (N, 3)
        segment_start: np.ndarray,  # (3,)
        segment_end: np.ndarray,  # (3,)
    ) -> np.ndarray:
        """Compute distance from each point to line segment."""
        v = segment_end - segment_start
        w = points - segment_start

        # Project onto segment
        c1 = np.sum(w * v, axis=1)
        c2 = np.dot(v, v)

        # Handle degenerate segment
        if c2 < 1e-10:
            return np.linalg.norm(w, axis=1)

        t = np.clip(c1 / c2, 0, 1)

        # Closest point on segment
        closest = segment_start + t[:, np.newaxis] * v

        return np.linalg.norm(points - closest, axis=1)


# =============================================================================
# Auto-Detection
# =============================================================================

class MarkerAutoDetector:
    """
    Auto-detect marker positions from mesh geometry.

    Uses extremity detection and symmetry analysis.
    """

    def detect(self, vertices: np.ndarray) -> MarkerSet:
        """
        Auto-detect markers from mesh vertices.

        Args:
            vertices: (N, 3) mesh vertex positions

        Returns:
            MarkerSet with detected positions
        """
        markers = MarkerSet()

        # Compute bounding box
        min_bound = vertices.min(axis=0)
        max_bound = vertices.max(axis=0)
        center = (min_bound + max_bound) / 2

        # Height is Y extent
        height = max_bound[1] - min_bound[1]

        # Head: highest point near center X
        center_mask = np.abs(vertices[:, 0] - center[0]) < height * 0.15
        if center_mask.any():
            head_candidates = vertices[center_mask]
            head_idx = np.argmax(head_candidates[:, 1])
            markers.head = tuple(head_candidates[head_idx])
        else:
            markers.head = tuple(max_bound * [0, 1, 0] + center * [1, 0, 1])

        # Feet: lowest points, split left/right
        foot_height_threshold = min_bound[1] + height * 0.1
        low_verts = vertices[vertices[:, 1] < foot_height_threshold]

        if len(low_verts) > 0:
            left_foot_mask = low_verts[:, 0] > center[0]
            right_foot_mask = low_verts[:, 0] < center[0]

            if left_foot_mask.any():
                left_foot_verts = low_verts[left_foot_mask]
                markers.left_foot = tuple(left_foot_verts[np.argmin(left_foot_verts[:, 1])])

            if right_foot_mask.any():
                right_foot_verts = low_verts[right_foot_mask]
                markers.right_foot = tuple(right_foot_verts[np.argmin(right_foot_verts[:, 1])])

        # Hips: center of mass at ~55% height
        hip_y = min_bound[1] + height * 0.52
        markers.hips = (float(center[0]), float(hip_y), float(center[2]))

        # Hands: extremities on X axis
        left_idx = np.argmax(vertices[:, 0])
        right_idx = np.argmin(vertices[:, 0])
        markers.left_hand = tuple(vertices[left_idx])
        markers.right_hand = tuple(vertices[right_idx])

        # Validate and fill missing with estimates
        self._fill_missing(markers, center, height)

        logger.info(f"Auto-detected markers: {list(markers.to_dict().keys())}")
        return markers

    def _fill_missing(self, markers: MarkerSet, center: np.ndarray, height: float):
        """Fill in any missing required markers with estimates."""
        if markers.head is None:
            markers.head = (center[0], center[1] + height * 0.45, center[2])

        if markers.hips is None:
            markers.hips = (center[0], center[1], center[2])

        if markers.left_foot is None:
            markers.left_foot = (center[0] + height * 0.1, center[1] - height * 0.5, center[2])

        if markers.right_foot is None:
            markers.right_foot = (center[0] - height * 0.1, center[1] - height * 0.5, center[2])

        if markers.left_hand is None:
            markers.left_hand = (center[0] + height * 0.4, center[1] + height * 0.2, center[2])

        if markers.right_hand is None:
            markers.right_hand = (center[0] - height * 0.4, center[1] + height * 0.2, center[2])


# =============================================================================
# Main AutoRigger Class
# =============================================================================

class AutoRigger:
    """
    Main auto-rigging interface.

    Usage:
        rigger = AutoRigger()
        rigger.load_mesh('/path/to/model.obj')
        markers = rigger.auto_detect_markers()  # or set manually
        result = rigger.rig(markers, output_path='/path/to/output.radiance')
    """

    def __init__(self):
        self.mesh_vertices: Optional[np.ndarray] = None
        self.mesh_faces: Optional[np.ndarray] = None
        self.mesh_colors: Optional[np.ndarray] = None
        self.mesh_path: Optional[str] = None

        self.fitter = SkeletonFitter()
        self.skinner = DirectGaussianSkinner(max_influences=4, falloff='smooth')
        self.detector = MarkerAutoDetector()

    def load_mesh(self, path: str) -> int:
        """
        Load mesh from file.

        Supports: OBJ, FBX, GLTF, GLB

        Returns:
            Number of vertices loaded
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mesh not found: {path}")

        self.mesh_path = str(path)
        suffix = path.suffix.lower()

        if suffix == '.obj':
            self._load_obj(path)
        elif suffix in ('.gltf', '.glb'):
            self._load_gltf(path)
        elif suffix == '.fbx':
            self._load_fbx(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        logger.info(f"Loaded mesh: {len(self.mesh_vertices):,} vertices from {path.name}")
        return len(self.mesh_vertices)

    def _load_obj(self, path: Path):
        """Load OBJ file with texture support."""
        from PIL import Image

        vertices = []
        faces = []
        colors = []
        uvs = []
        face_uvs = []  # UV indices per face
        mtl_file = None
        textures = {}  # material_name -> PIL Image
        current_material = None
        face_materials = []  # material per face

        obj_dir = path.parent

        # First pass: parse OBJ
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'mtllib':
                    # Join all parts after mtllib (filename may have spaces)
                    mtl_file = obj_dir / ' '.join(parts[1:])

                elif parts[0] == 'usemtl':
                    # Join all parts (material name may have spaces)
                    current_material = ' '.join(parts[1:])

                elif parts[0] == 'v':
                    # Vertex (may include RGB after XYZ)
                    coords = [float(x) for x in parts[1:4]]
                    vertices.append(coords)

                    if len(parts) >= 7:
                        rgb = [float(x) for x in parts[4:7]]
                        colors.append(rgb)

                elif parts[0] == 'vt':
                    # Texture coordinate
                    uv = [float(x) for x in parts[1:3]]
                    uvs.append(uv)

                elif parts[0] == 'f':
                    # Face (handle v, v/vt, v/vt/vn, v//vn formats)
                    face_verts = []
                    face_uv_indices = []
                    for p in parts[1:]:
                        indices = p.split('/')
                        v_idx = int(indices[0]) - 1  # OBJ is 1-indexed
                        face_verts.append(v_idx)

                        if len(indices) > 1 and indices[1]:
                            uv_idx = int(indices[1]) - 1
                            face_uv_indices.append(uv_idx)

                    # Triangulate if needed
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i+1]])
                        face_materials.append(current_material)
                        if len(face_uv_indices) >= i + 2:
                            face_uvs.append([face_uv_indices[0], face_uv_indices[i], face_uv_indices[i+1]])

        # Load MTL file if present
        if mtl_file and mtl_file.exists():
            current_mtl = None
            with open(mtl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0] == 'newmtl':
                        current_mtl = parts[1]
                    elif parts[0] == 'map_Kd' and current_mtl:
                        # Diffuse texture
                        tex_path = obj_dir / parts[1]
                        if tex_path.exists():
                            try:
                                textures[current_mtl] = Image.open(tex_path).convert('RGB')
                            except Exception:
                                pass

        self.mesh_vertices = np.array(vertices, dtype=np.float32)
        self.mesh_faces = np.array(faces, dtype=np.int32) if faces else None

        # Compute vertex colors from textures if available
        if uvs and face_uvs and textures:
            uvs = np.array(uvs, dtype=np.float32)
            vertex_colors = np.full((len(vertices), 3), 0.7, dtype=np.float32)
            vertex_counts = np.zeros(len(vertices), dtype=np.int32)

            for face_idx, (face, face_uv) in enumerate(zip(faces, face_uvs)):
                mat_name = face_materials[face_idx] if face_idx < len(face_materials) else None
                tex = textures.get(mat_name) if mat_name else (list(textures.values())[0] if textures else None)

                if tex is not None:
                    tex_w, tex_h = tex.size
                    for v_idx, uv_idx in zip(face, face_uv):
                        u, v = uvs[uv_idx]
                        # Sample texture (flip V for OpenGL convention)
                        px = int(u * tex_w) % tex_w
                        py = int((1 - v) * tex_h) % tex_h
                        r, g, b = tex.getpixel((px, py))
                        vertex_colors[v_idx] += np.array([r/255, g/255, b/255])
                        vertex_counts[v_idx] += 1

            # Average colors for vertices sampled multiple times
            mask = vertex_counts > 0
            vertex_colors[mask] /= vertex_counts[mask, np.newaxis]
            self.mesh_colors = vertex_colors
        elif colors:
            self.mesh_colors = np.array(colors, dtype=np.float32)
        else:
            self.mesh_colors = None

    def _load_gltf(self, path: Path):
        """Load GLTF/GLB file."""
        try:
            import pygltflib
            gltf = pygltflib.GLTF2().load(str(path))
            # TODO: Full GLTF parsing
            raise NotImplementedError("GLTF loading not yet implemented - use OBJ for now")
        except ImportError:
            raise ImportError("pygltflib required for GLTF. Install with: pip install pygltflib")

    def _load_fbx(self, path: Path):
        """Load FBX file."""
        try:
            import pyassimp
            # TODO: FBX loading via assimp
            raise NotImplementedError("FBX loading not yet implemented - use OBJ for now")
        except ImportError:
            raise ImportError("pyassimp required for FBX. Install with: pip install pyassimp")

    def auto_detect_markers(self) -> MarkerSet:
        """Auto-detect markers from loaded mesh."""
        if self.mesh_vertices is None:
            raise RuntimeError("No mesh loaded. Call load_mesh() first.")

        return self.detector.detect(self.mesh_vertices)

    def rig(
        self,
        markers: MarkerSet,
        output_path: Optional[str] = None,
        entity_id: str = "",
        display_name: str = "",
        densify: bool = True,
    ) -> Dict[str, Any]:
        """
        Rig the loaded mesh with the provided markers.

        Args:
            markers: Marker positions (from auto_detect or manual)
            output_path: Output .radiance file path
            entity_id: Entity ID for scene protocol
            display_name: Human-readable name
            densify: Add Gaussians at face centers/edges

        Returns:
            Result dict with success, output_path, gaussian_count, bone_count
        """
        if self.mesh_vertices is None:
            raise RuntimeError("No mesh loaded. Call load_mesh() first.")

        # Validate markers
        valid, msg = markers.validate()
        if not valid:
            return {'success': False, 'message': msg}

        # Auto-generate output path
        if not output_path:
            output_path = str(Path(self.mesh_path).with_suffix('.radiance'))

        try:
            # 1. Fit skeleton to markers
            logger.info("Fitting skeleton to markers...")
            bones = self.fitter.fit(markers)

            # 2. Generate Gaussians from mesh
            logger.info("Generating Gaussians from mesh...")
            gaussian_positions, gaussian_colors = self._mesh_to_gaussians(densify)

            # 3. Compute skinning weights
            logger.info("Computing skinning weights...")
            bone_indices, bone_weights = self.skinner.compute_weights(
                gaussian_positions, bones
            )

            # 4. Save as .radiance
            logger.info(f"Saving to {output_path}...")
            self._save_radiance(
                output_path,
                gaussian_positions,
                gaussian_colors,
                bones,
                bone_indices,
                bone_weights,
                entity_id or Path(output_path).stem,
                display_name or Path(output_path).stem,
            )

            return {
                'success': True,
                'output_path': output_path,
                'gaussian_count': len(gaussian_positions),
                'bone_count': len(bones),
                'message': f"Created rigged radiance: {len(gaussian_positions):,} Gaussians, {len(bones)} bones",
            }

        except Exception as e:
            logger.exception("Auto-rigging failed")
            return {'success': False, 'message': str(e)}

    def _mesh_to_gaussians(self, densify: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Convert mesh to Gaussian positions and colors."""
        positions = [self.mesh_vertices]

        # Use existing vertex colors or default to gray
        if self.mesh_colors is not None and len(self.mesh_colors) == len(self.mesh_vertices):
            colors = [self.mesh_colors]
        else:
            colors = [np.full((len(self.mesh_vertices), 3), 0.7, dtype=np.float32)]

        if densify and self.mesh_faces is not None:
            # Add face centers
            v0 = self.mesh_vertices[self.mesh_faces[:, 0]]
            v1 = self.mesh_vertices[self.mesh_faces[:, 1]]
            v2 = self.mesh_vertices[self.mesh_faces[:, 2]]
            face_centers = (v0 + v1 + v2) / 3
            positions.append(face_centers)

            # Average colors for face centers
            if self.mesh_colors is not None:
                c0 = self.mesh_colors[self.mesh_faces[:, 0]]
                c1 = self.mesh_colors[self.mesh_faces[:, 1]]
                c2 = self.mesh_colors[self.mesh_faces[:, 2]]
                face_colors = (c0 + c1 + c2) / 3
            else:
                face_colors = np.full((len(face_centers), 3), 0.7, dtype=np.float32)
            colors.append(face_colors)

            # Add edge midpoints
            edge_mids = np.vstack([
                (v0 + v1) / 2,
                (v1 + v2) / 2,
                (v2 + v0) / 2,
            ])
            positions.append(edge_mids)

            if self.mesh_colors is not None:
                edge_colors = np.vstack([
                    (c0 + c1) / 2,
                    (c1 + c2) / 2,
                    (c2 + c0) / 2,
                ])
            else:
                edge_colors = np.full((len(edge_mids), 3), 0.7, dtype=np.float32)
            colors.append(edge_colors)

        return np.vstack(positions), np.vstack(colors)

    def _save_radiance(
        self,
        output_path: str,
        positions: np.ndarray,
        colors: np.ndarray,
        bones: List[FittedBone],
        bone_indices: np.ndarray,
        bone_weights: np.ndarray,
        entity_id: str,
        display_name: str,
    ):
        """Save as .radiance format."""
        from ..core.semantic_world.radiance_format import RadianceAsset, save_radiance

        n = len(positions)

        # Create asset
        asset = RadianceAsset()
        asset.entity_id = entity_id
        asset.display_name = display_name

        asset.positions = positions.astype(np.float32)

        # Default Gaussian parameters
        asset.scales = np.full((n, 3), 0.005, dtype=np.float32)  # Small splats
        asset.rotations = np.zeros((n, 4), dtype=np.float32)
        asset.rotations[:, 3] = 1.0  # Identity quaternion
        asset.opacities = np.ones(n, dtype=np.float32)

        # Colors to SH DC
        # SH DC formula: color = 0.5 + SH_C0 * sh_dc, so sh_dc = (color - 0.5) / SH_C0
        SH_C0 = 0.28209479177387814
        asset.sh_dc = (colors - 0.5) / SH_C0

        # Skeleton
        from ..core.semantic_world.radiance_format import RadianceBone, RadianceSkeleton
        asset.skeleton = RadianceSkeleton()
        for bone in bones:
            asset.skeleton.bones.append(RadianceBone(
                name=bone.name,
                parent_index=bone.parent_index,
                position=tuple(bone.head_position),
                rotation=tuple(bone.local_rotation),
                scale=(1.0, 1.0, 1.0),
            ))

        # Humanoid mapping
        asset.skeleton.humanoid_map = {bone.humanoid_name: i for i, bone in enumerate(bones)}

        # Skinning weights (4 bones per vertex)
        asset.skin_bone_indices = bone_indices.astype(np.uint16)
        asset.skin_bone_weights = bone_weights.astype(np.float32)

        # Semantic labels from bone assignments
        asset.semantic_labels = self._compute_semantic_labels(bone_indices, bone_weights, bones)

        save_radiance(asset, output_path)

    def _compute_semantic_labels(
        self,
        bone_indices: np.ndarray,
        bone_weights: np.ndarray,
        bones: List[FittedBone],
    ) -> np.ndarray:
        """Compute body region labels from dominant bone."""
        n = len(bone_indices)
        labels = np.zeros(n, dtype=np.uint8)

        # Map humanoid bones to body regions
        humanoid_to_region = {
            'head': 1, 'neck': 1,
            'spine': 2, 'chest': 2, 'upperChest': 2, 'hips': 2,
            'leftShoulder': 3, 'leftUpperArm': 3, 'leftLowerArm': 3,
            'rightShoulder': 4, 'rightUpperArm': 4, 'rightLowerArm': 4,
            'leftUpperLeg': 5, 'leftLowerLeg': 5, 'leftFoot': 5, 'leftToes': 5,
            'rightUpperLeg': 6, 'rightLowerLeg': 6, 'rightFoot': 6, 'rightToes': 6,
            'leftHand': 7,
            'rightHand': 8,
        }

        for i in range(n):
            dominant_bone_idx = bone_indices[i, 0]
            if dominant_bone_idx < len(bones):
                humanoid_name = bones[dominant_bone_idx].humanoid_name
                labels[i] = humanoid_to_region.get(humanoid_name, 0)

        return labels


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-rig mesh to Gaussian splats"
    )
    parser.add_argument("mesh", help="Input mesh file (OBJ, FBX, GLTF)")
    parser.add_argument("-o", "--output", help="Output .radiance file")
    parser.add_argument("--name", help="Display name")
    parser.add_argument("--no-densify", action="store_true", help="Skip densification")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(message)s'
    )

    rigger = AutoRigger()
    rigger.load_mesh(args.mesh)

    markers = rigger.auto_detect_markers()
    print(f"Auto-detected markers: {list(markers.to_dict().keys())}")

    result = rigger.rig(
        markers,
        output_path=args.output,
        display_name=args.name or "",
        densify=not args.no_densify,
    )

    if result['success']:
        print(f"Created: {result['output_path']}")
        print(f"  Gaussians: {result['gaussian_count']:,}")
        print(f"  Bones: {result['bone_count']}")
    else:
        print(f"Failed: {result['message']}")
