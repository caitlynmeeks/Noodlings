"""
Radiance Format - Semantic Gaussian Splat file format for Noodlings.

.radiance files contain:
- Standard Gaussian parameters (position, scale, rotation, SH)
- Skeleton definition for animation
- Per-Gaussian skinning weights for LBS deformation
- Semantic labels (body part, region)
- Optional CLIP embeddings for natural language queries
- Optional spring bone physics parameters

Every Gaussian knows what it represents. Every frame is query-able.

Author: Caitlyn + Claude
Date: December 2025

FUTURE: Progressive Streaming (v2 format)
=========================================
For massive worlds with composited Gaussians, progressive loading would allow
instant rough display that refines over time. Design options:

1. SH Band Streaming
   - Load DC coefficients first (flat color, instant silhouette)
   - Stream higher SH bands progressively (view-dependent shading)
   - Minimal format change: split GAUS into GAUS_DC + GAUS_SH chunks

2. Multi-Resolution LOD Chunks
   - GAUS_LOD0: 1K coarse Gaussians (instant rough shape)
   - GAUS_LOD1: 10K medium
   - GAUS_LOD2: 100K+ full detail
   - New chunk: LODS (summary with counts, byte offsets, target screen sizes)

3. Spatial Tiling (for world-scale)
   - Octree subdivision of Gaussians
   - Stream tiles near camera first
   - Chunk: TILE with spatial bounds + Gaussian range

4. Importance Sorting
   - Sort Gaussians by visual importance (opacity * size)
   - First N Gaussians give most visual impact
   - Progressive count increase without format change

Current format (v1) stores everything linearly. When we need progressive
loading, we'll extend with optional LOD/tile chunks while maintaining
backward compatibility (v1 readers ignore unknown chunks).
"""

import struct
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, BinaryIO
from enum import IntEnum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

RADIANCE_MAGIC = b'RADI'
RADIANCE_VERSION = 0x00010000  # v1.0

# Feature flags
FLAG_HAS_SKELETON = 0x0001
FLAG_HAS_SKINNING = 0x0002
FLAG_HAS_SEMANTICS = 0x0004
FLAG_HAS_CLIP = 0x0008
FLAG_HAS_SPRINGS = 0x0010
FLAG_ANIMATED = 0x0020

# Chunk types
CHUNK_GAUS = b'GAUS'  # Gaussian parameters
CHUNK_SKEL = b'SKEL'  # Skeleton
CHUNK_SKIN = b'SKIN'  # Skinning weights
CHUNK_SEMA = b'SEMA'  # Semantic labels
CHUNK_CLIP = b'CLIP'  # CLIP embeddings
CHUNK_SPRG = b'SPRG'  # Spring bones
CHUNK_META = b'META'  # Metadata


class BodyRegion(IntEnum):
    """Body region classification for semantic labels."""
    OTHER = 0
    HEAD = 1
    TORSO = 2
    LEFT_ARM = 3
    RIGHT_ARM = 4
    LEFT_LEG = 5
    RIGHT_LEG = 6
    LEFT_HAND = 7
    RIGHT_HAND = 8
    TAIL = 9
    ACCESSORY = 10


BODY_REGION_NAMES = {
    BodyRegion.OTHER: "other",
    BodyRegion.HEAD: "head",
    BodyRegion.TORSO: "torso",
    BodyRegion.LEFT_ARM: "left_arm",
    BodyRegion.RIGHT_ARM: "right_arm",
    BodyRegion.LEFT_LEG: "left_leg",
    BodyRegion.RIGHT_LEG: "right_leg",
    BodyRegion.LEFT_HAND: "left_hand",
    BodyRegion.RIGHT_HAND: "right_hand",
    BodyRegion.TAIL: "tail",
    BodyRegion.ACCESSORY: "accessory",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RadianceBone:
    """Skeleton bone."""
    name: str
    parent_index: int = -1
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class RadianceSkeleton:
    """Skeleton definition."""
    bones: List[RadianceBone] = field(default_factory=list)
    humanoid_map: Dict[str, int] = field(default_factory=dict)

    @property
    def bone_count(self) -> int:
        return len(self.bones)


@dataclass
class SpringChain:
    """Spring bone physics chain."""
    name: str
    bone_indices: List[int] = field(default_factory=list)
    stiffness: float = 1.0
    gravity_power: float = 0.0
    gravity_dir: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    drag_force: float = 0.4
    hit_radius: float = 0.02


@dataclass
class SpringCollider:
    """Spring bone collider."""
    bone_index: int
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.1


@dataclass
class RadianceMetadata:
    """Asset metadata."""
    entity_type: str = "noodling"
    entity_id: str = ""
    display_name: str = ""
    author: str = ""
    created: str = ""
    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tags: List[str] = field(default_factory=list)


# =============================================================================
# Radiance Asset
# =============================================================================

class RadianceAsset:
    """
    A .radiance file containing semantic Gaussian splat data.

    Every Gaussian knows:
    - Where it is (position, scale, rotation)
    - What color it radiates (spherical harmonics)
    - What bone(s) move it (skinning)
    - What body part it represents (semantic label)
    - What entity it belongs to (metadata)
    """

    def __init__(self):
        # Core Gaussian data
        self.positions: Optional[np.ndarray] = None      # (N, 3) float32
        self.scales: Optional[np.ndarray] = None         # (N, 3) float32
        self.rotations: Optional[np.ndarray] = None      # (N, 4) float32 quaternion
        self.opacities: Optional[np.ndarray] = None      # (N,) float32
        self.sh_dc: Optional[np.ndarray] = None          # (N, 3) float32
        self.sh_rest: Optional[np.ndarray] = None        # (N, 45) float32, optional

        # Skeleton
        self.skeleton: Optional[RadianceSkeleton] = None

        # Skinning weights per Gaussian
        self.skin_bone_indices: Optional[np.ndarray] = None  # (N, 4) uint16
        self.skin_bone_weights: Optional[np.ndarray] = None  # (N, 4) float32

        # Semantic labels per Gaussian
        self.body_regions: Optional[np.ndarray] = None       # (N,) uint8
        self.semantic_labels: List[str] = []                 # String table

        # CLIP embeddings (optional)
        self.clip_embeddings: Optional[np.ndarray] = None    # (N, dim) float32

        # Spring bones (optional)
        self.spring_chains: List[SpringChain] = []
        self.spring_colliders: List[SpringCollider] = []

        # Metadata
        self.metadata: RadianceMetadata = RadianceMetadata()

        # SH degree (0=DC only, 1-3=higher bands)
        self.sh_degree: int = 0

    @property
    def gaussian_count(self) -> int:
        """Number of Gaussians."""
        if self.positions is not None:
            return self.positions.shape[0]
        return 0

    @property
    def bone_count(self) -> int:
        """Number of bones."""
        if self.skeleton:
            return self.skeleton.bone_count
        return 0

    @property
    def has_skeleton(self) -> bool:
        return self.skeleton is not None and self.skeleton.bone_count > 0

    @property
    def has_skinning(self) -> bool:
        return self.skin_bone_indices is not None

    @property
    def has_semantics(self) -> bool:
        return self.body_regions is not None

    @property
    def has_clip(self) -> bool:
        return self.clip_embeddings is not None

    @property
    def has_springs(self) -> bool:
        return len(self.spring_chains) > 0

    def get_semantic_label(self, index: int) -> str:
        """Get semantic label for a Gaussian."""
        if index < 0 or index >= len(self.semantic_labels):
            return ""
        return self.semantic_labels[index]

    def get_body_region(self, index: int) -> str:
        """Get body region name for a Gaussian."""
        if self.body_regions is None or index < 0 or index >= len(self.body_regions):
            return "other"
        region = BodyRegion(self.body_regions[index])
        return BODY_REGION_NAMES.get(region, "other")

    def get_covariance_matrix(self, index: int) -> np.ndarray:
        """
        Get 3x3 covariance matrix for a Gaussian.

        Used for collision detection.
        """
        if self.scales is None or self.rotations is None:
            return np.eye(3)

        scale = self.scales[index]
        quat = self.rotations[index]

        # Scale matrix (diagonal of squared scales)
        S = np.diag(scale ** 2)

        # Rotation matrix from quaternion
        R = self._quaternion_to_matrix(quat)

        # Covariance: R @ S @ R.T
        return R @ S @ R.T

    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
        x, y, z, w = q

        # Rotation matrix
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float32)

    def compute_bounds(self):
        """Compute bounding box from Gaussian positions."""
        if self.positions is None or len(self.positions) == 0:
            return

        self.metadata.bounds_min = tuple(self.positions.min(axis=0))
        self.metadata.bounds_max = tuple(self.positions.max(axis=0))
        self.metadata.center = tuple((self.positions.min(axis=0) + self.positions.max(axis=0)) / 2)

    def apply_skinning(self, bone_rotations: Dict[str, Tuple[float, float, float]]) -> np.ndarray:
        """
        Apply Linear Blend Skinning (LBS) deformation.

        Args:
            bone_rotations: Dict mapping bone name to euler rotation (degrees)

        Returns:
            Deformed positions (N, 3)
        """
        if not self.has_skeleton or not self.has_skinning:
            return self.positions.copy() if self.positions is not None else np.array([])

        # Build bone transform matrices
        bone_matrices = []
        for bone in self.skeleton.bones:
            # Start with identity
            M = np.eye(4, dtype=np.float32)

            # Apply rotation if specified
            if bone.name in bone_rotations:
                rx, ry, rz = bone_rotations[bone.name]
                R = self._euler_to_matrix(rx, ry, rz)
                M[:3, :3] = R

            bone_matrices.append(M)

        bone_matrices = np.array(bone_matrices)

        # Apply LBS
        deformed = np.zeros_like(self.positions)

        for i in range(self.gaussian_count):
            pos = np.append(self.positions[i], 1.0)  # Homogeneous

            for j in range(4):
                bone_idx = self.skin_bone_indices[i, j]
                weight = self.skin_bone_weights[i, j]

                if weight > 0 and bone_idx < len(bone_matrices):
                    deformed[i] += weight * (bone_matrices[bone_idx] @ pos)[:3]

        return deformed

    def _euler_to_matrix(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """Convert euler angles (degrees) to rotation matrix."""
        rx, ry, rz = np.radians([rx, ry, rz])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def save(self, path: str):
        """Save to .radiance file."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.radiance')

        with open(path, 'wb') as f:
            self._write_header(f)
            self._write_gaussian_chunk(f)

            if self.has_skeleton:
                self._write_skeleton_chunk(f)

            if self.has_skinning:
                self._write_skinning_chunk(f)

            if self.has_semantics:
                self._write_semantic_chunk(f)

            if self.has_clip:
                self._write_clip_chunk(f)

            if self.has_springs:
                self._write_spring_chunk(f)

            self._write_metadata_chunk(f)

        logger.info(f"Saved radiance: {path} ({self.gaussian_count} Gaussians)")

    @classmethod
    def load(cls, path: str) -> 'RadianceAsset':
        """Load from .radiance file."""
        asset = cls()

        with open(path, 'rb') as f:
            asset._read_header(f)

            while True:
                chunk_header = f.read(16)
                if len(chunk_header) < 16:
                    break

                chunk_type = chunk_header[:4]
                chunk_size = struct.unpack('<I', chunk_header[4:8])[0]
                chunk_data = f.read(chunk_size)

                if chunk_type == CHUNK_GAUS:
                    asset._parse_gaussian_chunk(chunk_data)
                elif chunk_type == CHUNK_SKEL:
                    asset._parse_skeleton_chunk(chunk_data)
                elif chunk_type == CHUNK_SKIN:
                    asset._parse_skinning_chunk(chunk_data)
                elif chunk_type == CHUNK_SEMA:
                    asset._parse_semantic_chunk(chunk_data)
                elif chunk_type == CHUNK_CLIP:
                    asset._parse_clip_chunk(chunk_data)
                elif chunk_type == CHUNK_SPRG:
                    asset._parse_spring_chunk(chunk_data)
                elif chunk_type == CHUNK_META:
                    asset._parse_metadata_chunk(chunk_data)

        logger.info(f"Loaded radiance: {path} ({asset.gaussian_count} Gaussians)")
        return asset

    def _write_header(self, f: BinaryIO):
        """Write file header."""
        flags = 0
        if self.has_skeleton:
            flags |= FLAG_HAS_SKELETON
        if self.has_skinning:
            flags |= FLAG_HAS_SKINNING
        if self.has_semantics:
            flags |= FLAG_HAS_SEMANTICS
        if self.has_clip:
            flags |= FLAG_HAS_CLIP
        if self.has_springs:
            flags |= FLAG_HAS_SPRINGS

        chunk_count = 2  # GAUS + META always
        if self.has_skeleton:
            chunk_count += 1
        if self.has_skinning:
            chunk_count += 1
        if self.has_semantics:
            chunk_count += 1
        if self.has_clip:
            chunk_count += 1
        if self.has_springs:
            chunk_count += 1

        header = struct.pack(
            '<4sIIIII8s',
            RADIANCE_MAGIC,
            RADIANCE_VERSION,
            self.gaussian_count,
            self.bone_count,
            chunk_count,
            flags,
            b'\x00' * 8
        )
        f.write(header)

    def _read_header(self, f: BinaryIO):
        """Read file header."""
        header = f.read(32)
        magic, version, gauss_count, bone_count, chunk_count, flags, _ = struct.unpack(
            '<4sIIIII8s', header
        )

        if magic != RADIANCE_MAGIC:
            raise ValueError(f"Not a radiance file (magic: {magic})")

        if version > RADIANCE_VERSION:
            logger.warning(f"Radiance version {version:#x} newer than supported {RADIANCE_VERSION:#x}")

    def _write_chunk_header(self, f: BinaryIO, chunk_type: bytes, size: int, flags: int = 0):
        """Write chunk header."""
        header = struct.pack('<4sIII', chunk_type, size, flags, 0)
        f.write(header)

    def _write_gaussian_chunk(self, f: BinaryIO):
        """Write GAUS chunk."""
        if self.positions is None:
            return

        # Calculate size
        n = self.gaussian_count
        has_sh_rest = self.sh_rest is not None and self.sh_degree > 0

        # Position(12) + Scale(12) + Rotation(16) + Opacity(4) + SH_DC(12) = 56 bytes
        # + SH_rest(180) if degree 3
        per_gaussian = 56
        if has_sh_rest:
            per_gaussian += 45 * 4  # 45 floats

        chunk_size = per_gaussian * n
        flags = self.sh_degree & 0x0F

        self._write_chunk_header(f, CHUNK_GAUS, chunk_size, flags)

        # Write data
        for i in range(n):
            f.write(struct.pack('<3f', *self.positions[i]))
            f.write(struct.pack('<3f', *self.scales[i]))
            f.write(struct.pack('<4f', *self.rotations[i]))
            f.write(struct.pack('<f', self.opacities[i]))
            f.write(struct.pack('<3f', *self.sh_dc[i]))
            if has_sh_rest:
                f.write(struct.pack('<45f', *self.sh_rest[i]))

    def _parse_gaussian_chunk(self, data: bytes):
        """Parse GAUS chunk."""
        # Determine SH degree from data size
        n = self.gaussian_count or (len(data) // 56)  # Minimum size per Gaussian

        if n == 0:
            return

        per_gaussian_min = 56
        per_gaussian_full = 56 + 45 * 4

        if len(data) >= n * per_gaussian_full:
            has_sh_rest = True
            self.sh_degree = 3
        else:
            has_sh_rest = False
            self.sh_degree = 0

        self.positions = np.zeros((n, 3), dtype=np.float32)
        self.scales = np.zeros((n, 3), dtype=np.float32)
        self.rotations = np.zeros((n, 4), dtype=np.float32)
        self.opacities = np.zeros(n, dtype=np.float32)
        self.sh_dc = np.zeros((n, 3), dtype=np.float32)
        if has_sh_rest:
            self.sh_rest = np.zeros((n, 45), dtype=np.float32)

        offset = 0
        for i in range(n):
            self.positions[i] = struct.unpack_from('<3f', data, offset)
            offset += 12
            self.scales[i] = struct.unpack_from('<3f', data, offset)
            offset += 12
            self.rotations[i] = struct.unpack_from('<4f', data, offset)
            offset += 16
            self.opacities[i] = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            self.sh_dc[i] = struct.unpack_from('<3f', data, offset)
            offset += 12
            if has_sh_rest:
                self.sh_rest[i] = struct.unpack_from('<45f', data, offset)
                offset += 180

    def _write_skeleton_chunk(self, f: BinaryIO):
        """Write SKEL chunk."""
        if not self.skeleton:
            return

        # Calculate size
        bone_data_size = len(self.skeleton.bones) * (32 + 4 + 12 + 16 + 12)  # name + parent + pos + rot + scale
        humanoid_data_size = len(self.skeleton.humanoid_map) * (32 + 4)
        chunk_size = 8 + bone_data_size + humanoid_data_size

        self._write_chunk_header(f, CHUNK_SKEL, chunk_size)

        # Write counts
        f.write(struct.pack('<II', len(self.skeleton.bones), len(self.skeleton.humanoid_map)))

        # Write bones
        for bone in self.skeleton.bones:
            name_bytes = bone.name.encode('utf-8')[:31].ljust(32, b'\x00')
            f.write(name_bytes)
            f.write(struct.pack('<i', bone.parent_index))
            f.write(struct.pack('<3f', *bone.position))
            f.write(struct.pack('<4f', *bone.rotation))
            f.write(struct.pack('<3f', *bone.scale))

        # Write humanoid map
        for humanoid_name, bone_idx in self.skeleton.humanoid_map.items():
            name_bytes = humanoid_name.encode('utf-8')[:31].ljust(32, b'\x00')
            f.write(name_bytes)
            f.write(struct.pack('<I', bone_idx))

    def _parse_skeleton_chunk(self, data: bytes):
        """Parse SKEL chunk."""
        bone_count, humanoid_count = struct.unpack_from('<II', data, 0)
        offset = 8

        self.skeleton = RadianceSkeleton()

        # Read bones
        for _ in range(bone_count):
            name = data[offset:offset+32].rstrip(b'\x00').decode('utf-8')
            offset += 32
            parent_idx = struct.unpack_from('<i', data, offset)[0]
            offset += 4
            position = struct.unpack_from('<3f', data, offset)
            offset += 12
            rotation = struct.unpack_from('<4f', data, offset)
            offset += 16
            scale = struct.unpack_from('<3f', data, offset)
            offset += 12

            self.skeleton.bones.append(RadianceBone(
                name=name,
                parent_index=parent_idx,
                position=position,
                rotation=rotation,
                scale=scale
            ))

        # Read humanoid map
        for _ in range(humanoid_count):
            name = data[offset:offset+32].rstrip(b'\x00').decode('utf-8')
            offset += 32
            bone_idx = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            self.skeleton.humanoid_map[name] = bone_idx

    def _write_skinning_chunk(self, f: BinaryIO):
        """Write SKIN chunk."""
        if self.skin_bone_indices is None:
            return

        n = self.gaussian_count
        chunk_size = n * (8 + 16)  # 4 uint16 + 4 float32

        self._write_chunk_header(f, CHUNK_SKIN, chunk_size)

        for i in range(n):
            f.write(struct.pack('<4H', *self.skin_bone_indices[i]))
            f.write(struct.pack('<4f', *self.skin_bone_weights[i]))

    def _parse_skinning_chunk(self, data: bytes):
        """Parse SKIN chunk."""
        n = self.gaussian_count
        self.skin_bone_indices = np.zeros((n, 4), dtype=np.uint16)
        self.skin_bone_weights = np.zeros((n, 4), dtype=np.float32)

        offset = 0
        for i in range(n):
            self.skin_bone_indices[i] = struct.unpack_from('<4H', data, offset)
            offset += 8
            self.skin_bone_weights[i] = struct.unpack_from('<4f', data, offset)
            offset += 16

    def _write_semantic_chunk(self, f: BinaryIO):
        """Write SEMA chunk."""
        if self.body_regions is None:
            return

        n = self.gaussian_count

        # Build string table
        unique_labels = list(set(self.semantic_labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

        # Encode strings
        string_data = b''
        string_offsets = []
        for label in unique_labels:
            string_offsets.append(len(string_data))
            string_data += label.encode('utf-8') + b'\x00'

        # Per-Gaussian data: region(1) + flags(1) + label_offset(2) = 4 bytes
        per_gaussian_size = 4

        # String table header: count(4) + offsets(count*4) + strings
        string_table_size = 4 + len(unique_labels) * 4 + len(string_data)

        chunk_size = n * per_gaussian_size + string_table_size

        self._write_chunk_header(f, CHUNK_SEMA, chunk_size)

        # Write per-Gaussian data
        for i in range(n):
            region = self.body_regions[i] if i < len(self.body_regions) else 0
            label = self.semantic_labels[i] if i < len(self.semantic_labels) else ""
            label_idx = label_to_idx.get(label, 0)
            f.write(struct.pack('<BBH', region, 0, label_idx))

        # Write string table
        f.write(struct.pack('<I', len(unique_labels)))
        for offset in string_offsets:
            f.write(struct.pack('<I', offset))
        f.write(string_data)

    def _parse_semantic_chunk(self, data: bytes):
        """Parse SEMA chunk."""
        n = self.gaussian_count
        self.body_regions = np.zeros(n, dtype=np.uint8)
        label_indices = []

        offset = 0
        for i in range(n):
            region, flags, label_idx = struct.unpack_from('<BBH', data, offset)
            offset += 4
            self.body_regions[i] = region
            label_indices.append(label_idx)

        # Read string table
        string_count = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        string_offsets = []
        for _ in range(string_count):
            string_offsets.append(struct.unpack_from('<I', data, offset)[0])
            offset += 4

        string_data_start = offset
        unique_labels = []
        for i, str_offset in enumerate(string_offsets):
            end = data.find(b'\x00', string_data_start + str_offset)
            label = data[string_data_start + str_offset:end].decode('utf-8')
            unique_labels.append(label)

        # Reconstruct labels
        self.semantic_labels = [unique_labels[idx] if idx < len(unique_labels) else "" for idx in label_indices]

    def _write_clip_chunk(self, f: BinaryIO):
        """Write CLIP chunk."""
        if self.clip_embeddings is None:
            return

        n, dim = self.clip_embeddings.shape
        chunk_size = 8 + n * dim * 4  # header + float32 data

        self._write_chunk_header(f, CHUNK_CLIP, chunk_size)

        f.write(struct.pack('<II', dim, 0))  # dim, quantization (0=float32)
        f.write(self.clip_embeddings.astype(np.float32).tobytes())

    def _parse_clip_chunk(self, data: bytes):
        """Parse CLIP chunk."""
        dim, quant = struct.unpack_from('<II', data, 0)
        n = self.gaussian_count

        if quant == 0:  # float32
            self.clip_embeddings = np.frombuffer(data[8:], dtype=np.float32).reshape((n, dim))

    def _write_spring_chunk(self, f: BinaryIO):
        """Write SPRG chunk."""
        # Calculate size
        # Per chain: name(32) + bone_count(4) + bones(N*4) + stiffness(4) + gravity_power(4) + gravity_dir(12) + drag(4) + radius(4) = 32 + 4 + N*4 + 28
        chain_size = sum(32 + 4 + len(c.bone_indices) * 4 + 28 for c in self.spring_chains)
        collider_size = len(self.spring_colliders) * 20
        chunk_size = 8 + chain_size + collider_size

        self._write_chunk_header(f, CHUNK_SPRG, chunk_size)

        f.write(struct.pack('<II', len(self.spring_chains), len(self.spring_colliders)))

        # Write chains
        for chain in self.spring_chains:
            name_bytes = chain.name.encode('utf-8')[:31].ljust(32, b'\x00')
            f.write(name_bytes)
            f.write(struct.pack('<I', len(chain.bone_indices)))
            for idx in chain.bone_indices:
                f.write(struct.pack('<I', idx))
            f.write(struct.pack('<f', chain.stiffness))
            f.write(struct.pack('<f', chain.gravity_power))
            f.write(struct.pack('<3f', *chain.gravity_dir))
            f.write(struct.pack('<f', chain.drag_force))
            f.write(struct.pack('<f', chain.hit_radius))

        # Write colliders
        for collider in self.spring_colliders:
            f.write(struct.pack('<I', collider.bone_index))
            f.write(struct.pack('<3f', *collider.offset))
            f.write(struct.pack('<f', collider.radius))

    def _parse_spring_chunk(self, data: bytes):
        """Parse SPRG chunk."""
        chain_count, collider_count = struct.unpack_from('<II', data, 0)
        offset = 8

        self.spring_chains = []
        self.spring_colliders = []

        # Read chains
        for _ in range(chain_count):
            name = data[offset:offset+32].rstrip(b'\x00').decode('utf-8')
            offset += 32
            bone_count = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            bone_indices = list(struct.unpack_from(f'<{bone_count}I', data, offset))
            offset += bone_count * 4
            stiffness = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            gravity_power = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            gravity_dir = struct.unpack_from('<3f', data, offset)
            offset += 12
            drag_force = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            hit_radius = struct.unpack_from('<f', data, offset)[0]
            offset += 4

            self.spring_chains.append(SpringChain(
                name=name,
                bone_indices=bone_indices,
                stiffness=stiffness,
                gravity_power=gravity_power,
                gravity_dir=gravity_dir,
                drag_force=drag_force,
                hit_radius=hit_radius
            ))

        # Read colliders
        for _ in range(collider_count):
            bone_idx = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            coll_offset = struct.unpack_from('<3f', data, offset)
            offset += 12
            radius = struct.unpack_from('<f', data, offset)[0]
            offset += 4

            self.spring_colliders.append(SpringCollider(
                bone_index=bone_idx,
                offset=coll_offset,
                radius=radius
            ))

    def _write_metadata_chunk(self, f: BinaryIO):
        """Write META chunk."""
        # Build tags string
        tags_data = b'\x00'.join(t.encode('utf-8') for t in self.metadata.tags) + b'\x00' if self.metadata.tags else b''

        chunk_size = 16 + 64 + 64 + 64 + 32 + 36 + 4 + len(tags_data)

        self._write_chunk_header(f, CHUNK_META, chunk_size)

        f.write(self.metadata.entity_type.encode('utf-8')[:15].ljust(16, b'\x00'))
        f.write(self.metadata.entity_id.encode('utf-8')[:63].ljust(64, b'\x00'))
        f.write(self.metadata.display_name.encode('utf-8')[:63].ljust(64, b'\x00'))
        f.write(self.metadata.author.encode('utf-8')[:63].ljust(64, b'\x00'))
        f.write(self.metadata.created.encode('utf-8')[:31].ljust(32, b'\x00'))
        f.write(struct.pack('<3f', *self.metadata.bounds_min))
        f.write(struct.pack('<3f', *self.metadata.bounds_max))
        f.write(struct.pack('<3f', *self.metadata.center))
        f.write(struct.pack('<I', len(self.metadata.tags)))
        f.write(tags_data)

    def _parse_metadata_chunk(self, data: bytes):
        """Parse META chunk."""
        offset = 0

        self.metadata.entity_type = data[offset:offset+16].rstrip(b'\x00').decode('utf-8')
        offset += 16
        self.metadata.entity_id = data[offset:offset+64].rstrip(b'\x00').decode('utf-8')
        offset += 64
        self.metadata.display_name = data[offset:offset+64].rstrip(b'\x00').decode('utf-8')
        offset += 64
        self.metadata.author = data[offset:offset+64].rstrip(b'\x00').decode('utf-8')
        offset += 64
        self.metadata.created = data[offset:offset+32].rstrip(b'\x00').decode('utf-8')
        offset += 32
        self.metadata.bounds_min = struct.unpack_from('<3f', data, offset)
        offset += 12
        self.metadata.bounds_max = struct.unpack_from('<3f', data, offset)
        offset += 12
        self.metadata.center = struct.unpack_from('<3f', data, offset)
        offset += 12
        tag_count = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        if tag_count > 0 and offset < len(data):
            tags_data = data[offset:].rstrip(b'\x00')
            self.metadata.tags = [t.decode('utf-8') for t in tags_data.split(b'\x00') if t]

    # -------------------------------------------------------------------------
    # Import/Export
    # -------------------------------------------------------------------------

    @classmethod
    def from_ply(cls, ply_path: str) -> 'RadianceAsset':
        """Import from standard Gaussian PLY file."""
        asset = cls()

        with open(ply_path, 'rb') as f:
            # Parse header
            header_lines = []
            vertex_count = 0
            property_names = []

            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)

                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('property'):
                    parts = line.split()
                    property_names.append(parts[-1])
                elif line == 'end_header':
                    break

            # Read binary data
            bytes_per_vertex = len(property_names) * 4
            data = f.read(vertex_count * bytes_per_vertex)

        # Parse vertices
        asset.positions = np.zeros((vertex_count, 3), dtype=np.float32)
        asset.scales = np.zeros((vertex_count, 3), dtype=np.float32)
        asset.rotations = np.zeros((vertex_count, 4), dtype=np.float32)
        asset.opacities = np.zeros(vertex_count, dtype=np.float32)
        asset.sh_dc = np.zeros((vertex_count, 3), dtype=np.float32)

        # Find property indices
        try:
            x_idx = property_names.index('x')
            y_idx = property_names.index('y')
            z_idx = property_names.index('z')
        except ValueError:
            raise ValueError("PLY missing position properties")

        # Scale indices
        scale_indices = []
        for name in ['scale_0', 'scale_1', 'scale_2']:
            if name in property_names:
                scale_indices.append(property_names.index(name))

        # Rotation indices
        rot_indices = []
        for name in ['rot_0', 'rot_1', 'rot_2', 'rot_3']:
            if name in property_names:
                rot_indices.append(property_names.index(name))

        # Opacity
        opacity_idx = property_names.index('opacity') if 'opacity' in property_names else -1

        # SH DC
        sh_dc_indices = []
        for name in ['f_dc_0', 'f_dc_1', 'f_dc_2']:
            if name in property_names:
                sh_dc_indices.append(property_names.index(name))

        # Parse data
        for i in range(vertex_count):
            offset = i * bytes_per_vertex
            values = struct.unpack_from(f'{len(property_names)}f', data, offset)

            asset.positions[i] = [values[x_idx], values[y_idx], values[z_idx]]

            if len(scale_indices) == 3:
                asset.scales[i] = [np.exp(values[idx]) for idx in scale_indices]

            if len(rot_indices) == 4:
                asset.rotations[i] = [values[idx] for idx in rot_indices]
            else:
                asset.rotations[i] = [0, 0, 0, 1]

            if opacity_idx >= 0:
                asset.opacities[i] = 1.0 / (1.0 + np.exp(-values[opacity_idx]))  # Sigmoid

            if len(sh_dc_indices) == 3:
                asset.sh_dc[i] = [values[idx] for idx in sh_dc_indices]

        asset.compute_bounds()
        asset.metadata.display_name = Path(ply_path).stem

        logger.info(f"Imported PLY: {ply_path} ({vertex_count} Gaussians)")
        return asset

    def export_ply(self, ply_path: str):
        """Export to standard Gaussian PLY (loses semantic data)."""
        if self.positions is None:
            raise ValueError("No Gaussian data to export")

        n = self.gaussian_count

        with open(ply_path, 'wb') as f:
            # Header
            header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
"""
            f.write(header.encode('ascii'))

            # Data
            for i in range(n):
                pos = self.positions[i]
                scale = np.log(self.scales[i]) if self.scales is not None else [0, 0, 0]
                rot = self.rotations[i] if self.rotations is not None else [0, 0, 0, 1]
                opacity = -np.log(1.0 / max(0.001, self.opacities[i]) - 1) if self.opacities is not None else 0
                sh_dc = self.sh_dc[i] if self.sh_dc is not None else [0, 0, 0]

                f.write(struct.pack('<3f', *pos))
                f.write(struct.pack('<3f', *scale))
                f.write(struct.pack('<4f', *rot))
                f.write(struct.pack('<f', opacity))
                f.write(struct.pack('<3f', *sh_dc))

        logger.info(f"Exported PLY: {ply_path}")

    def import_skeleton_from_vrm(self, vrm_path: str):
        """Import skeleton from VRM file."""
        from .vrm_parser import parse_vrm

        avatar = parse_vrm(vrm_path)

        self.skeleton = RadianceSkeleton()
        for bone in avatar.skeleton.bones:
            self.skeleton.bones.append(RadianceBone(
                name=bone.name,
                parent_index=bone.parent_index,
                position=(bone.transform.position.x, bone.transform.position.y, bone.transform.position.z),
                rotation=(bone.transform.rotation.x, bone.transform.rotation.y,
                         bone.transform.rotation.z, bone.transform.rotation.w),
                scale=(bone.transform.scale.x, bone.transform.scale.y, bone.transform.scale.z)
            ))
        self.skeleton.humanoid_map = dict(avatar.skeleton.humanoid_map)

        # Import spring bones
        for chain in avatar.spring_bones.chains:
            self.spring_chains.append(SpringChain(
                name=chain.name,
                bone_indices=list(chain.bone_indices),
                stiffness=chain.stiffness,
                gravity_power=chain.gravity_power,
                gravity_dir=(chain.gravity_dir.x, chain.gravity_dir.y, chain.gravity_dir.z),
                drag_force=chain.drag_force,
                hit_radius=chain.hit_radius
            ))

        for collider in avatar.spring_bones.colliders:
            self.spring_colliders.append(SpringCollider(
                bone_index=collider.bone_index,
                offset=(collider.offset.x, collider.offset.y, collider.offset.z),
                radius=collider.radius
            ))

        logger.info(f"Imported skeleton from VRM: {len(self.skeleton.bones)} bones, "
                   f"{len(self.spring_chains)} spring chains")

    def compute_skinning_from_vrm(self, vrm_path: str):
        """Compute skinning weights by transferring from VRM mesh vertices."""
        from .gaussian_adapter import create_skeleton_binding_from_vrm

        if self.positions is None:
            raise ValueError("No Gaussian positions to skin")

        binding = create_skeleton_binding_from_vrm(
            vrm_path,
            self.positions,
            self.metadata.entity_id
        )

        self.skin_bone_indices = binding.bone_indices
        self.skin_bone_weights = binding.bone_weights
        self.body_regions = np.array([
            BodyRegion[region.upper().replace(' ', '_')] if region.upper().replace(' ', '_') in BodyRegion.__members__
            else BodyRegion.OTHER
            for region in binding.body_regions
        ], dtype=np.uint8)
        self.semantic_labels = binding.semantic_labels

        logger.info(f"Computed skinning from VRM: {self.gaussian_count} Gaussians skinned")


# =============================================================================
# Module Interface
# =============================================================================

def load_radiance(path: str) -> RadianceAsset:
    """Load a .radiance file."""
    return RadianceAsset.load(path)


def save_radiance(asset: RadianceAsset, path: str):
    """Save a .radiance file."""
    asset.save(path)


def ply_to_radiance(ply_path: str, radiance_path: str,
                    vrm_path: Optional[str] = None,
                    entity_id: str = "",
                    display_name: str = "") -> RadianceAsset:
    """
    Convert PLY to radiance, optionally enriching with VRM data.

    Args:
        ply_path: Input PLY file
        radiance_path: Output radiance file
        vrm_path: Optional VRM for skeleton/skinning
        entity_id: Entity identifier
        display_name: Display name

    Returns:
        Created RadianceAsset
    """
    asset = RadianceAsset.from_ply(ply_path)

    asset.metadata.entity_id = entity_id or Path(ply_path).stem
    asset.metadata.display_name = display_name or Path(ply_path).stem

    if vrm_path:
        asset.import_skeleton_from_vrm(vrm_path)
        asset.compute_skinning_from_vrm(vrm_path)

    asset.save(radiance_path)
    return asset


__all__ = [
    'RadianceAsset',
    'RadianceBone',
    'RadianceSkeleton',
    'RadianceMetadata',
    'SpringChain',
    'SpringCollider',
    'BodyRegion',
    'BODY_REGION_NAMES',
    'load_radiance',
    'save_radiance',
    'ply_to_radiance',
]
