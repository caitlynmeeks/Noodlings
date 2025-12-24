"""
RadianceComponent - Visual representation for entities using Gaussian splats.

Like Unity's MeshRenderer, but for semantic Gaussian radiance fields.
Every Gaussian knows what it represents. Every frame is query-able.

The component provides three levels of control:
- Entity-level: tint, emission, visibility (fast, common)
- Region-level: override by body part (semantic, expressive)
- Per-Gaussian: individual control for FX (precise, powerful)

Author: Caitlyn + Claude (NinaK)
Date: December 2025
"""

import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class RenderMode(Enum):
    """How this component is rendered in the scene."""
    STATIC = "static"      # GPU-resident, instanced (environments, props)
    DYNAMIC = "dynamic"    # CPU-accessible, per-frame updates (characters)
    FX = "fx"              # Special effects, per-Gaussian control


class LightingMode(Enum):
    """How lighting is applied to this component."""
    BAKED = "baked"        # Use SH coefficients as-is (fastest)
    DYNAMIC = "dynamic"    # Apply runtime lights (Phase 4)
    UNLIT = "unlit"        # No lighting, pure emission


# =============================================================================
# Override Data Classes
# =============================================================================

@dataclass
class Color:
    """RGBA color."""
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

    def __iter__(self):
        return iter((self.r, self.g, self.b, self.a))

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    def to_rgb(self) -> Tuple[float, float, float]:
        return (self.r, self.g, self.b)

    @classmethod
    def from_tuple(cls, t: Tuple) -> 'Color':
        if len(t) == 3:
            return cls(t[0], t[1], t[2], 1.0)
        return cls(t[0], t[1], t[2], t[3])

    @classmethod
    def white(cls) -> 'Color':
        return cls(1.0, 1.0, 1.0, 1.0)

    @classmethod
    def black(cls) -> 'Color':
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def clear(cls) -> 'Color':
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclass
class MaterialOverride:
    """Material properties that can be overridden at runtime."""
    tint: Color = field(default_factory=Color.white)
    emission: Color = field(default_factory=Color.clear)
    alpha_mult: float = 1.0
    roughness_mult: float = 1.0
    scale_mult: float = 1.0  # Gaussian size multiplier

    def is_default(self) -> bool:
        """Check if this override has any effect."""
        return (
            self.tint.r == 1.0 and self.tint.g == 1.0 and
            self.tint.b == 1.0 and self.tint.a == 1.0 and
            self.emission.r == 0.0 and self.emission.g == 0.0 and
            self.emission.b == 0.0 and
            self.alpha_mult == 1.0 and
            self.roughness_mult == 1.0 and
            self.scale_mult == 1.0
        )


@dataclass
class GaussianOverride:
    """Per-Gaussian override (sparse - only stored when modified)."""
    tint: Optional[Color] = None
    emission: Optional[Color] = None
    alpha: Optional[float] = None
    scale_mult: Optional[float] = None
    visible: bool = True


@dataclass
class RegionOverride:
    """Override for a semantic region (e.g., "left_arm", "head")."""
    tint: Optional[Color] = None
    emission: Optional[Color] = None
    alpha_mult: Optional[float] = None
    visible: bool = True


# =============================================================================
# Transform
# =============================================================================

@dataclass
class Transform3D:
    """3D transform with position, rotation (euler), and scale."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Euler degrees
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        # Scale
        S = np.diag([self.scale[0], self.scale[1], self.scale[2], 1.0])

        # Rotation (euler to matrix)
        rx, ry, rz = np.radians(self.rotation)

        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        R = Rz @ Ry @ Rx

        # Translation
        T = np.eye(4)
        T[0, 3] = self.position[0]
        T[1, 3] = self.position[1]
        T[2, 3] = self.position[2]

        return T @ R @ S


# =============================================================================
# Radiance Component
# =============================================================================

class RadianceComponent:
    """
    Visual representation component using Gaussian splats.

    Attach to any entity (Noodling, Prim, etc.) to give it a visual form.

    Usage:
        # Create and attach
        radiance = RadianceComponent()
        radiance.load_asset("red_fire_imp.radiance")

        # Entity-level control
        radiance.material.tint = Color(1, 0.5, 0.5)  # Pink tint
        radiance.material.emission = Color(0.2, 0, 0)  # Red glow

        # Region-level control
        radiance.set_region_override("left_arm", RegionOverride(
            tint=Color(0.5, 0.5, 1.0),  # Blue arm
            emission=Color(0, 0, 0.3),   # Glowing
        ))

        # Per-Gaussian control (for FX)
        for i in radiance.query_radius(point, 0.1):
            radiance.set_gaussian_override(i, GaussianOverride(
                tint=Color(0.2, 0.2, 0.2),  # Burn mark
            ))
    """

    def __init__(self, entity_id: str = ""):
        # Identity
        self.entity_id: str = entity_id
        self.entity_type: str = "noodling"

        # Asset reference
        self._asset_path: str = ""
        self._asset: Optional['RadianceAsset'] = None
        self._asset_dirty: bool = False

        # Transform (world-space placement)
        self.transform: Transform3D = Transform3D()

        # Render settings
        self.render_mode: RenderMode = RenderMode.DYNAMIC
        self.lighting_mode: LightingMode = LightingMode.BAKED

        # Visibility
        self.visible: bool = True
        self.cast_shadows: bool = True
        self.receive_shadows: bool = True

        # Entity-level material override
        self.material: MaterialOverride = MaterialOverride()

        # Region-level overrides (by body part name)
        self._region_overrides: Dict[str, RegionOverride] = {}

        # Per-Gaussian overrides (sparse dict: index -> override)
        self._gaussian_overrides: Dict[int, GaussianOverride] = {}

        # Skeleton/animation state
        self._skeleton_pose: Optional[Dict[str, Tuple[float, float, float]]] = None

        # Cached world-space data (for queries)
        self._world_positions: Optional[np.ndarray] = None
        self._world_positions_dirty: bool = True
        self._spatial_index: Optional[Any] = None  # BVH or KDTree

        # Dirty flags for renderer
        self._transform_dirty: bool = True
        self._material_dirty: bool = True
        self._overrides_dirty: bool = True

    # =========================================================================
    # Asset Loading
    # =========================================================================

    @property
    def asset_path(self) -> str:
        return self._asset_path

    @property
    def asset(self) -> Optional['RadianceAsset']:
        return self._asset

    @property
    def is_loaded(self) -> bool:
        return self._asset is not None

    def load_asset(self, path: str) -> bool:
        """
        Load a .radiance asset.

        Args:
            path: Path to .radiance file

        Returns:
            True if loaded successfully
        """
        from .semantic_world.radiance_format import load_radiance

        try:
            resolved_path = Path(path)
            if not resolved_path.exists():
                logger.error(f"Radiance asset not found: {path}")
                return False

            self._asset = load_radiance(str(resolved_path))
            self._asset_path = str(resolved_path)
            self._asset_dirty = False
            self._world_positions_dirty = True
            self._spatial_index = None

            # Auto-detect render mode
            if self._asset.has_skeleton:
                self.render_mode = RenderMode.DYNAMIC
            else:
                self.render_mode = RenderMode.STATIC

            logger.info(f"Loaded radiance: {path} "
                       f"({self._asset.gaussian_count} Gaussians, "
                       f"skeleton={self._asset.has_skeleton})")
            return True

        except Exception as e:
            logger.error(f"Failed to load radiance {path}: {e}")
            return False

    def unload_asset(self):
        """Unload the current asset."""
        self._asset = None
        self._asset_path = ""
        self._world_positions = None
        self._spatial_index = None
        self._gaussian_overrides.clear()
        self._region_overrides.clear()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def gaussian_count(self) -> int:
        """Number of Gaussians in the asset."""
        if self._asset:
            return self._asset.gaussian_count
        return 0

    @property
    def has_skeleton(self) -> bool:
        """Does this asset have a skeleton for animation?"""
        if self._asset:
            return self._asset.has_skeleton
        return False

    @property
    def bone_names(self) -> List[str]:
        """Get list of bone names."""
        if self._asset and self._asset.skeleton:
            return [b.name for b in self._asset.skeleton.bones]
        return []

    @property
    def body_regions(self) -> Set[str]:
        """Get set of unique body regions in this asset."""
        if not self._asset or not self._asset.has_semantics:
            return set()

        regions = set()
        for i in range(self._asset.gaussian_count):
            regions.add(self._asset.get_body_region(i))
        return regions

    # =========================================================================
    # Transform
    # =========================================================================

    def set_position(self, x: float, y: float, z: float):
        """Set world position."""
        self.transform.position = (x, y, z)
        self._transform_dirty = True
        self._world_positions_dirty = True

    def set_rotation(self, x: float, y: float, z: float):
        """Set rotation (euler degrees)."""
        self.transform.rotation = (x, y, z)
        self._transform_dirty = True
        self._world_positions_dirty = True

    def set_scale(self, x: float, y: float, z: float):
        """Set scale."""
        self.transform.scale = (x, y, z)
        self._transform_dirty = True
        self._world_positions_dirty = True

    # =========================================================================
    # Entity-Level Material
    # =========================================================================

    def set_tint(self, r: float, g: float, b: float, a: float = 1.0):
        """Set entity-wide tint color."""
        self.material.tint = Color(r, g, b, a)
        self._material_dirty = True

    def set_emission(self, r: float, g: float, b: float):
        """Set entity-wide emission color."""
        self.material.emission = Color(r, g, b, 1.0)
        self._material_dirty = True

    def set_alpha(self, alpha: float):
        """Set entity-wide alpha multiplier."""
        self.material.alpha_mult = max(0.0, min(1.0, alpha))
        self._material_dirty = True

    # =========================================================================
    # Region-Level Overrides
    # =========================================================================

    def set_region_override(self, region: str, override: RegionOverride):
        """
        Set override for a body region.

        Args:
            region: Body region name (e.g., "left_arm", "head", "torso")
            override: The override to apply
        """
        self._region_overrides[region] = override
        self._overrides_dirty = True

    def clear_region_override(self, region: str):
        """Clear override for a body region."""
        if region in self._region_overrides:
            del self._region_overrides[region]
            self._overrides_dirty = True

    def clear_all_region_overrides(self):
        """Clear all region overrides."""
        self._region_overrides.clear()
        self._overrides_dirty = True

    def get_region_override(self, region: str) -> Optional[RegionOverride]:
        """Get override for a region, if any."""
        return self._region_overrides.get(region)

    # =========================================================================
    # Per-Gaussian Overrides
    # =========================================================================

    def set_gaussian_override(self, index: int, override: GaussianOverride):
        """
        Set override for a specific Gaussian.

        Args:
            index: Gaussian index
            override: The override to apply
        """
        if 0 <= index < self.gaussian_count:
            self._gaussian_overrides[index] = override
            self._overrides_dirty = True

    def clear_gaussian_override(self, index: int):
        """Clear override for a specific Gaussian."""
        if index in self._gaussian_overrides:
            del self._gaussian_overrides[index]
            self._overrides_dirty = True

    def clear_all_gaussian_overrides(self):
        """Clear all per-Gaussian overrides."""
        self._gaussian_overrides.clear()
        self._overrides_dirty = True

    def get_gaussian_override(self, index: int) -> Optional[GaussianOverride]:
        """Get override for a Gaussian, if any."""
        return self._gaussian_overrides.get(index)

    # =========================================================================
    # Spatial Queries
    # =========================================================================

    def _ensure_world_positions(self):
        """Ensure world positions are computed."""
        if not self._world_positions_dirty and self._world_positions is not None:
            return

        if not self._asset or self._asset.positions is None:
            self._world_positions = None
            return

        # Get local positions
        local_pos = self._asset.positions

        # Apply skeleton pose if present
        if self._skeleton_pose and self._asset.has_skeleton and self._asset.has_skinning:
            local_pos = self._asset.apply_skinning(self._skeleton_pose)

        # Transform to world space
        matrix = self.transform.to_matrix()

        # Homogeneous coordinates
        ones = np.ones((len(local_pos), 1), dtype=np.float32)
        homogeneous = np.hstack([local_pos, ones])

        # Transform
        world_homogeneous = (matrix @ homogeneous.T).T
        self._world_positions = world_homogeneous[:, :3].astype(np.float32)

        self._world_positions_dirty = False
        self._spatial_index = None  # Invalidate spatial index

    def _ensure_spatial_index(self):
        """Build spatial index for fast queries."""
        if self._spatial_index is not None:
            return

        self._ensure_world_positions()
        if self._world_positions is None:
            return

        try:
            from scipy.spatial import cKDTree
            self._spatial_index = cKDTree(self._world_positions)
        except ImportError:
            logger.warning("scipy not available, spatial queries will be slow")

    def get_position(self, index: int) -> Optional[Tuple[float, float, float]]:
        """Get world position of a Gaussian."""
        self._ensure_world_positions()
        if self._world_positions is None or index < 0 or index >= len(self._world_positions):
            return None
        return tuple(self._world_positions[index])

    def query_radius(self, point: Tuple[float, float, float], radius: float) -> List[int]:
        """
        Find all Gaussians within radius of a point.

        Args:
            point: World-space query point
            radius: Search radius

        Returns:
            List of Gaussian indices
        """
        self._ensure_spatial_index()
        if self._spatial_index is None:
            return []

        indices = self._spatial_index.query_ball_point(point, radius)
        return indices

    def query_nearest(self, point: Tuple[float, float, float], k: int = 1) -> List[Tuple[int, float]]:
        """
        Find k nearest Gaussians to a point.

        Args:
            point: World-space query point
            k: Number of neighbors

        Returns:
            List of (index, distance) tuples
        """
        self._ensure_spatial_index()
        if self._spatial_index is None:
            return []

        distances, indices = self._spatial_index.query(point, k=k)
        if k == 1:
            return [(int(indices), float(distances))]
        return [(int(i), float(d)) for i, d in zip(indices, distances)]

    def raycast(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        max_distance: float = 100.0
    ) -> Optional[Dict[str, Any]]:
        """
        Cast a ray and find the first Gaussian hit.

        Args:
            origin: Ray origin (world space)
            direction: Ray direction (normalized)
            max_distance: Maximum ray distance

        Returns:
            Hit info dict or None
        """
        self._ensure_world_positions()
        if self._world_positions is None or self._asset is None:
            return None

        # Normalize direction
        dir_arr = np.array(direction, dtype=np.float32)
        dir_arr = dir_arr / np.linalg.norm(dir_arr)
        origin_arr = np.array(origin, dtype=np.float32)

        # Simple ray-sphere test for each Gaussian
        # TODO: Use BVH for acceleration
        best_t = max_distance
        best_idx = -1

        positions = self._world_positions
        scales = self._asset.scales if self._asset.scales is not None else np.ones((len(positions), 3)) * 0.01

        for i in range(len(positions)):
            # Sphere radius from max scale
            radius = float(np.max(scales[i])) * 2.0

            # Ray-sphere intersection
            oc = origin_arr - positions[i]
            b = np.dot(oc, dir_arr)
            c = np.dot(oc, oc) - radius * radius
            discriminant = b * b - c

            if discriminant > 0:
                t = -b - np.sqrt(discriminant)
                if 0 < t < best_t:
                    best_t = t
                    best_idx = i

        if best_idx < 0:
            return None

        hit_point = origin_arr + dir_arr * best_t

        return {
            'hit': True,
            'index': best_idx,
            'distance': float(best_t),
            'position': tuple(hit_point),
            'body_part': self._asset.get_semantic_label(best_idx) if self._asset else "",
            'body_region': self._asset.get_body_region(best_idx) if self._asset else "",
            'entity_id': self.entity_id,
        }

    # =========================================================================
    # Animation
    # =========================================================================

    def set_skeleton_pose(self, pose: Dict[str, Tuple[float, float, float]]):
        """
        Set skeleton pose for LBS deformation.

        Args:
            pose: Dict mapping bone name to euler rotation (degrees)
        """
        self._skeleton_pose = pose
        self._world_positions_dirty = True
        self._transform_dirty = True

    def clear_skeleton_pose(self):
        """Reset to bind pose."""
        self._skeleton_pose = None
        self._world_positions_dirty = True
        self._transform_dirty = True

    # =========================================================================
    # Render Data Export
    # =========================================================================

    def get_render_data(self) -> Optional[Dict[str, Any]]:
        """
        Get data needed for rendering.

        Returns dict with:
        - positions: (N, 3) world-space positions
        - scales: (N, 3) scales
        - rotations: (N, 4) quaternions
        - colors: (N, 3) colors with overrides applied
        - opacities: (N,) opacities with overrides applied
        - entity_id: for click-to-inspect
        """
        if not self._asset:
            return None

        self._ensure_world_positions()

        # Start with asset data
        positions = self._world_positions
        scales = self._asset.scales.copy() * self.material.scale_mult
        rotations = self._asset.rotations
        opacities = self._asset.opacities.copy() if self._asset.opacities is not None else np.ones(self.gaussian_count)

        # Get base colors from SH DC
        if self._asset.sh_dc is not None:
            # SH DC are in [-inf, inf], convert to [0, 1] RGB
            # Using sigmoid activation
            colors = 1.0 / (1.0 + np.exp(-self._asset.sh_dc))
        else:
            colors = np.ones((self.gaussian_count, 3), dtype=np.float32)

        # Apply entity-level material
        colors = colors * np.array([self.material.tint.r, self.material.tint.g, self.material.tint.b])
        colors = colors + np.array([self.material.emission.r, self.material.emission.g, self.material.emission.b])
        opacities = opacities * self.material.alpha_mult

        # Apply region overrides
        for region, override in self._region_overrides.items():
            if not override.visible:
                # Hide entire region
                for i in range(self.gaussian_count):
                    if self._asset.get_body_region(i) == region:
                        opacities[i] = 0.0
            else:
                for i in range(self.gaussian_count):
                    if self._asset.get_body_region(i) == region:
                        if override.tint:
                            colors[i] *= np.array([override.tint.r, override.tint.g, override.tint.b])
                        if override.emission:
                            colors[i] += np.array([override.emission.r, override.emission.g, override.emission.b])
                        if override.alpha_mult is not None:
                            opacities[i] *= override.alpha_mult

        # Apply per-Gaussian overrides
        for idx, override in self._gaussian_overrides.items():
            if not override.visible:
                opacities[idx] = 0.0
            else:
                if override.tint:
                    colors[idx] *= np.array([override.tint.r, override.tint.g, override.tint.b])
                if override.emission:
                    colors[idx] += np.array([override.emission.r, override.emission.g, override.emission.b])
                if override.alpha is not None:
                    opacities[idx] = override.alpha

        # Clamp colors
        colors = np.clip(colors, 0.0, 1.0)

        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'colors': colors.astype(np.float32),
            'opacities': opacities.astype(np.float32),
            'entity_id': self.entity_id,
            'render_mode': self.render_mode.value,
            'lighting_mode': self.lighting_mode.value,
            'visible': self.visible,
            'cast_shadows': self.cast_shadows,
            'receive_shadows': self.receive_shadows,
        }

    def clear_dirty_flags(self):
        """Clear dirty flags after rendering."""
        self._transform_dirty = False
        self._material_dirty = False
        self._overrides_dirty = False

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'asset_path': self._asset_path,
            'transform': {
                'position': self.transform.position,
                'rotation': self.transform.rotation,
                'scale': self.transform.scale,
            },
            'render_mode': self.render_mode.value,
            'lighting_mode': self.lighting_mode.value,
            'visible': self.visible,
            'cast_shadows': self.cast_shadows,
            'receive_shadows': self.receive_shadows,
            'material': {
                'tint': self.material.tint.to_tuple(),
                'emission': self.material.emission.to_tuple(),
                'alpha_mult': self.material.alpha_mult,
                'roughness_mult': self.material.roughness_mult,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RadianceComponent':
        """Deserialize from dictionary."""
        component = cls(entity_id=data.get('entity_id', ''))
        component.entity_type = data.get('entity_type', 'noodling')

        if 'transform' in data:
            t = data['transform']
            component.transform.position = tuple(t.get('position', (0, 0, 0)))
            component.transform.rotation = tuple(t.get('rotation', (0, 0, 0)))
            component.transform.scale = tuple(t.get('scale', (1, 1, 1)))

        component.render_mode = RenderMode(data.get('render_mode', 'dynamic'))
        component.lighting_mode = LightingMode(data.get('lighting_mode', 'baked'))
        component.visible = data.get('visible', True)
        component.cast_shadows = data.get('cast_shadows', True)
        component.receive_shadows = data.get('receive_shadows', True)

        if 'material' in data:
            m = data['material']
            component.material.tint = Color.from_tuple(m.get('tint', (1, 1, 1, 1)))
            component.material.emission = Color.from_tuple(m.get('emission', (0, 0, 0, 0)))
            component.material.alpha_mult = m.get('alpha_mult', 1.0)
            component.material.roughness_mult = m.get('roughness_mult', 1.0)

        # Load asset if path specified
        if data.get('asset_path'):
            component.load_asset(data['asset_path'])

        return component


# =============================================================================
# Type hints for cross-module imports
# =============================================================================

# Forward reference to avoid circular import
if False:  # TYPE_CHECKING
    from .semantic_world.radiance_format import RadianceAsset


__all__ = [
    'RadianceComponent',
    'RenderMode',
    'LightingMode',
    'Color',
    'MaterialOverride',
    'GaussianOverride',
    'RegionOverride',
    'Transform3D',
]
