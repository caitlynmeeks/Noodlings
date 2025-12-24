"""
RadianceSceneBuilder - Composes multiple RadianceComponents into a renderable scene.

This is where the magic happens:
- Collect all entities with RadianceComponents
- Separate static (GPU-resident) from dynamic (CPU-accessible)
- Build unified render buffers
- Enable semantic queries across the whole scene

The hybrid approach:
- Static environment: loaded once, GPU-resident
- Dynamic characters: CPU holds semantic cache, GPU renders
- Queries: always CPU-side, instant

Author: Caitlyn + Claude (NinaK)
Date: December 2025
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Scene Data Structures
# =============================================================================

@dataclass
class SceneLight:
    """A light in the scene."""
    light_type: str = "point"  # point, directional, spot
    position: Tuple[float, float, float] = (0.0, 5.0, 0.0)
    direction: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 10.0
    spot_angle: float = 45.0
    cast_shadows: bool = True


@dataclass
class SceneCamera:
    """Camera for rendering."""
    position: Tuple[float, float, float] = (0.0, 1.5, 3.0)
    target: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 60.0
    near: float = 0.1
    far: float = 100.0


@dataclass
class RenderBatch:
    """
    A batch of Gaussians ready for rendering.

    All arrays are concatenated from multiple components.
    entity_indices maps each Gaussian back to its source component.
    """
    # Gaussian data (all concatenated)
    positions: np.ndarray          # (N, 3)
    scales: np.ndarray             # (N, 3)
    rotations: np.ndarray          # (N, 4)
    colors: np.ndarray             # (N, 3)
    opacities: np.ndarray          # (N,)

    # Per-Gaussian metadata
    entity_indices: np.ndarray     # (N,) int - which component
    gaussian_indices: np.ndarray   # (N,) int - index within component

    # Component list (for lookups)
    components: List['RadianceComponent']

    # Stats
    total_gaussians: int = 0
    static_gaussians: int = 0
    dynamic_gaussians: int = 0


@dataclass
class SceneHit:
    """Result of a scene-wide raycast or query."""
    hit: bool = False
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance: float = float('inf')

    # Entity info
    entity_id: str = ""
    entity_type: str = ""
    component_index: int = -1

    # Gaussian info
    gaussian_index: int = -1
    body_part: str = ""
    body_region: str = ""


# =============================================================================
# Radiance Scene Builder
# =============================================================================

class RadianceSceneBuilder:
    """
    Collects RadianceComponents and builds renderable scene data.

    Usage:
        builder = RadianceSceneBuilder()

        # Register components
        builder.add_component(red_radiance)
        builder.add_component(environment_radiance)
        builder.add_component(prop_radiance)

        # Build render batch
        batch = builder.build_render_batch()

        # Render
        renderer.render_batch(batch, camera)

        # Scene-wide queries
        hit = builder.raycast(ray_origin, ray_direction)
        nearby = builder.query_radius(point, radius)
    """

    def __init__(self):
        # Components by entity_id
        self._components: Dict[str, 'RadianceComponent'] = {}

        # Ordered list for rendering
        self._component_list: List['RadianceComponent'] = []

        # Static vs dynamic separation
        self._static_components: List['RadianceComponent'] = []
        self._dynamic_components: List['RadianceComponent'] = []

        # Cached render batch
        self._cached_batch: Optional[RenderBatch] = None
        self._batch_dirty: bool = True

        # Scene-wide spatial index
        self._scene_positions: Optional[np.ndarray] = None
        self._scene_entity_map: Optional[np.ndarray] = None  # Maps position index to component
        self._scene_spatial_index: Optional[Any] = None

        # Lighting
        self.lights: List[SceneLight] = []

        # Environment
        self.ambient_color: Tuple[float, float, float] = (0.2, 0.2, 0.2)
        self.environment_intensity: float = 1.0

    # =========================================================================
    # Component Management
    # =========================================================================

    def add_component(self, component: 'RadianceComponent') -> bool:
        """
        Add a RadianceComponent to the scene.

        Args:
            component: The component to add

        Returns:
            True if added successfully
        """
        if not component.is_loaded:
            logger.warning(f"Cannot add unloaded component: {component.entity_id}")
            return False

        entity_id = component.entity_id or f"entity_{len(self._components)}"

        if entity_id in self._components:
            logger.warning(f"Replacing existing component: {entity_id}")

        self._components[entity_id] = component
        self._rebuild_component_lists()
        self._batch_dirty = True

        logger.debug(f"Added component: {entity_id} "
                    f"({component.gaussian_count} Gaussians, "
                    f"mode={component.render_mode.value})")
        return True

    def remove_component(self, entity_id: str) -> bool:
        """Remove a component from the scene."""
        if entity_id in self._components:
            del self._components[entity_id]
            self._rebuild_component_lists()
            self._batch_dirty = True
            return True
        return False

    def get_component(self, entity_id: str) -> Optional['RadianceComponent']:
        """Get a component by entity ID."""
        return self._components.get(entity_id)

    def clear(self):
        """Remove all components."""
        self._components.clear()
        self._component_list.clear()
        self._static_components.clear()
        self._dynamic_components.clear()
        self._cached_batch = None
        self._batch_dirty = True
        self._scene_spatial_index = None

    def _rebuild_component_lists(self):
        """Rebuild ordered component lists."""
        from ..radiance_component import RenderMode

        self._component_list = list(self._components.values())
        self._static_components = [
            c for c in self._component_list
            if c.render_mode == RenderMode.STATIC
        ]
        self._dynamic_components = [
            c for c in self._component_list
            if c.render_mode != RenderMode.STATIC
        ]

    # =========================================================================
    # Render Batch Building
    # =========================================================================

    def build_render_batch(self, force_rebuild: bool = False) -> Optional[RenderBatch]:
        """
        Build a render batch from all components.

        Args:
            force_rebuild: Rebuild even if not dirty

        Returns:
            RenderBatch ready for renderer
        """
        if not force_rebuild and not self._batch_dirty and self._cached_batch:
            return self._cached_batch

        if not self._component_list:
            return None

        # Collect data from all visible components
        all_positions = []
        all_scales = []
        all_rotations = []
        all_colors = []
        all_opacities = []
        all_entity_indices = []
        all_gaussian_indices = []

        visible_components = []
        static_count = 0
        dynamic_count = 0

        for comp_idx, component in enumerate(self._component_list):
            if not component.visible:
                continue

            render_data = component.get_render_data()
            if render_data is None:
                continue

            n = len(render_data['positions'])
            if n == 0:
                continue

            visible_components.append(component)

            all_positions.append(render_data['positions'])
            all_scales.append(render_data['scales'])
            all_rotations.append(render_data['rotations'])
            all_colors.append(render_data['colors'])
            all_opacities.append(render_data['opacities'])

            # Track which component each Gaussian belongs to
            all_entity_indices.append(np.full(n, len(visible_components) - 1, dtype=np.int32))
            all_gaussian_indices.append(np.arange(n, dtype=np.int32))

            # Count by type
            from ..radiance_component import RenderMode
            if component.render_mode == RenderMode.STATIC:
                static_count += n
            else:
                dynamic_count += n

        if not all_positions:
            return None

        # Concatenate all arrays
        batch = RenderBatch(
            positions=np.vstack(all_positions).astype(np.float32),
            scales=np.vstack(all_scales).astype(np.float32),
            rotations=np.vstack(all_rotations).astype(np.float32),
            colors=np.vstack(all_colors).astype(np.float32),
            opacities=np.concatenate(all_opacities).astype(np.float32),
            entity_indices=np.concatenate(all_entity_indices),
            gaussian_indices=np.concatenate(all_gaussian_indices),
            components=visible_components,
            total_gaussians=len(all_positions[0]) if len(all_positions) == 1 else sum(len(p) for p in all_positions),
            static_gaussians=static_count,
            dynamic_gaussians=dynamic_count,
        )

        # Update stats
        batch.total_gaussians = len(batch.positions)

        self._cached_batch = batch
        self._batch_dirty = False

        # Invalidate scene spatial index
        self._scene_spatial_index = None

        logger.debug(f"Built render batch: {batch.total_gaussians} Gaussians "
                    f"(static={batch.static_gaussians}, dynamic={batch.dynamic_gaussians})")

        return batch

    def mark_dirty(self):
        """Mark the batch as needing rebuild."""
        self._batch_dirty = True
        self._scene_spatial_index = None

    # =========================================================================
    # Scene-Wide Queries
    # =========================================================================

    def _ensure_scene_spatial_index(self):
        """Build scene-wide spatial index."""
        if self._scene_spatial_index is not None:
            return

        batch = self.build_render_batch()
        if batch is None:
            return

        try:
            from scipy.spatial import cKDTree
            self._scene_spatial_index = cKDTree(batch.positions)
            self._scene_positions = batch.positions
            self._scene_entity_map = batch.entity_indices
        except ImportError:
            logger.warning("scipy not available for scene queries")

    def raycast(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        max_distance: float = 100.0
    ) -> SceneHit:
        """
        Cast a ray through the entire scene.

        Args:
            origin: Ray origin
            direction: Ray direction (will be normalized)
            max_distance: Maximum ray distance

        Returns:
            SceneHit with closest hit info
        """
        best_hit = SceneHit()
        best_distance = max_distance

        for comp_idx, component in enumerate(self._component_list):
            if not component.visible:
                continue

            hit_data = component.raycast(origin, direction, best_distance)
            if hit_data and hit_data.get('hit') and hit_data['distance'] < best_distance:
                best_distance = hit_data['distance']
                best_hit = SceneHit(
                    hit=True,
                    position=hit_data['position'],
                    distance=hit_data['distance'],
                    entity_id=component.entity_id,
                    entity_type=component.entity_type,
                    component_index=comp_idx,
                    gaussian_index=hit_data['index'],
                    body_part=hit_data.get('body_part', ''),
                    body_region=hit_data.get('body_region', ''),
                )

        return best_hit

    def query_radius(
        self,
        point: Tuple[float, float, float],
        radius: float
    ) -> List[SceneHit]:
        """
        Find all Gaussians within radius of a point, across all components.

        Args:
            point: Query point
            radius: Search radius

        Returns:
            List of SceneHit for each match
        """
        results = []

        for comp_idx, component in enumerate(self._component_list):
            if not component.visible:
                continue

            indices = component.query_radius(point, radius)
            for idx in indices:
                pos = component.get_position(idx)
                if pos:
                    results.append(SceneHit(
                        hit=True,
                        position=pos,
                        distance=np.linalg.norm(np.array(pos) - np.array(point)),
                        entity_id=component.entity_id,
                        entity_type=component.entity_type,
                        component_index=comp_idx,
                        gaussian_index=idx,
                        body_part=component.asset.get_semantic_label(idx) if component.asset else '',
                        body_region=component.asset.get_body_region(idx) if component.asset else '',
                    ))

        # Sort by distance
        results.sort(key=lambda h: h.distance)
        return results

    def query_semantic(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SceneHit]:
        """
        Natural language query across all components.

        Requires CLIP embeddings in assets.

        Args:
            query: Natural language query (e.g., "left hand")
            top_k: Number of results

        Returns:
            List of SceneHit sorted by relevance
        """
        from .semantic_query import get_semantic_query_engine

        engine = get_semantic_query_engine()
        if engine is None:
            logger.warning("Semantic query engine not initialized")
            return []

        # Query returns matches across registered entities
        result = engine.query_text(query, top_k=top_k)
        if not result:
            return []

        hits = []
        for match in result.matches:
            # Find the component
            component = self._components.get(match.entity_id)
            if component:
                comp_idx = self._component_list.index(component) if component in self._component_list else -1
                pos = component.get_position(match.gaussian_index) if match.gaussian_index >= 0 else None

                hits.append(SceneHit(
                    hit=True,
                    position=pos or (0, 0, 0),
                    entity_id=match.entity_id,
                    entity_type=component.entity_type,
                    component_index=comp_idx,
                    gaussian_index=match.gaussian_index,
                    body_part=match.body_part,
                    body_region=match.body_region if hasattr(match, 'body_region') else '',
                ))

        return hits

    # =========================================================================
    # Lighting
    # =========================================================================

    def add_light(self, light: SceneLight):
        """Add a light to the scene."""
        self.lights.append(light)

    def remove_light(self, index: int):
        """Remove a light by index."""
        if 0 <= index < len(self.lights):
            self.lights.pop(index)

    def clear_lights(self):
        """Remove all lights."""
        self.lights.clear()

    def add_point_light(
        self,
        position: Tuple[float, float, float],
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        range: float = 10.0
    ):
        """Convenience method to add a point light."""
        self.add_light(SceneLight(
            light_type="point",
            position=position,
            color=color,
            intensity=intensity,
            range=range,
        ))

    def add_directional_light(
        self,
        direction: Tuple[float, float, float],
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        intensity: float = 1.0
    ):
        """Convenience method to add a directional light."""
        self.add_light(SceneLight(
            light_type="directional",
            direction=direction,
            color=color,
            intensity=intensity,
        ))

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get scene statistics."""
        total_gaussians = 0
        static_gaussians = 0
        dynamic_gaussians = 0

        from ..radiance_component import RenderMode

        for component in self._component_list:
            n = component.gaussian_count
            total_gaussians += n
            if component.render_mode == RenderMode.STATIC:
                static_gaussians += n
            else:
                dynamic_gaussians += n

        return {
            'component_count': len(self._component_list),
            'static_components': len(self._static_components),
            'dynamic_components': len(self._dynamic_components),
            'total_gaussians': total_gaussians,
            'static_gaussians': static_gaussians,
            'dynamic_gaussians': dynamic_gaussians,
            'light_count': len(self.lights),
            'batch_dirty': self._batch_dirty,
        }


# =============================================================================
# Global Scene Builder Instance
# =============================================================================

_scene_builder: Optional[RadianceSceneBuilder] = None


def get_scene_builder() -> RadianceSceneBuilder:
    """Get the global scene builder instance."""
    global _scene_builder
    if _scene_builder is None:
        _scene_builder = RadianceSceneBuilder()
    return _scene_builder


def reset_scene_builder():
    """Reset the global scene builder."""
    global _scene_builder
    if _scene_builder:
        _scene_builder.clear()
    _scene_builder = RadianceSceneBuilder()


# =============================================================================
# Type hints
# =============================================================================

if False:  # TYPE_CHECKING
    from ..radiance_component import RadianceComponent


__all__ = [
    'RadianceSceneBuilder',
    'RenderBatch',
    'SceneLight',
    'SceneCamera',
    'SceneHit',
    'get_scene_builder',
    'reset_scene_builder',
]
