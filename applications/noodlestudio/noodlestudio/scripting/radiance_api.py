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
#   Radiance API - Scripting interface for Gaussian splat visual components.
#
#   Provides JavaScript-accessible interface for: - Loading a...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.scripting.radiance_api
# PURPOSE:  Radiance Api
# LAYER:    Studio / Scripting API
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   RadianceComponentJS, SceneBuilderJS, RadianceAPI, get_radiance_api(), reset_radiance_api()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# JS-friendly wrapper classes
# =============================================================================

class RadianceComponentJS:
    """
    JavaScript-friendly wrapper for RadianceComponent.

    Provides simplified methods that work well with JavaScript interop.
    """

    def __init__(self, component: 'RadianceComponent'):
        self._component = component

    @property
    def entity_id(self) -> str:
        return self._component.entity_id

    @property
    def is_loaded(self) -> bool:
        return self._component.is_loaded

    @property
    def gaussian_count(self) -> int:
        return self._component.gaussian_count

    @property
    def visible(self) -> bool:
        return self._component.visible

    @visible.setter
    def visible(self, value: bool):
        self._component.visible = value

    # =========================================================================
    # Transform
    # =========================================================================

    def set_position(self, x: float, y: float, z: float):
        """Set world position."""
        self._component.set_position(x, y, z)

    def set_rotation(self, x: float, y: float, z: float):
        """Set rotation (euler degrees)."""
        self._component.set_rotation(x, y, z)

    def set_scale(self, x: float, y: float, z: float):
        """Set scale."""
        self._component.set_scale(x, y, z)

    def get_position(self) -> Dict[str, float]:
        """Get current position."""
        pos = self._component.transform.position
        return {'x': pos[0], 'y': pos[1], 'z': pos[2]}

    def get_rotation(self) -> Dict[str, float]:
        """Get current rotation (euler degrees)."""
        rot = self._component.transform.rotation
        return {'x': rot[0], 'y': rot[1], 'z': rot[2]}

    # =========================================================================
    # Entity-Level Material
    # =========================================================================

    def set_tint(self, r: float, g: float, b: float, a: float = 1.0):
        """Set entity-wide tint color."""
        self._component.set_tint(r, g, b, a)

    def set_emission(self, r: float, g: float, b: float):
        """Set entity-wide emission color."""
        self._component.set_emission(r, g, b)

    def set_alpha(self, alpha: float):
        """Set entity-wide alpha multiplier."""
        self._component.set_alpha(alpha)

    def get_tint(self) -> Dict[str, float]:
        """Get current tint color."""
        t = self._component.material.tint
        return {'r': t.r, 'g': t.g, 'b': t.b, 'a': t.a}

    def get_emission(self) -> Dict[str, float]:
        """Get current emission color."""
        e = self._component.material.emission
        return {'r': e.r, 'g': e.g, 'b': e.b}

    # =========================================================================
    # Region Overrides
    # =========================================================================

    def set_region_override(self, region: str, override: Dict[str, Any]):
        """
        Set override for a body region.

        Args:
            region: Body region name (e.g., "left_arm", "head", "torso")
            override: Dict with optional keys:
                - tint: {r, g, b, a}
                - emission: {r, g, b}
                - alpha_mult: float
                - visible: bool

        Example:
            radiance.set_region_override("left_arm", {
                tint: {r: 0.5, g: 0.5, b: 1.0},
                emission: {r: 0, g: 0, b: 0.3}
            });
        """
        from ..core.radiance_component import RegionOverride, Color

        region_override = RegionOverride()

        if 'tint' in override:
            t = override['tint']
            region_override.tint = Color(
                t.get('r', 1), t.get('g', 1), t.get('b', 1), t.get('a', 1)
            )

        if 'emission' in override:
            e = override['emission']
            region_override.emission = Color(
                e.get('r', 0), e.get('g', 0), e.get('b', 0), 1.0
            )

        if 'alpha_mult' in override:
            region_override.alpha_mult = float(override['alpha_mult'])

        if 'visible' in override:
            region_override.visible = bool(override['visible'])

        self._component.set_region_override(region, region_override)

    def clear_region_override(self, region: str):
        """Clear override for a body region."""
        self._component.clear_region_override(region)

    def clear_all_region_overrides(self):
        """Clear all region overrides."""
        self._component.clear_all_region_overrides()

    def get_body_regions(self) -> List[str]:
        """Get list of body regions in this asset."""
        return list(self._component.body_regions)

    # =========================================================================
    # Per-Gaussian Overrides
    # =========================================================================

    def set_gaussian_override(self, index: int, override: Dict[str, Any]):
        """
        Set override for a specific Gaussian.

        Args:
            index: Gaussian index
            override: Dict with optional keys:
                - tint: {r, g, b}
                - emission: {r, g, b}
                - alpha: float
                - scale_mult: float
                - visible: bool

        Example:
            // Create burn mark
            radiance.set_gaussian_override(idx, {
                tint: {r: 0.2, g: 0.2, b: 0.2}
            });
        """
        from ..core.radiance_component import GaussianOverride, Color

        gauss_override = GaussianOverride()

        if 'tint' in override:
            t = override['tint']
            gauss_override.tint = Color(t.get('r', 1), t.get('g', 1), t.get('b', 1), 1)

        if 'emission' in override:
            e = override['emission']
            gauss_override.emission = Color(e.get('r', 0), e.get('g', 0), e.get('b', 0), 1)

        if 'alpha' in override:
            gauss_override.alpha = float(override['alpha'])

        if 'scale_mult' in override:
            gauss_override.scale_mult = float(override['scale_mult'])

        if 'visible' in override:
            gauss_override.visible = bool(override['visible'])

        self._component.set_gaussian_override(index, gauss_override)

    def clear_gaussian_override(self, index: int):
        """Clear override for a specific Gaussian."""
        self._component.clear_gaussian_override(index)

    def clear_all_gaussian_overrides(self):
        """Clear all per-Gaussian overrides."""
        self._component.clear_all_gaussian_overrides()

    # =========================================================================
    # Spatial Queries
    # =========================================================================

    def query_radius(self, x: float, y: float, z: float, radius: float) -> List[int]:
        """
        Find all Gaussians within radius of a point.

        Returns list of Gaussian indices.
        """
        return self._component.query_radius((x, y, z), radius)

    def query_nearest(self, x: float, y: float, z: float, k: int = 1) -> List[Dict[str, Any]]:
        """
        Find k nearest Gaussians to a point.

        Returns list of {index, distance} dicts.
        """
        results = self._component.query_nearest((x, y, z), k)
        return [{'index': idx, 'distance': dist} for idx, dist in results]

    def raycast(
        self,
        ox: float, oy: float, oz: float,  # origin
        dx: float, dy: float, dz: float,  # direction
        max_distance: float = 100.0
    ) -> Optional[Dict[str, Any]]:
        """
        Cast a ray and find the first Gaussian hit.

        Returns hit info dict or None.
        """
        return self._component.raycast((ox, oy, oz), (dx, dy, dz), max_distance)

    def get_gaussian_position(self, index: int) -> Optional[Dict[str, float]]:
        """Get world position of a Gaussian."""
        pos = self._component.get_position(index)
        if pos:
            return {'x': pos[0], 'y': pos[1], 'z': pos[2]}
        return None

    def get_semantic_label(self, index: int) -> str:
        """Get semantic label (body part) for a Gaussian."""
        if self._component.asset:
            return self._component.asset.get_semantic_label(index)
        return ""

    def get_body_region(self, index: int) -> str:
        """Get body region for a Gaussian."""
        if self._component.asset:
            return self._component.asset.get_body_region(index)
        return ""

    # =========================================================================
    # Asset Management
    # =========================================================================

    def load_asset(self, path: str) -> bool:
        """Load a .radiance asset."""
        return self._component.load_asset(path)

    def unload_asset(self):
        """Unload the current asset."""
        self._component.unload_asset()


class SceneBuilderJS:
    """
    JavaScript-friendly wrapper for RadianceSceneBuilder.
    """

    def __init__(self, builder: 'RadianceSceneBuilder'):
        self._builder = builder

    def get_stats(self) -> Dict[str, Any]:
        """Get scene statistics."""
        return self._builder.get_stats()

    def raycast(
        self,
        ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float,
        max_distance: float = 100.0
    ) -> Dict[str, Any]:
        """
        Cast a ray through the entire scene.

        Returns hit info dict.
        """
        hit = self._builder.raycast((ox, oy, oz), (dx, dy, dz), max_distance)
        return {
            'hit': hit.hit,
            'position': {'x': hit.position[0], 'y': hit.position[1], 'z': hit.position[2]},
            'distance': hit.distance,
            'entity_id': hit.entity_id,
            'entity_type': hit.entity_type,
            'gaussian_index': hit.gaussian_index,
            'body_part': hit.body_part,
            'body_region': hit.body_region,
        }

    def query_radius(
        self,
        x: float, y: float, z: float,
        radius: float
    ) -> List[Dict[str, Any]]:
        """
        Find all Gaussians within radius across all components.
        """
        hits = self._builder.query_radius((x, y, z), radius)
        return [
            {
                'entity_id': h.entity_id,
                'gaussian_index': h.gaussian_index,
                'distance': h.distance,
                'body_part': h.body_part,
                'body_region': h.body_region,
            }
            for h in hits
        ]

    def add_point_light(
        self,
        x: float, y: float, z: float,
        r: float = 1.0, g: float = 1.0, b: float = 1.0,
        intensity: float = 1.0,
        range: float = 10.0
    ):
        """Add a point light to the scene."""
        self._builder.add_point_light(
            (x, y, z), (r, g, b), intensity, range
        )

    def add_directional_light(
        self,
        dx: float, dy: float, dz: float,
        r: float = 1.0, g: float = 1.0, b: float = 1.0,
        intensity: float = 1.0
    ):
        """Add a directional light to the scene."""
        self._builder.add_directional_light(
            (dx, dy, dz), (r, g, b), intensity
        )

    def clear_lights(self):
        """Remove all lights."""
        self._builder.clear_lights()


# =============================================================================
# Main Radiance API
# =============================================================================

class RadianceAPI:
    """
    Radiance scripting API.

    Provides access to RadianceComponents and scene composition
    from JavaScript in ScriptedFacets.

    Example:
        function process(inputs, context) {
            // Get component for an entity
            var red = context.noodle.radiance.get("red_fire_anklebiter");

            // Tint it red when angry
            if (inputs.affect.valence < -0.5) {
                red.set_tint(1.0, 0.3, 0.3);
                red.set_region_override("head", {
                    emission: {r: 0.5, g: 0, b: 0}
                });
            }

            // Query the scene
            var hit = context.noodle.radiance.scene.raycast(0, 1, 0, 0, 0, -1);
            if (hit.hit) {
                console.log("Looking at: " + hit.body_part);
            }
        }
    """

    def __init__(self):
        # Component registry (entity_id -> RadianceComponentJS)
        self._components: Dict[str, RadianceComponentJS] = {}

        # Scene builder (lazy init)
        self._scene_builder: Optional['RadianceSceneBuilder'] = None
        self._scene_js: Optional[SceneBuilderJS] = None

    def get(self, entity_id: str) -> Optional[RadianceComponentJS]:
        """
        Get RadianceComponent wrapper for an entity.

        Args:
            entity_id: Entity ID

        Returns:
            RadianceComponentJS wrapper or None
        """
        return self._components.get(entity_id)

    def create(self, entity_id: str, asset_path: Optional[str] = None) -> RadianceComponentJS:
        """
        Create a new RadianceComponent.

        Args:
            entity_id: Entity ID
            asset_path: Optional path to .radiance file to load

        Returns:
            RadianceComponentJS wrapper
        """
        from ..core.radiance_component import RadianceComponent

        component = RadianceComponent(entity_id=entity_id)

        if asset_path:
            component.load_asset(asset_path)

        js_wrapper = RadianceComponentJS(component)
        self._components[entity_id] = js_wrapper

        # Add to scene builder if exists
        if self._scene_builder and component.is_loaded:
            self._scene_builder.add_component(component)

        return js_wrapper

    def register(self, component: 'RadianceComponent') -> RadianceComponentJS:
        """
        Register an existing RadianceComponent.

        Args:
            component: The RadianceComponent to register

        Returns:
            RadianceComponentJS wrapper
        """
        js_wrapper = RadianceComponentJS(component)
        self._components[component.entity_id] = js_wrapper

        # Add to scene builder if exists
        if self._scene_builder and component.is_loaded:
            self._scene_builder.add_component(component)

        return js_wrapper

    def remove(self, entity_id: str) -> bool:
        """
        Remove a RadianceComponent.

        Args:
            entity_id: Entity ID to remove

        Returns:
            True if removed
        """
        if entity_id in self._components:
            del self._components[entity_id]

            if self._scene_builder:
                self._scene_builder.remove_component(entity_id)

            return True
        return False

    def list_entities(self) -> List[str]:
        """Get list of all registered entity IDs."""
        return list(self._components.keys())

    @property
    def scene(self) -> SceneBuilderJS:
        """
        Get scene builder wrapper for scene-wide operations.

        Example:
            var hit = context.noodle.radiance.scene.raycast(0, 1, 0, 0, 0, -1);
        """
        if self._scene_js is None:
            from ..core.semantic_world.radiance_scene_builder import get_scene_builder
            self._scene_builder = get_scene_builder()
            self._scene_js = SceneBuilderJS(self._scene_builder)
        return self._scene_js

    def rebuild_scene(self):
        """Force rebuild of scene render batch."""
        if self._scene_builder:
            self._scene_builder.mark_dirty()

    # =========================================================================
    # Convenience methods for common operations
    # =========================================================================

    def set_all_tint(self, r: float, g: float, b: float, a: float = 1.0):
        """Set tint for all components."""
        for wrapper in self._components.values():
            wrapper.set_tint(r, g, b, a)

    def reset_all_overrides(self):
        """Clear all overrides on all components."""
        for wrapper in self._components.values():
            wrapper.clear_all_region_overrides()
            wrapper.clear_all_gaussian_overrides()

    def scene_raycast(
        self,
        ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float,
        max_distance: float = 100.0
    ) -> Dict[str, Any]:
        """Convenience method for scene raycast."""
        return self.scene.raycast(ox, oy, oz, dx, dy, dz, max_distance)


# =============================================================================
# Global API instance
# =============================================================================

_radiance_api: Optional[RadianceAPI] = None


def get_radiance_api() -> RadianceAPI:
    """Get global RadianceAPI instance."""
    global _radiance_api
    if _radiance_api is None:
        _radiance_api = RadianceAPI()
    return _radiance_api


def reset_radiance_api():
    """Reset global RadianceAPI."""
    global _radiance_api
    _radiance_api = RadianceAPI()


# =============================================================================
# Type hints
# =============================================================================

if False:  # TYPE_CHECKING
    from ..core.radiance_component import RadianceComponent
    from ..core.semantic_world.radiance_scene_builder import RadianceSceneBuilder


__all__ = [
    'RadianceAPI',
    'RadianceComponentJS',
    'SceneBuilderJS',
    'get_radiance_api',
    'reset_radiance_api',
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
