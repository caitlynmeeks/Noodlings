"""
RadianceViewport Component

A focused Gaussian splat renderer for the UI canvas system.

This component ONLY concerns itself with:
- Rendering Gaussian splats (RadianceComponents)
- Camera controls (orbit, pan, zoom)
- Semantic query passthrough for informed Gaussians

It does NOT concern itself with:
- What a "noodling" or "prop" is
- Parsing recipe.yaml files
- Stage loading
- Personality traits or any domain concepts

Whatever system needs to display Gaussians sends RadianceComponents here.
The viewport renders them. That's it.

Phase 3c (Jan 2026)
"""

import logging
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

from ..component import UIComponent, register_component

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration for the viewport."""
    distance: float = 3.0
    elevation: float = 15.0  # degrees
    azimuth: float = 180.0   # degrees (180 = facing front)
    target: Tuple[float, float, float] = (0.0, 0.8, 0.0)
    fov: float = 45.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraConfig':
        return cls(
            distance=data.get("distance", 3.0),
            elevation=data.get("elevation", 15.0),
            azimuth=data.get("azimuth", 180.0),
            target=tuple(data.get("target", [0.0, 0.8, 0.0])),
            fov=data.get("fov", 45.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance": self.distance,
            "elevation": self.elevation,
            "azimuth": self.azimuth,
            "target": list(self.target),
            "fov": self.fov,
        }


@register_component
class RadianceViewport(UIComponent):
    """
    3D Gaussian splat viewport component.

    A focused renderer that displays RadianceComponents. Send it components,
    it renders them. Camera controls are built in.

    Properties:
        camera: Camera configuration (distance, elevation, azimuth, target, fov)
        background: Background color (hex string)
        show_skeleton: Whether to display bone skeleton overlay
        interactive: Whether camera controls are enabled

    Events:
        onLoad: Triggered when radiance loads
        onClick: Triggered on click (with 3D ray info for semantic queries)
    """

    component_type = "RadianceViewport"

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.camera = CameraConfig()
        self.background: str = "#000000"
        self.show_skeleton: bool = False
        self.interactive: bool = True

        # Default size
        self.geometry.width = 512
        self.geometry.height = 512

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add RadianceViewport-specific properties to serialization."""
        data["camera"] = self.camera.to_dict()
        if self.background != "#000000":
            data["background"] = self.background
        if self.show_skeleton:
            data["show_skeleton"] = True
        if not self.interactive:
            data["interactive"] = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RadianceViewport':
        """Deserialize from dictionary."""
        viewport = cls(name=data.get("name", ""))

        # Base properties
        viewport.geometry.x = data.get("x", 0)
        viewport.geometry.y = data.get("y", 0)
        viewport.geometry.width = data.get("width", 512)
        viewport.geometry.height = data.get("height", 512)

        if "anchors" in data:
            from ..component import Anchors
            viewport.anchors = Anchors.from_list(data["anchors"])

        viewport.visible = data.get("visible", True)
        viewport.enabled = data.get("enabled", True)

        # RadianceViewport-specific
        if "camera" in data:
            viewport.camera = CameraConfig.from_dict(data["camera"])
        viewport.background = data.get("background", "#000000")
        viewport.show_skeleton = data.get("show_skeleton", False)
        viewport.interactive = data.get("interactive", True)

        return viewport


# ============================================================================
# Qt Widget Implementation
# ============================================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
    from PyQt6.QtCore import Qt, QTimer, QPoint, pyqtSignal
    from PyQt6.QtGui import QPixmap, QImage, QMouseEvent, QWheelEvent
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


if QT_AVAILABLE:
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    import threading

    class RadianceViewportWidget(QWidget):
        """
        Qt widget that renders Gaussian splats.

        This is a focused renderer. Give it RadianceComponents, it renders them.

        Public API:
            set_component(component)  - Set single RadianceComponent
            add_component(component)  - Add to multi-component scene
            clear()                   - Clear all components
            load_file(path)           - Load a .radiance or .ply file

        Semantic Queries:
            raycast(x, y)             - Get semantic info at screen position
            query_radius(pos, r)      - Find Gaussians near a 3D point

        Camera:
            set_camera(distance, elevation, azimuth, target)
            focus_on(position)
            frame_all()
        """

        # Render quality
        PREVIEW_SIZE = 128
        FULL_RENDER_DELAY = 300  # ms

        # Signals
        componentLoaded = pyqtSignal(str, int)  # entity_id, gaussian_count
        renderComplete = pyqtSignal(dict)  # render info

        def __init__(self, component: RadianceViewport, parent=None):
            super().__init__(parent)
            self.component = component
            self.setObjectName(component.name or "radiance_viewport")

            # Renderer
            self._renderer = None

            # Scene content (RadianceSceneBuilder for multi-component)
            self._scene_builder = None

            # Camera state
            self._azimuth = component.camera.azimuth
            self._elevation = component.camera.elevation
            self._distance = component.camera.distance
            self._target = list(component.camera.target)
            self._fov = component.camera.fov

            # Mouse interaction
            self._last_mouse_pos = QPoint()
            self._is_orbiting = False
            self._is_panning = False
            self._is_interacting = False

            # Render state
            self._render_lock = threading.Lock()
            self._is_rendering = False
            self._pending_render = False
            self._executor = ThreadPoolExecutor(max_workers=1)

            # Setup
            self._setup_ui()
            self._setup_renderer()
            self._setup_timers()

        def _setup_ui(self):
            """Create the display label."""
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            self._image_label = QLabel()
            self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._image_label.setStyleSheet(
                f"background-color: {self.component.background};"
            )
            layout.addWidget(self._image_label)

            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def _setup_renderer(self):
            """Initialize the Gaussian renderer."""
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available - viewport disabled")
                return

            try:
                from noodlestudio.core.gaussian_renderer import GaussianRenderer
                self._renderer = GaussianRenderer()
                logger.debug(f"Viewport renderer initialized: {self._renderer.device}")
            except Exception as e:
                logger.warning(f"Failed to init renderer: {e}")

        def _setup_timers(self):
            """Setup render timing."""
            self._render_timer = QTimer()
            self._render_timer.setSingleShot(True)
            self._render_timer.timeout.connect(self._request_full_render)

            self._interaction_timer = QTimer()
            self._interaction_timer.setSingleShot(True)
            self._interaction_timer.timeout.connect(self._end_interaction)

        # =====================================================================
        # Public API: Content Management
        # =====================================================================

        def set_component(self, component: 'RadianceComponent'):
            """
            Set a single RadianceComponent to render.

            Clears any existing scene and renders just this component.
            """
            self._ensure_scene_builder()
            self._scene_builder.clear()
            self._scene_builder.add_component(component)
            self.componentLoaded.emit(
                component.entity_id,
                component.gaussian_count
            )
            self._request_render()

        def add_component(self, component: 'RadianceComponent'):
            """
            Add a RadianceComponent to the scene.

            For multi-component rendering (multiple characters, props, etc.)
            """
            self._ensure_scene_builder()
            self._scene_builder.add_component(component)
            self.componentLoaded.emit(
                component.entity_id,
                component.gaussian_count
            )
            self._request_render()

        def remove_component(self, entity_id: str):
            """Remove a component from the scene by ID."""
            if self._scene_builder:
                self._scene_builder.remove_component(entity_id)
                self._request_render()

        def clear(self):
            """Clear all components from the scene."""
            if self._scene_builder:
                self._scene_builder.clear()
            self._request_render()

        def load_file(self, path: str, entity_id: str = "loaded"):
            """
            Load a .radiance or .ply file directly.

            Args:
                path: Path to the file
                entity_id: ID to assign to the loaded component
            """
            try:
                from noodlestudio.core.radiance_component import RadianceComponent

                component = RadianceComponent(entity_id)
                if component.load_asset(path):
                    self.set_component(component)
                    logger.info(f"Loaded: {path} ({component.gaussian_count:,} Gaussians)")
                else:
                    logger.error(f"Failed to load: {path}")
            except Exception as e:
                logger.error(f"Error loading file: {e}")

        def _ensure_scene_builder(self):
            """Ensure scene builder is initialized."""
            if self._scene_builder is None:
                from noodlestudio.core.semantic_world.radiance_scene_builder import (
                    RadianceSceneBuilder
                )
                self._scene_builder = RadianceSceneBuilder()

        # =====================================================================
        # Public API: Scene Info
        # =====================================================================

        def get_stats(self) -> Dict[str, Any]:
            """Get scene statistics."""
            if self._scene_builder:
                return self._scene_builder.get_stats()
            return {'component_count': 0, 'total_gaussians': 0}

        def get_component(self, entity_id: str) -> Optional['RadianceComponent']:
            """Get a component by entity ID."""
            if self._scene_builder:
                return self._scene_builder.get_component(entity_id)
            return None

        # =====================================================================
        # Public API: Camera Control
        # =====================================================================

        def set_camera(
            self,
            distance: Optional[float] = None,
            elevation: Optional[float] = None,
            azimuth: Optional[float] = None,
            target: Optional[Tuple[float, float, float]] = None,
            fov: Optional[float] = None
        ):
            """Set camera parameters."""
            if distance is not None:
                self._distance = distance
            if elevation is not None:
                self._elevation = max(-89, min(89, elevation))
            if azimuth is not None:
                self._azimuth = azimuth
            if target is not None:
                self._target = list(target)
            if fov is not None:
                self._fov = fov
            self._request_render()

        def focus_on(self, position: Tuple[float, float, float], distance: float = 3.0):
            """Focus camera on a 3D position."""
            self._target = list(position)
            self._distance = distance
            self._request_render()

        def frame_all(self):
            """Frame all content in view."""
            # TODO: Calculate bounds from scene_builder
            self._distance = 5.0
            self._target = [0.0, 0.5, 0.0]
            self._elevation = 20
            self._azimuth = 180
            self._request_render()

        # =====================================================================
        # Public API: Semantic Queries
        # =====================================================================

        def raycast(self, screen_x: int, screen_y: int) -> Optional[Dict[str, Any]]:
            """
            Cast a ray from screen position into the scene.

            Returns semantic info about what was hit (body part, entity, etc.)
            """
            if not self._scene_builder:
                return None

            # TODO: Implement screen-to-world raycast
            # This would use the camera params to construct a ray
            # and query the scene builder
            return None

        def query_at_world_position(
            self,
            position: Tuple[float, float, float],
            radius: float = 0.1
        ) -> list:
            """
            Query Gaussians near a world position.

            Returns list of hits with semantic info.
            """
            if not self._scene_builder:
                return []
            return self._scene_builder.query_radius(position, radius)

        # =====================================================================
        # Rendering
        # =====================================================================

        def _can_render(self) -> bool:
            """Check if we have content to render."""
            if not self._renderer:
                return False
            if not self._scene_builder:
                return False
            stats = self._scene_builder.get_stats()
            return stats.get('total_gaussians', 0) > 0

        def _request_render(self, preview: bool = False):
            """Request a render update."""
            if not self._can_render():
                return

            with self._render_lock:
                if self._is_rendering:
                    self._pending_render = True
                    return
                self._is_rendering = True

            # Determine resolution
            if preview or self._is_interacting:
                width = height = self.PREVIEW_SIZE
            else:
                width = max(1, self.width())
                height = max(1, self.height())

            self._executor.submit(self._do_render, width, height)

        def _request_full_render(self):
            """Request a full-resolution render."""
            self._request_render(preview=False)

        def _do_render(self, width: int, height: int):
            """Perform render in background thread."""
            try:
                from noodlestudio.core.gaussian_renderer import create_orbit_camera

                camera = create_orbit_camera(
                    distance=self._distance,
                    elevation=self._elevation,
                    azimuth=self._azimuth,
                    target=tuple(self._target),
                    fov=self._fov,
                    width=width,
                    height=height,
                )

                bg_color = self._parse_background_color()

                image, alpha, info = self._renderer.render_scene(
                    self._scene_builder, camera, background=bg_color
                )

                QTimer.singleShot(0, lambda: self._on_render_complete(image, alpha, info))

            except Exception as e:
                logger.error(f"Render error: {e}")
                import traceback
                traceback.print_exc()
                with self._render_lock:
                    self._is_rendering = False

        def _parse_background_color(self) -> Tuple[float, float, float]:
            """Parse hex background color to RGB tuple."""
            bg = self.component.background
            if bg.startswith('#') and len(bg) >= 7:
                try:
                    r = int(bg[1:3], 16) / 255.0
                    g = int(bg[3:5], 16) / 255.0
                    b = int(bg[5:7], 16) / 255.0
                    return (r, g, b)
                except (ValueError, IndexError):
                    pass
            return (0.0, 0.0, 0.0)

        def _on_render_complete(self, image, alpha, info):
            """Handle completed render."""
            with self._render_lock:
                self._is_rendering = False
                has_pending = self._pending_render
                self._pending_render = False

            if has_pending:
                self._request_render()
                return

            # Convert to QPixmap and display
            if image is not None:
                if isinstance(image, np.ndarray):
                    h, w = image.shape[:2]
                    if image.shape[2] == 3:
                        qimg = QImage(image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                    else:
                        qimg = QImage(image.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)

                    pixmap = QPixmap.fromImage(qimg)
                    scaled = pixmap.scaled(
                        self.width(), self.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self._image_label.setPixmap(scaled)

            self.renderComplete.emit(info)

        def _end_interaction(self):
            """End interaction mode and render at full resolution."""
            self._is_interacting = False
            self._request_full_render()

        # =====================================================================
        # Mouse/Keyboard Events
        # =====================================================================

        def mousePressEvent(self, event: QMouseEvent):
            if not self.component.interactive:
                return

            self._last_mouse_pos = event.pos()
            self._is_interacting = True
            self._interaction_timer.stop()

            if event.button() == Qt.MouseButton.LeftButton:
                self._is_orbiting = True
            elif event.button() == Qt.MouseButton.RightButton:
                self._is_panning = True

        def mouseReleaseEvent(self, event: QMouseEvent):
            self._is_orbiting = False
            self._is_panning = False
            self._interaction_timer.start(self.FULL_RENDER_DELAY)

        def mouseMoveEvent(self, event: QMouseEvent):
            if not self.component.interactive:
                return

            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()

            if self._is_orbiting:
                self._azimuth += delta.x() * 0.5
                self._elevation = max(-89, min(89, self._elevation - delta.y() * 0.5))
                self._request_render(preview=True)

            elif self._is_panning:
                pan_speed = 0.005 * self._distance
                self._target[0] -= delta.x() * pan_speed
                self._target[1] += delta.y() * pan_speed
                self._request_render(preview=True)

        def wheelEvent(self, event: QWheelEvent):
            if not self.component.interactive:
                return

            delta = event.angleDelta().y()
            zoom_factor = 1.1 if delta < 0 else 0.9
            self._distance = max(0.5, min(20.0, self._distance * zoom_factor))

            self._is_interacting = True
            self._interaction_timer.start(self.FULL_RENDER_DELAY)
            self._request_render(preview=True)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self._scene_builder:
                self._render_timer.start(100)

        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_F:
                self._distance = 3.0
                self._target = [0.0, 0.8, 0.0]
                self._request_render()
            elif event.key() == Qt.Key.Key_A:
                self.frame_all()


# Type hint import guard
if False:  # TYPE_CHECKING
    from noodlestudio.core.radiance_component import RadianceComponent
