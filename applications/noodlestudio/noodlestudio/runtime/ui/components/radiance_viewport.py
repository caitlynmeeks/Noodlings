"""
RadianceViewport Component

Embeds a Gaussian splat renderer. The key component that makes NoodleStudio
applications 3D-capable.

This is just another UI component - a "3D game" is simply a canvas with
a fullscreen RadianceViewport.
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

    Embeds the GaussianRenderer to display radiance assets (Gaussian splats).
    Supports orbit camera controls and stage loading.

    Properties:
        stage: Stage name to load and display
        camera: Camera configuration (distance, elevation, azimuth, target, fov)
        background: Background color (hex string)
        show_skeleton: Whether to display bone skeleton overlay
        interactive: Whether camera controls are enabled

    Events:
        onLoad: Triggered when stage/radiance loads
        onClick: Triggered on click (with 3D ray info)
    """

    component_type = "RadianceViewport"

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.stage: Optional[str] = None
        self.camera = CameraConfig()
        self.background: str = "#000000"
        self.show_skeleton: bool = False
        self.interactive: bool = True

        # Runtime state (not serialized)
        self._radiance_component = None
        self._renderer = None

        # Default to fill parent
        self.geometry.width = 512
        self.geometry.height = 512

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add RadianceViewport-specific properties to serialization."""
        if self.stage:
            data["stage"] = self.stage
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
        viewport.stage = data.get("stage")
        if "camera" in data:
            viewport.camera = CameraConfig.from_dict(data["camera"])
        viewport.background = data.get("background", "#000000")
        viewport.show_skeleton = data.get("show_skeleton", False)
        viewport.interactive = data.get("interactive", True)

        return viewport


# ============================================================================
# Qt Widget Implementation (for QtWidgetRenderer)
# ============================================================================

# Check for required dependencies
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

        This is the actual Qt implementation used by QtWidgetRenderer
        when it encounters a RadianceViewport component.
        """

        # Render quality
        PREVIEW_SIZE = 128
        FULL_RENDER_DELAY = 300  # ms

        def __init__(self, component: RadianceViewport, parent=None):
            super().__init__(parent)
            self.component = component
            self.setObjectName(component.name or "radiance_viewport")

            # Renderer
            self._renderer = None
            self._radiance_component = None

            # Camera state (from component config)
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

            # UI
            self._setup_ui()
            self._setup_renderer()
            self._setup_timers()

            # Load stage if specified
            if component.stage:
                QTimer.singleShot(100, lambda: self._load_stage(component.stage))

        def _setup_ui(self):
            """Create the display label."""
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            self._image_label = QLabel()
            self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._image_label.setStyleSheet(f"background-color: {self.component.background};")
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
                logger.info(f"Viewport renderer: {self._renderer.device}")
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

        def _load_stage(self, stage_name: str):
            """Load a stage and its radiances."""
            # TODO: Implement stage loading
            # For now, this is a placeholder
            logger.info(f"Would load stage: {stage_name}")

        def load_radiance(self, path: str):
            """Load a radiance file directly."""
            try:
                from noodlestudio.core.radiance_component import RadianceComponent
                self._radiance_component = RadianceComponent(self.component.name)
                self._radiance_component.load_asset(path)
                self._request_render()
            except Exception as e:
                logger.error(f"Failed to load radiance: {e}")

        def set_radiance_component(self, component):
            """Set a RadianceComponent directly."""
            self._radiance_component = component
            self._request_render()

        # --- Rendering ---

        def _request_render(self, preview: bool = False):
            """Request a render update."""
            if not self._renderer or not self._radiance_component:
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
                width = self.width()
                height = self.height()

            # Submit to thread pool
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

                image, alpha, info = self._renderer.render_component(
                    self._radiance_component, camera
                )

                # Update UI on main thread
                QTimer.singleShot(0, lambda: self._on_render_complete(image, alpha, info))

            except Exception as e:
                logger.error(f"Render error: {e}")
                with self._render_lock:
                    self._is_rendering = False

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
                import numpy as np
                if isinstance(image, np.ndarray):
                    h, w = image.shape[:2]
                    if image.shape[2] == 3:
                        # RGB
                        qimg = QImage(image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                    else:
                        # RGBA
                        qimg = QImage(image.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)

                    pixmap = QPixmap.fromImage(qimg)

                    # Scale to widget size
                    scaled = pixmap.scaled(
                        self.width(), self.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self._image_label.setPixmap(scaled)

        def _end_interaction(self):
            """End interaction mode and render at full resolution."""
            self._is_interacting = False
            self._request_full_render()

        # --- Mouse Events ---

        def mousePressEvent(self, event: QMouseEvent):
            """Handle mouse press."""
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
            """Handle mouse release."""
            self._is_orbiting = False
            self._is_panning = False
            self._interaction_timer.start(self.FULL_RENDER_DELAY)

        def mouseMoveEvent(self, event: QMouseEvent):
            """Handle mouse move for orbit/pan."""
            if not self.component.interactive:
                return

            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()

            if self._is_orbiting:
                # Orbit camera
                self._azimuth += delta.x() * 0.5
                self._elevation = max(-89, min(89, self._elevation - delta.y() * 0.5))
                self._request_render(preview=True)

            elif self._is_panning:
                # Pan camera target
                pan_speed = 0.005 * self._distance
                self._target[0] -= delta.x() * pan_speed
                self._target[1] += delta.y() * pan_speed
                self._request_render(preview=True)

        def wheelEvent(self, event: QWheelEvent):
            """Handle scroll for zoom."""
            if not self.component.interactive:
                return

            delta = event.angleDelta().y()
            zoom_factor = 1.1 if delta < 0 else 0.9
            self._distance = max(0.5, min(20.0, self._distance * zoom_factor))

            self._is_interacting = True
            self._interaction_timer.start(self.FULL_RENDER_DELAY)
            self._request_render(preview=True)

        def resizeEvent(self, event):
            """Handle resize."""
            super().resizeEvent(event)
            if self._radiance_component:
                self._render_timer.start(100)  # Debounce resize renders

        def keyPressEvent(self, event):
            """Handle key press."""
            if event.key() == Qt.Key.Key_F:
                # Focus/fit
                self._distance = 3.0
                self._target = [0.0, 0.8, 0.0]
                self._request_render()
            elif event.key() == Qt.Key.Key_A:
                # Frame all
                self._distance = 5.0
                self._target = [0.0, 0.5, 0.0]
                self._elevation = 20
                self._azimuth = 180
                self._request_render()
