"""
Gaussian Viewer Panel - Native MPS-accelerated Gaussian splat viewport.

Unity-style camera controls:
- Left drag: Orbit around target
- Right drag: Pan
- Scroll: Zoom in/out
- F: Focus/fit all
- Middle drag: Pan (alternate)

Performance notes:
- Renders at low resolution during interaction
- Full resolution render when interaction stops
- Uses background thread to prevent UI freeze

Author: Caitlyn + Claude (NinaK)
Date: December 2025
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QSizePolicy, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent, QWheelEvent, QKeyEvent

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - native rendering disabled")


class GaussianViewerPanel(QWidget):
    """
    Full-viewport Gaussian splat viewer with Unity-style camera controls.

    Signals:
        radianceLoaded: Emitted when a radiance file is loaded (path, component)
        selectionChanged: Emitted when the selection changes
    """

    radianceLoaded = pyqtSignal(str, object)  # path, RadianceComponent
    selectionChanged = pyqtSignal(object)  # RadianceComponent or None
    _renderComplete = pyqtSignal(object, object, object)  # image, alpha, info

    # Render quality settings
    PREVIEW_SIZE = 128  # Fast preview during interaction
    FULL_RENDER_DELAY = 500  # ms to wait before full render

    def __init__(self, parent=None):
        super().__init__(parent)

        # Radiance data
        self._radiance_component = None
        self._current_path = None

        # Native renderer
        self._renderer = None

        # Camera state
        self._orbit_azimuth = 30.0
        self._orbit_elevation = 15.0
        self._orbit_distance = 3.0
        self._target = [0.0, 0.8, 0.0]  # Look-at target
        self._fov = 45.0

        # Mouse interaction
        self._last_mouse_pos = QPoint()
        self._is_orbiting = False
        self._is_panning = False
        self._is_interacting = False

        # Animation
        self._auto_rotate = False
        self._rotate_timer = None

        # Render state
        self._render_lock = threading.Lock()
        self._is_rendering = False
        self._pending_render = False
        self._last_render_params = None

        # Thread pool for background rendering
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._setup_ui()
        self._setup_renderer()
        self._setup_timers()

        # Connect render complete signal
        self._renderComplete.connect(self._on_render_complete)

    def _setup_renderer(self):
        """Initialize the native Gaussian renderer."""
        if not TORCH_AVAILABLE:
            return

        try:
            from noodlestudio.core.gaussian_renderer import GaussianRenderer, GSPLAT_AVAILABLE
            self._renderer = GaussianRenderer()
            logger.info(f"Gaussian renderer: {self._renderer.device}, GPU: {self._renderer.use_gpu}")

            # Show warning if falling back to software rendering
            if not self._renderer.use_gpu:
                self._show_software_render_warning(GSPLAT_AVAILABLE)

        except Exception as e:
            logger.warning(f"Failed to init renderer: {e}")
            self._renderer = None

    def _show_software_render_warning(self, gsplat_available: bool):
        """Show a one-time warning about software rendering performance."""
        from PyQt6.QtWidgets import QMessageBox
        from PyQt6.QtCore import QSettings

        # Check if user already dismissed this warning
        settings = QSettings("NoodleStudio", "NoodleStudio")
        if settings.value("suppress_software_render_warning", False, type=bool):
            return

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Software Rendering Mode")

        if not gsplat_available:
            msg.setText("GPU acceleration is not available.")
            msg.setInformativeText(
                "The gsplat-mps library is not installed. Gaussian rendering will use "
                "software mode which is significantly slower (~0.1 FPS vs 120+ FPS).\n\n"
                "To enable GPU acceleration:\n"
                "1. Clone: git clone https://github.com/prkrmx/gsplat-mps\n"
                "2. Install: pip install --no-build-isolation ./gsplat-mps\n\n"
                "Requires: macOS with Apple Silicon (M1/M2/M3)"
            )
        else:
            msg.setText("GPU acceleration is disabled.")
            msg.setInformativeText(
                "gsplat-mps is installed but GPU rendering is disabled. "
                "This may be because MPS (Metal Performance Shaders) is not available "
                "on this system.\n\n"
                "Software rendering is significantly slower (~0.1 FPS vs 120+ FPS)."
            )

        msg.setStandardButtons(QMessageBox.StandardButton.Ok)

        # Add "Don't show again" checkbox
        checkbox = msg.checkBox()
        if checkbox is None:
            from PyQt6.QtWidgets import QCheckBox
            checkbox = QCheckBox("Don't show this again")
            msg.setCheckBox(checkbox)

        msg.exec()

        # Save preference if checked
        if checkbox and checkbox.isChecked():
            settings.setValue("suppress_software_render_warning", True)

    def _setup_timers(self):
        """Set up animation and render timers."""
        # Full render timer - fires after interaction stops
        self._full_render_timer = QTimer(self)
        self._full_render_timer.setSingleShot(True)
        self._full_render_timer.timeout.connect(self._request_full_render)

        # Auto-rotate timer
        self._rotate_timer = QTimer(self)
        self._rotate_timer.timeout.connect(self._rotate_step)

    def _setup_ui(self):
        """Build the UI - minimal, viewport-focused."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 4)
        toolbar.setSpacing(4)

        self.load_btn = QPushButton("Load")
        self.load_btn.setFixedWidth(60)
        self.load_btn.clicked.connect(self._load_file)
        toolbar.addWidget(self.load_btn)

        self.focus_btn = QPushButton("F")
        self.focus_btn.setFixedWidth(30)
        self.focus_btn.setToolTip("Focus/Fit All (F)")
        self.focus_btn.clicked.connect(self._focus_all)
        toolbar.addWidget(self.focus_btn)

        self.rotate_btn = QPushButton("Rotate")
        self.rotate_btn.setCheckable(True)
        self.rotate_btn.setFixedWidth(60)
        self.rotate_btn.clicked.connect(self._toggle_auto_rotate)
        toolbar.addWidget(self.rotate_btn)

        toolbar.addStretch()

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self.status_label)

        # Render indicator
        self.render_indicator = QLabel("")
        self.render_indicator.setStyleSheet("color: #cc8800; font-size: 11px;")
        self.render_indicator.setFixedWidth(80)
        toolbar.addWidget(self.render_indicator)

        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar)
        toolbar_widget.setStyleSheet("background-color: #2a2a2a;")
        toolbar_widget.setFixedHeight(32)
        layout.addWidget(toolbar_widget)

        # Viewport (the main event)
        self.viewport = QLabel()
        self.viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewport.setMinimumSize(256, 256)
        self.viewport.setStyleSheet("background-color: #1a1a1c;")
        self.viewport.setText("Drag .radiance file here or click Load")
        self.viewport.setMouseTracking(True)

        # Enable drops
        self.setAcceptDrops(True)

        layout.addWidget(self.viewport)

        # Set focus policy for keyboard
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Style
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1c;
                color: #d2d2d2;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:checked {
                background-color: #2d5c8f;
            }
        """)

    # =========================================================================
    # File Loading
    # =========================================================================

    def _load_file(self):
        """Open file dialog to load .radiance file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Radiance Asset",
            str(Path.home()),
            "Radiance Files (*.radiance);;All Files (*)"
        )
        if path:
            self.load_radiance(path)

    def load_radiance(self, path: str) -> bool:
        """Load a radiance file."""
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return False

        try:
            from noodlestudio.core.radiance_component import RadianceComponent

            name = Path(path).stem
            component = RadianceComponent(entity_id=name)

            if component.load_asset(path):
                # Set reasonable default Gaussian scale
                component.material.scale_mult = 3.0

                self._radiance_component = component
                self._current_path = path
                self._focus_all()

                # Update status
                self.status_label.setText(
                    f"{name} | {component.gaussian_count:,} Gaussians"
                )

                # Do initial preview render
                self._request_preview_render()

                # Emit signals
                self.radianceLoaded.emit(path, component)
                self.selectionChanged.emit(component)

                logger.info(f"Loaded: {path}")
                return True
            else:
                QMessageBox.warning(self, "Error", f"Failed to load:\n{path}")
                return False

        except Exception as e:
            logger.error(f"Load error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Error:\n{str(e)}")
            return False

    # =========================================================================
    # Camera Controls
    # =========================================================================

    def _focus_all(self):
        """Fit the model in view (like Unity's F key)."""
        if not self._radiance_component:
            return

        # Get bounds from asset
        asset = self._radiance_component._asset
        if asset is None:
            return

        positions = asset.positions
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        center = (min_pos + max_pos) / 2
        extent = np.max(max_pos - min_pos)

        # Set target to center
        self._target = center.tolist()

        # Set distance to fit
        self._orbit_distance = extent * 1.8

        # Reset angles for nice view
        self._orbit_azimuth = 30.0
        self._orbit_elevation = 15.0

        self._request_preview_render()
        self._schedule_full_render()

    def _orbit(self, dx: float, dy: float):
        """Orbit camera around target."""
        self._orbit_azimuth += dx * 0.5
        self._orbit_elevation = max(-89, min(89, self._orbit_elevation - dy * 0.5))

    def _pan(self, dx: float, dy: float):
        """Pan the view (move target)."""
        # Calculate pan in camera space
        scale = self._orbit_distance * 0.002

        # Simple pan in world XY for now
        az_rad = np.radians(self._orbit_azimuth)
        self._target[0] -= dx * scale * np.cos(az_rad)
        self._target[2] -= dx * scale * np.sin(az_rad)
        self._target[1] += dy * scale

    def _zoom(self, delta: float):
        """Zoom in/out."""
        factor = 1.0 - delta * 0.001
        self._orbit_distance = max(0.1, min(100.0, self._orbit_distance * factor))

    # =========================================================================
    # Mouse Events
    # =========================================================================

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        self._last_mouse_pos = event.pos()
        self._is_interacting = True

        if event.button() == Qt.MouseButton.LeftButton:
            self._is_orbiting = True
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._is_panning = True

        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        self._is_orbiting = False
        self._is_panning = False
        self._is_interacting = False

        # Schedule full render after interaction ends
        self._schedule_full_render()

        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if not self._radiance_component:
            return

        delta = event.pos() - self._last_mouse_pos
        self._last_mouse_pos = event.pos()

        if self._is_orbiting:
            self._orbit(delta.x(), delta.y())
            self._request_preview_render()
        elif self._is_panning:
            self._pan(delta.x(), delta.y())
            self._request_preview_render()

        event.accept()

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zoom."""
        if not self._radiance_component:
            return

        delta = event.angleDelta().y()
        self._zoom(delta)
        self._request_preview_render()
        self._schedule_full_render()

        event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_F:
            self._focus_all()
        elif event.key() == Qt.Key.Key_R:
            self._toggle_auto_rotate()
        else:
            super().keyPressEvent(event)

    # =========================================================================
    # Drag and Drop
    # =========================================================================

    def dragEnterEvent(self, event):
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().endswith('.radiance'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Handle drop."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.endswith('.radiance'):
                self.load_radiance(path)
                return

    # =========================================================================
    # Rendering
    # =========================================================================

    def _request_preview_render(self):
        """Request a fast preview render (low resolution)."""
        self._do_render_async(self.PREVIEW_SIZE, self.PREVIEW_SIZE)

    def _request_full_render(self):
        """Request a full resolution render."""
        w = max(64, self.viewport.width())
        h = max(64, self.viewport.height())
        # Cap at reasonable size to prevent multi-minute renders
        max_size = 512
        if w > max_size:
            h = int(h * max_size / w)
            w = max_size
        if h > max_size:
            w = int(w * max_size / h)
            h = max_size
        self._do_render_async(w, h)

    def _schedule_full_render(self):
        """Schedule a full render after a delay."""
        self._full_render_timer.stop()
        self._full_render_timer.start(self.FULL_RENDER_DELAY)

    def _do_render_async(self, width: int, height: int):
        """Submit render to background thread."""
        if not self._renderer or not self._radiance_component:
            return

        # Check if already rendering
        with self._render_lock:
            if self._is_rendering:
                self._pending_render = (width, height)
                return
            self._is_rendering = True

        # Update UI
        self.render_indicator.setText("Rendering...")

        # Capture current camera state
        params = {
            'width': width,
            'height': height,
            'distance': self._orbit_distance,
            'elevation': self._orbit_elevation,
            'azimuth': self._orbit_azimuth,
            'target': tuple(self._target),
            'fov': self._fov
        }

        # Submit to thread pool
        self._executor.submit(self._render_worker, params)

    def _render_worker(self, params: Dict[str, Any]):
        """Background render worker."""
        try:
            from noodlestudio.core.gaussian_renderer import create_orbit_camera

            camera = create_orbit_camera(
                distance=params['distance'],
                elevation=params['elevation'],
                azimuth=params['azimuth'],
                target=params['target'],
                fov=params['fov'],
                width=params['width'],
                height=params['height']
            )

            image, alpha, info = self._renderer.render_component(
                self._radiance_component,
                camera,
                background=(0.1, 0.1, 0.11)
            )

            # Emit signal to update UI on main thread
            self._renderComplete.emit(image, alpha, info)

        except Exception as e:
            logger.error(f"Render error: {e}")
            import traceback
            traceback.print_exc()
            self._renderComplete.emit(None, None, None)

    def _on_render_complete(self, image, alpha, info):
        """Handle render completion on main thread."""
        # Clear rendering flag
        with self._render_lock:
            self._is_rendering = False
            pending = self._pending_render
            self._pending_render = None

        # Update UI
        self.render_indicator.setText("")

        if image is not None:
            # Convert to QImage
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            h, w, c = img_np.shape
            bytes_per_line = c * w
            qimage = QImage(img_np.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Scale to viewport size
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(
                self.viewport.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.viewport.setPixmap(scaled)

        # Process pending render
        if pending:
            self._do_render_async(pending[0], pending[1])

    def resizeEvent(self, event):
        """Handle resize - schedule full render."""
        super().resizeEvent(event)
        if self._radiance_component:
            self._schedule_full_render()

    # =========================================================================
    # Auto-Rotate
    # =========================================================================

    def _toggle_auto_rotate(self):
        """Toggle auto-rotation."""
        self._auto_rotate = not self._auto_rotate
        self.rotate_btn.setChecked(self._auto_rotate)

        if self._auto_rotate:
            self._rotate_timer.start(100)  # Slower for performance
        else:
            self._rotate_timer.stop()
            self._schedule_full_render()

    def _rotate_step(self):
        """One step of auto-rotation."""
        self._orbit_azimuth += 3.0  # Larger step, less frequent
        if self._orbit_azimuth >= 360:
            self._orbit_azimuth -= 360
        self._request_preview_render()

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def component(self) -> Optional['RadianceComponent']:
        """Get the current RadianceComponent."""
        return self._radiance_component

    @property
    def current_path(self) -> Optional[str]:
        """Get the current file path."""
        return self._current_path

    def set_gaussian_scale(self, scale: float):
        """Set the Gaussian scale multiplier."""
        if self._radiance_component:
            self._radiance_component.material.scale_mult = scale
            self._request_preview_render()
            self._schedule_full_render()

    def get_camera_state(self) -> Dict[str, Any]:
        """Get current camera state for serialization."""
        return {
            'azimuth': self._orbit_azimuth,
            'elevation': self._orbit_elevation,
            'distance': self._orbit_distance,
            'target': self._target.copy(),
            'fov': self._fov
        }

    def set_camera_state(self, state: Dict[str, Any]):
        """Restore camera state."""
        self._orbit_azimuth = state.get('azimuth', 30.0)
        self._orbit_elevation = state.get('elevation', 15.0)
        self._orbit_distance = state.get('distance', 3.0)
        self._target = state.get('target', [0.0, 0.8, 0.0])
        self._fov = state.get('fov', 45.0)
        self._request_preview_render()
        self._schedule_full_render()

    def refresh(self):
        """Force a re-render (call after Inspector changes)."""
        self._request_preview_render()
        self._schedule_full_render()
