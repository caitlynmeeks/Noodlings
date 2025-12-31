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
import time
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
        meshImported: Emitted when a mesh is imported (source_path, mesh_type, output_radiance_path)
        selectionChanged: Emitted when the selection changes
    """

    radianceLoaded = pyqtSignal(str, object)  # path, RadianceComponent
    meshImported = pyqtSignal(str, str, str)  # source_path, mesh_type, output_radiance_path
    selectionChanged = pyqtSignal(object)  # RadianceComponent or None
    boneSelectionChanged = pyqtSignal(str)  # bone_name or "" for deselect
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

        # Bone selection (synced with inspector)
        self._selected_bone_name = ""
        self._selected_bone_position = None

        # Mouse interaction
        self._last_mouse_pos = QPoint()
        self._mouse_press_pos = QPoint()  # For detecting clicks vs drags
        self._is_orbiting = False
        self._is_panning = False
        self._is_interacting = False
        self._space_held = False  # Spacebar for grab/pan mode

        # Animation
        self._auto_rotate = False
        self._rotate_timer = None

        # Skeleton visualization
        self._show_skeleton = False
        self._bone_screen_positions = []  # [(bone_name, screen_pos, world_pos), ...]
        self._bone_segments = []  # [(bone_name, start_pos, end_pos, world_pos), ...] for line hit testing
        self._bone_hit_radius = 20  # Pixels for joint click detection (was 8, now much larger)
        self._bone_line_hit_radius = 12  # Pixels for line segment click detection

        # Render state
        self._render_lock = threading.Lock()
        self._is_rendering = False
        self._pending_render = False
        self._last_render_params = None

        # FPS tracking
        self._render_start_time = 0.0
        self._last_render_time_ms = 0.0
        self._last_fps = 0.0
        self._last_visible_count = 0
        self._last_backend = "none"

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

        # Continuous render timer for live FPS updates
        self._continuous_render_timer = QTimer(self)
        self._continuous_render_timer.timeout.connect(self._continuous_render_tick)
        self._continuous_render_timer.start(100)  # 10 FPS baseline

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

        # Import menu for meshes and VRMs
        from PyQt6.QtWidgets import QMenu
        self.import_btn = QPushButton("Import")
        self.import_btn.setFixedWidth(60)
        import_menu = QMenu(self)
        import_menu.addAction("Mesh (OBJ)...", self._import_mesh)
        import_menu.addAction("VRM Avatar...", self._import_vrm)
        self.import_btn.setMenu(import_menu)
        toolbar.addWidget(self.import_btn)

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

        self.bones_btn = QPushButton("Bones")
        self.bones_btn.setCheckable(True)
        self.bones_btn.setFixedWidth(60)
        self.bones_btn.setToolTip("Show skeleton (B)")
        self.bones_btn.clicked.connect(self._toggle_skeleton)
        toolbar.addWidget(self.bones_btn)

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

        # Viewport container (for overlay positioning)
        from PyQt6.QtWidgets import QFrame
        viewport_container = QFrame()
        viewport_container.setStyleSheet("background-color: #1a1a1c;")
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)

        # Viewport (the main event)
        self.viewport = QLabel()
        self.viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewport.setMinimumSize(256, 256)
        self.viewport.setStyleSheet("background-color: #1a1a1c;")
        self.viewport.setText("Drag .radiance file here or click Load")
        self.viewport.setMouseTracking(True)
        viewport_layout.addWidget(self.viewport)

        # FPS/Stats overlay (bottom-left corner)
        self.stats_label = QLabel(viewport_container)
        self.stats_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: #00ff88;
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 3px;
            }
        """)
        self.stats_label.setText("")
        self.stats_label.adjustSize()
        self.stats_label.move(8, 8)  # Will be repositioned in resizeEvent
        self.stats_label.hide()  # Hidden until we have stats

        # Enable drops
        self.setAcceptDrops(True)

        layout.addWidget(viewport_container)
        self._viewport_container = viewport_container

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

    def _import_mesh(self):
        """Import and auto-rig a mesh file (OBJ)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Mesh",
            str(Path.home()),
            "Mesh Files (*.obj);;All Files (*)"
        )
        if not path:
            return

        # Show progress
        self.status_label.setText("Auto-rigging mesh...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from noodlestudio.tools.auto_rigger import AutoRigger

            # Output path next to input
            input_path = Path(path)
            output_path = input_path.with_suffix('.radiance')

            # Run auto-rigger
            rigger = AutoRigger()
            rigger.load_mesh(str(input_path))
            markers = rigger.auto_detect_markers()

            result = rigger.rig(
                markers=markers,
                output_path=str(output_path),
                entity_id=input_path.stem,
                display_name=input_path.stem
            )

            if result.get('success') and output_path.exists():
                self.status_label.setText(f"Rigged: {input_path.stem}")
                self.load_radiance(str(output_path))
                # Notify that mesh was imported
                self.meshImported.emit(str(input_path), 'obj', str(output_path))
            else:
                msg = result.get('message', 'Unknown error')
                QMessageBox.warning(self, "Error", f"Auto-rigging failed:\n{msg}")
                self.status_label.setText("")

        except Exception as e:
            logger.error(f"Import mesh error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Import failed:\n{str(e)}")
            self.status_label.setText("")

    def _import_vrm(self):
        """Import and convert a VRM avatar to radiance."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import VRM Avatar",
            str(Path.home()),
            "VRM Files (*.vrm);;All Files (*)"
        )
        if not path:
            return

        # Show progress
        self.status_label.setText("Converting VRM...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from noodlestudio.tools.vrm_to_radiance import vrm_to_radiance
            from noodlestudio.core.semantic_world.radiance_format import save_radiance

            # Output path next to input
            input_path = Path(path)
            output_path = input_path.with_suffix('.radiance')

            # Run conversion with densification
            asset = vrm_to_radiance(
                str(input_path),
                entity_id=input_path.stem,
                display_name=input_path.stem,
                densify=True
            )

            # Save the asset
            save_radiance(asset, str(output_path))

            if output_path.exists():
                self.status_label.setText(f"Converted: {input_path.stem}")
                self.load_radiance(str(output_path))
                # Notify that VRM was imported
                self.meshImported.emit(str(input_path), 'vrm', str(output_path))
            else:
                QMessageBox.warning(self, "Error", "VRM conversion failed")
                self.status_label.setText("")

        except Exception as e:
            logger.error(f"Import VRM error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Import failed:\n{str(e)}")
            self.status_label.setText("")

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
        """Orbit camera around target (Unity-style: drag right = rotate right)."""
        self._orbit_azimuth -= dx * 0.5  # Flipped: drag right rotates model right
        self._orbit_elevation = max(-89, min(89, self._orbit_elevation + dy * 0.5))  # Flipped: drag up tilts up

    def _pan(self, dx: float, dy: float):
        """Pan the view (move target) in screen space."""
        # Calculate camera vectors for proper screen-space panning
        az_rad = np.radians(self._orbit_azimuth)
        el_rad = np.radians(self._orbit_elevation)

        # Camera position
        eye = np.array([
            self._target[0] + self._orbit_distance * np.cos(el_rad) * np.sin(az_rad),
            self._target[1] + self._orbit_distance * np.sin(el_rad),
            self._target[2] + self._orbit_distance * np.cos(el_rad) * np.cos(az_rad)
        ])

        # Forward vector (towards target)
        forward = np.array(self._target) - eye
        forward = forward / np.linalg.norm(forward)

        # Right vector (screen X direction)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
        else:
            right = np.array([1.0, 0.0, 0.0])

        # Up vector (screen Y direction)
        up = np.cross(right, forward)

        # Scale based on distance
        scale = self._orbit_distance * 0.002

        # Move target in grab mode: drag direction = world moves with cursor
        # Negate dx for correct horizontal behavior (drag left = world moves left)
        self._target[0] += (-right[0] * dx + up[0] * dy) * scale
        self._target[1] += (-right[1] * dx + up[1] * dy) * scale
        self._target[2] += (-right[2] * dx + up[2] * dy) * scale

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
        self._mouse_press_pos = event.pos()  # Remember for click detection
        self._had_focus_on_press = self.hasFocus()  # Track if we had focus
        self._is_interacting = True

        if event.button() == Qt.MouseButton.LeftButton:
            # Spacebar + left click = pan (grab mode)
            if self._space_held:
                self._is_panning = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self._is_orbiting = True
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._is_panning = True

        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        # Check if this was a click (not a drag) - check for bone click or deselect
        if event.button() == Qt.MouseButton.LeftButton:
            delta = event.pos() - self._mouse_press_pos
            if abs(delta.x()) < 5 and abs(delta.y()) < 5:
                # This was a click, not a drag
                # Only process bone selection if we already had focus
                # (clicking to give focus shouldn't deselect)
                if getattr(self, '_had_focus_on_press', True):
                    # Try to hit test a bone if skeleton is visible
                    if self._show_skeleton:
                        clicked_bone = self._hit_test_bone(event.pos())
                        if clicked_bone:
                            bone_name, world_pos = clicked_bone
                            self.select_bone(bone_name, world_pos)
                        else:
                            self.deselect_bone()
                    else:
                        self.deselect_bone()

        self._is_orbiting = False
        self._is_panning = False
        self._is_interacting = False

        # Reset cursor based on spacebar state
        if self._space_held:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

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
            # F = Focus on selected bone, or whole model if nothing selected
            if self._selected_bone_name and self._selected_bone_position:
                self.focus_on_position(self._selected_bone_position, distance=0.5)
            else:
                self._focus_all()
        elif event.key() == Qt.Key.Key_A:
            # A = Frame All (always focuses whole model)
            self._focus_all()
        elif event.key() == Qt.Key.Key_R:
            self._toggle_auto_rotate()
        elif event.key() == Qt.Key.Key_B:
            self._toggle_skeleton()
        elif event.key() == Qt.Key.Key_Space:
            # Spacebar = grab/pan mode
            if not event.isAutoRepeat():
                self._space_held = True
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif event.key() == Qt.Key.Key_Escape:
            # Escape = Deselect bone
            self.deselect_bone()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release."""
        if event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._space_held = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

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

        # Track render start time
        self._render_start_time = time.perf_counter()

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
        # Calculate render time
        render_end = time.perf_counter()
        self._last_render_time_ms = (render_end - self._render_start_time) * 1000

        # Clear rendering flag
        with self._render_lock:
            self._is_rendering = False
            pending = self._pending_render
            self._pending_render = None

        # Update UI
        self.render_indicator.setText("")

        if image is not None:
            # Extract stats from render info
            if info:
                self._last_visible_count = info.get('visible', 0)
                self._last_backend = info.get('backend', 'software')

            # Calculate FPS (avoid div by zero)
            if self._last_render_time_ms > 0:
                self._last_fps = 1000.0 / self._last_render_time_ms
            else:
                self._last_fps = 0.0

            # Update stats overlay
            self._update_stats_display()

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

            # Draw skeleton overlay if enabled
            if self._show_skeleton:
                scaled = self._draw_skeleton_overlay(scaled)

            self.viewport.setPixmap(scaled)

        # Process pending render
        if pending:
            self._do_render_async(pending[0], pending[1])

    def _update_stats_display(self):
        """Update the FPS/stats overlay."""
        if not self._radiance_component:
            self.stats_label.hide()
            return

        # Format stats text
        backend_str = "GPU" if "gsplat" in self._last_backend else "CPU"
        stats_text = (
            f"{self._last_fps:5.1f} FPS | "
            f"{self._last_render_time_ms:5.1f}ms | "
            f"{self._last_visible_count:,} visible | "
            f"{backend_str}"
        )

        self.stats_label.setText(stats_text)
        self.stats_label.adjustSize()

        # Position at bottom-left of viewport container
        self._reposition_stats_label()
        self.stats_label.show()

    def _reposition_stats_label(self):
        """Position stats label at bottom-left of viewport."""
        if hasattr(self, '_viewport_container') and hasattr(self, 'stats_label'):
            container_height = self._viewport_container.height()
            label_height = self.stats_label.height()
            margin = 8
            self.stats_label.move(margin, container_height - label_height - margin)

    def resizeEvent(self, event):
        """Handle resize - schedule full render and reposition stats."""
        super().resizeEvent(event)
        # Reposition stats overlay
        self._reposition_stats_label()
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

    def _continuous_render_tick(self):
        """Continuous render for live FPS updates."""
        if self._radiance_component and not self._is_interacting and not self._auto_rotate:
            # Only render if we have a component and aren't already animating
            self._request_full_render()

    # =========================================================================
    # Skeleton Visualization
    # =========================================================================

    def _toggle_skeleton(self):
        """Toggle skeleton visualization."""
        self._show_skeleton = not self._show_skeleton
        self.bones_btn.setChecked(self._show_skeleton)
        # Re-render to show/hide skeleton
        if self._radiance_component:
            self._schedule_full_render()

    def _get_bone_world_positions(self):
        """Compute world positions for all bones."""
        if not self._radiance_component or not self._radiance_component._asset:
            return None, None

        asset = self._radiance_component._asset
        if not asset.skeleton or not asset.skeleton.bones:
            return None, None

        bones = asset.skeleton.bones
        world_positions = []
        parent_indices = []

        # Compute world positions by accumulating local transforms
        for bone in bones:
            pos = np.array(bone.position)
            parent_idx = bone.parent_index

            # Walk up hierarchy to get world position
            current_idx = parent_idx
            while current_idx >= 0 and current_idx < len(bones):
                parent_bone = bones[current_idx]
                pos = pos + np.array(parent_bone.position)
                current_idx = parent_bone.parent_index

            world_positions.append(pos)
            parent_indices.append(parent_idx)

        return np.array(world_positions), parent_indices

    def _project_to_screen(self, world_pos, width, height):
        """Project 3D world position to 2D screen coordinates.

        Uses the same camera convention as the Gaussian renderer.
        """
        # Build camera exactly like create_orbit_camera
        az_rad = np.radians(self._orbit_azimuth)
        el_rad = np.radians(self._orbit_elevation)

        # Camera position (eye) in world space
        eye_x = self._target[0] + self._orbit_distance * np.cos(el_rad) * np.sin(az_rad)
        eye_y = self._target[1] + self._orbit_distance * np.sin(el_rad)
        eye_z = self._target[2] + self._orbit_distance * np.cos(el_rad) * np.cos(az_rad)
        eye = np.array([eye_x, eye_y, eye_z])

        target_arr = np.array(self._target)

        # Forward vector (points towards target)
        forward = target_arr - eye
        forward = forward / np.linalg.norm(forward)

        # Right and up vectors - must match renderer convention exactly
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)  # Flipped order to match screen coords
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Transform world position to camera space
        rel_pos = world_pos - eye
        cam_x = np.dot(rel_pos, right)
        cam_y = np.dot(rel_pos, up)
        cam_z = np.dot(rel_pos, forward)  # Positive Z = in front of camera

        if cam_z <= 0.01:  # Behind camera
            return None

        # Perspective projection (same as renderer)
        # fy = height / (2 * tan(fov/2))
        fov_rad = np.radians(self._fov)
        fy = height / (2 * np.tan(fov_rad / 2))
        fx = fy  # Square pixels

        # screen_x = fx * (cam_x / cam_z) + cx
        # screen_y = height - (fy * (cam_y / cam_z) + cy) for screen coords (Y flipped)
        screen_x = fx * (cam_x / cam_z) + width / 2
        screen_y = height / 2 - fy * (cam_y / cam_z)  # Flip Y for screen coordinates

        return (int(screen_x), int(screen_y))

    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
        # Vector from line start to point
        dx = x2 - x1
        dy = y2 - y1

        # Length squared of line segment
        len_sq = dx * dx + dy * dy
        if len_sq < 0.0001:
            # Line segment is a point
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        # Projection parameter (0 = at start, 1 = at end)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))

        # Closest point on line segment
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        # Distance from point to closest point on segment
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

    def _hit_test_bone(self, click_pos: QPoint):
        """Check if click position hits a bone joint or line segment.

        Returns (bone_name, world_position) if hit, None otherwise.
        """
        if not self._bone_screen_positions:
            return None

        # Map click position from GaussianViewerPanel to viewport QLabel
        # (viewport is nested: GaussianViewerPanel -> viewport_container -> viewport)
        viewport_pos = self.viewport.mapFrom(self, click_pos)
        if not self.viewport.rect().contains(viewport_pos):
            return None

        # Scale click position to match pixmap coordinates
        pixmap = self.viewport.pixmap()
        if pixmap is None:
            return None

        # Calculate offset from centering (QLabel centers the pixmap)
        viewport_w = self.viewport.width()
        viewport_h = self.viewport.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()

        offset_x = (viewport_w - pixmap_w) // 2
        offset_y = (viewport_h - pixmap_h) // 2

        # Convert click to pixmap coordinates
        px = viewport_pos.x() - offset_x
        py = viewport_pos.y() - offset_y

        closest_bone = None
        closest_dist = float('inf')

        # First check joints (priority - they're easier to target intentionally)
        for bone_name, screen_pos, world_pos in self._bone_screen_positions:
            if screen_pos is None:
                continue

            dx = px - screen_pos[0]
            dy = py - screen_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < self._bone_hit_radius and dist < closest_dist:
                closest_dist = dist
                closest_bone = (bone_name, world_pos)

        # If we found a joint hit, return it
        if closest_bone is not None:
            return closest_bone

        # Otherwise check line segments (bones between joints)
        for bone_name, start_pos, end_pos, world_pos in self._bone_segments:
            if start_pos is None or end_pos is None:
                continue

            dist = self._point_to_line_distance(
                px, py,
                start_pos[0], start_pos[1],
                end_pos[0], end_pos[1]
            )

            if dist < self._bone_line_hit_radius and dist < closest_dist:
                closest_dist = dist
                closest_bone = (bone_name, world_pos)

        return closest_bone

    def _draw_skeleton_overlay(self, pixmap):
        """Draw skeleton bones on top of the rendered image with capsule visualization."""
        from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont
        from PyQt6.QtCore import QPointF

        world_positions, parent_indices = self._get_bone_world_positions()
        if world_positions is None:
            self._bone_screen_positions = []
            self._bone_segments = []
            return pixmap

        # Get bone names
        bone_names = []
        if self._radiance_component and self._radiance_component._asset:
            skeleton = self._radiance_component._asset.skeleton
            if skeleton and skeleton.bones:
                bone_names = [b.name for b in skeleton.bones]

        # Pad names if needed
        while len(bone_names) < len(world_positions):
            bone_names.append(f"bone_{len(bone_names)}")

        # Get image dimensions
        width = pixmap.width()
        height = pixmap.height()

        # Project all bone positions to screen and store for hit testing
        screen_positions = []
        self._bone_screen_positions = []

        for i, pos in enumerate(world_positions):
            screen_pos = self._project_to_screen(pos, width, height)
            screen_positions.append(screen_pos)
            self._bone_screen_positions.append((bone_names[i], screen_pos, tuple(pos)))

        # Build bone segment list for line hit testing
        self._bone_segments = []
        for i, (screen_pos, parent_idx) in enumerate(zip(screen_positions, parent_indices)):
            if screen_pos is None:
                continue
            if parent_idx >= 0 and parent_idx < len(screen_positions):
                parent_pos = screen_positions[parent_idx]
                if parent_pos is not None:
                    # Store segment for hit testing (bone_name, child_pos, parent_pos, world_pos)
                    self._bone_segments.append((
                        bone_names[i],
                        screen_pos,
                        parent_pos,
                        tuple(world_positions[i])
                    ))

        # Draw on pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Colors
        bone_color = QColor(0, 180, 180, 180)  # Teal for bones
        selected_color = QColor(255, 100, 50, 255)  # Orange for selected
        joint_color = QColor(255, 200, 0, 220)  # Yellow for joints
        selected_joint_color = QColor(255, 80, 30, 255)  # Red-orange for selected

        # Draw bones (capsule-style: thicker lines from child to parent)
        for i, (screen_pos, parent_idx) in enumerate(zip(screen_positions, parent_indices)):
            if screen_pos is None:
                continue
            if parent_idx >= 0 and parent_idx < len(screen_positions):
                parent_pos = screen_positions[parent_idx]
                if parent_pos is not None:
                    # Check if this bone or parent is selected
                    is_selected = (bone_names[i] == self._selected_bone_name or
                                   bone_names[parent_idx] == self._selected_bone_name)

                    # Draw outer glow/capsule (thicker for easier clicking)
                    if is_selected:
                        outer_pen = QPen(selected_color)
                        outer_pen.setWidth(10)
                        outer_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    else:
                        outer_pen = QPen(QColor(0, 100, 100, 120))
                        outer_pen.setWidth(8)
                        outer_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

                    painter.setPen(outer_pen)
                    painter.drawLine(screen_pos[0], screen_pos[1],
                                     parent_pos[0], parent_pos[1])

                    # Draw inner line
                    if is_selected:
                        inner_pen = QPen(QColor(255, 200, 150, 255))
                    else:
                        inner_pen = QPen(bone_color)
                    inner_pen.setWidth(3)
                    inner_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(inner_pen)
                    painter.drawLine(screen_pos[0], screen_pos[1],
                                     parent_pos[0], parent_pos[1])

        # Draw joints (circles at each bone position - larger for easier clicking)
        for i, screen_pos in enumerate(screen_positions):
            if screen_pos is None:
                continue

            is_selected = bone_names[i] == self._selected_bone_name

            if is_selected:
                # Selected joint - larger and highlighted
                painter.setPen(QPen(selected_color, 2))
                painter.setBrush(QBrush(selected_joint_color))
                radius = 8
            else:
                # Normal joint
                painter.setPen(QPen(QColor(200, 150, 0, 200), 1))
                painter.setBrush(QBrush(joint_color))
                radius = 6

            painter.drawEllipse(screen_pos[0] - radius, screen_pos[1] - radius,
                               radius * 2, radius * 2)

        # Draw label for selected bone
        if self._selected_bone_name:
            for i, (bone_name, screen_pos, world_pos) in enumerate(self._bone_screen_positions):
                if bone_name == self._selected_bone_name and screen_pos is not None:
                    # Draw bone name label
                    font = QFont("Monaco", 9)
                    painter.setFont(font)

                    # Background for label
                    label_text = bone_name
                    fm = painter.fontMetrics()
                    text_rect = fm.boundingRect(label_text)

                    label_x = screen_pos[0] + 10
                    label_y = screen_pos[1] - 10

                    bg_rect = text_rect.adjusted(-4, -2, 4, 2)
                    bg_rect.moveTo(label_x - 4, label_y - fm.ascent() - 2)

                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
                    painter.drawRoundedRect(bg_rect, 3, 3)

                    # Draw text
                    painter.setPen(QPen(selected_color))
                    painter.drawText(label_x, label_y, label_text)
                    break

        painter.end()
        return pixmap

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

    def focus_on_position(self, position: tuple, distance: float = 0.5):
        """Focus camera on a specific world position.

        Args:
            position: (x, y, z) world coordinates
            distance: Camera distance from target (default 0.5 for close-up)
        """
        self._target = list(position)
        self._orbit_distance = distance
        logger.info(f"Camera focused on {position}, distance={distance}")
        self._request_preview_render()
        self._schedule_full_render()

    def focus_on_bone(self, bone_name: str, position: tuple):
        """Focus camera on a bone position and select it (called from Inspector)."""
        logger.info(f"Focusing on bone '{bone_name}'")
        self._selected_bone_name = bone_name
        self._selected_bone_position = position
        self.focus_on_position(position, distance=0.5)

    def select_bone(self, bone_name: str, position: tuple = None):
        """Select a bone (updates internal state and notifies inspector)."""
        self._selected_bone_name = bone_name
        self._selected_bone_position = position
        self.boneSelectionChanged.emit(bone_name)
        logger.info(f"Bone selected: '{bone_name}'")
        # Re-render to show selection highlight
        if self._show_skeleton:
            self._schedule_full_render()

    def deselect_bone(self):
        """Deselect current bone."""
        if self._selected_bone_name:
            self._selected_bone_name = ""
            self._selected_bone_position = None
            self.boneSelectionChanged.emit("")
            logger.info("Bone deselected")
            # Re-render to hide selection highlight
            if self._show_skeleton:
                self._schedule_full_render()

    def set_bone_selection(self, bone_name: str):
        """Set bone selection from inspector dropdown (computes position internally).

        This is called when the inspector's bone dropdown changes.
        Does NOT emit boneSelectionChanged (to prevent infinite loops).
        """
        if not bone_name:
            self._selected_bone_name = ""
            self._selected_bone_position = None
        else:
            self._selected_bone_name = bone_name

            # Compute world position for this bone
            world_pos = self._get_bone_world_position(bone_name)
            self._selected_bone_position = world_pos

        # Re-render to show/hide selection highlight
        if self._show_skeleton:
            self._schedule_full_render()

    def _get_bone_world_position(self, bone_name: str):
        """Compute world position for a bone by name."""
        if not self._radiance_component or not self._radiance_component._asset:
            return None

        asset = self._radiance_component._asset
        if not asset.skeleton or not asset.skeleton.bones:
            return None

        bones = asset.skeleton.bones
        bone_names = [b.name for b in bones]

        try:
            bone_idx = bone_names.index(bone_name)
        except ValueError:
            return None

        # Walk up hierarchy to get world position
        pos = [0.0, 0.0, 0.0]
        current_idx = bone_idx
        visited = set()

        while current_idx >= 0 and current_idx < len(bones):
            if current_idx in visited:
                break
            visited.add(current_idx)
            bone = bones[current_idx]
            pos[0] += bone.position[0]
            pos[1] += bone.position[1]
            pos[2] += bone.position[2]
            current_idx = bone.parent_index

        return tuple(pos)

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
