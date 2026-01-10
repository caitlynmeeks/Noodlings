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
#   Character Overlay Window
#
#   Transparent VRM character overlay that floats over the main UI.
#   QOpenGLWidget can't composite transparently as an embedded widget,
#   so we use a separate frameless overlay window that follows the parent.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.overlay
# PURPOSE:  Character Overlay Window
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   CharacterOverlayWindow
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import QMainWindow
    from PyQt6.QtCore import Qt, QTimer
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


if QT_AVAILABLE:
    from .components.vrm_viewport import VRMViewport, VRMViewportWidget

    class CharacterOverlayWindow(QMainWindow):
        """
        Transparent overlay window for VRM character display.

        This window is frameless and transparent, allowing the character
        to float over the main UI without a background box. Uses a timer
        to track the parent window position.

        Usage:
            overlay = CharacterOverlayWindow(
                parent_window=main_window,
                vrm_path="/path/to/character.vrm"
            )
            overlay.show()

        The overlay will automatically follow the parent window's position.
        """

        def __init__(
            self,
            parent_window: QMainWindow,
            vrm_path: str,
            size: Tuple[int, int] = (300, 400),
            offset: Tuple[int, int] = (20, 100),
            anchor: str = "right"
        ):
            """
            Initialize the character overlay window.

            Args:
                parent_window: The main window to follow
                vrm_path: Path to .vrm file
                size: (width, height) of overlay window
                offset: (x, y) offset from anchor position
                anchor: "right" or "left" side of parent window
            """
            super().__init__()
            self.parent_window = parent_window
            self._size = size
            self._offset = offset
            self._anchor = anchor

            # Frameless, transparent, stays on top, no taskbar entry
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint |
                Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

            # Create VRMViewport component with transparency
            component = VRMViewport("character_overlay")
            component.transparent = True
            component.vrm_path = vrm_path
            component.show_grid = False
            component.show_skeleton = False
            component.interactive = False  # No camera controls on overlay

            # Camera setup for portrait view (head/upper body)
            component.camera.distance = 2.0
            component.camera.elevation = 5
            component.camera.azimuth = 175  # Slightly off center
            component.camera.target = (0.0, 0.85, 0.0)  # Center on upper body

            # Create the viewport widget
            self.viewport = VRMViewportWidget(component, self)
            self.setCentralWidget(self.viewport)
            self.setFixedSize(*size)

            # Timer to follow parent window position
            self._follow_timer = QTimer()
            self._follow_timer.timeout.connect(self._follow_parent)
            self._follow_timer.start(50)  # 20 FPS position updates

            logger.info(f"CharacterOverlayWindow created: {vrm_path}")

        def _follow_parent(self):
            """Update position to follow parent window."""
            if self.parent_window and self.parent_window.isVisible():
                geo = self.parent_window.geometry()

                if self._anchor == "right":
                    # Position on right side of parent
                    x = geo.right() - self.width() - self._offset[0]
                else:
                    # Position on left side of parent
                    x = geo.left() + self._offset[0]

                y = geo.top() + self._offset[1]
                self.move(x, y)

                # Match visibility with parent
                if not self.isVisible():
                    self.show()
            else:
                # Hide when parent is hidden
                if self.isVisible():
                    self.hide()

        def set_muscles(self, muscles: Dict[str, float]):
            """
            Apply muscle values to the character.

            Args:
                muscles: Dict mapping muscle name to value (-1 to 1)
            """
            if self.viewport:
                self.viewport.set_muscles(muscles)

        def set_blend_shapes(self, shapes: Dict[str, float]):
            """
            Apply blend shape weights to the character.

            Args:
                shapes: Dict mapping shape name to weight (0 to 1)
            """
            if self.viewport:
                self.viewport.set_blend_shapes(shapes)

        def set_camera(
            self,
            distance: Optional[float] = None,
            elevation: Optional[float] = None,
            azimuth: Optional[float] = None,
            target: Optional[Tuple[float, float, float]] = None
        ):
            """
            Adjust camera parameters.

            Args:
                distance: Distance from target
                elevation: Vertical angle in degrees
                azimuth: Horizontal angle in degrees
                target: (x, y, z) look-at target
            """
            if self.viewport:
                self.viewport.set_camera(
                    distance=distance,
                    elevation=elevation,
                    azimuth=azimuth,
                    target=target
                )

        def set_anchor(self, anchor: str, offset: Optional[Tuple[int, int]] = None):
            """
            Change the anchor position.

            Args:
                anchor: "right" or "left"
                offset: Optional new (x, y) offset
            """
            self._anchor = anchor
            if offset:
                self._offset = offset
            self._follow_parent()

        def closeEvent(self, event):
            """Stop timer when closing."""
            self._follow_timer.stop()
            super().closeEvent(event)


# Fallback when Qt not available
if not QT_AVAILABLE:
    class CharacterOverlayWindow:
        """Stub when PyQt6 is not available."""

        def __init__(self, *args, **kwargs):
            logger.warning("CharacterOverlayWindow requires PyQt6")

        def show(self):
            pass

        def hide(self):
            pass

        def set_muscles(self, muscles):
            pass

        def set_blend_shapes(self, shapes):
            pass


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
