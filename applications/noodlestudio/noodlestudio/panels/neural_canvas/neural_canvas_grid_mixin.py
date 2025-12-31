"""
Neural Canvas Grid Mixin - Grid snapping and display

Contains:
- toggle_grid_snap: Toggle grid snapping and visibility
- set_grid_size: Set grid size in pixels
- _draw_grid: Draw grid lines on scene
- _clear_grid: Remove grid lines from scene

Author: Noodlings Project
Date: December 2025
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor


class NeuralCanvasGridMixin:
    """Mixin providing grid operations for NeuralCanvasView."""

    def toggle_grid_snap(self, enabled: bool):
        """Toggle grid snapping and grid visibility."""
        self.snap_to_grid = enabled
        self.grid_visible = enabled

        # Save to settings
        from PyQt6.QtCore import QSettings
        settings = QSettings('Noodlings', 'NeuralCanvas')
        settings.setValue('grid/snap_enabled', enabled)

        if enabled:
            self._draw_grid()
        else:
            self._clear_grid()

        print(f"[Neural Canvas] Grid snapping: {'ON' if enabled else 'OFF'}")

    def set_grid_size(self, size: int):
        """Set grid size in pixels."""
        self.grid_size = size

        # Save to settings
        from PyQt6.QtCore import QSettings
        settings = QSettings('Noodlings', 'NeuralCanvas')
        settings.setValue('grid/size', size)

        # Redraw grid if visible
        if self.grid_visible:
            self._clear_grid()
            self._draw_grid()

        print(f"[Neural Canvas] Grid size: {size}px")

    def _draw_grid(self):
        """Draw grid lines on scene."""
        if not self.scene:
            print("[Neural Canvas] Cannot draw grid - no scene")
            return

        # Clear any existing grid first
        self._clear_grid()

        from PyQt6.QtWidgets import QGraphicsLineItem
        scene_rect = self.scene.sceneRect()
        grid_size = self.grid_size

        # Faint gray for grid lines (matches Facets Editor)
        grid_pen = QPen(QColor("#333333"), 1, Qt.PenStyle.DotLine)

        # Draw vertical lines
        x = scene_rect.left()
        while x <= scene_rect.right():
            if x % grid_size == 0:
                try:
                    line = self.scene.addLine(
                        x, scene_rect.top(),
                        x, scene_rect.bottom(),
                        grid_pen
                    )
                    line.setZValue(-100)  # Behind everything
                    self.grid_lines.append(line)
                except Exception as e:
                    print(f"[Neural Canvas] Error adding vertical grid line: {e}")
                    break
            x += grid_size

        # Draw horizontal lines
        y = scene_rect.top()
        while y <= scene_rect.bottom():
            if y % grid_size == 0:
                try:
                    line = self.scene.addLine(
                        scene_rect.left(), y,
                        scene_rect.right(), y,
                        grid_pen
                    )
                    line.setZValue(-100)  # Behind everything
                    self.grid_lines.append(line)
                except Exception as e:
                    print(f"[Neural Canvas] Error adding horizontal grid line: {e}")
                    break
            y += grid_size

        print(f"[Neural Canvas] Drew {len(self.grid_lines)} grid lines")

    def _clear_grid(self):
        """Remove grid lines from scene."""
        if not self.scene:
            self.grid_lines.clear()
            return

        # Safely remove each line
        for line in list(self.grid_lines):  # Copy list to avoid modification during iteration
            try:
                if line.scene() == self.scene:  # Verify item is still in scene
                    self.scene.removeItem(line)
            except Exception as e:
                print(f"[Neural Canvas] Error removing grid line: {e}")

        self.grid_lines.clear()
        print("[Neural Canvas] Cleared grid lines")
