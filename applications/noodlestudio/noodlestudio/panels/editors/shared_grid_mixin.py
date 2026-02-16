"""Shared grid mixin for editor views.

Provides grid snap, grid rendering, and grid persistence.
Assumes self is a QGraphicsView subclass.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor


class SharedGridMixin:
    """Grid snapping and rendering for canvas editor views.

    Configure behavior via class attributes on the concrete view:
        GRID_PEN: QPen for grid lines
        GRID_Z_VALUE: Z-value for grid lines (behind content)
        GRID_SETTINGS_ORG: QSettings organization name
        GRID_SETTINGS_APP: QSettings application name
    """

    GRID_PEN = QPen(QColor("#333333"), 1, Qt.PenStyle.DotLine)
    GRID_Z_VALUE = -100
    GRID_SETTINGS_ORG = "Noodlings"
    GRID_SETTINGS_APP = "SharedEditor"

    def _init_grid_state(self):
        """Initialize grid mixin state. Call from concrete view __init__."""
        from PyQt6.QtCore import QSettings
        settings = QSettings(self.GRID_SETTINGS_ORG, self.GRID_SETTINGS_APP)

        self._snap_to_grid = settings.value("grid/snap_enabled", False, type=bool)
        self._grid_size = settings.value("grid/size", 20, type=int)
        self._grid_visible = self._snap_to_grid
        self._grid_lines = []

    def toggle_grid(self, enabled: bool):
        """Toggle grid snapping and grid visibility."""
        self._snap_to_grid = enabled
        self._grid_visible = enabled

        from PyQt6.QtCore import QSettings
        settings = QSettings(self.GRID_SETTINGS_ORG, self.GRID_SETTINGS_APP)
        settings.setValue("grid/snap_enabled", enabled)

        if enabled:
            self._draw_grid()
        else:
            self._clear_grid()

    def set_grid_size(self, size: int):
        """Set grid size in pixels."""
        self._grid_size = size

        from PyQt6.QtCore import QSettings
        settings = QSettings(self.GRID_SETTINGS_ORG, self.GRID_SETTINGS_APP)
        settings.setValue("grid/size", size)

        if self._grid_visible:
            self._clear_grid()
            self._draw_grid()

    def snap_position(self, x: float, y: float) -> tuple:
        """Snap (x, y) to grid if snapping is enabled. Returns (x, y)."""
        if not self._snap_to_grid:
            return (x, y)
        gs = self._grid_size
        return (round(x / gs) * gs, round(y / gs) * gs)

    def _draw_grid(self):
        """Draw grid lines on the scene."""
        scene = self.scene()
        if scene is None:
            return

        self._clear_grid()

        scene_rect = scene.sceneRect()
        gs = self._grid_size

        x = scene_rect.left()
        while x <= scene_rect.right():
            if x % gs == 0:
                line = scene.addLine(
                    x, scene_rect.top(), x, scene_rect.bottom(), self.GRID_PEN
                )
                line.setZValue(self.GRID_Z_VALUE)
                self._grid_lines.append(line)
            x += gs

        y = scene_rect.top()
        while y <= scene_rect.bottom():
            if y % gs == 0:
                line = scene.addLine(
                    scene_rect.left(), y, scene_rect.right(), y, self.GRID_PEN
                )
                line.setZValue(self.GRID_Z_VALUE)
                self._grid_lines.append(line)
            y += gs

    def _clear_grid(self):
        """Remove grid lines from the scene."""
        scene = self.scene()
        for line in list(self._grid_lines):
            try:
                if scene is not None and line.scene() == scene:
                    scene.removeItem(line)
            except RuntimeError:
                pass
        self._grid_lines.clear()
