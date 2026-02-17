"""NeuralCanvasPanel wrapper for depth-stack navigation.

Wraps the existing NeuralCanvasPanel as a depth view that can be pushed
onto the UnifiedEditorPanel's view stack when the user double-clicks a
NeuralCanvasFacet node at the assembly level.
"""

import os

from PyQt6.QtWidgets import QWidget, QVBoxLayout

from ..neural_canvas.neural_canvas_panel import NeuralCanvasPanel


class NeuralCanvasDepthView(QWidget):
    """Depth-stack adapter for NeuralCanvasPanel.

    Implements the DepthViewProtocol interface (load_data, save_data,
    get_breadcrumb_label, has_unsaved_changes) by delegating to the
    embedded NeuralCanvasPanel.

    Path resolution: The data_path passed to load_data() may be relative
    to the project root. The context dict carries 'project_root' from
    the parent stack frame so the path can be resolved.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._panel = NeuralCanvasPanel()
        layout.addWidget(self._panel)

        self._data_path = None
        self._context = {}

    # ==================== DepthViewProtocol ====================

    def load_data(self, data_path: str, context: dict) -> None:
        """Load a .nncanvas file into the embedded NeuralCanvasPanel.

        Args:
            data_path: Path to .nncanvas file (may be relative to project).
            context: Must contain 'project_root' for relative path resolution.
        """
        self._data_path = data_path
        self._context = context or {}

        resolved = self._resolve_path(data_path, self._context)
        if resolved and os.path.exists(resolved):
            self._panel._load_from_file(resolved)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self._panel.canvas_view.frame_all_nodes)

    def save_data(self) -> None:
        """Persist any unsaved changes via NeuralCanvasPanel's save mechanism."""
        self._panel.save_if_dirty()

    def get_breadcrumb_label(self) -> str:
        """Return the graph name for the breadcrumb bar."""
        if self._panel.graph and self._panel.graph.name:
            name = self._panel.graph.name
            if name != "No neural canvas loaded":
                return name
        return "Neural Canvas"

    def has_unsaved_changes(self) -> bool:
        """NC auto-saves on every modification, so no unsaved changes accumulate."""
        return False

    # ==================== Internals ====================

    @staticmethod
    def _resolve_path(data_path: str, context: dict) -> str:
        """Resolve a potentially-relative .nncanvas path against the project root.

        Args:
            data_path: Path from the Facet's nncanvas_path field.
            context: Dict with optional 'project_root' key.

        Returns:
            Absolute path to the .nncanvas file.
        """
        if os.path.isabs(data_path):
            return data_path
        project_root = context.get('project_root', '')
        if project_root:
            return os.path.join(project_root, data_path)
        return data_path
