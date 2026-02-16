"""Unified editor panel with depth-stack navigation.

Hosts a stack of domain views (assembly editor, neural canvas, etc.).
Only the top view is visible. Navigation via double-click (dive in)
and breadcrumb bar (ascend). Each view implements DepthViewProtocol.
"""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QKeyEvent

from .breadcrumb_bar import BreadcrumbBar


class _StackFrame:
    """One entry in the depth stack."""

    __slots__ = ("view", "label", "context")

    def __init__(self, view, label: str, context: dict):
        self.view = view
        self.label = label
        self.context = context


class UnifiedEditorPanel(QWidget):
    """Panel shell that hosts a stack of depth views with breadcrumb navigation.

    Signals:
        facetSelected(object): Emitted when a facet is selected (for Inspector).
        assemblyModified(): Emitted when the current assembly is modified.
        depthChanged(int): Emitted when the stack depth changes (push/pop).
    """

    facetSelected = pyqtSignal(object)
    assemblyModified = pyqtSignal()
    depthChanged = pyqtSignal(int)

    # Class-level depth view registry: facet_type_name -> view_class
    _depth_view_registry = {}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack = []  # list of _StackFrame
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Breadcrumb bar at top
        self._breadcrumb = BreadcrumbBar()
        self._breadcrumb.segmentClicked.connect(self._on_breadcrumb_clicked)
        layout.addWidget(self._breadcrumb)

        # View container (views are added/removed here)
        self._view_container = QWidget()
        self._view_layout = QVBoxLayout(self._view_container)
        self._view_layout.setContentsMargins(0, 0, 0, 0)
        self._view_layout.setSpacing(0)
        layout.addWidget(self._view_container, stretch=1)

        # Empty state
        self._empty_label = QLabel("No assembly loaded")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #888888; font-size: 14px;")
        self._view_layout.addWidget(self._empty_label)

        self._update_breadcrumb()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ==================== Stack operations ====================

    def push_view(self, view, breadcrumb_label: str, context: dict = None):
        """Push a new depth view onto the stack.

        Hides the previous top view (preserving its state).
        Shows the new view. Updates the breadcrumb bar.

        Args:
            view: A widget implementing DepthViewProtocol.
            breadcrumb_label: Short label for the breadcrumb bar.
            context: Optional context dict passed to the view.
        """
        # Hide current top view
        if self._stack:
            self._stack[-1].view.hide()

        # Hide empty state
        self._empty_label.hide()

        # Add new view to container
        self._view_layout.addWidget(view)
        view.show()

        # Push frame
        frame = _StackFrame(view, breadcrumb_label, context or {})
        self._stack.append(frame)

        self._update_breadcrumb()
        self.depthChanged.emit(len(self._stack))

    def pop_to(self, depth_index: int):
        """Pop views down to the given depth index (0-based).

        Calls save_data() on each popped view. Shows the view at depth_index.

        Args:
            depth_index: Target depth. 0 = root view only.
        """
        if depth_index < 0 or depth_index >= len(self._stack):
            return

        # Pop frames from top down to (but not including) target
        while len(self._stack) > depth_index + 1:
            frame = self._stack.pop()
            self._save_and_remove_view(frame)

        # Show the new top
        if self._stack:
            self._stack[-1].view.show()

        self._update_breadcrumb()
        self.depthChanged.emit(len(self._stack))

    def pop_one(self):
        """Pop the top view (Backspace). No-op if at root or empty."""
        if len(self._stack) <= 1:
            return
        self.pop_to(len(self._stack) - 2)

    def current_view(self):
        """Return the top view on the stack, or None if empty."""
        if self._stack:
            return self._stack[-1].view
        return None

    def depth(self) -> int:
        """Current stack depth (0 = empty, 1 = root view only)."""
        return len(self._stack)

    def clear_stack(self):
        """Pop and save all views. Shows empty state."""
        while self._stack:
            frame = self._stack.pop()
            self._save_and_remove_view(frame)

        self._empty_label.show()
        self._update_breadcrumb()
        self.depthChanged.emit(0)

    def save_all(self):
        """Save all views in the stack (Ctrl+S cascade)."""
        for frame in self._stack:
            if hasattr(frame.view, "save_data"):
                frame.view.save_data()

    # ==================== Depth view registry ====================

    @classmethod
    def register_depth_view(cls, facet_type_name: str, view_class):
        """Register a view class for a facet type.

        Args:
            facet_type_name: e.g. "NeuralCanvasFacet", "GroupFacet"
            view_class: Class to instantiate when double-clicking this type.
        """
        cls._depth_view_registry[facet_type_name] = view_class

    @classmethod
    def get_registered_view_class(cls, facet_type_name: str):
        """Look up the view class for a facet type, or None."""
        return cls._depth_view_registry.get(facet_type_name)

    def on_double_click_container(self, facet_type: str, data_path: str,
                                  breadcrumb_label: str, context: dict = None):
        """Handle double-click on a container facet (dispatches to registry).

        If the facet type has a registered view class, instantiates it,
        calls load_data(), and pushes it onto the stack.

        Args:
            facet_type: The facet type name (e.g. "NeuralCanvasFacet").
            data_path: Path to the data file for this depth level.
            breadcrumb_label: Label for the breadcrumb.
            context: Additional context for load_data().
        """
        view_class = self._depth_view_registry.get(facet_type)
        if view_class is None:
            return

        ctx = context or {}
        view = view_class(parent=self._view_container)

        if hasattr(view, "load_data"):
            view.load_data(data_path, ctx)

        self.push_view(view, breadcrumb_label, ctx)

    # ==================== Key handling ====================

    def keyPressEvent(self, event: QKeyEvent):
        """Handle Backspace for depth navigation."""
        if event.key() == Qt.Key.Key_Backspace:
            self.pop_one()
        else:
            super().keyPressEvent(event)

    # ==================== Internals ====================

    def _on_breadcrumb_clicked(self, depth_index: int):
        """Handle breadcrumb segment click."""
        self.pop_to(depth_index)

    def _update_breadcrumb(self):
        """Sync breadcrumb bar with current stack."""
        if not self._stack:
            self._breadcrumb.clear()
            return
        labels = [frame.label for frame in self._stack]
        self._breadcrumb.set_path(labels)

    def _save_and_remove_view(self, frame: _StackFrame):
        """Save a view's data and remove it from the layout."""
        if hasattr(frame.view, "save_data"):
            frame.view.save_data()
        frame.view.hide()
        self._view_layout.removeWidget(frame.view)
        frame.view.setParent(None)
        frame.view.deleteLater()
