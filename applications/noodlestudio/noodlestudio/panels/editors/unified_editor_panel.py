"""Unified editor panel with depth-stack navigation.

Hosts a stack of domain views (assembly editor, neural canvas, etc.).
Only the top view is visible. Navigation via double-click (dive in)
and breadcrumb bar (ascend). Each view implements DepthViewProtocol.

C.6: Facade methods matching old FacetsEditorPanel API so the main
window can call the same methods on this panel. Delegates to the root
AssemblyEditorView on the stack.
"""

import os
from typing import Optional

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

    # NC depth view forwarding signals (connected when NC view is pushed)
    ncNodeSelected = pyqtSignal(str)
    ncParamChanged = pyqtSignal(str, str, object)
    ncGraphLoaded = pyqtSignal()

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

        # Toolbar area (holds the current view's toolbar widget)
        self._toolbar_area = QVBoxLayout()
        self._toolbar_area.setContentsMargins(0, 0, 0, 0)
        self._toolbar_area.setSpacing(0)
        layout.addLayout(self._toolbar_area)
        self._current_toolbar = None

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

        # Auto-wire depth navigation signal if the view supports it
        if hasattr(view, 'containerDoubleClicked'):
            view.containerDoubleClicked.connect(self._on_container_double_clicked)

        # Auto-wire facetSelected forwarding from assembly views
        if hasattr(view, 'facetSelected'):
            view.facetSelected.connect(self.facetSelected)
        if hasattr(view, 'assemblyModified'):
            view.assemblyModified.connect(self.assemblyModified)

        # Auto-wire NC signal forwarding from depth views
        self._connect_nc_signals(view)

        # Embed the view's toolbar if it provides one
        self._swap_toolbar(view)

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

        # Show the new top and swap toolbar
        if self._stack:
            self._stack[-1].view.show()
            self._swap_toolbar(self._stack[-1].view)
        else:
            self._clear_toolbar()

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

        self._clear_toolbar()
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

    # ==================== Depth navigation dispatch ====================

    def _on_container_double_clicked(self, facet_type: str, data_path: str,
                                     label: str):
        """Handle containerDoubleClicked from any depth view.

        Inherits context from the current top stack frame so that
        project_root and other metadata flow from parent to child.
        """
        ctx = {}
        if self._stack:
            ctx = dict(self._stack[-1].context)
        self.on_double_click_container(facet_type, data_path, label, ctx)

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

    def _swap_toolbar(self, view):
        """Show the toolbar for the given view, hiding any previous toolbar."""
        self._clear_toolbar()
        if hasattr(view, 'get_toolbar_widget'):
            toolbar = view.get_toolbar_widget()
            if toolbar is not None:
                self._toolbar_area.addWidget(toolbar)
                toolbar.show()
                self._current_toolbar = toolbar

    def _clear_toolbar(self):
        """Hide and remove the current toolbar from the toolbar area."""
        if self._current_toolbar is not None:
            self._current_toolbar.hide()
            self._toolbar_area.removeWidget(self._current_toolbar)
            self._current_toolbar = None

    def _save_and_remove_view(self, frame: _StackFrame):
        """Save a view's data, disconnect signals, and remove from the layout."""
        if hasattr(frame.view, "save_data"):
            frame.view.save_data()

        # Disconnect all auto-wired signals
        for signal_name in ('containerDoubleClicked', 'facetSelected', 'assemblyModified'):
            if hasattr(frame.view, signal_name):
                try:
                    sig = getattr(frame.view, signal_name)
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass

        # Disconnect NC forwarding signals
        self._disconnect_nc_signals(frame.view)

        frame.view.hide()
        self._view_layout.removeWidget(frame.view)
        frame.view.setParent(None)
        frame.view.deleteLater()

    def _connect_nc_signals(self, view):
        """Connect NC panel signals from a NeuralCanvasDepthView for forwarding."""
        panel = getattr(view, '_panel', None)
        if panel is None:
            return
        if hasattr(panel, 'node_selected'):
            panel.node_selected.connect(self.ncNodeSelected)
        if hasattr(panel, 'graph_loaded'):
            panel.graph_loaded.connect(self.ncGraphLoaded)
        canvas_view = getattr(panel, 'canvas_view', None)
        if canvas_view and hasattr(canvas_view, 'node_param_changed'):
            canvas_view.node_param_changed.connect(self.ncParamChanged)

    def _disconnect_nc_signals(self, view):
        """Disconnect NC panel signals when a depth view is popped."""
        panel = getattr(view, '_panel', None)
        if panel is None:
            return
        for sig_name in ('node_selected', 'graph_loaded'):
            if hasattr(panel, sig_name):
                try:
                    getattr(panel, sig_name).disconnect()
                except (TypeError, RuntimeError):
                    pass
        canvas_view = getattr(panel, 'canvas_view', None)
        if canvas_view and hasattr(canvas_view, 'node_param_changed'):
            try:
                canvas_view.node_param_changed.disconnect()
            except (TypeError, RuntimeError):
                pass

    # ==================== Facade API (C.6) ====================
    #
    # These methods match the old FacetsEditorPanel interface so the
    # main window can call the same methods during parallel testing.
    # Each delegates to the root AssemblyEditorView on the stack.

    def _root_view(self):
        """Return the root (level 0) AssemblyEditorView, or None."""
        from .assembly_editor_view import AssemblyEditorView
        if self._stack and isinstance(self._stack[0].view, AssemblyEditorView):
            return self._stack[0].view
        return None

    def _ensure_root_view(self):
        """Create and push an AssemblyEditorView if the stack is empty."""
        from .assembly_editor_view import AssemblyEditorView
        if not self._stack or not isinstance(self._stack[0].view, AssemblyEditorView):
            view = AssemblyEditorView(parent=self._view_container)
            self.push_view(view, "assembly")
        return self._root_view()

    # -- Assembly loading --

    def load_assembly_from_data(self, assembly, force_reload=False,
                                source_path=None):
        """Load a parsed FacetAssembly into the editor.

        If the stack is empty, creates a root AssemblyEditorView first.
        Pops any depth views back to root before loading.
        """
        if self.depth() > 1:
            self.pop_to(0)

        root = self._ensure_root_view()
        if root:
            root.load_assembly_from_data(
                assembly, source_path=source_path, force_reload=force_reload
            )

    def clear_editor(self):
        """Show empty state (no noodling selected)."""
        self.clear_stack()

    def set_current_agent(self, agent_id: str):
        """Store the current agent ID for pause/resume."""
        root = self._root_view()
        if root:
            root.current_agent_id = agent_id

    def load_assembly(self, assembly_path):
        """Load assembly from a file path (used by soft_restart)."""
        from ...core.facet_system import FacetAssembly

        path = str(assembly_path)
        if not os.path.exists(path):
            return

        assembly = FacetAssembly.load_yaml(path)
        if assembly:
            self.load_assembly_from_data(
                assembly, source_path=path, force_reload=True
            )

    # -- Ensemble --

    def set_ensemble_noodlings(self, noodlings: list):
        """Configure the noodling selector for ensemble mode."""
        root = self._ensure_root_view()
        if root and hasattr(root, 'set_ensemble_noodlings'):
            root.set_ensemble_noodlings(noodlings)

    def clear_ensemble_noodlings(self):
        """Hide the noodling selector, return to single mode."""
        root = self._root_view()
        if root and hasattr(root, 'clear_ensemble_noodlings'):
            root.clear_ensemble_noodlings()

    def select_noodling(self, noodling_id: str):
        """Programmatic noodling selection (turn-taking)."""
        root = self._root_view()
        if root and hasattr(root, 'select_noodling'):
            root.select_noodling(noodling_id)

    # -- Execution events --

    def _handle_execution_event(self, event: dict):
        """Forward execution event to the root AssemblyEditorView."""
        root = self._root_view()
        if root and hasattr(root, '_handle_execution_event'):
            root._handle_execution_event(event)

    # -- Save --

    def save_if_dirty(self):
        """Save all views in the stack."""
        self.save_all()

    def _save_assembly_to_disk(self):
        """Persist the root assembly to disk."""
        root = self._root_view()
        if root and hasattr(root, '_save_assembly_to_disk'):
            root._save_assembly_to_disk()

    # -- Refresh --

    def refresh_node_for_facet(self, facet_id: str):
        """Refresh a specific node's visual after external property change."""
        root = self._root_view()
        if root and hasattr(root, 'refresh_node_for_facet'):
            root.refresh_node_for_facet(facet_id)

    # -- Pause state --

    def set_pause_state(self, paused: bool):
        """Set cognition pause state on the root view."""
        root = self._root_view()
        if root is None:
            return
        root.cognition_paused = paused
        if hasattr(root, '_pause_button'):
            root._pause_button.setChecked(paused)

    # -- NC graph access --

    def get_current_nc_graph(self):
        """Return the NC graph from the current depth view, if applicable."""
        top = self.current_view()
        panel = getattr(top, '_panel', None)
        if panel and hasattr(panel, 'graph'):
            return panel.graph
        return None

    # -- Properties for inspector / external access --

    @property
    def current_assembly(self):
        """The loaded FacetAssembly, or None."""
        root = self._root_view()
        return root._assembly if root else None

    @property
    def current_assembly_path(self) -> Optional[str]:
        """Path to the loaded assembly YAML, or None."""
        root = self._root_view()
        return root._assembly_path if root else None

    @property
    def current_agent_id(self) -> Optional[str]:
        """The current agent ID for pause/resume."""
        root = self._root_view()
        return root.current_agent_id if root else None

    @property
    def node_graphics(self) -> dict:
        """Map of facet_id -> FacetNodeItem (for inspector execution I/O)."""
        root = self._root_view()
        return root._node_items if root else {}

    @property
    def cognition_paused(self) -> bool:
        root = self._root_view()
        return root.cognition_paused if root else False

    @cognition_paused.setter
    def cognition_paused(self, value: bool):
        root = self._root_view()
        if root:
            root.cognition_paused = value

    @property
    def pause_button(self):
        """The pause/resume QPushButton (for scene_hierarchy_utils_mixin)."""
        root = self._root_view()
        return root._pause_button if root and hasattr(root, '_pause_button') else None

    @property
    def bottom_pause_btn(self):
        """Compatibility stub -- returns the same pause button."""
        return self.pause_button
