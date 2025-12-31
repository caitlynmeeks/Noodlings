"""
Scene Hierarchy Panel - Unity-style entity tree

Shows all prims in the noodleMUSH world:
- Rooms (with exits)
- Users (Noodlers)
- Noodlings
- Prims (WANTED POSTER, RADIO, etc.)

Click to select -> Inspector shows editable properties

Author: Caitlyn + Claude
Date: November 17, 2025
Refactored: December 30, 2025 (split into mixins)
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

from ..core.scene_graph import SceneGraph

# Import all mixins
from .scene_hierarchy_ui_setup_mixin import SceneHierarchyUISetupMixin
from .scene_hierarchy_graph_mixin import SceneHierarchyGraphMixin
from .scene_hierarchy_stage_mixin import SceneHierarchyStageMixin
from .scene_hierarchy_refresh_mixin import SceneHierarchyRefreshMixin
from .scene_hierarchy_selection_mixin import SceneHierarchySelectionMixin
from .scene_hierarchy_context_menu_mixin import SceneHierarchyContextMenuMixin
from .scene_hierarchy_export_mixin import SceneHierarchyExportMixin
from .scene_hierarchy_create_mixin import SceneHierarchyCreateMixin
from .scene_hierarchy_derez_mixin import SceneHierarchyDerezMixin
from .scene_hierarchy_utils_mixin import SceneHierarchyUtilsMixin


class SceneHierarchy(
    SceneHierarchyUISetupMixin,
    SceneHierarchyGraphMixin,
    SceneHierarchyStageMixin,
    SceneHierarchyRefreshMixin,
    SceneHierarchySelectionMixin,
    SceneHierarchyContextMenuMixin,
    SceneHierarchyExportMixin,
    SceneHierarchyCreateMixin,
    SceneHierarchyDerezMixin,
    SceneHierarchyUtilsMixin,
    QWidget
):
    """
    Unity-style Scene Hierarchy panel.

    Tree structure (entities directly at root, no Stage wrapper):
    |- main (r=10, f=5)              [zone]
    |- Red                [||]       [noodling]
    |- Servnak            [||]       [noodling]
    |- WANTED POSTER                 [prop]
    |- RADIO                         [prop]
    +- My Folder/                    [user-created folder]
       +- LAMP                       [prop]

    Stage is selected via dropdown at top of panel, not shown in tree.
    Users can create folders to organize content.

    Supports project structure (Stages/xxx/...) and legacy format.
    """

    entitySelected = pyqtSignal(str, dict)  # (entity_type, entity_data)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_base = "http://localhost:8081/api"
        self.current_room = "room_000"  # Start at Nexus (legacy)
        self.current_stage = None  # New project format
        self.project_manager = None  # Set via set_project_manager()

        # Scene graph - the canonical data model for hierarchy
        self.scene_graph = SceneGraph(self)
        self.scene_graph.nodeAdded.connect(self._on_graph_changed)
        self.scene_graph.nodeRemoved.connect(self._on_graph_changed)
        self.scene_graph.nodeReparented.connect(self._on_node_reparented)
        self.scene_graph.nodeRenamed.connect(self._on_node_renamed)

        # Map tree items to node IDs for quick lookup
        # Note: QTreeWidgetItem is not hashable, so we use id(item) as key
        self._item_id_to_node_id = {}  # {id(QTreeWidgetItem): node_id}
        self._node_id_to_item = {}  # {node_id: QTreeWidgetItem}

        # Track expanded state (survives tree rebuild)
        self.expanded_items = set()

        # Track selected item (survives tree rebuild)
        self.selected_item_path = None

        # Derez confirmation settings
        self.derez_confirm = True  # Show confirmation dialog

        # Track agent pause states
        self.agent_pause_states = {}  # {agent_id: bool}

        # Server state - controls whether full hierarchy is shown
        self._server_running = False

        # Flag to prevent refresh during edits
        self._suppress_refresh = False
        self._editing_item = None  # Track item being inline edited

        # Dirty flag - True when hierarchy has unsaved changes
        self._dirty = False

        # Initialize UI directly on this widget
        self.init_ui(self)

        # NO MORE POLLING TIMER - Event-driven updates only
        # Refresh happens when:
        # 1. Project opened/changed (set_project_manager)
        # 2. Server state changes (set_server_state)
        # 3. User explicitly requests (F5 / refresh button)
        # 4. Server sends WebSocket event (future)
        # Local changes update tree surgically, not via rebuild

    def set_project_manager(self, project_manager):
        """Set project manager reference for loading from project structure."""
        self.project_manager = project_manager
        print(f"[SceneHierarchy] set_project_manager called, project_manager={project_manager is not None}")
        # Refresh stage selector when project changes
        self.populate_stage_selector()
        # Do initial refresh (event-driven, no polling timer)
        self.refresh_scene()
