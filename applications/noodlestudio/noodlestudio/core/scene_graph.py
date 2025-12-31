"""
Scene Graph - Unity-style scene hierarchy manager

The SceneGraph manages the tree of SceneNodes, providing:
- CRUD operations (create, read, update, delete)
- Reparenting with ordering
- Path-based lookup ("Alicia/skeleton/leftHand")
- YAML persistence
- Signals for UI updates

This is the canonical source of truth for scene organization.
The Stage View panel reflects this graph.

Author: Commander Spock + Cadet Caity
Date: December 27, 2025
"""

from typing import List, Dict, Any, Optional, Callable, Set
from pathlib import Path
import os

try:
    from PyQt6.QtCore import QObject, pyqtSignal
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from .scene_node import SceneNode, SceneNodeType


class SceneGraph(QObject if HAS_PYQT else object):
    """
    Manages the scene hierarchy tree.

    The graph stores all nodes in a flat dict (by ID) with parent/children
    references forming the tree structure. Root-level nodes have parent_id=None.

    Signals (PyQt6):
        nodeAdded(node_id: str)
        nodeRemoved(node_id: str)
        nodeReparented(node_id: str, old_parent_id: str, new_parent_id: str)
        nodeRenamed(node_id: str, new_name: str)
        graphLoaded()
        graphSaved()
    """

    if HAS_PYQT:
        nodeAdded = pyqtSignal(str)
        nodeRemoved = pyqtSignal(str)
        nodeReparented = pyqtSignal(str, str, str)  # node_id, old_parent, new_parent
        nodeRenamed = pyqtSignal(str, str)
        nodeVisibilityChanged = pyqtSignal(str, bool)
        graphLoaded = pyqtSignal()
        graphSaved = pyqtSignal()

    def __init__(self, parent=None):
        if HAS_PYQT:
            super().__init__(parent)
        else:
            super().__init__()

        self._nodes: Dict[str, SceneNode] = {}
        self._root_ids: List[str] = []  # Ordered list of root-level node IDs
        self._dirty: bool = False        # Has unsaved changes?
        self._file_path: Optional[str] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def nodes(self) -> Dict[str, SceneNode]:
        """All nodes by ID (read-only view)."""
        return self._nodes

    @property
    def root_ids(self) -> List[str]:
        """IDs of root-level nodes (ordered)."""
        return self._root_ids.copy()

    @property
    def is_dirty(self) -> bool:
        """True if there are unsaved changes."""
        return self._dirty

    @property
    def file_path(self) -> Optional[str]:
        """Path to the hierarchy file, if loaded/saved."""
        return self._file_path

    # -------------------------------------------------------------------------
    # Node Access
    # -------------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> List[SceneNode]:
        """Get ordered children of a node."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_parent(self, node_id: str) -> Optional[SceneNode]:
        """Get parent of a node."""
        node = self._nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        return self._nodes.get(node.parent_id)

    def get_root_nodes(self) -> List[SceneNode]:
        """Get root-level nodes in order."""
        return [self._nodes[nid] for nid in self._root_ids if nid in self._nodes]

    def get_all_nodes(self) -> List[SceneNode]:
        """Get all nodes (no particular order)."""
        return list(self._nodes.values())

    def get_nodes_by_type(self, node_type: SceneNodeType) -> List[SceneNode]:
        """Get all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    # -------------------------------------------------------------------------
    # Path-based Access
    # -------------------------------------------------------------------------

    def find_by_path(self, path: str) -> Optional[SceneNode]:
        """
        Find a node by path (e.g., "Characters/Alicia/skeleton/leftHand").

        Path segments are matched by name. Returns None if not found.
        """
        parts = path.strip('/').split('/')
        if not parts:
            return None

        # Find root node with matching name
        current = None
        for rid in self._root_ids:
            node = self._nodes.get(rid)
            if node and node.name == parts[0]:
                current = node
                break

        if not current:
            return None

        # Traverse children
        for part in parts[1:]:
            found = None
            for cid in current.children_ids:
                child = self._nodes.get(cid)
                if child and child.name == part:
                    found = child
                    break
            if not found:
                return None
            current = found

        return current

    def find_by_name(self, name: str) -> List[SceneNode]:
        """Find all nodes with a given name."""
        return [n for n in self._nodes.values() if n.name == name]

    def get_path(self, node_id: str) -> str:
        """Get the full path of a node (e.g., "Characters/Alicia")."""
        node = self._nodes.get(node_id)
        if not node:
            return ""

        parts = [node.name]
        current = node
        while current.parent_id:
            parent = self._nodes.get(current.parent_id)
            if not parent:
                break
            parts.insert(0, parent.name)
            current = parent

        return '/'.join(parts)

    # -------------------------------------------------------------------------
    # Node Creation
    # -------------------------------------------------------------------------

    def add_node(self, node: SceneNode) -> SceneNode:
        """
        Add a node to the graph.

        If parent_id is set, the node is added as a child.
        Otherwise, it's added at root level.
        """
        self._nodes[node.id] = node

        if node.parent_id:
            parent = self._nodes.get(node.parent_id)
            if parent and node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
        else:
            if node.id not in self._root_ids:
                self._root_ids.append(node.id)

        self._dirty = True
        if HAS_PYQT:
            self.nodeAdded.emit(node.id)

        return node

    def create_folder(self, name: str, parent_id: Optional[str] = None) -> SceneNode:
        """Create a new folder node."""
        node = SceneNode.create_folder(name, parent_id)
        return self.add_node(node)

    def create_node(self, name: str, node_type: SceneNodeType,
                    parent_id: Optional[str] = None,
                    asset_path: Optional[str] = None) -> SceneNode:
        """Create a new node of any type."""
        node = SceneNode.create(name, node_type, parent_id, asset_path)
        return self.add_node(node)

    # -------------------------------------------------------------------------
    # Node Modification
    # -------------------------------------------------------------------------

    def rename_node(self, node_id: str, new_name: str) -> bool:
        """Rename a node. Returns False if node is virtual or not found."""
        node = self._nodes.get(node_id)
        if not node:
            return False
        if node.is_virtual:
            return False  # Can't rename bone nodes

        old_name = node.name
        node.name = new_name
        self._dirty = True

        if HAS_PYQT:
            self.nodeRenamed.emit(node_id, new_name)

        return True

    def set_visibility(self, node_id: str, visible: bool) -> bool:
        """Set node visibility."""
        node = self._nodes.get(node_id)
        if not node:
            return False

        node.is_visible = visible
        self._dirty = True

        if HAS_PYQT:
            self.nodeVisibilityChanged.emit(node_id, visible)

        return True

    def set_expanded(self, node_id: str, expanded: bool) -> bool:
        """Set node expansion state (UI only, not saved as dirty)."""
        node = self._nodes.get(node_id)
        if not node:
            return False

        node.is_expanded = expanded
        # Don't mark dirty for UI-only state
        return True

    def set_transform(self, node_id: str,
                      position: tuple = None,
                      rotation: tuple = None,
                      scale: tuple = None) -> bool:
        """Set node local transform."""
        node = self._nodes.get(node_id)
        if not node:
            return False

        if position is not None:
            node.local_position = tuple(position)
        if rotation is not None:
            node.local_rotation = tuple(rotation)
        if scale is not None:
            node.local_scale = tuple(scale)

        self._dirty = True
        return True

    # -------------------------------------------------------------------------
    # Reparenting
    # -------------------------------------------------------------------------

    def reparent(self, node_id: str, new_parent_id: Optional[str],
                 index: int = -1) -> bool:
        """
        Move a node to a new parent.

        Args:
            node_id: The node to move
            new_parent_id: New parent (None for root level)
            index: Position in new parent's children (-1 = append)

        Returns:
            True if successful, False if invalid operation.
        """
        node = self._nodes.get(node_id)
        if not node:
            return False

        if node.is_virtual:
            return False  # Can't move bone nodes

        # Prevent parenting to self or descendant
        if new_parent_id:
            if new_parent_id == node_id:
                return False
            if self._is_descendant(new_parent_id, node_id):
                return False

        old_parent_id = node.parent_id

        # Remove from old parent
        if old_parent_id:
            old_parent = self._nodes.get(old_parent_id)
            if old_parent and node_id in old_parent.children_ids:
                old_parent.children_ids.remove(node_id)
        else:
            if node_id in self._root_ids:
                self._root_ids.remove(node_id)

        # Add to new parent
        node.parent_id = new_parent_id
        if new_parent_id:
            new_parent = self._nodes.get(new_parent_id)
            if new_parent:
                if index < 0 or index >= len(new_parent.children_ids):
                    new_parent.children_ids.append(node_id)
                else:
                    new_parent.children_ids.insert(index, node_id)
        else:
            if index < 0 or index >= len(self._root_ids):
                self._root_ids.append(node_id)
            else:
                self._root_ids.insert(index, node_id)

        self._dirty = True

        if HAS_PYQT:
            self.nodeReparented.emit(node_id, old_parent_id or "", new_parent_id or "")

        return True

    def _is_descendant(self, node_id: str, potential_ancestor_id: str) -> bool:
        """Check if node_id is a descendant of potential_ancestor_id."""
        current = self._nodes.get(node_id)
        while current:
            if current.id == potential_ancestor_id:
                return True
            current = self._nodes.get(current.parent_id) if current.parent_id else None
        return False

    def reorder_child(self, parent_id: Optional[str], child_id: str, new_index: int) -> bool:
        """Reorder a child within its current parent."""
        if parent_id:
            parent = self._nodes.get(parent_id)
            if not parent or child_id not in parent.children_ids:
                return False
            parent.children_ids.remove(child_id)
            parent.children_ids.insert(min(new_index, len(parent.children_ids)), child_id)
        else:
            if child_id not in self._root_ids:
                return False
            self._root_ids.remove(child_id)
            self._root_ids.insert(min(new_index, len(self._root_ids)), child_id)

        self._dirty = True
        return True

    # -------------------------------------------------------------------------
    # Node Deletion
    # -------------------------------------------------------------------------

    def delete_node(self, node_id: str, recursive: bool = True) -> bool:
        """
        Delete a node.

        Args:
            node_id: Node to delete
            recursive: If True, delete all descendants too

        Returns:
            True if deleted, False if node is virtual or not found.
        """
        node = self._nodes.get(node_id)
        if not node:
            return False

        if node.is_virtual:
            return False  # Can't delete bone nodes

        if recursive:
            # Delete children first (copy list to avoid modification during iteration)
            for child_id in list(node.children_ids):
                self.delete_node(child_id, recursive=True)

        # Remove from parent
        if node.parent_id:
            parent = self._nodes.get(node.parent_id)
            if parent and node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
        else:
            if node_id in self._root_ids:
                self._root_ids.remove(node_id)

        # Remove from graph
        del self._nodes[node_id]
        self._dirty = True

        if HAS_PYQT:
            self.nodeRemoved.emit(node_id)

        return True

    def clear(self):
        """Remove all nodes."""
        self._nodes.clear()
        self._root_ids.clear()
        self._dirty = True

    # -------------------------------------------------------------------------
    # Bone Integration
    # -------------------------------------------------------------------------

    def populate_bones(self, radiance_node_id: str, bone_names: List[str],
                       bone_parents: Dict[str, str] = None) -> List[SceneNode]:
        """
        Create virtual bone nodes under a RadianceComponent.

        Args:
            radiance_node_id: The radiance node to add bones to
            bone_names: List of bone names
            bone_parents: Dict mapping bone_name -> parent_bone_name

        Returns:
            List of created bone nodes.
        """
        radiance_node = self._nodes.get(radiance_node_id)
        if not radiance_node:
            return []

        # Create a "skeleton" folder under the radiance
        skeleton_folder = SceneNode.create_folder("skeleton", radiance_node_id)
        skeleton_folder.is_virtual = True  # Can't delete skeleton folder
        self.add_node(skeleton_folder)

        bone_parents = bone_parents or {}
        created_bones: Dict[str, SceneNode] = {}

        # First pass: create all bone nodes
        for bone_name in bone_names:
            bone_node = SceneNode.create_bone(bone_name, skeleton_folder.id, bone_name)
            created_bones[bone_name] = bone_node

        # Second pass: set up parent relationships
        for bone_name, bone_node in created_bones.items():
            parent_bone_name = bone_parents.get(bone_name)
            if parent_bone_name and parent_bone_name in created_bones:
                bone_node.parent_id = created_bones[parent_bone_name].id
            else:
                bone_node.parent_id = skeleton_folder.id

        # Add all bones to graph
        for bone_node in created_bones.values():
            self.add_node(bone_node)

        return list(created_bones.values())

    def remove_bones(self, radiance_node_id: str):
        """Remove all bone nodes under a radiance component."""
        radiance_node = self._nodes.get(radiance_node_id)
        if not radiance_node:
            return

        # Find and remove skeleton folder
        for child_id in list(radiance_node.children_ids):
            child = self._nodes.get(child_id)
            if child and child.name == "skeleton" and child.is_virtual:
                # Remove all descendants (bones)
                self._remove_subtree(child_id)
                break

    def _remove_subtree(self, node_id: str):
        """Remove a node and all its descendants (even virtual ones)."""
        node = self._nodes.get(node_id)
        if not node:
            return

        for child_id in list(node.children_ids):
            self._remove_subtree(child_id)

        if node.parent_id:
            parent = self._nodes.get(node.parent_id)
            if parent and node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
        else:
            if node_id in self._root_ids:
                self._root_ids.remove(node_id)

        del self._nodes[node_id]

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire graph to a dictionary."""
        # Only serialize non-virtual nodes (bones are regenerated on load)
        nodes_data = []
        for node in self._nodes.values():
            if not node.is_virtual:
                nodes_data.append(node.to_dict())

        return {
            'version': 1,
            'root_order': [rid for rid in self._root_ids
                          if rid in self._nodes and not self._nodes[rid].is_virtual],
            'nodes': nodes_data
        }

    def from_dict(self, data: Dict[str, Any]):
        """Load graph from dictionary."""
        self.clear()

        version = data.get('version', 1)
        nodes_data = data.get('nodes', [])

        # First pass: create all nodes
        for node_data in nodes_data:
            node = SceneNode.from_dict(node_data)
            self._nodes[node.id] = node

        # Set root order
        self._root_ids = data.get('root_order', [])

        # Validate root_ids
        self._root_ids = [rid for rid in self._root_ids if rid in self._nodes]

        # Add any root nodes not in root_order
        for node in self._nodes.values():
            if not node.parent_id and node.id not in self._root_ids:
                self._root_ids.append(node.id)

        self._dirty = False

    def save(self, file_path: str = None) -> bool:
        """Save hierarchy to YAML file."""
        if not YAML_AVAILABLE:
            print("Warning: PyYAML not available, cannot save hierarchy")
            return False

        path = file_path or self._file_path
        if not path:
            return False

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

            self._file_path = path
            self._dirty = False

            if HAS_PYQT:
                self.graphSaved.emit()

            return True
        except Exception as e:
            print(f"Error saving hierarchy: {e}")
            return False

    def load(self, file_path: str) -> bool:
        """Load hierarchy from YAML file."""
        if not YAML_AVAILABLE:
            print("Warning: PyYAML not available, cannot load hierarchy")
            return False

        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)

            self.from_dict(data)
            self._file_path = file_path

            if HAS_PYQT:
                self.graphLoaded.emit()

            return True
        except Exception as e:
            print(f"Error loading hierarchy: {e}")
            return False

    # -------------------------------------------------------------------------
    # Traversal
    # -------------------------------------------------------------------------

    def traverse(self, callback: Callable[[SceneNode, int], None],
                 start_id: str = None):
        """
        Traverse the tree depth-first, calling callback(node, depth) for each.

        If start_id is None, traverses from all roots.
        """
        if start_id:
            node = self._nodes.get(start_id)
            if node:
                self._traverse_node(node, 0, callback)
        else:
            for rid in self._root_ids:
                node = self._nodes.get(rid)
                if node:
                    self._traverse_node(node, 0, callback)

    def _traverse_node(self, node: SceneNode, depth: int,
                       callback: Callable[[SceneNode, int], None]):
        callback(node, depth)
        for child_id in node.children_ids:
            child = self._nodes.get(child_id)
            if child:
                self._traverse_node(child, depth + 1, callback)

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_unique_name(self, base_name: str, parent_id: Optional[str] = None) -> str:
        """Generate a unique name for a new node under parent."""
        existing_names: Set[str] = set()

        if parent_id:
            parent = self._nodes.get(parent_id)
            if parent:
                for cid in parent.children_ids:
                    child = self._nodes.get(cid)
                    if child:
                        existing_names.add(child.name)
        else:
            for rid in self._root_ids:
                node = self._nodes.get(rid)
                if node:
                    existing_names.add(node.name)

        if base_name not in existing_names:
            return base_name

        # Try adding numbers
        counter = 1
        while f"{base_name} ({counter})" in existing_names:
            counter += 1

        return f"{base_name} ({counter})"

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes
