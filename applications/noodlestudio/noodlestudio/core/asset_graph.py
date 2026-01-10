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
#   Asset Graph - Manages the asset folder hierarchy.
#
#   Provides: - CRUD operations for asset nodes - Reparenting...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.asset_graph
# PURPOSE:  Asset Graph - Manages the asset folder hierarchy.
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AssetGraph
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal
import yaml
import os

from .asset_node import AssetNode, AssetNodeType


class AssetGraph(QObject):
    """
    Manages the asset folder hierarchy.

    The graph maintains a flat dict of nodes by ID, with parent/children
    relationships encoded in the nodes themselves.
    """

    # Signals
    nodeAdded = pyqtSignal(str)           # node_id
    nodeRemoved = pyqtSignal(str)         # node_id
    nodeReparented = pyqtSignal(str, str, str)  # node_id, old_parent, new_parent
    nodeRenamed = pyqtSignal(str, str)    # node_id, new_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes: Dict[str, AssetNode] = {}
        self.root_ids: List[str] = []  # IDs of top-level nodes

    def clear(self):
        """Clear all nodes."""
        self.nodes.clear()
        self.root_ids.clear()

    def get_node(self, node_id: str) -> Optional[AssetNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def create_node(self, name: str, node_type: AssetNodeType,
                    parent_id: Optional[str] = None,
                    asset_path: Optional[str] = None) -> AssetNode:
        """Create a new node and add it to the graph."""
        node = AssetNode.create(name, node_type, parent_id, asset_path)
        self._add_node(node)
        return node

    def create_folder(self, name: str, parent_id: Optional[str] = None) -> AssetNode:
        """Create a new folder node."""
        return self.create_node(name, AssetNodeType.FOLDER, parent_id)

    def _add_node(self, node: AssetNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node

        # Update parent's children list
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent and node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
        else:
            if node.id not in self.root_ids:
                self.root_ids.append(node.id)

        self.nodeAdded.emit(node.id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its children from the graph."""
        node = self.nodes.get(node_id)
        if not node:
            return False

        # Remove children first (recursive)
        for child_id in list(node.children_ids):
            self.remove_node(child_id)

        # Remove from parent's children list
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent and node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
        else:
            if node_id in self.root_ids:
                self.root_ids.remove(node_id)

        # Remove node
        del self.nodes[node_id]
        self.nodeRemoved.emit(node_id)
        return True

    def reparent(self, node_id: str, new_parent_id: Optional[str]) -> bool:
        """Move a node to a new parent (or root if new_parent_id is None)."""
        node = self.nodes.get(node_id)
        if not node:
            return False

        # Can't reparent to self or descendant
        if new_parent_id:
            if new_parent_id == node_id:
                return False
            if self._is_descendant(new_parent_id, node_id):
                return False

        old_parent_id = node.parent_id

        # Remove from old parent
        if old_parent_id:
            old_parent = self.nodes.get(old_parent_id)
            if old_parent and node_id in old_parent.children_ids:
                old_parent.children_ids.remove(node_id)
        else:
            if node_id in self.root_ids:
                self.root_ids.remove(node_id)

        # Add to new parent
        node.parent_id = new_parent_id
        if new_parent_id:
            new_parent = self.nodes.get(new_parent_id)
            if new_parent and node_id not in new_parent.children_ids:
                new_parent.children_ids.append(node_id)
        else:
            if node_id not in self.root_ids:
                self.root_ids.append(node_id)

        self.nodeReparented.emit(node_id, old_parent_id or '', new_parent_id or '')
        return True

    def rename_node(self, node_id: str, new_name: str) -> bool:
        """Rename a node."""
        node = self.nodes.get(node_id)
        if not node:
            return False

        node.name = new_name
        self.nodeRenamed.emit(node_id, new_name)
        return True

    def _is_descendant(self, node_id: str, potential_ancestor_id: str) -> bool:
        """Check if node_id is a descendant of potential_ancestor_id."""
        node = self.nodes.get(node_id)
        while node and node.parent_id:
            if node.parent_id == potential_ancestor_id:
                return True
            node = self.nodes.get(node.parent_id)
        return False

    def find_by_path(self, asset_path: str) -> Optional[AssetNode]:
        """Find a node by its asset path."""
        for node in self.nodes.values():
            if node.asset_path == asset_path:
                return node
        return None

    def get_children(self, node_id: Optional[str]) -> List[AssetNode]:
        """Get children of a node (or root nodes if node_id is None)."""
        if node_id is None:
            return [self.nodes[nid] for nid in self.root_ids if nid in self.nodes]
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def save(self, path: str):
        """Save the graph to a YAML file."""
        data = {
            'version': 1,
            'root_ids': self.root_ids,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()}
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load(self, path: str) -> bool:
        """Load the graph from a YAML file."""
        if not os.path.exists(path):
            return False

        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}

            self.clear()

            # Load nodes
            nodes_data = data.get('nodes', {})
            for node_id, node_data in nodes_data.items():
                node = AssetNode.from_dict(node_data)
                self.nodes[node_id] = node

            # Load root_ids
            self.root_ids = data.get('root_ids', [])

            return True

        except Exception as e:
            print(f"[AssetGraph] Error loading {path}: {e}")
            return False

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
