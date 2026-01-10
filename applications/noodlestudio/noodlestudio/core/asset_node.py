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
#   Asset Node - Data model for organizing assets in a folder hierarchy.
#
#   Unlike SceneNode (which represents entities in a scene), ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.asset_node
# PURPOSE:  Asset Node
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AssetNodeType, AssetNode
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import uuid


class AssetNodeType(Enum):
    """Type of asset node."""
    FOLDER = "folder"          # User-created folder for organization
    NOODLING = "noodling"      # AI character definition
    STAGE = "stage"            # Scene/level
    PRIM = "prim"              # 3D object template
    RADIANCE = "radiance"      # Gaussian splat model
    MESH = "mesh"              # Imported mesh
    GENERATION = "generation"  # AI-generated content


@dataclass
class AssetNode:
    """
    A node in the asset hierarchy.

    Nodes form a tree structure for organizing assets.
    Folders are purely organizational - the actual assets
    remain in their original disk locations.
    """
    id: str                                # UUID
    name: str                              # Display name
    node_type: AssetNodeType               # Type of node
    parent_id: Optional[str] = None        # None = root level
    children_ids: List[str] = field(default_factory=list)  # Ordered children

    # Asset reference (for non-folder types)
    asset_path: Optional[str] = None       # Path to asset (relative to project)

    # Additional metadata for specific types
    metadata: Dict[str, Any] = field(default_factory=dict)

    # UI state
    is_expanded: bool = True

    @staticmethod
    def create(name: str, node_type: AssetNodeType,
               parent_id: Optional[str] = None,
               asset_path: Optional[str] = None) -> 'AssetNode':
        """Factory method to create a new node with generated UUID."""
        return AssetNode(
            id=str(uuid.uuid4()),
            name=name,
            node_type=node_type,
            parent_id=parent_id,
            asset_path=asset_path
        )

    @staticmethod
    def create_folder(name: str, parent_id: Optional[str] = None) -> 'AssetNode':
        """Convenience method to create a folder node."""
        return AssetNode.create(name, AssetNodeType.FOLDER, parent_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML storage."""
        data = {
            'id': self.id,
            'name': self.name,
            'type': self.node_type.value,
        }

        # Only include non-default values
        if self.parent_id:
            data['parent_id'] = self.parent_id
        if self.children_ids:
            data['children'] = self.children_ids
        if self.asset_path:
            data['asset_path'] = self.asset_path
        if self.metadata:
            data['metadata'] = self.metadata
        if not self.is_expanded:
            data['expanded'] = False

        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AssetNode':
        """Deserialize from dictionary."""
        node = AssetNode(
            id=data['id'],
            name=data['name'],
            node_type=AssetNodeType(data['type']),
            parent_id=data.get('parent_id'),
            children_ids=data.get('children', []),
            asset_path=data.get('asset_path'),
            metadata=data.get('metadata', {}),
        )
        node.is_expanded = data.get('expanded', True)
        return node

    def __repr__(self) -> str:
        return f"AssetNode({self.name!r}, {self.node_type.value})"

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
