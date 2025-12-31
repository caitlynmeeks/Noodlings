"""
Scene Node - Unity-style scene hierarchy data model

A SceneNode represents any entity in the scene hierarchy:
- Folders (user-created organization)
- Radiances (Gaussian splat components)
- Noodlings (AI characters)
- Props (world objects)
- Bones (virtual, from skeleton)
- Zones (spatial regions)

Each node has a parent and ordered children, enabling arbitrary
hierarchies like Unity's GameObject tree.

Author: Commander Spock + Cadet Caity
Date: December 27, 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import uuid


class SceneNodeType(Enum):
    """Type of scene node."""
    FOLDER = "folder"          # User-created folder for organization
    RADIANCE = "radiance"      # RadianceComponent (Gaussian splat)
    NOODLING = "noodling"      # AI character
    PROP = "prop"              # World object
    BONE = "bone"              # Virtual node from skeleton
    ZONE = "zone"              # Spatial region


@dataclass
class SceneNode:
    """
    A node in the scene hierarchy.

    Nodes form a tree structure with parent/children relationships.
    Each node references an asset (by path) or is a pure organizational
    element (folder).

    Bone nodes are special: they are virtual (auto-populated from skeleton
    data) and cannot be deleted or renamed by the user.
    """
    id: str                                # UUID
    name: str                              # Display name
    node_type: SceneNodeType               # Type of node
    parent_id: Optional[str] = None        # None = root level
    children_ids: List[str] = field(default_factory=list)  # Ordered children

    # Asset reference (for non-folder types)
    asset_path: Optional[str] = None       # Path to asset file (relative to project)

    # For BONE type - name in the skeleton
    bone_name: Optional[str] = None

    # Local transform (relative to parent)
    # Used when parenting props to bones
    local_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Euler degrees
    local_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Flags
    is_virtual: bool = False               # True for bone nodes (can't delete)
    is_expanded: bool = True               # UI expansion state
    is_visible: bool = True                # Visibility in viewport
    is_locked: bool = False                # Prevent modifications

    # Metadata (for extensibility)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(name: str, node_type: SceneNodeType,
               parent_id: Optional[str] = None,
               asset_path: Optional[str] = None) -> 'SceneNode':
        """Factory method to create a new node with generated UUID."""
        return SceneNode(
            id=str(uuid.uuid4()),
            name=name,
            node_type=node_type,
            parent_id=parent_id,
            asset_path=asset_path
        )

    @staticmethod
    def create_folder(name: str, parent_id: Optional[str] = None) -> 'SceneNode':
        """Convenience method to create a folder node."""
        return SceneNode.create(name, SceneNodeType.FOLDER, parent_id)

    @staticmethod
    def create_bone(name: str, parent_id: str, bone_name: str) -> 'SceneNode':
        """Create a virtual bone node."""
        node = SceneNode.create(name, SceneNodeType.BONE, parent_id)
        node.bone_name = bone_name
        node.is_virtual = True
        return node

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML storage."""
        data = {
            'id': self.id,
            'name': self.name,
            'type': self.node_type.value,
        }

        # Only include non-default values to keep YAML clean
        if self.parent_id:
            data['parent_id'] = self.parent_id
        if self.children_ids:
            data['children'] = self.children_ids
        if self.asset_path:
            data['asset_path'] = self.asset_path
        if self.bone_name:
            data['bone_name'] = self.bone_name

        # Transform (only if non-identity)
        if self.local_position != (0.0, 0.0, 0.0):
            data['position'] = list(self.local_position)
        if self.local_rotation != (0.0, 0.0, 0.0):
            data['rotation'] = list(self.local_rotation)
        if self.local_scale != (1.0, 1.0, 1.0):
            data['scale'] = list(self.local_scale)

        # Flags (only if non-default)
        if self.is_virtual:
            data['virtual'] = True
        if not self.is_expanded:
            data['expanded'] = False
        if not self.is_visible:
            data['visible'] = False
        if self.is_locked:
            data['locked'] = True

        if self.metadata:
            data['metadata'] = self.metadata

        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SceneNode':
        """Deserialize from dictionary."""
        node = SceneNode(
            id=data['id'],
            name=data['name'],
            node_type=SceneNodeType(data['type']),
            parent_id=data.get('parent_id'),
            children_ids=data.get('children', []),
            asset_path=data.get('asset_path'),
            bone_name=data.get('bone_name'),
        )

        # Transform
        if 'position' in data:
            node.local_position = tuple(data['position'])
        if 'rotation' in data:
            node.local_rotation = tuple(data['rotation'])
        if 'scale' in data:
            node.local_scale = tuple(data['scale'])

        # Flags
        node.is_virtual = data.get('virtual', False)
        node.is_expanded = data.get('expanded', True)
        node.is_visible = data.get('visible', True)
        node.is_locked = data.get('locked', False)
        node.metadata = data.get('metadata', {})

        return node

    def __repr__(self) -> str:
        return f"SceneNode({self.name!r}, {self.node_type.value})"
