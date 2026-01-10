# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Prim Import/Export - USD-Compatible Scene Interchange
#
#   3D worlds need to share objects between different tools.
#   This module handles importing and exporting "prims" (primitive
#   objects like furniture, props, and scene elements) using a
#   USD-augmented format. USD (Universal Scene Description) is the
#   same format used by Pixar and major studios. Export a room
#   from Blender, import it here, and Noodlings can interact with
#   every object - complete with semantic physics and permissions.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.prim_import_export
# PURPOSE:  USD-compatible import/export for scene objects
# LAYER:    Backend / World Management
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PrimExporter          Export prims to USD-augmented .prim files
#   PrimImporter          Import prims from USD-augmented .prim files
#
# KEY FUNCTIONS:
#   export_third_prim()   Preserve the sacred Third Prim Ever
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Prim Import/Export with USD Compatibility

Implements USD-augmented format for noodleMUSH prims.
Uses Universal Scene Description (USD) as base format with custom schemas for:
- Noodling agents
- Semantic physics (POD)
- noodleMUSH-specific metadata

Compatible with Maya, Houdini, and other USD tools.
"""

import json
import os
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from world import World
from physics_object_descriptor import PhysicsObjectDescriptor
from permissions import PermissionSet, EntityMetadata, Permission

logger = logging.getLogger(__name__)


class PrimExporter:
    """
    Export prims to USD-augmented .prim format.

    Format uses USD-compatible JSON representation with custom schemas
    for Noodling-specific features.
    """

    def __init__(self, world: World):
        """
        Initialize exporter.

        Args:
            world: World instance
        """
        self.world = world

    def export_prim(self, obj_id: str, file_path: str) -> bool:
        """
        Export prim to .prim file (USD-augmented JSON).

        Args:
            obj_id: Object ID to export
            file_path: Destination file path

        Returns:
            True if successful
        """
        obj = self.world.get_object(obj_id)
        if not obj:
            logger.error(f"Object not found: {obj_id}")
            return False

        try:
            # Build USD-compatible prim data
            prim_data = self._build_prim_data(obj)

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(prim_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {obj_id} to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed for {obj_id}: {e}")
            return False

    def _build_prim_data(self, obj: Dict) -> Dict[str, Any]:
        """
        Build USD-augmented prim data structure.

        Args:
            obj: Object dictionary from world state

        Returns:
            USD-compatible prim data
        """
        # Extract metadata (public only)
        # Handle old objects without metadata field
        obj_id = obj.get('uid', 'unknown')  # Legacy objects may not have uid
        metadata = self.world.get_entity_metadata(obj_id) if obj_id != 'unknown' else None

        if metadata:
            export_metadata = {
                'creator': metadata.creator,
                'created_at': metadata.created_at,
                'export_date': datetime.now().isoformat(),
                'tags': metadata.tags,
                'notes': metadata.notes
            }
        else:
            # Legacy object - use available fields
            export_metadata = {
                'creator': obj.get('owner', 'unknown'),
                'created_at': obj.get('created', datetime.now().isoformat()),
                'export_date': datetime.now().isoformat(),
                'tags': [],
                'notes': 'Legacy prim exported without full metadata'
            }

        # Extract POD if present
        physics_data = None
        if obj.get('pod'):
            try:
                pod = PhysicsObjectDescriptor.from_dict(obj['pod'])
                physics_data = {
                    'mass': pod.mass,
                    'friction': pod.friction,
                    'velocity': pod.velocity,
                    'elasticity': pod.elasticity,
                    'softness': pod.softness,
                    'material': pod.material,
                    'semantic_properties': pod.semantic_properties,
                    'state': pod.state,
                    'tags': list(pod.tags),
                    'metadata': pod.metadata
                }
            except Exception as e:
                logger.warning(f"Could not extract POD: {e}")

        # Extract script if present
        script_data = None
        if obj.get('script') and obj['script'].get('name'):
            script_data = {
                'name': obj['script']['name'],
                'code': obj['script'].get('code', ''),
                'state': obj['script'].get('state', {})
            }

        # Extract permissions (handle old objects gracefully)
        permissions_data = None
        if metadata and hasattr(metadata, 'permissions') and metadata.permissions:
            try:
                permissions_data = {
                    'base': metadata.permissions.base.value,
                    'next_owner': metadata.permissions.next_owner.value,
                    'group': metadata.permissions.group.value,
                    'everyone': metadata.permissions.everyone.value,
                    'allow_copy_by_others': metadata.permissions.allow_copy_by_others,
                    'locked': metadata.permissions.locked
                }
            except Exception as e:
                logger.warning(f"Could not extract permissions: {e}")

        # Build USD-augmented structure
        return {
            # USD standard fields
            'usd_version': '0.10.5',  # USD format version
            'format_version': '1.0',  # Noodling augmentation version

            # USD prim definition
            'def': 'Xform',  # USD prim type (transformation node)
            'name': obj['name'],
            'typeName': obj.get('type', 'prop'),

            # USD metadata
            'customData': {
                'description': obj.get('description', ''),
                'properties': obj.get('properties', {}),
                'noodlemush': export_metadata
            },

            # USD custom attributes (Noodling extensions)
            'customAttributes': {
                # Noodling semantic physics
                'noodling:physics': physics_data,

                # Noodling script
                'noodling:script': script_data,

                # Noodling permissions
                'noodling:permissions': permissions_data,

                # Custom user data
                'noodling:customData': obj.get('custom_data', {})
            },

            # USD transform (identity by default - Studio will set)
            'xformOp:translate': [0.0, 0.0, 0.0],
            'xformOp:rotateXYZ': [0.0, 0.0, 0.0],
            'xformOp:scale': [1.0, 1.0, 1.0],

            # USD material binding (placeholder)
            'material:binding': None
        }


class PrimImporter:
    """
    Import prims from USD-augmented .prim files.

    Handles both:
    - Pure USD files (imports geometry/transforms only)
    - USD-augmented files (imports Noodling extensions too)
    """

    def __init__(self, world: World):
        """
        Initialize importer.

        Args:
            world: World instance
        """
        self.world = world

    def import_prim(
        self,
        file_path: str,
        room_id: str,
        importer_user: str
    ) -> Optional[str]:
        """
        Import prim from .prim file.

        Args:
            file_path: Path to .prim file
            room_id: Room to spawn in
            importer_user: User importing the prim

        Returns:
            New object ID if successful, None otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                prim_data = json.load(f)

            # Validate format
            if not self._validate_prim_data(prim_data):
                logger.error(f"Invalid prim data in {file_path}")
                return None

            # Import prim
            obj_id = self._create_prim_from_data(
                prim_data,
                room_id,
                importer_user
            )

            if obj_id:
                logger.info(f"Imported {file_path} as {obj_id}")
            else:
                logger.error(f"Failed to import {file_path}")

            return obj_id

        except Exception as e:
            logger.error(f"Import failed for {file_path}: {e}")
            return None

    def _validate_prim_data(self, data: Dict) -> bool:
        """
        Validate prim data structure.

        Args:
            data: Prim data from file

        Returns:
            True if valid
        """
        # Check required USD fields
        if 'name' not in data:
            logger.error("Missing required field: name")
            return False

        if 'def' not in data:
            logger.warning("Missing USD 'def' field - assuming Xform")

        # Check format version compatibility
        format_version = data.get('format_version', '1.0')
        if not self._is_compatible_version(format_version):
            logger.error(f"Incompatible format version: {format_version}")
            return False

        # Validate string lengths
        if len(data['name']) > 256:
            logger.error("Name too long (max 256 characters)")
            return False

        description = data.get('customData', {}).get('description', '')
        if len(description) > 4096:
            logger.error("Description too long (max 4096 characters)")
            return False

        return True

    def _is_compatible_version(self, version: str) -> bool:
        """Check if format version is compatible."""
        # Version 1.x are all compatible
        return version.startswith('1.')

    def _resolve_name_collision(self, base_name: str) -> str:
        """
        Resolve name collision by adding (N) suffix.

        Args:
            base_name: Desired name

        Returns:
            Unique name (possibly with (N) suffix)
        """
        # Check if name exists
        existing_names = set()
        for obj_id, obj in self.world.objects.items():
            existing_names.add(obj['name'])

        # If no collision, use base name
        if base_name not in existing_names:
            return base_name

        # Find next available number
        n = 1
        while f"{base_name} ({n})" in existing_names:
            n += 1

        return f"{base_name} ({n})"

    def _create_prim_from_data(
        self,
        data: Dict,
        room_id: str,
        importer_user: str
    ) -> Optional[str]:
        """
        Create prim from imported data.

        Args:
            data: Prim data from file
            room_id: Room to spawn in
            importer_user: User importing

        Returns:
            New object ID
        """
        # Extract fields
        base_name = data['name']
        custom_data = data.get('customData', {})
        description = custom_data.get('description', '')
        properties = custom_data.get('properties', {})
        noodlemush_meta = custom_data.get('noodlemush', {})

        # Handle name collision (add (1), (2), etc.)
        name = self._resolve_name_collision(base_name)

        custom_attrs = data.get('customAttributes', {})

        # Import POD if present
        pod = None
        physics_data = custom_attrs.get('noodling:physics')
        if physics_data:
            pod = PhysicsObjectDescriptor(
                mass=physics_data.get('mass', 'medium'),
                friction=physics_data.get('friction', 'medium'),
                velocity=physics_data.get('velocity', 'stationary'),
                elasticity=physics_data.get('elasticity', 'normal'),
                softness=physics_data.get('softness', 'normal'),
                material=physics_data.get('material', 'unknown'),
                semantic_properties=physics_data.get('semantic_properties', []),
                state=physics_data.get('state', 'normal'),
                metadata=physics_data.get('metadata', {}),
                tags=physics_data.get('tags', [])
            )

        # Import permissions
        permissions = None
        perms_data = custom_attrs.get('noodling:permissions')
        if perms_data:
            permissions = PermissionSet(
                # Use next_owner permissions for imported copy
                base=Permission(perms_data.get('next_owner', Permission.OWNER_DEFAULT.value)),
                next_owner=Permission(perms_data.get('next_owner', Permission.OWNER_DEFAULT.value)),
                group=Permission(perms_data.get('group', Permission.NONE.value)),
                everyone=Permission(perms_data.get('everyone', Permission.NONE.value)),
                allow_copy_by_others=perms_data.get('allow_copy_by_others', False),
                locked=perms_data.get('locked', False)
            )

        # Create object
        obj_id = self.world.create_object(
            name=name,
            description=description,
            owner=importer_user,
            location=room_id,
            portable=properties.get('portable', True),
            takeable=properties.get('takeable', True),
            obj_type=data.get('typeName', 'prop'),
            spawned_by=importer_user,
            pod=pod,
            permissions=permissions
        )

        # Preserve creator from file
        metadata = self.world.get_entity_metadata(obj_id)
        if metadata and noodlemush_meta.get('creator'):
            metadata.creator = noodlemush_meta['creator']
            # Add import note
            metadata.notes = f"Imported from {os.path.basename(file_path)} on {datetime.now().isoformat()}"
            if noodlemush_meta.get('notes'):
                metadata.notes += f"\n\nOriginal notes: {noodlemush_meta['notes']}"
            self.world.update_entity_metadata(obj_id, metadata)

        # Attach script if present
        script_data = custom_attrs.get('noodling:script')
        if script_data and script_data.get('name'):
            obj = self.world.get_object(obj_id)
            if obj:
                obj['script'] = {
                    'name': script_data['name'],
                    'code': script_data.get('code'),
                    'state': script_data.get('state', {}),
                    'version': 1,
                    'compiled': False  # Will be compiled on server start
                }
                self.world.save_all()

        # Store custom data
        custom_user_data = custom_attrs.get('noodling:customData', {})
        if custom_user_data:
            obj = self.world.get_object(obj_id)
            if obj:
                obj['custom_data'] = custom_user_data
                self.world.save_all()

        return obj_id


# ===== Helper Functions =====

def export_third_prim(world: World, output_path: str = "third_prim_ever.prim") -> bool:
    """
    Export the sacred Third Prim Ever.

    Lieutenant Caitlyn lost the first and second prims in a devastating QA session.
    This function ensures the third prim is preserved for eternity.

    Args:
        world: World instance
        output_path: Where to save the file

    Returns:
        True if successful
    """
    # Find "Third Prim Ever" or "THIRD PRIM EVER"
    third_prim_id = None

    for obj_id, obj in world.objects.items():
        name_lower = obj['name'].lower()
        if 'third' in name_lower and 'prim' in name_lower and 'ever' in name_lower:
            third_prim_id = obj_id
            break

    if not third_prim_id:
        logger.error("Third Prim Ever not found!")
        return False

    # Export it
    exporter = PrimExporter(world)
    success = exporter.export_prim(third_prim_id, output_path)

    if success:
        logger.info(f" Third Prim Ever preserved to {output_path}")
        logger.info("   The first and second prims were lost, but the third shall endure.")
    else:
        logger.error(" Failed to preserve Third Prim Ever")

    return success


# ===== Testing =====

if __name__ == '__main__':
    # Test export/import
    from world import World

    world = World("world")

    # Create test prim
    test_pod = PhysicsObjectDescriptor(
        mass="negligible",
        material="pure information",
        semantic_properties=["sacred", "historic"]
    )

    obj_id = world.create_object(
        name="Test Prim",
        description="A test object for export",
        owner="user_test",
        location="room_000",
        spawned_by="user_test",
        pod=test_pod
    )

    print(f"Created test prim: {obj_id}")

    # Export
    exporter = PrimExporter(world)
    success = exporter.export_prim(obj_id, "test_export.prim")
    print(f"Export: {'' if success else '✗'}")

    # Import
    importer = PrimImporter(world)
    imported_id = importer.import_prim("test_export.prim", "room_000", "user_test2")
    print(f"Import: {'' if imported_id else '✗'}")

    if imported_id:
        imported_obj = world.get_object(imported_id)
        print(f"Imported as: {imported_id}")
        print(f"Name: {imported_obj['name']}")
        print(f"Owner: {imported_obj['owner']}")
        print(f"Has POD: {imported_obj.get('pod') is not None}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
