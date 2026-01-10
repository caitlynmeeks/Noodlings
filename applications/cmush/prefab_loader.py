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
#   Prefab Loader - Character Template Management
#
#   Think of prefabs like cookie cutters for characters. Instead
#   of building each Noodling from scratch, you define a template
#   (prefab) that describes personality, cognitive settings, and
#   identity. This module loads those YAML templates and creates
#   new Noodlings from them. You can duplicate, modify, import,
#   and export prefabs - sharing characters between projects.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.prefab_loader
# PURPOSE:  Load, save, and manage character template files
# LAYER:    Backend / World Management
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PrefabLoader          Manages .prefab files (YAML templates)
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Prefab Loader - Character template management system.

Loads, saves, and manages .prefab files (character templates).
Prefabs define initial transistor configurations, instruction prompts,
and character settings.
"""

import yaml
import os
import shutil
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PrefabLoader:
    """
    Manages character prefab files (.prefab format).

    Prefabs are templates for spawning Noodlings with pre-configured
    cognitive transistors, instruction prompts, and personality settings.
    """

    def __init__(self, prefabs_dir: str = "prefabs"):
        """
        Initialize prefab loader.

        Args:
            prefabs_dir: Directory containing .prefab files
        """
        self.prefabs_dir = Path(prefabs_dir)
        self.prefabs_dir.mkdir(exist_ok=True)

        # Cache: id -> prefab data
        self._cache = {}
        self._cache_valid = False

    def _invalidate_cache(self):
        """Invalidate cache - force reload on next access."""
        self._cache_valid = False

    def _load_cache(self):
        """Load all prefabs into cache."""
        if self._cache_valid:
            return

        self._cache.clear()

        for filepath in self.prefabs_dir.glob("*.prefab"):
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)

                prefab_id = data.get('metadata', {}).get('id')
                if not prefab_id:
                    logger.warning(f"Prefab {filepath} missing metadata.id, skipping")
                    continue

                self._cache[prefab_id] = {
                    'data': data,
                    'filepath': filepath
                }

            except Exception as e:
                logger.error(f"Failed to load prefab {filepath}: {e}")

        self._cache_valid = True
        logger.info(f"Loaded {len(self._cache)} prefabs from {self.prefabs_dir}")

    def load(self, prefab_id: str) -> Optional[Dict]:
        """
        Load prefab by unique ID.

        Args:
            prefab_id: Unique identifier (e.g., "com.noodlings.characters.red_fire_anklebiter")

        Returns:
            Prefab data dict or None if not found
        """
        self._load_cache()

        cached = self._cache.get(prefab_id)
        if cached:
            return cached['data'].copy()

        # Try loading by filename if not found by ID
        filename = f"{prefab_id}.prefab"
        filepath = self.prefabs_dir / filename

        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
                self._invalidate_cache()  # Refresh cache
                return data
            except Exception as e:
                logger.error(f"Failed to load prefab {filepath}: {e}")
                return None

        return None

    def save(self, prefab_id: str, data: Dict):
        """
        Save prefab to disk.

        Args:
            prefab_id: Unique identifier
            data: Prefab data dict

        Updates modified timestamp automatically.
        """
        # Update metadata
        if 'metadata' not in data:
            data['metadata'] = {}

        data['metadata']['id'] = prefab_id
        data['metadata']['modified'] = datetime.now().strftime('%Y-%m-%d')

        # Generate UUID if not present
        if 'uuid' not in data['metadata']:
            data['metadata']['uuid'] = str(uuid.uuid4())

        # Save to file
        filename = f"{prefab_id}.prefab"
        filepath = self.prefabs_dir / filename

        try:
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            logger.info(f"Saved prefab: {prefab_id}")
            self._invalidate_cache()

        except Exception as e:
            logger.error(f"Failed to save prefab {prefab_id}: {e}")
            raise

    def list_all(self) -> List[Dict]:
        """
        List all available prefabs.

        Returns:
            List of prefab metadata dicts
        """
        self._load_cache()

        results = []
        for prefab_id, cached in self._cache.items():
            metadata = cached['data'].get('metadata', {})
            results.append({
                'id': prefab_id,
                'name': metadata.get('name', prefab_id),
                'version': metadata.get('version', '1.0.0'),
                'description': metadata.get('description', ''),
                'tags': metadata.get('tags', []),
                'modified': metadata.get('modified', 'unknown')
            })

        return sorted(results, key=lambda x: x['name'])

    def duplicate(self, source_id: str, new_name: str, new_id: Optional[str] = None) -> str:
        """
        Duplicate prefab with new name and ID.

        Args:
            source_id: Source prefab ID
            new_name: Display name for duplicate
            new_id: Optional custom ID (auto-generated if None)

        Returns:
            New prefab ID

        Example:
            new_id = loader.duplicate(
                'com.noodlings.characters.red_fire_anklebiter',
                'Purple Fire Anklebiter'
            )
            # Returns: 'com.noodlings.characters.purple_fire_anklebiter'
        """
        source_data = self.load(source_id)
        if not source_data:
            raise ValueError(f"Source prefab not found: {source_id}")

        # Generate new ID if not provided
        if not new_id:
            # Convert name to ID format
            name_slug = new_name.lower().replace(' ', '_')
            # Keep same category as source
            source_parts = source_id.split('.')
            if len(source_parts) >= 3:
                category = '.'.join(source_parts[:-1])
                new_id = f"{category}.{name_slug}"
            else:
                new_id = f"com.noodlings.characters.{name_slug}"

        # Create duplicate
        dup_data = source_data.copy()
        dup_data['metadata'] = source_data.get('metadata', {}).copy()
        dup_data['metadata']['id'] = new_id
        dup_data['metadata']['name'] = new_name
        dup_data['metadata']['created'] = datetime.now().strftime('%Y-%m-%d')
        dup_data['metadata']['modified'] = datetime.now().strftime('%Y-%m-%d')
        dup_data['metadata']['version'] = '1.0.0'

        # Update character name if exists
        if 'character' in dup_data:
            pass  # Don't auto-update character fields - let user edit

        self.save(new_id, dup_data)
        return new_id

    def delete(self, prefab_id: str):
        """
        Delete prefab file.

        Args:
            prefab_id: Prefab to delete

        Moves to trash (.deleted) rather than permanent deletion.
        """
        cached = self._cache.get(prefab_id)
        if not cached:
            # Try finding by filename
            filename = f"{prefab_id}.prefab"
            filepath = self.prefabs_dir / filename
        else:
            filepath = cached['filepath']

        if not filepath.exists():
            raise FileNotFoundError(f"Prefab not found: {prefab_id}")

        # Move to deleted folder instead of permanent delete
        deleted_dir = self.prefabs_dir / ".deleted"
        deleted_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{filepath.stem}_{timestamp}.prefab"
        backup_path = deleted_dir / backup_name

        shutil.move(str(filepath), str(backup_path))
        logger.info(f"Deleted prefab {prefab_id} (moved to {backup_path})")

        self._invalidate_cache()

    def export_prefab(self, prefab_id: str, dest_path: str):
        """
        Export prefab to external file.

        Args:
            prefab_id: Prefab to export
            dest_path: Destination file path

        Creates standalone .prefab file for sharing.
        """
        data = self.load(prefab_id)
        if not data:
            raise ValueError(f"Prefab not found: {prefab_id}")

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Exported prefab {prefab_id} to {dest_path}")

    def import_prefab(self, source_path: str, new_id: Optional[str] = None) -> str:
        """
        Import prefab from external file.

        Args:
            source_path: Path to .prefab file
            new_id: Optional new ID (auto-generated if None)

        Returns:
            Imported prefab ID
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        with open(source, 'r') as f:
            data = yaml.safe_load(f)

        # Use provided ID or generate from file
        if new_id:
            prefab_id = new_id
        else:
            # Try to get ID from metadata
            prefab_id = data.get('metadata', {}).get('id')
            if not prefab_id:
                # Generate from filename
                name_slug = source.stem
                prefab_id = f"user.imported.{name_slug}"

        # Check for conflicts
        existing = self.load(prefab_id)
        if existing:
            # Add timestamp to avoid collision
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefab_id = f"{prefab_id}_{timestamp}"

        self.save(prefab_id, data)
        logger.info(f"Imported prefab as {prefab_id}")

        return prefab_id

    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate prefab data structure.

        Args:
            data: Prefab data dict

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Required: metadata
        if 'metadata' not in data:
            errors.append("Missing 'metadata' section")
        else:
            metadata = data['metadata']
            if 'id' not in metadata:
                errors.append("Missing metadata.id")
            if 'name' not in metadata:
                errors.append("Missing metadata.name")

        # Required: character
        if 'character' not in data:
            errors.append("Missing 'character' section")
        else:
            char = data['character']
            if 'species' not in char:
                errors.append("Missing character.species")
            if 'identity_prompt' not in char:
                errors.append("Missing character.identity_prompt")

        # Validate cognitive_components if present
        if 'cognitive_components' in data:
            components = data['cognitive_components']
            for comp_name, comp_config in components.items():
                if 'type' not in comp_config:
                    errors.append(f"Component '{comp_name}' missing 'type'")
                if 'salience' in comp_config:
                    sal = comp_config['salience']
                    if not (0.0 <= sal <= 1.0):
                        errors.append(f"Component '{comp_name}' salience out of range: {sal}")

        return (len(errors) == 0, errors)


# Example usage and testing
if __name__ == '__main__':
    loader = PrefabLoader("prefabs")

    # List all prefabs
    prefabs = loader.list_all()
    print(f"Found {len(prefabs)} prefabs:")
    for p in prefabs:
        print(f"  {p['id']} - {p['name']} (v{p['version']})")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
