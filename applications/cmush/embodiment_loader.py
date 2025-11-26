"""
Embodiment Loader - Physical body template management system.

Loads, saves, and manages .embodiment files (body templates).
Embodiments define physical structure, characteristics, and mutable state.
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


class EmbodimentLoader:
    """
    Manages embodiment asset files (.embodiment format).

    Embodiments represent the physical body of a Noodling including:
    - Body architecture (quadruped, biped, hovering, disembodied)
    - Physical characteristics (fur, eyes, limbs)
    - Mutable state (injuries, energy, worn items)
    """

    def __init__(self, embodiments_dir: str = "assets/embodiments"):
        """
        Initialize embodiment loader.

        Args:
            embodiments_dir: Directory containing .embodiment files
        """
        self.embodiments_dir = Path(embodiments_dir)
        self.embodiments_dir.mkdir(parents=True, exist_ok=True)

        # Cache: id -> embodiment data
        self._cache = {}
        self._cache_valid = False

    def _invalidate_cache(self):
        """Invalidate cache - force reload on next access."""
        self._cache_valid = False

    def _load_cache(self):
        """Load all embodiments into cache."""
        if self._cache_valid:
            return

        self._cache.clear()

        for filepath in self.embodiments_dir.glob("*.embodiment"):
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)

                embodiment_id = data.get('metadata', {}).get('id')
                if not embodiment_id:
                    logger.warning(f"Embodiment {filepath} missing metadata.id, skipping")
                    continue

                self._cache[embodiment_id] = {
                    'data': data,
                    'filepath': filepath
                }

            except Exception as e:
                logger.error(f"Failed to load embodiment {filepath}: {e}")

        self._cache_valid = True
        logger.info(f"Loaded {len(self._cache)} embodiments from {self.embodiments_dir}")

    def load(self, embodiment_id: str) -> Optional[Dict]:
        """
        Load embodiment by unique ID.

        Args:
            embodiment_id: Unique identifier (e.g., "com.noodlings.embodiments.one_eyed_black_cat")

        Returns:
            Embodiment data dict or None if not found
        """
        self._load_cache()

        cached = self._cache.get(embodiment_id)
        if cached:
            return cached['data'].copy()

        # Try loading by filename if not found by ID
        filename = f"{embodiment_id}.embodiment"
        filepath = self.embodiments_dir / filename

        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
                self._invalidate_cache()  # Refresh cache
                return data
            except Exception as e:
                logger.error(f"Failed to load embodiment {filepath}: {e}")
                return None

        return None

    def save(self, embodiment_id: str, data: Dict):
        """
        Save embodiment to disk.

        Args:
            embodiment_id: Unique identifier
            data: Embodiment data dict

        Updates modified timestamp automatically.
        """
        # Update metadata
        if 'metadata' not in data:
            data['metadata'] = {}

        data['metadata']['id'] = embodiment_id
        data['metadata']['modified'] = datetime.now().strftime('%Y-%m-%d')

        # Generate UUID if not present
        if 'uuid' not in data['metadata']:
            data['metadata']['uuid'] = str(uuid.uuid4())

        # Save to file
        filename = f"{embodiment_id}.embodiment"
        filepath = self.embodiments_dir / filename

        try:
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            logger.info(f"Saved embodiment: {embodiment_id}")
            self._invalidate_cache()

        except Exception as e:
            logger.error(f"Failed to save embodiment {embodiment_id}: {e}")
            raise

    def list_all(self) -> List[Dict]:
        """
        List all available embodiments.

        Returns:
            List of embodiment metadata dicts
        """
        self._load_cache()

        results = []
        for embodiment_id, cached in self._cache.items():
            metadata = cached['data'].get('metadata', {})
            results.append({
                'id': embodiment_id,
                'name': metadata.get('name', embodiment_id),
                'version': metadata.get('version', '1.0.0'),
                'description': metadata.get('description', ''),
                'tags': metadata.get('tags', []),
                'modified': metadata.get('modified', 'unknown')
            })

        return sorted(results, key=lambda x: x['name'])

    def duplicate(self, source_id: str, new_name: str, new_id: Optional[str] = None) -> str:
        """
        Duplicate embodiment with new name and ID.

        Args:
            source_id: Source embodiment ID
            new_name: Display name for duplicate
            new_id: Optional custom ID (auto-generated if None)

        Returns:
            New embodiment ID

        Example:
            new_id = loader.duplicate(
                'com.noodlings.embodiments.one_eyed_black_cat',
                'Two-Eyed Black Cat'
            )
        """
        source_data = self.load(source_id)
        if not source_data:
            raise ValueError(f"Source embodiment not found: {source_id}")

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
                new_id = f"com.noodlings.embodiments.{name_slug}"

        # Create duplicate
        dup_data = source_data.copy()
        dup_data['metadata'] = source_data.get('metadata', {}).copy()
        dup_data['metadata']['id'] = new_id
        dup_data['metadata']['name'] = new_name
        dup_data['metadata']['created'] = datetime.now().strftime('%Y-%m-%d')
        dup_data['metadata']['modified'] = datetime.now().strftime('%Y-%m-%d')
        dup_data['metadata']['version'] = '1.0.0'

        self.save(new_id, dup_data)
        return new_id

    def delete(self, embodiment_id: str):
        """
        Delete embodiment file.

        Args:
            embodiment_id: Embodiment to delete

        Moves to trash (.deleted) rather than permanent deletion.
        """
        cached = self._cache.get(embodiment_id)
        if not cached:
            # Try finding by filename
            filename = f"{embodiment_id}.embodiment"
            filepath = self.embodiments_dir / filename
        else:
            filepath = cached['filepath']

        if not filepath.exists():
            raise FileNotFoundError(f"Embodiment not found: {embodiment_id}")

        # Move to deleted folder instead of permanent delete
        deleted_dir = self.embodiments_dir / ".deleted"
        deleted_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{filepath.stem}_{timestamp}.embodiment"
        backup_path = deleted_dir / backup_name

        shutil.move(str(filepath), str(backup_path))
        logger.info(f"Deleted embodiment {embodiment_id} (moved to {backup_path})")

        self._invalidate_cache()

    def export_embodiment(self, embodiment_id: str, dest_path: str):
        """
        Export embodiment to external file.

        Args:
            embodiment_id: Embodiment to export
            dest_path: Destination file path

        Creates standalone .embodiment file for sharing.
        """
        data = self.load(embodiment_id)
        if not data:
            raise ValueError(f"Embodiment not found: {embodiment_id}")

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Exported embodiment {embodiment_id} to {dest_path}")

    def import_embodiment(self, source_path: str, new_id: Optional[str] = None) -> str:
        """
        Import embodiment from external file.

        Args:
            source_path: Path to .embodiment file
            new_id: Optional new ID (auto-generated if None)

        Returns:
            Imported embodiment ID
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        with open(source, 'r') as f:
            data = yaml.safe_load(f)

        # Use provided ID or generate from file
        if new_id:
            embodiment_id = new_id
        else:
            # Try to get ID from metadata
            embodiment_id = data.get('metadata', {}).get('id')
            if not embodiment_id:
                # Generate from filename
                name_slug = source.stem
                embodiment_id = f"user.imported.{name_slug}"

        # Check for conflicts
        existing = self.load(embodiment_id)
        if existing:
            # Add timestamp to avoid collision
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            embodiment_id = f"{embodiment_id}_{timestamp}"

        self.save(embodiment_id, data)
        logger.info(f"Imported embodiment as {embodiment_id}")

        return embodiment_id

    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate embodiment data structure.

        Args:
            data: Embodiment data dict

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

        # Required: embodiment
        if 'embodiment' not in data:
            errors.append("Missing 'embodiment' section")
        else:
            emb = data['embodiment']

            # Required: architecture
            if 'architecture' not in emb:
                errors.append("Missing embodiment.architecture")
            else:
                arch = emb['architecture']
                if 'form' not in arch:
                    errors.append("Missing embodiment.architecture.form")
                if 'locomotion' not in arch:
                    errors.append("Missing embodiment.architecture.locomotion")

            # Required: characteristics
            if 'characteristics' not in emb:
                errors.append("Missing embodiment.characteristics")
            else:
                chars = emb['characteristics']
                if 'size' not in chars:
                    errors.append("Missing embodiment.characteristics.size")

            # Optional but recommended: state
            if 'state' not in emb:
                logger.warning("Embodiment missing 'state' section (empty state will be created)")

        return (len(errors) == 0, errors)

    def get_default_embodiment(self) -> Dict:
        """
        Get default embodiment for Noodlings without specified body.

        Returns:
            Default embodiment data dict
        """
        return {
            'metadata': {
                'id': 'com.noodlings.embodiments.default_noodling',
                'name': 'Default Noodling',
                'version': '1.0.0',
                'description': 'Default embodiment for unspecified Noodlings',
                'tags': ['default', 'amorphous']
            },
            'embodiment': {
                'architecture': {
                    'form': 'amorphous',
                    'limb_count': 0,
                    'has_tail': False,
                    'has_wings': False,
                    'locomotion': ['float', 'shift', 'manifest']
                },
                'characteristics': {
                    'size': 'small',
                    'substance': 'thought_patterns',
                    'tangible': False
                },
                'state': {
                    'coherence': 1.0
                },
                'movement': {
                    'baseSpeed': 0.5,
                    'canSwim': False,
                    'canFly': True,
                    'canClimb': False
                },
                'senses': {
                    'vision': 'full',
                    'hearing': 'full',
                    'smell': 'none',
                    'touch': 'none',
                    'proprioception': 'full'
                },
                'worn_items': []
            }
        }


# Example usage and testing
if __name__ == '__main__':
    loader = EmbodimentLoader("assets/embodiments")

    # List all embodiments
    embodiments = loader.list_all()
    print(f"Found {len(embodiments)} embodiments:")
    for e in embodiments:
        print(f"  {e['id']} - {e['name']} (v{e['version']})")
