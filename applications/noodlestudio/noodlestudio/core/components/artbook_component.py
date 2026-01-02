"""
Artbook Component - Reference art collection for characters.

Stores reference images, concept art, mood boards - anything visual
that helps define a character's appearance.

Features:
- Thumbnail gallery in Inspector
- Drag-and-drop import
- Persistent storage per-entity
- Other components can query for reference images

Author: Caitlyn + Claude
Date: January 2026
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from ..component_base import (
    ComponentBase,
    ComponentCategory,
    PropertySpec,
    register_component,
)

logger = logging.getLogger(__name__)


@register_component
class ArtbookComponent(ComponentBase):
    """
    Reference art collection component.

    Attach to any entity to store reference images.
    Inspector shows a thumbnail gallery with add/remove controls.
    """

    def __init__(self, entity_id: str = ""):
        super().__init__(entity_id)

        # List of image file paths
        self._art_files: List[str] = []

        # Optional notes/description per image
        self._art_notes: Dict[str, str] = {}

        # Gallery display settings
        self._thumbnail_size: int = 80
        self._columns: int = 4

    # ==========================================================================
    # ComponentBase implementation
    # ==========================================================================

    @property
    def component_type(self) -> str:
        return "artbook"

    @property
    def display_name(self) -> str:
        return "Artbook"

    @property
    def category(self) -> ComponentCategory:
        return ComponentCategory.ART_REFERENCE

    @property
    def description(self) -> str:
        return "Reference art collection for visual design."

    @property
    def icon(self) -> Optional[str]:
        return None  # Could return path to icon

    @property
    def property_specs(self) -> List[PropertySpec]:
        return [
            PropertySpec(
                name="thumbnail_size",
                display_name="Thumbnail Size",
                property_type="int",
                default=80,
                min_value=40,
                max_value=200,
                description="Size of thumbnail images in pixels"
            ),
            PropertySpec(
                name="columns",
                display_name="Columns",
                property_type="int",
                default=4,
                min_value=1,
                max_value=8,
                description="Number of columns in gallery grid"
            ),
        ]

    # ==========================================================================
    # Art management
    # ==========================================================================

    @property
    def art_files(self) -> List[str]:
        """Get list of art file paths."""
        return self._art_files.copy()

    @property
    def art_count(self) -> int:
        """Number of art files in collection."""
        return len(self._art_files)

    @property
    def thumbnail_size(self) -> int:
        return self._thumbnail_size

    @thumbnail_size.setter
    def thumbnail_size(self, value: int):
        self._thumbnail_size = max(40, min(200, value))
        self._mark_dirty()

    @property
    def columns(self) -> int:
        return self._columns

    @columns.setter
    def columns(self, value: int):
        self._columns = max(1, min(8, value))
        self._mark_dirty()

    def add_art(self, file_path: str, note: str = "") -> bool:
        """
        Add an image to the collection.

        Args:
            file_path: Path to image file
            note: Optional note/description

        Returns:
            True if added, False if already exists or invalid
        """
        # Normalize path
        path = str(Path(file_path).resolve())

        # Check if already present
        if path in self._art_files:
            logger.debug(f"Art already in collection: {path}")
            return False

        # Verify file exists
        if not Path(path).exists():
            logger.warning(f"Art file not found: {path}")
            # Still add it - file might appear later
            # return False

        self._art_files.append(path)
        if note:
            self._art_notes[path] = note

        self._mark_dirty()
        logger.info(f"Added art to {self._entity_id}: {path}")
        return True

    def remove_art(self, file_path: str) -> bool:
        """
        Remove an image from the collection.

        Args:
            file_path: Path to remove

        Returns:
            True if removed, False if not found
        """
        path = str(Path(file_path).resolve())

        if path not in self._art_files:
            return False

        self._art_files.remove(path)
        self._art_notes.pop(path, None)

        self._mark_dirty()
        logger.info(f"Removed art from {self._entity_id}: {path}")
        return True

    def reorder_art(self, from_index: int, to_index: int) -> bool:
        """
        Reorder images in the collection.

        Args:
            from_index: Current index
            to_index: New index

        Returns:
            True if reordered successfully
        """
        if not (0 <= from_index < len(self._art_files)):
            return False
        if not (0 <= to_index < len(self._art_files)):
            return False

        item = self._art_files.pop(from_index)
        self._art_files.insert(to_index, item)

        self._mark_dirty()
        return True

    def get_note(self, file_path: str) -> str:
        """Get note for an image."""
        path = str(Path(file_path).resolve())
        return self._art_notes.get(path, "")

    def set_note(self, file_path: str, note: str):
        """Set note for an image."""
        path = str(Path(file_path).resolve())
        if path in self._art_files:
            if note:
                self._art_notes[path] = note
            else:
                self._art_notes.pop(path, None)
            self._mark_dirty()

    def clear(self):
        """Remove all art from collection."""
        self._art_files.clear()
        self._art_notes.clear()
        self._mark_dirty()

    # ==========================================================================
    # Querying (for other components)
    # ==========================================================================

    def get_random_art(self) -> Optional[str]:
        """Get a random image path (for variety in prompts, etc.)."""
        import random
        if not self._art_files:
            return None
        return random.choice(self._art_files)

    def get_art_with_note(self, keyword: str) -> List[str]:
        """Get art files whose notes contain keyword."""
        keyword_lower = keyword.lower()
        return [
            path for path in self._art_files
            if keyword_lower in self._art_notes.get(path, "").lower()
        ]

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = self._base_to_dict()
        data.update({
            'art_files': self._art_files,
            'art_notes': self._art_notes,
            'thumbnail_size': self._thumbnail_size,
            'columns': self._columns,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], entity_id: str = "") -> 'ArtbookComponent':
        """Deserialize from dictionary."""
        component = cls(entity_id)
        component._base_from_dict(data)

        component._art_files = data.get('art_files', [])
        component._art_notes = data.get('art_notes', {})
        component._thumbnail_size = data.get('thumbnail_size', 80)
        component._columns = data.get('columns', 4)

        return component


__all__ = ['ArtbookComponent']
