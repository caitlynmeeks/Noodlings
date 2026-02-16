"""Depth view protocol for stackable editor views.

Any view that can be pushed onto the UnifiedEditorPanel's depth stack
must implement this interface.
"""

from abc import ABC, abstractmethod


class DepthViewProtocol(ABC):
    """Contract for stackable editor views."""

    @abstractmethod
    def load_data(self, data_path: str, context: dict) -> None:
        """Load and render the data for this depth level.

        Args:
            data_path: Path to the data file (assembly.yaml, .nncanvas, etc.)
            context: Additional context (project_root, noodling_id, etc.)
        """

    @abstractmethod
    def save_data(self) -> None:
        """Persist any unsaved changes to disk."""

    @abstractmethod
    def get_breadcrumb_label(self) -> str:
        """Short label for the breadcrumb bar."""

    @abstractmethod
    def has_unsaved_changes(self) -> bool:
        """Whether this view has modifications that need saving."""
