"""
Layout Manager - Save/Load panel configurations

Like Unity's layout presets - save your favorite panel arrangements.

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtCore import QSettings
from pathlib import Path
import json


class LayoutManager:
    """
    Manages panel layout configurations.

    Saves/loads:
    - Panel visibility
    - Panel sizes
    - Dock positions
    - Splitter states
    """

    def __init__(self, app_name: str = "NoodleStudio"):
        self.settings = QSettings("Noodlings", app_name)
        self.layouts_dir = Path.home() / ".noodlestudio" / "layouts"
        self.layouts_dir.mkdir(parents=True, exist_ok=True)

    def save_layout(self, window, layout_name: str):
        """
        Save current window layout.

        Args:
            window: QMainWindow instance
            layout_name: Name for this layout (e.g., "Default", "Demo Mode")
        """
        layout_data = {
            'geometry': window.saveGeometry().toHex().data().decode(),
            'state': window.saveState().toHex().data().decode()
        }

        # Save to file
        layout_file = self.layouts_dir / f"{layout_name}.json"
        with open(layout_file, 'w') as f:
            json.dump(layout_data, f, indent=2)

        print(f"Layout '{layout_name}' saved to {layout_file}")

    def load_layout(self, window, layout_name: str) -> bool:
        """
        Load saved layout.

        Args:
            window: QMainWindow instance
            layout_name: Layout name to load

        Returns:
            True if loaded successfully
        """
        layout_file = self.layouts_dir / f"{layout_name}.json"

        if not layout_file.exists():
            print(f"Layout '{layout_name}' not found")
            return False

        try:
            with open(layout_file, 'r') as f:
                layout_data = json.load(f)

            geometry = bytes.fromhex(layout_data['geometry'])
            state = bytes.fromhex(layout_data['state'])

            window.restoreGeometry(geometry)
            window.restoreState(state)

            print(f"Layout '{layout_name}' loaded")
            return True

        except Exception as e:
            print(f"Error loading layout: {e}")
            return False

    def list_layouts(self) -> list:
        """List all saved layouts."""
        return [f.stem for f in self.layouts_dir.glob("*.json")]

    def delete_layout(self, layout_name: str):
        """Delete a saved layout."""
        layout_file = self.layouts_dir / f"{layout_name}.json"
        if layout_file.exists():
            layout_file.unlink()
            print(f"Layout '{layout_name}' deleted")
