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

        # Preference file for "last used layout"
        self.prefs_file = self.layouts_dir / "preferences.json"

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
        Load saved layout with improved error handling.

        Args:
            window: QMainWindow instance
            layout_name: Layout name to load

        Returns:
            True if loaded successfully
        """
        layout_file = self.layouts_dir / f"{layout_name}.json"

        if not layout_file.exists():
            print(f"Layout '{layout_name}' not found at {layout_file}")
            return False

        try:
            with open(layout_file, 'r') as f:
                layout_data = json.load(f)

            print(f"Loading layout '{layout_name}'...")

            # Validate data before attempting restore
            if 'geometry' not in layout_data or 'state' not in layout_data:
                print(f"  Invalid layout file - missing geometry or state")
                return False

            success = False

            # Restore geometry first (safer)
            try:
                geometry = bytes.fromhex(layout_data['geometry'])
                result_geo = window.restoreGeometry(geometry)
                print(f"  Geometry restored: {result_geo}")
                if result_geo:
                    success = True
            except Exception as e:
                print(f"  Geometry restore failed: {e}")
                # Continue anyway - state might still work

            # Restore state (this is where crashes happen)
            # Skip state restoration to prevent C++ segfaults
            # State restoration can crash if saved with different widget structure
            try:
                # Validate state data first
                state_hex = layout_data.get('state', '')
                if not state_hex or len(state_hex) < 10:
                    print(f"  Skipping state restore - invalid data")
                else:
                    # Validate hex format
                    state = bytes.fromhex(state_hex)
                    # Skip restoreState to prevent crashes
                    # TODO: Implement safe state restoration with version checking
                    print(f"  State restore skipped (prevents crashes)")
            except Exception as e:
                print(f"  State validation failed (non-fatal): {e}")
                # Don't return False - geometry might have worked

            if success:
                print(f"Layout '{layout_name}' loaded (partial or full)")
                # Save as last used
                self.set_last_used_layout(layout_name)
                return True
            else:
                print(f"Layout '{layout_name}' failed to load")
                return False

        except Exception as e:
            print(f"Error loading layout '{layout_name}': {e}")
            import traceback
            traceback.print_exc()
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

    def set_last_used_layout(self, layout_name: str):
        """
        Save the last used layout name (like Unity's last scene).

        Args:
            layout_name: The layout that was just loaded
        """
        prefs = {}
        if self.prefs_file.exists():
            try:
                with open(self.prefs_file, 'r') as f:
                    prefs = json.load(f)
            except:
                pass

        prefs['last_used_layout'] = layout_name

        with open(self.prefs_file, 'w') as f:
            json.dump(prefs, f, indent=2)

        print(f"Last used layout: '{layout_name}'")

    def get_last_used_layout(self) -> str | None:
        """
        Get the last used layout (like Unity reopening last scene).

        Returns:
            Layout name or None if no preference saved
        """
        if not self.prefs_file.exists():
            return None

        try:
            with open(self.prefs_file, 'r') as f:
                prefs = json.load(f)
                return prefs.get('last_used_layout')
        except:
            return None
