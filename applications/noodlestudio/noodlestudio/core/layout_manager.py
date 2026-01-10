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
#   Layout Manager - Save/Load panel configurations
#
#   Like Unity's layout presets - save your favorite panel ar...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.layout_manager
# PURPOSE:  Layout Manager - Save/Load panel configurations
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   LayoutManager
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

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
        # Use versioned state for safer restoration
        STATE_VERSION = 1

        layout_data = {
            'version': STATE_VERSION,
            'geometry': window.saveGeometry().toHex().data().decode(),
            'state': window.saveState(STATE_VERSION).toHex().data().decode()
        }

        # Save to file
        layout_file = self.layouts_dir / f"{layout_name}.json"
        with open(layout_file, 'w') as f:
            json.dump(layout_data, f, indent=2)

        print(f"Layout '{layout_name}' saved to {layout_file}")

    def load_layout(self, window, layout_name: str) -> bool:
        """
        Load saved layout with safe state restoration.

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

            # Check version compatibility
            STATE_VERSION = 1
            saved_version = layout_data.get('version', 0)

            success = False

            # Restore geometry first (safer)
            try:
                from PyQt6.QtWidgets import QApplication
                from PyQt6.QtCore import QRect

                geometry = bytes.fromhex(layout_data['geometry'])
                result_geo = window.restoreGeometry(geometry)

                # Constrain window to screen bounds (for Parsec/remote desktop)
                screen = QApplication.primaryScreen()
                if screen:
                    screen_geom = screen.availableGeometry()
                    window_geom = window.frameGeometry()

                    # If window extends beyond screen, move/resize it
                    if not screen_geom.contains(window_geom):
                        # Constrain width and height
                        new_width = min(window_geom.width(), screen_geom.width())
                        new_height = min(window_geom.height(), screen_geom.height())

                        # Constrain position
                        new_x = max(screen_geom.x(), min(window_geom.x(), screen_geom.right() - new_width))
                        new_y = max(screen_geom.y(), min(window_geom.y(), screen_geom.bottom() - new_height))

                        window.setGeometry(new_x, new_y, new_width, new_height)
                        print(f"  Window constrained to screen bounds: {new_width}x{new_height} at ({new_x},{new_y})")

                print(f"  Geometry restored: {result_geo}")
                if result_geo:
                    success = True
            except Exception as e:
                print(f"  Geometry restore failed: {e}")
                # Continue anyway - state might still work

            # Restore state with version checking
            try:
                state_hex = layout_data.get('state', '')
                if not state_hex or len(state_hex) < 10:
                    print(f"  Skipping state restore - invalid data")
                elif saved_version != STATE_VERSION:
                    print(f"  Skipping state restore - version mismatch (saved: {saved_version}, expected: {STATE_VERSION})")
                else:
                    state = bytes.fromhex(state_hex)
                    result_state = window.restoreState(state, STATE_VERSION)
                    print(f"  State restored: {result_state}")
                    if result_state:
                        success = True
            except Exception as e:
                print(f"  State restore failed (non-fatal): {e}")
                # Don't return False - geometry might have worked

            if success:
                print(f"Layout '{layout_name}' loaded successfully")
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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
