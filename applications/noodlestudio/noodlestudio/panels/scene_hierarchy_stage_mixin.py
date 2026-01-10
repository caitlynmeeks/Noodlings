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
#   Scene Hierarchy Stage Mixin - Stage selector management
#
#   Contains: - populate_stage_selector: Populate dropdown wi...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.scene_hierarchy_stage_mixin
# PURPOSE:  Scene Hierarchy Stage Mixin
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SceneHierarchyStageMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os

from PyQt6.QtWidgets import QMessageBox


class SceneHierarchyStageMixin:
    """Mixin providing stage selector management for SceneHierarchy."""

    def populate_stage_selector(self):
        """Populate stage selector with available stages/rooms."""
        import json
        import yaml

        # Block signals during population to avoid triggering on_stage_changed
        self.stage_selector.blockSignals(True)
        self.stage_selector.clear()

        try:
            # Check if project is open - use new format
            if self.project_manager and self.project_manager.is_project_open():
                stages = self.project_manager.list_stages()

                for stage_name in stages:
                    stage_path = self.project_manager.get_stage_path(stage_name)
                    if stage_path:
                        # Try to load stage.yaml for display name
                        display_name = stage_name
                        stage_yaml = os.path.join(stage_path, "stage.yaml")
                        if os.path.exists(stage_yaml):
                            try:
                                with open(stage_yaml, 'r') as f:
                                    stage_data = yaml.safe_load(f) or {}
                                    display_name = stage_data.get('name', stage_name)
                            except:
                                pass

                        display_text = f"{display_name} ({stage_name})"
                        self.stage_selector.addItem(display_text, stage_name)

                        # Select current stage
                        if stage_name == self.current_stage:
                            self.stage_selector.setCurrentText(display_text)

                # If no stage selected, select first one
                if not self.current_stage and stages:
                    self.current_stage = stages[0]
                    self.stage_selector.setCurrentIndex(0)

            else:
                # Legacy mode - load from rooms.json
                rooms_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../../cmush/world/rooms.json"
                )
                if os.path.exists(rooms_path):
                    with open(rooms_path, 'r') as f:
                        rooms_data = json.load(f)

                    for room_id, room_data in rooms_data.items():
                        room_name = room_data.get('name', room_id)
                        display_text = f"{room_name} ({room_id})"
                        self.stage_selector.addItem(display_text, room_id)

                        # Select current room
                        if room_id == self.current_room:
                            self.stage_selector.setCurrentText(display_text)

        except Exception as e:
            print(f"Error populating stage selector: {e}")

        self.stage_selector.blockSignals(False)

    def on_stage_changed(self, text):
        """Handle stage selection change."""
        if not text:
            return

        # Get stage/room id from combo box data
        new_id = self.stage_selector.currentData()

        # Check if this is project mode or legacy mode
        if self.project_manager and self.project_manager.is_project_open():
            # Project mode - using stages
            if new_id == self.current_stage:
                return

            stage_name = text.split(' (')[0] if ' (' in text else text

            # Check for unsaved changes
            if self._dirty:
                reply = QMessageBox.question(
                    self,
                    "Unsaved Changes",
                    f"Save changes to current stage before switching to '{stage_name}'?",
                    QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Save
                )

                if reply == QMessageBox.StandardButton.Cancel:
                    # Restore the combo box selection
                    self._restore_stage_selector()
                    return
                elif reply == QMessageBox.StandardButton.Save:
                    self._save_hierarchy()

            # Switch to new stage
            self.current_stage = new_id
            self._dirty = False  # Reset dirty flag for new stage
            self.refresh_scene()
            # Clear Inspector - no entity selected after stage switch
            self.entitySelected.emit("", {})
        else:
            # Legacy mode - using rooms
            if new_id == self.current_room:
                return

            room_name = text.split(' (')[0] if ' (' in text else text
            reply = QMessageBox.question(
                self,
                "Teleport to Stage?",
                f"Teleport your character to stage '{room_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.teleport_to_stage(new_id)

            self.current_room = new_id
            self.refresh_scene()

    def teleport_to_stage(self, room_id):
        """Send teleport command to noodleMUSH."""
        try:
            # TODO: Implement teleport API endpoint
            # For now, just log it
            print(f"Teleporting to stage: {room_id}")
        except Exception as e:
            print(f"Error teleporting: {e}")

    def _restore_stage_selector(self):
        """Restore stage selector to current stage (used when Cancel is pressed)."""
        self.stage_selector.blockSignals(True)
        for i in range(self.stage_selector.count()):
            if self.stage_selector.itemData(i) == self.current_stage:
                self.stage_selector.setCurrentIndex(i)
                break
        self.stage_selector.blockSignals(False)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
