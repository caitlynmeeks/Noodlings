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
#   Main Window Project Mixin - Project management operations
#
#   Contains: - new_project, open_project: Project creation/o...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.main_window_project_mixin
# PURPOSE:  Main Window Project Mixin
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MainWindowProjectMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import json
from pathlib import Path
from typing import List

from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from PyQt6.QtCore import QTimer, QStandardPaths


class MainWindowProjectMixin:
    """Mixin providing project management for MainWindow."""

    def new_project(self):
        """Show Project Chooser dialog for creating/opening projects."""
        self._show_project_chooser()

    def _close_project(self):
        """Close the current project and return to empty state."""
        if not self.project_manager.is_project_open():
            return
        self.project_manager.close_project()
        self.statusBar().showMessage("Project closed", 3000)

    def open_project(self):
        """Open an existing NoodleStudio project."""
        project_path = QFileDialog.getExistingDirectory(
            self, "Open Project", os.path.expanduser("~/Documents")
        )
        if not project_path:
            return

        if self.project_manager.open_project(project_path):
            self.statusBar().showMessage(
                f"Opened project: {self.project_manager.current_project_name}", 3000
            )
        else:
            QMessageBox.warning(
                self, "Error",
                "Failed to open project.\nNot a valid NoodleStudio project."
            )

    def _update_window_title(self, _text=None):
        """Update window title to reflect current project and stage."""
        parts = ["NoodleStudio"]
        if self.project_manager.is_project_open():
            parts.append(self.project_manager.current_project_name)
            # Get stage display name from the stage selector (not the internal key)
            if hasattr(self, 'hierarchy') and self.hierarchy.current_stage:
                selector_text = self.hierarchy.stage_selector.currentText()
                # Stage selector format is "Display Name (dir_key)"
                if ' (' in selector_text:
                    stage_display = selector_text.split(' (')[0]
                else:
                    stage_display = selector_text
                if stage_display:
                    parts.append(stage_display)
        self.setWindowTitle(" \u2014 ".join(parts))

    def on_project_opened(self, project_path: str):
        """Handle project opened event."""
        try:
            import subprocess
            # Stop server when switching projects
            subprocess.run(['pkill', '-f', 'python.*server.py'])

            # Save to recent projects
            self.add_to_recent_projects(project_path)
            self.update_recent_projects_menu()

            # Refresh assets panel
            if hasattr(self, 'assets'):
                self.assets.refresh()

            # Show offline card (server is stopped, deferred in MVP)
            if hasattr(self, 'world_view') and self.world_view:
                self.world_view.show_offline_card()

            # Refresh hierarchy with new project structure
            if hasattr(self, 'hierarchy'):
                self.hierarchy.clear_for_project_change()
                self.hierarchy.populate_stage_selector()
                self.hierarchy.refresh_scene()
                self.hierarchy.set_server_state(False)
                # Update title when stage changes
                try:
                    self.hierarchy.stage_selector.currentTextChanged.disconnect(
                        self._update_window_title)
                except (TypeError, RuntimeError):
                    pass  # Not yet connected
                self.hierarchy.stage_selector.currentTextChanged.connect(
                    self._update_window_title)

            # Update window title (after hierarchy is populated with stage info)
            self._update_window_title()

            QTimer.singleShot(500, self.update_connection_status)

            # Refresh spatial view (deferred in MVP)
            if hasattr(self, 'spatial_view') and self.spatial_view:
                self.spatial_view.set_project_manager(self.project_manager)

            # Update Noodle Code with project path
            if hasattr(self, 'noodle_code_engine'):
                from pathlib import Path
                self.noodle_code_engine.set_project_path(Path(project_path))

            # Load editor access settings from build.yaml
            self._load_editor_access_from_build_config(project_path)

            print(f"Project opened: {project_path}")

        except Exception as e:
            import traceback
            print(f"[MainWindow] Error in on_project_opened: {e}")
            traceback.print_exc()

    def on_project_closed(self):
        """Handle project closed event."""
        self._update_window_title()

        if hasattr(self, 'assets'):
            self.assets.refresh()

        if hasattr(self, 'hierarchy'):
            self.hierarchy.current_stage = None
            self.hierarchy.current_room = "room_000"
            self.hierarchy.populate_stage_selector()
            self.hierarchy.refresh_scene()

        print("Project closed")

    def new_noodling(self):
        """Create a new Noodling in the current project."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        name, ok = QInputDialog.getText(
            self, "New Noodling", "Noodling Name:", text="New Noodling"
        )
        if not ok or not name:
            return

        desc, ok = QInputDialog.getText(
            self, "New Noodling", "Description (optional):"
        )

        path = self.project_manager.create_noodling(name, desc if ok else "")
        if path:
            self.statusBar().showMessage(f"Created noodling: {name}", 3000)
            if hasattr(self, 'assets'):
                self.assets.refresh()
        else:
            QMessageBox.warning(
                self, "Error", "Failed to create noodling. Name may already exist."
            )

    def new_stage(self):
        """Create a new Stage in the current project."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        existing_stages = set(self.project_manager.list_stages())

        # Generate unique default name
        default_name = "New Stage"
        if default_name.lower().replace(' ', '_') in [s.lower() for s in existing_stages]:
            counter = 1
            while f"{default_name} ({counter})".lower().replace(' ', '_') in [
                s.lower() for s in existing_stages
            ]:
                counter += 1
            default_name = f"{default_name} ({counter})"

        name, ok = QInputDialog.getText(
            self, "New Stage", "Stage Name:", text=default_name
        )
        if not ok or not name:
            return

        # Try to create, auto-increment if name exists
        original_name = name
        path = self.project_manager.create_stage(name, "")
        if not path:
            counter = 1
            while not path and counter < 100:
                name = f"{original_name} ({counter})"
                path = self.project_manager.create_stage(name, "")
                counter += 1

        if path:
            self.statusBar().showMessage(f"Created stage: {name}", 3000)
            if hasattr(self, 'assets'):
                self.assets.refresh()
            if hasattr(self, 'hierarchy'):
                stage_folder_name = os.path.basename(path)
                self.hierarchy.populate_stage_selector()
                self.hierarchy.current_stage = stage_folder_name
                for i in range(self.hierarchy.stage_selector.count()):
                    if self.hierarchy.stage_selector.itemData(i) == stage_folder_name:
                        self.hierarchy.stage_selector.setCurrentIndex(i)
                        break
                self.hierarchy.refresh_scene()
        else:
            QMessageBox.warning(self, "Error", "Failed to create stage.")

    def new_prim(self):
        """Create a new Prim template in the current project."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        name, ok = QInputDialog.getText(
            self, "New Prim", "Prim Name:", text="New Prim"
        )
        if not ok or not name:
            return

        desc, ok = QInputDialog.getText(
            self, "New Prim", "Text description (for MUD):", text=f"a {name.lower()}"
        )

        path = self.project_manager.create_prim(name, "", desc if ok else "")
        if path:
            self.statusBar().showMessage(f"Created prim: {name}", 3000)
            if hasattr(self, 'assets'):
                self.assets.refresh()
        else:
            QMessageBox.warning(
                self, "Error", "Failed to create prim. Name may already exist."
            )

    def save_project(self):
        """Save the current project and cascade to all dirty subsystems."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "No project is open.")
            return

        if not self.project_manager.save_project():
            QMessageBox.warning(self, "Error", "Failed to save project.")
            return

        # Cascade: flush all dirty editors
        # 1. Unified editor -- save all depth views (assembly + NC)
        editor = getattr(self, 'unified_editor', None)
        if editor and hasattr(editor, 'save_if_dirty'):
            editor.save_if_dirty()

        # 3. Inspector -- flush any pending property changes
        inspector = getattr(self, 'inspector', None)
        if inspector and hasattr(inspector, 'save_changes'):
            inspector.save_changes()

        self.statusBar().showMessage("Project saved", 3000)

    def save_stage(self):
        """Save the current stage hierarchy."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "No project is open.")
            return

        if hasattr(self, 'scene_hierarchy') and self.scene_hierarchy:
            if self.scene_hierarchy.save_stage():
                self.statusBar().showMessage("Stage saved", 3000)
            else:
                QMessageBox.warning(self, "Error", "Failed to save stage.")
        else:
            QMessageBox.warning(self, "Error", "Scene hierarchy not available.")

    def _on_play_toggled(self, checked: bool):
        """Play/Stop toggle for the current stage."""
        manager = getattr(self, 'guide_performance_manager', None)
        if not manager:
            self.statusBar().showMessage("Performance manager not ready", 3000)
            if hasattr(self, '_play_button'):
                self._play_button.setChecked(False)
            return

        if checked:
            # Start performance from current stage
            hierarchy = getattr(self, 'hierarchy', None)
            if not hierarchy or not getattr(hierarchy, 'current_stage', None):
                self.statusBar().showMessage("No stage selected", 3000)
                self._play_button.setChecked(False)
                return

            stage_path = self.project_manager.get_stage_path(hierarchy.current_stage)
            if not stage_path:
                self.statusBar().showMessage("Stage path not found", 3000)
                self._play_button.setChecked(False)
                return

            manager.start_ensemble_from_stage(stage_path, hierarchy.current_stage)
            self._play_button.setText("Stop")
            self.statusBar().showMessage("Playing...", 3000)
        else:
            # Stop performance
            manager.stop_performance()
            self._play_button.setText("Play")
            self.statusBar().showMessage("Stopped", 3000)

    def import_noodling_folder(self):
        """Import a noodling folder into the current project."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Import Noodling Folder", os.path.expanduser("~")
        )
        if not folder:
            return

        # Check if it's a valid noodling folder
        if not os.path.exists(os.path.join(folder, "noodling.yaml")):
            if not os.path.exists(os.path.join(folder, "recipe.yaml")):
                QMessageBox.warning(
                    self, "Invalid Folder",
                    "This doesn't appear to be a valid noodling folder.\n"
                    "Expected noodling.yaml or recipe.yaml."
                )
                return

        import shutil
        noodling_name = os.path.basename(folder)
        target = os.path.join(self.project_manager.get_noodlings_path(), noodling_name)

        if os.path.exists(target):
            reply = QMessageBox.question(
                self, "Noodling Exists",
                f"A noodling named '{noodling_name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            shutil.rmtree(target)

        shutil.copytree(folder, target)
        self.statusBar().showMessage(f"Imported noodling: {noodling_name}", 3000)
        if hasattr(self, 'assets'):
            self.assets.refresh()

    def export_noodling(self):
        """Export a noodling to a folder."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        noodlings = self.project_manager.list_noodlings()
        if not noodlings:
            QMessageBox.information(self, "No Noodlings", "No noodlings to export.")
            return

        name, ok = QInputDialog.getItem(
            self, "Export Noodling", "Select noodling:", noodlings, 0, False
        )
        if not ok:
            return

        target_dir = QFileDialog.getExistingDirectory(
            self, "Export To", os.path.expanduser("~")
        )
        if not target_dir:
            return

        import shutil
        source = self.project_manager.get_noodling_path(name)
        target = os.path.join(target_dir, name)

        if os.path.exists(target):
            shutil.rmtree(target)

        shutil.copytree(source, target)
        self.statusBar().showMessage(f"Exported noodling to: {target}", 3000)

    def export_unity_package(self):
        """Export a noodling to Unity-compatible .noodling package."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        noodlings = self.project_manager.list_noodlings()
        if not noodlings:
            QMessageBox.information(self, "No Noodlings", "No noodlings to export.")
            return

        name, ok = QInputDialog.getItem(
            self, "Export Unity Package", "Select noodling:", noodlings, 0, False
        )
        if not ok:
            return

        target_dir = QFileDialog.getExistingDirectory(
            self, "Export Unity Package To", os.path.expanduser("~")
        )
        if not target_dir:
            return

        try:
            from .noodling_package_exporter import NoodlingPackageExporter, ExportOptions
            from pathlib import Path

            exporter = NoodlingPackageExporter(self.project_manager)
            package_path = exporter.export(
                name,
                Path(target_dir),
                ExportOptions(include_plays=True)
            )

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported Unity package to:\n{package_path}\n\n"
                "Drag this folder into your Unity project's Assets."
            )
            self.statusBar().showMessage(f"Exported Unity package: {package_path}", 5000)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export Unity package:\n{str(e)}"
            )

    def _load_editor_access_from_build_config(self, project_path: str):
        """
        Load editor access settings from build.yaml.

        Called when a project is opened to apply editor access restrictions.

        Args:
            project_path: Path to the project directory
        """
        from pathlib import Path as PathLib
        build_yaml = PathLib(project_path) / "build.yaml"

        if not build_yaml.exists():
            # No build.yaml - reset to default (allow)
            if hasattr(self, 'set_editor_access'):
                self.set_editor_access(access="allow")
            return

        try:
            from .build_config import BuildConfig
            config = BuildConfig.from_yaml(build_yaml)

            if hasattr(self, 'set_editor_access'):
                self.set_editor_access(
                    access=config.editor.access,
                    password_hash=config.editor.password_hash,
                    keyboard_shortcut=config.editor.keyboard_shortcut
                )
                print(f"Editor access loaded: {config.editor.access}")
        except Exception as e:
            print(f"Warning: Could not load editor access from build.yaml: {e}")
            # Reset to default on error
            if hasattr(self, 'set_editor_access'):
                self.set_editor_access(access="allow")

    def show_build_settings(self):
        """Show the Build Settings dialog."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        project_path = Path(self.project_manager.current_project_path)

        from ..dialogs.build_settings_dialog import BuildSettingsDialog
        dialog = BuildSettingsDialog(project_path, self)
        if dialog.exec():
            self.statusBar().showMessage("Build settings saved", 3000)
            # Refresh assets to show build.yaml if newly created
            if hasattr(self, 'assets'):
                self.assets.refresh()
            # Reload editor access settings
            self._load_editor_access_from_build_config(str(project_path))

    def build_application(self):
        """Build a standalone application from the current project."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        project_path = Path(self.project_manager.current_project_path)

        # Check for build.yaml or create one
        build_yaml = project_path / "build.yaml"
        if not build_yaml.exists():
            reply = QMessageBox.question(
                self, "No Build Configuration",
                "This project doesn't have a build.yaml file.\n\n"
                "Create a default build configuration?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            # Create default build.yaml
            from ..appbuilder.builder import create_default_build_yaml
            create_default_build_yaml(project_path, self.project_manager.current_project_name)
            self.statusBar().showMessage("Created build.yaml - please edit and try again", 5000)

            # Refresh assets to show the new file
            if hasattr(self, 'assets'):
                self.assets.refresh()
            return

        # Get output location
        default_name = f"{self.project_manager.current_project_name}.app"
        desktop = os.path.expanduser("~/Desktop")
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Build Application",
            os.path.join(desktop, default_name),
            "macOS Application (*.app)"
        )
        if not output_path:
            return

        # Run build
        self._run_build(project_path, Path(output_path))

    def _run_build(self, project_path: Path, output_path: Path):
        """Execute the build process with progress dialog."""
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

        # Create progress dialog
        progress = QProgressDialog(
            "Preparing build...", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Building Application")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        cancelled = False

        def on_progress(percent: int, message: str):
            nonlocal cancelled
            if progress.wasCanceled():
                cancelled = True
                return
            progress.setValue(percent)
            progress.setLabelText(message)
            QApplication.processEvents()

        try:
            from ..appbuilder import Builder, BuildConfig

            # Load config
            config = BuildConfig.load(project_path)
            builder = Builder(config)
            builder.on_progress(on_progress)

            # Run build
            result = builder.build(str(output_path))

            progress.close()

            if cancelled:
                # Clean up partial build
                if output_path.exists():
                    import shutil
                    shutil.rmtree(output_path)
                self.statusBar().showMessage("Build cancelled", 3000)
                return

            if result.success:
                # Format size
                size_mb = result.total_size_bytes / (1024 * 1024)
                time_s = result.build_time_seconds

                QMessageBox.information(
                    self, "Build Complete",
                    f"Application built successfully!\n\n"
                    f"Location: {output_path}\n"
                    f"Size: {size_mb:.1f} MB\n"
                    f"Time: {time_s:.1f} seconds\n"
                    f"Files: {result.total_files}"
                )

                # Offer to reveal in Finder
                reply = QMessageBox.question(
                    self, "Reveal in Finder",
                    "Would you like to reveal the application in Finder?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import subprocess
                    subprocess.run(['open', '-R', str(output_path)])

            else:
                error_text = "\n".join(result.errors) if result.errors else "Unknown error"
                QMessageBox.critical(
                    self, "Build Failed",
                    f"Failed to build application.\n\n{error_text}"
                )

        except FileNotFoundError as e:
            progress.close()
            QMessageBox.critical(
                self, "Build Error",
                f"Build configuration error:\n\n{str(e)}"
            )
        except Exception as e:
            progress.close()
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Build Error",
                f"Unexpected error during build:\n\n{str(e)}"
            )

    def migrate_legacy_data(self):
        """Run the migration tool to convert legacy data."""
        from PyQt6.QtWidgets import QApplication

        reply = QMessageBox.question(
            self, "Migrate Legacy Data",
            "This will migrate data from the legacy noodleMUSH format\n"
            "to a new PROJECT_SPEC.md compliant project.\n\n"
            "Choose a location to create the new project.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        if reply != QMessageBox.StandardButton.Ok:
            return

        target = QFileDialog.getExistingDirectory(
            self, "Create Migrated Project In", os.path.expanduser("~/Documents")
        )
        if not target:
            return

        name, ok = QInputDialog.getText(
            self, "Project Name", "Name for migrated project:", text="MigratedProject"
        )
        if not ok or not name:
            return

        target_path = os.path.join(target, name)
        if os.path.exists(target_path):
            QMessageBox.warning(self, "Error", "A folder with that name already exists.")
            return

        from .project_migrator import migrate_to_project

        source_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))

        self.statusBar().showMessage("Migrating legacy data...", 0)
        QApplication.processEvents()

        success, report = migrate_to_project(source_root, target_path, dry_run=False)

        if success:
            QMessageBox.information(
                self, "Migration Complete",
                f"Successfully migrated to:\n{target_path}\n\n"
                "Open the new project to continue."
            )
            self.project_manager.open_project(target_path)
        else:
            QMessageBox.warning(
                self, "Migration Failed",
                "Migration encountered errors.\n\nSee console for details."
            )
            print(report)

        self.statusBar().clearMessage()

    # ========== RECENT PROJECTS ==========

    def get_settings_path(self) -> Path:
        """Get path to NoodleStudio settings file."""
        config_dir = Path(QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation
        ))
        config_dir = config_dir / "NoodleStudio"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "settings.json"

    def load_recent_projects(self) -> List[str]:
        """Load recent projects list from settings."""
        settings_path = self.get_settings_path()
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    return settings.get('recent_projects', [])
            except (FileNotFoundError, json.JSONDecodeError):
                return []
        return []

    def save_recent_projects(self, projects: List[str]):
        """Save recent projects list to settings."""
        settings_path = self.get_settings_path()
        settings = {}
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        settings['recent_projects'] = projects
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)

    def add_to_recent_projects(self, project_path: str):
        """Add a project to the recent projects list."""
        recent = self.load_recent_projects()
        if project_path in recent:
            recent.remove(project_path)
        recent.insert(0, project_path)
        recent = recent[:10]
        self.save_recent_projects(recent)

    def update_recent_projects_menu(self):
        """Update the Recent Projects menu with current list."""
        self.recent_projects_menu.clear()
        recent = self.load_recent_projects()

        if not recent:
            action = self.recent_projects_menu.addAction("(No recent projects)")
            action.setEnabled(False)
            return

        for project_path in recent:
            if not os.path.exists(project_path):
                continue
            project_name = os.path.basename(project_path)
            action = self.recent_projects_menu.addAction(project_name)
            action.triggered.connect(
                lambda checked, p=project_path: self.open_recent_project(p)
            )

        if recent:
            self.recent_projects_menu.addSeparator()
            clear_action = self.recent_projects_menu.addAction("Clear Recent Projects")
            clear_action.triggered.connect(self.clear_recent_projects)

    def open_recent_project(self, project_path: str):
        """Open a project from the recent projects list."""
        if os.path.exists(project_path):
            self.project_manager.open_project(project_path)
        else:
            QMessageBox.warning(
                self, "Project Not Found",
                f"Project no longer exists:\n{project_path}"
            )
            recent = self.load_recent_projects()
            if project_path in recent:
                recent.remove(project_path)
                self.save_recent_projects(recent)
                self.update_recent_projects_menu()

    def clear_recent_projects(self):
        """Clear the recent projects list."""
        self.save_recent_projects([])
        self.update_recent_projects_menu()

    def auto_open_last_project(self):
        """Always show chooser on launch (Logic Pro model)."""
        self._show_project_chooser()

    def _show_project_chooser(self):
        """Show the Project Chooser dialog (Logic Pro model: hide editor until project chosen)."""
        from ..dialogs.project_chooser_dialog import ProjectChooserDialog
        was_visible = self.isVisible()
        if was_visible:
            self.hide()
        dialog = ProjectChooserDialog(
            recent_projects=self.load_recent_projects(),
            parent=self
        )
        dialog.projectSelected.connect(self._on_project_chosen)
        dialog.exec()
        if was_visible:
            self.show()

    def _on_project_chosen(self, project_path: str):
        """Handle project selection from chooser dialog."""
        self.project_manager.open_project(project_path)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
