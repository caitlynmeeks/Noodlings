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
#   Asset Inspector Mixin - Assets panel item inspection
#
#   Handles inspection of items selected from the Assets pane...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.inspector_asset
# PURPOSE:  Inspector Asset
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AssetInspectorMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QDoubleSpinBox, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
import os
import yaml
import json


class AssetInspectorMixin:
    """
    Mixin providing asset inspection methods.

    Requires host class to have:
    - self.properties_layout (QVBoxLayout)
    - self.entity_header (QLabel)
    - self.create_property_group(title)
    - self.add_text_field(group, label, value, read_only)
    """

    # ========== MAIN DISPATCHER ==========

    def load_asset_properties(self, entity_data: dict):
        """
        Load asset properties based on asset sub-type.

        Args:
            entity_data: {'asset_type': str, 'path': str}
        """
        asset_type = entity_data.get('asset_type', 'file')
        path = entity_data.get('path', '')
        name = os.path.basename(path) if path else 'Unknown'

        # Update header with asset name
        self.entity_header.setText(name)

        # Dispatch to sub-type handler
        dispatch = {
            'folder': self._load_asset_folder,
            'noodling': self._load_asset_noodling,
            'stage': self._load_asset_stage,
            'radiance': self._load_asset_radiance,
            'vrm': self._load_asset_vrm,
            'image': self._load_asset_image,
            'audio': self._load_asset_audio,
            'script': self._load_asset_script,
            'neural_canvas': self._load_asset_neural_canvas,
            'yaml': self._load_asset_yaml,
            'zone': self._load_asset_yaml,
        }

        handler = dispatch.get(asset_type)
        if handler:
            handler(path)
        else:
            self._load_asset_generic(path, asset_type)

    # ========== ASSET TYPE HANDLERS ==========

    def _load_asset_folder(self, path: str):
        """Show folder properties."""
        group = self.create_property_group("Folder Info")

        name = os.path.basename(path)
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", "Folder", read_only=True)

        # Relative path
        rel_path = self._get_relative_path(path)
        self.add_text_field(group, "Path", rel_path, read_only=True)

        # Item count
        try:
            items = [f for f in os.listdir(path) if not f.startswith('.')]
            item_count = len(items)
        except:
            item_count = 0
        self.add_text_field(group, "Contains", f"{item_count} items", read_only=True)

        self.properties_layout.addWidget(group)
        self.properties_layout.addStretch()

    def _load_asset_noodling(self, path: str):
        """Show Noodling (recipe.yaml) properties."""
        # Find the recipe.yaml
        if os.path.isdir(path):
            recipe_path = os.path.join(path, 'recipe.yaml')
        else:
            recipe_path = path

        # Load recipe data
        recipe_data = {}
        if os.path.exists(recipe_path):
            try:
                with open(recipe_path, 'r') as f:
                    recipe_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[Inspector] Error loading recipe: {e}")

        # Basic Info
        group = self.create_property_group("Noodling Info")
        name = recipe_data.get('name', os.path.basename(os.path.dirname(recipe_path)))
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", "Noodling", read_only=True)
        self.properties_layout.addWidget(group)

        # Affect Baseline
        affect = recipe_data.get('affect_baseline', {})
        if affect:
            affect_group = self.create_property_group("Affect Baseline")
            self.add_text_field(affect_group, "Valence", f"{affect.get('valence', 0.0):.2f}", read_only=True)
            self.add_text_field(affect_group, "Arousal", f"{affect.get('arousal', 0.5):.2f}", read_only=True)
            self.add_text_field(affect_group, "Dominance", f"{affect.get('dominance', 0.5):.2f}", read_only=True)
            self.add_text_field(affect_group, "Boredom", f"{affect.get('boredom', 0.0):.2f}", read_only=True)
            self.add_text_field(affect_group, "Sorrow", f"{affect.get('sorrow', 0.0):.2f}", read_only=True)
            self.properties_layout.addWidget(affect_group)

        # Assembly Reference
        assembly_ref = recipe_data.get('facet_assembly', recipe_data.get('assembly', ''))
        if assembly_ref:
            assembly_group = self.create_property_group("Cognitive Assembly")
            self.add_text_field(assembly_group, "Assembly", assembly_ref, read_only=True)
            self.properties_layout.addWidget(assembly_group)

        # Actions
        self._add_asset_actions([
            ("Rez", lambda: self._rez_noodling(path)),
            ("Edit Recipe", lambda: self._open_in_system(recipe_path))
        ])

        self.properties_layout.addStretch()

    def _load_asset_stage(self, path: str):
        """Show Stage (stage.yaml) properties."""
        if os.path.isdir(path):
            stage_path = os.path.join(path, 'stage.yaml')
        else:
            stage_path = path

        stage_data = {}
        if os.path.exists(stage_path):
            try:
                with open(stage_path, 'r') as f:
                    stage_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[Inspector] Error loading stage: {e}")

        # Basic Info
        group = self.create_property_group("Stage Info")
        name = stage_data.get('name', os.path.basename(os.path.dirname(stage_path)))
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", "Stage", read_only=True)
        self.properties_layout.addWidget(group)

        # Statistics
        stats_group = self.create_property_group("Contents")
        stage_dir = os.path.dirname(stage_path) if os.path.isfile(stage_path) else path
        zones_dir = os.path.join(stage_dir, 'Zones')
        zone_count = 0
        instance_count = 0

        if os.path.exists(zones_dir):
            for item in os.listdir(zones_dir):
                if not item.startswith('.'):
                    zone_path = os.path.join(zones_dir, item)
                    if os.path.isdir(zone_path):
                        zone_count += 1
                        instances_dir = os.path.join(zone_path, 'Instances')
                        if os.path.exists(instances_dir):
                            instance_count += len([f for f in os.listdir(instances_dir) if not f.startswith('.')])

        self.add_text_field(stats_group, "Zones", str(zone_count), read_only=True)
        self.add_text_field(stats_group, "Instances", str(instance_count), read_only=True)
        self.properties_layout.addWidget(stats_group)

        # Actions
        self._add_asset_actions([
            ("Open Stage", lambda: self._open_stage(path))
        ])

        self.properties_layout.addStretch()

    def _load_asset_radiance(self, path: str):
        """Show .radiance file properties."""
        group = self.create_property_group("Radiance Info")

        name = os.path.basename(path)
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", "Gaussian Splat", read_only=True)
        self.add_text_field(group, "File Size", self._format_file_size(path), read_only=True)

        self.properties_layout.addWidget(group)

        # Try to load radiance metadata
        try:
            from noodlestudio.core.semantic_world.radiance_format import load_radiance
            asset = load_radiance(path)

            stats_group = self.create_property_group("Statistics")

            if hasattr(asset, 'positions') and asset.positions is not None:
                self.add_text_field(stats_group, "Gaussians", f"{len(asset.positions):,}", read_only=True)

            if hasattr(asset, 'skeleton') and asset.skeleton:
                bone_count = len(asset.skeleton.get('bones', []))
                self.add_text_field(stats_group, "Skeleton", f"{bone_count} bones", read_only=True)
            else:
                self.add_text_field(stats_group, "Skeleton", "None", read_only=True)

            if hasattr(asset, 'semantic_labels') and asset.semantic_labels:
                unique_labels = set(asset.semantic_labels)
                self.add_text_field(stats_group, "Semantic Labels", str(len(unique_labels)), read_only=True)

            has_clip = hasattr(asset, 'clip_embeddings') and asset.clip_embeddings is not None
            self.add_text_field(stats_group, "CLIP Embeddings", "Yes" if has_clip else "No", read_only=True)

            self.properties_layout.addWidget(stats_group)

        except Exception as e:
            print(f"[Inspector] Error loading radiance metadata: {e}")

        # Actions
        self._add_asset_actions([
            ("Add to Stage", lambda: self._add_to_stage(path, 'radiance')),
            ("Open in Viewer", lambda: self._open_in_gaussian_viewer(path))
        ])

        self.properties_layout.addStretch()

    def _load_asset_vrm(self, path: str):
        """Show VRM avatar model properties."""
        group = self.create_property_group("VRM Info")

        name = os.path.basename(path)
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", "Avatar Model", read_only=True)
        self.add_text_field(group, "File Size", self._format_file_size(path), read_only=True)

        self.properties_layout.addWidget(group)

        # Try to load VRM metadata
        try:
            from noodlestudio.core.semantic_world.vrm_parser import VRMParser
            parser = VRMParser()
            vrm_data = parser.parse(path)

            stats_group = self.create_property_group("Model Info")

            if 'skeleton' in vrm_data:
                bone_count = len(vrm_data['skeleton'].get('bones', []))
                self.add_text_field(stats_group, "Bones", str(bone_count), read_only=True)

            if 'materials' in vrm_data:
                mat_count = len(vrm_data['materials'])
                self.add_text_field(stats_group, "Materials", str(mat_count), read_only=True)

            self.properties_layout.addWidget(stats_group)

        except Exception as e:
            print(f"[Inspector] Error loading VRM metadata: {e}")

        # Import Settings
        settings_group = self.create_property_group("Import Settings")

        densify_cb = QCheckBox("Densify mesh")
        densify_cb.setChecked(True)
        densify_cb.setStyleSheet("color: #D2D2D2;")
        settings_group.content.layout().addRow(densify_cb)

        face_cb = QCheckBox("Add face centers")
        face_cb.setChecked(True)
        face_cb.setStyleSheet("color: #D2D2D2;")
        settings_group.content.layout().addRow(face_cb)

        scale_spin = QDoubleSpinBox()
        scale_spin.setRange(0.01, 100.0)
        scale_spin.setValue(1.0)
        scale_spin.setDecimals(2)
        scale_spin.setStyleSheet("background-color: #1E1E1E; color: #D2D2D2; padding: 4px;")
        settings_group.content.layout().addRow("Scale:", scale_spin)

        self.properties_layout.addWidget(settings_group)

        # Actions
        self._add_asset_actions([
            ("Add to Stage", lambda: self._add_to_stage(path, 'vrm')),
            ("Import as Radiance", lambda: self._import_vrm_as_radiance(
                path, densify_cb.isChecked(), face_cb.isChecked(), scale_spin.value()
            )),
            ("Preview", lambda: self._preview_vrm(path))
        ])

        self.properties_layout.addStretch()

    def _load_asset_image(self, path: str):
        """Show image file properties."""
        group = self.create_property_group("Image Info")

        name = os.path.basename(path)
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", "Image", read_only=True)
        self.add_text_field(group, "File Size", self._format_file_size(path), read_only=True)

        # Dimensions
        try:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self.add_text_field(group, "Dimensions", f"{pixmap.width()} x {pixmap.height()}", read_only=True)
        except:
            pass

        self.properties_layout.addWidget(group)

        # Thumbnail preview
        preview_group = self.create_property_group("Preview")
        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setMinimumHeight(150)
        preview_label.setStyleSheet("background-color: #1A1A1A; border: 1px solid #3A3A3A;")

        try:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                preview_label.setPixmap(scaled)
        except:
            preview_label.setText("Preview unavailable")

        preview_group.content.layout().addRow(preview_label)
        self.properties_layout.addWidget(preview_group)

        # Actions
        self._add_asset_actions([
            ("Open in System Viewer", lambda: self._open_in_system(path))
        ])

        self.properties_layout.addStretch()

    def _load_asset_audio(self, path: str):
        """Show audio file properties."""
        group = self.create_property_group("Audio Info")

        name = os.path.basename(path)
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", "Audio", read_only=True)
        self.add_text_field(group, "File Size", self._format_file_size(path), read_only=True)

        self.properties_layout.addWidget(group)

        # Audio metadata for WAV
        if path.lower().endswith('.wav'):
            try:
                import wave
                with wave.open(path, 'rb') as wav:
                    meta_group = self.create_property_group("Audio Metadata")

                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    duration = frames / float(rate)
                    self.add_text_field(meta_group, "Duration", f"{duration:.2f}s", read_only=True)
                    self.add_text_field(meta_group, "Sample Rate", f"{rate} Hz", read_only=True)
                    self.add_text_field(meta_group, "Channels", str(wav.getnchannels()), read_only=True)

                    self.properties_layout.addWidget(meta_group)
            except:
                pass

        # Actions
        self._add_asset_actions([
            ("Play", lambda: self._play_audio(path)),
            ("Stop", lambda: self._stop_audio())
        ])

        self.properties_layout.addStretch()

    def _load_asset_script(self, path: str):
        """Show script file properties."""
        group = self.create_property_group("Script Info")

        name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        lang = "Python" if ext == '.py' else "JavaScript" if ext == '.js' else "Script"

        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", lang, read_only=True)
        self.add_text_field(group, "File Size", self._format_file_size(path), read_only=True)

        try:
            with open(path, 'r') as f:
                line_count = sum(1 for _ in f)
            self.add_text_field(group, "Lines", str(line_count), read_only=True)
        except:
            pass

        self.properties_layout.addWidget(group)

        # Code preview
        preview_group = self.create_property_group("Code Preview")
        code_view = QTextEdit()
        code_view.setReadOnly(True)
        code_view.setFont(QFont("Menlo", 11))
        code_view.setStyleSheet("""
            QTextEdit {
                background-color: #1A1A1A;
                color: #D2D2D2;
                border: 1px solid #3A3A3A;
                padding: 8px;
            }
        """)
        code_view.setMinimumHeight(200)
        code_view.setMaximumHeight(300)

        try:
            with open(path, 'r') as f:
                content = f.read(8192)
                if len(content) >= 8192:
                    content += "\n\n... (truncated)"
                code_view.setPlainText(content)
        except Exception as e:
            code_view.setPlainText(f"Error reading file: {e}")

        preview_group.content.layout().addRow(code_view)
        self.properties_layout.addWidget(preview_group)

        # Actions
        self._add_asset_actions([
            ("Open in Editor", lambda: self._open_in_system(path))
        ])

        self.properties_layout.addStretch()

    def _load_asset_neural_canvas(self, path: str):
        """Show Neural Canvas (.nncanvas) properties."""
        group = self.create_property_group("Neural Canvas Info")

        name = os.path.basename(path)
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", "Neural Canvas", read_only=True)

        self.properties_layout.addWidget(group)

        # Try to load canvas metadata
        try:
            with open(path, 'r') as f:
                canvas_data = json.load(f)

            stats_group = self.create_property_group("Statistics")
            nodes = canvas_data.get('nodes', [])
            connections = canvas_data.get('connections', [])
            self.add_text_field(stats_group, "Nodes", str(len(nodes)), read_only=True)
            self.add_text_field(stats_group, "Connections", str(len(connections)), read_only=True)
            self.properties_layout.addWidget(stats_group)
        except Exception as e:
            print(f"[Inspector] Error loading neural canvas: {e}")

        # Actions
        self._add_asset_actions([
            ("Open in Editor", lambda: self._open_neural_canvas(path))
        ])

        self.properties_layout.addStretch()

    def _load_asset_yaml(self, path: str):
        """Show generic YAML file properties."""
        group = self.create_property_group("YAML File")

        name = os.path.basename(path)
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", "YAML Configuration", read_only=True)
        self.add_text_field(group, "File Size", self._format_file_size(path), read_only=True)

        self.properties_layout.addWidget(group)

        # Actions
        self._add_asset_actions([
            ("Open in Editor", lambda: self._open_in_system(path))
        ])

        self.properties_layout.addStretch()

    def _load_asset_generic(self, path: str, asset_type: str):
        """Show generic file properties for unknown types."""
        group = self.create_property_group("File Info")

        name = os.path.basename(path)
        self.add_text_field(group, "Name", name, read_only=True)
        self.add_text_field(group, "Type", asset_type.title() if asset_type else "File", read_only=True)
        self.add_text_field(group, "File Size", self._format_file_size(path), read_only=True)

        self.properties_layout.addWidget(group)

        # Actions
        self._add_asset_actions([
            ("Open", lambda: self._open_in_system(path))
        ])

        self.properties_layout.addStretch()

    # ========== ACTION HELPERS ==========

    def _add_asset_actions(self, actions: list):
        """Add action buttons to inspector."""
        actions_group = self.create_property_group("Actions")
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)

        for label, callback in actions:
            btn = QPushButton(label)
            btn.setStyleSheet(self._action_button_style())
            btn.clicked.connect(callback)
            actions_layout.addWidget(btn)

        actions_layout.addStretch()
        actions_widget = QWidget()
        actions_widget.setLayout(actions_layout)
        actions_group.content.layout().addRow(actions_widget)
        self.properties_layout.addWidget(actions_group)

    def _action_button_style(self) -> str:
        """Common style for action buttons."""
        return """
            QPushButton {
                background-color: #3a3a3a;
                color: #D2D2D2;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """

    def _format_file_size(self, path: str) -> str:
        """Format file size as human-readable string."""
        try:
            size_bytes = os.path.getsize(path)
            if size_bytes > 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
            elif size_bytes > 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes} bytes"
        except:
            return "Unknown"

    def _get_relative_path(self, path: str) -> str:
        """Get path relative to project."""
        main_window = self.window()
        if hasattr(main_window, 'project_manager') and main_window.project_manager:
            project_path = main_window.project_manager.current_project_path
            if project_path and path.startswith(project_path):
                return os.path.relpath(path, project_path)
        return path

    # ========== ACTION CALLBACKS ==========

    def _open_in_system(self, path: str):
        """Open file with system default application."""
        import subprocess
        subprocess.run(['open', path])

    def _rez_noodling(self, path: str):
        """Rez a noodling from the Assets panel."""
        main_window = self.window()
        if hasattr(main_window, 'assets') and hasattr(main_window.assets, 'agentRezzed'):
            main_window.assets.agentRezzed.emit(path)
        print(f"[Inspector] Rez requested for: {path}")

    def _open_stage(self, path: str):
        """Open a stage in the Stage View."""
        main_window = self.window()
        if hasattr(main_window, 'hierarchy'):
            stage_name = os.path.basename(path) if os.path.isdir(path) else os.path.basename(os.path.dirname(path))
            main_window.hierarchy.load_stage(stage_name)
            print(f"[Inspector] Opening stage: {stage_name}")

    def _open_in_gaussian_viewer(self, path: str):
        """Open radiance file in Gaussian Viewer."""
        main_window = self.window()
        if hasattr(main_window, 'gaussian_viewer'):
            main_window.gaussian_viewer.load_radiance_file(path)
            if hasattr(main_window, 'center_tabs'):
                for i in range(main_window.center_tabs.count()):
                    if 'Gaussian' in main_window.center_tabs.tabText(i):
                        main_window.center_tabs.setCurrentIndex(i)
                        break
        print(f"[Inspector] Opening in Gaussian Viewer: {path}")

    def _import_vrm_as_radiance(self, path: str, densify: bool, face_centers: bool, scale: float):
        """Import VRM as radiance asset."""
        base_name = os.path.splitext(os.path.basename(path))[0]
        output_dir = os.path.dirname(path)
        output_path = os.path.join(output_dir, f"{base_name}.radiance")

        try:
            from noodlestudio.tools.vrm_to_radiance import convert_vrm_to_radiance
            convert_vrm_to_radiance(
                path,
                output_path,
                densify=densify,
                face_centers=face_centers,
                scale=scale
            )
            QMessageBox.information(self, "Import Complete", f"Created: {os.path.basename(output_path)}")
            print(f"[Inspector] VRM imported as: {output_path}")
        except Exception as e:
            QMessageBox.warning(self, "Import Failed", f"Error: {e}")
            print(f"[Inspector] VRM import failed: {e}")

    def _preview_vrm(self, path: str):
        """Preview VRM in VRM Preview panel."""
        main_window = self.window()
        if hasattr(main_window, 'vrm_preview'):
            main_window.vrm_preview.load_vrm(path)
        print(f"[Inspector] VRM preview requested: {path}")

    def _add_to_stage(self, path: str, asset_type: str):
        """Add an asset to the current stage as a prop."""
        main_window = self.window()
        if hasattr(main_window, 'hierarchy') and main_window.hierarchy:
            main_window.hierarchy.add_asset_as_prop(asset_type, path)
        else:
            print(f"[Inspector] Cannot add to stage - hierarchy not available")

    def _play_audio(self, path: str):
        """Play audio file."""
        try:
            from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PyQt6.QtCore import QUrl

            if not hasattr(self, '_audio_player'):
                self._audio_player = QMediaPlayer()
                self._audio_output = QAudioOutput()
                self._audio_player.setAudioOutput(self._audio_output)

            self._audio_player.setSource(QUrl.fromLocalFile(path))
            self._audio_player.play()
            print(f"[Inspector] Playing audio: {path}")
        except Exception as e:
            print(f"[Inspector] Audio playback error: {e}")
            self._open_in_system(path)

    def _stop_audio(self):
        """Stop audio playback."""
        if hasattr(self, '_audio_player'):
            self._audio_player.stop()
            print("[Inspector] Audio stopped")

    def _open_neural_canvas(self, path: str):
        """Open neural canvas in editor."""
        main_window = self.window()
        if hasattr(main_window, 'neural_canvas'):
            main_window.neural_canvas.load_canvas(path)
            if hasattr(main_window, 'center_tabs'):
                for i in range(main_window.center_tabs.count()):
                    if 'Neural' in main_window.center_tabs.tabText(i):
                        main_window.center_tabs.setCurrentIndex(i)
                        break
        print(f"[Inspector] Opening Neural Canvas: {path}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
