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
#   Radiance Inspector - Import settings and properties for Gaussian splat assets.
#
#   Shows in the Inspector panel when a RadianceComponent is ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.radiance_inspector
# PURPOSE:  Radiance Inspector
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   RadianceInspector
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import logging
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSlider, QDoubleSpinBox, QCheckBox,
    QLineEdit, QScrollArea, QFrame, QColorDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor

logger = logging.getLogger(__name__)


class RadianceInspector(QWidget):
    """
    Inspector widget for RadianceComponent properties.

    Displays asset info and provides controls for:
    - Gaussian scale
    - Material tint and emission
    - Region visibility
    - Export options
    """

    # Emitted when any property changes
    propertyChanged = pyqtSignal()
    # Emitted when user requests focus on a bone (bone_name, position_xyz)
    focusBoneRequested = pyqtSignal(str, tuple)
    # Emitted when user selects a bone from dropdown (bone_name)
    boneSelected = pyqtSignal(str)
    # Emitted when inspector wants to transfer keyboard focus to viewer
    requestViewerFocus = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._component = None
        self._on_change_callback: Optional[Callable] = None
        self._updating = False
        self._updating_bone_combo = False  # Prevent signal loops

        # Debounce timer for change notifications (prevents crashes from rapid updates)
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(100)  # 100ms debounce
        self._debounce_timer.timeout.connect(self._emit_debounced_change)

        self._setup_ui()

    def _setup_ui(self):
        """Build the inspector UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header = QLabel("Radiance Asset")
        header.setStyleSheet("font-weight: bold; font-size: 13px; color: #cc66cc;")
        layout.addWidget(header)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        # === File Info ===
        file_group = QGroupBox("File Info")
        file_layout = QFormLayout(file_group)
        file_layout.setSpacing(4)

        self.name_label = QLabel("-")
        file_layout.addRow("Name:", self.name_label)

        self.path_label = QLabel("-")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #888; font-size: 10px;")
        file_layout.addRow("Path:", self.path_label)

        self.size_label = QLabel("-")
        file_layout.addRow("Size:", self.size_label)

        content_layout.addWidget(file_group)

        # === Gaussian Stats ===
        stats_group = QGroupBox("Gaussian Data")
        stats_layout = QFormLayout(stats_group)
        stats_layout.setSpacing(4)

        self.count_label = QLabel("-")
        stats_layout.addRow("Gaussians:", self.count_label)

        self.skeleton_label = QLabel("-")
        stats_layout.addRow("Skeleton:", self.skeleton_label)

        self.regions_label = QLabel("-")
        self.regions_label.setWordWrap(True)
        stats_layout.addRow("Body Regions:", self.regions_label)

        content_layout.addWidget(stats_group)

        # === Skeleton / Bones ===
        skeleton_group = QGroupBox("Skeleton")
        skeleton_layout = QFormLayout(skeleton_group)
        skeleton_layout.setSpacing(6)

        # Bone selector dropdown
        from PyQt6.QtWidgets import QComboBox
        bone_layout = QHBoxLayout()
        self.bone_combo = QComboBox()
        self.bone_combo.setMinimumWidth(120)
        self.bone_combo.addItem("(none)")
        bone_layout.addWidget(self.bone_combo)

        self.focus_bone_btn = QPushButton("Focus")
        self.focus_bone_btn.setFixedWidth(50)
        self.focus_bone_btn.clicked.connect(self._focus_on_bone)
        bone_layout.addWidget(self.focus_bone_btn)

        skeleton_layout.addRow("Bone:", bone_layout)

        # Bone details (shown when bone is selected)
        self.bone_position_label = QLabel("-")
        self.bone_position_label.setStyleSheet("color: #888;")
        skeleton_layout.addRow("Position:", self.bone_position_label)

        self.bone_parent_label = QLabel("-")
        self.bone_parent_label.setStyleSheet("color: #888;")
        skeleton_layout.addRow("Parent:", self.bone_parent_label)

        self.bone_children_label = QLabel("-")
        self.bone_children_label.setStyleSheet("color: #888;")
        skeleton_layout.addRow("Children:", self.bone_children_label)

        # Connect combo change to update bone details
        self.bone_combo.currentTextChanged.connect(self._on_bone_selected)

        content_layout.addWidget(skeleton_group)

        # === Display Settings ===
        display_group = QGroupBox("Display")
        display_layout = QFormLayout(display_group)
        display_layout.setSpacing(6)

        # Gaussian Scale
        scale_layout = QHBoxLayout()
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(50)
        self.scale_slider.setValue(3)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.scale_slider)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 50.0)
        self.scale_spin.setValue(3.0)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setFixedWidth(70)
        self.scale_spin.valueChanged.connect(self._on_scale_spin_changed)
        scale_layout.addWidget(self.scale_spin)

        display_layout.addRow("Scale:", scale_layout)

        # Alpha
        alpha_layout = QHBoxLayout()
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(100)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        alpha_layout.addWidget(self.alpha_slider)

        self.alpha_label = QLabel("100%")
        self.alpha_label.setFixedWidth(40)
        alpha_layout.addWidget(self.alpha_label)

        display_layout.addRow("Alpha:", alpha_layout)

        # Sharpness (inverse of Gaussian scale - higher = crisper edges)
        sharpness_layout = QHBoxLayout()
        self.sharpness_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpness_slider.setMinimum(0)
        self.sharpness_slider.setMaximum(100)
        self.sharpness_slider.setValue(50)  # 50% = normal
        self.sharpness_slider.valueChanged.connect(self._on_sharpness_changed)
        sharpness_layout.addWidget(self.sharpness_slider)

        self.sharpness_label = QLabel("50%")
        self.sharpness_label.setFixedWidth(40)
        sharpness_layout.addWidget(self.sharpness_label)

        display_layout.addRow("Sharpness:", sharpness_layout)

        content_layout.addWidget(display_group)

        # === Material ===
        material_group = QGroupBox("Material")
        material_layout = QFormLayout(material_group)
        material_layout.setSpacing(6)

        # Tint color
        tint_layout = QHBoxLayout()
        self.tint_btn = QPushButton("")
        self.tint_btn.setFixedSize(24, 24)
        self.tint_btn.setStyleSheet("background-color: #ffffff; border: 1px solid #555;")
        self.tint_btn.clicked.connect(self._pick_tint_color)
        tint_layout.addWidget(self.tint_btn)

        self.tint_label = QLabel("1.0, 1.0, 1.0")
        self.tint_label.setStyleSheet("color: #888;")
        tint_layout.addWidget(self.tint_label)
        tint_layout.addStretch()

        material_layout.addRow("Tint:", tint_layout)

        # Emission color
        emission_layout = QHBoxLayout()
        self.emission_btn = QPushButton("")
        self.emission_btn.setFixedSize(24, 24)
        self.emission_btn.setStyleSheet("background-color: #000000; border: 1px solid #555;")
        self.emission_btn.clicked.connect(self._pick_emission_color)
        emission_layout.addWidget(self.emission_btn)

        self.emission_label = QLabel("0.0, 0.0, 0.0")
        self.emission_label.setStyleSheet("color: #888;")
        emission_layout.addWidget(self.emission_label)
        emission_layout.addStretch()

        material_layout.addRow("Emission:", emission_layout)

        content_layout.addWidget(material_group)

        # === Actions ===
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(4)

        show_folder_btn = QPushButton("Show in Folder")
        show_folder_btn.clicked.connect(self._show_in_folder)
        actions_layout.addWidget(show_folder_btn)

        reload_btn = QPushButton("Reload Asset")
        reload_btn.clicked.connect(self._reload_asset)
        actions_layout.addWidget(reload_btn)

        content_layout.addWidget(actions_group)

        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Style
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #d2d2d2;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QSlider::groove:horizontal {
                background-color: #3a3a3a;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background-color: #6a6a6a;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background-color: #8a8a8a;
            }
            QDoubleSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                padding: 2px;
            }
        """)

    def set_component(self, component, path: str = None):
        """Set the RadianceComponent to inspect."""
        self._component = component

        if component is None:
            self._clear()
            return

        # Update file info
        self.name_label.setText(component.entity_id)
        if path:
            self.path_label.setText(str(path))
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                self.size_label.setText(f"{size_mb:.2f} MB")
            else:
                self.size_label.setText("-")
        else:
            self.path_label.setText("-")
            self.size_label.setText("-")

        # Update stats
        self.count_label.setText(f"{component.gaussian_count:,}")
        # Show skeleton status with bone count
        if component.has_skeleton:
            bone_count = len(component.bone_names) if component.bone_names else 0
            self.skeleton_label.setText(f"Yes ({bone_count} bones)")
        else:
            self.skeleton_label.setText("No")

        # Populate bone dropdown
        self.bone_combo.clear()
        self.bone_combo.addItem("(none)")
        if component.bone_names:
            for bone_name in component.bone_names:
                self.bone_combo.addItem(bone_name)

        regions = component.body_regions
        if regions:
            self.regions_label.setText(", ".join(sorted(regions)))
        else:
            self.regions_label.setText("-")

        # Update display controls
        self._updating = True
        self.scale_slider.setValue(int(component.material.scale_mult))
        self.scale_spin.setValue(component.material.scale_mult)
        self.alpha_slider.setValue(int(component.material.alpha_mult * 100))
        self.alpha_label.setText(f"{int(component.material.alpha_mult * 100)}%")

        # Update sharpness slider (reverse of _on_sharpness_changed)
        sharpness_mult = getattr(component.material, 'sharpness_mult', 1.0)
        if sharpness_mult >= 1.0:
            # 2.0-1.0 maps to 0-50
            sharpness_value = int((2.0 - sharpness_mult) * 50.0)
        else:
            # 1.0-0.5 maps to 50-100
            sharpness_value = int(50 + (1.0 - sharpness_mult) * 100.0)
        sharpness_value = max(0, min(100, sharpness_value))
        self.sharpness_slider.setValue(sharpness_value)
        self.sharpness_label.setText(f"{sharpness_value}%")

        # Update tint button
        tint = component.material.tint
        self.tint_btn.setStyleSheet(
            f"background-color: rgb({int(tint.r*255)},{int(tint.g*255)},{int(tint.b*255)}); border: 1px solid #555;"
        )
        self.tint_label.setText(f"{tint.r:.2f}, {tint.g:.2f}, {tint.b:.2f}")

        # Update emission button
        em = component.material.emission
        self.emission_btn.setStyleSheet(
            f"background-color: rgb({int(em.r*255)},{int(em.g*255)},{int(em.b*255)}); border: 1px solid #555;"
        )
        self.emission_label.setText(f"{em.r:.2f}, {em.g:.2f}, {em.b:.2f}")

        self._updating = False

    def _clear(self):
        """Clear all fields."""
        self.name_label.setText("-")
        self.path_label.setText("-")
        self.size_label.setText("-")
        self.count_label.setText("-")
        self.skeleton_label.setText("-")
        self.regions_label.setText("-")
        self.bone_combo.clear()
        self.bone_combo.addItem("(none)")

    def set_on_change_callback(self, callback: Callable):
        """Set callback for property changes."""
        self._on_change_callback = callback

    def _notify_change(self):
        """Notify listeners of property change (debounced to prevent crashes)."""
        self.propertyChanged.emit()
        # Debounce the callback to prevent rapid re-renders during typing
        self._debounce_timer.start()

    def _emit_debounced_change(self):
        """Actually emit the change callback after debounce period."""
        if self._on_change_callback:
            self._on_change_callback()

    # =========================================================================
    # Property Change Handlers
    # =========================================================================

    def _on_scale_changed(self, value: int):
        """Handle scale slider change."""
        if self._updating:
            return
        if self._component:
            self._component.material.scale_mult = float(value)
            self._updating = True
            self.scale_spin.setValue(float(value))
            self._updating = False
            self._notify_change()

    def _on_scale_spin_changed(self, value: float):
        """Handle scale spinbox change."""
        if self._updating:
            return
        if self._component:
            self._component.material.scale_mult = value
            self._updating = True
            self.scale_slider.setValue(int(value))
            self._updating = False
            self._notify_change()

    def _on_alpha_changed(self, value: int):
        """Handle alpha slider change."""
        if self._updating:
            return
        if self._component:
            self._component.material.alpha_mult = value / 100.0
            self.alpha_label.setText(f"{value}%")
            self._notify_change()

    def _on_sharpness_changed(self, value: int):
        """Handle sharpness slider change.

        0% = very soft (sharpness_mult = 2.0)
        50% = normal (sharpness_mult = 1.0)
        100% = very sharp (sharpness_mult = 0.5)
        """
        if self._updating:
            return
        if self._component:
            # Convert slider 0-100 to sharpness_mult
            # 0 -> 2.0, 50 -> 1.0, 100 -> 0.5
            if value <= 50:
                # 0-50 maps to 2.0-1.0
                self._component.material.sharpness_mult = 2.0 - (value / 50.0)
            else:
                # 50-100 maps to 1.0-0.5
                self._component.material.sharpness_mult = 1.0 - (value - 50) / 100.0
            self.sharpness_label.setText(f"{value}%")
            self._notify_change()

    def _pick_tint_color(self):
        """Open color picker for tint."""
        if not self._component:
            return

        tint = self._component.material.tint
        initial = QColor(int(tint.r*255), int(tint.g*255), int(tint.b*255))

        color = QColorDialog.getColor(initial, self, "Select Tint Color")
        if color.isValid():
            from noodlestudio.core.radiance_component import Color
            self._component.material.tint = Color(
                color.redF(), color.greenF(), color.blueF()
            )
            self.tint_btn.setStyleSheet(
                f"background-color: {color.name()}; border: 1px solid #555;"
            )
            self.tint_label.setText(
                f"{color.redF():.2f}, {color.greenF():.2f}, {color.blueF():.2f}"
            )
            self._notify_change()

    def _pick_emission_color(self):
        """Open color picker for emission."""
        if not self._component:
            return

        em = self._component.material.emission
        initial = QColor(int(em.r*255), int(em.g*255), int(em.b*255))

        color = QColorDialog.getColor(initial, self, "Select Emission Color")
        if color.isValid():
            from noodlestudio.core.radiance_component import Color
            self._component.material.emission = Color(
                color.redF(), color.greenF(), color.blueF()
            )
            self.emission_btn.setStyleSheet(
                f"background-color: {color.name()}; border: 1px solid #555;"
            )
            self.emission_label.setText(
                f"{color.redF():.2f}, {color.greenF():.2f}, {color.blueF():.2f}"
            )
            self._notify_change()

    # =========================================================================
    # Actions
    # =========================================================================

    def _show_in_folder(self):
        """Open the containing folder."""
        path = self.path_label.text()
        if path and path != "-" and os.path.exists(path):
            folder = os.path.dirname(path)
            import subprocess
            import sys
            if sys.platform == 'darwin':
                subprocess.run(['open', folder])
            elif sys.platform == 'win32':
                os.startfile(folder)
            else:
                subprocess.run(['xdg-open', folder])

    def _reload_asset(self):
        """Reload the asset from disk."""
        path = self.path_label.text()
        if path and path != "-" and self._component:
            if self._component.load_asset(path):
                logger.info(f"Reloaded: {path}")
                self._notify_change()

    def _focus_on_bone(self):
        """Focus camera on selected bone."""
        bone_name = self.bone_combo.currentText()
        if bone_name == "(none)" or not self._component:
            return

        # Get bone list from component
        bone_names = self._component.bone_names
        if not bone_names:
            logger.warning("No bones available")
            return

        try:
            bone_idx = bone_names.index(bone_name)
        except ValueError:
            logger.warning(f"Bone '{bone_name}' not found in bone list")
            return

        # Get skeleton from asset
        asset = self._component.asset
        if not asset or not asset.skeleton:
            logger.warning("No skeleton available")
            return

        skeleton = asset.skeleton
        if bone_idx >= len(skeleton.bones):
            logger.warning(f"Bone index {bone_idx} out of range")
            return

        # Compute world position by traversing parent chain
        # Each bone.position is in local space relative to parent
        world_pos = [0.0, 0.0, 0.0]
        current_idx = bone_idx

        while current_idx >= 0 and current_idx < len(skeleton.bones):
            bone = skeleton.bones[current_idx]
            # Add this bone's local position
            world_pos[0] += bone.position[0]
            world_pos[1] += bone.position[1]
            world_pos[2] += bone.position[2]
            # Move to parent
            current_idx = bone.parent_index
            # Safety: prevent infinite loop
            if current_idx == bone_idx:
                break

        pos = (world_pos[0], world_pos[1], world_pos[2])
        logger.info(f"Focusing on bone '{bone_name}' at {pos}")
        self.focusBoneRequested.emit(bone_name, pos)
        # Also request viewer to take keyboard focus so F key works immediately
        self.requestViewerFocus.emit()

    def set_selected_bone(self, bone_name: str):
        """Set the selected bone in the dropdown (called from viewer).

        Uses a flag to prevent signal loops.
        """
        self._updating_bone_combo = True
        try:
            if not bone_name:
                # Deselect - set to "(none)"
                self.bone_combo.setCurrentIndex(0)
            else:
                # Find and select the bone
                idx = self.bone_combo.findText(bone_name)
                if idx >= 0:
                    self.bone_combo.setCurrentIndex(idx)
        finally:
            self._updating_bone_combo = False

    def _on_bone_selected(self, bone_name: str):
        """Update bone details when selection changes."""
        # Emit signal to viewer if this was a manual selection (not programmatic)
        if not self._updating_bone_combo:
            self.boneSelected.emit(bone_name if bone_name != "(none)" else "")

        if bone_name == "(none)" or not self._component:
            self.bone_position_label.setText("-")
            self.bone_parent_label.setText("-")
            self.bone_children_label.setText("-")
            return

        asset = self._component.asset
        if not asset or not asset.skeleton:
            return

        skeleton = asset.skeleton
        bone_names = [b.name for b in skeleton.bones]

        try:
            bone_idx = bone_names.index(bone_name)
        except ValueError:
            return

        bone = skeleton.bones[bone_idx]

        # Position (local)
        pos = bone.position
        self.bone_position_label.setText(f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        # Parent
        if bone.parent_index >= 0 and bone.parent_index < len(skeleton.bones):
            parent_name = skeleton.bones[bone.parent_index].name
            self.bone_parent_label.setText(parent_name)
        else:
            self.bone_parent_label.setText("(root)")

        # Children
        children = [b.name for i, b in enumerate(skeleton.bones) if b.parent_index == bone_idx]
        if children:
            self.bone_children_label.setText(", ".join(children[:3]) + ("..." if len(children) > 3 else ""))
        else:
            self.bone_children_label.setText("(none)")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
