"""
Radiance Inspector - Import settings and properties for Gaussian splat assets.

Shows in the Inspector panel when a RadianceComponent is selected.
Like Unity's Model Importer inspector.

Author: Caitlyn + Claude (NinaK)
Date: December 2025
"""

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

    def __init__(self, parent=None):
        super().__init__(parent)

        self._component = None
        self._on_change_callback: Optional[Callable] = None
        self._updating = False

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
        self.scale_spin.setRange(0.1, 50.0)
        self.scale_spin.setValue(3.0)
        self.scale_spin.setDecimals(1)
        self.scale_spin.setSingleStep(0.5)
        self.scale_spin.setFixedWidth(60)
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
        self.skeleton_label.setText("Yes" if component.has_skeleton else "No")

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
