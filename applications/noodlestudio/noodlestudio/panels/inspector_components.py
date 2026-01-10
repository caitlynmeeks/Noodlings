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
#   Component Inspector Mixin - Display and edit entity components
#
#   Renders components from a ComponentCollection in the Insp...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.inspector_components
# PURPOSE:  Inspector Components
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ComponentInspectorMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ComponentInspectorMixin:
    """
    Mixin providing component inspection UI generation.

    Requires host class to have:
    - self.properties_layout (QVBoxLayout)
    - CollapsibleSection imported
    """

    def create_components_section_new(self, collection: 'ComponentCollection') -> Optional[QWidget]:
        """
        Create Inspector UI for all components in a collection.

        Args:
            collection: ComponentCollection to display

        Returns:
            Container widget with all component sections, or None if empty
        """
        if not collection or len(collection) == 0:
            return None

        # Import here to avoid circular imports
        from noodlestudio.widgets.collapsible_section import CollapsibleSection
        from noodlestudio.core.component_base import CATEGORY_COLORS

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)

        # Header
        header = QLabel("Components")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(11)
        header.setFont(header_font)
        header.setStyleSheet("color: #D2D2D2; padding: 4px 0;")
        container_layout.addWidget(header)

        # Add section for each component
        for component in collection:
            section = self._create_component_section(component)
            if section:
                container_layout.addWidget(section)

        # Add Component button
        add_btn = QPushButton("+ Add Component")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #AAAAAA;
                border: 1px dashed #555555;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                color: #FFFFFF;
                border-color: #777777;
            }
        """)
        add_btn.clicked.connect(lambda: self._show_add_component_menu(collection))
        container_layout.addWidget(add_btn)

        return container

    def _create_component_section(self, component: 'ComponentBase') -> QWidget:
        """
        Create CollapsibleSection for a single component.

        Args:
            component: Component to display

        Returns:
            CollapsibleSection widget
        """
        from noodlestudio.widgets.collapsible_section import CollapsibleSection
        from noodlestudio.core.component_base import CATEGORY_COLORS

        # Create section with component name
        section = CollapsibleSection(component.display_name)

        # Apply category border color
        border_color = component.border_color
        section.header.setStyleSheet(f"""
            QFrame {{
                background-color: #3A3A3A;
                border-left: 3px solid {border_color};
                border-radius: 0px;
            }}
        """)

        # Description (if any)
        if component.description:
            desc_label = QLabel(component.description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #888888; font-size: 9pt; font-style: italic; padding: 4px 0;")
            section.add_widget(desc_label)

        # Enabled checkbox
        enabled_check = QCheckBox("Enabled")
        enabled_check.setChecked(component.enabled)
        enabled_check.setStyleSheet("color: #CCCCCC;")
        enabled_check.stateChanged.connect(
            lambda state: self._on_component_property_changed(component, 'enabled', state == Qt.CheckState.Checked.value)
        )
        section.add_widget(enabled_check)

        # Property fields from PropertySpec
        for spec in component.property_specs:
            field_widget = self._create_property_field(component, spec)
            if field_widget:
                # Create row with label
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 2, 0, 2)
                row_layout.setSpacing(8)

                label = QLabel(f"{spec.display_name}:")
                label.setStyleSheet("color: #AAAAAA;")
                label.setFixedWidth(100)
                row_layout.addWidget(label)
                row_layout.addWidget(field_widget, 1)

                section.add_widget(row)

        # Component-specific custom UI
        custom_ui = self._create_component_custom_ui(component)
        if custom_ui:
            section.add_widget(custom_ui)

        # Remove button
        remove_btn = QPushButton("Remove Component")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #AA5555;
                border: none;
                padding: 4px;
                font-size: 9pt;
            }
            QPushButton:hover {
                color: #FF6666;
                text-decoration: underline;
            }
        """)
        remove_btn.clicked.connect(lambda: self._on_remove_component(component))
        section.add_widget(remove_btn)

        return section

    def _create_property_field(self, component: 'ComponentBase', spec: 'PropertySpec') -> Optional[QWidget]:
        """
        Create appropriate widget for a PropertySpec.

        Args:
            component: Component owning the property
            spec: PropertySpec describing the property

        Returns:
            Widget for editing the property
        """
        current_value = component.get_property(spec.name)
        if current_value is None:
            current_value = spec.default

        widget = None

        if spec.property_type == 'string':
            widget = QLineEdit(str(current_value or ''))
            widget.setStyleSheet("background-color: #2A2A2A; color: #D2D2D2; border: 1px solid #3A3A3A; padding: 4px;")
            if spec.readonly:
                widget.setReadOnly(True)
                widget.setStyleSheet(widget.styleSheet() + " color: #888888;")
            else:
                widget.editingFinished.connect(
                    lambda w=widget, s=spec: self._on_component_property_changed(
                        component, s.name, w.text()
                    )
                )

        elif spec.property_type == 'text':
            widget = QTextEdit(str(current_value or ''))
            widget.setStyleSheet("background-color: #2A2A2A; color: #D2D2D2; border: 1px solid #3A3A3A; padding: 4px;")
            widget.setMaximumHeight(80)
            if spec.readonly:
                widget.setReadOnly(True)
            else:
                widget.textChanged.connect(
                    lambda w=widget, s=spec: self._on_component_property_changed(
                        component, s.name, w.toPlainText()
                    )
                )

        elif spec.property_type == 'int':
            widget = QSpinBox()
            widget.setStyleSheet("background-color: #2A2A2A; color: #D2D2D2; border: 1px solid #3A3A3A; padding: 4px;")
            if spec.min_value is not None:
                widget.setMinimum(int(spec.min_value))
            if spec.max_value is not None:
                widget.setMaximum(int(spec.max_value))
            widget.setValue(int(current_value or 0))
            if spec.readonly:
                widget.setReadOnly(True)
            else:
                widget.valueChanged.connect(
                    lambda val, s=spec: self._on_component_property_changed(component, s.name, val)
                )

        elif spec.property_type == 'float':
            widget = QDoubleSpinBox()
            widget.setStyleSheet("background-color: #2A2A2A; color: #D2D2D2; border: 1px solid #3A3A3A; padding: 4px;")
            widget.setDecimals(2)
            if spec.min_value is not None:
                widget.setMinimum(float(spec.min_value))
            if spec.max_value is not None:
                widget.setMaximum(float(spec.max_value))
            widget.setValue(float(current_value or 0.0))
            if spec.readonly:
                widget.setReadOnly(True)
            else:
                widget.valueChanged.connect(
                    lambda val, s=spec: self._on_component_property_changed(component, s.name, val)
                )

        elif spec.property_type == 'bool':
            widget = QCheckBox()
            widget.setChecked(bool(current_value))
            if spec.readonly:
                widget.setEnabled(False)
            else:
                widget.stateChanged.connect(
                    lambda state, s=spec: self._on_component_property_changed(
                        component, s.name, state == Qt.CheckState.Checked.value
                    )
                )

        elif spec.property_type == 'dropdown' and spec.options:
            widget = QComboBox()
            widget.setStyleSheet("background-color: #2A2A2A; color: #D2D2D2; border: 1px solid #3A3A3A; padding: 4px;")
            widget.addItems(spec.options)
            if current_value in spec.options:
                widget.setCurrentText(str(current_value))
            if spec.readonly:
                widget.setEnabled(False)
            else:
                widget.currentTextChanged.connect(
                    lambda val, s=spec: self._on_component_property_changed(component, s.name, val)
                )

        elif spec.property_type == 'file':
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

            path_field = QLineEdit(str(current_value or ''))
            path_field.setStyleSheet("background-color: #2A2A2A; color: #D2D2D2; border: 1px solid #3A3A3A; padding: 4px;")
            path_field.setReadOnly(True)

            browse_btn = QPushButton("...")
            browse_btn.setFixedWidth(30)
            browse_btn.setStyleSheet("background-color: #3A3A3A; color: #D2D2D2;")
            browse_btn.clicked.connect(
                lambda checked, s=spec, pf=path_field: self._browse_file(component, s, pf)
            )

            layout.addWidget(path_field, 1)
            layout.addWidget(browse_btn)

        if widget and spec.description:
            widget.setToolTip(spec.description)

        return widget

    def _create_component_custom_ui(self, component: 'ComponentBase') -> Optional[QWidget]:
        """
        Create component-specific custom UI (beyond PropertySpecs).

        For example, ArtbookComponent gets a thumbnail gallery.

        Args:
            component: Component to create UI for

        Returns:
            Custom widget, or None
        """
        # Check for ArtbookComponent
        if component.component_type == 'artbook':
            return self._create_artbook_gallery(component)

        # Check for FacetAssemblyComponent
        if component.component_type == 'facet_assembly':
            return self._create_facet_assembly_ui(component)

        return None

    def _create_artbook_gallery(self, artbook: 'ArtbookComponent') -> QWidget:
        """
        Create thumbnail gallery for ArtbookComponent.

        Args:
            artbook: ArtbookComponent to display

        Returns:
            Gallery widget
        """
        from pathlib import Path

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(4)

        # Gallery label
        gallery_label = QLabel(f"Reference Art ({artbook.art_count} images)")
        gallery_label.setStyleSheet("color: #AAAAAA; font-weight: bold;")
        layout.addWidget(gallery_label)

        # Thumbnail grid
        from PyQt6.QtCore import QSize

        gallery = QListWidget()
        gallery.setViewMode(QListWidget.ViewMode.IconMode)
        gallery.setIconSize(QSize(artbook.thumbnail_size, artbook.thumbnail_size))
        gallery.setSpacing(4)
        gallery.setMaximumHeight(200)
        gallery.setStyleSheet("""
            QListWidget {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
            }
            QListWidget::item {
                background-color: #3A3A3A;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #4A4A4A;
                border: 1px solid #FF9800;
            }
        """)

        # Add thumbnails
        for art_path in artbook.art_files:
            path = Path(art_path)
            if path.exists():
                pixmap = QPixmap(str(path))
                if not pixmap.isNull():
                    thumb = pixmap.scaled(
                        artbook.thumbnail_size, artbook.thumbnail_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    item = QListWidgetItem(QIcon(thumb), path.name)
                    item.setData(Qt.ItemDataRole.UserRole, str(path))
                    item.setToolTip(artbook.get_note(str(path)) or str(path))
                    gallery.addItem(item)
            else:
                # Missing file indicator
                item = QListWidgetItem(path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                item.setForeground(Qt.GlobalColor.red)
                item.setToolTip(f"File not found: {path}")
                gallery.addItem(item)

        layout.addWidget(gallery)

        # Add/Remove buttons
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 4, 0, 0)
        btn_layout.setSpacing(8)

        add_btn = QPushButton("+ Add Art")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #4A4A4A;
                border-radius: 3px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
        """)
        add_btn.clicked.connect(lambda: self._add_art_to_artbook(artbook, gallery))

        remove_btn = QPushButton("- Remove")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #AA7777;
                border: 1px solid #4A4A4A;
                border-radius: 3px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                color: #FF9999;
            }
        """)
        remove_btn.clicked.connect(lambda: self._remove_art_from_artbook(artbook, gallery))

        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        layout.addWidget(btn_row)

        return container

    def _create_facet_assembly_ui(self, assembly_component: 'FacetAssemblyComponent') -> QWidget:
        """
        Create custom UI for FacetAssemblyComponent.

        Shows:
        - Assembly file picker with reload button
        - Input/output pad bindings
        - Execution statistics

        Args:
            assembly_component: FacetAssemblyComponent to display

        Returns:
            Custom widget with assembly controls
        """
        from pathlib import Path

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # Store references for updates
        self._assembly_component = assembly_component
        self._assembly_stats_labels = {}

        # --- Bindings Section ---
        if assembly_component.assembly:
            # Input Bindings
            input_pads = assembly_component.input_pads
            if input_pads:
                inputs_label = QLabel("Input Bindings")
                inputs_label.setStyleSheet("color: #AAAAAA; font-weight: bold; margin-top: 8px;")
                layout.addWidget(inputs_label)

                for pad_name in input_pads:
                    row = self._create_binding_row(
                        assembly_component,
                        pad_name,
                        assembly_component._input_bindings.get(pad_name, ''),
                        is_input=True
                    )
                    layout.addWidget(row)

            # Output Bindings
            output_pads = assembly_component.output_pads
            if output_pads:
                outputs_label = QLabel("Output Bindings")
                outputs_label.setStyleSheet("color: #AAAAAA; font-weight: bold; margin-top: 8px;")
                layout.addWidget(outputs_label)

                for pad_name in output_pads:
                    row = self._create_binding_row(
                        assembly_component,
                        pad_name,
                        assembly_component._output_bindings.get(pad_name, ''),
                        is_input=False
                    )
                    layout.addWidget(row)

        # --- Statistics Section ---
        stats_label = QLabel("Statistics")
        stats_label.setStyleSheet("color: #AAAAAA; font-weight: bold; margin-top: 8px;")
        layout.addWidget(stats_label)

        stats_container = QWidget()
        stats_container.setStyleSheet("""
            QWidget {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        stats_layout = QFormLayout(stats_container)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_layout.setSpacing(4)

        stats = assembly_component.get_statistics()

        # Executions
        exec_label = QLabel(str(stats.get('execution_count', 0)))
        exec_label.setStyleSheet("color: #D2D2D2;")
        stats_layout.addRow("Executions:", exec_label)
        self._assembly_stats_labels['executions'] = exec_label

        # Total Tokens
        tokens_label = QLabel(f"{stats.get('total_tokens', 0):,}")
        tokens_label.setStyleSheet("color: #D2D2D2;")
        stats_layout.addRow("Total Tokens:", tokens_label)
        self._assembly_stats_labels['tokens'] = tokens_label

        # Last Execution Time
        last_time = stats.get('last_execution_time', 0)
        time_label = QLabel(f"{last_time:.3f}s" if last_time > 0 else "-")
        time_label.setStyleSheet("color: #D2D2D2;")
        stats_layout.addRow("Last Run:", time_label)
        self._assembly_stats_labels['last_time'] = time_label

        # Average Tokens
        avg_tokens = stats.get('avg_tokens', 0)
        avg_label = QLabel(f"{avg_tokens:.0f}" if avg_tokens > 0 else "-")
        avg_label.setStyleSheet("color: #D2D2D2;")
        stats_layout.addRow("Avg Tokens:", avg_label)
        self._assembly_stats_labels['avg_tokens'] = avg_label

        # Running status
        status_text = "Running" if stats.get('is_running') else "Idle"
        if stats.get('run_in_cognition_loop'):
            status_text = "Continuous" if stats.get('is_running') else "Continuous (paused)"
        status_label = QLabel(status_text)
        status_label.setStyleSheet(
            "color: #4CAF50;" if stats.get('is_running') else "color: #888888;"
        )
        stats_layout.addRow("Status:", status_label)
        self._assembly_stats_labels['status'] = status_label

        layout.addWidget(stats_container)

        # --- Action Buttons ---
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 8, 0, 0)
        btn_layout.setSpacing(8)

        # Run Once button (for one-shot testing)
        run_btn = QPushButton("Run Once")
        run_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A5A3A;
                color: #CCCCCC;
                border: 1px solid #4A6A4A;
                border-radius: 3px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #4A6A4A;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #2A2A2A;
                color: #666666;
            }
        """)
        run_btn.setToolTip("Execute assembly once with current inputs")
        run_btn.clicked.connect(lambda: self._run_assembly_once(assembly_component))
        run_btn.setEnabled(assembly_component.assembly is not None)
        btn_layout.addWidget(run_btn)

        # Refresh Stats button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #AAAAAA;
                border: 1px solid #4A4A4A;
                border-radius: 3px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                color: #CCCCCC;
            }
        """)
        refresh_btn.setToolTip("Refresh statistics")
        refresh_btn.clicked.connect(lambda: self._refresh_assembly_stats(assembly_component))
        btn_layout.addWidget(refresh_btn)

        btn_layout.addStretch()
        layout.addWidget(btn_row)

        return container

    def _create_binding_row(
        self,
        assembly_component: 'FacetAssemblyComponent',
        pad_name: str,
        current_binding: str,
        is_input: bool
    ) -> QWidget:
        """
        Create a row for input/output binding.

        Args:
            assembly_component: The component
            pad_name: Name of the pad
            current_binding: Current binding expression
            is_input: True for input binding, False for output

        Returns:
            Row widget with label, text field, and clear button
        """
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 2, 0, 2)
        row_layout.setSpacing(4)

        # Pad name label
        label = QLabel(f"{pad_name}:")
        label.setStyleSheet("color: #AAAAAA;")
        label.setFixedWidth(80)
        row_layout.addWidget(label)

        # Binding field
        binding_field = QLineEdit(current_binding)
        binding_field.setPlaceholderText("{component.property}" if is_input else "component.property")
        binding_field.setStyleSheet("""
            QLineEdit {
                background-color: #2A2A2A;
                color: #D2D2D2;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 4px;
            }
            QLineEdit:focus {
                border-color: #5A5A5A;
            }
        """)
        binding_field.editingFinished.connect(
            lambda: self._on_binding_changed(
                assembly_component, pad_name, binding_field.text(), is_input
            )
        )
        row_layout.addWidget(binding_field, 1)

        # Clear button
        clear_btn = QPushButton("x")
        clear_btn.setFixedSize(20, 20)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #AA5555;
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #FF6666;
            }
        """)
        clear_btn.setToolTip("Clear binding")
        clear_btn.clicked.connect(
            lambda: self._clear_binding(assembly_component, pad_name, binding_field, is_input)
        )
        row_layout.addWidget(clear_btn)

        return row

    def _on_binding_changed(
        self,
        assembly_component: 'FacetAssemblyComponent',
        pad_name: str,
        value: str,
        is_input: bool
    ):
        """Handle binding change."""
        if is_input:
            if value:
                assembly_component.bind_input(pad_name, value)
            else:
                assembly_component.unbind_input(pad_name)
        else:
            if value:
                assembly_component.bind_output(pad_name, value)
            else:
                assembly_component.unbind_output(pad_name)

        logger.debug(f"{'Input' if is_input else 'Output'} binding {pad_name} = {value}")

    def _clear_binding(
        self,
        assembly_component: 'FacetAssemblyComponent',
        pad_name: str,
        field: QLineEdit,
        is_input: bool
    ):
        """Clear a binding."""
        field.setText("")
        if is_input:
            assembly_component.unbind_input(pad_name)
        else:
            assembly_component.unbind_output(pad_name)

    def _run_assembly_once(self, assembly_component: 'FacetAssemblyComponent'):
        """Run the assembly once (for testing)."""
        import asyncio

        if not assembly_component.assembly:
            return

        async def run():
            result = await assembly_component.run({})
            # Refresh stats after run
            self._refresh_assembly_stats(assembly_component)
            return result

        # Try to get running loop or create new one
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(run())
        except RuntimeError:
            # No running loop - use Qt's event loop integration
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: asyncio.run(run()))

    def _refresh_assembly_stats(self, assembly_component: 'FacetAssemblyComponent'):
        """Refresh the statistics display."""
        if not hasattr(self, '_assembly_stats_labels'):
            return

        stats = assembly_component.get_statistics()

        if 'executions' in self._assembly_stats_labels:
            self._assembly_stats_labels['executions'].setText(str(stats.get('execution_count', 0)))

        if 'tokens' in self._assembly_stats_labels:
            self._assembly_stats_labels['tokens'].setText(f"{stats.get('total_tokens', 0):,}")

        if 'last_time' in self._assembly_stats_labels:
            last_time = stats.get('last_execution_time', 0)
            self._assembly_stats_labels['last_time'].setText(
                f"{last_time:.3f}s" if last_time > 0 else "-"
            )

        if 'avg_tokens' in self._assembly_stats_labels:
            avg_tokens = stats.get('avg_tokens', 0)
            self._assembly_stats_labels['avg_tokens'].setText(
                f"{avg_tokens:.0f}" if avg_tokens > 0 else "-"
            )

        if 'status' in self._assembly_stats_labels:
            status_text = "Running" if stats.get('is_running') else "Idle"
            if stats.get('run_in_cognition_loop'):
                status_text = "Continuous" if stats.get('is_running') else "Continuous (paused)"
            self._assembly_stats_labels['status'].setText(status_text)
            self._assembly_stats_labels['status'].setStyleSheet(
                "color: #4CAF50;" if stats.get('is_running') else "color: #888888;"
            )

    # ==========================================================================
    # Event handlers
    # ==========================================================================

    def _on_component_property_changed(self, component: 'ComponentBase', prop_name: str, value: Any):
        """Handle property change from UI."""
        old_value = component.get_property(prop_name)
        if old_value != value:
            component.set_property(prop_name, value)
            logger.debug(f"Component {component.component_type}.{prop_name} = {value}")

            # Trigger save (if host class supports it)
            if hasattr(self, '_save_component_changes'):
                self._save_component_changes(component)

    def _on_remove_component(self, component: 'ComponentBase'):
        """Handle component removal request."""
        from PyQt6.QtWidgets import QMessageBox

        result = QMessageBox.question(
            self,
            "Remove Component",
            f"Remove {component.display_name} from this entity?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if result == QMessageBox.StandardButton.Yes:
            # Signal to parent to handle removal
            if hasattr(self, '_remove_component_from_entity'):
                self._remove_component_from_entity(component)

    def _show_add_component_menu(self, collection: 'ComponentCollection'):
        """Show menu of available components to add."""
        from PyQt6.QtWidgets import QMenu
        from noodlestudio.core.component_base import component_registry, ComponentCategory

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2A2A2A;
                color: #D2D2D2;
                border: 1px solid #3A3A3A;
            }
            QMenu::item:selected {
                background-color: #4A4A4A;
            }
        """)

        # Group by category
        for category in ComponentCategory:
            types_in_category = component_registry.get_by_category(category)
            if not types_in_category:
                continue

            # Category submenu
            category_menu = menu.addMenu(category.value.title())

            for type_name in types_in_category:
                # Skip if already present (for singletons)
                if type_name in collection:
                    info = component_registry.get_display_info(type_name)
                    if info.get('singleton', True):
                        continue

                info = component_registry.get_display_info(type_name)
                action = category_menu.addAction(info.get('display_name', type_name))
                action.setData(type_name)
                action.triggered.connect(
                    lambda checked, t=type_name: self._add_component_to_entity(t, collection)
                )

        # Show menu at cursor
        menu.exec(self.cursor().pos())

    def _add_component_to_entity(self, type_name: str, collection: 'ComponentCollection'):
        """Add a new component to the current entity."""
        component = collection.add(type_name)
        if component:
            logger.info(f"Added component: {type_name}")
            # Refresh inspector
            if hasattr(self, '_refresh_components_display'):
                self._refresh_components_display()

    def _browse_file(self, component: 'ComponentBase', spec: 'PropertySpec', path_field: QLineEdit):
        """Browse for a file."""
        file_filter = spec.file_filter or "All Files (*)"
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {spec.display_name}",
            "",
            file_filter
        )
        if path:
            path_field.setText(path)
            self._on_component_property_changed(component, spec.name, path)

    def _add_art_to_artbook(self, artbook: 'ArtbookComponent', gallery: QListWidget):
        """Add art files to artbook."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Reference Art",
            "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp);;All Files (*)"
        )

        for path in paths:
            if artbook.add_art(path):
                # Add to gallery
                from pathlib import Path
                p = Path(path)
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    thumb = pixmap.scaled(
                        artbook.thumbnail_size, artbook.thumbnail_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    item = QListWidgetItem(QIcon(thumb), p.name)
                    item.setData(Qt.ItemDataRole.UserRole, path)
                    gallery.addItem(item)

    def _remove_art_from_artbook(self, artbook: 'ArtbookComponent', gallery: QListWidget):
        """Remove selected art from artbook."""
        selected = gallery.currentItem()
        if selected:
            path = selected.data(Qt.ItemDataRole.UserRole)
            if artbook.remove_art(path):
                gallery.takeItem(gallery.row(selected))


__all__ = ['ComponentInspectorMixin']

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
