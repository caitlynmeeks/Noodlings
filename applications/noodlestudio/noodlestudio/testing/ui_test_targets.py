# ──────────────────────────────────────────────────────────────
#
#   UI Test Targets - Resolve target specifications to coordinates
#
#   Finds UI elements by various selectors (menu, panel, button, etc.)
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.testing.ui_test_targets
# PURPOSE:  Target resolution for UI tests
# LAYER:    Studio / Testing
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, Any, Tuple, Optional
from PyQt6.QtWidgets import (
    QWidget, QMainWindow, QMenuBar, QMenu,
    QPushButton, QLineEdit, QTextEdit, QLabel,
    QDockWidget, QTabWidget, QTreeView, QListView,
    QGraphicsView, QApplication
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QPoint, QRect


class UITestTargetResolver:
    """
    Resolves target specifications to screen coordinates.

    Supports various target types:
    - menu: Menu bar items
    - menu_item: Items in open menus
    - panel: Dock panels
    - button: Push buttons
    - field: Input fields
    - facet_node: Facet nodes in Facets panel
    - etc.
    """

    def __init__(self, main_window: QMainWindow):
        self.window = main_window

    async def resolve(self, target: Dict[str, Any]) -> Tuple[int, int]:
        """
        Resolve a target specification to (x, y) coordinates.

        Args:
            target: Target specification dict

        Returns:
            Tuple of (x, y) global screen coordinates
        """
        # Direct coordinates
        if 'x' in target and 'y' in target:
            return (target['x'], target['y'])

        # Menu bar
        if 'menu' in target:
            return await self._resolve_menu(target['menu'])

        # Menu item (in open menu)
        if 'menu_item' in target:
            return await self._resolve_menu_item(target['menu_item'])

        # Panel
        if 'panel' in target:
            return await self._resolve_panel(target['panel'], target.get('area', 'center'))

        # Panel tab
        if 'panel_tab' in target:
            return await self._resolve_panel_tab(target['panel_tab'])

        # Dialog
        if 'dialog' in target:
            return await self._resolve_dialog(target['dialog'])

        # Button
        if 'button' in target:
            return await self._resolve_button(target['button'])

        # Field (QLineEdit, QTextEdit)
        if 'field' in target:
            return await self._resolve_field(target['field'])

        # Inspector field
        if 'inspector_field' in target:
            return await self._resolve_inspector_field(target['inspector_field'])

        # Facet node
        if 'facet_node' in target:
            return await self._resolve_facet_node(target['facet_node'])

        # Facet pad
        if 'facet_pad' in target:
            return await self._resolve_facet_pad(target['facet_pad'])

        # Hierarchy item
        if 'item_in_hierarchy' in target:
            return await self._resolve_hierarchy_item(target['item_in_hierarchy'])

        # Noodling in stage
        if 'noodling_in_stage' in target:
            return await self._resolve_noodling_in_stage(target['noodling_in_stage'])

        # Chat input
        if 'chat_input' in target:
            return await self._resolve_chat_input()

        # Generic element by name
        if 'element' in target:
            return await self._resolve_element_by_name(target['element'])

        raise ValueError(f"Unknown target specification: {target}")

    # ═══════════════════════════════════════════════════════════
    # Menu Resolution
    # ═══════════════════════════════════════════════════════════

    async def _resolve_menu(self, menu_name: str) -> Tuple[int, int]:
        """Find a menu in the menu bar."""
        menubar = self.window.menuBar()
        if not menubar:
            raise ValueError("No menu bar found")

        for action in menubar.actions():
            if action.text().replace('&', '') == menu_name:
                # Get the geometry of this menu item
                rect = menubar.actionGeometry(action)
                center = rect.center()
                global_pos = menubar.mapToGlobal(center)
                return (global_pos.x(), global_pos.y())

        raise ValueError(f"Menu not found: {menu_name}")

    async def _resolve_menu_item(self, item_name: str) -> Tuple[int, int]:
        """Find an item in the currently open menu."""
        # Find active popup menu
        app = QApplication.instance()
        for widget in app.topLevelWidgets():
            if isinstance(widget, QMenu) and widget.isVisible():
                for action in widget.actions():
                    text = action.text().replace('&', '')
                    if text == item_name or item_name in text:
                        rect = widget.actionGeometry(action)
                        center = rect.center()
                        global_pos = widget.mapToGlobal(center)
                        return (global_pos.x(), global_pos.y())

        raise ValueError(f"Menu item not found: {item_name}")

    # ═══════════════════════════════════════════════════════════
    # Panel Resolution
    # ═══════════════════════════════════════════════════════════

    async def _resolve_panel(self, panel_name: str, area: str = 'center') -> Tuple[int, int]:
        """Find a panel by name."""
        panel_map = {
            'stage': ['stage_panel', 'StagePanel', 'stage'],
            'hierarchy': ['hierarchy_panel', 'HierarchyPanel', 'scene_hierarchy'],
            'facets': ['facets_panel', 'FacetsEditorPanel', 'facets_editor'],
            'inspector': ['inspector_panel', 'InspectorPanel', 'inspector'],
            'neural_canvas': ['neural_canvas_panel', 'NeuralCanvasPanel', 'nncanvas'],
            'chat': ['chat_panel', 'ChatPanel', 'conversation'],
            'console': ['console_panel', 'ConsolePanel', 'console'],
            'noodlecode': ['noodle_code_panel', 'NoodleCodePanel', 'noodlecode'],
        }

        # Get possible attribute names
        attr_names = panel_map.get(panel_name.lower(), [panel_name])

        panel = None
        for attr in attr_names:
            if hasattr(self.window, attr):
                panel = getattr(self.window, attr)
                break

        if panel is None:
            # Try finding by object name
            panel = self.window.findChild(QWidget, panel_name)

        if panel is None:
            raise ValueError(f"Panel not found: {panel_name}")

        # Get position based on area
        rect = panel.rect()
        if area == 'center':
            pos = rect.center()
        elif area == 'top':
            pos = QPoint(rect.center().x(), rect.top() + 20)
        elif area == 'bottom':
            pos = QPoint(rect.center().x(), rect.bottom() - 20)
        elif area == 'left':
            pos = QPoint(rect.left() + 20, rect.center().y())
        elif area == 'right':
            pos = QPoint(rect.right() - 20, rect.center().y())
        elif area == 'empty':
            # Try to find empty area (for context menus)
            pos = QPoint(rect.center().x(), rect.center().y())
        else:
            pos = rect.center()

        global_pos = panel.mapToGlobal(pos)
        return (global_pos.x(), global_pos.y())

    async def _resolve_panel_tab(self, tab_name: str) -> Tuple[int, int]:
        """Find a tab in tab widgets."""
        # Look for QTabWidget children
        for tab_widget in self.window.findChildren(QTabWidget):
            for i in range(tab_widget.count()):
                if tab_widget.tabText(i) == tab_name:
                    # Get tab bar position
                    tab_bar = tab_widget.tabBar()
                    rect = tab_bar.tabRect(i)
                    center = rect.center()
                    global_pos = tab_bar.mapToGlobal(center)
                    return (global_pos.x(), global_pos.y())

        raise ValueError(f"Tab not found: {tab_name}")

    # ═══════════════════════════════════════════════════════════
    # Dialog Resolution
    # ═══════════════════════════════════════════════════════════

    async def _resolve_dialog(self, dialog_name: str) -> Tuple[int, int]:
        """Find an open dialog."""
        app = QApplication.instance()
        for widget in app.topLevelWidgets():
            if widget.isVisible() and widget != self.window:
                title = widget.windowTitle()
                if dialog_name in title or title in dialog_name:
                    rect = widget.rect()
                    center = rect.center()
                    global_pos = widget.mapToGlobal(center)
                    return (global_pos.x(), global_pos.y())

        raise ValueError(f"Dialog not found: {dialog_name}")

    async def _resolve_button(self, button_text: str) -> Tuple[int, int]:
        """Find a button by text."""
        # Search in active dialog first
        app = QApplication.instance()
        search_widgets = [self.window]

        for widget in app.topLevelWidgets():
            if widget.isVisible() and widget != self.window:
                search_widgets.insert(0, widget)  # Prioritize dialogs

        for parent in search_widgets:
            for button in parent.findChildren(QPushButton):
                if button.text() == button_text and button.isVisible():
                    rect = button.rect()
                    center = rect.center()
                    global_pos = button.mapToGlobal(center)
                    return (global_pos.x(), global_pos.y())

        raise ValueError(f"Button not found: {button_text}")

    async def _resolve_field(self, field_name: str) -> Tuple[int, int]:
        """Find an input field by label or placeholder."""
        # Search dialogs first, then main window
        app = QApplication.instance()
        search_widgets = [self.window]

        for widget in app.topLevelWidgets():
            if widget.isVisible() and widget != self.window:
                search_widgets.insert(0, widget)

        for parent in search_widgets:
            # Look for QLineEdit
            for field in parent.findChildren(QLineEdit):
                if not field.isVisible():
                    continue

                # Check placeholder text
                if field.placeholderText() and field_name.lower() in field.placeholderText().lower():
                    return self._widget_center(field)

                # Check object name
                if field_name.lower() in field.objectName().lower():
                    return self._widget_center(field)

                # Check associated label
                label = self._find_label_for_widget(parent, field)
                if label and field_name.lower() in label.text().lower():
                    return self._widget_center(field)

            # Look for QTextEdit
            for field in parent.findChildren(QTextEdit):
                if not field.isVisible():
                    continue

                if field_name.lower() in field.objectName().lower():
                    return self._widget_center(field)

                label = self._find_label_for_widget(parent, field)
                if label and field_name.lower() in label.text().lower():
                    return self._widget_center(field)

        raise ValueError(f"Field not found: {field_name}")

    # ═══════════════════════════════════════════════════════════
    # Inspector Resolution
    # ═══════════════════════════════════════════════════════════

    async def _resolve_inspector_field(self, field_name: str) -> Tuple[int, int]:
        """Find a field in the Inspector panel."""
        inspector = getattr(self.window, 'inspector_panel', None)
        if inspector is None:
            inspector = self.window.findChild(QWidget, 'inspector')

        if inspector is None:
            raise ValueError("Inspector panel not found")

        # Look for field by label
        for field in inspector.findChildren((QLineEdit, QTextEdit)):
            if not field.isVisible():
                continue

            label = self._find_label_for_widget(inspector, field)
            if label and field_name.lower() in label.text().lower().replace(':', ''):
                return self._widget_center(field)

            if field_name.lower() in field.objectName().lower():
                return self._widget_center(field)

        raise ValueError(f"Inspector field not found: {field_name}")

    # ═══════════════════════════════════════════════════════════
    # Facets Resolution
    # ═══════════════════════════════════════════════════════════

    async def _resolve_facet_node(self, node_name: str) -> Tuple[int, int]:
        """Find a facet node in the Facets editor."""
        facets_panel = getattr(self.window, 'facets_panel', None)
        if facets_panel is None:
            facets_panel = self.window.findChild(QWidget, 'facets_editor')

        if facets_panel is None:
            raise ValueError("Facets panel not found")

        # Access the node graphics
        if hasattr(facets_panel, 'node_graphics'):
            for facet_id, node_gfx in facets_panel.node_graphics.items():
                if hasattr(node_gfx, 'facet'):
                    if node_gfx.facet.name == node_name or node_name in node_gfx.facet.name:
                        # Get scene position and convert to global
                        scene_pos = node_gfx.scenePos()
                        view = facets_panel.view
                        view_pos = view.mapFromScene(scene_pos.x(), scene_pos.y())
                        global_pos = view.viewport().mapToGlobal(view_pos)
                        return (global_pos.x(), global_pos.y())

        raise ValueError(f"Facet node not found: {node_name}")

    async def _resolve_facet_pad(self, pad_spec: str) -> Tuple[int, int]:
        """
        Find a facet pad.

        pad_spec format: "FacetName.pad_name"
        """
        parts = pad_spec.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid pad spec: {pad_spec} (expected 'FacetName.pad_name')")

        facet_name, pad_name = parts

        facets_panel = getattr(self.window, 'facets_panel', None)
        if facets_panel is None:
            raise ValueError("Facets panel not found")

        if hasattr(facets_panel, 'node_graphics'):
            for facet_id, node_gfx in facets_panel.node_graphics.items():
                if hasattr(node_gfx, 'facet') and node_gfx.facet.name == facet_name:
                    # Find the pad
                    pad_gfx = None

                    if hasattr(node_gfx, 'input_pads') and pad_name in node_gfx.input_pads:
                        pad_gfx = node_gfx.input_pads[pad_name]
                    elif hasattr(node_gfx, 'output_pads') and pad_name in node_gfx.output_pads:
                        pad_gfx = node_gfx.output_pads[pad_name]

                    if pad_gfx:
                        scene_pos = pad_gfx.scenePos()
                        view = facets_panel.view
                        view_pos = view.mapFromScene(scene_pos.x(), scene_pos.y())
                        global_pos = view.viewport().mapToGlobal(view_pos)
                        return (global_pos.x(), global_pos.y())

        raise ValueError(f"Facet pad not found: {pad_spec}")

    # ═══════════════════════════════════════════════════════════
    # Hierarchy/Stage Resolution
    # ═══════════════════════════════════════════════════════════

    async def _resolve_hierarchy_item(self, item_name: str) -> Tuple[int, int]:
        """Find an item in the scene hierarchy."""
        hierarchy = getattr(self.window, 'hierarchy', None)
        if hierarchy is None:
            hierarchy = self.window.findChild(QWidget, 'scene_hierarchy')

        if hierarchy is None:
            raise ValueError("Hierarchy panel not found")

        # This would need to search the tree model
        # For now, return center of hierarchy panel
        return self._widget_center(hierarchy)

    async def _resolve_noodling_in_stage(self, noodling_name: str) -> Tuple[int, int]:
        """Find a noodling in the Stage view."""
        stage = getattr(self.window, 'stage_panel', None)
        if stage is None:
            stage = self.window.findChild(QWidget, 'stage')

        if stage is None:
            raise ValueError("Stage panel not found")

        # This would need to search stage for noodling graphics
        # For now, return center of stage
        return self._widget_center(stage)

    # ═══════════════════════════════════════════════════════════
    # Chat Resolution
    # ═══════════════════════════════════════════════════════════

    async def _resolve_chat_input(self) -> Tuple[int, int]:
        """Find the chat input field."""
        chat = getattr(self.window, 'chat_panel', None)
        if chat is None:
            chat = self.window.findChild(QWidget, 'chat')

        if chat:
            # Look for input field
            for field in chat.findChildren((QLineEdit, QTextEdit)):
                if field.isVisible():
                    return self._widget_center(field)

        raise ValueError("Chat input not found")

    # ═══════════════════════════════════════════════════════════
    # Generic Resolution
    # ═══════════════════════════════════════════════════════════

    async def _resolve_element_by_name(self, element_name: str) -> Tuple[int, int]:
        """Find an element by object name or text."""
        # Search by object name
        widget = self.window.findChild(QWidget, element_name)
        if widget and widget.isVisible():
            return self._widget_center(widget)

        # Search by button text
        for button in self.window.findChildren(QPushButton):
            if button.text() == element_name and button.isVisible():
                return self._widget_center(button)

        # Search by label text
        for label in self.window.findChildren(QLabel):
            if label.text() == element_name and label.isVisible():
                return self._widget_center(label)

        raise ValueError(f"Element not found: {element_name}")

    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

    def _widget_center(self, widget: QWidget) -> Tuple[int, int]:
        """Get global center coordinates of a widget."""
        rect = widget.rect()
        center = rect.center()
        global_pos = widget.mapToGlobal(center)
        return (global_pos.x(), global_pos.y())

    def _find_label_for_widget(self, parent: QWidget, widget: QWidget) -> Optional[QLabel]:
        """Find the label associated with a widget (e.g., in a form layout)."""
        widget_pos = widget.pos()

        for label in parent.findChildren(QLabel):
            if not label.isVisible():
                continue

            label_pos = label.pos()

            # Check if label is to the left of widget (typical form layout)
            if abs(label_pos.y() - widget_pos.y()) < 30:
                if label_pos.x() < widget_pos.x():
                    return label

        return None
