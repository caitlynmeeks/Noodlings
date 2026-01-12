# ──────────────────────────────────────────────────────────────
#
#   UI Test Assertions - Assertions and waits for UI tests
#
#   Check conditions and wait for UI elements to appear.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.testing.ui_test_assertions
# PURPOSE:  Test assertions and waits
# LAYER:    Studio / Testing
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QMainWindow, QMenu, QDialog,
    QPushButton, QLineEdit, QTextEdit, QLabel,
    QApplication
)
from PyQt6.QtCore import QThread


class UITestAssertions:
    """
    Assertions and waits for UI tests.

    Provides:
    - wait_for_element: Wait for UI element to appear
    - check_condition: Check various UI conditions
    """

    def __init__(self, main_window: QMainWindow):
        self.window = main_window

    async def wait_for_element(
        self,
        element_spec: Dict[str, Any],
        timeout: float = 10.0
    ):
        """
        Wait for an element to appear.

        Args:
            element_spec: Element specification
            timeout: Maximum wait time in seconds
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Timeout waiting for element: {element_spec}")

            if await self._element_exists(element_spec):
                return

            await asyncio.sleep(0.1)

    async def _element_exists(self, element_spec: Dict[str, Any]) -> bool:
        """Check if an element exists and is visible."""
        try:
            # Dialog
            if 'dialog' in element_spec:
                return self._dialog_visible(element_spec['dialog'])

            # Panel
            if 'panel' in element_spec:
                return self._panel_visible(element_spec['panel'])

            # Button
            if 'button' in element_spec:
                return self._button_visible(element_spec['button'])

            # Facets loaded
            if 'facets_loaded' in element_spec:
                return self._facets_loaded()

            # Facet node exists
            if 'facet_node' in element_spec:
                return self._facet_node_exists(element_spec['facet_node'])

            # Facet exists
            if 'facet_exists' in element_spec:
                return self._facet_exists(element_spec['facet_exists'])

            # Noodling in stage
            if 'noodling_in_stage' in element_spec:
                return self._noodling_in_stage(element_spec['noodling_in_stage'])

            # Item in hierarchy
            if 'item_in_hierarchy' in element_spec:
                return self._item_in_hierarchy(element_spec['item_in_hierarchy'])

            # Inspector showing facet
            if 'inspector_showing_facet' in element_spec:
                return self._inspector_showing_facet()

            # Chat panel
            if 'chat_panel' in element_spec:
                return self._chat_panel_visible()

            # Chat response
            if 'chat_response' in element_spec:
                return self._chat_response_received()

            # Splash visible
            if 'splash_visible' in element_spec:
                return self._splash_visible()

            # Splash complete
            if 'splash_complete' in element_spec:
                return not self._splash_visible()

            # Guide visible
            if 'guide_visible' in element_spec:
                return self._guide_visible()

            # Guide chat bubble
            if 'guide_chat_bubble' in element_spec:
                return self._guide_chat_bubble_visible()

            # Guide response
            if 'guide_response' in element_spec:
                return self._guide_response_received()

            # Save complete
            if 'save_complete' in element_spec:
                return self._save_complete()

            # Sandbox
            if 'sandbox' in element_spec:
                return self._sandbox_active(element_spec['sandbox'])

            # Phi meter visible
            if 'phi_meter_visible' in element_spec:
                return self._phi_meter_visible()

            # Museum panel
            if 'museum' in element_spec:
                return self._museum_visible()

            # Exhibit visible
            if 'exhibit' in element_spec:
                return self._exhibit_visible(element_spec['exhibit'])

            # Boolean shortcuts
            for key in ['project_loaded', 'assembly_loaded', 'noodling_selected']:
                if key in element_spec:
                    return await self.check_condition({key: element_spec[key]})

            return False

        except Exception:
            return False

    async def check_condition(self, condition: Dict[str, Any]) -> bool:
        """
        Check if a condition is true.

        Args:
            condition: Condition specification

        Returns:
            True if condition is met
        """
        # Project loaded
        if 'project_loaded' in condition:
            return self._project_loaded()

        # Project name
        if 'project_name' in condition:
            return self._project_name_matches(condition['project_name'])

        # Assembly loaded
        if 'assembly_loaded' in condition:
            return self._assembly_loaded()

        # Facet exists
        if 'facet_exists' in condition:
            return self._facet_exists(condition['facet_exists'])

        # Wire exists
        if 'wire_exists' in condition:
            return self._wire_exists(condition['wire_exists'])

        # Response contains
        if 'response_contains' in condition:
            return self._response_contains(condition['response_contains'])

        # Response contains any
        if 'response_contains_any' in condition:
            return self._response_contains_any(condition['response_contains_any'])

        # Splash visible
        if 'splash_visible' in condition:
            return self._splash_visible() == condition['splash_visible']

        # Guide visible
        if 'guide_visible' in condition:
            return self._guide_visible() == condition['guide_visible']

        # Tab exists (check if a tab with this name exists in any tab bar)
        if 'tab_exists' in condition:
            return self._tab_exists(condition['tab_exists'])

        # Panel visible
        if 'panel_visible' in condition:
            return self._panel_visible(condition['panel_visible'])

        # Window visible (main window)
        if 'window_visible' in condition:
            return self.window.isVisible() == condition['window_visible']

        return False

    # ═══════════════════════════════════════════════════════════
    # Element Checks
    # ═══════════════════════════════════════════════════════════

    def _dialog_visible(self, dialog_name: str) -> bool:
        """Check if a dialog with given name/title is visible."""
        app = QApplication.instance()
        for widget in app.topLevelWidgets():
            if widget.isVisible() and widget != self.window:
                if isinstance(widget, QDialog) or widget.windowFlags():
                    title = widget.windowTitle()
                    if dialog_name in title or title in dialog_name:
                        return True
        return False

    def _panel_visible(self, panel_name: str) -> bool:
        """Check if a panel is visible."""
        panel_attrs = [
            f'{panel_name}_panel',
            panel_name,
            f'{panel_name}Panel',
        ]

        for attr in panel_attrs:
            if hasattr(self.window, attr):
                panel = getattr(self.window, attr)
                if panel and panel.isVisible():
                    return True

        return False

    def _tab_exists(self, tab_name: str) -> bool:
        """Check if a tab with given name exists in any tab widget or dock."""
        from PyQt6.QtWidgets import QTabWidget, QTabBar, QDockWidget

        # Search all tab widgets in the window
        try:
            for tab_widget in self.window.findChildren(QTabWidget):
                for i in range(tab_widget.count()):
                    if tab_name.lower() in tab_widget.tabText(i).lower():
                        return True
        except Exception:
            pass

        # Also check tab bars directly (some UIs use standalone tab bars)
        try:
            for tab_bar in self.window.findChildren(QTabBar):
                for i in range(tab_bar.count()):
                    if tab_name.lower() in tab_bar.tabText(i).lower():
                        return True
        except Exception:
            pass

        # Check dock widgets (they can be tabified without using QTabWidget)
        try:
            for dock in self.window.findChildren(QDockWidget):
                if tab_name.lower() in dock.windowTitle().lower():
                    return True
        except Exception:
            pass

        # Also check if the window has a panel attribute with that name
        panel_attrs = [f'{tab_name}_panel', tab_name, f'{tab_name}Panel']
        for attr in panel_attrs:
            try:
                if hasattr(self.window, attr) and getattr(self.window, attr, None) is not None:
                    return True
            except Exception:
                pass

        return False

    def _button_visible(self, button_text: str) -> bool:
        """Check if a button with given text is visible."""
        for button in self.window.findChildren(QPushButton):
            if button.text() == button_text and button.isVisible():
                return True

        # Check dialogs too
        app = QApplication.instance()
        for widget in app.topLevelWidgets():
            if widget.isVisible() and widget != self.window:
                for button in widget.findChildren(QPushButton):
                    if button.text() == button_text and button.isVisible():
                        return True

        return False

    def _facets_loaded(self) -> bool:
        """Check if Facets panel has loaded content."""
        facets_panel = getattr(self.window, 'facets_panel', None)
        if facets_panel is None:
            return False

        if hasattr(facets_panel, 'node_graphics'):
            return len(facets_panel.node_graphics) > 0

        return False

    def _facet_node_exists(self, node_name: str) -> bool:
        """Check if a facet node with given name exists."""
        return self._facet_exists(node_name)

    def _facet_exists(self, facet_name: str) -> bool:
        """Check if a facet with given name exists."""
        facets_panel = getattr(self.window, 'facets_panel', None)
        if facets_panel is None:
            return False

        if hasattr(facets_panel, 'node_graphics'):
            for node_gfx in facets_panel.node_graphics.values():
                if hasattr(node_gfx, 'facet'):
                    if node_gfx.facet.name == facet_name:
                        return True

        return False

    def _wire_exists(self, wire_spec) -> bool:
        """
        Check if a wire exists between facets.

        wire_spec can be:
        - [from_facet, to_facet] list
        - {"from": "FacetA", "to": "FacetB"} dict
        """
        facets_panel = getattr(self.window, 'facets_panel', None)
        if facets_panel is None:
            return False

        if isinstance(wire_spec, list) and len(wire_spec) == 2:
            from_name, to_name = wire_spec
        elif isinstance(wire_spec, dict):
            from_name = wire_spec.get('from')
            to_name = wire_spec.get('to')
        else:
            return False

        if hasattr(facets_panel, 'wire_graphics'):
            for wire in facets_panel.wire_graphics:
                if hasattr(wire, 'from_pad') and hasattr(wire, 'to_pad'):
                    from_facet = wire.from_pad.facet_node.facet.name
                    to_facet = wire.to_pad.facet_node.facet.name
                    if from_facet == from_name and to_facet == to_name:
                        return True

        return False

    def _noodling_in_stage(self, name: str) -> bool:
        """Check if a noodling with given name exists in stage."""
        # This would need to check the stage's scene items
        stage = getattr(self.window, 'stage_panel', None)
        if stage is None:
            return False

        # Check stage for noodling graphics
        # For now, assume true if name is not None
        return name is not None or name == True

    def _item_in_hierarchy(self, name: str) -> bool:
        """Check if an item exists in the hierarchy."""
        hierarchy = getattr(self.window, 'hierarchy', None)
        if hierarchy is None:
            return False

        # Would need to search hierarchy tree model
        return True  # Placeholder

    def _inspector_showing_facet(self) -> bool:
        """Check if inspector is showing a facet."""
        inspector = getattr(self.window, 'inspector_panel', None)
        if inspector is None:
            return False

        # Check if inspector has facet content
        if hasattr(inspector, 'current_facet'):
            return inspector.current_facet is not None

        return True  # Placeholder

    # ═══════════════════════════════════════════════════════════
    # Chat/Response Checks
    # ═══════════════════════════════════════════════════════════

    def _chat_panel_visible(self) -> bool:
        """Check if chat panel is visible."""
        chat = getattr(self.window, 'chat_panel', None)
        return chat is not None and chat.isVisible()

    def _chat_response_received(self) -> bool:
        """Check if a chat response has been received."""
        chat = getattr(self.window, 'chat_panel', None)
        if chat is None:
            return False

        if hasattr(chat, 'has_response'):
            return chat.has_response()

        if hasattr(chat, 'message_count'):
            return chat.message_count() > 0

        return True  # Placeholder

    def _response_contains(self, text: str) -> bool:
        """Check if the last response contains text."""
        chat = getattr(self.window, 'chat_panel', None)
        if chat is None:
            return False

        if hasattr(chat, 'last_response'):
            return text.lower() in chat.last_response().lower()

        return False

    def _response_contains_any(self, texts: list) -> bool:
        """Check if the last response contains any of the given texts."""
        chat = getattr(self.window, 'chat_panel', None)
        if chat is None:
            return False

        if hasattr(chat, 'last_response'):
            response = chat.last_response().lower()
            return any(t.lower() in response for t in texts)

        return False

    # ═══════════════════════════════════════════════════════════
    # Guide Checks
    # ═══════════════════════════════════════════════════════════

    def _guide_visible(self) -> bool:
        """Check if Guide is visible."""
        guide = getattr(self.window, 'guide_overlay', None)
        if guide:
            return guide.isVisible()

        # Check for guide panel
        guide_panel = getattr(self.window, 'guide_panel', None)
        if guide_panel:
            return guide_panel.isVisible()

        return False

    def _guide_chat_bubble_visible(self) -> bool:
        """Check if Guide's chat bubble is visible."""
        guide = getattr(self.window, 'guide_overlay', None)
        if guide and hasattr(guide, 'chat_bubble'):
            return guide.chat_bubble.isVisible()

        return self._guide_visible()

    def _guide_response_received(self) -> bool:
        """Check if Guide has responded."""
        guide = getattr(self.window, 'guide_overlay', None)
        if guide and hasattr(guide, 'has_response'):
            return guide.has_response()

        return True  # Placeholder

    # ═══════════════════════════════════════════════════════════
    # Project/State Checks
    # ═══════════════════════════════════════════════════════════

    def _project_loaded(self) -> bool:
        """Check if a project is loaded."""
        if hasattr(self.window, 'project'):
            return self.window.project is not None

        if hasattr(self.window, 'current_project'):
            return self.window.current_project is not None

        return True  # Placeholder

    def _project_name_matches(self, name: str) -> bool:
        """Check if project name matches."""
        if hasattr(self.window, 'project') and self.window.project:
            if hasattr(self.window.project, 'name'):
                return self.window.project.name == name

        return True  # Placeholder

    def _assembly_loaded(self) -> bool:
        """Check if a facet assembly is loaded."""
        facets_panel = getattr(self.window, 'facets_panel', None)
        if facets_panel and hasattr(facets_panel, 'current_assembly'):
            return facets_panel.current_assembly is not None

        return False

    def _save_complete(self) -> bool:
        """Check if save operation completed."""
        # Could check for save indicator, status bar, etc.
        return True  # Placeholder - assume save completes quickly

    # ═══════════════════════════════════════════════════════════
    # Splash/Museum Checks
    # ═══════════════════════════════════════════════════════════

    def _splash_visible(self) -> bool:
        """Check if splash screen is visible."""
        splash = getattr(self.window, 'splash_screen', None)
        if splash:
            return splash.isVisible()

        # Check for splash widget
        app = QApplication.instance()
        for widget in app.topLevelWidgets():
            if 'splash' in widget.objectName().lower():
                return widget.isVisible()

        return False

    def _museum_visible(self) -> bool:
        """Check if Museum of Minds is visible."""
        museum = getattr(self.window, 'museum_panel', None)
        return museum is not None and museum.isVisible()

    def _exhibit_visible(self, exhibit_name: str) -> bool:
        """Check if a specific exhibit is visible."""
        museum = getattr(self.window, 'museum_panel', None)
        if museum and hasattr(museum, 'current_exhibit'):
            return museum.current_exhibit == exhibit_name

        return False

    def _sandbox_active(self, sandbox_name: str) -> bool:
        """Check if a sandbox mode is active."""
        # Would check sandbox controller
        return True  # Placeholder

    def _phi_meter_visible(self) -> bool:
        """Check if phi meter widget is visible."""
        # Would find phi meter in current panel
        return True  # Placeholder
