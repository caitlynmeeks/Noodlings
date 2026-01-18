# ──────────────────────────────────────────────────────────────
#
#   UI Test Actions - Action implementations for UI tests
#
#   Each action type (click, type, drag, etc.) is implemented here.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.testing.ui_test_actions
# PURPOSE:  Test action implementations
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
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .ui_test_runner import UITestRunner


class UITestActions:
    """
    Implements UI test actions.

    Actions use the computer use controller to interact with the UI
    and the ghost cursor to visualize what's happening.
    """

    def __init__(self, runner: 'UITestRunner'):
        self.runner = runner

    @property
    def computer_use(self):
        return self.runner.computer_use

    @property
    def ghost(self):
        return self.runner.ghost

    @property
    def targets(self):
        return self.runner.targets

    @property
    def assertions(self):
        return self.runner.assertions

    async def execute(self, step: Dict[str, Any]):
        """
        Execute a test step.

        Args:
            step: Step specification dict with 'action' and parameters
        """
        action = step.get('action', 'unknown')

        # Dispatch to handler
        handler = getattr(self, f'_action_{action}', None)
        if handler is None:
            raise ValueError(f"Unknown action: {action}")

        await handler(step)

    # ═══════════════════════════════════════════════════════════
    # Click Actions
    # ═══════════════════════════════════════════════════════════

    async def _action_click(self, step: Dict[str, Any]):
        """Left click on a target."""
        x, y = await self.targets.resolve(step['target'])

        # Visualize with ghost cursor
        if self.ghost:
            await self._visualize_click(x, y)

        # Perform click
        if self.computer_use:
            await self._do_click(x, y, 'left')

    async def _action_right_click(self, step: Dict[str, Any]):
        """Right click on a target (context menu)."""
        x, y = await self.targets.resolve(step['target'])

        if self.ghost:
            await self._visualize_click(x, y)

        if self.computer_use:
            await self._do_click(x, y, 'right')

    async def _action_double_click(self, step: Dict[str, Any]):
        """Double click on a target."""
        x, y = await self.targets.resolve(step['target'])

        if self.ghost:
            await self._visualize_double_click(x, y)

        if self.computer_use:
            await self._do_double_click(x, y)

    # ═══════════════════════════════════════════════════════════
    # Type Actions
    # ═══════════════════════════════════════════════════════════

    async def _action_type(self, step: Dict[str, Any]):
        """Type text (at current focus or target)."""
        text = step.get('text', '')

        # Click target first if specified
        if 'target' in step:
            x, y = await self.targets.resolve(step['target'])
            if self.ghost:
                await self._visualize_click(x, y)
            if self.computer_use:
                await self._do_click(x, y, 'left')
            await asyncio.sleep(0.1)

        # Type the text
        if self.computer_use:
            await self._do_type(text)

    async def _action_clear_and_type(self, step: Dict[str, Any]):
        """Select all and type (replace existing text)."""
        text = step.get('text', '')

        # Click target first if specified
        if 'target' in step:
            x, y = await self.targets.resolve(step['target'])
            if self.ghost:
                await self._visualize_click(x, y)
            if self.computer_use:
                # Triple-click to select all
                await self._do_click(x, y, 'left')
                await asyncio.sleep(0.05)
                await self._do_click(x, y, 'left')
                await asyncio.sleep(0.05)
                await self._do_click(x, y, 'left')
            await asyncio.sleep(0.1)

        # Type the new text (replaces selection)
        if self.computer_use:
            await self._do_type(text)

    # ═══════════════════════════════════════════════════════════
    # Drag Actions
    # ═══════════════════════════════════════════════════════════

    async def _action_drag(self, step: Dict[str, Any]):
        """Drag from one target to another."""
        x1, y1 = await self.targets.resolve(step['from'])
        x2, y2 = await self.targets.resolve(step['to'])

        if self.ghost:
            await self._visualize_drag(x1, y1, x2, y2)

        if self.computer_use:
            await self._do_drag(x1, y1, x2, y2)

    # ═══════════════════════════════════════════════════════════
    # Key Actions
    # ═══════════════════════════════════════════════════════════

    async def _action_key(self, step: Dict[str, Any]):
        """Press a single key."""
        key = step.get('key', '')

        if self.computer_use:
            await self._do_key(key)

    async def _action_key_combo(self, step: Dict[str, Any]):
        """Press a key combination (e.g., Cmd+S)."""
        keys = step.get('keys', [])

        if self.computer_use:
            await self._do_key_combo(keys)

    # ═══════════════════════════════════════════════════════════
    # Wait Actions
    # ═══════════════════════════════════════════════════════════

    async def _action_wait_for(self, step: Dict[str, Any]):
        """Wait for an element to appear."""
        element_spec = step.get('element', step.get('for', {}))
        timeout = self._parse_timeout(step.get('timeout', '10s'))

        await self.assertions.wait_for_element(element_spec, timeout)

    async def _action_wait(self, step: Dict[str, Any]):
        """Wait for a fixed duration."""
        duration = self._parse_timeout(step.get('duration', '1s'))
        await asyncio.sleep(duration)

    # ═══════════════════════════════════════════════════════════
    # Assert Actions
    # ═══════════════════════════════════════════════════════════

    async def _action_assert(self, step: Dict[str, Any]):
        """Assert a condition is true."""
        condition = step.get('condition', {})
        expected = step.get('expected', True)

        result = await self.assertions.check_condition(condition)

        if result != expected:
            raise AssertionError(f"Expected {expected}, got {result} for condition {condition}")

    async def _action_assert_visual(self, step: Dict[str, Any]):
        """
        Assert current UI matches a baseline screenshot.

        Usage in .ui-test.yaml:
            - action: assert_visual
              baseline: "ajo_imported"
              threshold: 0.95
              ignore_regions:
                - [0, 0, 100, 30]  # Ignore title bar area
        """
        import base64

        baseline = step.get('baseline')
        threshold = step.get('threshold', 0.95)
        ignore_regions = step.get('ignore_regions', [])

        if not baseline:
            raise ValueError("assert_visual requires 'baseline' parameter")

        # Take screenshot
        if not self.computer_use:
            raise RuntimeError("Computer use controller required for visual verification")

        b64_data, width, height = self.computer_use.screenshot(add_rulers=False)

        # Verify against baseline
        from .visual_verifier import VisualVerifier
        verifier = VisualVerifier()
        result = verifier.verify(baseline, b64_data, threshold, ignore_regions)

        if not result.passed:
            # Save the failed screenshot for debugging
            fail_path = verifier.baselines_dir / f"{baseline}_FAILED.png"
            fail_path.write_bytes(base64.b64decode(b64_data))

            diff_path = None
            if result.diff_image:
                diff_path = verifier.baselines_dir / f"{baseline}_DIFF.png"
                diff_path.write_bytes(result.diff_image)

            error_msg = (
                f"Visual verification failed: {result.message}\n"
                f"Failed screenshot: {fail_path}"
            )
            if diff_path:
                error_msg += f"\nDiff image: {diff_path}"

            raise AssertionError(error_msg)

        print(f"    VISUAL PASS: {baseline} ({result.similarity:.1%} similarity)")

    async def _action_assert_file_exists(self, step: Dict[str, Any]):
        """Assert a file exists."""
        import os
        path = step.get('path', '')

        # Resolve relative to project root
        if not os.path.isabs(path):
            # Try common bases
            bases = [
                os.getcwd(),
                os.path.expanduser('~/git/noodlings_clean'),
            ]
            for base in bases:
                full_path = os.path.join(base, path)
                if os.path.exists(full_path):
                    return

            raise AssertionError(f"File not found: {path}")

        if not os.path.exists(path):
            raise AssertionError(f"File not found: {path}")

    # ═══════════════════════════════════════════════════════════
    # Utility Actions
    # ═══════════════════════════════════════════════════════════

    async def _action_log(self, step: Dict[str, Any]):
        """Log a message."""
        message = step.get('message', '')
        print(f"    LOG: {message}")

    async def _action_screenshot(self, step: Dict[str, Any]):
        """Take a screenshot."""
        path = step.get('path', 'screenshot.png')

        if self.computer_use and hasattr(self.computer_use, 'screenshot'):
            # ComputerUseController.screenshot() is synchronous
            self.computer_use.screenshot(path)
            print(f"    Screenshot saved: {path}")

    # ═══════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════

    def _parse_timeout(self, timeout_str: str) -> float:
        """Parse timeout string like '5s' or '500ms' to seconds."""
        if isinstance(timeout_str, (int, float)):
            return float(timeout_str)

        timeout_str = str(timeout_str).lower().strip()

        if timeout_str.endswith('ms'):
            return float(timeout_str[:-2]) / 1000
        elif timeout_str.endswith('s'):
            return float(timeout_str[:-1])
        else:
            return float(timeout_str)

    # ═══════════════════════════════════════════════════════════
    # Low-level Computer Use Wrappers
    # ═══════════════════════════════════════════════════════════

    async def _do_click(self, x: int, y: int, button: str = 'left'):
        """Perform a click via computer use."""
        # ComputerUseController methods are synchronous (return bool)
        if hasattr(self.computer_use, 'click'):
            self.computer_use.click(x, y, button)
        elif hasattr(self.computer_use, 'left_click') and button == 'left':
            self.computer_use.left_click(x, y)
        elif hasattr(self.computer_use, 'right_click') and button == 'right':
            self.computer_use.right_click(x, y)
        else:
            # Fallback: simulate with Qt
            from PyQt6.QtCore import QPoint, Qt
            from PyQt6.QtTest import QTest
            from PyQt6.QtWidgets import QApplication

            widget = QApplication.widgetAt(x, y)
            if widget:
                local_pos = widget.mapFromGlobal(QPoint(x, y))
                btn = Qt.MouseButton.LeftButton if button == 'left' else Qt.MouseButton.RightButton
                QTest.mouseClick(widget, btn, pos=local_pos)

    async def _do_double_click(self, x: int, y: int):
        """Perform a double click."""
        # ComputerUseController methods are synchronous
        if hasattr(self.computer_use, 'double_click'):
            self.computer_use.double_click(x, y)
        else:
            # Fallback: two clicks
            await self._do_click(x, y, 'left')
            await asyncio.sleep(0.05)
            await self._do_click(x, y, 'left')

    async def _do_type(self, text: str):
        """Type text via computer use."""
        # ComputerUseController methods are synchronous
        if hasattr(self.computer_use, 'type_text'):
            self.computer_use.type_text(text)
        else:
            # Fallback: simulate with Qt
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtTest import QTest

            widget = QApplication.focusWidget()
            if widget:
                QTest.keyClicks(widget, text)

    async def _do_drag(self, x1: int, y1: int, x2: int, y2: int):
        """Perform a drag operation."""
        # ComputerUseController methods are synchronous
        if hasattr(self.computer_use, 'drag'):
            self.computer_use.drag(x1, y1, x2, y2)
        else:
            # Fallback: mouse down, move, up
            print(f"    [Drag not implemented - would drag from ({x1},{y1}) to ({x2},{y2})]")

    async def _do_key(self, key: str):
        """Press a key."""
        # ComputerUseController methods are synchronous
        if hasattr(self.computer_use, 'key'):
            self.computer_use.key(key)
        else:
            # Fallback: simulate with Qt
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtTest import QTest
            from PyQt6.QtCore import Qt

            key_map = {
                'Return': Qt.Key.Key_Return,
                'Enter': Qt.Key.Key_Return,
                'Escape': Qt.Key.Key_Escape,
                'Tab': Qt.Key.Key_Tab,
                'Backspace': Qt.Key.Key_Backspace,
                'Delete': Qt.Key.Key_Delete,
            }

            qt_key = key_map.get(key)
            if qt_key:
                widget = QApplication.focusWidget()
                if widget:
                    QTest.keyClick(widget, qt_key)

    async def _do_key_combo(self, keys: list):
        """Press a key combination."""
        # ComputerUseController methods are synchronous
        if hasattr(self.computer_use, 'key_combo'):
            self.computer_use.key_combo(keys)
        elif hasattr(self.computer_use, 'hotkey'):
            self.computer_use.hotkey(*keys)
        else:
            # Fallback: Qt simulation
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtTest import QTest
            from PyQt6.QtCore import Qt

            modifier_map = {
                'Cmd': Qt.KeyboardModifier.ControlModifier,  # Mac Cmd = Ctrl
                'Ctrl': Qt.KeyboardModifier.ControlModifier,
                'Shift': Qt.KeyboardModifier.ShiftModifier,
                'Alt': Qt.KeyboardModifier.AltModifier,
                'Option': Qt.KeyboardModifier.AltModifier,
            }

            modifiers = Qt.KeyboardModifier.NoModifier
            key = None

            for k in keys:
                if k in modifier_map:
                    modifiers |= modifier_map[k]
                else:
                    # Assume it's the main key
                    key = getattr(Qt.Key, f'Key_{k.upper()}', None)

            if key:
                widget = QApplication.focusWidget()
                if widget:
                    QTest.keyClick(widget, key, modifiers)

    # ═══════════════════════════════════════════════════════════
    # Ghost Cursor Visualization
    # ═══════════════════════════════════════════════════════════

    async def _visualize_click(self, x: int, y: int):
        """Visualize a click with ghost cursor."""
        if self.ghost and hasattr(self.ghost, 'visualize_click_async'):
            await self.ghost.visualize_click_async(x, y)
        elif self.ghost and hasattr(self.ghost, 'visualize_click'):
            self.ghost.visualize_click(x, y, 'left', lambda: None)
            await asyncio.sleep(0.3)

    async def _visualize_double_click(self, x: int, y: int):
        """Visualize a double click."""
        if self.ghost and hasattr(self.ghost, 'visualize_double_click_async'):
            await self.ghost.visualize_double_click_async(x, y)
        else:
            await self._visualize_click(x, y)
            await asyncio.sleep(0.1)
            await self._visualize_click(x, y)

    async def _visualize_drag(self, x1: int, y1: int, x2: int, y2: int):
        """Visualize a drag operation."""
        if self.ghost and hasattr(self.ghost, 'visualize_drag_async'):
            await self.ghost.visualize_drag_async(x1, y1, x2, y2)
        elif self.ghost and hasattr(self.ghost, 'visualize_drag'):
            self.ghost.visualize_drag(x1, y1, x2, y2, lambda: None)
            await asyncio.sleep(0.5)
