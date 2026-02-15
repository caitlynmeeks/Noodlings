# ──────────────────────────────────────────────────────────────
#   Server Infrastructure Tests
#
#   Verifies: _cmush_dir() path resolution, start.sh structure,
#   server toggle signal behavior.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_server_startup
# PURPOSE:  Server Infrastructure Tests
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestCmushDirResolution:
    """_cmush_dir() must resolve to the real applications/cmush/ directory."""

    def test_cmush_dir_resolves_to_existing_directory(self):
        """_cmush_dir() must point to applications/cmush/ which contains start.sh."""
        from noodlestudio.core.main_window_server_mixin import MainWindowServerMixin

        cmush_dir = MainWindowServerMixin._cmush_dir()
        assert os.path.isdir(cmush_dir), f"cmush dir not found: {cmush_dir}"
        assert os.path.isfile(os.path.join(cmush_dir, 'start.sh')), \
            f"start.sh not found in {cmush_dir}"
        assert os.path.isfile(os.path.join(cmush_dir, 'server.py')), \
            f"server.py not found in {cmush_dir}"
        assert 'applications/cmush' in cmush_dir or 'applications\\cmush' in cmush_dir, \
            f"Path should contain applications/cmush: {cmush_dir}"

    def test_start_sh_is_executable(self):
        """start.sh must have execute permission."""
        from noodlestudio.core.main_window_server_mixin import MainWindowServerMixin

        start = os.path.join(MainWindowServerMixin._cmush_dir(), 'start.sh')
        assert os.access(start, os.X_OK), f"start.sh is not executable: {start}"

    def test_start_sh_trap_before_server(self):
        """Cleanup trap must be registered before blocking server.py call."""
        from noodlestudio.core.main_window_server_mixin import MainWindowServerMixin

        start = os.path.join(MainWindowServerMixin._cmush_dir(), 'start.sh')
        with open(start) as f:
            content = f.read()
        trap_pos = content.find('trap ')
        server_pos = content.find('$PYTHON server.py')
        assert trap_pos > -1, "No trap statement found in start.sh"
        assert server_pos > -1, "No server.py invocation found in start.sh"
        assert trap_pos < server_pos, \
            "trap must come before blocking server.py call"


class TestServerToggleSignals:
    """Programmatic setChecked must not re-fire on_server_toggled."""

    def test_toggle_update_does_not_retrigger(self, qapp):
        """update_connection_status must block signals when setting toggle state."""
        from PyQt6.QtWidgets import QCheckBox
        from noodlestudio.core.main_window_server_mixin import MainWindowServerMixin

        # Build a minimal object that has the mixin's method + a toggle
        class StubServerHost(MainWindowServerMixin):
            def __init__(self):
                self.server_toggle = QCheckBox()
                self._toggle_call_count = 0
                # Wire the toggle to a counter
                self.server_toggle.toggled.connect(self._count_toggle)

            def _count_toggle(self, checked):
                self._toggle_call_count += 1

            def is_server_running(self):
                return False  # Simulate offline

        host = StubServerHost()
        # Start with toggle checked (opposite of server state = offline)
        host.server_toggle.setChecked(True)
        host._toggle_call_count = 0  # Reset counter after setup
        # Now update_connection_status should set toggle to False (server offline)
        # Without blockSignals, this fires toggled signal
        host.update_connection_status()
        assert host._toggle_call_count == 0, \
            f"Toggle signal fired {host._toggle_call_count} times (should be 0 with blockSignals)"
