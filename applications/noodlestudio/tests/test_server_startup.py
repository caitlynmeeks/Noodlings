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
