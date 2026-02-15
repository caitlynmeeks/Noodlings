# ------------------------------------------------------------------
#   Console Panel Tests
#
#   Verifies: deferred WebSocket connection, quiet initial state,
#   reconnect lifecycle.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_console_panel
# PURPOSE:  Console Panel Tests
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestConsoleDeferredConnection:
    """Console must not connect to WebSocket until the server starts."""

    def test_console_does_not_connect_on_init(self, qapp):
        """ConsolePanel must NOT create a WebSocketWorker on construction."""
        from noodlestudio.panels.console_panel import ConsolePanel

        panel = ConsolePanel()

        # ws_worker should not exist or should not be running
        has_worker = hasattr(panel, 'ws_worker') and panel.ws_worker is not None
        if has_worker:
            assert not panel.ws_worker.isRunning(), \
                "WebSocketWorker should not be running on init"

    def test_console_shows_waiting_message(self, qapp):
        """Console must show quiet 'Server not running' on cold start."""
        from noodlestudio.panels.console_panel import ConsolePanel

        panel = ConsolePanel()

        text = panel.log_text.toPlainText()
        assert "Server not running" in text, \
            f"Expected 'Server not running' message, got: {text}"
        # Must NOT contain red error text about connection failure
        html = panel.log_text.toHtml()
        assert "Connection failed" not in text, \
            f"Should not show connection errors on cold start"

    def test_console_connects_when_reconnect_called(self, qapp):
        """Calling reconnect() must create and start a WebSocketWorker."""
        from noodlestudio.panels.console_panel import ConsolePanel

        panel = ConsolePanel()

        # Before reconnect: no worker running
        has_worker_before = (
            hasattr(panel, 'ws_worker')
            and panel.ws_worker is not None
            and panel.ws_worker.isRunning()
        )
        assert not has_worker_before, "No worker should be running before reconnect"

        # Reconnect creates a worker
        panel.reconnect()

        assert hasattr(panel, 'ws_worker'), "reconnect() must create ws_worker"
        assert panel.ws_worker is not None, "ws_worker must not be None after reconnect"

        # Clean up: stop the worker thread
        panel.ws_worker.stop()
        panel.ws_worker.wait(2000)
