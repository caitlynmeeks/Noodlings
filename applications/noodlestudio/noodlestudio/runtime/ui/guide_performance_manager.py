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
#   Guide Performance Manager
#
#   Orchestrates the lifecycle of guided play performances.
#   Creates/destroys the GuidePerformanceWindow, toggles demo
#   mode on ComputerUseController, and syncs the [D] button
#   state on NoodleCode panel.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.guide_performance_manager
# PURPOSE:  Performance Lifecycle Orchestrator
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   GuidePerformanceManager
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QTimer

from ..channels import ChannelBus, ChannelMessage
from ..brenda import BrendaDirector, CHANNEL_USER_INPUT
from ..guide_cue_handler import GuideCueHandler

logger = logging.getLogger(__name__)

# Well-known guide VRM location relative to project root
_GUIDE_VRM_RELATIVE = Path("noodlings/guide/Radiances/AjoMajo.vrm")


class GuidePerformanceManager:
    """
    Orchestrates the lifecycle of guided play performances.

    Coordinates:
    - GuidePerformanceWindow (floating VRM + dialogue panel)
    - ComputerUseController (demo mode for ghost cursor)
    - NoodleCode panel [D] button state
    - GuideCueHandler (Brenda direction injection)

    Usage:
        manager = GuidePerformanceManager(main_window)
        manager.set_engine(noodle_code_engine)
        manager.start_performance("Let's Consciousness!", vrm_path="/path/to/ajo.vrm")
        # ... performance runs ...
        manager.stop_performance()
    """

    def __init__(self, main_window):
        """
        Initialize the performance manager.

        Args:
            main_window: MainWindow instance (parent for floating window)
        """
        self._main_window = main_window
        self._window = None  # GuidePerformanceWindow
        self._engine = None
        self._guide_cue_handler = None
        self._noodle_code_panel = None

        # Play system (created per-performance when play_path is provided)
        self._channel_bus = None
        self._director = None      # BrendaDirector
        self._tick_timer = None    # QTimer for Brenda.tick()

        logger.info("GuidePerformanceManager initialized")

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_engine(self, engine):
        """
        Set the NoodleCode engine for LLM communication.

        Args:
            engine: NoodleCodeEngine instance
        """
        self._engine = engine

    def set_noodle_code_panel(self, panel):
        """
        Set reference to the NoodleCode panel for [D] button sync.

        Args:
            panel: NoodleCodePanel instance
        """
        self._noodle_code_panel = panel

    def set_guide_cue_handler(self, handler):
        """
        Set the GuideCueHandler for Brenda direction.

        Args:
            handler: GuideCueHandler instance
        """
        self._guide_cue_handler = handler

    # =========================================================================
    # PERFORMANCE LIFECYCLE
    # =========================================================================

    def start_performance(self, play_title: str, vrm_path: Optional[str] = None,
                          play_path: Optional[str] = None):
        """
        Start a guided performance.

        Creates the floating window, enables demo mode, and optionally
        loads a .play.yaml via BrendaDirector + GuideCueHandler.

        Args:
            play_title: Title displayed in the window header
            vrm_path: Optional path to VRM character file
            play_path: Optional path to .play.yaml file. When provided,
                       creates the full play pipeline (ChannelBus ->
                       BrendaDirector -> GuideCueHandler -> window).
        """
        if self._window:
            logger.warning("Performance already active, stopping first")
            self.stop_performance()

        # Enable demo mode on ComputerUseController
        self._set_demo_mode(True)

        # --- Set up play pipeline if play_path provided ---
        if play_path:
            self._setup_play_pipeline(play_path)

        # Create the floating performance window
        from .guide_performance_window import GuidePerformanceWindow

        self._window = GuidePerformanceWindow(
            parent_window=self._main_window
        )
        self._window.show_play_header(play_title)

        # Wire engine
        if self._engine:
            self._window.set_engine(self._engine)

        # Wire guide cue handler (may have been set externally or by _setup_play_pipeline)
        if self._guide_cue_handler:
            self._window.set_guide_cue_handler(self._guide_cue_handler)

            # Also wire ComputerUseController to GuideCueHandler
            try:
                from noodlestudio.core.computer_use_controller import (
                    get_computer_use_controller
                )
                controller = get_computer_use_controller()
                self._guide_cue_handler.set_computer_use_controller(controller)
            except Exception as e:
                logger.debug(
                    f"Could not wire ComputerUseController to GuideCueHandler: {e}"
                )

        # Connect user input to channel bus for Brenda tracking
        if self._channel_bus and hasattr(self._window, 'messageSent') and self._window.messageSent:
            self._window.messageSent.connect(self._on_user_message)

        # Load VRM -- use provided path, or auto-discover guide VRM
        if not vrm_path:
            vrm_path = self._discover_guide_vrm()
        if vrm_path:
            print(f"[GuidePerformance] Loading VRM: {vrm_path}", flush=True)
            self._window.set_vrm(vrm_path)
        else:
            print("[GuidePerformance] No VRM found", flush=True)

        self._window.show()

        # Start Brenda after window is visible (so first cue arrives with UI ready)
        if self._director:
            self._director.start()
            print(f"[GuidePerformance] Brenda directing: {play_title}", flush=True)

        logger.info(f"Performance started: {play_title}")

    def _setup_play_pipeline(self, play_path: str):
        """
        Create the full play pipeline: ChannelBus -> BrendaDirector -> GuideCueHandler.

        Args:
            play_path: Path to the .play.yaml file
        """
        path = Path(play_path)
        if not path.exists():
            print(f"[GuidePerformance] Play file not found: {play_path}", flush=True)
            return

        # Create channel bus for this performance
        self._channel_bus = ChannelBus()

        # Create and load Brenda
        self._director = BrendaDirector(self._channel_bus)
        if not self._director.load_play(str(path)):
            print(f"[GuidePerformance] Failed to load play: {play_path}", flush=True)
            self._director = None
            self._channel_bus = None
            return

        # Create guide cue handler on the same bus
        self._guide_cue_handler = GuideCueHandler(self._channel_bus, "guide")

        # Start tick timer for Brenda (200ms interval)
        self._tick_timer = QTimer()
        self._tick_timer.timeout.connect(self._director.tick)
        self._tick_timer.start(200)

        print(f"[GuidePerformance] Play pipeline ready: {path.stem}", flush=True)

    def _on_user_message(self, message: str):
        """
        Forward user message to the channel bus so Brenda can track it.

        Args:
            message: The user's message text
        """
        if not self._channel_bus:
            return

        self._channel_bus.publish(
            CHANNEL_USER_INPUT,
            ChannelMessage(
                channel=CHANNEL_USER_INPUT,
                from_noodling="user",
                timestamp=time.time(),
                payload={"text": message}
            )
        )

    def stop_performance(self):
        """
        Stop the current performance.

        Closes the floating window, stops Brenda, disables demo mode,
        and tears down the play pipeline.
        """
        # Stop Brenda tick timer
        if self._tick_timer:
            self._tick_timer.stop()
            self._tick_timer = None

        # Stop director
        if self._director:
            self._director.stop()
            self._director = None

        # Clear play pipeline references
        self._channel_bus = None
        self._guide_cue_handler = None

        if self._window:
            self._window.close()
            self._window = None

        # Disable demo mode
        self._set_demo_mode(False)

        logger.info("Performance stopped")

    @property
    def is_active(self) -> bool:
        """Whether a performance is currently active."""
        return self._window is not None and self._window.isVisible()

    @property
    def window(self):
        """The current GuidePerformanceWindow (or None)."""
        return self._window

    # =========================================================================
    # VRM DISCOVERY
    # =========================================================================

    def _discover_guide_vrm(self) -> Optional[str]:
        """
        Auto-discover the guide character VRM file.

        Looks for AjoMajo.vrm at the well-known path relative to the
        project root (noodlings/guide/Radiances/AjoMajo.vrm).

        Returns:
            Absolute path to VRM file, or None if not found
        """
        # Walk up from this file to find the project root
        # noodlings_clean/applications/noodlestudio/noodlestudio/runtime/ui/
        try:
            studio_dir = Path(__file__).resolve().parent.parent.parent.parent
            project_root = studio_dir.parent.parent

            vrm_path = project_root / _GUIDE_VRM_RELATIVE
            if vrm_path.exists():
                print(f"[GuidePerformance] Auto-discovered VRM: {vrm_path}", flush=True)
                return str(vrm_path)

            print(f"[GuidePerformance] VRM not found at {vrm_path}", flush=True)
        except Exception as e:
            logger.debug(f"Could not discover guide VRM: {e}")

        return None

    # =========================================================================
    # DEMO MODE
    # =========================================================================

    def _set_demo_mode(self, enabled: bool):
        """
        Toggle demo mode on ComputerUseController and sync [D] button.

        Args:
            enabled: Whether to enable demo mode
        """
        # Toggle on ComputerUseController
        try:
            from noodlestudio.core.computer_use_controller import (
                get_computer_use_controller
            )
            controller = get_computer_use_controller()
            controller.demo_mode = enabled
        except Exception as e:
            logger.debug(f"Could not set demo mode: {e}")

        # Sync [D] button on NoodleCode panel
        if self._noodle_code_panel:
            try:
                btn = getattr(self._noodle_code_panel, 'demo_mode_btn', None)
                if btn:
                    btn.setChecked(enabled)
            except Exception as e:
                logger.debug(f"Could not sync [D] button: {e}")


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
