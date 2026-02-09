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
#   Loads Ajo's facet assembly, creates a FacetExecutor, and
#   routes user messages through the assembly pipeline. The
#   window is a pure renderer; all cognition lives here.
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

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QThread, QTimer, pyqtSignal

from ..channels import ChannelBus, ChannelMessage
from ..brenda import BrendaDirector, CHANNEL_USER_INPUT
from ..guide_cue_handler import GuideCueHandler

logger = logging.getLogger(__name__)

# Well-known guide paths relative to project root
_GUIDE_VRM_RELATIVE = Path("noodlings/guide/Radiances/AjoMajo.vrm")
_GUIDE_ASSEMBLY_RELATIVE = Path("noodlings/guide/assembly.yaml")


# =============================================================================
# Assembly Worker (QThread for async FacetExecutor.execute)
# =============================================================================

class _AssemblyWorker(QThread):
    """Worker thread that executes a facet assembly in its own event loop."""

    resultReady = pyqtSignal(object)    # ExecutionResult
    errorOccurred = pyqtSignal(str)     # Error message

    def __init__(self, executor, assembly, message: str, context: dict):
        super().__init__()
        self._executor = executor
        self._assembly = assembly
        self._message = message
        self._context = context

    def run(self):
        """Execute the assembly in a dedicated asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._executor.execute(
                    self._assembly,
                    incoming_data=self._message,
                    context=self._context
                )
            )
            self.resultReady.emit(result)
        except Exception as e:
            logger.error(f"Assembly execution failed: {e}")
            self.errorOccurred.emit(str(e))
        finally:
            loop.close()


# =============================================================================
# Guide Performance Manager
# =============================================================================

class GuidePerformanceManager:
    """
    Orchestrates the lifecycle of guided play performances.

    Loads Ajo's facet assembly, creates a FacetExecutor with a
    HeadlessLLMClient, and routes user messages through the assembly.
    The window is a pure renderer -- all cognition lives here.

    Coordinates:
    - GuidePerformanceWindow (floating VRM + dialogue panel)
    - FacetExecutor (assembly execution engine)
    - HeadlessLLMClient (LLM backend from editor settings)
    - FACSMapper (sentiment -> VRM expressions)
    - ComputerUseController (demo mode for ghost cursor)
    - NoodleCode panel [D] button state
    - GuideCueHandler (Brenda direction injection)

    Usage:
        manager = GuidePerformanceManager(main_window)
        manager.start_performance("Ajo Alive", vrm_path="/path/to/ajo.vrm")
        # ... user chats, assembly runs, expressions animate ...
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
        self._noodle_code_panel = None

        # Assembly execution
        self._assembly = None       # FacetAssembly
        self._assembly_path = None  # Path to assembly YAML (for editor)
        self._executor = None       # FacetExecutor
        self._llm_client = None     # HeadlessLLMClient
        self._worker = None         # _AssemblyWorker (current execution)

        # Conversation state
        self._conversation_history: List[Dict] = []
        self._last_user_message: str = ""

        # Play system (created per-performance when play_path is provided)
        self._channel_bus = None
        self._director = None       # BrendaDirector
        self._guide_cue_handler = None
        self._tick_timer = None     # QTimer for Brenda.tick()
        self._play_title = None     # For restoring header after test mode

        # FACS expression test mode
        self._expression_test_timer = None
        self._expression_test_index = 0
        self._expression_test_mapper = None

        logger.info("GuidePerformanceManager initialized")

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

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
    # LLM CLIENT FROM EDITOR SETTINGS
    # =========================================================================

    def _create_llm_client(self):
        """
        Create a HeadlessLLMClient from the editor's provider settings.

        Reads from ProviderManager and ModelLabelManager singletons so
        Ajo respects whatever provider the user has configured.

        Returns:
            HeadlessLLMClient configured for facet execution
        """
        from noodlestudio.core.provider_manager import get_provider_manager
        from noodlestudio.core.model_label_manager import get_model_label_manager
        from noodlestudio.runtime.llm_client import HeadlessLLMClient, LLMConfig

        label_mgr = get_model_label_manager()
        provider_mgr = get_provider_manager()

        # Get primary provider from Large label
        provider_id, model_name = label_mgr.get_model_for_label("Large")
        provider = provider_mgr.get_provider(provider_id) if provider_id else None

        # Build SMALL/MEDIUM/LARGE -> actual model name mapping
        model_labels = {}
        for label in ["Small", "Medium", "Large"]:
            pid, mname = label_mgr.get_model_for_label(label)
            if mname:
                model_labels[label.upper()] = mname

        config = LLMConfig(
            provider=provider.type if provider else "ollama",
            model=model_name or "",
            api_key=provider.api_key if provider else "",
            base_url=provider.base_url if provider else "",
            model_labels=model_labels
        )

        print(f"[GuidePerformance] LLM client: provider={config.provider}, "
              f"labels={model_labels}", flush=True)

        return HeadlessLLMClient(config)

    # =========================================================================
    # ASSEMBLY LOADING
    # =========================================================================

    def _discover_guide_assembly(self) -> Optional[str]:
        """
        Auto-discover the guide assembly YAML file.

        Returns:
            Absolute path to assembly.yaml, or None if not found
        """
        try:
            studio_dir = Path(__file__).resolve().parent.parent.parent.parent
            project_root = studio_dir.parent.parent

            assembly_path = project_root / _GUIDE_ASSEMBLY_RELATIVE
            if assembly_path.exists():
                print(f"[GuidePerformance] Assembly: {assembly_path}", flush=True)
                return str(assembly_path)

            print(f"[GuidePerformance] Assembly not found at {assembly_path}", flush=True)
        except Exception as e:
            logger.debug(f"Could not discover guide assembly: {e}")

        return None

    def _load_assembly(self) -> bool:
        """
        Load Ajo's facet assembly and create the execution pipeline.

        Returns:
            True if assembly loaded successfully
        """
        from noodlestudio.core.facet_system import FacetAssembly
        from noodlestudio.core.facet_executor import FacetExecutor

        assembly_path = self._discover_guide_assembly()
        if not assembly_path:
            print("[GuidePerformance] No assembly found, Ajo cannot think", flush=True)
            return False

        try:
            self._assembly = FacetAssembly.load_yaml(assembly_path)
            self._assembly_path = assembly_path
            print(f"[GuidePerformance] Loaded assembly: {self._assembly.name} "
                  f"({len(self._assembly.facets)} facets, "
                  f"{len(self._assembly.connections)} connections)", flush=True)
        except Exception as e:
            logger.error(f"Failed to load assembly: {e}")
            print(f"[GuidePerformance] Assembly load failed: {e}", flush=True)
            return False

        # Create LLM client from editor settings
        try:
            self._llm_client = self._create_llm_client()
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            print(f"[GuidePerformance] LLM client creation failed: {e}", flush=True)
            return False

        # Create executor (event bus disabled -- it requires an asyncio loop
        # in the main thread which conflicts with Qt. Enable only when the
        # facets editor needs live visualization of guide execution.)
        self._executor = FacetExecutor(
            llm_client=self._llm_client,
            channel_bus=self._channel_bus,
            use_event_bus=False
        )

        print("[GuidePerformance] Assembly execution pipeline ready", flush=True)
        return True

    # =========================================================================
    # PERFORMANCE LIFECYCLE
    # =========================================================================

    def start_performance(self, play_title: str, vrm_path: Optional[str] = None,
                          play_path: Optional[str] = None):
        """
        Start a guided performance.

        Creates the floating window, loads the assembly, and enables
        demo mode. Optionally loads a .play.yaml via BrendaDirector.

        Args:
            play_title: Title displayed in the window header
            vrm_path: Optional path to VRM character file
            play_path: Optional path to .play.yaml file
        """
        if self._window:
            logger.warning("Performance already active, stopping first")
            self.stop_performance()

        # Enable demo mode on ComputerUseController
        self._set_demo_mode(True)

        # --- Set up play pipeline if play_path provided ---
        if play_path:
            self._setup_play_pipeline(play_path)

        # --- Load assembly and create execution pipeline ---
        self._load_assembly()

        # Create the floating performance window
        from .guide_performance_window import GuidePerformanceWindow

        self._window = GuidePerformanceWindow(
            parent_window=self._main_window
        )
        self._play_title = play_title
        self._window.show_play_header(play_title)

        # Wire guide cue handler (may have been set externally or by _setup_play_pipeline)
        if self._guide_cue_handler:
            # Wire ComputerUseController to GuideCueHandler
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

        # Connect user message signals
        self._window.messageSubmitted.connect(self._on_user_message_for_assembly)

        # Ctrl+Shift+F - FACS expression test mode
        from PyQt6.QtGui import QShortcut, QKeySequence
        facs_shortcut = QShortcut(QKeySequence("Ctrl+Shift+F"), self._window)
        facs_shortcut.activated.connect(self.toggle_expression_test)

        if self._channel_bus and hasattr(self._window, 'messageSent') and self._window.messageSent:
            self._window.messageSent.connect(self._on_user_message_for_channel)

        # Load VRM -- use provided path, or auto-discover guide VRM
        if not vrm_path:
            vrm_path = self._discover_guide_vrm()
        if vrm_path:
            print(f"[GuidePerformance] Loading VRM: {vrm_path}", flush=True)
            self._window.set_vrm(vrm_path)
        else:
            print("[GuidePerformance] No VRM found", flush=True)

        self._window.show()

        # Load assembly into facets editor for visibility
        if self._assembly:
            try:
                facets_editor = getattr(self._main_window, 'facets_editor', None)
                if facets_editor and hasattr(facets_editor, 'load_assembly_from_data'):
                    source = str(self._assembly_path) if self._assembly_path else None
                    facets_editor.load_assembly_from_data(
                        self._assembly, force_reload=True, source_path=source
                    )
                    print(f"[GuidePerformance] Assembly loaded in facets editor", flush=True)
            except Exception as e:
                print(f"[GuidePerformance] Could not load assembly in editor: {e}", flush=True)

        # Start Brenda after window is visible
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

    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================

    def _on_user_message_for_assembly(self, message: str):
        """
        Execute Ajo's assembly in response to a user message.

        Runs on a worker thread via _AssemblyWorker so the UI stays responsive.

        Args:
            message: The user's message text
        """
        if not self._assembly or not self._executor:
            if self._window:
                self._window._show_error("Assembly not loaded.")
                self._window.set_busy(False)
            return

        # Prevent overlapping executions
        if self._worker and self._worker.isRunning():
            return

        self._last_user_message = message

        # Set window to busy state
        if self._window:
            self._window.set_busy(True)

        # Build execution context
        context = {
            'conversation_history': self._conversation_history,
        }

        # Inject Brenda direction if available
        if self._guide_cue_handler:
            direction = self._guide_cue_handler.build_system_prompt_addition()
            if direction:
                context['brenda_direction'] = direction

        # Execute assembly on worker thread
        self._worker = _AssemblyWorker(
            self._executor, self._assembly, message, context
        )
        self._worker.resultReady.connect(self._on_assembly_result)
        self._worker.errorOccurred.connect(self._on_assembly_error)
        self._worker.start()

    def _on_assembly_result(self, result):
        """
        Handle completed assembly execution.

        Displays the response, drives expressions via affect pipeline,
        reports to Brenda, and updates conversation history.

        Args:
            result: ExecutionResult from FacetExecutor
        """
        if not self._window:
            return

        # Display the response
        response = result.response
        if response and response != '[No output]':
            self._window.append_guide_text(response)
        else:
            self._window._show_error("No response generated.")

        # Drive expressions from sentiment facet output
        sentiment_raw = result.facet_outputs.get('sentiment', {})
        sentiment_output = sentiment_raw.get('out')
        print(f"[AFFECT] Sentiment facet outputs: {sentiment_raw}", flush=True)
        if sentiment_output:
            self._apply_affect(sentiment_output)
        else:
            print("[AFFECT] Sentiment facet produced NO output", flush=True)

        # Report to GuideCueHandler for Brenda feedback
        if self._guide_cue_handler and response and response.strip():
            self._guide_cue_handler.report_response(
                response.strip(), self._last_user_message
            )

        # Update conversation history
        self._conversation_history.append({
            'role': 'user', 'content': self._last_user_message
        })
        self._conversation_history.append({
            'role': 'assistant', 'content': response
        })

        # Clear busy state
        self._window.set_busy(False)
        self._worker = None

        print(f"[GuidePerformance] Assembly done: {result.total_time:.2f}s, "
              f"{result.total_tokens} tokens", flush=True)

    def _on_assembly_error(self, error_msg: str):
        """
        Handle assembly execution error.

        Args:
            error_msg: Error description
        """
        if self._window:
            self._window._show_error(f"Assembly error: {error_msg}")
            self._window.set_busy(False)
        self._worker = None
        print(f"[GuidePerformance] Assembly error: {error_msg}", flush=True)

    def _on_user_message_for_channel(self, message: str):
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

    # =========================================================================
    # AFFECT PIPELINE
    # =========================================================================

    def _apply_affect(self, affect_text: str):
        """
        Parse sentiment JSON and drive VRM expressions.

        Runs the full affect pipeline:
        sentiment JSON -> Affect -> FACSMapper -> VRM blendshapes

        Args:
            affect_text: JSON string from the sentiment facet
        """
        print(f"[AFFECT] Raw sentiment text: {affect_text!r}", flush=True)

        try:
            affect_data = json.loads(affect_text)
            valence = float(affect_data.get('valence', 0.5))
            arousal = float(affect_data.get('arousal', 0.5))
            dominance = float(affect_data.get('dominance', 0.5))
            print(f"[AFFECT] Parsed: valence={valence:.2f} arousal={arousal:.2f} "
                  f"dominance={dominance:.2f}", flush=True)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[AFFECT] JSON parse FAILED: {e}", flush=True)
            return

        from noodlestudio.runtime.facs_mapper import FACSMapper, Affect

        mapper = FACSMapper()
        affect_state = Affect(
            valence=valence * 2 - 1,  # Remap 0..1 to -1..1 for FACSMapper
            arousal=arousal,
            dominance=dominance,
            sorrow=max(0.0, (1.0 - valence) * 0.5),
            boredom=max(0.0, (1.0 - arousal) * 0.3)
        )

        vrm_shapes = mapper.map_affect_to_vrm(affect_state)
        print(f"[AFFECT] VRM blend shapes ({len(vrm_shapes)}): {vrm_shapes}", flush=True)

        if self._window and vrm_shapes:
            self._window.set_blend_shapes(vrm_shapes)
        elif not vrm_shapes:
            print("[AFFECT] FACSMapper produced EMPTY shapes", flush=True)

    # =========================================================================
    # FACS EXPRESSION TEST MODE
    # =========================================================================

    _EXPRESSION_TEST_PRESETS = None  # Lazy-loaded

    @classmethod
    def _get_test_presets(cls):
        """Lazy-load test presets to avoid import at module level."""
        if cls._EXPRESSION_TEST_PRESETS is None:
            from noodlestudio.runtime.facs_mapper import Affect
            cls._EXPRESSION_TEST_PRESETS = [
                ("Neutral",    Affect(valence=0.0,  arousal=0.3, dominance=0.5, sorrow=0.0, boredom=0.0)),
                ("Happy",      Affect(valence=0.8,  arousal=0.6, dominance=0.7, sorrow=0.0, boredom=0.0)),
                ("Very Happy", Affect(valence=1.0,  arousal=0.9, dominance=0.8, sorrow=0.0, boredom=0.0)),
                ("Sad",        Affect(valence=-0.7, arousal=0.2, dominance=0.2, sorrow=0.8, boredom=0.0)),
                ("Angry",      Affect(valence=-0.6, arousal=0.9, dominance=0.9, sorrow=0.0, boredom=0.0)),
                ("Surprised",  Affect(valence=0.3,  arousal=0.9, dominance=0.3, sorrow=0.0, boredom=0.0)),
                ("Afraid",     Affect(valence=-0.5, arousal=0.8, dominance=0.1, sorrow=0.0, boredom=0.0)),
                ("Bored",      Affect(valence=-0.2, arousal=0.1, dominance=0.3, sorrow=0.0, boredom=0.8)),
                ("Contempt",   Affect(valence=-0.3, arousal=0.4, dominance=0.8, sorrow=0.0, boredom=0.2)),
            ]
        return cls._EXPRESSION_TEST_PRESETS

    def toggle_expression_test(self):
        """Toggle FACS expression test mode (Ctrl+Shift+F).

        Cycles through 9 preset affect states at 2.5s intervals to
        visually verify the morph target pipeline. Shows the current
        expression name in the window header.
        """
        if not self._window:
            return

        if self._expression_test_timer and self._expression_test_timer.isActive():
            self._stop_expression_test()
        else:
            self._start_expression_test()

    def _start_expression_test(self):
        """Start cycling through test expressions."""
        from noodlestudio.runtime.facs_mapper import FACSMapper

        self._expression_test_index = 0
        self._expression_test_mapper = FACSMapper()

        if not self._expression_test_timer:
            self._expression_test_timer = QTimer()
            self._expression_test_timer.timeout.connect(self._next_test_expression)

        self._apply_test_expression()
        self._expression_test_timer.start(2500)
        print("[FACS TEST] Started expression test cycle", flush=True)

    def _stop_expression_test(self):
        """Stop test mode and reset to neutral."""
        if self._expression_test_timer:
            self._expression_test_timer.stop()

        from noodlestudio.runtime.facs_mapper import FACSMapper, Affect
        neutral = Affect(valence=0.0, arousal=0.3, dominance=0.5, sorrow=0.0, boredom=0.0)
        shapes = FACSMapper().map_affect_to_vrm(neutral)
        if self._window:
            self._window.set_blend_shapes(shapes)
            self._window.show_play_header(self._play_title or "Guide")
        print("[FACS TEST] Stopped", flush=True)

    def _next_test_expression(self):
        """Advance to the next test expression."""
        presets = self._get_test_presets()
        self._expression_test_index = (
            (self._expression_test_index + 1) % len(presets)
        )
        self._apply_test_expression()

    def _apply_test_expression(self):
        """Apply the current test expression."""
        presets = self._get_test_presets()
        name, affect = presets[self._expression_test_index]
        shapes = self._expression_test_mapper.map_affect_to_vrm(affect)

        if self._window:
            self._window.set_blend_shapes(shapes)
            self._window.show_play_header(f"FACS Test: {name}")

        print(f"[FACS TEST] {name}: {shapes}", flush=True)

    # =========================================================================
    # PERFORMANCE LIFECYCLE (STOP)
    # =========================================================================

    def stop_performance(self):
        """
        Stop the current performance.

        Closes the floating window, stops the executor, stops Brenda,
        disables demo mode, and tears down the play pipeline.
        """
        # Stop any running worker
        if self._worker and self._worker.isRunning():
            self._worker.wait(2000)
            self._worker = None

        # Stop Brenda tick timer
        if self._tick_timer:
            self._tick_timer.stop()
            self._tick_timer = None

        # Stop director
        if self._director:
            self._director.stop()
            self._director = None

        # Close LLM client session
        if self._llm_client:
            # HeadlessLLMClient.close() is async; schedule in a temp loop
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._llm_client.close())
                loop.close()
            except Exception:
                pass
            self._llm_client = None

        # Clear execution state
        self._assembly = None
        self._executor = None
        self._conversation_history = []

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

        Returns:
            Absolute path to VRM file, or None if not found
        """
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
