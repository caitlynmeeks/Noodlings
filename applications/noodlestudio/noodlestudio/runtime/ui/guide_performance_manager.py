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
#   Delegates cognition to NoodlingPerformer instances and wires
#   their signals to the rendering window and facets editor.
#   Supports single-noodling mode (backward compatible) and
#   ensemble mode (multiple noodlings on a shared stage).
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
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from PyQt6.QtCore import QTimer

from ..channels import ChannelBus, ChannelMessage
from ..brenda import BrendaDirector, CHANNEL_USER_INPUT
from ..guide_cue_handler import GuideCueHandler
from .noodling_performer import NoodlingPerformer, create_llm_client

logger = logging.getLogger(__name__)

# Well-known guide paths relative to project root
_GUIDE_VRM_RELATIVE = Path("noodlings/guide/Radiances/AjoMajo.vrm")
_GUIDE_ASSEMBLY_RELATIVE = Path("noodlings/guide/assembly.yaml")

# Yuki's assembly path (relative to studio_dir = applications/noodlestudio/)
_YUKI_ASSEMBLY_RELATIVE = Path("library/noodlings/yuki_cyberfox/assembly.yaml")


# =============================================================================
# Guide Performance Manager
# =============================================================================

class GuidePerformanceManager:
    """
    Orchestrates the lifecycle of guided play performances.

    Delegates cognition to NoodlingPerformer and wires its signals
    to the rendering window and facets editor. The window is a pure
    renderer; all cognition lives in the performer.

    Coordinates:
    - GuidePerformanceWindow (floating VRM + dialogue panel)
    - NoodlingPerformer (assembly execution, affect, typed text)
    - FACSMapper (via performer -- sentiment -> VRM expressions)
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

        # Performer (owns assembly, executor, history, affect, playback)
        self._performer = None  # NoodlingPerformer

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

        # Assembly editor live visualization
        self._assembly_editor = None       # Cached reference
        self._current_execution_id = None  # Current execution for event tracking

        # Ensemble mode (two noodlings on shared stage)
        self._ensemble_mode = False
        self._performers = {}        # noodling_id -> NoodlingPerformer
        self._turn_queue = []        # Remaining noodling_ids in current turn
        self._turn_responses = {}    # noodling_id -> response from this round
        self._pending_message = None # User message waiting for execution
        self._active_noodling_id = 'default'  # Which noodling's events are emitted

        # Ensemble conversation history (chronological all-speaker transcript)
        self._ensemble_history: List[Dict] = []

        # Stage metadata (populated by _discover_stage_instances)
        self._stage_description = None   # str from stage.yaml
        self._instance_metadata = {}     # noodling_id -> discovery dict

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
    # HIERARCHY SYNC
    # =========================================================================

    def on_hierarchy_noodling_selected(self, noodling_id: str):
        """
        Handle noodling selection from hierarchy during active performance.

        Sets the selected performer as active speaker in the window and
        switches the facets editor to their assembly.

        Args:
            noodling_id: The noodling instance ID (e.g. 'ajo', 'yuki')
        """
        if not self._ensemble_mode or not self._window:
            return

        performer = self._performers.get(noodling_id)
        if not performer:
            return

        self._window.set_active_speaker(noodling_id)

        # Switch facets editor to this performer's assembly
        editor = self._get_facets_editor()
        if editor and hasattr(editor, 'select_noodling'):
            editor.select_noodling(noodling_id)

    # =========================================================================
    # ASSEMBLY DISCOVERY
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
                logger.info(f"Assembly: {assembly_path}")
                return str(assembly_path)

            logger.warning(f"Assembly not found at {assembly_path}")
        except Exception as e:
            logger.debug(f"Could not discover guide assembly: {e}")

        return None

    def _discover_yuki_assembly(self) -> Optional[str]:
        """
        Auto-discover Yuki's assembly YAML file.

        Returns:
            Absolute path to Yuki's assembly.yaml, or None if not found
        """
        try:
            studio_dir = Path(__file__).resolve().parent.parent.parent.parent
            assembly_path = studio_dir / _YUKI_ASSEMBLY_RELATIVE
            if assembly_path.exists():
                logger.info(f"Yuki assembly: {assembly_path}")
                return str(assembly_path)
            logger.warning(f"Yuki assembly not found at {assembly_path}")
        except Exception as e:
            logger.debug(f"Could not discover Yuki assembly: {e}")
        return None

    # =========================================================================
    # STAGE INSTANCE DISCOVERY
    # =========================================================================

    def _discover_stage_instances(self, stage_path: str) -> List[dict]:
        """
        Discover noodling instances from a stage directory.

        Reads the ``Instances/`` subdirectory and parses each
        ``instance.yaml`` to resolve noodling template paths.
        Also loads stage.yaml description and stores it on
        ``self._stage_description``.

        Args:
            stage_path: Absolute path to a stage directory
                        (e.g. ``…/Stages/the_nexus``)

        Returns:
            List of dicts, each containing:
            - ``noodling_id``: Directory name (used as performer key)
            - ``name``: Display name (from overrides or dir name)
            - ``assembly_path``: Absolute path to assembly.yaml
            - ``noodling_path``: Absolute path to noodling template dir
            - ``vrm_path``: Absolute path to VRM file (or None)
            - ``description``: One-liner from noodling.yaml (or None)
            - ``appearance``: Prose block from recipe.yaml (or None)
            - ``affect_baseline``: PAD baseline dict from recipe.yaml (or None)
        """
        stage_dir = Path(stage_path)

        # Load stage description
        stage_yaml = stage_dir / 'stage.yaml'
        if stage_yaml.exists():
            try:
                with open(stage_yaml) as f:
                    stage_data = yaml.safe_load(f)
                self._stage_description = stage_data.get('description')
            except Exception as e:
                logger.warning(f"Could not read stage.yaml: {e}")

        instances_dir = stage_dir / 'Instances'
        if not instances_dir.is_dir():
            return []

        results = []
        for instance_dir in sorted(instances_dir.iterdir()):
            if not instance_dir.is_dir():
                continue

            instance_yaml = instance_dir / 'instance.yaml'
            if not instance_yaml.exists():
                continue

            with open(instance_yaml) as f:
                data = yaml.safe_load(f)

            noodling_ref = data.get('noodling', '')
            noodling_path = (instance_dir / noodling_ref).resolve()

            if not noodling_path.is_dir():
                logger.warning(
                    f"Noodling template not found: {noodling_path} "
                    f"(from instance {instance_dir.name})"
                )
                continue

            assembly_path = noodling_path / 'assembly.yaml'
            if not assembly_path.exists():
                logger.warning(f"No assembly.yaml in {noodling_path}")
                continue

            # Display name from overrides, falling back to dir name
            name = data.get('overrides', {}).get('name', instance_dir.name)

            # Load noodling.yaml for VRM path and description
            vrm_path = None
            description = None
            noodling_yaml = noodling_path / 'noodling.yaml'
            if noodling_yaml.exists():
                with open(noodling_yaml) as f:
                    noodling_data = yaml.safe_load(f)
                description = noodling_data.get('description')
                vrm_ref = noodling_data.get('vrm_path', '')
                if vrm_ref:
                    vrm_resolved = (noodling_path / vrm_ref).resolve()
                    if vrm_resolved.exists():
                        vrm_path = str(vrm_resolved)

            # Load recipe.yaml for appearance and affect baseline
            appearance = None
            affect_baseline = None
            recipe_yaml = noodling_path / 'recipe.yaml'
            if recipe_yaml.exists():
                try:
                    with open(recipe_yaml) as f:
                        recipe_data = yaml.safe_load(f)
                    appearance = recipe_data.get('appearance')
                    if appearance and isinstance(appearance, str):
                        appearance = appearance.strip()

                    # Extract PAD baseline from affect dimensions
                    affect = recipe_data.get('affect', {})
                    dimensions = affect.get('dimensions', [])
                    if dimensions:
                        affect_baseline = {}
                        for dim in dimensions:
                            dim_name = dim.get('name')
                            baseline = dim.get('baseline')
                            if dim_name and baseline is not None:
                                affect_baseline[dim_name] = baseline
                except Exception as e:
                    logger.warning(
                        f"Could not read recipe.yaml for "
                        f"{instance_dir.name}: {e}"
                    )

            results.append({
                'noodling_id': instance_dir.name,
                'name': name,
                'assembly_path': str(assembly_path),
                'noodling_path': str(noodling_path),
                'vrm_path': vrm_path,
                'description': description,
                'appearance': appearance,
                'affect_baseline': affect_baseline,
            })

        return results

    # =========================================================================
    # PERFORMANCE LIFECYCLE
    # =========================================================================

    def start_ensemble_from_stage(self, stage_path: str,
                                   play_title: str = "Ensemble"):
        """
        Start an ensemble performance from stage instance definitions.

        Reads the ``Instances/`` directory in the given stage, resolves
        each noodling template, loads assemblies, creates performers,
        and opens the performance window. If only one instance is found,
        starts a single-performer performance instead.

        Args:
            stage_path: Absolute path to the stage directory
            play_title: Title displayed in the window header
        """
        instances = self._discover_stage_instances(stage_path)
        if not instances:
            logger.error(f"No valid instances found in {stage_path}")
            return

        if self._window:
            logger.warning("Performance already active, stopping first")
            self.stop_performance()

        ensemble = len(instances) > 1
        self._ensemble_mode = ensemble
        self._set_demo_mode(True)

        # Store instance metadata for ensemble awareness
        self._instance_metadata = {
            info['noodling_id']: info for info in instances
        }

        # Create performers for each instance
        performers = {}
        for info in instances:
            try:
                llm_client = create_llm_client()
            except Exception as e:
                logger.error(
                    f"Failed to create LLM client for {info['name']}: {e}"
                )
                continue

            performer = NoodlingPerformer(
                noodling_id=info['noodling_id'],
                name=info['name'],
                llm_client=llm_client
            )
            if performer.load_assembly(info['assembly_path']):
                performers[info['noodling_id']] = performer
            else:
                logger.error(
                    f"Failed to load assembly for {info['name']}"
                )

        if not performers:
            logger.error("No performers could be created")
            return

        self._performers = performers
        # Primary performer for backward compat (first in order)
        self._performer = next(iter(performers.values()))

        # Create window
        from .guide_performance_window import GuidePerformanceWindow

        self._window = GuidePerformanceWindow(
            parent_window=self._main_window,
            ensemble_mode=ensemble
        )
        self._play_title = play_title
        self._window.show_play_header(play_title)

        # Connect user message
        self._window.messageSubmitted.connect(self._on_user_message)

        # Wire each performer
        for nid, performer in performers.items():
            if ensemble:
                self._wire_ensemble_performer(performer, self._window, nid)
            else:
                self._wire_performer_to_window(performer, self._window)

        # Load VRM files from stage instance data
        for info in instances:
            if info['vrm_path']:
                if ensemble:
                    self._window.set_vrm(
                        info['vrm_path'],
                        noodling_id=info['noodling_id']
                    )
                else:
                    self._window.set_vrm(info['vrm_path'])

        self._window.show()

        # Set performer names in ensemble mode
        if ensemble:
            for info in instances:
                self._window.set_performer_name(
                    info['noodling_id'], info['name']
                )

        # Set up assembly editor
        try:
            editor = getattr(self._main_window, 'unified_editor', None)
            if editor:
                if ensemble and hasattr(editor, 'set_ensemble_noodlings'):
                    noodlings = [
                        {
                            'id': nid,
                            'name': p.name,
                            'assembly': p.assembly,
                            'assembly_path': p.assembly_path,
                        }
                        for nid, p in performers.items()
                    ]
                    editor.set_ensemble_noodlings(noodlings)
                elif hasattr(editor, 'load_assembly_from_data'):
                    first = self._performer
                    editor.load_assembly_from_data(
                        first.assembly, force_reload=True,
                        source_path=first.assembly_path
                    )

                center_tabs = getattr(self._main_window, 'center_tabs', None)
                if center_tabs:
                    for i in range(center_tabs.count()):
                        if center_tabs.tabText(i) == "Assembly":
                            center_tabs.setCurrentIndex(i)
                            break
        except Exception as e:
            logger.debug(f"Could not load assembly in editor: {e}")

        logger.info(
            f"Ensemble from stage started: {play_title} "
            f"({', '.join(p.name for p in performers.values())})"
        )

    def start_ensemble(self, play_title: str = "Ensemble"):
        """
        Start an ensemble performance with two noodlings on a shared stage.

        Creates Ajo and Yuki performers, loads their assemblies, creates
        a shared window with two VRM viewports, and wires turn-taking.

        Args:
            play_title: Title displayed in the window header
        """
        if self._window:
            logger.warning("Performance already active, stopping first")
            self.stop_performance()

        self._ensemble_mode = True
        self._set_demo_mode(True)

        # Create shared LLM client
        try:
            llm_client = create_llm_client()
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            return

        # --- Create Ajo performer ---
        ajo_assembly = self._discover_guide_assembly()
        if not ajo_assembly:
            logger.error("Ajo assembly not found")
            return

        ajo = NoodlingPerformer(
            noodling_id='ajo', name='Ajo', llm_client=llm_client
        )
        if not ajo.load_assembly(ajo_assembly):
            logger.error("Failed to load Ajo assembly")
            return

        # --- Create Yuki performer (needs separate LLM client) ---
        try:
            yuki_llm = create_llm_client()
        except Exception as e:
            logger.error(f"Failed to create second LLM client: {e}")
            return

        yuki_assembly = self._discover_yuki_assembly()
        if not yuki_assembly:
            logger.error("Yuki assembly not found")
            return

        yuki = NoodlingPerformer(
            noodling_id='yuki', name='Yuki', llm_client=yuki_llm
        )
        if not yuki.load_assembly(yuki_assembly):
            logger.error("Failed to load Yuki assembly")
            return

        self._performers = {'ajo': ajo, 'yuki': yuki}
        self._performer = ajo  # Primary performer for backward compat

        # --- Create ensemble window ---
        from .guide_performance_window import GuidePerformanceWindow

        self._window = GuidePerformanceWindow(
            parent_window=self._main_window,
            ensemble_mode=True
        )
        self._play_title = play_title
        self._window.show_play_header(play_title)

        # Connect user message to ensemble handler
        self._window.messageSubmitted.connect(self._on_user_message)

        # Wire each performer to the window
        for nid, performer in self._performers.items():
            self._wire_ensemble_performer(performer, self._window, nid)

        # Load VRM for Ajo (Yuki has no VRM yet -- shows placeholder)
        ajo_vrm = self._discover_guide_vrm()
        if ajo_vrm:
            self._window.set_vrm(ajo_vrm, noodling_id='ajo')

        self._window.show()

        # Set performer names in stage view
        self._window.set_performer_name('ajo', 'Ajo Majo')
        self._window.set_performer_name('yuki', 'Yuki Cyberfox')

        # Set up assembly editor noodling selector + load Ajo's assembly
        try:
            editor = getattr(self._main_window, 'unified_editor', None)
            if editor:
                noodlings = [
                    {'id': 'ajo', 'name': 'Ajo Majo',
                     'assembly': ajo.assembly,
                     'assembly_path': ajo.assembly_path},
                    {'id': 'yuki', 'name': 'Yuki Cyberfox',
                     'assembly': yuki.assembly,
                     'assembly_path': yuki.assembly_path},
                ]
                if hasattr(editor, 'set_ensemble_noodlings'):
                    editor.set_ensemble_noodlings(noodlings)
                elif hasattr(editor, 'load_assembly_from_data'):
                    # Fallback: just load Ajo's assembly
                    editor.load_assembly_from_data(
                        ajo.assembly, force_reload=True,
                        source_path=ajo.assembly_path
                    )

                center_tabs = getattr(self._main_window, 'center_tabs', None)
                if center_tabs:
                    for i in range(center_tabs.count()):
                        if center_tabs.tabText(i) == "Assembly":
                            center_tabs.setCurrentIndex(i)
                            break
        except Exception as e:
            logger.debug(f"Could not load assembly in editor: {e}")

        logger.info(f"Ensemble started: {play_title} (Ajo + Yuki)")

    def _wire_ensemble_performer(self, performer: NoodlingPerformer,
                                  window, noodling_id: str):
        """
        Wire a performer's signals for ensemble mode.

        Routes VRM updates to the correct viewport and dialogue text
        includes the noodling name. Turn advancement happens on
        executionFinished.

        Args:
            performer: NoodlingPerformer to wire
            window: GuidePerformanceWindow (ensemble mode)
            noodling_id: Identifier for routing
        """
        name = performer.name

        # Affect -> VRM blend shapes (routed by noodling_id)
        performer.affectReady.connect(
            lambda shapes, nid=noodling_id: window.set_blend_shapes(
                shapes, noodling_id=nid
            )
        )

        # Plain text response (no performance script)
        performer.responseReady.connect(
            lambda text, nid=noodling_id, n=name: window.append_noodling_text(
                nid, n, text
            )
        )

        # Performance script -> typed text delivery with name prefix
        performer.performanceReady.connect(
            lambda script, nid=noodling_id, n=name: window.begin_noodling_text(
                nid, n
            )
        )
        performer.characterRevealed.connect(window.append_character)
        performer.speakingStateChanged.connect(
            lambda speaking, p=performer, nid=noodling_id: (
                window.set_speaking_mode(
                    speaking, p.speaking_intensity, noodling_id=nid
                )
            )
        )
        performer.performanceFinished.connect(
            lambda nid=noodling_id: self._on_ensemble_performance_finished(nid)
        )

        # Execution lifecycle (turn-based)
        performer.executionStarted.connect(
            lambda n=name: window.set_busy(True, name=n)
        )
        performer.executionFinished.connect(
            lambda nid=noodling_id: self._on_ensemble_turn_finished(nid)
        )
        performer.errorOccurred.connect(
            lambda msg, nid=noodling_id: self._on_ensemble_error(nid, msg)
        )

        # Live viz: per-facet completion for facets editor
        performer.facetCompleted.connect(self._on_facet_completed)

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

        # --- Create LLM client and performer ---
        assembly_path = self._discover_guide_assembly()
        if not assembly_path:
            logger.warning("No assembly found, Ajo cannot think")
            return

        try:
            llm_client = create_llm_client()
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            return

        self._performer = NoodlingPerformer(
            noodling_id='ajo',
            name='Ajo',
            llm_client=llm_client
        )

        if self._channel_bus:
            self._performer.set_channel_bus(self._channel_bus)

        if not self._performer.load_assembly(assembly_path):
            logger.error("Failed to load assembly")
            self._performer = None
            return

        # Create the floating performance window
        from .guide_performance_window import GuidePerformanceWindow

        self._window = GuidePerformanceWindow(
            parent_window=self._main_window
        )
        self._play_title = play_title
        self._window.show_play_header(play_title)

        # Wire guide cue handler (may have been set externally or by _setup_play_pipeline)
        if self._guide_cue_handler:
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

        # Connect user message to manager (routes to performer)
        self._window.messageSubmitted.connect(self._on_user_message)

        # Ctrl+Shift+F - FACS expression test mode
        from PyQt6.QtGui import QShortcut, QKeySequence
        facs_shortcut = QShortcut(QKeySequence("Ctrl+Shift+F"), self._window)
        facs_shortcut.activated.connect(self.toggle_expression_test)

        if self._channel_bus and hasattr(self._window, 'messageSent') and self._window.messageSent:
            self._window.messageSent.connect(self._on_user_message_for_channel)

        # Wire performer signals to window
        self._wire_performer_to_window(self._performer, self._window)

        # Load VRM -- use provided path, or auto-discover guide VRM
        if not vrm_path:
            vrm_path = self._discover_guide_vrm()
        if vrm_path:
            logger.info(f"Loading VRM: {vrm_path}")
            self._window.set_vrm(vrm_path)
        else:
            logger.warning("No VRM found")

        self._window.show()

        # Load assembly into editor for visibility
        if self._performer.assembly:
            try:
                editor = getattr(self._main_window, 'unified_editor', None)
                if editor and hasattr(editor, 'load_assembly_from_data'):
                    source = self._performer.assembly_path
                    editor.load_assembly_from_data(
                        self._performer.assembly, force_reload=True,
                        source_path=source
                    )
                    logger.info("Assembly loaded in editor")

                    # Switch to Assembly tab so user sees the graph
                    center_tabs = getattr(self._main_window, 'center_tabs', None)
                    if center_tabs:
                        for i in range(center_tabs.count()):
                            if center_tabs.tabText(i) == "Assembly":
                                center_tabs.setCurrentIndex(i)
                                break
            except Exception as e:
                logger.debug(f"Could not load assembly in editor: {e}")

        # Start Brenda after window is visible
        if self._director:
            self._director.start()
            logger.info(f"Brenda directing: {play_title}")

        logger.info(f"Performance started: {play_title}")

    def _wire_performer_to_window(self, performer: NoodlingPerformer, window):
        """
        Connect a performer's signals to a window for rendering.

        Args:
            performer: NoodlingPerformer to wire
            window: GuidePerformanceWindow to render into
        """
        # Affect -> VRM blend shapes
        performer.affectReady.connect(
            lambda shapes: window.set_blend_shapes(shapes)
        )

        # Plain text response (no performance script)
        performer.responseReady.connect(
            lambda text: window.append_guide_text(text)
        )

        # Performance script -> typed text delivery
        performer.performanceReady.connect(
            lambda script: window.begin_guide_text()
        )
        performer.characterRevealed.connect(window.append_character)
        performer.speakingStateChanged.connect(
            lambda speaking: self._on_speaking_state_changed(performer, speaking)
        )
        performer.performanceFinished.connect(
            lambda: self._on_performance_finished()
        )

        # Execution lifecycle
        performer.executionStarted.connect(
            lambda: window.set_busy(True)
        )
        performer.executionFinished.connect(
            lambda: self._on_execution_finished()
        )
        performer.errorOccurred.connect(
            lambda msg: self._on_execution_error(msg)
        )

        # Live viz: per-facet completion for facets editor
        performer.facetCompleted.connect(self._on_facet_completed)

    def _setup_play_pipeline(self, play_path: str):
        """
        Create the full play pipeline: ChannelBus -> BrendaDirector -> GuideCueHandler.

        Args:
            play_path: Path to the .play.yaml file
        """
        path = Path(play_path)
        if not path.exists():
            logger.warning(f"Play file not found: {play_path}")
            return

        # Create channel bus for this performance
        self._channel_bus = ChannelBus()

        # Create and load Brenda
        self._director = BrendaDirector(self._channel_bus)
        if not self._director.load_play(str(path)):
            logger.error(f"Failed to load play: {play_path}")
            self._director = None
            self._channel_bus = None
            return

        # Create guide cue handler on the same bus
        self._guide_cue_handler = GuideCueHandler(self._channel_bus, "guide")

        # Start tick timer for Brenda (200ms interval)
        self._tick_timer = QTimer()
        self._tick_timer.timeout.connect(self._director.tick)
        self._tick_timer.start(200)

        logger.info(f"Play pipeline ready: {path.stem}")

    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================

    def _on_user_message(self, message: str):
        """
        Route a user message to the performer(s) for assembly execution.

        In single mode, sends directly to the performer.
        In ensemble mode, starts turn-taking sequence.

        Args:
            message: The user's message text
        """
        if self._ensemble_mode:
            self._on_user_message_ensemble(message)
            return

        if not self._performer:
            if self._window:
                self._window._show_error("Assembly not loaded.")
                self._window.set_busy(False)
            return

        # Build extra context (Brenda direction)
        extra_context = {}
        if self._guide_cue_handler:
            direction = self._guide_cue_handler.build_system_prompt_addition()
            if direction:
                extra_context['brenda_direction'] = direction

        # Execute via performer
        self._performer.execute(message, extra_context if extra_context else None)

        # Emit live execution events for facets editor visualization
        self._current_execution_id = str(uuid.uuid4())[:8]
        self._emit_execution_start_events()

    # =========================================================================
    # ENSEMBLE TURN-TAKING
    # =========================================================================

    def _on_user_message_ensemble(self, message: str):
        """
        Start the ensemble turn-taking sequence.

        User types a message -> Ajo responds first -> Yuki responds next
        (aware of what Ajo said) -> wait for user.

        Args:
            message: The user's message text
        """
        if not self._performers:
            if self._window:
                self._window._show_error("No performers loaded.")
                self._window.set_busy(False)
            return

        self._turn_responses = {}
        self._turn_queue = list(self._performers.keys())  # ['ajo', 'yuki']
        self._pending_message = message

        # Record user message in shared ensemble history
        self._ensemble_history.append({
            'role': 'User',
            'content': message,
        })

        # Start events are emitted per-noodling in _advance_ensemble_turn()
        self._advance_ensemble_turn()

    def _format_ensemble_history(self) -> str:
        """
        Format the shared ensemble history for injection into prompts.

        Returns the last 30 messages as a chronological transcript with
        speaker names. Keeps the window small to avoid token bloat while
        giving each noodling full awareness of the conversation.

        Returns:
            Formatted transcript string, or "(No previous conversation)"
        """
        if not self._ensemble_history:
            return "(No previous conversation)"

        recent = self._ensemble_history[-30:]
        lines = []
        for msg in recent:
            lines.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(lines)

    def _format_present_entities(self, exclude_nid: str) -> str:
        """
        Format descriptions of other noodlings for awareness injection.

        Builds a prose block like:
            Also here with you:
            - Ajo Majo: A small chibi axolotl... Currently seems happy and energetic.
            - Krampus: A seven-year-old boy... Currently seems determined.

        Args:
            exclude_nid: Noodling ID to exclude (you don't describe yourself)

        Returns:
            Formatted entity descriptions, or empty string if alone
        """
        lines = []
        for nid, meta in self._instance_metadata.items():
            if nid == exclude_nid:
                continue

            name = meta.get('name', nid)
            # Prefer appearance (richer prose), fall back to description
            desc = meta.get('appearance') or meta.get('description') or ''

            # Append current mood if available
            performer = self._performers.get(nid)
            mood_str = ''
            if performer and performer.last_affect:
                mood_str = self._describe_affect(performer.last_affect)

            if desc and mood_str:
                lines.append(f"- {name}: {desc} Currently seems {mood_str}.")
            elif desc:
                lines.append(f"- {name}: {desc}")
            elif mood_str:
                lines.append(f"- {name}: Currently seems {mood_str}.")
            else:
                lines.append(f"- {name}")

        if not lines:
            return ""
        return "Also here with you:\n" + "\n".join(lines)

    @staticmethod
    def _describe_affect(pad: Dict) -> str:
        """
        Convert PAD values to a brief natural-language mood description.

        Args:
            pad: Dict with valence (-1..1), arousal (0..1), dominance (0..1)

        Returns:
            Brief mood phrase (e.g. "happy and energetic")
        """
        valence = pad.get('valence', 0.0)
        arousal = pad.get('arousal', 0.5)
        dominance = pad.get('dominance', 0.5)

        # Valence descriptor (-1..1 range)
        if valence > 0.4:
            v_word = 'happy'
        elif valence > 0.1:
            v_word = 'pleasant'
        elif valence > -0.1:
            v_word = 'neutral'
        elif valence > -0.4:
            v_word = 'subdued'
        else:
            v_word = 'unhappy'

        # Arousal descriptor
        if arousal > 0.7:
            a_word = 'energetic'
        elif arousal > 0.5:
            a_word = 'engaged'
        elif arousal > 0.3:
            a_word = 'calm'
        else:
            a_word = 'quiet'

        # Dominance modifier (only if notably high or low)
        d_mod = ''
        if dominance > 0.7:
            d_mod = ' and confident'
        elif dominance < 0.3:
            d_mod = ' and uncertain'

        return f"{v_word} and {a_word}{d_mod}"

    def _advance_ensemble_turn(self):
        """Advance to the next noodling in the turn sequence."""
        if not self._turn_queue:
            # All turns complete
            self._active_noodling_id = 'default'
            if self._window:
                self._window.set_active_speaker(None)
                self._window.set_busy(False)
            self._emit_execution_complete_events()
            return

        nid = self._turn_queue.pop(0)
        performer = self._performers.get(nid)
        if not performer:
            self._advance_ensemble_turn()
            return

        # Highlight active speaker in stage view
        if self._window:
            self._window.set_active_speaker(nid)

        # Tag events with this noodling's ID
        self._active_noodling_id = nid
        self._current_execution_id = str(uuid.uuid4())[:8]
        self._emit_execution_start_events()

        # Build rich perception context
        extra_context = {
            'stage_context': self._stage_description or '',
            'present_entities': self._format_present_entities(nid),
            'ensemble_history': self._format_ensemble_history(),
            'conversation_history': self._format_ensemble_history(),
        }

        if self._guide_cue_handler:
            direction = self._guide_cue_handler.build_system_prompt_addition()
            if direction:
                extra_context['brenda_direction'] = direction

        # Include previous noodlings' responses from this round
        for prev_nid, response in self._turn_responses.items():
            extra_context[f'{prev_nid}_said'] = response

        # Add mood cross-pollination (other noodlings' affect state)
        for other_nid, other_performer in self._performers.items():
            if other_nid != nid and other_performer.last_affect:
                extra_context[f'{other_nid}_mood'] = self._describe_affect(
                    other_performer.last_affect
                )

        performer.execute(self._pending_message, extra_context)

    def _on_ensemble_turn_finished(self, noodling_id: str):
        """
        Handle one noodling finishing its turn.

        Stores the response and advances to the next noodling.

        Args:
            noodling_id: The noodling that just finished
        """
        performer = self._performers.get(noodling_id)
        if performer:
            self._turn_responses[noodling_id] = performer.last_response

            # Record noodling response in shared ensemble history
            if performer.last_response:
                self._ensemble_history.append({
                    'role': performer.name,
                    'content': performer.last_response,
                })

        self._advance_ensemble_turn()

    def _on_ensemble_performance_finished(self, noodling_id: str):
        """
        Handle typed text playback finishing for an ensemble performer.

        Finalizes the text block and turns off speaking animation.

        Args:
            noodling_id: The noodling whose performance finished
        """
        if self._window:
            self._window.end_noodling_text()
            self._window.set_speaking_mode(False, noodling_id=noodling_id)

    def _on_ensemble_error(self, noodling_id: str, error_msg: str):
        """
        Handle assembly error for an ensemble performer.

        Args:
            noodling_id: The noodling that errored
            error_msg: Error description
        """
        if self._window:
            performer = self._performers.get(noodling_id)
            name = performer.name if performer else noodling_id
            self._window._show_error(f"{name}: {error_msg}")

        # Emit live execution events for facets editor visualization
        self._current_execution_id = str(uuid.uuid4())[:8]
        self._emit_execution_start_events()

    # =========================================================================
    # PERFORMER SIGNAL HANDLERS
    # =========================================================================

    def _on_execution_finished(self):
        """Handle completed assembly execution (covers both plain text and typed).

        Clears window busy state, emits live viz completion events,
        and reports to Brenda. For performance scripts, this fires
        after PerformancePlayer finishes (via performanceFinished chain).
        """
        if self._window:
            self._window.set_busy(False)

        # Emit live viz completion events
        self._emit_execution_complete_events()

        # Report to Brenda for non-performance responses
        # (performance responses report in _on_performance_finished)
        if (self._guide_cue_handler and self._performer
                and self._performer.last_response):
            self._guide_cue_handler.report_response(
                self._performer.last_response.strip(),
                self._performer._last_user_message
            )

    def _on_execution_error(self, error_msg: str):
        """Handle assembly execution error.

        Args:
            error_msg: Error description
        """
        if self._window:
            self._window._show_error(error_msg)
            self._window.set_busy(False)

        # Emit error events for facets editor visualization
        self._emit_execution_error_events(error_msg)

    def _on_speaking_state_changed(self, performer: NoodlingPerformer,
                                   speaking: bool):
        """
        Update VRM speaking animation in response to PerformancePlayer state.

        Args:
            performer: The performer whose speaking state changed
            speaking: True when typing active, False on pause or finished
        """
        if self._window and self._window._vrm_viewport:
            intensity = performer.speaking_intensity
            self._window._vrm_viewport.set_speaking_mode(speaking, intensity)

    def _on_performance_finished(self):
        """Handle completed typed text playback.

        Finalizes the typed-text block and turns off speaking animation.
        Busy state and Brenda reporting are handled by _on_execution_finished
        which fires after this (via the performer's signal chain).
        """
        if self._window:
            self._window.end_guide_text()

        # Turn off speaking animation
        if self._window and self._window._vrm_viewport:
            self._window._vrm_viewport.set_speaking_mode(False)

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
    # PER-FACET COMPLETION (LIVE VIZ)
    # =========================================================================

    def _on_facet_completed(self, facet_id: str, outputs: dict):
        """
        Handle individual facet completion for live visualization.

        The mood-first expression and affect pipeline are already handled
        inside NoodlingPerformer. This method only drives the facets editor
        live visualization (node pulsing/coloring).

        Args:
            facet_id: The ID of the completed facet
            outputs: The facet's output dict
        """
        eid = self._current_execution_id
        if not eid:
            return

        # Facet turns green immediately
        self._emit_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_complete',
            'source_id': facet_id,
            'execution_id': eid,
            'data': {'execution_id': eid, 'outputs': outputs}
        })

        # Emit data flow events for this facet's outgoing connections
        if facet_id == 'sentiment':
            self._emit_execution_event({
                'type': 'facet_execution',
                'subtype': 'data_flow',
                'from_facet': 'sentiment',
                'to_facet': 'outgoing',
                'execution_id': eid,
            })
        elif facet_id == 'response':
            self._emit_execution_event({
                'type': 'facet_execution',
                'subtype': 'data_flow',
                'from_facet': 'response',
                'to_facet': 'performance',
                'execution_id': eid,
            })
            # Performance facet start (it runs immediately after response)
            self._emit_execution_event({
                'type': 'facet_execution',
                'subtype': 'facet_start',
                'source_id': 'performance',
                'execution_id': eid,
                'data': {'execution_id': eid}
            })
        elif facet_id == 'performance':
            self._emit_execution_event({
                'type': 'facet_execution',
                'subtype': 'data_flow',
                'from_facet': 'performance',
                'to_facet': 'outgoing',
                'execution_id': eid,
            })

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
        logger.info("FACS test: started expression test cycle")

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
        logger.info("FACS test: stopped")

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

        logger.info(f"FACS test: {name}: {shapes}")

    # =========================================================================
    # FACETS EDITOR LIVE VISUALIZATION
    # =========================================================================

    def _get_facets_editor(self):
        """
        Get the assembly editor panel (cached).

        On first call, looks up ``main_window.unified_editor`` and caches
        the result so subsequent calls are free.

        Returns:
            UnifiedEditorPanel or None
        """
        if self._assembly_editor is None:
            editor = getattr(self._main_window, 'unified_editor', None)
            if editor is not None:
                self._assembly_editor = editor
        return self._assembly_editor

    def _emit_execution_event(self, event: dict):
        """
        Feed an execution event directly to the facets editor.

        Bypasses the WebSocket layer entirely -- events are delivered
        synchronously on the Qt main thread, exactly as
        _handle_execution_event expects.

        In ensemble mode, each event is tagged with ``noodling_id`` so
        the facets editor can filter events by the selected noodling.

        Args:
            event: Execution event dict matching the WebSocket protocol
        """
        if 'noodling_id' not in event:
            event['noodling_id'] = self._active_noodling_id
        editor = self._get_facets_editor()
        if editor and hasattr(editor, '_handle_execution_event'):
            editor._handle_execution_event(event)

    def _emit_execution_start_events(self):
        """
        Emit events when assembly execution begins.

        INCOMING completes immediately (pass-through), then Response and
        Sentiment facets start processing. They stay in 'processing' state
        (yellow pulse) until the result arrives from the worker thread.
        """
        eid = self._current_execution_id
        if not eid:
            return

        user_message = ''
        if self._performer:
            user_message = self._performer._last_user_message

        # Cycle begins
        self._emit_execution_event({
            'type': 'facet_execution',
            'subtype': 'cycle_start',
            'execution_id': eid,
            'data': {'execution_id': eid}
        })

        # INCOMING receives user input
        self._emit_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'incoming',
            'execution_id': eid,
            'data': {
                'execution_id': eid,
                'inputs': {'user_message': user_message}
            }
        })

        def _after_incoming():
            if self._current_execution_id != eid:
                return  # Stale execution, skip

            # INCOMING completes (pass-through)
            self._emit_execution_event({
                'type': 'facet_execution',
                'subtype': 'facet_complete',
                'source_id': 'incoming',
                'execution_id': eid,
                'data': {
                    'execution_id': eid,
                    'outputs': {'out': user_message}
                }
            })

            # Data flows from INCOMING to both parallel facets
            for target in ('response', 'sentiment'):
                self._emit_execution_event({
                    'type': 'facet_execution',
                    'subtype': 'data_flow',
                    'from_facet': 'incoming',
                    'to_facet': target,
                    'execution_id': eid,
                })

            # Response and Sentiment start processing (stay yellow until result)
            for facet_id in ('response', 'sentiment'):
                self._emit_execution_event({
                    'type': 'facet_execution',
                    'subtype': 'facet_start',
                    'source_id': facet_id,
                    'execution_id': eid,
                    'data': {
                        'execution_id': eid,
                        'inputs': {'in': user_message}
                    }
                })

        QTimer.singleShot(80, _after_incoming)

    def _emit_execution_complete_events(self, result=None):
        """
        Emit events when assembly execution completes.

        Individual facet completions are already emitted by _on_facet_completed
        via the per-facet callback. This method handles OUTGOING and the
        cycle_complete event.

        Args:
            result: ExecutionResult from FacetExecutor (optional)
        """
        eid = self._current_execution_id
        if not eid:
            return

        def _complete_outgoing():
            if self._current_execution_id != eid:
                return

            # OUTGOING starts
            self._emit_execution_event({
                'type': 'facet_execution',
                'subtype': 'facet_start',
                'source_id': 'outgoing',
                'execution_id': eid,
                'data': {'execution_id': eid, 'inputs': {}}
            })

            def _finish_cycle():
                if self._current_execution_id != eid:
                    return

                # OUTGOING completes
                self._emit_execution_event({
                    'type': 'facet_execution',
                    'subtype': 'facet_complete',
                    'source_id': 'outgoing',
                    'execution_id': eid,
                    'data': {'execution_id': eid, 'outputs': {}}
                })

                # Cycle ends
                self._emit_execution_event({
                    'type': 'facet_execution',
                    'subtype': 'cycle_complete',
                    'execution_id': eid,
                    'data': {'execution_id': eid}
                })

                self._current_execution_id = None

            QTimer.singleShot(100, _finish_cycle)

        QTimer.singleShot(150, _complete_outgoing)

    def _emit_execution_error_events(self, error_msg: str):
        """
        Emit error events for facets editor visualization.

        Shows red flash on processing nodes when execution fails.

        Args:
            error_msg: Error description
        """
        eid = self._current_execution_id
        if not eid:
            return

        # Error on any processing facets
        for facet_id in ('response', 'sentiment', 'performance'):
            self._emit_execution_event({
                'type': 'facet_execution',
                'subtype': 'facet_error',
                'source_id': facet_id,
                'execution_id': eid,
                'data': {'execution_id': eid, 'error': error_msg}
            })

        # End the cycle
        self._emit_execution_event({
            'type': 'facet_execution',
            'subtype': 'cycle_complete',
            'execution_id': eid,
            'data': {'execution_id': eid}
        })

        self._current_execution_id = None

    # =========================================================================
    # PERFORMANCE LIFECYCLE (STOP)
    # =========================================================================

    def stop_performance(self):
        """
        Stop the current performance.

        Closes the floating window, stops all performers, stops Brenda,
        disables demo mode, and tears down the play pipeline.
        """
        # Stop all ensemble performers
        if self._ensemble_mode:
            for nid, performer in self._performers.items():
                performer.stop()
            self._performers = {}
            self._turn_queue = []
            self._turn_responses = {}
            self._active_noodling_id = 'default'
            self._ensemble_mode = False
            self._ensemble_history = []
            self._stage_description = None
            self._instance_metadata = {}

            # Clear facets editor noodling selector
            try:
                editor = self._get_facets_editor()
                if editor and hasattr(editor, 'clear_ensemble_noodlings'):
                    editor.clear_ensemble_noodlings()
            except Exception:
                pass

        # Stop primary performer
        if self._performer:
            self._performer.stop()
            self._performer = None

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

        Returns:
            Absolute path to VRM file, or None if not found
        """
        try:
            studio_dir = Path(__file__).resolve().parent.parent.parent.parent
            project_root = studio_dir.parent.parent

            vrm_path = project_root / _GUIDE_VRM_RELATIVE
            if vrm_path.exists():
                logger.info(f"Auto-discovered VRM: {vrm_path}")
                return str(vrm_path)

            logger.warning(f"VRM not found at {vrm_path}")
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


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
