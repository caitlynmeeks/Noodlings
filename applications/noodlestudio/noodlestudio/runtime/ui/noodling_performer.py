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
#   Noodling Performer
#
#   One noodling's complete cognition + playback pipeline.
#   Owns a FacetAssembly, FacetExecutor, conversation history,
#   affect state, and PerformancePlayer. Emits signals for
#   UI rendering -- no window references, no editor coupling.
#
#   Extracted from GuidePerformanceManager to enable ensemble
#   mode (multiple noodlings on a shared stage).
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.noodling_performer
# PURPOSE:  Per-Noodling Cognition + Playback
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NoodlingPerformer
#   _AssemblyWorker
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
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

logger = logging.getLogger(__name__)


# =============================================================================
# Assembly Worker (QThread for async FacetExecutor.execute)
# =============================================================================

class _AssemblyWorker(QThread):
    """Worker thread that executes a facet assembly in its own event loop.

    Emits facetCompleted for each individual facet as it finishes, enabling
    mood-first expression updates (Sentiment completes before Response).
    """

    resultReady = pyqtSignal(object)    # ExecutionResult
    errorOccurred = pyqtSignal(str)     # Error message
    facetCompleted = pyqtSignal(str, object)  # (facet_id, outputs_dict)
    facetTraceReady = pyqtSignal(str, object)  # (facet_id, trace_dict)
    streamTokenReady = pyqtSignal(str, str)   # (facet_id, text) for streaming delivery

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
            def _on_facet_done(facet_id, outputs):
                self.facetCompleted.emit(facet_id, outputs)
                # Emit trace data if available
                for facet in self._assembly.facets:
                    if facet.id == facet_id:
                        trace = getattr(facet, '_last_trace', None)
                        if trace:
                            trace_with_id = dict(trace)
                            trace_with_id['facet_id'] = facet_id
                            trace_with_id['facet_name'] = facet.name
                            trace_with_id['facet_type'] = facet.facet_type
                            self.facetTraceReady.emit(facet_id, trace_with_id)
                        break

            def _on_stream_token(facet_id, text):
                self.streamTokenReady.emit(facet_id, text)

            result = loop.run_until_complete(
                self._executor.execute(
                    self._assembly,
                    incoming_data=self._message,
                    context=self._context,
                    on_facet_complete=_on_facet_done,
                    on_stream_token=_on_stream_token
                )
            )
            self.resultReady.emit(result)
        except Exception as e:
            logger.error(f"Assembly execution failed: {e}")
            self.errorOccurred.emit(str(e))
        finally:
            loop.close()


# =============================================================================
# LLM Client Creation (Shared Utility)
# =============================================================================

def create_llm_client():
    """
    Create a HeadlessLLMClient from the editor's provider settings.

    Reads from ProviderManager and ModelLabelManager singletons so
    noodlings respect whatever provider the user has configured.

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

    # Read model prefixes for all labels
    label_prefixes = {}
    for label in label_mgr.get_all_labels():
        prefix = label_mgr.get_model_prefix(label)
        if prefix:
            label_prefixes[label.upper()] = prefix

    config = LLMConfig(
        provider=provider.type if provider else "ollama",
        model=model_name or "",
        api_key=provider.api_key if provider else "",
        base_url=provider.base_url if provider else "",
        model_labels=model_labels,
        label_prefixes=label_prefixes
    )

    logger.info(f"LLM client: provider={config.provider}, labels={model_labels}")

    return HeadlessLLMClient(config)


# =============================================================================
# Noodling Performer
# =============================================================================

class NoodlingPerformer(QObject):
    """
    One noodling's complete cognition + playback pipeline.

    Encapsulates everything needed to run a single noodling:
    - Assembly loading (FacetAssembly from YAML)
    - FacetExecutor creation
    - _AssemblyWorker management (spawn, connect signals)
    - Conversation history (per-noodling, formatted for prompt injection)
    - Affect state (sentiment -> FACSMapper -> VRM blend shapes)
    - Performance playback (PerformancePlayer for typed text)

    No window references, no editor coupling. Emits signals for
    UI rendering so the manager can wire them to any window.

    Signals:
        responseReady(str): Raw response text for conversation history
        performanceReady(dict): Performance script JSON for typed delivery
        affectReady(dict): VRM blend shape weights from sentiment facet
        facetCompleted(str, object): (facet_id, outputs) for live viz
        executionStarted(): Assembly execution began
        executionFinished(): Assembly execution completed (text ready)
        errorOccurred(str): Assembly execution failed
        characterRevealed(str): Single character from PerformancePlayer
        speakingStateChanged(bool): Speaking animation from PerformancePlayer
        performanceFinished(): Typed text delivery completed
    """

    # Cognition signals
    responseReady = pyqtSignal(str)          # raw response text
    performanceReady = pyqtSignal(dict)      # performance_script JSON
    affectReady = pyqtSignal(dict)           # VRM blend shapes
    facetCompleted = pyqtSignal(str, object) # (facet_id, outputs) for live viz
    turnTraceReady = pyqtSignal(str, list)   # (noodling_id, [trace_dicts])
    executionStarted = pyqtSignal()
    executionFinished = pyqtSignal()
    errorOccurred = pyqtSignal(str)

    # Playback signals (proxied from PerformancePlayer)
    characterRevealed = pyqtSignal(str)
    speakingStateChanged = pyqtSignal(bool)
    performanceFinished = pyqtSignal()

    def __init__(self, noodling_id: str, name: str, llm_client,
                 parent: Optional[QObject] = None):
        """
        Initialize a noodling performer.

        Args:
            noodling_id: Unique identifier (e.g. 'ajo', 'yuki')
            name: Display name (e.g. 'Ajo', 'Yuki')
            llm_client: HeadlessLLMClient for LLM calls
            parent: Optional QObject parent
        """
        super().__init__(parent)
        self._noodling_id = noodling_id
        self._name = name
        self._llm_client = llm_client

        # Assembly execution
        self._assembly = None       # FacetAssembly
        self._assembly_path = None  # Path to assembly YAML
        self._executor = None       # FacetExecutor

        # Worker thread (one per execution, replaced each message)
        self._worker = None

        # Conversation state
        self._conversation_history: List[Dict] = []
        self._last_user_message: str = ""
        self._last_response: str = ""

        # Mood-first tracking
        self._sentiment_applied_early = False

        # Stored affect state (populated by _apply_affect)
        self._last_pad_values: Optional[Dict] = None

        # Streaming delivery state
        self._streaming_mode: Optional[str] = None
        self._streaming_first_token: bool = True

        # Performance player (created lazily)
        self._performance_player = None

        # Channel bus reference (set by manager for channel publishing)
        self._channel_bus = None

        # Pause gate: when True, execute() returns immediately
        self._paused = False

        logger.info(f"NoodlingPerformer[{noodling_id}] initialized: {name}")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def noodling_id(self) -> str:
        """Unique identifier for this noodling."""
        return self._noodling_id

    @property
    def name(self) -> str:
        """Display name for this noodling."""
        return self._name

    @property
    def paused(self) -> bool:
        """Whether this performer's cognition is paused."""
        return self._paused

    def set_paused(self, paused: bool):
        """Set the pause state. When paused, execute() is a no-op."""
        self._paused = paused
        logger.info(f"NoodlingPerformer[{self._noodling_id}] {'paused' if paused else 'resumed'}")

    @property
    def last_response(self) -> str:
        """The most recent response text."""
        return self._last_response

    @property
    def conversation_history(self) -> List[Dict]:
        """The conversation history list."""
        return self._conversation_history

    @property
    def assembly(self):
        """The loaded FacetAssembly (or None)."""
        return self._assembly

    @property
    def assembly_path(self) -> Optional[str]:
        """Path to the loaded assembly YAML (or None)."""
        return self._assembly_path

    @property
    def is_executing(self) -> bool:
        """Whether an assembly execution is in progress."""
        return self._worker is not None and self._worker.isRunning()

    @property
    def last_affect(self) -> Optional[Dict]:
        """Most recent PAD values from the sentiment facet.

        Returns a dict with ``valence``, ``arousal``, ``dominance`` keys
        (raw values as parsed from the Mood Reader output), or None if
        no affect has been applied yet.
        """
        return self._last_pad_values

    # =========================================================================
    # CHANNEL BUS
    # =========================================================================

    def set_channel_bus(self, bus):
        """
        Set the ChannelBus for inter-noodling communication.

        Args:
            bus: ChannelBus instance (shared across ensemble)
        """
        self._channel_bus = bus

    # =========================================================================
    # ASSEMBLY LOADING
    # =========================================================================

    def load_assembly(self, assembly_path: str) -> bool:
        """
        Load a facet assembly and create the execution pipeline.

        Args:
            assembly_path: Absolute path to assembly YAML file

        Returns:
            True if assembly loaded successfully
        """
        from noodlestudio.core.facet_system import FacetAssembly
        from noodlestudio.core.facet_executor import FacetExecutor

        try:
            self._assembly = FacetAssembly.load_yaml(assembly_path)
            self._assembly_path = assembly_path
            logger.info(f"[{self._noodling_id}] Loaded assembly: "
                        f"{self._assembly.name} "
                        f"({len(self._assembly.facets)} facets, "
                        f"{len(self._assembly.connections)} connections)")
        except Exception as e:
            logger.error(f"[{self._noodling_id}] Failed to load assembly: {e}")
            return False

        # Create executor (event bus disabled -- conflicts with Qt thread model)
        self._executor = FacetExecutor(
            llm_client=self._llm_client,
            channel_bus=self._channel_bus,
            use_event_bus=False
        )

        logger.info(f"[{self._noodling_id}] Assembly execution pipeline ready")
        return True

    # =========================================================================
    # CONVERSATION HISTORY
    # =========================================================================

    def _format_history(self) -> str:
        """
        Format conversation history for injection into facet prompts.

        Keeps the last 20 messages (10 exchanges) to avoid token bloat.

        Returns:
            Formatted conversation string, or "(No previous conversation)"
        """
        if not self._conversation_history:
            return "(No previous conversation)"

        lines = []
        recent = self._conversation_history[-20:]
        for msg in recent:
            role = "User" if msg['role'] == 'user' else self._name
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    # =========================================================================
    # MESSAGE EXECUTION
    # =========================================================================

    def execute(self, user_message: str, extra_context: Optional[dict] = None):
        """
        Execute this noodling's assembly with a user message.

        Runs on a worker thread via _AssemblyWorker so the UI stays responsive.
        Emits executionStarted immediately, then responseReady/performanceReady
        when the assembly completes, or errorOccurred on failure.

        Args:
            user_message: The user's message text
            extra_context: Additional context (e.g. channel messages from
                          other noodlings, Brenda direction)
        """
        if not self._assembly or not self._executor:
            self.errorOccurred.emit("Assembly not loaded.")
            return

        # Pause gate: don't execute while paused
        if self._paused:
            return

        # Prevent overlapping executions
        if self._worker and self._worker.isRunning():
            return

        self._last_user_message = user_message
        self._sentiment_applied_early = False

        # Stop any in-progress performance playback
        if self._performance_player and self._performance_player.is_playing:
            self._performance_player.stop()

        # Detect streaming delivery mode from the Response facet
        self._streaming_mode = None
        self._streaming_first_token = True
        if self._assembly and hasattr(self._assembly, 'facets'):
            for facet in self._assembly.facets:
                if facet.id == 'response' and hasattr(facet, 'delivery'):
                    if facet.delivery in ('stream_animated', 'stream_raw'):
                        self._streaming_mode = facet.delivery

        self.executionStarted.emit()

        # Build execution context
        context = {
            'conversation_history': self._format_history(),
        }
        if extra_context:
            context.update(extra_context)

        # Execute assembly on worker thread
        # Reset per-turn trace collection
        self._current_turn_traces = []

        self._worker = _AssemblyWorker(
            self._executor, self._assembly, user_message, context
        )
        self._worker.resultReady.connect(self._on_assembly_result)
        self._worker.errorOccurred.connect(self._on_assembly_error)
        self._worker.facetCompleted.connect(self._on_facet_completed)
        self._worker.facetTraceReady.connect(self._on_facet_trace)

        # Wire streaming if delivery mode is streaming
        if self._streaming_mode:
            self._worker.streamTokenReady.connect(self._on_stream_token)

        self._worker.start()

    # =========================================================================
    # ASSEMBLY RESULT HANDLING
    # =========================================================================

    def _on_assembly_result(self, result):
        """
        Handle completed assembly execution.

        Detects performance scripts from the Performance facet and routes
        to the PerformancePlayer for typed text delivery. Falls back to
        immediate text emission for plain responses.

        Args:
            result: ExecutionResult from FacetExecutor
        """
        # Extract raw response text for conversation history
        raw_response = result.facet_outputs.get('response', {}).get('out', '')
        outgoing_output = result.response

        # Check if outgoing output is a performance script
        performance_script = None
        if outgoing_output and outgoing_output != '[No output]':
            try:
                parsed = json.loads(outgoing_output)
                if isinstance(parsed, dict) and parsed.get('type') == 'performance_script':
                    performance_script = parsed
                    if not raw_response:
                        raw_response = parsed.get('text', '')
            except (json.JSONDecodeError, ValueError):
                pass

        # Store the raw response text
        self._last_response = raw_response or outgoing_output or ''

        # Emit response signal
        if self._streaming_mode and not self._streaming_first_token:
            # Streaming was active -- tokens already delivered via _on_stream_token.
            # Just finish the streaming playback.
            if self._streaming_mode == 'stream_animated' and self._performance_player:
                self._performance_player.finish_streaming()
            elif self._streaming_mode == 'stream_raw':
                self.executionFinished.emit()
        elif performance_script:
            # Name prefix FIRST (begin_noodling_text inserts "Name: "),
            # then start character reveal. Previous order caused first
            # char to appear before the name prefix.
            self.performanceReady.emit(performance_script)
            self._play_performance(performance_script)
        elif outgoing_output and outgoing_output != '[No output]':
            self.responseReady.emit(outgoing_output)
            raw_response = raw_response or outgoing_output
        else:
            self.errorOccurred.emit("No response generated.")

        # Drive expressions from sentiment facet output (skip if mood-first
        # already applied it via _on_facet_completed)
        if not self._sentiment_applied_early:
            sentiment_raw = result.facet_outputs.get('sentiment', {})
            sentiment_output = sentiment_raw.get('out')
            if sentiment_output:
                self._apply_affect(sentiment_output)

        # Update conversation history with actual text (not performance JSON)
        self._conversation_history.append({
            'role': 'user', 'content': self._last_user_message
        })
        self._conversation_history.append({
            'role': 'assistant', 'content': raw_response or outgoing_output
        })

        # Publish to channel bus if configured
        if self._channel_bus and self._last_response:
            import time
            from ..channels import ChannelMessage
            self._channel_bus.publish(
                '#room.chat',
                ChannelMessage(
                    channel='#room.chat',
                    from_noodling=self._noodling_id,
                    timestamp=time.time(),
                    payload={'text': self._last_response}
                )
            )

        # Signal execution finished (unless performance player is running)
        if not performance_script:
            self.executionFinished.emit()

        # Emit collected traces for this turn
        traces = getattr(self, '_current_turn_traces', [])
        if traces:
            self.turnTraceReady.emit(self._noodling_id, list(traces))
            self._current_turn_traces = []

        self._worker = None

        logger.info(f"[{self._noodling_id}] Assembly done: "
                    f"{result.total_time:.2f}s, {result.total_tokens} tokens")

    def _on_assembly_error(self, error_msg: str):
        """
        Handle assembly execution error.

        Args:
            error_msg: Error description
        """
        self.errorOccurred.emit(error_msg)
        self.executionFinished.emit()
        self._worker = None
        logger.error(f"[{self._noodling_id}] Assembly error: {error_msg}")

    # =========================================================================
    # PER-FACET COMPLETION (MOOD-FIRST EXPRESSION)
    # =========================================================================

    def _on_facet_completed(self, facet_id: str, outputs: dict):
        """
        Handle individual facet completion for mood-first expression updates.

        Sentiment completes in ~1s (SMALL model), while Response takes ~5-15s
        (LARGE model). This lets us apply the expression change immediately.

        Args:
            facet_id: The ID of the completed facet
            outputs: The facet's output dict
        """
        if facet_id == 'sentiment':
            sentiment_output = outputs.get('out')
            if sentiment_output:
                self._apply_affect(sentiment_output)
                self._sentiment_applied_early = True
                logger.info(f"[{self._noodling_id}] Mood-first: "
                            "expression applied before response text")

        # Forward to live viz
        self.facetCompleted.emit(facet_id, outputs)

    def _on_facet_trace(self, facet_id: str, trace: dict):
        """Collect per-facet trace data for the current turn.

        Args:
            facet_id: The ID of the completed facet
            trace: Dict with system_prompt, formatted_prompt, output, etc.
        """
        if not hasattr(self, '_current_turn_traces'):
            self._current_turn_traces = []
        self._current_turn_traces.append(trace)

    # =========================================================================
    # STREAMING DELIVERY
    # =========================================================================

    def _on_stream_token(self, facet_id: str, text: str):
        """
        Handle streaming text token from the LLM.

        For stream_animated: emit performanceReady on first token (name prefix
        first), then feed tokens to PerformancePlayer for typed reveal.
        For stream_raw: emit chars directly via characterRevealed.

        Args:
            facet_id: The facet that produced this token
            text: The text chunk from the LLM
        """
        if not text:
            return

        if self._streaming_first_token:
            self._streaming_first_token = False
            # Emit performanceReady on first token so the name prefix is inserted
            stub_script = {
                'type': 'performance_script',
                'text': '',
                'characters': [],
                'speaking_intensity': 0.7
            }
            self.performanceReady.emit(stub_script)

            if self._streaming_mode == 'stream_animated':
                # Initialize the performance player for streaming
                from .performance_player import PerformancePlayer
                if self._performance_player is None:
                    self._performance_player = PerformancePlayer()
                    self._performance_player.characterRevealed.connect(
                        self.characterRevealed
                    )
                    self._performance_player.speakingStateChanged.connect(
                        self.speakingStateChanged
                    )
                    self._performance_player.finished.connect(
                        self._on_performance_finished
                    )
                self._performance_player.start_streaming()

        if self._streaming_mode == 'stream_animated':
            if self._performance_player:
                self._performance_player.append_text(text)
        elif self._streaming_mode == 'stream_raw':
            # Emit characters directly, no typing animation
            for char in text:
                self.characterRevealed.emit(char)

    # =========================================================================
    # AFFECT PIPELINE
    # =========================================================================

    def _apply_affect(self, affect_text: str):
        """
        Parse sentiment JSON and emit VRM blend shapes.

        Runs the full affect pipeline:
        sentiment JSON -> Affect -> FACSMapper -> VRM blendshapes

        Args:
            affect_text: JSON string from the sentiment facet
        """
        logger.info(f"[{self._noodling_id}] Raw sentiment: {affect_text!r}")

        try:
            affect_data = json.loads(affect_text)
            valence = float(affect_data.get('valence', 0.5))
            arousal = float(affect_data.get('arousal', 0.5))
            dominance = float(affect_data.get('dominance', 0.5))
            logger.info(f"[{self._noodling_id}] Affect: "
                        f"valence={valence:.2f} arousal={arousal:.2f} "
                        f"dominance={dominance:.2f}")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"[{self._noodling_id}] Affect JSON parse failed: {e}")
            return

        # Store raw PAD values for ensemble cross-pollination
        self._last_pad_values = {
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
        }

        from noodlestudio.runtime.facs_mapper import FACSMapper, Affect

        mapper = FACSMapper()
        affect_state = Affect(
            valence=valence,  # Mood Reader outputs -1..1 natively
            arousal=arousal,
            dominance=dominance,
            sorrow=max(0.0, -valence * 0.5),   # Derived from negative valence
            boredom=max(0.0, (1.0 - arousal) * 0.3)
        )

        vrm_shapes = mapper.map_affect_to_vrm(affect_state)
        logger.info(f"[{self._noodling_id}] VRM blend shapes ({len(vrm_shapes)})")

        if vrm_shapes:
            self.affectReady.emit(vrm_shapes)

    # =========================================================================
    # PERFORMANCE PLAYBACK (TYPED TEXT DELIVERY)
    # =========================================================================

    def _play_performance(self, script: dict):
        """
        Play a performance script via PerformancePlayer.

        Creates the player on first use and wires its signals to
        this performer's proxy signals.

        Args:
            script: Performance script dict from the Performance facet
        """
        from .performance_player import PerformancePlayer

        # Create player on first use
        if self._performance_player is None:
            self._performance_player = PerformancePlayer()
            self._performance_player.characterRevealed.connect(
                self.characterRevealed
            )
            self._performance_player.speakingStateChanged.connect(
                self.speakingStateChanged
            )
            self._performance_player.finished.connect(
                self._on_performance_finished
            )

        self._performance_player.play(script)

    def _on_performance_finished(self):
        """Handle completed performance playback."""
        self.performanceFinished.emit()
        self.executionFinished.emit()

    def pause_animation(self):
        """Freeze typing animation at current position."""
        if self._performance_player:
            self._performance_player.pause()

    def resume_animation(self):
        """Continue typing animation from current position."""
        if self._performance_player:
            self._performance_player.resume()

    @property
    def speaking_intensity(self) -> float:
        """Current speaking animation intensity."""
        if self._performance_player:
            return self._performance_player.speaking_intensity
        return 0.7

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def stop(self):
        """Stop all execution and clean up."""
        # Stop performance player
        if self._performance_player:
            self._performance_player.stop()
            self._performance_player = None

        # Stop any running worker
        if self._worker and self._worker.isRunning():
            self._worker.wait(2000)
            self._worker = None

        # Close LLM client session
        if self._llm_client:
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
        self._last_pad_values = None

        logger.info(f"[{self._noodling_id}] Performer stopped")


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
