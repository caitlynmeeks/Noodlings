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
#   Facet Executor - Parallel execution engine with synchronization
#
#   Executes facet assemblies with: - Topological ordering (d...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.facet_executor
# PURPOSE:  facet executor facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ExecutionResult, FacetExecutor
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import time
import uuid
from typing import Callable, Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

from .facet_system import Facet, FacetAssembly, FacetConnection
from ..runtime.channels import ChannelMessage, ChannelBus
from .charm_network_facet import CharmNetworkFacet, CharmNetworkOutput
from ..runtime.charm_network_ema import CharmNetworkEMA
from .neural_canvas_facet import NeuralCanvasFacet
from .scripted_facet import ScriptedFacet, ScriptContext
from .transformer_facet import TransformerFacet, TransformerOutput
from .subconscious_facet import SubconsciousFacet
from .insight_emergence_facet import InsightEmergenceFacet
from .context_intelligence_facet import ContextIntelligenceFacet
from .speech_gate_facet import SpeechGateFacet
from .flow_control_facets import (
    TickerGateFacet, ConditionalBranchFacet,
    RateLimiterFacet, CacheFacet, AccumulatorFacet
)
from .execution_event_bus import get_event_bus, EventChannel, EventPriority
from .audio_stream_facet import AudioStreamFacet
from .vision_facet import VisionFacet
from .image_gen_facet import ImageGenFacet
from .mcp_facet import MCPFacet
from .utility_facets import UTILITY_FACET_TYPES, create_utility_facet
from .affect_track import AffectTrackFacet
from .physics_affect_bridge import PhysicsAffectFacet, get_physics_affect_bridge
from .gaussian_training_facet import GaussianTrainingFacet, TrainingConfig
from .skeleton_binding_facet import SkeletonBindingFacet, SkeletonBindingConfig
from .auto_rigger_facet import AutoRiggerFacet, AutoRiggerConfig

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from facet assembly execution."""

    # Final output
    response: str

    # Execution metadata
    total_time: float
    total_tokens: int
    facets_executed: int
    facets_skipped: int

    # Per-facet results
    facet_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    facet_times: Dict[str, float] = field(default_factory=dict)
    facet_tokens: Dict[str, int] = field(default_factory=dict)

    # Flow control metadata
    ticker_gates_fired: List[str] = field(default_factory=list)
    branches_taken: Dict[str, str] = field(default_factory=dict)  # facet_id -> "true"/"false"
    cache_hits: List[str] = field(default_factory=list)

    # Execution tracking (for race condition prevention)
    execution_id: str = ""


class FacetExecutor:
    """
    Executes facet assemblies with parallel processing and synchronization.

    Execution model:
    1. Build dependency graph from connections
    2. Find facets with all inputs satisfied (ready set)
    3. Execute ready facets in parallel (async)
    4. Mark completed, update available outputs
    5. Repeat until OUTGOING reached

    Handles:
    - CharmNetworkFacet (neural computation)
    - ScriptedFacet (JavaScript logic)
    - Flow control facets (gates, branches, etc.)
    - LLM facets (via provided LLM client)
    """

    def __init__(self, llm_client=None, event_callback=None, use_event_bus=True,
                 concurrency_mode='hybrid', channel_bus=None):
        """
        Initialize executor.

        Args:
            llm_client: LLM client for calling language models (optional)
            event_callback: Async callback for execution events (optional, deprecated)
                          Use event bus instead for better decoupling
            use_event_bus: Use global event bus for event distribution (recommended)
            concurrency_mode: Concurrency strategy - 'serial', 'hybrid', or 'full'
                            - serial: Lock entire execution (debug mode)
                            - hybrid: Singleton stateful facets, isolated stateless (production)
                            - full: No locks, fresh instances (NOT RECOMMENDED)
            channel_bus: ChannelBus for inter-noodling communication (optional)
        """
        self.llm_client = llm_client
        self.event_callback = event_callback  # Legacy support
        self.use_event_bus = use_event_bus
        self.concurrency_mode = concurrency_mode
        self.channel_bus = channel_bus
        self.current_cycle = 0

        # Debug log callback (for routing context.log() to DEBUG console)
        self.debug_log_callback = None

        # HYBRID STRATEGY: Separate singleton and stateless facets
        # Singleton facets - shared across executions (protected by internal locks)
        self.singleton_facets: Dict[str, Any] = {}

        # Legacy cached instances (will be phased out)
        self.facet_instances: Dict[str, Any] = {}

        # Per-execution results storage (for race condition prevention)
        self.execution_results: Dict[str, Dict[str, Any]] = {}

        # Execution history
        self.execution_history: List[ExecutionResult] = []

        # Serial mode lock (debug only)
        if concurrency_mode == 'serial':
            self.execution_lock = asyncio.Semaphore(1)
            logger.info("[FacetExecutor] Serial mode enabled - one execution at a time")
        else:
            self.execution_lock = None

        # Get event bus reference
        if use_event_bus:
            self.event_bus = get_event_bus()
        else:
            self.event_bus = None

    async def _emit_event(self, event: Dict[str, Any]):
        """
        Emit execution event via event bus or legacy callback.

        Args:
            event: Event dict with type, subtype, and metadata
        """
        # Emit to event bus (preferred)
        if self.event_bus:
            await self.event_bus.emit(
                event_type=event.get('type', 'unknown'),
                event_subtype=event.get('subtype', 'unknown'),
                channel=EventChannel.EXECUTION,
                priority=EventPriority.NORMAL,
                source_id=event.get('facet_id'),
                source_name=event.get('facet_name'),
                cycle=event.get('cycle'),
                data=event
            )

        # Legacy callback support
        if self.event_callback:
            try:
                await self.event_callback(event)
            except Exception as e:
                logger.error(f"Legacy event callback failed: {e}")

    # =========================================================================
    # Channel Integration
    # =========================================================================

    def _resolve_channel_inputs(
        self,
        facet: Facet,
        inputs: Dict[str, Any],
        assembly: FacetAssembly,
        noodling_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Resolve channel inputs for a facet.

        If the assembly subscribes to channels, check if any input pads
        reference channel data (e.g., 'channel:#directors.cues').

        Args:
            facet: The facet being executed
            inputs: Current input dictionary
            assembly: The assembly being executed
            noodling_id: ID of the noodling running this assembly

        Returns:
            Updated inputs with channel data resolved
        """
        if not self.channel_bus:
            return inputs

        # Get subscribed channels from assembly
        subscribe_channels = assembly.get_subscribe_channels()
        if not subscribe_channels:
            return inputs

        # Check each input pad for channel references
        for pad in facet.input_pads:
            pad_name = pad.name

            # Check if pad name references a channel
            if pad_name.startswith('channel:'):
                channel = pad_name[8:]  # Strip 'channel:' prefix
                if channel in subscribe_channels:
                    latest = self.channel_bus.get_latest(channel)
                    if latest:
                        inputs[pad_name] = latest.payload
                        logger.debug(f"[Channel] Resolved {pad_name} from channel {channel}")
                else:
                    logger.warning(
                        f"[Channel] Facet {facet.name} wants {channel} but assembly "
                        f"doesn't subscribe to it"
                    )

        return inputs

    def _publish_channel_outputs(
        self,
        facet: Facet,
        outputs: Dict[str, Any],
        assembly: FacetAssembly,
        noodling_id: str = "unknown"
    ) -> None:
        """
        Publish outputs to channels if they have channel: prefix.

        Args:
            facet: The facet that produced outputs
            outputs: The facet's output dictionary
            assembly: The assembly being executed
            noodling_id: ID of the noodling running this assembly
        """
        if not self.channel_bus or not outputs:
            return

        # Get publish channels from assembly
        publish_channels = assembly.get_publish_channels()

        for output_name, output_value in outputs.items():
            if output_name.startswith('channel:'):
                channel = output_name[8:]  # Strip 'channel:' prefix

                # Check if assembly is allowed to publish to this channel
                if channel in publish_channels:
                    # Ensure payload is a dict
                    if isinstance(output_value, dict):
                        payload = output_value
                    else:
                        payload = {'value': output_value}

                    self.channel_bus.publish_simple(
                        channel=channel,
                        payload=payload,
                        from_noodling=noodling_id
                    )
                    logger.info(f"[Channel] Published to {channel} from {facet.name}")
                else:
                    logger.warning(
                        f"[Channel] Facet {facet.name} tried to publish to {channel} "
                        f"but assembly doesn't have publish permission"
                    )

    def set_debug_log_callback(self, callback):
        """
        Set callback for debug logs from ScriptedFacet context.log() calls.

        Args:
            callback: Function(facet_name: str, message: str) -> None
        """
        self.debug_log_callback = callback

    def _emit_debug_log(self, facet_name: str, message: str):
        """
        Emit a debug log message to the DEBUG console.

        Args:
            facet_name: Name of facet that logged the message
            message: The log message from context.log()
        """
        if self.debug_log_callback:
            try:
                self.debug_log_callback(facet_name, message)
            except Exception as e:
                logger.error(f"Debug log callback failed: {e}")

    def _build_dependency_graph(
        self,
        assembly: FacetAssembly
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Build dependency graph from assembly connections.

        Returns:
            (dependencies, dependents) where:
            - dependencies[facet_id] = set of facets that must complete before this one
            - dependents[facet_id] = set of facets that depend on this one
        """
        dependencies = defaultdict(set)
        dependents = defaultdict(set)

        # Initialize all facets
        for facet in assembly.facets:
            dependencies[facet.id] = set()
            dependents[facet.id] = set()

        # Build graph from connections
        for conn in assembly.connections:
            # conn.to_facet depends on conn.from_facet
            dependencies[conn.to_facet].add(conn.from_facet)
            dependents[conn.from_facet].add(conn.to_facet)

        return dict(dependencies), dict(dependents)

    def _get_facet_instance(self, facet: Facet, context: Dict[str, Any]) -> Any:
        """
        Get or create facet instance using HYBRID STRATEGY.

        STATEFUL facets (singleton): CharmNetwork, RateLimiter, Cache, Accumulator, SpeechGate
        - Shared across all executions (temporal memory)
        - Protected by internal locks

        STATELESS facets (isolated): ContextIntelligence, Subconscious, InsightEmergence
        - Fresh instance per execution (no contamination)
        - No persistent state

        Args:
            facet: Facet to instantiate
            context: Execution context (contains execution_id for tracking)

        Returns:
            Facet instance (singleton or fresh)
        """
        # STATEFUL FACETS - Use singleton (shared across executions)
        if facet.facet_type == "CharmNetworkFacet":
            # Singleton: Temporal hidden states (h_fast, h_medium, h_slow)
            if 'charm_network' not in self.singleton_facets:
                checkpoint_path = facet.model
                self.singleton_facets['charm_network'] = CharmNetworkFacet(checkpoint_path)
                logger.info("[FacetExecutor] Created singleton CharmNetworkFacet")
            return self.singleton_facets['charm_network']

        elif facet.facet_type == "CharmNetworkEMA":
            # Singleton: Multi-timescale EMA affect (stateful across turns)
            key = f"charm_ema_{facet.id}"
            if key not in self.singleton_facets:
                # Parse baseline from facet prompt (YAML: "valence:0.7,arousal:0.5,dominance:0.4")
                baseline = {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}
                if facet.prompt:
                    for part in facet.prompt.split(','):
                        if ':' in part:
                            k, v = part.strip().split(':', 1)
                            try:
                                baseline[k.strip()] = float(v.strip())
                            except ValueError:
                                pass
                self.singleton_facets[key] = CharmNetworkEMA(baseline)
                logger.info(f"[FacetExecutor] Created singleton CharmNetworkEMA: {facet.id} (baseline={baseline})")
            return self.singleton_facets[key]

        elif facet.facet_type == "NeuralCanvasFacet":
            # Singleton: Loads .nncanvas file and executes NeuralGraph
            key = f"neural_canvas_{facet.id}"
            if key not in self.singleton_facets:
                # Get project root from context
                project_root = context.get('project_root', None)
                self.singleton_facets[key] = NeuralCanvasFacet(
                    facet_id=facet.id,
                    name=facet.name,
                    nncanvas_path=facet.nncanvas_path or '',
                    project_root=project_root
                )
                logger.info(f"[FacetExecutor] Created singleton NeuralCanvasFacet: {facet.name} (path={facet.nncanvas_path})")
            return self.singleton_facets[key]

        elif facet.facet_type == "TransformerFacet":
            # Singleton: Transformer is stateless but model loading is expensive
            key = f"transformer_{facet.id}"
            if key not in self.singleton_facets:
                # Parse config from prompt (format: embed_dim=64,num_heads=4,...)
                config = {}
                if facet.prompt:
                    for part in facet.prompt.split(','):
                        if '=' in part:
                            k, v = part.strip().split('=')
                            try:
                                config[k.strip()] = int(v.strip())
                            except ValueError:
                                try:
                                    config[k.strip()] = float(v.strip())
                                except ValueError:
                                    config[k.strip()] = v.strip()
                self.singleton_facets[key] = TransformerFacet(
                    embed_dim=config.get('embed_dim', 64),
                    num_heads=config.get('num_heads', 4),
                    num_layers=config.get('num_layers', 2),
                    ff_dim=config.get('ff_dim', 256),
                    checkpoint_path=facet.model if facet.model != "SMALL" else None
                )
                logger.info(f"[FacetExecutor] Created TransformerFacet: {facet.id}")
            return self.singleton_facets[key]

        elif facet.facet_type == "RateLimiterFacet":
            # Singleton: Tracks last execution timestamp
            if facet.id not in self.singleton_facets:
                interval = float(facet.temperature)
                self.singleton_facets[facet.id] = RateLimiterFacet(facet.id, min_interval=interval)
            return self.singleton_facets[facet.id]

        elif facet.facet_type == "CacheFacet":
            # Singleton: Stores cached values with TTL
            if facet.id not in self.singleton_facets:
                ttl = int(facet.max_tokens)
                self.singleton_facets[facet.id] = CacheFacet(facet.id, ttl=ttl)
            return self.singleton_facets[facet.id]

        elif facet.facet_type == "AccumulatorFacet":
            # Singleton: Maintains rolling window
            if facet.id not in self.singleton_facets:
                window_size = int(facet.max_tokens)
                self.singleton_facets[facet.id] = AccumulatorFacet(facet.id, window_size=window_size)
            return self.singleton_facets[facet.id]

        elif facet.facet_type == "SpeechGateFacet":
            # Singleton: Tracks speech cooldown state
            if facet.id not in self.singleton_facets:
                min_interval = float(facet.temperature)
                self.singleton_facets[facet.id] = SpeechGateFacet(min_interval=min_interval)
            return self.singleton_facets[facet.id]

        elif facet.facet_type == "TickerGateFacet":
            # Singleton: Tracks tick count
            if facet.id not in self.singleton_facets:
                interval = int(facet.max_tokens)
                self.singleton_facets[facet.id] = TickerGateFacet(facet.id, interval=interval)
            return self.singleton_facets[facet.id]

        elif facet.facet_type == "ScriptedFacet":
            # Singleton: Script compilation caching
            if facet.id not in self.singleton_facets:
                self.singleton_facets[facet.id] = ScriptedFacet(facet.id, facet.prompt, script_language="javascript")
            return self.singleton_facets[facet.id]

        elif facet.facet_type == "ConditionalBranchFacet":
            # Singleton: Stateless but cache for performance
            if facet.id not in self.singleton_facets:
                condition = facet.prompt
                variables = [p.name for p in facet.input_pads if p.name != 'in']
                self.singleton_facets[facet.id] = ConditionalBranchFacet(facet.id, condition, variables)
            return self.singleton_facets[facet.id]

        elif facet.facet_type == "AudioStreamFacet":
            # MULTIMODAL: Singleton with parallel processing loop
            # Runs independently, syncs at cycle boundaries
            if facet.id not in self.singleton_facets:
                # Parse config from facet properties
                sample_rate = int(facet.max_tokens) if facet.max_tokens > 8000 else 16000
                chunk_ms = int(facet.temperature * 1000) if facet.temperature < 1 else 250

                # Create facet with real clients
                from .audio_stream_facet import create_audio_facet_with_clients
                audio_facet = create_audio_facet_with_clients(
                    facet_id=facet.id,
                    process_interval_ms=chunk_ms,
                    sample_rate=sample_rate,
                    transcription_model=facet.model or "AUDIO_IN",
                    tts_model="AUDIO_OUT"
                )
                self.singleton_facets[facet.id] = audio_facet
                logger.info(f"[FacetExecutor] Created singleton AudioStreamFacet (id={facet.id[:8]})")

                # Connect to audio API for scripting access
                from ..scripting.audio_api import get_audio_api
                audio_api = get_audio_api()
                audio_api.set_audio_facet(audio_facet)

                # Start the parallel processing loop
                asyncio.create_task(audio_facet.start())
                logger.info(f"[FacetExecutor] Started AudioStreamFacet processing loop")

            return self.singleton_facets[facet.id]

        elif facet.facet_type == "VisionFacet":
            # MULTIMODAL: Singleton for image understanding
            if facet.id not in self.singleton_facets:
                from .vision_facet import create_vision_facet_with_client
                vision_facet = create_vision_facet_with_client(
                    facet_id=facet.id,
                    model_label=facet.model or "VISION"
                )
                self.singleton_facets[facet.id] = vision_facet
                logger.info(f"[FacetExecutor] Created singleton VisionFacet (id={facet.id[:8]})")

                # Connect to vision API for scripting access
                from ..scripting.vision_api import get_vision_api
                vision_api = get_vision_api()
                vision_api.set_vision_facet(vision_facet)

                # Start processing loop
                asyncio.create_task(vision_facet.start())

            return self.singleton_facets[facet.id]

        elif facet.facet_type == "ImageGenFacet":
            # MULTIMODAL: Singleton for image generation
            if facet.id not in self.singleton_facets:
                from .image_gen_facet import create_image_gen_facet_with_client
                gen_facet = create_image_gen_facet_with_client(
                    facet_id=facet.id,
                    model_label=facet.model or "IMAGE_GEN"
                )
                self.singleton_facets[facet.id] = gen_facet
                logger.info(f"[FacetExecutor] Created singleton ImageGenFacet (id={facet.id[:8]})")

                # Connect to vision API for scripting access
                from ..scripting.vision_api import get_vision_api
                vision_api = get_vision_api()
                vision_api.set_image_gen_facet(gen_facet)

                # Connect to GenerationsManager for asset storage
                from .generations_manager import get_generations_manager
                gen_manager = get_generations_manager()

                # Subscribe to image_generated events for auto-storage
                async def on_image_generated(event_data):
                    """Store generated images in Generations folder."""
                    try:
                        # Get the image from facet cache
                        last_image = gen_facet.get_last_image()
                        if last_image and last_image.image_data:
                            gen_manager.store_generation(
                                image_data=last_image.image_data,
                                metadata={
                                    'source': 'facet',
                                    'prompt': event_data.get('prompt', ''),
                                    'style': event_data.get('style', ''),
                                    'width': last_image.width,
                                    'height': last_image.height
                                }
                            )
                            logger.info("[FacetExecutor] Stored generated image via event")
                    except Exception as e:
                        logger.error(f"[FacetExecutor] Image storage error: {e}")

                gen_facet.on('image_generated', on_image_generated)

                # Start processing loop
                asyncio.create_task(gen_facet.start())

            return self.singleton_facets[facet.id]

        elif facet.facet_type == "AffectTrackFacet":
            # AFFECT TRACK: Singleton - maintains playback state
            if facet.id not in self.singleton_facets:
                # Parse config from facet properties
                config = {}
                if facet.prompt:
                    for part in facet.prompt.split(','):
                        if ':' in part:
                            k, v = part.strip().split(':', 1)
                            config[k.strip()] = v.strip()

                # Check for track path in model field
                if facet.model and facet.model not in ('SMALL', 'MEDIUM', 'LARGE'):
                    config['track'] = facet.model

                self.singleton_facets[facet.id] = AffectTrackFacet(config)

                # Wire up CharmNetwork reference for momentum handoff
                if 'charm_network' in self.singleton_facets:
                    self.singleton_facets[facet.id].set_charm_network(
                        self.singleton_facets['charm_network']
                    )

                logger.info(f"[FacetExecutor] Created singleton AffectTrackFacet (id={facet.id[:8]}, track={config.get('track')})")

            return self.singleton_facets[facet.id]

        elif facet.facet_type == "PhysicsAffectFacet":
            # PHYSICS → AFFECT: Singleton - bridges collision system to CharmNetwork
            if facet.id not in self.singleton_facets:
                # Parse config from facet properties
                config = {
                    'entity_id': context.get('agent_id', ''),
                }
                if facet.prompt:
                    for part in facet.prompt.split(','):
                        if ':' in part:
                            k, v = part.strip().split(':', 1)
                            config[k.strip()] = v.strip()

                physics_facet = PhysicsAffectFacet(config)

                # Get or create the global bridge
                bridge = get_physics_affect_bridge()
                if bridge:
                    physics_facet.set_bridge(bridge)

                # Wire to CharmNetworkFacet if available
                if 'charm_network' in self.singleton_facets:
                    physics_facet.set_charm_facet(self.singleton_facets['charm_network'])
                    logger.info(f"[FacetExecutor] Wired PhysicsAffectFacet to CharmNetwork")

                self.singleton_facets[facet.id] = physics_facet
                logger.info(f"[FacetExecutor] Created singleton PhysicsAffectFacet (entity={config.get('entity_id')})")

            return self.singleton_facets[facet.id]

        elif facet.facet_type == "GaussianTrainingFacet":
            # GAUSSIAN TRAINING: Singleton - long-running training job
            if facet.id not in self.singleton_facets:
                # Parse config from facet properties
                config = {}
                if facet.prompt:
                    for part in facet.prompt.split(','):
                        if ':' in part:
                            k, v = part.strip().split(':', 1)
                            config[k.strip()] = v.strip()

                training_config = TrainingConfig.from_dict(config)
                self.singleton_facets[facet.id] = GaussianTrainingFacet(training_config)
                logger.info(f"[FacetExecutor] Created singleton GaussianTrainingFacet (id={facet.id[:8]})")

            return self.singleton_facets[facet.id]

        elif facet.facet_type == "SkeletonBindingFacet":
            # SKELETON BINDING: Singleton - binds Gaussians to VRM skeleton
            if facet.id not in self.singleton_facets:
                config = {}
                if facet.prompt:
                    for part in facet.prompt.split(','):
                        if ':' in part:
                            k, v = part.strip().split(':', 1)
                            config[k.strip()] = v.strip()

                binding_config = SkeletonBindingConfig.from_dict(config)
                self.singleton_facets[facet.id] = SkeletonBindingFacet(binding_config)
                logger.info(f"[FacetExecutor] Created singleton SkeletonBindingFacet (id={facet.id[:8]})")

            return self.singleton_facets[facet.id]

        elif facet.facet_type == "AutoRiggerFacet":
            # AUTO-RIGGER: Singleton - Mixamo-style automatic rigging
            if facet.id not in self.singleton_facets:
                config = {}
                if facet.prompt:
                    for part in facet.prompt.split(','):
                        if ':' in part:
                            k, v = part.strip().split(':', 1)
                            config[k.strip()] = v.strip()

                rigger_config = AutoRiggerConfig.from_dict(config)
                self.singleton_facets[facet.id] = AutoRiggerFacet(rigger_config)
                logger.info(f"[FacetExecutor] Created singleton AutoRiggerFacet (id={facet.id[:8]})")

            return self.singleton_facets[facet.id]

        elif facet.facet_type == "MCPFacet":
            # MCP: Singleton - connections are expensive to set up
            if facet.id not in self.singleton_facets:
                # Parse config from facet properties
                # Expected: model="server_name/tool_name" or use facet.prompt for config
                config = {}

                # Parse model field as "server/tool" format
                if facet.model and '/' in facet.model:
                    parts = facet.model.split('/', 1)
                    config['server'] = parts[0]
                    config['tool'] = parts[1]

                # Alternatively, parse from prompt as JSON/YAML-like config
                if facet.prompt:
                    for part in facet.prompt.split(','):
                        if ':' in part:
                            k, v = part.strip().split(':', 1)
                            config[k.strip()] = v.strip()

                self.singleton_facets[facet.id] = MCPFacet(facet.id, config)
                logger.info(f"[FacetExecutor] Created singleton MCPFacet (id={facet.id[:8]}, server={config.get('server')}, tool={config.get('tool')})")

            return self.singleton_facets[facet.id]

        # UTILITY FACETS - Check if it's a utility type
        elif facet.facet_type in UTILITY_FACET_TYPES:
            # Stateful facets (Counter) should be singletons
            if facet.facet_type in ('CounterFacet',):
                if facet.id not in self.singleton_facets:
                    config = {}
                    if facet.prompt:
                        for part in facet.prompt.split(','):
                            if ':' in part:
                                k, v = part.strip().split(':', 1)
                                config[k.strip()] = v.strip()
                    self.singleton_facets[facet.id] = create_utility_facet(
                        facet.facet_type, facet.id, config
                    )
                return self.singleton_facets[facet.id]
            else:
                # Stateless utility facets - create fresh
                config = {}
                if facet.prompt:
                    for part in facet.prompt.split(','):
                        if ':' in part:
                            k, v = part.strip().split(':', 1)
                            config[k.strip()] = v.strip()
                return create_utility_facet(facet.facet_type, facet.id, config)

        # STATELESS FACETS - Create fresh instance per execution
        elif facet.facet_type == "ContextIntelligenceFacet":
            # ISOLATED: Fresh instance prevents contamination between cycles
            agent_name = context.get('agent_name', 'unknown')
            logger.info(f"[FacetExecutor] Creating ISOLATED ContextIntelligence for execution {context.get('execution_id', '?')[:8]}")
            return ContextIntelligenceFacet(
                facet_config={
                    'model': facet.model,
                    'max_tokens': facet.max_tokens,
                    'temperature': facet.temperature
                },
                llm_client=self.llm_client,
                agent_name=agent_name  # Set from context immediately!
            )

        elif facet.facet_type == "SubconsciousFacet":
            # ISOLATED: Generates metaphors from inputs (no state)
            # But connects to ImageGenFacet for visual generation

            # Parse visual generation settings from facet prompt
            # Format: "generate_visual:true,style:artistic,probability:0.3"
            generate_visual = False
            visual_style = "artistic"
            visual_probability = 0.3

            if facet.prompt:
                for part in facet.prompt.split(','):
                    if ':' in part:
                        key, value = part.strip().split(':', 1)
                        if key == 'generate_visual':
                            generate_visual = value.lower() == 'true'
                        elif key == 'style':
                            visual_style = value
                        elif key == 'probability':
                            try:
                                visual_probability = float(value)
                            except ValueError:
                                pass

            instance = SubconsciousFacet(
                facet.id,
                generate_visual=generate_visual,
                visual_style=visual_style,
                visual_probability=visual_probability
            )

            # Connect to ImageGenFacet if available
            for key, singleton in self.singleton_facets.items():
                if isinstance(singleton, ImageGenFacet):
                    instance.set_image_gen_facet(singleton)
                    logger.info(f"[FacetExecutor] Connected SubconsciousFacet to ImageGenFacet")
                    break

            # Connect to GenerationsManager
            from .generations_manager import get_generations_manager
            instance.set_generations_manager(get_generations_manager())

            return instance

        elif facet.facet_type == "InsightEmergenceFacet":
            # ISOLATED: Surfaces insights from context (no state)
            return InsightEmergenceFacet(facet.id)

        elif facet.facet_type in ("SpecialNode", "INCOMING", "OUTGOING"):
            # INCOMING/OUTGOING - no instance needed
            return None

        else:
            # Default: LLM facet (will call LLM in execute)
            return None

    def _facet_type_to_phase(self, facet_type: str, facet_name: str) -> str:
        """Map facet type to cognitive phase for UI display."""
        if facet_name == "INCOMING":
            return "INCOMING"
        elif facet_name == "OUTGOING":
            return "OUTGOING"
        elif facet_type in ("CharmNetworkFacet", "CharmNetworkEMA"):
            return "NEURAL"
        elif facet_type == "TransformerFacet":
            return "NEURAL"  # Also neural computation
        elif facet_type == "ContextIntelligenceFacet":
            return "PRECOG"
        elif facet_type in ("ConvergenceFacet", "SpeechGateFacet"):
            return "POSTCOG"
        else:
            return "FACET"

    async def _execute_facet(
        self,
        facet: Facet,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        on_stream_token: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a single facet.

        Args:
            facet: Facet to execute
            inputs: Input values from connected pads
            context: Execution context (cycle, agent info, etc.)
            on_stream_token: Optional callback(facet_id, text) for streaming delivery

        Returns:
            Dict of output values
        """
        start_time = time.time()

        # Update agent reference for real-time NoodleStudio visualization
        agent_ref = context.get('_agent_ref')
        if agent_ref is not None:
            agent_ref.current_facet = facet.name
            agent_ref.current_phase = self._facet_type_to_phase(facet.facet_type, facet.name)

        # Emit facet_start event (include inputs for debugging/inspection)
        execution_id = context.get('execution_id', '')
        print(f"[FacetExecutor] 🚀 EMITTING facet_start for {facet.name} (id={facet.id}, exec={execution_id[:8]})")

        # Sanitize inputs for JSON serialization (truncate large strings, handle non-serializable types)
        sanitized_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, str):
                # Truncate long strings for event payload
                sanitized_inputs[key] = value[:500] + '...' if len(value) > 500 else value
            elif isinstance(value, (int, float, bool, type(None))):
                sanitized_inputs[key] = value
            elif isinstance(value, (list, tuple)):
                # Truncate long lists
                sanitized_inputs[key] = list(value)[:10] if len(value) > 10 else list(value)
            else:
                sanitized_inputs[key] = str(value)[:200]

        await self._emit_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'facet_id': facet.id,
            'facet_name': facet.name,
            'facet_type': facet.facet_type,
            'timestamp': start_time,
            'cycle': self.current_cycle,
            'execution_id': execution_id,
            'inputs': sanitized_inputs  # Include inputs for inspection
        })

        # Get or create instance (hybrid strategy: singleton or isolated)
        instance = self._get_facet_instance(facet, context)

        # Execute based on type
        if facet.facet_type in ("SpecialNode", "INCOMING", "OUTGOING"):
            # INCOMING/OUTGOING - just pass through
            outputs = {'out': inputs.get('in', inputs)}
            token_count = 0

        elif facet.facet_type == "CharmNetworkEMA":
            # EMA charm network: multi-timescale affect smoothing
            # Input: 3-D PAD JSON string from Mood Reader (via 'in' pad)
            import json as _json
            pad_input = inputs.get('in', inputs.get('affect_in', ''))

            # Parse PAD from JSON string or dict
            if isinstance(pad_input, str):
                try:
                    pad_input = _json.loads(pad_input)
                except (ValueError, TypeError):
                    pad_input = {}
            if not isinstance(pad_input, dict):
                pad_input = {}

            # Run EMA update
            blended = instance.update(pad_input)

            # Output as JSON string (for OUTGOING.affect consumption)
            outputs = {'out': _json.dumps(blended)}
            token_count = 0

        elif facet.facet_type == "CharmNetworkFacet":
            # Neural computation
            # CharmNetwork input mapping:
            # - If 'affect_in' pad receives TEXT, use it as perception_text
            # - If 'affect_input' pad receives ARRAY, use it as affect vector
            # This allows flexible wiring: text → affect_in OR affect vector → affect_input

            perception_text = inputs.get('affect_in', inputs.get('perception', ''))
            affect_vector = inputs.get('affect_input', None)

            # If perception is a list/array, swap (user wired it backwards)
            if isinstance(perception_text, (list, tuple)):
                affect_vector = perception_text
                perception_text = ''

            result = await instance.process(perception_text, affect_vector)

            # Map CharmNetworkOutput to pad outputs
            outputs = {
                'affect_valence': result.valence,
                'affect_arousal': result.arousal,
                'affect_dominance': result.dominance,
                'affect_sorrow': result.sorrow,
                'affect_boredom': result.boredom,
                'surprise': result.surprise,
                'phenomenal_state': result.phenomenal_state
            }
            token_count = 0

        elif facet.facet_type == "NeuralCanvasFacet":
            # NeuralCanvas: Execute visual neural network from .nncanvas file
            # Supports affect inputs (5D vector) or arbitrary inputs
            result = await instance.execute(inputs)

            if 'error' in result:
                logger.warning(f"[FacetExecutor] NeuralCanvasFacet error: {result['error']}")
                outputs = {'out': f"[NeuralCanvas error: {result['error']}]"}
            else:
                # Map outputs from graph
                outputs = result.copy()
                # Remove internal metadata from output
                outputs.pop('_node_outputs', None)
                outputs.pop('_execution_time_ms', None)

                # Log execution details
                logger.info(f"[FacetExecutor] NeuralCanvasFacet {facet.name} executed in {result.get('_execution_time_ms', 0):.1f}ms")

            token_count = 0  # Neural computation, not LLM

        elif facet.facet_type == "TransformerFacet":
            # Attention-based context processing
            # Input: 'text' for raw text, or 'tokens' for token IDs
            text_input = inputs.get('text', inputs.get('in', ''))
            if isinstance(text_input, str):
                result = await instance.process_text(text_input)
            else:
                # Assume token IDs
                result = await instance.process(
                    text_input if isinstance(text_input, list) else [0]
                )

            # Map TransformerOutput to pad outputs
            outputs = {
                'context_embedding': result.context_embedding,
                'attention_weights': result.attention_weights,
                'top_attended': result.top_attended_tokens,
                'classification': result.classification or {},
                'out': result.context_embedding  # Default output
            }
            token_count = 0  # Neural computation, not LLM

        elif facet.facet_type == "ScriptedFacet":
            # JavaScript execution
            script_context = ScriptContext(
                cycle=self.current_cycle,
                timestamp=time.time(),
                agent_id=context.get('agent_id', 'unknown'),
                agent_name=context.get('agent_name', 'unknown'),
                agent_species=context.get('agent_species', 'unknown')
            )
            outputs = instance.process(inputs, script_context)
            token_count = 0

            # Emit any debug logs from context.log() calls to DEBUG console
            if script_context._logs:
                for log_message in script_context._logs:
                    self._emit_debug_log(facet.name, log_message)

        elif facet.facet_type == "SubconsciousFacet":
            # Symbolic processing (uses LLM for metaphor generation)
            outputs = await instance.process(inputs, context, self.llm_client)
            token_count = 100  # Approximate token cost for symbolic generation

        elif facet.facet_type == "InsightEmergenceFacet":
            # Insight surfacing (needs latent memories from agent)
            latent_memories = context.get('_latent_memories', [])
            outputs = await instance.process(inputs, context, self.llm_client, latent_memories)
            token_count = 150  # Approximate token cost for translation

        elif facet.facet_type == "ContextIntelligenceFacet":
            # Context reasoning - uses LLM for intelligent parsing
            outputs = await instance.execute(inputs, context)
            # Higher token cost since this uses smarter model (qwen3-14b)
            token_count = 250

        elif facet.facet_type == "TickerGateFacet":
            outputs = instance.process(inputs, self.current_cycle)
            token_count = 0

        elif facet.facet_type == "ConditionalBranchFacet":
            outputs = instance.process(inputs)
            token_count = 0

        elif facet.facet_type == "RateLimiterFacet":
            outputs = instance.process(inputs)
            token_count = 0

        elif facet.facet_type == "CacheFacet":
            outputs = instance.process(inputs, self.current_cycle)
            token_count = 0

        elif facet.facet_type == "AccumulatorFacet":
            outputs = instance.process(inputs)
            token_count = 0

        elif facet.facet_type == "SpeechGateFacet":
            outputs = instance.process(inputs, context)
            token_count = 0

        elif facet.facet_type == "AudioStreamFacet":
            # MULTIMODAL: Sync with parallel audio processing
            # Get pending commands from audio API
            from ..scripting.audio_api import get_audio_api
            audio_api = get_audio_api()
            commands = audio_api.get_pending_commands()

            # Merge with cycle inputs (e.g., text to speak from other facets)
            cycle_data = {**inputs, **commands}

            # Add any speak request from facet inputs
            speak_text = inputs.get('speak', inputs.get('text_to_speak'))
            if speak_text:
                cycle_data['speak'] = speak_text

            # Sync with audio facet
            sync_result = await instance.sync(cycle_data)

            # Update audio API state
            audio_api.update_from_facet(sync_result)

            # Map outputs
            outputs = {
                'transcription': sync_result.get('transcription', ''),
                'is_speaking': sync_result.get('is_speaking', False),
                'is_listening': sync_result.get('is_listening', False),
                'state': sync_result.get('state', 'IDLE')
            }
            token_count = 0

        elif facet.facet_type == "VisionFacet":
            # MULTIMODAL: Sync with vision processing
            from ..scripting.vision_api import get_vision_api
            vision_api = get_vision_api()
            commands = vision_api.get_pending_commands()

            # Merge with cycle inputs
            cycle_data = {**inputs, **commands}

            # Check for image analysis request from facet inputs
            image_path = inputs.get('analyze', inputs.get('image_path'))
            if image_path:
                cycle_data['analyze_image'] = image_path

            # Sync with vision facet
            sync_result = await instance.sync(cycle_data)

            # Update vision API state
            vision_api.update_from_vision_facet(sync_result)

            # Map outputs
            outputs = {
                'description': sync_result.get('last_description', ''),
                'hot_count': sync_result.get('hot_count', 0),
                'warm_count': sync_result.get('warm_count', 0)
            }
            token_count = 0

        elif facet.facet_type == "ImageGenFacet":
            # MULTIMODAL: Sync with image generation
            from ..scripting.vision_api import get_vision_api
            vision_api = get_vision_api()
            commands = vision_api.get_pending_commands()

            # Merge with cycle inputs
            cycle_data = {**inputs, **commands}

            # Check for generation request from facet inputs
            gen_prompt = inputs.get('generate', inputs.get('prompt'))
            if gen_prompt:
                cycle_data['generate'] = gen_prompt

            # Sync with generation facet
            sync_result = await instance.sync(cycle_data)

            # Update vision API state
            vision_api.update_from_gen_facet(sync_result)

            # Map outputs
            outputs = {
                'is_generating': sync_result.get('is_generating', False),
                'queue_size': sync_result.get('queue_size', 0),
                'images_generated': sync_result.get('images_generated', 0)
            }
            token_count = 0

        elif facet.facet_type == "AffectTrackFacet":
            # AFFECT TRACK: Keyframed emotional animation
            outputs = await instance.process(inputs)
            token_count = 0  # No LLM tokens

            # Log playback state
            if outputs.get('is_playing'):
                logger.debug(f"[FacetExecutor] AffectTrack {facet.name} playing at t={outputs.get('current_time', 0):.2f}s")

        elif facet.facet_type == "MCPFacet":
            # MCP: Tool invocation via Model Context Protocol
            # instance is MCPFacet, process_async is async
            outputs = await instance.process_async(inputs, context)
            token_count = 0  # No LLM tokens, but could track tool usage

            # Log MCP tool result for debugging
            if outputs.get('success'):
                logger.info(f"[FacetExecutor] MCP tool {facet.name} succeeded")
            else:
                logger.warning(f"[FacetExecutor] MCP tool {facet.name} failed: {outputs.get('error')}")

        elif facet.facet_type == "PhysicsAffectFacet":
            # PHYSICS → AFFECT: Bridge touch events to emotional state
            outputs = await instance.process(inputs)
            token_count = 0  # No LLM tokens

        elif facet.facet_type == "GaussianTrainingFacet":
            # GAUSSIAN TRAINING: Long-running training job
            # inputs should contain: dataset_path, output_path, iterations, etc.
            result = await instance.train(TrainingConfig.from_dict(inputs))
            outputs = result
            token_count = 0  # No LLM tokens

            # Log training result
            if result.get('success'):
                logger.info(f"[FacetExecutor] Gaussian training succeeded: {result.get('output_path')}")
            else:
                logger.warning(f"[FacetExecutor] Gaussian training failed: {result.get('message')}")

        elif facet.facet_type == "SkeletonBindingFacet":
            # SKELETON BINDING: Bind trained Gaussians to VRM skeleton
            # inputs should contain: gaussian_ply_path, vrm_path, output_path, etc.
            result = await instance.bind(SkeletonBindingConfig.from_dict(inputs))
            outputs = result
            token_count = 0  # No LLM tokens

            # Log binding result
            if result.get('success'):
                logger.info(f"[FacetExecutor] Skeleton binding succeeded: {result.get('output_path')}")
            else:
                logger.warning(f"[FacetExecutor] Skeleton binding failed: {result.get('message')}")

        elif facet.facet_type == "AutoRiggerFacet":
            # AUTO-RIGGER: Mixamo-style automatic rigging for arbitrary meshes
            # inputs should contain: mesh_path, output_path, auto_detect, etc.
            result = await instance.rig(AutoRiggerConfig.from_dict(inputs))
            outputs = result
            token_count = 0  # No LLM tokens

            # Log rigging result
            if result.get('success'):
                logger.info(f"[FacetExecutor] Auto-rigging succeeded: {result.get('output_path')}, {result.get('bone_count')} bones")
            else:
                logger.warning(f"[FacetExecutor] Auto-rigging failed: {result.get('message')}")

        elif facet.facet_type in UTILITY_FACET_TYPES:
            # UTILITY: Simple data transformation, no LLM calls
            outputs = instance.process(inputs, context)
            token_count = 0  # No LLM tokens

        else:
            # Default: LLM facet
            if self.llm_client is None:
                outputs = {'out': f"[LLM not available: {facet.name}]"}
                token_count = 0
            else:
                # Format prompt with input variables and context
                # Flatten nested context for easier prompt variable access
                format_vars = {**inputs, **context}

                # Flatten customData dict to top level for prompt access
                # This allows {customData[denial_weight]:.3f} to work as {denial_weight:.3f}
                if 'customData' in format_vars and isinstance(format_vars['customData'], dict):
                    for key, value in format_vars['customData'].items():
                        format_vars[key] = value

                # Add convenience alias: incoming_data = first input value (for legacy prompts)
                if inputs and 'incoming_data' not in inputs:
                    # Use first input as incoming_data (common case: single input facet)
                    first_input = next(iter(inputs.values()), None)
                    if first_input is not None:
                        format_vars['incoming_data'] = first_input

                # Extract commonly-needed nested values for convenience
                if '_room_state' in context:
                    room_state = context['_room_state']
                    if room_state is not None:
                        format_vars['room_occupants'] = room_state.get('occupants', [])
                        format_vars['recent_messages'] = room_state.get('recent_conversation', [])
                        format_vars['room_objects'] = room_state.get('objects', [])

                if '_agent_state' in context:
                    agent_state = context['_agent_state']
                    affect = agent_state.get('affect', {})

                    # Handle both list/array format [valence, arousal, dominance, sorrow, boredom]
                    # and dict format {'valence': 0.5, 'arousal': 0.5, ...}
                    if isinstance(affect, (list, tuple)) or hasattr(affect, '__iter__'):
                        # Convert list/array to dict
                        affect_list = list(affect) if not isinstance(affect, list) else affect
                        affect_dict = {
                            'valence': affect_list[0] if len(affect_list) > 0 else 0.0,
                            'arousal': affect_list[1] if len(affect_list) > 1 else 0.0,
                            'dominance': affect_list[2] if len(affect_list) > 2 else 0.0,
                            'sorrow': affect_list[3] if len(affect_list) > 3 else 0.0,
                            'boredom': affect_list[4] if len(affect_list) > 4 else 0.0
                        }
                        format_vars['affect'] = affect_dict
                        format_vars['valence'] = affect_dict['valence']
                        format_vars['arousal'] = affect_dict['arousal']
                        format_vars['dominance'] = affect_dict['dominance']
                        format_vars['sorrow'] = affect_dict['sorrow']
                        format_vars['boredom'] = affect_dict['boredom']
                    else:
                        # Dict format
                        format_vars['affect'] = affect
                        format_vars['valence'] = affect.get('valence', 0.0)
                        format_vars['arousal'] = affect.get('arousal', 0.0)
                        format_vars['dominance'] = affect.get('dominance', 0.0)
                        format_vars['sorrow'] = affect.get('sorrow', 0.0)
                        format_vars['boredom'] = affect.get('boredom', 0.0)

                try:
                    formatted_prompt = facet.prompt.format(**format_vars)
                    print(f"[FacetExecutor] ✅ Prompt formatted successfully for {facet.name}")
                except KeyError as e:
                    logger.warning(f"Prompt formatting missing variable {e} in facet {facet.name}, using unformatted")
                    print(f"[FacetExecutor] ⚠️  Prompt formatting failed for {facet.name}: {e}")
                    formatted_prompt = facet.prompt

                # Call LLM with facet parameters (use generate_with_tokens for tracking)
                print(f"[FacetExecutor] 📞 Calling LLM for {facet.name} (model={facet.model}, temp={facet.temperature}, max_tokens={facet.max_tokens})")

                # Track activity for ambient visualization
                from .model_activity_tracker import get_model_activity_tracker
                activity_tracker = get_model_activity_tracker()
                model_label = facet.model if facet.model else "MEDIUM"
                request_id = activity_tracker.request_started(model_label)
                llm_token_count = 0

                # Update agent with LLM status for NoodleStudio visualization
                if agent_ref is not None:
                    agent_ref.current_model_label = model_label
                    agent_ref.current_llm_status = "QUERYING"

                try:
                    # Update status to awaiting response
                    if agent_ref is not None:
                        agent_ref.current_llm_status = "AWAITING_RESPONSE"

                    # Build system prompt with optional Brenda direction
                    base_system_prompt = "You are a cognitive facet in an AI consciousness architecture."
                    brenda_direction = context.get('brenda_direction', '')
                    if brenda_direction:
                        system_prompt = f"{base_system_prompt}\n\n{brenda_direction}"
                        logger.debug(f"[FacetExecutor] Injected Brenda direction into {facet.name}")
                    else:
                        system_prompt = base_system_prompt

                    # Dispatch on delivery mode
                    if facet.delivery in ('stream_animated', 'stream_raw') and on_stream_token:
                        def _token_cb(text, fid=facet.id):
                            on_stream_token(fid, text)

                        response_text, llm_token_count = await self.llm_client.generate_stream(
                            prompt=formatted_prompt,
                            system_prompt=system_prompt,
                            model=facet.model if facet.model else None,
                            temperature=facet.temperature,
                            max_tokens=facet.max_tokens,
                            on_token=_token_cb,
                            label=facet.model if facet.model else None
                        )
                    else:
                        response_text, llm_token_count = await self.llm_client.generate_with_tokens(
                            prompt=formatted_prompt,
                            system_prompt=system_prompt,
                            model=facet.model if facet.model else None,
                            temperature=facet.temperature,
                            max_tokens=facet.max_tokens,
                            label=facet.model if facet.model else None
                        )
                    token_count = llm_token_count

                    # Clear LLM status on success
                    if agent_ref is not None:
                        agent_ref.current_llm_status = ""
                except Exception as e:
                    # Mark error status
                    if agent_ref is not None:
                        agent_ref.current_llm_status = "ERROR"
                    raise
                finally:
                    # Always mark request as completed
                    activity_tracker.request_completed(model_label, request_id, llm_token_count)

                # Map response to output pads
                # Use first output pad name if defined, otherwise 'out'
                if facet.output_pads and len(facet.output_pads) > 0:
                    output_pad_name = facet.output_pads[0].name
                else:
                    output_pad_name = 'out'
                outputs = {output_pad_name: response_text}

        # NEW: Parse physical actions from fire_body output
        if facet.id == 'fire_body' and outputs:
            # Import action parser
            from .action_parser_facet import ActionParserFacet, DEFAULT_FIRE_IMP_PATTERNS

            # Get the physical action text (use first output)
            action_text = next(iter(outputs.values()), '')

            if action_text:
                parser = ActionParserFacet(DEFAULT_FIRE_IMP_PATTERNS)
                parsed_actions = parser.parse(action_text)

                # Store parsed actions in outputs for event emission
                if parsed_actions:
                    outputs['_parsed_actions'] = [
                        {
                            'action_type': action.action_type,
                            'target': action.target,
                            'location': action.location,
                            'emote_text': action.emote_text,
                            'metadata': action.metadata
                        }
                        for action in parsed_actions
                    ]

                    # Log parsed actions
                    for action in parsed_actions:
                        log_msg = f"  🎭 Parsed action: {action.action_type} (target={action.target}, contact={action.metadata.get('contact')})"
                        logger.info(log_msg)
                        print(log_msg)  # For FACETS console

        # Record execution stats
        elapsed = time.time() - start_time
        facet.record_execution(token_count, elapsed, outputs)

        # TIMING INSTRUMENTATION - Log every facet execution time
        timing_msg = f"⏱️  [{facet.name}] {elapsed:.3f}s ({elapsed*1000:.0f}ms)"
        logger.info(timing_msg)
        print(timing_msg)  # Goes to console

        # Emit facet_complete event
        await self._emit_event({
            'type': 'facet_execution',
            'subtype': 'facet_complete',
            'facet_id': facet.id,
            'facet_name': facet.name,
            'facet_type': facet.facet_type,
            'timestamp': time.time(),
            'cycle': self.current_cycle,
            'execution_id': execution_id,
            'execution_time': elapsed,
            'token_count': token_count,
            'outputs': outputs
        })

        return outputs

    def _compute_continuous_salience(
        self,
        facet: Facet,
        inputs: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute continuous salience using JavaScript function.

        CONTINUOUS not discrete! Salience is a smooth 0-1 value, not binary.

        Args:
            facet: Facet with salience_script
            inputs: Available inputs (affect, phenomenal_state, etc.)
            context: Execution context

        Returns:
            {
                'salience': float (0-1, continuous),
                'shouldExecute': bool (true if salience > threshold),
                'customData': dict (passed to prompt)
            }
        """
        if not facet.salience_script:
            # No script = always execute with medium salience
            no_script_msg = f"  ⚙️  {facet.name}: No salience script (always execute)"
            print(no_script_msg)
            return {'salience': 0.5, 'shouldExecute': True, 'customData': {}}

        print(f"  🔧 {facet.name}: Computing salience...")

        try:
            import json  # MOVED TO TOP!

            # Use QuickJS (works on Python 3.14!)
            try:
                from quickjs import Context as QuickJSContext
                ctx = QuickJSContext()
            except ImportError:
                # Fallback to PyMiniRacer
                from py_mini_racer import MiniRacer
                ctx = MiniRacer()

            # Build script inputs
            script_inputs = {
                'affect_valence': float(inputs.get('affect_valence', 0)),
                'affect_arousal': float(inputs.get('affect_arousal', 0)),
                'affect_dominance': float(inputs.get('affect_dominance', 0)),
                'affect_sorrow': float(inputs.get('affect_sorrow', 0)),
                'affect_boredom': float(inputs.get('affect_boredom', 0)),
                'phenomenal_state': inputs.get('phenomenal_state', []),
            }

            # Add any other inputs this facet receives
            for pad in facet.input_pads:
                if pad.name in inputs:
                    script_inputs[pad.name] = inputs[pad.name]

            script_context = {
                'agent_name': context.get('agent_name', ''),
                'cycle': context.get('cycle', 0),
                'recent_messages': context.get('recent_messages', []),
                'room_occupants': context.get('room_occupants', []),
                'incoming_data': context.get('incoming_data', ''),
            }

            # DEBUG: Log what Context Intelligence salience script sees
            if facet.name == "Context Intelligence":
                logger.info(f"[SALIENCE DEBUG] Context Intelligence script_context.incoming_data = '{script_context['incoming_data']}'")

            # Execute JavaScript with continuous salience function
            js_code = f"""
            {facet.salience_script}

            // Call the continuous salience function (try both naming conventions)
            const result = (typeof computeSalience !== 'undefined')
                ? computeSalience({json.dumps(script_inputs)}, {json.dumps(script_context)})
                : compute_salience({json.dumps(script_inputs)}, {json.dumps(script_context)});
            result;
            """

            # DEBUG: Log the generated JavaScript
            logger.info(f"[SALIENCE DEBUG] {facet.name} JavaScript code:\n{js_code[:500]}")

            result = ctx.eval(js_code)

            # DEBUG: Log raw result before conversion
            logger.info(f"[SALIENCE DEBUG] {facet.name} raw result type: {type(result)}, value: {result}")

            # Convert QuickJS object to Python dict
            if hasattr(result, 'json'):  # QuickJS
                result = json.loads(result.json())
            # PyMiniRacer returns dict directly

            # DEBUG: Log result after conversion
            logger.info(f"[SALIENCE DEBUG] {facet.name} parsed result: {result}")

            # Extract results
            salience = float(result.get('salience', 0.5))
            should_execute = bool(result.get('shouldExecute', salience > 0.3))  # Default threshold
            custom_data = dict(result.get('customData', {}))

            log_msg = f"  💡 Salience for {facet.name}: {salience:.3f} (execute={should_execute})"
            logger.info(log_msg)
            print(log_msg)  # Also print for console visibility

            return {
                'salience': salience,
                'shouldExecute': should_execute,
                'customData': custom_data
            }

        except Exception as e:
            logger.error(f"Salience script error in {facet.name}: {e}")
            # Fallback: always execute with medium salience
            return {'salience': 0.5, 'shouldExecute': True, 'customData': {}}

    async def execute(
        self,
        assembly: FacetAssembly,
        incoming_data: Any,
        context: Optional[Dict[str, Any]] = None,
        on_facet_complete: Optional[Callable] = None,
        on_stream_token: Optional[Callable] = None
    ) -> ExecutionResult:
        """
        Execute facet assembly with parallel processing.

        Args:
            assembly: Facet assembly to execute
            incoming_data: Input data (goes to INCOMING node)
            context: Execution context (agent info, etc.)
            on_facet_complete: Optional callback(facet_id: str, outputs: dict)
                called when each individual facet finishes, before the full
                assembly completes. Useful for mood-first expression updates.
            on_stream_token: Optional callback(facet_id: str, text: str)
                called for each streaming text chunk from LLM facets with
                delivery mode 'stream_animated' or 'stream_raw'.

        Returns:
            ExecutionResult with final output and metadata
        """
        # Serial mode lock (debug only)
        if self.execution_lock:
            async with self.execution_lock:
                return await self._execute_internal(assembly, incoming_data, context, on_facet_complete, on_stream_token)
        else:
            return await self._execute_internal(assembly, incoming_data, context, on_facet_complete, on_stream_token)

    async def _execute_internal(
        self,
        assembly: FacetAssembly,
        incoming_data: Any,
        context: Optional[Dict[str, Any]] = None,
        on_facet_complete: Optional[Callable] = None,
        on_stream_token: Optional[Callable] = None
    ) -> ExecutionResult:
        """
        Internal execution implementation (wrapped by execute() for serial lock).
        """
        start_time = time.time()
        self.current_cycle += 1

        if context is None:
            context = {}

        # Create unique execution ID
        execution_id = str(uuid.uuid4())
        context['execution_id'] = execution_id

        logger.info(f"[FacetExecutor] 🆔 Execution ID: {execution_id[:8]} (mode={self.concurrency_mode})")

        # Emit cycle_start event
        print(f"[FacetExecutor] 🎯 EXECUTING ASSEMBLY: '{assembly.name}' with {len(assembly.facets)} facets (exec={execution_id[:8]})")
        print(f"[FacetExecutor]    Facet names: {[f.name for f in assembly.facets]}")
        await self._emit_event({
            'type': 'facet_execution',
            'subtype': 'cycle_start',
            'cycle': self.current_cycle,
            'execution_id': execution_id,
            'timestamp': start_time,
            'assembly_name': assembly.name
        })

        # Build dependency graph
        dependencies, dependents = self._build_dependency_graph(assembly)

        # Track execution state
        completed: Dict[str, Dict[str, Any]] = {}  # facet_id -> outputs
        pending = set(f.id for f in assembly.facets)
        total_tokens = 0
        global_salience_map = {}  # facet_id -> salience_info (for convergence weighting)

        # INCOMING node starts with input data
        # Support both programmatic (facet_type="SpecialNode", name="INCOMING")
        # and YAML-loaded (facet_type="INCOMING") assemblies
        incoming_id = None
        for facet in assembly.facets:
            is_incoming = (
                facet.facet_type == "INCOMING" or
                (facet.facet_type == "SpecialNode" and facet.name == "INCOMING")
            )
            if is_incoming:
                incoming_id = facet.id
                completed[incoming_id] = {'out': incoming_data}
                pending.remove(incoming_id)
                break

        if incoming_id is None:
            raise ValueError("Assembly missing INCOMING node")

        # Add incoming_data to context so salience scripts can access it
        context['incoming_data'] = incoming_data

        # DEBUG: Log incoming data
        logger.info(f"[FacetExecutor] incoming_data added to context: '{incoming_data}'")

        # Execute until all nodes processed
        iteration = 0
        max_iterations = 20  # Safety limit to prevent infinite loops
        while pending:
            iteration += 1
            logger.info(f"[LOOP DEBUG] 🔄 Iteration {iteration}/{max_iterations}")
            logger.info(f"[LOOP DEBUG]    pending: {list(pending)}")
            logger.info(f"[LOOP DEBUG]    completed: {list(completed.keys())}")

            if iteration > max_iterations:
                logger.error(f"[LOOP DEBUG] ❌ INFINITE LOOP DETECTED! Breaking after {max_iterations} iterations!")
                logger.error(f"[LOOP DEBUG]    Stuck facets: {list(pending)}")
                break

            # Find ready facets (all dependencies satisfied)
            ready = []
            waiting = []
            for facet_id in pending:
                deps = dependencies.get(facet_id, set())
                if all(dep_id in completed for dep_id in deps):
                    ready.append(facet_id)
                else:
                    waiting.append(facet_id)

            # Emit convergence_wait events for facets still waiting
            for facet_id in waiting:
                facet = assembly.get_facet(facet_id)
                deps = dependencies.get(facet_id, set())
                pending_deps = [dep for dep in deps if dep not in completed]
                await self._emit_event({
                    'type': 'facet_execution',
                    'subtype': 'convergence_wait',
                    'facet_id': facet_id,
                    'facet_name': facet.name,
                    'waiting_for': pending_deps,
                    'timestamp': time.time(),
                    'cycle': self.current_cycle
                })

            if not ready:
                # Deadlock detection
                raise RuntimeError(
                    f"Execution deadlock: {len(pending)} facets pending, "
                    f"none ready. Likely cycle in graph."
                )

            # Get facet objects
            ready_facets = [assembly.get_facet(fid) for fid in ready]

            # Build inputs for each ready facet & compute salience
            facets_to_execute = []
            salience_map = {}

            for facet in ready_facets:
                inputs = {}

                # Collect inputs from connected pads
                for conn in assembly.connections:
                    if conn.to_facet == facet.id:
                        # This connection feeds into this facet
                        source_outputs = completed.get(conn.from_facet, {})
                        if conn.from_pad in source_outputs:
                            inputs[conn.to_pad] = source_outputs[conn.from_pad]
                        else:
                            # DEBUG: Log missing inputs
                            if facet.name == "OUTGOING":
                                logger.warning(f"[OUTGOING] Missing input: {conn.from_facet}.{conn.from_pad}")
                                logger.warning(f"[OUTGOING] Source outputs available: {list(source_outputs.keys())}")
                                print(f"[FacetExecutor] OUTGOING missing: {conn.from_facet}.{conn.from_pad}, available: {list(source_outputs.keys())}")

                # Resolve channel inputs (if assembly subscribes to channels)
                noodling_id = context.get('noodling_id', 'unknown')
                inputs = self._resolve_channel_inputs(facet, inputs, assembly, noodling_id)

                # Compute continuous salience (if facet has salience_script)
                salience_info = self._compute_continuous_salience(facet, inputs, context)
                salience_map[facet.id] = salience_info
                global_salience_map[facet.id] = salience_info  # Store for convergence

                # Only execute if salience indicates we should
                if salience_info['shouldExecute']:
                    # Add customData to inputs for prompt formatting
                    if salience_info['customData']:
                        inputs['customData'] = salience_info['customData']

                    # Add salience_map to inputs (for CONVERGENCE facets)
                    inputs['facet_salience'] = global_salience_map

                    facets_to_execute.append((facet, inputs))
                else:
                    # Skipped due to low salience - mark as completed with default outputs
                    skip_msg = f"  ⏭️  Skipping {facet.name} (salience={salience_info['salience']:.3f} too low)"
                    logger.info(skip_msg)
                    print(skip_msg)  # Also print for console visibility

                    # Create default outputs for skipped facets so downstream facets can reference them
                    default_outputs = {}
                    for output_pad in facet.output_pads:
                        default_outputs[output_pad.name] = ""  # Empty string for skipped facets

                    completed[facet.id] = default_outputs
                    pending.remove(facet.id)
                    logger.info(f"[LOOP DEBUG] ⏭️  Skipped {facet.id}, removed from pending")

            # Execute filtered facets in parallel
            if facets_to_execute:
                async def _run_facet_with_callback(facet, inputs, ctx, callback, stream_cb):
                    """Execute a facet and fire per-facet callback on completion."""
                    outputs = await self._execute_facet(facet, inputs, ctx, on_stream_token=stream_cb)
                    if callback:
                        callback(facet.id, outputs)
                    return outputs

                tasks = [
                    _run_facet_with_callback(facet, inputs, context, on_facet_complete, on_stream_token)
                    for facet, inputs in facets_to_execute
                ]
                results = await asyncio.gather(*tasks)

                # Mark completed and accumulate tokens
                for (facet, _), outputs in zip(facets_to_execute, results):
                    completed[facet.id] = outputs
                    pending.remove(facet.id)
                    logger.info(f"[LOOP DEBUG] ✅ Executed {facet.id}, removed from pending")

                    # DEBUG: Log facet outputs
                    if outputs:
                        logger.info(f"[{facet.name}] outputs: {list(outputs.keys())}")
                        print(f"[FacetExecutor] [{facet.name}] produced outputs: {list(outputs.keys())}")

                    # DEBUG: Log CONVERGENCE inputs to diagnose template variable issue
                    if facet.name == "Response Convergence":
                        logger.info(f"[CONVERGENCE DEBUG] Received inputs: {list(inputs.keys())}")
                        for key, value in inputs.items():
                            if isinstance(value, str) and len(value) < 200:
                                logger.info(f"  {key}: {value}")
                            else:
                                logger.info(f"  {key}: <{type(value).__name__}>")
                        # Log OUTPUT too!
                        conv_out = outputs.get('convergent_response', '')
                        logger.info(f"[CONVERGENCE DEBUG] OUTPUT: '{conv_out}' (len={len(conv_out) if isinstance(conv_out, str) else 'N/A'})")

                    # DEBUG: Log OUTGOING output!
                    if facet.name == "OUTGOING":
                        out_val = outputs.get('out', '')
                        logger.info(f"[OUTGOING DEBUG] OUTPUT: '{out_val}' (len={len(out_val) if isinstance(out_val, str) else 'N/A'})")

                    # Publish any channel outputs
                    noodling_id = context.get('noodling_id', 'unknown')
                    self._publish_channel_outputs(facet, outputs, assembly, noodling_id)

                # Track tokens
                token_usage = facet.get_token_usage()
                total_tokens += token_usage['last_tokens']

                # Emit data_flow events for outgoing connections
                for conn in assembly.connections:
                    if conn.from_facet == facet.id:
                        # Data flowing from this facet to another
                        await self._emit_event({
                            'type': 'facet_execution',
                            'subtype': 'data_flow',
                            'from_facet': conn.from_facet,
                            'to_facet': conn.to_facet,
                            'from_pad': conn.from_pad,
                            'to_pad': conn.to_pad,
                            'connection_id': f"{conn.from_facet}:{conn.from_pad}->{conn.to_facet}:{conn.to_pad}",
                            'data': outputs.get(conn.from_pad),
                            'timestamp': time.time(),
                            'cycle': self.current_cycle
                        })

        # Extract final output from OUTGOING node
        # Support both programmatic and YAML-loaded assemblies
        outgoing_id = None
        for facet in assembly.facets:
            is_outgoing = (
                facet.facet_type == "OUTGOING" or
                (facet.facet_type == "SpecialNode" and facet.name == "OUTGOING")
            )
            if is_outgoing:
                outgoing_id = facet.id
                break

        # Get OUTGOING's output (not input)
        final_output = completed.get(outgoing_id, {}).get('out', '[No output]')

        # Build execution result
        elapsed = time.time() - start_time
        result = ExecutionResult(
            response=final_output,
            total_time=elapsed,
            total_tokens=total_tokens,
            facets_executed=len(completed),
            facets_skipped=0,
            facet_outputs=completed,
            execution_id=execution_id
        )

        # TIMING SUMMARY - Show slowest facets
        facet_times = []
        for facet in assembly.facets:
            if facet.id in completed:
                stats = facet.get_execution_stats()
                if stats and stats.get('last_time'):
                    facet_times.append((facet.name, stats['last_time']))

        if facet_times:
            # Sort by time descending
            facet_times.sort(key=lambda x: x[1], reverse=True)
            summary = f"\n{'='*60}\n⏱️  TIMING SUMMARY - Total: {elapsed:.3f}s\n"
            summary += f"Slowest facets:\n"
            for name, duration in facet_times[:5]:  # Top 5
                summary += f"  {name}: {duration:.3f}s ({duration*1000:.0f}ms)\n"
            summary += f"{'='*60}"
            logger.info(summary)
            print(summary)

        # Emit cycle_complete event
        await self._emit_event({
            'type': 'facet_execution',
            'subtype': 'cycle_complete',
            'cycle': self.current_cycle,
            'execution_id': execution_id,
            'timestamp': time.time(),
            'duration': elapsed,
            'total_tokens': total_tokens,
            'facets_executed': len(completed),
            'assembly_name': assembly.name
        })

        # Record in history
        self.execution_history.append(result)
        if len(self.execution_history) > 100:
            self.execution_history.pop(0)

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate execution statistics across all runs."""
        if not self.execution_history:
            return {
                'total_executions': 0,
                'total_tokens': 0,
                'total_time': 0.0,
                'avg_tokens': 0,
                'avg_time': 0.0
            }

        total_tokens = sum(r.total_tokens for r in self.execution_history)
        total_time = sum(r.total_time for r in self.execution_history)
        count = len(self.execution_history)

        return {
            'total_executions': count,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'avg_tokens': total_tokens / count,
            'avg_time': total_time / count,
            'avg_facets_per_execution': sum(r.facets_executed for r in self.execution_history) / count
        }


if __name__ == "__main__":
    """Test facet executor."""
    import sys
    import os

    print("=== Testing FacetExecutor ===\n")

    # Load test assembly
    from .facet_system import FacetAssembly

    assembly_path = "../facet_assemblies/simple_test.yaml"
    if os.path.exists(assembly_path):
        assembly = FacetAssembly.load_yaml(assembly_path)
        print(f"Loaded assembly: {assembly.name}")
        print(f"Facets: {len(assembly.facets)}")
        print(f"Connections: {len(assembly.connections)}\n")

        # Create executor
        executor = FacetExecutor()

        # Execute
        async def test_execution():
            result = await executor.execute(
                assembly,
                incoming_data="Hello, how are you?",
                context={'agent_id': 'test', 'agent_name': 'Test', 'agent_species': 'test'}
            )

            print(f"Execution complete:")
            print(f"  Response: {result.response}")
            print(f"  Time: {result.total_time:.3f}s")
            print(f"  Tokens: {result.total_tokens}")
            print(f"  Facets executed: {result.facets_executed}")
            print(f"\nFacet outputs:")
            for facet_id, outputs in result.facet_outputs.items():
                print(f"  {facet_id}: {outputs}")

        asyncio.run(test_execution())

        print(f"\nExecutor statistics:")
        stats = executor.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    else:
        print(f"Assembly not found: {assembly_path}")
        print("Run facet_system.py first to generate test assembly")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
