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
#   Timeline Recorder - Bridge from ExecutionEventBus to Cognitive Timeline
#
#   Listens to facet execution events and builds timeline dat...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.timeline_recorder
# PURPOSE:  Timeline Recorder
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FacetRecord, CycleRecord, AffectSample, AgentTimeline, TimelineSession
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from PyQt6.QtCore import QObject, pyqtSignal

from .execution_event_bus import (
    get_event_bus,
    Event,
    EventChannel,
    EventListener
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FacetRecord:
    """
    Record of a single facet execution.

    Contains everything needed to inspect the facet in the timeline:
    - Timing (when it started, how long it took)
    - Inputs/outputs (for debugging)
    - Token usage (for cost tracking)
    - Prompt (for LLM facets, for A/B testing)
    """
    facet_id: str
    facet_name: str
    facet_type: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    token_count: int = 0
    salience: float = 0.5
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None  # For LLM facets
    execution_id: str = ""
    cycle: int = 0


@dataclass
class CycleRecord:
    """
    Record of a complete cognition cycle.

    A cycle represents one "thought" - from incoming stimulus
    through all facets to outgoing response.
    """
    cycle_id: str
    cycle_number: int
    cycle_type: str  # 'reactive' | 'autonomous'
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    incoming_text: str = ""
    outgoing_text: str = ""
    assembly_name: str = ""
    total_tokens: int = 0
    facets: List[FacetRecord] = field(default_factory=list)


@dataclass
class AffectSample:
    """
    Single affect sample for waveform visualization.
    """
    timestamp: float
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    sorrow: float = 0.0
    boredom: float = 0.0


@dataclass
class AgentTimeline:
    """
    Timeline data for a single agent.
    """
    agent_id: str
    agent_name: str
    cycles: List[CycleRecord] = field(default_factory=list)
    affect_samples: List[AffectSample] = field(default_factory=list)

    # Currently recording cycle (in progress)
    _current_cycle: Optional[CycleRecord] = field(default=None, repr=False)
    _pending_facets: Dict[str, FacetRecord] = field(default_factory=dict, repr=False)


@dataclass
class TimelineSession:
    """
    Complete recording session across all agents.
    """
    session_id: str
    start_time: float
    agents: Dict[str, AgentTimeline] = field(default_factory=dict)

    def get_or_create_agent(self, agent_id: str, agent_name: str = "") -> AgentTimeline:
        """Get existing agent timeline or create new one."""
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentTimeline(
                agent_id=agent_id,
                agent_name=agent_name or agent_id
            )
        return self.agents[agent_id]


# =============================================================================
# Timeline Recorder
# =============================================================================

class TimelineRecorder(QObject):
    """
    Records facet execution events to timeline data structures.

    Subscribes to ExecutionEventBus and emits Qt signals when
    new data is available for the ProfilerPanel.

    Signals:
        cycleRecorded(CycleRecord): Emitted when a cycle completes
        facetStarted(FacetRecord): Emitted when a facet starts
        facetCompleted(FacetRecord): Emitted when a facet completes
        affectUpdated(str, AffectSample): Emitted for affect changes (agent_id, sample)
    """

    # Qt Signals for ProfilerPanel
    cycleRecorded = pyqtSignal(object)  # CycleRecord
    facetStarted = pyqtSignal(object)   # FacetRecord
    facetCompleted = pyqtSignal(object) # FacetRecord
    affectUpdated = pyqtSignal(str, object)  # agent_id, AffectSample

    def __init__(self, parent=None):
        super().__init__(parent)

        # Current recording session
        self.session = TimelineSession(
            session_id=f"session_{int(time.time())}",
            start_time=time.time()
        )

        # Recording state
        self.is_recording = True
        self.session_start = time.time()

        # Event bus connection
        self.event_bus = None
        self.listeners: List[EventListener] = []

        # Stats
        self.total_cycles_recorded = 0
        self.total_facets_recorded = 0

        logger.info("[TimelineRecorder] Initialized")

    def start_recording(self):
        """Start recording events from the bus."""
        if self.event_bus is not None:
            return  # Already connected

        try:
            # Try to get event bus - this may fail outside async context
            self.event_bus = get_event_bus()
            self.is_recording = True

            # Register listeners for facet execution events
            self.listeners = [
                self.event_bus.register_listener(
                    self._on_cycle_start,
                    event_type='facet_execution',
                    event_subtype='cycle_start',
                    channel=EventChannel.EXECUTION
                ),
                self.event_bus.register_listener(
                    self._on_facet_start,
                    event_type='facet_execution',
                    event_subtype='facet_start',
                    channel=EventChannel.EXECUTION
                ),
                self.event_bus.register_listener(
                    self._on_facet_complete,
                    event_type='facet_execution',
                    event_subtype='facet_complete',
                    channel=EventChannel.EXECUTION
                ),
                self.event_bus.register_listener(
                    self._on_cycle_complete,
                    event_type='facet_execution',
                    event_subtype='cycle_complete',
                    channel=EventChannel.EXECUTION
                ),
            ]

            logger.info("[TimelineRecorder] Started recording (4 listeners registered)")

        except RuntimeError as e:
            # No event loop - defer until server starts
            logger.warning(f"[TimelineRecorder] Deferred recording start: {e}")
            self.is_recording = False
            self.event_bus = None

    def stop_recording(self):
        """Stop recording events."""
        if self.event_bus and self.listeners:
            for listener in self.listeners:
                self.event_bus.unregister_listener(listener)
            self.listeners = []

        self.is_recording = False
        logger.info("[TimelineRecorder] Stopped recording")

    def clear_session(self):
        """Clear current session and start fresh."""
        self.session = TimelineSession(
            session_id=f"session_{int(time.time())}",
            start_time=time.time()
        )
        self.session_start = time.time()
        self.total_cycles_recorded = 0
        self.total_facets_recorded = 0
        logger.info("[TimelineRecorder] Session cleared")

    # =========================================================================
    # Event Handlers (async callbacks from EventBus)
    # =========================================================================

    async def _on_cycle_start(self, event: Event):
        """Handle cycle_start event."""
        if not self.is_recording:
            return

        data = event.data
        execution_id = data.get('execution_id', '')
        cycle_number = data.get('cycle', 0)
        assembly_name = data.get('assembly_name', 'Unknown')

        # Get agent ID from context (if available) or use execution_id
        agent_id = data.get('agent_id', 'default')
        agent_name = data.get('agent_name', 'Agent')

        # Get or create agent timeline
        agent = self.session.get_or_create_agent(agent_id, agent_name)

        # Create new cycle record
        cycle = CycleRecord(
            cycle_id=execution_id,
            cycle_number=cycle_number,
            cycle_type='reactive',  # TODO: detect autonomous cycles
            start_time=event.timestamp - self.session_start,
            assembly_name=assembly_name
        )

        agent._current_cycle = cycle
        agent._pending_facets = {}

        logger.debug(f"[TimelineRecorder] Cycle {cycle_number} started: {assembly_name}")

    async def _on_facet_start(self, event: Event):
        """Handle facet_start event."""
        if not self.is_recording:
            return

        data = event.data

        # Create facet record
        facet = FacetRecord(
            facet_id=data.get('facet_id', ''),
            facet_name=data.get('facet_name', 'Unknown'),
            facet_type=data.get('facet_type', 'Unknown'),
            start_time=event.timestamp - self.session_start,
            execution_id=data.get('execution_id', ''),
            cycle=data.get('cycle', 0),
            inputs=data.get('inputs', {})
        )

        # Store pending facet (will be completed when facet_complete arrives)
        agent_id = data.get('agent_id', 'default')
        agent = self.session.get_or_create_agent(agent_id)
        agent._pending_facets[facet.facet_id] = facet

        # Emit signal for real-time visualization
        self.facetStarted.emit(facet)

        logger.debug(f"[TimelineRecorder] Facet started: {facet.facet_name}")

    async def _on_facet_complete(self, event: Event):
        """Handle facet_complete event."""
        if not self.is_recording:
            return

        data = event.data
        facet_id = data.get('facet_id', '')

        # Find pending facet
        agent_id = data.get('agent_id', 'default')
        agent = self.session.get_or_create_agent(agent_id)

        facet = agent._pending_facets.pop(facet_id, None)
        if not facet:
            # Create facet if we missed the start event
            facet = FacetRecord(
                facet_id=facet_id,
                facet_name=data.get('facet_name', 'Unknown'),
                facet_type=data.get('facet_type', 'Unknown'),
                start_time=event.timestamp - self.session_start - data.get('execution_time', 0),
                execution_id=data.get('execution_id', ''),
                cycle=data.get('cycle', 0)
            )

        # Complete the facet record
        facet.end_time = event.timestamp - self.session_start
        facet.duration_ms = data.get('execution_time', 0) * 1000
        facet.token_count = data.get('token_count', 0)
        facet.outputs = data.get('outputs', {})

        # Extract affect values if this is CharmNetwork output
        outputs = facet.outputs
        if 'affect_valence' in outputs:
            sample = AffectSample(
                timestamp=facet.end_time,
                valence=outputs.get('affect_valence', 0),
                arousal=outputs.get('affect_arousal', 0),
                dominance=outputs.get('affect_dominance', 0),
                sorrow=outputs.get('affect_sorrow', 0),
                boredom=outputs.get('affect_boredom', 0)
            )
            agent.affect_samples.append(sample)
            self.affectUpdated.emit(agent_id, sample)

        # Add to current cycle
        if agent._current_cycle:
            agent._current_cycle.facets.append(facet)

        self.total_facets_recorded += 1

        # Emit signal for real-time visualization
        self.facetCompleted.emit(facet)

        logger.debug(f"[TimelineRecorder] Facet completed: {facet.facet_name} ({facet.duration_ms:.0f}ms)")

    async def _on_cycle_complete(self, event: Event):
        """Handle cycle_complete event."""
        if not self.is_recording:
            return

        data = event.data
        agent_id = data.get('agent_id', 'default')
        agent = self.session.get_or_create_agent(agent_id)

        if agent._current_cycle:
            cycle = agent._current_cycle

            # Complete the cycle record
            cycle.end_time = event.timestamp - self.session_start
            cycle.duration_ms = data.get('duration', 0) * 1000
            cycle.total_tokens = data.get('total_tokens', 0)

            # Add to agent's cycle list
            agent.cycles.append(cycle)
            agent._current_cycle = None

            self.total_cycles_recorded += 1

            # Emit signal for ProfilerPanel
            self.cycleRecorded.emit(cycle)

            logger.info(f"[TimelineRecorder] Cycle {cycle.cycle_number} completed: "
                       f"{len(cycle.facets)} facets, {cycle.duration_ms:.0f}ms, "
                       f"{cycle.total_tokens} tokens")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_all_cycles(self) -> List[CycleRecord]:
        """Get all recorded cycles across all agents."""
        cycles = []
        for agent in self.session.agents.values():
            cycles.extend(agent.cycles)
        return sorted(cycles, key=lambda c: c.start_time)

    def get_agent_cycles(self, agent_id: str) -> List[CycleRecord]:
        """Get cycles for a specific agent."""
        agent = self.session.agents.get(agent_id)
        return agent.cycles if agent else []

    def get_facets_in_range(self, start_time: float, end_time: float) -> List[FacetRecord]:
        """Get all facets that executed within a time range."""
        facets = []
        for agent in self.session.agents.values():
            for cycle in agent.cycles:
                for facet in cycle.facets:
                    if start_time <= facet.start_time <= end_time:
                        facets.append(facet)
        return sorted(facets, key=lambda f: f.start_time)

    def get_affect_samples(self, agent_id: str) -> List[AffectSample]:
        """Get affect samples for an agent."""
        agent = self.session.agents.get(agent_id)
        return agent.affect_samples if agent else []

    def get_stats(self) -> Dict[str, Any]:
        """Get recording statistics."""
        return {
            'session_id': self.session.session_id,
            'recording': self.is_recording,
            'duration_seconds': time.time() - self.session_start,
            'agents': len(self.session.agents),
            'total_cycles': self.total_cycles_recorded,
            'total_facets': self.total_facets_recorded,
            'agent_details': {
                agent_id: {
                    'name': agent.agent_name,
                    'cycles': len(agent.cycles),
                    'affect_samples': len(agent.affect_samples)
                }
                for agent_id, agent in self.session.agents.items()
            }
        }


# =============================================================================
# Global Singleton
# =============================================================================

_global_timeline_recorder: Optional[TimelineRecorder] = None

def get_timeline_recorder() -> TimelineRecorder:
    """
    Get global timeline recorder instance (singleton).

    The recorder will attempt to connect to the EventBus when start_recording()
    is called. If no async event loop is running, it will defer connection
    until the server starts.

    Returns:
        Global TimelineRecorder instance
    """
    global _global_timeline_recorder

    if _global_timeline_recorder is None:
        _global_timeline_recorder = TimelineRecorder()
        # Don't auto-start here - let ProfilerPanel control recording
        # This avoids the "no running event loop" error at startup

    return _global_timeline_recorder


def reset_timeline_recorder():
    """Reset global timeline recorder (for testing)."""
    global _global_timeline_recorder

    if _global_timeline_recorder:
        _global_timeline_recorder.stop_recording()

    _global_timeline_recorder = None


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    """Test timeline recorder with mock events."""

    print("=== Testing TimelineRecorder ===\n")

    async def test_recorder():
        from .execution_event_bus import ExecutionEventBus

        # Create event bus and recorder
        bus = ExecutionEventBus()
        bus.start()

        recorder = TimelineRecorder()
        recorder.event_bus = bus
        recorder.start_recording()

        # Simulate cognition cycle
        await bus.emit(
            'facet_execution', 'cycle_start',
            data={
                'execution_id': 'exec_001',
                'cycle': 1,
                'assembly_name': 'red_fire_anklebiter',
                'agent_id': 'red'
            }
        )

        await bus.emit(
            'facet_execution', 'facet_start',
            data={
                'facet_id': 'charm_net',
                'facet_name': 'CharmNetwork',
                'facet_type': 'CharmNetworkFacet',
                'execution_id': 'exec_001',
                'cycle': 1,
                'agent_id': 'red',
                'inputs': {'perception': 'hello red'}
            }
        )

        await asyncio.sleep(0.1)

        await bus.emit(
            'facet_execution', 'facet_complete',
            data={
                'facet_id': 'charm_net',
                'facet_name': 'CharmNetwork',
                'facet_type': 'CharmNetworkFacet',
                'execution_id': 'exec_001',
                'cycle': 1,
                'agent_id': 'red',
                'execution_time': 0.05,
                'token_count': 0,
                'outputs': {
                    'affect_valence': 0.3,
                    'affect_arousal': 0.7,
                    'affect_dominance': 0.5,
                    'affect_sorrow': 0.1,
                    'affect_boredom': 0.2
                }
            }
        )

        await bus.emit(
            'facet_execution', 'cycle_complete',
            data={
                'execution_id': 'exec_001',
                'cycle': 1,
                'agent_id': 'red',
                'duration': 0.5,
                'total_tokens': 150
            }
        )

        await asyncio.sleep(0.2)

        # Check results
        stats = recorder.get_stats()
        print(f"Stats: {stats}")

        cycles = recorder.get_all_cycles()
        print(f"\nRecorded {len(cycles)} cycles:")
        for cycle in cycles:
            print(f"  Cycle {cycle.cycle_number}: {len(cycle.facets)} facets, {cycle.duration_ms:.0f}ms")
            for facet in cycle.facets:
                print(f"    - {facet.facet_name}: {facet.duration_ms:.0f}ms")

        await bus.stop()
        print("\n=== Test complete ===")

    asyncio.run(test_recorder())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
