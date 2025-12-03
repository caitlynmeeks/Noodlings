"""
Facet Executor - Parallel execution engine with synchronization

Executes facet assemblies with:
- Topological ordering (dependencies respected)
- Parallel execution where possible
- Synchronization gates (wait for all inputs)
- Flow control integration
- Token tracking
- Error handling and recovery

Like electrical current through a circuit - facets execute when all
inputs are ready, multiple paths run in parallel, converges at sync points.

Author: Commander Spock + Cadet Caity
Date: November 28, 2025
"""

import asyncio
import time
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

from .facet_system import Facet, FacetAssembly, FacetConnection
from .charm_network_facet import CharmNetworkFacet, CharmNetworkOutput
from .scripted_facet import ScriptedFacet, ScriptContext
from .flow_control_facets import (
    TickerGateFacet, ConditionalBranchFacet,
    RateLimiterFacet, CacheFacet, AccumulatorFacet
)
from .execution_event_bus import get_event_bus, EventChannel, EventPriority

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

    def __init__(self, llm_client=None, event_callback=None, use_event_bus=True):
        """
        Initialize executor.

        Args:
            llm_client: LLM client for calling language models (optional)
            event_callback: Async callback for execution events (optional, deprecated)
                          Use event bus instead for better decoupling
            use_event_bus: Use global event bus for event distribution (recommended)
        """
        self.llm_client = llm_client
        self.event_callback = event_callback  # Legacy support
        self.use_event_bus = use_event_bus
        self.current_cycle = 0

        # Facet instances (cached)
        self.facet_instances: Dict[str, Any] = {}

        # Execution history
        self.execution_history: List[ExecutionResult] = []

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

    def _get_facet_instance(self, facet: Facet) -> Any:
        """
        Get or create facet instance.

        Caches instances for stateful facets (CharmNetwork, ScriptedFacet, etc.)
        """
        if facet.id in self.facet_instances:
            return self.facet_instances[facet.id]

        # Create instance based on type
        if facet.facet_type == "CharmNetworkFacet":
            # Extract checkpoint path from metadata or facet config
            checkpoint_path = facet.model  # Stored in model field for special facets
            instance = CharmNetworkFacet(checkpoint_path)
            self.facet_instances[facet.id] = instance
            return instance

        elif facet.facet_type == "ScriptedFacet":
            # Extract script from prompt field
            instance = ScriptedFacet(facet.id, facet.prompt, script_language="javascript")
            self.facet_instances[facet.id] = instance
            return instance

        elif facet.facet_type == "TickerGateFacet":
            # Parse config from facet metadata
            interval = int(facet.max_tokens)  # Using max_tokens as interval storage (hack)
            instance = TickerGateFacet(facet.id, interval=interval)
            self.facet_instances[facet.id] = instance
            return instance

        elif facet.facet_type == "ConditionalBranchFacet":
            # Parse condition from prompt
            condition = facet.prompt
            variables = [p.name for p in facet.input_pads if p.name != 'in']
            instance = ConditionalBranchFacet(facet.id, condition, variables)
            self.facet_instances[facet.id] = instance
            return instance

        elif facet.facet_type == "RateLimiterFacet":
            interval = float(facet.temperature)  # Using temperature as interval (hack)
            instance = RateLimiterFacet(facet.id, min_interval=interval)
            self.facet_instances[facet.id] = instance
            return instance

        elif facet.facet_type == "CacheFacet":
            ttl = int(facet.max_tokens)  # Using max_tokens as TTL
            instance = CacheFacet(facet.id, ttl=ttl)
            self.facet_instances[facet.id] = instance
            return instance

        elif facet.facet_type == "AccumulatorFacet":
            window_size = int(facet.max_tokens)
            instance = AccumulatorFacet(facet.id, window_size=window_size)
            self.facet_instances[facet.id] = instance
            return instance

        elif facet.facet_type == "SpecialNode":
            # INCOMING/OUTGOING - no instance needed
            return None

        else:
            # Default: LLM facet (will call LLM in execute)
            return None

    async def _execute_facet(
        self,
        facet: Facet,
        inputs: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single facet.

        Args:
            facet: Facet to execute
            inputs: Input values from connected pads
            context: Execution context (cycle, agent info, etc.)

        Returns:
            Dict of output values
        """
        start_time = time.time()

        # Emit facet_start event
        await self._emit_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'facet_id': facet.id,
            'facet_name': facet.name,
            'facet_type': facet.facet_type,
            'timestamp': start_time,
            'cycle': self.current_cycle
        })

        # Get or create instance
        instance = self._get_facet_instance(facet)

        # Execute based on type
        if facet.facet_type == "SpecialNode":
            # INCOMING/OUTGOING - just pass through
            outputs = {'out': inputs.get('in', inputs)}
            token_count = 0

        elif facet.facet_type == "CharmNetworkFacet":
            # Neural computation
            perception = inputs.get('perception', '')
            affect_in = inputs.get('affect_input')
            result = instance.process(perception, affect_in)

            # Map CharmNetworkOutput to pad outputs
            outputs = {
                'valence': result.valence,
                'arousal': result.arousal,
                'fear': result.fear,
                'sorrow': result.sorrow,
                'boredom': result.boredom,
                'surprise': result.surprise,
                'phenomenal_state': result.phenomenal_state
            }
            token_count = 0

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

        else:
            # Default: LLM facet
            if self.llm_client is None:
                outputs = {'out': f"[LLM not available: {facet.name}]"}
                token_count = 0
            else:
                # Format prompt with input variables and context
                # Flatten nested context for easier prompt variable access
                format_vars = {**inputs, **context}

                # Add convenience alias: incoming_data = first input value (for legacy prompts)
                if inputs and 'incoming_data' not in inputs:
                    # Use first input as incoming_data (common case: single input facet)
                    first_input = next(iter(inputs.values()), None)
                    if first_input is not None:
                        format_vars['incoming_data'] = first_input

                # Extract commonly-needed nested values for convenience
                if '_room_state' in context:
                    room_state = context['_room_state']
                    format_vars['room_occupants'] = room_state.get('occupants', [])
                    format_vars['recent_messages'] = room_state.get('recent_conversation', [])
                    format_vars['room_objects'] = room_state.get('objects', [])

                if '_agent_state' in context:
                    agent_state = context['_agent_state']
                    format_vars['affect'] = agent_state.get('affect', {})
                    format_vars['valence'] = agent_state.get('affect', {}).get('valence', 0.0)
                    format_vars['arousal'] = agent_state.get('affect', {}).get('arousal', 0.0)
                    format_vars['dominance'] = agent_state.get('affect', {}).get('dominance', 0.0)
                    format_vars['sorrow'] = agent_state.get('affect', {}).get('sorrow', 0.0)
                    format_vars['boredom'] = agent_state.get('affect', {}).get('boredom', 0.0)

                try:
                    formatted_prompt = facet.prompt.format(**format_vars)
                except KeyError as e:
                    logger.warning(f"Prompt formatting missing variable {e} in facet {facet.name}, using unformatted")
                    formatted_prompt = facet.prompt

                # Call LLM with facet parameters (use generate_with_tokens for tracking)
                response_text, token_count = await self.llm_client.generate_with_tokens(
                    prompt=formatted_prompt,
                    system_prompt="You are a cognitive facet in an AI consciousness architecture.",
                    model=facet.model if facet.model else None,
                    temperature=facet.temperature,
                    max_tokens=facet.max_tokens
                )

                # Map response to output pads
                # Primary output goes to 'out' pad, can be customized per facet
                outputs = {'out': response_text}

        # Record execution stats
        elapsed = time.time() - start_time
        facet.record_execution(token_count, elapsed, outputs)

        # Emit facet_complete event
        await self._emit_event({
            'type': 'facet_execution',
            'subtype': 'facet_complete',
            'facet_id': facet.id,
            'facet_name': facet.name,
            'facet_type': facet.facet_type,
            'timestamp': time.time(),
            'cycle': self.current_cycle,
            'execution_time': elapsed,
            'token_count': token_count,
            'outputs': outputs
        })

        return outputs

    async def execute(
        self,
        assembly: FacetAssembly,
        incoming_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute facet assembly with parallel processing.

        Args:
            assembly: Facet assembly to execute
            incoming_data: Input data (goes to INCOMING node)
            context: Execution context (agent info, etc.)

        Returns:
            ExecutionResult with final output and metadata
        """
        start_time = time.time()
        self.current_cycle += 1

        if context is None:
            context = {}

        # Emit cycle_start event
        await self._emit_event({
            'type': 'facet_execution',
            'subtype': 'cycle_start',
            'cycle': self.current_cycle,
            'timestamp': start_time,
            'assembly_name': assembly.name
        })

        # Build dependency graph
        dependencies, dependents = self._build_dependency_graph(assembly)

        # Track execution state
        completed: Dict[str, Dict[str, Any]] = {}  # facet_id -> outputs
        pending = set(f.id for f in assembly.facets)
        total_tokens = 0

        # INCOMING node starts with input data
        incoming_id = None
        for facet in assembly.facets:
            if facet.facet_type == "SpecialNode" and facet.name == "INCOMING":
                incoming_id = facet.id
                completed[incoming_id] = {'out': incoming_data}
                pending.remove(incoming_id)
                break

        if incoming_id is None:
            raise ValueError("Assembly missing INCOMING node")

        # Execute until all nodes processed
        while pending:
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

            # Build inputs for each ready facet
            facet_inputs = []
            for facet in ready_facets:
                inputs = {}

                # Collect inputs from connected pads
                for conn in assembly.connections:
                    if conn.to_facet == facet.id:
                        # This connection feeds into this facet
                        source_outputs = completed.get(conn.from_facet, {})
                        if conn.from_pad in source_outputs:
                            inputs[conn.to_pad] = source_outputs[conn.from_pad]

                facet_inputs.append(inputs)

            # Execute ready facets in parallel
            tasks = [
                self._execute_facet(facet, inputs, context)
                for facet, inputs in zip(ready_facets, facet_inputs)
            ]
            results = await asyncio.gather(*tasks)

            # Mark completed and accumulate tokens
            for facet, outputs in zip(ready_facets, results):
                completed[facet.id] = outputs
                pending.remove(facet.id)

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
        outgoing_id = None
        for facet in assembly.facets:
            if facet.facet_type == "SpecialNode" and facet.name == "OUTGOING":
                outgoing_id = facet.id
                break

        final_output = completed.get(outgoing_id, {}).get('in', '[No output]')

        # Build execution result
        elapsed = time.time() - start_time
        result = ExecutionResult(
            response=final_output,
            total_time=elapsed,
            total_tokens=total_tokens,
            facets_executed=len(completed),
            facets_skipped=0,
            facet_outputs=completed
        )

        # Emit cycle_complete event
        await self._emit_event({
            'type': 'facet_execution',
            'subtype': 'cycle_complete',
            'cycle': self.current_cycle,
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
