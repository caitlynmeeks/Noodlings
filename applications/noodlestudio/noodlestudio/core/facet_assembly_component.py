"""
Facet Assembly Component - Attachable facet assemblies for any entity

This is THE key architectural unification: Facet Assemblies are now attachable
components that can go on ANY object (Noodling, Prim, UI element). Each assembly
gets a "run_in_cognition_loop" checkbox - checked means continuous (thinking),
unchecked means one-shot (event-triggered).

This makes Facets the universal visual logic language for everything in NoodleStudio.

Key concepts:
- Multiple assemblies per entity (each is independent)
- singleton = False (unlike most components)
- Two execution modes:
  - Continuous: Runs in cognition loop every tick_rate seconds
  - One-shot: Runs once on demand (via events, scripts, or manual trigger)

Author: Caitlyn + Claude
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import asyncio
import time
import uuid
import logging

from .component_base import (
    ComponentBase,
    ComponentCategory,
    PropertySpec,
    register_component,
)
from .facet_system import FacetAssembly

logger = logging.getLogger(__name__)


@dataclass
class AssemblyEvent:
    """Event emitted by FacetAssemblyComponent."""
    event_type: str  # 'complete', 'state_change', 'error'
    assembly_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventEmitter:
    """Simple event emitter for component events."""

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event_type: str, callback: Callable) -> None:
        """Add event listener."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def off(self, event_type: str, callback: Callable) -> None:
        """Remove event listener."""
        if event_type in self._listeners:
            self._listeners[event_type] = [
                cb for cb in self._listeners[event_type] if cb != callback
            ]

    def emit(self, event: AssemblyEvent) -> None:
        """Emit event to all listeners."""
        callbacks = self._listeners.get(event.event_type, [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")


@register_component
class FacetAssemblyComponent(ComponentBase):
    """
    Attachable facet assembly component.

    This component allows ANY entity (Noodling, Prim, UI element) to have
    facet-based logic. The key UX is the "Run in cognition loop" checkbox:

    - CHECKED: Assembly runs continuously every tick_rate seconds
    - UNCHECKED: Assembly runs once on demand (event-triggered)

    Properties:
        assembly_path: Path to .yaml assembly file (relative to project)
        run_in_cognition_loop: Whether to run continuously
        tick_rate: Seconds between cognitive ticks (if continuous)
        auto_run_on_attach: Run once when component is attached

    Events:
        OnComplete: Fires when one-shot execution finishes
        OnStateChange: Fires when continuous assembly state changes
        OnError: Fires on execution error

    Usage:
        # Get assembly component from entity
        assembly = entity.GetComponent("facet_assembly", "translate-chinese")

        # Run one-shot manually
        result = await assembly.run({"text": "Hello world"})

        # Check continuous state
        if assembly.is_running:
            current_state = assembly.last_output
    """

    def __init__(self, entity_id: str = "", assembly_path: str = ""):
        super().__init__(entity_id)

        # Assembly reference
        self._assembly_path: str = assembly_path
        self._assembly: Optional[FacetAssembly] = None
        self._assembly_name: str = ""

        # THE CHECKBOX - continuous vs one-shot mode
        self._run_in_cognition_loop: bool = False

        # Continuous mode settings
        self._tick_rate: float = 0.1  # 100ms default

        # One-shot settings
        self._auto_run_on_attach: bool = False

        # Runtime state
        self._is_running: bool = False
        self._last_output: Optional[Dict[str, Any]] = None
        self._last_error: Optional[str] = None
        self._execution_count: int = 0
        self._total_tokens: int = 0
        self._last_execution_time: float = 0.0

        # Input/output bindings (component_name.property -> pad_name)
        self._input_bindings: Dict[str, str] = {}
        self._output_bindings: Dict[str, str] = {}

        # Events
        self._events = EventEmitter()

        # Cognition loop task
        self._cognition_task: Optional[asyncio.Task] = None

        # Executor reference (set by cognition manager)
        self._executor = None

    # ==========================================================================
    # ComponentBase abstract properties
    # ==========================================================================

    @property
    def component_type(self) -> str:
        return "facet_assembly"

    @property
    def display_name(self) -> str:
        if self._assembly_name:
            return f"Facet Assembly: {self._assembly_name}"
        return "Facet Assembly"

    @property
    def category(self) -> ComponentCategory:
        return ComponentCategory.CHARM

    @property
    def singleton(self) -> bool:
        # MULTIPLE assemblies allowed per entity!
        return False

    @property
    def description(self) -> str:
        return "Visual logic assembly - runs as continuous cognition or one-shot on events"

    @property
    def property_specs(self) -> List[PropertySpec]:
        return [
            PropertySpec(
                name='assembly_path',
                display_name='Assembly',
                property_type='file',
                description='Path to assembly YAML file',
                file_filter='Assembly Files (*.yaml *.yml);;All Files (*)'
            ),
            PropertySpec(
                name='run_in_cognition_loop',
                display_name='Run in cognition loop',
                property_type='bool',
                default=False,
                description='If checked, runs continuously. If unchecked, runs on-demand.'
            ),
            PropertySpec(
                name='tick_rate',
                display_name='Tick Rate (seconds)',
                property_type='float',
                default=0.1,
                min_value=0.01,
                max_value=60.0,
                description='How often to run in continuous mode'
            ),
            PropertySpec(
                name='auto_run_on_attach',
                display_name='Auto-run on attach',
                property_type='bool',
                default=False,
                description='Run assembly once when component is added'
            ),
        ]

    # ==========================================================================
    # Properties with getters/setters
    # ==========================================================================

    @property
    def assembly_path(self) -> str:
        return self._assembly_path

    @assembly_path.setter
    def assembly_path(self, value: str):
        if value != self._assembly_path:
            self._assembly_path = value
            self._assembly = None  # Force reload
            self._mark_dirty()

    @property
    def run_in_cognition_loop(self) -> bool:
        return self._run_in_cognition_loop

    @run_in_cognition_loop.setter
    def run_in_cognition_loop(self, value: bool):
        if value != self._run_in_cognition_loop:
            self._run_in_cognition_loop = value
            self._mark_dirty()

            # Start/stop cognition loop based on checkbox
            if value:
                self._start_cognition_loop()
            else:
                self._stop_cognition_loop()

    @property
    def tick_rate(self) -> float:
        return self._tick_rate

    @tick_rate.setter
    def tick_rate(self, value: float):
        self._tick_rate = max(0.01, min(60.0, value))
        self._mark_dirty()

    @property
    def auto_run_on_attach(self) -> bool:
        return self._auto_run_on_attach

    @auto_run_on_attach.setter
    def auto_run_on_attach(self, value: bool):
        self._auto_run_on_attach = value
        self._mark_dirty()

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def last_output(self) -> Optional[Dict[str, Any]]:
        return self._last_output

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def assembly(self) -> Optional[FacetAssembly]:
        """Get loaded assembly (lazy load)."""
        if self._assembly is None and self._assembly_path:
            self._load_assembly()
        return self._assembly

    @property
    def input_pads(self) -> List[str]:
        """Get list of input pad names from assembly."""
        if not self.assembly:
            return []
        # INCOMING node's output pad is effectively the assembly input
        for facet in self.assembly.facets:
            if facet.name == "INCOMING":
                return [p.name for p in facet.output_pads]
        return ['in']

    @property
    def output_pads(self) -> List[str]:
        """Get list of output pad names from assembly."""
        if not self.assembly:
            return []
        # OUTGOING node's input pad is effectively the assembly output
        for facet in self.assembly.facets:
            if facet.name == "OUTGOING":
                return [p.name for p in facet.input_pads]
        return ['out']

    # ==========================================================================
    # Events
    # ==========================================================================

    @property
    def OnComplete(self) -> EventEmitter:
        """Event fired when one-shot execution completes."""
        return self._events

    @property
    def OnStateChange(self) -> EventEmitter:
        """Event fired when continuous assembly state changes."""
        return self._events

    @property
    def OnError(self) -> EventEmitter:
        """Event fired on execution error."""
        return self._events

    def add_listener(self, event_type: str, callback: Callable) -> None:
        """Add event listener (Unity-style API)."""
        self._events.on(event_type, callback)

    def remove_listener(self, event_type: str, callback: Callable) -> None:
        """Remove event listener."""
        self._events.off(event_type, callback)

    # ==========================================================================
    # Lifecycle
    # ==========================================================================

    def on_added(self, entity: Any) -> None:
        """Called when component is added to entity."""
        super().on_added(entity)

        # Load assembly
        if self._assembly_path:
            self._load_assembly()

        # Auto-run if configured
        if self._auto_run_on_attach:
            asyncio.create_task(self.run({}))

        # Start cognition loop if configured
        if self._run_in_cognition_loop:
            self._start_cognition_loop()

    def on_removed(self) -> None:
        """Called when component is removed."""
        self._stop_cognition_loop()
        super().on_removed()

    # ==========================================================================
    # Assembly loading
    # ==========================================================================

    def _load_assembly(self) -> bool:
        """Load assembly from path."""
        if not self._assembly_path:
            return False

        try:
            path = Path(self._assembly_path)
            if path.exists():
                self._assembly = FacetAssembly.load_yaml(str(path))
                self._assembly_name = self._assembly.name
                logger.info(f"Loaded assembly: {self._assembly_name} from {path}")
                return True
            else:
                logger.warning(f"Assembly file not found: {path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load assembly: {e}")
            self._last_error = str(e)
            return False

    def reload_assembly(self) -> bool:
        """Force reload assembly from disk."""
        self._assembly = None
        return self._load_assembly()

    # ==========================================================================
    # Execution
    # ==========================================================================

    async def run(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run assembly once (one-shot mode).

        Args:
            inputs: Input values for INCOMING node
            context: Optional execution context

        Returns:
            Output values from OUTGOING node
        """
        if not self.assembly:
            error = "No assembly loaded"
            self._last_error = error
            self._events.emit(AssemblyEvent('error', self.id, {'error': error}))
            return {'error': error}

        if not self._executor:
            error = "No executor available"
            self._last_error = error
            self._events.emit(AssemblyEvent('error', self.id, {'error': error}))
            return {'error': error}

        self._is_running = True
        start_time = time.time()

        try:
            # Build context
            exec_context = {
                'component_id': self.id,
                'entity_id': self._entity_id,
                **(context or {})
            }

            # Resolve input bindings
            resolved_inputs = self._resolve_input_bindings(inputs)

            # Execute assembly
            result = await self._executor.execute(
                self.assembly,
                resolved_inputs,
                exec_context
            )

            # Store output
            self._last_output = result.facet_outputs
            self._last_error = None
            self._execution_count += 1
            self._total_tokens += result.total_tokens
            self._last_execution_time = time.time() - start_time

            # Apply output bindings
            self._apply_output_bindings(result.facet_outputs)

            # Emit completion event
            self._events.emit(AssemblyEvent('complete', self.id, {
                'response': result.response,
                'outputs': result.facet_outputs,
                'tokens': result.total_tokens,
                'time': self._last_execution_time
            }))

            return {
                'response': result.response,
                'outputs': result.facet_outputs,
                'success': True
            }

        except Exception as e:
            logger.error(f"Assembly execution error: {e}")
            self._last_error = str(e)
            self._events.emit(AssemblyEvent('error', self.id, {'error': str(e)}))
            return {'error': str(e), 'success': False}

        finally:
            self._is_running = False

    def _resolve_input_bindings(self, inputs: Dict[str, Any]) -> Any:
        """Resolve input bindings to get actual values."""
        # For now, just pass through inputs
        # TODO: Resolve {component.property} bindings
        if 'in' in inputs:
            return inputs['in']
        return inputs

    def _apply_output_bindings(self, outputs: Dict[str, Any]) -> None:
        """Apply output bindings to target components."""
        # TODO: Resolve {component.property} bindings and set values
        pass

    # ==========================================================================
    # Cognition Loop
    # ==========================================================================

    def _start_cognition_loop(self) -> None:
        """Start continuous cognition loop."""
        if self._cognition_task and not self._cognition_task.done():
            return  # Already running

        self._cognition_task = asyncio.create_task(self._cognition_loop())
        logger.info(f"Started cognition loop for assembly: {self._assembly_name}")

    def _stop_cognition_loop(self) -> None:
        """Stop continuous cognition loop."""
        if self._cognition_task:
            self._cognition_task.cancel()
            self._cognition_task = None
            logger.info(f"Stopped cognition loop for assembly: {self._assembly_name}")

    async def _cognition_loop(self) -> None:
        """Main cognition loop - runs assembly at tick_rate."""
        while self._run_in_cognition_loop:
            try:
                # Gather inputs from bindings
                inputs = self._gather_bound_inputs()

                # Execute
                old_output = self._last_output
                result = await self.run(inputs)

                # Emit state change if output changed
                if result.get('success') and result.get('outputs') != old_output:
                    self._events.emit(AssemblyEvent('state_change', self.id, {
                        'outputs': result.get('outputs'),
                        'previous': old_output
                    }))

                # Wait for next tick
                await asyncio.sleep(self._tick_rate)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cognition loop error: {e}")
                self._events.emit(AssemblyEvent('error', self.id, {'error': str(e)}))
                await asyncio.sleep(self._tick_rate)

    def _gather_bound_inputs(self) -> Dict[str, Any]:
        """Gather input values from bindings."""
        # TODO: Resolve bindings from other components
        return {}

    # ==========================================================================
    # Input/Output Bindings
    # ==========================================================================

    def bind_input(self, pad_name: str, source: str) -> None:
        """
        Bind an input pad to a source.

        Args:
            pad_name: Name of input pad
            source: Source expression (e.g., "text_field.value", "slider.value")
        """
        self._input_bindings[pad_name] = source
        self._mark_dirty()

    def bind_output(self, pad_name: str, target: str) -> None:
        """
        Bind an output pad to a target.

        Args:
            pad_name: Name of output pad
            target: Target expression (e.g., "result_label.text", "progress.value")
        """
        self._output_bindings[pad_name] = target
        self._mark_dirty()

    def unbind_input(self, pad_name: str) -> None:
        """Remove input binding."""
        if pad_name in self._input_bindings:
            del self._input_bindings[pad_name]
            self._mark_dirty()

    def unbind_output(self, pad_name: str) -> None:
        """Remove output binding."""
        if pad_name in self._output_bindings:
            del self._output_bindings[pad_name]
            self._mark_dirty()

    # ==========================================================================
    # Executor management
    # ==========================================================================

    def set_executor(self, executor) -> None:
        """Set the FacetExecutor instance for running assemblies."""
        self._executor = executor

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self._execution_count,
            'total_tokens': self._total_tokens,
            'last_execution_time': self._last_execution_time,
            'avg_tokens': (
                self._total_tokens / self._execution_count
                if self._execution_count > 0 else 0
            ),
            'is_running': self._is_running,
            'run_in_cognition_loop': self._run_in_cognition_loop,
        }

    # ==========================================================================
    # Serialization
    # ==========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML storage."""
        data = self._base_to_dict()
        data.update({
            'assembly_path': self._assembly_path,
            'assembly_name': self._assembly_name,
            'run_in_cognition_loop': self._run_in_cognition_loop,
            'tick_rate': self._tick_rate,
            'auto_run_on_attach': self._auto_run_on_attach,
            'input_bindings': self._input_bindings,
            'output_bindings': self._output_bindings,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], entity_id: str = "") -> 'FacetAssemblyComponent':
        """Deserialize from dictionary."""
        component = cls(entity_id=entity_id, assembly_path=data.get('assembly_path', ''))
        component._base_from_dict(data)
        component._assembly_name = data.get('assembly_name', '')
        component._run_in_cognition_loop = data.get('run_in_cognition_loop', False)
        component._tick_rate = data.get('tick_rate', 0.1)
        component._auto_run_on_attach = data.get('auto_run_on_attach', False)
        component._input_bindings = data.get('input_bindings', {})
        component._output_bindings = data.get('output_bindings', {})
        return component


__all__ = [
    'FacetAssemblyComponent',
    'AssemblyEvent',
    'EventEmitter',
]
