"""
Tests for FacetAssemblyComponent - Facets as Universal Components

This tests the core architecture that allows facet assemblies to be
attached to ANY entity (Noodling, Prim, UI element).

Author: Caitlyn + Claude
Date: January 2026
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mark all async tests for anyio
pytestmark = pytest.mark.anyio

from noodlestudio.core.facet_assembly_component import (
    FacetAssemblyComponent,
    AssemblyEvent,
    EventEmitter,
)
from noodlestudio.core.component_base import (
    ComponentCategory,
    component_registry,
)
from noodlestudio.core.facet_system import FacetAssembly, Facet


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def simple_assembly_yaml():
    """Create a minimal assembly YAML for testing."""
    # Use dedent to remove leading whitespace
    from textwrap import dedent
    return dedent("""
        name: "Test Assembly"
        version: "1.0.0"
        description: "Simple test assembly"

        facets:
          - id: "incoming"
            name: "INCOMING"
            type: "SpecialNode"
            prompt: ""
            model: ""
            temperature: 0.7
            max_tokens: 100
            position: {x: 100, y: 200}
            inputs: []
            outputs:
              - name: "out"
                type: "output"
                description: "Raw input"

          - id: "test_facet"
            name: "Test"
            type: "LLMFacet"
            prompt: "Echo: {in}"
            model: "SMALL"
            temperature: 0.7
            max_tokens: 100
            position: {x: 300, y: 200}
            inputs:
              - name: "in"
                type: "input"
                description: "Input"
            outputs:
              - name: "out"
                type: "output"
                description: "Output"

          - id: "outgoing"
            name: "OUTGOING"
            type: "SpecialNode"
            prompt: ""
            model: ""
            temperature: 0.7
            max_tokens: 100
            position: {x: 500, y: 200}
            inputs:
              - name: "in"
                type: "input"
                description: "Final response"
            outputs: []

        connections:
          - from: "incoming.out"
            to: "test_facet.in"
          - from: "test_facet.out"
            to: "outgoing.in"
    """).strip()


@pytest.fixture
def temp_assembly_file(simple_assembly_yaml):
    """Create a temporary assembly file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(simple_assembly_yaml)
        f.flush()  # Ensure content is written to disk
        temp_path = f.name
    # File is now closed and fully written
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def mock_executor():
    """Create a mock FacetExecutor."""
    executor = MagicMock()
    # Mock execute to return a result
    mock_result = MagicMock()
    mock_result.response = "Test response"
    mock_result.facet_outputs = {'OUTGOING': {'out': 'Test output'}}
    mock_result.total_tokens = 100
    mock_result.total_time = 0.5
    executor.execute = AsyncMock(return_value=mock_result)
    return executor


# ==============================================================================
# Component Creation Tests
# ==============================================================================

class TestFacetAssemblyComponentCreation:
    """Tests for creating FacetAssemblyComponent instances."""

    def test_create_empty_component(self):
        """Test creating component without assembly path."""
        component = FacetAssemblyComponent()
        assert component.component_type == "facet_assembly"
        assert component.assembly_path == ""
        assert component.assembly is None
        assert not component.run_in_cognition_loop
        assert component.tick_rate == 0.1

    def test_create_with_assembly_path(self, temp_assembly_file):
        """Test creating component with assembly path."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        assert component.assembly_path == temp_assembly_file

    def test_component_type_and_category(self):
        """Test component type and category are correct."""
        component = FacetAssemblyComponent()
        assert component.component_type == "facet_assembly"
        assert component.category == ComponentCategory.CHARM
        assert not component.singleton  # Multiple allowed!

    def test_display_name_without_assembly(self):
        """Test display name when no assembly is loaded."""
        component = FacetAssemblyComponent()
        assert component.display_name == "Facet Assembly"

    def test_display_name_with_assembly(self, temp_assembly_file):
        """Test display name includes assembly name after loading."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        assert "Test Assembly" in component.display_name

    def test_property_specs(self):
        """Test property specs are defined correctly."""
        component = FacetAssemblyComponent()
        specs = component.property_specs
        spec_names = [s.name for s in specs]
        assert 'assembly_path' in spec_names
        assert 'run_in_cognition_loop' in spec_names
        assert 'tick_rate' in spec_names
        assert 'auto_run_on_attach' in spec_names


# ==============================================================================
# Assembly Loading Tests
# ==============================================================================

class TestAssemblyLoading:
    """Tests for loading facet assemblies."""

    def test_load_assembly_from_file(self, temp_assembly_file):
        """Test loading assembly from YAML file."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        result = component._load_assembly()
        assert result is True
        assert component.assembly is not None
        assert component.assembly.name == "Test Assembly"

    def test_load_assembly_missing_file(self):
        """Test loading assembly with non-existent file."""
        component = FacetAssemblyComponent(assembly_path="/nonexistent/path.yaml")
        result = component._load_assembly()
        assert result is False
        assert component.assembly is None

    def test_lazy_load_assembly(self, temp_assembly_file):
        """Test lazy loading via property access."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        # Assembly not loaded yet
        assert component._assembly is None
        # Access triggers load
        assembly = component.assembly
        assert assembly is not None
        assert assembly.name == "Test Assembly"

    def test_reload_assembly(self, temp_assembly_file):
        """Test reloading assembly from disk."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        assert component.assembly is not None
        # Reload
        component._assembly = None  # Clear cache
        result = component.reload_assembly()
        assert result is True
        assert component.assembly is not None


# ==============================================================================
# Execution Tests
# ==============================================================================

class TestAssemblyExecution:
    """Tests for one-shot assembly execution."""

    @pytest.mark.asyncio
    async def test_run_without_executor(self, temp_assembly_file):
        """Test run fails gracefully without executor."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        result = await component.run({"in": "test input"})
        assert 'error' in result
        assert 'executor' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_run_without_assembly(self, mock_executor):
        """Test run fails gracefully without assembly."""
        component = FacetAssemblyComponent()
        component.set_executor(mock_executor)
        result = await component.run({"in": "test input"})
        assert 'error' in result
        assert 'assembly' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_run_with_executor(self, temp_assembly_file, mock_executor):
        """Test successful one-shot execution."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        component.set_executor(mock_executor)

        result = await component.run({"in": "Hello world"})

        assert result['success'] is True
        assert 'response' in result
        mock_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_statistics(self, temp_assembly_file, mock_executor):
        """Test execution statistics are tracked."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        component.set_executor(mock_executor)

        await component.run({"in": "test"})

        stats = component.get_statistics()
        assert stats['execution_count'] == 1
        assert stats['total_tokens'] == 100


# ==============================================================================
# Event Tests
# ==============================================================================

class TestAssemblyEvents:
    """Tests for event emission."""

    def test_event_emitter_basic(self):
        """Test basic event emission."""
        emitter = EventEmitter()
        received = []
        emitter.on('complete', lambda e: received.append(e))
        emitter.emit(AssemblyEvent('complete', 'test-id', {'data': 'test'}))
        assert len(received) == 1
        assert received[0].event_type == 'complete'

    def test_event_emitter_multiple_listeners(self):
        """Test multiple listeners for same event."""
        emitter = EventEmitter()
        received = []
        emitter.on('complete', lambda e: received.append('a'))
        emitter.on('complete', lambda e: received.append('b'))
        emitter.emit(AssemblyEvent('complete', 'test-id'))
        assert received == ['a', 'b']

    def test_event_emitter_remove_listener(self):
        """Test removing event listener."""
        emitter = EventEmitter()
        received = []
        callback = lambda e: received.append(e)
        emitter.on('complete', callback)
        emitter.off('complete', callback)
        emitter.emit(AssemblyEvent('complete', 'test-id'))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_complete_event_fires(self, temp_assembly_file, mock_executor):
        """Test OnComplete event fires after execution."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        component.set_executor(mock_executor)

        received = []
        component.add_listener('complete', lambda e: received.append(e))

        await component.run({"in": "test"})

        assert len(received) == 1
        assert received[0].event_type == 'complete'

    @pytest.mark.asyncio
    async def test_error_event_fires(self, temp_assembly_file):
        """Test OnError event fires on execution error."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        # No executor set - will cause error

        received = []
        component.add_listener('error', lambda e: received.append(e))

        await component.run({"in": "test"})

        assert len(received) == 1
        assert received[0].event_type == 'error'


# ==============================================================================
# Binding Tests
# ==============================================================================

class TestInputOutputBindings:
    """Tests for input/output bindings."""

    def test_bind_input(self):
        """Test binding an input pad."""
        component = FacetAssemblyComponent()
        component.bind_input('text', 'text_field.value')
        assert 'text' in component._input_bindings
        assert component._input_bindings['text'] == 'text_field.value'

    def test_bind_output(self):
        """Test binding an output pad."""
        component = FacetAssemblyComponent()
        component.bind_output('result', 'result_label.text')
        assert 'result' in component._output_bindings
        assert component._output_bindings['result'] == 'result_label.text'

    def test_unbind_input(self):
        """Test removing input binding."""
        component = FacetAssemblyComponent()
        component.bind_input('text', 'text_field.value')
        component.unbind_input('text')
        assert 'text' not in component._input_bindings

    def test_unbind_output(self):
        """Test removing output binding."""
        component = FacetAssemblyComponent()
        component.bind_output('result', 'result_label.text')
        component.unbind_output('result')
        assert 'result' not in component._output_bindings


# ==============================================================================
# Serialization Tests
# ==============================================================================

class TestSerialization:
    """Tests for component serialization."""

    def test_to_dict_basic(self):
        """Test serialization to dictionary."""
        component = FacetAssemblyComponent()
        component._assembly_path = "assemblies/test.yaml"
        component._run_in_cognition_loop = True
        component._tick_rate = 0.5

        data = component.to_dict()

        assert data['type'] == 'facet_assembly'
        assert data['assembly_path'] == "assemblies/test.yaml"
        assert data['run_in_cognition_loop'] is True
        assert data['tick_rate'] == 0.5

    def test_to_dict_with_bindings(self):
        """Test serialization includes bindings."""
        component = FacetAssemblyComponent()
        component.bind_input('text', 'input.value')
        component.bind_output('result', 'output.text')

        data = component.to_dict()

        assert 'input_bindings' in data
        assert data['input_bindings']['text'] == 'input.value'
        assert 'output_bindings' in data
        assert data['output_bindings']['result'] == 'output.text'

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            'type': 'facet_assembly',
            'id': 'test-id',
            'assembly_path': 'assemblies/test.yaml',
            'assembly_name': 'Test Assembly',
            'run_in_cognition_loop': True,
            'tick_rate': 0.5,
            'auto_run_on_attach': True,
            'input_bindings': {'text': 'input.value'},
            'output_bindings': {'result': 'output.text'},
        }

        component = FacetAssemblyComponent.from_dict(data)

        assert component.assembly_path == 'assemblies/test.yaml'
        assert component._assembly_name == 'Test Assembly'
        assert component.run_in_cognition_loop is True
        assert component.tick_rate == 0.5
        assert component.auto_run_on_attach is True
        assert component._input_bindings['text'] == 'input.value'
        assert component._output_bindings['result'] == 'output.text'

    def test_round_trip(self):
        """Test serialization round-trip preserves data."""
        component = FacetAssemblyComponent()
        component._assembly_path = "assemblies/test.yaml"
        component._assembly_name = "Test Assembly"
        component._run_in_cognition_loop = True
        component._tick_rate = 0.25
        component.bind_input('text', 'field.value')
        component.bind_output('result', 'label.text')

        data = component.to_dict()
        restored = FacetAssemblyComponent.from_dict(data)

        assert restored.assembly_path == component.assembly_path
        assert restored._assembly_name == component._assembly_name
        assert restored.run_in_cognition_loop == component.run_in_cognition_loop
        assert restored.tick_rate == component.tick_rate
        assert restored._input_bindings == component._input_bindings
        assert restored._output_bindings == component._output_bindings


# ==============================================================================
# Component Registry Tests
# ==============================================================================

class TestComponentRegistry:
    """Tests for component registry integration."""

    def test_registered_with_registry(self):
        """Test FacetAssemblyComponent is registered."""
        cls = component_registry.get_class('facet_assembly')
        assert cls is FacetAssemblyComponent

    def test_create_via_registry(self):
        """Test creating component via registry."""
        component = component_registry.create('facet_assembly', entity_id='test-entity')
        assert component is not None
        assert isinstance(component, FacetAssemblyComponent)
        assert component.entity_id == 'test-entity'

    def test_display_info(self):
        """Test getting display info from registry."""
        info = component_registry.get_display_info('facet_assembly')
        assert info['type'] == 'facet_assembly'
        assert info['category'] == 'charm'
        assert info['singleton'] is False


# ==============================================================================
# Cognition Loop Tests
# ==============================================================================

class TestCognitionLoop:
    """Tests for continuous cognition loop mode."""

    def test_run_in_cognition_loop_property(self):
        """Test setting run_in_cognition_loop property."""
        component = FacetAssemblyComponent()
        assert not component.run_in_cognition_loop
        # Setting to True would try to start async task, so just test the internal value
        component._run_in_cognition_loop = True
        assert component.run_in_cognition_loop
        component._run_in_cognition_loop = False  # Reset to avoid warning

    def test_tick_rate_bounds(self):
        """Test tick rate is bounded."""
        component = FacetAssemblyComponent()
        component.tick_rate = 0.001  # Below min
        assert component.tick_rate >= 0.01
        component.tick_rate = 100  # Above max
        assert component.tick_rate <= 60.0

    @pytest.mark.asyncio
    async def test_cognition_loop_starts(self, temp_assembly_file, mock_executor):
        """Test cognition loop starts when checkbox is checked."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        component.set_executor(mock_executor)
        component._tick_rate = 0.01  # Fast for testing

        # Enable cognition loop
        component.run_in_cognition_loop = True

        # Wait a bit for loop to start
        await asyncio.sleep(0.05)

        # Should have executed at least once
        assert mock_executor.execute.called

        # Stop the loop
        component.run_in_cognition_loop = False
        await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_cognition_loop_stops(self, temp_assembly_file, mock_executor):
        """Test cognition loop stops when checkbox is unchecked."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        component.set_executor(mock_executor)
        component._tick_rate = 0.01

        component.run_in_cognition_loop = True
        await asyncio.sleep(0.03)

        call_count_before = mock_executor.execute.call_count

        component.run_in_cognition_loop = False
        await asyncio.sleep(0.05)

        # No additional calls after stopping
        assert mock_executor.execute.call_count == call_count_before


# ==============================================================================
# Input/Output Pad Discovery Tests
# ==============================================================================

class TestPadDiscovery:
    """Tests for discovering assembly input/output pads."""

    def test_input_pads_empty_without_assembly(self):
        """Test input_pads returns empty list without assembly."""
        component = FacetAssemblyComponent()
        assert component.input_pads == []

    def test_output_pads_empty_without_assembly(self):
        """Test output_pads returns empty list without assembly."""
        component = FacetAssemblyComponent()
        assert component.output_pads == []

    def test_input_pads_from_assembly(self, temp_assembly_file):
        """Test input_pads returns INCOMING node's output pads."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        pads = component.input_pads
        assert 'out' in pads  # INCOMING has 'out' output

    def test_output_pads_from_assembly(self, temp_assembly_file):
        """Test output_pads returns OUTGOING node's input pads."""
        component = FacetAssemblyComponent(assembly_path=temp_assembly_file)
        component._load_assembly()
        pads = component.output_pads
        assert 'in' in pads  # OUTGOING has 'in' input


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
