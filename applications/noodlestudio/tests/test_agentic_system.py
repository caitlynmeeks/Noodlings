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
#   Test Suite for Agentic Facet System
#
#   Tests for: 1. Utility Facets - All 31 types 2. MCP Integr...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_agentic_system
# PURPOSE:  Test Suite for Agentic Facet System
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestMathFacets, TestLogicFacets, TestStringFacets, TestArrayFacets, TestDataFacets
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import asyncio
import json
import sys
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# UTILITY FACETS TESTS
# =============================================================================

class TestMathFacets:
    """Test math utility facets."""

    def test_add(self):
        from noodlestudio.core.utility_facets import MathAddFacet
        facet = MathAddFacet("test_add")
        result = facet.process({'a': 5, 'b': 3})
        assert result['result'] == 8
        assert result['out'] == 8

    def test_subtract(self):
        from noodlestudio.core.utility_facets import MathSubtractFacet
        facet = MathSubtractFacet("test_sub")
        result = facet.process({'a': 10, 'b': 4})
        assert result['result'] == 6

    def test_multiply(self):
        from noodlestudio.core.utility_facets import MathMultiplyFacet
        facet = MathMultiplyFacet("test_mul")
        result = facet.process({'a': 7, 'b': 6})
        assert result['result'] == 42

    def test_divide(self):
        from noodlestudio.core.utility_facets import MathDivideFacet
        facet = MathDivideFacet("test_div")
        result = facet.process({'a': 20, 'b': 4})
        assert result['result'] == 5

    def test_divide_by_zero(self):
        from noodlestudio.core.utility_facets import MathDivideFacet
        facet = MathDivideFacet("test_div_zero")
        result = facet.process({'a': 10, 'b': 0})
        assert result['result'] == 0
        assert 'error' in result

    def test_min(self):
        from noodlestudio.core.utility_facets import MathMinFacet
        facet = MathMinFacet("test_min")
        result = facet.process({'a': 5, 'b': 3})
        assert result['result'] == 3

    def test_max(self):
        from noodlestudio.core.utility_facets import MathMaxFacet
        facet = MathMaxFacet("test_max")
        result = facet.process({'a': 5, 'b': 3})
        assert result['result'] == 5

    def test_clamp(self):
        from noodlestudio.core.utility_facets import MathClampFacet
        facet = MathClampFacet("test_clamp")

        # Value within range
        result = facet.process({'value': 5, 'min': 0, 'max': 10})
        assert result['result'] == 5

        # Value below min
        result = facet.process({'value': -5, 'min': 0, 'max': 10})
        assert result['result'] == 0

        # Value above max
        result = facet.process({'value': 15, 'min': 0, 'max': 10})
        assert result['result'] == 10

    def test_abs(self):
        from noodlestudio.core.utility_facets import MathAbsFacet
        facet = MathAbsFacet("test_abs")
        result = facet.process({'value': -5})
        assert result['result'] == 5


class TestLogicFacets:
    """Test logic utility facets."""

    def test_and_true(self):
        from noodlestudio.core.utility_facets import LogicAndFacet
        facet = LogicAndFacet("test_and")
        result = facet.process({'a': True, 'b': True})
        assert result['result'] is True

    def test_and_false(self):
        from noodlestudio.core.utility_facets import LogicAndFacet
        facet = LogicAndFacet("test_and")
        result = facet.process({'a': True, 'b': False})
        assert result['result'] is False

    def test_or(self):
        from noodlestudio.core.utility_facets import LogicOrFacet
        facet = LogicOrFacet("test_or")
        result = facet.process({'a': False, 'b': True})
        assert result['result'] is True

    def test_not(self):
        from noodlestudio.core.utility_facets import LogicNotFacet
        facet = LogicNotFacet("test_not")
        result = facet.process({'value': True})
        assert result['result'] is False

    def test_compare_equal(self):
        from noodlestudio.core.utility_facets import LogicCompareFacet
        facet = LogicCompareFacet("test_cmp", operator='==')
        result = facet.process({'a': 5, 'b': 5})
        assert result['result'] is True

    def test_compare_greater(self):
        from noodlestudio.core.utility_facets import LogicCompareFacet
        facet = LogicCompareFacet("test_cmp", operator='>')
        result = facet.process({'a': 10, 'b': 5})
        assert result['result'] is True

    def test_switch_true(self):
        from noodlestudio.core.utility_facets import LogicSwitchFacet
        facet = LogicSwitchFacet("test_switch")
        result = facet.process({
            'condition': True,
            'true_value': 'yes',
            'false_value': 'no'
        })
        assert result['result'] == 'yes'

    def test_switch_false(self):
        from noodlestudio.core.utility_facets import LogicSwitchFacet
        facet = LogicSwitchFacet("test_switch")
        result = facet.process({
            'condition': False,
            'true_value': 'yes',
            'false_value': 'no'
        })
        assert result['result'] == 'no'


class TestStringFacets:
    """Test string utility facets."""

    def test_concat(self):
        from noodlestudio.core.utility_facets import StringConcatFacet
        facet = StringConcatFacet("test_concat", separator=' ')
        result = facet.process({'a': 'Hello', 'b': 'World'})
        assert result['result'] == 'Hello World'

    def test_split(self):
        from noodlestudio.core.utility_facets import StringSplitFacet
        facet = StringSplitFacet("test_split", delimiter=',')
        result = facet.process({'value': 'a,b,c'})
        assert result['result'] == ['a', 'b', 'c']
        assert result['first'] == 'a'
        assert result['last'] == 'c'
        assert result['count'] == 3

    def test_replace(self):
        from noodlestudio.core.utility_facets import StringReplaceFacet
        facet = StringReplaceFacet("test_replace")
        result = facet.process({
            'value': 'Hello World',
            'search': 'World',
            'replace': 'Noodlings'
        })
        assert result['result'] == 'Hello Noodlings'

    def test_format(self):
        from noodlestudio.core.utility_facets import StringFormatFacet
        facet = StringFormatFacet("test_format")
        result = facet.process({
            'template': '{name} is {age} years old',
            'name': 'Alice',
            'age': 30
        })
        assert result['result'] == 'Alice is 30 years old'

    def test_length(self):
        from noodlestudio.core.utility_facets import StringLengthFacet
        facet = StringLengthFacet("test_len")
        result = facet.process({'value': 'Hello'})
        assert result['result'] == 5

    def test_contains_true(self):
        from noodlestudio.core.utility_facets import StringContainsFacet
        facet = StringContainsFacet("test_contains")
        result = facet.process({'value': 'Hello World', 'search': 'World'})
        assert result['result'] is True

    def test_contains_false(self):
        from noodlestudio.core.utility_facets import StringContainsFacet
        facet = StringContainsFacet("test_contains")
        result = facet.process({'value': 'Hello World', 'search': 'Noodle'})
        assert result['result'] is False

    def test_regex_match(self):
        from noodlestudio.core.utility_facets import StringRegexFacet
        facet = StringRegexFacet("test_regex")
        result = facet.process({
            'value': 'The number is 42',
            'pattern': r'(\d+)'
        })
        assert result['result'] is True
        assert result['match'] == '42'
        assert '42' in result['groups']


class TestArrayFacets:
    """Test array utility facets."""

    def test_get_element(self):
        from noodlestudio.core.utility_facets import ArrayGetFacet
        facet = ArrayGetFacet("test_get")
        result = facet.process({'array': ['a', 'b', 'c'], 'index': 1})
        assert result['result'] == 'b'
        assert result['found'] is True

    def test_get_out_of_bounds(self):
        from noodlestudio.core.utility_facets import ArrayGetFacet
        facet = ArrayGetFacet("test_get")
        result = facet.process({'array': ['a', 'b', 'c'], 'index': 10})
        assert result['found'] is False

    def test_first(self):
        from noodlestudio.core.utility_facets import ArrayFirstFacet
        facet = ArrayFirstFacet("test_first")
        result = facet.process({'array': [1, 2, 3]})
        assert result['result'] == 1
        assert result['empty'] is False

    def test_last(self):
        from noodlestudio.core.utility_facets import ArrayLastFacet
        facet = ArrayLastFacet("test_last")
        result = facet.process({'array': [1, 2, 3]})
        assert result['result'] == 3

    def test_join(self):
        from noodlestudio.core.utility_facets import ArrayJoinFacet
        facet = ArrayJoinFacet("test_join", separator=', ')
        result = facet.process({'array': ['a', 'b', 'c']})
        assert result['result'] == 'a, b, c'

    def test_length(self):
        from noodlestudio.core.utility_facets import ArrayLengthFacet
        facet = ArrayLengthFacet("test_len")
        result = facet.process({'array': [1, 2, 3, 4, 5]})
        assert result['result'] == 5


class TestDataFacets:
    """Test data/control utility facets."""

    def test_pass_through(self):
        from noodlestudio.core.utility_facets import PassThroughFacet
        facet = PassThroughFacet("test_pass")
        result = facet.process({'in': 'hello'})
        assert result['out'] == 'hello'

    def test_gate_open(self):
        from noodlestudio.core.utility_facets import GateFacet
        facet = GateFacet("test_gate")
        result = facet.process({'value': 'data', 'gate': True})
        assert result['result'] == 'data'
        assert result['passed'] is True

    def test_gate_closed(self):
        from noodlestudio.core.utility_facets import GateFacet
        facet = GateFacet("test_gate")
        result = facet.process({'value': 'data', 'gate': False})
        assert result['result'] is None
        assert result['passed'] is False

    def test_counter(self):
        from noodlestudio.core.utility_facets import CounterFacet
        facet = CounterFacet("test_counter")
        result1 = facet.process({})
        assert result1['count'] == 1
        result2 = facet.process({})
        assert result2['count'] == 2
        result3 = facet.process({'reset': True})
        assert result3['count'] == 0

    def test_json_parse(self):
        from noodlestudio.core.utility_facets import JSONParseFacet
        facet = JSONParseFacet("test_parse")
        result = facet.process({'value': '{"name": "test", "count": 42}'})
        assert result['success'] is True
        assert result['result']['name'] == 'test'
        assert result['result']['count'] == 42

    def test_json_parse_invalid(self):
        from noodlestudio.core.utility_facets import JSONParseFacet
        facet = JSONParseFacet("test_parse")
        result = facet.process({'value': 'not json'})
        assert result['success'] is False
        assert 'error' in result

    def test_json_stringify(self):
        from noodlestudio.core.utility_facets import JSONStringifyFacet
        facet = JSONStringifyFacet("test_stringify")
        result = facet.process({'value': {'name': 'test', 'count': 42}})
        assert result['success'] is True
        parsed = json.loads(result['result'])
        assert parsed['name'] == 'test'

    def test_get_property(self):
        from noodlestudio.core.utility_facets import GetPropertyFacet
        facet = GetPropertyFacet("test_get_prop")
        result = facet.process({
            'object': {'name': 'Alice', 'age': 30},
            'key': 'name'
        })
        assert result['result'] == 'Alice'
        assert result['found'] is True

    def test_set_property(self):
        from noodlestudio.core.utility_facets import SetPropertyFacet
        facet = SetPropertyFacet("test_set_prop")
        result = facet.process({
            'object': {'name': 'Alice'},
            'key': 'age',
            'value': 30
        })
        assert result['result']['name'] == 'Alice'
        assert result['result']['age'] == 30


class TestUtilityFacetFactory:
    """Test factory function."""

    def test_create_utility_facet(self):
        from noodlestudio.core.utility_facets import create_utility_facet

        # Simple facet
        facet = create_utility_facet('MathAddFacet', 'test_id')
        assert facet is not None

        # Facet with config
        facet = create_utility_facet('LogicCompareFacet', 'test_id', {'operator': '>'})
        result = facet.process({'a': 10, 'b': 5})
        assert result['result'] is True

    def test_create_unknown_facet(self):
        from noodlestudio.core.utility_facets import create_utility_facet

        facet = create_utility_facet('UnknownFacet', 'test_id')
        assert facet is None


# =============================================================================
# MCP INTEGRATION TESTS
# =============================================================================

class TestMCPManager:
    """Test MCP Manager (without actual server connection)."""

    def test_manager_singleton(self):
        from noodlestudio.core.mcp_manager import MCPManager

        manager1 = MCPManager.instance()
        manager2 = MCPManager.instance()
        assert manager1 is manager2

    def test_env_expansion(self):
        from noodlestudio.core.mcp_manager import MCPManager
        import os

        manager = MCPManager.instance()

        # Test environment variable expansion
        os.environ['TEST_VAR'] = 'test_value'
        result = manager._expand_env('prefix_${TEST_VAR}_suffix')
        assert result == 'prefix_test_value_suffix'

        # Test non-existent variable (should remain unchanged)
        result = manager._expand_env('${NONEXISTENT_VAR}')
        assert result == '${NONEXISTENT_VAR}'

    def test_server_config(self):
        from noodlestudio.core.mcp_manager import MCPServerConfig, MCPServerType

        config = MCPServerConfig(
            name='test_server',
            type=MCPServerType.LOCAL,
            command='npx',
            args=['-y', '@modelcontextprotocol/server-filesystem']
        )

        assert config.name == 'test_server'
        assert config.type == MCPServerType.LOCAL


class TestMCPFacet:
    """Test MCP Facet (mocked, no actual server)."""

    def test_facet_init(self):
        from noodlestudio.core.mcp_facet import MCPFacet

        facet = MCPFacet('test_id', {
            'server': 'filesystem',
            'tool': 'read_file'
        })

        assert facet.server_name == 'filesystem'
        assert facet.tool_name == 'read_file'

    def test_get_input_schema_no_connection(self):
        from noodlestudio.core.mcp_facet import MCPFacet

        facet = MCPFacet('test_id', {
            'server': 'filesystem',
            'tool': 'read_file'
        })

        # Without connection, should return empty schema
        schema = facet.get_input_schema()
        assert schema['type'] == 'object'


# =============================================================================
# PLAYER RUNTIME TESTS
# =============================================================================

class TestPlayer:
    """Test headless Player runtime."""

    def test_player_init(self):
        from noodlestudio.player import Player, PlayerConfig

        config = PlayerConfig(
            llm_provider='ollama',
            verbose=True
        )
        player = Player(config)

        assert player.config.llm_provider == 'ollama'
        assert player.config.verbose is True

    def test_load_assembly_not_found(self):
        from noodlestudio.player import Player

        player = Player()
        result = player.load_assembly('/nonexistent/path.yaml')
        assert result is False

    def test_load_assembly_success(self):
        from noodlestudio.player import Player
        import tempfile
        import yaml

        # Create temporary assembly file (using correct YAML format)
        assembly_data = {
            'name': 'Test Assembly',
            'facets': [
                {
                    'id': 'incoming',
                    'name': 'INCOMING',
                    'type': 'SpecialNode',
                    'prompt': '',
                    'model': '',
                    'temperature': 0.7,
                    'max_tokens': 100,
                    'position': {'x': 0, 'y': 0},
                    'inputs': [],
                    'outputs': [{'name': 'out', 'type': 'output'}]
                },
                {
                    'id': 'outgoing',
                    'name': 'OUTGOING',
                    'type': 'SpecialNode',
                    'prompt': '',
                    'model': '',
                    'temperature': 0.7,
                    'max_tokens': 100,
                    'position': {'x': 100, 'y': 0},
                    'inputs': [{'name': 'in', 'type': 'input'}],
                    'outputs': [{'name': 'out', 'type': 'output'}]
                }
            ],
            'connections': [
                {
                    'from': 'incoming.out',
                    'to': 'outgoing.in'
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(assembly_data, f)
            temp_path = f.name

        player = Player()
        result = player.load_assembly(temp_path)
        assert result is True
        assert player.assembly is not None
        assert player.assembly.name == 'Test Assembly'

        # Cleanup
        import os
        os.unlink(temp_path)

    def test_run_without_assembly(self):
        """Test running Player without loading an assembly first."""
        import asyncio
        from noodlestudio.player import Player

        player = Player()
        result = asyncio.run(player.run("Hello"))

        assert result['error'] == 'No assembly loaded'


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_utility_facet_chain(self):
        """Test chaining utility facets together."""
        from noodlestudio.core.utility_facets import (
            StringSplitFacet, ArrayFirstFacet, StringLengthFacet
        )

        # Split "hello,world,test" -> get first -> get length
        split = StringSplitFacet("split", delimiter=',')
        first = ArrayFirstFacet("first")
        length = StringLengthFacet("length")

        split_result = split.process({'value': 'hello,world,test'})
        first_result = first.process({'array': split_result['result']})
        length_result = length.process({'value': first_result['result']})

        assert length_result['result'] == 5  # len("hello") == 5

    def test_json_round_trip(self):
        """Test JSON parse -> modify -> stringify."""
        from noodlestudio.core.utility_facets import (
            JSONParseFacet, SetPropertyFacet, JSONStringifyFacet
        )

        parse = JSONParseFacet("parse")
        set_prop = SetPropertyFacet("set")
        stringify = JSONStringifyFacet("stringify")

        # Parse JSON
        parsed = parse.process({'value': '{"name": "test"}'})

        # Add property
        modified = set_prop.process({
            'object': parsed['result'],
            'key': 'count',
            'value': 42
        })

        # Stringify back
        stringified = stringify.process({'value': modified['result']})

        # Verify round trip
        final = json.loads(stringified['result'])
        assert final['name'] == 'test'
        assert final['count'] == 42


# =============================================================================
# SCRIPTING API TESTS - Unity-style access
# =============================================================================

class TestFacetProxy:
    """Tests for FacetProxy Unity-style methods."""

    def test_facet_proxy_properties(self):
        """Test FacetProxy property access."""
        from noodlestudio.scripting.agents_api import FacetProxy
        from noodlestudio.core.facet_system import Facet, FacetPad, PadType

        facet = Facet(
            id="test_facet",
            name="Test Facet",
            facet_type="LLMFacet",
            prompt="Hello {input}",
            model="MEDIUM",
            temperature=0.8,
            max_tokens=100,
            position={'x': 100, 'y': 200}
        )
        facet.input_pads = [FacetPad(name="input", pad_type=PadType.INPUT, description="Input")]
        facet.output_pads = [FacetPad(name="out", pad_type=PadType.OUTPUT, description="Output")]

        proxy = FacetProxy(facet, None)

        assert proxy.get_id() == "test_facet"
        assert proxy.get_name() == "Test Facet"
        assert proxy.get_type() == "LLMFacet"
        assert proxy.get_prompt() == "Hello {input}"
        assert proxy.get_model() == "MEDIUM"
        assert proxy.get_temperature() == 0.8

    def test_facet_proxy_setters(self):
        """Test FacetProxy property setters."""
        from noodlestudio.scripting.agents_api import FacetProxy
        from noodlestudio.core.facet_system import Facet

        facet = Facet(
            id="test",
            name="Test",
            facet_type="LLMFacet",
            prompt="Original"
        )
        proxy = FacetProxy(facet, None)

        assert proxy.set_prompt("New prompt") is True
        assert proxy.get_prompt() == "New prompt"

        assert proxy.set_model("LARGE") is True
        assert proxy.get_model() == "LARGE"

        assert proxy.set_temperature(0.5) is True
        assert proxy.get_temperature() == 0.5

    def test_facet_proxy_enabled(self):
        """Test enable/disable facet."""
        from noodlestudio.scripting.agents_api import FacetProxy
        from noodlestudio.core.facet_system import Facet

        facet = Facet(id="test", name="Test", facet_type="LLMFacet", prompt="")
        proxy = FacetProxy(facet, None)

        assert proxy.is_enabled() is True  # Default enabled
        assert proxy.set_enabled(False) is True
        assert proxy.is_enabled() is False

    def test_facet_proxy_pads(self):
        """Test input/output pad access."""
        from noodlestudio.scripting.agents_api import FacetProxy
        from noodlestudio.core.facet_system import Facet, FacetPad, PadType

        facet = Facet(id="test", name="Test", facet_type="LLMFacet", prompt="")
        facet.input_pads = [
            FacetPad(name="a", pad_type=PadType.INPUT, description="Input A"),
            FacetPad(name="b", pad_type=PadType.INPUT, description="Input B")
        ]
        facet.output_pads = [
            FacetPad(name="out", pad_type=PadType.OUTPUT, description="Output")
        ]

        proxy = FacetProxy(facet, None)
        inputs = proxy.get_inputs()
        outputs = proxy.get_outputs()

        assert len(inputs) == 2
        assert inputs[0]['name'] == 'a'
        assert len(outputs) == 1
        assert outputs[0]['name'] == 'out'


class TestFacetAssemblyProxy:
    """Tests for FacetAssemblyProxy Unity-style methods."""

    def _create_test_assembly(self):
        """Create a test assembly with multiple facets."""
        from noodlestudio.core.facet_system import (
            FacetAssembly, Facet, FacetConnection, FacetPad, PadType
        )

        assembly = FacetAssembly(name="Test Assembly")

        # Add INCOMING
        incoming = Facet(
            id="incoming", name="INCOMING", facet_type="SpecialNode", prompt=""
        )
        incoming.output_pads = [FacetPad(name="out", pad_type=PadType.OUTPUT)]
        assembly.facets.append(incoming)

        # Add two LLM facets
        llm1 = Facet(
            id="llm1", name="Analyzer", facet_type="LLMFacet",
            prompt="Analyze", model="MEDIUM"
        )
        llm1.input_pads = [FacetPad(name="in", pad_type=PadType.INPUT)]
        llm1.output_pads = [FacetPad(name="out", pad_type=PadType.OUTPUT)]
        assembly.facets.append(llm1)

        llm2 = Facet(
            id="llm2", name="Generator", facet_type="LLMFacet",
            prompt="Generate", model="LARGE"
        )
        llm2.input_pads = [FacetPad(name="in", pad_type=PadType.INPUT)]
        llm2.output_pads = [FacetPad(name="out", pad_type=PadType.OUTPUT)]
        assembly.facets.append(llm2)

        # Add OUTGOING
        outgoing = Facet(
            id="outgoing", name="OUTGOING", facet_type="SpecialNode", prompt=""
        )
        outgoing.input_pads = [FacetPad(name="in", pad_type=PadType.INPUT)]
        assembly.facets.append(outgoing)

        # Add connections
        assembly.connections.append(FacetConnection("incoming", "out", "llm1", "in"))
        assembly.connections.append(FacetConnection("llm1", "out", "llm2", "in"))
        assembly.connections.append(FacetConnection("llm2", "out", "outgoing", "in"))

        return assembly

    def test_get_facets_by_type(self):
        """Test getting facets by type (like GetComponents<T>)."""
        from noodlestudio.scripting.agents_api import FacetAssemblyProxy

        assembly = self._create_test_assembly()
        proxy = FacetAssemblyProxy(assembly)

        llm_facets = proxy.get_facets_by_type("LLMFacet")
        assert len(llm_facets) == 2

        special_facets = proxy.get_facets_by_type("SpecialNode")
        assert len(special_facets) == 2

    def test_find_facets(self):
        """Test finding facets with predicate."""
        from noodlestudio.scripting.agents_api import FacetAssemblyProxy

        assembly = self._create_test_assembly()
        proxy = FacetAssemblyProxy(assembly)

        large_llms = proxy.find_facets({'type': 'LLMFacet', 'model': 'LARGE'})
        assert len(large_llms) == 1
        assert large_llms[0].get_name() == "Generator"

    def test_get_connections(self):
        """Test getting all connections."""
        from noodlestudio.scripting.agents_api import FacetAssemblyProxy

        assembly = self._create_test_assembly()
        proxy = FacetAssemblyProxy(assembly)

        conns = proxy.get_connections()
        assert len(conns) == 3

    def test_get_connections_from(self):
        """Test getting connections from a specific facet."""
        from noodlestudio.scripting.agents_api import FacetAssemblyProxy

        assembly = self._create_test_assembly()
        proxy = FacetAssemblyProxy(assembly)

        conns = proxy.get_connections_from("llm1")
        assert len(conns) == 1
        assert conns[0]['to_facet'] == "llm2"

    def test_get_connections_to(self):
        """Test getting connections to a specific facet."""
        from noodlestudio.scripting.agents_api import FacetAssemblyProxy

        assembly = self._create_test_assembly()
        proxy = FacetAssemblyProxy(assembly)

        conns = proxy.get_connections_to("llm2")
        assert len(conns) == 1
        assert conns[0]['from_facet'] == "llm1"

    def test_duplicate_facet(self):
        """Test duplicating a facet."""
        from noodlestudio.scripting.agents_api import FacetAssemblyProxy

        assembly = self._create_test_assembly()
        proxy = FacetAssemblyProxy(assembly)

        original_count = proxy.get_facet_count()
        clone_id = proxy.duplicate_facet("llm1", "Analyzer Clone")

        assert clone_id is not None
        assert proxy.get_facet_count() == original_count + 1

        clone = proxy.get_facet(clone_id)
        assert clone is not None
        assert clone.get_name() == "Analyzer Clone"
        assert clone.get_type() == "LLMFacet"

    def test_get_incoming_outgoing(self):
        """Test getting INCOMING and OUTGOING facets."""
        from noodlestudio.scripting.agents_api import FacetAssemblyProxy

        assembly = self._create_test_assembly()
        proxy = FacetAssemblyProxy(assembly)

        incoming = proxy.get_incoming()
        assert incoming is not None
        assert incoming.get_id() == "incoming"

        outgoing = proxy.get_outgoing()
        assert outgoing is not None
        assert outgoing.get_id() == "outgoing"


class TestNeuralNetworkProxy:
    """Tests for NeuralNetworkProxy Unity-style methods."""

    def test_list_nodes(self):
        """Test listing all nodes."""
        from noodlestudio.scripting.neural_api import NeuralAPI

        api = NeuralAPI()
        network = api.create_network("Test")

        # Create some nodes
        lstm_id = network.create_node("LSTM", position=[100, 100])
        gru_id = network.create_node("GRU", position=[200, 100])

        nodes = network.list_nodes()
        assert len(nodes) == 2

    def test_get_nodes_by_type(self):
        """Test getting nodes by type."""
        from noodlestudio.scripting.neural_api import NeuralAPI

        api = NeuralAPI()
        network = api.create_network("Test")

        # Create mixed nodes
        network.create_node("LSTM", position=[100, 100])
        network.create_node("LSTM", position=[200, 100])
        network.create_node("GRU", position=[300, 100])

        lstms = network.get_nodes_by_type("LSTM")
        assert len(lstms) == 2

        grus = network.get_nodes_by_type("GRU")
        assert len(grus) == 1

    def test_get_node_count(self):
        """Test node count."""
        from noodlestudio.scripting.neural_api import NeuralAPI

        api = NeuralAPI()
        network = api.create_network("Test")

        assert network.get_node_count() == 0
        network.create_node("LSTM", position=[100, 100])
        assert network.get_node_count() == 1
        network.create_node("GRU", position=[200, 100])
        assert network.get_node_count() == 2

    def test_set_node_name(self):
        """Test setting node name."""
        from noodlestudio.scripting.neural_api import NeuralAPI

        api = NeuralAPI()
        network = api.create_network("Test")

        node_id = network.create_node("LSTM", position=[100, 100])
        assert network.set_node_name(node_id, "My LSTM") is True

        node = network.get_node(node_id)
        assert node['name'] == "My LSTM"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
