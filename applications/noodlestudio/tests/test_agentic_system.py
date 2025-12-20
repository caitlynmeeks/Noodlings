"""
Test Suite for Agentic Facet System

Tests for:
1. Utility Facets - All 31 types
2. MCP Integration - Manager and Facet
3. Player Runtime - Headless execution

Run with: pytest tests/test_agentic_system.py -v

Author: Caitlyn + Claude
Date: December 20, 2025
"""

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
