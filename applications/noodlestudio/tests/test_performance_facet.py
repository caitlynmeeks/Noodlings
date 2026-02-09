# ──────────────────────────────────────────────────────────────
#   Tests for Performance ScriptedFacet
#
#   Tests the JavaScript process() function from the Performance
#   facet in Ajo's assembly. Verifies JSON output format,
#   punctuation delays, and speaking intensity.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import json
import os
import pytest
import yaml
from unittest.mock import MagicMock


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def performance_script():
    """Load the Performance facet's JavaScript from assembly.yaml."""
    test_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_dir, "..", "..", ".."))
    assembly_path = os.path.join(project_root, "noodlings", "guide", "assembly.yaml")

    if not os.path.exists(assembly_path):
        pytest.skip(f"Assembly not found at {assembly_path}")

    with open(assembly_path) as f:
        data = yaml.safe_load(f)

    for facet in data['facets']:
        if facet['id'] == 'performance':
            return facet['prompt']

    pytest.skip("Performance facet not found in assembly")


@pytest.fixture
def scripted_facet(performance_script):
    """Create a ScriptedFacet instance with the Performance script."""
    try:
        from noodlestudio.core.scripted_facet import ScriptedFacet
    except ImportError:
        pytest.skip("ScriptedFacet not available")

    return ScriptedFacet("performance_test", performance_script)


def _execute_performance(scripted_facet, text):
    """Execute the Performance script and parse the JSON output."""
    from noodlestudio.core.scripted_facet import ScriptContext

    # Use a mock noodle_api to avoid AffectAPI.to_dict() crash in tests
    mock_api = MagicMock()
    mock_api.to_dict.return_value = {}

    context = ScriptContext(
        cycle=1, timestamp=0,
        agent_id='test', agent_name='test', agent_species='test',
        _noodle_api=mock_api
    )
    outputs = scripted_facet.process({'in': text}, context)

    assert 'out' in outputs
    result = json.loads(outputs['out'])
    return result


# =============================================================================
# Output Format Tests
# =============================================================================

class TestPerformanceFacetOutput:
    """Tests for the Performance facet's JSON output format."""

    def test_output_is_valid_json(self, scripted_facet):
        """Performance facet produces valid JSON output."""
        result = _execute_performance(scripted_facet, "Hello world")
        assert isinstance(result, dict)

    def test_output_has_type_field(self, scripted_facet):
        """Output has type='performance_script'."""
        result = _execute_performance(scripted_facet, "Test")
        assert result['type'] == 'performance_script'

    def test_output_preserves_text(self, scripted_facet):
        """Output text field matches the input text."""
        text = "Hello, this is a test."
        result = _execute_performance(scripted_facet, text)
        assert result['text'] == text

    def test_output_has_characters_array(self, scripted_facet):
        """Output has characters array with one entry per character."""
        text = "Hello"
        result = _execute_performance(scripted_facet, text)
        assert len(result['characters']) == len(text)

    def test_output_has_speaking_intensity(self, scripted_facet):
        """Output has speaking_intensity field."""
        result = _execute_performance(scripted_facet, "X")
        assert 'speaking_intensity' in result
        assert isinstance(result['speaking_intensity'], (int, float))
        assert result['speaking_intensity'] > 0

    def test_characters_have_c_and_d_fields(self, scripted_facet):
        """Each character entry has 'c' (char) and 'd' (delay) fields."""
        result = _execute_performance(scripted_facet, "AB")
        for entry in result['characters']:
            assert 'c' in entry
            assert 'd' in entry


# =============================================================================
# Punctuation Delay Tests
# =============================================================================

class TestPunctuationDelays:
    """Tests for punctuation pause durations."""

    def _get_delay_for_char(self, scripted_facet, text, char_index):
        """Get the delay for a specific character index."""
        result = _execute_performance(scripted_facet, text)
        return result['characters'][char_index]['d']

    def test_period_has_longer_delay(self, scripted_facet):
        """Period gets a longer delay than regular characters."""
        regular_delay = self._get_delay_for_char(scripted_facet, "abc", 0)
        period_delay = self._get_delay_for_char(scripted_facet, "a.", 1)
        assert period_delay > regular_delay

    def test_comma_has_medium_delay(self, scripted_facet):
        """Comma gets a medium delay (longer than regular, shorter than period)."""
        regular_delay = self._get_delay_for_char(scripted_facet, "abc", 0)
        comma_delay = self._get_delay_for_char(scripted_facet, "a,b", 1)
        assert comma_delay > regular_delay

    def test_exclamation_has_longer_delay(self, scripted_facet):
        """Exclamation mark gets a longer delay."""
        regular_delay = self._get_delay_for_char(scripted_facet, "abc", 0)
        excl_delay = self._get_delay_for_char(scripted_facet, "a!", 1)
        assert excl_delay > regular_delay

    def test_question_has_longer_delay(self, scripted_facet):
        """Question mark gets a longer delay."""
        regular_delay = self._get_delay_for_char(scripted_facet, "abc", 0)
        q_delay = self._get_delay_for_char(scripted_facet, "a?", 1)
        assert q_delay > regular_delay

    def test_space_has_shorter_delay(self, scripted_facet):
        """Space gets a shorter delay than regular characters."""
        regular_delay = self._get_delay_for_char(scripted_facet, "abc", 0)
        space_delay = self._get_delay_for_char(scripted_facet, "a b", 1)
        assert space_delay < regular_delay


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestPerformanceFacetEdgeCases:
    """Edge cases for the Performance facet."""

    def test_empty_input(self, scripted_facet):
        """Empty input produces valid output with empty characters."""
        result = _execute_performance(scripted_facet, "")
        assert result['type'] == 'performance_script'
        assert result['text'] == ''
        assert result['characters'] == []

    def test_newline_has_pause(self, scripted_facet):
        """Newlines get a pause delay."""
        regular_delay = 35  # base_delay from script
        result = _execute_performance(scripted_facet, "a\nb")
        newline_entry = result['characters'][1]
        assert newline_entry['c'] == '\n'
        assert newline_entry['d'] > regular_delay

    def test_long_text(self, scripted_facet):
        """Long text produces correct number of characters."""
        text = "a" * 500
        result = _execute_performance(scripted_facet, text)
        assert len(result['characters']) == 500

    def test_output_roundtrips_json(self, scripted_facet):
        """Output survives JSON serialization/deserialization."""
        result = _execute_performance(scripted_facet, "Test, please.")
        json_str = json.dumps(result)
        restored = json.loads(json_str)
        assert restored['text'] == result['text']
        assert len(restored['characters']) == len(result['characters'])
