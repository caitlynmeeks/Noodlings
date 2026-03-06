# ──────────────────────────────────────────────────────────────
#
#   Tests for Facet Output Viewer (Commit D)
#
#   Verifies that the inspector shows a "Last Output" section
#   for LLM facets, displaying the most recent facet execution output.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   tests.test_facet_output_viewer
# PURPOSE:  Verify Last Output viewer in inspector
# LAYER:    Tests
# ──────────────────────────────────────────────────────────────

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


class TestFacetOutputViewer:
    """Verify Last Output section in LLM facet inspector."""

    def _make_facet(self, **kwargs):
        from noodlestudio.core.facet_system import Facet
        defaults = {
            'id': 'test_facet',
            'name': 'Test Response',
            'facet_type': 'LLMFacet',
            'prompt': 'You are a helpful assistant.',
        }
        defaults.update(kwargs)
        return Facet(**defaults)

    def test_last_output_none_shows_placeholder(self):
        """When _last_output is None, display '(no output yet)'."""
        facet = self._make_facet()
        assert facet._last_output is None
        # The inspector reads this field directly; this test validates
        # the data model works correctly for the viewer
        assert facet.get_last_output() is None

    def test_last_output_after_execution(self):
        """After record_execution, _last_output contains the output dict."""
        facet = self._make_facet()
        outputs = {'out': 'Hello, I am a noodling!'}
        facet.record_execution(token_count=50, execution_time=0.5, outputs=outputs)

        assert facet._last_output is not None
        assert facet._last_output == outputs
        assert facet.get_last_output()['out'] == 'Hello, I am a noodling!'

    def test_last_output_updates_on_subsequent_execution(self):
        """Each execution overwrites _last_output."""
        facet = self._make_facet()

        facet.record_execution(42, 0.3, {'out': 'First response'})
        assert facet._last_output['out'] == 'First response'

        facet.record_execution(55, 0.4, {'out': 'Second response'})
        assert facet._last_output['out'] == 'Second response'

    def test_last_output_first_value_extraction(self):
        """Viewer extracts first value from output dict for display."""
        facet = self._make_facet()
        outputs = {'response': 'The ocean is deep', 'confidence': '0.95'}
        facet.record_execution(60, 0.5, outputs)

        # Simulate what the inspector does: get first value from dict
        last = facet._last_output
        display = next(iter(last.values()), '') if isinstance(last, dict) else str(last)
        assert display == 'The ocean is deep'
