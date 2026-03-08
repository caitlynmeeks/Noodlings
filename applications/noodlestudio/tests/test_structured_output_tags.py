# ------------------------------------------------------------------
#   Structured Output Tag Tests
#
#   Verifies: JavaScript tag parser behavior (via assembly YAML
#   inspection), Python streaming tag detection in PerformancePlayer,
#   NoodlingPerformer _last_actions/_last_thoughts storage, and
#   formatChanged signal emission.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_structured_output_tags
# PURPOSE:  Commit 2 -- Tag parsing + streaming detection
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

NOODLING_BASE = os.path.join(
    os.path.dirname(__file__), '..',
    'library', 'templates', 'Getting Started', 'Noodlings'
)


# =============================================================================
# JavaScript parser structure tests (static YAML inspection)
# =============================================================================

class TestJavaScriptParserPresence:
    """Performance facet JS must contain all required parser functions."""

    @pytest.fixture
    def assembly_paths(self):
        base = NOODLING_BASE
        return {
            'ajo': os.path.join(base, 'ajo_majo', 'assembly.yaml'),
            'krampus': os.path.join(base, 'krampus', 'assembly.yaml'),
            'juanita': os.path.join(base, 'juanita', 'assembly.yaml'),
        }

    def _get_perf_js(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        perf = next(fct for fct in data['facets'] if fct['id'] == 'performance')
        return perf['prompt']

    def test_all_have_parse_structured_output(self, assembly_paths):
        for nid, path in assembly_paths.items():
            js = self._get_perf_js(path)
            assert 'parseStructuredOutput' in js, f"{nid}: missing parseStructuredOutput"

    def test_all_have_get_thoughts(self, assembly_paths):
        for nid, path in assembly_paths.items():
            js = self._get_perf_js(path)
            assert 'getThoughts' in js, f"{nid}: missing getThoughts"

    def test_all_have_get_actions(self, assembly_paths):
        for nid, path in assembly_paths.items():
            js = self._get_perf_js(path)
            assert 'getActions' in js, f"{nid}: missing getActions"

    def test_all_have_get_visible_text(self, assembly_paths):
        for nid, path in assembly_paths.items():
            js = self._get_perf_js(path)
            assert 'getVisibleText' in js, f"{nid}: missing getVisibleText"

    def test_all_output_has_thoughts_and_actions_fields(self, assembly_paths):
        """The returned JSON must include thoughts and actions arrays."""
        for nid, path in assembly_paths.items():
            js = self._get_perf_js(path)
            assert 'thoughts: thoughts' in js, f"{nid}: missing thoughts field"
            assert 'actions: actions' in js, f"{nid}: missing actions field"

    def test_thought_type_excluded_from_display(self, assembly_paths):
        """THOUGHT segments must be skipped (continue) in char loop."""
        for nid, path in assembly_paths.items():
            js = self._get_perf_js(path)
            assert "if (seg.type === 'thought') continue" in js, \
                f"{nid}: thought exclusion missing"

    def test_fmt_field_on_chars(self, assembly_paths):
        """Each char entry must include a fmt field for type."""
        for nid, path in assembly_paths.items():
            js = self._get_perf_js(path)
            assert 'fmt: seg.type' in js, f"{nid}: fmt field missing"


# =============================================================================
# Python streaming tag detection tests
# =============================================================================

class TestStreamingTagDetection:
    """PerformancePlayer.append_text() must detect and strip SPOKEN/ACTION/THOUGHT."""

    @pytest.fixture
    def player(self, qapp):
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        p = PerformancePlayer()
        p.start_streaming()
        return p

    def test_untagged_line_defaults_to_spoken(self, player):
        """A plain line with no tag is treated as SPOKEN."""
        formats_seen = []
        player.formatChanged.connect(formats_seen.append)

        # Use a long string; first char is consumed by _reveal_next_streaming
        # synchronously, so check for a suffix that survives
        player.append_text("Hello there from Ajo\n")

        # No format change emitted (already spoken)
        assert formats_seen == []
        # After streaming starts, first char consumed; rest still in buffer
        assert 'ello there from Ajo' in player._stream_buffer

    def test_spoken_tag_stripped_from_buffer(self, player):
        """SPOKEN: prefix is stripped before adding to stream buffer."""
        player.append_text('SPOKEN: "Hello world from Ajo!"\n')
        # First char consumed; check a suffix that is unambiguously content
        assert 'ello world from Ajo' in player._stream_buffer
        assert 'SPOKEN:' not in player._stream_buffer

    def test_action_tag_detected_and_stripped(self, player):
        """ACTION: prefix is stripped and formatChanged emitted."""
        formats_seen = []
        player.formatChanged.connect(formats_seen.append)

        player.append_text("ACTION: *tilts head curiously*\n")

        assert 'action' in formats_seen
        # First char consumed; check a suffix
        assert 'ilts head curiously' in player._stream_buffer
        assert 'ACTION:' not in player._stream_buffer

    def test_thought_line_not_added_to_buffer(self, player):
        """THOUGHT: lines are discarded -- not added to stream buffer."""
        player.append_text("THOUGHT: He suspects something.\n")
        # THOUGHT content must never appear in the stream buffer
        assert 'He suspects something' not in player._stream_buffer

    def test_format_change_emits_signal(self, player):
        """formatChanged emits when type changes between lines."""
        seen = []
        player.formatChanged.connect(seen.append)

        player.append_text("Hello\n")           # spoken (no change)
        player.append_text("ACTION: *waves*\n") # action -> change emitted
        player.append_text("SPOKEN: Yes\n")     # back to spoken -> change emitted

        assert 'action' in seen
        assert 'spoken' in seen

    def test_finish_streaming_flushes_partial_line(self, player):
        """finish_streaming() flushes any partial line without trailing \\n."""
        # Add a short line first to trigger streaming start, clearing state
        player.append_text("Hi\n")
        # Now add a partial line (no \n) - it stays in _line_buffer
        player.append_text("SPOKEN: Hello world partial")
        assert 'Hello world partial' not in player._stream_buffer  # in line buffer

        player.finish_streaming()
        # After flush, content must be in stream buffer
        assert 'ello world partial' in player._stream_buffer

    def test_format_reset_on_start_streaming(self, qapp):
        """start_streaming() resets _current_format to 'spoken'."""
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()
        player.start_streaming()
        player.append_text("ACTION: *waves*\n")
        assert player._current_format == 'action'

        player.start_streaming()
        assert player._current_format == 'spoken'


# =============================================================================
# NoodlingPerformer structured output storage
# =============================================================================

class TestPerformerStructuredOutputStorage:
    """NoodlingPerformer stores _last_actions from performance_script."""

    @pytest.fixture
    def performer(self, qapp):
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from conftest import FakeLLMClient
        return NoodlingPerformer(
            noodling_id='test', name='Test',
            llm_client=FakeLLMClient()
        )

    def test_last_actions_initially_empty(self, performer):
        assert performer._last_actions == []

    def test_last_thoughts_initially_empty(self, performer):
        assert performer._last_thoughts == []

    def test_performer_has_format_changed_signal(self, performer):
        """NoodlingPerformer must expose a formatChanged signal."""
        assert hasattr(performer, 'formatChanged')

    def test_assembly_result_stores_actions(self, performer, qapp):
        """_on_assembly_result stores actions from performance_script."""
        import json

        script = {
            'type': 'performance_script',
            'text': 'Hello',
            'characters': [],
            'speaking_intensity': 0.7,
            'thoughts': ['He wonders'],
            'actions': ['*waves*', '*smiles*'],
        }

        class FakeResult:
            facet_outputs = {
                'response': {'out': 'Hello'},
                'outgoing': {'out': json.dumps(script)},
            }
            response = json.dumps(script)
            total_time = 0.0
            total_tokens = 0

        performer._last_user_message = 'test'
        performer._on_assembly_result(FakeResult())

        assert performer._last_actions == ['*waves*', '*smiles*']
        assert performer._last_thoughts == ['He wonders']

    def test_assembly_result_uses_visible_text_for_response(self, performer, qapp):
        """_last_response uses visible text (no THOUGHT content) from script."""
        import json

        script = {
            'type': 'performance_script',
            'text': 'Hello there',   # visible text (no thoughts)
            'characters': [],
            'speaking_intensity': 0.7,
            'thoughts': ['private thought'],
            'actions': [],
        }

        class FakeResult:
            facet_outputs = {
                'response': {'out': ''},
                'outgoing': {'out': json.dumps(script)},
            }
            response = json.dumps(script)
            total_time = 0.0
            total_tokens = 0

        performer._last_user_message = 'test'
        performer._on_assembly_result(FakeResult())

        assert performer._last_response == 'Hello there'
        assert 'private thought' not in performer._last_response
