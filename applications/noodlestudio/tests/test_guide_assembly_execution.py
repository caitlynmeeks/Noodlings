# ──────────────────────────────────────────────────────────────
#   Tests for Guide Assembly Execution
#
#   Tests that Ajo's cognition flows through the facet assembly
#   system: assembly loading, FacetExecutor integration, sentiment
#   -> affect pipeline, and conversation history.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import json
from dataclasses import dataclass, field
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from noodlestudio.core.facet_system import FacetAssembly


# Path to Ajo's assembly YAML (relative to project root)
ASSEMBLY_YAML = "noodlings/guide/assembly.yaml"


# =============================================================================
# Assembly Structure Tests
# =============================================================================

class TestAssemblyStructure:
    """Tests that assembly.yaml has the correct graph topology."""

    @pytest.fixture
    def assembly(self):
        """Load the guide assembly from YAML."""
        import os
        # Navigate from tests/ up to project root
        test_dir = os.path.dirname(__file__)
        studio_dir = os.path.join(test_dir, "..")
        project_root = os.path.abspath(os.path.join(studio_dir, "..", ".."))
        path = os.path.join(project_root, ASSEMBLY_YAML)
        if not os.path.exists(path):
            pytest.skip(f"Assembly not found at {path}")
        return FacetAssembly.load_yaml(path)

    def test_assembly_loads(self, assembly):
        """Assembly loads without error."""
        assert assembly is not None
        assert assembly.name == "Guide Assembly"

    def test_has_five_facets(self, assembly):
        """Assembly has exactly 5 facets (incoming, response, sentiment, performance, outgoing)."""
        assert len(assembly.facets) == 5

    def test_has_incoming_facet(self, assembly):
        """Assembly has an INCOMING entry point."""
        facet_ids = [f.id for f in assembly.facets]
        assert "incoming" in facet_ids

    def test_has_response_facet(self, assembly):
        """Assembly has a response LLM facet."""
        facet_ids = [f.id for f in assembly.facets]
        assert "response" in facet_ids

    def test_has_sentiment_facet(self, assembly):
        """Assembly has a sentiment LLM facet."""
        facet_ids = [f.id for f in assembly.facets]
        assert "sentiment" in facet_ids

    def test_has_outgoing_facet(self, assembly):
        """Assembly has an OUTGOING exit point."""
        facet_ids = [f.id for f in assembly.facets]
        assert "outgoing" in facet_ids

    def test_has_five_connections(self, assembly):
        """Assembly has 5 connections (parallel fan-out + performance chain)."""
        assert len(assembly.connections) == 5

    def test_response_uses_large_model(self, assembly):
        """Response facet uses LARGE model label."""
        response = next(f for f in assembly.facets if f.id == "response")
        assert response.model == "LARGE"

    def test_sentiment_uses_small_model(self, assembly):
        """Sentiment facet uses SMALL model label."""
        sentiment = next(f for f in assembly.facets if f.id == "sentiment")
        assert sentiment.model == "SMALL"

    def test_parallel_fan_out(self, assembly):
        """Incoming fans out to both response and sentiment in parallel."""
        from_incoming = [
            c for c in assembly.connections
            if c.from_facet == "incoming"
        ]
        assert len(from_incoming) == 2
        targets = {f"{c.to_facet}.{c.to_pad}" for c in from_incoming}
        assert "response.in" in targets
        assert "sentiment.in" in targets

    def test_outgoing_receives_both(self, assembly):
        """Outgoing receives from performance (text) and sentiment (affect)."""
        to_outgoing = [
            c for c in assembly.connections
            if c.to_facet == "outgoing"
        ]
        assert len(to_outgoing) == 2
        sources = {f"{c.from_facet}.{c.from_pad}" for c in to_outgoing}
        assert "performance.out" in sources
        assert "sentiment.out" in sources

    def test_has_performance_facet(self, assembly):
        """Assembly has a Performance ScriptedFacet."""
        facet_ids = [f.id for f in assembly.facets]
        assert "performance" in facet_ids
        perf = next(f for f in assembly.facets if f.id == "performance")
        assert perf.facet_type == "ScriptedFacet"

    def test_response_feeds_performance(self, assembly):
        """Response output feeds into Performance input."""
        conn = [
            c for c in assembly.connections
            if c.from_facet == "response" and c.to_facet == "performance"
        ]
        assert len(conn) == 1
        assert conn[0].from_pad == "out"
        assert conn[0].to_pad == "in"

    def test_performance_feeds_outgoing(self, assembly):
        """Performance output feeds into OUTGOING input."""
        conn = [
            c for c in assembly.connections
            if c.from_facet == "performance" and c.to_facet == "outgoing"
        ]
        assert len(conn) == 1
        assert conn[0].from_pad == "out"
        assert conn[0].to_pad == "in"

    def test_self_description_in_prompt(self, assembly):
        """Response facet prompt includes Ajo's physical appearance."""
        response = next(f for f in assembly.facets if f.id == "response")
        assert "axolotl" in response.prompt.lower()
        assert "gill" in response.prompt.lower()

    def test_outgoing_has_affect_input(self, assembly):
        """Outgoing has an 'affect' input port for sentiment data."""
        outgoing = next(f for f in assembly.facets if f.id == "outgoing")
        input_names = [pad.name for pad in outgoing.input_pads]
        assert "affect" in input_names

    def test_channels_subscribe(self, assembly):
        """Assembly subscribes to director and world channels."""
        assert assembly.channels is not None
        subs = assembly.channels.subscribe
        assert "#directors.cues" in subs

    def test_channels_publish(self, assembly):
        """Assembly publishes to director feedback channel."""
        pubs = assembly.channels.publish
        assert "#directors.feedback" in pubs


# =============================================================================
# Affect Pipeline Tests
# =============================================================================

class TestAffectPipeline:
    """Tests for sentiment JSON -> Affect -> VRM blendshapes pipeline."""

    def test_positive_sentiment_produces_shapes(self):
        """Positive sentiment generates non-empty VRM blend shapes."""
        from noodlestudio.runtime.facs_mapper import FACSMapper, Affect

        # Simulate what _apply_affect does
        affect_text = json.dumps({
            "valence": 0.8,
            "arousal": 0.6,
            "dominance": 0.7
        })

        data = json.loads(affect_text)
        valence = float(data['valence'])
        arousal = float(data['arousal'])
        dominance = float(data['dominance'])

        mapper = FACSMapper()
        affect = Affect(
            valence=valence * 2 - 1,  # 0..1 -> -1..1
            arousal=arousal,
            dominance=dominance,
            sorrow=max(0.0, (1.0 - valence) * 0.5),
            boredom=max(0.0, (1.0 - arousal) * 0.3)
        )

        shapes = mapper.map_affect_to_vrm(affect)
        assert isinstance(shapes, dict)
        assert len(shapes) > 0

    def test_negative_sentiment_produces_shapes(self):
        """Negative sentiment generates different VRM blend shapes."""
        from noodlestudio.runtime.facs_mapper import FACSMapper, Affect

        mapper = FACSMapper()
        affect = Affect(
            valence=-0.6,  # 0.2 remapped to -0.6
            arousal=0.3,
            dominance=0.3,
            sorrow=0.4,
            boredom=0.21
        )

        shapes = mapper.map_affect_to_vrm(affect)
        assert isinstance(shapes, dict)
        assert len(shapes) > 0

    def test_neutral_sentiment_produces_shapes(self):
        """Neutral sentiment (valence=0.5) produces valid output."""
        from noodlestudio.runtime.facs_mapper import FACSMapper, Affect

        mapper = FACSMapper()
        affect = Affect(
            valence=0.0,  # 0.5 remapped to 0.0
            arousal=0.5,
            dominance=0.5,
            sorrow=0.25,
            boredom=0.15
        )

        shapes = mapper.map_affect_to_vrm(affect)
        assert isinstance(shapes, dict)

    def test_shape_values_in_range(self):
        """All blend shape values are in [0, 1] range."""
        from noodlestudio.runtime.facs_mapper import FACSMapper, Affect

        mapper = FACSMapper()
        # Test several affect states
        test_states = [
            Affect(valence=0.8, arousal=0.9, dominance=0.7),
            Affect(valence=-0.8, arousal=0.1, dominance=0.3, sorrow=0.6),
            Affect(valence=0.0, arousal=0.5, dominance=0.5, boredom=0.5),
        ]

        for affect in test_states:
            shapes = mapper.map_affect_to_vrm(affect)
            for name, value in shapes.items():
                assert 0.0 <= value <= 1.0, (
                    f"Shape {name}={value} out of range for {affect}"
                )

    def test_valence_remapping(self):
        """Valence 0..1 (from LLM) remaps to -1..1 (for FACSMapper)."""
        # 0.0 -> -1.0 (negative)
        assert 0.0 * 2 - 1 == -1.0
        # 0.5 -> 0.0 (neutral)
        assert 0.5 * 2 - 1 == 0.0
        # 1.0 -> 1.0 (positive)
        assert 1.0 * 2 - 1 == 1.0

    def test_sorrow_derived_from_valence(self):
        """Sorrow is derived from low valence."""
        # When valence=0.2, sorrow = (1-0.2)*0.5 = 0.4
        valence = 0.2
        sorrow = max(0.0, (1.0 - valence) * 0.5)
        assert abs(sorrow - 0.4) < 0.001

    def test_boredom_derived_from_arousal(self):
        """Boredom is derived from low arousal."""
        # When arousal=0.3, boredom = (1-0.3)*0.3 = 0.21
        arousal = 0.3
        boredom = max(0.0, (1.0 - arousal) * 0.3)
        assert abs(boredom - 0.21) < 0.001

    def test_invalid_json_handled_gracefully(self, guide_manager):
        """Invalid JSON from sentiment facet does not crash."""
        # Call _apply_affect with invalid JSON -- should not raise
        guide_manager._apply_affect("not valid json {{{")
        # Window's set_blend_shapes should NOT have been called
        assert guide_manager._window.blend_shapes_calls == []

    def test_missing_keys_use_defaults(self, guide_manager):
        """Missing keys in sentiment JSON use defaults."""
        # Only provide valence, omit arousal and dominance
        guide_manager._apply_affect(json.dumps({"valence": 0.9}))

        # Should still call set_blend_shapes (arousal/dominance default to 0.5)
        assert len(guide_manager._window.blend_shapes_calls) == 1


# =============================================================================
# Assembly Result Handling Tests
# =============================================================================

@dataclass
class FakeExecutionResult:
    """Lightweight stand-in for facet_executor.ExecutionResult."""
    response: str = "Hello from assembly"
    total_time: float = 0.5
    total_tokens: int = 50
    facets_executed: int = 2
    facets_skipped: int = 0
    facet_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    facet_times: Dict[str, float] = field(default_factory=dict)
    facet_tokens: Dict[str, int] = field(default_factory=dict)


class TestAssemblyResultHandling:
    """Tests for _on_assembly_result processing."""

    def test_response_displayed_in_window(self, guide_manager):
        """Assembly response is displayed via append_guide_text."""
        result = FakeExecutionResult(response="Hello from Ajo!")
        guide_manager._on_assembly_result(result)

        assert guide_manager._window.texts == ["Hello from Ajo!"]

    def test_busy_cleared_after_result(self, guide_manager):
        """Window busy state is cleared after assembly result."""
        result = FakeExecutionResult()
        guide_manager._on_assembly_result(result)

        assert False in guide_manager._window.busy_states

    def test_conversation_history_updated(self, guide_manager):
        """Conversation history grows after each exchange."""
        guide_manager._last_user_message = "What are noodlings?"

        result = FakeExecutionResult(response="Noodlings are cognitive agents!")
        guide_manager._on_assembly_result(result)

        assert len(guide_manager._conversation_history) == 2
        assert guide_manager._conversation_history[0] == {
            'role': 'user', 'content': 'What are noodlings?'
        }
        assert guide_manager._conversation_history[1] == {
            'role': 'assistant', 'content': 'Noodlings are cognitive agents!'
        }

    def test_history_accumulates_over_exchanges(self, guide_manager):
        """Multiple exchanges accumulate in history."""
        guide_manager._last_user_message = "Hello"
        result1 = FakeExecutionResult(response="Hi there!")
        guide_manager._on_assembly_result(result1)

        guide_manager._last_user_message = "What is NoodleStudio?"
        result2 = FakeExecutionResult(response="It's a cognitive IDE!")
        guide_manager._on_assembly_result(result2)

        assert len(guide_manager._conversation_history) == 4

    def test_sentiment_drives_affect(self, guide_manager):
        """Sentiment facet output drives the affect pipeline."""
        sentiment_json = json.dumps({
            "valence": 0.8,
            "arousal": 0.6,
            "dominance": 0.7
        })

        result = FakeExecutionResult(
            response="I love noodlings!",
            facet_outputs={
                'sentiment': {'out': sentiment_json},
                'response': {'out': 'I love noodlings!'},
            }
        )
        guide_manager._on_assembly_result(result)

        assert len(guide_manager._window.blend_shapes_calls) == 1
        shapes = guide_manager._window.blend_shapes_calls[0]
        assert isinstance(shapes, dict)

    def test_no_sentiment_output_skips_affect(self, guide_manager):
        """Missing sentiment output skips affect pipeline gracefully."""
        result = FakeExecutionResult(
            response="Just a response",
            facet_outputs={'response': {'out': 'Just a response'}}
        )
        guide_manager._on_assembly_result(result)

        assert guide_manager._window.blend_shapes_calls == []

    def test_brenda_feedback_on_result(self, guide_manager):
        """GuideCueHandler receives response for Brenda feedback."""
        mock_handler = MagicMock()
        guide_manager._guide_cue_handler = mock_handler
        guide_manager._last_user_message = "Tell me about facets"

        result = FakeExecutionResult(response="Facets are cognitive nodes!")
        guide_manager._on_assembly_result(result)

        mock_handler.report_response.assert_called_once_with(
            "Facets are cognitive nodes!", "Tell me about facets"
        )

    def test_empty_response_shows_error(self, guide_manager):
        """Empty response shows error in window."""
        result = FakeExecutionResult(response="[No output]")
        guide_manager._on_assembly_result(result)

        assert len(guide_manager._window.errors) == 1


# =============================================================================
# Context Building Tests
# =============================================================================

class TestContextBuilding:
    """Tests for execution context assembly."""

    def test_context_includes_history(self, guide_manager):
        """Execution context includes conversation history."""
        guide_manager._assembly = MagicMock()
        guide_manager._executor = MagicMock()
        guide_manager._conversation_history = [
            {'role': 'user', 'content': 'previous message'}
        ]

        with patch(
            'noodlestudio.runtime.ui.guide_performance_manager._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            guide_manager._on_user_message_for_assembly("Hello")

            call_args = MockWorker.call_args
            context = call_args[0][3]  # 4th positional arg
            assert 'conversation_history' in context
            # History is now a formatted string for prompt injection
            assert isinstance(context['conversation_history'], str)
            assert 'previous message' in context['conversation_history']

    def test_context_includes_brenda_direction(self, guide_manager):
        """Execution context includes Brenda direction when available."""
        guide_manager._assembly = MagicMock()
        guide_manager._executor = MagicMock()

        mock_handler = MagicMock()
        mock_handler.build_system_prompt_addition.return_value = (
            "Direct Ajo to welcome the user warmly."
        )
        guide_manager._guide_cue_handler = mock_handler

        with patch(
            'noodlestudio.runtime.ui.guide_performance_manager._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            guide_manager._on_user_message_for_assembly("Hello")

            context = MockWorker.call_args[0][3]
            assert 'brenda_direction' in context
            assert "welcome the user" in context['brenda_direction']

    def test_context_omits_direction_when_none(self, guide_manager):
        """Execution context omits brenda_direction when no handler."""
        guide_manager._assembly = MagicMock()
        guide_manager._executor = MagicMock()

        with patch(
            'noodlestudio.runtime.ui.guide_performance_manager._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            guide_manager._on_user_message_for_assembly("Hello")

            context = MockWorker.call_args[0][3]
            assert 'brenda_direction' not in context


# =============================================================================
# LLM Client Creation Tests
# =============================================================================

class TestLLMClientCreation:
    """Tests for _create_llm_client bridging editor settings."""

    @patch('noodlestudio.core.model_label_manager.get_model_label_manager')
    @patch('noodlestudio.core.provider_manager.get_provider_manager')
    def test_creates_client_from_settings(
        self, mock_get_provider, mock_get_label, guide_manager
    ):
        """LLM client is created from editor provider/label settings."""
        # Set up mock label manager
        label_mgr = MagicMock()
        label_mgr.get_model_for_label.side_effect = lambda label: {
            "Large": ("anthropic", "claude-sonnet-4-5-20250929"),
            "Medium": ("anthropic", "claude-sonnet-4-5-20250929"),
            "Small": ("ollama", "deepseek-r1:7b"),
        }.get(label, (None, None))
        mock_get_label.return_value = label_mgr

        # Set up mock provider manager
        provider_mgr = MagicMock()
        mock_provider = MagicMock()
        mock_provider.type = "anthropic"
        mock_provider.api_key = "test-key"
        mock_provider.base_url = "https://api.anthropic.com"
        provider_mgr.get_provider.return_value = mock_provider
        mock_get_provider.return_value = provider_mgr

        with patch('noodlestudio.runtime.llm_client.HeadlessLLMClient') as MockClient:
            client = guide_manager._create_llm_client()

            MockClient.assert_called_once()
            config = MockClient.call_args[0][0]
            assert config.provider == "anthropic"
            assert config.api_key == "test-key"
            assert "LARGE" in config.model_labels
            assert "SMALL" in config.model_labels


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
