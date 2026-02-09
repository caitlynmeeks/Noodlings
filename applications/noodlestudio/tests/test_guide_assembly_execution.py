# ------------------------------------------------------------------
#   Tests for Guide Assembly Execution
#
#   Tests that Ajo's cognition flows through the facet assembly
#   system: assembly loading, FacetExecutor integration, sentiment
#   -> affect pipeline, and conversation history.
#
#   Cognition logic lives in NoodlingPerformer. Tests exercise the
#   performer directly for affect, result handling, and context
#   building. Manager-level tests verify wiring (Brenda feedback).
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ------------------------------------------------------------------

import json
from dataclasses import dataclass, field
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import pytest

from noodlestudio.core.facet_system import FacetAssembly
from conftest import FakeLLMClient, SignalCollector, StubFacetsEditor, StubMainWindow


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

    def test_invalid_json_handled_gracefully(self, performer):
        """Invalid JSON from sentiment facet does not crash."""
        collector = SignalCollector()
        performer.affectReady.connect(collector)

        # Call _apply_affect with invalid JSON -- should not raise
        performer._apply_affect("not valid json {{{")

        # affectReady should NOT have been emitted
        assert collector.values == []

    def test_missing_keys_use_defaults(self, performer):
        """Missing keys in sentiment JSON use defaults."""
        collector = SignalCollector()
        performer.affectReady.connect(collector)

        # Only provide valence, omit arousal and dominance
        performer._apply_affect(json.dumps({"valence": 0.9}))

        # Should still emit affectReady (arousal/dominance default to 0.5)
        assert len(collector.values) == 1
        shapes = collector.values[0]
        assert isinstance(shapes, dict)
        assert len(shapes) > 0


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
    """Tests for NoodlingPerformer._on_assembly_result processing."""

    def test_response_emits_signal(self, performer):
        """Assembly response emits responseReady signal."""
        collector = SignalCollector()
        performer.responseReady.connect(collector)

        result = FakeExecutionResult(response="Hello from Ajo!")
        performer._on_assembly_result(result)

        assert collector.values == ["Hello from Ajo!"]

    def test_execution_finished_after_result(self, performer):
        """executionFinished signal fires after plain text result."""
        collector = SignalCollector()
        performer.executionFinished.connect(collector)

        result = FakeExecutionResult(response="Hello!")
        performer._on_assembly_result(result)

        assert len(collector.values) == 1

    def test_conversation_history_updated(self, performer):
        """Conversation history grows after each exchange."""
        performer._last_user_message = "What are noodlings?"

        result = FakeExecutionResult(response="Noodlings are cognitive agents!")
        performer._on_assembly_result(result)

        assert len(performer.conversation_history) == 2
        assert performer.conversation_history[0] == {
            'role': 'user', 'content': 'What are noodlings?'
        }
        assert performer.conversation_history[1] == {
            'role': 'assistant', 'content': 'Noodlings are cognitive agents!'
        }

    def test_history_accumulates_over_exchanges(self, performer):
        """Multiple exchanges accumulate in history."""
        performer._last_user_message = "Hello"
        result1 = FakeExecutionResult(response="Hi there!")
        performer._on_assembly_result(result1)

        performer._last_user_message = "What is NoodleStudio?"
        result2 = FakeExecutionResult(response="It's a cognitive IDE!")
        performer._on_assembly_result(result2)

        assert len(performer.conversation_history) == 4

    def test_sentiment_drives_affect(self, performer):
        """Sentiment facet output drives the affect pipeline."""
        collector = SignalCollector()
        performer.affectReady.connect(collector)

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
        performer._on_assembly_result(result)

        assert len(collector.values) == 1
        shapes = collector.values[0]
        assert isinstance(shapes, dict)

    def test_no_sentiment_output_skips_affect(self, performer):
        """Missing sentiment output skips affect pipeline gracefully."""
        collector = SignalCollector()
        performer.affectReady.connect(collector)

        result = FakeExecutionResult(
            response="Just a response",
            facet_outputs={'response': {'out': 'Just a response'}}
        )
        performer._on_assembly_result(result)

        assert collector.values == []

    def test_brenda_feedback_on_result(self, guide_manager):
        """GuideCueHandler receives response via manager's execution finish."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        # Set up a performer with last response data
        p = NoodlingPerformer('ajo', 'Ajo', FakeLLMClient())
        p._last_response = "Facets are cognitive nodes!"
        p._last_user_message = "Tell me about facets"
        guide_manager._performer = p

        # Stub cue handler that records calls
        class StubCueHandler:
            def __init__(self):
                self.reported = []

            def report_response(self, response, user_msg):
                self.reported.append((response, user_msg))

        handler = StubCueHandler()
        guide_manager._guide_cue_handler = handler

        # Trigger the execution finished handler
        guide_manager._on_execution_finished()

        assert len(handler.reported) == 1
        assert handler.reported[0] == (
            "Facets are cognitive nodes!", "Tell me about facets"
        )

    def test_empty_response_shows_error(self, performer):
        """Empty response emits errorOccurred signal."""
        collector = SignalCollector()
        performer.errorOccurred.connect(collector)

        result = FakeExecutionResult(response="[No output]")
        performer._on_assembly_result(result)

        assert len(collector.values) == 1


# =============================================================================
# Context Building Tests
# =============================================================================

class TestContextBuilding:
    """Tests for execution context assembly (now in NoodlingPerformer)."""

    def test_context_includes_history(self, performer):
        """Execution context includes conversation history."""
        performer._assembly = True   # Truthy sentinel
        performer._executor = True
        performer._conversation_history = [
            {'role': 'user', 'content': 'previous message'}
        ]

        with patch(
            'noodlestudio.runtime.ui.noodling_performer._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            performer.execute("Hello")

            call_args = MockWorker.call_args
            context = call_args[0][3]  # 4th positional arg
            assert 'conversation_history' in context
            # History is now a formatted string for prompt injection
            assert isinstance(context['conversation_history'], str)
            assert 'previous message' in context['conversation_history']

    def test_context_includes_brenda_direction(self, guide_manager):
        """Manager passes Brenda direction as extra_context to performer."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        p = NoodlingPerformer('ajo', 'Ajo', FakeLLMClient())
        p._assembly = True
        p._executor = True
        guide_manager._performer = p

        class StubCueHandler:
            def build_system_prompt_addition(self):
                return "Direct Ajo to welcome the user warmly."

        guide_manager._guide_cue_handler = StubCueHandler()

        with patch(
            'noodlestudio.runtime.ui.noodling_performer._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            guide_manager._on_user_message("Hello")

            context = MockWorker.call_args[0][3]
            assert 'brenda_direction' in context
            assert "welcome the user" in context['brenda_direction']

    def test_context_omits_direction_when_none(self, guide_manager):
        """Execution context omits brenda_direction when no handler."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        p = NoodlingPerformer('ajo', 'Ajo', FakeLLMClient())
        p._assembly = True
        p._executor = True
        guide_manager._performer = p
        guide_manager._guide_cue_handler = None

        with patch(
            'noodlestudio.runtime.ui.noodling_performer._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            guide_manager._on_user_message("Hello")

            context = MockWorker.call_args[0][3]
            assert 'brenda_direction' not in context


# =============================================================================
# LLM Client Creation Tests
# =============================================================================

class TestLLMClientCreation:
    """Tests for create_llm_client utility function."""

    @patch('noodlestudio.core.model_label_manager.get_model_label_manager')
    @patch('noodlestudio.core.provider_manager.get_provider_manager')
    def test_creates_client_from_settings(
        self, mock_get_provider, mock_get_label
    ):
        """LLM client is created from editor provider/label settings."""
        from noodlestudio.runtime.ui.noodling_performer import create_llm_client

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
            client = create_llm_client()

            MockClient.assert_called_once()
            config = MockClient.call_args[0][0]
            assert config.provider == "anthropic"
            assert config.api_key == "test-key"
            assert "LARGE" in config.model_labels
            assert "SMALL" in config.model_labels


# Made with love. Use with love.
# Caitlyn Meeks 2026
