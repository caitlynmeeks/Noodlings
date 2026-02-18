# ------------------------------------------------------------------
#   Ensemble Awareness Tests
#
#   Verifies: perception context building, present_entities formatting,
#   affect cross-pollination, template variable injection in assembly
#   prompts, and solo mode fallback.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_ensemble_awareness
# PURPOSE:  Ensemble Awareness Tests (D.1.5c + D.1.5d)
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

LIBRARY_DIR = os.path.join(os.path.dirname(__file__), '..', 'library')


def _make_ensemble_manager():
    """Build a manager with 3 performers for testing."""
    from noodlestudio.runtime.ui.guide_performance_manager import (
        GuidePerformanceManager,
    )
    from conftest import StubMainWindow, StubWindow, FakeLLMClient
    from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

    manager = GuidePerformanceManager(StubMainWindow())
    manager._ensemble_mode = True
    manager._window = StubWindow()

    # Create performers with fake LLM clients
    ajo = NoodlingPerformer(
        noodling_id='ajo', name='Ajo Majo',
        llm_client=FakeLLMClient()
    )
    krampus = NoodlingPerformer(
        noodling_id='krampus', name='Krampus',
        llm_client=FakeLLMClient()
    )
    juanita = NoodlingPerformer(
        noodling_id='juanita', name='Juanita',
        llm_client=FakeLLMClient()
    )

    manager._performers = {
        'ajo': ajo, 'krampus': krampus, 'juanita': juanita
    }
    manager._performer = ajo

    # Populate instance metadata (as _discover_stage_instances would)
    manager._instance_metadata = {
        'ajo': {
            'noodling_id': 'ajo',
            'name': 'Ajo Majo',
            'description': 'A curious axolotl guide',
            'appearance': 'A small chibi axolotl with pink-lavender coloring and cute nubby external gills',
            'affect_baseline': {'valence': 0.4, 'arousal': 0.4, 'dominance': 0.3},
        },
        'krampus': {
            'noodling_id': 'krampus',
            'name': 'Krampus',
            'description': 'A seven-year-old Alpine enforcer',
            'appearance': 'A seven-year-old boy with tiny horns and an oversized wreath',
            'affect_baseline': {'valence': 0.4, 'arousal': 0.6, 'dominance': 0.3},
        },
        'juanita': {
            'noodling_id': 'juanita',
            'name': 'Juanita',
            'description': 'A curious explorer from Lanzarote',
            'appearance': 'A girl with a backpack of interesting things',
            'affect_baseline': {'valence': 0.5, 'arousal': 0.3, 'dominance': 0.4},
        },
    }

    manager._stage_description = "The Nexus -- a shared space where noodlings meet"

    return manager


class TestPerceptionContext:
    """_advance_ensemble_turn must build rich extra_context for each noodling."""

    def test_extra_context_contains_required_keys(self):
        """Extra context must contain stage_context, present_entities,
        ensemble_history, and conversation_history."""
        manager = _make_ensemble_manager()

        # Seed ensemble history
        manager._ensemble_history = [
            {'role': 'User', 'content': 'Hello!'},
        ]

        # Set up for Ajo's turn
        manager._turn_queue = ['ajo']
        manager._pending_message = "What do you see?"
        manager._turn_responses = {}

        # Capture what execute() receives by patching it
        captured_context = {}
        original_execute = manager._performers['ajo'].execute
        def capturing_execute(msg, ctx=None):
            captured_context.update(ctx or {})
        manager._performers['ajo'].execute = capturing_execute

        manager._advance_ensemble_turn()

        assert 'stage_context' in captured_context
        assert 'present_entities' in captured_context
        assert 'ensemble_history' in captured_context
        assert 'conversation_history' in captured_context

    def test_stage_context_populated(self):
        """stage_context must contain the stage description."""
        manager = _make_ensemble_manager()
        manager._turn_queue = ['ajo']
        manager._pending_message = "Hello"
        manager._turn_responses = {}

        captured = {}
        manager._performers['ajo'].execute = lambda msg, ctx=None: captured.update(ctx or {})
        manager._advance_ensemble_turn()

        assert 'Nexus' in captured['stage_context']

    def test_present_entities_excludes_current_noodling(self):
        """present_entities for Ajo must NOT mention Ajo, but include others."""
        manager = _make_ensemble_manager()
        manager._turn_queue = ['ajo']
        manager._pending_message = "Hello"
        manager._turn_responses = {}

        captured = {}
        manager._performers['ajo'].execute = lambda msg, ctx=None: captured.update(ctx or {})
        manager._advance_ensemble_turn()

        entities = captured['present_entities']
        assert 'Krampus' in entities
        assert 'Juanita' in entities
        assert 'Ajo Majo' not in entities

    def test_present_entities_includes_appearance(self):
        """present_entities must include the appearance text."""
        manager = _make_ensemble_manager()
        manager._turn_queue = ['ajo']
        manager._pending_message = "Hello"
        manager._turn_responses = {}

        captured = {}
        manager._performers['ajo'].execute = lambda msg, ctx=None: captured.update(ctx or {})
        manager._advance_ensemble_turn()

        entities = captured['present_entities']
        assert 'horns' in entities  # Krampus appearance
        assert 'backpack' in entities  # Juanita appearance

    def test_ensemble_history_in_context(self):
        """ensemble_history must contain formatted conversation transcript."""
        manager = _make_ensemble_manager()
        manager._ensemble_history = [
            {'role': 'User', 'content': 'Hello!'},
            {'role': 'Ajo Majo', 'content': 'Hi there!'},
        ]
        manager._turn_queue = ['krampus']
        manager._pending_message = "What do you think?"
        manager._turn_responses = {}

        captured = {}
        manager._performers['krampus'].execute = lambda msg, ctx=None: captured.update(ctx or {})
        manager._advance_ensemble_turn()

        assert 'User: Hello!' in captured['ensemble_history']
        assert 'Ajo Majo: Hi there!' in captured['ensemble_history']

    def test_conversation_history_equals_ensemble_history(self):
        """conversation_history must be populated with ensemble history
        for backward compatibility with solo-mode prompts."""
        manager = _make_ensemble_manager()
        manager._ensemble_history = [
            {'role': 'User', 'content': 'Test'},
        ]
        manager._turn_queue = ['ajo']
        manager._pending_message = "Hello"
        manager._turn_responses = {}

        captured = {}
        manager._performers['ajo'].execute = lambda msg, ctx=None: captured.update(ctx or {})
        manager._advance_ensemble_turn()

        assert captured['conversation_history'] == captured['ensemble_history']


class TestMoodCrossPollination:
    """Other noodlings' mood state must appear in perception context."""

    def test_mood_appears_in_context_after_affect(self):
        """After Ajo's turn sets affect, Krampus should see Ajo's mood."""
        manager = _make_ensemble_manager()

        # Simulate Ajo having responded with a mood (-1..1 valence)
        manager._performers['ajo']._last_pad_values = {
            'valence': 0.6, 'arousal': 0.7, 'dominance': 0.5
        }

        manager._turn_queue = ['krampus']
        manager._pending_message = "Hello"
        manager._turn_responses = {'ajo': 'Hi there!'}

        captured = {}
        manager._performers['krampus'].execute = lambda msg, ctx=None: captured.update(ctx or {})
        manager._advance_ensemble_turn()

        assert 'ajo_mood' in captured
        assert isinstance(captured['ajo_mood'], str)
        assert len(captured['ajo_mood']) > 0

    def test_mood_not_included_for_self(self):
        """A noodling's own mood must NOT be in its context."""
        manager = _make_ensemble_manager()
        manager._performers['ajo']._last_pad_values = {
            'valence': 0.6, 'arousal': 0.7, 'dominance': 0.5
        }

        manager._turn_queue = ['ajo']
        manager._pending_message = "Hello"
        manager._turn_responses = {}

        captured = {}
        manager._performers['ajo'].execute = lambda msg, ctx=None: captured.update(ctx or {})
        manager._advance_ensemble_turn()

        assert 'ajo_mood' not in captured

    def test_mood_in_present_entities_description(self):
        """present_entities must include mood phrasing when affect is available."""
        manager = _make_ensemble_manager()
        manager._performers['krampus']._last_pad_values = {
            'valence': 0.6, 'arousal': 0.8, 'dominance': 0.5
        }

        manager._turn_queue = ['ajo']
        manager._pending_message = "Hello"
        manager._turn_responses = {}

        captured = {}
        manager._performers['ajo'].execute = lambda msg, ctx=None: captured.update(ctx or {})
        manager._advance_ensemble_turn()

        entities = captured['present_entities']
        # Krampus should have a mood descriptor
        assert 'Currently seems' in entities

    def test_no_mood_for_noodling_without_affect(self):
        """Noodlings without stored affect must not have mood in context."""
        manager = _make_ensemble_manager()
        # No affect set on any performer

        manager._turn_queue = ['ajo']
        manager._pending_message = "Hello"
        manager._turn_responses = {}

        captured = {}
        manager._performers['ajo'].execute = lambda msg, ctx=None: captured.update(ctx or {})
        manager._advance_ensemble_turn()

        assert 'krampus_mood' not in captured
        assert 'juanita_mood' not in captured


class TestFormatPresentEntities:
    """_format_present_entities must build correct prose."""

    def test_excludes_specified_noodling(self):
        """Must exclude the noodling specified by exclude_nid."""
        manager = _make_ensemble_manager()
        result = manager._format_present_entities('ajo')
        assert 'Ajo Majo' not in result
        assert 'Krampus' in result
        assert 'Juanita' in result

    def test_all_excluded_returns_empty(self):
        """If only one noodling exists and it's excluded, return empty."""
        manager = _make_ensemble_manager()
        manager._instance_metadata = {
            'ajo': manager._instance_metadata['ajo']
        }
        result = manager._format_present_entities('ajo')
        assert result == ""

    def test_starts_with_header(self):
        """Non-empty result must start with 'Also here with you:'."""
        manager = _make_ensemble_manager()
        result = manager._format_present_entities('ajo')
        assert result.startswith("Also here with you:")


class TestDescribeAffect:
    """_describe_affect must produce natural language from PAD values."""

    def test_happy_energetic(self):
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        result = GuidePerformanceManager._describe_affect(
            {'valence': 0.7, 'arousal': 0.8, 'dominance': 0.5}
        )
        assert 'happy' in result
        assert 'energetic' in result

    def test_unhappy_quiet(self):
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        result = GuidePerformanceManager._describe_affect(
            {'valence': -0.7, 'arousal': 0.1, 'dominance': 0.5}
        )
        assert 'unhappy' in result
        assert 'quiet' in result

    def test_high_dominance_adds_confident(self):
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        result = GuidePerformanceManager._describe_affect(
            {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.9}
        )
        assert 'confident' in result

    def test_low_dominance_adds_uncertain(self):
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        result = GuidePerformanceManager._describe_affect(
            {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.1}
        )
        assert 'uncertain' in result


class TestAssemblyPromptAwareness:
    """Assembly prompts must contain awareness template variables."""

    @pytest.fixture
    def assembly_paths(self):
        base = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started', 'Noodlings'
        )
        return {
            'ajo': os.path.join(base, 'ajo_majo', 'assembly.yaml'),
            'krampus': os.path.join(base, 'krampus', 'assembly.yaml'),
            'juanita': os.path.join(base, 'juanita', 'assembly.yaml'),
        }

    def test_all_assemblies_have_stage_context_var(self, assembly_paths):
        """All Response prompts must contain {stage_context}."""
        for nid, path in assembly_paths.items():
            with open(path) as f:
                data = yaml.safe_load(f)
            response = next(f for f in data['facets'] if f['id'] == 'response')
            assert '{stage_context}' in response['prompt'], \
                f"{nid} Response prompt missing {{stage_context}}"

    def test_all_assemblies_have_present_entities_var(self, assembly_paths):
        """All Response prompts must contain {present_entities}."""
        for nid, path in assembly_paths.items():
            with open(path) as f:
                data = yaml.safe_load(f)
            response = next(f for f in data['facets'] if f['id'] == 'response')
            assert '{present_entities}' in response['prompt'], \
                f"{nid} Response prompt missing {{present_entities}}"

    def test_all_assemblies_have_ensemble_history_var(self, assembly_paths):
        """All Response prompts must contain {ensemble_history}."""
        for nid, path in assembly_paths.items():
            with open(path) as f:
                data = yaml.safe_load(f)
            response = next(f for f in data['facets'] if f['id'] == 'response')
            assert '{ensemble_history}' in response['prompt'], \
                f"{nid} Response prompt missing {{ensemble_history}}"

    def test_all_assemblies_retain_conversation_history(self, assembly_paths):
        """All Response prompts must still contain {conversation_history}
        for solo mode backward compatibility."""
        for nid, path in assembly_paths.items():
            with open(path) as f:
                data = yaml.safe_load(f)
            response = next(f for f in data['facets'] if f['id'] == 'response')
            assert '{conversation_history}' in response['prompt'], \
                f"{nid} Response prompt missing {{conversation_history}}"

    def test_assemblies_still_parse(self, assembly_paths):
        """All assemblies must still be valid YAML after edits."""
        from noodlestudio.core.facet_system import FacetAssembly
        for nid, path in assembly_paths.items():
            assembly = FacetAssembly.load_yaml(path)
            assert assembly is not None
            assert len(assembly.facets) >= 4  # incoming, response, sentiment, performance, outgoing


class TestSoloModeFallback:
    """Solo mode (non-ensemble) must still work with conversation_history."""

    def test_solo_mode_uses_conversation_history(self):
        """In solo mode, execute() builds context with conversation_history,
        and the template falls back gracefully when ensemble vars are absent."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from conftest import StubMainWindow, StubWindow, FakeLLMClient
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        manager = GuidePerformanceManager(StubMainWindow())
        manager._ensemble_mode = False
        manager._window = StubWindow()

        ajo = NoodlingPerformer(
            noodling_id='ajo', name='Ajo Majo',
            llm_client=FakeLLMClient()
        )
        manager._performer = ajo

        # In solo mode, _on_user_message does NOT inject ensemble vars
        # The performer's execute() builds its own conversation_history
        # FacetExecutor catches KeyError for {ensemble_history} etc. and
        # falls back to the unformatted prompt -- not ideal but functional
        captured = {}
        original_execute = ajo.execute
        def capturing_execute(msg, ctx=None):
            captured.update(ctx or {})
        ajo.execute = capturing_execute

        manager._on_user_message("test")

        # Solo mode should NOT have ensemble-specific keys
        assert 'ensemble_history' not in captured
        assert 'present_entities' not in captured


class TestSpeakerSpotlight:
    """set_active_speaker must be called during turn-taking for spotlight."""

    def test_speaker_set_on_each_turn(self):
        """Each noodling's turn must set them as active speaker."""
        manager = _make_ensemble_manager()
        manager._turn_queue = ['ajo', 'krampus']
        manager._pending_message = "Hello"
        manager._turn_responses = {}

        # Capture execute calls instead of actually executing
        for nid in manager._performers:
            manager._performers[nid].execute = lambda msg, ctx=None: None

        manager._advance_ensemble_turn()  # Ajo's turn

        calls = manager._window._active_speaker_calls
        assert calls[-1] == 'ajo'

    def test_speaker_cleared_when_all_turns_done(self):
        """After all turns complete, speaker must be set to None."""
        manager = _make_ensemble_manager()
        manager._turn_queue = []  # All turns done
        manager._pending_message = "Hello"

        manager._advance_ensemble_turn()

        calls = manager._window._active_speaker_calls
        assert calls[-1] is None

    def test_speaker_changes_between_noodlings(self):
        """Active speaker must change as turns advance."""
        manager = _make_ensemble_manager()
        manager._turn_queue = ['ajo', 'krampus']
        manager._pending_message = "Hello"
        manager._turn_responses = {}

        for nid in manager._performers:
            manager._performers[nid].execute = lambda msg, ctx=None: None

        # First turn: Ajo
        manager._advance_ensemble_turn()
        assert manager._window._active_speaker_calls[-1] == 'ajo'

        # Simulate Ajo finishing, advance to Krampus
        manager._performers['ajo']._last_response = "Hi!"
        manager._on_ensemble_turn_finished('ajo')

        # Krampus turn queued, but turn_queue was modified
        assert manager._window._active_speaker_calls[-1] == 'krampus'
