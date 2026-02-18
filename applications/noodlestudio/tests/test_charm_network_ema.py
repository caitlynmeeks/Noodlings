# ------------------------------------------------------------------
#   Charm Network EMA Tests
#
#   Verifies: EMA update dynamics, multi-turn accumulation, baseline
#   drift, PAD clamping, and assembly integration.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_charm_network_ema
# PURPOSE:  Charm Network EMA Tests (D.1.5g)
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


def _make_ema(valence=0.0, arousal=0.5, dominance=0.5):
    from noodlestudio.runtime.charm_network_ema import CharmNetworkEMA
    return CharmNetworkEMA({
        'valence': valence, 'arousal': arousal, 'dominance': dominance
    })


class TestEMAUpdate:
    """Single update must move fast layer significantly, slow barely."""

    def test_fast_layer_responds_strongly(self):
        """A single high-arousal input must move the fast layer significantly."""
        ema = _make_ema(arousal=0.5)
        ema.update({'valence': 0.0, 'arousal': 1.0, 'dominance': 0.5})

        # Fast alpha=0.7 -> fast.arousal = 0.5 * 0.3 + 1.0 * 0.7 = 0.85
        assert ema.fast['arousal'] == pytest.approx(0.85, abs=0.01)

    def test_slow_layer_barely_moves(self):
        """A single input must barely move the slow layer."""
        ema = _make_ema(arousal=0.5)
        ema.update({'valence': 0.0, 'arousal': 1.0, 'dominance': 0.5})

        # Slow alpha=0.03 -> slow.arousal = 0.5 * 0.97 + 1.0 * 0.03 = 0.515
        assert ema.slow['arousal'] == pytest.approx(0.515, abs=0.01)

    def test_medium_layer_moderate_response(self):
        """Medium layer must respond moderately."""
        ema = _make_ema(arousal=0.5)
        ema.update({'valence': 0.0, 'arousal': 1.0, 'dominance': 0.5})

        # Medium alpha=0.15 -> medium.arousal = 0.5 * 0.85 + 1.0 * 0.15 = 0.575
        assert ema.medium['arousal'] == pytest.approx(0.575, abs=0.01)

    def test_blended_output_between_layers(self):
        """Blended output must be between fast and slow values."""
        ema = _make_ema(arousal=0.5)
        result = ema.update({'valence': 0.0, 'arousal': 1.0, 'dominance': 0.5})

        assert ema.slow['arousal'] < result['arousal'] < ema.fast['arousal']

    def test_all_three_dimensions_updated(self):
        """All PAD dimensions must be updated."""
        ema = _make_ema()
        result = ema.update({'valence': 0.8, 'arousal': 0.9, 'dominance': 0.7})

        assert 'valence' in result
        assert 'arousal' in result
        assert 'dominance' in result
        assert result['valence'] > 0.0  # Moved from baseline 0
        assert result['arousal'] > 0.5  # Moved from baseline 0.5

    def test_negative_valence(self):
        """Negative valence input must drive output negative."""
        ema = _make_ema(valence=0.0)
        result = ema.update({'valence': -0.8, 'arousal': 0.5, 'dominance': 0.5})

        assert result['valence'] < 0.0


class TestMultiTurnAccumulation:
    """Multiple high-arousal inputs must meaningfully drift the slow layer."""

    def test_slow_layer_drifts_after_10_turns(self):
        """10 high-arousal inputs must move the slow layer meaningfully."""
        ema = _make_ema(arousal=0.5)

        for _ in range(10):
            ema.update({'valence': 0.0, 'arousal': 1.0, 'dominance': 0.5})

        # After 10 turns: slow should have drifted noticeably from 0.5
        assert ema.slow['arousal'] > 0.6

    def test_fast_layer_near_target_after_many_turns(self):
        """Fast layer must converge near the sustained input."""
        ema = _make_ema(arousal=0.5)

        for _ in range(20):
            ema.update({'valence': 0.0, 'arousal': 1.0, 'dominance': 0.5})

        # Fast alpha=0.7 converges very quickly
        assert ema.fast['arousal'] > 0.99

    def test_output_rises_steadily(self):
        """Blended output must rise with each turn under sustained input."""
        ema = _make_ema(arousal=0.3)

        outputs = []
        for _ in range(5):
            result = ema.update({'valence': 0.0, 'arousal': 0.9, 'dominance': 0.5})
            outputs.append(result['arousal'])

        # Each output should be >= previous (monotonically increasing)
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i - 1]


class TestBaselineDrift:
    """Neutral inputs must cause state to trend back toward baseline."""

    def test_drift_toward_baseline_from_extreme(self):
        """After high input, drift_toward_baseline must pull back."""
        ema = _make_ema(arousal=0.5)

        # Push to extreme
        for _ in range(10):
            ema.update({'valence': 0.0, 'arousal': 1.0, 'dominance': 0.5})

        high_fast = ema.fast['arousal']

        # Drift back
        for _ in range(10):
            ema.drift_toward_baseline()

        assert ema.fast['arousal'] < high_fast

    def test_drift_converges_to_baseline(self):
        """Many drift steps must converge near baseline."""
        ema = _make_ema(arousal=0.5)

        # Push far from baseline
        for _ in range(20):
            ema.update({'valence': 0.0, 'arousal': 1.0, 'dominance': 0.5})

        # Drift many times
        for _ in range(100):
            ema.drift_toward_baseline(rate=0.1)

        # All layers should be near baseline
        assert ema.fast['arousal'] == pytest.approx(0.5, abs=0.05)
        assert ema.medium['arousal'] == pytest.approx(0.5, abs=0.05)
        assert ema.slow['arousal'] == pytest.approx(0.5, abs=0.1)

    def test_drift_does_not_overshoot(self):
        """Drift must not push past the baseline."""
        ema = _make_ema(arousal=0.5)

        # Push high
        ema.update({'valence': 0.0, 'arousal': 1.0, 'dominance': 0.5})

        # Drift
        for _ in range(50):
            ema.drift_toward_baseline()

        # Should be between high and baseline, not below baseline
        assert ema.fast['arousal'] >= 0.49  # baseline 0.5, allow tiny float err


class TestPADClamping:
    """Output must stay in valid PAD ranges."""

    def test_valence_clamped_to_minus_one_one(self):
        """Valence must stay in [-1, 1]."""
        ema = _make_ema(valence=0.9)
        result = ema.update({'valence': 2.0, 'arousal': 0.5, 'dominance': 0.5})
        assert -1.0 <= result['valence'] <= 1.0

    def test_negative_valence_clamped(self):
        """Extreme negative valence must be clamped to -1."""
        ema = _make_ema(valence=-0.9)
        result = ema.update({'valence': -5.0, 'arousal': 0.5, 'dominance': 0.5})
        assert result['valence'] >= -1.0

    def test_arousal_clamped_to_zero_one(self):
        """Arousal must stay in [0, 1]."""
        ema = _make_ema(arousal=0.1)
        result = ema.update({'valence': 0.0, 'arousal': -1.0, 'dominance': 0.5})
        assert 0.0 <= result['arousal'] <= 1.0

    def test_dominance_clamped_to_zero_one(self):
        """Dominance must stay in [0, 1]."""
        ema = _make_ema(dominance=0.9)
        result = ema.update({'valence': 0.0, 'arousal': 0.5, 'dominance': 3.0})
        assert 0.0 <= result['dominance'] <= 1.0


class TestGetState:
    """get_state() must return complete inspector-ready state."""

    def test_state_contains_all_layers(self):
        ema = _make_ema()
        state = ema.get_state()
        assert 'fast' in state
        assert 'medium' in state
        assert 'slow' in state
        assert 'output' in state
        assert 'baseline' in state

    def test_state_output_matches_blend(self):
        ema = _make_ema()
        ema.update({'valence': 0.5, 'arousal': 0.8, 'dominance': 0.6})
        state = ema.get_state()

        # Output should match what _blend returns
        blended = ema._blend()
        assert state['output']['valence'] == pytest.approx(blended['valence'], abs=0.001)
        assert state['output']['arousal'] == pytest.approx(blended['arousal'], abs=0.001)

    def test_state_baseline_is_original(self):
        ema = _make_ema(valence=0.3, arousal=0.4, dominance=0.6)
        ema.update({'valence': 0.9, 'arousal': 0.9, 'dominance': 0.9})
        state = ema.get_state()

        assert state['baseline']['valence'] == pytest.approx(0.3)
        assert state['baseline']['arousal'] == pytest.approx(0.4)
        assert state['baseline']['dominance'] == pytest.approx(0.6)


class TestCharacterBaselines:
    """Character-specific baselines from assembly.yaml must be correct."""

    def test_ajo_baseline(self):
        from noodlestudio.runtime.charm_network_ema import CharmNetworkEMA
        ema = CharmNetworkEMA({'valence': 0.7, 'arousal': 0.5, 'dominance': 0.4})
        state = ema.get_state()
        assert state['baseline']['valence'] == pytest.approx(0.7)
        assert state['baseline']['arousal'] == pytest.approx(0.5)
        assert state['baseline']['dominance'] == pytest.approx(0.4)

    def test_krampus_baseline(self):
        from noodlestudio.runtime.charm_network_ema import CharmNetworkEMA
        ema = CharmNetworkEMA({'valence': 0.5, 'arousal': 0.7, 'dominance': 0.3})
        state = ema.get_state()
        assert state['baseline']['arousal'] == pytest.approx(0.7)

    def test_juanita_baseline(self):
        from noodlestudio.runtime.charm_network_ema import CharmNetworkEMA
        ema = CharmNetworkEMA({'valence': 0.6, 'arousal': 0.3, 'dominance': 0.6})
        state = ema.get_state()
        assert state['baseline']['dominance'] == pytest.approx(0.6)


class TestAssemblyIntegration:
    """Assembly with CharmNetworkEMA facet must parse and wire correctly."""

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

    def test_all_assemblies_have_charm_facet(self, assembly_paths):
        """All assemblies must contain a CharmNetworkEMA facet."""
        for nid, path in assembly_paths.items():
            with open(path) as f:
                data = yaml.safe_load(f)
            charm = [f for f in data['facets'] if f['type'] == 'CharmNetworkEMA']
            assert len(charm) == 1, f"{nid} missing CharmNetworkEMA facet"

    def test_charm_wired_between_sentiment_and_outgoing(self, assembly_paths):
        """Charm must receive from sentiment.out and send to outgoing.affect."""
        for nid, path in assembly_paths.items():
            with open(path) as f:
                data = yaml.safe_load(f)
            connections = data['connections']
            from_conns = {(c['from'], c['to']) for c in connections}

            assert ('sentiment.out', 'charm.in') in from_conns, \
                f"{nid} missing sentiment.out -> charm.in"
            assert ('charm.out', 'outgoing.affect') in from_conns, \
                f"{nid} missing charm.out -> outgoing.affect"
            # Old direct sentiment->outgoing should NOT exist
            assert ('sentiment.out', 'outgoing.affect') not in from_conns, \
                f"{nid} still has direct sentiment.out -> outgoing.affect"

    def test_charm_facet_has_baseline_in_prompt(self, assembly_paths):
        """Charm facet prompt must contain baseline PAD values."""
        for nid, path in assembly_paths.items():
            with open(path) as f:
                data = yaml.safe_load(f)
            charm = next(f for f in data['facets'] if f['type'] == 'CharmNetworkEMA')
            assert 'valence:' in charm['prompt']
            assert 'arousal:' in charm['prompt']
            assert 'dominance:' in charm['prompt']

    def test_mood_reader_uses_negative_one_to_one_valence(self, assembly_paths):
        """All Mood Reader prompts must specify -1.0 to +1.0 valence range."""
        for nid, path in assembly_paths.items():
            with open(path) as f:
                data = yaml.safe_load(f)
            sentiment = next(f for f in data['facets'] if f['id'] == 'sentiment')
            assert '-1.0' in sentiment['prompt'], \
                f"{nid} Mood Reader still uses 0..1 valence"

    def test_assemblies_parse_with_charm_facet(self, assembly_paths):
        """All assemblies must parse successfully with CharmNetworkEMA."""
        from noodlestudio.core.facet_system import FacetAssembly
        for nid, path in assembly_paths.items():
            assembly = FacetAssembly.load_yaml(path)
            assert assembly is not None
            facet_ids = {f.id for f in assembly.facets}
            assert 'charm' in facet_ids, f"{nid} missing charm facet after parse"

    def test_facet_executor_dispatches_ema(self):
        """FacetExecutor must create CharmNetworkEMA singleton for EMA facet."""
        from noodlestudio.core.facet_executor import FacetExecutor
        from noodlestudio.core.facet_system import Facet

        executor = FacetExecutor(llm_client=None, use_event_bus=False)

        facet = Facet(
            id='charm_test',
            name='Charm Network',
            facet_type='CharmNetworkEMA',
            prompt='valence:0.5,arousal:0.5,dominance:0.5',
        )

        instance = executor._get_facet_instance(facet, {})
        from noodlestudio.runtime.charm_network_ema import CharmNetworkEMA
        assert isinstance(instance, CharmNetworkEMA)

    def test_facet_executor_ema_singleton(self):
        """Same facet ID must return the same EMA instance (singleton)."""
        from noodlestudio.core.facet_executor import FacetExecutor
        from noodlestudio.core.facet_system import Facet

        executor = FacetExecutor(llm_client=None, use_event_bus=False)

        facet = Facet(
            id='charm_test',
            name='Charm Network',
            facet_type='CharmNetworkEMA',
            prompt='valence:0.5,arousal:0.5,dominance:0.5',
        )

        instance1 = executor._get_facet_instance(facet, {})
        instance2 = executor._get_facet_instance(facet, {})
        assert instance1 is instance2

    def test_facet_executor_parses_baseline(self):
        """FacetExecutor must parse baseline from facet prompt."""
        from noodlestudio.core.facet_executor import FacetExecutor
        from noodlestudio.core.facet_system import Facet

        executor = FacetExecutor(llm_client=None, use_event_bus=False)

        facet = Facet(
            id='charm_parse',
            name='Charm Network',
            facet_type='CharmNetworkEMA',
            prompt='valence:0.7,arousal:0.3,dominance:0.6',
        )

        instance = executor._get_facet_instance(facet, {})
        assert instance.baseline['valence'] == pytest.approx(0.7)
        assert instance.baseline['arousal'] == pytest.approx(0.3)
        assert instance.baseline['dominance'] == pytest.approx(0.6)
