# ------------------------------------------------------------------
#
#   Charm Network EMA
#
#   Multi-timescale affect via exponential moving averages.
#   Three EMA tracks (fast, medium, slow) process incoming PAD
#   values and produce a blended output that drifts naturally
#   between turns.
#
#   This is the MVP implementation of charm networks. The full
#   LSTM+GRU architecture (NoodlingModelPhase4) drops in as a
#   replacement when training is complete -- same inputs, same
#   outputs, same position in the assembly.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.runtime.charm_network_ema
# PURPOSE:  Multi-Timescale Affect EMA
# LAYER:    Studio / Runtime
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# PAD dimension keys
_PAD_KEYS = ('valence', 'arousal', 'dominance')

# EMA alpha values per timescale
_ALPHA_FAST = 0.7     # Reacts in 1-2 turns
_ALPHA_MEDIUM = 0.15  # Smooths over ~10 turns
_ALPHA_SLOW = 0.03    # Session-level drift

# Blend weights for combined output
_WEIGHT_FAST = 0.5
_WEIGHT_MEDIUM = 0.3
_WEIGHT_SLOW = 0.2

# Valid ranges for clamping
_VALENCE_MIN, _VALENCE_MAX = -1.0, 1.0
_UNIT_MIN, _UNIT_MAX = 0.0, 1.0


def _ema(current: float, target: float, alpha: float) -> float:
    """Single-step exponential moving average."""
    return current * (1.0 - alpha) + target * alpha


def _clamp_pad(pad: Dict) -> Dict:
    """Clamp PAD values to valid ranges.

    Valence: [-1, 1], arousal: [0, 1], dominance: [0, 1].
    """
    return {
        'valence': max(_VALENCE_MIN, min(_VALENCE_MAX,
                                         pad.get('valence', 0.0))),
        'arousal': max(_UNIT_MIN, min(_UNIT_MAX,
                                      pad.get('arousal', 0.5))),
        'dominance': max(_UNIT_MIN, min(_UNIT_MAX,
                                        pad.get('dominance', 0.5))),
    }


class CharmNetworkEMA:
    """Multi-timescale affect via exponential moving averages.

    Implements Varela's temporal nesting principle: faster processes
    nest within slower processes. The fast track gives immediate
    reactions, the medium track gives conversational mood, the slow
    track gives dispositional state.

    Args:
        baseline_pad: Character-specific baseline PAD values.
            Dict with ``valence`` (-1..1), ``arousal`` (0..1),
            ``dominance`` (0..1).
    """

    def __init__(self, baseline_pad: Dict):
        self.baseline = _clamp_pad(baseline_pad)
        self.fast = dict(self.baseline)
        self.medium = dict(self.baseline)
        self.slow = dict(self.baseline)

    def update(self, mood_input: Dict) -> Dict:
        """Process new mood reading through all three timescales.

        Args:
            mood_input: PAD dict from Mood Reader output.

        Returns:
            Blended PAD output (weighted combination of all three
            timescales).
        """
        clamped = _clamp_pad(mood_input)

        for key in _PAD_KEYS:
            self.fast[key] = _ema(self.fast[key], clamped[key], _ALPHA_FAST)
            self.medium[key] = _ema(self.medium[key], clamped[key],
                                    _ALPHA_MEDIUM)
            self.slow[key] = _ema(self.slow[key], clamped[key], _ALPHA_SLOW)

        return self._blend()

    def drift_toward_baseline(self, rate: float = 0.05):
        """Between turns, gently pull all layers toward character baseline.

        Args:
            rate: Base drift rate. Fast drifts at rate*3, medium at
                rate, slow at rate*0.3.
        """
        for key in _PAD_KEYS:
            self.fast[key] = _ema(self.fast[key], self.baseline[key],
                                  rate * 3.0)
            self.medium[key] = _ema(self.medium[key], self.baseline[key],
                                    rate)
            self.slow[key] = _ema(self.slow[key], self.baseline[key],
                                  rate * 0.3)

    def get_state(self) -> Dict:
        """Full internal state for inspector visualization.

        Returns:
            Dict with ``fast``, ``medium``, ``slow``, ``output``,
            and ``baseline`` sub-dicts.
        """
        return {
            'fast': dict(self.fast),
            'medium': dict(self.medium),
            'slow': dict(self.slow),
            'output': self._blend(),
            'baseline': dict(self.baseline),
        }

    def _blend(self) -> Dict:
        """Weighted blend of all three timescales.

        Returns:
            Blended PAD dict, clamped to valid ranges.
        """
        result = {}
        for key in _PAD_KEYS:
            raw = (self.fast[key] * _WEIGHT_FAST
                   + self.medium[key] * _WEIGHT_MEDIUM
                   + self.slow[key] * _WEIGHT_SLOW)
            result[key] = raw

        return _clamp_pad(result)
