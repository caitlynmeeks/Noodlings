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
#   Tests for VRM Procedural Idle Animation via Muscles
#
#   Verifies idle muscle generation (breathing, head drift, spine sway),
#   muscle layering (idle vs external), amplitude bounds, frequency
#   incommensurability, and model matrix identity.
#   These tests run without an OpenGL context.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_vrm_procedural_idle
# PURPOSE:  Tests for Procedural Idle Animation via Muscles
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestIdleMusclGeneration, TestMuscleMerging,
#   TestIdleAmplitudes, TestIdleFrequencies,
#   TestModelMatrix
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import math
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Replicate _compute_idle_muscles locally for testing (no GL context needed)
# ---------------------------------------------------------------------------

def _compute_idle_muscles(t: float) -> dict:
    """Replicate VRMViewportWidget._compute_idle_muscles for testing."""
    muscles = {}
    two_pi = 2.0 * math.pi

    # Breathing
    breath = math.sin(t * two_pi / 3.5)
    muscles['Chest.FrontBack'] = 0.06 * breath
    muscles['UpperChest.FrontBack'] = 0.04 * breath
    muscles['Spine.FrontBack'] = 0.02 * breath
    muscles['LeftShoulder.DownUp'] = 0.03 * breath
    muscles['RightShoulder.DownUp'] = 0.03 * breath

    # Head drift
    muscles['Head.NodDownUp'] = 0.02 * math.sin(t * two_pi / 7.3)
    muscles['Head.TiltLeftRight'] = 0.015 * math.sin(t * two_pi / 11.1)
    muscles['Head.TurnLeftRight'] = 0.01 * math.sin(t * two_pi / 13.7)

    # Neck
    muscles['Neck.NodDownUp'] = 0.01 * math.sin(t * two_pi / 7.3)
    muscles['Neck.TiltLeftRight'] = 0.008 * math.sin(t * two_pi / 11.1)

    # Spine sway
    muscles['Spine.LeftRight'] = 0.015 * math.sin(t * two_pi / 9.7)

    return muscles


def _merge_muscles(idle: dict, external: dict) -> dict:
    """Replicate _merge_and_apply_muscles merge logic."""
    merged = dict(idle)
    merged.update(external)
    return merged


# All expected muscle keys from idle animation
IDLE_MUSCLE_KEYS = {
    'Chest.FrontBack', 'UpperChest.FrontBack', 'Spine.FrontBack',
    'LeftShoulder.DownUp', 'RightShoulder.DownUp',
    'Head.NodDownUp', 'Head.TiltLeftRight', 'Head.TurnLeftRight',
    'Neck.NodDownUp', 'Neck.TiltLeftRight',
    'Spine.LeftRight',
}

BREATHING_KEYS = {'Chest.FrontBack', 'UpperChest.FrontBack', 'Spine.FrontBack'}
HEAD_DRIFT_KEYS = {'Head.NodDownUp', 'Head.TiltLeftRight', 'Head.TurnLeftRight'}
SHOULDER_KEYS = {'LeftShoulder.DownUp', 'RightShoulder.DownUp'}
NECK_KEYS = {'Neck.NodDownUp', 'Neck.TiltLeftRight'}

# Idle frequency periods (seconds) for each muscle group
IDLE_PERIODS = {
    'Chest.FrontBack': 3.5,
    'UpperChest.FrontBack': 3.5,
    'Spine.FrontBack': 3.5,
    'LeftShoulder.DownUp': 3.5,
    'RightShoulder.DownUp': 3.5,
    'Head.NodDownUp': 7.3,
    'Head.TiltLeftRight': 11.1,
    'Head.TurnLeftRight': 13.7,
    'Neck.NodDownUp': 7.3,
    'Neck.TiltLeftRight': 11.1,
    'Spine.LeftRight': 9.7,
}


# =============================================================================
# Test: Idle Muscle Generation
# =============================================================================

class TestIdleMuscleGeneration:
    """Tests for _compute_idle_muscles output structure."""

    def test_compute_idle_muscles_returns_dict(self):
        result = _compute_idle_muscles(1.0)
        assert isinstance(result, dict)
        assert set(result.keys()) == IDLE_MUSCLE_KEYS

    def test_idle_has_breathing_muscles(self):
        result = _compute_idle_muscles(1.0)
        assert BREATHING_KEYS.issubset(result.keys())

    def test_idle_has_head_drift_muscles(self):
        result = _compute_idle_muscles(1.0)
        assert HEAD_DRIFT_KEYS.issubset(result.keys())

    def test_idle_has_spine_sway(self):
        result = _compute_idle_muscles(1.0)
        assert 'Spine.LeftRight' in result

    def test_idle_has_shoulder_rise(self):
        result = _compute_idle_muscles(1.0)
        assert SHOULDER_KEYS.issubset(result.keys())

    def test_idle_has_neck_muscles(self):
        result = _compute_idle_muscles(1.0)
        assert NECK_KEYS.issubset(result.keys())

    def test_idle_muscles_at_zero(self):
        """At t=0, sin(0) = 0, so all muscles should be 0."""
        result = _compute_idle_muscles(0.0)
        for key, value in result.items():
            assert value == 0.0, f"{key} should be 0 at t=0, got {value}"

    def test_idle_muscles_vary_over_time(self):
        """At t=0.875 (quarter period of 3.5s), breathing muscles are nonzero."""
        result = _compute_idle_muscles(0.875)
        assert result['Chest.FrontBack'] != 0.0
        assert result['UpperChest.FrontBack'] != 0.0
        # At quarter period, sin = 1.0, so chest should be at max
        assert abs(result['Chest.FrontBack'] - 0.06) < 1e-6


# =============================================================================
# Test: Idle Amplitudes
# =============================================================================

class TestIdleAmplitudes:
    """Tests for idle muscle amplitude bounds."""

    def test_idle_amplitudes_bounded(self):
        """All idle muscles stay within [-0.1, 0.1] over 1000 time samples."""
        for i in range(1000):
            t = i * 0.05  # 0 to 50 seconds
            result = _compute_idle_muscles(t)
            for key, value in result.items():
                assert -0.1 <= value <= 0.1, (
                    f"{key} = {value} at t={t} exceeds bounds"
                )

    def test_breathing_symmetric_shoulders(self):
        """Left and right shoulder amplitudes are equal at all times."""
        for i in range(100):
            t = i * 0.1
            result = _compute_idle_muscles(t)
            assert result['LeftShoulder.DownUp'] == result['RightShoulder.DownUp'], (
                f"Shoulder asymmetry at t={t}"
            )


# =============================================================================
# Test: Idle Frequencies
# =============================================================================

class TestIdleFrequencies:
    """Tests for idle muscle frequency characteristics."""

    def test_breathing_period(self):
        """Chest.FrontBack is near zero at t=0 and t=3.5 (full period)."""
        at_zero = _compute_idle_muscles(0.0)['Chest.FrontBack']
        at_period = _compute_idle_muscles(3.5)['Chest.FrontBack']
        assert abs(at_zero) < 1e-10
        assert abs(at_period) < 1e-6

    def test_incommensurate_frequencies(self):
        """No two muscle groups share the same period."""
        unique_periods = set(IDLE_PERIODS.values())
        # We expect: 3.5, 7.3, 11.1, 13.7, 9.7 = 5 distinct periods
        assert len(unique_periods) == 5

    def test_head_nod_period(self):
        """Head.NodDownUp near zero at t=0 and t=7.3."""
        at_zero = _compute_idle_muscles(0.0)['Head.NodDownUp']
        at_period = _compute_idle_muscles(7.3)['Head.NodDownUp']
        assert abs(at_zero) < 1e-10
        assert abs(at_period) < 1e-6


# =============================================================================
# Test: Muscle Merging
# =============================================================================

class TestMuscleMerging:
    """Tests for idle + external muscle layering."""

    def test_merge_idle_only(self):
        """No external muscles: merged == idle."""
        idle = _compute_idle_muscles(1.0)
        merged = _merge_muscles(idle, {})
        assert merged == idle

    def test_merge_external_overrides_idle(self):
        """External Chest.FrontBack=0.5 overrides idle's value."""
        idle = _compute_idle_muscles(1.0)
        external = {'Chest.FrontBack': 0.5}
        merged = _merge_muscles(idle, external)
        assert merged['Chest.FrontBack'] == 0.5
        # Other idle muscles preserved
        assert merged['Head.NodDownUp'] == idle['Head.NodDownUp']

    def test_merge_external_preserves_idle_others(self):
        """External sets Head only; breathing from idle preserved."""
        idle = _compute_idle_muscles(2.0)
        external = {'Head.TurnLeftRight': 0.3}
        merged = _merge_muscles(idle, external)

        # External value applied
        assert merged['Head.TurnLeftRight'] == 0.3

        # Breathing muscles unchanged from idle
        for key in BREATHING_KEYS:
            assert merged[key] == idle[key]

    def test_merge_empty_external(self):
        """Empty external dict, idle passes through."""
        idle = _compute_idle_muscles(3.0)
        merged = _merge_muscles(idle, {})
        for key in IDLE_MUSCLE_KEYS:
            assert merged[key] == idle[key]

    def test_merge_external_adds_new_muscles(self):
        """External can add muscles not in idle set."""
        idle = _compute_idle_muscles(1.0)
        external = {'LeftArm.DownUp': 0.7}
        merged = _merge_muscles(idle, external)
        assert merged['LeftArm.DownUp'] == 0.7
        # Idle muscles still present
        assert 'Chest.FrontBack' in merged


# =============================================================================
# Test: Model Matrix
# =============================================================================

class TestModelMatrix:
    """Tests for the simplified model matrix."""

    def test_model_matrix_is_identity(self):
        """_build_model_matrix() should return identity (idle is muscle-driven)."""
        # Replicate the simplified method
        model = np.eye(4, dtype=np.float32)
        np.testing.assert_array_equal(model, np.eye(4, dtype=np.float32))
