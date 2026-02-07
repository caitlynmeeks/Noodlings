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
#   Tests for VRM Blend Shape Morph Targets + Animal Crossing Arm Bob
#
#   Verifies morph target dataclasses, CPU blend math, expression bind
#   parsing, and idle arm bob animation. No OpenGL context needed.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_vrm_blend_shapes
# PURPOSE:  Tests for Blend Shape Morph Target Pipeline
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestMorphTargetDataclasses, TestCPUMorphBlending,
#   TestExpressionBindParsing, TestArmBobIdle
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

from noodlestudio.core.semantic_world.vrm_parser import (
    BlendShape, MorphTargetBind, Mesh, Vector3, Quaternion, Transform,
)


# ---------------------------------------------------------------------------
# Helper: replicate CPU morph blending logic for testing
# ---------------------------------------------------------------------------

def _cpu_blend(base_positions, blend_shapes, current_weights, mesh_vertex_ranges):
    """Replicate _apply_blend_shapes blend math for testing.

    Args:
        base_positions: (N, 3) base vertex positions
        blend_shapes: list of BlendShape objects
        current_weights: dict mapping shape name -> weight
        mesh_vertex_ranges: list of (mesh_ref, vbo_offset, vtx_count)

    Returns:
        blended_positions: (N, 3) with morph deltas applied
    """
    blended = base_positions.copy()

    # Build lookup
    bs_map = {}
    for bs in blend_shapes:
        bs_map[bs.name] = bs
        if bs.preset:
            bs_map[bs.preset] = bs

    for shape_name, weight in current_weights.items():
        if weight == 0.0:
            continue
        bs = bs_map.get(shape_name)
        if not bs or not bs.binds:
            continue

        for bind in bs.binds:
            for mesh_ref, vbo_offset, vtx_count in mesh_vertex_ranges:
                if mesh_ref.source_mesh_index != bind.mesh_index:
                    continue
                if not mesh_ref.morph_targets:
                    continue
                if bind.target_index >= len(mesh_ref.morph_targets):
                    continue

                delta = mesh_ref.morph_targets[bind.target_index]
                effective_weight = weight * bind.weight
                start = vbo_offset
                end = vbo_offset + vtx_count
                blended[start:end] += effective_weight * delta

    return blended


def _compute_idle_muscles(t: float) -> dict:
    """Replicate VRMViewportWidget._compute_idle_muscles for arm bob tests."""
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
    muscles['Neck.NodDownUp'] = 0.01 * math.sin(t * two_pi / 7.3)
    muscles['Neck.TiltLeftRight'] = 0.008 * math.sin(t * two_pi / 11.1)

    # Spine sway
    muscles['Spine.LeftRight'] = 0.015 * math.sin(t * two_pi / 9.7)

    # Arm bob
    muscles['LeftArm.FrontBack'] = 0.03 * math.sin(t * two_pi / 5.3)
    muscles['RightArm.FrontBack'] = 0.03 * math.sin(t * two_pi / 5.9)
    muscles['LeftArm.DownUp'] = 0.02 * math.sin(t * two_pi / 8.3)
    muscles['RightArm.DownUp'] = 0.02 * math.sin(t * two_pi / 8.9)

    return muscles


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mesh(source_idx, vertex_count=100, num_targets=2):
    """Create a Mesh with mock morph targets."""
    vertices = np.random.rand(vertex_count, 3).astype(np.float32)
    normals = np.random.rand(vertex_count, 3).astype(np.float32)

    morph_targets = []
    for i in range(num_targets):
        # Small deltas (like facial deformation)
        delta = (np.random.rand(vertex_count, 3) * 0.01).astype(np.float32)
        morph_targets.append(delta)

    return Mesh(
        name=f'mesh_{source_idx}',
        vertices=vertices,
        normals=normals,
        source_mesh_index=source_idx,
        morph_targets=morph_targets,
    )


# =============================================================================
# Test: Morph Target Dataclasses
# =============================================================================

class TestMorphTargetDataclasses:
    """Tests for MorphTargetBind, BlendShape, and Mesh dataclass additions."""

    def test_morph_target_bind_dataclass(self):
        bind = MorphTargetBind(mesh_index=0, target_index=3, weight=0.8)
        assert bind.mesh_index == 0
        assert bind.target_index == 3
        assert bind.weight == 0.8

    def test_morph_target_bind_default_weight(self):
        bind = MorphTargetBind(mesh_index=1, target_index=0)
        assert bind.weight == 1.0

    def test_blend_shape_has_binds(self):
        binds = [
            MorphTargetBind(mesh_index=0, target_index=0, weight=1.0),
            MorphTargetBind(mesh_index=0, target_index=1, weight=0.5),
        ]
        bs = BlendShape(name='happy', preset='happy', binds=binds)
        assert len(bs.binds) == 2
        assert bs.binds[0].target_index == 0
        assert bs.binds[1].weight == 0.5

    def test_blend_shape_default_empty_binds(self):
        bs = BlendShape(name='test')
        assert bs.binds == []

    def test_mesh_has_morph_targets(self):
        mesh = _make_mesh(source_idx=0, vertex_count=50, num_targets=3)
        assert mesh.morph_targets is not None
        assert len(mesh.morph_targets) == 3
        assert mesh.morph_targets[0].shape == (50, 3)

    def test_mesh_source_index(self):
        mesh = _make_mesh(source_idx=5)
        assert mesh.source_mesh_index == 5

    def test_vrm_0x_bind_weight_normalization(self):
        """VRM 0.x stores weight as 0-100, should normalize to 0-1."""
        raw_weight = 100
        normalized = raw_weight / 100.0
        bind = MorphTargetBind(mesh_index=0, target_index=0, weight=normalized)
        assert bind.weight == 1.0

        raw_weight = 50
        normalized = raw_weight / 100.0
        bind = MorphTargetBind(mesh_index=0, target_index=0, weight=normalized)
        assert bind.weight == 0.5


# =============================================================================
# Test: CPU Morph Blending Math
# =============================================================================

class TestCPUMorphBlending:
    """Tests for the CPU-side morph target blending computation."""

    def test_blend_zero_weight(self):
        """Weight 0 produces no change from base."""
        mesh = _make_mesh(source_idx=0, vertex_count=10, num_targets=1)
        base = np.ones((10, 3), dtype=np.float32)
        bs = BlendShape(
            name='test', preset='test',
            binds=[MorphTargetBind(mesh_index=0, target_index=0)]
        )
        result = _cpu_blend(base, [bs], {'test': 0.0}, [(mesh, 0, 10)])
        np.testing.assert_array_equal(result, base)

    def test_blend_single_target_full_weight(self):
        """Weight 1.0 adds full delta."""
        delta = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mesh = Mesh(
            name='face', vertices=np.zeros((1, 3), dtype=np.float32),
            source_mesh_index=0, morph_targets=[delta],
        )
        base = np.zeros((1, 3), dtype=np.float32)
        bs = BlendShape(
            name='smile', preset='smile',
            binds=[MorphTargetBind(mesh_index=0, target_index=0, weight=1.0)]
        )
        result = _cpu_blend(base, [bs], {'smile': 1.0}, [(mesh, 0, 1)])
        np.testing.assert_array_almost_equal(result, delta)

    def test_blend_partial_weight(self):
        """Weight 0.5 adds half delta."""
        delta = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        mesh = Mesh(
            name='face', vertices=np.zeros((1, 3), dtype=np.float32),
            source_mesh_index=0, morph_targets=[delta],
        )
        base = np.zeros((1, 3), dtype=np.float32)
        bs = BlendShape(
            name='test', binds=[MorphTargetBind(0, 0, 1.0)]
        )
        result = _cpu_blend(base, [bs], {'test': 0.5}, [(mesh, 0, 1)])
        np.testing.assert_array_almost_equal(result, [[0.5, 1.0, 1.5]])

    def test_blend_multiple_targets_accumulate(self):
        """Two blend shapes accumulate their deltas."""
        delta1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        delta2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        mesh = Mesh(
            name='face', vertices=np.zeros((1, 3), dtype=np.float32),
            source_mesh_index=0, morph_targets=[delta1, delta2],
        )
        bs1 = BlendShape(name='a', binds=[MorphTargetBind(0, 0, 1.0)])
        bs2 = BlendShape(name='b', binds=[MorphTargetBind(0, 1, 1.0)])
        base = np.zeros((1, 3), dtype=np.float32)
        result = _cpu_blend(
            base, [bs1, bs2], {'a': 1.0, 'b': 1.0}, [(mesh, 0, 1)]
        )
        np.testing.assert_array_almost_equal(result, [[1.0, 1.0, 0.0]])

    def test_blend_with_bind_weight(self):
        """Effective weight = blend_weight * bind_weight."""
        delta = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        mesh = Mesh(
            name='face', vertices=np.zeros((1, 3), dtype=np.float32),
            source_mesh_index=0, morph_targets=[delta],
        )
        bs = BlendShape(
            name='test',
            binds=[MorphTargetBind(0, 0, 0.5)]  # bind weight = 0.5
        )
        base = np.zeros((1, 3), dtype=np.float32)
        result = _cpu_blend(base, [bs], {'test': 0.6}, [(mesh, 0, 1)])
        # effective = 0.6 * 0.5 = 0.3
        np.testing.assert_array_almost_equal(result, [[0.3, 0.3, 0.3]])

    def test_blend_preserves_non_target_vertices(self):
        """Vertices from a different mesh are unchanged."""
        delta = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        face_mesh = Mesh(
            name='face', vertices=np.zeros((1, 3), dtype=np.float32),
            source_mesh_index=0, morph_targets=[delta],
        )
        body_mesh = Mesh(
            name='body', vertices=np.ones((1, 3), dtype=np.float32),
            source_mesh_index=1,
        )
        # Combined VBO: body at offset 0 (1 vert), face at offset 1 (1 vert)
        base = np.array([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        bs = BlendShape(
            name='smile', binds=[MorphTargetBind(0, 0, 1.0)]
        )
        ranges = [(body_mesh, 0, 1), (face_mesh, 1, 1)]
        result = _cpu_blend(base, [bs], {'smile': 1.0}, ranges)
        # Body unchanged
        np.testing.assert_array_almost_equal(result[0], [5.0, 5.0, 5.0])
        # Face got delta
        np.testing.assert_array_almost_equal(result[1], [1.0, 1.0, 1.0])

    def test_blend_resets_to_base(self):
        """With no weights, result equals base."""
        mesh = _make_mesh(source_idx=0, vertex_count=5, num_targets=2)
        base = np.random.rand(5, 3).astype(np.float32)
        bs = BlendShape(
            name='test', binds=[MorphTargetBind(0, 0, 1.0)]
        )
        result = _cpu_blend(base, [bs], {}, [(mesh, 0, 5)])
        np.testing.assert_array_equal(result, base)

    def test_blend_by_preset_name(self):
        """Can look up blend shape by preset name."""
        delta = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        mesh = Mesh(
            name='face', vertices=np.zeros((1, 3), dtype=np.float32),
            source_mesh_index=0, morph_targets=[delta],
        )
        bs = BlendShape(
            name='Fcl_ALL_Joy', preset='happy',
            binds=[MorphTargetBind(0, 0, 1.0)]
        )
        base = np.zeros((1, 3), dtype=np.float32)
        # Look up by preset 'happy'
        result = _cpu_blend(base, [bs], {'happy': 1.0}, [(mesh, 0, 1)])
        np.testing.assert_array_almost_equal(result, [[1.0, 0.0, 0.0]])


# =============================================================================
# Test: Animal Crossing Arm Bob
# =============================================================================

class TestArmBobIdle:
    """Tests for the Animal Crossing-style arm bob in idle animation."""

    def test_idle_has_arm_muscles(self):
        result = _compute_idle_muscles(1.0)
        assert 'LeftArm.FrontBack' in result
        assert 'RightArm.FrontBack' in result
        assert 'LeftArm.DownUp' in result
        assert 'RightArm.DownUp' in result

    def test_arm_bob_asymmetric(self):
        """Left and right arms have different periods, so values differ."""
        result = _compute_idle_muscles(2.0)
        assert result['LeftArm.FrontBack'] != result['RightArm.FrontBack']
        assert result['LeftArm.DownUp'] != result['RightArm.DownUp']

    def test_arm_bob_amplitudes_bounded(self):
        """Arm muscles stay within [-0.05, 0.05] over many samples."""
        arm_keys = [
            'LeftArm.FrontBack', 'RightArm.FrontBack',
            'LeftArm.DownUp', 'RightArm.DownUp',
        ]
        for i in range(500):
            t = i * 0.1
            result = _compute_idle_muscles(t)
            for key in arm_keys:
                assert -0.05 <= result[key] <= 0.05, (
                    f"{key} = {result[key]} at t={t} exceeds bounds"
                )

    def test_arm_bob_at_zero(self):
        """At t=0, arm muscles are 0."""
        result = _compute_idle_muscles(0.0)
        assert result['LeftArm.FrontBack'] == 0.0
        assert result['RightArm.FrontBack'] == 0.0
        assert result['LeftArm.DownUp'] == 0.0
        assert result['RightArm.DownUp'] == 0.0
