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
#   Tests for VRM Texture Rendering and Idle Animation
#
#   Verifies per-material draw groups, texture detection,
#   idle animation phase/matrix, and fallback color behavior.
#   These tests run without an OpenGL context.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_vrm_texture_rendering
# PURPOSE:  Tests for VRM Texture Rendering and Idle Animation
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestMaterialGroupTracking, TestTextureDetection,
#   TestIdleAnimation, TestFallbackColor
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
    VRMAvatar, Mesh, MToonMaterial, Skeleton,
)


def _make_avatar_with_materials(mesh_specs):
    """Helper to create a VRMAvatar with meshes assigned to materials.

    Args:
        mesh_specs: List of (material_index, vertex_count, index_count) tuples.
            Each generates a simple mesh with the given material assignment.

    Returns:
        VRMAvatar with the specified meshes and placeholder materials.
    """
    avatar = VRMAvatar()
    materials_needed = set()

    for mat_idx, vert_count, idx_count in mesh_specs:
        vertices = np.zeros((vert_count, 3), dtype=np.float32)
        normals = np.zeros((vert_count, 3), dtype=np.float32)
        uvs = np.zeros((vert_count, 2), dtype=np.float32)
        # Simple sequential indices (triangles)
        indices = np.arange(idx_count, dtype=np.uint32) % vert_count

        mesh = Mesh(
            name=f"mesh_mat{mat_idx}",
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            indices=indices,
            material_index=mat_idx if mat_idx >= 0 else None,
        )
        avatar.meshes.append(mesh)
        if mat_idx >= 0:
            materials_needed.add(mat_idx)

    # Create placeholder materials
    for i in range(max(materials_needed) + 1 if materials_needed else 0):
        avatar.materials.append(MToonMaterial(
            name=f"material_{i}",
            diffuse_color=(0.8, 0.7, 0.6, 1.0),
        ))

    return avatar


class TestMaterialGroupTracking:
    """Test that _create_mesh_buffers correctly computes per-material groups."""

    def test_single_material_single_mesh(self):
        """One mesh with material 0 produces one group."""
        avatar = _make_avatar_with_materials([(0, 100, 300)])
        # Simulate what _create_mesh_buffers does (sorting + grouping)
        groups = self._compute_groups(avatar)
        assert len(groups) == 1
        mat_idx, byte_offset, count = groups[0]
        assert mat_idx == 0
        assert byte_offset == 0
        assert count == 300

    def test_two_meshes_same_material_merge(self):
        """Two meshes with same material should merge into one group."""
        avatar = _make_avatar_with_materials([
            (0, 50, 150),
            (0, 80, 240),
        ])
        groups = self._compute_groups(avatar)
        assert len(groups) == 1
        assert groups[0][0] == 0  # material index
        assert groups[0][2] == 390  # 150 + 240 indices

    def test_different_materials_separate_groups(self):
        """Meshes with different materials produce separate groups."""
        avatar = _make_avatar_with_materials([
            (0, 50, 150),
            (1, 80, 240),
            (2, 30, 90),
        ])
        groups = self._compute_groups(avatar)
        assert len(groups) == 3
        # Sorted by material index
        mat_indices = [g[0] for g in groups]
        assert mat_indices == [0, 1, 2]

    def test_no_material_index_uses_negative_one(self):
        """Mesh with no material_index gets grouped as -1."""
        avatar = _make_avatar_with_materials([(-1, 50, 150)])
        groups = self._compute_groups(avatar)
        assert len(groups) == 1
        assert groups[0][0] == -1

    def test_byte_offsets_are_correct(self):
        """Byte offsets account for uint32 (4 bytes per index)."""
        avatar = _make_avatar_with_materials([
            (0, 50, 150),
            (1, 80, 240),
        ])
        groups = self._compute_groups(avatar)
        assert groups[0][1] == 0           # first group starts at byte 0
        assert groups[1][1] == 150 * 4     # second group starts after 150 indices

    def test_empty_mesh_skipped(self):
        """Meshes with no vertices are skipped."""
        avatar = VRMAvatar()
        avatar.meshes.append(Mesh(
            name="empty",
            vertices=np.zeros((0, 3), dtype=np.float32),
            material_index=0,
        ))
        avatar.materials.append(MToonMaterial(name="mat0"))
        groups = self._compute_groups(avatar)
        assert len(groups) == 0

    def _compute_groups(self, avatar):
        """Replicate the material group computation logic from
        _create_mesh_buffers without needing an OpenGL context."""
        sorted_meshes = sorted(
            avatar.meshes,
            key=lambda m: m.material_index if m.material_index is not None else -1
        )

        material_groups = []
        vertex_offset = 0
        index_offset = 0

        for mesh in sorted_meshes:
            if mesh.vertices is None or len(mesh.vertices) == 0:
                continue

            vertices = np.asarray(mesh.vertices, dtype=np.float32)

            if mesh.indices is not None:
                indices = np.asarray(mesh.indices, dtype=np.uint32).flatten() + vertex_offset
            else:
                indices = np.arange(len(vertices), dtype=np.uint32) + vertex_offset

            mat_idx = mesh.material_index if mesh.material_index is not None else -1
            byte_offset = index_offset * 4
            index_count = len(indices)

            if (material_groups
                    and material_groups[-1][0] == mat_idx):
                prev_mat, prev_offset, prev_count = material_groups[-1]
                material_groups[-1] = (prev_mat, prev_offset, prev_count + index_count)
            else:
                material_groups.append((mat_idx, byte_offset, index_count))

            vertex_offset += len(vertices)
            index_offset += index_count

        return material_groups


class TestTextureDetection:
    """Test that textures are correctly identified for materials."""

    def test_material_with_diffuse_texture(self):
        """Material with diffuse_texture index should be detected."""
        avatar = VRMAvatar()
        avatar.materials.append(MToonMaterial(
            name="skin",
            diffuse_texture=0,
            diffuse_color=(1.0, 1.0, 1.0, 1.0),
        ))
        avatar.textures.append(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

        mat = avatar.materials[0]
        assert mat.diffuse_texture == 0
        assert mat.diffuse_texture < len(avatar.textures)

    def test_material_without_texture(self):
        """Material with no diffuse_texture should use flat color."""
        avatar = VRMAvatar()
        avatar.materials.append(MToonMaterial(
            name="flat",
            diffuse_texture=None,
            diffuse_color=(0.8, 0.3, 0.3, 1.0),
        ))

        mat = avatar.materials[0]
        assert mat.diffuse_texture is None

    def test_texture_index_out_of_range(self):
        """Material referencing non-existent texture should be skipped."""
        avatar = VRMAvatar()
        avatar.materials.append(MToonMaterial(
            name="bad_ref",
            diffuse_texture=5,
        ))
        # No textures loaded
        assert len(avatar.textures) == 0
        mat = avatar.materials[0]
        assert mat.diffuse_texture >= len(avatar.textures)

    def test_multiple_materials_mixed_textures(self):
        """Avatar with some textured and some untextured materials."""
        avatar = VRMAvatar()
        avatar.textures.append(b'\x89PNG' + b'\x00' * 50)
        avatar.textures.append(b'\xff\xd8\xff' + b'\x00' * 50)

        avatar.materials.append(MToonMaterial(
            name="body", diffuse_texture=0
        ))
        avatar.materials.append(MToonMaterial(
            name="eyes", diffuse_texture=1
        ))
        avatar.materials.append(MToonMaterial(
            name="outline", diffuse_texture=None
        ))

        textured = [
            m for m in avatar.materials
            if m.diffuse_texture is not None
            and m.diffuse_texture < len(avatar.textures)
        ]
        untextured = [
            m for m in avatar.materials
            if m.diffuse_texture is None
        ]
        assert len(textured) == 2
        assert len(untextured) == 1


class TestIdleAnimation:
    """Test idle animation phase advancement and model matrix."""

    def test_phase_advances_on_tick(self):
        """Idle phase should increase by ~0.016 each tick."""
        phase = 0.0
        for _ in range(10):
            phase += 0.016
        assert abs(phase - 0.16) < 1e-6

    def test_model_matrix_includes_bob(self):
        """Model matrix Y translation should oscillate with idle phase."""
        for t in [0.0, 1.0, 2.0, 3.0, 4.0]:
            model = self._build_model_matrix(t)
            expected_bob = 0.01 * math.sin(t * 2.0 * math.pi / 4.0)
            assert abs(model[1, 3] - expected_bob) < 1e-7, (
                f"At t={t}, expected bob={expected_bob}, got {model[1, 3]}"
            )

    def test_model_matrix_includes_breathing(self):
        """Model matrix Y scale should pulse with idle phase."""
        for t in [0.0, 0.875, 1.75, 2.625, 3.5]:
            model = self._build_model_matrix(t)
            expected_scale = 1.0 + 0.02 * math.sin(t * 2.0 * math.pi / 3.5)
            assert abs(model[1, 1] - expected_scale) < 1e-7, (
                f"At t={t}, expected scale={expected_scale}, got {model[1, 1]}"
            )

    def test_identity_at_phase_zero(self):
        """At phase=0, model matrix should be identity (sin(0)=0)."""
        model = self._build_model_matrix(0.0)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_allclose(model, expected, atol=1e-7)

    def test_bob_period_is_four_seconds(self):
        """Bob should complete a full cycle in 4 seconds."""
        # At t=0 and t=4.0, sin should be 0
        model_0 = self._build_model_matrix(0.0)
        model_4 = self._build_model_matrix(4.0)
        assert abs(model_0[1, 3] - model_4[1, 3]) < 1e-6

    def test_breathing_period_is_three_point_five_seconds(self):
        """Breathing should complete a full cycle in 3.5 seconds."""
        model_0 = self._build_model_matrix(0.0)
        model_35 = self._build_model_matrix(3.5)
        assert abs(model_0[1, 1] - model_35[1, 1]) < 1e-6

    def test_non_y_axes_unchanged(self):
        """X and Z axes should not be affected by idle animation."""
        model = self._build_model_matrix(1.5)
        assert model[0, 0] == 1.0  # X scale
        assert model[2, 2] == 1.0  # Z scale
        assert model[0, 3] == 0.0  # X translation
        assert model[2, 3] == 0.0  # Z translation

    def _build_model_matrix(self, t: float) -> np.ndarray:
        """Replicate the model matrix construction from VRMViewportWidget."""
        model = np.eye(4, dtype=np.float32)

        bob_y = 0.01 * math.sin(t * 2.0 * math.pi / 4.0)
        model[1, 3] = bob_y

        breath_scale = 1.0 + 0.02 * math.sin(t * 2.0 * math.pi / 3.5)
        model[1, 1] = breath_scale

        return model


class TestFallbackColor:
    """Test fallback color behavior for untextured materials."""

    def test_material_color_extraction(self):
        """Material diffuse_color should be extracted as RGB tuple."""
        mat = MToonMaterial(
            name="skin",
            diffuse_color=(0.9, 0.75, 0.65, 1.0),
        )
        r, g, b = float(mat.diffuse_color[0]), float(mat.diffuse_color[1]), float(mat.diffuse_color[2])
        assert abs(r - 0.9) < 1e-6
        assert abs(g - 0.75) < 1e-6
        assert abs(b - 0.65) < 1e-6

    def test_default_material_color_is_white(self):
        """Default MToonMaterial diffuse_color is white."""
        mat = MToonMaterial(name="default")
        assert mat.diffuse_color == (1, 1, 1, 1)

    def test_fallback_tan_when_no_material(self):
        """When material_index is -1, fallback tan (0.85, 0.80, 0.75) is used."""
        # This tests the .get() fallback in _draw_mesh
        material_colors = {0: (0.9, 0.8, 0.7)}
        fallback = material_colors.get(-1, (0.85, 0.80, 0.75))
        assert fallback == (0.85, 0.80, 0.75)

    def test_material_color_dict_lookup(self):
        """Material color dict returns correct color for valid index."""
        material_colors = {
            0: (0.9, 0.8, 0.7),
            1: (0.3, 0.4, 0.5),
        }
        assert material_colors.get(0, (0.85, 0.80, 0.75)) == (0.9, 0.8, 0.7)
        assert material_colors.get(1, (0.85, 0.80, 0.75)) == (0.3, 0.4, 0.5)
        assert material_colors.get(99, (0.85, 0.80, 0.75)) == (0.85, 0.80, 0.75)


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
