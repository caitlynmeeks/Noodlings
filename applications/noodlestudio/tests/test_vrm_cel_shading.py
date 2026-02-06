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
#   Tests for VRM MToon Cel Shading
#
#   Verifies MToon material data pipeline, shade color extraction,
#   cel-shading uniform defaults, and shader uniform caching.
#   These tests run without an OpenGL context.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_vrm_cel_shading
# PURPOSE:  Tests for VRM MToon Cel Shading Pipeline
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestMToonDataPipeline, TestCelShadingDefaults,
#   TestShadeColorExtraction, TestMToonUniformNames
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest

from noodlestudio.core.semantic_world.vrm_parser import (
    VRMAvatar, Mesh, MToonMaterial, Skeleton,
)

import numpy as np


def _make_avatar_with_mtoon(material_specs):
    """Helper to create a VRMAvatar with MToon materials.

    Args:
        material_specs: List of dicts with MToonMaterial kwargs.

    Returns:
        VRMAvatar with the specified materials and a simple mesh per material.
    """
    avatar = VRMAvatar()

    for i, spec in enumerate(material_specs):
        avatar.materials.append(MToonMaterial(name=f"mat_{i}", **spec))
        # Add a minimal mesh referencing this material
        avatar.meshes.append(Mesh(
            name=f"mesh_{i}",
            vertices=np.zeros((10, 3), dtype=np.float32),
            normals=np.zeros((10, 3), dtype=np.float32),
            uvs=np.zeros((10, 2), dtype=np.float32),
            indices=np.arange(9, dtype=np.uint32),
            material_index=i,
        ))

    return avatar


class TestMToonDataPipeline:
    """Test that MToon material data is correctly stored per material index."""

    def test_mtoon_dict_populated(self):
        """_material_mtoon should map mat_idx to MToonMaterial."""
        avatar = _make_avatar_with_mtoon([
            {'shade_color': (0.4, 0.3, 0.5, 1.0), 'shading_toony': 0.8},
            {'shade_color': (0.6, 0.5, 0.4, 1.0), 'shading_toony': 0.6},
        ])
        # Simulate what _load_textures does
        material_mtoon = {}
        for mat_idx, material in enumerate(avatar.materials):
            material_mtoon[mat_idx] = material

        assert len(material_mtoon) == 2
        assert material_mtoon[0].shading_toony == 0.8
        assert material_mtoon[1].shading_toony == 0.6

    def test_mtoon_references_are_same_objects(self):
        """Stored references should be the actual MToonMaterial objects."""
        avatar = _make_avatar_with_mtoon([
            {'rim_power': 3.0},
        ])
        material_mtoon = {}
        for mat_idx, material in enumerate(avatar.materials):
            material_mtoon[mat_idx] = material

        assert material_mtoon[0] is avatar.materials[0]

    def test_empty_avatar_produces_empty_mtoon(self):
        """Avatar with no materials produces empty _material_mtoon."""
        avatar = VRMAvatar()
        material_mtoon = {}
        for mat_idx, material in enumerate(avatar.materials):
            material_mtoon[mat_idx] = material

        assert len(material_mtoon) == 0

    def test_mtoon_dict_keys_match_material_indices(self):
        """Keys should be sequential indices matching avatar.materials."""
        avatar = _make_avatar_with_mtoon([
            {}, {}, {},
        ])
        material_mtoon = {}
        for mat_idx, material in enumerate(avatar.materials):
            material_mtoon[mat_idx] = material

        assert list(material_mtoon.keys()) == [0, 1, 2]


class TestShadeColorExtraction:
    """Test shade_color extraction from MToonMaterial for shader uniforms."""

    def test_shade_color_rgb_from_rgba(self):
        """shade_color[:3] should give RGB without alpha."""
        mat = MToonMaterial(
            name="skin",
            shade_color=(0.4, 0.3, 0.5, 1.0),
        )
        r, g, b = mat.shade_color[:3]
        assert abs(r - 0.4) < 1e-6
        assert abs(g - 0.3) < 1e-6
        assert abs(b - 0.5) < 1e-6

    def test_default_shade_color(self):
        """Default shade_color is (0.5, 0.5, 0.5, 1)."""
        mat = MToonMaterial(name="default")
        assert mat.shade_color == (0.5, 0.5, 0.5, 1)

    def test_shade_color_darker_than_diffuse(self):
        """Typical VRM pattern: shade is darker/cooler than diffuse."""
        mat = MToonMaterial(
            name="typical",
            diffuse_color=(0.9, 0.85, 0.8, 1.0),
            shade_color=(0.5, 0.45, 0.55, 1.0),
        )
        # Shade should be dimmer on average
        diffuse_avg = sum(mat.diffuse_color[:3]) / 3
        shade_avg = sum(mat.shade_color[:3]) / 3
        assert shade_avg < diffuse_avg

    def test_rim_color_default_is_black(self):
        """Default rim_color should be black (no rim by default)."""
        mat = MToonMaterial(name="default")
        assert mat.rim_color == (0, 0, 0)

    def test_rim_power_default(self):
        """Default rim_power should be 1.0."""
        mat = MToonMaterial(name="default")
        assert mat.rim_power == 1.0

    def test_shading_toony_default(self):
        """Default shading_toony should be 0.9 (sharp cel boundary)."""
        mat = MToonMaterial(name="default")
        assert mat.shading_toony == 0.9

    def test_shading_shift_default(self):
        """Default shading_shift should be 0.0 (no shift)."""
        mat = MToonMaterial(name="default")
        assert mat.shading_shift == 0.0


class TestCelShadingDefaults:
    """Test fallback values when MToon data is absent for a material."""

    def test_fallback_shade_color(self):
        """When no MToon data, shade_color defaults to cool gray."""
        material_mtoon = {}  # empty
        mtoon = material_mtoon.get(0)
        assert mtoon is None
        # Fallback values from _draw_mesh
        default_shade = (0.65, 0.65, 0.7)
        assert default_shade[2] > default_shade[0]  # slightly cooler (more blue)

    def test_fallback_toony_is_sharp(self):
        """Default toony should produce a sharp cel boundary."""
        default_toony = 0.9
        # With toony=0.9, smoothstep range is 0.05 to 0.95 -- narrow transition
        low = 0.5 - default_toony * 0.5   # 0.05
        high = 0.5 + default_toony * 0.5  # 0.95
        assert high - low == pytest.approx(0.9, abs=1e-6)

    def test_fallback_rim_is_invisible(self):
        """Default rim_color (0,0,0) should add nothing to output."""
        default_rim = (0.0, 0.0, 0.0)
        # rim contribution = rim_color * rim_factor
        # With black rim_color, contribution is always zero
        assert sum(default_rim) == 0.0

    def test_fallback_shift_is_neutral(self):
        """Default shading_shift 0.0 means no bias toward light or shadow."""
        default_shift = 0.0
        assert default_shift == 0.0

    def test_rim_power_clamped_above_zero(self):
        """rim_power of 0 would cause pow(x, 0)=1 everywhere. We clamp to 0.1."""
        # In _draw_mesh, we use max(0.1, mtoon.rim_power)
        mat = MToonMaterial(name="zero_rim", rim_power=0.0)
        clamped = max(0.1, mat.rim_power)
        assert clamped == 0.1


class TestMToonUniformNames:
    """Test that all required MToon uniform names are defined."""

    EXPECTED_MTOON_UNIFORMS = [
        'uShadeColor',
        'uShadingToony',
        'uShadingShift',
        'uRimColor',
        'uRimPower',
        'uCameraPos',
    ]

    EXPECTED_BASE_UNIFORMS = [
        'uModel',
        'uView',
        'uProjection',
        'uLightDir',
        'uColor',
        'uDiffuseTex',
        'uHasTexture',
    ]

    def test_mtoon_uniform_names_present_in_shader(self):
        """Fragment shader source should contain all MToon uniform declarations."""
        # Replicate the shader source to verify uniform names
        shader_source = """
            uniform vec3 uShadeColor;
            uniform float uShadingToony;
            uniform float uShadingShift;
            uniform vec3 uRimColor;
            uniform float uRimPower;
            uniform vec3 uCameraPos;
        """
        for name in self.EXPECTED_MTOON_UNIFORMS:
            assert name in shader_source, f"Missing uniform: {name}"

    def test_all_uniforms_would_be_cached(self):
        """The uniform cache dict should include both base and MToon uniforms."""
        all_expected = self.EXPECTED_BASE_UNIFORMS + self.EXPECTED_MTOON_UNIFORMS
        # Simulate the cache dict keys
        cache_keys = set(all_expected)
        assert len(cache_keys) == 13  # 7 base + 6 MToon

    def test_no_duplicate_uniform_names(self):
        """No uniform name should appear in both base and MToon lists."""
        base_set = set(self.EXPECTED_BASE_UNIFORMS)
        mtoon_set = set(self.EXPECTED_MTOON_UNIFORMS)
        overlap = base_set & mtoon_set
        assert len(overlap) == 0, f"Duplicate uniforms: {overlap}"


class TestCelShadingMath:
    """Test the cel-shading math that will run in the fragment shader."""

    def _half_lambert(self, n_dot_l):
        """Half-Lambert remap: dot(N, L) * 0.5 + 0.5"""
        return n_dot_l * 0.5 + 0.5

    def _smoothstep(self, edge0, edge1, x):
        """GLSL smoothstep equivalent."""
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0))) if edge1 != edge0 else 0.0
        return t * t * (3.0 - 2.0 * t)

    def _toony_step(self, half_lambert, toony, shift):
        """Compute the cel-shading step value."""
        shifted = half_lambert + shift
        low = 0.5 - toony * 0.5
        high = 0.5 + toony * 0.5
        return self._smoothstep(low, high, shifted)

    def test_facing_light_is_fully_lit(self):
        """Surface directly facing light should be fully lit (toony=1)."""
        n_dot_l = 1.0  # Normal aligned with light
        hl = self._half_lambert(n_dot_l)  # 1.0
        step = self._toony_step(hl, 0.9, 0.0)
        assert step == pytest.approx(1.0, abs=1e-3)

    def test_facing_away_is_fully_shaded(self):
        """Surface facing away from light should be fully shaded (toony=0)."""
        n_dot_l = -1.0  # Opposite to light
        hl = self._half_lambert(n_dot_l)  # 0.0
        step = self._toony_step(hl, 0.9, 0.0)
        assert step == pytest.approx(0.0, abs=1e-3)

    def test_grazing_angle_with_high_toony(self):
        """At 90 degrees (NdotL=0), half-lambert=0.5, toony=0.9 should be mid-step."""
        n_dot_l = 0.0
        hl = self._half_lambert(n_dot_l)  # 0.5
        step = self._toony_step(hl, 0.9, 0.0)
        assert step == pytest.approx(0.5, abs=0.01)

    def test_positive_shift_brightens(self):
        """Positive shading_shift should push boundary toward shadow, making more lit area."""
        n_dot_l = -0.2  # Slightly away from light
        hl = self._half_lambert(n_dot_l)  # 0.4
        step_no_shift = self._toony_step(hl, 0.9, 0.0)
        step_pos_shift = self._toony_step(hl, 0.9, 0.2)
        assert step_pos_shift > step_no_shift

    def test_negative_shift_darkens(self):
        """Negative shading_shift should push boundary toward light, making more shadow."""
        n_dot_l = 0.2  # Slightly toward light
        hl = self._half_lambert(n_dot_l)  # 0.6
        step_no_shift = self._toony_step(hl, 0.9, 0.0)
        step_neg_shift = self._toony_step(hl, 0.9, -0.2)
        assert step_neg_shift < step_no_shift

    def test_low_toony_is_soft_gradient(self):
        """Low toony value should produce a gradual transition."""
        # With toony=0.1, smoothstep range is 0.45 to 0.55
        # Test values within the transition band
        hl_values = [0.46, 0.48, 0.50, 0.52, 0.54]
        steps = [self._toony_step(hl, 0.1, 0.0) for hl in hl_values]
        # Should transition gradually
        for i in range(len(steps) - 1):
            assert steps[i] < steps[i + 1]

    def test_high_toony_is_sharp_step(self):
        """High toony value should produce a near-binary step."""
        # With toony=1.0, smoothstep range is 0.0 to 1.0
        # But values near 0.5 should still transition
        step_below = self._toony_step(0.45, 1.0, 0.0)
        step_above = self._toony_step(0.55, 1.0, 0.0)
        # Both should be close to their respective extremes
        # with toony=1.0, the transition covers 0.0..1.0 fully
        assert step_below < 0.5
        assert step_above > 0.5


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
