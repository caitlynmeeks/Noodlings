# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.test_scene_protocol
# PURPOSE:  Test Scene Protocol with semantic queries
# LAYER:    Backend / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Test Scene Protocol Integration with Semantic Queries.

Tests the wiring between:
- Scene Protocol (SceneStateManager)
- Gaussian Adapter (RadianceAsset loading)
- Semantic Query Engine (CLIP natural language)
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "noodlestudio"))

def test_imports():
    """Test that all imports work."""
    print("=== Test 1: Imports ===\n")

    from scene_protocol_integration import (
        SCENE_PROTOCOL_AVAILABLE,
        GAUSSIAN_ADAPTER_AVAILABLE,
        SEMANTIC_QUERY_AVAILABLE,
    )

    print(f"Scene Protocol available: {SCENE_PROTOCOL_AVAILABLE}")
    print(f"Gaussian Adapter available: {GAUSSIAN_ADAPTER_AVAILABLE}")
    print(f"Semantic Query available: {SEMANTIC_QUERY_AVAILABLE}")

    return SEMANTIC_QUERY_AVAILABLE


def test_init_semantic_engine():
    """Test initializing the semantic query engine."""
    print("\n=== Test 2: Initialize Semantic Engine ===\n")

    from scene_protocol_integration import (
        init_semantic_query_engine,
        get_semantic_query_engine,
    )

    success = init_semantic_query_engine()
    print(f"Initialization: {'SUCCESS' if success else 'FAILED'}")

    engine = get_semantic_query_engine()
    print(f"Engine acquired: {engine is not None}")

    return success


def test_register_radiance():
    """Test registering a radiance asset."""
    print("\n=== Test 3: Register Radiance Asset ===\n")

    from scene_protocol_integration import register_entity_radiance

    # Use our test radiance file
    radiance_path = Path(__file__).parent.parent.parent / "external/vrm_samples/alicia_textured.radiance"

    if not radiance_path.exists():
        print(f"Radiance file not found: {radiance_path}")
        print("Skipping this test")
        return True

    success = register_entity_radiance(
        entity_id="alicia",
        radiance_path=str(radiance_path),
        display_name="Alicia",
        entity_type="noodling"
    )

    print(f"Registration: {'SUCCESS' if success else 'FAILED'}")
    return success


def test_semantic_query():
    """Test running semantic queries."""
    print("\n=== Test 4: Semantic Queries ===\n")

    from scene_protocol_integration import query_scene_semantic

    queries = ["head", "left hand", "torso", "foot"]

    for query in queries:
        result = query_scene_semantic(query, top_k=3)

        if result:
            print(f"\nQuery: '{query}' ({result['search_time_ms']:.1f}ms)")
            for match in result['matches'][:3]:
                print(f"  - {match['body_part']}: {match['similarity']:.3f}")
        else:
            print(f"\nQuery: '{query}' - FAILED (no result)")
            return False

    return True


def test_visible_body_parts():
    """Test computing visible body parts."""
    print("\n=== Test 5: Visible Body Parts ===\n")

    from scene_protocol_integration import (
        get_entity_visible_body_parts,
        init_scene_state_manager,
    )

    # Initialize scene state manager
    init_scene_state_manager("test", "Test Scene")

    # Test with manual positions
    visible = get_entity_visible_body_parts(
        perceiver_id="player1",
        target_id="alicia",
        perceiver_pos=[0, 1.5, 2],
        perceiver_facing=[0, 0, -1],
        fov=120.0
    )

    print(f"Visible body parts: {len(visible)}")
    if visible:
        print(f"Sample parts: {visible[:10]}")

    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Scene Protocol Semantic Integration Tests")
    print("=" * 60)

    all_passed = True

    # Test 1
    try:
        if not test_imports():
            print("\nSemantic Query not available - skipping further tests")
            sys.exit(0)
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        all_passed = False

    # Test 2
    try:
        if not test_init_semantic_engine():
            all_passed = False
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        all_passed = False

    # Test 3
    try:
        if not test_register_radiance():
            all_passed = False
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        all_passed = False

    # Test 4
    try:
        if not test_semantic_query():
            all_passed = False
    except Exception as e:
        print(f"Test 4 FAILED: {e}")
        all_passed = False

    # Test 5
    try:
        if not test_visible_body_parts():
            all_passed = False
    except Exception as e:
        print(f"Test 5 FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
