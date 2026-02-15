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
#   Test CLIP embedding generation and natural language queries.
#
#   Tests the semantic query system with actual radiance assets.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_clip_queries
# PURPOSE:  Tests for clip queries
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   test_clip_embedding_generation(), test_semantic_query_engine(), test_with_real_radiance()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import sys
import time
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_clip_embedding_generation():
    """Test generating CLIP embeddings from semantic labels."""
    print("\n=== Test 1: CLIP Embedding Generation ===\n")

    from noodlestudio.core.semantic_world.semantic_query import (
        CLIPEmbeddingGenerator,
        populate_asset_embeddings,
    )
    from noodlestudio.core.semantic_world.radiance_format import RadianceAsset

    # Create a simple test asset with semantic labels
    asset = RadianceAsset()
    import numpy as np

    # Create 10 test Gaussians
    n = 10
    asset.positions = np.random.randn(n, 3).astype(np.float32)
    asset.scales = np.ones((n, 3), dtype=np.float32) * 0.01
    asset.rotations = np.zeros((n, 4), dtype=np.float32)
    asset.rotations[:, 3] = 1.0  # w = 1 for identity quaternion
    asset.opacities = np.ones(n, dtype=np.float32)
    asset.sh_dc = np.ones((n, 3), dtype=np.float32) * 0.5

    # Add semantic labels
    asset.semantic_labels = [
        "head", "head",
        "leftHand", "leftHand",
        "rightHand", "rightHand",
        "torso", "torso",
        "leftFoot", "rightFoot"
    ]

    print(f"Created test asset with {n} Gaussians")
    print(f"Labels: {set(asset.semantic_labels)}")

    # Generate embeddings
    print("\nGenerating CLIP embeddings...")
    start = time.time()

    generator = CLIPEmbeddingGenerator()
    success = populate_asset_embeddings(asset, generator)

    elapsed = time.time() - start
    print(f"Generation time: {elapsed*1000:.1f}ms")

    if success:
        print(f"Success! Generated {asset.clip_embeddings.shape} embeddings")
        print(f"Embedding dimension: {asset.clip_embeddings.shape[1]}")

        # Check that similar labels have similar embeddings
        head_emb = asset.clip_embeddings[0]  # first head
        hand_emb = asset.clip_embeddings[2]  # left hand

        similarity = np.dot(head_emb, hand_emb)
        print(f"Head-Hand similarity: {similarity:.3f}")

        # Same label should have identical embeddings
        head1 = asset.clip_embeddings[0]
        head2 = asset.clip_embeddings[1]
        same_similarity = np.dot(head1, head2)
        print(f"Head-Head similarity: {same_similarity:.3f} (should be 1.0)")

        return True
    else:
        print("Failed to generate embeddings")
        return False


def test_semantic_query_engine():
    """Test the full semantic query engine with natural language."""
    print("\n=== Test 2: Semantic Query Engine ===\n")

    from noodlestudio.core.semantic_world.semantic_query import (
        SemanticQueryEngine,
    )
    from noodlestudio.core.semantic_world.radiance_format import RadianceAsset
    import numpy as np

    # Create test asset
    asset = RadianceAsset()
    n = 20
    asset.positions = np.random.randn(n, 3).astype(np.float32)
    asset.scales = np.ones((n, 3), dtype=np.float32) * 0.01
    asset.rotations = np.zeros((n, 4), dtype=np.float32)
    asset.rotations[:, 3] = 1.0
    asset.opacities = np.ones(n, dtype=np.float32)
    asset.sh_dc = np.ones((n, 3), dtype=np.float32) * 0.5

    # Diverse semantic labels
    asset.semantic_labels = [
        "head", "head", "head", "head",
        "leftHand", "leftHand", "leftHand",
        "rightHand", "rightHand", "rightHand",
        "torso", "torso", "torso", "torso",
        "leftFoot", "leftFoot",
        "rightFoot", "rightFoot",
        "leftArm", "rightArm"
    ]

    # Create engine with auto-generation
    print("Creating SemanticQueryEngine with auto-generation...")
    engine = SemanticQueryEngine(auto_generate_embeddings=True)

    # Register entity (should auto-generate embeddings)
    print("Registering test entity 'alicia'...")
    engine.register_entity('alicia', asset, display_name='Alicia')

    print(f"Asset now has CLIP embeddings: {asset.has_clip}")

    # Test queries
    queries = [
        "hand",
        "left hand",
        "head",
        "foot",
        "arm",
        "the character's head",
    ]

    print("\n--- Query Results ---")
    for query in queries:
        result = engine.query_text(query, top_k=3)
        print(f"\nQuery: '{query}' ({result.search_time_ms:.1f}ms)")
        for match in result.matches[:3]:
            print(f"  - {match.body_part}: {match.similarity:.3f}")

    return True


def test_with_real_radiance():
    """Test with actual radiance file if available."""
    print("\n=== Test 3: Real Radiance Asset ===\n")

    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent.parent.parent.parent.parent / "external/vrm_samples/alicia_textured.radiance",
        Path("/Users/caitlyn/git/noodlings_clean/external/vrm_samples/alicia_textured.radiance"),
    ]
    radiance_path = None
    for p in possible_paths:
        if p.exists():
            radiance_path = p
            break

    if radiance_path is None:
        print(f"Radiance file not found in any of: {[str(p) for p in possible_paths]}")
        print("Skipping real asset test")
        return True


    from noodlestudio.core.semantic_world.radiance_format import load_radiance
    from noodlestudio.core.semantic_world.semantic_query import (
        SemanticQueryEngine,
    )

    print(f"Loading: {radiance_path}")
    asset = load_radiance(str(radiance_path))

    print(f"Gaussians: {asset.gaussian_count}")
    print(f"Has skeleton: {asset.has_skeleton}")
    print(f"Has semantics: {asset.has_semantics}")
    print(f"Has CLIP (before): {asset.has_clip}")

    if asset.semantic_labels:
        unique = set(asset.semantic_labels[:100])  # Sample first 100
        print(f"Sample labels: {unique}")

    # Create engine and register
    engine = SemanticQueryEngine(auto_generate_embeddings=True)
    engine.register_entity('alicia', asset, display_name='Alicia')

    print(f"Has CLIP (after): {asset.has_clip}")

    if asset.has_clip:
        # Test queries
        queries = ["head", "left hand", "torso", "foot"]
        print("\n--- Query Results ---")
        for query in queries:
            result = engine.query_text(query, top_k=3)
            print(f"\nQuery: '{query}' ({result.search_time_ms:.1f}ms)")
            for match in result.matches[:3]:
                print(f"  - {match.body_part}: {match.similarity:.3f} @ {match.position}")

    return True


if __name__ == '__main__':
    print("=" * 60)
    print("CLIP Embedding & Query Test Suite")
    print("=" * 60)

    all_passed = True

    # Test 1: Basic embedding generation
    try:
        if not test_clip_embedding_generation():
            all_passed = False
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: Semantic query engine
    try:
        if not test_semantic_query_engine():
            all_passed = False
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 3: Real radiance file
    try:
        if not test_with_real_radiance():
            all_passed = False
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
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
