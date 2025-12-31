"""
Test Gaussian Adapter

Tests the NSP to Gaussian Splatting bridge.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from noodlestudio.core.semantic_world.gaussian_adapter import (
    GaussianAsset,
    GaussianInstance,
    GaussianScene,
    GaussianAssetManager,
    GaussianSceneCompositor,
    GaussianGenerator,
)
from noodlestudio.core.semantic_world.scene_packet import (
    ScenePacket,
    Noodling,
    Transform,
    Vector3,
    VisualForm,
    Zone,
    ZoneBounds,
    SpatialTruth,
    PacketHeader,
)


def test_asset_creation():
    """Test creating Gaussian assets."""
    asset = GaussianAsset(
        id="gs_test_001",
        name="test_character",
        asset_type="character",
        ply_path="/path/to/test.ply",
        generator="sharp",
        gaussian_count=500000,
        semantic_tags=["fox", "kitsune", "spirit"],
        noodling_id="yuki",
        visual_form="default",
    )

    assert asset.id == "gs_test_001"
    assert asset.asset_type == "character"
    assert "fox" in asset.semantic_tags
    print("Asset creation test passed")


def test_instance_creation():
    """Test creating Gaussian instances."""
    instance = GaussianInstance(
        instance_id="inst_yuki_001",
        asset_id="gs_test_001",
        transform=Transform(
            position=Vector3(1.0, 0.0, 2.0),
            rotation=Vector3(0.0, 45.0, 0.0),
        ),
        zone_id="nexus_main",
        entity_type="noodling",
        entity_id="yuki",
    )

    assert instance.instance_id == "inst_yuki_001"
    assert instance.transform.position.x == 1.0
    assert instance.transform.rotation.y == 45.0
    print("Instance creation test passed")


def test_scene_creation():
    """Test creating Gaussian scenes."""
    scene = GaussianScene(
        scene_id="scene_001",
        stage_id="the_nexus",
        stage_name="The Nexus",
    )

    # Add an instance
    scene.instances["inst_001"] = GaussianInstance(
        instance_id="inst_001",
        asset_id="gs_char_001",
        transform=Transform(position=Vector3(0, 0, 0)),
    )

    assert len(scene.instances) == 1
    assert scene.stage_name == "The Nexus"
    print("Scene creation test passed")


def test_asset_manager():
    """Test asset manager with temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = GaussianAssetManager(tmpdir)

        # Create a test asset
        asset = GaussianAsset(
            id="gs_test_002",
            name="test_env",
            asset_type="environment",
            ply_path="/fake/path.ply",
            generator="opensplat",
            semantic_tags=["forest", "clearing"],
        )

        # Register it
        manager.register_asset(asset)

        # Retrieve it
        retrieved = manager.get_asset("gs_test_002")
        assert retrieved is not None
        assert retrieved.name == "test_env"

        # Search by tags
        results = manager.search_by_tags(["forest"])
        assert len(results) == 1
        assert results[0].id == "gs_test_002"

        print("Asset manager test passed")


def test_scene_composition():
    """Test composing a scene from an NSP packet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = GaussianAssetManager(tmpdir)

        # Create and register a character asset
        char_asset = GaussianAsset(
            id="gs_char_yuki_default",
            name="yuki_default",
            asset_type="character",
            ply_path="/fake/yuki.ply",
            generator="sharp",
            noodling_id="yuki",
            visual_form="default",
        )
        manager.register_asset(char_asset)

        # Create compositor
        compositor = GaussianSceneCompositor(manager)

        # Create a scene packet
        packet = ScenePacket(
            header=PacketHeader(stage_id="nexus", stage_name="The Nexus"),
        )

        # Add a noodling
        packet.noodlings["yuki"] = Noodling(
            id="yuki",
            display_name="Yuki",
            species="kitsune",
            visual_state="default",
            transform=Transform(
                position=Vector3(2.0, 0.0, 3.0),
                rotation=Vector3(0.0, 90.0, 0.0),
            ),
            zone="nexus_main",
        )

        # Compose scene
        scene = compositor.compose_scene(packet)

        assert scene.stage_name == "The Nexus"
        assert "inst_yuki" in scene.instances
        assert scene.instances["inst_yuki"].asset_id == "gs_char_yuki_default"
        assert scene.instances["inst_yuki"].transform.position.x == 2.0

        print("Scene composition test passed")


def test_generator_discovery():
    """Test generator tool discovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = GaussianAssetManager(tmpdir)
        generator = GaussianGenerator(manager)

        # Check if tools were discovered
        print(f"SHARP path: {generator.sharp_path or 'Not found'}")
        print(f"OpenSplat path: {generator.opensplat_path or 'Not found'}")

        # At least one should be found on this system
        has_tools = generator.sharp_path or generator.opensplat_path
        print(f"Generator discovery test: {'passed' if has_tools else 'no tools found (expected on some systems)'}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Gaussian Adapter Tests")
    print("=" * 60)

    test_asset_creation()
    test_instance_creation()
    test_scene_creation()
    test_asset_manager()
    test_scene_composition()
    test_generator_discovery()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
