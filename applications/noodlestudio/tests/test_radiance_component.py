"""
Test RadianceComponent System

Tests the complete pipeline:
1. RadianceComponent - loading, transforms, overrides
2. RadianceSceneBuilder - multi-asset composition
3. GaussianRenderer - batch rendering
4. RadianceAPI - scripting interface

Run with:
    cd applications/noodlestudio
    PYTHONPATH=.:../.. python tests/test_radiance_component.py
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np


def find_test_radiance():
    """Find a test .radiance file."""
    candidates = [
        Path(__file__).parent.parent.parent.parent / "external/vrm_samples/alicia_textured.radiance",
        Path(__file__).parent.parent.parent / "cmush/world/radiances/alicia.radiance",
        Path.home() / "git/noodlings_clean/external/vrm_samples/alicia_textured.radiance",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def test_radiance_component_basics():
    """Test RadianceComponent creation and loading."""
    print("\n=== Test 1: RadianceComponent Basics ===\n")

    from noodlestudio.core.radiance_component import (
        RadianceComponent, RenderMode, LightingMode, Color
    )

    # Create component
    component = RadianceComponent(entity_id="test_entity")
    print(f"Created component: {component.entity_id}")
    print(f"Is loaded: {component.is_loaded}")

    # Find and load asset
    radiance_path = find_test_radiance()
    if not radiance_path:
        print("No test .radiance file found - creating synthetic test")
        return test_synthetic_component()

    success = component.load_asset(radiance_path)
    print(f"Loaded asset: {success}")
    print(f"Gaussian count: {component.gaussian_count}")
    print(f"Has skeleton: {component.has_skeleton}")
    print(f"Render mode: {component.render_mode.value}")

    # Test transform
    component.set_position(1.0, 0.0, 2.0)
    component.set_rotation(0, 45, 0)
    print(f"Position set: {component.transform.position}")
    print(f"Rotation set: {component.transform.rotation}")

    # Test material
    component.set_tint(1.0, 0.5, 0.5, 1.0)
    component.set_emission(0.1, 0.0, 0.0)
    print(f"Tint: {component.material.tint.to_tuple()}")
    print(f"Emission: {component.material.emission.to_rgb()}")

    return component


def test_synthetic_component():
    """Test with synthetic data when no .radiance file available."""
    print("Testing with synthetic RadianceAsset...")

    from noodlestudio.core.radiance_component import RadianceComponent
    from noodlestudio.core.semantic_world.radiance_format import RadianceAsset

    # Create synthetic asset
    asset = RadianceAsset()
    n = 1000
    asset.positions = np.random.randn(n, 3).astype(np.float32) * 0.5
    asset.positions[:, 1] += 1.0  # Center around y=1
    asset.scales = np.ones((n, 3), dtype=np.float32) * 0.02
    asset.rotations = np.zeros((n, 4), dtype=np.float32)
    asset.rotations[:, 3] = 1.0  # Identity quaternion
    asset.opacities = np.ones(n, dtype=np.float32)
    asset.sh_dc = np.random.rand(n, 3).astype(np.float32)

    # Create component with synthetic asset
    component = RadianceComponent(entity_id="synthetic_test")
    component._asset = asset
    component._asset_path = "synthetic"

    print(f"Synthetic component created: {component.gaussian_count} Gaussians")
    return component


def test_region_overrides(loaded_radiance_component):
    """Test region-level overrides."""
    component = loaded_radiance_component

    from noodlestudio.core.radiance_component import RegionOverride, Color

    # Get available regions
    regions = component.body_regions
    print(f"Available body regions: {regions}")

    if regions:
        # Override first region
        test_region = list(regions)[0]
        component.set_region_override(test_region, RegionOverride(
            tint=Color(0.5, 0.5, 1.0),
            emission=Color(0.0, 0.0, 0.3),
            alpha_mult=0.8
        ))
        print(f"Set override for '{test_region}'")

        override = component.get_region_override(test_region)
        print(f"  Tint: {override.tint.to_rgb()}")
        print(f"  Emission: {override.emission.to_rgb()}")
        print(f"  Alpha mult: {override.alpha_mult}")
    else:
        print("No body regions available (asset may lack semantics)")

    return True


def test_gaussian_overrides(loaded_radiance_component):
    """Test per-Gaussian overrides."""
    component = loaded_radiance_component

    from noodlestudio.core.radiance_component import GaussianOverride, Color

    # Override some random Gaussians
    indices_to_override = [0, 10, 50, 100]

    for idx in indices_to_override:
        if idx < component.gaussian_count:
            component.set_gaussian_override(idx, GaussianOverride(
                tint=Color(0.2, 0.2, 0.2),  # Darken
                alpha=0.5
            ))

    print(f"Set overrides for {len(indices_to_override)} Gaussians")
    print(f"Total overrides stored: {len(component._gaussian_overrides)}")

    # Test clearing
    component.clear_gaussian_override(0)
    print(f"After clearing one: {len(component._gaussian_overrides)}")

    return True


def test_spatial_queries(loaded_radiance_component):
    """Test spatial query methods."""
    component = loaded_radiance_component

    # Query radius
    center = (0.0, 1.0, 0.0)
    radius = 0.5
    nearby = component.query_radius(center, radius)
    print(f"Gaussians within {radius}m of {center}: {len(nearby)}")

    # Query nearest
    nearest = component.query_nearest(center, k=5)
    print(f"5 nearest to {center}:")
    for idx, dist in nearest:
        print(f"  Index {idx}: {dist:.3f}m")

    # Raycast
    origin = (0, 1, 3)
    direction = (0, 0, -1)
    hit = component.raycast(origin, direction)
    if hit:
        print(f"Raycast hit: index={hit['index']}, distance={hit['distance']:.2f}")
        print(f"  Body part: {hit.get('body_part', 'N/A')}")
    else:
        print("Raycast: no hit")

    return True


def test_render_data(loaded_radiance_component):
    """Test render data export."""
    component = loaded_radiance_component

    render_data = component.get_render_data()

    if render_data:
        print(f"Render data keys: {list(render_data.keys())}")
        print(f"Positions shape: {render_data['positions'].shape}")
        print(f"Colors shape: {render_data['colors'].shape}")
        print(f"Opacities shape: {render_data['opacities'].shape}")
        print(f"Entity ID: {render_data['entity_id']}")

        # Check that overrides were applied
        colors = render_data['colors']
        print(f"Color range: [{colors.min():.2f}, {colors.max():.2f}]")
    else:
        print("No render data (asset not loaded)")

    return render_data is not None


def test_scene_builder():
    """Test RadianceSceneBuilder."""
    print("\n=== Test 6: Scene Builder ===\n")

    from noodlestudio.core.radiance_component import RadianceComponent
    from noodlestudio.core.semantic_world.radiance_scene_builder import (
        RadianceSceneBuilder, get_scene_builder
    )

    builder = RadianceSceneBuilder()

    # Create multiple components
    radiance_path = find_test_radiance()

    if radiance_path:
        # Component 1: Character at origin
        char1 = RadianceComponent(entity_id="character_1")
        char1.load_asset(radiance_path)
        char1.set_position(0, 0, 0)
        builder.add_component(char1)

        # Component 2: Same character offset
        char2 = RadianceComponent(entity_id="character_2")
        char2.load_asset(radiance_path)
        char2.set_position(2, 0, 0)
        char2.set_tint(0.5, 1.0, 0.5)  # Green tint
        builder.add_component(char2)

        print(f"Added 2 components to scene")
    else:
        # Synthetic components
        from noodlestudio.core.semantic_world.radiance_format import RadianceAsset

        for i, offset in enumerate([(0, 0, 0), (2, 0, 0), (0, 0, 2)]):
            asset = RadianceAsset()
            n = 500
            asset.positions = np.random.randn(n, 3).astype(np.float32) * 0.3
            asset.positions += np.array(offset)
            asset.positions[:, 1] += 1.0
            asset.scales = np.ones((n, 3), dtype=np.float32) * 0.02
            asset.rotations = np.zeros((n, 4), dtype=np.float32)
            asset.rotations[:, 3] = 1.0
            asset.opacities = np.ones(n, dtype=np.float32)
            asset.sh_dc = np.random.rand(n, 3).astype(np.float32)

            comp = RadianceComponent(entity_id=f"synthetic_{i}")
            comp._asset = asset
            comp._asset_path = "synthetic"
            builder.add_component(comp)

        print(f"Added 3 synthetic components to scene")

    # Build batch
    batch = builder.build_render_batch()
    if batch:
        print(f"Render batch built:")
        print(f"  Total Gaussians: {batch.total_gaussians}")
        print(f"  Components: {len(batch.components)}")
        print(f"  Positions shape: {batch.positions.shape}")

    # Scene stats
    stats = builder.get_stats()
    print(f"Scene stats: {stats}")

    # Scene raycast
    hit = builder.raycast((0, 1, 5), (0, 0, -1))
    print(f"Scene raycast hit: {hit.hit}")
    if hit.hit:
        print(f"  Entity: {hit.entity_id}")
        print(f"  Distance: {hit.distance:.2f}")

    return batch is not None


def test_renderer():
    """Test GaussianRenderer with batch."""
    print("\n=== Test 7: Batch Rendering ===\n")

    try:
        import torch
    except ImportError:
        print("PyTorch not available - skipping render test")
        return True

    from noodlestudio.core.gaussian_renderer import (
        GaussianRenderer, CameraParams, create_orbit_camera
    )
    from noodlestudio.core.radiance_component import RadianceComponent
    from noodlestudio.core.semantic_world.radiance_scene_builder import RadianceSceneBuilder

    # Create scene
    builder = RadianceSceneBuilder()

    radiance_path = find_test_radiance()
    if radiance_path:
        comp = RadianceComponent(entity_id="render_test")
        comp.load_asset(radiance_path)
        comp.set_tint(1.0, 0.8, 0.8)  # Slight pink
        builder.add_component(comp)
    else:
        # Synthetic
        from noodlestudio.core.semantic_world.radiance_format import RadianceAsset
        asset = RadianceAsset()
        n = 1000
        asset.positions = np.random.randn(n, 3).astype(np.float32) * 0.5
        asset.positions[:, 1] += 1.0
        asset.scales = np.ones((n, 3), dtype=np.float32) * 0.03
        asset.rotations = np.zeros((n, 4), dtype=np.float32)
        asset.rotations[:, 3] = 1.0
        asset.opacities = np.ones(n, dtype=np.float32)
        asset.sh_dc = np.random.rand(n, 3).astype(np.float32)

        comp = RadianceComponent(entity_id="synthetic_render")
        comp._asset = asset
        comp._asset_path = "synthetic"
        builder.add_component(comp)

    # Create renderer and camera
    renderer = GaussianRenderer()
    camera = create_orbit_camera(
        distance=2.5,
        elevation=15,
        azimuth=30,
        target=(0, 0.8, 0),
        width=256,
        height=256
    )

    # Render scene
    image, alpha, info = renderer.render_scene(builder, camera, background=(0.1, 0.1, 0.15))

    print(f"Rendered image: {image.shape}")
    print(f"Visible Gaussians: {info['visible']}/{info['total']}")
    print(f"Components: {info.get('components', 'N/A')}")
    print(f"Device: {info['device']}")

    # Save test image
    try:
        from PIL import Image
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        output_path = Path(__file__).parent / "test_render_output.png"
        pil_img.save(output_path)
        print(f"Saved test render to: {output_path}")
    except ImportError:
        print("PIL not available - skipping image save")

    return True


def test_scripting_api():
    """Test RadianceAPI scripting interface."""
    print("\n=== Test 8: Scripting API ===\n")

    from noodlestudio.scripting.radiance_api import RadianceAPI, get_radiance_api

    api = get_radiance_api()

    # Create component via API
    radiance_path = find_test_radiance()
    if radiance_path:
        wrapper = api.create("api_test", radiance_path)
    else:
        # Create with synthetic
        from noodlestudio.core.radiance_component import RadianceComponent
        from noodlestudio.core.semantic_world.radiance_format import RadianceAsset

        asset = RadianceAsset()
        n = 500
        asset.positions = np.random.randn(n, 3).astype(np.float32) * 0.5
        asset.positions[:, 1] += 1.0
        asset.scales = np.ones((n, 3), dtype=np.float32) * 0.02
        asset.rotations = np.zeros((n, 4), dtype=np.float32)
        asset.rotations[:, 3] = 1.0
        asset.opacities = np.ones(n, dtype=np.float32)
        asset.sh_dc = np.random.rand(n, 3).astype(np.float32)

        comp = RadianceComponent(entity_id="api_test")
        comp._asset = asset
        comp._asset_path = "synthetic"
        wrapper = api.register(comp)

    print(f"Created via API: {wrapper.entity_id}")
    print(f"Gaussian count: {wrapper.gaussian_count}")

    # Test JS-friendly methods
    wrapper.set_tint(1.0, 0.5, 0.5)
    print(f"Tint set: {wrapper.get_tint()}")

    wrapper.set_position(1.0, 0.0, 2.0)
    print(f"Position: {wrapper.get_position()}")

    # Test region override (JS-style dict)
    wrapper.set_region_override("head", {
        'tint': {'r': 0.8, 'g': 0.8, 'b': 1.0},
        'emission': {'r': 0.0, 'g': 0.0, 'b': 0.1}
    })
    print("Set region override for 'head'")

    # List entities
    entities = api.list_entities()
    print(f"Registered entities: {entities}")

    # Scene access
    scene = api.scene
    stats = scene.get_stats()
    print(f"Scene stats: {stats}")

    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RadianceComponent System Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Basics
    try:
        component = test_radiance_component_basics()
        if component is None:
            all_passed = False
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        component = None

    # Test 2: Region overrides
    if component:
        try:
            test_region_overrides(component)
        except Exception as e:
            print(f"Test 2 FAILED: {e}")
            all_passed = False

    # Test 3: Gaussian overrides
    if component:
        try:
            test_gaussian_overrides(component)
        except Exception as e:
            print(f"Test 3 FAILED: {e}")
            all_passed = False

    # Test 4: Spatial queries
    if component:
        try:
            test_spatial_queries(component)
        except Exception as e:
            print(f"Test 4 FAILED: {e}")
            all_passed = False

    # Test 5: Render data
    if component:
        try:
            test_render_data(component)
        except Exception as e:
            print(f"Test 5 FAILED: {e}")
            all_passed = False

    # Test 6: Scene builder
    try:
        test_scene_builder()
    except Exception as e:
        print(f"Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 7: Renderer
    try:
        test_renderer()
    except Exception as e:
        print(f"Test 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 8: Scripting API
    try:
        test_scripting_api()
    except Exception as e:
        print(f"Test 8 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
