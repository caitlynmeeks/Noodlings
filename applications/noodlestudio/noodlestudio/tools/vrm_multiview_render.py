"""
VRM Multi-View Renderer - Generate training views for Gaussian splatting.

Renders a VRM avatar from multiple camera angles and outputs in nerfstudio format
(transforms.json + images) for training with OpenSplat.

The pipeline:
1. Parse VRM and extract mesh/textures
2. Render from N orbit positions around the model
3. Output transforms.json with camera poses
4. Output initial point cloud from mesh vertices (reconstruction.ply)

Author: Caitlyn + Claude
Date: December 2025
"""

import json
import logging
import math
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraParams:
    """Camera parameters for a single view."""
    # Image dimensions
    width: int
    height: int
    # Intrinsics
    focal_x: float
    focal_y: float
    cx: float
    cy: float
    # Extrinsics (camera-to-world transform)
    transform_matrix: np.ndarray  # 4x4
    # File path (relative)
    file_path: str


def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """
    Create a camera-to-world matrix looking from eye toward target.

    Uses OpenGL convention: camera looks down -Z axis.
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    # Forward vector (from eye to target)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Right vector
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 0.001:
        # Looking straight up/down, use different up
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recompute up to ensure orthogonal
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix (OpenGL: -Z is forward)
    # Columns are right, up, -forward (camera axes in world space)
    rotation = np.eye(4)
    rotation[:3, 0] = right
    rotation[:3, 1] = up
    rotation[:3, 2] = -forward  # Camera looks down -Z
    rotation[:3, 3] = eye

    return rotation


def create_orbit_cameras(
    num_views: int = 64,
    num_elevations: int = 3,
    distance: float = 2.5,
    target: Tuple[float, float, float] = (0.0, 0.85, 0.0),
    elevation_range: Tuple[float, float] = (0, 30),
    width: int = 1024,
    height: int = 1024,
    fov_degrees: float = 50.0,
) -> List[CameraParams]:
    """
    Create orbit cameras around a target point.

    Args:
        num_views: Total views per elevation ring
        num_elevations: Number of elevation levels
        distance: Camera distance from target
        target: Point to orbit around (typically model center)
        elevation_range: (min, max) elevation in degrees
        width, height: Image dimensions
        fov_degrees: Field of view

    Returns:
        List of CameraParams for each view
    """
    cameras = []

    # Compute focal length from FOV
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))

    view_idx = 0
    elevations = np.linspace(elevation_range[0], elevation_range[1], num_elevations)
    views_per_elevation = num_views // num_elevations

    target_arr = np.array(target)

    for elev_deg in elevations:
        elev_rad = math.radians(elev_deg)

        for i in range(views_per_elevation):
            azimuth_deg = (360.0 * i / views_per_elevation)
            azimuth_rad = math.radians(azimuth_deg)

            # Camera position (spherical to cartesian, Y-up)
            # At elev=0, camera is at target height
            # At elev=30, camera is above target
            x = distance * math.cos(elev_rad) * math.sin(azimuth_rad)
            y = distance * math.sin(elev_rad)
            z = distance * math.cos(elev_rad) * math.cos(azimuth_rad)

            cam_pos = np.array([
                target[0] + x,
                target[1] + y,
                target[2] + z
            ])

            # Build camera-to-world matrix
            transform = look_at_matrix(cam_pos, target_arr)

            camera = CameraParams(
                width=width,
                height=height,
                focal_x=focal,
                focal_y=focal,
                cx=width / 2.0,
                cy=height / 2.0,
                transform_matrix=transform,
                file_path=f"images/frame_{view_idx:05d}.png",
            )
            cameras.append(camera)
            view_idx += 1

    return cameras


def cameras_to_transforms_json(cameras: List[CameraParams]) -> Dict[str, Any]:
    """Convert camera list to nerfstudio transforms.json format."""
    frames = []

    for cam in cameras:
        frame = {
            "file_path": cam.file_path,
            "w": cam.width,
            "h": cam.height,
            "fl_x": cam.focal_x,
            "fl_y": cam.focal_y,
            "cx": cam.cx,
            "cy": cam.cy,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
            "transform_matrix": cam.transform_matrix.tolist(),
        }
        frames.append(frame)

    return {
        "camera_model": "OPENCV",
        "ply_file_path": "reconstruction.ply",
        "frames": frames,
    }


def render_vrm_multiview(
    vrm_path: str,
    output_dir: str,
    num_views: int = 64,
    num_elevations: int = 3,
    resolution: int = 1024,
    distance: float = 2.5,
    fov_degrees: float = 45.0,
    background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> str:
    """
    Render a VRM model from multiple views for Gaussian splatting training.

    Args:
        vrm_path: Path to input VRM file
        output_dir: Directory to write output (images/ + transforms.json)
        num_views: Total number of views
        num_elevations: Number of elevation levels
        resolution: Image resolution (square)
        distance: Camera distance
        fov_degrees: Field of view
        background: Background color (RGB 0-1)

    Returns:
        Path to output directory
    """
    import torch

    # Try to import rendering dependencies
    try:
        import trimesh
        import pyrender
        HAS_PYRENDER = True
    except ImportError:
        HAS_PYRENDER = False
        logger.warning("pyrender not available, trying software rendering")

    vrm_path = Path(vrm_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)

    logger.info(f"Rendering VRM multi-view: {vrm_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Views: {num_views} ({num_elevations} elevations)")
    logger.info(f"  Resolution: {resolution}x{resolution}")

    # Parse VRM to get mesh
    from ..core.semantic_world.vrm_parser import parse_vrm
    avatar = parse_vrm(str(vrm_path))

    # Compute model bounds for camera placement
    all_verts = np.vstack([m.vertices for m in avatar.meshes])
    model_min = all_verts.min(axis=0)
    model_max = all_verts.max(axis=0)
    model_center = (model_min + model_max) / 2.0
    model_height = model_max[1] - model_min[1]
    model_width = max(model_max[0] - model_min[0], model_max[2] - model_min[2])
    model_size = max(model_height, model_width)

    # Target slightly above center (chest height for humanoids)
    target = (float(model_center[0]), float(model_min[1] + model_height * 0.5), float(model_center[2]))

    # Adjust distance based on model size and FOV
    # We want the full model to fit in frame
    if distance <= 0:
        # Calculate distance needed to fit model in view
        half_fov = math.radians(fov_degrees) / 2.0
        distance = (model_size / 2.0) / math.tan(half_fov) * 1.3  # 1.3x for margin

    logger.info(f"  Model: height={model_height:.2f}m, size={model_size:.2f}m")
    logger.info(f"  Target: {target}")
    logger.info(f"  Camera distance: {distance:.2f}m")

    # Create orbit cameras
    cameras = create_orbit_cameras(
        num_views=num_views,
        num_elevations=num_elevations,
        distance=distance,
        target=target,
        width=resolution,
        height=resolution,
        fov_degrees=fov_degrees,
    )

    if HAS_PYRENDER:
        # Use pyrender for GPU-accelerated rendering
        rendered_count = _render_with_pyrender(
            avatar, cameras, output_path, background
        )
    else:
        # Fallback to software rendering
        rendered_count = _render_software(
            avatar, cameras, output_path, background
        )

    logger.info(f"  Rendered {rendered_count} views")

    # Write transforms.json
    transforms = cameras_to_transforms_json(cameras)
    transforms_path = output_path / "transforms.json"
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    logger.info(f"  Wrote: {transforms_path}")

    # Write initial point cloud (reconstruction.ply)
    _write_initial_pointcloud(avatar, output_path / "reconstruction.ply")

    return str(output_path)


def _render_with_pyrender(avatar, cameras, output_path, background):
    """Render using pyrender (GPU accelerated)."""
    import trimesh
    import pyrender
    from PIL import Image
    import numpy as np

    # Build scene from VRM meshes
    scene = pyrender.Scene(bg_color=[*background, 1.0], ambient_light=[0.3, 0.3, 0.3])

    for mesh_idx, mesh in enumerate(avatar.meshes):
        # Create trimesh
        if mesh.indices is not None:
            faces = mesh.indices.reshape(-1, 3)
        else:
            # No indices - assume triangle list
            faces = np.arange(len(mesh.vertices)).reshape(-1, 3)

        # Get vertex colors from texture if available
        vertex_colors = None
        if mesh.uvs is not None and mesh.material_index is not None:
            # Try to sample texture colors
            vertex_colors = _sample_texture_at_uvs(
                avatar, mesh.material_index, mesh.uvs
            )

        if vertex_colors is None:
            vertex_colors = np.ones((len(mesh.vertices), 4)) * 0.8

        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=faces,
            vertex_colors=vertex_colors,
        )

        # Convert to pyrender mesh
        pm = pyrender.Mesh.from_trimesh(tm, smooth=True)
        scene.add(pm)

    # Add a fixed key light
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    key_light_pose = np.eye(4)
    key_light_pose[:3, :3] = look_at_matrix(
        np.array([2, 3, 2]), np.array([0, 0, 0])
    )[:3, :3]
    scene.add(key_light, pose=key_light_pose)

    # Add fill light
    fill_light = pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=1.0)
    fill_light_pose = np.eye(4)
    fill_light_pose[:3, :3] = look_at_matrix(
        np.array([-2, 1, -1]), np.array([0, 0, 0])
    )[:3, :3]
    scene.add(fill_light, pose=fill_light_pose)

    # Create offscreen renderer
    r = pyrender.OffscreenRenderer(
        cameras[0].width, cameras[0].height
    )

    rendered = 0
    for cam_params in cameras:
        # Create pyrender camera
        yfov = 2.0 * np.arctan(cam_params.height / (2.0 * cam_params.focal_y))
        camera = pyrender.PerspectiveCamera(
            yfov=yfov,
            aspectRatio=cam_params.width / cam_params.height,
        )

        # Camera pose - pyrender expects camera-to-world transform
        # Our look_at_matrix already produces this
        pose = cam_params.transform_matrix.copy()

        # Add camera to scene
        cam_node = scene.add(camera, pose=pose)

        # Render
        color, depth = r.render(scene)

        # Save image
        img_path = output_path / cam_params.file_path
        Image.fromarray(color).save(img_path)

        # Remove camera for next iteration
        scene.remove_node(cam_node)

        rendered += 1
        if rendered % 10 == 0:
            logger.info(f"    Rendered {rendered}/{len(cameras)}")

    r.delete()
    return rendered


def _render_software(avatar, cameras, output_path, background):
    """Software rendering fallback using our Gaussian renderer."""
    from ..core.gaussian_renderer import GaussianRenderer, GaussianCamera
    from ..core.radiance_component import RadianceComponent
    from .vrm_to_radiance import vrm_to_radiance
    from PIL import Image
    import tempfile
    import numpy as np

    # First convert VRM to radiance (densified)
    with tempfile.NamedTemporaryFile(suffix='.radiance', delete=False) as tmp:
        tmp_path = tmp.name

    logger.info("  Converting VRM to Gaussians for software rendering...")
    asset = vrm_to_radiance(
        str(avatar),  # This won't work - we need the path
        output_path=tmp_path,
        densify=True,
        use_adaptive_scale=True,
    )

    # Use our Gaussian renderer
    component = RadianceComponent('render')
    component._asset = asset

    renderer = GaussianRenderer()

    rendered = 0
    for cam_params in cameras:
        # Convert camera params to our format
        camera = GaussianCamera(
            width=cam_params.width,
            height=cam_params.height,
            fx=cam_params.focal_x,
            fy=cam_params.focal_y,
            cx=cam_params.cx,
            cy=cam_params.cy,
            c2w=cam_params.transform_matrix,
        )

        image, alpha, info = renderer.render_component(
            component, camera, background=background
        )

        # Save image
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_path = output_path / cam_params.file_path
        Image.fromarray(img_np).save(img_path)

        rendered += 1
        if rendered % 10 == 0:
            logger.info(f"    Rendered {rendered}/{len(cameras)}")

    return rendered


def _sample_texture_at_uvs(avatar, material_idx: int, uvs: np.ndarray) -> Optional[np.ndarray]:
    """Sample texture colors at UV coordinates."""
    from PIL import Image
    import io

    if material_idx >= len(avatar.materials):
        return None

    material = avatar.materials[material_idx]
    if material.diffuse_texture is None:
        # Return diffuse color
        color = material.diffuse_color
        return np.tile(color, (len(uvs), 1))

    tex_idx = material.diffuse_texture
    if tex_idx >= len(avatar.textures):
        return None

    # Load texture
    tex_data = avatar.textures[tex_idx]
    try:
        img = Image.open(io.BytesIO(tex_data)).convert('RGBA')
        img_arr = np.array(img) / 255.0
    except Exception:
        return None

    # Sample at UVs
    h, w = img_arr.shape[:2]
    u = np.clip(uvs[:, 0], 0, 1) * (w - 1)
    v = np.clip(1 - uvs[:, 1], 0, 1) * (h - 1)  # Flip V

    # Nearest neighbor sampling
    u_i = u.astype(int)
    v_i = v.astype(int)

    colors = img_arr[v_i, u_i]
    return colors


def _write_initial_pointcloud(avatar, output_path: Path):
    """Write initial point cloud in PLY format for OpenSplat initialization."""
    # Collect all vertices
    all_positions = []
    all_colors = []

    for mesh in avatar.meshes:
        all_positions.append(mesh.vertices)

        # Get colors
        if mesh.uvs is not None and mesh.material_index is not None:
            colors = _sample_texture_at_uvs(avatar, mesh.material_index, mesh.uvs)
            if colors is not None:
                all_colors.append((colors[:, :3] * 255).astype(np.uint8))
            else:
                all_colors.append(np.full((len(mesh.vertices), 3), 180, dtype=np.uint8))
        else:
            all_colors.append(np.full((len(mesh.vertices), 3), 180, dtype=np.uint8))

    positions = np.vstack(all_positions).astype(np.float32)
    colors = np.vstack(all_colors).astype(np.uint8)

    # Write binary PLY
    n_points = len(positions)

    header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n_points):
            f.write(struct.pack('<fff', *positions[i]))
            f.write(struct.pack('<BBB', *colors[i]))

    logger.info(f"  Wrote initial point cloud: {output_path} ({n_points:,} points)")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Render VRM from multiple views for Gaussian splatting training"
    )
    parser.add_argument("vrm", help="Path to VRM file")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-n", "--num-views", type=int, default=64,
                       help="Number of views (default: 64)")
    parser.add_argument("--elevations", type=int, default=3,
                       help="Number of elevation levels (default: 3)")
    parser.add_argument("-r", "--resolution", type=int, default=1024,
                       help="Image resolution (default: 1024)")
    parser.add_argument("-d", "--distance", type=float, default=0,
                       help="Camera distance (0 = auto)")
    parser.add_argument("--fov", type=float, default=45.0,
                       help="Field of view in degrees (default: 45)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(message)s'
    )

    vrm_path = Path(args.vrm)
    if not vrm_path.exists():
        print(f"Error: VRM file not found: {vrm_path}")
        exit(1)

    if args.output:
        output_dir = args.output
    else:
        output_dir = str(vrm_path.parent / f"{vrm_path.stem}_views")

    result = render_vrm_multiview(
        vrm_path=str(vrm_path),
        output_dir=output_dir,
        num_views=args.num_views,
        num_elevations=args.elevations,
        resolution=args.resolution,
        distance=args.distance,
        fov_degrees=args.fov,
    )

    print(f"Output: {result}")
    print(f"Next step: opensplat {result} -o {vrm_path.stem}_trained.ply -n 7000")
