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
#   Training API - Scripting interface for Gaussian splat training.
#
#   Provides methods for: - Camera view generation (importanc...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.scripting.training_api
# PURPOSE:  Training Api
# LAYER:    Studio / Scripting API
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TrainingAPI
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, List, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class TrainingAPI:
    """
    Training API for ScriptedFacet context.

    Access via: context.noodle.training
    """

    def __init__(self, context: Any = None):
        self._context = context

    async def detectHeadPosition(self, vrm_path: str) -> List[float]:
        """
        Auto-detect head bone world position from VRM.

        Args:
            vrm_path: Path to VRM file

        Returns:
            [x, y, z] world position of head bone
        """
        try:
            from noodlestudio.core.semantic_world.vrm_parser import VRMParser

            parser = VRMParser()
            vrm_data = parser.parse(vrm_path)

            if vrm_data.skeleton and vrm_data.skeleton.humanoid_map:
                head_idx = vrm_data.skeleton.humanoid_map.get('head')
                if head_idx is not None and head_idx < len(vrm_data.skeleton.bones):
                    # Accumulate position up parent chain
                    pos = [0.0, 0.0, 0.0]
                    current_idx = head_idx
                    visited = set()
                    while current_idx >= 0 and current_idx < len(vrm_data.skeleton.bones):
                        if current_idx in visited:
                            break
                        visited.add(current_idx)
                        b = vrm_data.skeleton.bones[current_idx]
                        pos[0] += b.position[0]
                        pos[1] += b.position[1]
                        pos[2] += b.position[2]
                        current_idx = b.parent_index
                    return pos

        except Exception as e:
            logger.warning(f"Could not detect head from VRM: {e}")

        # Default fallback
        return [0.0, 1.35, 0.0]

    async def generateFaceDetailCameras(
        self,
        head_position: List[float],
        output_dir: str,
        body_views: int = 24,
        face_views: int = 48,
        detail_views: int = 8,
    ) -> List[Dict]:
        """
        Generate importance-weighted camera views for face detail training.

        Args:
            head_position: [x, y, z] world position of head bone
            output_dir: Directory to write transforms.json
            body_views: Number of full-body turntable views
            face_views: Number of face-focused orbital views
            detail_views: Views per detail region (lips, eyes, brow)

        Returns:
            List of camera view dicts with region info
        """
        from noodlestudio.tools.face_detail_camera import FaceDetailCameraGenerator

        generator = FaceDetailCameraGenerator()
        generator.set_head_position(head_position)

        views = generator.generate_views(
            body_views=body_views,
            face_views=face_views,
            detail_views_per_region=detail_views,
        )

        from pathlib import Path
        transforms_path = Path(output_dir) / "transforms.json"
        generator.export_transforms(str(transforms_path), views)

        # Return view info
        return [
            {
                'region': v.region,
                'distance': v.distance,
                'azimuth': v.azimuth,
                'elevation': v.elevation,
            }
            for v in views
        ]

    async def renderVRMViews(
        self,
        vrm_path: str,
        transforms_path: str,
        output_dir: str,
        resolution: int = 1024,
        onProgress: Optional[Callable] = None,
    ) -> Dict:
        """
        Render VRM from camera views defined in transforms.json.

        Args:
            vrm_path: Path to VRM file
            transforms_path: Path to transforms.json
            output_dir: Directory to write rendered images
            resolution: Image resolution (square)
            onProgress: Optional callback(progress_0_to_1)

        Returns:
            Dict with output_dir and count
        """
        import json
        import numpy as np
        from pathlib import Path
        from PIL import Image

        from noodlestudio.core.radiance_component import RadianceComponent
        from noodlestudio.core.gaussian_renderer import GaussianRenderer, GaussianCamera
        from noodlestudio.tools.vrm_to_radiance import vrm_to_radiance
        from noodlestudio.core.semantic_world.radiance_format import save_radiance

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load transforms
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)

        frames = transforms.get('frames', [])

        # Convert VRM to radiance for rendering
        temp_radiance = output_path.parent / "temp_render.radiance"
        asset = vrm_to_radiance(
            vrm_path,
            entity_id="render_ref",
            display_name="Render Reference",
            densify=True,
        )
        save_radiance(asset, str(temp_radiance))

        # Load as component
        component = RadianceComponent("render_ref")
        component.load_asset(str(temp_radiance))

        # Create renderer
        renderer = GaussianRenderer()

        # Render each frame
        for i, frame in enumerate(frames):
            matrix = np.array(frame['transform_matrix'])
            eye = matrix[:3, 3]
            forward = -matrix[:3, 2]
            target = eye + forward

            camera = GaussianCamera(
                width=resolution,
                height=resolution,
                fov_y=50.0,
                position=tuple(eye),
                target=tuple(target),
                up=(0, 1, 0),
            )

            image, alpha, info = renderer.render_component(component, camera)

            img_path = output_path / f"frame_{i:04d}.png"
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
                img.save(str(img_path))
            else:
                image.save(str(img_path))

            if onProgress and i % 5 == 0:
                onProgress((i + 1) / len(frames))

        # Cleanup
        temp_radiance.unlink(missing_ok=True)

        return {
            'output_dir': str(output_path),
            'count': len(frames),
        }

    async def train(
        self,
        dataset_path: str,
        iterations: int = 30000,
        output_path: Optional[str] = None,
        continue_from: Optional[str] = None,
        transforms_override: Optional[str] = None,
        sh_degree: int = 2,
        onProgress: Optional[Callable] = None,
    ) -> Dict:
        """
        Run OpenSplat Gaussian training.

        Args:
            dataset_path: Path to dataset directory (with transforms.json and images/)
            iterations: Number of training iterations
            output_path: Path for output PLY (default: dataset_path/output.ply)
            continue_from: Optional PLY to continue training from
            transforms_override: Optional path to alternative transforms.json
            sh_degree: Spherical harmonics degree (0-3)
            onProgress: Optional callback(progress_0_to_1)

        Returns:
            Dict with output_path and success status
        """
        import subprocess
        import shutil
        from pathlib import Path

        dataset_dir = Path(dataset_path)
        if output_path is None:
            output_path = str(dataset_dir / "output.ply")

        # Find OpenSplat
        opensplat = self._find_opensplat()
        if not opensplat:
            raise RuntimeError("OpenSplat not found")

        # Handle transforms override
        original_transforms = None
        if transforms_override:
            original_transforms = dataset_dir / "transforms.json"
            backup = dataset_dir / "transforms_backup.json"
            shutil.copy(original_transforms, backup)
            shutil.copy(transforms_override, original_transforms)

        try:
            cmd = [
                opensplat,
                str(dataset_dir),
                "-o", output_path,
                "-n", str(iterations),
                "--sh-degree", str(sh_degree),
            ]

            if continue_from:
                cmd.extend(["--input", continue_from])

            logger.info(f"Running: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

            for line in process.stdout:
                if onProgress and "iteration" in line.lower():
                    try:
                        parts = line.split()
                        for p in parts:
                            if p.isdigit():
                                progress = min(1.0, int(p) / iterations)
                                onProgress(progress)
                                break
                    except:
                        pass

            process.wait()

            return {
                'output_path': output_path,
                'success': process.returncode == 0,
            }

        finally:
            # Restore original transforms if we overrode them
            if original_transforms and original_transforms.exists():
                backup = dataset_dir / "transforms_backup.json"
                if backup.exists():
                    shutil.copy(backup, original_transforms)
                    backup.unlink()

    async def filterTransforms(
        self,
        transforms_path: str,
        exclude_regions: List[str] = None,
        include_regions: List[str] = None,
        output_path: str = None,
    ) -> str:
        """
        Filter transforms.json to include/exclude specific regions.

        Args:
            transforms_path: Path to source transforms.json
            exclude_regions: Regions to exclude (e.g., ['body'])
            include_regions: Regions to include (exclusive with exclude)
            output_path: Output path (default: adds _filtered suffix)

        Returns:
            Path to filtered transforms.json
        """
        import json
        from pathlib import Path

        with open(transforms_path, 'r') as f:
            transforms = json.load(f)

        frames = transforms.get('frames', [])
        filtered_frames = []

        for frame in frames:
            region = frame.get('_region', 'unknown')

            if include_regions:
                if region in include_regions:
                    filtered_frames.append(frame)
            elif exclude_regions:
                if region not in exclude_regions:
                    filtered_frames.append(frame)
            else:
                filtered_frames.append(frame)

        transforms['frames'] = filtered_frames

        if output_path is None:
            p = Path(transforms_path)
            output_path = str(p.parent / f"{p.stem}_filtered.json")

        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        logger.info(f"Filtered {len(frames)} -> {len(filtered_frames)} frames")
        return output_path

    async def convertToRadiance(
        self,
        ply_path: str,
        output_path: str,
        filter: bool = True,
        min_opacity: float = 0.8,
        max_scale: float = 0.05,
        max_brightness: float = 2.0,
    ) -> Dict:
        """
        Convert trained PLY to .radiance format.

        Args:
            ply_path: Path to trained PLY file
            output_path: Output .radiance path
            filter: Whether to filter artifacts
            min_opacity: Minimum opacity threshold
            max_scale: Maximum Gaussian scale
            max_brightness: Maximum SH brightness

        Returns:
            Dict with gaussian_count and success
        """
        from noodlestudio.tools.vrm_to_radiance import ply_to_radiance
        from noodlestudio.core.semantic_world.radiance_format import save_radiance

        try:
            asset = ply_to_radiance(
                ply_path,
                entity_id="trained",
                display_name="Trained Gaussians",
                filter_gaussians=filter,
                min_opacity=min_opacity,
                max_scale=max_scale,
            )

            save_radiance(asset, output_path)

            return {
                'success': True,
                'gaussian_count': asset.gaussian_count,
                'output_path': output_path,
            }

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
            }

    def _find_opensplat(self) -> Optional[str]:
        """Find OpenSplat executable."""
        from pathlib import Path

        candidates = [
            Path(__file__).parent.parent.parent.parent / "external/OpenSplat/build/opensplat",
            Path.home() / "git/OpenSplat/build/opensplat",
            "/usr/local/bin/opensplat",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
