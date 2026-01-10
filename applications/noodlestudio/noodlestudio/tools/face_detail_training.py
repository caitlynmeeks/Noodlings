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
#   Face Detail Training Pipeline - Multi-stage Gaussian training with facial importance weighting.
#
#   This pipeline: 1. Generates importance-weighted camera vi...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tools.face_detail_training
# PURPOSE:  Face Detail Training
# LAYER:    Studio / Tools
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TrainingConfig, FaceDetailTrainingPipeline, train_face_detail()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import json
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import logging
import numpy as np

from .face_detail_camera import FaceDetailCameraGenerator, CameraView

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for face-detail Gaussian training."""
    # View distribution
    body_views: int = 24
    face_views: int = 48
    detail_views_per_region: int = 8

    # Training parameters
    iterations: int = 30000
    sh_degree: int = 2

    # Multi-stage refinement
    enable_refinement: bool = True
    refinement_iterations: int = 10000  # Extra iterations with face-only views

    # Output
    output_name: str = "face_detail"

    # Filtering (for trained PLY)
    min_opacity: float = 0.8
    max_scale: float = 0.05


class FaceDetailTrainingPipeline:
    """
    Pipeline for training face-detail-focused Gaussian splats.

    Usage:
        pipeline = FaceDetailTrainingPipeline(
            vrm_path="/path/to/avatar.vrm",
            output_dir="/path/to/output"
        )
        result = pipeline.run()
    """

    def __init__(
        self,
        vrm_path: Optional[str] = None,
        mesh_path: Optional[str] = None,
        output_dir: str = "./face_detail_output",
        opensplat_path: Optional[str] = None,
    ):
        self.vrm_path = vrm_path
        self.mesh_path = mesh_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find OpenSplat
        self.opensplat_path = opensplat_path or self._find_opensplat()

        self.config = TrainingConfig()
        self._progress_callback: Optional[Callable] = None

    def _find_opensplat(self) -> Optional[str]:
        """Find OpenSplat executable."""
        candidates = [
            Path(__file__).parent.parent.parent.parent.parent / "external/OpenSplat/build/opensplat",
            Path.home() / "git/OpenSplat/build/opensplat",
            "/usr/local/bin/opensplat",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def set_config(self, **kwargs):
        """Update training configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set callback for progress updates: callback(stage, progress_0_to_1)"""
        self._progress_callback = callback

    def _report_progress(self, stage: str, progress: float):
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(stage, progress)
        logger.info(f"[{stage}] {progress*100:.1f}%")

    def generate_cameras(self, head_position: List[float]) -> List[CameraView]:
        """Generate importance-weighted camera views."""
        self._report_progress("cameras", 0.0)

        generator = FaceDetailCameraGenerator()
        generator.set_head_position(head_position)

        views = generator.generate_views(
            body_views=self.config.body_views,
            face_views=self.config.face_views,
            detail_views_per_region=self.config.detail_views_per_region,
        )

        # Export transforms
        transforms_path = self.output_dir / "transforms.json"
        generator.export_transforms(str(transforms_path), views)

        self._report_progress("cameras", 1.0)
        return views

    def render_training_images(
        self,
        views: List[CameraView],
        render_func: Optional[Callable] = None,
    ) -> Path:
        """
        Render training images from camera views.

        Args:
            views: Camera views to render
            render_func: Optional custom render function(view, output_path)
                        If not provided, uses built-in VRM renderer

        Returns:
            Path to images directory
        """
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        self._report_progress("render", 0.0)

        if render_func:
            # Use custom renderer
            for i, view in enumerate(views):
                output_path = images_dir / f"frame_{i:04d}.png"
                render_func(view, str(output_path))
                self._report_progress("render", (i + 1) / len(views))
        else:
            # Use built-in VRM renderer
            self._render_vrm_views(views, images_dir)

        self._report_progress("render", 1.0)
        return images_dir

    def _render_vrm_views(self, views: List[CameraView], output_dir: Path):
        """Render VRM model from camera views using our Gaussian renderer."""
        if not self.vrm_path:
            raise ValueError("VRM path required for built-in renderer")

        # Import here to avoid circular imports
        from noodlestudio.core.radiance_component import RadianceComponent
        from noodlestudio.core.gaussian_renderer import GaussianRenderer
        from noodlestudio.tools.vrm_to_radiance import vrm_to_radiance
        from noodlestudio.core.semantic_world.radiance_format import save_radiance
        from PIL import Image

        # Convert VRM to radiance (temporary, for rendering reference views)
        logger.info("Converting VRM to radiance for reference rendering...")
        temp_radiance = self.output_dir / "temp_vrm.radiance"

        asset = vrm_to_radiance(
            self.vrm_path,
            entity_id="training_ref",
            display_name="Training Reference",
            densify=True  # Use dense version for better reference
        )
        save_radiance(asset, str(temp_radiance))

        # Load as component
        component = RadianceComponent("training_ref")
        component.load_asset(str(temp_radiance))

        # Create renderer
        renderer = GaussianRenderer()

        # Render each view
        for i, view in enumerate(views):
            # Extract camera parameters from view matrix
            matrix = view.transform_matrix
            eye = matrix[:3, 3]
            forward = -matrix[:3, 2]  # -Z is forward in OpenGL
            target = eye + forward

            # Create camera
            from noodlestudio.core.gaussian_renderer import GaussianCamera
            camera = GaussianCamera(
                width=1024,
                height=1024,
                fov_y=50.0,
                position=tuple(eye),
                target=tuple(target),
                up=(0, 1, 0),
            )

            # Render
            image, alpha, info = renderer.render_component(component, camera)

            # Save
            output_path = output_dir / f"frame_{i:04d}.png"
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
                img.save(str(output_path))
            else:
                image.save(str(output_path))

            if i % 10 == 0:
                self._report_progress("render", (i + 1) / len(views))

        # Cleanup temp file
        temp_radiance.unlink(missing_ok=True)

    def run_opensplat(
        self,
        dataset_dir: Path,
        output_ply: Path,
        iterations: int,
        continue_from: Optional[Path] = None,
    ) -> bool:
        """Run OpenSplat training."""
        if not self.opensplat_path:
            raise ValueError("OpenSplat not found. Please build it first.")

        cmd = [
            self.opensplat_path,
            str(dataset_dir),
            "-o", str(output_ply),
            "-n", str(iterations),
            "--sh-degree", str(self.config.sh_degree),
        ]

        if continue_from and continue_from.exists():
            cmd.extend(["--input", str(continue_from)])

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

            # Monitor progress
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.debug(line)
                    # Parse iteration progress if possible
                    if "iteration" in line.lower():
                        try:
                            # Try to extract iteration number
                            parts = line.split()
                            for p in parts:
                                if p.isdigit():
                                    progress = min(1.0, int(p) / iterations)
                                    self._report_progress("training", progress)
                                    break
                        except:
                            pass

            process.wait()
            return process.returncode == 0

        except Exception as e:
            logger.error(f"OpenSplat failed: {e}")
            return False

    def convert_to_radiance(self, ply_path: Path, output_path: Path) -> bool:
        """Convert trained PLY to .radiance format with filtering."""
        try:
            from noodlestudio.tools.vrm_to_radiance import ply_to_radiance
            from noodlestudio.core.semantic_world.radiance_format import save_radiance

            asset = ply_to_radiance(
                str(ply_path),
                entity_id=self.config.output_name,
                display_name=self.config.output_name,
                filter_gaussians=True,
                min_opacity=self.config.min_opacity,
                max_scale=self.config.max_scale,
            )

            save_radiance(asset, str(output_path))
            logger.info(f"Converted to radiance: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False

    def run(
        self,
        head_position: Optional[List[float]] = None,
        skip_render: bool = False,
        existing_images_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run the complete face-detail training pipeline.

        Args:
            head_position: [x, y, z] of head bone. If None, auto-detected from VRM.
            skip_render: If True, skip rendering (use existing images)
            existing_images_dir: Path to existing training images

        Returns:
            Dict with results: {
                'success': bool,
                'radiance_path': str,
                'ply_path': str,
                'view_count': int,
                'gaussian_count': int,
            }
        """
        result = {
            'success': False,
            'radiance_path': None,
            'ply_path': None,
            'view_count': 0,
            'gaussian_count': 0,
        }

        try:
            # Step 1: Determine head position
            if head_position is None:
                head_position = self._detect_head_position()

            logger.info(f"Head position: {head_position}")

            # Step 2: Generate cameras
            views = self.generate_cameras(head_position)
            result['view_count'] = len(views)
            logger.info(f"Generated {len(views)} camera views")

            # Step 3: Render training images
            if existing_images_dir:
                images_dir = Path(existing_images_dir)
            elif not skip_render:
                images_dir = self.render_training_images(views)
            else:
                images_dir = self.output_dir / "images"

            # Step 4: Run OpenSplat (Stage 1: Full training)
            stage1_ply = self.output_dir / "stage1.ply"
            self._report_progress("training", 0.0)

            success = self.run_opensplat(
                self.output_dir,
                stage1_ply,
                self.config.iterations,
            )

            if not success:
                logger.error("Stage 1 training failed")
                return result

            self._report_progress("training", 0.7)

            # Step 5: Optional refinement with face-only views
            final_ply = stage1_ply
            if self.config.enable_refinement and self.config.refinement_iterations > 0:
                logger.info("Running face refinement stage...")

                # Create face-only transforms
                face_views = [v for v in views if v.region != 'body']
                face_transforms_path = self.output_dir / "transforms_face.json"

                generator = FaceDetailCameraGenerator()
                generator.set_head_position(head_position)
                generator.export_transforms(str(face_transforms_path), face_views)

                # Rename for OpenSplat to use
                original_transforms = self.output_dir / "transforms.json"
                backup_transforms = self.output_dir / "transforms_full.json"
                shutil.copy(original_transforms, backup_transforms)
                shutil.copy(face_transforms_path, original_transforms)

                # Run refinement
                stage2_ply = self.output_dir / "stage2_refined.ply"
                success = self.run_opensplat(
                    self.output_dir,
                    stage2_ply,
                    self.config.refinement_iterations,
                    continue_from=stage1_ply,
                )

                # Restore original transforms
                shutil.copy(backup_transforms, original_transforms)

                if success and stage2_ply.exists():
                    final_ply = stage2_ply

            self._report_progress("training", 1.0)

            # Step 6: Convert to radiance format
            radiance_path = self.output_dir / f"{self.config.output_name}.radiance"
            self._report_progress("conversion", 0.0)

            if self.convert_to_radiance(final_ply, radiance_path):
                result['success'] = True
                result['radiance_path'] = str(radiance_path)
                result['ply_path'] = str(final_ply)

                # Get Gaussian count
                from noodlestudio.core.semantic_world.radiance_format import load_radiance
                asset = load_radiance(str(radiance_path))
                result['gaussian_count'] = asset.gaussian_count

            self._report_progress("conversion", 1.0)

            logger.info(f"Training complete: {result['gaussian_count']} Gaussians")
            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return result

    def _detect_head_position(self) -> List[float]:
        """Auto-detect head position from VRM skeleton."""
        if self.vrm_path:
            try:
                from noodlestudio.core.semantic_world.vrm_parser import VRMParser

                parser = VRMParser()
                vrm_data = parser.parse(self.vrm_path)

                # Find head bone
                if vrm_data.skeleton and vrm_data.skeleton.humanoid_map:
                    head_idx = vrm_data.skeleton.humanoid_map.get('head')
                    if head_idx is not None and head_idx < len(vrm_data.skeleton.bones):
                        bone = vrm_data.skeleton.bones[head_idx]
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


def train_face_detail(
    vrm_path: str,
    output_dir: str,
    iterations: int = 30000,
    enable_refinement: bool = True,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """
    Convenience function for face-detail Gaussian training.

    Args:
        vrm_path: Path to VRM avatar file
        output_dir: Output directory for training artifacts
        iterations: Number of training iterations
        enable_refinement: Whether to run face refinement stage
        progress_callback: Optional callback(stage, progress)

    Returns:
        Dict with training results
    """
    pipeline = FaceDetailTrainingPipeline(
        vrm_path=vrm_path,
        output_dir=output_dir,
    )

    pipeline.set_config(
        iterations=iterations,
        enable_refinement=enable_refinement,
    )

    if progress_callback:
        pipeline.set_progress_callback(progress_callback)

    return pipeline.run()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python face_detail_training.py <vrm_path> <output_dir>")
        sys.exit(1)

    vrm_path = sys.argv[1]
    output_dir = sys.argv[2]

    result = train_face_detail(vrm_path, output_dir)

    if result['success']:
        print(f"\nTraining complete!")
        print(f"  Radiance: {result['radiance_path']}")
        print(f"  Gaussians: {result['gaussian_count']}")
    else:
        print("\nTraining failed!")
        sys.exit(1)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
