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
#   Face Detail Camera Generator - Importance-weighted camera views for facial expression training.
#
#   Generates camera transforms that focus on expressive faci...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tools.face_detail_camera
# PURPOSE:  Face Detail Camera
# LAYER:    Studio / Tools
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   CameraView, FaceRegion, FaceDetailCameraGenerator, generate_face_detail_cameras()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraView:
    """Single camera view for training."""
    transform_matrix: np.ndarray  # 4x4 camera-to-world matrix
    region: str  # 'body', 'face', 'lips', 'eyes', 'brow', 'profile'
    distance: float
    azimuth: float  # degrees
    elevation: float  # degrees

    def to_nerf_format(self) -> Dict:
        """Convert to NeRFStudio transforms.json format."""
        return {
            "transform_matrix": self.transform_matrix.tolist(),
            "region": self.region,
        }


@dataclass
class FaceRegion:
    """Defines a facial region of interest."""
    name: str
    center: Tuple[float, float, float]  # world coordinates
    importance: float  # 0-1, higher = more views
    view_distance: float  # camera distance for close-ups
    elevation_range: Tuple[float, float]  # min/max elevation angles
    azimuth_range: Tuple[float, float]  # min/max azimuth (0=front)


class FaceDetailCameraGenerator:
    """
    Generates camera views with importance weighting for facial details.

    Usage:
        generator = FaceDetailCameraGenerator()
        generator.set_head_position([0, 1.5, 0])  # From skeleton
        views = generator.generate_views(
            body_views=24,
            face_views=48,
            detail_views=36  # lips, eyes, brow
        )
        generator.export_transforms("output/transforms.json", views)
    """

    # Default facial landmark offsets from head bone (in head-local space)
    # These are approximate for a standard humanoid
    DEFAULT_LANDMARKS = {
        'head_center': (0.0, 0.0, 0.0),
        'face_center': (0.0, -0.02, 0.08),  # slightly forward and down from head
        'lips': (0.0, -0.06, 0.10),
        'left_eye': (-0.03, 0.02, 0.09),
        'right_eye': (0.03, 0.02, 0.09),
        'brow_center': (0.0, 0.05, 0.09),
        'left_brow': (-0.035, 0.05, 0.08),
        'right_brow': (0.035, 0.05, 0.08),
        'nose_tip': (0.0, -0.02, 0.12),
        'chin': (0.0, -0.10, 0.08),
        'left_cheek': (-0.05, -0.02, 0.06),
        'right_cheek': (0.05, -0.02, 0.06),
    }

    def __init__(self):
        self.head_position = np.array([0.0, 1.5, 0.0])
        self.head_rotation = np.eye(3)  # Head orientation matrix
        self.landmarks = {}
        self._compute_landmarks()

        # Region definitions with importance weights
        self.regions = self._define_regions()

    def _compute_landmarks(self):
        """Compute world-space landmark positions from head pose."""
        for name, offset in self.DEFAULT_LANDMARKS.items():
            local_pos = np.array(offset)
            world_pos = self.head_position + self.head_rotation @ local_pos
            self.landmarks[name] = world_pos

    def set_head_position(self, position: List[float], rotation: Optional[np.ndarray] = None):
        """Set the head bone world position and optional rotation."""
        self.head_position = np.array(position)
        if rotation is not None:
            self.head_rotation = rotation
        self._compute_landmarks()
        self.regions = self._define_regions()

    def _define_regions(self) -> List[FaceRegion]:
        """Define facial regions with importance weights."""
        return [
            # Lips - highest importance for speech and emotion
            FaceRegion(
                name='lips',
                center=tuple(self.landmarks['lips']),
                importance=1.0,
                view_distance=0.15,
                elevation_range=(-20, 20),
                azimuth_range=(-60, 60),
            ),
            # Eyes - critical for emotion
            FaceRegion(
                name='left_eye',
                center=tuple(self.landmarks['left_eye']),
                importance=0.9,
                view_distance=0.12,
                elevation_range=(-15, 30),
                azimuth_range=(-80, 20),
            ),
            FaceRegion(
                name='right_eye',
                center=tuple(self.landmarks['right_eye']),
                importance=0.9,
                view_distance=0.12,
                elevation_range=(-15, 30),
                azimuth_range=(-20, 80),
            ),
            # Brow - important for concern, surprise, anger
            FaceRegion(
                name='brow',
                center=tuple(self.landmarks['brow_center']),
                importance=0.85,
                view_distance=0.18,
                elevation_range=(0, 45),
                azimuth_range=(-45, 45),
            ),
            # Face center - general facial structure
            FaceRegion(
                name='face',
                center=tuple(self.landmarks['face_center']),
                importance=0.7,
                view_distance=0.25,
                elevation_range=(-30, 45),
                azimuth_range=(-90, 90),
            ),
            # Profile views - nose, chin, cheekbones
            FaceRegion(
                name='profile_left',
                center=tuple(self.landmarks['left_cheek']),
                importance=0.6,
                view_distance=0.20,
                elevation_range=(-15, 30),
                azimuth_range=(-120, -60),
            ),
            FaceRegion(
                name='profile_right',
                center=tuple(self.landmarks['right_cheek']),
                importance=0.6,
                view_distance=0.20,
                elevation_range=(-15, 30),
                azimuth_range=(60, 120),
            ),
        ]

    def _create_look_at_matrix(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = np.array([0, 1, 0])
    ) -> np.ndarray:
        """Create a 4x4 camera-to-world matrix looking at target from eye."""
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Camera-to-world matrix (OpenGL convention: -Z forward)
        matrix = np.eye(4)
        matrix[:3, 0] = right
        matrix[:3, 1] = up
        matrix[:3, 2] = -forward
        matrix[:3, 3] = eye

        return matrix

    def _orbit_position(
        self,
        center: np.ndarray,
        distance: float,
        azimuth: float,
        elevation: float
    ) -> np.ndarray:
        """Calculate camera position on orbital sphere around center."""
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)

        x = center[0] + distance * np.cos(el_rad) * np.sin(az_rad)
        y = center[1] + distance * np.sin(el_rad)
        z = center[2] + distance * np.cos(el_rad) * np.cos(az_rad)

        return np.array([x, y, z])

    def generate_body_views(self, count: int = 24) -> List[CameraView]:
        """Generate full-body turntable views."""
        views = []
        body_center = self.head_position.copy()
        body_center[1] -= 0.5  # Center on torso, not head

        for i in range(count):
            azimuth = (i / count) * 360.0
            elevation = 15.0  # Slight upward angle
            distance = 2.5

            eye = self._orbit_position(body_center, distance, azimuth, elevation)
            matrix = self._create_look_at_matrix(eye, body_center)

            views.append(CameraView(
                transform_matrix=matrix,
                region='body',
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
            ))

        return views

    def generate_face_views(self, count: int = 48) -> List[CameraView]:
        """Generate face-focused orbital views."""
        views = []
        face_center = np.array(self.landmarks['face_center'])

        # Multiple elevation rings
        elevations = [-15, 0, 15, 30]
        views_per_elevation = count // len(elevations)

        for elevation in elevations:
            for i in range(views_per_elevation):
                # Front-focused: more views in -60 to +60 range
                # Use sinusoidal distribution for front-weighting
                t = i / views_per_elevation
                azimuth = 120 * np.sin(t * np.pi) - 60  # -60 to +60

                distance = 0.35
                eye = self._orbit_position(face_center, distance, azimuth, elevation)
                matrix = self._create_look_at_matrix(eye, face_center)

                views.append(CameraView(
                    transform_matrix=matrix,
                    region='face',
                    distance=distance,
                    azimuth=azimuth,
                    elevation=elevation,
                ))

        return views

    def generate_detail_views(self, views_per_region: int = 6) -> List[CameraView]:
        """Generate extreme close-up views of detailed facial regions."""
        views = []

        for region in self.regions:
            if region.importance < 0.8:  # Only highest importance regions
                continue

            center = np.array(region.center)

            # Calculate view count based on importance
            region_views = int(views_per_region * region.importance)

            for i in range(region_views):
                # Sample within region's angle ranges
                t = i / max(1, region_views - 1)

                az_min, az_max = region.azimuth_range
                el_min, el_max = region.elevation_range

                azimuth = az_min + t * (az_max - az_min)
                # Vary elevation across views
                elevation = el_min + (i % 3) / 2 * (el_max - el_min)

                eye = self._orbit_position(center, region.view_distance, azimuth, elevation)
                matrix = self._create_look_at_matrix(eye, center)

                views.append(CameraView(
                    transform_matrix=matrix,
                    region=region.name,
                    distance=region.view_distance,
                    azimuth=azimuth,
                    elevation=elevation,
                ))

        return views

    def generate_views(
        self,
        body_views: int = 24,
        face_views: int = 48,
        detail_views_per_region: int = 8,
    ) -> List[CameraView]:
        """
        Generate complete view set with importance weighting.

        Default distribution:
        - 24 body views (turntable)
        - 48 face views (orbital around head)
        - ~40 detail views (lips, eyes, brow close-ups)

        Total: ~112 views with heavy face weighting
        """
        all_views = []

        all_views.extend(self.generate_body_views(body_views))
        all_views.extend(self.generate_face_views(face_views))
        all_views.extend(self.generate_detail_views(detail_views_per_region))

        logger.info(f"Generated {len(all_views)} views:")
        logger.info(f"  Body: {body_views}")
        logger.info(f"  Face: {face_views}")
        logger.info(f"  Detail: {len(all_views) - body_views - face_views}")

        return all_views

    def export_transforms(
        self,
        output_path: str,
        views: List[CameraView],
        image_width: int = 1024,
        image_height: int = 1024,
        fov: float = 50.0,
    ):
        """Export views to NeRFStudio transforms.json format."""
        # Calculate focal length from FOV
        focal_length = 0.5 * image_width / np.tan(0.5 * np.radians(fov))

        transforms = {
            "camera_model": "OPENCV",
            "fl_x": focal_length,
            "fl_y": focal_length,
            "cx": image_width / 2,
            "cy": image_height / 2,
            "w": image_width,
            "h": image_height,
            "frames": []
        }

        for i, view in enumerate(views):
            frame = {
                "file_path": f"images/frame_{i:04d}.png",
                "transform_matrix": view.transform_matrix.tolist(),
                # Metadata for debugging
                "_region": view.region,
                "_azimuth": view.azimuth,
                "_elevation": view.elevation,
                "_distance": view.distance,
            }
            transforms["frames"].append(frame)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        logger.info(f"Exported {len(views)} camera transforms to {output_path}")

        # Also export a summary
        summary = self._generate_summary(views)
        summary_path = output_path.parent / "camera_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)

        return output_path

    def _generate_summary(self, views: List[CameraView]) -> str:
        """Generate human-readable summary of view distribution."""
        regions = {}
        for v in views:
            regions[v.region] = regions.get(v.region, 0) + 1

        lines = [
            "Face Detail Camera Distribution",
            "=" * 40,
            f"Total views: {len(views)}",
            "",
            "By region:",
        ]

        for region, count in sorted(regions.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(views)
            lines.append(f"  {region}: {count} ({pct:.1f}%)")

        lines.extend([
            "",
            "Region importance (higher = more detail):",
            "  lips: 1.0 (speech, emotion)",
            "  eyes: 0.9 (subtle emotion)",
            "  brow: 0.85 (concern, surprise)",
            "  face: 0.7 (general structure)",
            "  profile: 0.6 (silhouette)",
            "  body: 0.3 (context)",
        ])

        return "\n".join(lines)


def generate_face_detail_cameras(
    head_position: List[float],
    output_dir: str,
    body_views: int = 24,
    face_views: int = 48,
    detail_views: int = 8,
) -> str:
    """
    Convenience function to generate face-detail camera transforms.

    Args:
        head_position: [x, y, z] world position of head bone
        output_dir: Directory to write transforms.json
        body_views: Number of full-body turntable views
        face_views: Number of face-focused orbital views
        detail_views: Views per detail region (lips, eyes, brow)

    Returns:
        Path to generated transforms.json
    """
    generator = FaceDetailCameraGenerator()
    generator.set_head_position(head_position)

    views = generator.generate_views(
        body_views=body_views,
        face_views=face_views,
        detail_views_per_region=detail_views,
    )

    output_path = Path(output_dir) / "transforms.json"
    generator.export_transforms(str(output_path), views)

    return str(output_path)


if __name__ == "__main__":
    # Test with default head position
    import sys

    logging.basicConfig(level=logging.INFO)

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/face_cameras"

    generator = FaceDetailCameraGenerator()
    generator.set_head_position([0, 1.35, 0])  # Typical head height

    views = generator.generate_views()
    generator.export_transforms(f"{output_dir}/transforms.json", views)

    print(f"\nGenerated {len(views)} camera views to {output_dir}/")
    print("\nTo render training images from these cameras:")
    print("  1. Load your model in Blender/Unity")
    print("  2. Import transforms.json")
    print("  3. Render each camera to images/frame_XXXX.png")
    print("  4. Run OpenSplat: ./opensplat <output_dir> -n 30000")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
