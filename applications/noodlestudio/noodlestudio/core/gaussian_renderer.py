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
#   Gaussian Renderer - GPU-accelerated via gsplat-mps or PyTorch fallback.
#
#   This renderer implements 3D Gaussian Splatting with two b...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.gaussian_renderer
# PURPOSE:  Gaussian Renderer
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   CameraParams, GaussianRenderer, create_orbit_camera(), render_turntable()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import math
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import gsplat-mps for GPU acceleration
GSPLAT_AVAILABLE = False
try:
    import sys
    # Add gsplat-mps to path if needed
    gsplat_path = "/Users/thistlequell/git/gsplat-mps"
    if gsplat_path not in sys.path:
        sys.path.insert(0, gsplat_path)

    from gsplat.project_gaussians import project_gaussians as gsplat_project
    from gsplat.rasterize import rasterize_gaussians as gsplat_rasterize
    GSPLAT_AVAILABLE = True
    logger.info("gsplat-mps GPU acceleration available")
except ImportError as e:
    logger.warning(f"gsplat-mps not available, using software rendering: {e}")


@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    # Intrinsics
    fx: float = 500.0  # Focal length X
    fy: float = 500.0  # Focal length Y
    cx: float = 256.0  # Principal point X
    cy: float = 256.0  # Principal point Y

    # Image size
    width: int = 512
    height: int = 512

    # Clipping planes
    near: float = 0.1
    far: float = 100.0

    # View transform (camera pose)
    view_matrix: Optional[np.ndarray] = None  # 4x4 world-to-camera

    def get_K(self, device: torch.device) -> torch.Tensor:
        """Get 3x3 intrinsic matrix."""
        return torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

    def get_view_matrix(self, device: torch.device) -> torch.Tensor:
        """Get 4x4 view matrix."""
        if self.view_matrix is not None:
            return torch.tensor(self.view_matrix, dtype=torch.float32, device=device)
        # Default: look down -Z axis
        return torch.eye(4, dtype=torch.float32, device=device)


class GaussianRenderer:
    """
    Pure PyTorch Gaussian splatting renderer.

    Optimized for Apple Silicon via MPS, with CPU fallback.
    """

    def __init__(self, device: Optional[str] = None, force_software: bool = False):
        """
        Initialize renderer.

        Args:
            device: 'mps', 'cuda', 'cpu', or None for auto-detect
            force_software: If True, use software rendering even if GPU available
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("GaussianRenderer using MPS (Apple Silicon)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("GaussianRenderer using CUDA")
            else:
                self.device = torch.device("cpu")
                logger.info("GaussianRenderer using CPU")
        else:
            self.device = torch.device(device)

        # GPU acceleration
        self.use_gpu = GSPLAT_AVAILABLE and not force_software and str(self.device) == "mps"
        if self.use_gpu:
            logger.info("Using gsplat-mps GPU acceleration")
        else:
            logger.info("Using software rendering")

        # Pre-allocate common tensors
        self._setup_constants()

    def _setup_constants(self):
        """Set up constant tensors."""
        # SH to RGB conversion constant (DC component)
        self.SH_C0 = 0.28209479177387814

    def _render_gpu(
        self,
        positions: torch.Tensor,      # (N, 3) Gaussian centers
        scales: torch.Tensor,          # (N, 3) Gaussian scales
        rotations: torch.Tensor,       # (N, 4) Quaternions (x, y, z, w)
        opacities: torch.Tensor,       # (N,) or (N, 1) opacity values
        colors: torch.Tensor,          # (N, 3) RGB colors [0, 1]
        camera: CameraParams,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        GPU-accelerated rendering via gsplat-mps.

        ~200+ FPS on M3 Ultra for 50K Gaussians at 512x512.
        """
        N = positions.shape[0]
        H, W = camera.height, camera.width

        # Move to device
        positions = positions.to(self.device).float()
        scales = scales.to(self.device).contiguous()
        opacities = opacities.to(self.device).view(-1, 1).contiguous()
        colors = colors.to(self.device).contiguous()

        # gsplat uses (w, x, y, z) quaternion format
        # Our data is (x, y, z, w) - reorder
        rotations = rotations.to(self.device)
        quats = torch.cat([
            rotations[:, 3:4],  # w
            rotations[:, 0:1],  # x
            rotations[:, 1:2],  # y
            rotations[:, 2:3],  # z
        ], dim=1).contiguous()

        # gsplat expects:
        # - Positions in world space
        # - Objects with positive Z in camera space are visible
        # - View matrix transforms world -> camera
        #
        # Our camera convention: camera at eye looking at target
        # gsplat convention: camera at origin, +Z points into scene
        #
        # Strategy: Transform positions to camera space, then use identity view
        view_matrix = camera.get_view_matrix(self.device)

        # Transform positions to camera space
        pos_homo = torch.cat([positions, torch.ones(N, 1, device=self.device)], dim=1)
        pos_cam = (view_matrix @ pos_homo.T).T[:, :3]

        # gsplat expects +Z to be in front of camera
        # Our view has +Z in front, but gsplat clips at z < clip_thresh
        # Make sure positions are in front of camera
        pos_gsplat = pos_cam.contiguous()

        # Use identity view matrix since we pre-transformed
        viewmat = torch.eye(4, device=self.device, dtype=torch.float32)

        # Build projection matrix from intrinsics
        fov_y = 2 * math.atan(H / (2 * camera.fy))
        aspect = W / H
        near, far = camera.near, camera.far
        f = 1.0 / math.tan(fov_y / 2)
        projmat = torch.tensor([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0]
        ], device=self.device, dtype=torch.float32).contiguous()

        # Tile bounds
        BLOCK = 16
        tile_bounds = ((W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK, 1)

        # Project Gaussians to 2D
        xys, depths, radii, conics, num_tiles_hit, cov3d = gsplat_project(
            pos_gsplat,
            scales,
            1.0,  # glob_scale
            quats,
            viewmat,
            projmat,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            H,
            W,
            tile_bounds,
            0.01,  # clip_thresh
        )

        # Count visible
        visible = (radii > 0).sum().item()

        if visible == 0:
            bg = torch.tensor(background, device=self.device)
            image = bg.view(1, 1, 3).expand(H, W, 3).contiguous()
            alpha = torch.zeros(H, W, device=self.device)
            return image, alpha, {'visible': 0, 'total': N, 'device': str(self.device), 'backend': 'gsplat-mps'}

        # Rasterize
        bg_tensor = torch.tensor(background, device=self.device, dtype=torch.float32)
        out_img = gsplat_rasterize(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            colors,
            opacities,
            H,
            W,
            background=bg_tensor,
            return_alpha=False,
        )

        # Alpha channel (approximate)
        alpha = torch.ones(H, W, device=self.device)

        info = {
            'visible': visible,
            'total': N,
            'device': str(self.device),
            'backend': 'gsplat-mps',
        }

        return out_img, alpha, info

    def render(
        self,
        positions: torch.Tensor,      # (N, 3) Gaussian centers
        scales: torch.Tensor,          # (N, 3) Gaussian scales
        rotations: torch.Tensor,       # (N, 4) Quaternions (x, y, z, w)
        opacities: torch.Tensor,       # (N,) or (N, 1) opacity values
        sh_dc: torch.Tensor,           # (N, 3) DC spherical harmonics (color)
        camera: CameraParams,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Render Gaussians to an image.

        Args:
            positions: (N, 3) Gaussian center positions in world space
            scales: (N, 3) Gaussian scales (standard deviations)
            rotations: (N, 4) Quaternion rotations (x, y, z, w)
            opacities: (N,) Opacity values [0, 1]
            sh_dc: (N, 3) DC spherical harmonics (base color)
            camera: Camera parameters
            background: Background RGB color

        Returns:
            image: (H, W, 3) RGB image
            alpha: (H, W) Alpha channel
            info: Dictionary with rendering statistics
        """
        # Convert SH DC to RGB colors
        sh_dc_tensor = sh_dc if isinstance(sh_dc, torch.Tensor) else torch.from_numpy(sh_dc)
        colors = sh_dc_tensor * self.SH_C0 + 0.5
        colors = torch.clamp(colors, 0, 1)

        # Use GPU renderer if available
        if self.use_gpu:
            return self._render_gpu(positions, scales, rotations, opacities, colors, camera, background)

        N = positions.shape[0]
        H, W = camera.height, camera.width

        # Move to device
        positions = positions.to(self.device)
        scales = scales.to(self.device)
        rotations = rotations.to(self.device)
        opacities = opacities.to(self.device).view(-1)
        colors = colors.to(self.device)

        # Get camera matrices
        K = camera.get_K(self.device)
        view = camera.get_view_matrix(self.device)

        # 1. Transform positions to camera space
        pos_homo = torch.cat([positions, torch.ones(N, 1, device=self.device)], dim=1)
        pos_cam = (view @ pos_homo.T).T[:, :3]  # (N, 3) in camera space

        # 2. Filter by depth (only render in front of camera)
        depth = pos_cam[:, 2]
        valid_mask = (depth > camera.near) & (depth < camera.far)

        if valid_mask.sum() == 0:
            # No visible Gaussians
            bg = torch.tensor(background, device=self.device)
            image = bg.view(1, 1, 3).expand(H, W, 3).contiguous()
            alpha = torch.zeros(H, W, device=self.device)
            return image, alpha, {'visible': 0, 'total': N, 'device': str(self.device)}

        # 3. Project to screen space
        pos_cam_valid = pos_cam[valid_mask]
        depth_valid = depth[valid_mask]

        # Perspective projection
        pos_proj = pos_cam_valid[:, :2] / pos_cam_valid[:, 2:3]  # (N', 2)
        pos_screen = pos_proj @ K[:2, :2].T + K[:2, 2]  # (N', 2) pixel coords

        # Flip Y for image coordinates (Y=0 at top)
        pos_screen[:, 1] = H - pos_screen[:, 1]

        # 4. Compute 2D covariance from 3D Gaussian
        scales_valid = scales[valid_mask]
        rotations_valid = rotations[valid_mask]

        # Build 3D covariance matrix: R @ S @ S.T @ R.T
        cov_3d = self._build_covariance_3d(scales_valid, rotations_valid)

        # Project to 2D covariance
        cov_2d = self._project_covariance(cov_3d, pos_cam_valid, K)

        # 5. Sort by depth (front-to-back)
        sort_indices = torch.argsort(depth_valid)

        pos_screen = pos_screen[sort_indices]
        cov_2d = cov_2d[sort_indices]
        opacities_valid = opacities[valid_mask][sort_indices]
        colors_valid = colors[valid_mask][sort_indices]  # Colors already converted from SH

        # 6. Rasterize with alpha blending
        image, alpha = self._rasterize(
            pos_screen, cov_2d, opacities_valid, colors_valid,
            H, W, background
        )

        info = {
            'visible': valid_mask.sum().item(),
            'total': N,
            'device': str(self.device),
        }

        return image, alpha, info

    def _build_covariance_3d(
        self,
        scales: torch.Tensor,     # (N, 3)
        rotations: torch.Tensor,  # (N, 4) quaternion (x, y, z, w)
    ) -> torch.Tensor:
        """Build 3D covariance matrices from scales and rotations."""
        N = scales.shape[0]

        # Convert quaternion to rotation matrix
        R = self._quaternion_to_matrix(rotations)  # (N, 3, 3)

        # Scale matrix (diagonal)
        S = torch.diag_embed(scales)  # (N, 3, 3)

        # Covariance: R @ S @ S.T @ R.T = R @ (S^2) @ R.T
        S_sq = S @ S.transpose(-1, -2)
        cov = R @ S_sq @ R.transpose(-1, -2)

        return cov  # (N, 3, 3)

    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (x, y, z, w) to rotation matrix."""
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Rotation matrix elements
        r00 = 1 - 2*y*y - 2*z*z
        r01 = 2*x*y - 2*z*w
        r02 = 2*x*z + 2*y*w
        r10 = 2*x*y + 2*z*w
        r11 = 1 - 2*x*x - 2*z*z
        r12 = 2*y*z - 2*x*w
        r20 = 2*x*z - 2*y*w
        r21 = 2*y*z + 2*x*w
        r22 = 1 - 2*x*x - 2*y*y

        R = torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ], dim=-2)

        return R  # (N, 3, 3)

    def _project_covariance(
        self,
        cov_3d: torch.Tensor,     # (N, 3, 3)
        pos_cam: torch.Tensor,    # (N, 3) camera space positions
        K: torch.Tensor,          # (3, 3) intrinsic matrix
    ) -> torch.Tensor:
        """Project 3D covariance to 2D screen space covariance."""
        N = cov_3d.shape[0]

        # Jacobian of perspective projection
        z = pos_cam[:, 2:3]  # (N, 1)
        fx, fy = K[0, 0], K[1, 1]

        # J = [[fx/z, 0, -fx*x/z^2],
        #      [0, fy/z, -fy*y/z^2]]
        x, y = pos_cam[:, 0], pos_cam[:, 1]
        z_sq = z.squeeze() ** 2

        J = torch.zeros(N, 2, 3, device=self.device)
        J[:, 0, 0] = fx / z.squeeze()
        J[:, 0, 2] = -fx * x / z_sq
        J[:, 1, 1] = fy / z.squeeze()
        J[:, 1, 2] = -fy * y / z_sq

        # 2D covariance: J @ cov_3d @ J.T
        cov_2d = J @ cov_3d @ J.transpose(-1, -2)  # (N, 2, 2)

        # Add small regularization for numerical stability
        cov_2d[:, 0, 0] += 0.3
        cov_2d[:, 1, 1] += 0.3

        return cov_2d  # (N, 2, 2)

    def _rasterize(
        self,
        pos_screen: torch.Tensor,     # (N, 2) screen positions
        cov_2d: torch.Tensor,         # (N, 2, 2) 2D covariances
        opacities: torch.Tensor,      # (N,)
        colors: torch.Tensor,         # (N, 3)
        H: int, W: int,
        background: Tuple[float, float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rasterize Gaussians with alpha blending.

        Uses a simplified tile-based approach.
        """
        N = pos_screen.shape[0]

        # Initialize output
        bg = torch.tensor(background, device=self.device)
        image = bg.view(1, 1, 3).expand(H, W, 3).clone()
        alpha = torch.zeros(H, W, device=self.device)

        # Create pixel coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        pixels = torch.stack([x_coords, y_coords], dim=-1).float()  # (H, W, 2)

        # Process Gaussians (already sorted front-to-back)
        # For efficiency, batch process in chunks
        chunk_size = min(1000, N)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            pos_chunk = pos_screen[start:end]      # (C, 2)
            cov_chunk = cov_2d[start:end]          # (C, 2, 2)
            opa_chunk = opacities[start:end]       # (C,)
            col_chunk = colors[start:end]          # (C, 3)

            # Compute Gaussian values for each pixel
            # diff = pixels - pos: (H, W, C, 2)
            diff = pixels.unsqueeze(2) - pos_chunk.unsqueeze(0).unsqueeze(0)

            # Inverse covariance
            cov_det = cov_chunk[:, 0, 0] * cov_chunk[:, 1, 1] - cov_chunk[:, 0, 1] * cov_chunk[:, 1, 0]
            cov_det = torch.clamp(cov_det, min=1e-6)

            cov_inv = torch.zeros_like(cov_chunk)
            cov_inv[:, 0, 0] = cov_chunk[:, 1, 1] / cov_det
            cov_inv[:, 1, 1] = cov_chunk[:, 0, 0] / cov_det
            cov_inv[:, 0, 1] = -cov_chunk[:, 0, 1] / cov_det
            cov_inv[:, 1, 0] = -cov_chunk[:, 1, 0] / cov_det

            # Mahalanobis distance: diff.T @ cov_inv @ diff
            # (H, W, C, 2) @ (C, 2, 2) -> (H, W, C, 2)
            temp = torch.einsum('hwci,cij->hwcj', diff, cov_inv)
            # (H, W, C, 2) * (H, W, C, 2) -> sum -> (H, W, C)
            maha_dist = (temp * diff).sum(dim=-1)

            # Gaussian value: exp(-0.5 * maha_dist)
            gauss_val = torch.exp(-0.5 * maha_dist)  # (H, W, C)

            # Alpha contribution
            alpha_contrib = opa_chunk.view(1, 1, -1) * gauss_val  # (H, W, C)

            # Alpha blend (front-to-back within chunk)
            for i in range(end - start):
                a = alpha_contrib[:, :, i]
                transmittance = 1 - alpha

                # Accumulate color
                image = image + transmittance.unsqueeze(-1) * a.unsqueeze(-1) * col_chunk[i].view(1, 1, 3)

                # Update alpha
                alpha = alpha + transmittance * a

        # Clamp output
        image = torch.clamp(image, 0, 1)
        alpha = torch.clamp(alpha, 0, 1)

        return image, alpha

    def render_from_radiance(
        self,
        radiance_path: str,
        camera: CameraParams,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Render a .radiance file.

        Args:
            radiance_path: Path to .radiance file
            camera: Camera parameters
            background: Background color

        Returns:
            image, alpha, info (same as render())
        """
        from .semantic_world.radiance_format import load_radiance

        asset = load_radiance(radiance_path)

        positions = torch.from_numpy(asset.positions)
        scales = torch.from_numpy(asset.scales)
        rotations = torch.from_numpy(asset.rotations)
        opacities = torch.from_numpy(asset.opacities)
        sh_dc = torch.from_numpy(asset.sh_dc)

        return self.render(positions, scales, rotations, opacities, sh_dc, camera, background)

    def render_batch(
        self,
        batch: 'RenderBatch',
        camera: CameraParams,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Render a batch of Gaussians from multiple components.

        This is the main entry point for multi-asset scene rendering.
        The batch is pre-built by RadianceSceneBuilder.

        Args:
            batch: RenderBatch from RadianceSceneBuilder.build_render_batch()
            camera: Camera parameters
            background: Background color

        Returns:
            image: (H, W, 3) RGB image
            alpha: (H, W) Alpha channel
            info: Dictionary with rendering statistics
        """
        # Convert batch arrays to tensors
        positions = torch.from_numpy(batch.positions)
        scales = torch.from_numpy(batch.scales)
        rotations = torch.from_numpy(batch.rotations)
        opacities = torch.from_numpy(batch.opacities)

        # Colors are already computed with overrides in get_render_data()
        # Convert to SH-like format (we'll bypass SH conversion since colors are final)
        colors = torch.from_numpy(batch.colors)

        # Render using modified path that takes final colors
        image, alpha, info = self._render_with_colors(
            positions, scales, rotations, opacities, colors, camera, background
        )

        # Add batch-specific info
        info['components'] = len(batch.components)
        info['static_gaussians'] = batch.static_gaussians
        info['dynamic_gaussians'] = batch.dynamic_gaussians

        return image, alpha, info

    def _render_with_colors(
        self,
        positions: torch.Tensor,      # (N, 3) Gaussian centers
        scales: torch.Tensor,          # (N, 3) Gaussian scales
        rotations: torch.Tensor,       # (N, 4) Quaternions
        opacities: torch.Tensor,       # (N,) opacity values
        colors: torch.Tensor,          # (N, 3) FINAL RGB colors [0, 1]
        camera: CameraParams,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Render with pre-computed colors (no SH conversion).

        Used by render_batch() where colors already have overrides applied.
        Uses GPU acceleration when available.
        """
        # Use GPU renderer if available
        if self.use_gpu:
            return self._render_gpu(positions, scales, rotations, opacities, colors, camera, background)

        N = positions.shape[0]
        H, W = camera.height, camera.width

        # Move to device
        positions = positions.to(self.device)
        scales = scales.to(self.device)
        rotations = rotations.to(self.device)
        opacities = opacities.to(self.device).view(-1)
        colors = colors.to(self.device)

        # Get camera matrices
        K = camera.get_K(self.device)
        view = camera.get_view_matrix(self.device)

        # 1. Transform positions to camera space
        pos_homo = torch.cat([positions, torch.ones(N, 1, device=self.device)], dim=1)
        pos_cam = (view @ pos_homo.T).T[:, :3]

        # 2. Filter by depth
        depth = pos_cam[:, 2]
        valid_mask = (depth > camera.near) & (depth < camera.far)

        if valid_mask.sum() == 0:
            bg = torch.tensor(background, device=self.device)
            image = bg.view(1, 1, 3).expand(H, W, 3).contiguous()
            alpha = torch.zeros(H, W, device=self.device)
            return image, alpha, {'visible': 0, 'total': N, 'device': str(self.device)}

        # 3. Project to screen space
        pos_cam_valid = pos_cam[valid_mask]
        depth_valid = depth[valid_mask]

        pos_proj = pos_cam_valid[:, :2] / pos_cam_valid[:, 2:3]
        pos_screen = pos_proj @ K[:2, :2].T + K[:2, 2]
        pos_screen[:, 1] = H - pos_screen[:, 1]

        # 4. Compute 2D covariance
        scales_valid = scales[valid_mask]
        rotations_valid = rotations[valid_mask]
        cov_3d = self._build_covariance_3d(scales_valid, rotations_valid)
        cov_2d = self._project_covariance(cov_3d, pos_cam_valid, K)

        # 5. Sort by depth
        sort_indices = torch.argsort(depth_valid)
        pos_screen = pos_screen[sort_indices]
        cov_2d = cov_2d[sort_indices]
        opacities_valid = opacities[valid_mask][sort_indices]
        colors_valid = colors[valid_mask][sort_indices]

        # 6. Rasterize (colors already final, no SH conversion needed)
        image, alpha = self._rasterize(
            pos_screen, cov_2d, opacities_valid, colors_valid,
            H, W, background
        )

        info = {
            'visible': valid_mask.sum().item(),
            'total': N,
            'device': str(self.device),
        }

        return image, alpha, info

    def render_scene(
        self,
        scene_builder: 'RadianceSceneBuilder',
        camera: CameraParams,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Render a complete scene from a RadianceSceneBuilder.

        Convenience method that builds the batch and renders in one call.

        Args:
            scene_builder: The scene builder with components added
            camera: Camera parameters
            background: Background color

        Returns:
            image, alpha, info (same as render())
        """
        batch = scene_builder.build_render_batch()
        if batch is None:
            H, W = camera.height, camera.width
            bg = torch.tensor(background, device=self.device)
            image = bg.view(1, 1, 3).expand(H, W, 3).contiguous()
            alpha = torch.zeros(H, W, device=self.device)
            return image, alpha, {'visible': 0, 'total': 0, 'components': 0, 'device': str(self.device)}

        return self.render_batch(batch, camera, background)

    def render_component(
        self,
        component: 'RadianceComponent',
        camera: CameraParams,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Render a single RadianceComponent.

        This applies the component's transform, material overrides, and
        region/gaussian overrides.

        Args:
            component: The RadianceComponent to render
            camera: Camera parameters
            background: Background color

        Returns:
            image, alpha, info (same as render())
        """
        if not component.is_loaded:
            H, W = camera.height, camera.width
            bg = torch.tensor(background, device=self.device)
            image = bg.view(1, 1, 3).expand(H, W, 3).contiguous()
            alpha = torch.zeros(H, W, device=self.device)
            return image, alpha, {'visible': 0, 'total': 0, 'device': str(self.device)}

        render_data = component.get_render_data()
        if render_data is None:
            H, W = camera.height, camera.width
            bg = torch.tensor(background, device=self.device)
            image = bg.view(1, 1, 3).expand(H, W, 3).contiguous()
            alpha = torch.zeros(H, W, device=self.device)
            return image, alpha, {'visible': 0, 'total': 0, 'device': str(self.device)}

        positions = torch.from_numpy(render_data['positions'])
        scales = torch.from_numpy(render_data['scales'])
        rotations = torch.from_numpy(render_data['rotations'])
        opacities = torch.from_numpy(render_data['opacities'])
        colors = torch.from_numpy(render_data['colors'])

        image, alpha, info = self._render_with_colors(
            positions, scales, rotations, opacities, colors, camera, background
        )

        info['entity_id'] = component.entity_id

        return image, alpha, info


def create_orbit_camera(
    distance: float = 3.0,
    elevation: float = 0.0,  # degrees
    azimuth: float = 0.0,    # degrees
    target: Tuple[float, float, float] = (0.0, 0.8, 0.0),
    fov: float = 60.0,
    width: int = 512,
    height: int = 512,
) -> CameraParams:
    """
    Create a camera orbiting around a target point.

    Args:
        distance: Distance from target
        elevation: Vertical angle in degrees (0 = horizontal, 90 = top-down)
        azimuth: Horizontal angle in degrees (0 = front, 90 = right)
        target: Point to look at
        fov: Field of view in degrees
        width, height: Image dimensions

    Returns:
        CameraParams with view matrix set
    """
    # Convert to radians
    elev_rad = math.radians(elevation)
    azim_rad = math.radians(azimuth)

    # Camera position in spherical coordinates
    x = target[0] + distance * math.cos(elev_rad) * math.sin(azim_rad)
    y = target[1] + distance * math.sin(elev_rad)
    z = target[2] + distance * math.cos(elev_rad) * math.cos(azim_rad)

    eye = np.array([x, y, z])
    target_arr = np.array(target)
    up = np.array([0.0, 1.0, 0.0])

    # Look-at matrix (positive Z points towards target - in front of camera)
    forward = target_arr - eye  # Points TOWARDS target
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        # Handle case where forward is parallel to up
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / np.linalg.norm(right)

    up_new = np.cross(forward, right)

    # View matrix (world-to-camera)
    # With this convention: positive Z = in front of camera
    view = np.eye(4)
    view[0, :3] = right
    view[1, :3] = up_new
    view[2, :3] = forward  # Z axis points TOWARDS target (positive Z = in front)
    view[:3, 3] = np.array([
        -np.dot(right, eye),
        -np.dot(up_new, eye),
        -np.dot(forward, eye)
    ])

    # Focal length from FOV
    fy = height / (2 * math.tan(math.radians(fov / 2)))
    fx = fy  # Square pixels

    return CameraParams(
        fx=fx, fy=fy,
        cx=width / 2, cy=height / 2,
        width=width, height=height,
        view_matrix=view
    )


def render_turntable(
    radiance_path: str,
    output_dir: str,
    num_frames: int = 36,
    elevation: float = 15.0,
    distance: float = 2.5,
    resolution: int = 512,
) -> list:
    """
    Render a turntable animation of a .radiance asset.

    Args:
        radiance_path: Path to .radiance file
        output_dir: Directory to save frames
        num_frames: Number of frames (360/num_frames degrees per frame)
        elevation: Camera elevation in degrees
        distance: Camera distance from center
        resolution: Image resolution

    Returns:
        List of output frame paths
    """
    import os
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    renderer = GaussianRenderer()
    frame_paths = []

    for i in range(num_frames):
        azimuth = (i / num_frames) * 360

        camera = create_orbit_camera(
            distance=distance,
            elevation=elevation,
            azimuth=azimuth,
            width=resolution,
            height=resolution,
        )

        image, alpha, info = renderer.render_from_radiance(
            radiance_path, camera, background=(0.1, 0.1, 0.1)
        )

        # Convert to PIL image
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        pil_img.save(frame_path)
        frame_paths.append(frame_path)

        logger.info(f"Rendered frame {i+1}/{num_frames}: {info['visible']}/{info['total']} visible")

    return frame_paths


__all__ = [
    'GaussianRenderer',
    'CameraParams',
    'create_orbit_camera',
    'render_turntable',
]


# =============================================================================
# Type hints for cross-module imports
# =============================================================================

if False:  # TYPE_CHECKING
    from .semantic_world.radiance_scene_builder import RenderBatch, RadianceSceneBuilder
    from .radiance_component import RadianceComponent

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
