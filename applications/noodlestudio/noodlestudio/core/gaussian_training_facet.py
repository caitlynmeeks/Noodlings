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
#   Gaussian Training Facet - Train 3D Gaussian splats from images.
#
#   This facet wraps Gaussian splatting training, allowing us...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.gaussian_training_facet
# PURPOSE:  gaussian training facet facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TrainingBackend, TrainingStatus, TrainingProgress, TrainingConfig, GaussianTrainingFacet
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import logging
import subprocess
import time
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from enum import Enum

logger = logging.getLogger(__name__)


class TrainingBackend(Enum):
    """Available training backends."""
    OPENSPLAT = "opensplat"    # CLI wrapper (simpler, proven)
    NATIVE = "native"          # Python/gsplat (more control)


class TrainingStatus(Enum):
    """Training job status."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    TRAINING = "training"
    DENSIFYING = "densifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingProgress:
    """Current training progress."""
    status: TrainingStatus = TrainingStatus.IDLE
    iteration: int = 0
    total_iterations: int = 0
    loss: float = 0.0
    gaussian_count: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    message: str = ""

    @property
    def progress_percent(self) -> float:
        if self.total_iterations <= 0:
            return 0.0
        return 100.0 * self.iteration / self.total_iterations

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'iteration': self.iteration,
            'total_iterations': self.total_iterations,
            'progress_percent': self.progress_percent,
            'loss': self.loss,
            'gaussian_count': self.gaussian_count,
            'elapsed_seconds': self.elapsed_seconds,
            'eta_seconds': self.eta_seconds,
            'message': self.message,
        }


@dataclass
class TrainingConfig:
    """Configuration for Gaussian training."""
    # Required
    dataset_path: str = ""          # Path to images + transforms.json
    output_path: str = ""           # Output .ply path

    # Training parameters
    iterations: int = 30000         # Total training iterations
    sh_degree: int = 2              # Spherical harmonics degree (0-3)

    # Densification
    densify_from: int = 500         # Start densification at this iteration
    densify_until: int = 15000      # Stop densification at this iteration
    densify_every: int = 100        # Densify every N iterations

    # Backend
    backend: TrainingBackend = TrainingBackend.OPENSPLAT
    opensplat_path: str = ""        # Path to opensplat binary (auto-detected if empty)

    # Post-processing
    filter_output: bool = True      # Apply background artifact filtering
    min_opacity: float = 0.8        # Filtering threshold
    max_scale: float = 0.05         # Filtering threshold
    max_brightness: float = 2.0     # Filtering threshold

    # Output format
    convert_to_radiance: bool = True  # Convert .ply to .radiance
    entity_id: str = ""               # Entity ID for .radiance metadata
    display_name: str = ""            # Display name for .radiance metadata

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                if key == 'backend':
                    value = TrainingBackend(value)
                setattr(config, key, value)
        return config


class GaussianTrainingFacet:
    """
    Facet for training 3D Gaussian splats.

    This is a long-running facet - training can take 10-30 minutes.
    Progress is reported via callbacks and can be monitored from the UI.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.progress = TrainingProgress()
        self._process: Optional[subprocess.Popen] = None
        self._cancelled = False
        self._progress_callbacks: List[Callable[[TrainingProgress], None]] = []

    def on_progress(self, callback: Callable[[TrainingProgress], None]):
        """Register a progress callback."""
        self._progress_callbacks.append(callback)

    def _emit_progress(self):
        """Notify all progress callbacks."""
        for cb in self._progress_callbacks:
            try:
                cb(self.progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def cancel(self):
        """Cancel ongoing training."""
        self._cancelled = True
        if self._process:
            self._process.terminate()
            self.progress.status = TrainingStatus.CANCELLED
            self.progress.message = "Training cancelled by user"
            self._emit_progress()

    async def train(self, config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
        """
        Run Gaussian splatting training.

        Args:
            config: Training configuration (uses self.config if not provided)

        Returns:
            Dict with 'success', 'output_path', 'gaussian_count', 'message'
        """
        if config:
            self.config = config

        self._cancelled = False
        self.progress = TrainingProgress(
            status=TrainingStatus.INITIALIZING,
            total_iterations=self.config.iterations,
            message="Initializing training..."
        )
        self._emit_progress()

        # Validate config
        if not self.config.dataset_path:
            return self._fail("No dataset path specified")
        if not Path(self.config.dataset_path).exists():
            return self._fail(f"Dataset path does not exist: {self.config.dataset_path}")

        # Auto-generate output path if not specified
        if not self.config.output_path:
            dataset_name = Path(self.config.dataset_path).name
            self.config.output_path = str(
                Path(self.config.dataset_path) / f"{dataset_name}_trained.ply"
            )

        # Run training based on backend
        if self.config.backend == TrainingBackend.OPENSPLAT:
            result = await self._train_opensplat()
        else:
            result = await self._train_native()

        if not result.get('success'):
            return result

        # Post-processing: convert to .radiance
        if self.config.convert_to_radiance:
            result = await self._convert_to_radiance(result['output_path'])

        return result

    async def _train_opensplat(self) -> Dict[str, Any]:
        """Train using OpenSplat CLI."""

        # Find OpenSplat binary
        opensplat_path = self.config.opensplat_path
        if not opensplat_path:
            # Try common locations
            candidates = [
                Path(__file__).parent.parent.parent.parent.parent.parent /
                    "external/OpenSplat/build/opensplat",
                Path.home() / "git/noodlings_clean/external/OpenSplat/build/opensplat",
                Path("/usr/local/bin/opensplat"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    opensplat_path = str(candidate)
                    break

        if not opensplat_path or not Path(opensplat_path).exists():
            return self._fail("OpenSplat binary not found. Build it first.")

        # Build command
        cmd = [
            opensplat_path,
            self.config.dataset_path,
            "-o", self.config.output_path,
            "-n", str(self.config.iterations),
            "--sh-degree", str(self.config.sh_degree),
        ]

        logger.info(f"Starting OpenSplat: {' '.join(cmd)}")
        self.progress.status = TrainingStatus.TRAINING
        self.progress.message = "Training started..."
        self._emit_progress()

        start_time = time.time()

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Parse output for progress
            async for line in self._read_process_output():
                if self._cancelled:
                    break
                self._parse_opensplat_output(line, start_time)

            # Wait for completion
            return_code = self._process.wait()

            if self._cancelled:
                return self._fail("Training cancelled")

            if return_code != 0:
                return self._fail(f"OpenSplat exited with code {return_code}")

            # Success
            self.progress.status = TrainingStatus.COMPLETED
            self.progress.elapsed_seconds = time.time() - start_time
            self.progress.message = "Training completed successfully"
            self._emit_progress()

            return {
                'success': True,
                'output_path': self.config.output_path,
                'gaussian_count': self.progress.gaussian_count,
                'message': self.progress.message,
            }

        except Exception as e:
            logger.exception("OpenSplat training failed")
            return self._fail(str(e))
        finally:
            self._process = None

    async def _read_process_output(self):
        """Async generator for reading process output."""
        while True:
            line = self._process.stdout.readline()
            if not line and self._process.poll() is not None:
                break
            if line:
                yield line.strip()
            await asyncio.sleep(0.01)  # Yield to event loop

    def _parse_opensplat_output(self, line: str, start_time: float):
        """Parse OpenSplat output line for progress info."""
        # Example lines:
        # "Step 1000: 0.0523456 (3%)"
        # "Added 500 gaussians, new count 52500"
        # "Culled 200 gaussians, remaining 52300"

        if line.startswith("Step "):
            try:
                # Parse "Step 1000: 0.0523456 (3%)"
                parts = line.split()
                iteration = int(parts[1].rstrip(':'))
                loss = float(parts[2])

                self.progress.iteration = iteration
                self.progress.loss = loss
                self.progress.elapsed_seconds = time.time() - start_time

                # Estimate ETA
                if iteration > 0:
                    rate = iteration / self.progress.elapsed_seconds
                    remaining = self.config.iterations - iteration
                    self.progress.eta_seconds = remaining / rate if rate > 0 else 0

                self.progress.message = f"Step {iteration}/{self.config.iterations}, Loss: {loss:.6f}"
                self._emit_progress()

            except (ValueError, IndexError):
                pass

        elif "gaussians" in line.lower():
            try:
                # Parse gaussian count updates
                if "remaining" in line:
                    parts = line.split()
                    count = int(parts[-1])
                    self.progress.gaussian_count = count
                elif "new count" in line:
                    parts = line.split()
                    count = int(parts[-1])
                    self.progress.gaussian_count = count
            except (ValueError, IndexError):
                pass

    async def _train_native(self) -> Dict[str, Any]:
        """Train using native Python/gsplat (more integrated)."""
        # This would use gsplat-mps directly for finer control
        # For now, fall back to OpenSplat
        logger.warning("Native training not yet implemented, falling back to OpenSplat")
        self.config.backend = TrainingBackend.OPENSPLAT
        return await self._train_opensplat()

    async def _convert_to_radiance(self, ply_path: str) -> Dict[str, Any]:
        """Convert trained PLY to .radiance format."""
        self.progress.message = "Converting to .radiance format..."
        self._emit_progress()

        try:
            # Import here to avoid circular imports
            from noodlestudio.tools.vrm_to_radiance import ply_to_radiance_filtered

            radiance_path = Path(ply_path).with_suffix('.radiance')

            asset = ply_to_radiance_filtered(
                ply_path=ply_path,
                output_path=str(radiance_path),
                entity_id=self.config.entity_id or Path(ply_path).stem,
                display_name=self.config.display_name or Path(ply_path).stem,
                filter_gaussians=self.config.filter_output,
                min_opacity=self.config.min_opacity,
                max_scale=self.config.max_scale,
                max_sh_brightness=self.config.max_brightness,
            )

            self.progress.gaussian_count = len(asset.positions)
            self.progress.message = f"Converted to {radiance_path.name} ({self.progress.gaussian_count:,} Gaussians)"
            self._emit_progress()

            return {
                'success': True,
                'output_path': str(radiance_path),
                'ply_path': ply_path,
                'gaussian_count': self.progress.gaussian_count,
                'message': self.progress.message,
            }

        except Exception as e:
            logger.exception("Failed to convert to .radiance")
            return {
                'success': True,  # Training succeeded, just conversion failed
                'output_path': ply_path,
                'gaussian_count': self.progress.gaussian_count,
                'message': f"Training succeeded but .radiance conversion failed: {e}",
                'conversion_error': str(e),
            }

    def _fail(self, message: str) -> Dict[str, Any]:
        """Mark training as failed."""
        self.progress.status = TrainingStatus.FAILED
        self.progress.message = message
        self._emit_progress()
        logger.error(f"Gaussian training failed: {message}")
        return {
            'success': False,
            'message': message,
        }

    # === Facet Interface ===

    def process(
        self,
        inputs: Dict[str, Any],
        context: Any = None,
    ) -> Dict[str, Any]:
        """
        Synchronous facet interface (wraps async train).

        Inputs:
            dataset_path: Path to training images
            output_path: (optional) Output path
            iterations: (optional) Number of iterations
            ... other TrainingConfig fields

        Outputs:
            success: bool
            output_path: str
            gaussian_count: int
            progress: dict
        """
        # Build config from inputs
        config = TrainingConfig.from_dict(inputs)

        # Run training
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.train(config))
        finally:
            loop.close()

        result['progress'] = self.progress.to_dict()
        return result


# === Scripting API Extension ===

class GaussianTrainingAPI:
    """
    Scripting API for Gaussian training.

    Exposed as context.noodle.training in ScriptedFacets.

    Example:
        let result = await context.noodle.training.train({
            dataset_path: '/path/to/images',
            iterations: 30000,
            onProgress: (p) => console.log(`${p.progress_percent}%`)
        });
    """

    def __init__(self, runtime=None):
        self._runtime = runtime
        self._active_training: Optional[GaussianTrainingFacet] = None

    async def train(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start Gaussian training.

        Args:
            config: Training configuration dict
                - dataset_path: Path to images + transforms.json
                - output_path: (optional) Output path
                - iterations: (default 30000)
                - sh_degree: (default 2)
                - onProgress: (optional) Progress callback

        Returns:
            Result dict with success, output_path, gaussian_count
        """
        facet = GaussianTrainingFacet()
        self._active_training = facet

        # Register progress callback if provided
        on_progress = config.pop('onProgress', None)
        if on_progress and callable(on_progress):
            facet.on_progress(lambda p: on_progress(p.to_dict()))

        training_config = TrainingConfig.from_dict(config)
        result = await facet.train(training_config)

        self._active_training = None
        return result

    def cancel(self):
        """Cancel active training."""
        if self._active_training:
            self._active_training.cancel()

    def getProgress(self) -> Optional[Dict[str, Any]]:
        """Get current training progress."""
        if self._active_training:
            return self._active_training.progress.to_dict()
        return None


# Export for facet registration
FACET_TYPE = "gaussian_training"
FACET_CLASS = GaussianTrainingFacet

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
