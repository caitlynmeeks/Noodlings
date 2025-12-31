"""
Skeleton Binding Facet - Bind Gaussians to VRM skeleton.

Wraps the bind_gaussians_to_skeleton tool as a facet for use in
facet assemblies, enabling skeleton-bound Gaussian avatar creation.

Pipeline:
1. Load trained Gaussians (.ply from training)
2. Load VRM (for skeleton and skinning weights)
3. Transfer skinning weights to Gaussians
4. Save as .radiance with skeleton binding

Author: Caitlyn + Claude
Date: December 24, 2025
"""

import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkeletonBindingConfig:
    """Configuration for skeleton binding."""
    # Required
    gaussian_ply_path: str = ""     # Trained Gaussians (.ply)
    vrm_path: str = ""              # VRM file (for skeleton)
    output_path: str = ""           # Output .radiance path

    # Metadata
    entity_id: str = ""             # Entity ID for scene protocol
    display_name: str = ""          # Human-readable name

    # Options
    k_neighbors: int = 4            # Neighbors for weight interpolation

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SkeletonBindingConfig':
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class SkeletonBindingFacet:
    """
    Facet for binding trained Gaussians to VRM skeleton.

    Takes trained Gaussians (.ply) and a VRM file, transfers skinning
    weights from mesh vertices to Gaussian positions, and outputs
    a skeleton-rigged .radiance file.
    """

    def __init__(self, config: Optional[SkeletonBindingConfig] = None):
        self.config = config or SkeletonBindingConfig()

    async def bind(self, config: Optional[SkeletonBindingConfig] = None) -> Dict[str, Any]:
        """
        Bind Gaussians to skeleton.

        Args:
            config: Binding configuration

        Returns:
            Dict with 'success', 'output_path', 'gaussian_count', 'bone_count', 'message'
        """
        if config:
            self.config = config

        # Validate
        if not self.config.gaussian_ply_path:
            return self._fail("No gaussian_ply_path specified")
        if not self.config.vrm_path:
            return self._fail("No vrm_path specified")

        ply_path = Path(self.config.gaussian_ply_path)
        vrm_path = Path(self.config.vrm_path)

        if not ply_path.exists():
            return self._fail(f"PLY file not found: {ply_path}")
        if not vrm_path.exists():
            return self._fail(f"VRM file not found: {vrm_path}")

        # Auto-generate output path
        if not self.config.output_path:
            self.config.output_path = str(ply_path.with_suffix('.radiance'))

        # Run binding
        try:
            # Import here to avoid circular imports
            from ..tools.bind_gaussians_to_skeleton import bind_gaussians_to_skeleton

            logger.info(f"Binding Gaussians to skeleton:")
            logger.info(f"  PLY: {ply_path}")
            logger.info(f"  VRM: {vrm_path}")
            logger.info(f"  Output: {self.config.output_path}")

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            asset = await loop.run_in_executor(
                None,
                lambda: bind_gaussians_to_skeleton(
                    gaussian_ply_path=str(ply_path),
                    vrm_path=str(vrm_path),
                    output_path=self.config.output_path,
                    entity_id=self.config.entity_id or ply_path.stem,
                    display_name=self.config.display_name or ply_path.stem,
                )
            )

            gaussian_count = len(asset.positions) if hasattr(asset, 'positions') else 0
            bone_count = len(asset.bones) if hasattr(asset, 'bones') and asset.bones else 0

            return {
                'success': True,
                'output_path': self.config.output_path,
                'gaussian_count': gaussian_count,
                'bone_count': bone_count,
                'message': f"Created rigged radiance: {gaussian_count:,} Gaussians, {bone_count} bones",
            }

        except Exception as e:
            logger.exception("Skeleton binding failed")
            return self._fail(str(e))

    def _fail(self, message: str) -> Dict[str, Any]:
        """Mark binding as failed."""
        logger.error(f"Skeleton binding failed: {message}")
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
        Synchronous facet interface (wraps async bind).

        Inputs:
            gaussian_ply_path: Path to trained Gaussians
            vrm_path: Path to VRM file
            output_path: (optional) Output path
            entity_id: (optional) Entity ID
            display_name: (optional) Display name

        Outputs:
            success: bool
            output_path: str
            gaussian_count: int
            bone_count: int
            message: str
        """
        config = SkeletonBindingConfig.from_dict(inputs)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.bind(config))
        finally:
            loop.close()

        return result


# === Scripting API Extension ===

class SkeletonBindingAPI:
    """
    Scripting API for skeleton binding.

    Exposed as context.noodle.binding in ScriptedFacets.

    Example:
        let result = await context.noodle.binding.bind({
            gaussian_ply_path: '/path/to/trained.ply',
            vrm_path: '/path/to/avatar.vrm',
            display_name: 'My Avatar'
        });
    """

    def __init__(self, runtime=None):
        self._runtime = runtime

    async def bind(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bind Gaussians to skeleton.

        Args:
            config: Configuration dict
                - gaussian_ply_path: Path to trained PLY
                - vrm_path: Path to VRM file
                - output_path: (optional) Output path
                - entity_id: (optional) Entity ID
                - display_name: (optional) Display name

        Returns:
            Result dict with success, output_path, gaussian_count, bone_count
        """
        facet = SkeletonBindingFacet()
        binding_config = SkeletonBindingConfig.from_dict(config)
        return await facet.bind(binding_config)


# Export for facet registration
FACET_TYPE = "skeleton_binding"
FACET_CLASS = SkeletonBindingFacet
