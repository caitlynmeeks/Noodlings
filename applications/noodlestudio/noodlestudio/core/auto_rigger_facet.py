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
#   Auto-Rigger Facet - Automatic rigging for arbitrary meshes.
#
#   Wraps the AutoRigger tool as a facet for use in facet ass...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.auto_rigger_facet
# PURPOSE:  auto rigger facet facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AutoRiggerConfig, AutoRiggerFacet, AutoRiggerAPI
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
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AutoRiggerConfig:
    """Configuration for auto-rigging."""
    # Required
    mesh_path: str = ""         # Path to mesh file (OBJ, FBX, GLTF)
    output_path: str = ""       # Output .radiance path

    # Marker detection
    auto_detect: bool = True    # Auto-detect markers from mesh

    # Manual markers (if auto_detect=False)
    markers: Optional[Dict[str, Tuple[float, float, float]]] = None

    # Options
    densify: bool = True        # Add face centers and edge midpoints
    max_influences: int = 4     # Max bone influences per Gaussian
    weight_falloff: str = "smooth"  # linear, quadratic, smooth

    # Metadata
    entity_id: str = ""
    display_name: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AutoRiggerConfig':
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class AutoRiggerFacet:
    """
    Facet for auto-rigging arbitrary meshes.

    Takes a mesh file and produces a rigged Gaussian avatar with:
    - Auto-detected (or manual) bone markers
    - Fitted humanoid skeleton
    - Direct Gaussian-to-bone skinning weights
    """

    def __init__(self, config: Optional[AutoRiggerConfig] = None):
        self.config = config or AutoRiggerConfig()

    async def rig(self, config: Optional[AutoRiggerConfig] = None) -> Dict[str, Any]:
        """
        Run auto-rigging.

        Args:
            config: Rigging configuration

        Returns:
            Dict with 'success', 'output_path', 'gaussian_count', 'bone_count', 'message'
        """
        if config:
            self.config = config

        # Validate
        if not self.config.mesh_path:
            return self._fail("No mesh_path specified")

        mesh_path = Path(self.config.mesh_path)
        if not mesh_path.exists():
            return self._fail(f"Mesh file not found: {mesh_path}")

        # Auto-generate output path
        if not self.config.output_path:
            self.config.output_path = str(mesh_path.with_suffix('.radiance'))

        try:
            # Import here to avoid circular imports
            from ..tools.auto_rigger import AutoRigger, MarkerSet

            logger.info(f"Auto-rigging mesh: {mesh_path}")

            # Create rigger
            rigger = AutoRigger()
            rigger.skinner.max_influences = self.config.max_influences
            rigger.skinner.falloff = self.config.weight_falloff

            # Load mesh
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, rigger.load_mesh, str(mesh_path))

            # Get markers
            if self.config.auto_detect:
                markers = await loop.run_in_executor(None, rigger.auto_detect_markers)
                logger.info(f"Auto-detected markers: {list(markers.to_dict().keys())}")
            elif self.config.markers:
                markers = MarkerSet.from_dict(self.config.markers)
            else:
                return self._fail("No markers provided and auto_detect=False")

            # Run rigging
            result = await loop.run_in_executor(
                None,
                lambda: rigger.rig(
                    markers,
                    output_path=self.config.output_path,
                    entity_id=self.config.entity_id or mesh_path.stem,
                    display_name=self.config.display_name or mesh_path.stem,
                    densify=self.config.densify,
                )
            )

            return result

        except Exception as e:
            logger.exception("Auto-rigging failed")
            return self._fail(str(e))

    def _fail(self, message: str) -> Dict[str, Any]:
        """Mark rigging as failed."""
        logger.error(f"Auto-rigging failed: {message}")
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
        Synchronous facet interface (wraps async rig).

        Inputs:
            mesh_path: Path to mesh file
            output_path: (optional) Output path
            auto_detect: (optional) Auto-detect markers
            markers: (optional) Manual marker positions
            densify: (optional) Densify Gaussians

        Outputs:
            success: bool
            output_path: str
            gaussian_count: int
            bone_count: int
            message: str
        """
        config = AutoRiggerConfig.from_dict(inputs)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.rig(config))
        finally:
            loop.close()

        return result


# === Scripting API Extension ===

class AutoRiggerAPI:
    """
    Scripting API for auto-rigging.

    Exposed as context.noodle.rigger in ScriptedFacets.

    Example:
        let result = await context.noodle.rigger.rig({
            mesh_path: '/path/to/model.obj',
            auto_detect: true,
            display_name: 'Conker'
        });
    """

    def __init__(self, runtime=None):
        self._runtime = runtime

    async def rig(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-rig a mesh to Gaussian splats.

        Args:
            config: Configuration dict
                - mesh_path: Path to mesh file
                - output_path: (optional) Output path
                - auto_detect: (default true) Auto-detect markers
                - markers: (optional) Manual marker positions
                - densify: (default true) Add face/edge Gaussians

        Returns:
            Result dict with success, output_path, gaussian_count, bone_count
        """
        facet = AutoRiggerFacet()
        rigger_config = AutoRiggerConfig.from_dict(config)
        return await facet.rig(rigger_config)

    async def detect_markers(self, mesh_path: str) -> Dict[str, Any]:
        """
        Auto-detect markers from mesh without rigging.

        Useful for UI preview before committing to rig.
        """
        from ..tools.auto_rigger import AutoRigger

        rigger = AutoRigger()
        rigger.load_mesh(mesh_path)
        markers = rigger.auto_detect_markers()

        return {
            'success': True,
            'markers': markers.to_dict(),
        }


# Export for facet registration
FACET_TYPE = "auto_rigger"
FACET_CLASS = AutoRiggerFacet

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
