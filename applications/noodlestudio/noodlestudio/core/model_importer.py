"""
Model Importer - Unified model import for NoodleStudio.

Supports importing 3D models from various formats and converting them
to semantic Gaussian splats (.radiance) with optional humanoid rigging.

Supported Formats:
    - VRM (.vrm) - VR avatars with humanoid mapping and spring bones
    - GLTF/GLB (.gltf, .glb) - Web standard, similar to VRM
    - FBX (.fbx) - Unity/Unreal standard (requires pyassimp)
    - OBJ (.obj) - Simple meshes without rigging

The Pipeline:
    1. Parse model file -> UnifiedMesh + Skeleton
    2. Map to humanoid rig (optional) -> MuscleBinding
    3. Generate Gaussians from mesh vertices
    4. Sample texture colors
    5. Output .radiance file

Muscle System (like Unity Mecanim):
    Instead of storing raw bone rotations, we use a muscle-space
    representation where animations are defined by muscle values
    in a normalized range. This allows retargeting between different
    rigs with the same humanoid mapping.

Author: Caitlyn + NinaK
Date: December 2025
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from enum import IntEnum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Unified Mesh Format (intermediate representation)
# =============================================================================

@dataclass
class UnifiedVertex:
    """A single vertex with all attributes."""
    position: Tuple[float, float, float]
    normal: Optional[Tuple[float, float, float]] = None
    uv: Optional[Tuple[float, float]] = None
    color: Optional[Tuple[float, float, float, float]] = None
    bone_indices: Tuple[int, int, int, int] = (-1, -1, -1, -1)
    bone_weights: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class UnifiedMaterial:
    """Material with texture references."""
    name: str = ""
    diffuse_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    diffuse_texture_idx: int = -1
    normal_texture_idx: int = -1
    metallic: float = 0.0
    roughness: float = 0.5
    emission_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class UnifiedTexture:
    """Texture data."""
    name: str = ""
    width: int = 0
    height: int = 0
    channels: int = 4
    data: Optional[np.ndarray] = None  # (H, W, C) uint8


@dataclass
class UnifiedBone:
    """Skeleton bone."""
    name: str = ""
    index: int = 0
    parent_index: int = -1
    local_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    local_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    humanoid_bone: str = ""  # e.g., "leftUpperArm", "head"


@dataclass
class UnifiedMesh:
    """A single mesh with material and skinning."""
    name: str = ""
    vertices: List[UnifiedVertex] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)  # Triangle indices
    material_idx: int = -1


@dataclass
class UnifiedModel:
    """Complete imported model."""
    name: str = ""
    source_path: str = ""
    source_format: str = ""  # "vrm", "gltf", "fbx", "obj"

    meshes: List[UnifiedMesh] = field(default_factory=list)
    materials: List[UnifiedMaterial] = field(default_factory=list)
    textures: List[UnifiedTexture] = field(default_factory=list)
    bones: List[UnifiedBone] = field(default_factory=list)

    # Humanoid mapping (bone_name -> humanoid_bone_name)
    humanoid_map: Dict[str, str] = field(default_factory=dict)

    # Spring bones (for hair/cloth physics)
    spring_chains: List[Dict[str, Any]] = field(default_factory=list)

    # Blend shapes / morph targets
    blend_shapes: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return sum(len(m.vertices) for m in self.meshes)

    @property
    def bone_count(self) -> int:
        return len(self.bones)

    @property
    def is_rigged(self) -> bool:
        return len(self.bones) > 0

    @property
    def is_humanoid(self) -> bool:
        return len(self.humanoid_map) > 0


# =============================================================================
# Muscle System (Mecanim-style)
# =============================================================================

class HumanoidBone(IntEnum):
    """Standard humanoid bones (Unity-compatible naming)."""
    HIPS = 0
    SPINE = 1
    CHEST = 2
    UPPER_CHEST = 3
    NECK = 4
    HEAD = 5

    LEFT_SHOULDER = 6
    LEFT_UPPER_ARM = 7
    LEFT_LOWER_ARM = 8
    LEFT_HAND = 9

    RIGHT_SHOULDER = 10
    RIGHT_UPPER_ARM = 11
    RIGHT_LOWER_ARM = 12
    RIGHT_HAND = 13

    LEFT_UPPER_LEG = 14
    LEFT_LOWER_LEG = 15
    LEFT_FOOT = 16
    LEFT_TOES = 17

    RIGHT_UPPER_LEG = 18
    RIGHT_LOWER_LEG = 19
    RIGHT_FOOT = 20
    RIGHT_TOES = 21

    # Fingers (optional)
    LEFT_THUMB_PROXIMAL = 22
    LEFT_THUMB_INTERMEDIATE = 23
    LEFT_THUMB_DISTAL = 24
    LEFT_INDEX_PROXIMAL = 25
    LEFT_INDEX_INTERMEDIATE = 26
    LEFT_INDEX_DISTAL = 27
    # ... more fingers


# Humanoid bone name to enum mapping
HUMANOID_BONE_NAMES = {
    "hips": HumanoidBone.HIPS,
    "spine": HumanoidBone.SPINE,
    "chest": HumanoidBone.CHEST,
    "upperChest": HumanoidBone.UPPER_CHEST,
    "neck": HumanoidBone.NECK,
    "head": HumanoidBone.HEAD,
    "leftShoulder": HumanoidBone.LEFT_SHOULDER,
    "leftUpperArm": HumanoidBone.LEFT_UPPER_ARM,
    "leftLowerArm": HumanoidBone.LEFT_LOWER_ARM,
    "leftHand": HumanoidBone.LEFT_HAND,
    "rightShoulder": HumanoidBone.RIGHT_SHOULDER,
    "rightUpperArm": HumanoidBone.RIGHT_UPPER_ARM,
    "rightLowerArm": HumanoidBone.RIGHT_LOWER_ARM,
    "rightHand": HumanoidBone.RIGHT_HAND,
    "leftUpperLeg": HumanoidBone.LEFT_UPPER_LEG,
    "leftLowerLeg": HumanoidBone.LEFT_LOWER_LEG,
    "leftFoot": HumanoidBone.LEFT_FOOT,
    "leftToes": HumanoidBone.LEFT_TOES,
    "rightUpperLeg": HumanoidBone.RIGHT_UPPER_LEG,
    "rightLowerLeg": HumanoidBone.RIGHT_LOWER_LEG,
    "rightFoot": HumanoidBone.RIGHT_FOOT,
    "rightToes": HumanoidBone.RIGHT_TOES,
}


@dataclass
class MuscleDefinition:
    """
    Defines how a muscle maps to bone rotation.

    A muscle is a normalized value [-1, 1] that controls a bone's
    rotation around a specific axis within a defined range.
    """
    name: str
    bone: HumanoidBone
    axis: Tuple[float, float, float]  # Rotation axis
    min_angle: float  # Degrees at muscle = -1
    max_angle: float  # Degrees at muscle = +1
    default_value: float = 0.0


# Standard muscle definitions (simplified subset)
STANDARD_MUSCLES = [
    # Spine
    MuscleDefinition("Spine Front-Back", HumanoidBone.SPINE, (1, 0, 0), -40, 40),
    MuscleDefinition("Spine Left-Right", HumanoidBone.SPINE, (0, 0, 1), -40, 40),
    MuscleDefinition("Spine Twist", HumanoidBone.SPINE, (0, 1, 0), -40, 40),

    # Head
    MuscleDefinition("Head Nod", HumanoidBone.HEAD, (1, 0, 0), -40, 40),
    MuscleDefinition("Head Tilt", HumanoidBone.HEAD, (0, 0, 1), -40, 40),
    MuscleDefinition("Head Turn", HumanoidBone.HEAD, (0, 1, 0), -75, 75),

    # Left Arm
    MuscleDefinition("Left Arm Down-Up", HumanoidBone.LEFT_UPPER_ARM, (0, 0, 1), -60, 100),
    MuscleDefinition("Left Arm Front-Back", HumanoidBone.LEFT_UPPER_ARM, (1, 0, 0), -60, 100),
    MuscleDefinition("Left Arm Twist", HumanoidBone.LEFT_UPPER_ARM, (0, 1, 0), -90, 90),
    MuscleDefinition("Left Forearm Stretch", HumanoidBone.LEFT_LOWER_ARM, (0, 0, 1), -80, 80),
    MuscleDefinition("Left Forearm Twist", HumanoidBone.LEFT_LOWER_ARM, (0, 1, 0), -90, 90),

    # Right Arm (mirror of left)
    MuscleDefinition("Right Arm Down-Up", HumanoidBone.RIGHT_UPPER_ARM, (0, 0, -1), -60, 100),
    MuscleDefinition("Right Arm Front-Back", HumanoidBone.RIGHT_UPPER_ARM, (1, 0, 0), -60, 100),
    MuscleDefinition("Right Arm Twist", HumanoidBone.RIGHT_UPPER_ARM, (0, 1, 0), -90, 90),
    MuscleDefinition("Right Forearm Stretch", HumanoidBone.RIGHT_LOWER_ARM, (0, 0, -1), -80, 80),
    MuscleDefinition("Right Forearm Twist", HumanoidBone.RIGHT_LOWER_ARM, (0, 1, 0), -90, 90),

    # Legs
    MuscleDefinition("Left Upper Leg Front-Back", HumanoidBone.LEFT_UPPER_LEG, (1, 0, 0), -90, 50),
    MuscleDefinition("Left Upper Leg In-Out", HumanoidBone.LEFT_UPPER_LEG, (0, 0, 1), -60, 60),
    MuscleDefinition("Left Lower Leg Stretch", HumanoidBone.LEFT_LOWER_LEG, (1, 0, 0), -80, 80),

    MuscleDefinition("Right Upper Leg Front-Back", HumanoidBone.RIGHT_UPPER_LEG, (1, 0, 0), -90, 50),
    MuscleDefinition("Right Upper Leg In-Out", HumanoidBone.RIGHT_UPPER_LEG, (0, 0, -1), -60, 60),
    MuscleDefinition("Right Lower Leg Stretch", HumanoidBone.RIGHT_LOWER_LEG, (1, 0, 0), -80, 80),
]


@dataclass
class MuscleBinding:
    """
    Binds a model's skeleton to the muscle system.

    This allows any humanoid model to use the same muscle-based
    animations regardless of its specific bone hierarchy.
    """
    model_name: str = ""

    # Mapping from model bone index to humanoid bone
    bone_to_humanoid: Dict[int, HumanoidBone] = field(default_factory=dict)

    # Rest pose (T-pose or A-pose)
    rest_rotations: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)

    # Per-muscle calibration (for models with different proportions)
    muscle_scales: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Model Parsers (Abstract)
# =============================================================================

class ModelParser(ABC):
    """Abstract base class for model parsers."""

    @abstractmethod
    def can_parse(self, path: str) -> bool:
        """Check if this parser can handle the file."""
        pass

    @abstractmethod
    def parse(self, path: str) -> UnifiedModel:
        """Parse the model file into UnifiedModel format."""
        pass


class VRMParser(ModelParser):
    """Parser for VRM files (delegates to vrm_parser.py)."""

    def can_parse(self, path: str) -> bool:
        return Path(path).suffix.lower() == '.vrm'

    def parse(self, path: str) -> UnifiedModel:
        from .semantic_world.vrm_parser import parse_vrm

        avatar = parse_vrm(path)
        model = UnifiedModel(
            name=avatar.metadata.title or Path(path).stem,
            source_path=path,
            source_format="vrm",
        )

        # Convert meshes
        for mesh in avatar.meshes:
            unified_mesh = UnifiedMesh(name=mesh.name, material_idx=mesh.material_index or 0)

            for i, pos in enumerate(mesh.vertices):
                vert = UnifiedVertex(
                    position=tuple(pos),
                    normal=tuple(mesh.normals[i]) if mesh.normals is not None else None,
                    uv=tuple(mesh.uvs[i]) if mesh.uvs is not None else None,
                )
                if mesh.joint_indices is not None:
                    vert.bone_indices = tuple(mesh.joint_indices[i])
                if mesh.joint_weights is not None:
                    vert.bone_weights = tuple(mesh.joint_weights[i])
                unified_mesh.vertices.append(vert)

            if mesh.indices is not None:
                unified_mesh.indices = list(mesh.indices)

            model.meshes.append(unified_mesh)

        # Convert materials
        for mat in avatar.materials:
            unified_mat = UnifiedMaterial(
                name=mat.name,
                diffuse_color=mat.diffuse_color if hasattr(mat, 'diffuse_color') else (1, 1, 1, 1),
                diffuse_texture_idx=mat.diffuse_texture if hasattr(mat, 'diffuse_texture') else -1,
            )
            model.materials.append(unified_mat)

        # Convert textures
        import io
        from PIL import Image
        for tex_data in avatar.textures:
            if isinstance(tex_data, bytes):
                try:
                    img = Image.open(io.BytesIO(tex_data)).convert('RGBA')
                    tex = UnifiedTexture(
                        width=img.size[0],
                        height=img.size[1],
                        channels=4,
                        data=np.array(img, dtype=np.uint8),
                    )
                    model.textures.append(tex)
                except:
                    model.textures.append(UnifiedTexture())

        # Convert skeleton
        for bone in avatar.skeleton.bones:
            unified_bone = UnifiedBone(
                name=bone.name,
                index=bone.index,
                parent_index=bone.parent_index,
                local_position=(bone.transform.position.x, bone.transform.position.y, bone.transform.position.z),
                local_rotation=(bone.transform.rotation.x, bone.transform.rotation.y,
                               bone.transform.rotation.z, bone.transform.rotation.w),
                humanoid_bone=bone.humanoid_bone or "",
            )
            model.bones.append(unified_bone)

        # Humanoid mapping
        for humanoid_name, bone_idx in avatar.skeleton.humanoid_map.items():
            if bone_idx < len(model.bones):
                model.humanoid_map[model.bones[bone_idx].name] = humanoid_name

        return model


class GLTFParser(ModelParser):
    """Parser for GLTF/GLB files."""

    def can_parse(self, path: str) -> bool:
        suffix = Path(path).suffix.lower()
        return suffix in ['.gltf', '.glb']

    def parse(self, path: str) -> UnifiedModel:
        # TODO: Implement GLTF parsing
        # Could use pygltflib or similar
        raise NotImplementedError("GLTF parsing not yet implemented")


class FBXParser(ModelParser):
    """Parser for FBX files (requires pyassimp)."""

    def can_parse(self, path: str) -> bool:
        return Path(path).suffix.lower() == '.fbx'

    def parse(self, path: str) -> UnifiedModel:
        # TODO: Implement FBX parsing via pyassimp
        raise NotImplementedError("FBX parsing not yet implemented")


class OBJParser(ModelParser):
    """Parser for OBJ files (simple meshes)."""

    def can_parse(self, path: str) -> bool:
        return Path(path).suffix.lower() == '.obj'

    def parse(self, path: str) -> UnifiedModel:
        # TODO: Implement OBJ parsing
        raise NotImplementedError("OBJ parsing not yet implemented")


# =============================================================================
# Model Import Manager
# =============================================================================

class ModelImporter:
    """
    Unified model import manager.

    Automatically selects the appropriate parser based on file format
    and provides conversion to .radiance format.
    """

    def __init__(self):
        self.parsers: List[ModelParser] = [
            VRMParser(),
            GLTFParser(),
            FBXParser(),
            OBJParser(),
        ]

    def can_import(self, path: str) -> bool:
        """Check if the file format is supported."""
        return any(p.can_parse(path) for p in self.parsers)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file extensions."""
        return ['.vrm', '.gltf', '.glb', '.fbx', '.obj']

    def import_model(self, path: str) -> UnifiedModel:
        """
        Import a 3D model file.

        Args:
            path: Path to the model file

        Returns:
            UnifiedModel instance
        """
        for parser in self.parsers:
            if parser.can_parse(path):
                logger.info(f"Importing {path} with {parser.__class__.__name__}")
                return parser.parse(path)

        raise ValueError(f"Unsupported file format: {path}")

    def convert_to_radiance(
        self,
        model: UnifiedModel,
        output_path: str,
        gaussian_scale: float = 0.005,
        include_textures: bool = True,
    ) -> str:
        """
        Convert a UnifiedModel to .radiance format.

        Args:
            model: The imported model
            output_path: Where to save the .radiance file
            gaussian_scale: Base scale for Gaussians
            include_textures: Whether to sample texture colors

        Returns:
            Path to the saved .radiance file
        """
        from .semantic_world.radiance_format import (
            RadianceAsset, RadianceSkeleton, RadianceBone
        )

        # Collect all vertices
        positions = []
        normals = []
        colors = []
        bone_indices = []
        bone_weights = []

        for mesh in model.meshes:
            for vert in mesh.vertices:
                positions.append(vert.position)
                normals.append(vert.normal or (0, 0, 1))
                bone_indices.append(vert.bone_indices)
                bone_weights.append(vert.bone_weights)

                # Get color from texture or material
                color = (0.5, 0.5, 0.5)  # Default gray
                if include_textures and vert.uv is not None:
                    mat_idx = mesh.material_idx
                    if 0 <= mat_idx < len(model.materials):
                        mat = model.materials[mat_idx]
                        tex_idx = mat.diffuse_texture_idx
                        if 0 <= tex_idx < len(model.textures):
                            tex = model.textures[tex_idx]
                            if tex.data is not None:
                                u, v = vert.uv
                                px = int((u % 1.0) * (tex.width - 1))
                                py = int((1.0 - v % 1.0) * (tex.height - 1))
                                px = max(0, min(tex.width - 1, px))
                                py = max(0, min(tex.height - 1, py))
                                color = tuple(tex.data[py, px, :3] / 255.0)
                colors.append(color)

        positions = np.array(positions, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)

        # Create Gaussians
        n = len(positions)
        scales = np.full((n, 3), gaussian_scale, dtype=np.float32)
        rotations = np.zeros((n, 4), dtype=np.float32)
        rotations[:, 3] = 1.0  # Identity quaternion
        opacities = np.ones(n, dtype=np.float32)

        # Convert colors to SH DC
        SH_C0 = 0.28209479177387814
        sh_dc = (colors - 0.5) / SH_C0

        # Create RadianceAsset
        asset = RadianceAsset()
        asset.positions = positions
        asset.scales = scales
        asset.rotations = rotations
        asset.opacities = opacities
        asset.sh_dc = sh_dc.astype(np.float32)

        # Transfer skeleton
        asset.skeleton = RadianceSkeleton()
        for bone in model.bones:
            asset.skeleton.bones.append(RadianceBone(
                name=bone.name,
                parent_index=bone.parent_index,
                position=bone.local_position,
                rotation=bone.local_rotation,
                scale=bone.local_scale,
            ))

        # Skinning weights
        if model.is_rigged:
            asset.skin_bone_indices = np.array(bone_indices, dtype=np.uint16)
            asset.skin_bone_weights = np.array(bone_weights, dtype=np.float32)

        # Metadata
        asset.metadata.entity_id = model.name
        asset.metadata.display_name = model.name
        asset.compute_bounds()

        # Save
        asset.save(output_path)
        logger.info(f"Saved radiance: {output_path} ({n:,} Gaussians)")

        return output_path

    def create_muscle_binding(self, model: UnifiedModel) -> Optional[MuscleBinding]:
        """
        Create a muscle binding for a humanoid model.

        Returns None if the model is not humanoid.
        """
        if not model.is_humanoid:
            return None

        binding = MuscleBinding(model_name=model.name)

        for bone in model.bones:
            if bone.humanoid_bone and bone.humanoid_bone in HUMANOID_BONE_NAMES:
                humanoid = HUMANOID_BONE_NAMES[bone.humanoid_bone]
                binding.bone_to_humanoid[bone.index] = humanoid
                binding.rest_rotations[bone.index] = bone.local_rotation

        return binding


# Module-level importer instance
_importer = ModelImporter()


def import_model(path: str) -> UnifiedModel:
    """Import a 3D model file."""
    return _importer.import_model(path)


def convert_to_radiance(model: UnifiedModel, output_path: str, **kwargs) -> str:
    """Convert a model to .radiance format."""
    return _importer.convert_to_radiance(model, output_path, **kwargs)


def get_supported_formats() -> List[str]:
    """Get list of supported file extensions."""
    return _importer.get_supported_formats()


__all__ = [
    'UnifiedModel',
    'UnifiedMesh',
    'UnifiedVertex',
    'UnifiedMaterial',
    'UnifiedTexture',
    'UnifiedBone',
    'ModelImporter',
    'ModelParser',
    'MuscleBinding',
    'MuscleDefinition',
    'HumanoidBone',
    'HUMANOID_BONE_NAMES',
    'STANDARD_MUSCLES',
    'import_model',
    'convert_to_radiance',
    'get_supported_formats',
]
