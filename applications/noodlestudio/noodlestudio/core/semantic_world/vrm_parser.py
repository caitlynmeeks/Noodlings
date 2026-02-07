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
#   VRM Parser - Parse VRM avatar files for Gaussian Splatting conversion.
#
#   VRM is a glTF-based format for humanoid avatars, popular ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.semantic_world.vrm_parser
# PURPOSE:  Vrm Parser
# LAYER:    Studio / Semantic World
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Vector3, Quaternion, Transform, Bone, Skeleton
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import json
import struct
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, BinaryIO
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @classmethod
    def from_list(cls, data) -> 'Vector3':
        """Parse from list [x,y,z] or dict {x,y,z}."""
        if isinstance(data, dict):
            return cls(x=data.get('x', 0), y=data.get('y', 0), z=data.get('z', 0))
        return cls(x=data[0], y=data[1], z=data[2])


@dataclass
class Quaternion:
    """Quaternion rotation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float32)

    @classmethod
    def from_list(cls, data: List[float]) -> 'Quaternion':
        return cls(x=data[0], y=data[1], z=data[2], w=data[3])


@dataclass
class Transform:
    """Transform with position, rotation, scale."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))


@dataclass
class Bone:
    """Skeleton bone."""
    name: str
    index: int
    parent_index: int = -1
    transform: Transform = field(default_factory=Transform)
    children: List[int] = field(default_factory=list)
    humanoid_bone: Optional[str] = None  # VRM humanoid bone mapping


@dataclass
class Skeleton:
    """Complete skeleton definition."""
    bones: List[Bone] = field(default_factory=list)
    root_bone_index: int = 0
    humanoid_map: Dict[str, int] = field(default_factory=dict)  # humanoid_bone_name -> bone_index
    inverse_bind_matrices: Optional[np.ndarray] = None  # (N, 4, 4) row-major, one per bone

    def get_bone_by_name(self, name: str) -> Optional[Bone]:
        for bone in self.bones:
            if bone.name == name:
                return bone
        return None

    def get_bone_by_humanoid(self, humanoid_name: str) -> Optional[Bone]:
        idx = self.humanoid_map.get(humanoid_name)
        if idx is not None and 0 <= idx < len(self.bones):
            return self.bones[idx]
        return None


@dataclass
class MorphTargetBind:
    """Links a blend shape expression to a mesh's morph target."""
    mesh_index: int          # Index into glTF meshes array
    target_index: int        # Index into primitive.targets array
    weight: float = 1.0      # Bind weight (0-1)


@dataclass
class BlendShape:
    """Morph target / blend shape."""
    name: str
    positions: Optional[np.ndarray] = None  # Delta positions
    normals: Optional[np.ndarray] = None    # Delta normals
    preset: Optional[str] = None            # VRM preset (joy, angry, sorrow, fun, etc.)
    is_binary: bool = False                 # Binary (on/off) vs continuous
    binds: List['MorphTargetBind'] = field(default_factory=list)


@dataclass
class SpringBoneCollider:
    """Collision sphere for spring bone simulation."""
    bone_index: int
    offset: Vector3 = field(default_factory=Vector3)
    radius: float = 0.1


@dataclass
class SpringBoneChain:
    """Spring bone chain for hair/cloth physics."""
    name: str
    bone_indices: List[int] = field(default_factory=list)
    stiffness: float = 1.0
    gravity_power: float = 0.0
    gravity_dir: Vector3 = field(default_factory=lambda: Vector3(0, -1, 0))
    drag_force: float = 0.4
    hit_radius: float = 0.02
    colliders: List[int] = field(default_factory=list)  # Indices into collider list


@dataclass
class SpringBoneSystem:
    """Complete spring bone physics system."""
    chains: List[SpringBoneChain] = field(default_factory=list)
    colliders: List[SpringBoneCollider] = field(default_factory=list)


@dataclass
class MToonMaterial:
    """MToon cel-shading material parameters."""
    name: str
    # Base colors
    diffuse_color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    shade_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1)
    # Textures (indices into texture list)
    diffuse_texture: Optional[int] = None
    shade_texture: Optional[int] = None
    normal_texture: Optional[int] = None
    emission_texture: Optional[int] = None
    # Shading parameters
    shading_shift: float = 0.0
    shading_toony: float = 0.9
    # Rim lighting
    rim_color: Tuple[float, float, float] = (0, 0, 0)
    rim_power: float = 1.0
    # Outline
    outline_width: float = 0.0
    outline_color: Tuple[float, float, float, float] = (0, 0, 0, 1)


@dataclass
class Mesh:
    """Mesh data with skinning."""
    name: str
    vertices: np.ndarray           # (N, 3) positions
    normals: Optional[np.ndarray] = None  # (N, 3)
    uvs: Optional[np.ndarray] = None      # (N, 2)
    indices: Optional[np.ndarray] = None  # (M,) triangle indices
    # Skinning
    joint_indices: Optional[np.ndarray] = None  # (N, 4) bone indices per vertex
    joint_weights: Optional[np.ndarray] = None  # (N, 4) weights per vertex
    # Material
    material_index: Optional[int] = None
    # Morph targets
    morph_targets: Optional[List[np.ndarray]] = None         # List of (N,3) position deltas
    morph_target_normals: Optional[List[np.ndarray]] = None  # List of (N,3) normal deltas
    source_mesh_index: int = -1                              # Original glTF mesh index


@dataclass
class VRMMetadata:
    """VRM avatar metadata."""
    title: str = ""
    version: str = ""
    author: str = ""
    contact: str = ""
    reference: str = ""
    allowed_user: str = "Everyone"  # OnlyAuthor, ExplicitlyLicensedPerson, Everyone
    license_name: str = ""


@dataclass
class VRMAvatar:
    """Complete parsed VRM avatar."""
    metadata: VRMMetadata = field(default_factory=VRMMetadata)
    skeleton: Skeleton = field(default_factory=Skeleton)
    meshes: List[Mesh] = field(default_factory=list)
    blend_shapes: List[BlendShape] = field(default_factory=list)
    spring_bones: SpringBoneSystem = field(default_factory=SpringBoneSystem)
    materials: List[MToonMaterial] = field(default_factory=list)
    textures: List[bytes] = field(default_factory=list)  # Raw texture data
    texture_names: List[str] = field(default_factory=list)

    # Summary stats
    @property
    def vertex_count(self) -> int:
        return sum(m.vertices.shape[0] for m in self.meshes)

    @property
    def bone_count(self) -> int:
        return len(self.skeleton.bones)

    @property
    def spring_chain_count(self) -> int:
        return len(self.spring_bones.chains)


# =============================================================================
# glTF/GLB Parsing
# =============================================================================

class GLTFParser:
    """Parse glTF 2.0 / GLB files."""

    GLTF_MAGIC = 0x46546C67  # 'glTF'
    JSON_CHUNK = 0x4E4F534A  # 'JSON'
    BIN_CHUNK = 0x004E4942   # 'BIN\x00'

    # Component type to numpy dtype
    COMPONENT_TYPES = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }

    # Type to component count
    TYPE_COUNTS = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16,
    }

    def __init__(self):
        self.json_data: Dict[str, Any] = {}
        self.binary_data: bytes = b''
        self.buffers: List[bytes] = []

    def parse_file(self, path: str) -> Dict[str, Any]:
        """Parse a .glb, .vrm, or .gltf file."""
        path = Path(path)

        # Check file magic to determine format (VRM is GLB format)
        with open(path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]

        if magic == self.GLTF_MAGIC:
            return self._parse_glb(path)
        else:
            return self._parse_gltf(path)

    def _parse_glb(self, path: Path) -> Dict[str, Any]:
        """Parse binary GLB file."""
        with open(path, 'rb') as f:
            # Header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != self.GLTF_MAGIC:
                raise ValueError(f"Not a valid GLB file (magic: {hex(magic)})")

            version = struct.unpack('<I', f.read(4))[0]
            total_length = struct.unpack('<I', f.read(4))[0]

            if version != 2:
                raise ValueError(f"Unsupported glTF version: {version}")

            # Read chunks
            while f.tell() < total_length:
                chunk_length = struct.unpack('<I', f.read(4))[0]
                chunk_type = struct.unpack('<I', f.read(4))[0]
                chunk_data = f.read(chunk_length)

                if chunk_type == self.JSON_CHUNK:
                    self.json_data = json.loads(chunk_data.decode('utf-8'))
                elif chunk_type == self.BIN_CHUNK:
                    self.binary_data = chunk_data
                    self.buffers = [chunk_data]

        return self.json_data

    def _parse_gltf(self, path: Path) -> Dict[str, Any]:
        """Parse JSON .gltf file with external buffers."""
        with open(path, 'r') as f:
            self.json_data = json.load(f)

        # Load external buffers
        base_dir = path.parent
        for buffer in self.json_data.get('buffers', []):
            if 'uri' in buffer:
                uri = buffer['uri']
                if uri.startswith('data:'):
                    # Embedded base64
                    import base64
                    data_start = uri.index(',') + 1
                    self.buffers.append(base64.b64decode(uri[data_start:]))
                else:
                    # External file
                    buffer_path = base_dir / uri
                    with open(buffer_path, 'rb') as bf:
                        self.buffers.append(bf.read())

        if self.buffers:
            self.binary_data = self.buffers[0]

        return self.json_data

    def get_accessor_data(self, accessor_index: int) -> np.ndarray:
        """Get data from an accessor as numpy array."""
        accessor = self.json_data['accessors'][accessor_index]
        buffer_view = self.json_data['bufferViews'][accessor['bufferView']]

        # Get buffer
        buffer_index = buffer_view.get('buffer', 0)
        buffer_data = self.buffers[buffer_index]

        # Calculate offset
        offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)

        # Get dtype and count
        dtype = self.COMPONENT_TYPES[accessor['componentType']]
        component_count = self.TYPE_COUNTS[accessor['type']]
        count = accessor['count']

        # Read data
        byte_stride = buffer_view.get('byteStride', 0)

        if byte_stride == 0:
            # Tightly packed
            data = np.frombuffer(
                buffer_data,
                dtype=dtype,
                count=count * component_count,
                offset=offset
            )
            if component_count > 1:
                data = data.reshape((count, component_count))
        else:
            # Strided
            element_size = np.dtype(dtype).itemsize * component_count
            data = np.zeros((count, component_count), dtype=dtype)
            for i in range(count):
                item_offset = offset + i * byte_stride
                item_data = np.frombuffer(
                    buffer_data,
                    dtype=dtype,
                    count=component_count,
                    offset=item_offset
                )
                data[i] = item_data

        return data

    def get_image_data(self, image_index: int) -> bytes:
        """Get raw image data from an image."""
        image = self.json_data['images'][image_index]

        if 'bufferView' in image:
            buffer_view = self.json_data['bufferViews'][image['bufferView']]
            buffer_index = buffer_view.get('buffer', 0)
            offset = buffer_view.get('byteOffset', 0)
            length = buffer_view['byteLength']
            return self.buffers[buffer_index][offset:offset + length]
        elif 'uri' in image:
            uri = image['uri']
            if uri.startswith('data:'):
                import base64
                data_start = uri.index(',') + 1
                return base64.b64decode(uri[data_start:])

        return b''


# =============================================================================
# VRM Parser
# =============================================================================

class VRMParser:
    """Parse VRM avatar files."""

    # VRM 1.0 humanoid bone names
    HUMANOID_BONES = [
        'hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
        'leftEye', 'rightEye', 'jaw',
        'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
        'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
        'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
        'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes',
        # Fingers
        'leftThumbMetacarpal', 'leftThumbProximal', 'leftThumbDistal',
        'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
        'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
        'leftRingProximal', 'leftRingIntermediate', 'leftRingDistal',
        'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
        'rightThumbMetacarpal', 'rightThumbProximal', 'rightThumbDistal',
        'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
        'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
        'rightRingProximal', 'rightRingIntermediate', 'rightRingDistal',
        'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal',
    ]

    # VRM expression presets
    EXPRESSION_PRESETS = [
        'happy', 'angry', 'sad', 'relaxed', 'surprised',
        'aa', 'ih', 'ou', 'ee', 'oh',  # Visemes
        'blink', 'blinkLeft', 'blinkRight',
        'lookUp', 'lookDown', 'lookLeft', 'lookRight',
        'neutral',
    ]

    def __init__(self):
        self.gltf = GLTFParser()
        self.avatar = VRMAvatar()

    def parse(self, path: str) -> VRMAvatar:
        """Parse a VRM file and return the avatar data."""
        logger.info(f"Parsing VRM: {path}")

        # Parse base glTF
        json_data = self.gltf.parse_file(path)

        # Detect VRM version and parse extensions
        extensions = json_data.get('extensions', {})

        if 'VRMC_vrm' in extensions:
            self._parse_vrm_1_0(json_data)
        elif 'VRM' in extensions:
            self._parse_vrm_0_x(json_data)
        else:
            logger.warning("No VRM extension found, parsing as plain glTF")

        # Parse common glTF data
        self._parse_skeleton(json_data)
        self._parse_meshes(json_data)
        self._parse_textures(json_data)

        logger.info(f"Parsed VRM: {self.avatar.vertex_count} vertices, "
                   f"{self.avatar.bone_count} bones, "
                   f"{len(self.avatar.blend_shapes)} blend shapes, "
                   f"{self.avatar.spring_chain_count} spring chains")

        return self.avatar

    def _parse_vrm_1_0(self, json_data: Dict[str, Any]):
        """Parse VRM 1.0 extensions."""
        extensions = json_data.get('extensions', {})

        # VRMC_vrm - Core VRM data
        vrm_ext = extensions.get('VRMC_vrm', {})

        # Metadata
        meta = vrm_ext.get('meta', {})
        self.avatar.metadata = VRMMetadata(
            title=meta.get('name', ''),
            version=meta.get('version', '1.0'),
            author=', '.join(meta.get('authors', [])),
            license_name=meta.get('licenseUrl', ''),
            allowed_user=meta.get('allowedUserName', 'Everyone'),
        )

        # Humanoid bone mapping
        humanoid = vrm_ext.get('humanoid', {})
        human_bones = humanoid.get('humanBones', {})
        for bone_name, bone_data in human_bones.items():
            if 'node' in bone_data:
                self.avatar.skeleton.humanoid_map[bone_name] = bone_data['node']

        # Expressions (blend shapes) - one BlendShape per expression
        expressions = vrm_ext.get('expressions', {})
        preset_exps = expressions.get('preset', {})
        for preset_name, exp_data in preset_exps.items():
            binds = []
            for bind in exp_data.get('morphTargetBinds', []):
                binds.append(MorphTargetBind(
                    mesh_index=bind.get('node', 0),
                    target_index=bind.get('index', 0),
                    weight=bind.get('weight', 1.0),
                ))
            if binds:
                bs = BlendShape(
                    name=preset_name,
                    preset=preset_name,
                    is_binary=exp_data.get('isBinary', False),
                    binds=binds,
                )
                self.avatar.blend_shapes.append(bs)

        # VRMC_springBone - Hair/cloth physics
        spring_ext = extensions.get('VRMC_springBone', {})
        self._parse_spring_bones_1_0(spring_ext)

        # VRMC_materials_mtoon - Cel shading
        # (Parsed per-material in _parse_meshes)

    def _parse_vrm_0_x(self, json_data: Dict[str, Any]):
        """Parse VRM 0.x extension (legacy format)."""
        vrm_ext = json_data.get('extensions', {}).get('VRM', {})

        # Metadata
        meta = vrm_ext.get('meta', {})
        self.avatar.metadata = VRMMetadata(
            title=meta.get('title', ''),
            version=meta.get('version', '0.x'),
            author=meta.get('author', ''),
            contact=meta.get('contactInformation', ''),
            reference=meta.get('reference', ''),
            allowed_user=meta.get('allowedUserName', 'Everyone'),
            license_name=meta.get('licenseName', ''),
        )

        # Humanoid
        humanoid = vrm_ext.get('humanoid', {})
        for bone_data in humanoid.get('humanBones', []):
            bone_name = bone_data.get('bone', '')
            node_index = bone_data.get('node', -1)
            if bone_name and node_index >= 0:
                self.avatar.skeleton.humanoid_map[bone_name] = node_index

        # Blend shapes - one BlendShape per group with all its binds
        blend_shape_master = vrm_ext.get('blendShapeMaster', {})
        for group in blend_shape_master.get('blendShapeGroups', []):
            binds = []
            for bind in group.get('binds', []):
                binds.append(MorphTargetBind(
                    mesh_index=bind.get('mesh', 0),
                    target_index=bind.get('index', 0),
                    weight=bind.get('weight', 100) / 100.0,  # VRM 0.x uses 0-100
                ))
            bs = BlendShape(
                name=group.get('name', ''),
                preset=group.get('presetName', ''),
                is_binary=group.get('isBinary', False),
                binds=binds,
            )
            self.avatar.blend_shapes.append(bs)

        # Spring bones (0.x format)
        secondary_anim = vrm_ext.get('secondaryAnimation', {})
        self._parse_spring_bones_0_x(secondary_anim)

    def _parse_spring_bones_1_0(self, spring_ext: Dict[str, Any]):
        """Parse VRMC_springBone extension."""
        # Colliders
        for collider_data in spring_ext.get('colliders', []):
            shape = collider_data.get('shape', {})
            if 'sphere' in shape:
                sphere = shape['sphere']
                collider = SpringBoneCollider(
                    bone_index=collider_data.get('node', 0),
                    offset=Vector3.from_list(sphere.get('offset', [0, 0, 0])),
                    radius=sphere.get('radius', 0.1),
                )
                self.avatar.spring_bones.colliders.append(collider)

        # Collider groups
        collider_groups = spring_ext.get('colliderGroups', [])

        # Springs
        for spring_data in spring_ext.get('springs', []):
            joints = spring_data.get('joints', [])
            if not joints:
                continue

            chain = SpringBoneChain(
                name=spring_data.get('name', f'spring_{len(self.avatar.spring_bones.chains)}'),
                bone_indices=[j.get('node', 0) for j in joints],
                stiffness=joints[0].get('stiffness', 1.0) if joints else 1.0,
                gravity_power=joints[0].get('gravityPower', 0.0) if joints else 0.0,
                drag_force=joints[0].get('dragForce', 0.4) if joints else 0.4,
                hit_radius=joints[0].get('hitRadius', 0.02) if joints else 0.02,
            )

            # Resolve collider groups
            for group_idx in spring_data.get('colliderGroups', []):
                if group_idx < len(collider_groups):
                    chain.colliders.extend(collider_groups[group_idx].get('colliders', []))

            self.avatar.spring_bones.chains.append(chain)

    def _parse_spring_bones_0_x(self, secondary_anim: Dict[str, Any]):
        """Parse VRM 0.x secondary animation (spring bones)."""
        # Collider groups
        for group_data in secondary_anim.get('colliderGroups', []):
            bone_idx = group_data.get('node', 0)
            for collider_data in group_data.get('colliders', []):
                collider = SpringBoneCollider(
                    bone_index=bone_idx,
                    offset=Vector3.from_list(collider_data.get('offset', [0, 0, 0])),
                    radius=collider_data.get('radius', 0.1),
                )
                self.avatar.spring_bones.colliders.append(collider)

        # Bone groups (springs)
        for group_data in secondary_anim.get('boneGroups', []):
            chain = SpringBoneChain(
                name=group_data.get('comment', f'spring_{len(self.avatar.spring_bones.chains)}'),
                bone_indices=group_data.get('bones', []),
                stiffness=group_data.get('stiffiness', 1.0),  # Note: typo in VRM 0.x spec
                gravity_power=group_data.get('gravityPower', 0.0),
                gravity_dir=Vector3.from_list(group_data.get('gravityDir', [0, -1, 0])),
                drag_force=group_data.get('dragForce', 0.4),
                hit_radius=group_data.get('hitRadius', 0.02),
                colliders=group_data.get('colliderGroups', []),
            )
            self.avatar.spring_bones.chains.append(chain)

    def _parse_skeleton(self, json_data: Dict[str, Any]):
        """Parse skeleton from glTF nodes."""
        nodes = json_data.get('nodes', [])

        # Find skin (skeleton)
        skins = json_data.get('skins', [])
        if skins:
            skin = skins[0]
            joint_indices = list(skin.get('joints', []))

            # Get inverse bind matrices (one per joint in the skin)
            ibm_accessor = skin.get('inverseBindMatrices')
            inverse_bind_matrices = None
            original_joint_count = len(joint_indices)
            if ibm_accessor is not None:
                ibm_raw = self.gltf.get_accessor_data(ibm_accessor)
                # glTF stores mat4 as 16 floats in column-major order.
                # Reshape to (N, 4, 4) then transpose each matrix to row-major
                # (matching our convention - we upload with GL_TRUE transpose).
                inverse_bind_matrices = ibm_raw.reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float32)

            # Save original humanoid map (glTF node indices) before modifying
            # This is critical: we'll modify the map values to bone list indices,
            # but we need the original glTF indices for the lookup
            original_humanoid_map = dict(self.avatar.skeleton.humanoid_map)

            # IMPORTANT: Add any humanoid bones that aren't in the skin's joint list
            # Some VRMs have incomplete skins that exclude leg bones
            for hname, gltf_node_idx in original_humanoid_map.items():
                if gltf_node_idx not in joint_indices and gltf_node_idx < len(nodes):
                    joint_indices.append(gltf_node_idx)
                    logger.info(f"Added missing humanoid bone: {hname} (node {gltf_node_idx})")

            # Build bone list from joints
            for i, joint_idx in enumerate(joint_indices):
                node = nodes[joint_idx]

                bone = Bone(
                    name=node.get('name', f'bone_{i}'),
                    index=i,
                )

                # Transform
                if 'translation' in node:
                    bone.transform.position = Vector3.from_list(node['translation'])
                if 'rotation' in node:
                    bone.transform.rotation = Quaternion.from_list(node['rotation'])
                if 'scale' in node:
                    bone.transform.scale = Vector3.from_list(node['scale'])

                # Children
                bone.children = [
                    joint_indices.index(c)
                    for c in node.get('children', [])
                    if c in joint_indices
                ]

                # Humanoid mapping - check against original (unmodified) glTF node indices
                for hname, original_gltf_idx in original_humanoid_map.items():
                    if original_gltf_idx == joint_idx:
                        bone.humanoid_bone = hname
                        # Update map to use bone list index (0-based index in our bone array)
                        self.avatar.skeleton.humanoid_map[hname] = i
                        break

                self.avatar.skeleton.bones.append(bone)

            # Find parents
            for i, joint_idx in enumerate(joint_indices):
                node = nodes[joint_idx]
                for j, other_idx in enumerate(joint_indices):
                    other_node = nodes[other_idx]
                    if joint_idx in other_node.get('children', []):
                        self.avatar.skeleton.bones[i].parent_index = j
                        break

            # Find root
            for i, bone in enumerate(self.avatar.skeleton.bones):
                if bone.parent_index == -1:
                    self.avatar.skeleton.root_bone_index = i
                    break

            # Store inverse bind matrices on skeleton.
            # The IBM accessor provides matrices for the original skin joints.
            # Extra humanoid bones added beyond that get identity matrices.
            num_bones = len(self.avatar.skeleton.bones)
            if inverse_bind_matrices is not None:
                ibm_full = np.zeros((num_bones, 4, 4), dtype=np.float32)
                # Identity for all bones as default
                for i in range(num_bones):
                    ibm_full[i] = np.eye(4, dtype=np.float32)
                # Copy parsed matrices for original skin joints
                ibm_count = min(len(inverse_bind_matrices), original_joint_count)
                ibm_full[:ibm_count] = inverse_bind_matrices[:ibm_count]
                self.avatar.skeleton.inverse_bind_matrices = ibm_full

    def _parse_meshes(self, json_data: Dict[str, Any]):
        """Parse meshes with skinning data and morph targets."""
        meshes = json_data.get('meshes', [])

        for mesh_idx, mesh_data in enumerate(meshes):
            for primitive in mesh_data.get('primitives', []):
                attributes = primitive.get('attributes', {})

                # Positions (required)
                if 'POSITION' not in attributes:
                    continue

                positions = self.gltf.get_accessor_data(attributes['POSITION'])

                mesh = Mesh(
                    name=mesh_data.get('name', f'mesh_{len(self.avatar.meshes)}'),
                    vertices=positions,
                    source_mesh_index=mesh_idx,
                )

                # Normals
                if 'NORMAL' in attributes:
                    mesh.normals = self.gltf.get_accessor_data(attributes['NORMAL'])

                # UVs
                if 'TEXCOORD_0' in attributes:
                    mesh.uvs = self.gltf.get_accessor_data(attributes['TEXCOORD_0'])

                # Indices
                if 'indices' in primitive:
                    mesh.indices = self.gltf.get_accessor_data(primitive['indices'])

                # Skinning
                if 'JOINTS_0' in attributes:
                    mesh.joint_indices = self.gltf.get_accessor_data(attributes['JOINTS_0'])
                if 'WEIGHTS_0' in attributes:
                    mesh.joint_weights = self.gltf.get_accessor_data(attributes['WEIGHTS_0'])

                # Material
                mesh.material_index = primitive.get('material')

                # Morph targets (blend shape vertex deltas)
                targets = primitive.get('targets', [])
                if targets:
                    mesh.morph_targets = []
                    mesh.morph_target_normals = []
                    for target in targets:
                        if 'POSITION' in target:
                            delta_pos = self.gltf.get_accessor_data(target['POSITION'])
                            mesh.morph_targets.append(
                                np.asarray(delta_pos, dtype=np.float32)
                            )
                        if 'NORMAL' in target:
                            delta_norm = self.gltf.get_accessor_data(target['NORMAL'])
                            mesh.morph_target_normals.append(
                                np.asarray(delta_norm, dtype=np.float32)
                            )

                self.avatar.meshes.append(mesh)

        # Parse materials
        self._parse_materials(json_data)

    def _parse_materials(self, json_data: Dict[str, Any]):
        """Parse materials, looking for MToon extensions."""
        materials = json_data.get('materials', [])

        for mat_data in materials:
            extensions = mat_data.get('extensions', {})

            # Check for MToon
            mtoon_ext = extensions.get('VRMC_materials_mtoon', {})

            material = MToonMaterial(
                name=mat_data.get('name', f'material_{len(self.avatar.materials)}'),
            )

            # Base color from PBR
            pbr = mat_data.get('pbrMetallicRoughness', {})
            if 'baseColorFactor' in pbr:
                material.diffuse_color = tuple(pbr['baseColorFactor'])
            if 'baseColorTexture' in pbr:
                material.diffuse_texture = pbr['baseColorTexture'].get('index')

            # MToon specific
            if mtoon_ext:
                if 'shadeColorFactor' in mtoon_ext:
                    material.shade_color = tuple(mtoon_ext['shadeColorFactor'])
                material.shading_shift = mtoon_ext.get('shadingShiftFactor', 0.0)
                material.shading_toony = mtoon_ext.get('shadingToonyFactor', 0.9)

                # Outline
                outline = mtoon_ext.get('outlineWidthMode', 'none')
                if outline != 'none':
                    material.outline_width = mtoon_ext.get('outlineWidthFactor', 0.0)
                    if 'outlineColorFactor' in mtoon_ext:
                        material.outline_color = tuple(mtoon_ext['outlineColorFactor'])

            self.avatar.materials.append(material)

    def _parse_textures(self, json_data: Dict[str, Any]):
        """Parse and extract textures."""
        textures = json_data.get('textures', [])
        images = json_data.get('images', [])

        for tex_data in textures:
            source_idx = tex_data.get('source')
            if source_idx is not None and source_idx < len(images):
                image_data = self.gltf.get_image_data(source_idx)
                self.avatar.textures.append(image_data)
                self.avatar.texture_names.append(
                    images[source_idx].get('name', f'texture_{len(self.avatar.textures)}')
                )


# =============================================================================
# Export Functions
# =============================================================================

def export_skeleton_json(avatar: VRMAvatar, output_path: str):
    """Export skeleton to JSON for Gaussian skinning."""
    skeleton_data = {
        'bone_count': len(avatar.skeleton.bones),
        'root_bone': avatar.skeleton.root_bone_index,
        'humanoid_map': avatar.skeleton.humanoid_map,
        'bones': []
    }

    for bone in avatar.skeleton.bones:
        bone_data = {
            'name': bone.name,
            'index': bone.index,
            'parent': bone.parent_index,
            'children': bone.children,
            'humanoid_bone': bone.humanoid_bone,
            'position': [bone.transform.position.x, bone.transform.position.y, bone.transform.position.z],
            'rotation': [bone.transform.rotation.x, bone.transform.rotation.y,
                        bone.transform.rotation.z, bone.transform.rotation.w],
            'scale': [bone.transform.scale.x, bone.transform.scale.y, bone.transform.scale.z],
        }
        skeleton_data['bones'].append(bone_data)

    with open(output_path, 'w') as f:
        json.dump(skeleton_data, f, indent=2)

    logger.info(f"Exported skeleton to {output_path}")


def export_skinning_json(avatar: VRMAvatar, output_path: str):
    """Export skinning weights for vertex-to-bone mapping."""
    skinning_data = {
        'vertex_count': avatar.vertex_count,
        'bone_count': len(avatar.skeleton.bones),
        'meshes': []
    }

    for mesh in avatar.meshes:
        mesh_data = {
            'name': mesh.name,
            'vertex_count': mesh.vertices.shape[0],
            'has_skinning': mesh.joint_indices is not None,
        }

        if mesh.joint_indices is not None:
            # Convert to list for JSON
            mesh_data['joint_indices'] = mesh.joint_indices.tolist()
            mesh_data['joint_weights'] = mesh.joint_weights.tolist()

        skinning_data['meshes'].append(mesh_data)

    with open(output_path, 'w') as f:
        json.dump(skinning_data, f, indent=2)

    logger.info(f"Exported skinning to {output_path}")


def export_spring_bones_json(avatar: VRMAvatar, output_path: str):
    """Export spring bone definitions for physics simulation."""
    spring_data = {
        'chain_count': len(avatar.spring_bones.chains),
        'collider_count': len(avatar.spring_bones.colliders),
        'chains': [],
        'colliders': []
    }

    for chain in avatar.spring_bones.chains:
        chain_data = {
            'name': chain.name,
            'bone_indices': chain.bone_indices,
            'stiffness': chain.stiffness,
            'gravity_power': chain.gravity_power,
            'gravity_dir': [chain.gravity_dir.x, chain.gravity_dir.y, chain.gravity_dir.z],
            'drag_force': chain.drag_force,
            'hit_radius': chain.hit_radius,
            'colliders': chain.colliders,
        }
        spring_data['chains'].append(chain_data)

    for collider in avatar.spring_bones.colliders:
        collider_data = {
            'bone_index': collider.bone_index,
            'offset': [collider.offset.x, collider.offset.y, collider.offset.z],
            'radius': collider.radius,
        }
        spring_data['colliders'].append(collider_data)

    with open(output_path, 'w') as f:
        json.dump(spring_data, f, indent=2)

    logger.info(f"Exported spring bones to {output_path}")


def export_blend_shapes_json(avatar: VRMAvatar, output_path: str):
    """Export blend shape definitions."""
    blend_data = {
        'count': len(avatar.blend_shapes),
        'blend_shapes': []
    }

    for bs in avatar.blend_shapes:
        bs_data = {
            'name': bs.name,
            'preset': bs.preset,
            'is_binary': bs.is_binary,
        }
        blend_data['blend_shapes'].append(bs_data)

    with open(output_path, 'w') as f:
        json.dump(blend_data, f, indent=2)

    logger.info(f"Exported blend shapes to {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_vrm(path: str) -> VRMAvatar:
    """
    Parse a VRM file and return the avatar data.

    Args:
        path: Path to .vrm file (GLB format)

    Returns:
        VRMAvatar with skeleton, meshes, blend shapes, spring bones
    """
    parser = VRMParser()
    return parser.parse(path)


def vrm_to_gaussian_package(vrm_path: str, output_dir: str) -> str:
    """
    Convert VRM to a Gaussian-ready package.

    Creates:
        output_dir/
            skeleton.json      - Bone hierarchy for LBS
            skinning.json      - Vertex weights for Gaussian binding
            springbones.json   - Physics simulation params
            blendshapes.json   - Expression definitions
            textures/          - Extracted textures

    Args:
        vrm_path: Path to input .vrm file
        output_dir: Directory to write output files

    Returns:
        Path to output directory
    """
    import os

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse VRM
    avatar = parse_vrm(vrm_path)

    # Export JSON files
    export_skeleton_json(avatar, str(output_path / 'skeleton.json'))
    export_skinning_json(avatar, str(output_path / 'skinning.json'))
    export_spring_bones_json(avatar, str(output_path / 'springbones.json'))
    export_blend_shapes_json(avatar, str(output_path / 'blendshapes.json'))

    # Export textures
    textures_dir = output_path / 'textures'
    textures_dir.mkdir(exist_ok=True)

    for i, (tex_data, tex_name) in enumerate(zip(avatar.textures, avatar.texture_names)):
        # Determine extension from magic bytes
        if tex_data.startswith(b'\x89PNG'):
            ext = '.png'
        elif tex_data.startswith(b'\xff\xd8\xff'):
            ext = '.jpg'
        else:
            ext = '.bin'

        tex_path = textures_dir / f'{tex_name}{ext}'
        with open(tex_path, 'wb') as f:
            f.write(tex_data)

    # Write metadata
    meta_path = output_path / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump({
            'title': avatar.metadata.title,
            'author': avatar.metadata.author,
            'version': avatar.metadata.version,
            'vertex_count': avatar.vertex_count,
            'bone_count': avatar.bone_count,
            'blend_shape_count': len(avatar.blend_shapes),
            'spring_chain_count': avatar.spring_chain_count,
            'material_count': len(avatar.materials),
            'texture_count': len(avatar.textures),
        }, f, indent=2)

    logger.info(f"Created Gaussian package at {output_path}")
    return str(output_path)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vrm_parser.py <path_to_vrm>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    vrm_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './vrm_output'

    result = vrm_to_gaussian_package(vrm_path, output_dir)
    print(f"Output: {result}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
