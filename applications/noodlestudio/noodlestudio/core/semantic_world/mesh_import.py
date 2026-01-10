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
#   Mesh Import Pipeline
#
#   Import arbitrary 3D meshes (glTF, FBX, OBJ) for conversio...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.semantic_world.mesh_import
# PURPOSE:  Mesh Import Pipeline
# LAYER:    Studio / Semantic World
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Vector2, Vector3, MeshMaterial, MeshPrimitive, ImportedMesh
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
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, BinaryIO
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Vector2:
    x: float = 0.0
    y: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @classmethod
    def from_list(cls, data: List[float]) -> 'Vector3':
        return cls(x=data[0], y=data[1], z=data[2])


@dataclass
class MeshMaterial:
    """Material definition for a mesh."""
    name: str
    # PBR properties
    base_color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    metallic: float = 0.0
    roughness: float = 0.5
    # Textures (indices or paths)
    base_color_texture: Optional[str] = None
    normal_texture: Optional[str] = None
    metallic_roughness_texture: Optional[str] = None
    emissive_texture: Optional[str] = None
    emissive_factor: Tuple[float, float, float] = (0, 0, 0)
    # Rendering
    alpha_mode: str = "OPAQUE"  # OPAQUE, MASK, BLEND
    alpha_cutoff: float = 0.5
    double_sided: bool = False


@dataclass
class MeshPrimitive:
    """A single drawable primitive within a mesh."""
    name: str
    vertices: np.ndarray          # (N, 3) positions
    normals: Optional[np.ndarray] = None  # (N, 3)
    uvs: Optional[np.ndarray] = None      # (N, 2)
    indices: Optional[np.ndarray] = None  # (M,) triangle indices
    material_index: Optional[int] = None

    @property
    def vertex_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def triangle_count(self) -> int:
        if self.indices is not None:
            return len(self.indices) // 3
        return self.vertex_count // 3


@dataclass
class ImportedMesh:
    """Complete imported mesh with all data."""
    name: str
    primitives: List[MeshPrimitive] = field(default_factory=list)
    materials: List[MeshMaterial] = field(default_factory=list)
    textures: List[bytes] = field(default_factory=list)
    texture_names: List[str] = field(default_factory=list)

    # Bounds
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Source info
    source_path: str = ""
    source_format: str = ""

    @property
    def vertex_count(self) -> int:
        return sum(p.vertex_count for p in self.primitives)

    @property
    def triangle_count(self) -> int:
        return sum(p.triangle_count for p in self.primitives)

    def compute_bounds(self):
        """Compute bounding box from all vertices."""
        if not self.primitives:
            return

        all_verts = np.vstack([p.vertices for p in self.primitives])
        self.bounds_min = np.min(all_verts, axis=0)
        self.bounds_max = np.max(all_verts, axis=0)

    @property
    def bounds_center(self) -> np.ndarray:
        return (self.bounds_min + self.bounds_max) / 2

    @property
    def bounds_size(self) -> np.ndarray:
        return self.bounds_max - self.bounds_min


# =============================================================================
# glTF Parser (reuse from VRM parser, extended)
# =============================================================================

class GLTFParser:
    """Parse glTF 2.0 / GLB files."""

    GLTF_MAGIC = 0x46546C67  # 'glTF'
    JSON_CHUNK = 0x4E4F534A  # 'JSON'
    BIN_CHUNK = 0x004E4942   # 'BIN\x00'

    COMPONENT_TYPES = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }

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
        self.base_path: Path = Path('.')

    def parse_file(self, path: str) -> Dict[str, Any]:
        """Parse a .glb or .gltf file."""
        path = Path(path)
        self.base_path = path.parent

        with open(path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            f.seek(0)

        if magic == self.GLTF_MAGIC:
            return self._parse_glb(path)
        else:
            return self._parse_gltf(path)

    def _parse_glb(self, path: Path) -> Dict[str, Any]:
        """Parse binary GLB file."""
        with open(path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            version = struct.unpack('<I', f.read(4))[0]
            total_length = struct.unpack('<I', f.read(4))[0]

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

        for buffer in self.json_data.get('buffers', []):
            if 'uri' in buffer:
                uri = buffer['uri']
                if uri.startswith('data:'):
                    import base64
                    data_start = uri.index(',') + 1
                    self.buffers.append(base64.b64decode(uri[data_start:]))
                else:
                    buffer_path = self.base_path / uri
                    with open(buffer_path, 'rb') as bf:
                        self.buffers.append(bf.read())

        if self.buffers:
            self.binary_data = self.buffers[0]

        return self.json_data

    def get_accessor_data(self, accessor_index: int) -> np.ndarray:
        """Get data from an accessor as numpy array."""
        accessor = self.json_data['accessors'][accessor_index]
        buffer_view = self.json_data['bufferViews'][accessor['bufferView']]

        buffer_index = buffer_view.get('buffer', 0)
        buffer_data = self.buffers[buffer_index]
        offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)

        dtype = self.COMPONENT_TYPES[accessor['componentType']]
        component_count = self.TYPE_COUNTS[accessor['type']]
        count = accessor['count']

        byte_stride = buffer_view.get('byteStride', 0)

        if byte_stride == 0:
            data = np.frombuffer(
                buffer_data, dtype=dtype, count=count * component_count, offset=offset
            )
            if component_count > 1:
                data = data.reshape((count, component_count))
        else:
            data = np.zeros((count, component_count), dtype=dtype)
            for i in range(count):
                item_offset = offset + i * byte_stride
                item_data = np.frombuffer(
                    buffer_data, dtype=dtype, count=component_count, offset=item_offset
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
            else:
                image_path = self.base_path / uri
                with open(image_path, 'rb') as f:
                    return f.read()

        return b''


# =============================================================================
# Mesh Importer
# =============================================================================

class MeshImporter:
    """Import 3D meshes from various formats."""

    SUPPORTED_FORMATS = {
        '.gltf': 'glTF',
        '.glb': 'glTF',
        '.obj': 'OBJ',
        '.fbx': 'FBX',
    }

    def __init__(self):
        self.gltf = GLTFParser()

    def import_mesh(self, path: str) -> ImportedMesh:
        """
        Import a mesh file.

        Args:
            path: Path to mesh file (.gltf, .glb, .obj, .fbx)

        Returns:
            ImportedMesh with geometry and materials
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}")

        logger.info(f"Importing mesh: {path}")

        if suffix in ['.gltf', '.glb']:
            mesh = self._import_gltf(path)
        elif suffix == '.obj':
            mesh = self._import_obj(path)
        elif suffix == '.fbx':
            mesh = self._import_fbx(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        mesh.source_path = str(path)
        mesh.source_format = self.SUPPORTED_FORMATS[suffix]
        mesh.compute_bounds()

        logger.info(f"Imported: {mesh.vertex_count} vertices, {mesh.triangle_count} triangles, "
                   f"{len(mesh.materials)} materials")

        return mesh

    def _import_gltf(self, path: Path) -> ImportedMesh:
        """Import glTF/GLB file."""
        json_data = self.gltf.parse_file(str(path))

        mesh = ImportedMesh(name=path.stem)

        # Parse materials
        for mat_data in json_data.get('materials', []):
            material = self._parse_gltf_material(mat_data)
            mesh.materials.append(material)

        # Parse textures
        for tex_data in json_data.get('textures', []):
            source_idx = tex_data.get('source')
            if source_idx is not None:
                image_data = self.gltf.get_image_data(source_idx)
                mesh.textures.append(image_data)
                images = json_data.get('images', [])
                name = images[source_idx].get('name', f'texture_{len(mesh.textures)}')
                mesh.texture_names.append(name)

        # Parse meshes
        for mesh_data in json_data.get('meshes', []):
            mesh_name = mesh_data.get('name', 'mesh')
            for prim_idx, prim_data in enumerate(mesh_data.get('primitives', [])):
                primitive = self._parse_gltf_primitive(prim_data, f"{mesh_name}_{prim_idx}")
                if primitive:
                    mesh.primitives.append(primitive)

        return mesh

    def _parse_gltf_material(self, mat_data: Dict) -> MeshMaterial:
        """Parse a glTF material."""
        material = MeshMaterial(name=mat_data.get('name', 'material'))

        pbr = mat_data.get('pbrMetallicRoughness', {})

        if 'baseColorFactor' in pbr:
            material.base_color = tuple(pbr['baseColorFactor'])

        if 'baseColorTexture' in pbr:
            material.base_color_texture = str(pbr['baseColorTexture'].get('index'))

        material.metallic = pbr.get('metallicFactor', 0.0)
        material.roughness = pbr.get('roughnessFactor', 0.5)

        if 'metallicRoughnessTexture' in pbr:
            material.metallic_roughness_texture = str(pbr['metallicRoughnessTexture'].get('index'))

        if 'normalTexture' in mat_data:
            material.normal_texture = str(mat_data['normalTexture'].get('index'))

        if 'emissiveTexture' in mat_data:
            material.emissive_texture = str(mat_data['emissiveTexture'].get('index'))

        if 'emissiveFactor' in mat_data:
            material.emissive_factor = tuple(mat_data['emissiveFactor'])

        material.alpha_mode = mat_data.get('alphaMode', 'OPAQUE')
        material.alpha_cutoff = mat_data.get('alphaCutoff', 0.5)
        material.double_sided = mat_data.get('doubleSided', False)

        return material

    def _parse_gltf_primitive(self, prim_data: Dict, name: str) -> Optional[MeshPrimitive]:
        """Parse a glTF mesh primitive."""
        attributes = prim_data.get('attributes', {})

        if 'POSITION' not in attributes:
            return None

        positions = self.gltf.get_accessor_data(attributes['POSITION'])

        primitive = MeshPrimitive(
            name=name,
            vertices=positions,
        )

        if 'NORMAL' in attributes:
            primitive.normals = self.gltf.get_accessor_data(attributes['NORMAL'])

        if 'TEXCOORD_0' in attributes:
            primitive.uvs = self.gltf.get_accessor_data(attributes['TEXCOORD_0'])

        if 'indices' in prim_data:
            primitive.indices = self.gltf.get_accessor_data(prim_data['indices'])

        primitive.material_index = prim_data.get('material')

        return primitive

    def _import_obj(self, path: Path) -> ImportedMesh:
        """Import OBJ file (basic support)."""
        mesh = ImportedMesh(name=path.stem)

        vertices = []
        normals = []
        uvs = []
        faces = []

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                if parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vt':
                    uvs.append([float(parts[1]), float(parts[2])])
                elif parts[0] == 'f':
                    face = []
                    for p in parts[1:]:
                        indices = p.split('/')
                        v_idx = int(indices[0]) - 1
                        face.append(v_idx)
                    # Triangulate if needed
                    for i in range(1, len(face) - 1):
                        faces.extend([face[0], face[i], face[i + 1]])

        if vertices:
            primitive = MeshPrimitive(
                name=path.stem,
                vertices=np.array(vertices, dtype=np.float32),
                normals=np.array(normals, dtype=np.float32) if normals else None,
                uvs=np.array(uvs, dtype=np.float32) if uvs else None,
                indices=np.array(faces, dtype=np.uint32) if faces else None,
            )
            mesh.primitives.append(primitive)

        return mesh

    def _import_fbx(self, path: Path) -> ImportedMesh:
        """Import FBX file (requires external tool or library)."""
        # FBX is proprietary - options:
        # 1. Use Autodesk FBX SDK (C++ with Python bindings)
        # 2. Convert to glTF using Blender CLI
        # 3. Use assimp library

        # For now, try converting via Blender if available
        try:
            mesh = self._fbx_via_blender(path)
            if mesh:
                return mesh
        except Exception as e:
            logger.warning(f"Blender conversion failed: {e}")

        raise NotImplementedError(
            "FBX import requires Blender installed. "
            "Please convert to glTF format first, or install Blender."
        )

    def _fbx_via_blender(self, path: Path) -> Optional[ImportedMesh]:
        """Convert FBX to glTF using Blender CLI."""
        import shutil

        blender_path = shutil.which('blender')
        if not blender_path:
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"{path.stem}.glb"

            script = f'''
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath="{path}")
bpy.ops.export_scene.gltf(filepath="{output_path}", export_format='GLB')
'''
            script_path = Path(tmpdir) / "convert.py"
            with open(script_path, 'w') as f:
                f.write(script)

            result = subprocess.run(
                [blender_path, '--background', '--python', str(script_path)],
                capture_output=True,
                timeout=60
            )

            if result.returncode == 0 and output_path.exists():
                return self._import_gltf(output_path)

        return None


# =============================================================================
# Gaussian Conversion Pipeline
# =============================================================================

@dataclass
class GaussianConversionConfig:
    """Configuration for mesh-to-Gaussian conversion."""
    # Multi-view rendering
    num_views: int = 36          # Number of camera positions
    view_distance: float = 2.0   # Distance from center
    image_resolution: int = 512  # Render resolution

    # Gaussian training
    num_iterations: int = 2000
    num_gaussians: int = 50000

    # Output
    output_format: str = "ply"


class MeshToGaussianPipeline:
    """
    Convert imported meshes to Gaussian splats.

    Pipeline:
    1. Generate multi-view renders of the mesh
    2. Run COLMAP for camera poses
    3. Train Gaussians via OpenSplat
    4. Output .ply file
    """

    def __init__(self, config: GaussianConversionConfig = None):
        self.config = config or GaussianConversionConfig()
        self.importer = MeshImporter()

    def convert(self, mesh_path: str, output_dir: str) -> str:
        """
        Convert a mesh file to Gaussian splats.

        Args:
            mesh_path: Path to input mesh
            output_dir: Directory for output files

        Returns:
            Path to output .ply file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting mesh to Gaussians: {mesh_path}")

        # 1. Import mesh
        mesh = self.importer.import_mesh(mesh_path)

        # 2. Generate multi-view renders
        views_dir = output_dir / "views"
        views_dir.mkdir(exist_ok=True)
        cameras = self._generate_views(mesh, views_dir)

        # 3. Run COLMAP (or use known cameras)
        cameras_json = output_dir / "cameras.json"
        self._write_cameras(cameras, cameras_json)

        # 4. Train Gaussians
        output_ply = output_dir / f"{mesh.name}_splat.ply"
        self._train_gaussians(views_dir, cameras_json, output_ply)

        # 5. Write metadata
        metadata = {
            "source_mesh": mesh_path,
            "vertex_count": mesh.vertex_count,
            "triangle_count": mesh.triangle_count,
            "bounds_min": mesh.bounds_min.tolist(),
            "bounds_max": mesh.bounds_max.tolist(),
            "num_views": self.config.num_views,
            "num_iterations": self.config.num_iterations,
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Conversion complete: {output_ply}")
        return str(output_ply)

    def _generate_views(self, mesh: ImportedMesh, output_dir: Path) -> List[Dict]:
        """Generate multi-view renders of the mesh."""
        # This would ideally use a proper renderer (e.g., Blender, PyTorch3D)
        # For now, we'll generate camera positions and assume external rendering

        cameras = []
        center = mesh.bounds_center
        radius = np.linalg.norm(mesh.bounds_size) / 2 * self.config.view_distance

        for i in range(self.config.num_views):
            # Spiral camera positions
            t = i / self.config.num_views
            theta = t * 4 * np.pi  # 2 rotations
            phi = np.pi * 0.2 + t * np.pi * 0.6  # 20° to 80° elevation

            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.cos(phi)
            z = center[2] + radius * np.sin(phi) * np.sin(theta)

            camera = {
                "id": i,
                "position": [x, y, z],
                "target": center.tolist(),
                "up": [0, 1, 0],
                "fov": 60,
                "image_path": str(output_dir / f"view_{i:03d}.png"),
            }
            cameras.append(camera)

        logger.info(f"Generated {len(cameras)} camera positions")
        return cameras

    def _write_cameras(self, cameras: List[Dict], output_path: Path):
        """Write cameras.json for OpenSplat."""
        # Convert to OpenSplat format
        output = []
        for cam in cameras:
            pos = np.array(cam["position"])
            target = np.array(cam["target"])
            up = np.array(cam["up"])

            # Compute rotation matrix
            forward = target - pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            rotation = np.array([right, up, -forward]).T

            output.append({
                "id": cam["id"],
                "img_name": Path(cam["image_path"]).name,
                "width": self.config.image_resolution,
                "height": self.config.image_resolution,
                "position": pos.tolist(),
                "rotation": rotation.flatten().tolist(),
                "fx": self.config.image_resolution / (2 * np.tan(np.radians(cam["fov"]) / 2)),
                "fy": self.config.image_resolution / (2 * np.tan(np.radians(cam["fov"]) / 2)),
            })

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    def _train_gaussians(self, views_dir: Path, cameras_json: Path, output_ply: Path):
        """Run OpenSplat training."""
        import shutil

        opensplat_path = shutil.which('opensplat')
        if not opensplat_path:
            # Try local build
            local_path = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "external" / "OpenSplat" / "build" / "opensplat"
            if local_path.exists():
                opensplat_path = str(local_path)
            else:
                raise RuntimeError("OpenSplat not found. Please install or build it.")

        cmd = [
            opensplat_path,
            str(views_dir.parent),
            "-n", str(self.config.num_iterations),
            "-o", str(output_ply),
        ]

        logger.info(f"Running OpenSplat: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"OpenSplat failed: {result.stderr}")
            raise RuntimeError(f"OpenSplat training failed: {result.stderr}")


# =============================================================================
# Factory Functions
# =============================================================================

def import_mesh(path: str) -> ImportedMesh:
    """Import a mesh file."""
    importer = MeshImporter()
    return importer.import_mesh(path)


def mesh_to_gaussians(
    mesh_path: str,
    output_dir: str,
    num_views: int = 36,
    num_iterations: int = 2000,
) -> str:
    """
    Convert a mesh to Gaussian splats.

    Args:
        mesh_path: Path to input mesh (.gltf, .glb, .obj, .fbx)
        output_dir: Output directory
        num_views: Number of training views
        num_iterations: Training iterations

    Returns:
        Path to output .ply file
    """
    config = GaussianConversionConfig(
        num_views=num_views,
        num_iterations=num_iterations,
    )
    pipeline = MeshToGaussianPipeline(config)
    return pipeline.convert(mesh_path, output_dir)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Mesh Import Test")
    print("=" * 40)

    # Test OBJ parsing (simple cube)
    obj_content = """
# Simple cube
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1

vn  0  0 -1
vn  0  0  1
vn  0 -1  0
vn  0  1  0
vn -1  0  0
vn  1  0  0

f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 4 3 7 8
f 1 4 8 5
f 2 6 7 3
"""

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
        f.write(obj_content)
        obj_path = f.name

    importer = MeshImporter()
    mesh = importer.import_mesh(obj_path)

    print(f"Imported mesh: {mesh.name}")
    print(f"  Primitives: {len(mesh.primitives)}")
    print(f"  Vertices: {mesh.vertex_count}")
    print(f"  Triangles: {mesh.triangle_count}")
    print(f"  Bounds min: {mesh.bounds_min}")
    print(f"  Bounds max: {mesh.bounds_max}")
    print(f"  Center: {mesh.bounds_center}")
    print(f"  Size: {mesh.bounds_size}")

    # Clean up
    Path(obj_path).unlink()

    print("\nTest complete!")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
