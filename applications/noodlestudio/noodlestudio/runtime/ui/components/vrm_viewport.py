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
#   VRMViewport Component
#
#   OpenGL-based VRM avatar renderer for the UI canvas system.
#   Uses the muscle system for rig-agnostic animation.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.vrm_viewport
# PURPOSE:  VRMViewport Component
# LAYER:    Studio / UI Components
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   CameraConfig, VRMViewport, VRMViewportWidget
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import math
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from ..component import UIComponent, register_component

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration for the viewport."""
    distance: float = 2.5
    elevation: float = 10.0   # degrees
    azimuth: float = 180.0    # degrees (180 = facing front)
    target: Tuple[float, float, float] = (0.0, 0.9, 0.0)
    fov: float = 45.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraConfig':
        return cls(
            distance=data.get("distance", 2.5),
            elevation=data.get("elevation", 10.0),
            azimuth=data.get("azimuth", 180.0),
            target=tuple(data.get("target", [0.0, 0.9, 0.0])),
            fov=data.get("fov", 45.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance": self.distance,
            "elevation": self.elevation,
            "azimuth": self.azimuth,
            "target": list(self.target),
            "fov": self.fov,
        }


@register_component
class VRMViewport(UIComponent):
    """
    OpenGL VRM avatar viewport component.

    Renders VRM avatars with skeletal animation support. Uses the muscle
    system for rig-agnostic animation - send muscle values, not bone rotations.

    Properties:
        vrm_path: Path to .vrm file
        camera: Camera configuration
        background: Background color (hex)
        show_skeleton: Display bone overlay
        show_grid: Display ground grid
        interactive: Enable camera controls

    Events:
        onLoad: VRM loaded successfully
        onClick: Click on viewport (with bone raycast)
        onPoseApplied: Pose was applied to avatar
    """

    component_type = "VRMViewport"

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.vrm_path: str = ""
        self.camera = CameraConfig()
        self.background: str = "#1e1e1e"
        self.show_skeleton: bool = False
        self.show_grid: bool = False
        self.interactive: bool = True
        self.transparent: bool = False  # Clear alpha for compositing over UI

        # Default size
        self.geometry.width = 512
        self.geometry.height = 512

    def _serialize_properties(self, data: Dict[str, Any]) -> None:
        """Add VRMViewport-specific properties to serialization."""
        if self.vrm_path:
            data["vrm_path"] = self.vrm_path
        data["camera"] = self.camera.to_dict()
        if self.background != "#1e1e1e":
            data["background"] = self.background
        if self.show_skeleton:
            data["show_skeleton"] = True
        if self.show_grid:
            data["show_grid"] = True
        if not self.interactive:
            data["interactive"] = False
        if self.transparent:
            data["transparent"] = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VRMViewport':
        """Deserialize from dictionary."""
        viewport = cls(name=data.get("name", ""))
        viewport._apply_base_properties(data)

        # Override geometry defaults
        viewport.geometry.width = data.get("width", 512)
        viewport.geometry.height = data.get("height", 512)

        # VRMViewport-specific
        viewport.vrm_path = data.get("vrm_path", "")
        if "camera" in data:
            viewport.camera = CameraConfig.from_dict(data["camera"])
        viewport.background = data.get("background", "#1e1e1e")
        viewport.show_skeleton = data.get("show_skeleton", False)
        viewport.show_grid = data.get("show_grid", False)
        viewport.interactive = data.get("interactive", True)
        viewport.transparent = data.get("transparent", False)

        return viewport


# ============================================================================
# Qt Widget Implementation
# ============================================================================

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout
    from PyQt6.QtCore import Qt, QPoint, QByteArray, QTimer, pyqtSignal
    from PyQt6.QtGui import QMouseEvent, QWheelEvent, QSurfaceFormat, QImage
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    import OpenGL.GL as GL
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


# ============================================================================
# Raw ctypes GL helpers for texture upload
# PyOpenGL 3.1.0 + OpenGL_accelerate 3.1.10 has a broken array-type handler
# on Python 3.14 that prevents glGenTextures from working. These helpers
# bypass the PyOpenGL wrapper and call the C API directly via ctypes.
# ============================================================================
if OPENGL_AVAILABLE:
    import ctypes as _ct
    import ctypes.util as _ct_util

    _gl_lib = _ct.cdll.LoadLibrary(_ct_util.find_library('OpenGL'))

    _raw_glGenTextures = _gl_lib.glGenTextures
    _raw_glGenTextures.argtypes = [_ct.c_int, _ct.POINTER(_ct.c_uint)]
    _raw_glGenTextures.restype = None

    _raw_glBindTexture = _gl_lib.glBindTexture
    _raw_glBindTexture.argtypes = [_ct.c_uint, _ct.c_uint]
    _raw_glBindTexture.restype = None

    _raw_glTexImage2D = _gl_lib.glTexImage2D
    _raw_glTexImage2D.argtypes = [
        _ct.c_uint, _ct.c_int, _ct.c_int,
        _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.c_uint, _ct.c_uint, _ct.c_void_p
    ]
    _raw_glTexImage2D.restype = None

    def _gl_gen_texture() -> int:
        """Generate one GL texture ID via raw ctypes."""
        buf = (_ct.c_uint * 1)()
        _raw_glGenTextures(1, buf)
        return int(buf[0])

    def _gl_bind_texture(tex_id: int):
        """Bind a 2D texture via raw ctypes."""
        _raw_glBindTexture(0x0DE1, _ct.c_uint(tex_id))  # GL_TEXTURE_2D

    def _gl_tex_image_2d(width: int, height: int, data: bytes):
        """Upload RGBA texture data via raw ctypes."""
        _raw_glTexImage2D(
            0x0DE1,  # GL_TEXTURE_2D
            0,       # level
            0x1908,  # GL_RGBA (internalformat)
            width, height, 0,
            0x1908,  # GL_RGBA (format)
            0x1401,  # GL_UNSIGNED_BYTE
            data     # ctypes handles bytes natively
        )


if QT_AVAILABLE and OPENGL_AVAILABLE and NUMPY_AVAILABLE:

    class VRMViewportWidget(QOpenGLWidget):
        """
        Qt OpenGL widget that renders VRM avatars.

        Public API:
            load_vrm(path)           - Load VRM file
            set_muscles(muscles)     - Apply muscle values (dict)
            set_blend_shapes(shapes) - Apply blend shape weights
            set_camera(...)          - Set camera parameters
            get_bone_at(x, y)        - Raycast to find bone at screen pos

        Muscle Animation:
            Instead of setting bone rotations directly, send muscle values.
            The widget uses PoseRetargeter internally to convert to bone rotations.
        """

        # Signals
        vrmLoaded = pyqtSignal(str, int, int)  # path, bone_count, vertex_count
        poseApplied = pyqtSignal(int)  # muscle_count
        clicked = pyqtSignal(int, int, str)  # x, y, bone_name (or empty)

        def __init__(self, component: VRMViewport, parent=None):
            # Set up OpenGL format
            fmt = QSurfaceFormat()
            fmt.setSamples(4)  # MSAA
            fmt.setDepthBufferSize(24)
            fmt.setAlphaBufferSize(8)  # Enable alpha channel for transparency
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            QSurfaceFormat.setDefaultFormat(fmt)

            super().__init__(parent)
            self.component = component
            self.setObjectName(component.name or "vrm_viewport")

            # Enable transparency if requested
            if component.transparent:
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
                self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

            # Avatar data
            self._avatar = None           # Parsed VRM data
            self._muscle_binding = None   # MuscleBinding for retargeting
            self._retargeter = None       # PoseRetargeter instance

            # Display data (GPU buffers)
            self._mesh: Optional[Dict] = None
            self._skeleton: Optional[Dict] = None
            self._bone_matrices: Optional[np.ndarray] = None

            # Current pose (muscle values)
            self._current_muscles: Dict[str, float] = {}
            self._current_blend_shapes: Dict[str, float] = {}
            self._bone_rotations: Dict[str, Tuple[float, float, float]] = {}

            # Camera state (copy from component)
            self._azimuth = component.camera.azimuth
            self._elevation = component.camera.elevation
            self._distance = component.camera.distance
            self._target = list(component.camera.target)
            self._fov = component.camera.fov

            # Mouse interaction
            self._last_mouse_pos = QPoint()
            self._is_orbiting = False
            self._is_panning = False

            # Shaders (initialized in initializeGL)
            self._shader_mesh = None
            self._shader_line = None

            # Per-material rendering
            self._material_groups = []   # (material_index, byte_offset, index_count)
            self._gl_textures = {}       # material_index -> GL texture ID
            self._material_colors = {}   # material_index -> (r, g, b)
            self._material_mtoon = {}    # material_index -> MToonMaterial

            # Cached uniform locations (set after shader compilation)
            self._mesh_uniform_locs = {}

            # Idle animation
            self._idle_phase = 0.0       # Animation phase in seconds
            self._idle_timer = None      # QTimer for animation

            # Grid buffers
            self._grid_vao = 0
            self._grid_vbo = 0
            self._grid_count = 0

            # Skeleton buffers
            self._skeleton_vao = 0
            self._skeleton_vbo = 0

            self.setMinimumSize(200, 200)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # =====================================================================
        # OpenGL Setup
        # =====================================================================

        def initializeGL(self):
            """Initialize OpenGL resources."""
            if self.component.transparent:
                # Transparent background - alpha = 0 where no geometry
                GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            else:
                bg = self._parse_color(self.component.background)
                GL.glClearColor(*bg, 1.0)

            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glEnable(GL.GL_MULTISAMPLE)

            self._create_shaders()
            self._create_grid()

            # Load VRM if path specified
            if self.component.vrm_path:
                self._load_vrm_deferred()

        def _load_vrm_deferred(self):
            """Load VRM after OpenGL is initialized."""
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.load_vrm(self.component.vrm_path))

        def _create_shaders(self):
            """Create shader programs."""
            # Mesh shader (simplified - no skinning yet for phase 1)
            vertex_mesh = """
            #version 330 core
            layout(location = 0) in vec3 aPos;
            layout(location = 1) in vec3 aNormal;
            layout(location = 2) in vec2 aUV;

            uniform mat4 uModel;
            uniform mat4 uView;
            uniform mat4 uProjection;

            out vec3 vNormal;
            out vec3 vWorldPos;
            out vec2 vUV;

            void main() {
                vec4 worldPos = uModel * vec4(aPos, 1.0);
                vWorldPos = worldPos.xyz;
                vNormal = mat3(transpose(inverse(uModel))) * aNormal;
                vUV = aUV;
                gl_Position = uProjection * uView * worldPos;
            }
            """

            fragment_mesh = """
            #version 330 core
            in vec3 vNormal;
            in vec3 vWorldPos;
            in vec2 vUV;

            uniform vec3 uLightDir;
            uniform vec3 uColor;
            uniform sampler2D uDiffuseTex;
            uniform int uHasTexture;

            // MToon cel-shading uniforms
            uniform vec3 uShadeColor;
            uniform float uShadingToony;
            uniform float uShadingShift;
            uniform vec3 uRimColor;
            uniform float uRimPower;
            uniform vec3 uCameraPos;

            out vec4 FragColor;

            void main() {
                vec3 normal = normalize(vNormal);

                vec3 baseColor;
                float alpha = 1.0;
                if (uHasTexture == 1) {
                    vec4 texColor = texture(uDiffuseTex, vUV);
                    baseColor = texColor.rgb * uColor;
                    alpha = texColor.a;
                } else {
                    baseColor = uColor;
                }

                // Alpha cutout for hair/accessories
                if (alpha < 0.5) discard;

                // MToon cel shading: half-Lambert with stepped boundary
                float halfLambert = dot(normal, uLightDir) * 0.5 + 0.5;
                float shifted = halfLambert + uShadingShift;
                float toony = smoothstep(
                    0.5 - uShadingToony * 0.5,
                    0.5 + uShadingToony * 0.5,
                    shifted
                );

                // Blend between shade color and lit color
                vec3 shadedColor = baseColor * uShadeColor;
                vec3 color = mix(shadedColor, baseColor, toony);

                // Ambient fill to prevent pure black shadows
                color = max(color, baseColor * 0.15);

                // Rim lighting (view-dependent edge glow)
                vec3 viewDir = normalize(uCameraPos - vWorldPos);
                float rim = 1.0 - max(dot(viewDir, normal), 0.0);
                rim = pow(rim, uRimPower);
                color += uRimColor * rim;

                FragColor = vec4(color, alpha);
            }
            """

            self._shader_mesh = self._compile_shader(vertex_mesh, fragment_mesh)

            # Line shader (skeleton, grid)
            vertex_line = """
            #version 330 core
            layout(location = 0) in vec3 aPos;
            layout(location = 1) in vec3 aColor;

            uniform mat4 uView;
            uniform mat4 uProjection;

            out vec3 vColor;

            void main() {
                vColor = aColor;
                gl_Position = uProjection * uView * vec4(aPos, 1.0);
            }
            """

            fragment_line = """
            #version 330 core
            in vec3 vColor;
            out vec4 FragColor;

            void main() {
                FragColor = vec4(vColor, 1.0);
            }
            """

            self._shader_line = self._compile_shader(vertex_line, fragment_line)

        def _compile_shader(self, vertex_src: str, fragment_src: str) -> int:
            """Compile and link shader program."""
            program = GL.glCreateProgram()

            vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
            GL.glShaderSource(vs, vertex_src)
            GL.glCompileShader(vs)
            if not GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(vs)
                logger.error(f"Vertex shader error: {error}")

            fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
            GL.glShaderSource(fs, fragment_src)
            GL.glCompileShader(fs)
            if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
                error = GL.glGetShaderInfoLog(fs)
                logger.error(f"Fragment shader error: {error}")

            GL.glAttachShader(program, vs)
            GL.glAttachShader(program, fs)
            GL.glLinkProgram(program)

            if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
                error = GL.glGetProgramInfoLog(program)
                logger.error(f"Shader link error: {error}")

            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)

            return program

        def _create_grid(self):
            """Create ground grid."""
            lines = []
            grid_size = 5
            grid_step = 0.5

            for i in range(-grid_size, grid_size + 1):
                x = i * grid_step
                color = [0.25, 0.25, 0.25]
                lines.extend([x, 0, -grid_size * grid_step, *color])
                lines.extend([x, 0, grid_size * grid_step, *color])
                lines.extend([-grid_size * grid_step, 0, x, *color])
                lines.extend([grid_size * grid_step, 0, x, *color])

            data = np.array(lines, dtype=np.float32)
            self._grid_count = len(lines) // 6

            self._grid_vao = GL.glGenVertexArrays(1)
            self._grid_vbo = GL.glGenBuffers(1)

            GL.glBindVertexArray(self._grid_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)

            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24,
                                     GL.ctypes.c_void_p(12))
            GL.glEnableVertexAttribArray(1)

            GL.glBindVertexArray(0)

        # =====================================================================
        # Public API: VRM Loading
        # =====================================================================

        def load_vrm(self, path: str):
            """
            Load a VRM file.

            Args:
                path: Path to .vrm file (absolute or relative to project)
            """
            try:
                from noodlestudio.core.semantic_world.vrm_parser import parse_vrm

                logger.info(f"Loading VRM: {path}")

                # Parse VRM
                self._avatar = parse_vrm(path)

                if not self._avatar:
                    logger.error(f"Failed to parse VRM: {path}")
                    return

                # Create GPU buffers
                self._create_mesh_buffers()
                self._load_textures()
                self._cache_mesh_uniforms()
                self._create_skeleton_data()

                # Center camera
                self._center_camera()

                # Start idle animation
                if not self._idle_timer:
                    self._idle_timer = QTimer(self)
                    self._idle_timer.timeout.connect(self._tick_idle)
                    self._idle_timer.start(16)  # ~60fps

                # Emit signal
                bone_count = len(self._avatar.skeleton.bones) if self._avatar.skeleton else 0
                vertex_count = sum(
                    len(m.vertices) for m in self._avatar.meshes
                ) if self._avatar.meshes else 0

                self.vrmLoaded.emit(path, bone_count, vertex_count)
                logger.info(f"VRM loaded: {path} ({bone_count} bones, {vertex_count:,} verts)")

                self.update()

            except Exception as e:
                logger.error(f"Failed to load VRM: {e}")
                import traceback
                traceback.print_exc()

        def _create_mesh_buffers(self):
            """Create OpenGL buffers from avatar mesh data.

            Builds a single combined VAO with all meshes, but tracks
            per-material draw groups so each material can bind its own
            texture and diffuse color during rendering.
            """
            if not self._avatar or not self._avatar.meshes:
                logger.warning("No meshes in avatar")
                return

            # Sort meshes by material_index so same-material primitives
            # are contiguous in the index buffer
            sorted_meshes = sorted(
                self._avatar.meshes,
                key=lambda m: m.material_index if m.material_index is not None else -1
            )

            # Combine all meshes, tracking per-material groups
            all_vertices = []
            all_normals = []
            all_uvs = []
            all_indices = []
            self._material_groups = []  # (material_index, byte_offset, index_count)

            vertex_offset = 0
            index_offset = 0  # in number of indices (not bytes)

            for mesh in sorted_meshes:
                if mesh.vertices is None or len(mesh.vertices) == 0:
                    continue

                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                all_vertices.append(vertices)

                if mesh.normals is not None:
                    all_normals.append(np.asarray(mesh.normals, dtype=np.float32))
                else:
                    all_normals.append(np.zeros_like(vertices))

                if mesh.uvs is not None:
                    all_uvs.append(np.asarray(mesh.uvs, dtype=np.float32))
                else:
                    all_uvs.append(np.zeros((len(vertices), 2), dtype=np.float32))

                if mesh.indices is not None:
                    indices = np.asarray(mesh.indices, dtype=np.uint32).flatten() + vertex_offset
                    all_indices.append(indices)
                else:
                    indices = np.arange(len(vertices), dtype=np.uint32) + vertex_offset
                    all_indices.append(indices)

                mat_idx = mesh.material_index if mesh.material_index is not None else -1
                # byte_offset = index position * 4 bytes per uint32
                byte_offset = index_offset * 4
                index_count = len(indices)

                # Try to merge with previous group if same material
                if (self._material_groups
                        and self._material_groups[-1][0] == mat_idx):
                    prev_mat, prev_offset, prev_count = self._material_groups[-1]
                    self._material_groups[-1] = (prev_mat, prev_offset, prev_count + index_count)
                else:
                    self._material_groups.append((mat_idx, byte_offset, index_count))

                vertex_offset += len(vertices)
                index_offset += index_count

            if not all_vertices:
                logger.warning("No vertex data found")
                return

            vertices = np.vstack(all_vertices)
            normals = np.vstack(all_normals)
            uvs = np.vstack(all_uvs)
            indices = np.concatenate(all_indices)

            logger.info(
                f"Combined mesh: {len(vertices)} verts, {len(indices)} indices, "
                f"{len(self._material_groups)} material groups"
            )

            # Store for rendering
            self._mesh = {
                'vertices': vertices,
                'normals': normals,
                'uvs': uvs,
                'indices': indices,
                'vao': 0,
            }

            # Create VAO
            self._mesh['vao'] = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(self._mesh['vao'])

            # Position (location 0)
            vbo_pos = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_pos)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(0)

            # Normal (location 1)
            vbo_norm = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_norm)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, normals.nbytes, normals, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(1)

            # UV (location 2)
            vbo_uv = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_uv)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(2)

            # Index buffer
            ebo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

            GL.glBindVertexArray(0)

        def _create_skeleton_data(self):
            """Extract skeleton data for visualization."""
            if not self._avatar or not self._avatar.skeleton:
                return

            self._skeleton = {
                'bones': self._avatar.skeleton.bones,
                'humanoid_map': dict(self._avatar.skeleton.humanoid_map)
                    if hasattr(self._avatar.skeleton, 'humanoid_map') else {},
            }

        def _load_textures(self):
            """Load textures from avatar materials and upload to GL.

            For each material that references a diffuse_texture, decodes the
            image bytes via QImage and uploads to an OpenGL texture object.
            Also extracts per-material diffuse colors for the uColor uniform.
            """
            if not self._avatar:
                return

            self._gl_textures = {}
            self._material_colors = {}
            self._material_mtoon = {}

            for mat_idx, material in enumerate(self._avatar.materials):
                # Store MToon material reference for cel-shading uniforms
                self._material_mtoon[mat_idx] = material

                # Extract diffuse color (RGB from RGBA)
                dc = material.diffuse_color
                self._material_colors[mat_idx] = (
                    float(dc[0]), float(dc[1]), float(dc[2])
                )

                # Load diffuse texture if available
                tex_idx = material.diffuse_texture
                if tex_idx is None:
                    continue
                if tex_idx >= len(self._avatar.textures):
                    logger.warning(
                        f"Material '{material.name}' references texture {tex_idx} "
                        f"but only {len(self._avatar.textures)} textures available"
                    )
                    continue

                raw_bytes = self._avatar.textures[tex_idx]
                if not raw_bytes:
                    continue

                try:
                    qimg = QImage.fromData(QByteArray(raw_bytes))
                    if qimg.isNull():
                        logger.warning(
                            f"Failed to decode texture {tex_idx} for material "
                            f"'{material.name}'"
                        )
                        continue

                    # Convert to RGBA8888 for GL upload
                    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
                    # glTF/VRM UVs use top-left origin (V=0 at top).
                    # OpenGL glTexImage2D uploads the first row of pixels
                    # to the bottom of the texture (where V=0 maps).
                    # So no vertical flip is needed -- the conventions
                    # align when the image is uploaded as-is.

                    w, h = qimg.width(), qimg.height()
                    ptr = qimg.bits()
                    ptr.setsize(w * h * 4)
                    raw_bytes_data = bytes(ptr)

                    # Use raw ctypes GL calls for texture upload.
                    # PyOpenGL 3.1.0 + OpenGL_accelerate 3.1.10 has a
                    # broken array-type handler on Python 3.14 that
                    # causes CArgObject errors in glGenTextures.
                    tex_id = _gl_gen_texture()
                    _gl_bind_texture(tex_id)
                    _gl_tex_image_2d(w, h, raw_bytes_data)
                    GL.glTexParameteri(
                        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
                    )
                    GL.glTexParameteri(
                        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
                    )
                    GL.glTexParameteri(
                        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT
                    )
                    GL.glTexParameteri(
                        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT
                    )
                    _gl_bind_texture(0)

                    self._gl_textures[mat_idx] = tex_id
                    logger.info(
                        f"Loaded texture {tex_idx} ({w}x{h}) for material "
                        f"'{material.name}'"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to load texture {tex_idx} for material "
                        f"'{material.name}': {e}"
                    )

            logger.info(
                f"Loaded {len(self._gl_textures)} textures, "
                f"{len(self._material_colors)} material colors"
            )

        def _cache_mesh_uniforms(self):
            """Cache uniform locations for the mesh shader to avoid
            per-frame glGetUniformLocation calls."""
            if not self._shader_mesh:
                return
            prog = self._shader_mesh
            self._mesh_uniform_locs = {
                'uModel': GL.glGetUniformLocation(prog, "uModel"),
                'uView': GL.glGetUniformLocation(prog, "uView"),
                'uProjection': GL.glGetUniformLocation(prog, "uProjection"),
                'uLightDir': GL.glGetUniformLocation(prog, "uLightDir"),
                'uColor': GL.glGetUniformLocation(prog, "uColor"),
                'uDiffuseTex': GL.glGetUniformLocation(prog, "uDiffuseTex"),
                'uHasTexture': GL.glGetUniformLocation(prog, "uHasTexture"),
                # MToon cel-shading uniforms
                'uShadeColor': GL.glGetUniformLocation(prog, "uShadeColor"),
                'uShadingToony': GL.glGetUniformLocation(prog, "uShadingToony"),
                'uShadingShift': GL.glGetUniformLocation(prog, "uShadingShift"),
                'uRimColor': GL.glGetUniformLocation(prog, "uRimColor"),
                'uRimPower': GL.glGetUniformLocation(prog, "uRimPower"),
                'uCameraPos': GL.glGetUniformLocation(prog, "uCameraPos"),
            }

        def _center_camera(self):
            """Center camera on loaded model."""
            if self._mesh and len(self._mesh['vertices']) > 0:
                verts = self._mesh['vertices']
                center = verts.mean(axis=0)
                bounds_min = verts.min(axis=0)
                bounds_max = verts.max(axis=0)
                size = (bounds_max - bounds_min).max()

                # Center on upper body (typical VRM is centered at feet)
                self._target = [float(center[0]), float(center[1]), float(center[2])]
                self._distance = max(1.5, size * 1.2)
                self._elevation = 10

        # =====================================================================
        # Public API: Muscle Animation
        # =====================================================================

        def set_muscles(self, muscles: Dict[str, float]):
            """
            Apply muscle values to the avatar.

            This is the PRIMARY animation interface. Send normalized muscle
            values (-1 to 1), and the widget handles retargeting to bone
            rotations internally.

            Args:
                muscles: Dict mapping muscle name to value, e.g.:
                    {'Head.TurnLeftRight': 0.3, 'RightArm.DownUp': 0.5}
            """
            self._current_muscles = muscles.copy()
            self._apply_pose()
            self.poseApplied.emit(len(muscles))
            self.update()

        def set_blend_shapes(self, shapes: Dict[str, float]):
            """
            Apply blend shape weights.

            Args:
                shapes: Dict mapping shape name to weight (0-1), e.g.:
                    {'happy': 0.6, 'blink_left': 0.0}
            """
            self._current_blend_shapes = shapes.copy()
            # TODO: Apply to mesh morph targets
            self.update()

        def _apply_pose(self):
            """Apply current muscle values to skeleton."""
            if not self._current_muscles:
                return

            # For Phase 1, we'll skip the retargeter and just store muscles
            # Phase 2 will implement proper bone matrix computation
            # TODO: Implement muscle → bone rotation → bone matrices
            pass

        # =====================================================================
        # Public API: Camera
        # =====================================================================

        def set_camera(
            self,
            distance: Optional[float] = None,
            elevation: Optional[float] = None,
            azimuth: Optional[float] = None,
            target: Optional[Tuple[float, float, float]] = None,
            fov: Optional[float] = None
        ):
            """Set camera parameters."""
            if distance is not None:
                self._distance = max(0.5, distance)
            if elevation is not None:
                self._elevation = max(-89, min(89, elevation))
            if azimuth is not None:
                self._azimuth = azimuth
            if target is not None:
                self._target = list(target)
            if fov is not None:
                self._fov = fov
            self.update()

        def frame_all(self):
            """Frame all content in view."""
            self._center_camera()
            self.update()

        # =====================================================================
        # Rendering
        # =====================================================================

        def resizeGL(self, w: int, h: int):
            GL.glViewport(0, 0, w, h)

        def paintGL(self):
            """Render the scene."""
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            aspect = self.width() / max(1, self.height())
            view = self._view_matrix()
            proj = self._projection_matrix(aspect)

            # Draw grid (skip if transparent - don't want floating grid)
            if self.component.show_grid and not self.component.transparent:
                self._draw_grid(view, proj)

            # Draw mesh
            if self._mesh and self._mesh.get('vao'):
                self._draw_mesh(view, proj)

            # Draw skeleton overlay
            if self.component.show_skeleton and self._skeleton:
                self._draw_skeleton(view, proj)

        def _view_matrix(self) -> np.ndarray:
            """Compute view matrix from orbit camera."""
            az_rad = math.radians(self._azimuth)
            el_rad = math.radians(self._elevation)

            x = self._distance * math.cos(el_rad) * math.sin(az_rad)
            y = self._distance * math.sin(el_rad)
            z = self._distance * math.cos(el_rad) * math.cos(az_rad)

            pos = np.array([
                self._target[0] + x,
                self._target[1] + y,
                self._target[2] + z
            ], dtype=np.float32)

            target = np.array(self._target, dtype=np.float32)
            forward = target - pos
            forward = forward / np.linalg.norm(forward)

            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            view = np.eye(4, dtype=np.float32)
            view[0, :3] = right
            view[1, :3] = up
            view[2, :3] = -forward
            view[0, 3] = -np.dot(right, pos)
            view[1, 3] = -np.dot(up, pos)
            view[2, 3] = np.dot(forward, pos)

            return view

        def _projection_matrix(self, aspect: float) -> np.ndarray:
            """Compute perspective projection matrix."""
            fov_rad = math.radians(self._fov)
            f = 1.0 / math.tan(fov_rad / 2.0)
            near, far = 0.01, 100.0

            proj = np.zeros((4, 4), dtype=np.float32)
            proj[0, 0] = f / aspect
            proj[1, 1] = f
            proj[2, 2] = (far + near) / (near - far)
            proj[2, 3] = (2 * far * near) / (near - far)
            proj[3, 2] = -1.0

            return proj

        def _draw_grid(self, view: np.ndarray, proj: np.ndarray):
            """Draw ground grid."""
            GL.glUseProgram(self._shader_line)
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._shader_line, "uView"),
                1, GL.GL_TRUE, view
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._shader_line, "uProjection"),
                1, GL.GL_TRUE, proj
            )

            GL.glBindVertexArray(self._grid_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self._grid_count)
            GL.glBindVertexArray(0)

        def _draw_mesh(self, view: np.ndarray, proj: np.ndarray):
            """Draw avatar mesh with per-material MToon cel-shading."""
            GL.glUseProgram(self._shader_mesh)

            locs = self._mesh_uniform_locs

            # Model matrix with idle animation
            model = self._build_model_matrix()
            GL.glUniformMatrix4fv(locs.get('uModel', -1), 1, GL.GL_TRUE, model)
            GL.glUniformMatrix4fv(locs.get('uView', -1), 1, GL.GL_TRUE, view)
            GL.glUniformMatrix4fv(
                locs.get('uProjection', -1), 1, GL.GL_TRUE, proj
            )

            # Light direction (shared across all materials)
            light_dir = np.array([0.5, 0.7, 0.5], dtype=np.float32)
            light_dir = light_dir / np.linalg.norm(light_dir)
            GL.glUniform3fv(locs.get('uLightDir', -1), 1, light_dir)

            # Camera position for rim lighting (same computation as _view_matrix)
            az_rad = math.radians(self._azimuth)
            el_rad = math.radians(self._elevation)
            cam_pos = np.array([
                self._target[0] + self._distance * math.cos(el_rad) * math.sin(az_rad),
                self._target[1] + self._distance * math.sin(el_rad),
                self._target[2] + self._distance * math.cos(el_rad) * math.cos(az_rad),
            ], dtype=np.float32)
            GL.glUniform3fv(locs.get('uCameraPos', -1), 1, cam_pos)

            loc_color = locs.get('uColor', -1)
            loc_tex = locs.get('uDiffuseTex', -1)
            loc_has_tex = locs.get('uHasTexture', -1)
            loc_shade_color = locs.get('uShadeColor', -1)
            loc_shading_toony = locs.get('uShadingToony', -1)
            loc_shading_shift = locs.get('uShadingShift', -1)
            loc_rim_color = locs.get('uRimColor', -1)
            loc_rim_power = locs.get('uRimPower', -1)

            GL.glBindVertexArray(self._mesh['vao'])

            for mat_idx, byte_offset, count in self._material_groups:
                # Bind texture if available for this material
                if mat_idx in self._gl_textures:
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    _gl_bind_texture(self._gl_textures[mat_idx])
                    GL.glUniform1i(loc_tex, 0)
                    GL.glUniform1i(loc_has_tex, 1)
                else:
                    GL.glUniform1i(loc_has_tex, 0)

                # Set material diffuse color
                color = self._material_colors.get(mat_idx, (0.85, 0.80, 0.75))
                GL.glUniform3f(loc_color, *color)

                # Set MToon cel-shading uniforms
                mtoon = self._material_mtoon.get(mat_idx)
                if mtoon:
                    GL.glUniform3f(loc_shade_color, *mtoon.shade_color[:3])
                    GL.glUniform1f(loc_shading_toony, mtoon.shading_toony)
                    GL.glUniform1f(loc_shading_shift, mtoon.shading_shift)
                    GL.glUniform3f(loc_rim_color, *mtoon.rim_color)
                    GL.glUniform1f(loc_rim_power, max(0.1, mtoon.rim_power))
                else:
                    GL.glUniform3f(loc_shade_color, 0.65, 0.65, 0.7)
                    GL.glUniform1f(loc_shading_toony, 0.9)
                    GL.glUniform1f(loc_shading_shift, 0.0)
                    GL.glUniform3f(loc_rim_color, 0.0, 0.0, 0.0)
                    GL.glUniform1f(loc_rim_power, 1.0)

                # Draw this material group
                GL.glDrawElements(
                    GL.GL_TRIANGLES, count, GL.GL_UNSIGNED_INT,
                    GL.ctypes.c_void_p(byte_offset)
                )

            _gl_bind_texture(0)
            GL.glBindVertexArray(0)

        def _draw_skeleton(self, view: np.ndarray, proj: np.ndarray):
            """Draw skeleton overlay."""
            if not self._skeleton or 'bones' not in self._skeleton:
                return

            bones = self._skeleton['bones']
            if not bones:
                return

            # Build line data from bone hierarchy
            lines = []
            for bone in bones:
                if bone.parent_index >= 0 and bone.parent_index < len(bones):
                    parent = bones[bone.parent_index]

                    # Get positions (using rest pose)
                    child_pos = self._get_bone_world_position(bone, bones)
                    parent_pos = self._get_bone_world_position(parent, bones)

                    if child_pos is not None and parent_pos is not None:
                        # Cyan color for humanoid bones
                        color = [0.0, 0.9, 0.9]
                        lines.extend([*parent_pos, *color])
                        lines.extend([*child_pos, *color])

            if not lines:
                return

            data = np.array(lines, dtype=np.float32)

            # Create/update skeleton buffer
            if self._skeleton_vao == 0:
                self._skeleton_vao = GL.glGenVertexArrays(1)
                self._skeleton_vbo = GL.glGenBuffers(1)

            GL.glBindVertexArray(self._skeleton_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._skeleton_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)

            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24,
                                     GL.ctypes.c_void_p(12))
            GL.glEnableVertexAttribArray(1)

            # Draw
            GL.glUseProgram(self._shader_line)
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._shader_line, "uView"),
                1, GL.GL_TRUE, view
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._shader_line, "uProjection"),
                1, GL.GL_TRUE, proj
            )

            GL.glDrawArrays(GL.GL_LINES, 0, len(lines) // 6)
            GL.glBindVertexArray(0)

        def _get_bone_world_position(self, bone, all_bones) -> Optional[List[float]]:
            """Get world position of a bone (rest pose)."""
            try:
                # Accumulate transforms up the hierarchy
                pos = np.array([
                    bone.transform.position.x,
                    bone.transform.position.y,
                    bone.transform.position.z
                ], dtype=np.float32)

                current = bone
                while current.parent_index >= 0 and current.parent_index < len(all_bones):
                    parent = all_bones[current.parent_index]
                    parent_pos = np.array([
                        parent.transform.position.x,
                        parent.transform.position.y,
                        parent.transform.position.z
                    ], dtype=np.float32)
                    pos = pos + parent_pos
                    current = parent

                return pos.tolist()
            except Exception:
                return None

        # =====================================================================
        # Idle Animation
        # =====================================================================

        def _tick_idle(self):
            """Advance idle animation phase and request repaint."""
            self._idle_phase += 0.016  # ~16ms per tick
            self.update()

        def _build_model_matrix(self) -> np.ndarray:
            """Build model matrix with idle animation transforms.

            Applies a subtle Y-axis bob (breathing rhythm) and Y-axis
            scale pulse (chest expansion) driven by sine waves.
            """
            t = self._idle_phase
            model = np.eye(4, dtype=np.float32)

            # Y bob (period ~4s, amplitude ~0.01 units)
            bob_y = 0.01 * math.sin(t * 2.0 * math.pi / 4.0)
            model[1, 3] = bob_y

            # Chest expansion via Y scale (period ~3.5s, amplitude ~0.02)
            breath_scale = 1.0 + 0.02 * math.sin(t * 2.0 * math.pi / 3.5)
            model[1, 1] = breath_scale

            return model

        # =====================================================================
        # Mouse Interaction
        # =====================================================================

        def mousePressEvent(self, event: QMouseEvent):
            if not self.component.interactive:
                return

            self._last_mouse_pos = event.pos()
            if event.button() == Qt.MouseButton.LeftButton:
                self._is_orbiting = True
            elif event.button() == Qt.MouseButton.RightButton:
                self._is_panning = True

        def mouseReleaseEvent(self, event: QMouseEvent):
            self._is_orbiting = False
            self._is_panning = False

        def mouseMoveEvent(self, event: QMouseEvent):
            if not self.component.interactive:
                return

            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()

            if self._is_orbiting:
                self._azimuth += delta.x() * 0.5
                self._elevation = max(-89, min(89, self._elevation - delta.y() * 0.5))
                self.update()

            elif self._is_panning:
                pan_speed = 0.003 * self._distance
                self._target[0] -= delta.x() * pan_speed
                self._target[1] += delta.y() * pan_speed
                self.update()

        def wheelEvent(self, event: QWheelEvent):
            if not self.component.interactive:
                return

            delta = event.angleDelta().y()
            factor = 0.9 if delta > 0 else 1.1
            self._distance = max(0.5, min(20.0, self._distance * factor))
            self.update()

        def keyPressEvent(self, event):
            """Handle key presses."""
            if event.key() == Qt.Key.Key_F:
                self.frame_all()
            elif event.key() == Qt.Key.Key_G:
                self.component.show_grid = not self.component.show_grid
                self.update()
            elif event.key() == Qt.Key.Key_S:
                self.component.show_skeleton = not self.component.show_skeleton
                self.update()

        # =====================================================================
        # Utilities
        # =====================================================================

        def _parse_color(self, hex_color: str) -> Tuple[float, float, float]:
            """Parse hex color to RGB tuple."""
            if hex_color.startswith('#') and len(hex_color) >= 7:
                try:
                    r = int(hex_color[1:3], 16) / 255.0
                    g = int(hex_color[3:5], 16) / 255.0
                    b = int(hex_color[5:7], 16) / 255.0
                    return (r, g, b)
                except (ValueError, IndexError):
                    pass
            return (0.12, 0.12, 0.12)


# Fallback widget when OpenGL not available
if not (QT_AVAILABLE and OPENGL_AVAILABLE and NUMPY_AVAILABLE):
    class VRMViewportWidget(QWidget if QT_AVAILABLE else object):
        """Fallback widget when OpenGL is not available."""

        def __init__(self, component, parent=None):
            if QT_AVAILABLE:
                super().__init__(parent)
                from PyQt6.QtWidgets import QLabel, QVBoxLayout
                layout = QVBoxLayout(self)
                label = QLabel("OpenGL not available")
                label.setStyleSheet("color: #888;")
                layout.addWidget(label)


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
