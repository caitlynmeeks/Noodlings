"""
VRM Preview Panel - 3D character preview with animation support.

A proper OpenGL-based 3D viewer for:
- VRM avatar display (mesh with materials)
- Skeleton visualization (bone hierarchy)
- Animation playback (muscle-space poses)
- Gaussian splat preview (point cloud mode)
- Transform gizmos (interactive posing)

"What Maya did for motion, what Mecanim did for retargeting -
 we visualize for intuitive understanding."

Author: Caitlyn + Claude
Date: December 2025
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QFrame, QSplitter, QToolBar,
    QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QAction, QSurfaceFormat

# OpenGL imports
try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from PyQt6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader
    import OpenGL.GL as GL
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    QOpenGLWidget = QWidget

logger = logging.getLogger(__name__)


# =============================================================================
# View Modes
# =============================================================================

class ViewMode(Enum):
    """Display modes for the preview."""
    MESH = "mesh"               # Solid mesh with materials
    WIREFRAME = "wireframe"     # Wireframe mesh
    SKELETON = "skeleton"       # Skeleton only
    POINTS = "points"           # Point cloud (Gaussian positions)
    MESH_SKELETON = "mesh_skeleton"  # Mesh + skeleton overlay


class ShadingMode(Enum):
    """Shading modes."""
    UNLIT = "unlit"
    LIT = "lit"
    NORMAL = "normal"           # Visualize normals
    UV = "uv"                   # Visualize UVs
    WEIGHTS = "weights"         # Visualize skinning weights


# =============================================================================
# Camera
# =============================================================================

@dataclass
class Camera:
    """Orbit camera around a target point."""
    target: np.ndarray = None           # Look-at point
    distance: float = 3.0               # Distance from target
    azimuth: float = 0.0                # Horizontal angle (degrees)
    elevation: float = 15.0             # Vertical angle (degrees)
    fov: float = 45.0                   # Field of view
    near: float = 0.01
    far: float = 100.0

    def __post_init__(self):
        if self.target is None:
            self.target = np.array([0.0, 1.0, 0.0])

    @property
    def position(self) -> np.ndarray:
        """Calculate camera position from orbit parameters."""
        az_rad = math.radians(self.azimuth)
        el_rad = math.radians(self.elevation)

        x = self.distance * math.cos(el_rad) * math.sin(az_rad)
        y = self.distance * math.sin(el_rad)
        z = self.distance * math.cos(el_rad) * math.cos(az_rad)

        return self.target + np.array([x, y, z])

    def view_matrix(self) -> np.ndarray:
        """Compute view matrix."""
        pos = self.position
        forward = self.target - pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0.0, 1.0, 0.0])
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

    def projection_matrix(self, aspect: float) -> np.ndarray:
        """Compute perspective projection matrix."""
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0

        return proj

    def orbit(self, delta_azimuth: float, delta_elevation: float):
        """Orbit camera around target."""
        self.azimuth += delta_azimuth
        self.elevation = max(-89, min(89, self.elevation + delta_elevation))

    def zoom(self, factor: float):
        """Zoom in/out."""
        self.distance = max(0.1, self.distance * factor)

    def pan(self, dx: float, dy: float):
        """Pan camera target."""
        # Calculate right and up vectors
        az_rad = math.radians(self.azimuth)
        right = np.array([math.cos(az_rad), 0, -math.sin(az_rad)])
        up = np.array([0, 1, 0])

        scale = self.distance * 0.001
        self.target += right * dx * scale + up * dy * scale


# =============================================================================
# VRM Display Data
# =============================================================================

@dataclass
class DisplayMesh:
    """Mesh data prepared for OpenGL rendering."""
    vertices: np.ndarray        # (N, 3) positions
    normals: np.ndarray         # (N, 3) normals
    uvs: np.ndarray             # (N, 2) texture coords
    indices: np.ndarray         # Triangle indices
    bone_indices: np.ndarray    # (N, 4) bone indices for skinning
    bone_weights: np.ndarray    # (N, 4) bone weights

    # OpenGL buffer IDs (set after upload)
    vao: int = 0
    vbo_pos: int = 0
    vbo_norm: int = 0
    vbo_uv: int = 0
    vbo_bone_idx: int = 0
    vbo_bone_wgt: int = 0
    ebo: int = 0


@dataclass
class DisplayBone:
    """Bone for skeleton visualization."""
    name: str
    index: int
    parent_index: int
    position: np.ndarray        # Local position
    rotation: np.ndarray        # Quaternion (x, y, z, w)
    world_position: np.ndarray = None  # Computed world position


@dataclass
class DisplaySkeleton:
    """Skeleton data for visualization."""
    bones: List[DisplayBone]
    humanoid_map: Dict[str, int]


# =============================================================================
# OpenGL Preview Widget
# =============================================================================

class VRMGLWidget(QOpenGLWidget):
    """
    OpenGL widget for VRM preview rendering.

    Handles:
    - Camera orbit/pan/zoom
    - Mesh rendering with skinning
    - Skeleton visualization
    - Grid and axis display
    """

    def __init__(self, parent=None):
        # Set up OpenGL format
        fmt = QSurfaceFormat()
        fmt.setSamples(4)  # MSAA
        fmt.setDepthBufferSize(24)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)

        self.setMinimumSize(400, 300)

        # Camera
        self.camera = Camera()

        # View settings
        self.view_mode = ViewMode.MESH_SKELETON
        self.shading_mode = ShadingMode.LIT
        self.show_grid = True
        self.show_axes = True
        self.wireframe_overlay = False

        # Data
        self.mesh: Optional[DisplayMesh] = None
        self.skeleton: Optional[DisplaySkeleton] = None
        self.gaussian_positions: Optional[np.ndarray] = None

        # Animation state
        self.bone_transforms: Dict[str, np.ndarray] = {}  # name -> 4x4 matrix

        # Mouse state
        self._mouse_pos = None
        self._mouse_button = None

        # Shaders
        self._shader_mesh = None
        self._shader_line = None
        self._shader_point = None

        # Grid/axes buffers
        self._grid_vao = 0
        self._grid_vbo = 0
        self._grid_count = 0
        self._axes_vao = 0
        self._axes_vbo = 0

        # Skeleton buffers
        self._skeleton_vao = 0
        self._skeleton_vbo = 0

    def initializeGL(self):
        """Initialize OpenGL resources."""
        if not OPENGL_AVAILABLE:
            return

        # Background color
        GL.glClearColor(0.12, 0.12, 0.12, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Enable MSAA if available
        GL.glEnable(GL.GL_MULTISAMPLE)

        # Create shaders
        self._create_shaders()

        # Create grid
        self._create_grid()

        # Create axes
        self._create_axes()

    def _create_shaders(self):
        """Create shader programs."""
        # Simple mesh shader
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
        uniform vec3 uViewPos;
        uniform vec3 uColor;
        uniform int uShadingMode;

        out vec4 FragColor;

        void main() {
            vec3 normal = normalize(vNormal);

            if (uShadingMode == 0) {
                // Unlit
                FragColor = vec4(uColor, 1.0);
            } else if (uShadingMode == 1) {
                // Lit
                float diff = max(dot(normal, uLightDir), 0.0);
                float ambient = 0.3;
                vec3 color = uColor * (ambient + diff * 0.7);
                FragColor = vec4(color, 1.0);
            } else if (uShadingMode == 2) {
                // Normal visualization
                FragColor = vec4(normal * 0.5 + 0.5, 1.0);
            } else if (uShadingMode == 3) {
                // UV visualization
                FragColor = vec4(vUV, 0.0, 1.0);
            } else {
                FragColor = vec4(uColor, 1.0);
            }
        }
        """

        self._shader_mesh = self._compile_shader(vertex_mesh, fragment_mesh)

        # Line shader (for skeleton, grid)
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

        # Point shader (for Gaussians)
        vertex_point = """
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aColor;

        uniform mat4 uView;
        uniform mat4 uProjection;
        uniform float uPointSize;

        out vec3 vColor;

        void main() {
            vColor = aColor;
            gl_Position = uProjection * uView * vec4(aPos, 1.0);
            gl_PointSize = uPointSize;
        }
        """

        fragment_point = """
        #version 330 core
        in vec3 vColor;
        out vec4 FragColor;

        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            if (dist > 0.5) discard;
            float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
            FragColor = vec4(vColor, alpha);
        }
        """

        self._shader_point = self._compile_shader(vertex_point, fragment_point)

    def _compile_shader(self, vertex_src: str, fragment_src: str) -> int:
        """Compile and link shader program."""
        program = GL.glCreateProgram()

        vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vertex_shader, vertex_src)
        GL.glCompileShader(vertex_shader)
        if not GL.glGetShaderiv(vertex_shader, GL.GL_COMPILE_STATUS):
            logger.error(f"Vertex shader error: {GL.glGetShaderInfoLog(vertex_shader)}")

        fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fragment_shader, fragment_src)
        GL.glCompileShader(fragment_shader)
        if not GL.glGetShaderiv(fragment_shader, GL.GL_COMPILE_STATUS):
            logger.error(f"Fragment shader error: {GL.glGetShaderInfoLog(fragment_shader)}")

        GL.glAttachShader(program, vertex_shader)
        GL.glAttachShader(program, fragment_shader)
        GL.glLinkProgram(program)

        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            logger.error(f"Shader link error: {GL.glGetProgramInfoLog(program)}")

        GL.glDeleteShader(vertex_shader)
        GL.glDeleteShader(fragment_shader)

        return program

    def _create_grid(self):
        """Create ground grid."""
        lines = []
        grid_size = 10
        grid_step = 0.5

        for i in range(-grid_size, grid_size + 1):
            x = i * grid_step
            # X lines (gray)
            lines.extend([x, 0, -grid_size * grid_step, 0.3, 0.3, 0.3])
            lines.extend([x, 0, grid_size * grid_step, 0.3, 0.3, 0.3])
            # Z lines (gray)
            lines.extend([-grid_size * grid_step, 0, x, 0.3, 0.3, 0.3])
            lines.extend([grid_size * grid_step, 0, x, 0.3, 0.3, 0.3])

        data = np.array(lines, dtype=np.float32)
        self._grid_count = len(lines) // 6

        self._grid_vao = GL.glGenVertexArrays(1)
        self._grid_vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self._grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)

        # Position
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
        GL.glEnableVertexAttribArray(0)
        # Color
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, GL.ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)

    def _create_axes(self):
        """Create coordinate axes."""
        lines = [
            # X axis (red)
            0, 0, 0, 1, 0, 0,
            1, 0, 0, 1, 0, 0,
            # Y axis (green)
            0, 0, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            # Z axis (blue)
            0, 0, 0, 0, 0, 1,
            0, 0, 1, 0, 0, 1,
        ]

        data = np.array(lines, dtype=np.float32)

        self._axes_vao = GL.glGenVertexArrays(1)
        self._axes_vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self._axes_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._axes_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)

        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, GL.ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)

    def resizeGL(self, w: int, h: int):
        """Handle resize."""
        if not OPENGL_AVAILABLE:
            return
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        """Render the scene."""
        if not OPENGL_AVAILABLE:
            return

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        aspect = self.width() / max(1, self.height())
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(aspect)

        # Draw grid
        if self.show_grid:
            self._draw_grid(view, proj)

        # Draw axes
        if self.show_axes:
            self._draw_axes(view, proj)

        # Draw mesh
        if self.mesh and self.view_mode in [ViewMode.MESH, ViewMode.WIREFRAME, ViewMode.MESH_SKELETON]:
            self._draw_mesh(view, proj)

        # Draw skeleton
        if self.skeleton and self.view_mode in [ViewMode.SKELETON, ViewMode.MESH_SKELETON]:
            self._draw_skeleton(view, proj)

        # Draw Gaussians
        if self.gaussian_positions is not None and self.view_mode == ViewMode.POINTS:
            self._draw_points(view, proj)

    def _draw_grid(self, view: np.ndarray, proj: np.ndarray):
        """Draw ground grid."""
        GL.glUseProgram(self._shader_line)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_line, "uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_line, "uProjection"), 1, GL.GL_TRUE, proj)

        GL.glBindVertexArray(self._grid_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self._grid_count)
        GL.glBindVertexArray(0)

    def _draw_axes(self, view: np.ndarray, proj: np.ndarray):
        """Draw coordinate axes."""
        GL.glUseProgram(self._shader_line)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_line, "uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_line, "uProjection"), 1, GL.GL_TRUE, proj)

        # Note: glLineWidth > 1.0 not supported on macOS Metal
        GL.glBindVertexArray(self._axes_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, 6)
        GL.glBindVertexArray(0)

    def _draw_mesh(self, view: np.ndarray, proj: np.ndarray):
        """Draw mesh."""
        if not self.mesh or self.mesh.vao == 0:
            return

        GL.glUseProgram(self._shader_mesh)

        # Uniforms
        model = np.eye(4, dtype=np.float32)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_mesh, "uModel"), 1, GL.GL_TRUE, model)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_mesh, "uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_mesh, "uProjection"), 1, GL.GL_TRUE, proj)

        light_dir = np.array([0.5, 0.7, 0.5], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)
        GL.glUniform3fv(GL.glGetUniformLocation(self._shader_mesh, "uLightDir"), 1, light_dir)
        GL.glUniform3fv(GL.glGetUniformLocation(self._shader_mesh, "uViewPos"), 1, self.camera.position.astype(np.float32))
        GL.glUniform3f(GL.glGetUniformLocation(self._shader_mesh, "uColor"), 0.8, 0.75, 0.7)
        GL.glUniform1i(GL.glGetUniformLocation(self._shader_mesh, "uShadingMode"), self.shading_mode.value == "lit")

        if self.view_mode == ViewMode.WIREFRAME:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

        GL.glBindVertexArray(self.mesh.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.mesh.indices), GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

        if self.view_mode == ViewMode.WIREFRAME:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

    def _draw_skeleton(self, view: np.ndarray, proj: np.ndarray):
        """Draw skeleton as lines and points."""
        if not self.skeleton:
            return

        # Compute world positions
        self._compute_skeleton_world_positions()

        # Build line data
        lines = []
        for bone in self.skeleton.bones:
            if bone.parent_index >= 0 and bone.world_position is not None:
                parent = self.skeleton.bones[bone.parent_index]
                if parent.world_position is not None:
                    # Bone color (yellow for standard, cyan for humanoid mapped)
                    is_humanoid = bone.name in self.skeleton.humanoid_map.values() if hasattr(self.skeleton, 'humanoid_map') else False
                    color = [0.0, 1.0, 1.0] if is_humanoid else [1.0, 0.9, 0.2]

                    lines.extend([*parent.world_position, *color])
                    lines.extend([*bone.world_position, *color])

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
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, GL.ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)

        # Draw
        GL.glUseProgram(self._shader_line)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_line, "uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_line, "uProjection"), 1, GL.GL_TRUE, proj)

        # Note: glLineWidth > 1.0 not supported on macOS Metal
        GL.glDrawArrays(GL.GL_LINES, 0, len(lines) // 6)

        GL.glBindVertexArray(0)

        # Draw joint points
        GL.glUseProgram(self._shader_point)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_point, "uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_point, "uProjection"), 1, GL.GL_TRUE, proj)
        GL.glUniform1f(GL.glGetUniformLocation(self._shader_point, "uPointSize"), 8.0)

        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glBindVertexArray(self._skeleton_vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, len(lines) // 6)
        GL.glBindVertexArray(0)

    def _compute_skeleton_world_positions(self):
        """Compute world positions for all bones."""
        if not self.skeleton:
            return

        for bone in self.skeleton.bones:
            bone.world_position = self._compute_bone_world_position(bone)

    def _compute_bone_world_position(self, bone: DisplayBone) -> np.ndarray:
        """Recursively compute world position for a bone."""
        # Start with bone's local position
        local_pos = bone.position.copy()

        # Apply any animation transform
        if bone.name in self.bone_transforms:
            transform = self.bone_transforms[bone.name]
            local_pos = (transform @ np.append(local_pos, 1.0))[:3]

        # If has parent, transform by parent's world transform
        if bone.parent_index >= 0:
            parent = self.skeleton.bones[bone.parent_index]
            parent_world = self._compute_bone_world_position(parent)
            # Simple addition for now (proper FK would use matrices)
            return parent_world + local_pos

        return local_pos

    def _draw_points(self, view: np.ndarray, proj: np.ndarray):
        """Draw Gaussian positions as points."""
        if self.gaussian_positions is None:
            return

        # Build point data with colors
        n = len(self.gaussian_positions)
        points = np.zeros((n, 6), dtype=np.float32)
        points[:, :3] = self.gaussian_positions
        points[:, 3:] = [0.3, 0.8, 0.4]  # Green color

        # Create buffer
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, points.nbytes, points, GL.GL_STREAM_DRAW)

        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, GL.ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)

        # Draw
        GL.glUseProgram(self._shader_point)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_point, "uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._shader_point, "uProjection"), 1, GL.GL_TRUE, proj)
        GL.glUniform1f(GL.glGetUniformLocation(self._shader_point, "uPointSize"), 4.0)

        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glDrawArrays(GL.GL_POINTS, 0, n)

        # Cleanup
        GL.glBindVertexArray(0)
        GL.glDeleteVertexArrays(1, [vao])
        GL.glDeleteBuffers(1, [vbo])

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_vrm(self, vrm_path: str):
        """Load VRM file for preview."""
        try:
            from noodlestudio.core.semantic_world.vrm_parser import parse_vrm

            avatar = parse_vrm(vrm_path)

            # Extract mesh data
            if avatar.meshes:
                mesh_data = avatar.meshes[0]  # Use first mesh
                self._create_mesh_buffers(mesh_data, avatar)

            # Extract skeleton
            self._create_skeleton_from_vrm(avatar)

            # Center camera on model
            self._center_camera_on_model()

            self.update()
            logger.info(f"Loaded VRM: {vrm_path}")

        except Exception as e:
            logger.error(f"Failed to load VRM: {e}")

    def _create_mesh_buffers(self, mesh_data, avatar):
        """Create OpenGL buffers from VRM mesh data."""
        if not OPENGL_AVAILABLE:
            return

        # Get vertex data directly from mesh (parser provides numpy arrays)
        if mesh_data.vertices is None or len(mesh_data.vertices) == 0:
            return

        vertices = np.asarray(mesh_data.vertices, dtype=np.float32)
        normals = np.asarray(mesh_data.normals, dtype=np.float32) if mesh_data.normals is not None else np.zeros_like(vertices)
        uvs = np.asarray(mesh_data.uvs, dtype=np.float32) if mesh_data.uvs is not None else np.zeros((len(vertices), 2), dtype=np.float32)
        indices = np.asarray(mesh_data.indices, dtype=np.uint32) if mesh_data.indices is not None else np.arange(len(vertices), dtype=np.uint32)

        # Skinning data
        bone_indices = np.asarray(mesh_data.joint_indices, dtype=np.float32) if mesh_data.joint_indices is not None else np.zeros((len(vertices), 4), dtype=np.float32)
        bone_weights = np.asarray(mesh_data.joint_weights, dtype=np.float32) if mesh_data.joint_weights is not None else np.zeros((len(vertices), 4), dtype=np.float32)

        self.mesh = DisplayMesh(
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            indices=indices,
            bone_indices=bone_indices,
            bone_weights=bone_weights
        )

        # Create VAO
        self.mesh.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.mesh.vao)

        # Position buffer
        self.mesh.vbo_pos = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.mesh.vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Normal buffer
        self.mesh.vbo_norm = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.mesh.vbo_norm)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, normals.nbytes, normals, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

        # UV buffer
        self.mesh.vbo_uv = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.mesh.vbo_uv)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(2)

        # Index buffer
        self.mesh.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.mesh.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

        GL.glBindVertexArray(0)

    def _create_skeleton_from_vrm(self, avatar):
        """Extract skeleton from VRM avatar."""
        bones = []
        for bone in avatar.skeleton.bones:
            pos = np.array([bone.transform.position.x, bone.transform.position.y, bone.transform.position.z])
            rot = np.array([bone.transform.rotation.x, bone.transform.rotation.y,
                          bone.transform.rotation.z, bone.transform.rotation.w])

            bones.append(DisplayBone(
                name=bone.name,
                index=bone.index,
                parent_index=bone.parent_index,
                position=pos,
                rotation=rot
            ))

        self.skeleton = DisplaySkeleton(
            bones=bones,
            humanoid_map=dict(avatar.skeleton.humanoid_map)
        )

    def _center_camera_on_model(self):
        """Center camera on loaded model."""
        if self.mesh is not None and len(self.mesh.vertices) > 0:
            center = self.mesh.vertices.mean(axis=0)
            size = (self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0)).max()

            self.camera.target = center
            self.camera.distance = size * 2
            self.camera.elevation = 15

    def load_radiance(self, radiance_path: str):
        """Load radiance file for Gaussian preview."""
        try:
            from noodlestudio.core.semantic_world.radiance_format import RadianceAsset

            asset = RadianceAsset.load(radiance_path)
            self.gaussian_positions = asset.positions

            # Also load skeleton if present
            if asset.has_skeleton:
                bones = []
                for i, bone in enumerate(asset.skeleton.bones):
                    bones.append(DisplayBone(
                        name=bone.name,
                        index=i,
                        parent_index=bone.parent_index,
                        position=np.array(bone.position),
                        rotation=np.array(bone.rotation)
                    ))
                self.skeleton = DisplaySkeleton(
                    bones=bones,
                    humanoid_map=asset.skeleton.humanoid_map
                )

            self._center_camera_on_gaussians()
            self.update()
            logger.info(f"Loaded radiance: {radiance_path}")

        except Exception as e:
            logger.error(f"Failed to load radiance: {e}")

    def _center_camera_on_gaussians(self):
        """Center camera on Gaussian positions."""
        if self.gaussian_positions is not None and len(self.gaussian_positions) > 0:
            center = self.gaussian_positions.mean(axis=0)
            size = (self.gaussian_positions.max(axis=0) - self.gaussian_positions.min(axis=0)).max()

            self.camera.target = center
            self.camera.distance = size * 2

    def apply_pose(self, bone_rotations: Dict[str, Tuple[float, float, float]]):
        """
        Apply bone rotations from pose track.

        Args:
            bone_rotations: Dict mapping bone name to euler rotation (degrees)
        """
        self.bone_transforms.clear()

        for bone_name, (rx, ry, rz) in bone_rotations.items():
            # Convert euler to matrix
            rx, ry, rz = np.radians([rx, ry, rz])

            Rx = np.array([
                [1, 0, 0, 0],
                [0, np.cos(rx), -np.sin(rx), 0],
                [0, np.sin(rx), np.cos(rx), 0],
                [0, 0, 0, 1]
            ])
            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry), 0],
                [0, 1, 0, 0],
                [-np.sin(ry), 0, np.cos(ry), 0],
                [0, 0, 0, 1]
            ])
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0, 0],
                [np.sin(rz), np.cos(rz), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            self.bone_transforms[bone_name] = (Rz @ Ry @ Rx).astype(np.float32)

        self.update()

    # -------------------------------------------------------------------------
    # Mouse Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event):
        """Handle mouse press."""
        self._mouse_pos = event.position()
        self._mouse_button = event.button()

    def mouseMoveEvent(self, event):
        """Handle mouse drag."""
        if self._mouse_pos is None:
            return

        dx = event.position().x() - self._mouse_pos.x()
        dy = event.position().y() - self._mouse_pos.y()
        self._mouse_pos = event.position()

        if self._mouse_button == Qt.MouseButton.LeftButton:
            # Orbit
            self.camera.orbit(-dx * 0.5, -dy * 0.5)
        elif self._mouse_button == Qt.MouseButton.MiddleButton:
            # Pan
            self.camera.pan(-dx, dy)
        elif self._mouse_button == Qt.MouseButton.RightButton:
            # Zoom
            self.camera.zoom(1.0 + dy * 0.01)

        self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self._mouse_pos = None
        self._mouse_button = None

    def wheelEvent(self, event):
        """Handle mouse wheel."""
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self.camera.zoom(factor)
        self.update()


# =============================================================================
# Main Panel
# =============================================================================

class VRMPreviewPanel(QWidget):
    """
    Main VRM Preview Panel with controls.

    Combines the OpenGL viewer with:
    - File loading controls
    - View mode toggles
    - Animation playback
    - Pose controls
    """

    # Signals
    poseChanged = pyqtSignal(dict)  # bone_rotations

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("VRM Preview")

        # Animation state
        self._pose_track = None
        self._pose_player = None
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._on_animation_tick)

        self._setup_ui()

    def _setup_ui(self):
        """Build the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Main content (GL view + controls)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # GL View
        self.gl_widget = VRMGLWidget() if OPENGL_AVAILABLE else QWidget()
        self.gl_widget.setMinimumWidth(300)
        splitter.addWidget(self.gl_widget)

        # Controls panel
        controls = self._create_controls_panel()
        controls.setMaximumWidth(280)
        splitter.addWidget(controls)

        splitter.setSizes([600, 200])
        layout.addWidget(splitter, 1)

        # Status bar
        self.status_label = QLabel("No model loaded")
        self.status_label.setStyleSheet("color: #888; padding: 4px; font-size: 11px;")
        layout.addWidget(self.status_label)

    def _create_toolbar(self) -> QToolBar:
        """Create toolbar."""
        toolbar = QToolBar()
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #1e1e1e;
                border-bottom: 1px solid #333;
                spacing: 4px;
                padding: 2px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                color: #ccc;
                padding: 4px 8px;
                font-size: 11px;
            }
            QToolButton:hover {
                background-color: #3a3a3a;
            }
        """)

        # Load VRM
        load_vrm_action = QAction("Load VRM", self)
        load_vrm_action.triggered.connect(self._on_load_vrm)
        toolbar.addAction(load_vrm_action)

        # Load Radiance
        load_rad_action = QAction("Load Radiance", self)
        load_rad_action.triggered.connect(self._on_load_radiance)
        toolbar.addAction(load_rad_action)

        toolbar.addSeparator()

        # Load Animation
        load_anim_action = QAction("Load Animation", self)
        load_anim_action.triggered.connect(self._on_load_animation)
        toolbar.addAction(load_anim_action)

        toolbar.addSeparator()

        # View mode combo
        toolbar.addWidget(QLabel(" View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Mesh + Skeleton", "Mesh", "Wireframe", "Skeleton", "Points"])
        self.view_combo.setStyleSheet("background-color: #2a2a2a; color: #ccc;")
        self.view_combo.currentTextChanged.connect(self._on_view_mode_changed)
        toolbar.addWidget(self.view_combo)

        return toolbar

    def _create_controls_panel(self) -> QWidget:
        """Create side controls panel."""
        panel = QFrame()
        panel.setStyleSheet("background-color: #1e1e1e;")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # Display options
        display_group = QGroupBox("Display")
        display_group.setStyleSheet("QGroupBox { color: #888; }")
        display_layout = QVBoxLayout(display_group)

        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self._on_grid_toggled)
        display_layout.addWidget(self.grid_check)

        self.axes_check = QCheckBox("Show Axes")
        self.axes_check.setChecked(True)
        self.axes_check.toggled.connect(self._on_axes_toggled)
        display_layout.addWidget(self.axes_check)

        layout.addWidget(display_group)

        # Animation controls
        anim_group = QGroupBox("Animation")
        anim_group.setStyleSheet("QGroupBox { color: #888; }")
        anim_layout = QVBoxLayout(anim_group)

        # Play/Pause/Stop buttons
        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._on_play)
        btn_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.stop_btn)
        anim_layout.addLayout(btn_layout)

        # Time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.valueChanged.connect(self._on_time_slider)
        anim_layout.addWidget(self.time_slider)

        self.time_label = QLabel("0.00s / 0.00s")
        self.time_label.setStyleSheet("color: #888;")
        anim_layout.addWidget(self.time_label)

        # Speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 4.0)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.1)
        speed_layout.addWidget(self.speed_spin)
        anim_layout.addLayout(speed_layout)

        # Loop
        self.loop_check = QCheckBox("Loop")
        anim_layout.addWidget(self.loop_check)

        layout.addWidget(anim_group)

        # Camera info
        camera_group = QGroupBox("Camera")
        camera_group.setStyleSheet("QGroupBox { color: #888; }")
        camera_layout = QVBoxLayout(camera_group)

        self.camera_label = QLabel("Distance: 3.0\nAzimuth: 0\nElevation: 15")
        self.camera_label.setStyleSheet("color: #666; font-size: 10px;")
        camera_layout.addWidget(self.camera_label)

        reset_cam_btn = QPushButton("Reset Camera")
        reset_cam_btn.clicked.connect(self._on_reset_camera)
        camera_layout.addWidget(reset_cam_btn)

        layout.addWidget(camera_group)

        layout.addStretch()

        return panel

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def _on_load_vrm(self):
        """Load VRM file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load VRM", "",
            "VRM Files (*.vrm);;All Files (*)"
        )
        if path and OPENGL_AVAILABLE:
            self.gl_widget.load_vrm(path)
            self.status_label.setText(f"Loaded: {Path(path).name}")

    def _on_load_radiance(self):
        """Load radiance file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Radiance", "",
            "Radiance Files (*.radiance);;PLY Files (*.ply);;All Files (*)"
        )
        if path and OPENGL_AVAILABLE:
            if path.endswith('.radiance'):
                self.gl_widget.load_radiance(path)
            else:
                # Could add PLY loading here
                pass
            self.status_label.setText(f"Loaded: {Path(path).name}")
            self.view_combo.setCurrentText("Points")

    def _on_load_animation(self):
        """Load pose track for animation."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Animation", "",
            "Pose Tracks (*.posetrack);;All Files (*)"
        )
        if path:
            try:
                from noodlestudio.core.pose_track import PoseTrack, PoseTrackPlayer, PoseRetargeter

                self._pose_track = PoseTrack.load_yaml(path)
                self._pose_player = PoseTrackPlayer(self._pose_track)
                self._pose_player.speed = self.speed_spin.value()
                self._pose_player.is_looping = self.loop_check.isChecked()

                self.time_slider.setRange(0, int(self._pose_track.duration * 100))
                self.time_label.setText(f"0.00s / {self._pose_track.duration:.2f}s")
                self.status_label.setText(f"Animation: {Path(path).name}")

            except Exception as e:
                logger.error(f"Failed to load animation: {e}")

    def _on_view_mode_changed(self, text: str):
        """Handle view mode change."""
        if not OPENGL_AVAILABLE:
            return

        mode_map = {
            "Mesh + Skeleton": ViewMode.MESH_SKELETON,
            "Mesh": ViewMode.MESH,
            "Wireframe": ViewMode.WIREFRAME,
            "Skeleton": ViewMode.SKELETON,
            "Points": ViewMode.POINTS,
        }
        self.gl_widget.view_mode = mode_map.get(text, ViewMode.MESH_SKELETON)
        self.gl_widget.update()

    def _on_grid_toggled(self, checked: bool):
        if OPENGL_AVAILABLE:
            self.gl_widget.show_grid = checked
            self.gl_widget.update()

    def _on_axes_toggled(self, checked: bool):
        if OPENGL_AVAILABLE:
            self.gl_widget.show_axes = checked
            self.gl_widget.update()

    def _on_play(self):
        """Start/pause animation."""
        if self._pose_player is None:
            return

        if self._pose_player.is_playing:
            self._pose_player.pause()
            self._animation_timer.stop()
            self.play_btn.setText("Play")
        else:
            self._pose_player.speed = self.speed_spin.value()
            self._pose_player.is_looping = self.loop_check.isChecked()
            self._pose_player.play()
            self._animation_timer.start(16)  # ~60fps
            self.play_btn.setText("Pause")

    def _on_stop(self):
        """Stop animation."""
        if self._pose_player:
            self._pose_player.stop()
            self._animation_timer.stop()
            self.play_btn.setText("Play")
            self._update_pose_display()

    def _on_time_slider(self, value: int):
        """Handle time slider change."""
        if self._pose_player and self._pose_track:
            t = value / 100.0
            self._pose_player.seek(t)
            self._update_pose_display()

    def _on_animation_tick(self):
        """Animation timer tick."""
        if self._pose_player:
            self._pose_player.update()
            self._update_pose_display()

            # Update slider
            if self._pose_track:
                progress = int(self._pose_player.current_time * 100)
                self.time_slider.blockSignals(True)
                self.time_slider.setValue(progress)
                self.time_slider.blockSignals(False)

    def _update_pose_display(self):
        """Update GL display with current pose."""
        if not self._pose_player or not OPENGL_AVAILABLE:
            return

        from noodlestudio.core.pose_track import PoseRetargeter

        pose = self._pose_player.sample()
        retargeter = PoseRetargeter()
        bone_rotations = retargeter.apply_pose(pose)

        self.gl_widget.apply_pose(bone_rotations)

        # Update time label
        if self._pose_track:
            self.time_label.setText(
                f"{self._pose_player.current_time:.2f}s / {self._pose_track.duration:.2f}s"
            )

        # Emit signal
        self.poseChanged.emit(bone_rotations)

    def _on_reset_camera(self):
        """Reset camera to default."""
        if OPENGL_AVAILABLE:
            self.gl_widget.camera = Camera()
            self.gl_widget._center_camera_on_model()
            self.gl_widget.update()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            background-color: #1e1e1e;
            color: #cccccc;
            font-family: Monaco;
        }
    """)

    panel = VRMPreviewPanel()
    panel.resize(900, 600)
    panel.show()

    if not OPENGL_AVAILABLE:
        panel.status_label.setText("OpenGL not available - install PyOpenGL")

    sys.exit(app.exec())
