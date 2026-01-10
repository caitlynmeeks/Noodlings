# VRMViewport Component

**Status**: Implementation Spec
**Date**: 2026-01-08
**Authors**: Caity + Claude
**Priority**: High (enables Guide experience)

---

## Overview

`VRMViewport` is a UIComponent that renders VRM avatars using OpenGL. It's the standard 3D renderer counterpart to `RadianceViewport` (Gaussian splats).

### Why VRMViewport?

| RadianceViewport | VRMViewport |
|------------------|-------------|
| Gaussian splat rendering | Standard mesh + skeletal animation |
| Experimental, cutting-edge | Battle-tested, reliable |
| MPS/GPU rasterization | OpenGL (PyOpenGL + Qt) |
| Novel visual style | Familiar 3D rendering |

**Decision**: Ship Guide experience with VRMViewport now, add RadianceViewport as upgrade later.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VRMViewport (UIComponent)                │
│  - Serializes to ui.yaml                                     │
│  - Properties: vrm_path, camera, background, interactive     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ creates
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  VRMViewportWidget (QOpenGLWidget)           │
│  - OpenGL 3.3 Core Profile                                   │
│  - Mesh rendering with shaders                               │
│  - Skeleton visualization                                    │
│  - Camera orbit/pan/zoom                                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ uses
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    VRMAvatar (data container)                │
│  - Mesh data (vertices, normals, UVs, indices)              │
│  - Skeleton (bones, humanoid mapping)                        │
│  - Blend shapes (expressions)                                │
│  - MuscleBinding (from model_importer)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ animated by
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Muscle System Integration                   │
│  - Receives muscle values (47 normalized floats)            │
│  - PoseRetargeter → bone rotations                          │
│  - Applies to skeleton each frame                            │
└─────────────────────────────────────────────────────────────┘
```

---

## File Location

```
noodlestudio/runtime/ui/components/
├── radiance_viewport.py    # Existing - Gaussian splats
├── vrm_viewport.py         # NEW - OpenGL VRM rendering
└── __init__.py             # Register both
```

---

## UIComponent Definition

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `vrm_path` | string | "" | Path to .vrm file (relative to project) |
| `camera` | CameraConfig | default | Camera position/orientation |
| `background` | string | "#1e1e1e" | Background color (hex) |
| `show_skeleton` | bool | false | Overlay skeleton visualization |
| `show_grid` | bool | false | Show ground grid |
| `interactive` | bool | true | Enable camera controls |
| `transparent` | bool | false | Clear alpha channel for UI compositing |

### Events

| Event | Payload | Description |
|-------|---------|-------------|
| `onLoad` | `{vrm_path, bone_count, vertex_count}` | VRM loaded successfully |
| `onClick` | `{x, y, bone_name?}` | Click with optional bone hit |
| `onPoseApplied` | `{muscle_count}` | Pose was applied |

### YAML Example

```yaml
# In ui.yaml
components:
  - type: VRMViewport
    name: avatar_view
    x: 0
    y: 0
    width: 512
    height: 512
    anchors: [left, top, bottom]
    vrm_path: Radiances/AjoMajo.vrm
    camera:
      distance: 2.5
      elevation: 10
      azimuth: 180
      target: [0, 0.9, 0]
    background: "#1a1a2e"
    interactive: true
```

### Transparent Mode (Character Overlay)

Use `transparent: true` to render the character over the UI with no background:

```yaml
  - type: VRMViewport
    name: guide_character
    transparent: true        # Alpha = 0 where no geometry
    x: 50
    y: 100
    width: 300
    height: 400
    vrm_path: Radiances/AjoMajo.vrm
    interactive: false       # Disable camera controls
    camera:
      distance: 1.5
      elevation: 5
      azimuth: 170
      target: [0, 0.8, 0]
```

This allows the character to walk around the foreground of the UI.

**Important**: QOpenGLWidget can't composite transparently with sibling widgets. For true transparency, use a **separate frameless overlay window**:

```python
from PyQt6.QtCore import Qt

class CharacterOverlay(QMainWindow):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window

        # Frameless, transparent, stays on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Create VRMViewport with transparent=True
        component = VRMViewport("guide")
        component.transparent = True
        component.vrm_path = "path/to/avatar.vrm"

        self.viewport = VRMViewportWidget(component, self)
        self.setCentralWidget(self.viewport)

    def follow_parent(self):
        """Call periodically to track parent window position."""
        geo = self.parent_window.geometry()
        self.move(geo.right() - self.width(), geo.top() + 100)
```

This creates a character that floats over the UI and follows the main window.

---

## Implementation

### vrm_viewport.py

```python
"""
VRMViewport Component - OpenGL VRM avatar renderer for UI canvas.

Like RadianceViewport but for standard 3D mesh + skeletal animation.
Uses the muscle system for rig-agnostic animation.
"""

import logging
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass

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

        return viewport


# ============================================================================
# Qt Widget Implementation
# ============================================================================

try:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout
    from PyQt6.QtCore import Qt, QPoint, pyqtSignal
    from PyQt6.QtGui import QMouseEvent, QWheelEvent, QSurfaceFormat
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    import OpenGL.GL as GL
    import numpy as np
    QT_AVAILABLE = True
    OPENGL_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    OPENGL_AVAILABLE = False


if QT_AVAILABLE and OPENGL_AVAILABLE:

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

        def __init__(self, component: VRMViewport, parent=None):
            # Set up OpenGL format
            fmt = QSurfaceFormat()
            fmt.setSamples(4)  # MSAA
            fmt.setDepthBufferSize(24)
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            QSurfaceFormat.setDefaultFormat(fmt)

            super().__init__(parent)
            self.component = component
            self.setObjectName(component.name or "vrm_viewport")

            # Avatar data
            self._avatar = None           # Parsed VRM data
            self._muscle_binding = None   # MuscleBinding for retargeting
            self._retargeter = None       # PoseRetargeter instance

            # Display data (GPU buffers)
            self._mesh = None             # DisplayMesh with VAO/VBO
            self._skeleton = None         # DisplaySkeleton

            # Current pose (muscle values)
            self._current_muscles: Dict[str, float] = {}
            self._current_blend_shapes: Dict[str, float] = {}

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

            # Grid/axes buffers
            self._grid_vao = 0
            self._grid_vbo = 0
            self._grid_count = 0

            self.setMinimumSize(200, 200)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # =====================================================================
        # OpenGL Setup
        # =====================================================================

        def initializeGL(self):
            """Initialize OpenGL resources."""
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
                self._load_vrm_internal(self.component.vrm_path)

        def _create_shaders(self):
            """Create shader programs."""
            # Mesh shader with skeletal animation support
            vertex_mesh = """
            #version 330 core
            layout(location = 0) in vec3 aPos;
            layout(location = 1) in vec3 aNormal;
            layout(location = 2) in vec2 aUV;
            layout(location = 3) in vec4 aBoneIndices;
            layout(location = 4) in vec4 aBoneWeights;

            uniform mat4 uModel;
            uniform mat4 uView;
            uniform mat4 uProjection;
            uniform mat4 uBoneMatrices[128];
            uniform bool uUseSkinning;

            out vec3 vNormal;
            out vec3 vWorldPos;
            out vec2 vUV;

            void main() {
                vec4 pos = vec4(aPos, 1.0);
                vec3 norm = aNormal;

                if (uUseSkinning) {
                    mat4 skinMatrix =
                        aBoneWeights.x * uBoneMatrices[int(aBoneIndices.x)] +
                        aBoneWeights.y * uBoneMatrices[int(aBoneIndices.y)] +
                        aBoneWeights.z * uBoneMatrices[int(aBoneIndices.z)] +
                        aBoneWeights.w * uBoneMatrices[int(aBoneIndices.w)];

                    pos = skinMatrix * pos;
                    norm = mat3(skinMatrix) * norm;
                }

                vec4 worldPos = uModel * pos;
                vWorldPos = worldPos.xyz;
                vNormal = mat3(transpose(inverse(uModel))) * norm;
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

            out vec4 FragColor;

            void main() {
                vec3 normal = normalize(vNormal);
                float diff = max(dot(normal, uLightDir), 0.0);
                float ambient = 0.35;
                vec3 color = uColor * (ambient + diff * 0.65);
                FragColor = vec4(color, 1.0);
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
                logger.error(f"Vertex shader error: {GL.glGetShaderInfoLog(vs)}")

            fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
            GL.glShaderSource(fs, fragment_src)
            GL.glCompileShader(fs)
            if not GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS):
                logger.error(f"Fragment shader error: {GL.glGetShaderInfoLog(fs)}")

            GL.glAttachShader(program, vs)
            GL.glAttachShader(program, fs)
            GL.glLinkProgram(program)

            if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
                logger.error(f"Shader link error: {GL.glGetProgramInfoLog(program)}")

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
            self._load_vrm_internal(path)

        def _load_vrm_internal(self, path: str):
            """Internal VRM loading."""
            try:
                from noodlestudio.core.semantic_world.vrm_parser import parse_vrm
                from noodlestudio.core.model_importer import ModelImporter
                from noodlestudio.core.pose_track import PoseRetargeter

                # Parse VRM
                self._avatar = parse_vrm(path)

                # Create muscle binding
                importer = ModelImporter()
                self._muscle_binding = importer.create_muscle_binding_from_avatar(
                    self._avatar
                )

                # Create retargeter
                self._retargeter = PoseRetargeter(self._muscle_binding)

                # Create GPU buffers
                self._create_mesh_buffers()
                self._create_skeleton_data()

                # Center camera
                self._center_camera()

                # Emit signal
                bone_count = len(self._avatar.skeleton.bones) if self._avatar.skeleton else 0
                vertex_count = sum(len(m.vertices) for m in self._avatar.meshes) if self._avatar.meshes else 0
                self.vrmLoaded.emit(path, bone_count, vertex_count)

                logger.info(f"VRM loaded: {path} ({bone_count} bones, {vertex_count} verts)")
                self.update()

            except Exception as e:
                logger.error(f"Failed to load VRM: {e}")
                import traceback
                traceback.print_exc()

        def _create_mesh_buffers(self):
            """Create OpenGL buffers from avatar mesh data."""
            if not self._avatar or not self._avatar.meshes:
                return

            # Combine all meshes for simplicity (could support multiple later)
            mesh = self._avatar.meshes[0]

            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            normals = np.asarray(mesh.normals, dtype=np.float32) if mesh.normals is not None else np.zeros_like(vertices)
            uvs = np.asarray(mesh.uvs, dtype=np.float32) if mesh.uvs is not None else np.zeros((len(vertices), 2), dtype=np.float32)
            indices = np.asarray(mesh.indices, dtype=np.uint32) if mesh.indices is not None else np.arange(len(vertices), dtype=np.uint32)

            # Skinning data
            bone_indices = np.asarray(mesh.joint_indices, dtype=np.float32) if mesh.joint_indices is not None else np.zeros((len(vertices), 4), dtype=np.float32)
            bone_weights = np.asarray(mesh.joint_weights, dtype=np.float32) if mesh.joint_weights is not None else np.zeros((len(vertices), 4), dtype=np.float32)

            # Store for rendering
            self._mesh = {
                'vertices': vertices,
                'normals': normals,
                'uvs': uvs,
                'indices': indices,
                'bone_indices': bone_indices,
                'bone_weights': bone_weights,
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

            # Bone indices (location 3)
            vbo_bi = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_bi)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, bone_indices.nbytes, bone_indices, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(3, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(3)

            # Bone weights (location 4)
            vbo_bw = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_bw)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, bone_weights.nbytes, bone_weights, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(4, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(4)

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
                'humanoid_map': dict(self._avatar.skeleton.humanoid_map),
            }

        def _center_camera(self):
            """Center camera on loaded model."""
            if self._mesh and len(self._mesh['vertices']) > 0:
                verts = self._mesh['vertices']
                center = verts.mean(axis=0)
                size = (verts.max(axis=0) - verts.min(axis=0)).max()

                self._target = [center[0], center[1], center[2]]
                self._distance = size * 1.5
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
            if not self._retargeter or not self._current_muscles:
                return

            # Convert muscles to bone rotations via retargeter
            from noodlestudio.core.pose_track import PoseState

            pose = PoseState(
                muscles=self._current_muscles,
                blend_shapes=self._current_blend_shapes,
            )

            self._bone_rotations = self._retargeter.apply_pose(pose)

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

        def focus_on_bone(self, bone_name: str):
            """Focus camera on a specific bone."""
            # TODO: Implement bone world position lookup
            pass

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

            # Draw grid
            if self.component.show_grid:
                self._draw_grid(view, proj)

            # Draw mesh
            if self._mesh:
                self._draw_mesh(view, proj)

            # Draw skeleton overlay
            if self.component.show_skeleton and self._skeleton:
                self._draw_skeleton(view, proj)

        def _view_matrix(self) -> np.ndarray:
            """Compute view matrix from orbit camera."""
            import math

            az_rad = math.radians(self._azimuth)
            el_rad = math.radians(self._elevation)

            x = self._distance * math.cos(el_rad) * math.sin(az_rad)
            y = self._distance * math.sin(el_rad)
            z = self._distance * math.cos(el_rad) * math.cos(az_rad)

            pos = np.array([
                self._target[0] + x,
                self._target[1] + y,
                self._target[2] + z
            ])

            target = np.array(self._target)
            forward = target - pos
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

        def _projection_matrix(self, aspect: float) -> np.ndarray:
            """Compute perspective projection matrix."""
            import math

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
            """Draw avatar mesh."""
            GL.glUseProgram(self._shader_mesh)

            model = np.eye(4, dtype=np.float32)
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._shader_mesh, "uModel"),
                1, GL.GL_TRUE, model
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._shader_mesh, "uView"),
                1, GL.GL_TRUE, view
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._shader_mesh, "uProjection"),
                1, GL.GL_TRUE, proj
            )

            # Light and color
            light_dir = np.array([0.5, 0.7, 0.5], dtype=np.float32)
            light_dir = light_dir / np.linalg.norm(light_dir)
            GL.glUniform3fv(
                GL.glGetUniformLocation(self._shader_mesh, "uLightDir"),
                1, light_dir
            )
            GL.glUniform3f(
                GL.glGetUniformLocation(self._shader_mesh, "uColor"),
                0.85, 0.80, 0.75
            )

            # Skinning - disabled for now (TODO: implement bone matrices)
            GL.glUniform1i(
                GL.glGetUniformLocation(self._shader_mesh, "uUseSkinning"),
                0
            )

            GL.glBindVertexArray(self._mesh['vao'])
            GL.glDrawElements(
                GL.GL_TRIANGLES,
                len(self._mesh['indices']),
                GL.GL_UNSIGNED_INT,
                None
            )
            GL.glBindVertexArray(0)

        def _draw_skeleton(self, view: np.ndarray, proj: np.ndarray):
            """Draw skeleton overlay."""
            # TODO: Implement skeleton visualization
            pass

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
```

---

## Renderer Integration

The `QtWidgetRenderer` needs to know how to create `VRMViewportWidget`:

### renderer.py update

```python
# In _create_widget method, add:

elif component.component_type == "VRMViewport":
    from .components.vrm_viewport import VRMViewportWidget
    widget = VRMViewportWidget(component, parent)
```

### components/__init__.py update

```python
from .vrm_viewport import VRMViewport, VRMViewportWidget
```

---

## Usage in Let's Consciousness

### ui.yaml

```yaml
canvas:
  width: 1024
  height: 768
  background: "#0d0d14"

components:
  # VRM avatar viewport (left side)
  - type: VRMViewport
    name: guide_viewport
    x: 0
    y: 0
    width: 512
    height: 768
    anchors: [left, top, bottom]
    vrm_path: Radiances/AjoMajo.vrm
    camera:
      distance: 2.0
      elevation: 5
      azimuth: 180
      target: [0, 0.85, 0]
    background: "#0d0d14"
    show_skeleton: false
    interactive: true

  # Speech bubble (right side)
  - type: Panel
    name: speech_panel
    x: 520
    y: 50
    width: 480
    height: 300
    anchors: [right, top]
    background: "#1a1a2e"
    border_radius: 12
    children:
      - type: Label
        name: speech_text
        x: 16
        y: 16
        width: 448
        height: 268
        text: "Hello! I'm Guide..."
        font_size: 16
        color: "#e0e0e0"
        wrap: true

  # Input area (bottom right)
  - type: TextInput
    name: user_input
    x: 520
    y: 680
    width: 480
    height: 60
    anchors: [right, bottom]
    placeholder: "Ask Guide something..."
```

### Wiring Animation

The facet assembly can send muscle values to the viewport:

```yaml
# assembly.yaml
nodes:
  - type: PoseTrackFacet
    name: idle_pose
    properties:
      track: idle.posetrack
      loop: true

  - type: ScriptedFacet
    name: pose_output
    script: |
      // Get current pose muscles
      const muscles = context.noodle.pose.getMuscles();

      // Send to viewport (via binding)
      context.setOutput('muscles', muscles);

connections:
  - from: idle_pose.pose
    to: pose_output.input
```

The `muscles` output binds to the viewport's `set_muscles` method.

---

## Implementation Checklist

### Phase 1: Basic Rendering
- [ ] Create `vrm_viewport.py` with UIComponent and Widget
- [ ] Register in components/__init__.py
- [ ] Add to renderer.py widget creation
- [ ] Test: Load VRM, display static mesh
- [ ] Test: Camera controls (orbit, pan, zoom)

### Phase 2: Muscle Integration
- [ ] Import PoseRetargeter in widget
- [ ] Implement `set_muscles()` method
- [ ] Compute bone matrices from muscle values
- [ ] Enable GPU skinning in shader
- [ ] Test: Apply muscle values, see mesh deform

### Phase 3: Polish
- [ ] Skeleton overlay visualization
- [ ] Blend shape support
- [ ] Bone raycast for onClick
- [ ] Performance optimization (bone matrix caching)

### Phase 4: Let's Consciousness
- [ ] Update ui.yaml to use VRMViewport
- [ ] Wire Guide's assembly to send muscles
- [ ] Test full loop: chat → facets → pose → viewport

---

## Testing

```bash
# Unit test for VRMViewport component
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest tests/test_vrm_viewport.py -v

# Manual test - standalone viewer
python -c "
from noodlestudio.runtime.ui.components.vrm_viewport import VRMViewport
from PyQt6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
# ... create widget, load VRM, show
"

# Integration test - Let's Consciousness
python -m noodlestudio.runtime.cli Projects/lets-consciousness --gui
```

---

## See Also

- [Animation Muscle System](animation-muscle-system.md) - Muscle definitions and retargeting
- [UI Canvas](ui-canvas.md) - UIComponent system
- [RadianceViewport](radiance-viewport.md) - Gaussian splat alternative

---

*"Standard 3D for standard experiences. Gaussian splats for the wow factor."*
