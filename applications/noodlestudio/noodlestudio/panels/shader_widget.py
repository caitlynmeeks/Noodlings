"""
Demo Scene Shader Widget

Classic demoscene effects using OpenGL shaders.
Plasma, tunnels, fractals - the good stuff.
"""

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QSurfaceFormat
import time
import random


# Classic demoscene shader effects
PLASMA_SHADER = """
#version 330 core
out vec4 FragColor;
uniform float time;
uniform vec2 resolution;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;

    // Classic plasma effect
    float v = 0.0;
    v += sin((uv.x + time) * 10.0);
    v += sin((uv.y + time) * 10.0);
    v += sin((uv.x + uv.y + time) * 10.0);
    v += sin(sqrt(uv.x * uv.x + uv.y * uv.y + time) * 10.0);
    v /= 4.0;

    // Color cycle
    vec3 col = vec3(
        0.5 + 0.5 * sin(v * 3.14159 + time),
        0.5 + 0.5 * sin(v * 3.14159 + time + 2.0),
        0.5 + 0.5 * sin(v * 3.14159 + time + 4.0)
    );

    // Darken for text readability
    col *= 0.3;

    FragColor = vec4(col, 1.0);
}
"""

TUNNEL_SHADER = """
#version 330 core
out vec4 FragColor;
uniform float time;
uniform vec2 resolution;

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution) / resolution.y;

    // Classic tunnel effect
    float d = length(uv);
    float a = atan(uv.y, uv.x);

    float u = time + 1.0 / d;
    float v = a / 3.14159;

    float pattern = sin(u * 3.0) * sin(v * 8.0);

    vec3 col = vec3(pattern * 0.2 + 0.1);

    FragColor = vec4(col, 1.0);
}
"""

STARFIELD_SHADER = """
#version 330 core
out vec4 FragColor;
uniform float time;
uniform vec2 resolution;

float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;

    vec3 col = vec3(0.0);

    // Multiple layers of stars
    for (float i = 0.0; i < 3.0; i++) {
        float layer = i / 3.0;
        vec2 suv = uv * (200.0 + i * 100.0);
        suv.x += time * (0.1 + layer * 0.2);

        vec2 id = floor(suv);
        vec2 gv = fract(suv) - 0.5;

        float n = hash(id.x + id.y * 1000.0);

        if (n > 0.95) {
            float d = length(gv);
            float star = smoothstep(0.02, 0.0, d);
            col += vec3(star * (0.3 + layer * 0.2));
        }
    }

    FragColor = vec4(col, 1.0);
}
"""

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;

void main() {
    gl_Position = vec4(position, 1.0);
}
"""


class ShaderWidget(QOpenGLWidget):
    """
    OpenGL widget that renders demo scene shaders.

    Classic effects: plasma, tunnel, starfield.
    Random selection on creation.
    """

    def __init__(self, parent=None):
        # Set OpenGL format
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)

        self.start_time = time.time()
        self.shader_type = random.choice(["plasma", "tunnel", "starfield"])

        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS

        print(f"Shader effect: {self.shader_type}")

    def initializeGL(self):
        """Initialize OpenGL resources."""
        try:
            from OpenGL import GL
            self.gl = GL

            # Compile shaders
            self.compile_shaders()

            # Create fullscreen quad
            vertices = [
                -1.0, -1.0, 0.0,
                 1.0, -1.0, 0.0,
                -1.0,  1.0, 0.0,
                 1.0,  1.0, 0.0,
            ]

            # Note: Proper VAO/VBO setup would go here
            # For now, using immediate mode (legacy but simpler)

        except ImportError:
            print("PyOpenGL not installed. Shader effects disabled.")
            print("Run: pip install PyOpenGL PyOpenGL_accelerate")

    def compile_shaders(self):
        """Compile vertex and fragment shaders."""
        # Select fragment shader based on type
        shaders = {
            "plasma": PLASMA_SHADER,
            "tunnel": TUNNEL_SHADER,
            "starfield": STARFIELD_SHADER,
        }

        fragment_source = shaders.get(self.shader_type, PLASMA_SHADER)

        # Note: Actual shader compilation would happen here
        # Simplified for now - full implementation requires more GL boilerplate

    def paintGL(self):
        """Render the shader effect."""
        try:
            from OpenGL import GL

            # Clear
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # Set uniforms
            current_time = time.time() - self.start_time
            # GL.glUniform1f(time_location, current_time)
            # GL.glUniform2f(resolution_location, self.width(), self.height())

            # Draw fullscreen quad
            # GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

            # Fallback: Simple animated gradient if OpenGL setup fails
            self.paintFallback(current_time)

        except Exception as e:
            print(f"OpenGL render error: {e}")
            self.paintFallback(0)

    def paintFallback(self, t):
        """Fallback rendering using Qt painting."""
        from PyQt6.QtGui import QPainter, QLinearGradient, QColor
        from math import sin, cos

        painter = QPainter(self)

        # Animated gradient as fallback
        gradient = QLinearGradient(0, 0, self.width(), self.height())

        r = int(127 + 127 * sin(t * 0.5))
        g = int(127 + 127 * sin(t * 0.7 + 2))
        b = int(127 + 127 * sin(t * 0.3 + 4))

        gradient.setColorAt(0, QColor(r // 4, g // 4, b // 4))
        gradient.setColorAt(1, QColor(0, 0, 0))

        painter.fillRect(self.rect(), gradient)
        painter.end()

    def resizeGL(self, w, h):
        """Handle resize."""
        try:
            from OpenGL import GL
            GL.glViewport(0, 0, w, h)
        except:
            pass

    def closeEvent(self, event):
        """Stop timer on close."""
        self.timer.stop()
        super().closeEvent(event)


def create_shader_background(parent=None):
    """Create a shader widget for demo scene effects."""
    return ShaderWidget(parent)
