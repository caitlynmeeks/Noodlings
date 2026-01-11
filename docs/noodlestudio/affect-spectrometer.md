# Affect Spectrometer

> Visualizing emotional state in affect space, like chromaticity diagrams for color

## The Insight

Color scientists visualize color spaces with chromaticity diagrams - 2D/3D representations that show the full gamut of possible colors. We can do the same for affect.

PAD (Pleasure-Arousal-Dominance) is a 3D space. Every emotional state maps to a point in this cube. By visualizing the space, we make the invisible visible - users can SEE where a noodling's emotions are, how they move, and where they're headed.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   COLOR SPACE                          AFFECT SPACE                     │
│                                                                         │
│   ┌─────────────┐                     ┌─────────────┐                  │
│   │  CIE 1931   │                     │    PAD      │                  │
│   │  Chromatic  │                     │   Cube      │                  │
│   │  Diagram    │                     │             │                  │
│   │    ___      │                     │  Arousal    │                  │
│   │   /   \     │                     │     ↑   ┌───┤                  │
│   │  |  ●  |    │        ═══▶         │     │  ╱│   │                  │
│   │   \___/     │                     │     │ ╱ │ ● │ ← Current        │
│   │             │                     │     │╱──┼───│   emotion        │
│   │  Current    │                     │  P ←┘   │   │                  │
│   │  color      │                     │        D↓   │                  │
│   └─────────────┘                     └─────────────┘                  │
│                                                                         │
│   "What color is this?"               "What emotion is this?"          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## The PAD Space

### Axes

| Axis | Range | Low (-1) | Neutral (0) | High (+1) |
|------|-------|----------|-------------|-----------|
| **P** (Pleasure) | -1 to +1 | Misery, pain | Neutral | Joy, ecstasy |
| **A** (Arousal) | 0 to 1 | Sleepy, calm | Alert | Excited, frantic |
| **D** (Dominance) | 0 to 1 | Submissive, helpless | Balanced | Dominant, in control |

### Archetypal Emotions as Regions

Emotions cluster in specific regions of PAD space:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                        HIGH AROUSAL                                     │
│                            ▲                                            │
│                            │                                            │
│     TERRIFIED              │              TRIUMPHANT                    │
│     (-P, +A, -D)           │              (+P, +A, +D)                  │
│     Fear, panic            │              Victory, power               │
│                            │                                            │
│                            │                                            │
│     ANGRY                  │              EXCITED                       │
│     (-P, +A, +D)           │              (+P, +A, -D)                  │
│     Rage, fury             │              Thrill, anticipation         │
│                            │                                            │
│  ◄──────────────────────── ● ────────────────────────►                 │
│  NEGATIVE VALENCE          │            POSITIVE VALENCE               │
│                            │                                            │
│     DEPRESSED              │              RELAXED                       │
│     (-P, -A, -D)           │              (+P, -A, +D)                  │
│     Despair, grief         │              Content, serene              │
│                            │                                            │
│                            │                                            │
│     SAD                    │              PEACEFUL                      │
│     (-P, -A, +D)           │              (+P, -A, -D)                  │
│     Melancholy             │              Calm, sleepy                 │
│                            │                                            │
│                            ▼                                            │
│                        LOW AROUSAL                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Visualization Modes

### 1. PAD Cube (3D)

The full 3D representation - a unit cube with the current emotion as a glowing point.

```
                    Arousal (1.0)
                         ▲
                        ╱│╲
                       ╱ │ ╲
                      ╱  │  ╲
                     ╱   │   ╲
                    ╱    │    ╲
            ┌─────╱─────┬┼─────╲─────┐
           ╱│    ╱      ││      ╲    │╲
          ╱ │   ╱       ││       ╲   │ ╲
         ╱  │  ╱   ┌────┼┼────┐   ╲  │  ╲
        ╱   │ ╱    │    ●│    │    ╲ │   ╲
       ╱    │╱     │  ╱  │    │     ╲│    ╲
      ┼─────┼──────┼─╱───┼────┼──────┼─────┼───▶ Dominance (1.0)
       ╲    │╲     │╱    │    │     ╱│    ╱
        ╲   │ ╲    │     │    │    ╱ │   ╱
         ╲  │  ╲   └─────┼────┘   ╱  │  ╱
          ╲ │   ╲        │       ╱   │ ╱
           ╲│    ╲       │      ╱    │╱
            └─────╲──────┼─────╱─────┘
                   ╲     │    ╱
                    ╲    │   ╱
                     ╲   │  ╱
                      ╲  │ ╱
                       ╲ │╱
                        ╲│
                         ▼
                    Pleasure (-1.0)

    ● = Current emotional state
    Trail shows recent history
```

**Features:**
- Rotatable/zoomable 3D cube
- Current state as glowing orb
- Temporal trail showing movement
- Semi-transparent faces for visibility
- Corner labels (archetypal emotions)
- Grid lines for reference

### 2. Chromaticity View (2D Projection)

Project 3D PAD space onto 2D, similar to the CIE xy chromaticity diagram.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Arousal                                                        │
│     ▲                                                           │
│     │         EXCITED          TRIUMPHANT                       │
│     │            ○                 ○                            │
│  1.0├─────────────────────────────────────                      │
│     │                    ╱╲                                     │
│     │     ANGRY        ╱  ╲        HAPPY                        │
│     │       ○         ╱    ╲         ○                          │
│     │               ╱   ●   ╲                                   │
│     │              ╱  ╱   ╲  ╲                                  │
│  0.5├────────────╱──╱─────╲──╲────────────                      │
│     │          ╱  ╱         ╲  ╲                                │
│     │        ╱  ╱    trail    ╲  ╲                              │
│     │      ╱  ╱                 ╲  ╲                            │
│     │    ╱  ╱                     ╲  ╲                          │
│     │  ╱  ╱                         ╲  ╲                        │
│  0.0├╱──╱─────────────────────────────╲──╲──                    │
│     │ SAD              PEACEFUL        CONTENT                  │
│     │  ○                  ○               ○                     │
│     └───┴───────────┴───────────┴───────────┴───────▶           │
│       -1.0        -0.5         0.0        +0.5      +1.0        │
│                                                    Pleasure     │
│                                                                 │
│  Dominance shown as: ○ size (bigger = more dominant)            │
│                  or: ○ color intensity                          │
│                  or: third axis coming "out" of screen          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- 2D scatter plot (P vs A)
- Dominance encoded as point size/color/opacity
- Emotion region labels
- Boundary showing "valid" emotional space
- Trail with fading history

### 3. Radar/Spider Chart

Three axes radiating from center - good for comparing multiple noodlings.

```
                         Arousal
                            │
                           1.0
                            │
                            │
                      ╱─────┼─────╲
                    ╱       │       ╲
                  ╱         │         ╲
                ╱           │           ╲
              ╱      ●──────┼            ╲
            ╱     ╱         │              ╲
          ╱    ╱            │                ╲
        ╱   ╱               │                  ╲
      1.0╱                  │                    ╲1.0
  ─────●────────────────────●────────────────────●─────
    Pleasure               0.0               Dominance
        ╲                   │                    ╱
          ╲                 │                  ╱
            ╲               │                ╱
              ╲             │              ╱
                ╲           │           ╱
                  ╲         │         ╱
                    ╲       │       ╱
                      ╲─────┼─────╱
                            │
                           -1.0
                            │
                    (Negative Pleasure)

    Filled triangle shows current state
    Can overlay multiple noodlings
```

**Features:**
- Three axes from center
- Current state forms a triangle
- Multiple noodlings as overlapping triangles
- Color-coded per noodling
- Good for comparison views

### 4. Temporal Spectrogram

Time on X axis, PAD values as stacked colored bands - like an audio spectrogram.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Pleasure  ██████████░░░░░░░░████████████████░░░░░░░░░░████████ │
│            ▓▓▓▓▓▓▓▓▓▓░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░▓▓▓▓▓▓▓▓ │
│                                                                 │
│  Arousal   ░░░░░░████████████░░░░░░░░████████████████░░░░░░░░░░ │
│            ░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░ │
│                                                                 │
│  Dominance ████████████░░░░░░░░░░░░████████░░░░░░░░████████████ │
│            ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░▓▓▓▓▓▓▓▓░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓ │
│                                                                 │
│            └────────────────────────────────────────────────────│
│             -60s        -40s        -20s         now            │
│                                                                 │
│  Events    ▼ User spoke    ▼ Noodling surprised    ▼ Resolved   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Time series view
- See emotional "weather patterns"
- Correlate with events (marked on timeline)
- Useful for session review/analysis

## Widget Implementation

### AffectSpectrometer Component

```python
class AffectSpectrometer(QMLWidgetWrapper):
    """
    Visualizes affect state in PAD space.

    Multiple visualization modes:
    - cube: 3D rotatable cube
    - chromaticity: 2D projection (like CIE diagram)
    - radar: Spider chart (good for comparison)
    - spectrogram: Time series view

    Properties:
    - mode: "cube" | "chromaticity" | "radar" | "spectrogram"
    - pleasure: float (-1 to 1)
    - arousal: float (0 to 1)
    - dominance: float (0 to 1)
    - show_trail: bool (show temporal history)
    - trail_length: int (seconds of history)
    - show_labels: bool (show emotion region labels)
    - show_grid: bool (show reference grid)
    - point_color: color (current state indicator)
    - trail_color: color (history trail)

    Channel bindings:
    - pleasure ← affect/valence (or affect/pleasure)
    - arousal ← affect/arousal
    - dominance ← affect/dominance
    """

    component_type = "AffectSpectrometer"
```

### QML Implementation (Cube Mode)

```qml
// affect_spectrometer_cube.qml
// @name: Affect Spectrometer (3D Cube)
// @category: Visualization / Affect
// @license: MIT

import QtQuick 2.15
import QtQuick3D 1.15
import QtQuick3D.Helpers 1.15

Item {
    id: root
    width: 300
    height: 300

    // Bindable properties
    property real pleasure: 0.0      // -1 to 1
    property real arousal: 0.5       // 0 to 1
    property real dominance: 0.5     // 0 to 1

    property bool showTrail: true
    property int trailLength: 30     // samples
    property bool showLabels: true
    property bool showGrid: true
    property color pointColor: "#e74c3c"
    property color trailColor: "#3498db"

    // Internal trail storage
    property var trailPoints: []

    View3D {
        anchors.fill: parent

        environment: SceneEnvironment {
            clearColor: "#1a1a2e"
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
        }

        // Camera with orbit controls
        PerspectiveCamera {
            id: camera
            position: Qt.vector3d(2.5, 2.5, 2.5)
            eulerRotation.x: -30
            eulerRotation.y: 45
        }

        OrbitCameraController {
            camera: camera
            origin: Qt.vector3d(0, 0, 0)
        }

        // Lighting
        DirectionalLight {
            eulerRotation.x: -30
            eulerRotation.y: -70
            ambientColor: "#333"
        }

        // The PAD Cube (wireframe)
        Model {
            id: cubeFrame
            source: "#Cube"
            scale: Qt.vector3d(1, 1, 1)
            materials: [
                PrincipledMaterial {
                    baseColor: "#ffffff"
                    opacity: 0.1
                    roughness: 1.0
                }
            ]
        }

        // Grid lines
        Repeater3D {
            model: root.showGrid ? 5 : 0
            Model {
                source: "#Rectangle"
                position: Qt.vector3d(0, index * 0.25 - 0.5, 0)
                scale: Qt.vector3d(1, 0.002, 1)
                materials: PrincipledMaterial {
                    baseColor: "#444"
                    opacity: 0.3
                }
            }
        }

        // Current state indicator (glowing sphere)
        Model {
            id: currentPoint
            source: "#Sphere"

            // Map PAD to cube coordinates
            // Pleasure: -1 to +1 → X: -0.5 to +0.5
            // Arousal: 0 to 1 → Y: -0.5 to +0.5
            // Dominance: 0 to 1 → Z: -0.5 to +0.5
            position: Qt.vector3d(
                root.pleasure * 0.5,
                (root.arousal - 0.5),
                (root.dominance - 0.5)
            )

            scale: Qt.vector3d(0.08, 0.08, 0.08)

            materials: [
                PrincipledMaterial {
                    baseColor: root.pointColor
                    emissiveColor: root.pointColor
                    emissiveFactor: 0.5
                }
            ]

            // Pulse animation
            SequentialAnimation on scale {
                loops: Animation.Infinite
                NumberAnimation {
                    to: 0.1
                    duration: 500
                    easing.type: Easing.InOutQuad
                }
                NumberAnimation {
                    to: 0.08
                    duration: 500
                    easing.type: Easing.InOutQuad
                }
            }
        }

        // Trail points
        Repeater3D {
            model: root.showTrail ? root.trailPoints.length : 0

            Model {
                source: "#Sphere"
                position: root.trailPoints[index].position
                scale: Qt.vector3d(0.02, 0.02, 0.02)
                opacity: (index + 1) / root.trailPoints.length * 0.6

                materials: PrincipledMaterial {
                    baseColor: root.trailColor
                    opacity: parent.opacity
                }
            }
        }

        // Axis labels (if enabled)
        // ... (text rendering in 3D)
    }

    // Corner labels (2D overlay)
    Item {
        anchors.fill: parent
        visible: root.showLabels

        Text {
            text: "Triumphant"
            color: "#2ecc71"
            font.pixelSize: 10
            anchors.top: parent.top
            anchors.right: parent.right
            anchors.margins: 8
        }

        Text {
            text: "Terrified"
            color: "#e74c3c"
            font.pixelSize: 10
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.margins: 8
        }

        Text {
            text: "Peaceful"
            color: "#3498db"
            font.pixelSize: 10
            anchors.bottom: parent.bottom
            anchors.right: parent.right
            anchors.margins: 8
        }

        Text {
            text: "Despairing"
            color: "#9b59b6"
            font.pixelSize: 10
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.margins: 8
        }
    }

    // Numeric readout
    Column {
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottomMargin: 8

        Text {
            text: "P: " + root.pleasure.toFixed(2) +
                  "  A: " + root.arousal.toFixed(2) +
                  "  D: " + root.dominance.toFixed(2)
            color: "#aaa"
            font.family: "monospace"
            font.pixelSize: 11
        }
    }

    // Trail update timer
    Timer {
        interval: 100  // 10 Hz sampling
        running: root.showTrail
        repeat: true
        onTriggered: {
            var newPoint = {
                position: Qt.vector3d(
                    root.pleasure * 0.5,
                    root.arousal - 0.5,
                    root.dominance - 0.5
                )
            };

            var trail = root.trailPoints.slice();
            trail.push(newPoint);

            // Limit trail length
            while (trail.length > root.trailLength) {
                trail.shift();
            }

            root.trailPoints = trail;
        }
    }
}
```

### QML Implementation (Chromaticity Mode)

```qml
// affect_spectrometer_2d.qml
// @name: Affect Spectrometer (2D Chromaticity)
// @category: Visualization / Affect

import QtQuick 2.15
import QtQuick.Shapes 1.15

Item {
    id: root
    width: 300
    height: 300

    property real pleasure: 0.0
    property real arousal: 0.5
    property real dominance: 0.5

    property bool showTrail: true
    property int trailLength: 100
    property bool showLabels: true
    property bool showRegions: true
    property color pointColor: "#e74c3c"
    property color trailColor: "#3498db"

    property var trailPoints: []

    // Background
    Rectangle {
        anchors.fill: parent
        color: "#1a1a2e"
        radius: 4
    }

    // Emotion region overlays (soft colored regions)
    Canvas {
        id: regionsCanvas
        anchors.fill: parent
        anchors.margins: 40
        visible: root.showRegions

        onPaint: {
            var ctx = getContext("2d");
            var w = width;
            var h = height;

            ctx.clearRect(0, 0, w, h);

            // Gradient regions for emotion spaces
            // Top-left: Angry (red)
            var gradient1 = ctx.createRadialGradient(0, 0, 0, 0, 0, w * 0.5);
            gradient1.addColorStop(0, "rgba(231, 76, 60, 0.3)");
            gradient1.addColorStop(1, "rgba(231, 76, 60, 0)");
            ctx.fillStyle = gradient1;
            ctx.fillRect(0, 0, w * 0.5, h * 0.5);

            // Top-right: Happy (green)
            var gradient2 = ctx.createRadialGradient(w, 0, 0, w, 0, w * 0.5);
            gradient2.addColorStop(0, "rgba(46, 204, 113, 0.3)");
            gradient2.addColorStop(1, "rgba(46, 204, 113, 0)");
            ctx.fillStyle = gradient2;
            ctx.fillRect(w * 0.5, 0, w * 0.5, h * 0.5);

            // Bottom-left: Sad (blue)
            var gradient3 = ctx.createRadialGradient(0, h, 0, 0, h, w * 0.5);
            gradient3.addColorStop(0, "rgba(52, 152, 219, 0.3)");
            gradient3.addColorStop(1, "rgba(52, 152, 219, 0)");
            ctx.fillStyle = gradient3;
            ctx.fillRect(0, h * 0.5, w * 0.5, h * 0.5);

            // Bottom-right: Peaceful (purple)
            var gradient4 = ctx.createRadialGradient(w, h, 0, w, h, w * 0.5);
            gradient4.addColorStop(0, "rgba(155, 89, 182, 0.3)");
            gradient4.addColorStop(1, "rgba(155, 89, 182, 0)");
            ctx.fillStyle = gradient4;
            ctx.fillRect(w * 0.5, h * 0.5, w * 0.5, h * 0.5);
        }
    }

    // Grid
    Canvas {
        id: gridCanvas
        anchors.fill: parent
        anchors.margins: 40

        onPaint: {
            var ctx = getContext("2d");
            var w = width;
            var h = height;

            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1;

            // Vertical grid lines
            for (var i = 0; i <= 4; i++) {
                var x = i * w / 4;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            }

            // Horizontal grid lines
            for (var j = 0; j <= 4; j++) {
                var y = j * h / 4;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }

            // Center crosshairs (stronger)
            ctx.strokeStyle = "#555";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(w / 2, 0);
            ctx.lineTo(w / 2, h);
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();
        }
    }

    // Trail
    Shape {
        anchors.fill: parent
        anchors.margins: 40
        visible: root.showTrail && root.trailPoints.length > 1

        ShapePath {
            strokeColor: root.trailColor
            strokeWidth: 2
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            joinStyle: ShapePath.RoundJoin

            PathPolyline {
                path: {
                    var points = [];
                    var w = root.width - 80;
                    var h = root.height - 80;

                    for (var i = 0; i < root.trailPoints.length; i++) {
                        var pt = root.trailPoints[i];
                        points.push(Qt.point(
                            (pt.p + 1) / 2 * w,  // Pleasure -1..1 → 0..w
                            (1 - pt.a) * h        // Arousal 0..1 → h..0 (flip Y)
                        ));
                    }
                    return points;
                }
            }
        }
    }

    // Current point
    Rectangle {
        id: currentPoint
        width: 12 + root.dominance * 12  // Size encodes dominance
        height: width
        radius: width / 2
        color: root.pointColor

        x: 40 + (root.pleasure + 1) / 2 * (parent.width - 80) - width / 2
        y: 40 + (1 - root.arousal) * (parent.height - 80) - height / 2

        // Glow effect
        Rectangle {
            anchors.centerIn: parent
            width: parent.width * 2
            height: width
            radius: width / 2
            color: root.pointColor
            opacity: 0.3
        }

        // Pulse
        SequentialAnimation on scale {
            loops: Animation.Infinite
            NumberAnimation { to: 1.2; duration: 500 }
            NumberAnimation { to: 1.0; duration: 500 }
        }
    }

    // Axis labels
    Text {
        text: "Pleasure →"
        color: "#888"
        font.pixelSize: 10
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottomMargin: 8
    }

    Text {
        text: "← Arousal"
        color: "#888"
        font.pixelSize: 10
        rotation: -90
        anchors.left: parent.left
        anchors.verticalCenter: parent.verticalCenter
        anchors.leftMargin: 8
    }

    // Corner emotion labels
    Text {
        visible: root.showLabels
        text: "Angry"
        color: "#e74c3c"
        font.pixelSize: 9
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.margins: 44
    }

    Text {
        visible: root.showLabels
        text: "Excited"
        color: "#2ecc71"
        font.pixelSize: 9
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 44
    }

    Text {
        visible: root.showLabels
        text: "Sad"
        color: "#3498db"
        font.pixelSize: 9
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.margins: 44
    }

    Text {
        visible: root.showLabels
        text: "Calm"
        color: "#9b59b6"
        font.pixelSize: 9
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        anchors.margins: 44
    }

    // Dominance indicator (separate readout since it's encoded in size)
    Row {
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 8
        spacing: 4

        Text {
            text: "D:"
            color: "#888"
            font.pixelSize: 10
        }

        Rectangle {
            width: 40
            height: 8
            color: "#333"
            radius: 2

            Rectangle {
                width: parent.width * root.dominance
                height: parent.height
                color: "#f39c12"
                radius: 2
            }
        }
    }

    // Trail update
    Timer {
        interval: 50
        running: root.showTrail
        repeat: true
        onTriggered: {
            var trail = root.trailPoints.slice();
            trail.push({
                p: root.pleasure,
                a: root.arousal,
                d: root.dominance
            });

            while (trail.length > root.trailLength) {
                trail.shift();
            }

            root.trailPoints = trail;
        }
    }
}
```

## Usage in UI

### Basic Usage

```yaml
# In ui.yaml
- type: QMLWidget
  qml_source: "qml/widgets/affect_spectrometer_2d.qml"
  width: 300
  height: 300
  properties:
    showTrail: true
    showLabels: true
    showRegions: true
  bindings:
    pleasure: "affect/valence"
    arousal: "affect/arousal"
    dominance: "affect/dominance"
```

### Multi-Noodling Comparison

```yaml
# Compare two noodlings' affect
- type: Panel
  layout: horizontal
  children:
    - type: QMLWidget
      qml_source: "qml/widgets/affect_spectrometer_2d.qml"
      properties:
        pointColor: "#e74c3c"  # Red for noodling A
      bindings:
        pleasure: "noodling_a/affect/valence"
        arousal: "noodling_a/affect/arousal"
        dominance: "noodling_a/affect/dominance"

    - type: QMLWidget
      qml_source: "qml/widgets/affect_spectrometer_2d.qml"
      properties:
        pointColor: "#3498db"  # Blue for noodling B
      bindings:
        pleasure: "noodling_b/affect/valence"
        arousal: "noodling_b/affect/arousal"
        dominance: "noodling_b/affect/dominance"
```

### With Full Dashboard

```yaml
# Complete affect monitoring dashboard
- type: Panel
  layout: vertical
  children:
    # Main spectrometer
    - type: QMLWidget
      qml_source: "qml/widgets/affect_spectrometer_cube.qml"
      flex: 1
      bindings:
        pleasure: "affect/valence"
        arousal: "affect/arousal"
        dominance: "affect/dominance"

    # Individual dimension meters
    - type: Panel
      layout: horizontal
      height: 100
      children:
        - type: QMLWidget
          qml_source: "qml/widgets/vu_meter.qml"
          properties:
            label: "Pleasure"
            min: -1
            max: 1
          bindings:
            value: "affect/valence"

        - type: QMLWidget
          qml_source: "qml/widgets/vu_meter.qml"
          properties:
            label: "Arousal"
          bindings:
            value: "affect/arousal"

        - type: QMLWidget
          qml_source: "qml/widgets/vu_meter.qml"
          properties:
            label: "Dominance"
          bindings:
            value: "affect/dominance"
```

## Advanced Features

### Emotion Attractor Basins

Show regions where emotions tend to "settle" - like gravity wells in the affect space.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│        ╭───╮                          ╭───╮                     │
│       ╱     ╲   ANGRY                ╱     ╲   EXCITED          │
│      │   ◉   │                      │   ◉   │                   │
│       ╲     ╱                        ╲     ╱                    │
│        ╰───╯                          ╰───╯                     │
│                         ●                                       │
│                       ╱   ╲                                     │
│                     ╱       ╲  ← Ball rolling toward basin     │
│                   ╱           ╲                                 │
│        ╭───╮    ╱               ╲    ╭───╮                      │
│       ╱     ╲                        ╱     ╲                    │
│      │   ◉   │   SAD            CALM│   ◉   │                   │
│       ╲     ╱                        ╲     ╱                    │
│        ╰───╯                          ╰───╯                     │
│                                                                 │
│  Attractor basins show where emotions naturally gravitate      │
│  Useful for understanding emotional dynamics                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Velocity Vectors

Show not just WHERE the emotion is, but WHERE IT'S HEADING.

```python
# In the widget, compute velocity from recent trail
dx = trail[-1].p - trail[-5].p
dy = trail[-1].a - trail[-5].a
velocity = (dx, dy)

# Draw arrow from current point in direction of velocity
```

### Emotional Weather Forecast

Predict where affect is likely to go based on current trajectory and known attractor dynamics.

```
Current: Anxious (P: -0.3, A: 0.8, D: 0.2)
Velocity: Calming ↓↘
Forecast: Likely to settle into Sad or Calm within 30s
```

## Kimii-Sensei Teaching Mode

The Affect Spectrometer becomes a teaching tool:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  🦎 KIMII-SENSEI: "Let's explore the emotion space!"           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │   Try moving the sliders to explore different emotions! │   │
│  │                                                         │   │
│  │              [AFFECT SPECTROMETER]                      │   │
│  │                    ●───→                                │   │
│  │                  ╱                                      │   │
│  │                ╱  "You're getting happier               │   │
│  │              ╱     and more excited!"                   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Pleasure: ──────────●──────── (+0.6)                          │
│  Arousal:  ────────────●────── (+0.7)                          │
│  Dominance: ─────●─────────── (+0.3)                           │
│                                                                 │
│  Current emotion: EXCITED (close to TRIUMPHANT)                │
│                                                                 │
│  [Challenge: Find CALM]  [Challenge: Find ANGRY]               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Kids can:
1. Use sliders to explore the space
2. Try to hit specific emotion targets
3. Watch the trail as they move through affect space
4. Learn the relationship between PAD values and named emotions

## Implementation Priority

1. **2D Chromaticity View** - Simplest, most immediately useful
2. **Trail System** - Shows temporal dynamics
3. **Region Labels** - Makes it interpretable
4. **3D Cube View** - Advanced/optional
5. **Radar View** - For comparison dashboards
6. **Spectrogram View** - For research/analysis

## References

- Mehrabian, A. & Russell, J.A. (1974). *An Approach to Environmental Psychology*
- CIE 1931 Color Space - inspiration for 2D projection approach
- Russell's Circumplex Model of Affect (2D precursor to PAD)

---

*"To understand emotions, first we must see them."*

Made with love by Caity & Claude
