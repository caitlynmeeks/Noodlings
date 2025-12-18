import QtQuick
import QtQuick3D
import QtQuick3D.Helpers

Rectangle {
    id: root
    color: "#1E1E1E"
    focus: true

    // Camera state - top-down by default
    property real cameraDistance: 250
    property real cameraYaw: 0
    property real cameraPitch: -90  // Top-down view
    property vector3d cameraTarget: Qt.vector3d(0, 0, -80)

    // Render mode: false = solid, true = wireframe
    property bool wireframeMode: true  // Default to wireframe
    // Ghost mode: shows backfaces (no occlusion), everything translucent
    property bool ghostMode: false

    // Limits from Python (fallbacks if not set)
    property real minDistance: minZoomDistance || 5
    property real maxDistance: maxZoomDistance || 20000
    property real nearClip: cameraNearClip || 0.1
    property real farClip: cameraFarClip || 50000

    // 3D View - FAST, CHEAP rendering
    View3D {
        id: view3d
        anchors.fill: parent

        environment: SceneEnvironment {
            clearColor: "#1A1A1A"
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.NoAA

            // Actual wireframe rendering - shows triangle edges
            debugSettings: DebugSettings {
                wireframeEnabled: wireframeMode
            }
        }

        // Orbit camera
        PerspectiveCamera {
            id: camera
            position: updateCameraPosition()
            eulerRotation: Qt.vector3d(cameraPitch, cameraYaw, 0)
            clipFar: farClip
            clipNear: nearClip

            function updateCameraPosition() {
                var yawRad = cameraYaw * Math.PI / 180
                var pitchRad = cameraPitch * Math.PI / 180

                return Qt.vector3d(
                    cameraTarget.x + cameraDistance * Math.cos(pitchRad) * Math.sin(yawRad),
                    cameraTarget.y + cameraDistance * Math.sin(-pitchRad),
                    cameraTarget.z + cameraDistance * Math.cos(pitchRad) * Math.cos(yawRad)
                )
            }
        }

        // Single light
        DirectionalLight {
            eulerRotation.x: -35
            eulerRotation.y: -25
            color: "#FFFFFF"
            brightness: 1.0
            ambientColor: "#404040"
        }

        // Ground plane
        Model {
            visible: !ghostMode
            source: "#Rectangle"
            scale: Qt.vector3d(50, 50, 1)
            eulerRotation.x: -90
            y: -1
            materials: DefaultMaterial {
                diffuseColor: "#252525"
                lighting: DefaultMaterial.NoLighting
            }
        }

        // Zone boxes - simple cubes
        Repeater3D {
            id: zoneRepeater
            model: zoneModel ? zoneModel.getZones() : []

            delegate: Node {
                id: zoneNode
                property var zoneData: modelData
                property real s: zoneData.radius
                position: Qt.vector3d(zoneData.x, zoneData.y, zoneData.z)

                // Zone cube
                Model {
                    source: "#Cube"
                    scale: Qt.vector3d(s, s, s)
                    materials: DefaultMaterial {
                        // Wireframe mode: gray so lines stand out
                        // Solid mode: use zone color
                        diffuseColor: wireframeMode ? "#444444" : (zoneData.selected ? "#FFFFFF" : zoneData.color)
                        opacity: ghostMode ? 0.2 : (zoneData.selected ? 1.0 : 0.8)
                        lighting: DefaultMaterial.NoLighting
                    }
                }

                // Selection highlight - yellow glow
                Model {
                    visible: zoneData.selected
                    source: "#Cube"
                    scale: Qt.vector3d(s * 1.12, s * 1.12, s * 1.12)
                    materials: DefaultMaterial {
                        diffuseColor: "#FFFF00"
                        opacity: 0.3
                        lighting: DefaultMaterial.NoLighting
                    }
                }
            }
        }
    }

    // 2D overlay for connection lines - DISABLED (projection math needs work)
    // TODO: Fix 3D-to-2D projection or use 3D line geometry instead
    Canvas {
        id: connectionCanvas
        anchors.fill: parent
        visible: false  // Hidden until projection is fixed

        property var connections: zoneModel ? zoneModel.getConnections() : []

        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
        }

        Connections {
            target: zoneModel
            function onZonesChanged() {
                connectionCanvas.connections = zoneModel.getConnections()
            }
        }
    }

    // Mouse interaction - Unity/Maya style controls
    // Alt+LMB: Tumble (orbit)
    // Alt+MMB: Track (pan)
    // Alt+RMB: Dolly (zoom)
    // Scroll: Zoom
    // LMB (no modifier): Select
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton | Qt.MiddleButton
        hoverEnabled: true

        property point lastPos
        property bool dragging: false

        onPressed: function(mouse) {
            lastPos = Qt.point(mouse.x, mouse.y)
            dragging = false
            root.forceActiveFocus()
        }

        onReleased: function(mouse) {
            // Only select if not dragging and no Alt modifier
            if (!dragging && mouse.button === Qt.LeftButton && !(mouse.modifiers & Qt.AltModifier)) {
                pickZoneAt(mouse.x, mouse.y)
            }
            dragging = false
        }

        onPositionChanged: function(mouse) {
            if (!(mouse.buttons & (Qt.LeftButton | Qt.RightButton | Qt.MiddleButton))) return

            var dx = mouse.x - lastPos.x
            var dy = mouse.y - lastPos.y

            if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
                dragging = true
            }

            lastPos = Qt.point(mouse.x, mouse.y)

            var hasAlt = mouse.modifiers & Qt.AltModifier

            // Alt + LMB: Tumble (orbit)
            if ((mouse.buttons & Qt.LeftButton) && hasAlt) {
                cameraYaw += dx * 0.5
                cameraPitch = Math.max(-89.9, Math.min(89.9, cameraPitch - dy * 0.5))
                camera.position = camera.updateCameraPosition()
                // connectionCanvas.requestPaint() - disabled
            }
            // Alt + MMB: Track (pan)
            else if ((mouse.buttons & Qt.MiddleButton) && hasAlt) {
                panCamera(dx, dy)
            }
            // MMB without Alt: Also pan (convenience)
            else if ((mouse.buttons & Qt.MiddleButton) && !hasAlt) {
                panCamera(dx, dy)
            }
            // Alt + RMB: Dolly (zoom)
            else if ((mouse.buttons & Qt.RightButton) && hasAlt) {
                var dollyAmount = (dx + dy) * 0.5
                var factor = dollyAmount > 0 ? 1.02 : 0.98
                cameraDistance = Math.max(minDistance, Math.min(maxDistance, cameraDistance * Math.pow(factor, Math.abs(dollyAmount))))
                camera.position = camera.updateCameraPosition()
                // connectionCanvas.requestPaint() - disabled
            }
            // RMB without Alt: Context menu handled by Qt (no drag action)
        }

        function panCamera(dx, dy) {
            var panSpeed = cameraDistance * 0.002
            var yawRad = cameraYaw * Math.PI / 180
            var pitchRad = cameraPitch * Math.PI / 180

            // Calculate right and up vectors based on camera orientation
            var rightX = Math.cos(yawRad)
            var rightZ = -Math.sin(yawRad)

            // For top-down view, up is forward (Z axis)
            // For perspective view, up is Y axis mixed with forward
            var upX, upY, upZ
            if (Math.abs(cameraPitch) > 80) {
                // Near top-down: up maps to Z movement
                upX = Math.sin(yawRad) * Math.sign(cameraPitch)
                upY = 0
                upZ = Math.cos(yawRad) * Math.sign(cameraPitch)
            } else {
                // Perspective: up maps to Y with some forward
                upX = 0
                upY = 1
                upZ = 0
            }

            cameraTarget.x -= dx * rightX * panSpeed
            cameraTarget.z -= dx * rightZ * panSpeed
            cameraTarget.x -= dy * upX * panSpeed
            cameraTarget.y += dy * upY * panSpeed
            cameraTarget.z -= dy * upZ * panSpeed

            camera.position = camera.updateCameraPosition()
            // connectionCanvas.requestPaint() - disabled
        }

        onWheel: function(wheel) {
            var factor = wheel.angleDelta.y > 0 ? 0.9 : 1.1
            cameraDistance = Math.max(minDistance, Math.min(maxDistance, cameraDistance * factor))
            camera.position = camera.updateCameraPosition()
            // connectionCanvas.requestPaint() - disabled
        }

        function pickZoneAt(screenX, screenY) {
            var zones = zoneModel ? zoneModel.getZones() : []
            var closestZone = null
            var closestDist = 40

            for (var i = 0; i < zones.length; i++) {
                var zone = zones[i]
                var screenPos = connectionCanvas.project3DTo2D(zone.x, zone.y, zone.z)
                if (!screenPos) continue

                var dist = Math.sqrt(
                    Math.pow(screenX - screenPos.x, 2) +
                    Math.pow(screenY - screenPos.y, 2)
                )
                if (dist < closestDist) {
                    closestDist = dist
                    closestZone = zone
                }
            }

            if (closestZone && zoneModel) {
                zoneModel.selectZone(closestZone.id)
            }
        }
    }

    // Keyboard handling
    Keys.onPressed: function(event) {
        if (event.key === Qt.Key_F) {
            focusSelected()
            event.accepted = true
        } else if (event.key === Qt.Key_A) {
            fitAll()
            event.accepted = true
        } else if (event.key === Qt.Key_W) {
            toggleWireframe()
            event.accepted = true
        } else if (event.key === Qt.Key_T) {
            toggleGhost()
            event.accepted = true
        }
    }

    // Render mode indicator
    Row {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.margins: 8
        spacing: 6

        Rectangle {
            width: modeText.width + 12
            height: 20
            radius: 3
            color: "#2A2A2A"

            Text {
                id: modeText
                anchors.centerIn: parent
                text: wireframeMode ? "WIRE" : "SOLID"
                color: wireframeMode ? "#88AAFF" : "#888888"
                font.pixelSize: 10
                font.bold: true
            }
        }

        Rectangle {
            visible: ghostMode
            width: ghostText.width + 12
            height: 20
            radius: 3
            color: "#2A2A2A"

            Text {
                id: ghostText
                anchors.centerIn: parent
                text: "X-RAY"
                color: "#88FF88"
                font.pixelSize: 10
                font.bold: true
            }
        }
    }

    // Platform-specific modifier key name
    property string modKey: Qt.platform.os === "osx" ? "⌥" : "Alt"

    // Help text
    Text {
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        anchors.margins: 8
        text: "W: Wireframe | T: Ghost | " + modKey + "+LMB: Tumble | " + modKey + "+MMB: Track | Scroll: Dolly | A: Top"
        color: "#555555"
        font.pixelSize: 10
    }

    // Model update listener
    Connections {
        target: zoneModel
        function onZonesChanged() {
            zoneRepeater.model = zoneModel.getZones()
        }
    }

    // API functions callable from Python
    function resetView() {
        cameraYaw = 0
        cameraPitch = -25
        cameraDistance = 150
        cameraTarget = Qt.vector3d(0, 0, 0)
        camera.position = camera.updateCameraPosition()
        // connectionCanvas.requestPaint() - disabled
    }

    function fitAll() {
        var zones = zoneModel ? zoneModel.getZones() : []
        if (zones.length === 0) {
            resetView()
            return
        }

        var minX = Infinity, maxX = -Infinity
        var minY = Infinity, maxY = -Infinity
        var minZ = Infinity, maxZ = -Infinity

        for (var i = 0; i < zones.length; i++) {
            var z = zones[i]
            var r = z.radius
            minX = Math.min(minX, z.x - r)
            maxX = Math.max(maxX, z.x + r)
            minY = Math.min(minY, z.y - r)
            maxY = Math.max(maxY, z.y + r)
            minZ = Math.min(minZ, z.z - r)
            maxZ = Math.max(maxZ, z.z + r)
        }

        var centerX = (minX + maxX) / 2
        var centerY = (minY + maxY) / 2
        var centerZ = (minZ + maxZ) / 2

        // For top-down view, extent is max of X and Z spans
        var extentX = maxX - minX
        var extentZ = maxZ - minZ
        var extent = Math.max(extentX, extentZ)

        // Calculate distance needed to see the whole scene (rough FOV calculation)
        // For 45 degree FOV, distance ~= extent / tan(22.5) ~= extent * 2.4
        var neededDistance = extent * 1.2

        cameraTarget = Qt.vector3d(centerX, centerY, centerZ)
        cameraDistance = Math.max(minDistance, Math.min(maxDistance, neededDistance))
        cameraYaw = 0
        cameraPitch = -90  // Snap to top-down view
        camera.position = camera.updateCameraPosition()
        // connectionCanvas.requestPaint() - disabled

        console.log("fitAll: center=(" + centerX + "," + centerY + "," + centerZ + ") extent=" + extent + " dist=" + cameraDistance)
    }

    function toggleWireframe() {
        wireframeMode = !wireframeMode
    }

    function toggleGhost() {
        ghostMode = !ghostMode
    }

    function focusSelected() {
        var zones = zoneModel ? zoneModel.getZones() : []
        for (var i = 0; i < zones.length; i++) {
            if (zones[i].selected) {
                var z = zones[i]
                cameraTarget = Qt.vector3d(z.x, z.y, z.z)
                cameraDistance = Math.max(minDistance, Math.min(maxDistance, z.radius * 5))
                camera.position = camera.updateCameraPosition()
                // connectionCanvas.requestPaint() - disabled
                return
            }
        }
    }

    function focusOnZone(zoneId) {
        var zones = zoneModel ? zoneModel.getZones() : []
        for (var i = 0; i < zones.length; i++) {
            if (zones[i].id === zoneId) {
                var z = zones[i]
                cameraTarget = Qt.vector3d(z.x, z.y, z.z)
                cameraDistance = Math.max(minDistance, Math.min(maxDistance, z.radius * 5))
                camera.position = camera.updateCameraPosition()
                // connectionCanvas.requestPaint() - disabled
                return
            }
        }
    }

    // Auto-fit on first load
    Component.onCompleted: {
        Qt.callLater(fitAll)
    }
}
