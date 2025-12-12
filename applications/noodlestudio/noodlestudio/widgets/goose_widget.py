"""
Goose Widget - The legendary gooseware overlay.

Features:
- Sprite-based animation (walk, flap, honk)
- Positional audio (honking gets louder as goose approaches)
- Random wandering behavior
- Triggered by: Konami code, Ctrl+Shift+G, or degoosification button
- Defeated by: Valid degoosification code

Origin Story: This is where Noodlings began - a year ago with ChatGPT conversation
downloader and a React nightmare. The goose persists. The goose endures.
"""

import random
from pathlib import Path
from PyQt6.QtWidgets import QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QSettings, QObject, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QTransform
from PyQt6.QtMultimedia import QSoundEffect, QAudioOutput
from PyQt6.QtCore import QUrl


class GooseWidget(QWidget):
    """
    Overlay widget that displays the animated goose walking across the screen.

    The goose:
    - Enters from one side
    - Walks with animated sprite
    - Stops occasionally to flap wings
    - Honks (with increasing volume as it approaches center)
    - Exits on the other side
    - Destroys itself when animation completes
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("Noodlings", "NoodleStudio")

        # Make transparent overlay (goose walks on top of everything!)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

        # Set transparent background
        self.setStyleSheet("background: transparent;")

        # Raise above siblings but don't use window flags (simpler approach)
        self.raise_()

        # Load goose assets
        assets_dir = Path.home() / "git" / "goose assets" / "assets"
        self.sprite_sheet = QPixmap(str(assets_dir / "goose-sheet-haha.png"))

        # Sprite sheet is 3x3 grid (1024x1024 total, so each frame is ~341x341)
        self.frame_size = 341
        self.frames = self._extract_frames()

        # Animation state
        self.current_frame = 0
        self.position = QPointF(-self.frame_size, 0)  # Start off-screen left
        self.direction = 1  # 1 = right, -1 = left
        self.state = "walking"  # walking, flapping, honking
        self.walk_speed = 5  # Faster waddle!
        self.tilt_angle = 0  # For South Park-style waddle
        self.tilt_direction = 1  # Oscillates +/-

        # Animation sequences (indices into sprite sheet) - Caitlyn's analysis!
        self.walk_cycle = [5, 6, 7]  # Waddle waddle
        self.flap_cycle = [5, 8, 4, 1, 2, 3, 1, 4]  # Dramatic wing sequence
        self.honk_cycle = [6, 5, 4, 1]  # Honking poses

        # Behavior timers
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._animate)
        self.anim_timer.start(100)  # 10 FPS animation

        self.behavior_timer = QTimer()
        self.behavior_timer.timeout.connect(self._change_behavior)
        self.behavior_timer.start(2000)  # Change behavior every 2 seconds

        # Audio setup
        self.honk_sound = QSoundEffect()
        honk_file = assets_dir / "goose-honks-haha- tyvm 164484__deleted_user_2104797__geese.ogg"
        if honk_file.exists():
            self.honk_sound.setSource(QUrl.fromLocalFile(str(honk_file)))
            self.honk_sound.setVolume(0.3)

        self.next_honk_time = random.randint(50, 100)  # Frames until next honk
        self.honk_countdown = self.next_honk_time

        # Position over parent
        if parent:
            self.setGeometry(parent.rect())

    def _extract_frames(self):
        """Extract individual goose frames from sprite sheet."""
        frames = []
        for row in range(3):
            for col in range(3):
                x = col * self.frame_size
                y = row * self.frame_size
                frame = self.sprite_sheet.copy(x, y, self.frame_size, self.frame_size)
                frames.append(frame)
        return frames

    def _animate(self):
        """Update animation frame and position."""
        # Move goose
        if self.state == "walking":
            self.position.setX(self.position.x() + self.walk_speed * self.direction)
            # South Park-style waddle (tilt back and forth)
            self.tilt_angle += 0.8 * self.tilt_direction
            if abs(self.tilt_angle) > 8:  # Max tilt 8 degrees
                self.tilt_direction *= -1

        # Update frame based on current state
        if self.state == "walking":
            cycle = self.walk_cycle
        elif self.state == "flapping":
            cycle = self.flap_cycle
        else:  # honking
            cycle = self.honk_cycle

        self.current_frame = (self.current_frame + 1) % len(cycle)

        # Check if goose has left the screen
        if self.position.x() > self.width() + self.frame_size:
            self.anim_timer.stop()
            self.behavior_timer.stop()
            self.deleteLater()
            return

        # Honk countdown
        self.honk_countdown -= 1
        if self.honk_countdown <= 0:
            self._honk()
            self.honk_countdown = random.randint(30, 80)

        # Update volume based on position (louder in center)
        center_x = self.width() / 2
        distance = abs(self.position.x() - center_x)
        max_distance = self.width() / 2
        volume = max(0.1, 1.0 - (distance / max_distance))
        self.honk_sound.setVolume(volume * 0.5)  # Scale down a bit

        self.update()

    def _change_behavior(self):
        """Randomly change goose behavior."""
        behaviors = ["walking", "flapping", "honking"]
        # Bias towards walking (70% of the time)
        weights = [0.7, 0.15, 0.15]
        self.state = random.choices(behaviors, weights=weights)[0]
        self.current_frame = 0

    def _honk(self):
        """Play honking sound."""
        if self.honk_sound.isPlaying():
            self.honk_sound.stop()
        self.honk_sound.play()

    def paintEvent(self, event):
        """Draw the goose at current position."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Clear background (transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

        # Get current frame based on state
        if self.state == "walking":
            frame_idx = self.walk_cycle[self.current_frame % len(self.walk_cycle)]
        elif self.state == "flapping":
            frame_idx = self.flap_cycle[self.current_frame % len(self.flap_cycle)]
        else:
            frame_idx = self.honk_cycle[self.current_frame % len(self.honk_cycle)]

        frame = self.frames[frame_idx]

        # Apply transformations (flip + South Park waddle)
        transform = QTransform()

        # Flip horizontally if walking left
        if self.direction == -1:
            transform.scale(-1, 1)

        # South Park-style waddle tilt (pivot at bottom center of sprite)
        if self.state == "walking":
            # Translate to pivot point (bottom center), rotate, translate back
            pivot_x = self.frame_size / 2
            pivot_y = self.frame_size
            transform.translate(pivot_x, pivot_y)
            transform.rotate(self.tilt_angle)
            transform.translate(-pivot_x, -pivot_y)

        frame = frame.transformed(transform, Qt.TransformationMode.SmoothTransformation)

        # Draw at current position
        y_pos = self.height() - self.frame_size - 50  # Near bottom of screen
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawPixmap(int(self.position.x()), int(y_pos), frame)

        painter.end()


class KonamiCodeDetector(QObject):
    """
    Detects the Konami code: ↑↑↓↓←→←→

    Emits goose_summoned signal when sequence is entered correctly.
    """

    goose_summoned = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.sequence = []
        self.konami = [
            Qt.Key.Key_Up, Qt.Key.Key_Up,
            Qt.Key.Key_Down, Qt.Key.Key_Down,
            Qt.Key.Key_Left, Qt.Key.Key_Right,
            Qt.Key.Key_Left, Qt.Key.Key_Right
        ]
        self.reset_timer = QTimer()
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self._reset)

    def key_pressed(self, key):
        """Process a key press for Konami code detection."""
        self.sequence.append(key)

        # Keep only last 8 keys (length of Konami code)
        if len(self.sequence) > len(self.konami):
            self.sequence.pop(0)

        # Check if we have the Konami code
        if self.sequence == self.konami:
            self.goose_summoned.emit()
            self._reset()

        # Reset sequence after 2 seconds of no input
        self.reset_timer.stop()
        self.reset_timer.start(2000)

    def _reset(self):
        """Reset the key sequence."""
        self.sequence = []
