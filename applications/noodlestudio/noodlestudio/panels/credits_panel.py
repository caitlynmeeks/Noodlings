"""
Demo Scene Style Credits

Horizontal scrolling credits with music.
ESC to exit, just like the old demos.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QFont, QColor, QPaintEvent, QKeyEvent
import random


class TextOverlay(QWidget):
    """Transparent overlay for scrolling credits text."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_credits = parent

        # Make transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event: QPaintEvent):
        """Draw the scrolling credits text."""
        if not hasattr(self.parent_credits, 'credits_lines'):
            return

        painter = QPainter(self)

        # BIGGER fonts for demo scene energy
        title_font = QFont("Courier New", 32, QFont.Weight.Bold)
        item_font = QFont("Courier New", 24)

        # Starting position
        x = self.parent_credits.scroll_x
        y = self.height() // 2  # Vertical center

        # Draw each line
        for line_type, text in self.parent_credits.credits_lines:
            if line_type == "title":
                painter.setFont(title_font)
                painter.setPen(QColor(255, 255, 255))  # White
            elif line_type == "item":
                painter.setFont(item_font)
                painter.setPen(QColor(200, 200, 200))  # Light gray
            else:  # spacer
                x += 40  # Add space
                continue

            # Draw text with glow effect
            painter.drawText(x, y, text)

            # Move x position - BIGGER spacing
            x += len(text) * 20 + 150  # Demo scene spacing

        painter.end()


class CreditsPanel(QWidget):
    """
    Full-screen scrolling credits with demo scene vibes.

    - Shader background (plasma/tunnel/starfield)
    - Horizontal scroll
    - Random music track
    - ESC to exit
    - Made with love
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Credits - Demo Scene Style")

        # Create layout with shader background
        layout = QStackedLayout(self)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)

        # Add shader widget as background
        try:
            from .shader_widget import create_shader_background
            self.shader_bg = create_shader_background(self)
            layout.addWidget(self.shader_bg)
            print("Shader background loaded!")
        except Exception as e:
            print(f"Shader background failed: {e}")
            self.setStyleSheet("background-color: #000000;")

        # Add transparent text overlay
        self.text_overlay = TextOverlay(self)
        layout.addWidget(self.text_overlay)

        # Scroll position (moved to overlay)
        self.scroll_x = self.width()
        self.scroll_speed = 2  # pixels per frame

        # Load credits
        from ..core.credits_data import CREDITS_SECTIONS, MUSIC_TRACKS
        self.sections = CREDITS_SECTIONS

        # Pick random music (conceptual)
        self.current_track = random.choice(MUSIC_TRACKS)

        # Build full credits text with spacing
        self.credits_lines = []
        for section in self.sections:
            # Title
            self.credits_lines.append(("title", section["title"]))
            self.credits_lines.append(("spacer", ""))

            # Items
            for item in section["items"]:
                self.credits_lines.append(("item", item))

            # Big spacer between sections
            for _ in range(5):
                self.credits_lines.append(("spacer", ""))

        # Add final message
        self.credits_lines.append(("title", "* * *  T H A N K  Y O U  * * *"))
        self.credits_lines.append(("spacer", ""))
        self.credits_lines.append(("item", "Press ESC to exit"))

        # Add lots of space at end so it scrolls off
        for _ in range(20):
            self.credits_lines.append(("spacer", ""))

        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_scroll)
        self.timer.start(16)  # ~60 FPS

        # Show music info
        self.show_music_info()

    def show_music_info(self):
        """Show what track is 'playing' and try to play it."""
        print(f"\n=== NOW PLAYING ===")
        print(f"Track: {self.current_track['name']}")
        print(f"Vibe: {self.current_track['description']}")
        print(f"Mood: {self.current_track['mood']}")

        # Try to play a MIDI file if available
        from ..core.midi_primitives import get_available_music, play_midi_file
        available = get_available_music()

        if available:
            midi_file = random.choice(available)
            print(f"Playing: {midi_file}")
            if play_midi_file(midi_file):
                print("Music started!")
            else:
                print("(pygame not installed - run: pip install pygame)")
        else:
            print(f"(Add .mid files to ~/.noodlestudio/music/ for music!)")
            print(f"Suggestion: Army of Me by Bjork")

        print(f"===================\n")

    def update_scroll(self):
        """Update scroll position and trigger repaint."""
        self.scroll_x -= self.scroll_speed

        # Calculate total width needed
        total_width = len(self.credits_lines) * 40  # rough estimate

        # Reset if scrolled past end
        if self.scroll_x < -total_width:
            self.scroll_x = self.width()

        # Update the text overlay
        if hasattr(self, 'text_overlay'):
            self.text_overlay.update()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key presses - ESC to exit."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Stop timer when closing."""
        self.timer.stop()
        super().closeEvent(event)


def show_credits(parent=None):
    """Show the credits screen as a floating window."""
    from PyQt6.QtCore import Qt

    credits = CreditsPanel(parent)

    # Make it a proper window
    credits.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)

    # Set size
    credits.resize(1024, 768)

    # Center on screen
    if parent:
        # Center relative to parent
        parent_geo = parent.geometry()
        x = parent_geo.x() + (parent_geo.width() - 1024) // 2
        y = parent_geo.y() + (parent_geo.height() - 768) // 2
        credits.move(x, y)
    else:
        # Center on screen
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - 1024) // 2
        y = (screen.height() - 768) // 2
        credits.move(x, y)

    credits.show()
    return credits
