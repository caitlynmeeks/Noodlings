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
#   View Project Button
#
#   The button that reveals NoodleStudio from within a published app.
#   Subtle but discoverable. The door to the studio.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.widgets.view_project_button
# PURPOSE:  View Project Button for App Mode
# LAYER:    Studio / Widgets
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ViewProjectButton
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Optional

from PyQt6.QtWidgets import QPushButton, QWidget, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QFont


class ViewProjectButton(QPushButton):
    """
    The "View Project" button shown in App Mode.

    When clicked, triggers the unfold animation to reveal NoodleStudio.
    Positioned at the bottom center of the window. Subtle but discoverable.

    Features:
    - Hover effect (slight glow)
    - Fade in/out animations
    - Customizable label text

    Usage:
        button = ViewProjectButton(parent_window)
        button.clicked.connect(fold_manager.unfold)
        button.show()
    """

    def __init__(self, parent: Optional[QWidget] = None, label: str = "View Project"):
        """
        Initialize the button.

        Args:
            parent: Parent widget
            label: Button text (default "View Project")
        """
        super().__init__(label, parent)

        self._setup_style()
        self._setup_opacity()

    def _setup_style(self):
        """Configure button appearance."""
        self.setFont(QFont("-apple-system, BlinkMacSystemFont, sans-serif", 11))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)
        self.setMinimumWidth(120)

        self.setStyleSheet("""
            QPushButton {
                background: rgba(60, 60, 60, 0.9);
                color: #cccccc;
                border: 1px solid rgba(100, 100, 100, 0.5);
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background: rgba(80, 80, 80, 0.95);
                color: #ffffff;
                border: 1px solid rgba(130, 130, 130, 0.7);
            }
            QPushButton:pressed {
                background: rgba(50, 50, 50, 1.0);
                color: #dddddd;
            }
        """)

    def _setup_opacity(self):
        """Set up opacity effect for fade animations."""
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setDuration(200)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def fade_in(self):
        """Fade the button in."""
        self._fade_animation.stop()
        self._fade_animation.setStartValue(self._opacity_effect.opacity())
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.start()
        self.show()

    def fade_out(self):
        """Fade the button out."""
        self._fade_animation.stop()
        self._fade_animation.setStartValue(self._opacity_effect.opacity())
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.finished.connect(self._on_fade_out_complete)
        self._fade_animation.start()

    def _on_fade_out_complete(self):
        """Hide button after fade out."""
        self._fade_animation.finished.disconnect(self._on_fade_out_complete)
        self.hide()

    def set_label(self, text: str):
        """
        Set button label text.

        Args:
            text: New button text
        """
        self.setText(text)


class ViewProjectBar(QWidget):
    """
    Bottom bar containing the View Project button.

    Positioned at the bottom of the window in App Mode.
    Contains the button centered horizontally.

    Usage:
        bar = ViewProjectBar(parent_window)
        bar.button.clicked.connect(fold_manager.unfold)
    """

    def __init__(self, parent: Optional[QWidget] = None, label: str = "View Project"):
        """
        Initialize the bar.

        Args:
            parent: Parent widget
            label: Button text
        """
        super().__init__(parent)

        self.setFixedHeight(48)
        self.setStyleSheet("""
            QWidget {
                background: rgba(30, 30, 30, 0.95);
                border-top: 1px solid rgba(60, 60, 60, 0.5);
            }
        """)

        # Create button
        self.button = ViewProjectButton(self, label)

        # Position button in center
        self._position_button()

    def _position_button(self):
        """Position button at center of bar."""
        # This will be called on resize
        self.button.move(
            (self.width() - self.button.width()) // 2,
            (self.height() - self.button.height()) // 2
        )

    def resizeEvent(self, event):
        """Reposition button on resize."""
        super().resizeEvent(event)
        self._position_button()


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
