"""
Toggle Switch Widget - Like LMStudio's server toggle

Clean linear toggle switch with green/gray states.

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush


class ToggleSwitch(QWidget):
    """
    LMStudio-style toggle switch.

    OFF: Gray background, knob on left
    ON: Green background, knob on right
    """

    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self._knob_position = 0  # 0 = left (off), 1 = right (on)

        self.setFixedSize(50, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Animation for smooth sliding
        self.animation = QPropertyAnimation(self, b"knob_position")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background track
        track_rect = QRect(0, 0, self.width(), self.height())
        track_color = QColor(76, 175, 80) if self._checked else QColor(120, 120, 120)  # Green or gray

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(track_rect, self.height() // 2, self.height() // 2)

        # Knob (white circle)
        knob_radius = self.height() - 6
        knob_x = int(3 + self._knob_position * (self.width() - knob_radius - 6))
        knob_y = self.height() // 2

        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(knob_x, 3, knob_radius, knob_radius)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()

    def toggle(self):
        """Toggle the switch."""
        self.setChecked(not self._checked)

    def setChecked(self, checked: bool):
        """Set switch state."""
        if self._checked == checked:
            return

        self._checked = checked

        # Animate knob movement
        self.animation.stop()
        self.animation.setStartValue(self._knob_position)
        self.animation.setEndValue(1.0 if checked else 0.0)
        self.animation.start()

        self.toggled.emit(checked)

    def isChecked(self) -> bool:
        """Get switch state."""
        return self._checked

    @pyqtProperty(float)
    def knob_position(self):
        return self._knob_position

    @knob_position.setter
    def knob_position(self, value):
        self._knob_position = value
        self.update()
