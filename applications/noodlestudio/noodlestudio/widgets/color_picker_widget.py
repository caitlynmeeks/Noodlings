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
#   Color Picker Widget - Procreate-style color wheel with palette
#
#   A professional color picker featuring: - HSV color wheel ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.widgets.color_picker_widget
# PURPOSE:  Color Picker Widget
# LAYER:    Studio / Widgets
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ColorWheelWidget, ColorSwatchButton, ColorPickerWidget, ColorPickerPopup, ColorFieldWidget
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import math
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QLabel, QPushButton, QSlider, QFrame,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF, QSize
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QConicalGradient,
    QRadialGradient, QLinearGradient, QMouseEvent, QPaintEvent,
    QImage, QPixmap
)


class ColorWheelWidget(QWidget):
    """
    HSV Color wheel with inner saturation/value square.

    The outer ring selects hue (0-360).
    The inner square selects saturation (x) and value (y).
    """

    colorChanged = pyqtSignal(QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Current color in HSV
        self._hue = 0.0  # 0-360
        self._saturation = 1.0  # 0-1
        self._value = 1.0  # 0-1
        self._alpha = 1.0  # 0-1

        # Wheel geometry
        self._wheel_width = 20  # Ring thickness
        self._dragging_wheel = False
        self._dragging_square = False

        # Cache the wheel image
        self._wheel_image: Optional[QImage] = None
        self._last_size = QSize()

    def sizeHint(self) -> QSize:
        return QSize(200, 200)

    def setColor(self, color: QColor):
        """Set the current color."""
        h, s, v, a = color.getHsvF()
        if h < 0:
            h = 0  # Qt returns -1 for achromatic
        self._hue = h * 360
        self._saturation = s
        self._value = v
        self._alpha = a
        self.update()

    def color(self) -> QColor:
        """Get the current color."""
        c = QColor.fromHsvF(self._hue / 360.0, self._saturation, self._value, self._alpha)
        return c

    def setAlpha(self, alpha: float):
        """Set alpha value (0-1)."""
        self._alpha = max(0.0, min(1.0, alpha))
        self.update()
        self.colorChanged.emit(self.color())

    def _get_geometry(self):
        """Calculate wheel and square geometry."""
        size = min(self.width(), self.height())
        center = QPointF(self.width() / 2, self.height() / 2)
        outer_radius = size / 2 - 2
        inner_radius = outer_radius - self._wheel_width

        # Square inscribed in inner circle (with some padding)
        square_radius = inner_radius * 0.7
        square_rect = QRectF(
            center.x() - square_radius,
            center.y() - square_radius,
            square_radius * 2,
            square_radius * 2
        )

        return center, outer_radius, inner_radius, square_rect

    def _build_wheel_image(self, size: int):
        """Build the color wheel image (cached)."""
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)

        center = size / 2
        outer_r = size / 2 - 2
        inner_r = outer_r - self._wheel_width

        # Draw pixel by pixel for smooth gradient
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                dist = math.sqrt(dx * dx + dy * dy)

                if inner_r <= dist <= outer_r:
                    # In the wheel ring - calculate hue from angle
                    angle = math.atan2(dy, dx)
                    hue = (math.degrees(angle) + 180) % 360
                    color = QColor.fromHsvF(hue / 360.0, 1.0, 1.0)
                    image.setPixelColor(x, y, color)

        return image

    def paintEvent(self, event: QPaintEvent):
        """Paint the color wheel and saturation/value square."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center, outer_r, inner_r, square_rect = self._get_geometry()
        size = int(min(self.width(), self.height()))

        # Rebuild wheel image if size changed
        if self._wheel_image is None or self._last_size != self.size():
            self._wheel_image = self._build_wheel_image(size)
            self._last_size = self.size()

        # Draw wheel image centered
        offset_x = (self.width() - size) / 2
        offset_y = (self.height() - size) / 2
        painter.drawImage(int(offset_x), int(offset_y), self._wheel_image)

        # Draw hue indicator on wheel
        hue_angle = math.radians(self._hue - 180)
        wheel_mid_r = (outer_r + inner_r) / 2
        hue_x = center.x() + wheel_mid_r * math.cos(hue_angle)
        hue_y = center.y() + wheel_mid_r * math.sin(hue_angle)

        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.setBrush(QBrush(QColor.fromHsvF(self._hue / 360.0, 1.0, 1.0)))
        painter.drawEllipse(QPointF(hue_x, hue_y), 6, 6)

        # Draw saturation/value square with current hue
        # Horizontal gradient: white to hue color (saturation)
        hue_color = QColor.fromHsvF(self._hue / 360.0, 1.0, 1.0)

        # Draw the SV square
        # First, fill with horizontal saturation gradient
        sat_grad = QLinearGradient(square_rect.left(), 0, square_rect.right(), 0)
        sat_grad.setColorAt(0, QColor(255, 255, 255))
        sat_grad.setColorAt(1, hue_color)
        painter.fillRect(square_rect, sat_grad)

        # Overlay vertical value gradient (transparent black on top)
        val_grad = QLinearGradient(0, square_rect.top(), 0, square_rect.bottom())
        val_grad.setColorAt(0, QColor(0, 0, 0, 0))
        val_grad.setColorAt(1, QColor(0, 0, 0, 255))
        painter.fillRect(square_rect, val_grad)

        # Draw square border
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(square_rect)

        # Draw SV indicator
        sv_x = square_rect.left() + self._saturation * square_rect.width()
        sv_y = square_rect.top() + (1 - self._value) * square_rect.height()

        # Draw crosshair
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawEllipse(QPointF(sv_x, sv_y), 6, 6)
        painter.setPen(QPen(QColor("#000000"), 1))
        painter.drawEllipse(QPointF(sv_x, sv_y), 7, 7)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for dragging."""
        self._handle_mouse(QPointF(event.pos()))

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for dragging."""
        if self._dragging_wheel or self._dragging_square:
            self._handle_mouse(QPointF(event.pos()))

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        self._dragging_wheel = False
        self._dragging_square = False

    def _handle_mouse(self, pos):
        """Process mouse position to update color."""
        center, outer_r, inner_r, square_rect = self._get_geometry()

        dx = pos.x() - center.x()
        dy = pos.y() - center.y()
        dist = math.sqrt(dx * dx + dy * dy)

        # Check if in wheel ring or should continue dragging wheel
        if (inner_r <= dist <= outer_r) or self._dragging_wheel:
            if not self._dragging_square:
                self._dragging_wheel = True
                angle = math.atan2(dy, dx)
                self._hue = (math.degrees(angle) + 180) % 360
                self.update()
                self.colorChanged.emit(self.color())
                return

        # Check if in square or should continue dragging square
        if square_rect.contains(pos) or self._dragging_square:
            if not self._dragging_wheel:
                self._dragging_square = True
                # Clamp to square bounds
                x = max(square_rect.left(), min(square_rect.right(), pos.x()))
                y = max(square_rect.top(), min(square_rect.bottom(), pos.y()))

                self._saturation = (x - square_rect.left()) / square_rect.width()
                self._value = 1.0 - (y - square_rect.top()) / square_rect.height()
                self.update()
                self.colorChanged.emit(self.color())


class ColorSwatchButton(QPushButton):
    """A clickable color swatch."""

    colorClicked = pyqtSignal(QColor)

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(24, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._on_clicked)
        self._update_style()

    def _update_style(self):
        """Update button style to show color."""
        hex_color = self._color.name()
        # Calculate contrasting border
        lightness = self._color.lightnessF()
        border_color = "#555555" if lightness > 0.5 else "#888888"

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {hex_color};
                border: 1px solid {border_color};
                border-radius: 3px;
            }}
            QPushButton:hover {{
                border: 2px solid #4a9eff;
            }}
        """)

    def setColor(self, color: QColor):
        """Set the swatch color."""
        self._color = color
        self._update_style()

    def color(self) -> QColor:
        return self._color

    def _on_clicked(self):
        self.colorClicked.emit(self._color)


class ColorPickerWidget(QWidget):
    """
    Complete color picker with wheel, palette, and hex input.

    Signals:
        colorChanged(QColor): Emitted when color changes
    """

    colorChanged = pyqtSignal(QColor)

    # Default palette colors (Procreate-inspired)
    DEFAULT_PALETTE = [
        "#ffffff", "#c0c0c0", "#808080", "#404040", "#000000",
        "#ff0000", "#ff8000", "#ffff00", "#80ff00", "#00ff00",
        "#00ff80", "#00ffff", "#0080ff", "#0000ff", "#8000ff",
        "#ff00ff", "#ff0080", "#ff6b6b", "#ffd93d", "#6bcb77",
        "#4d96ff", "#845ec2", "#ff9671", "#ffc75f", "#f9f871",
    ]

    def __init__(self, parent=None, show_alpha: bool = False):
        super().__init__(parent)
        self._show_alpha = show_alpha
        self._current_color = QColor("#ffffff")
        self._recent_colors: List[QColor] = []
        self._max_recent = 10

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Color wheel
        self._wheel = ColorWheelWidget()
        self._wheel.colorChanged.connect(self._on_wheel_changed)
        layout.addWidget(self._wheel)

        # Current color preview + hex input row
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)

        # Current color swatch (larger)
        self._current_swatch = QFrame()
        self._current_swatch.setFixedSize(40, 40)
        self._current_swatch.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
            }
        """)
        preview_row.addWidget(self._current_swatch)

        # Hex input
        hex_layout = QVBoxLayout()
        hex_layout.setSpacing(2)
        hex_label = QLabel("Hex")
        hex_label.setStyleSheet("color: #888888; font-size: 10px;")
        hex_layout.addWidget(hex_label)

        self._hex_input = QLineEdit("#ffffff")
        self._hex_input.setMaxLength(9)  # #RRGGBBAA
        self._hex_input.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 4px;
                color: #cccccc;
                font-family: monospace;
            }
            QLineEdit:focus {
                border-color: #4a9eff;
            }
        """)
        self._hex_input.editingFinished.connect(self._on_hex_changed)
        hex_layout.addWidget(self._hex_input)
        preview_row.addLayout(hex_layout)

        preview_row.addStretch()
        layout.addLayout(preview_row)

        # Alpha slider (optional)
        if self._show_alpha:
            alpha_row = QHBoxLayout()
            alpha_label = QLabel("Alpha")
            alpha_label.setStyleSheet("color: #888888; font-size: 10px;")
            alpha_label.setFixedWidth(40)
            alpha_row.addWidget(alpha_label)

            self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
            self._alpha_slider.setRange(0, 255)
            self._alpha_slider.setValue(255)
            self._alpha_slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 transparent, stop:1 #ffffff);
                    height: 12px;
                    border-radius: 6px;
                    border: 1px solid #3d3d3d;
                }
                QSlider::handle:horizontal {
                    background: #cccccc;
                    width: 14px;
                    margin: -2px 0;
                    border-radius: 7px;
                    border: 1px solid #555555;
                }
            """)
            self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
            alpha_row.addWidget(self._alpha_slider)
            layout.addLayout(alpha_row)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #3d3d3d;")
        separator.setFixedHeight(1)
        layout.addWidget(separator)

        # Palette label
        palette_label = QLabel("Palette")
        palette_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(palette_label)

        # Color palette grid
        palette_grid = QGridLayout()
        palette_grid.setSpacing(4)

        self._palette_buttons: List[ColorSwatchButton] = []
        cols = 5
        for i, hex_color in enumerate(self.DEFAULT_PALETTE):
            btn = ColorSwatchButton(QColor(hex_color))
            btn.colorClicked.connect(self._on_palette_clicked)
            palette_grid.addWidget(btn, i // cols, i % cols)
            self._palette_buttons.append(btn)

        layout.addLayout(palette_grid)

        # Recent colors
        recent_label = QLabel("Recent")
        recent_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(recent_label)

        self._recent_layout = QHBoxLayout()
        self._recent_layout.setSpacing(4)
        self._recent_buttons: List[ColorSwatchButton] = []

        # Create placeholder slots
        for i in range(self._max_recent):
            btn = ColorSwatchButton(QColor("#2d2d2d"))
            btn.colorClicked.connect(self._on_palette_clicked)
            btn.hide()  # Hidden until used
            self._recent_layout.addWidget(btn)
            self._recent_buttons.append(btn)

        self._recent_layout.addStretch()
        layout.addLayout(self._recent_layout)

        layout.addStretch()

    def setColor(self, color: QColor):
        """Set the current color."""
        self._current_color = color
        self._wheel.setColor(color)
        self._update_preview()
        self._hex_input.setText(color.name())

        if self._show_alpha and hasattr(self, '_alpha_slider'):
            self._alpha_slider.setValue(color.alpha())

    def color(self) -> QColor:
        """Get the current color."""
        return self._current_color

    def _update_preview(self):
        """Update the current color preview swatch."""
        hex_color = self._current_color.name()
        alpha = self._current_color.alpha()

        if alpha < 255:
            # Show checkerboard pattern for transparency
            self._current_swatch.setStyleSheet(f"""
                QFrame {{
                    background-color: {hex_color};
                    border: 1px solid #555555;
                    border-radius: 4px;
                }}
            """)
        else:
            self._current_swatch.setStyleSheet(f"""
                QFrame {{
                    background-color: {hex_color};
                    border: 1px solid #555555;
                    border-radius: 4px;
                }}
            """)

    def _add_to_recent(self, color: QColor):
        """Add a color to recent colors."""
        # Don't add duplicates
        for rc in self._recent_colors:
            if rc.name() == color.name():
                return

        # Add to front, remove old
        self._recent_colors.insert(0, QColor(color))
        if len(self._recent_colors) > self._max_recent:
            self._recent_colors.pop()

        # Update recent buttons
        for i, btn in enumerate(self._recent_buttons):
            if i < len(self._recent_colors):
                btn.setColor(self._recent_colors[i])
                btn.show()
            else:
                btn.hide()

    def _on_wheel_changed(self, color: QColor):
        """Handle color wheel changes."""
        self._current_color = color
        self._update_preview()
        self._hex_input.setText(color.name())
        self.colorChanged.emit(color)

    def _on_hex_changed(self):
        """Handle hex input changes."""
        hex_text = self._hex_input.text().strip()
        if not hex_text.startswith("#"):
            hex_text = "#" + hex_text

        color = QColor(hex_text)
        if color.isValid():
            self._current_color = color
            self._wheel.setColor(color)
            self._update_preview()
            self.colorChanged.emit(color)
            self._add_to_recent(color)

    def _on_alpha_changed(self, value: int):
        """Handle alpha slider changes."""
        self._current_color.setAlpha(value)
        self._wheel.setAlpha(value / 255.0)
        self._update_preview()
        self.colorChanged.emit(self._current_color)

    def _on_palette_clicked(self, color: QColor):
        """Handle palette swatch click."""
        self.setColor(color)
        self._add_to_recent(color)
        self.colorChanged.emit(color)


class ColorPickerPopup(QWidget):
    """
    A popup color picker that appears when clicking a color field.

    Shows the full ColorPickerWidget in a floating window.
    """

    colorChanged = pyqtSignal(QColor)
    closed = pyqtSignal()

    def __init__(self, parent=None, show_alpha: bool = False):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._picker = ColorPickerWidget(show_alpha=show_alpha)
        self._picker.colorChanged.connect(self.colorChanged.emit)
        layout.addWidget(self._picker)

        self.setStyleSheet("""
            ColorPickerPopup {
                background-color: #2d2d2d;
                border: 1px solid #4d4d4d;
                border-radius: 6px;
            }
        """)

        self.setFixedWidth(250)

    def setColor(self, color: QColor):
        """Set the current color."""
        self._picker.setColor(color)

    def color(self) -> QColor:
        """Get the current color."""
        return self._picker.color()

    def closeEvent(self, event):
        """Handle close event."""
        self.closed.emit()
        super().closeEvent(event)


class ColorFieldWidget(QWidget):
    """
    A color field widget for use in forms/inspectors.

    Shows a color swatch + hex value, click to open picker popup.
    """

    colorChanged = pyqtSignal(QColor)

    def __init__(self, parent=None, show_alpha: bool = False):
        super().__init__(parent)
        self._color = QColor("#ffffff")
        self._show_alpha = show_alpha
        self._popup: Optional[ColorPickerPopup] = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Color swatch button
        self._swatch = QPushButton()
        self._swatch.setFixedSize(24, 24)
        self._swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self._swatch.clicked.connect(self._show_picker)
        layout.addWidget(self._swatch)

        # Hex input
        self._hex_input = QLineEdit("#ffffff")
        self._hex_input.setMaxLength(9)
        self._hex_input.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 2px;
                padding: 2px 4px;
                color: #cccccc;
                font-family: monospace;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #4a9eff;
            }
        """)
        self._hex_input.editingFinished.connect(self._on_hex_changed)
        layout.addWidget(self._hex_input)

        self._update_swatch()

    def setColor(self, color: QColor):
        """Set the current color."""
        if isinstance(color, str):
            color = QColor(color)
        self._color = color
        self._hex_input.setText(color.name())
        self._update_swatch()

    def color(self) -> QColor:
        """Get the current color."""
        return self._color

    def _update_swatch(self):
        """Update the swatch button appearance."""
        hex_color = self._color.name()
        lightness = self._color.lightnessF()
        border_color = "#555555" if lightness > 0.5 else "#888888"

        self._swatch.setStyleSheet(f"""
            QPushButton {{
                background-color: {hex_color};
                border: 1px solid {border_color};
                border-radius: 3px;
            }}
            QPushButton:hover {{
                border: 2px solid #4a9eff;
            }}
        """)

    def _show_picker(self):
        """Show the color picker popup."""
        from PyQt6.QtWidgets import QApplication

        if self._popup is None:
            self._popup = ColorPickerPopup(show_alpha=self._show_alpha)
            self._popup.colorChanged.connect(self._on_picker_changed)

        self._popup.setColor(self._color)

        # Get screen geometry to ensure popup stays on-screen
        screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry()

        # Try positioning below the swatch first
        global_pos = self._swatch.mapToGlobal(self._swatch.rect().bottomLeft())
        popup_rect = self._popup.rect()
        popup_rect.moveTopLeft(global_pos)

        # Check if popup would go off right edge
        if popup_rect.right() > screen_rect.right():
            # Position to the left of the swatch instead
            global_pos.setX(screen_rect.right() - self._popup.width() - 10)

        # Check if popup would go off bottom edge
        if popup_rect.bottom() > screen_rect.bottom():
            # Position above the swatch
            global_pos = self._swatch.mapToGlobal(self._swatch.rect().topLeft())
            global_pos.setY(global_pos.y() - self._popup.sizeHint().height())

        self._popup.move(global_pos)
        self._popup.show()

    def _on_picker_changed(self, color: QColor):
        """Handle color picker changes."""
        self._color = color
        self._hex_input.setText(color.name())
        self._update_swatch()
        self.colorChanged.emit(color)

    def _on_hex_changed(self):
        """Handle hex input changes."""
        hex_text = self._hex_input.text().strip()
        if not hex_text.startswith("#"):
            hex_text = "#" + hex_text

        color = QColor(hex_text)
        if color.isValid():
            self._color = color
            self._update_swatch()
            self.colorChanged.emit(color)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
