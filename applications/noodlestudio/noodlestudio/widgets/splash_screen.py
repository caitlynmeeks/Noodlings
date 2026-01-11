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
#   Splash Screen
#
#   Customizable splash screen for published NoodleSTUDIO apps.
#   Supports custom images, text, and always includes NEC attribution.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.widgets.splash_screen
# PURPOSE:  Splash Screen for Published Apps
# LAYER:    Studio / Widgets
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SplashScreen
#   AttributionWidget
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import webbrowser
from typing import Callable, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QSize
)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter, QPainterPath

logger = logging.getLogger(__name__)


class LoadingIndicator(QWidget):
    """
    Animated loading indicator for splash screen.

    Supports styles: dots, bar, spinner, none
    """

    def __init__(self, style: str = "dots", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._style = style
        self._progress = 0
        self._animation_timer: Optional[QTimer] = None

        self.setFixedSize(80, 20)

        if style != "none":
            self._start_animation()

    def _start_animation(self):
        """Start the loading animation."""
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._tick)
        self._animation_timer.start(200)  # 5 FPS for dots

    def _tick(self):
        """Animation tick."""
        self._progress = (self._progress + 1) % 4
        self.update()

    def stop(self):
        """Stop the animation."""
        if self._animation_timer:
            self._animation_timer.stop()
            self._animation_timer = None

    def paintEvent(self, event):
        """Paint the loading indicator."""
        if self._style == "none":
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._style == "dots":
            self._paint_dots(painter)
        elif self._style == "bar":
            self._paint_bar(painter)
        elif self._style == "spinner":
            self._paint_spinner(painter)

        painter.end()

    def _paint_dots(self, painter: QPainter):
        """Paint dots loading indicator."""
        dot_count = 3
        dot_size = 6
        spacing = 12
        total_width = dot_count * dot_size + (dot_count - 1) * spacing
        start_x = (self.width() - total_width) // 2
        y = self.height() // 2 - dot_size // 2

        for i in range(dot_count):
            x = start_x + i * (dot_size + spacing)
            # Highlight dots up to progress
            if i <= self._progress:
                painter.setBrush(QColor(200, 200, 200))
            else:
                painter.setBrush(QColor(80, 80, 80))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(x, y, dot_size, dot_size)

    def _paint_bar(self, painter: QPainter):
        """Paint bar loading indicator."""
        bar_width = 60
        bar_height = 4
        x = (self.width() - bar_width) // 2
        y = self.height() // 2 - bar_height // 2

        # Background
        painter.setBrush(QColor(60, 60, 60))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(x, y, bar_width, bar_height, 2, 2)

        # Progress (cycling)
        progress_width = int(bar_width * ((self._progress + 1) / 4))
        painter.setBrush(QColor(150, 150, 150))
        painter.drawRoundedRect(x, y, progress_width, bar_height, 2, 2)

    def _paint_spinner(self, painter: QPainter):
        """Paint spinner loading indicator."""
        size = 16
        x = (self.width() - size) // 2
        y = (self.height() - size) // 2

        painter.translate(x + size // 2, y + size // 2)
        painter.rotate(self._progress * 90)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(150, 150, 150))
        painter.drawEllipse(-size // 2, -size // 2, 6, 6)


class AttributionWidget(QWidget):
    """
    The 'Made with NoodleSTUDIO' attribution.

    Required on all published apps. Clickable to open NEC link.

    Styles:
    - badge: Rounded rect with text
    - text: Simple inline text
    - minimal: Just "NoodleSTUDIO"
    """

    NEC_URL = "https://noodlings.ai/nec"

    clicked = pyqtSignal()

    def __init__(
        self,
        style: str = "badge",
        show_nec_link: bool = True,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._style = style
        self._show_nec_link = show_nec_link

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_ui()

    def _build_ui(self):
        """Build the attribution UI based on style."""
        if self._style == "badge":
            self._build_badge()
        elif self._style == "text":
            self._build_text()
        else:
            self._build_minimal()

    def _build_badge(self):
        """Build badge-style attribution."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        self.setStyleSheet("""
            AttributionWidget {
                background: rgba(40, 40, 40, 0.9);
                border: 1px solid rgba(80, 80, 80, 0.5);
                border-radius: 6px;
            }
        """)

        # "Made with NoodleSTUDIO"
        main_label = QLabel("Made with NoodleSTUDIO")
        main_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(main_label)

        # NEC link
        if self._show_nec_link:
            nec_label = QLabel("noodlings.ai/nec")
            nec_label.setStyleSheet("color: #888888; font-size: 10px;")
            nec_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(nec_label)

    def _build_text(self):
        """Build text-style attribution."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        if self._show_nec_link:
            text = "Made with NoodleSTUDIO - noodlings.ai/nec"
        else:
            text = "Made with NoodleSTUDIO"

        label = QLabel(text)
        label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(label)

    def _build_minimal(self):
        """Build minimal-style attribution."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        label = QLabel("NoodleSTUDIO")
        label.setStyleSheet("color: #666666; font-size: 10px;")
        layout.addWidget(label)

    def mousePressEvent(self, event):
        """Open NEC link on click."""
        self.clicked.emit()
        if self._show_nec_link:
            webbrowser.open(self.NEC_URL)
        super().mousePressEvent(event)


class SplashScreen(QWidget):
    """
    Customizable splash screen for published NoodleSTUDIO apps.

    Features:
    - Custom image or text-based splash
    - Fade in/out animations
    - Loading indicator (dots, bar, spinner)
    - Required attribution (Made with NoodleSTUDIO + NEC link)
    - Minimum 1.5s display

    Usage:
        config = {
            'title': 'My App',
            'subtitle': 'Welcome',
            'background': '#1a1a2e',
            'duration': 2.5,
        }
        splash = SplashScreen(config)
        splash.show_splash(on_complete=lambda: main_window.show())
    """

    # Signals
    fade_in_complete = pyqtSignal()
    fade_out_complete = pyqtSignal()

    # Minimum display time (seconds)
    MIN_DURATION = 1.5

    def __init__(
        self,
        config: Optional[Dict] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.config = config or {}

        # Window setup - frameless, stays on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Timing
        self._duration = max(self.MIN_DURATION, self.config.get('duration', 2.5))
        self._fade_in_duration = self.config.get('fade_in', 0.3)
        self._fade_out_duration = self.config.get('fade_out', 0.5)

        # State
        self._on_complete: Optional[Callable] = None
        self._loading_indicator: Optional[LoadingIndicator] = None

        # Opacity effect for fade
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        # Animation
        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Set default size
        self.setFixedSize(500, 350)

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build splash screen UI from config."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main content container
        content = QWidget()
        content.setObjectName("splashContent")

        bg_color = self.config.get('background', '#1a1a2e')
        content.setStyleSheet(f"""
            #splashContent {{
                background: {bg_color};
                border-radius: 8px;
            }}
        """)

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(40, 40, 40, 20)
        content_layout.setSpacing(12)

        # Custom image or text splash
        if self.config.get('image'):
            self._build_image_splash(content_layout)
        else:
            self._build_text_splash(content_layout)

        # Loading indicator
        show_loading = self.config.get('show_loading', True)
        if show_loading:
            loading_style = self.config.get('loading_style', 'dots')
            self._loading_indicator = LoadingIndicator(loading_style)
            indicator_container = QWidget()
            indicator_layout = QHBoxLayout(indicator_container)
            indicator_layout.setContentsMargins(0, 0, 0, 0)
            indicator_layout.addStretch()
            indicator_layout.addWidget(self._loading_indicator)
            indicator_layout.addStretch()
            content_layout.addWidget(indicator_container)

        # Spacer before attribution
        content_layout.addStretch()

        # Attribution (always present, required)
        attr_config = self.config.get('attribution', {})
        attr_style = attr_config.get('style', 'badge')
        show_nec = attr_config.get('show_nec_link', True)

        attribution = AttributionWidget(attr_style, show_nec)

        # Position attribution based on config
        attr_position = attr_config.get('position', 'bottom-center')
        attr_container = QWidget()
        attr_layout = QHBoxLayout(attr_container)
        attr_layout.setContentsMargins(10, 0, 10, 10)

        if attr_position == 'bottom-left':
            attr_layout.addWidget(attribution)
            attr_layout.addStretch()
        elif attr_position == 'bottom-right':
            attr_layout.addStretch()
            attr_layout.addWidget(attribution)
        else:  # bottom-center
            attr_layout.addStretch()
            attr_layout.addWidget(attribution)
            attr_layout.addStretch()

        content_layout.addWidget(attr_container)

        layout.addWidget(content)

    def _build_image_splash(self, layout: QVBoxLayout):
        """Build splash from custom image."""
        image_path = self.config['image']
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale to fit
                pixmap = pixmap.scaled(
                    420, 200,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                label = QLabel()
                label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
            else:
                logger.warning(f"Could not load splash image: {image_path}")
                self._build_text_splash(layout)
        except Exception as e:
            logger.error(f"Error loading splash image: {e}")
            self._build_text_splash(layout)

    def _build_text_splash(self, layout: QVBoxLayout):
        """Build text-based splash."""
        # Logo (optional)
        if self.config.get('logo'):
            logo_path = self.config['logo']
            try:
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        100, 100,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    logo_label = QLabel()
                    logo_label.setPixmap(pixmap)
                    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout.addWidget(logo_label)
            except Exception as e:
                logger.warning(f"Could not load logo: {e}")

        # Title
        title = self.config.get('title', 'NoodleSTUDIO')
        title_color = self.config.get('title_color', '#ffffff')
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {title_color};
        """)
        layout.addWidget(title_label)

        # Subtitle (optional)
        subtitle = self.config.get('subtitle')
        if subtitle:
            subtitle_color = self.config.get('subtitle_color', '#888888')
            subtitle_label = QLabel(subtitle)
            subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subtitle_label.setStyleSheet(f"""
                font-size: 14px;
                color: {subtitle_color};
            """)
            layout.addWidget(subtitle_label)

    def show_splash(self, on_complete: Optional[Callable] = None):
        """
        Show splash with animations, call on_complete when done.

        Args:
            on_complete: Callback when splash is finished
        """
        self._on_complete = on_complete

        # Center on screen
        self._center_on_screen()

        # Show
        self.show()
        self.raise_()

        logger.info(f"Showing splash for {self._duration}s")

        # Fade in
        self._fade_in()

    def _center_on_screen(self):
        """Center the splash on the primary screen."""
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QScreen
        screen = QApplication.primaryScreen()
        if screen:
            geometry = screen.availableGeometry()
            x = (geometry.width() - self.width()) // 2 + geometry.x()
            y = (geometry.height() - self.height()) // 2 + geometry.y()
            self.move(x, y)

    def _fade_in(self):
        """Animate fade in."""
        self._fade_animation.stop()
        self._fade_animation.setDuration(int(self._fade_in_duration * 1000))
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.finished.connect(self._on_fade_in_complete)
        self._fade_animation.start()

    def _on_fade_in_complete(self):
        """Called when fade in completes."""
        self._fade_animation.finished.disconnect(self._on_fade_in_complete)
        self.fade_in_complete.emit()

        # Schedule fade out after display time
        display_time = self._duration - self._fade_in_duration - self._fade_out_duration
        QTimer.singleShot(int(display_time * 1000), self._fade_out)

    def _fade_out(self):
        """Animate fade out, then call completion handler."""
        self._fade_animation.stop()
        self._fade_animation.setDuration(int(self._fade_out_duration * 1000))
        self._fade_animation.setStartValue(1.0)
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.finished.connect(self._on_fade_out_complete)
        self._fade_animation.start()

    def _on_fade_out_complete(self):
        """Called when fade out completes."""
        self._fade_animation.finished.disconnect(self._on_fade_out_complete)

        # Stop loading indicator
        if self._loading_indicator:
            self._loading_indicator.stop()

        self.fade_out_complete.emit()
        logger.info("Splash complete")

        # Hide and call completion
        self.hide()

        if self._on_complete:
            self._on_complete()

    def skip(self):
        """Skip remaining splash time (for dev/testing)."""
        if self._fade_animation.state() == QPropertyAnimation.State.Running:
            self._fade_animation.stop()

        self._fade_out()

    def mousePressEvent(self, event):
        """
        Handle click - skip splash if click_to_skip enabled.
        """
        if self.config.get('click_to_skip', False):
            # Only skip after minimum time
            self.skip()
        super().mousePressEvent(event)


def create_default_splash() -> SplashScreen:
    """Create the default NoodleSTUDIO splash screen."""
    return SplashScreen({
        'title': 'NoodleSTUDIO',
        'subtitle': 'Building minds, not black boxes.',
        'background': '#1a1a2e',
        'duration': 2.5,
        'show_loading': True,
        'loading_style': 'dots',
        'attribution': {
            'position': 'bottom-center',
            'style': 'badge',
            'show_nec_link': True,
        }
    })


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
