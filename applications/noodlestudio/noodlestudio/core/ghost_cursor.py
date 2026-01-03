"""
Ghost Cursor Overlay - Theatrical visualization of Computer Use actions

Displays a beautiful ghost cursor that moves with organic bezier curves,
click ripples, and motion trails. Makes Computer Use visible and magical.

"Watch as it edits itself..."

Author: Caitlyn + Claude
Date: January 2, 2026
"""

import math
import random
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QRadialGradient, QLinearGradient, QPolygonF
)


@dataclass
class CursorAnimation:
    """Represents an in-progress cursor movement animation."""
    start: QPointF
    end: QPointF
    control1: QPointF  # Bezier control point 1
    control2: QPointF  # Bezier control point 2
    duration_ms: int
    start_time: float

    def progress(self, current_time: float) -> float:
        """Get animation progress 0.0 to 1.0 with easing."""
        elapsed = current_time - self.start_time
        t = min(1.0, elapsed / (self.duration_ms / 1000.0))
        # Ease-in-out cubic for smooth acceleration/deceleration
        return self._ease_in_out_cubic(t)

    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic ease-in-out: slow start, fast middle, slow end."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def position_at(self, t: float) -> QPointF:
        """Get position along bezier curve at parameter t."""
        # Cubic bezier: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3 * self.start.x() + 3 * mt2 * t * self.control1.x() + \
            3 * mt * t2 * self.control2.x() + t3 * self.end.x()
        y = mt3 * self.start.y() + 3 * mt2 * t * self.control1.y() + \
            3 * mt * t2 * self.control2.y() + t3 * self.end.y()

        return QPointF(x, y)


@dataclass
class ClickRipple:
    """A click ripple effect."""
    center: QPointF
    start_time: float
    duration_ms: int = 400
    max_radius: float = 30.0
    color: QColor = field(default_factory=lambda: QColor(150, 150, 150))

    def progress(self, current_time: float) -> float:
        """Get ripple progress 0.0 to 1.0."""
        elapsed = current_time - self.start_time
        return min(1.0, elapsed / (self.duration_ms / 1000.0))

    def is_finished(self, current_time: float) -> bool:
        return self.progress(current_time) >= 1.0


@dataclass
class TrailPoint:
    """A point in the motion trail."""
    position: QPointF
    timestamp: float

    def age(self, current_time: float) -> float:
        """Age in seconds."""
        return current_time - self.timestamp


class GhostCursorOverlay(QWidget):
    """
    Transparent overlay that renders the ghost cursor and effects.

    Features:
    - Semi-transparent ghost cursor with glow
    - Organic bezier curve movement paths
    - Click ripple effects
    - Motion trail afterimages
    - Theatrical timing and easing
    """

    # Signals
    animationComplete = pyqtSignal()  # Emitted when movement animation finishes

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        # Make overlay transparent and click-through
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # State
        self._enabled = False
        self._cursor_pos = QPointF(0, 0)
        self._cursor_visible = False
        self._current_animation: Optional[CursorAnimation] = None
        self._ripples: List[ClickRipple] = []
        self._trail: List[TrailPoint] = []
        self._trail_max_age = 0.3  # seconds
        self._trail_max_points = 50

        # Visual settings
        self._cursor_color = QColor(180, 180, 180, 200)
        self._cursor_glow_color = QColor(255, 255, 255, 60)
        self._trail_color = QColor(150, 150, 150, 100)
        self._ripple_color = QColor(200, 200, 200, 150)

        # Breathing glow colors
        self._glow_color_white = QColor(255, 255, 255)  # Idle breathing
        self._glow_color_pink = QColor(255, 120, 180)   # Hot pink (moving/sparkle)
        self._glow_color_orange = QColor(255, 160, 80)  # Warm orange (moving/sparkle)

        # Animation settings
        self._base_move_duration = 400  # ms for short moves
        self._max_move_duration = 800   # ms for long moves

        # Breathing animation
        self._breath_phase = 0.0  # 0.0 to 1.0, cycles continuously
        self._breath_speed_idle = 0.4   # Slow gentle breathing when idle
        self._breath_speed_moving = 1.5  # Faster when moving

        # State: idle vs moving
        self._is_moving = False
        self._move_intensity = 0.0  # 0.0 = idle, 1.0 = full moving mode
        self._intensity_fade_speed = 3.0  # How fast to transition between states

        # Sparkle poofs (occasional color bursts when idle)
        self._sparkle_intensity = 0.0  # Current sparkle amount
        self._sparkle_color_mix = 0.0  # 0.0 = pink, 1.0 = orange
        self._time_until_sparkle = random.uniform(5.0, 10.0)  # Seconds until next sparkle
        self._last_time = self._current_time()

        # Insight flash - brief bright flash before moving
        self._insight_flash = 0.0  # 0.0 to 1.0, decays quickly

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.setInterval(16)  # ~60fps

        self.hide()

    def set_enabled(self, enabled: bool):
        """Enable or disable the ghost cursor overlay."""
        self._enabled = enabled
        if enabled:
            self.show()
            self.raise_()
        else:
            self.hide()
            self._cursor_visible = False
            self._current_animation = None
            self._ripples.clear()
            self._trail.clear()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def move_to(self, x: int, y: int, callback=None):
        """
        Animate the ghost cursor to a new position with beautiful bezier curves.

        Args:
            x, y: Target position (window-relative)
            callback: Optional function to call when animation completes
        """
        if not self._enabled:
            if callback:
                callback()
            return

        target = QPointF(x, y)

        # If cursor not visible yet, just appear at position
        if not self._cursor_visible:
            self._cursor_pos = target
            self._cursor_visible = True
            self._timer.start()
            self.update()
            if callback:
                callback()
            return

        # Calculate distance for duration scaling
        dx = target.x() - self._cursor_pos.x()
        dy = target.y() - self._cursor_pos.y()
        distance = math.sqrt(dx * dx + dy * dy)

        # Scale duration based on distance (longer moves take longer, but not linearly)
        duration = int(self._base_move_duration +
                      (self._max_move_duration - self._base_move_duration) *
                      min(1.0, distance / 500.0))

        # Generate organic bezier control points
        control1, control2 = self._generate_control_points(
            self._cursor_pos, target, distance
        )

        # Trigger insight flash - "watch me, I'm about to move!"
        self._insight_flash = 1.0

        # Create animation
        self._current_animation = CursorAnimation(
            start=QPointF(self._cursor_pos),
            end=target,
            control1=control1,
            control2=control2,
            duration_ms=duration,
            start_time=self._current_time()
        )

        self._animation_callback = callback

        if not self._timer.isActive():
            self._timer.start()

    def _generate_control_points(self, start: QPointF, end: QPointF,
                                  distance: float) -> Tuple[QPointF, QPointF]:
        """
        Generate bezier control points for an organic, sweeping curve.

        Creates curves that arc naturally like a confident hand gesture,
        not robotic straight lines or awkward S-curves.
        """
        dx = end.x() - start.x()
        dy = end.y() - start.y()

        # Perpendicular vector for arc direction
        perp_x = -dy
        perp_y = dx
        perp_len = math.sqrt(perp_x * perp_x + perp_y * perp_y)
        if perp_len > 0:
            perp_x /= perp_len
            perp_y /= perp_len

        # Arc magnitude scales with distance, with some randomness
        arc_magnitude = distance * random.uniform(0.15, 0.35)

        # Randomly choose arc direction (left or right of direct path)
        arc_sign = random.choice([-1, 1])

        # Control point 1: ~1/3 along path, arced out
        t1 = random.uniform(0.25, 0.4)
        c1_base_x = start.x() + dx * t1
        c1_base_y = start.y() + dy * t1
        c1_arc = arc_magnitude * random.uniform(0.8, 1.2)
        control1 = QPointF(
            c1_base_x + perp_x * c1_arc * arc_sign,
            c1_base_y + perp_y * c1_arc * arc_sign
        )

        # Control point 2: ~2/3 along path, arced (same direction for smooth curve)
        t2 = random.uniform(0.6, 0.75)
        c2_base_x = start.x() + dx * t2
        c2_base_y = start.y() + dy * t2
        c2_arc = arc_magnitude * random.uniform(0.6, 1.0)
        control2 = QPointF(
            c2_base_x + perp_x * c2_arc * arc_sign,
            c2_base_y + perp_y * c2_arc * arc_sign
        )

        return control1, control2

    def click_at(self, x: int, y: int, button: str = "left"):
        """Show a click ripple effect at the position."""
        if not self._enabled:
            return

        # Determine ripple color based on button
        if button == "right":
            color = QColor(200, 180, 150, 150)  # Slightly warm for right-click
        elif button == "double":
            color = QColor(220, 220, 220, 180)  # Brighter for double-click
        else:
            color = QColor(180, 180, 180, 150)  # Default gray

        ripple = ClickRipple(
            center=QPointF(x, y),
            start_time=self._current_time(),
            color=color
        )
        self._ripples.append(ripple)

        if not self._timer.isActive():
            self._timer.start()

    def type_indicator(self, x: int, y: int):
        """Show a subtle typing indicator at position."""
        if not self._enabled:
            return
        # Could add a pulsing glow or text cursor effect here
        pass

    def hide_cursor(self):
        """Hide the ghost cursor."""
        self._cursor_visible = False
        self.update()

    def _current_time(self) -> float:
        """Get current time in seconds."""
        return datetime.now().timestamp()

    def _on_timer(self):
        """Animation timer tick."""
        current = self._current_time()
        dt = current - self._last_time
        self._last_time = current
        needs_update = False

        # Update state: moving vs idle
        if self._current_animation:
            # Animation active = moving state
            self._is_moving = True
            # Ramp up intensity quickly
            self._move_intensity = min(1.0, self._move_intensity + dt * self._intensity_fade_speed * 2)
        else:
            # No animation = idle state
            self._is_moving = False
            # Fade intensity slowly back to idle
            self._move_intensity = max(0.0, self._move_intensity - dt * self._intensity_fade_speed)

        # Advance breathing animation (always runs when cursor visible)
        if self._cursor_visible:
            # Breath speed depends on state (lerp between idle and moving speeds)
            breath_speed = (self._breath_speed_idle +
                          (self._breath_speed_moving - self._breath_speed_idle) * self._move_intensity)
            self._breath_phase = (self._breath_phase + breath_speed * dt) % 1.0
            needs_update = True

            # Sparkle poofs when idle
            if self._move_intensity < 0.3:
                # Count down to next sparkle
                self._time_until_sparkle -= dt
                if self._time_until_sparkle <= 0:
                    # Trigger a sparkle!
                    self._sparkle_intensity = 1.0
                    self._sparkle_color_mix = random.random()  # Random pink/orange mix
                    self._time_until_sparkle = random.uniform(5.0, 10.0)

            # Decay sparkle intensity
            if self._sparkle_intensity > 0:
                self._sparkle_intensity = max(0.0, self._sparkle_intensity - dt * 2.0)
                needs_update = True

            # Decay insight flash (fast - it's a brief flicker)
            if self._insight_flash > 0:
                self._insight_flash = max(0.0, self._insight_flash - dt * 8.0)  # ~125ms to fade
                needs_update = True

        # Update cursor animation
        if self._current_animation:
            progress = self._current_animation.progress(current)
            self._cursor_pos = self._current_animation.position_at(progress)

            # Add to trail
            self._trail.append(TrailPoint(
                position=QPointF(self._cursor_pos),
                timestamp=current
            ))

            # Trim old trail points
            self._trail = [p for p in self._trail
                          if p.age(current) < self._trail_max_age][-self._trail_max_points:]

            needs_update = True

            if progress >= 1.0:
                self._current_animation = None
                self.animationComplete.emit()
                if hasattr(self, '_animation_callback') and self._animation_callback:
                    cb = self._animation_callback
                    self._animation_callback = None
                    cb()

        # Update ripples
        finished_ripples = []
        for ripple in self._ripples:
            if ripple.is_finished(current):
                finished_ripples.append(ripple)
            else:
                needs_update = True

        for ripple in finished_ripples:
            self._ripples.remove(ripple)

        # Clean up trail
        old_trail_len = len(self._trail)
        self._trail = [p for p in self._trail if p.age(current) < self._trail_max_age]
        if len(self._trail) != old_trail_len:
            needs_update = True

        if needs_update:
            self.update()
        elif not self._current_animation and not self._ripples and not self._trail:
            # Nothing to animate, stop timer
            pass  # Keep timer running for smooth experience

    def paintEvent(self, event):
        """Render the ghost cursor, trail, and effects."""
        if not self._enabled:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        current = self._current_time()

        # Draw motion trail
        self._draw_trail(painter, current)

        # Draw click ripples
        for ripple in self._ripples:
            self._draw_ripple(painter, ripple, current)

        # Draw ghost cursor
        if self._cursor_visible:
            self._draw_cursor(painter)

        painter.end()

    def _draw_trail(self, painter: QPainter, current_time: float):
        """Draw the motion trail as fading afterimages."""
        if len(self._trail) < 2:
            return

        # Draw trail as fading line segments
        for i in range(1, len(self._trail)):
            prev = self._trail[i - 1]
            curr = self._trail[i]

            # Calculate opacity based on age (newer = more opaque)
            age = curr.age(current_time)
            alpha = int(100 * (1 - age / self._trail_max_age))

            if alpha > 0:
                color = QColor(self._trail_color)
                color.setAlpha(alpha)

                pen = QPen(color)
                pen.setWidth(2)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(prev.position, curr.position)

    def _draw_ripple(self, painter: QPainter, ripple: ClickRipple, current_time: float):
        """Draw a click ripple effect."""
        progress = ripple.progress(current_time)

        # Radius expands, opacity fades
        radius = ripple.max_radius * progress
        alpha = int(ripple.color.alpha() * (1 - progress))

        if alpha > 0:
            color = QColor(ripple.color)
            color.setAlpha(alpha)

            # Draw outer ring
            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(ripple.center, radius, radius)

            # Draw inner filled circle (faster fade)
            inner_alpha = int(alpha * 0.3 * (1 - progress))
            if inner_alpha > 0:
                inner_color = QColor(color)
                inner_color.setAlpha(inner_alpha)
                painter.setBrush(QBrush(inner_color))
                painter.setPen(Qt.PenStyle.NoPen)
                inner_radius = radius * 0.5
                painter.drawEllipse(ripple.center, inner_radius, inner_radius)

    def _draw_cursor(self, painter: QPainter):
        """Draw the ghost cursor with state-dependent glow.

        Idle: Small, subtle, ephemeral white breathing with occasional sparkle poofs
        Insight flash: Brief bright burst right before movement
        Moving: Full colorful pink-orange dynamic magic
        """
        x, y = self._cursor_pos.x(), self._cursor_pos.y()

        # Cursor dimensions
        size = 20

        # Breathing calculation using sine wave for smooth pulsing
        breath = math.sin(self._breath_phase * math.pi * 2) * 0.5 + 0.5  # 0.0 to 1.0

        # Calculate idle (white) glow - small and ephemeral
        idle_base_radius = size * 0.8  # Much smaller base
        idle_radius = idle_base_radius + (breath * size * 0.2)  # Subtle growth
        idle_alpha = int(20 + breath * 25)  # Very soft alpha 20-45

        # Calculate moving (pink-orange) glow
        moving_base_radius = size * 2.0
        moving_radius = moving_base_radius + (breath * size * 0.8)  # More dramatic
        moving_alpha = int(60 + breath * 80)  # Brighter alpha 60-140

        # Interpolate pink-orange based on breath
        pink_r, pink_g, pink_b = 255, 120, 180
        orange_r, orange_g, orange_b = 255, 160, 80
        moving_r = int(pink_r + (orange_r - pink_r) * breath)
        moving_g = int(pink_g + (orange_g - pink_g) * breath)
        moving_b = int(pink_b + (orange_b - pink_b) * breath)

        # Blend between idle and moving based on intensity
        intensity = self._move_intensity
        glow_radius = idle_radius + (moving_radius - idle_radius) * intensity
        glow_alpha = int(idle_alpha + (moving_alpha - idle_alpha) * intensity)

        # Color: white when idle, pink-orange when moving
        # Also factor in sparkle poof (brief color burst when idle)
        sparkle = self._sparkle_intensity
        sparkle_r = int(pink_r + (orange_r - pink_r) * self._sparkle_color_mix)
        sparkle_g = int(pink_g + (orange_g - pink_g) * self._sparkle_color_mix)
        sparkle_b = int(pink_b + (orange_b - pink_b) * self._sparkle_color_mix)

        # Final color calculation
        # Base is white (255,255,255) for idle, moving_color for moving
        # Sparkle adds color burst when idle
        base_r = int(255 + (moving_r - 255) * intensity)
        base_g = int(255 + (moving_g - 255) * intensity)
        base_b = int(255 + (moving_b - 255) * intensity)

        # Add sparkle contribution (only visible when not moving)
        sparkle_contribution = sparkle * (1.0 - intensity)
        r = int(base_r + (sparkle_r - base_r) * sparkle_contribution)
        g = int(base_g + (sparkle_g - base_g) * sparkle_contribution)
        b = int(base_b + (sparkle_b - base_b) * sparkle_contribution)

        # Sparkle briefly boosts alpha
        if sparkle_contribution > 0:
            glow_alpha = min(255, int(glow_alpha + sparkle_contribution * 100))
            glow_radius += sparkle_contribution * size * 0.5

        # Insight flash - brief bright burst before movement
        # This is a "watch me!" moment that precedes the beautiful travel animation
        if self._insight_flash > 0:
            flash = self._insight_flash
            # Expand rapidly then contract
            flash_radius = size * 2.5 * flash
            flash_alpha = int(180 * flash)

            # Draw flash as expanding bright ring (cyan-white)
            flash_gradient = QRadialGradient(x, y, flash_radius)
            flash_gradient.setColorAt(0, QColor(220, 255, 255, flash_alpha))
            flash_gradient.setColorAt(0.3, QColor(180, 220, 255, int(flash_alpha * 0.7)))
            flash_gradient.setColorAt(0.6, QColor(150, 180, 220, int(flash_alpha * 0.3)))
            flash_gradient.setColorAt(1, QColor(150, 180, 220, 0))

            painter.setBrush(QBrush(flash_gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, y), flash_radius, flash_radius)

            # Also boost the main glow during flash
            glow_alpha = min(255, int(glow_alpha + flash * 60))
            glow_radius += flash * size * 0.3

        # Draw breathing glow (radial gradient)
        gradient = QRadialGradient(x, y, glow_radius)
        gradient.setColorAt(0, QColor(r, g, b, glow_alpha))
        gradient.setColorAt(0.4, QColor(r, g, b, int(glow_alpha * 0.5)))
        gradient.setColorAt(0.7, QColor(r, g, b, int(glow_alpha * 0.2)))
        gradient.setColorAt(1, QColor(r, g, b, 0))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(x, y), glow_radius, glow_radius)

        # Draw cursor arrow shape
        cursor_path = QPainterPath()
        cursor_path.moveTo(x, y)  # Tip
        cursor_path.lineTo(x, y + size)  # Down
        cursor_path.lineTo(x + size * 0.35, y + size * 0.7)  # Notch
        cursor_path.lineTo(x + size * 0.5, y + size * 0.9)  # Tail right
        cursor_path.lineTo(x + size * 0.65, y + size * 0.75)  # Tail up
        cursor_path.lineTo(x + size * 0.45, y + size * 0.55)  # Notch back
        cursor_path.lineTo(x + size * 0.7, y + size * 0.55)  # Right point
        cursor_path.closeSubpath()

        # Fill cursor with subtle color tint from the glow
        cursor_fill = QColor(
            int(180 + (r - 180) * 0.15),
            int(180 + (g - 180) * 0.15),
            int(180 + (b - 180) * 0.15),
            220
        )
        painter.setBrush(QBrush(cursor_fill))
        painter.setPen(QPen(QColor(100, 100, 100, 200), 1))
        painter.drawPath(cursor_path)


class GhostCursorController(QObject):
    """
    Controller that bridges ComputerUseController with GhostCursorOverlay.

    Intercepts computer use actions and visualizes them beautifully
    before (or during) execution.
    """

    def __init__(self, overlay: GhostCursorOverlay, parent=None):
        super().__init__(parent)
        self._overlay = overlay
        self._demo_mode = False
        self._action_delay = 200  # ms to pause after each action for visibility

    def set_demo_mode(self, enabled: bool):
        """Enable demo mode with visible cursor and delays."""
        self._demo_mode = enabled
        self._overlay.set_enabled(enabled)

    @property
    def demo_mode(self) -> bool:
        return self._demo_mode

    def visualize_move(self, x: int, y: int, callback=None):
        """Visualize a mouse move with animation."""
        if not self._demo_mode:
            if callback:
                callback()
            return

        self._overlay.move_to(x, y, callback)

    def visualize_click(self, x: int, y: int, button: str = "left", callback=None):
        """Visualize a click with cursor move and ripple."""
        if not self._demo_mode:
            if callback:
                callback()
            return

        def do_click():
            self._overlay.click_at(x, y, button)
            if callback:
                # Small delay after click for visibility
                QTimer.singleShot(self._action_delay, callback)

        self._overlay.move_to(x, y, do_click)

    def visualize_double_click(self, x: int, y: int, callback=None):
        """Visualize a double-click."""
        if not self._demo_mode:
            if callback:
                callback()
            return

        def do_clicks():
            self._overlay.click_at(x, y, "double")
            # Two ripples for double-click
            QTimer.singleShot(100, lambda: self._overlay.click_at(x, y, "double"))
            if callback:
                QTimer.singleShot(self._action_delay + 100, callback)

        self._overlay.move_to(x, y, do_clicks)

    def visualize_drag(self, start_x: int, start_y: int,
                       end_x: int, end_y: int, callback=None):
        """Visualize a drag operation."""
        if not self._demo_mode:
            if callback:
                callback()
            return

        def start_drag():
            self._overlay.click_at(start_x, start_y, "left")
            # Move to end position
            QTimer.singleShot(100, lambda: self._overlay.move_to(
                end_x, end_y,
                lambda: self._finish_drag(end_x, end_y, callback)
            ))

        self._overlay.move_to(start_x, start_y, start_drag)

    def _finish_drag(self, x: int, y: int, callback):
        """Finish drag with release ripple."""
        self._overlay.click_at(x, y, "left")
        if callback:
            QTimer.singleShot(self._action_delay, callback)

    def visualize_type(self, callback=None):
        """Visualize typing (subtle indicator)."""
        if not self._demo_mode:
            if callback:
                callback()
            return

        # Could add typing visualization here
        if callback:
            QTimer.singleShot(50, callback)

    def hide(self):
        """Hide the ghost cursor."""
        self._overlay.hide_cursor()


# Singleton management
_ghost_overlay: Optional[GhostCursorOverlay] = None
_ghost_controller: Optional[GhostCursorController] = None
_main_window_ref: Optional[QWidget] = None


class MainWindowResizeFilter(QObject):
    """Event filter to track main window resize events."""

    def __init__(self, overlay: GhostCursorOverlay, parent=None):
        super().__init__(parent)
        self._overlay = overlay

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.Resize:
            # Update overlay geometry to match window
            self._overlay.setGeometry(obj.rect())
        return False


def setup_ghost_cursor(main_window: QWidget) -> GhostCursorController:
    """
    Set up the ghost cursor system for a main window.

    Call this once during main window initialization.
    Returns the controller for enabling/disabling demo mode.
    """
    global _ghost_overlay, _ghost_controller, _main_window_ref

    _main_window_ref = main_window
    _ghost_overlay = GhostCursorOverlay(main_window)
    _ghost_overlay.setGeometry(main_window.rect())
    _ghost_controller = GhostCursorController(_ghost_overlay)

    # Install event filter to track resize
    resize_filter = MainWindowResizeFilter(_ghost_overlay, main_window)
    main_window.installEventFilter(resize_filter)

    print("[GhostCursor] Set up for main window")

    return _ghost_controller


def get_ghost_controller() -> Optional[GhostCursorController]:
    """Get the global ghost cursor controller."""
    return _ghost_controller


def get_ghost_overlay() -> Optional[GhostCursorOverlay]:
    """Get the global ghost cursor overlay."""
    return _ghost_overlay
