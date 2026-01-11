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
#   Panel Fold Manager
#
#   Manages the fold/unfold animation for transitioning between
#   App Mode (folded - panels hidden) and Studio Mode (unfolded).
#
#   "The studio was always there. Just waiting for you to unfold it."
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.panel_fold_manager
# PURPOSE:  Fold/Unfold Animation for App ↔ Studio Transition
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PanelFoldManager
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from typing import Callable, List, Optional

from PyQt6.QtCore import QObject, QTimer, QEasingCurve, pyqtSignal
from PyQt6.QtWidgets import QSplitter

logger = logging.getLogger(__name__)


class PanelFoldManager(QObject):
    """
    Manages panel fold/unfold animations.

    When folded (App Mode):
      - Left, right, and bottom panels have 0 width/height
      - Center panel fills the window
      - The "View Project" button is visible

    When unfolded (Studio Mode):
      - Panels animate to their default sizes
      - Full NoodleStudio interface is visible

    The animation uses ease-out for unfold (fast start, gentle landing)
    and ease-in for fold (gentle start, fast finish).

    Usage:
        manager = PanelFoldManager(main_splitter, top_splitter)
        manager.unfold()  # Reveal the studio
        manager.fold()    # Hide the studio
        manager.toggle()  # Toggle between modes
    """

    # Signals
    fold_started = pyqtSignal()       # Emitted when fold animation starts
    fold_complete = pyqtSignal()      # Emitted when fold animation completes
    unfold_started = pyqtSignal()     # Emitted when unfold animation starts
    unfold_complete = pyqtSignal()    # Emitted when unfold animation completes
    state_changed = pyqtSignal(bool)  # Emitted with is_folded state

    # Animation settings
    UNFOLD_DURATION_MS = 400
    FOLD_DURATION_MS = 300
    FRAME_INTERVAL_MS = 16  # ~60 FPS

    # Default panel sizes
    DEFAULT_LEFT = 250
    DEFAULT_RIGHT = 280
    DEFAULT_BOTTOM = 180

    def __init__(
        self,
        main_splitter: QSplitter,
        top_splitter: QSplitter,
        parent: Optional[QObject] = None
    ):
        """
        Initialize the fold manager.

        Args:
            main_splitter: Vertical splitter (top_area | bottom_panels)
            top_splitter: Horizontal splitter (left | center | right)
            parent: Optional parent QObject
        """
        super().__init__(parent)

        self._main_splitter = main_splitter
        self._top_splitter = top_splitter

        # State
        self._is_folded = False
        self._is_animating = False

        # Animation state
        self._animation_timer: Optional[QTimer] = None
        self._animation_progress = 0.0
        self._animation_start_sizes = {}
        self._animation_target_sizes = {}
        self._animation_duration = 0
        self._on_animation_complete: Optional[Callable] = None

        # Saved sizes (for returning to user's layout after fold)
        self._saved_top_sizes: List[int] = []
        self._saved_main_sizes: List[int] = []

    @property
    def is_folded(self) -> bool:
        """Check if panels are currently folded."""
        return self._is_folded

    @property
    def is_animating(self) -> bool:
        """Check if an animation is in progress."""
        return self._is_animating

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def fold(self, animated: bool = True):
        """
        Fold panels away (transition to App Mode).

        Args:
            animated: Whether to animate the transition
        """
        if self._is_folded or self._is_animating:
            return

        logger.info("Folding panels (entering App Mode)")
        self.fold_started.emit()

        # Save current sizes
        self._saved_top_sizes = self._top_splitter.sizes()
        self._saved_main_sizes = self._main_splitter.sizes()

        if animated:
            self._animate_fold()
        else:
            self._set_folded_sizes()
            self._is_folded = True
            self.fold_complete.emit()
            self.state_changed.emit(True)

    def unfold(self, animated: bool = True):
        """
        Unfold panels (transition to Studio Mode).

        Args:
            animated: Whether to animate the transition
        """
        if not self._is_folded or self._is_animating:
            return

        logger.info("Unfolding panels (entering Studio Mode)")
        self.unfold_started.emit()

        if animated:
            self._animate_unfold()
        else:
            self._set_unfolded_sizes()
            self._is_folded = False
            self.unfold_complete.emit()
            self.state_changed.emit(False)

    def toggle(self, animated: bool = True):
        """
        Toggle between folded and unfolded states.

        Args:
            animated: Whether to animate the transition
        """
        if self._is_folded:
            self.unfold(animated)
        else:
            self.fold(animated)

    def set_folded(self, folded: bool, animated: bool = True):
        """
        Set the fold state explicitly.

        Args:
            folded: True for folded (App Mode), False for unfolded (Studio Mode)
            animated: Whether to animate the transition
        """
        if folded:
            self.fold(animated)
        else:
            self.unfold(animated)

    def set_default_sizes(self, left: int, right: int, bottom: int):
        """
        Set default panel sizes for unfold.

        Args:
            left: Left panel width
            right: Right panel width
            bottom: Bottom panel height
        """
        self.DEFAULT_LEFT = left
        self.DEFAULT_RIGHT = right
        self.DEFAULT_BOTTOM = bottom

    # =========================================================================
    # ANIMATION
    # =========================================================================

    def _animate_fold(self):
        """Animate panels folding away."""
        self._is_animating = True

        # Calculate total sizes
        total_h = sum(self._top_splitter.sizes())
        total_v = sum(self._main_splitter.sizes())

        # Start and target
        self._animation_start_sizes = {
            'top': self._top_splitter.sizes(),
            'main': self._main_splitter.sizes(),
        }
        self._animation_target_sizes = {
            'top': [0, total_h, 0],  # Left=0, Center=all, Right=0
            'main': [total_v, 0],    # Top=all, Bottom=0
        }

        self._animation_duration = self.FOLD_DURATION_MS
        self._animation_progress = 0.0
        self._on_animation_complete = self._on_fold_complete

        self._start_animation_timer()

    def _animate_unfold(self):
        """Animate panels unfolding."""
        self._is_animating = True

        # Calculate total sizes
        total_h = sum(self._top_splitter.sizes())
        total_v = sum(self._main_splitter.sizes())

        # Determine target sizes (saved or default)
        if self._saved_top_sizes and sum(self._saved_top_sizes) > 0:
            target_top = self._saved_top_sizes
        else:
            center_width = total_h - self.DEFAULT_LEFT - self.DEFAULT_RIGHT
            target_top = [self.DEFAULT_LEFT, max(100, center_width), self.DEFAULT_RIGHT]

        if self._saved_main_sizes and sum(self._saved_main_sizes) > 0:
            target_main = self._saved_main_sizes
        else:
            top_height = total_v - self.DEFAULT_BOTTOM
            target_main = [max(100, top_height), self.DEFAULT_BOTTOM]

        self._animation_start_sizes = {
            'top': self._top_splitter.sizes(),
            'main': self._main_splitter.sizes(),
        }
        self._animation_target_sizes = {
            'top': target_top,
            'main': target_main,
        }

        self._animation_duration = self.UNFOLD_DURATION_MS
        self._animation_progress = 0.0
        self._on_animation_complete = self._on_unfold_complete

        self._start_animation_timer()

    def _start_animation_timer(self):
        """Start the animation timer."""
        if self._animation_timer:
            self._animation_timer.stop()

        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animation_tick)
        self._animation_timer.start(self.FRAME_INTERVAL_MS)

    def _animation_tick(self):
        """Called each animation frame."""
        # Update progress
        self._animation_progress += self.FRAME_INTERVAL_MS / self._animation_duration

        if self._animation_progress >= 1.0:
            self._animation_progress = 1.0
            self._apply_animation_frame(1.0)
            self._stop_animation()
            return

        # Apply easing
        if self._on_animation_complete == self._on_fold_complete:
            # Ease-in for fold (gentle start, fast finish)
            eased = self._ease_in(self._animation_progress)
        else:
            # Ease-out for unfold (fast start, gentle landing)
            eased = self._ease_out(self._animation_progress)

        self._apply_animation_frame(eased)

    def _apply_animation_frame(self, t: float):
        """
        Apply interpolated sizes for animation frame.

        Args:
            t: Progress 0.0 to 1.0 (after easing)
        """
        # Interpolate top splitter
        start_top = self._animation_start_sizes['top']
        target_top = self._animation_target_sizes['top']
        current_top = [
            int(start_top[i] + (target_top[i] - start_top[i]) * t)
            for i in range(len(start_top))
        ]
        self._top_splitter.setSizes(current_top)

        # Interpolate main splitter
        start_main = self._animation_start_sizes['main']
        target_main = self._animation_target_sizes['main']
        current_main = [
            int(start_main[i] + (target_main[i] - start_main[i]) * t)
            for i in range(len(start_main))
        ]
        self._main_splitter.setSizes(current_main)

    def _stop_animation(self):
        """Stop the animation and call completion handler."""
        if self._animation_timer:
            self._animation_timer.stop()
            self._animation_timer = None

        self._is_animating = False

        if self._on_animation_complete:
            self._on_animation_complete()

    def _on_fold_complete(self):
        """Called when fold animation completes."""
        self._is_folded = True
        logger.info("Fold complete (App Mode)")
        self.fold_complete.emit()
        self.state_changed.emit(True)

    def _on_unfold_complete(self):
        """Called when unfold animation completes."""
        self._is_folded = False
        logger.info("Unfold complete (Studio Mode)")
        self.unfold_complete.emit()
        self.state_changed.emit(False)

    # =========================================================================
    # INSTANT (NON-ANIMATED) TRANSITIONS
    # =========================================================================

    def _set_folded_sizes(self):
        """Set panel sizes to folded state immediately."""
        total_h = sum(self._top_splitter.sizes())
        total_v = sum(self._main_splitter.sizes())

        self._top_splitter.setSizes([0, total_h, 0])
        self._main_splitter.setSizes([total_v, 0])

    def _set_unfolded_sizes(self):
        """Set panel sizes to unfolded state immediately."""
        total_h = sum(self._top_splitter.sizes())
        total_v = sum(self._main_splitter.sizes())

        if self._saved_top_sizes:
            self._top_splitter.setSizes(self._saved_top_sizes)
        else:
            center = total_h - self.DEFAULT_LEFT - self.DEFAULT_RIGHT
            self._top_splitter.setSizes([self.DEFAULT_LEFT, center, self.DEFAULT_RIGHT])

        if self._saved_main_sizes:
            self._main_splitter.setSizes(self._saved_main_sizes)
        else:
            top = total_v - self.DEFAULT_BOTTOM
            self._main_splitter.setSizes([top, self.DEFAULT_BOTTOM])

    # =========================================================================
    # EASING FUNCTIONS
    # =========================================================================

    @staticmethod
    def _ease_out(t: float) -> float:
        """
        Ease-out cubic: fast start, gentle landing.

        Args:
            t: Progress 0.0 to 1.0

        Returns:
            Eased value 0.0 to 1.0
        """
        return 1 - (1 - t) ** 3

    @staticmethod
    def _ease_in(t: float) -> float:
        """
        Ease-in cubic: gentle start, fast finish.

        Args:
            t: Progress 0.0 to 1.0

        Returns:
            Eased value 0.0 to 1.0
        """
        return t ** 3


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
