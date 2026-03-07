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
#   Guide Performance Window
#
#   Pure renderer for noodling performances. Displays VRM
#   character(s), receives text and affect from facet assemblies.
#   Does NOT make LLM calls. Does NOT contain personality prompts.
#   All cognition happens in assemblies, orchestrated by
#   GuidePerformanceManager.
#
#   Supports single-noodling mode (one VRM viewport, 350x600) and
#   ensemble mode (two VRM viewports side by side, 650x650).
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.guide_performance_window
# PURPOSE:  Floating Guide Dialogue and VRM Panel
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   GuidePerformanceWindow
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QFrame
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QTextCursor, QColor, QTextCharFormat, QMouseEvent
    )
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


# =============================================================================
# Draggable Header
# =============================================================================

if QT_AVAILABLE:

    class _DraggableHeader(QLabel):
        """Header label that supports window dragging.

        Coordinates with GuidePerformanceWindow to pause follow-parent
        tracking during drag and recalculate offset on release so the
        window stays at the user-chosen position relative to the parent.
        """

        closeClicked = pyqtSignal()
        dragStarted = pyqtSignal()
        dragFinished = pyqtSignal()

        def __init__(self, text: str = "", parent=None):
            super().__init__(text, parent)
            self.drag_position = None

        def mousePressEvent(self, event: QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                self.drag_position = (
                    event.globalPosition().toPoint()
                    - self.window().frameGeometry().topLeft()
                )
                self.dragStarted.emit()
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event: QMouseEvent):
            if (event.buttons() == Qt.MouseButton.LeftButton
                    and self.drag_position is not None):
                self.window().move(
                    event.globalPosition().toPoint() - self.drag_position
                )
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                self.drag_position = None
                self.dragFinished.emit()
            super().mouseReleaseEvent(event)


# =============================================================================
# Thinking Indicator (minimal variant)
# =============================================================================

if QT_AVAILABLE:

    class _ThinkingIndicator(QFrame):
        """Compact thinking indicator with pulsing dot."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFrameStyle(QFrame.Shape.NoFrame)
            self._pulse_state = 0

            layout = QHBoxLayout(self)
            layout.setContentsMargins(8, 4, 8, 4)
            layout.setSpacing(6)

            self.dot = QLabel()
            self.dot.setFixedSize(6, 6)
            self._update_dot()
            layout.addWidget(self.dot)

            self.status_label = QLabel("")
            self.status_label.setStyleSheet(
                "color: #888888; font-family: 'SF Mono', monospace; font-size: 10px;"
            )
            layout.addWidget(self.status_label)
            layout.addStretch()

            self.setStyleSheet(
                "_ThinkingIndicator { background-color: #1A1A1A; }"
            )

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._pulse)
            self._timer.setInterval(400)
            self.hide()

        def _update_dot(self):
            colors = ['#555555', '#777777', '#999999', '#777777']
            color = colors[self._pulse_state % len(colors)]
            self.dot.setStyleSheet(
                f"background-color: {color}; border-radius: 3px;"
            )

        def _pulse(self):
            self._pulse_state = (self._pulse_state + 1) % 4
            self._update_dot()

        def set_status(self, text: str):
            self.status_label.setText(text)
            if not self.isVisible():
                self.show()
                self._timer.start()

        def clear(self):
            self._timer.stop()
            self.hide()


# =============================================================================
# Guide Performance Window
# =============================================================================

if QT_AVAILABLE:

    class GuidePerformanceWindow(QMainWindow):
        """
        Pure renderer for noodling performances.

        Supports two modes:

        **Single mode** (default): One VRM viewport, 350x600 window.
        Backward compatible with existing single-noodling performances.

        **Ensemble mode**: Two VRM viewports side by side, 650x650 window.
        Each viewport is independently addressable by noodling_id.
        Dialogue interleaves noodling text with name prefixes.

        Usage (single):
            window = GuidePerformanceWindow(parent_window=main_window)
            window.set_vrm("/path/to/ajo.vrm")
            window.show()

        Usage (ensemble):
            window = GuidePerformanceWindow(parent_window=main_window,
                                            ensemble_mode=True)
            window.set_vrm("/path/to/ajo.vrm", noodling_id='ajo')
            window.set_vrm("/path/to/yuki.vrm", noodling_id='yuki')
            window.show()
        """

        # Signal: user submitted a message for assembly execution
        messageSubmitted = pyqtSignal(str)

        # Signal: user sent a message (for channel bus forwarding)
        messageSent = pyqtSignal(str)

        # Signal: user clicked a performer name in the ensemble name bar
        noodlingSelected = pyqtSignal(str)

        def __init__(
            self,
            parent_window: QMainWindow,
            size: Tuple[int, int] = (350, 600),
            offset: Tuple[int, int] = (10, 60),
            ensemble_mode: bool = False,
        ):
            """
            Initialize the guide performance window.

            Args:
                parent_window: The main window to follow
                size: (width, height) of this window
                offset: (x, y) offset from right edge of parent
                ensemble_mode: If True, create two VRM viewports side by side
            """
            super().__init__()
            self.parent_window = parent_window
            self._ensemble_mode = ensemble_mode
            self._offset = offset

            # In ensemble mode with default single-mode size, widen
            if ensemble_mode and size == (350, 600):
                size = (900, 650)
            self._size = size

            # Frameless, stays on top, no taskbar entry
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint |
                Qt.WindowType.Tool
            )

            self.setFixedSize(*size)

            # Pre-init state needed by _build_ui / _build_ensemble_vrm_area
            self._performer_labels = {}

            self._build_ui()

            # Position once at the right edge of the parent, then stay put.
            if parent_window:
                geo = parent_window.geometry()
                x = geo.right() - size[0] - offset[0]
                y = geo.top() + offset[1]
                self.move(x, y)

            # VRM viewports (slot_key -> VRMViewportWidget)
            self._vrm_viewports = {}

            # Legacy alias for single mode (set when VRM loads)
            self._vrm_viewport = None

            # Ensemble: slot assignment for noodling_ids
            self._noodling_to_slot = {}

            # Noodling text colors for dialogue (noodling_id -> QColor)
            self._noodling_colors = {
                'ajo': QColor(176, 176, 176),       # #B0B0B0 warm gray
                'krampus': QColor(176, 160, 144),    # #B0A090 warm brownish gray
                'juanita': QColor(160, 176, 160),    # #A0B0A0 subtle sage gray
                'default': QColor(176, 176, 176),
            }

            # Track which noodling is currently in a typed-text block
            self._current_typing_noodling = None

            logger.info(
                f"GuidePerformanceWindow created (ensemble={ensemble_mode})"
            )

        # =================================================================
        # ENSEMBLE PROPERTIES
        # =================================================================

        @property
        def ensemble_mode(self) -> bool:
            """Whether this window is in ensemble mode."""
            return self._ensemble_mode

        # =================================================================
        # UI CONSTRUCTION
        # =================================================================

        def _build_ui(self):
            """Build the window layout."""
            container = QWidget()
            container.setStyleSheet("background-color: #020204;")
            main_layout = QVBoxLayout(container)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)

            # --- Header ---
            header_frame = QFrame()
            header_frame.setStyleSheet("""
                QFrame {
                    background-color: #252525;
                    border-bottom: 1px solid #333333;
                }
            """)
            header_layout = QHBoxLayout(header_frame)
            header_layout.setContentsMargins(10, 6, 6, 6)
            header_layout.setSpacing(0)

            self.header_label = _DraggableHeader("Performance")
            self.header_label.setStyleSheet("""
                color: #B0B0B0;
                font-family: 'SF Mono', 'Source Code Pro', monospace;
                font-size: 12px;
                font-weight: bold;
                background: transparent;
            """)
            header_layout.addWidget(self.header_label, stretch=1)

            close_btn = QPushButton("x")
            close_btn.setFixedSize(22, 22)
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    color: #888888;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    color: #CCCCCC;
                    background-color: #3A3A3A;
                    border-radius: 3px;
                }
            """)
            close_btn.clicked.connect(self.hide)
            header_layout.addWidget(close_btn)

            main_layout.addWidget(header_frame)

            # --- VRM Viewport Area ---
            self._vrm_containers = {}
            self._vrm_container_layouts = {}
            self._vrm_placeholders = {}

            if self._ensemble_mode:
                self._build_ensemble_vrm_area(main_layout)
            else:
                self._build_single_vrm_area(main_layout)

            # --- Thinking Indicator ---
            self.thinking_indicator = _ThinkingIndicator()
            main_layout.addWidget(self.thinking_indicator)

            # --- Dialogue Display ---
            self.dialogue_view = QTextEdit()
            self.dialogue_view.setReadOnly(True)
            self.dialogue_view.setStyleSheet("""
                QTextEdit {
                    background-color: #1A1A1A;
                    border: none;
                    color: #B0B0B0;
                    font-family: 'SF Mono', 'Source Code Pro', monospace;
                    font-size: 12px;
                    padding: 8px;
                    selection-background-color: #3A3A3A;
                }
                QScrollBar:vertical {
                    background: #1A1A1A;
                    width: 6px;
                }
                QScrollBar::handle:vertical {
                    background: #3A3A3A;
                    border-radius: 3px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)
            main_layout.addWidget(self.dialogue_view, stretch=1)

            # --- Input Area ---
            input_frame = QFrame()
            input_frame.setStyleSheet("""
                QFrame {
                    background-color: #2A2A2A;
                    border-top: 1px solid #3A3A3A;
                }
            """)
            input_layout = QHBoxLayout(input_frame)
            input_layout.setContentsMargins(6, 6, 6, 6)
            input_layout.setSpacing(6)

            placeholder = (
                "Talk to the ensemble..."
                if self._ensemble_mode
                else "Talk to Guide..."
            )
            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText(placeholder)
            self.input_field.setStyleSheet("""
                QLineEdit {
                    background-color: #1E1E1E;
                    border: 1px solid #3A3A3A;
                    border-radius: 4px;
                    color: #D2D2D2;
                    padding: 6px 10px;
                    font-family: 'SF Mono', 'Source Code Pro', monospace;
                    font-size: 12px;
                }
                QLineEdit:focus {
                    border: 1px solid #4FC3F7;
                }
            """)
            self.input_field.returnPressed.connect(self._on_send)
            input_layout.addWidget(self.input_field)

            self.send_button = QPushButton("Send")
            self.send_button.setStyleSheet("""
                QPushButton {
                    background-color: #4FC3F7;
                    border: none;
                    border-radius: 4px;
                    color: #1A1A1A;
                    padding: 6px 12px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #67D3FF; }
                QPushButton:pressed { background-color: #3AA3D7; }
                QPushButton:disabled {
                    background-color: #3A3A3A;
                    color: #666;
                }
            """)
            self.send_button.clicked.connect(self._on_send)
            input_layout.addWidget(self.send_button)

            main_layout.addWidget(input_frame)

            self.setCentralWidget(container)

        def _build_single_vrm_area(self, main_layout: QVBoxLayout):
            """Build the single-noodling VRM viewport area."""
            self.vrm_container = QFrame()
            self.vrm_container.setFixedHeight(250)
            self.vrm_container.setStyleSheet("""
                QFrame {
                    background-color: #020204;
                    border: none;
                }
            """)
            self.vrm_container_layout = QVBoxLayout(self.vrm_container)
            self.vrm_container_layout.setContentsMargins(0, 0, 0, 0)

            placeholder = QLabel("No character loaded")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                "color: #555555; font-size: 11px; background: transparent;"
            )
            self.vrm_container_layout.addWidget(placeholder)

            # Register in container dicts
            self._vrm_containers['default'] = self.vrm_container
            self._vrm_container_layouts['default'] = self.vrm_container_layout
            self._vrm_placeholders['default'] = placeholder

            # Legacy alias
            self._vrm_placeholder = placeholder

            main_layout.addWidget(self.vrm_container)

        def _build_ensemble_vrm_area(self, main_layout: QVBoxLayout):
            """Build the ensemble VRM viewport area (N viewports side by side)."""
            # --- Performer name bar ---
            name_bar = QFrame()
            name_bar.setFixedHeight(28)
            name_bar.setStyleSheet(
                "QFrame { background-color: #0A0A0C; border: none; }"
            )
            name_bar_layout = QHBoxLayout(name_bar)
            name_bar_layout.setContentsMargins(8, 2, 8, 2)
            name_bar_layout.setSpacing(8)

            for slot_key in ('left', 'center', 'right'):
                label = QLabel("\u2014")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setCursor(Qt.CursorShape.PointingHandCursor)
                label.setStyleSheet(
                    "color: #555; font-size: 11px; font-weight: bold; "
                    "background: transparent; padding: 2px 12px;"
                )
                label.installEventFilter(self)
                name_bar_layout.addWidget(label, stretch=1)
                self._performer_labels[slot_key] = label

            main_layout.addWidget(name_bar)

            # --- VRM viewports ---
            vrm_row = QFrame()
            vrm_row.setFixedHeight(280)
            vrm_row.setStyleSheet(
                "QFrame { background-color: #020204; border: none; }"
            )
            vrm_row_layout = QHBoxLayout(vrm_row)
            vrm_row_layout.setContentsMargins(0, 0, 0, 0)
            vrm_row_layout.setSpacing(1)

            for slot_key in ('left', 'center', 'right'):
                slot_container = QFrame()
                slot_container.setStyleSheet(
                    "QFrame { background-color: #020204; border: none; }"
                )
                slot_layout = QVBoxLayout(slot_container)
                slot_layout.setContentsMargins(0, 0, 0, 0)

                placeholder = QLabel("No character loaded")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                placeholder.setStyleSheet(
                    "color: #555555; font-size: 11px; "
                    "background: transparent;"
                )
                slot_layout.addWidget(placeholder)

                self._vrm_containers[slot_key] = slot_container
                self._vrm_container_layouts[slot_key] = slot_layout
                self._vrm_placeholders[slot_key] = placeholder

                vrm_row_layout.addWidget(slot_container, stretch=1)

            # Store reference for layout access
            self._vrm_row = vrm_row

            main_layout.addWidget(vrm_row)

        # =================================================================
        # VRM SLOT ROUTING
        # =================================================================

        def _get_slot(self, noodling_id: str = 'default') -> str:
            """
            Get the container slot key for a noodling_id.

            In single mode, always returns 'default'.
            In ensemble mode, assigns noodling_ids to 'left'/'center'/'right'
            slots in the order they are first seen.

            Args:
                noodling_id: Identifier for the noodling

            Returns:
                Slot key ('default', 'left', or 'right')
            """
            if not self._ensemble_mode:
                return 'default'

            # Already assigned?
            if noodling_id in self._noodling_to_slot:
                return self._noodling_to_slot[noodling_id]

            # Assign next available slot
            used_slots = set(self._noodling_to_slot.values())
            for slot in ('left', 'center', 'right'):
                if slot not in used_slots:
                    self._noodling_to_slot[noodling_id] = slot
                    return slot

            # All slots taken — return left as fallback
            return 'left'

        # =================================================================
        # PERFORMER NAME BAR
        # =================================================================

        def set_performer_name(self, noodling_id: str, name: str):
            """Set the display name for a noodling's stage slot."""
            slot = self._get_slot(noodling_id)
            label = self._performer_labels.get(slot)
            if label:
                label.setText(name)
                label.setStyleSheet(
                    "color: #D2D2D2; font-size: 11px; font-weight: bold; "
                    "background: transparent; padding: 2px 12px;"
                )

        def set_active_speaker(self, noodling_id: str = None):
            """Highlight which noodling is currently speaking/thinking.

            Sets name label styling and dims non-speaking VRM containers
            for a subtle stage-light spotlight effect.
            """
            active_slot = self._noodling_to_slot.get(noodling_id) if noodling_id else None

            for slot, label in self._performer_labels.items():
                if active_slot == slot:
                    label.setStyleSheet(
                        "color: #E8C547; font-size: 11px; font-weight: bold; "
                        "background: #1A1A0A; border-radius: 3px; "
                        "padding: 2px 12px;"
                    )
                else:
                    label.setStyleSheet(
                        "color: #888; font-size: 11px; font-weight: bold; "
                        "background: transparent; padding: 2px 12px;"
                    )

            # Dim non-speaking VRM containers (subtle background shift)
            for slot, container in self._vrm_containers.items():
                if slot == 'default':
                    continue  # Single mode container, skip
                if noodling_id is None:
                    # No speaker -- restore all containers to normal
                    container.setStyleSheet(
                        "QFrame { background-color: #020204; border: none; }"
                    )
                elif slot == active_slot:
                    # Active speaker -- full brightness
                    container.setStyleSheet(
                        "QFrame { background-color: #020204; border: none; }"
                    )
                else:
                    # Non-speaker -- subtle dim
                    container.setStyleSheet(
                        "QFrame { background-color: #010102; border: none; }"
                    )

        def eventFilter(self, obj, event):
            """Handle clicks on performer name labels."""
            from PyQt6.QtCore import QEvent
            if event.type() == QEvent.Type.MouseButtonPress:
                # Check if the clicked object is a performer label
                for slot, label in self._performer_labels.items():
                    if obj is label:
                        # Reverse lookup: slot -> noodling_id
                        for nid, s in self._noodling_to_slot.items():
                            if s == slot:
                                self.noodlingSelected.emit(nid)
                                return True
                        break
            return super().eventFilter(obj, event)

        # =================================================================
        # VRM
        # =================================================================

        def set_vrm(self, vrm_path: str, noodling_id: str = 'default'):
            """
            Load a VRM character model into a viewport.

            In single mode, noodling_id is ignored (uses default viewport).
            In ensemble mode, each noodling_id is assigned a viewport slot.

            Args:
                vrm_path: Path to .vrm file
                noodling_id: Identifier for the noodling
            """
            slot = self._get_slot(noodling_id)
            slot_container = self._vrm_containers.get(slot)
            slot_layout = self._vrm_container_layouts.get(slot)

            if not slot_container or not slot_layout:
                logger.warning(f"No VRM container for slot '{slot}'")
                return

            try:
                from .components.vrm_viewport import VRMViewport, VRMViewportWidget

                component = VRMViewport(f"character_{noodling_id}")
                component.transparent = False
                component.background = "#020204"
                component.vrm_path = vrm_path
                component.show_grid = False
                component.show_skeleton = False
                component.interactive = False

                # Portrait camera (head/upper body)
                component.camera.distance = 2.0
                component.camera.elevation = 5
                component.camera.azimuth = 175
                component.camera.target = (0.0, 0.85, 0.0)

                # Remove placeholder
                placeholder = self._vrm_placeholders.get(slot)
                if placeholder:
                    placeholder.setParent(None)
                    self._vrm_placeholders[slot] = None
                    # Clear legacy alias (single mode)
                    if not self._ensemble_mode:
                        self._vrm_placeholder = None

                # Remove old viewport if any
                old_viewport = self._vrm_viewports.get(slot)
                if old_viewport:
                    old_viewport.setParent(None)

                # Create and add the viewport widget
                viewport = VRMViewportWidget(component, slot_container)
                slot_layout.addWidget(viewport)
                self._vrm_viewports[slot] = viewport

                # Legacy alias for single mode
                if not self._ensemble_mode:
                    self._vrm_viewport = viewport

                logger.info(
                    f"VRM loaded for {noodling_id} (slot={slot}): {vrm_path}"
                )

            except Exception as e:
                logger.error(f"VRM load failed for {noodling_id}: {e}")
                placeholder = self._vrm_placeholders.get(slot)
                if placeholder:
                    placeholder.setText(f"VRM load failed: {e}")

        def show_name_card(self, noodling_id: str, noodling_name: str):
            """Show a name card for a noodling without a VRM.

            Displays the noodling's name on a dark card in the viewport
            slot, like a Zoom call with camera off.

            Args:
                noodling_id: Instance identifier
                noodling_name: Display name for the card
            """
            slot = self._get_slot(noodling_id)
            slot_layout = self._vrm_container_layouts.get(slot)

            if not slot_layout:
                logger.warning(f"No container for name card slot '{slot}'")
                return

            # Remove existing viewport if any
            old_viewport = self._vrm_viewports.get(slot)
            if old_viewport:
                old_viewport.setParent(None)
                self._vrm_viewports[slot] = None

            # Get or recreate placeholder
            placeholder = self._vrm_placeholders.get(slot)
            if placeholder is None:
                placeholder = QLabel()
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                slot_layout.addWidget(placeholder)
                self._vrm_placeholders[slot] = placeholder

            placeholder.setText(noodling_name)
            placeholder.setStyleSheet(
                "QLabel {"
                "  color: #D2D2D2;"
                "  background: #2a2a2a;"
                "  font-size: 18px;"
                "  font-weight: 500;"
                "}"
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.show()
            logger.info(f"Name card shown for {noodling_id}: {noodling_name}")

        def set_muscles(self, muscles: Dict[str, float],
                        noodling_id: str = 'default'):
            """Apply muscle values to a VRM character."""
            slot = self._get_slot(noodling_id)
            viewport = self._vrm_viewports.get(slot)
            if viewport:
                viewport.set_muscles(muscles)

        def set_blend_shapes(self, shapes: Dict[str, float],
                             noodling_id: str = 'default'):
            """Apply blend shape weights to a VRM character."""
            slot = self._get_slot(noodling_id)
            viewport = self._vrm_viewports.get(slot)
            if viewport:
                viewport.set_blend_shapes(shapes)

        def set_speaking_mode(self, active: bool, intensity: float = 0.7,
                              noodling_id: str = 'default'):
            """
            Toggle speaking animation on a VRM character.

            Args:
                active: True to enable speaking animation
                intensity: Speaking animation intensity (0.0 to 1.0)
                noodling_id: Which noodling's viewport to animate
            """
            slot = self._get_slot(noodling_id)
            viewport = self._vrm_viewports.get(slot)
            if viewport:
                viewport.set_speaking_mode(active, intensity)

        # =================================================================
        # DIALOGUE DISPLAY
        # =================================================================

        def show_play_header(self, title: str):
            """
            Set the header text to the play title.

            Args:
                title: Title of the play being performed
            """
            self.header_label.setText(title)

        def clear_dialogue(self):
            """Clear the dialogue display."""
            self.dialogue_view.clear()

        def dim_dialogue(self):
            """Dim all existing dialogue text to indicate stopped state."""
            cursor = self.dialogue_view.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(100, 100, 100))
            cursor.mergeCharFormat(fmt)

        def set_input_enabled(self, enabled: bool):
            """Enable or disable the message input area.

            Args:
                enabled: True to enable input, False to disable
            """
            self.input_field.setEnabled(enabled)
            self.send_button.setEnabled(enabled)

        def display_narration(self, text: str):
            """Display narration text -- no speaker, dimmed italic."""
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(136, 136, 136))  # #888
            fmt.setFontItalic(True)
            cursor.setCharFormat(fmt)
            cursor.insertText(f"{text}\n\n")
            self.dialogue_view.setTextCursor(cursor)
            self._scroll_to_bottom()

        # -- Noodling-aware dialogue methods (ensemble + single) --

        def begin_noodling_text(self, noodling_id: str, name: Optional[str]):
            """
            Begin a typed-text block for character-by-character delivery.

            In ensemble mode with a name, inserts "Name: " prefix.
            In single mode, inserts the guide icon prefix.

            Args:
                noodling_id: Identifier for the noodling
                name: Display name (e.g. "Ajo", "Yuki"), or None
            """
            self._current_typing_noodling = noodling_id

            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            color = self._noodling_colors.get(
                noodling_id, self._noodling_colors['default']
            )
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            cursor.setCharFormat(fmt)

            if self._ensemble_mode and name:
                cursor.insertText(f"{name}: ")
            else:
                cursor.insertText("\ua69c ")

            self.dialogue_view.setTextCursor(cursor)

        def end_noodling_text(self):
            """Finalize a typed-text block (add trailing newlines)."""
            self._current_typing_noodling = None

            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("\n\n")
            self.dialogue_view.setTextCursor(cursor)

        def append_noodling_text(self, noodling_id: str,
                                 name: Optional[str], text: str):
            """
            Append noodling text all at once with optional name prefix.

            In ensemble mode with a name, prefixes with "Name: ".
            In single mode, prefixes with the guide icon.

            Args:
                noodling_id: Identifier for the noodling
                name: Display name (e.g. "Ajo", "Yuki"), or None
                text: The message text
            """
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            color = self._noodling_colors.get(
                noodling_id, self._noodling_colors['default']
            )
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            cursor.setCharFormat(fmt)

            if self._ensemble_mode and name:
                cursor.insertText(f"{name}: {text}\n\n")
            else:
                cursor.insertText(f"\ua69c {text}\n\n")

            self.dialogue_view.setTextCursor(cursor)
            self._scroll_to_bottom()

        # -- Backward-compatible dialogue methods (single mode) --

        def append_guide_text(self, text: str):
            """
            Append guide (assistant) text to the dialogue (all at once).

            Backward-compatible wrapper around append_noodling_text().

            Args:
                text: Guide's message text (from assembly OUTGOING)
            """
            self.append_noodling_text('default', None, text)

        def begin_guide_text(self):
            """
            Begin a typed-text block for character-by-character delivery.

            Backward-compatible wrapper around begin_noodling_text().
            """
            self.begin_noodling_text('default', None)

        def append_character(self, char: str):
            """
            Append a single character to the dialogue for typed-text effect.

            Uses the color of the current typing noodling (set by
            begin_noodling_text or begin_guide_text).

            Args:
                char: Single character to append
            """
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            nid = self._current_typing_noodling or 'default'
            color = self._noodling_colors.get(
                nid, self._noodling_colors['default']
            )
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            cursor.setCharFormat(fmt)
            cursor.insertText(char)

            self.dialogue_view.setTextCursor(cursor)
            self._scroll_to_bottom()

        def end_guide_text(self):
            """
            Finalize a typed-text block (add trailing newlines).

            Backward-compatible wrapper around end_noodling_text().
            """
            self.end_noodling_text()

        def append_user_text(self, text: str):
            """
            Append user text to the dialogue.

            Args:
                text: User's message text
            """
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            fmt = QTextCharFormat()
            fmt.setBackground(QColor(60, 60, 60))
            fmt.setForeground(QColor(200, 200, 200))
            cursor.setCharFormat(fmt)

            lines = text.split('\n')
            for i, line in enumerate(lines):
                if i > 0:
                    cursor.insertText('\n')
                cursor.insertText(f"\u2b44 {line}" if line.strip() else "\u2b44")

            cursor.setCharFormat(QTextCharFormat())
            cursor.insertText('\n\n')

            self.dialogue_view.setTextCursor(cursor)
            self._scroll_to_bottom()

        def _scroll_to_bottom(self):
            """Scroll dialogue to bottom."""
            QTimer.singleShot(10, self._do_scroll)

        def _do_scroll(self):
            scrollbar = self.dialogue_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        # =================================================================
        # THINKING / BUSY STATE
        # =================================================================

        def set_thinking(self, noodling_id: str, name: Optional[str],
                         thinking: bool):
            """
            Show or hide the thinking indicator with a noodling name.

            Args:
                noodling_id: Identifier for the noodling
                name: Display name (e.g. "Ajo", "Yuki"), or None
                thinking: True to show, False to hide
            """
            if thinking:
                text = f"{name} is thinking..." if name else "Thinking..."
                self.thinking_indicator.set_status(text)
            else:
                self.thinking_indicator.clear()

        def set_busy(self, busy: bool, name: Optional[str] = None):
            """
            Toggle busy state (thinking indicator and input).

            Args:
                busy: True when assembly is executing, False when done
                name: Optional noodling name for the thinking indicator
            """
            self.input_field.setEnabled(not busy)
            self.send_button.setEnabled(not busy)
            if busy:
                text = f"{name} is thinking..." if name else "Thinking..."
                self.thinking_indicator.set_status(text)
            else:
                self.thinking_indicator.clear()
                self.input_field.setFocus()

        # =================================================================
        # SENDING MESSAGES
        # =================================================================

        def _on_send(self):
            """Handle user pressing Enter or clicking Send."""
            message = self.input_field.text().strip()
            if not message:
                return

            self.input_field.clear()

            # Display user message
            self.append_user_text(message)

            # Signal to manager for assembly execution
            self.messageSubmitted.emit(message)

            # Signal for channel bus forwarding
            self.messageSent.emit(message)

        def _show_error(self, text: str):
            """Show an error in the dialogue."""
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(200, 100, 100))
            cursor.setCharFormat(fmt)
            cursor.insertText(f"Error: {text}\n\n")
            self.dialogue_view.setTextCursor(cursor)
            self._scroll_to_bottom()

        # =================================================================
        # LIFECYCLE
        # =================================================================

        def closeEvent(self, event):
            """Clean up on close."""
            super().closeEvent(event)


# =============================================================================
# Fallback when Qt not available
# =============================================================================

if not QT_AVAILABLE:

    class GuidePerformanceWindow:
        """Stub when PyQt6 is not available."""

        messageSent = None
        messageSubmitted = None

        def __init__(self, *args, **kwargs):
            logger.warning("GuidePerformanceWindow requires PyQt6")

        def show(self): pass
        def hide(self): pass
        def close(self): pass
        def set_vrm(self, vrm_path, noodling_id='default'): pass
        def show_name_card(self, noodling_id, noodling_name): pass
        def set_muscles(self, muscles, noodling_id='default'): pass
        def set_blend_shapes(self, shapes, noodling_id='default'): pass
        def set_speaking_mode(self, active, intensity=0.7,
                              noodling_id='default'): pass
        def set_busy(self, busy, name=None): pass
        def set_thinking(self, noodling_id, name, thinking): pass
        def show_play_header(self, title): pass
        def clear_dialogue(self): pass
        def dim_dialogue(self): pass
        def set_input_enabled(self, enabled): pass
        def display_narration(self, text): pass
        def append_guide_text(self, text): pass
        def append_noodling_text(self, noodling_id, name, text): pass
        def begin_noodling_text(self, noodling_id, name): pass
        def end_noodling_text(self): pass
        def begin_guide_text(self): pass
        def append_character(self, char): pass
        def end_guide_text(self): pass
        def append_user_text(self, text): pass


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
