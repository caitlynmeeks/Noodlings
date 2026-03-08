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
#   Performance Panel
#
#   Pure renderer for noodling performances. Displays VRM
#   character(s), receives text and affect from facet assemblies.
#   Does NOT make LLM calls. Does NOT contain personality prompts.
#   All cognition happens in assemblies, orchestrated by
#   GuidePerformanceManager.
#
#   Embedded as a center-pane tab (like Unity's Game tab).
#   Always builds 3 VRM slots; use set_ensemble_visible() to
#   hide center/right for single-performer plays.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.guide_performance_window
# PURPOSE:  Embeddable Performance Panel (center-pane tab)
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PerformancePanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QFrame
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QTextCursor, QColor, QTextCharFormat
    )
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


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
# Performance Panel
# =============================================================================

if QT_AVAILABLE:

    class PerformancePanel(QWidget):
        """
        Pure renderer for noodling performances.

        Embeddable QWidget panel for the center-pane tab stack.
        Always builds 3 VRM viewport slots (left/center/right).
        Use set_ensemble_visible() to hide center/right for
        single-performer plays.

        Usage:
            panel = PerformancePanel(ensemble_mode=True)
            panel.set_vrm("/path/to/ajo.vrm", noodling_id='ajo')
        """

        # Signal: user submitted a message for assembly execution
        messageSubmitted = pyqtSignal(str)

        # Signal: user sent a message (for channel bus forwarding)
        messageSent = pyqtSignal(str)

        # Signal: user clicked a performer name in the ensemble name bar
        noodlingSelected = pyqtSignal(str)

        def __init__(self, ensemble_mode: bool = True, parent=None):
            """
            Initialize the performance panel.

            Args:
                ensemble_mode: If True, show all 3 VRM slots. If False,
                    hide center/right slots for single-performer display.
                parent: Parent widget (typically the center tab widget)
            """
            super().__init__(parent)
            self._ensemble_mode = ensemble_mode

            # Pre-init state needed by _build_ui / _build_ensemble_vrm_area
            self._performer_labels = {}

            self._build_ui()

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

            # Track current segment type for append_character() styling
            self._current_char_fmt = 'spoken'  # 'spoken' | 'action' | 'thought'

            # Apply initial ensemble visibility
            if not ensemble_mode:
                self.set_ensemble_visible(False)

            logger.info(
                f"PerformancePanel created (ensemble={ensemble_mode})"
            )

        # =================================================================
        # ENSEMBLE PROPERTIES
        # =================================================================

        @property
        def ensemble_mode(self) -> bool:
            """Whether this panel is in ensemble mode."""
            return self._ensemble_mode

        # =================================================================
        # UI CONSTRUCTION
        # =================================================================

        def _build_ui(self):
            """Build the panel layout."""
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

            self.header_label = QLabel("Performance")
            self.header_label.setStyleSheet("""
                color: #B0B0B0;
                font-family: 'SF Mono', 'Source Code Pro', monospace;
                font-size: 12px;
                font-weight: bold;
                background: transparent;
            """)
            header_layout.addWidget(self.header_label, stretch=1)

            main_layout.addWidget(header_frame)

            # --- VRM Viewport Area (always ensemble layout: 3 slots) ---
            self._vrm_containers = {}
            self._vrm_container_layouts = {}
            self._vrm_placeholders = {}

            self._build_ensemble_vrm_area(main_layout)

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

            # Set layout on self (QWidget, not QMainWindow)
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.addWidget(container)

        def _build_ensemble_vrm_area(self, main_layout: QVBoxLayout):
            """Build the ensemble VRM viewport area (3 viewports side by side)."""
            # --- Performer name bar ---
            self._name_bar = QFrame()
            self._name_bar.setFixedHeight(28)
            self._name_bar.setStyleSheet(
                "QFrame { background-color: #0A0A0C; border: none; }"
            )
            name_bar_layout = QHBoxLayout(self._name_bar)
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

            main_layout.addWidget(self._name_bar)

            # --- VRM viewports ---
            vrm_row = QFrame()
            vrm_row.setMinimumHeight(200)
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
        # ENSEMBLE VISIBILITY
        # =================================================================

        def set_ensemble_visible(self, visible: bool):
            """Show or hide the name bar and center/right VRM containers.

            For single-performer plays, call with visible=False to show
            only the left VRM slot.

            Args:
                visible: True for full ensemble, False for single-performer
            """
            self._ensemble_mode = visible

            # Name bar
            self._name_bar.setVisible(visible)

            # Center and right VRM containers
            for slot_key in ('center', 'right'):
                container = self._vrm_containers.get(slot_key)
                if container:
                    container.setVisible(visible)

        # =================================================================
        # VRM SLOT ROUTING
        # =================================================================

        def _get_slot(self, noodling_id: str = 'default') -> str:
            """
            Get the container slot key for a noodling_id.

            In single mode, always returns 'left' (the only visible slot).
            In ensemble mode, assigns noodling_ids to 'left'/'center'/'right'
            slots in the order they are first seen.

            Args:
                noodling_id: Identifier for the noodling

            Returns:
                Slot key ('left', 'center', or 'right')
            """
            if not self._ensemble_mode:
                return 'left'

            # Already assigned?
            if noodling_id in self._noodling_to_slot:
                return self._noodling_to_slot[noodling_id]

            # Assign next available slot
            used_slots = set(self._noodling_to_slot.values())
            for slot in ('left', 'center', 'right'):
                if slot not in used_slots:
                    self._noodling_to_slot[noodling_id] = slot
                    return slot

            # All slots taken -- return left as fallback
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

            Each noodling_id is assigned a viewport slot.

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
            self._current_char_fmt = 'spoken'

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
            self._current_char_fmt = 'spoken'   # Reset format at start of each block

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

        def on_format_changed(self, fmt: str):
            """Update current character format type for subsequent append_character() calls.

            Called by GuidePerformanceManager when a performer's formatChanged
            signal fires (streaming mode only).

            Args:
                fmt: 'spoken', 'action', or 'thought' (thought is a no-op
                     since thought lines are filtered before reaching here)
            """
            self._current_char_fmt = fmt

        def append_character(self, char: str):
            """
            Append a single character to the dialogue for typed-text effect.

            Uses the color of the current typing noodling (set by
            begin_noodling_text or begin_guide_text). Applies italic
            dim styling for 'action' segments.

            Args:
                char: Single character to append
            """
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            nid = self._current_typing_noodling or 'default'
            color = self._noodling_colors.get(
                nid, self._noodling_colors['default']
            )
            char_fmt = QTextCharFormat()

            if self._current_char_fmt == 'action':
                # Action text: italic + slightly dimmer gray
                char_fmt.setForeground(QColor(153, 153, 153))  # #999999
                char_fmt.setFontItalic(True)
            else:
                # Spoken (and any unrecognized type): normal color
                char_fmt.setForeground(color)
                char_fmt.setFontItalic(False)

            cursor.setCharFormat(char_fmt)
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

    # Backward-compatible alias
    GuidePerformanceWindow = PerformancePanel


# =============================================================================
# Fallback when Qt not available
# =============================================================================

if not QT_AVAILABLE:

    class PerformancePanel:
        """Stub when PyQt6 is not available."""

        messageSent = None
        messageSubmitted = None

        def __init__(self, *args, **kwargs):
            logger.warning("PerformancePanel requires PyQt6")

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
        def set_ensemble_visible(self, visible): pass
        def display_narration(self, text): pass
        def append_guide_text(self, text): pass
        def append_noodling_text(self, noodling_id, name, text): pass
        def begin_noodling_text(self, noodling_id, name): pass
        def end_noodling_text(self): pass
        def begin_guide_text(self): pass
        def append_character(self, char): pass
        def end_guide_text(self): pass
        def append_user_text(self, text): pass
        def on_format_changed(self, fmt): pass

    # Backward-compatible alias
    GuidePerformanceWindow = PerformancePanel


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
