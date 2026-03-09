# ──────────────────────────────────────────────────────────────
#
#   Performance Player
#
#   Plays a performance script character-by-character via QTimer.
#   Drives typed text delivery with natural punctuation pauses
#   and speaking animation state changes.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.performance_player
# PURPOSE:  Animal Crossing Style Text Delivery
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   PerformancePlayer
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import QObject, QTimer, pyqtSignal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

# Prefixes that introduce structured output segments.
# Checked at line start in streaming mode.
_TAG_PREFIXES = {
    'SPOKEN: ': 'spoken',
    'ACTION: ': 'action',
    'THOUGHT: ': 'thought',
}


# Pauses longer than this (ms) toggle speaking state off briefly
_PAUSE_THRESHOLD_MS = 150


if QT_AVAILABLE:

    class PerformancePlayer(QObject):
        """
        Plays a performance script as typed text with natural pauses.

        Takes a performance script dict (from the Performance ScriptedFacet)
        and reveals text one character at a time via QTimer. Punctuation
        triggers natural pauses. Speaking animation state changes are
        signaled so the VRM viewport can drive jaw/head motion.

        The script format is:
            {
                "type": "performance_script",
                "text": "Hello, world!",
                "characters": [
                    {"c": "H", "d": 35},
                    {"c": "e", "d": 35},
                    ...
                    {"c": "!", "d": 250}
                ],
                "speaking_intensity": 0.7
            }

        Signals:
            characterRevealed(str): Emitted for each character as revealed
            speakingStateChanged(bool): True when speaking, False on pause/done
            finished(): Emitted when all characters have been revealed
        """

        characterRevealed = pyqtSignal(str)
        speakingStateChanged = pyqtSignal(bool)
        formatChanged = pyqtSignal(str)   # emitted when segment type changes between lines
        finished = pyqtSignal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self._timer = QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._reveal_next)

            self._characters = []       # List of {c: str, d: int} entries
            self._index = 0             # Current position
            self._is_speaking = False   # Current speaking state
            self._speaking_intensity = 0.7
            self._paused = False

            # Streaming tag detection state
            self._line_buffer = ''          # Accumulate chars until \n
            self._current_format = 'spoken' # 'spoken' | 'action' | 'thought'

        @property
        def speaking_intensity(self) -> float:
            """The speaking animation intensity from the script."""
            return self._speaking_intensity

        @property
        def is_playing(self) -> bool:
            """True if a performance is currently playing."""
            return self._index < len(self._characters)

        def play(self, script: dict):
            """
            Start playing a performance script.

            Args:
                script: Performance script dict with 'characters' list
            """
            self.stop()
            self._characters = script.get('characters', [])
            self._speaking_intensity = script.get('speaking_intensity', 0.7)
            self._index = 0

            if not self._characters:
                self.finished.emit()
                return

            self._set_speaking(True)
            self._reveal_next()

        def stop(self):
            """Stop the current performance immediately."""
            self._timer.stop()
            self._characters = []
            self._index = 0
            self._paused = False
            if self._is_speaking:
                self._set_speaking(False)

        def pause(self):
            """Freeze typing animation at current position."""
            self._paused = True
            self._timer.stop()
            # Also pause streaming timer if active
            if hasattr(self, '_stream_timer') and self._stream_timer:
                self._stream_timer.stop()

        def resume(self):
            """Continue typing animation from current position."""
            if not self._paused:
                return
            self._paused = False
            # Resume buffered mode
            if self._index < len(self._characters):
                self._reveal_next()
            # Resume streaming mode
            elif (hasattr(self, '_stream_buffer')
                  and (self._stream_buffer or not self._stream_done)):
                self._reveal_next_streaming()

        def _reveal_next(self):
            """Reveal the next character and schedule the following one."""
            if self._index >= len(self._characters):
                # All characters revealed
                self._set_speaking(False)
                self.finished.emit()
                return

            entry = self._characters[self._index]
            char = entry.get('c', '')
            delay = entry.get('d', 35)
            self._index += 1

            # Emit the character
            self.characterRevealed.emit(char)

            # Manage speaking state based on pause duration
            if delay >= _PAUSE_THRESHOLD_MS:
                # Long pause -- signal speaking off during pause
                if self._is_speaking:
                    self._set_speaking(False)
            else:
                # Short delay -- ensure speaking is on
                if not self._is_speaking:
                    self._set_speaking(True)

            # Schedule next character (or finish)
            if self._index < len(self._characters):
                self._timer.start(delay)
            else:
                # Last character -- brief settle then finish
                QTimer.singleShot(100, self._finish)

        def _finish(self):
            """Final cleanup after last character."""
            self._set_speaking(False)
            self.finished.emit()

        def _set_speaking(self, speaking: bool):
            """Update speaking state and emit signal if changed."""
            if speaking != self._is_speaking:
                self._is_speaking = speaking
                self.speakingStateChanged.emit(speaking)

        # =================================================================
        # STREAMING MODE
        #
        # For stream_animated delivery: tokens arrive incrementally from
        # the LLM. We buffer them and reveal character-by-character with
        # natural typing delays (matching the buffered mode's timing).
        # =================================================================

        def start_streaming(self):
            """Initialize streaming state. Call before feeding tokens."""
            self.stop()
            self._stream_buffer = ""
            self._stream_done = False
            self._stream_timer = QTimer(self)
            self._stream_timer.setSingleShot(True)
            self._stream_timer.timeout.connect(self._reveal_next_streaming)
            self._stream_started = False
            # Reset tag detection state
            self._line_buffer = ''
            self._current_format = 'spoken'

        def append_text(self, text: str):
            """
            Buffer incoming tokens from streaming LLM with tag detection.

            Accumulates characters into a line buffer. When a newline arrives,
            the completed line is checked for SPOKEN/ACTION/THOUGHT prefixes:
            - THOUGHT lines are silently discarded (not added to stream buffer)
            - ACTION and SPOKEN lines emit formatChanged if the type changes
            - Untagged lines are treated as SPOKEN

            Starts character reveal once 3+ chars are buffered.
            """
            for ch in text:
                if ch == '\n':
                    self._flush_line_buffer()
                else:
                    self._line_buffer += ch

            # Start revealing once we have enough to avoid stutter
            if not self._stream_started and len(self._stream_buffer) >= 3:
                self._stream_started = True
                self._set_speaking(True)
                self._reveal_next_streaming()

        def _flush_line_buffer(self, add_newline: bool = True):
            """Process a completed line from the streaming buffer.

            Detects tag prefix, emits formatChanged if type changed,
            strips the prefix, and adds the content to the stream buffer
            (unless the line is a THOUGHT, which is silently dropped).

            Args:
                add_newline: If True (default), appends '\\n' after the content
                    (used when the line ended with an actual newline in the stream).
                    Pass False when flushing a partial line at stream end so no
                    spurious '\\n' is appended.
            """
            line = self._line_buffer
            self._line_buffer = ''

            if not line.strip():
                # Empty line -- add the newline separator to stream buffer
                if add_newline:
                    self._stream_buffer += '\n'
                return

            # Detect tag prefix
            new_format = 'spoken'
            content = line
            for prefix, fmt in _TAG_PREFIXES.items():
                if line.startswith(prefix):
                    new_format = fmt
                    content = line[len(prefix):]
                    break

            # THOUGHT lines are not displayed
            if new_format == 'thought':
                return

            # Emit formatChanged if type changed
            if new_format != self._current_format:
                self._current_format = new_format
                self.formatChanged.emit(new_format)

            # Add stripped content to stream buffer
            self._stream_buffer += content + ('\n' if add_newline else '')

        def finish_streaming(self):
            """Mark stream as done. Flushes line buffer, then drains stream buffer."""
            # Flush any partial line that didn't end with \n -- no trailing newline
            if self._line_buffer:
                self._flush_line_buffer(add_newline=False)

            self._stream_done = True
            # If we never started (very short response), start now
            if not self._stream_started and self._stream_buffer:
                self._stream_started = True
                self._set_speaking(True)
                self._reveal_next_streaming()

        def _reveal_next_streaming(self):
            """Pop a char from buffer, emit with typing delay, retry if buffer empty."""
            if not self._stream_buffer:
                if self._stream_done:
                    # All done
                    self._set_speaking(False)
                    self.finished.emit()
                    return
                else:
                    # Buffer empty but stream ongoing -- wait and retry
                    self._stream_timer.start(50)
                    return

            char = self._stream_buffer[0]
            self._stream_buffer = self._stream_buffer[1:]

            self.characterRevealed.emit(char)

            delay = self._char_delay(char)

            # Manage speaking state based on delay
            if delay >= _PAUSE_THRESHOLD_MS:
                if self._is_speaking:
                    self._set_speaking(False)
            else:
                if not self._is_speaking:
                    self._set_speaking(True)

            # Schedule next character
            self._stream_timer.start(delay)

        @staticmethod
        def _char_delay(char: str) -> int:
            """
            Calculate per-character delay in ms.

            Timing rules matching the buffered Performance ScriptedFacet:
            - Base: 35ms
            - Period: 220ms
            - Exclamation/Question: 250ms
            - Comma: 120ms
            - Colon/Semicolon: 150ms
            - Newline: 300ms
            - Space: 21ms
            """
            if char == '.':
                return 220
            elif char in ('!', '?'):
                return 250
            elif char == ',':
                return 120
            elif char in (':', ';'):
                return 150
            elif char == '\n':
                return 300
            elif char == ' ':
                return 21
            else:
                return 35
