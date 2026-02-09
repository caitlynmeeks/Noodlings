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
            if self._is_speaking:
                self._set_speaking(False)

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
