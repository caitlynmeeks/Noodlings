#!/usr/bin/env python3
"""
Run the Hello NoodleStudio demo play.

This script launches NoodleStudio and runs the demo play with ghost cursor visualization.

Usage:
    python run_demo_play.py [play_file] [--fast|--slow]

Examples:
    python run_demo_play.py                              # Run default demo
    python run_demo_play.py --fast                       # Run at 2x speed
    python run_demo_play.py path/to/play.yaml            # Run specific play
"""

import asyncio
import os
import sys

# Ensure we can import noodlestudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DemoScheduler:
    """Schedules and runs the demo within Qt's event loop."""

    def __init__(self, window, play_path: str, speed: float = 1.0):
        self.window = window
        self.play_path = play_path
        self.speed = speed
        self._step = 0
        self._runner = None
        self._beat_index = 0
        self._play_data = None

    def start(self):
        """Start the demo execution using Qt timers."""
        from PyQt6.QtCore import QTimer
        import yaml

        # Load play file
        with open(self.play_path, 'r') as f:
            self._play_data = yaml.safe_load(f)

        play_name = self._play_data.get('name', 'Unnamed Play')
        print(f"\n{'='*60}")
        print(f"DEMO: {play_name}")
        print(f"{'='*60}\n")

        # Initialize runner
        from noodlestudio.testing import DemoPlayRunner
        self._runner = DemoPlayRunner(self.window, visual_mode=True, speed=self.speed)

        def on_speak(character, text):
            print(f"[{character.upper()}]: {text}\n")

        self._runner.on_speak(on_speak)

        # Enable visual mode
        if self._runner._ghost:
            self._runner._ghost.set_demo_mode(True)
        if self._runner._computer_use:
            self._runner._computer_use.demo_mode = True

        # Start executing beats
        print(f"Starting {len(self._play_data.get('beats', []))} beats...\n")
        self._beat_index = 0
        self._execute_next_beat()

    def _execute_next_beat(self):
        """Execute the next beat in the sequence."""
        from PyQt6.QtCore import QTimer

        beats = self._play_data.get('beats', [])
        if self._beat_index >= len(beats):
            self._finish()
            return

        beat = beats[self._beat_index]
        self._beat_index += 1

        # Execute beat synchronously using timers
        self._execute_beat(beat)

    def _execute_beat(self, beat):
        """Execute a single beat."""
        from PyQt6.QtCore import QTimer

        beat_id = beat.get('id', 'unknown')
        print(f"\n--- Beat: {beat_id} ---")

        # Handle dialogue
        speaks = beat.get('speaks')
        if speaks:
            character = beat.get('character', 'narrator')
            print(f"[{character}] {speaks}")

        # Execute computer_use actions
        actions = beat.get('computer_use', [])
        self._execute_actions(actions, 0, lambda: self._beat_complete(beat))

    def _execute_actions(self, actions, index, callback):
        """Execute actions sequentially with proper timing."""
        from PyQt6.QtCore import QTimer

        if index >= len(actions):
            callback()
            return

        action = actions[index]
        delay = self._execute_action(action)

        # Schedule next action
        QTimer.singleShot(int(delay * 1000 / self.speed),
                         lambda: self._execute_actions(actions, index + 1, callback))

    def _execute_action(self, action) -> float:
        """Execute a single action. Returns delay in seconds."""
        import asyncio

        action_type = action.get('action')
        target = action.get('target')

        # Resolve target synchronously
        coords = None
        if target and self._runner._targets:
            try:
                # Run async resolve in sync context
                loop = asyncio.new_event_loop()
                coords = loop.run_until_complete(self._runner._targets.resolve(target))
                loop.close()
            except Exception as e:
                print(f"[Demo] Could not resolve target {target}: {e}")

        # Execute action
        if action_type == 'move':
            if coords and self._runner._computer_use:
                self._runner._computer_use.mouse_move(coords[0], coords[1])
            return 0.5

        elif action_type == 'click':
            button = action.get('button', 'left')
            if coords and self._runner._computer_use:
                self._runner._computer_use.click(coords[0], coords[1], button)
            return 0.3

        elif action_type == 'double_click':
            if coords and self._runner._computer_use:
                self._runner._computer_use.double_click(coords[0], coords[1])
            return 0.3

        elif action_type == 'type':
            text = action.get('text', '')
            if self._runner._computer_use:
                self._runner._computer_use.type_text(text)
            return len(text) * 0.08

        elif action_type == 'key':
            combo = action.get('combo', '')
            if self._runner._computer_use:
                self._runner._computer_use.key(combo)
            return 0.2

        elif action_type == 'highlight':
            if coords and self._runner._computer_use:
                self._runner._computer_use.mouse_move(coords[0], coords[1])
            duration = action.get('duration', '1s')
            return self._parse_duration(duration)

        elif action_type == 'wait':
            duration = action.get('duration', '500ms')
            return self._parse_duration(duration)

        elif action_type == 'wait_for':
            # For simplicity, just wait the timeout
            timeout = action.get('timeout', '3s')
            return self._parse_duration(timeout)

        return 0.1

    def _parse_duration(self, duration_str) -> float:
        """Parse duration string to seconds."""
        if isinstance(duration_str, (int, float)):
            return float(duration_str)

        duration_str = str(duration_str).lower().strip()
        if duration_str.endswith('ms'):
            return float(duration_str[:-2]) / 1000.0
        elif duration_str.endswith('s'):
            return float(duration_str[:-1])
        return 1.0

    def _beat_complete(self, beat):
        """Called when a beat is complete."""
        from PyQt6.QtCore import QTimer

        wait_after = beat.get('wait_after', 0.3)
        QTimer.singleShot(int(wait_after * 1000 / self.speed), self._execute_next_beat)

    def _finish(self):
        """Demo complete."""
        # Disable visual mode
        if self._runner._ghost:
            self._runner._ghost.set_demo_mode(False)
        if self._runner._computer_use:
            self._runner._computer_use.demo_mode = False

        success_msg = self._play_data.get('success', {}).get('message', 'Demo complete!')
        print(f"\n{'='*60}")
        print(f"COMPLETE: {success_msg}")
        print(f"{'='*60}\n")


def main():
    # Parse arguments
    play_path = None
    speed = 1.0

    args = sys.argv[1:]
    for arg in args:
        if arg == '--fast':
            speed = 2.0
        elif arg == '--slow':
            speed = 0.5
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('-'):
            play_path = arg

    # Default play file
    if play_path is None:
        # Go up to noodlings_clean root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        play_path = os.path.join(base_dir, 'docs', 'noodlestudio', 'plays', 'demos', 'hello_noodlestudio.play.yaml')

    if not os.path.exists(play_path):
        print(f"Play file not found: {play_path}")
        sys.exit(1)

    print(f"Running demo play: {play_path}")
    print(f"Speed: {speed}x")
    print()

    # Import Qt
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Import and create main window
    from noodlestudio.core.main_window import MainWindow

    window = MainWindow()
    window.show()

    # Create demo scheduler
    scheduler = DemoScheduler(window, play_path, speed)

    # Start demo after window is fully shown
    QTimer.singleShot(1500, scheduler.start)

    # Run Qt event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
