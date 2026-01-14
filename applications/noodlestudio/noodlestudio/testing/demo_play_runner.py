# ──────────────────────────────────────────────────────────────
#
#   Demo Play Runner - Execute .play.yaml files with visual ghost cursor
#
#   Runs Brenda plays with full computer_use visualization.
#   Shows ghost cursor moving, clicking, and typing while Ajo narrates.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.testing.demo_play_runner
# PURPOSE:  Demo Play Execution
# LAYER:    Studio / Testing
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication


@dataclass
class BeatResult:
    """Result of executing a beat."""
    beat_id: str
    success: bool
    speaks: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0


class DemoPlayRunner:
    """
    Runs .play.yaml demo files with ghost cursor visualization.

    Unlike UITestRunner (which runs test assertions), this runner
    executes theatrical demo plays with narration and visual flair.

    Usage:
        runner = DemoPlayRunner(main_window)
        await runner.run_play("docs/noodlestudio/plays/demos/hello_noodlestudio.play.yaml")
    """

    def __init__(self, main_window, visual_mode: bool = True, speed: float = 1.0):
        """
        Initialize the demo play runner.

        Args:
            main_window: The main NoodleStudio window
            visual_mode: If True, show ghost cursor during demos
            speed: Playback speed multiplier (1.0 = normal, 2.0 = double speed)
        """
        self.window = main_window
        self.visual_mode = visual_mode
        self.speed = speed

        # Initialize controllers
        self._computer_use = None
        self._ghost = None
        self._targets = None
        self._init_controllers()

        # Playback state
        self._play_data: Optional[Dict] = None
        self._current_beat_index = 0
        self._stop_requested = False

        # Callbacks
        self._on_speak: Optional[Callable[[str, str], None]] = None  # (character, text)
        self._on_beat_start: Optional[Callable[[str], None]] = None
        self._on_beat_complete: Optional[Callable[[str], None]] = None

    def _init_controllers(self):
        """Initialize computer use and ghost cursor controllers."""
        # Get computer use controller
        try:
            from ..core.computer_use_controller import get_computer_use_controller
            self._computer_use = get_computer_use_controller()
            if self._computer_use._main_window is None:
                self._computer_use.set_main_window(self.window)
        except ImportError:
            print("[DemoPlayRunner] WARNING: ComputerUseController not available")

        # Get ghost cursor controller
        try:
            from ..core.ghost_cursor import get_ghost_controller
            self._ghost = get_ghost_controller()
        except ImportError:
            print("[DemoPlayRunner] WARNING: GhostCursorController not available")

        # Initialize target resolver
        try:
            from .ui_test_targets import UITestTargetResolver
            self._targets = UITestTargetResolver(self.window)
        except ImportError:
            print("[DemoPlayRunner] WARNING: UITestTargetResolver not available")

    def on_speak(self, callback: Callable[[str, str], None]):
        """Register callback for when a character speaks."""
        self._on_speak = callback

    def on_beat_start(self, callback: Callable[[str], None]):
        """Register callback for beat start."""
        self._on_beat_start = callback

    def on_beat_complete(self, callback: Callable[[str], None]):
        """Register callback for beat completion."""
        self._on_beat_complete = callback

    async def run_play(self, play_path: str) -> List[BeatResult]:
        """
        Run a .play.yaml demo file.

        Args:
            play_path: Path to the play YAML file

        Returns:
            List of BeatResult for each beat
        """
        # Load play file
        path = Path(play_path)
        if not path.exists():
            print(f"[DemoPlayRunner] Play file not found: {play_path}")
            return [BeatResult(beat_id="load", success=False, error="File not found")]

        with open(path, 'r') as f:
            self._play_data = yaml.safe_load(f)

        play_name = self._play_data.get('name', 'Unnamed Play')
        print(f"\n{'='*60}")
        print(f"DEMO: {play_name}")
        print(f"{'='*60}\n")

        # Enable visual mode
        if self.visual_mode and self._ghost:
            self._ghost.set_demo_mode(True)

        if self._computer_use:
            self._computer_use.demo_mode = self.visual_mode

        # Execute beats
        results = []
        beats = self._play_data.get('beats', [])

        for i, beat in enumerate(beats):
            if self._stop_requested:
                break

            self._current_beat_index = i
            result = await self._execute_beat(beat)
            results.append(result)

            if not result.success:
                print(f"[DemoPlayRunner] Beat failed: {result.error}")
                break

        # Disable visual mode
        if self._ghost:
            self._ghost.set_demo_mode(False)

        if self._computer_use:
            self._computer_use.demo_mode = False

        # Print completion message
        success_msg = self._play_data.get('success', {}).get('message', 'Demo complete!')
        print(f"\n{'='*60}")
        print(f"COMPLETE: {success_msg}")
        print(f"{'='*60}\n")

        return results

    async def _execute_beat(self, beat: Dict[str, Any]) -> BeatResult:
        """Execute a single beat."""
        beat_id = beat.get('id', 'unknown')
        start_time = time.time()

        print(f"\n--- Beat: {beat_id} ---")

        if self._on_beat_start:
            self._on_beat_start(beat_id)

        try:
            # Handle dialogue
            speaks = beat.get('speaks')
            if speaks:
                character = beat.get('character', 'narrator')
                print(f"[{character}] {speaks}")

                if self._on_speak:
                    self._on_speak(character, speaks)

            # Execute computer_use actions
            actions = beat.get('computer_use', [])
            for action in actions:
                await self._execute_action(action)

            # Handle wait_after
            wait_after = beat.get('wait_after', 0)
            if wait_after > 0:
                await asyncio.sleep(wait_after / self.speed)

            duration = time.time() - start_time

            if self._on_beat_complete:
                self._on_beat_complete(beat_id)

            return BeatResult(
                beat_id=beat_id,
                success=True,
                speaks=speaks,
                duration=duration
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[DemoPlayRunner] Error in beat {beat_id}: {e}")
            print(tb)
            return BeatResult(
                beat_id=beat_id,
                success=False,
                error=f"{e}\n{tb}",
                duration=time.time() - start_time
            )

    async def _execute_action(self, action: Dict[str, Any]):
        """Execute a single computer_use action."""
        action_type = action.get('action')

        # Resolve target to coordinates
        target = action.get('target')
        coords = None
        if target and self._targets:
            try:
                coords = await self._targets.resolve(target)
            except Exception as e:
                print(f"[DemoPlayRunner] Could not resolve target {target}: {e}")

        # Execute action
        if action_type == 'move':
            if coords and self._computer_use:
                self._computer_use.mouse_move(coords[0], coords[1])
            await asyncio.sleep(0.3 / self.speed)

        elif action_type == 'click':
            button = action.get('button', 'left')
            if coords and self._computer_use:
                self._computer_use.click(coords[0], coords[1], button)
            await asyncio.sleep(0.2 / self.speed)

        elif action_type == 'double_click':
            if coords and self._computer_use:
                self._computer_use.double_click(coords[0], coords[1])
            await asyncio.sleep(0.2 / self.speed)

        elif action_type == 'type':
            text = action.get('text', '')
            if self._computer_use:
                self._computer_use.type_text(text)
            await asyncio.sleep(len(text) * 0.05 / self.speed)

        elif action_type == 'key':
            combo = action.get('combo', '')
            if self._computer_use:
                self._computer_use.key(combo)
            await asyncio.sleep(0.1 / self.speed)

        elif action_type == 'highlight':
            # Highlight is visual-only - move cursor and pause
            if coords and self._computer_use:
                self._computer_use.mouse_move(coords[0], coords[1])

            duration = action.get('duration', '1s')
            seconds = self._parse_duration(duration)
            await asyncio.sleep(seconds / self.speed)

        elif action_type == 'wait':
            duration = action.get('duration', '500ms')
            seconds = self._parse_duration(duration)
            await asyncio.sleep(seconds / self.speed)

        elif action_type == 'wait_for':
            # Wait for an element to appear
            element = action.get('element')
            timeout = action.get('timeout', '3s')
            timeout_seconds = self._parse_duration(timeout)

            start = time.time()
            while time.time() - start < timeout_seconds:
                if element and self._targets:
                    try:
                        await self._targets.resolve(element)
                        break  # Element found
                    except:
                        pass  # Not found yet
                await asyncio.sleep(0.1)

        else:
            print(f"[DemoPlayRunner] Unknown action type: {action_type}")

    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string like '500ms' or '2s' to seconds."""
        if isinstance(duration_str, (int, float)):
            return float(duration_str)

        duration_str = str(duration_str).lower().strip()

        if duration_str.endswith('ms'):
            return float(duration_str[:-2]) / 1000.0
        elif duration_str.endswith('s'):
            return float(duration_str[:-1])
        else:
            try:
                return float(duration_str)
            except:
                return 1.0

    def stop(self):
        """Request playback to stop."""
        self._stop_requested = True


async def run_demo_play(play_path: str, speed: float = 1.0):
    """
    Run a demo play from the command line.

    Requires NoodleStudio to be running.

    Args:
        play_path: Path to the .play.yaml file
        speed: Playback speed (1.0 = normal)
    """
    app = QApplication.instance()
    if not app:
        print("ERROR: NoodleStudio must be running")
        return

    # Find main window
    main_window = None
    for widget in app.topLevelWidgets():
        if widget.__class__.__name__ == 'MainWindow':
            main_window = widget
            break

    if not main_window:
        print("ERROR: Could not find NoodleStudio main window")
        return

    # Run the demo
    runner = DemoPlayRunner(main_window, visual_mode=True, speed=speed)

    def on_speak(character, text):
        # Could integrate with TTS here
        pass

    runner.on_speak(on_speak)
    results = await runner.run_play(play_path)

    # Summary
    successes = sum(1 for r in results if r.success)
    print(f"\nBeats: {successes}/{len(results)} successful")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m noodlestudio.testing.demo_play_runner <play_file.yaml> [speed]")
        sys.exit(1)

    play_path = sys.argv[1]
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    asyncio.run(run_demo_play(play_path, speed))
