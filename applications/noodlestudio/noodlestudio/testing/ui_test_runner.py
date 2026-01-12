# ──────────────────────────────────────────────────────────────
#
#   UI Test Runner - Execute automated UI tests
#
#   Uses NoodleCode's computer use to actually click the UI.
#   Not mocks - real integration testing.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.testing.ui_test_runner
# PURPOSE:  Main test runner
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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class StepResult:
    """Result of a single test step."""
    action: str
    success: bool
    duration: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestPhaseResult:
    """Result of a test phase."""
    name: str
    success: bool
    duration: float
    steps: List[StepResult] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TestResult:
    """Result of a complete test run."""
    name: str
    success: bool
    duration: float
    phases: List[TestPhaseResult] = field(default_factory=list)
    error: Optional[str] = None


class UITestRunner:
    """
    Run UI tests using NoodleCode's computer use infrastructure.

    This is NOT mocking - it actually clicks the UI.
    Ghost cursor visualizes what the test is doing.

    Usage:
        runner = UITestRunner(main_window)
        result = await runner.run_test_file("tests/ui/e2e/test_create_project.yaml")
    """

    def __init__(self, main_window, visual_mode: bool = True):
        """
        Initialize the UI test runner.

        Args:
            main_window: The main NoodleStudio window
            visual_mode: If True, show ghost cursor during tests
        """
        self.window = main_window
        self.visual_mode = visual_mode

        # Get computer use controller from NoodleCode
        self._computer_use = None
        self._ghost = None
        self._init_controllers()

        # Import action handlers
        from .ui_test_actions import UITestActions
        from .ui_test_targets import UITestTargetResolver
        from .ui_test_assertions import UITestAssertions

        self.actions = UITestActions(self)
        self.targets = UITestTargetResolver(self.window)
        self.assertions = UITestAssertions(self.window)

        # Test state
        self._current_test: Optional[Dict] = None
        self._current_phase: Optional[str] = None
        self._stop_requested = False

    def _init_controllers(self):
        """Initialize computer use and ghost cursor controllers."""
        # Try to get from NoodleCode panel
        if hasattr(self.window, 'noodle_code_panel'):
            panel = self.window.noodle_code_panel
            if hasattr(panel, 'computer_use_controller'):
                self._computer_use = panel.computer_use_controller

        # Try to get ghost cursor
        if hasattr(self.window, 'ghost_cursor_controller'):
            self._ghost = self.window.ghost_cursor_controller
        elif hasattr(self.window, '_ghost_controller'):
            self._ghost = self.window._ghost_controller

        # Fallback: create controllers if not found
        if self._computer_use is None:
            try:
                from ..core.computer_use_controller import get_computer_use_controller
                self._computer_use = get_computer_use_controller()
                # Ensure window is set
                if self._computer_use._main_window is None:
                    self._computer_use.set_main_window(self.window)
            except ImportError:
                print("[UITestRunner] WARNING: ComputerUseController not available")

        if self._ghost is None and self.visual_mode:
            try:
                # First try to get the existing global ghost controller
                from ..core.ghost_cursor import get_ghost_controller
                self._ghost = get_ghost_controller()
                
                if self._ghost is None:
                    # Fallback: create new ghost cursor
                    from ..core.ghost_cursor import GhostCursorOverlay, GhostCursorController
                    from PyQt6.QtCore import QThread
                    from PyQt6.QtWidgets import QApplication

                    # Only create ghost cursor on main thread (GUI widgets must be created there)
                    app = QApplication.instance()
                    if app and QThread.currentThread() == app.thread():
                        overlay = GhostCursorOverlay(self.window)
                        self._ghost = GhostCursorController(overlay)
                        overlay.show()
                    else:
                        print("[UITestRunner] WARNING: Cannot create ghost cursor from non-main thread")
            except ImportError:
                print("[UITestRunner] WARNING: GhostCursorController not available")
            except Exception as e:
                print(f"[UITestRunner] WARNING: Could not create ghost cursor: {e}")

    @property
    def computer_use(self):
        """Get the computer use controller."""
        return self._computer_use

    @property
    def ghost(self):
        """Get the ghost cursor controller."""
        return self._ghost

    async def run_test_file(self, path: str) -> TestResult:
        """
        Run a YAML test file.

        Args:
            path: Path to the test YAML file

        Returns:
            TestResult with pass/fail and details
        """
        if not YAML_AVAILABLE:
            return TestResult(
                name="Unknown",
                success=False,
                duration=0,
                error="PyYAML not available"
            )

        # Load test file
        try:
            with open(path) as f:
                test_spec = yaml.safe_load(f)
        except Exception as e:
            return TestResult(
                name=Path(path).stem,
                success=False,
                duration=0,
                error=f"Failed to load test file: {e}"
            )

        return await self.run_test(test_spec)

    async def run_test(self, test_spec: Dict[str, Any]) -> TestResult:
        """
        Run a test from a specification dict.

        Args:
            test_spec: Test specification dictionary

        Returns:
            TestResult with pass/fail and details
        """
        self._current_test = test_spec
        self._stop_requested = False

        test_name = test_spec.get('name', 'Unnamed Test')
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")

        if test_spec.get('description'):
            print(f"\n{test_spec['description']}\n")

        # Enable visual mode if requested
        if self.visual_mode and self._ghost:
            self._ghost.set_demo_mode(True)

        phase_results = []
        overall_success = True
        error_msg = None

        try:
            phases = test_spec.get('phases', [])

            for phase in phases:
                if self._stop_requested:
                    break

                phase_result = await self._run_phase(phase)
                phase_results.append(phase_result)

                if not phase_result.success:
                    overall_success = False
                    error_msg = f"Phase '{phase_result.name}' failed: {phase_result.error}"
                    break

        except Exception as e:
            overall_success = False
            error_msg = f"Test execution error: {e}"

        finally:
            # Disable demo mode
            if self._ghost:
                self._ghost.set_demo_mode(False)

        duration = time.time() - start_time

        # Print result
        print(f"\n{'='*60}")
        if overall_success:
            success_msg = test_spec.get('success', {}).get('message', 'Test passed!')
            print(f"SUCCESS: {success_msg}")
        else:
            print(f"FAILED: {error_msg}")
        print(f"Duration: {duration:.2f}s")
        print(f"{'='*60}\n")

        return TestResult(
            name=test_name,
            success=overall_success,
            duration=duration,
            phases=phase_results,
            error=error_msg
        )

    async def _run_phase(self, phase: Dict[str, Any]) -> TestPhaseResult:
        """Run a single test phase."""
        phase_name = phase.get('name', 'Unnamed Phase')
        self._current_phase = phase_name

        print(f"\n--- Phase: {phase_name} ---\n")

        start_time = time.time()
        step_results = []
        success = True
        error_msg = None

        steps = phase.get('steps', [])

        for i, step in enumerate(steps):
            if self._stop_requested:
                break

            step_result = await self._run_step(step, i + 1, len(steps))
            step_results.append(step_result)

            if not step_result.success:
                success = False
                error_msg = step_result.error
                break

        duration = time.time() - start_time

        return TestPhaseResult(
            name=phase_name,
            success=success,
            duration=duration,
            steps=step_results,
            error=error_msg
        )

    async def _run_step(self, step: Dict[str, Any], step_num: int, total_steps: int) -> StepResult:
        """Run a single test step."""
        action = step.get('action', 'unknown')
        start_time = time.time()

        # Print step info
        comment = step.get('comment', '')
        if comment:
            print(f"  [{step_num}/{total_steps}] {action}: {comment}")
        else:
            print(f"  [{step_num}/{total_steps}] {action}")

        try:
            # Execute the action
            await self.actions.execute(step)

            duration = time.time() - start_time

            # Small delay between steps for stability
            await asyncio.sleep(0.1)

            return StepResult(
                action=action,
                success=True,
                duration=duration
            )

        except AssertionError as e:
            duration = time.time() - start_time
            error_msg = f"Assertion failed: {e}"
            print(f"    FAILED: {error_msg}")
            return StepResult(
                action=action,
                success=False,
                duration=duration,
                error=error_msg
            )

        except Exception as e:
            import traceback
            duration = time.time() - start_time
            tb = traceback.format_exc()
            error_msg = f"Step failed: {e}"
            print(f"    ERROR: {error_msg}")
            print(f"    Traceback:\n{tb}")
            return StepResult(
                action=action,
                success=False,
                duration=duration,
                error=f"{error_msg}\n{tb}"
            )

    def stop(self):
        """Request the test to stop."""
        self._stop_requested = True
        print("\n[UITestRunner] Stop requested")


# ═══════════════════════════════════════════════════════════
# Command-line interface
# ═══════════════════════════════════════════════════════════

async def run_test_cli(test_path: str, visual: bool = True):
    """
    Run a UI test from command line.

    This requires the app to be running.
    """
    # Import Qt app
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if app is None:
        print("ERROR: NoodleStudio must be running to execute UI tests")
        print("Start the app first, then run tests.")
        sys.exit(1)

    # Find main window
    main_window = None
    for widget in app.topLevelWidgets():
        if widget.__class__.__name__ == 'MainWindow':
            main_window = widget
            break

    if main_window is None:
        print("ERROR: Could not find NoodleStudio main window")
        sys.exit(1)

    # Run test
    runner = UITestRunner(main_window, visual_mode=visual)
    result = await runner.run_test_file(test_path)

    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m noodlestudio.testing.ui_test_runner <test_file.yaml> [--no-visual]")
        sys.exit(1)

    test_path = sys.argv[1]
    visual = "--no-visual" not in sys.argv

    asyncio.run(run_test_cli(test_path, visual))
