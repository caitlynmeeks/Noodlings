"""
Soft Restart System - Restart app with state preservation

Saves current UI state, restarts the application, and restores state on startup.
Used for applying code changes that require restart (panel classes, mixins, etc.)

State preserved:
- Open project path
- Current stage
- Selected entity (type + UUID)
- Active tabs (center, bottom)
- Window geometry
- Facets editor state

Author: Caitlyn + Claude
Date: January 2, 2026
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox


# State file location
STATE_DIR = Path.home() / ".noodlestudio"
RESTART_STATE_FILE = STATE_DIR / ".restart_state"


@dataclass
class RestartState:
    """State to preserve across restart."""
    # Project
    project_path: Optional[str] = None
    current_stage: Optional[str] = None

    # Selection
    selected_entity_type: Optional[str] = None  # 'noodling', 'zone', 'prop', 'facet'
    selected_entity_id: Optional[str] = None

    # Tabs
    center_tab_index: int = 0
    bottom_tab_index: int = 0
    left_tab_index: int = 0

    # Window
    window_geometry: Optional[str] = None  # Base64 encoded QByteArray
    window_state: Optional[str] = None

    # Facets editor
    facets_assembly_path: Optional[str] = None

    # Noodle Code
    noodle_code_history: Optional[str] = None  # JSON of recent messages

    # Metadata
    timestamp: str = ""
    version: str = ""
    reason: str = ""  # Why restart was triggered

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RestartState':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def save_restart_state(main_window, reason: str = "User requested") -> bool:
    """
    Save current UI state for restoration after restart.

    Args:
        main_window: The MainWindow instance
        reason: Why the restart is happening

    Returns:
        True if state was saved successfully
    """
    try:
        try:
            from .. import __version__
        except ImportError:
            __version__ = "unknown"

        state = RestartState(
            timestamp=datetime.now().isoformat(),
            version=__version__,
            reason=reason,
        )

        # Project state
        if hasattr(main_window, 'project_manager'):
            pm = main_window.project_manager
            if pm.current_project_path:
                state.project_path = pm.current_project_path

        # Stage state
        if hasattr(main_window, 'hierarchy') and main_window.hierarchy.current_stage:
            state.current_stage = main_window.hierarchy.current_stage

        # Selection state
        if hasattr(main_window, 'inspector'):
            inspector = main_window.inspector
            if hasattr(inspector, 'current_mode') and inspector.current_mode:
                state.selected_entity_type = inspector.current_mode
                if hasattr(inspector, 'current_entity_id'):
                    state.selected_entity_id = inspector.current_entity_id

        # Tab states
        if hasattr(main_window, 'center_tabs'):
            state.center_tab_index = main_window.center_tabs.currentIndex()

        # Find bottom tabs (it's in the main splitter layout)
        # This is a bit fragile but works with current architecture
        central = main_window.centralWidget()
        if central:
            # Main splitter > bottom widget
            from PyQt6.QtWidgets import QSplitter, QTabWidget
            if isinstance(central, QSplitter) and central.count() > 1:
                bottom_widget = central.widget(1)
                if isinstance(bottom_widget, QTabWidget):
                    state.bottom_tab_index = bottom_widget.currentIndex()

        # Window geometry
        state.window_geometry = main_window.saveGeometry().toBase64().data().decode()
        state.window_state = main_window.saveState().toBase64().data().decode()

        # Facets editor state
        if hasattr(main_window, 'facets_editor'):
            fe = main_window.facets_editor
            if hasattr(fe, 'current_assembly_path') and fe.current_assembly_path:
                state.facets_assembly_path = str(fe.current_assembly_path)

        # Save to file
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESTART_STATE_FILE, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)

        print(f"[SoftRestart] State saved: {RESTART_STATE_FILE}")
        return True

    except Exception as e:
        print(f"[SoftRestart] Failed to save state: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_restart_state() -> Optional[RestartState]:
    """
    Load restart state if it exists.

    Returns:
        RestartState if file exists and is valid, None otherwise.
        The state file is deleted after reading.
    """
    if not RESTART_STATE_FILE.exists():
        return None

    try:
        with open(RESTART_STATE_FILE, 'r') as f:
            data = json.load(f)

        # Delete the file immediately so we don't restore twice
        RESTART_STATE_FILE.unlink()

        state = RestartState.from_dict(data)
        print(f"[SoftRestart] Loaded state from: {state.timestamp}")
        print(f"[SoftRestart] Reason: {state.reason}")
        return state

    except Exception as e:
        print(f"[SoftRestart] Failed to load state: {e}")
        # Clean up corrupted file
        try:
            RESTART_STATE_FILE.unlink()
        except:
            pass
        return None


def restore_state(main_window, state: RestartState):
    """
    Restore UI state after restart.

    Called from MainWindow after initial setup is complete.
    Uses QTimer.singleShot to defer operations until event loop is running.
    """
    print(f"[SoftRestart] Restoring state...")

    def do_restore():
        try:
            # 1. Restore window geometry first
            if state.window_geometry:
                from PyQt6.QtCore import QByteArray
                geometry = QByteArray.fromBase64(state.window_geometry.encode())
                main_window.restoreGeometry(geometry)

            if state.window_state:
                from PyQt6.QtCore import QByteArray
                win_state = QByteArray.fromBase64(state.window_state.encode())
                main_window.restoreState(win_state)

            # 2. Open project (this triggers other loading)
            if state.project_path and Path(state.project_path).exists():
                main_window.project_manager.open_project(state.project_path)

                # 3. After project loads, restore stage and selection
                def after_project_load():
                    try:
                        # Restore stage
                        if state.current_stage and hasattr(main_window, 'hierarchy'):
                            main_window.hierarchy.load_stage(state.current_stage)

                        # Restore tabs
                        if hasattr(main_window, 'center_tabs'):
                            main_window.center_tabs.setCurrentIndex(state.center_tab_index)

                        # Restore bottom tabs
                        central = main_window.centralWidget()
                        if central:
                            from PyQt6.QtWidgets import QSplitter, QTabWidget
                            if isinstance(central, QSplitter) and central.count() > 1:
                                bottom_widget = central.widget(1)
                                if isinstance(bottom_widget, QTabWidget):
                                    bottom_widget.setCurrentIndex(state.bottom_tab_index)

                        # Restore facets editor assembly
                        if state.facets_assembly_path and hasattr(main_window, 'facets_editor'):
                            assembly_path = Path(state.facets_assembly_path)
                            if assembly_path.exists():
                                main_window.facets_editor.load_assembly(assembly_path)

                        # TODO: Restore selection (requires entity to be loaded)
                        # if state.selected_entity_type and state.selected_entity_id:
                        #     main_window.hierarchy.select_entity(
                        #         state.selected_entity_type,
                        #         state.selected_entity_id
                        #     )

                        print("[SoftRestart] State restored successfully")
                        main_window.statusBar().showMessage(
                            f"Restarted and restored state ({state.reason})",
                            5000
                        )

                    except Exception as e:
                        print(f"[SoftRestart] Error restoring post-project state: {e}")

                # Wait for project to load
                QTimer.singleShot(500, after_project_load)
            else:
                # No project to restore, just restore tabs
                if hasattr(main_window, 'center_tabs'):
                    main_window.center_tabs.setCurrentIndex(state.center_tab_index)

                print("[SoftRestart] State restored (no project)")

        except Exception as e:
            print(f"[SoftRestart] Error restoring state: {e}")
            import traceback
            traceback.print_exc()

    # Defer to let MainWindow finish initialization
    QTimer.singleShot(100, do_restore)


def perform_soft_restart(main_window, reason: str = "Code changes applied"):
    """
    Perform a soft restart of NoodleStudio.

    1. Checks for unsaved changes
    2. Saves current state
    3. Restarts the application

    Args:
        main_window: The MainWindow instance
        reason: Why the restart is happening
    """
    # Check for unsaved changes
    if hasattr(main_window, 'has_unsaved_changes') and main_window.has_unsaved_changes():
        reply = QMessageBox.question(
            main_window,
            "Unsaved Changes",
            "You have unsaved changes. Save before restarting?",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel
        )

        if reply == QMessageBox.StandardButton.Cancel:
            return False
        elif reply == QMessageBox.StandardButton.Save:
            # Trigger save
            if hasattr(main_window, 'save_all'):
                main_window.save_all()

    # Save state
    if not save_restart_state(main_window, reason):
        QMessageBox.warning(
            main_window,
            "Restart Failed",
            "Could not save application state. Restart cancelled."
        )
        return False

    print("[SoftRestart] Restarting application...")

    # Get the command to restart
    python = sys.executable
    script = sys.argv[0]

    # Use os.execv to replace current process
    # This is cleaner than subprocess as it doesn't leave orphan processes
    try:
        # On macOS/Linux, execv replaces current process
        os.execv(python, [python, script] + sys.argv[1:])
    except Exception as e:
        print(f"[SoftRestart] execv failed: {e}, trying subprocess...")

        # Fallback to subprocess
        import subprocess
        subprocess.Popen([python, script] + sys.argv[1:])

        # Exit current process
        main_window.close()
        sys.exit(0)

    return True


def request_soft_restart(main_window, reason: str = "Apply changes"):
    """
    Request a soft restart with user confirmation.

    Shows a dialog explaining what will happen.
    """
    reply = QMessageBox.question(
        main_window,
        "Restart Required",
        f"To apply changes, NoodleStudio needs to restart.\n\n"
        f"Reason: {reason}\n\n"
        f"Your current state (project, tabs, selection) will be restored.\n\n"
        f"Restart now?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )

    if reply == QMessageBox.StandardButton.Yes:
        perform_soft_restart(main_window, reason)
