"""
NoodleSTUDIO Main Application

Entry point for the NoodleSTUDIO IDE.
"""

import sys
import os
import traceback
import atexit
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QFont, QColor

# NOTE: MainWindow imported AFTER QApplication is created
from .core.studio_acronyms import get_random_acronym
from . import __version__


# Global reference to main window for crash reporter
_main_window = None

# Sentinel file for crash detection
SENTINEL_DIR = Path.home() / ".noodlestudio"
SENTINEL_FILE = SENTINEL_DIR / ".running"
CRASH_INFO_FILE = SENTINEL_DIR / ".last_crash"


def create_sentinel():
    """Create sentinel file indicating NoodleStudio is running."""
    try:
        SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(SENTINEL_FILE, 'w') as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"started={datetime.now().isoformat()}\n")
            f.write(f"version={__version__}\n")
    except Exception as e:
        print(f"Could not create sentinel file: {e}")


def remove_sentinel():
    """Remove sentinel file on clean shutdown."""
    try:
        if SENTINEL_FILE.exists():
            SENTINEL_FILE.unlink()
    except Exception as e:
        print(f"Could not remove sentinel file: {e}")


def check_for_crash() -> bool:
    """
    Check if a previous session crashed.

    Returns True if crash detected (sentinel file exists from previous run).
    """
    if not SENTINEL_FILE.exists():
        return False

    # Read sentinel info
    try:
        sentinel_info = SENTINEL_FILE.read_text()
        print(f"[Crash Detection] Found sentinel from previous session:\n{sentinel_info}")

        # Save crash info for the recovery dialog
        crash_info = {
            'detected_at': datetime.now().isoformat(),
            'sentinel_info': sentinel_info,
        }

        # Try to get last log file for crash context
        logs_dir = Path(__file__).parent.parent / 'logs'
        if logs_dir.exists():
            log_files = sorted(logs_dir.glob('noodlestudio_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
            if log_files:
                last_log = log_files[0]
                crash_info['last_log'] = str(last_log)
                # Read last 100 lines of log
                try:
                    lines = last_log.read_text().splitlines()
                    crash_info['log_tail'] = '\n'.join(lines[-100:])
                except:
                    pass

        # Save crash info
        CRASH_INFO_FILE.write_text(str(crash_info))

        # Remove old sentinel
        SENTINEL_FILE.unlink()

        return True

    except Exception as e:
        print(f"[Crash Detection] Error reading sentinel: {e}")
        try:
            SENTINEL_FILE.unlink()
        except:
            pass
        return False


def save_crash_info(exc_type, exc_value, exc_tb):
    """Save crash information to file for post-mortem analysis."""
    try:
        SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
        crash_info = {
            'timestamp': datetime.now().isoformat(),
            'exception_type': exc_type.__name__ if exc_type else 'Unknown',
            'exception_value': str(exc_value) if exc_value else '',
            'traceback': ''.join(traceback.format_exception(exc_type, exc_value, exc_tb)) if exc_tb else '',
            'version': __version__,
        }

        # Include recent UI actions if recorder is available
        try:
            from .core.ui_action_recorder import get_ui_action_recorder
            recorder = get_ui_action_recorder()
            if recorder.is_recording():
                crash_info['ui_actions'] = recorder.get_crash_report_data()
        except Exception as e:
            crash_info['ui_actions_error'] = str(e)

        with open(CRASH_INFO_FILE, 'w') as f:
            import json
            json.dump(crash_info, f, indent=2)
    except Exception as e:
        print(f"Could not save crash info: {e}")


def install_crash_reporter():
    """Install global exception handler for crash reporting."""
    original_hook = sys.excepthook

    def crash_handler(exc_type, exc_value, exc_tb):
        """Handle uncaught exceptions by offering to report them."""
        # Don't catch KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            original_hook(exc_type, exc_value, exc_tb)
            return

        # Print to console first
        traceback.print_exception(exc_type, exc_value, exc_tb)

        # Save crash info for next session recovery
        save_crash_info(exc_type, exc_value, exc_tb)

        # Try to show crash dialog
        try:
            from .dialogs.bug_report_dialog import show_crash_report_dialog
            show_crash_report_dialog(_main_window, exc_type, exc_value, exc_tb)
        except Exception as e:
            # If crash dialog fails, show simple message
            print(f"Could not show crash dialog: {e}")
            try:
                QMessageBox.critical(
                    _main_window,
                    "NoodleStudio Crashed",
                    f"An unexpected error occurred:\n\n{exc_type.__name__}: {exc_value}\n\n"
                    "Please report this issue on GitHub."
                )
            except:
                pass

    sys.excepthook = crash_handler


def main():
    """Launch NoodleSTUDIO."""
    # Check for crash from previous session BEFORE creating app
    previous_crash = check_for_crash()

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Required for WebEngine to work - must be set BEFORE QApplication is created
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    # macOS: Set process name BEFORE creating QApplication
    import platform
    if platform.system() == "Darwin":
        try:
            from Foundation import NSBundle
            bundle = NSBundle.mainBundle()
            if bundle:
                info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
                if info:
                    info['CFBundleName'] = 'NoodleStudio'
        except ImportError:
            pass  # PyObjC not available

    app = QApplication(sys.argv)
    app.setApplicationName("NoodleStudio")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("Noodlings")
    app.setOrganizationDomain("noodlings.ai")

    # Install crash reporter and create sentinel
    install_crash_reporter()
    create_sentinel()

    # Install UI action recorder for crash debugging
    from .core.ui_action_recorder import get_ui_action_recorder
    ui_recorder = get_ui_action_recorder()
    ui_recorder.install(app)

    # Register cleanup on exit
    atexit.register(remove_sentinel)

    # Connect app aboutToQuit to clean up sentinel
    app.aboutToQuit.connect(remove_sentinel)

    # Create splash screen with random acronym
    splash = create_splash_screen()
    splash.show()
    app.processEvents()

    # Import MainWindow AFTER QApplication is created
    # (WebEngine on macOS requires QApplication to exist before import)
    from .core.main_window import MainWindow

    # Create main window (takes a moment to load)
    global _main_window
    window = MainWindow()
    _main_window = window

    # Keep splash visible for 7 seconds
    import time
    start_time = time.time()
    while time.time() - start_time < 7.0:
        app.processEvents()
        time.sleep(0.01)

    # Close splash and show main window maximized
    splash.finish(window)
    window.showMaximized()

    # Show crash recovery dialog if previous session crashed
    if previous_crash:
        show_crash_recovery_dialog(window)

    # Check for soft restart state and restore
    from .core.soft_restart import load_restart_state, restore_state
    restart_state = load_restart_state()
    if restart_state:
        restore_state(window, restart_state)

    sys.exit(app.exec())


def create_splash_screen():
    """Create a splash screen with a random STUDIO acronym - Green TUI aesthetic."""
    # Wider window for ASCII art
    pixmap = QPixmap(1200, 700)
    pixmap.fill(Qt.GlobalColor.black)

    # Draw on it
    painter = QPainter(pixmap)

    # ASCII Art Banner - moved down for better composition
    painter.setPen(QColor(0, 255, 0))  # Green
    font = QFont("Courier New", 8)
    painter.setFont(font)
    banner = (
        ":::.    :::.    ...         ...    :::::::-.   :::    .,::::::      .        :    ...    ::: .::::::.   ::   .:\n"
        "`;;;;,  `;;; .;;;;;;;.   .;;;;;;;.  ;;,   `';, ;;;    ;;;;''''      ;;,.    ;;;   ;;     ;;;;;;`    `  ,;;   ;;,\n"
        "  [[[[[. '[[,[[     \\[[,,[[     \\[[,`[[     [[ [[[     [[cccc       [[[[, ,[[[[, [['     [[['[==/[[[[,,[[[,,,[[[\n"
        "  $$$ \"Y$c$$$$$,     $$$$$$,     $$$ $$,    $$ $$'     $$\"\"\"\"       $$$$$$$$\"$$$ $$      $$$  '''    $\"$$$\"\"\"$$$\n"
        "  888    Y88\"888,_ _,88P\"888,_ _,88P 888_,o8P'o88oo,.__888oo,__     888 Y88\" 888o88    .d888 88b    dP 888   \"88o\n"
        "  MMM     YM  \"YMMMMMP\"   \"YMMMMMP\"  MMMMP\"`  \"\"\"\"YUMMM\"\"\"\"YUMMM    MMM  M'  \"MMM \"YmmMMMM\"\"  \"YMmMY\"  MMM    YMM"
    )
    painter.drawText(pixmap.rect().adjusted(10, 160, -10, -420),  # Moved further down: 160
                     Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
                     banner)

    # "NoodleSTUDIO" subtitle - positioned one line above acronym, larger font
    painter.setPen(QColor(100, 255, 100))  # Light green
    font = QFont("Courier New", 20)  # Larger: 16 → 20
    painter.setFont(font)
    painter.drawText(pixmap.rect().adjusted(40, 290, -40, -320),  # Moved lower: 275 → 290
                     Qt.AlignmentFlag.AlignCenter,
                     "STUDIO")

    # Random acronym
    acronym = get_random_acronym("all")

    # Acronym text - BIG and prominent
    font = QFont("Courier New", 20)
    painter.setFont(font)
    painter.setPen(QColor(100, 255, 100))  # Light green
    # Wrap text if too long
    text_rect = pixmap.rect().adjusted(60, 315, -60, -150)  # Adjusted for tighter spacing
    painter.drawText(text_rect,
                     Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                     acronym)

    # Version below acronym - moved up 2 lines
    font = QFont("Courier New", 16)
    painter.setFont(font)
    painter.setPen(QColor(80, 80, 80))  # Dark gray
    painter.drawText(pixmap.rect().adjusted(40, 470, -40, -40),  # Moved up: 500 → 470
                     Qt.AlignmentFlag.AlignCenter,
                     f"v{__version__}")

    painter.end()

    splash = QSplashScreen(pixmap)
    return splash


def show_crash_recovery_dialog(parent):
    """
    Show recovery dialog after detecting a crash from previous session.

    Offers user the chance to send a crash report to help improve Studio.
    """
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
    from PyQt6.QtGui import QFont
    from PyQt6.QtCore import Qt

    # Try to load crash info
    crash_info = {}
    if CRASH_INFO_FILE.exists():
        try:
            import json
            import ast
            content = CRASH_INFO_FILE.read_text()
            # Handle both JSON and Python dict repr formats
            try:
                crash_info = json.loads(content)
            except json.JSONDecodeError:
                crash_info = ast.literal_eval(content)
        except Exception as e:
            print(f"[Crash Recovery] Could not load crash info: {e}")

    dialog = QDialog(parent)
    dialog.setWindowTitle("Session Recovery")
    dialog.setModal(True)
    dialog.setMinimumWidth(500)
    dialog.setMinimumHeight(300)

    dialog.setStyleSheet("""
        QDialog {
            background-color: #1a1a1a;
        }
        QLabel {
            color: #cccccc;
        }
        QTextEdit {
            background-color: #2d2d2d;
            color: #888888;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 6px;
            font-family: monospace;
            font-size: 11px;
        }
        QPushButton {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 8px 16px;
            min-width: 100px;
        }
        QPushButton:hover {
            background-color: #3d3d3d;
        }
        QPushButton#send_btn {
            background-color: #76AF6A;
            border-color: #76AF6A;
            color: #000000;
        }
        QPushButton#send_btn:hover {
            background-color: #8BC77F;
        }
    """)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(16)
    layout.setContentsMargins(24, 24, 24, 24)

    # Title
    title = QLabel("Previous Session Ended Unexpectedly")
    title.setFont(QFont("", 16, QFont.Weight.DemiBold))
    title.setStyleSheet("color: #ffffff;")
    layout.addWidget(title)

    # Message
    message = QLabel(
        "It looks like NoodleStudio didn't shut down cleanly last time.\n\n"
        "Would you help us improve Studio by sending a crash report?\n"
        "This helps us identify and fix issues."
    )
    message.setWordWrap(True)
    message.setStyleSheet("color: #aaaaaa; line-height: 1.4;")
    layout.addWidget(message)

    # Show crash details if available
    if crash_info:
        details_label = QLabel("Details from last session:")
        details_label.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(details_label)

        details = QTextEdit()
        details.setReadOnly(True)
        details.setMaximumHeight(120)

        # Format crash info
        detail_lines = []
        if 'timestamp' in crash_info:
            detail_lines.append(f"Time: {crash_info['timestamp']}")
        if 'exception_type' in crash_info:
            detail_lines.append(f"Error: {crash_info['exception_type']}")
        if 'exception_value' in crash_info:
            val = crash_info['exception_value']
            if len(val) > 100:
                val = val[:100] + "..."
            detail_lines.append(f"Message: {val}")
        if 'version' in crash_info:
            detail_lines.append(f"Version: {crash_info['version']}")
        if 'sentinel_info' in crash_info:
            # Parse sentinel info for session details
            for line in crash_info['sentinel_info'].split('\n'):
                if line.startswith('started='):
                    detail_lines.append(f"Session started: {line.split('=', 1)[1]}")

        details.setPlainText('\n'.join(detail_lines) if detail_lines else "No details available")
        layout.addWidget(details)

    layout.addStretch()

    # Buttons
    btn_layout = QHBoxLayout()
    btn_layout.addStretch()

    dismiss_btn = QPushButton("Not Now")
    dismiss_btn.clicked.connect(dialog.reject)
    btn_layout.addWidget(dismiss_btn)

    send_btn = QPushButton("Send Crash Report")
    send_btn.setObjectName("send_btn")

    def send_report():
        """Open the full bug report dialog with crash context."""
        dialog.accept()
        try:
            from .dialogs.bug_report_dialog import BugReportDialog
            # Prepare crash info for the bug report dialog
            report_info = {
                'exception_type': crash_info.get('exception_type', 'Unknown'),
                'exception_value': crash_info.get('exception_value', 'Previous session ended unexpectedly'),
                'traceback': crash_info.get('traceback', crash_info.get('log_tail', '')),
            }
            report_dialog = BugReportDialog(parent, crash_info=report_info)
            report_dialog.exec()
        except Exception as e:
            print(f"[Crash Recovery] Could not open bug report dialog: {e}")

    send_btn.clicked.connect(send_report)
    btn_layout.addWidget(send_btn)

    layout.addLayout(btn_layout)

    # Clean up crash info file after showing dialog
    try:
        if CRASH_INFO_FILE.exists():
            CRASH_INFO_FILE.unlink()
    except:
        pass

    dialog.exec()


if __name__ == '__main__':
    main()
