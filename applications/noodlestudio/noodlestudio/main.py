"""
NoodleSTUDIO Main Application

Entry point for the NoodleSTUDIO IDE.
"""

import sys
import traceback
from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QFont, QColor

# NOTE: MainWindow imported AFTER QApplication is created
from .core.studio_acronyms import get_random_acronym
from . import __version__


# Global reference to main window for crash reporter
_main_window = None


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

    # Install crash reporter
    install_crash_reporter()

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


if __name__ == '__main__':
    main()
