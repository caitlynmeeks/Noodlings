"""
NoodleSTUDIO Main Application

Entry point for the NoodleSTUDIO IDE.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from .core.main_window import MainWindow


def main():
    """Launch NoodleSTUDIO."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

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
    app.setApplicationVersion("1.0.0-alpha")
    app.setOrganizationName("Consilience")
    app.setOrganizationDomain("noodlings.ai")

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
