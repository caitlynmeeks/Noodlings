"""
NoodleSTUDIO Main Application

Entry point for the NoodleSTUDIO IDE.
"""

import sys
from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QFont, QColor

from .core.main_window import MainWindow
from .core.studio_acronyms import get_random_acronym


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

    # Create splash screen with random acronym
    splash = create_splash_screen()
    splash.show()
    app.processEvents()

    # Create main window (takes a moment to load)
    window = MainWindow()

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
                     "NoodleSTUDIO")

    # Random acronym (pick one style randomly - NO AUTHOR ATTRIBUTION)
    import random
    style = random.choice(["compassionate", "pratchett", "coupland", "bjork"])
    acronym = get_random_acronym(style)

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
                     "v1.0.0-alpha")

    painter.end()

    splash = QSplashScreen(pixmap)
    return splash


if __name__ == '__main__':
    main()
