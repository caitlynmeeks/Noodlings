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

    # Close splash and show main window
    splash.finish(window)
    window.show()

    sys.exit(app.exec())


def create_splash_screen():
    """Create a splash screen with a random STUDIO acronym - Green TUI aesthetic."""
    # Create a larger pixmap - easy to read!
    pixmap = QPixmap(1000, 700)
    pixmap.fill(Qt.GlobalColor.black)

    # Draw on it
    painter = QPainter(pixmap)

    # ASCII Art Banner
    painter.setPen(QColor(0, 255, 0))  # Green
    font = QFont("Courier New", 7)  # Small for the banner
    painter.setFont(font)
    banner = (
        ":::.    :::.    ...         ...    :::::::-.   :::    .,::::::      .        :    ...    ::: .::::::.   ::   .:\n"
        "`;;;;,  `;;; .;;;;;;;.   .;;;;;;;.  ;;,   `';, ;;;    ;;;;''''      ;;,.    ;;;   ;;     ;;;;;;`    `  ,;;   ;;,\n"
        "  [[[[[. '[[,[[     \\[[,,[[     \\[[,`[[     [[ [[[     [[cccc       [[[[, ,[[[[, [['     [[['[==/[[[[,,[[[,,,[[[\\n"
        "  $$$ \"Y$c$$$$$,     $$$$$$,     $$$ $$,    $$ $$'     $$\"\"\"\"       $$$$$$$$\"$$$ $$      $$$  '''    $\"$$$\"\"\"$$$\n"
        "  888    Y88\"888,_ _,88P\"888,_ _,88P 888_,o8P'o88oo,.__888oo,__     888 Y88\" 888o88    .d888 88b    dP 888   \"88o\n"
        "  MMM     YM  \"YMMMMMP\"   \"YMMMMMP\"  MMMMP\"`  \"\"\"\"YUMMM\"\"\"\"YUMMM    MMM  M'  \"MMM \"YmmMMMM\"\"  \"YMmMY\"  MMM    YMM"
    )
    painter.drawText(pixmap.rect().adjusted(20, 20, -20, -550),
                     Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
                     banner)

    # Subtitle
    painter.setPen(QColor(100, 255, 100))  # Light green
    font = QFont("Courier New", 16)
    painter.setFont(font)
    painter.drawText(pixmap.rect().adjusted(40, 160, -40, -450),
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
    text_rect = pixmap.rect().adjusted(60, 280, -60, -150)
    painter.drawText(text_rect,
                     Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                     acronym)

    # Version at bottom
    font = QFont("Courier New", 16)
    painter.setFont(font)
    painter.setPen(QColor(80, 80, 80))  # Dark gray
    painter.drawText(pixmap.rect().adjusted(40, -80, -40, -40),
                     Qt.AlignmentFlag.AlignCenter,
                     "v1.0.0-alpha")

    painter.end()

    splash = QSplashScreen(pixmap)
    return splash


if __name__ == '__main__':
    main()
