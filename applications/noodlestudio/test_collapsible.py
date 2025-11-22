#!/usr/bin/env python3
"""
Minimal test case for CollapsibleSection double-trigger debugging.

Run this standalone to isolate the issue from NoodleStudio complexity.
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from noodlestudio.widgets.collapsible_section import CollapsibleSection


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CollapsibleSection Test")
        self.setGeometry(100, 100, 400, 300)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Add QScrollArea (like Inspector uses)
        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # Create test sections
        section1 = CollapsibleSection("Test Section 1")
        section1.add_widget(QLabel("Content for section 1"))
        section1.add_widget(QLabel("More content here"))

        section2 = CollapsibleSection("Test Section 2")
        section2.add_widget(QLabel("Content for section 2"))

        section3 = CollapsibleSection("Test Section 3")
        section3.add_widget(QLabel("Content for section 3"))

        scroll_layout.addWidget(section1)
        scroll_layout.addWidget(section2)
        scroll_layout.addWidget(section3)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        print("\n" + "="*60)
        print("COLLAPSIBLE SECTION TEST")
        print("="*60)
        print("Instructions:")
        print("1. Click each section header ONCE")
        print("2. Observe console output")
        print("3. Check if toggle() is called once or twice")
        print("="*60 + "\n")


def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
