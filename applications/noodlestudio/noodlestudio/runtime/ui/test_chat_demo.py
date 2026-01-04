#!/usr/bin/env python3
"""
Test script for chat components.

This creates a simple chat UI for visual verification.
Run from: applications/noodlestudio
Command: PYTHONPATH=.:../.. python3 noodlestudio/runtime/ui/test_chat_demo.py
"""

import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from noodlestudio.runtime.ui import (
    UILoader, QtWidgetRenderer, AnchoredWidget, UIEventDispatcher
)
from noodlestudio.runtime.ui.components.chat_history import MessageRole


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Load the demo chat UI
    loader = UILoader()
    demo_path = Path(__file__).parent / "demo_chat.yaml"

    if demo_path.exists():
        root = loader.load_file(demo_path)
        print(f"Loaded UI: {demo_path.name}")
    else:
        print(f"Demo file not found: {demo_path}")
        return 1

    # Create renderer
    renderer = QtWidgetRenderer()

    # Create event dispatcher (demo mode - no app)
    dispatcher = UIEventDispatcher(renderer)
    renderer.set_event_dispatcher(dispatcher.dispatch)

    # Create anchored widget container
    window = AnchoredWidget(root, renderer)
    window.setWindowTitle("Chat Demo - Phase 3b")
    window.resize(500, 500)

    # Add some demo messages after a short delay
    def add_demo_messages():
        chat_history = renderer.get_component("chat_history")
        if chat_history:
            # Add welcome message from noodling
            chat_history.add_message(
                role=MessageRole.NOODLING,
                content="Hello! I'm Red, your friendly AI companion. How can I help you today?",
                sender_name="Red"
            )

            # Add a system message
            chat_history.add_message(
                role=MessageRole.SYSTEM,
                content="This is a demo. Type a message and press Send."
            )

    QTimer.singleShot(500, add_demo_messages)

    window.show()

    print("Chat demo launched!")
    print(f"Window: {window.width()}x{window.height()}")
    print("Type a message and press Send or Enter to test.")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
