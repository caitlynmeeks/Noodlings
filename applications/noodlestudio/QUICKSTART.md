# NoodleSTUDIO Quickstart Guide

Get started building NoodleSTUDIO in 30 minutes.

## Prerequisites

- Python 3.10+ installed
- Qt 6 installed (will be installed via pip)
- noodleMUSH running (for testing with live data)
- Code editor (VSCode recommended)

## Step 1: Setup Environment

```bash
cd /Users/thistlequell/git/noodlings_clean/applications/noodleSTUDIO

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Create Initial File Structure

```bash
# Create all directories
mkdir -p noodlestudio/{core,panels,widgets,data,dialogs,resources/{icons,fonts,styles}}
mkdir -p tests

# Create __init__.py files
touch noodlestudio/__init__.py
touch noodlestudio/core/__init__.py
touch noodlestudio/panels/__init__.py
touch noodlestudio/widgets/__init__.py
touch noodlestudio/data/__init__.py
touch noodlestudio/dialogs/__init__.py
```

## Step 3: Write Minimal Main Application

Create `noodlestudio/main.py`:

```python
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
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setApplicationName("NoodleSTUDIO")
    app.setApplicationVersion("1.0.0-alpha")
    app.setOrganizationName("Noodlings Project")

    # Create main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

Create `noodlestudio/core/main_window.py`:

```python
"""
Main Window for NoodleSTUDIO.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QMenuBar, QMenu,
    QToolBar, QStatusBar, QLabel
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction


class MainWindow(QMainWindow):
    """
    Main application window for NoodleSTUDIO.

    Contains:
    - Menu bar (File, View, Agent, Session, Tools, Help)
    - Tool bar (quick actions)
    - Dockable panel area (panels go here)
    - Status bar
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("NoodleSTUDIO - Noodlings IDE")
        self.resize(1400, 900)

        self._setup_ui()
        self._setup_menu_bar()
        self._setup_tool_bar()
        self._setup_status_bar()

    def _setup_ui(self):
        """Build UI components."""
        # Central widget (temporary - will be removed once we add dock widgets)
        central = QWidget()
        layout = QVBoxLayout()
        welcome = QLabel("NoodleSTUDIO v1.0.0-alpha\n\nWelcome to the Noodlings IDE!")
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome)
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Apply dark theme (placeholder)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0e1a;
                color: #e0e0e0;
            }
            QLabel {
                font-size: 18px;
                color: #64b5f6;
            }
        """)

    def _setup_menu_bar(self):
        """Create menu bar."""
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self._create_action("&New Recipe...", "Ctrl+N"))
        file_menu.addAction(self._create_action("&Open Recipe...", "Ctrl+O"))
        file_menu.addAction(self._create_action("&Save Recipe", "Ctrl+S"))
        file_menu.addSeparator()
        file_menu.addAction(self._create_action("&Quit", "Ctrl+Q", self.close))

        # View Menu
        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self._create_action("Chat View", "Ctrl+1"))
        view_menu.addAction(self._create_action("Log View", "Ctrl+2"))
        view_menu.addAction(self._create_action("Recipe Editor", "Ctrl+3"))

        # Agent Menu
        agent_menu = menu_bar.addMenu("&Agent")
        agent_menu.addAction(self._create_action("Spawn Agent...", "Ctrl+Shift+N"))

        # Session Menu
        session_menu = menu_bar.addMenu("&Session")
        session_menu.addAction(self._create_action("Load Session..."))

        # Tools Menu
        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction(self._create_action("Preferences..."))

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self._create_action("Documentation"))
        help_menu.addAction(self._create_action("About NoodleSTUDIO"))

    def _setup_tool_bar(self):
        """Create tool bar."""
        tool_bar = QToolBar("Main Toolbar")
        tool_bar.setIconSize(QSize(24, 24))
        self.addToolBar(tool_bar)

        # Add actions (no icons yet - will add in Phase 1)
        tool_bar.addAction(self._create_action("New", "Ctrl+N"))
        tool_bar.addAction(self._create_action("Open", "Ctrl+O"))
        tool_bar.addAction(self._create_action("Save", "Ctrl+S"))
        tool_bar.addSeparator()
        tool_bar.addAction(self._create_action("Play", "Space"))
        tool_bar.addAction(self._create_action("Pause"))
        tool_bar.addSeparator()
        tool_bar.addAction(self._create_action("Spawn Agent", "Ctrl+Shift+N"))

    def _setup_status_bar(self):
        """Create status bar."""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # Connection status
        status_bar.showMessage("● Not connected to noodleMUSH")

    def _create_action(self, text: str, shortcut: str = "", slot=None) -> QAction:
        """
        Create a QAction with text, shortcut, and optional slot.

        Args:
            text: Action text
            shortcut: Keyboard shortcut (e.g., "Ctrl+N")
            slot: Slot to connect to (optional)

        Returns:
            QAction instance
        """
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        if slot:
            action.triggered.connect(slot)
        return action
```

Create `run_studio.py` (launcher):

```python
#!/usr/bin/env python3
"""
Convenience launcher for NoodleSTUDIO.

Usage:
    python run_studio.py
"""

import sys
from pathlib import Path

# Add noodlestudio to path
sys.path.insert(0, str(Path(__file__).parent))

from noodlestudio.main import main

if __name__ == '__main__':
    main()
```

Make it executable:
```bash
chmod +x run_studio.py
```

## Step 4: Test the Minimal App

```bash
python run_studio.py
```

You should see a window with:
- Title: "NoodleSTUDIO - Noodlings IDE"
- Menu bar with File, View, Agent, Session, Tools, Help
- Tool bar with placeholder buttons
- Status bar showing "● Not connected to noodleMUSH"
- Central area with welcome message

**If it works**, you're ready to start Phase 1!

## Step 5: Add First Panel (Chat View)

Create `noodlestudio/panels/chat_panel.py`:

```python
"""
Chat panel - embeds noodleMUSH web interface.
"""

from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl


class ChatPanel(QDockWidget):
    """
    Chat panel displaying noodleMUSH web interface.

    Embeds the existing web/index.html via QWebEngineView.
    """

    def __init__(self, parent: QWidget = None):
        super().__init__("Chat View", parent)

        # Don't allow closing (for now)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._setup_ui()

    def _setup_ui(self):
        """Build UI components."""
        # Create web view
        self.web_view = QWebEngineView()

        # Try to load noodleMUSH
        # If server is running, this will work
        # If not, will show error page
        self.web_view.setUrl(QUrl("http://localhost:8080"))

        # Set as widget
        self.setWidget(self.web_view)

    def reload(self):
        """Reload the web page."""
        self.web_view.reload()
```

Update `noodlestudio/core/main_window.py` to add the chat panel:

```python
# Add to imports at top:
from ..panels.chat_panel import ChatPanel

# Add to __init__ method after _setup_status_bar():
        self._setup_panels()

# Add new method:
    def _setup_panels(self):
        """Create and add dock panels."""
        # Chat panel
        self.chat_panel = ChatPanel(self)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.chat_panel
        )
```

Update `noodlestudio/panels/__init__.py`:

```python
"""Panels for NoodleSTUDIO."""

from .chat_panel import ChatPanel

__all__ = ['ChatPanel']
```

## Step 6: Test with Real noodleMUSH Server

1. **Start noodleMUSH** in another terminal:
   ```bash
   cd /Users/thistlequell/git/noodlings_clean/applications/cmush
   ./start.sh
   ```

2. **Launch NoodleSTUDIO**:
   ```bash
   cd /Users/thistlequell/git/noodlings_clean/applications/noodleSTUDIO
   python run_studio.py
   ```

3. **You should see**:
   - NoodleSTUDIO window
   - Chat panel on the right showing noodleMUSH interface
   - You can interact with noodleMUSH from within NoodleSTUDIO!

## Step 7: Next Steps

You've completed the minimal viable app! 🎉

### Phase 1 Tasks (Week 1)

Now implement:
1. **Log panel** (similar to chat panel, but shows logs)
2. **Recipe editor panel** (native Qt widgets)
3. **Layout save/load system** (save panel positions to JSON)
4. **Dark theme stylesheet** (create `resources/styles/dark.qss`)
5. **Panel visibility toggles** (View menu checkboxes)

### Recommended Order

1. Add log panel (copy chat_panel.py, modify for logs)
2. Add recipe editor panel (most complex - follow IMPLEMENTATION_PLAN.md Day 3-5)
3. Test docking/undocking/floating all panels
4. Add layout save/load (save to `~/.noodlestudio/layouts/default.json`)
5. Polish dark theme (create QSS file, load in main_window.py)

## Development Tips

### Hot Reloading

For faster development, use `python -m noodlestudio.main` and keep the app running. When you modify code, just close and rerun.

### Debugging

Add this to see Qt warnings:
```python
import os
os.environ['QT_DEBUG_PLUGINS'] = '1'
```

### Testing Layouts

Save your layout via code:
```python
# In main_window.py
def save_layout(self):
    """Save current layout to file."""
    state = self.saveState()
    with open('layout.dat', 'wb') as f:
        f.write(state.data())
```

Load it:
```python
def load_layout(self):
    """Load layout from file."""
    with open('layout.dat', 'rb') as f:
        state = QByteArray(f.read())
        self.restoreState(state)
```

### Using Qt Designer

If you prefer visual design:
1. Install Qt Designer: `pip install pyqt6-tools`
2. Run: `pyqt6-tools designer`
3. Design your UI, save as `.ui` file
4. Convert to Python: `pyuic6 panel.ui -o panel_ui.py`
5. Import and use: `from .panel_ui import Ui_Panel`

## Troubleshooting

### "ImportError: No module named PyQt6"
```bash
pip install PyQt6 PyQt6-WebEngine
```

### "Cannot load libQt6Core.so.6"
Qt libraries not found. Reinstall:
```bash
pip uninstall PyQt6
pip install PyQt6
```

### "QWebEngineView not found"
WebEngine not installed:
```bash
pip install PyQt6-WebEngine
```

### Chat panel shows blank/error
noodleMUSH server not running. Start it:
```bash
cd ../cmush && ./start.sh
```

### Window is too small/big
Adjust in main_window.py:
```python
self.resize(1920, 1080)  # Or whatever size you want
```

## Resources

- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [Qt Documentation](https://doc.qt.io/)
- [NoodleSTUDIO Architecture](docs/ARCHITECTURE.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [UI Mockups](docs/UI_MOCKUPS.md)

## Questions?

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design or ask Claude!

---

**Next Milestone**: Complete Phase 1 (Main window + Chat + Logs + Recipe Editor + Layouts)

**Estimated Time**: 1 week for Phase 1, 5 weeks total for v1.0

Good luck! 🧠✨
