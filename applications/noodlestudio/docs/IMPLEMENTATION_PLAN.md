# NoodleSTUDIO Implementation Plan

**Version**: 1.0
**Date**: November 15, 2025
**Timeline**: 5 weeks
**Framework**: PyQt6

## Directory Structure

```
applications/noodleSTUDIO/
├── docs/
│   ├── ARCHITECTURE.md          # System architecture (this file's companion)
│   ├── IMPLEMENTATION_PLAN.md   # This file
│   ├── API_REFERENCE.md         # API docs (to be created)
│   └── USER_GUIDE.md            # User manual (to be created)
│
├── noodlestudio/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── main_window.py       # QMainWindow subclass
│   │   ├── config.py            # App configuration
│   │   ├── layout_manager.py   # Save/load panel layouts
│   │   └── theme.py             # Color scheme, fonts, styling
│   │
│   ├── panels/
│   │   ├── __init__.py
│   │   ├── base_panel.py        # QDockWidget base class
│   │   ├── chat_panel.py        # Web view for chat
│   │   ├── log_panel.py         # Web view for logs
│   │   ├── recipe_editor.py     # Recipe editing UI
│   │   ├── phenomenal_state.py  # 40-D state visualization
│   │   ├── timeline.py          # Timeline profiler
│   │   ├── analytics.py         # Analytics dashboard
│   │   ├── operations.py        # Operations console
│   │   └── agent_manager.py     # Agent spawning/management
│   │
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── state_bars.py        # Bar charts for phenomenal state
│   │   ├── radar_chart.py       # Affect radar chart
│   │   ├── timeline_track.py    # Single timeline track
│   │   ├── playhead.py          # Timeline playhead control
│   │   ├── event_marker.py      # Timeline event markers
│   │   └── metric_card.py       # Analytics metric display
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── session_loader.py    # Load SessionProfiler JSON
│   │   ├── api_client.py        # HTTP client for noodleMUSH API
│   │   ├── websocket_client.py  # WebSocket client for live updates
│   │   └── recipe_manager.py    # Recipe CRUD operations
│   │
│   ├── dialogs/
│   │   ├── __init__.py
│   │   ├── spawn_agent.py       # Spawn agent dialog
│   │   ├── inspect_agent.py     # Agent inspection dialog
│   │   ├── preferences.py       # Preferences dialog
│   │   ├── export_segment.py    # Export time range dialog
│   │   └── about.py             # About dialog
│   │
│   └── resources/
│       ├── icons/               # SVG icons
│       ├── fonts/               # Source Code Pro, Roboto
│       └── styles/
│           └── dark.qss         # Qt stylesheet for dark theme
│
├── tests/
│   ├── test_session_loader.py
│   ├── test_recipe_manager.py
│   └── test_timeline.py
│
├── requirements.txt
├── setup.py
├── README.md
└── run_studio.py                # Convenience launcher
```

## File Breakdown by Implementation Phase

### Phase 1: Foundation (Week 1)

#### Day 1-2: Project Setup
- [x] Create directory structure
- [ ] Write `requirements.txt`
- [ ] Write `setup.py` for pip installation
- [ ] Create `run_studio.py` launcher
- [ ] Implement `core/config.py` (load from YAML)
- [ ] Implement `core/theme.py` (color constants, QSS stylesheet)

#### Day 3-4: Main Window
- [ ] Implement `core/main_window.py`:
  - QMainWindow with menu bar
  - Tool bar
  - Status bar
  - QDockWidget area (central widget)
- [ ] Implement `core/layout_manager.py`:
  - Save layout to JSON
  - Load layout from JSON
  - Restore dock positions, sizes, visibility
- [ ] Implement menu bar structure (File, View, Agent, Session, Tools, Help)

#### Day 5-7: Web Panels
- [ ] Implement `panels/base_panel.py`:
  - Base QDockWidget with common functionality
  - Title bar styling
  - Resize handling
- [ ] Implement `panels/chat_panel.py`:
  - QWebEngineView pointing to http://localhost:8080
  - Connection status indicator
  - Reload button
- [ ] Implement `panels/log_panel.py`:
  - QWebEngineView for logs OR native QTextEdit with ANSI color support
  - Filter controls (level, search)
  - Auto-scroll toggle
- [ ] Apply dark theme stylesheet
- [ ] Test panel docking/undocking/floating

**Deliverable**: Launchable app with Chat and Log panels, dockable layout

---

### Phase 2: Recipe Editor (Week 2)

#### Day 1-2: Recipe Data Layer
- [ ] Implement `data/recipe_manager.py`:
  - Load recipe from YAML (reuse existing recipe_loader.py)
  - Save recipe to YAML
  - Validate recipe
  - List available recipes
  - Create new recipe from template

#### Day 3-5: Recipe Editor UI
- [ ] Implement `panels/recipe_editor.py`:
  - Recipe selector dropdown
  - Identity section (name, species, description, age, pronouns, identity_prompt)
  - Personality sliders (8 dimensions)
  - Appetites sliders (8 dimensions)
  - Constraints section (language_mode, max_tokens, temperature, etc.)
  - Validation status indicator (green ✓ or red ✗ with error list)
  - Save/Save As/Reload buttons
- [ ] Implement `widgets/radar_chart.py`:
  - QPainter-based pentagon radar chart for appetites
  - Customizable colors, labels, scale
- [ ] Wire up signals:
  - valueChanged → validate()
  - save_clicked → recipe_manager.save()
  - reload_clicked → recipe_manager.load()

#### Day 6-7: Testing & Polish
- [ ] Test recipe loading/saving
- [ ] Test validation (invalid values, missing fields)
- [ ] Add tooltips to all fields (explain what each dimension means)
- [ ] Test Apply to Running Agent (if agent is spawned)

**Deliverable**: Fully functional recipe editor with live validation

---

### Phase 3: Phenomenal State View (Week 2)

#### Day 1-2: State Visualization Widgets
- [ ] Implement `widgets/state_bars.py`:
  - Custom QWidget for single state bar
  - Bar color (layer-specific)
  - Value label (-1.0 to 1.0)
  - Smooth animation on value change
- [ ] Implement layered state widget:
  - Fast layer (16 bars, green)
  - Medium layer (16 bars, orange)
  - Slow layer (8 bars, purple)
  - Collapsible sections

#### Day 3-4: Phenomenal State Panel
- [ ] Implement `panels/phenomenal_state.py`:
  - Agent selector dropdown
  - Time slider (for scrubbing) OR "Live" toggle
  - State bars widget
  - Affect radar chart widget
  - Update from session data or real-time feed

#### Day 5: Real-time Updates
- [ ] Implement `data/websocket_client.py`:
  - Connect to ws://localhost:8765
  - Listen for agent state updates
  - Emit Qt signals when state changes
- [ ] Wire up WebSocket → Phenomenal State View
- [ ] Test: Spawn agent in noodleMUSH, watch state update in real-time

**Deliverable**: Live phenomenal state visualization with 40-D bars + affect chart

---

### Phase 4: Timeline Profiler (Week 3)

#### Day 1-2: Data Loading
- [ ] Implement `data/session_loader.py`:
  - Parse SessionProfiler JSON
  - Build timeline data structures (per agent)
  - Index by timestamp for fast lookups
- [ ] Implement `data/api_client.py`:
  - GET /api/profiler/live-session
  - Poll every 1s in real-time mode
  - Append new data points to timeline

#### Day 3-5: Timeline Rendering
- [ ] Implement `widgets/timeline_track.py`:
  - QGraphicsPathItem for line chart
  - Configurable color, scale, label
  - Efficient rendering (only draw visible range)
- [ ] Implement `widgets/playhead.py`:
  - QGraphicsLineItem for vertical playhead line
  - Draggable (QGraphicsItem::ItemIsMovable)
  - Emit signal when moved
- [ ] Implement `widgets/event_marker.py`:
  - QGraphicsEllipseItem or QGraphicsPixmapItem for marker
  - Tooltip with event details
  - Clickable (emit signal when clicked)
- [ ] Implement `panels/timeline.py`:
  - QGraphicsView + QGraphicsScene
  - Add multiple tracks (Surprise, Valence, Arousal, Fear, HSI)
  - Add playhead
  - Add event markers (speech, thought, surprise spikes)
  - Zoom controls (scroll wheel, buttons)
  - Time axis labels (0s, 10s, 20s, ...)

#### Day 6-7: Playback Controls
- [ ] Add play/pause buttons
- [ ] Implement auto-advance (QTimer at 60fps)
- [ ] Implement keyboard shortcuts (Space, ←/→)
- [ ] Implement jump to spike/event
- [ ] Test scrubbing with session data

**Deliverable**: Fully functional timeline with scrubbing, playback, event markers

---

### Phase 5: Operations Console (Week 3)

#### Day 1-2: Operations Data
- [ ] Extend `data/api_client.py`:
  - GET /api/performance/operations
  - Parse PerformanceTracker data
  - Structure: List[Operation] with timestamp, agent, type, duration, status

#### Day 3-4: Operations UI
- [ ] Implement `panels/operations.py`:
  - QTableView or QListWidget for operations log
  - Columns: Timestamp, Agent, Operation, Duration, Status
  - Color-coded rows (green=fast, orange=medium, red=slow)
  - Filter controls (agent, operation type, status)
  - Search box
  - Auto-scroll toggle
  - Export button

#### Day 5: Real-time Updates
- [ ] Poll API every 1s for new operations
- [ ] Append to table
- [ ] Scroll to bottom if auto-scroll enabled
- [ ] Test with live noodleMUSH session

**Deliverable**: Real-time operations log with filtering and search

---

### Phase 6: Analytics Dashboard (Week 4)

#### Day 1-2: Metrics Calculation
- [ ] Implement analytics engine:
  - Load session data
  - Calculate HSI (already in session JSON)
  - Calculate avg surprise, speech rate
  - Calculate layer velocities
  - Calculate affect distributions (histograms)
  - Calculate performance metrics (LLM latency, etc.)

#### Day 3-5: Analytics UI
- [ ] Implement `panels/analytics.py`:
  - QTabWidget with 5 tabs:
    1. Metrics Overview (QGridLayout with MetricCards)
    2. Affect Distribution (QCharts histograms)
    3. Surprise Analysis (QCharts histogram + table)
    4. Performance Metrics (QCharts pie chart + stats)
    5. Consciousness Metrics (if available)
- [ ] Implement `widgets/metric_card.py`:
  - Display single metric with label, value, status, progress bar
- [ ] Use QCharts for visualizations (histograms, pie charts)

**Deliverable**: Analytics dashboard with 5 tabs of metrics

---

### Phase 7: Agent Manager (Week 4)

#### Day 1-2: Agent API Integration
- [ ] Extend `data/api_client.py`:
  - GET /api/agents/list
  - GET /api/agents/{agent_id}/state
  - POST /api/agents/spawn
  - DELETE /api/agents/{agent_id}
- [ ] Test API endpoints with noodleMUSH

#### Day 3-4: Agent Manager UI
- [ ] Implement `panels/agent_manager.py`:
  - List active agents (QListWidget or QScrollArea with custom widgets)
  - Show agent status, room, enlightenment
  - Inspect, Kill, Restart buttons per agent
  - Spawn section at bottom (recipe dropdown + Spawn button)
- [ ] Implement `dialogs/spawn_agent.py`:
  - Recipe selector
  - Room selector
  - Enlightenment toggle
  - Spawn confirmation
- [ ] Implement `dialogs/inspect_agent.py`:
  - Full agent state (phenomenal, affect, memory, etc.)
  - Conversation context
  - Recent operations

#### Day 5: Testing
- [ ] Test spawning agents
- [ ] Test killing agents
- [ ] Test inspection dialog

**Deliverable**: Agent manager with spawn/kill/inspect

---

### Phase 8: Polish & Testing (Week 5)

#### Day 1-2: Keyboard Shortcuts
- [ ] Implement command palette (Ctrl+/)
- [ ] Implement global shortcuts (Ctrl+N, Ctrl+O, etc.)
- [ ] Implement timeline shortcuts (Space, ←/→, etc.)
- [ ] Implement panel shortcuts (Ctrl+1-8)

#### Day 3-4: Preferences & Dialogs
- [ ] Implement `dialogs/preferences.py`:
  - Theme selection (dark/light)
  - Default layout
  - API endpoints
  - Refresh intervals
  - Keyboard shortcuts (editable)
- [ ] Implement `dialogs/about.py`:
  - Version info
  - Credits
  - Links to docs

#### Day 5: Error Handling
- [ ] Add try/catch blocks for all API calls
- [ ] Show error dialogs on failure
- [ ] Validate user input everywhere
- [ ] Log errors to console

#### Day 6-7: Testing & Documentation
- [ ] Write unit tests for key components
- [ ] Write integration tests (mock API responses)
- [ ] Write user guide (USER_GUIDE.md)
- [ ] Write API reference (API_REFERENCE.md)
- [ ] Record demo video

**Deliverable**: Polished, tested, documented NoodleSTUDIO v1.0

---

## Code Style Guidelines

### Python
- PEP 8 compliant
- Type hints for all function signatures
- Docstrings for all classes and public methods
- Use f-strings for string formatting

### Qt
- Use signals/slots for communication between components
- Avoid blocking the main thread (use QThread for long operations)
- Use QSettings for persistent configuration
- Use QSS for styling (avoid inline styles)

### Example Module Structure

```python
"""
Module docstring explaining what this module does.

Author: Your Name
Date: 2025-11-15
"""

from typing import Optional, List
from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal

from ..core.theme import COLORS, FONTS
from ..data.session_loader import SessionData


class PhenomenalStatePanel(QDockWidget):
    """
    Panel for visualizing 40-D phenomenal state in real-time.

    Displays:
    - Fast layer (16-D, green bars)
    - Medium layer (16-D, orange bars)
    - Slow layer (8-D, purple bars)
    - 5-D affect radar chart

    Signals:
        agent_changed: Emitted when user selects different agent
    """

    agent_changed = pyqtSignal(str)  # agent_id

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize phenomenal state panel.

        Args:
            parent: Parent widget
        """
        super().__init__("Phenomenal State", parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Build UI components."""
        # Implementation...
        pass

    def _connect_signals(self):
        """Connect Qt signals to slots."""
        # Implementation...
        pass

    def update_state(self, state: SessionData.TimelinePoint):
        """
        Update displayed state.

        Args:
            state: Timeline data point with phenomenal state
        """
        # Implementation...
        pass
```

## Testing Strategy

### Unit Tests
- Test data loading (session_loader, recipe_manager)
- Test calculations (HSI, metrics, etc.)
- Test API client (mock responses)

### Integration Tests
- Test panel communication (timeline → phenomenal state sync)
- Test layout save/load
- Test WebSocket client with mock server

### Manual Testing
- Test with real noodleMUSH session
- Test with 1000+ data points (performance)
- Test all keyboard shortcuts
- Test on different screen sizes

## Deployment

### Development
```bash
cd applications/noodleSTUDIO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_studio.py
```

### Distribution (v1.0)
- Use pyinstaller to create standalone executable
- Bundle resources (icons, fonts, QSS)
- Create DMG installer for macOS
- Write installation instructions

### Future (v2.0+)
- Port to Qt C++ for better performance
- Create Windows/Linux builds
- Publish to PyPI as `noodlestudio`

## Success Checklist

NoodleSTUDIO v1.0 is ready for release when:

- [ ] All 8 panels implemented and functional
- [ ] Layout save/load works correctly
- [ ] Real-time updates work (WebSocket + API)
- [ ] Timeline scrubbing is smooth (60fps)
- [ ] Recipe editor validates correctly
- [ ] Agent manager can spawn/kill/inspect
- [ ] Analytics dashboard shows all 5 tabs
- [ ] Operations console filters and searches
- [ ] Dark theme looks good on all panels
- [ ] Keyboard shortcuts work
- [ ] Error handling prevents crashes
- [ ] Documentation is complete (USER_GUIDE.md)
- [ ] Demo video shows all features
- [ ] Tested with real session data
- [ ] Performance is acceptable (no lag with 1000+ points)

---

**Next Steps**: Start with Phase 1 (Foundation). Get main window, menu bar, and web panels working first. Then iterate on data-driven native panels.
