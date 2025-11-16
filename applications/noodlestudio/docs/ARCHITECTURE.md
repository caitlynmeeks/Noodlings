# NoodleSTUDIO Architecture Specification

**Version**: 1.0
**Date**: November 15, 2025
**Framework**: Qt 6 (C++) or PyQt6/PySide6 (Python)
**Target Platform**: macOS (M3 Ultra primary)

## Overview

NoodleSTUDIO is a comprehensive IDE for developing, monitoring, and analyzing Noodling consciousness agents. It provides real-time visualization, recipe editing, performance profiling, and timeline analysis in a unified, flexible interface.

## Design Philosophy

1. **Flexibility First**: Drag-and-drop panels, save/load layouts, dock anywhere
2. **Web + Native Hybrid**: Chat/logs in web panels, analytics in native Qt
3. **Real-time Performance**: Sub-100ms updates, efficient rendering
4. **Data-Driven**: All views backed by session profiler data + performance tracker
5. **Epistemic Clarity**: Always show what you're looking at (session, agent, time range)

## Core Components

### 1. Main Window (QMainWindow)

**Structure**:
```
┌────────────────────────────────────────────────────┐
│  Menu Bar                                          │
├────────────────────────────────────────────────────┤
│  Tool Bar (Session, Agent, View Controls)         │
├────────────────────────────────────────────────────┤
│                                                    │
│  [Dockable Panel Area - Flexible Layout]          │
│                                                    │
│  Default Layout:                                   │
│  ┌───────────┬─────────────────┬────────────────┐ │
│  │           │                 │                │ │
│  │  Recipe   │   Chat View     │  Phenomenal    │ │
│  │  Editor   │   (Web Panel)   │  State View    │ │
│  │           │                 │                │ │
│  │  (Native) ├─────────────────┤  (Native)      │ │
│  │           │                 │                │ │
│  │           │   Log View      │                │ │
│  │           │   (Web Panel)   │                │ │
│  └───────────┴─────────────────┴────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │  Timeline Profiler (Native - Full Width)     │ │
│  │  [Unity-style scrubber with HSI/Surprise]    │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
└────────────────────────────────────────────────────┘
```

**Features**:
- All panels are QDockWidget instances
- Drag to reposition, stack, or float
- Save/load layouts to JSON
- Quick layout presets: "Development", "Analysis", "Performance", "Theater"

### 2. Panel Types

#### A. Web Panels (QWebEngineView)

**Chat View**:
- Embeds existing `web/index.html`
- Full noodleMUSH interface
- Auto-connects to WebSocket server

**Log View**:
- Real-time server logs
- Filterable by level (INFO, DEBUG, ERROR)
- Searchable
- Color-coded output

**Implementation Note**:
```python
from PyQt6.QtWebEngineWidgets import QWebEngineView

class ChatPanel(QDockWidget):
    def __init__(self):
        super().__init__("Chat View")
        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl("http://localhost:8080"))
        self.setWidget(self.web_view)
```

#### B. Native Qt Panels

##### 2.1 Recipe Editor

**Purpose**: Edit agent YAML recipes with live validation

**UI Components**:
- **Recipe Selector**: Dropdown of available recipes + "New" button
- **Identity Section**:
  - Name (QLineEdit)
  - Species (QLineEdit)
  - Description (QTextEdit, 200 chars)
  - Age (QLineEdit)
  - Pronouns (QLineEdit)
  - Identity Prompt (QTextEdit, monospace font)

- **Personality Section** (8-D slow layer):
  - Extraversion (QSlider 0-1, 2 decimals)
  - Impulsivity (QSlider 0-1)
  - Curiosity (QSlider 0-1)
  - Emotional Volatility (QSlider 0-1)
  - Vanity (QSlider 0-1)
  - Each slider shows: Label | Slider | Value | Info icon (tooltip)

- **Appetites Section** (8-D Phase 6):
  - Curiosity (QSlider 0-1)
  - Status (QSlider 0-1)
  - Mastery (QSlider 0-1)
  - Novelty (QSlider 0-1)
  - Safety (QSlider 0-1)
  - Social Bond (QSlider 0-1)
  - Comfort (QSlider 0-1)
  - Autonomy (QSlider 0-1)
  - Visual summary: Radar chart showing all 8 appetites

- **Constraints Section**:
  - Language Mode (QComboBox: verbal/nonverbal)
  - Max Tokens (QSpinBox, 10-1000)
  - Temperature (QDoubleSpinBox, 0.0-2.0, step 0.1)
  - Response Cooldown (QDoubleSpinBox, 0-60 seconds)
  - Enlightenment (QCheckBox)
  - Enforce Action Format (QCheckBox)

- **Validation Status**:
  - Green checkmark ✓ if valid
  - Red X ✗ with error list if invalid
  - Live validation as user types

- **Actions**:
  - Save (Ctrl+S)
  - Save As... (Ctrl+Shift+S)
  - Reload from disk
  - Apply to Running Agent (if spawned)

**Data Flow**:
```
Recipe YAML ──read──> RecipeEditor ──edit──> Save ──write──> Recipe YAML
                                      │
                                      └──Apply──> Running Agent (via WebSocket command)
```

##### 2.2 Phenomenal State View

**Purpose**: Real-time visualization of 40-D phenomenal state

**Layout**:
```
┌─────────────────────────────────────┐
│  Agent: [Dropdown]  Time: [Slider]  │
├─────────────────────────────────────┤
│  Fast Layer (16-D) - GREEN          │
│  ████████████░░░░░░░░░░░░░░░░  0.72 │
│  ███████░░░░░░░░░░░░░░░░░░░░░  0.54 │
│  ... (16 bars)                      │
├─────────────────────────────────────┤
│  Medium Layer (16-D) - ORANGE       │
│  ████████████████░░░░░░░░░░░  0.85 │
│  ... (16 bars)                      │
├─────────────────────────────────────┤
│  Slow Layer (8-D) - PURPLE          │
│  ██████████████████████░░░░░  0.92 │
│  ... (8 bars)                       │
├─────────────────────────────────────┤
│  [Radar Chart: 5-D Affect Vector]   │
│  Valence, Arousal, Fear, Sorrow,    │
│  Boredom plotted on pentagon        │
└─────────────────────────────────────┘
```

**Features**:
- Bar charts for each dimension (QProgressBar or custom QPainter)
- Color-coded by layer (Fast=green, Medium=orange, Slow=purple)
- Numeric values displayed next to bars
- Radar chart for affect (using QPainter or QChart)
- Updates at 10Hz when in real-time mode
- Scrubbing mode: Updates when timeline slider moves

**Implementation**:
```python
class PhenomenalStatePanel(QDockWidget):
    def __init__(self):
        super().__init__("Phenomenal State")
        self.state_widget = StateVisualizationWidget()
        self.setWidget(self.state_widget)

    def update_state(self, phenomenal_state: np.ndarray):
        fast = phenomenal_state[:16]
        medium = phenomenal_state[16:32]
        slow = phenomenal_state[32:40]
        self.state_widget.update_layers(fast, medium, slow)
```

##### 2.3 Timeline Profiler

**Purpose**: Unity-style timeline with scrubbing, annotations, metrics

**Inspiration**: Unity Timeline Editor, Chrome DevTools Performance Tab

**Layout**:
```
┌───────────────────────────────────────────────────────────────────┐
│  Session: [session_20251115_183422 ▼]  Agent: [agent_callie ▼]   │
├───────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  Surprise    ╱╲        ╱╲╲        ╱╲                   │ │  │
│  │  │  (0-1)    ══╱══╲══════╱══╲╲══════╱══╲════════════════  │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  Valence   ══════╱╲═══════════════════╱╲═══════════    │ │  │
│  │  │  (-1 to 1)     ══╱══╲══════════════════╱══╲═══════     │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  Arousal     ════════╱╲╲╱╲═════════════════            │ │  │
│  │  │  (0-1)       ════════╱══╲╲╱╲═════════════════          │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  HSI         ════════════════════════════════════       │ │  │
│  │  │  (variance)  ════════════════════════════════════       │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                              │  │
│  │  ╎ ← Playhead (draggable)                                   │  │
│  │  0s      10s      20s      30s      40s      50s      60s   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Event Markers:                                            │   │
│  │  💬 Speech   🧠 Thought   ⚡ Surprise Spike   🎭 Play Start │   │
│  └────────────────────────────────────────────────────────────┘   │
│  [▶ Play] [⏸ Pause] [⏮ Reset] [⏭ Jump to Spike]                 │
│  [Export Segment] [Ask @Kimmie]                                  │
└───────────────────────────────────────────────────────────────────┘
```

**Features**:
1. **Multi-track timeline** (QGraphicsView + QGraphicsScene):
   - Surprise track (0-1 scale)
   - Valence track (-1 to 1)
   - Arousal track (0-1)
   - Fear track (0-1)
   - HSI track (variance ratio)
   - Custom tracks: Add any metric from session profiler

2. **Playhead Control**:
   - Drag playhead to scrub through time
   - Play/pause buttons (auto-advance at 1x, 2x, 5x, 10x speed)
   - Jump to next/previous surprise spike
   - Jump to next/previous speech event

3. **Event Markers** (QGraphicsItem pins on timeline):
   - 💬 Speech events (green markers)
   - 🧠 Thought events (blue markers)
   - ⚡ Surprise spikes (yellow markers, triggered > 0.5)
   - 🎭 Play start/end (purple markers)
   - 🤔 Self-monitoring events (orange markers)
   - Click marker to jump to that time
   - Hover shows tooltip with event details

4. **Time Range Selection**:
   - Drag to select range
   - Right-click → Export segment as JSON
   - Right-click → Ask @Kimmie about this segment

5. **Zoom Controls**:
   - Scroll wheel: Zoom in/out on timeline
   - Fit to window
   - Zoom to selection

6. **Annotations**:
   - User can add text annotations at any time
   - Saved with session data
   - Example: "Notable: Callie laughed at Toad's joke here"

7. **Synchronized Views**:
   - Moving playhead updates:
     - Phenomenal State View (shows state at that time)
     - Conversation Context (shows messages at that time)
     - Operations Console (shows operations at that time)
   - All views scrub together

**Data Source**: `SessionProfiler` JSON files from `profiler_sessions/`

**Implementation Strategy**:
```python
class TimelinePanel(QDockWidget):
    playhead_moved = pyqtSignal(float)  # Emit current time

    def __init__(self):
        super().__init__("Timeline Profiler")
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        # Load session data
        self.session_data = {}  # Load from profiler JSON

        # Create timeline tracks
        self.surprise_track = TimelineTrack("Surprise", color="yellow")
        self.valence_track = TimelineTrack("Valence", color="green")
        # ... etc

        # Playhead
        self.playhead = PlayheadItem(x=0)
        self.scene.addItem(self.playhead)

    def scrub_to(self, time: float):
        """Move playhead and emit signal."""
        self.playhead.setX(time * self.pixels_per_second)
        self.playhead_moved.emit(time)
```

##### 2.4 Analytics Dashboard

**Purpose**: Statistical analysis of agent behavior over time

**Tabs**:

**Tab 1: Metrics Overview**
```
┌─────────────────────────────────────────┐
│  Time Range: [Last Hour ▼]             │
├─────────────────────────────────────────┤
│  ┌─────────────┬─────────────────────┐  │
│  │  HSI        │  0.0234             │  │
│  │  (Slow/Fast)│  ████░░░░░░  Poor   │  │
│  └─────────────┴─────────────────────┘  │
│  ┌─────────────┬─────────────────────┐  │
│  │  Avg Surprise│  0.185             │  │
│  │             │  ████████░░  Normal │  │
│  └─────────────┴─────────────────────┘  │
│  ┌─────────────┬─────────────────────┐  │
│  │  Speech Rate│  3.2 msg/min       │  │
│  │             │  ████████░░  Active │  │
│  └─────────────┴─────────────────────┘  │
│  ┌─────────────┬─────────────────────┐  │
│  │  Layer Vel  │  Fast:  0.042      │  │
│  │  (L2 norm)  │  Med:   0.018      │  │
│  │             │  Slow:  0.003  ✓   │  │
│  └─────────────┴─────────────────────┘  │
└─────────────────────────────────────────┘
```

**Tab 2: Affect Distribution**
- Histograms of valence, arousal, fear, sorrow, boredom
- Mean, median, std dev displayed
- Compare across agents (overlay histograms)

**Tab 3: Surprise Analysis**
- Histogram of surprise values
- Surprise spike frequency (spikes/hour)
- Correlation: Surprise vs. speech probability
- Table: Top 10 surprise spikes with context

**Tab 4: Performance Metrics**
- LLM latency: Histogram (ms)
- MLX forward pass latency: Histogram (ms)
- Total response time: P50, P95, P99
- Operations breakdown: Pie chart (% time in each operation)

**Tab 5: Consciousness Metrics**
- Integrated information Φ (if computed)
- Temporal Prediction Horizon (TPH)
- Surprise-Novelty Correlation (SNC)
- Personality Consistency Score (PCS)

**Implementation**: QTabWidget with QCharts for visualizations

##### 2.5 Operations Console

**Purpose**: Real-time operation log (like Chrome DevTools Console)

**Layout**:
```
┌──────────────────────────────────────────────────────────┐
│  [Filter: All ▼] [●●● Levels] [🔍 Search]               │
├──────────────────────────────────────────────────────────┤
│  [18:34:22.123] agent_callie | llm_generate_response |  │
│                 duration: 243ms | model: qwen3-4b      │
│  [18:34:22.367] agent_callie | mlx_forward_pass |      │
│                 duration: 8ms                          │
│  [18:34:22.375] agent_callie | surprise_computed |     │
│                 surprise: 0.185 (below threshold)      │
│  [18:34:22.380] agent_phi | received_stimulus |        │
│                 text: "Callie: How are you, Phi?"      │
│  [18:34:22.385] agent_phi | intuition_receiver |       │
│                 duration: 52ms | routing: "for_me"     │
│  [18:34:22.437] agent_phi | llm_generate_response |    │
│                 duration: 189ms | model: qwen3-4b      │
│  [18:34:22.626] agent_phi | speech_emitted |           │
│                 surprise: 0.421 (above threshold!)     │
│                 *meows happily, as if to say "I'm...   │
└──────────────────────────────────────────────────────────┘
```

**Features**:
- Color-coded by operation type
- Collapsible operation details (expand/collapse with ▶ icon)
- Filterable by agent, operation type, status
- Searchable (Ctrl+F)
- Auto-scroll when new operations arrive (toggle)
- Export to JSON/CSV
- Click operation → jump to that time in Timeline

**Data Source**: `PerformanceTracker` API

##### 2.6 Agent Manager

**Purpose**: Spawn, inspect, control agents

**Layout**:
```
┌──────────────────────────────────────────┐
│  Active Agents                           │
├──────────────────────────────────────────┤
│  ┌────────────────────────────────────┐  │
│  │  Callie (agent_callie)             │  │
│  │  🧠 Conscious | ⭐ Enlightened      │  │
│  │  Room: room_000                    │  │
│  │  [Inspect] [Kill] [Restart]        │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │  Phi (agent_phi)                   │  │
│  │  🧠 Conscious | 🎭 In Character     │  │
│  │  Room: room_000                    │  │
│  │  [Inspect] [Kill] [Restart]        │  │
│  └────────────────────────────────────┘  │
├──────────────────────────────────────────┤
│  [+ Spawn New Agent]                     │
│  Recipe: [callie ▼] [Spawn]             │
└──────────────────────────────────────────┘
```

**Features**:
- List all active agents
- Show status: Running, Paused, Error
- Inspect button → Opens detailed view:
  - Full phenomenal state
  - Conversation context
  - Memory contents
  - Recent operations
- Kill button → Gracefully stops agent
- Restart button → Reloads recipe and restarts
- Spawn dialog → Select recipe → Spawn in target room

### 3. Menu Bar Structure

**File**:
- New Recipe... (Ctrl+N)
- Open Recipe... (Ctrl+O)
- Save Recipe (Ctrl+S)
- Save Recipe As... (Ctrl+Shift+S)
- ---
- Open Session... (Load profiler JSON)
- Export Session Segment...
- ---
- Preferences
- ---
- Quit (Ctrl+Q)

**View**:
- Show/Hide panels (checkboxes):
  - ✓ Chat View
  - ✓ Log View
  - ✓ Recipe Editor
  - ✓ Phenomenal State View
  - ✓ Timeline Profiler
  - ✓ Analytics Dashboard
  - ✓ Operations Console
  - ✓ Agent Manager
- ---
- Layout Presets:
  - Development (Recipe Editor + Chat + Logs)
  - Analysis (Timeline + Analytics + Phenomenal State)
  - Performance (Operations + Analytics + Timeline)
  - Theater (Chat + Logs + Agent Manager)
- Save Current Layout...
- Load Layout...
- ---
- Reset to Default Layout

**Agent**:
- Spawn Agent... (Ctrl+Shift+N)
- Inspect Agent...
- Kill All Agents
- ---
- @enlighten [agent] on/off (Toggle enlightenment)

**Session**:
- Start New Session
- Load Session...
- Export Current Session
- ---
- Ask @Kimmie about Selection...

**Tools**:
- Validate All Recipes
- Benchmark Performance
- Export Metrics CSV
- ---
- noodleMUSH Server:
  - Start Server
  - Stop Server
  - Restart Server
  - View Server Logs

**Help**:
- Documentation
- Keyboard Shortcuts
- About NoodleSTUDIO

### 4. Tool Bar

Quick access icons:
```
[📄 New Recipe] [📁 Open] [💾 Save] | [▶ Play] [⏸ Pause] [⏹ Stop] |
[🤖 Spawn Agent] [🔍 Inspect] | [📊 Analytics] [⚡ Operations] |
Session: [session_20251115_183422 ▼] | Agent: [agent_callie ▼]
```

### 5. Status Bar

```
● Connected to noodleMUSH (ws://localhost:8765) |
Agents: 3 active |
Session: session_20251115_183422 |
Playhead: 45.2s / 183.7s |
HSI: 0.0234 (poor) |
FPS: 60
```

### 6. Keyboard Shortcuts

**Global**:
- `Ctrl+N`: New recipe
- `Ctrl+O`: Open recipe
- `Ctrl+S`: Save recipe
- `Ctrl+Shift+N`: Spawn agent
- `Ctrl+F`: Search (context-aware: searches active panel)
- `Ctrl+/`: Show command palette
- `Tab`: Toggle between Chat view and Log view
- `F11`: Toggle fullscreen

**Timeline**:
- `Space`: Play/pause
- `←/→`: Step backward/forward (1s)
- `Shift+←/→`: Jump to previous/next surprise spike
- `Ctrl+←/→`: Jump to previous/next speech event
- `Home`: Reset to start
- `End`: Jump to end
- `[/]`: Decrease/increase playback speed
- `Ctrl+E`: Export selected time range

**Panels**:
- `Ctrl+1-8`: Jump to panel (1=Chat, 2=Logs, 3=Recipe, 4=Phenomenal, 5=Timeline, 6=Analytics, 7=Operations, 8=Agents)
- `Ctrl+W`: Close current panel
- `Ctrl+Shift+W`: Close all panels except Chat

## Technology Stack

### Option A: PyQt6/PySide6 (Recommended)

**Pros**:
- Same language as rest of noodlings (Python)
- Easy integration with existing code (agent_bridge, session_profiler, etc.)
- Rapid development
- Excellent documentation
- QWebEngineView for web panels

**Cons**:
- Slightly slower than native C++
- Larger bundle size

**Dependencies**:
```
PyQt6
PyQt6-WebEngine
PyQt6-Charts
numpy
pyyaml
websockets
aiohttp
```

### Option B: Qt C++

**Pros**:
- Native performance
- Smaller binary
- Better for shipping standalone app

**Cons**:
- More complex integration with Python backend
- Need to rewrite data loading logic in C++
- Slower development

**Decision**: Use PyQt6 for v1.0. Port to C++ later if performance becomes issue.

## Data Architecture

### Real-time Data Flow

```
noodleMUSH Server (Python)
    ├── WebSocket (ws://localhost:8765)
    │   └── Chat events, agent state updates
    │
    ├── HTTP API (http://localhost:8081)
    │   ├── /api/profiler/live-session (SessionProfiler data)
    │   ├── /api/performance/operations (PerformanceTracker data)
    │   ├── /api/agents/list
    │   ├── /api/agents/{agent_id}/state
    │   └── /api/agents/spawn (POST)
    │
    └── SessionProfiler (writes to profiler_sessions/*.json)

NoodleSTUDIO (PyQt6)
    ├── WebSocket Client → Chat/Log panels
    ├── HTTP Client → Polling for metrics (1Hz)
    └── File Watcher → Detects new session files
```

### Session Data Storage

**Location**: `applications/cmush/profiler_sessions/`

**Format**: JSON (one file per session)

**Example**: `session_20251115_183422.json`
```json
{
  "metadata": {
    "session_id": "session_20251115_183422",
    "start_time": "2025-11-15 18:34:22",
    "agents": ["agent_callie", "agent_phi", "agent_toad"]
  },
  "duration": 183.7,
  "timelines": {
    "agent_callie": [
      {
        "timestamp": 0.0,
        "phenomenal_state": {
          "fast": [0.1, 0.2, ...],  // 16-D
          "medium": [...],          // 16-D
          "slow": [...],            // 8-D
          "full": [...]             // 40-D
        },
        "affect": {
          "valence": 0.68,
          "arousal": 0.54,
          "fear": 0.12,
          "sorrow": 0.08,
          "boredom": 0.15
        },
        "surprise": 0.185,
        "did_speak": false,
        "utterance": null,
        "hsi": {
          "hsi_slow_fast": 0.0234,
          "hsi_medium_fast": 0.184,
          "status": "poor_separation"
        },
        "event": "received_message",
        "conversation_context": [...]
      },
      ...
    ]
  }
}
```

**Loading Strategy**:
1. On startup, scan `profiler_sessions/` for available sessions
2. Load most recent session by default
3. User can open older sessions via File → Open Session
4. In real-time mode, poll API every 1s for new data points
5. In scrubbing mode, load all data once and seek through it

## UI/UX Design Principles

### Color Scheme (Dark Theme)

```
Background:       #0a0e1a (deep blue-black)
Panel Background: #131824 (slightly lighter)
Borders:          #2a3f5f (muted blue)
Text:             #e0e0e0 (light gray)
Accent (Primary): #64b5f6 (bright blue)

Layer Colors:
  Fast:   #66bb6a (green)
  Medium: #ffa726 (orange)
  Slow:   #ba68c8 (purple)

Affect Colors:
  Valence+: #66bb6a (green)
  Valence-: #ef5350 (red)
  Arousal:  #ffa726 (orange)
  Fear:     #ef5350 (red)
  Surprise: #64b5f6 (blue)

Status Colors:
  Good:     #66bb6a (green)
  Warning:  #ffa726 (orange)
  Error:    #ef5350 (red)
```

### Typography

```
Body:      14px Roboto
Monospace: 13px 'Source Code Pro'
Headers:   16px Roboto Bold
Captions:  12px Roboto Light
```

### Spacing

```
Panel Padding:    16px
Widget Spacing:   8px
Section Spacing:  24px
Margin:          12px
```

### Animation

- Smooth panel transitions: 200ms ease-in-out
- Timeline playhead: 16.67ms (60fps)
- Value updates: Spring animation (QPropertyAnimation)
- Panel resize: No animation (instant)

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Main window with menu bar
- [ ] QDockWidget infrastructure
- [ ] Layout save/load system
- [ ] Chat panel (QWebEngineView)
- [ ] Log panel (QWebEngineView)
- [ ] Basic styling (dark theme)

### Phase 2: Recipe Editor (Week 2)
- [ ] Recipe loader integration
- [ ] All input widgets (sliders, text fields, etc.)
- [ ] Live validation
- [ ] Save/load functionality
- [ ] Radar chart for appetites

### Phase 3: Phenomenal State View (Week 2)
- [ ] 40-D state visualization (bar charts)
- [ ] Layer color coding
- [ ] 5-D affect radar chart
- [ ] Real-time updates (WebSocket)

### Phase 4: Timeline Profiler (Week 3)
- [ ] QGraphicsView timeline rendering
- [ ] Multi-track support (Surprise, Valence, Arousal, etc.)
- [ ] Playhead control (drag + keyboard)
- [ ] Event markers (speech, thought, spikes)
- [ ] Zoom and pan
- [ ] Time range selection
- [ ] Synchronized view updates

### Phase 5: Operations Console (Week 3)
- [ ] Operation log display (QTableView)
- [ ] Filtering and search
- [ ] Real-time updates from PerformanceTracker
- [ ] Operation details expansion
- [ ] Export to JSON/CSV

### Phase 6: Analytics Dashboard (Week 4)
- [ ] Metrics overview tab
- [ ] Affect distribution histograms
- [ ] Surprise analysis
- [ ] Performance metrics
- [ ] Consciousness metrics (if available)

### Phase 7: Agent Manager (Week 4)
- [ ] List active agents
- [ ] Spawn dialog
- [ ] Inspect dialog (detailed state view)
- [ ] Kill/restart controls
- [ ] Integration with WebSocket commands

### Phase 8: Polish & Testing (Week 5)
- [ ] Keyboard shortcuts
- [ ] Command palette (Ctrl+/)
- [ ] Preferences dialog
- [ ] Error handling and validation
- [ ] Performance optimization
- [ ] Documentation
- [ ] Testing with real sessions

## Open Questions

1. **@Kimmie Integration**: How should we integrate Claude for timeline interpretation?
   - Option A: External Claude Desktop app (via MCP)
   - Option B: Embedded Claude API client
   - Option C: Export segment → Paste into Claude chat

2. **Real-time vs. Playback**: Should we support both modes simultaneously?
   - Proposal: Two modes toggled via button
   - Real-time mode: Live tail of data
   - Playback mode: Load entire session, scrub through it

3. **Multi-agent Timeline**: Should we show multiple agents on same timeline?
   - Proposal: Separate timeline tracks per agent (stacked)
   - Color-code each agent
   - Option to overlay or separate

4. **Performance**: Can PyQt6 handle 60fps timeline rendering?
   - Need to benchmark with realistic data
   - May need to downsample display (show every Nth point)
   - Use QGraphicsView optimizations (view culling, LOD)

5. **Distribution**: How do we package/distribute?
   - Option A: pyinstaller (standalone executable)
   - Option B: pip install noodlestudio (Python package)
   - Option C: Both

## Success Metrics

NoodleSTUDIO v1.0 is successful if:

1. **Usability**: Non-technical user can spawn agent, view its state, scrub timeline
2. **Performance**: 60fps timeline rendering with 1000+ data points
3. **Flexibility**: User can create custom layouts, save/load them
4. **Integration**: Seamless connection to noodleMUSH server
5. **Clarity**: All views clearly show what data they're displaying (agent, time, session)
6. **Extensibility**: Easy to add new panels/tracks/metrics

## Future Enhancements (Post v1.0)

- Multi-session comparison (diff two sessions side-by-side)
- Experiment tracking (tie sessions to specific ablations/configs)
- Integrated debugger (set breakpoints on surprise spikes)
- Record/replay system (deterministic playback)
- Plugin system (custom panels, custom metrics)
- Cloud sync (share sessions with collaborators)
- Mobile companion app (iOS/Android monitoring dashboard)
