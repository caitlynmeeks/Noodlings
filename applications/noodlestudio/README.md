# NoodleSTUDIO

**Version**: 1.0.0-alpha
**Status**: In Development
**Framework**: PyQt6
**Platform**: macOS (primary), Linux/Windows (future)

## Overview

NoodleSTUDIO is a comprehensive IDE for developing, monitoring, and analyzing Noodling consciousness agents. It provides real-time visualization, recipe editing, performance profiling, and timeline analysis in a unified, flexible interface.

Think of it as:
- **Unity Timeline** + **Chrome DevTools** for consciousness agents
- **Visual Studio Code** for Noodlings development
- **Grafana** for phenomenal state monitoring

## Features

### 🎛️ Flexible Panel Layout
- Drag-and-drop panels anywhere
- Save/load custom layouts
- Preset layouts: Development, Analysis, Performance, Theater
- Float panels on multiple monitors

### 📝 Recipe Editor
- Visual editor for agent YAML recipes
- Live validation with helpful error messages
- Sliders for personality (8-D) and appetites (8-D)
- Radar chart visualization
- Apply changes to running agents

### 🧠 Phenomenal State Visualization
- Real-time 40-D state display
- Color-coded layers (Fast/Medium/Slow)
- Bar charts for each dimension
- 5-D affect radar chart (Valence, Arousal, Fear, Sorrow, Boredom)

### 📊 Timeline Profiler (Unity-style)
- Scrub through time with playhead
- Multiple tracks: Surprise, Valence, Arousal, Fear, HSI
- Event markers: Speech 💬, Thought 🧠, Spikes ⚡
- Zoom, pan, play/pause
- Export time ranges for analysis

### 📈 Analytics Dashboard
- HSI (Hierarchical Separation Index) metrics
- Affect distributions
- Surprise spike analysis
- Performance metrics (LLM latency, operation times)
- Consciousness metrics (Φ, TPH, SNC, PCS)

### ⚡ Operations Console
- Real-time operation log
- Filter by agent, type, status
- Color-coded by duration (green=fast, red=slow)
- Click operation → jump to timeline
- Export to JSON/CSV

### 🤖 Agent Manager
- List active agents
- Spawn new agents from recipes
- Inspect agent state (memory, context, operations)
- Kill/restart agents
- Toggle enlightenment mode

### 💬 Integrated Chat & Logs
- Embedded noodleMUSH web interface
- Real-time server logs with filtering
- Tab/View toggle for quick switching

## Installation

### Requirements
- Python 3.10+
- Qt 6
- noodleMUSH server running (for live data)

### Setup
```bash
cd applications/noodleSTUDIO
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
python run_studio.py
```

Or from the project root:
```bash
cd applications/noodleSTUDIO
python -m noodlestudio.main
```

## Quick Start

1. **Start noodleMUSH server** (in another terminal):
   ```bash
   cd applications/cmush
   ./start.sh
   ```

2. **Launch NoodleSTUDIO**:
   ```bash
   cd applications/noodleSTUDIO
   python run_studio.py
   ```

3. **Open Chat panel** → Spawn an agent:
   ```
   @spawn callie
   ```

4. **Watch the magic**:
   - Phenomenal State View updates in real-time
   - Timeline shows surprise spikes
   - Operations Console logs each LLM call
   - Analytics Dashboard computes HSI

5. **Scrub through time**:
   - Drag the timeline playhead
   - See how state evolved over time
   - Click event markers to jump to moments

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

### Component Overview
```
NoodleSTUDIO (PyQt6)
├── Chat Panel (QWebEngineView) ─┐
├── Log Panel (QWebEngineView)   │ Web Panels
├── Recipe Editor                ─┤
├── Phenomenal State View        │
├── Timeline Profiler            │ Native Qt Panels
├── Analytics Dashboard          │
├── Operations Console           │
└── Agent Manager                ─┘

Data Sources:
├── noodleMUSH WebSocket (ws://localhost:8765)
├── noodleMUSH HTTP API (http://localhost:8081)
└── SessionProfiler JSON files (profiler_sessions/*.json)
```

## Development Status

Current status: **Phase 1 - Foundation** (Week 1)

### Roadmap
- [x] Architecture design
- [x] Documentation structure
- [ ] Main window + menu bar (Week 1)
- [ ] Web panels (Chat, Logs) (Week 1)
- [ ] Recipe editor (Week 2)
- [ ] Phenomenal state view (Week 2)
- [ ] Timeline profiler (Week 3)
- [ ] Operations console (Week 3)
- [ ] Analytics dashboard (Week 4)
- [ ] Agent manager (Week 4)
- [ ] Polish & testing (Week 5)

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for detailed timeline.

## Keyboard Shortcuts

### Global
- `Ctrl+N`: New recipe
- `Ctrl+O`: Open recipe
- `Ctrl+S`: Save recipe
- `Ctrl+Shift+N`: Spawn agent
- `Ctrl+F`: Search (context-aware)
- `Ctrl+/`: Command palette
- `Tab`: Toggle Chat ↔ Log view
- `F11`: Fullscreen

### Timeline
- `Space`: Play/pause
- `←/→`: Step backward/forward (1s)
- `Shift+←/→`: Jump to prev/next surprise spike
- `Ctrl+←/→`: Jump to prev/next speech event
- `Home/End`: Jump to start/end
- `[/]`: Decrease/increase playback speed

### Panels
- `Ctrl+1-8`: Jump to panel (1=Chat, 2=Logs, 3=Recipe, etc.)
- `Ctrl+W`: Close current panel
- `Ctrl+Shift+W`: Close all except Chat

## Configuration

Configuration file: `~/.noodlestudio/config.yaml`

```yaml
# NoodleSTUDIO Configuration

api:
  websocket_url: ws://localhost:8765
  http_url: http://localhost:8081
  refresh_interval_ms: 1000

ui:
  theme: dark  # dark, light
  default_layout: development  # development, analysis, performance, theater
  font_size: 14
  enable_animations: true

timeline:
  fps: 60
  default_playback_speed: 1.0
  auto_jump_to_spikes: false

performance:
  max_timeline_points: 10000  # Downsample if exceeded
  operation_buffer_size: 1000
  enable_profiling: false  # Profile NoodleSTUDIO itself

paths:
  sessions_dir: ../cmush/profiler_sessions
  recipes_dir: ../cmush/recipes
  layouts_dir: ~/.noodlestudio/layouts
```

## Documentation

- [Architecture Specification](docs/ARCHITECTURE.md) - System design, component breakdown
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Week-by-week development schedule
- [User Guide](docs/USER_GUIDE.md) *(Coming soon)* - How to use NoodleSTUDIO
- [API Reference](docs/API_REFERENCE.md) *(Coming soon)* - Python API docs

## Contributing

NoodleSTUDIO is part of the Noodlings project. See [../../CLAUDE.md](../../CLAUDE.md) for project context.

### Code Style
- PEP 8 for Python
- Type hints required
- Docstrings for all public APIs
- Use Qt signals/slots for component communication

### Testing
```bash
pytest tests/
```

## Troubleshooting

### "Cannot connect to noodleMUSH"
- Make sure noodleMUSH server is running: `cd ../cmush && ./start.sh`
- Check WebSocket URL in config.yaml

### "Timeline rendering is slow"
- Reduce `max_timeline_points` in config.yaml
- Disable animations: `enable_animations: false`
- Close unused panels

### "Recipe validation fails"
- Check YAML syntax
- Ensure all values are in valid ranges (0-1 for sliders)
- Check error messages in Recipe Editor status area

## License

Part of the Noodlings project. See [../../LICENSE](../../LICENSE).

## Credits

**Design & Architecture**: Caitlyn Griffith + Claude (Anthropic)
**Framework**: PyQt6 (The Qt Company)
**Inspiration**: Unity Timeline Editor, Chrome DevTools, VSCode

Special thanks to the noodleMUSH agents who served as test subjects. 🧠✨

---

**Status**: Pre-alpha. API will change. Expect bugs. 🐛

**Next Milestone**: Week 1 - Foundation (Main window + web panels)

**Questions?** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) or open an issue.
