# Architecture

System design at a glance.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NoodleStudio                              │
│                    (PyQt6 Desktop IDE)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Stage   │ │  Assets  │ │ Inspector│ │ Facets   │           │
│  │  View    │ │  Panel   │ │  Panel   │ │ Editor   │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────────────┐         │
│  │  Chat    │ │ Gaussian │ │     Neural Canvas       │         │
│  │  Panel   │ │ Viewer   │ │   (Visual Programming)  │         │
│  └────┬─────┘ └──────────┘ └─────────────────────────┘         │
└───────┼─────────────────────────────────────────────────────────┘
        │ WebSocket
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        NoodleMUSH                                │
│                   (Python WebSocket Server)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Scene State Manager                    │   │
│  │              (Canonical World Truth)                      │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────┼───────────────────────────────┐   │
│  │         ┌────────────────┴────────────────┐              │   │
│  │         ▼                                 ▼              │   │
│  │  ┌─────────────┐                  ┌─────────────┐        │   │
│  │  │   Red's     │                  │   Yuki's    │        │   │
│  │  │ Perception  │                  │ Perception  │        │   │
│  │  │   Slice     │                  │   Slice     │        │   │
│  │  └──────┬──────┘                  └──────┬──────┘        │   │
│  │         ▼                                ▼              │   │
│  │  ┌─────────────┐                  ┌─────────────┐        │   │
│  │  │   Red's     │                  │   Yuki's    │        │   │
│  │  │   Facet     │                  │   Facet     │        │   │
│  │  │  Assembly   │                  │  Assembly   │        │   │
│  │  └─────────────┘                  └─────────────┘        │   │
│  │       Noodlings (AI Characters)                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │  LLM Interface │  │  World State   │  │  Auth System   │     │
│  │  (8 providers) │  │  (Rooms/Props) │  │  (Local/Cloud) │     │
│  └────────────────┘  └────────────────┘  └────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Scene State Manager
Single source of truth for world state. All renders (text, 2D, 3D) are
projections of this canonical data.

### Perception Slices
Each Noodling gets a filtered view of the world - only what they can perceive.
No omniscient AI that knows everything.

### Facet Assemblies
Visual node graphs defining how a Noodling thinks. Data flows from INCOMING
(perception) through cognitive facets to OUTGOING (actions).

### Continuous Affect
5-dimensional emotional state (valence, arousal, dominance, boredom, sorrow)
that evolves continuously, not just in response to prompts.

---

## Data Flow

```
1. Event occurs (player speaks, time passes, prop moves)
           │
           ▼
2. Scene State Manager updates canonical state
           │
           ▼
3. Perception slices generated for each Noodling
           │
           ▼
4. Facet assemblies process perception
           │
           ├──► LLM calls (thinking, speaking)
           ├──► Neural networks (affect prediction)
           └──► Scripted logic (custom behavior)
           │
           ▼
5. Actions emitted (speech, movement, expressions)
           │
           ▼
6. Scene State Manager updates, loop continues
```

---

## Key Components

### NoodleMUSH (Server)

| Component | File | Purpose |
|-----------|------|---------|
| Server | `server.py` | WebSocket server, HTTP |
| Commands | `commands.py` | @rez, @observe, say, etc. |
| Agent Bridge | `agent_bridge.py` | Noodling lifecycle |
| World | `world.py` | Rooms, objects, state |
| LLM Interface | `llm_interface.py` | Multi-provider LLM |

### NoodleStudio (IDE)

| Component | File | Purpose |
|-----------|------|---------|
| Main Window | `main_window.py` | Application shell |
| Stage View | `scene_hierarchy.py` | Scene tree |
| Inspector | `inspector_panel.py` | Property editing |
| Facets Editor | `facets_editor_panel.py` | Cognitive graphs |
| Neural Canvas | `neural_canvas_view.py` | NN design |
| Gaussian Viewer | `gaussian_viewer_panel.py` | 3D preview |

### Core Systems

| System | Files | Purpose |
|--------|-------|---------|
| Facet Executor | `facet_executor.py` | Run assemblies |
| Gaussian Renderer | `gaussian_renderer.py` | GPU rendering |
| Radiance Format | `radiance_format.py` | .radiance I/O |
| Scene Protocol | `scene_state_manager.py` | World state |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Desktop UI | PyQt6 |
| Server | Python asyncio, WebSockets |
| Neural Networks | MLX (Apple Silicon), PyTorch |
| GPU Rendering | gsplat-mps (Metal) |
| LLM Providers | Ollama, OpenAI, Anthropic, + 5 more |
| Data Formats | YAML, Binary chunks (.radiance) |
| Web Client | Vanilla HTML/JS |

---

## Ports

| Port | Service |
|------|---------|
| 8080 | HTTP (web client) |
| 8765 | WebSocket (real-time) |
| 11434 | Ollama (LLM) |

---

## Project Structure

```
noodlings_clean/
├── applications/
│   ├── cmush/              # NoodleMUSH server (~50K lines)
│   └── noodlestudio/       # NoodleStudio IDE (~80K lines)
├── docs/                   # You are here
├── papers/                 # Academic papers
└── external/               # Third-party tools
```
