# Noodle Code

**AI-powered coding assistant integrated into NoodleStudio**

*Last updated: January 2, 2026*

---

## Overview

Noodle Code is an embedded Claude-style AI assistant that can:
- **See and control** NoodleStudio's UI (Computer Use)
- **Read, write, and edit** project files
- **Search** the codebase with glob and grep
- **Execute** shell commands and GitHub CLI
- **Hot reload** modified Python modules without restart
- **Soft restart** NoodleStudio preserving state
- **Access** project-specific knowledge via NOODLE_CODE.md

---

## Quick Start

1. Open the **Noodle Code** panel (View > Noodle Code)
2. Type a request and press Enter or click Send
3. Use **A-/A+** buttons to adjust font size
4. Watch tool executions appear inline

**Example prompts:**
- "Take a screenshot and describe what you see"
- "Search for all files containing 'facet'"
- "Create a new Python file for utility functions"
- "List open GitHub issues"

---

## Architecture

```
NoodleCodePanel (UI)
       │
       ▼
NoodleCodeEngine (orchestration)
       │
       ├──▶ NoodleCodeTools (file ops, search, bash)
       │
       ├──▶ ComputerUseController (screenshots, input injection)
       │
       ├──▶ GitHub CLI (issues, PRs, repo ops)
       │
       └──▶ NOODLE_CODE.md (project knowledge)
```

---

## Core Components

### NoodleCodePanel (`panels/noodle_code_panel.py`)

Chat interface with:
- **Message bubbles** - User (green), Assistant (blue), Error (orange)
- **Tool indicators** - Shows tool name, brief args, status
- **Font controls** - A-/A+ buttons with size persistence
- **Streaming** - Real-time response display

### NoodleCodeEngine (`core/noodle_code_engine.py`)

Backend orchestration:
- Manages conversation history
- Builds context-aware system prompt
- Loads NOODLE_CODE.md knowledge base
- Routes to configured model (Noodle Code label or Large fallback)
- Handles streaming with tool use loop

### NoodleCodeTools (`core/noodle_code_tools.py`)

All tool implementations with security sandboxing.

### ComputerUseController (`core/computer_use_controller.py`)

UI interaction via:
- `QWidget.grab()` for screenshots
- `QTest.mouseClick/keyClick` for input injection

---

## Available Tools

### File Operations

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `read_file` | Read file contents with line numbers | `path`, `offset?`, `limit?` |
| `write_file` | Create or overwrite file | `path`, `content` |
| `edit_file` | Replace exact string in file | `path`, `old_string`, `new_string` |
| `glob` | Find files by pattern | `pattern`, `path?` |
| `grep` | Search file contents | `pattern`, `path?`, `glob_pattern?`, `context_lines?` |
| `list_directory` | List folder contents | `path?`, `recursive?`, `max_depth?` |

### System

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `bash` | Run shell command | `command`, `timeout?` |
| `hot_reload` | Reload Python module without restart | `module_name` or `file_path` |
| `soft_restart` | Restart NoodleStudio preserving state | `reason?`, `confirm` |

### GitHub CLI

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `github` | GitHub CLI operations | `command` (e.g., "issue list", "pr view 42") |

**Examples:**
```
github(command="issue list")
github(command="issue view 42")
github(command="issue create --title 'Bug: X' --label bug")
github(command="pr list")
github(command="pr create --title 'Feature'")
github(command="search issues 'crash'")
```

### Computer Use (UI Control)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `computer_use` | See and interact with UI | `action`, `coordinate?`, `text?`, etc. |

**Actions:**
- `screenshot` - Capture window + **UI Element Map** with exact coordinates
- `ui_elements` - Get just the element map (no screenshot)
- `left_click` - Click at [x, y]
- `right_click` - Right-click at [x, y]
- `double_click` - Double-click at [x, y]
- `type` - Type text into focused widget
- `key` - Press key combo (e.g., "ctrl+s", "enter")
- `scroll` - Scroll at position
- `drag` - Drag from start to end

**UI Element Map (key feature):**

Screenshots include a structured list of all clickable UI elements with EXACT coordinates from Qt's widget tree. No vision-based coordinate guessing needed.

```
CLICKABLE UI ELEMENTS (name -> click at x,y):

TABS:
  Tab: Stage -> (22, 11)
  Tab: Assets -> (78, 11)
  Tab: Noodle Code -> (257, 11)
  Tab: Facets Editor -> (502, 11)

BUTTONS:
  Button: Send -> (1305, 582)
  Button: D -> (320, 582)

INPUT FIELDS:
  Input: Ask Noodle Code... -> (880, 582)
```

**Workflow:**
```
1. computer_use(action="screenshot")       # Get image + element map
2. Find element in the map: "Tab: Facets Editor -> (502, 11)"
3. computer_use(action="left_click", coordinate=[502, 11])
4. computer_use(action="screenshot")       # Verify result
```

**Why UI Element Map?**

Vision models struggle to read coordinates from images accurately. They often hallucinate UI text and estimate positions incorrectly. By querying Qt's actual widget tree, we get pixel-perfect coordinates every time.

---

## Model Routing

Noodle Code uses the model label system for routing:

1. **"Noodle Code" label** - Dedicated AI assistant model (Settings > Models)
2. **"Large" label fallback** - If Noodle Code not configured
3. **Default** - `claude-sonnet-4-20250514` if nothing configured

### Configuring the Model

1. Open **Settings > Models**
2. Find **Label Assignments** section
3. Assign a capable model to **"Noodle Code"** label
4. Recommended: Claude Opus 4.5 or Claude Sonnet 4

---

## NOODLE_CODE.md Knowledge Base

Like CLAUDE.md for Claude Code, NOODLE_CODE.md provides project context.

### How It Works

1. On startup, Noodle Code looks for `NOODLE_CODE.md`:
   - First in project root
   - Falls back to NoodleStudio default
2. Content is injected into the system prompt
3. Truncated to 8K chars to preserve context budget

### Creating Project-Specific Context

Copy the template to your project:
```bash
cp applications/noodlestudio/NOODLE_CODE.md ~/MyProject/
```

Then customize the "Project-Specific Context" section:
```markdown
## Project-Specific Context

### Current Focus
Working on multiplayer networking

### Key Files
- networking/connection_manager.py
- stages/multiplayer_stage/

### Conventions
- All network messages use msgpack
- Tests required for all new code
```

### Template Contents

The default NOODLE_CODE.md includes:
- NoodleStudio architecture overview
- Scripting API reference
- Common operations guide
- Computer Use workflow
- GitHub CLI examples
- Tips for effective assistance

---

## System Labels

Two protected model labels for routing:

### Noodle Code Label

For the AI assistant panel. Assign a smart model with:
- Strong coding ability
- Tool use support
- Good context handling

### Computer Use Label

For UI automation. Assign a model with:
- Vision capability
- `computer_use` tool support
- Currently: `claude-3-5-sonnet-20241022` or `claude-sonnet-4-20250514`

---

## Hot Reload

Modify Python code and apply changes without restart:

```
hot_reload(module_name="noodlestudio.core.utility_facets")
```

**Safe to reload:**
- Facet types (`*_facet.py`)
- Tool implementations
- Scripting APIs
- Utility modules

**Requires soft_restart:**
- Panel classes
- Mixins
- Core singletons
- Main window components

---

## Soft Restart

When hot reload isn't enough:

```
soft_restart(reason="Modified inspector panel", confirm=true)
```

Preserves:
- Current project
- Open tabs
- Window layout
- Panel states

---

## Security

### File Operations
- All paths resolved relative to project
- `../` escape attempts blocked
- Write warnings for critical files

### Bash Commands
- Timeout enforced (default 30s)
- Output truncated if too long

### GitHub CLI
- Shell operators blocked (`; | & $ ` etc.)
- Requires `gh auth login` first

### Computer Use
- Only affects NoodleStudio window
- Coordinates relative to window (not screen)

---

## UI Features

### Chat Display

Messages use a minimal Claude Code-style format:
- **User messages**: `⭄` prefix with selection-style background
- **Assistant messages**: `꩜` (Cham spiral) prefix
- Single text area for easy copy/paste across messages
- Chat history persists between sessions (~/.noodlestudio/noodlecode_history.json)

### Control Buttons

| Button | Function |
|--------|----------|
| **D** | Demo mode - show Ghost Cursor during Computer Use |
| **C** | Copy chat history to clipboard |
| **A-/A+** | Decrease/increase font size |
| **Send/Stop** | Send message or stop generation |

### Ghost Cursor (Demo Mode)

When **D** (demo mode) is enabled, Computer Use actions display a theatrical ghost cursor overlay:

**Visual States:**
- **Idle**: Small, ephemeral white breathing glow with occasional sparkle poofs
- **Insight Flash**: Brief bright cyan-white burst before movement ("watch me!")
- **Moving**: Pink-orange glow following beautiful bezier curves
- **Click**: Expanding ripple effect at click location

**Implementation:**
- Bezier curve movement with ease-in-out-cubic timing
- Motion trail with fading afterimages
- Click ripples that expand and fade
- All coordinates match actual click positions

**Key Files:**
- `core/ghost_cursor.py` - GhostCursorOverlay, GhostCursorController
- Stop button hides ghost cursor when generation is interrupted

### Keyboard Shortcuts

- **Enter** - Send message
- **Escape** - Stop generation (if running)
- **Up/Down** - Navigate input history

---

## Testing

Run Noodle Code tests:

```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest tests/test_noodle_code.py -v
```

Test coverage:
- Tool execution
- Message formatting
- Engine orchestration
- Computer Use controller
- GitHub CLI integration

---

## Troubleshooting

### "Engine not initialized"

Project hasn't loaded yet. Wait for project to open.

### "API error 404: model not found"

Invalid model ID in settings. Check:
1. Settings > Models
2. Noodle Code label assignment
3. Use real API model IDs (e.g., `claude-sonnet-4-20250514`)

### "gh: not authenticated"

Run in terminal:
```bash
gh auth login
```

### Computer Use not working

1. Check ComputerUseController initialized (main_window.py)
2. Verify window is visible and focused
3. Coordinates are window-relative, not screen

### Hot reload fails

Module must already be imported. For new files:
1. Restart NoodleStudio once
2. Then hot reload will work

---

## File Structure

```
noodlestudio/
├── panels/
│   └── noodle_code_panel.py          # Chat UI
├── core/
│   ├── noodle_code_engine.py         # LLM orchestration
│   ├── noodle_code_tools.py          # Tool implementations
│   ├── computer_use_controller.py    # UI interaction + UI Element Map
│   ├── ghost_cursor.py               # Demo mode cursor overlay
│   ├── noodle_code_profiles.py       # Personality profiles
│   ├── noodle_code_profiles/         # Profile markdown files
│   ├── hot_reload.py                 # Module reloading
│   └── soft_restart.py               # State-preserving restart
└── NOODLE_CODE.md                    # Default knowledge base
```

---

## See Also

- [Facet System](facets.md) - Cognitive node pipelines
- [Scripting API](scripting.md) - context.noodle.* reference
- [Testing](testing.md) - Regression test guide
- [Model Labels](../noodlemush/model-routing.md) - Label system

---

## Changelog

### January 3, 2026

- **UI Element Map** - Screenshots now include exact coordinates from Qt widget tree
- **Ghost Cursor** - Theatrical cursor overlay with bezier curves, insight flash, sparkles
- **Chat Redesign** - Claude Code-style single text area with `⭄`/`꩜` prefixes
- **Chat Persistence** - History saves between sessions
- **Copy History** - C button copies full conversation
- **Image Fix** - Screenshots now actually sent to vision model (was broken!)
- **Profiles** - Personality profiles (default, creative, architect, reviewer, mlx)

### January 2, 2026

- **Computer Use** - Screenshot, click, type, scroll, drag
- **GitHub CLI** - Issues, PRs, repo operations
- **NOODLE_CODE.md** - Project knowledge base (like CLAUDE.md)
- **System Labels** - "Noodle Code" and "Computer Use" protected labels
- **Font Controls** - A-/A+ with persistence
- **Model Capabilities** - Badges in Settings > Models (vision, tools, computer_use, thinking, pdf)
