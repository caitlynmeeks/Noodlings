# Shell Mode: Editor/Runtime Duality

**Status**: Core Architecture
**Last Updated**: January 2026
**Authors**: Caitlyn + Claude

---

## The Core Principle

> "There is no 'runtime' separate from the 'editor.' There is only NoodleStudio, with different UI shells and permission levels."

This is foundational. Unlike traditional game engines (Unity, Unreal) that have separate editor and player binaries, NoodleStudio is **one application** that presents different faces based on context.

When someone "plays" a NoodleStudio project, they're running NoodleStudio. They just don't see the editor panels.

---

## Why This Matters

### The Permeable Boundary

Traditional software drew a hard line: creators use the editor, consumers use the player. You're either a developer or a user. This created barriers to learning.

Some systems got it right:

- **HyperCard** - Every stack was editable. The authoring tool was the player. A generation learned to program because the tools were just... there.

- **The early Web** - View Source. The whole truth of the page was accessible. People learned HTML by looking at pages they liked.

- **Emacs** - A text editor that's also a Lisp machine. You can inspect and modify any part of it while using it.

All of these created communities of tinkerers, learners, creators. Because the boundary was permeable.

NoodleStudio follows this tradition. The "player" isn't a separate thing - it's the editor with panels hidden. And you can always peek behind the curtain.

### Connection to Christopher Alexander

In architecture, "defensive spaces" cut off possibility. Gated communities, buildings that say "this is not for you." Alexander argued for permeable, inviting spaces.

Proprietary "player" binaries are defensive software architecture. They say: "consume this, don't understand it."

NoodleStudio asks: what if we just... didn't do that?

---

## Publisher Permission Levels

The creator of a NoodleStudio project chooses how permeable to make the boundary:

| Level | Description | Use Case |
|-------|-------------|----------|
| **Locked** | Pure experience, no peeking | Commercial games, controlled experiences |
| **View Source** | Can see how things are built, cannot modify | Educational, "learn from this" |
| **Sandbox** | Can modify, changes don't persist / can reset | Safe experimentation |
| **Full Access** | Complete editor access | Open creation, community building |

Set in `project.yaml`:

```yaml
name: "Let's Consciousness!"
version: "0.1.0"
permission: view_source
ui: ui.yaml
```

---

## Shell Mode Architecture

### What Is Shell Mode?

When NoodleStudio opens a project with a `ui.yaml` and permission < `full_access`, it enters **shell mode**:

1. Editor panels (Inspector, NNCanvas, Assets, etc.) are hidden
2. The project's `ui.yaml` renders in the central widget
3. A "View Source" affordance appears (if permission allows)
4. User interacts with the project through its custom UI

### Implementation

```python
# In main_window.py (conceptual)

def open_project(self, project_path):
    project = Project.load(project_path)

    if project.ui_yaml and project.permission != Permission.FULL_ACCESS:
        self._enter_shell_mode(project)
    else:
        self._enter_editor_mode(project)

def _enter_shell_mode(self, project):
    """Hide editor, show project's UI shell."""
    # Hide all dock panels
    for dock in self.dock_widgets:
        dock.hide()

    # Hide menu bar (or show minimal version)
    self.menuBar().hide()

    # Render project's ui.yaml in central widget
    self.central_widget.load_ui(project.ui_yaml)

    # Show View Source button if permission allows
    if project.permission in [Permission.VIEW_SOURCE, Permission.SANDBOX]:
        self._show_view_source_button()

def _exit_shell_mode(self):
    """Reveal editor panels."""
    for dock in self.dock_widgets:
        dock.show()
    self.menuBar().show()
    self._hide_view_source_button()
```

### The View Source Affordance

For `view_source` and `sandbox` permissions, a subtle button allows users to reveal the editor:

```
┌─────────────────────────────────────────┐
│  [Let's Consciousness!]      [</>]      │  <- View Source button
│  ┌─────────────────┐  ┌──────────────┐  │
│  │  Guide Avatar   │  │   Speech     │  │
│  │                 │  │   Bubble     │  │
│  └─────────────────┘  └──────────────┘  │
│  ┌─────────────────────────────────────┐│
│  │  Ask Guide something...             ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

Clicking `[</>]` reveals the editor panels. The user sees Guide's facet assembly, recipe, the NNCanvas - everything that powers the experience.

For `view_source`: Panels are read-only. User can inspect but not modify.
For `sandbox`: User can modify, but changes reset on close.
For `locked`: No button. No peeking.

---

## The Meta-Demo

This architecture enables a powerful demonstration:

1. Launch NoodleStudio
2. Open "Let's Consciousness!" project
3. User sees Guide, chats with the axolotl
4. User clicks "View Source"
5. Editor panels appear - **same window, same app**
6. User sees Guide's assembly, facets, recipe
7. "This is what NoodleStudio is. You've been using it the whole time."

At no point did the user "become a developer." They just kept going deeper. The ramp was always there.

---

## Practical Implications

### For Project Structure

Every NoodleStudio project can have:

```
MyProject/
├── project.yaml      # Name, version, permission level
├── ui.yaml           # Custom UI shell (optional)
├── Noodlings/        # AI characters
├── Stages/           # 3D environments
└── ...
```

If `ui.yaml` exists and permission < full_access, shell mode activates.

### For the Build System

There is no separate "player build." Building a project packages:
- NoodleStudio (the one binary)
- Project assets
- A flag to start in shell mode

The "standalone app" IS NoodleStudio, pre-configured to open that project in shell mode.

### For Testing

`runtime/cli.py` exists as a **test harness** to validate UI components in isolation. It is NOT the shipping architecture. The shipping architecture is NoodleStudio with shell mode.

### For New Features

When implementing features, ask:
- Does this work in shell mode?
- Does permission level affect visibility?
- Can View Source reveal this?

---

## What Shell Mode Is NOT

- **NOT a separate binary** - Same executable, different mode
- **NOT a "player"** - We avoid this term; it implies passivity
- **NOT locked down** - Even "locked" permission could be unlocked by the user modifying project.yaml (we trust users, we just set defaults)
- **NOT a security boundary** - This is about UX defaults, not DRM

---

## Naming Conventions

From `build-system.md`:

> We avoid "player" (sounds passive). The runtime/output is referred to as:
> - **In code**: `shell_mode` or `project_mode`
> - **In UI**: "Run Project" / "Open Project"
> - **In docs**: "shell mode" or "project experience"

---

## Reference Implementation

See:
- `main_window.py` - Shell mode entry/exit
- `project.py` - Permission level handling
- `ui_canvas.md` - How ui.yaml renders
- `guide-implementation-brief.md` - Let's Consciousness! as shell mode example

---

## Historical Note

This architecture was established January 2026, documented in `claudechat/projects/noodling-studio/con-splo-spec.md` (the "Consciousness Exploratorium" specification). The principle emerged from the design of "Let's Consciousness!" (formerly "Carnival of Consciousness") - the showcase app that teaches consciousness concepts using NoodleStudio itself.

The insight: if Carnival IS NoodleStudio with a different UI, then the demo becomes the product becomes the learning tool. No separate paths for "users" and "creators."

---

*See also: [Design Philosophy](design-philosophy.md) | [Build System](build-system.md) | [UI Canvas](ui-canvas.md)*
