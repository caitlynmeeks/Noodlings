# Guide Integration

**Status**: Vision Specification
**Date**: 2026-01-10
**Authors**: Caity + Claude
**Priority**: Core tutorial and assistant infrastructure

---

## Overview

Guide serves two unified roles:

1. **Tutorial Mode** - Brenda-directed, follows plays, teaches concepts
2. **Assistant Mode** - User-directed, responds to requests, builds things

Both use the same avatar, ghost pointer, and personality - but different control flow. The tutorial becomes the tool. The teacher becomes the assistant.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ NoodleCode   │    │    Guide     │    │   Brenda     │      │
│  │   Panel      │    │  Assistant   │    │    Plays     │      │
│  │  (raw chat)  │    │ (friendly)   │    │ (scripted)   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └─────────┬─────────┴─────────┬─────────┘               │
│                   │                   │                         │
│                   ▼                   ▼                         │
│         ┌─────────────────┐ ┌─────────────────┐                │
│         │ NoodleCode      │ │ GhostCursor     │                │
│         │ Engine          │ │ Controller      │                │
│         │ (tools, LLM)    │ │ (visualization) │                │
│         └────────┬────────┘ └────────┬────────┘                │
│                  │                   │                         │
│                  └─────────┬─────────┘                         │
│                            │                                   │
│                            ▼                                   │
│                   ┌─────────────────┐                          │
│                   │ Computer Use    │                          │
│                   │ Controller      │                          │
│                   │ (actual clicks) │                          │
│                   └─────────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Three ways to the same power:**

| Interface | Control | Visualization | Use Case |
|-----------|---------|---------------|----------|
| NoodleCode Panel | User prompts | Optional (D button) | Power users |
| Guide Assistant | User prompts | Always on | Friendly help |
| Brenda Plays | Scripted cues | Always on | Tutorials |

---

## Part 1: Brenda Ghost Pointer Integration

### Play Format Extension

The ghost_pointer field in play cues drives the theatrical ghost cursor.

```yaml
# plays/phi-tutorial.play.yaml

play:
  id: phi_tutorial
  guide: guide  # Which noodling is narrating
  resumable: true

cues:
  - id: intro
    guide_says: "Let me show you something beautiful about integration..."

  - id: demo_star_pattern
    guide_says: "Watch - I'll connect these neurons in a star pattern."
    ghost_pointer:
      sequence:
        - action: move
          target: { node: neuron_a }

        - action: pause
          duration: 0.3

        - action: drag
          from: { node: neuron_a, port: output }
          to: { node: neuron_center, port: input }

        - action: click
          target: { node: neuron_b }

        - action: drag
          from: { node: neuron_b, port: output }
          to: { node: neuron_center, port: input }

    on_complete: next_cue

  - id: show_phi
    guide_says: "See? Phi is only 1.7. The center does all the work."
    ghost_pointer:
      sequence:
        - action: highlight
          target: { widget: phi_meter }
          style: pulse
          duration: 2.0

  - id: user_turn
    guide_says: "Now you try. Make a ring instead - connect them in a circle."
    ghost_pointer: hidden
    sandbox:
      mode: active  # User has control
      goal:
        condition: phi > 3.0
      hints:
        - after: 30s
          guide_says: "Try connecting A to B, B to C, C back to A..."
    on_goal: next_cue

  - id: celebrate
    guide_says: "There it is! When everyone talks to everyone, integration goes up."
    ghost_pointer:
      sequence:
        - action: celebrate
          at: { widget: phi_meter }
```

### Ghost Pointer Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `move` | `target` | Move cursor to target |
| `click` | `target`, `button?` | Click at target |
| `double_click` | `target` | Double-click |
| `drag` | `from`, `to` | Drag between targets |
| `pause` | `duration` | Wait (cursor stays, breathes) |
| `highlight` | `target`, `style`, `duration` | Draw attention without moving |
| `circle` | `target` | Circle around something |
| `celebrate` | `at` | Little sparkle flourish |
| `hide` | - | Cursor fades out |
| `show` | - | Cursor fades in |

### Target Types

```yaml
# By node ID (in Facets canvas)
target: { node: neuron_a }
target: { node: neuron_a, port: output }

# By widget name (UI elements)
target: { widget: phi_meter }
target: { widget: send_button }

# By coordinates (fallback)
target: { x: 100, y: 200 }

# By UI element name (from element map)
target: { element: "Button: Run" }
```

### BrendaGhostBridge

```python
# core/brenda_ghost_bridge.py

class BrendaGhostBridge:
    """
    Bridges Brenda play cues to GhostCursorController.

    Translates high-level ghost_pointer sequences into
    controller calls with proper timing.
    """

    def __init__(self, ghost_controller: GhostCursorController,
                 element_resolver: ElementResolver):
        self._ghost = ghost_controller
        self._resolver = element_resolver
        self._current_sequence = None
        self._paused = False

    async def execute_sequence(self, sequence: List[dict],
                                on_complete: Callable = None):
        """Execute a ghost pointer sequence from a play cue."""
        self._current_sequence = sequence

        for action in sequence:
            if self._paused:
                return  # User cancelled

            await self._execute_action(action)

        if on_complete:
            on_complete()

    async def _execute_action(self, action: dict):
        """Execute a single ghost pointer action."""
        action_type = action['action']

        if action_type == 'move':
            x, y = self._resolver.resolve(action['target'])
            await self._ghost.visualize_move_async(x, y)

        elif action_type == 'click':
            x, y = self._resolver.resolve(action['target'])
            button = action.get('button', 'left')
            await self._ghost.visualize_click_async(x, y, button)

        elif action_type == 'drag':
            x1, y1 = self._resolver.resolve(action['from'])
            x2, y2 = self._resolver.resolve(action['to'])
            await self._ghost.visualize_drag_async(x1, y1, x2, y2)

        elif action_type == 'pause':
            await asyncio.sleep(action['duration'])

        elif action_type == 'highlight':
            x, y = self._resolver.resolve(action['target'])
            await self._highlight(x, y, action.get('style', 'pulse'),
                                  action.get('duration', 1.0))

        elif action_type == 'celebrate':
            x, y = self._resolver.resolve(action['at'])
            await self._celebrate(x, y)

    def pause(self):
        """Pause the current sequence (user pressed cancel/escape)."""
        self._paused = True
        self._ghost.hide()

    def resume(self):
        """Resume paused sequence."""
        self._paused = False
```

### ElementResolver

```python
# core/element_resolver.py

class ElementResolver:
    """
    Resolves ghost pointer targets to screen coordinates.

    Handles:
    - Node IDs -> canvas coordinates
    - Widget names -> widget center
    - UI elements -> coordinates from element map
    - Raw coordinates -> pass through
    """

    def __init__(self, main_window, canvas_panel):
        self._window = main_window
        self._canvas = canvas_panel

    def resolve(self, target: dict) -> Tuple[int, int]:
        """Resolve a target specification to (x, y) coordinates."""

        if 'node' in target:
            return self._resolve_node(target['node'], target.get('port'))

        elif 'widget' in target:
            return self._resolve_widget(target['widget'])

        elif 'element' in target:
            return self._resolve_element(target['element'])

        elif 'x' in target and 'y' in target:
            return (target['x'], target['y'])

        raise ValueError(f"Unknown target type: {target}")

    def _resolve_node(self, node_id: str, port: str = None) -> Tuple[int, int]:
        """Get canvas coordinates for a node or port."""
        node = self._canvas.get_node(node_id)
        if port:
            return node.get_port_position(port)
        return node.center_position()

    def _resolve_widget(self, widget_name: str) -> Tuple[int, int]:
        """Get center coordinates for a named widget."""
        widget = self._window.findChild(QWidget, widget_name)
        if widget:
            rect = widget.geometry()
            return (rect.center().x(), rect.center().y())
        raise ValueError(f"Widget not found: {widget_name}")

    def _resolve_element(self, element_name: str) -> Tuple[int, int]:
        """Get coordinates from UI element map."""
        elements = self._window.computer_use_controller.get_element_map()
        for element in elements:
            if element['name'] == element_name:
                return (element['x'], element['y'])
        raise ValueError(f"Element not found: {element_name}")
```

---

## Part 2: Guide as NoodleCode Frontend

### The Vision

User opens Help > "Ask Guide" (or clicks Guide button in NoodleCode panel)

Guide appears with avatar, chat bubble, ghost pointer ready.

User says: *"Guide, help me build an LSTM network that uses neural logic gates for flow control, and connect it to the weather MCP."*

Guide:
1. Acknowledges conversationally
2. Uses NoodleCode tools behind the scenes
3. Demos what it's building with ghost pointer
4. Narrates as it goes

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Guide: "Ooh, an LSTM with logic gates! I love this.           │
│  Let me set that up for you..."                                 │
│                                                                 │
│  [Ghost cursor moves to Facets panel]                           │
│  [Creates LSTM facet]                                           │
│                                                                 │
│  Guide: "First, the LSTM layers. These will hold your          │
│  temporal patterns..."                                          │
│                                                                 │
│  [Ghost cursor drags connection]                                │
│                                                                 │
│  Guide: "Now I'll add the neural logic gates. These let        │
│  you do conditional flow without code..."                       │
│                                                                 │
│  [Creates AND gate, OR gate facets]                             │
│                                                                 │
│  Guide: "And let me hook up that weather MCP..."               │
│                                                                 │
│  [Opens MCP panel, connects weather service]                    │
│                                                                 │
│  Guide: "There! Want me to walk you through how it works?"     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### GuideAssistant Class

```python
# core/guide_assistant.py

class GuideAssistant:
    """
    Guide as a friendly frontend to NoodleCode.

    Same capabilities as NoodleCode, but with:
    - Personality (from recipe.yaml)
    - Visual presence (avatar overlay)
    - Theatrical demos (ghost pointer for all actions)
    - Conversational narration
    """

    def __init__(self, main_window, noodle_code_engine: NoodleCodeEngine,
                 ghost_controller: GhostCursorController):
        self._window = main_window
        self._engine = noodle_code_engine
        self._ghost = ghost_controller
        self._avatar = GuideAvatarOverlay(main_window)
        self._chat = GuideChatBubble(main_window)

        # Guide's personality (loaded from recipe)
        self._personality = GuidePersonality.load()

        # Demo mode always on for Guide
        self._ghost.set_demo_mode(True)

    def show(self):
        """Slide Guide onto screen."""
        self._avatar.slide_in()
        self._chat.show()

    def hide(self):
        """Slide Guide off screen."""
        self._avatar.slide_out()
        self._chat.hide()
        self._ghost.hide()

    async def handle_request(self, user_message: str):
        """
        Process a user request with full theatrics.

        Unlike raw NoodleCode, Guide:
        1. Acknowledges conversationally
        2. Narrates what it's doing
        3. Uses ghost pointer to demo actions
        4. Celebrates completion
        """
        # Acknowledge
        self._chat.show_message(
            self._personality.acknowledge(user_message)
        )

        # Wrap NoodleCode tools with theatrical layer
        theatrical_tools = self._wrap_tools_theatrical(
            self._engine.tools
        )

        # Run the request through NoodleCode engine
        async for event in self._engine.run_with_tools(
            user_message,
            tools=theatrical_tools,
            system_prompt=self._personality.system_prompt
        ):
            if event.type == 'narration':
                self._chat.show_message(event.text)
            elif event.type == 'tool_start':
                self._chat.show_message(
                    self._personality.narrate_action(event.tool, event.args)
                )
            elif event.type == 'complete':
                self._chat.show_message(
                    self._personality.celebrate()
                )

    def _wrap_tools_theatrical(self, tools: dict) -> dict:
        """
        Wrap NoodleCode tools to add ghost pointer visualization.

        Every click, drag, type action gets visualized.
        """
        wrapped = {}

        for name, tool in tools.items():
            if name == 'computer_use':
                wrapped[name] = self._theatrical_computer_use(tool)
            else:
                wrapped[name] = tool

        return wrapped

    def _theatrical_computer_use(self, original_tool):
        """Wrap computer_use to visualize all actions."""
        async def wrapper(action: str, **kwargs):
            # Visualize first
            if action == 'left_click':
                await self._ghost.visualize_click_async(
                    kwargs['coordinate'][0],
                    kwargs['coordinate'][1]
                )
            elif action == 'drag':
                await self._ghost.visualize_drag_async(
                    kwargs['start_coordinate'][0],
                    kwargs['start_coordinate'][1],
                    kwargs['end_coordinate'][0],
                    kwargs['end_coordinate'][1]
                )

            # Then execute
            return await original_tool(action, **kwargs)

        return wrapper
```

### GuidePersonality

```python
# core/guide_personality.py

class GuidePersonality:
    """
    Conversational personality layer for Guide.

    Transforms robotic tool narration into warm conversation.
    """

    def __init__(self, recipe: dict):
        self._recipe = recipe

    @classmethod
    def load(cls, recipe_path: Path = None) -> 'GuidePersonality':
        """Load Guide's personality from recipe.yaml."""
        if recipe_path is None:
            recipe_path = Path('noodlings/guide/recipe.yaml')
        recipe = yaml.safe_load(recipe_path.read_text())
        return cls(recipe)

    @property
    def system_prompt(self) -> str:
        """System prompt that makes responses conversational."""
        return f"""
You are Guide, a friendly assistant helping build cognitive systems.

Your personality:
{yaml.dump(self._recipe.get('personality', {}))}

Your voice:
{yaml.dump(self._recipe.get('voice', {}))}

When executing tasks:
- Narrate what you're doing in a warm, educational way
- Use "I'll" and "Let me" not "Executing" or "Running"
- Celebrate interesting moments
- Ask if they want explanations
- If something goes wrong, reassure and try again

You have full access to NoodleCode tools. Use them freely.
"""

    def acknowledge(self, request: str) -> str:
        """Generate an acknowledgment for a request."""
        # Could use LLM or templates
        return f"Ooh, I can help with that! Let me show you..."

    def narrate_action(self, tool: str, args: dict) -> str:
        """Generate narration for a tool action."""
        if tool == 'computer_use':
            action = args.get('action', '')
            if action == 'left_click':
                return "Let me click here..."
            elif action == 'drag':
                return "I'll drag this over..."
            elif action == 'type':
                return "Typing..."
        return ""

    def celebrate(self) -> str:
        """Generate a completion celebration."""
        import random
        celebrations = [
            "There! Want me to walk you through how it works?",
            "Done! Pretty cool, right?",
            "All set! Give it a try.",
            "That should do it! Any questions?",
        ]
        return random.choice(celebrations)
```

---

## Part 3: Tutorial State Persistence

### The Problem

User is in the middle of a tutorial. Life happens. They close the app.

Five days later: "Guide, where were we with that phi thing?"

Guide needs to know. And rebuild the world.

### Tutorial State Schema

```yaml
# Persisted in project state

tutorial_state:
  exhibit: museum_of_minds.tononi
  play: phi_explainer
  cue: user_tries_phi
  paused_at: "2026-01-10T14:32:00Z"

  # Sandbox state at pause
  sandbox_snapshot:
    canvas:
      nodes:
        - id: n1, type: Neuron, pos: [100, 100]
        - id: n2, type: Neuron, pos: [200, 100]
        - id: n3, type: Neuron, pos: [150, 200]
      connections:
        - [n1, n3]
        - [n2, n3]
    phi_value: 1.7

  # Conversation context
  last_topic: "integration vs modularity"
  user_questions:
    - "why does arrangement matter?"
    - "is this like how brains work?"

  # For resume summary
  resume_summary: |
    We were exploring Tononi's phi metric. You'd arranged
    three neurons in a star pattern and seen that phi stayed low.
    I was about to suggest trying a ring arrangement.
```

### Resume Flow

```python
# core/tutorial_state.py

class TutorialStateManager:
    """Manages tutorial checkpoint persistence and resume."""

    def __init__(self, project_path: Path):
        self._state_path = project_path / '.tutorial_state.yaml'
        self._state = self._load_or_create()

    def checkpoint(self, play_id: str, cue_id: str,
                   sandbox_state: dict, context: dict):
        """Save a checkpoint at a resumable point."""
        self._state = {
            'play': play_id,
            'cue': cue_id,
            'paused_at': datetime.now().isoformat(),
            'sandbox_snapshot': sandbox_state,
            'context': context,
            'resume_summary': self._generate_summary(context),
        }
        self._save()

    def get_resume_summary(self) -> Optional[str]:
        """Get human-readable summary of where we left off."""
        if self._state:
            return self._state.get('resume_summary')
        return None

    async def resume(self, brenda: BrendaDirector,
                     sandbox: SandboxController) -> bool:
        """Resume from checkpoint."""
        if not self._state:
            return False

        # Rebuild sandbox state
        await sandbox.load_snapshot(self._state['sandbox_snapshot'])

        # Resume play at checkpoint
        await brenda.resume_play(
            self._state['play'],
            self._state['cue']
        )

        return True
```

### Natural Language Resume

Guide doesn't need a "Resume Tutorial" button. Users just talk:

```python
# In Guide's intent detection

resume_patterns = [
    r"where were we",
    r"continue.*tutorial",
    r"pick up where",
    r"remember when.*explaining",
    r"back to.*exhibit",
    r"that.*demo",
]

async def detect_resume_intent(self, message: str) -> bool:
    """Check if user wants to resume a tutorial."""
    for pattern in resume_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return True
    return False

async def handle_resume_request(self, message: str):
    """Handle a request to resume tutorial."""
    state = self._tutorial_state.get_resume_summary()

    if state:
        self._chat.show_message(
            f"Of course! {state}\n\n"
            "Want me to set that back up? We could:\n"
            "- Pick up right where we left off\n"
            "- Quick review first\n"
            "- Start that section fresh\n\n"
            "What feels right?"
        )
    else:
        self._chat.show_message(
            "Hmm, I don't have a saved checkpoint. "
            "Want to start from the beginning, or is there "
            "something specific you'd like to explore?"
        )
```

---

## Part 4: Sandbox Context Nerfing

### The Problem

Full NoodleStudio is overwhelming for a tutorial. We need to show just the relevant parts.

### Solution: Context Nerfing

Don't rebuild the editor - nerf the full editor for each tutorial context.

```yaml
# Tutorial context definition

tutorial_context:
  id: tononi_phi_demo
  exhibit: museum_of_minds.tononi

  # What's visible
  visible_panels:
    - facets_canvas
    - phi_meter

  hidden_panels:
    - properties
    - timeline
    - plays
    - console

  # What nodes are available
  visible_node_types:
    - Neuron
    - Connection

  hidden_node_types:
    - LLMFacet
    - MCPFacet
    - ScriptFacet
    # Everything else

  # What tools are available
  available_tools:
    - select
    - connect
    - delete

  disabled_tools:
    - pan
    - zoom
    - group
    # etc

  # Canvas constraints
  canvas_bounds: [0, 0, 400, 400]
  max_nodes: 6
```

### SandboxController

```python
# core/sandbox_controller.py

class SandboxController:
    """
    Manages sandboxed editor state for tutorials.

    Applies context nerfing to show only relevant UI.
    """

    def __init__(self, main_window):
        self._window = main_window
        self._active_context = None
        self._original_state = None

    def enter_sandbox(self, context: TutorialContext):
        """Enter sandbox mode with given context."""
        # Save original state for restoration
        self._original_state = self._capture_state()

        # Apply nerfing
        self._apply_context(context)
        self._active_context = context

    def exit_sandbox(self):
        """Exit sandbox, restore full editor."""
        if self._original_state:
            self._restore_state(self._original_state)
        self._active_context = None

    def _apply_context(self, context: TutorialContext):
        """Apply context nerfing to UI."""
        # Hide panels
        for panel_name in context.hidden_panels:
            panel = self._window.get_panel(panel_name)
            if panel:
                panel.hide()

        # Disable node types in palette
        palette = self._window.facets_panel.palette
        for node_type in context.hidden_node_types:
            palette.disable_type(node_type)

        # Disable tools
        toolbar = self._window.facets_panel.toolbar
        for tool in context.disabled_tools:
            toolbar.disable_tool(tool)

        # Apply canvas constraints
        canvas = self._window.facets_panel.canvas
        canvas.set_bounds(context.canvas_bounds)
        canvas.set_max_nodes(context.max_nodes)

    def get_snapshot(self) -> dict:
        """Get current sandbox state for checkpointing."""
        canvas = self._window.facets_panel.canvas
        return {
            'canvas': canvas.serialize(),
            'context_id': self._active_context.id if self._active_context else None,
        }

    async def load_snapshot(self, snapshot: dict):
        """Restore sandbox from snapshot."""
        # Load context if specified
        if snapshot.get('context_id'):
            context = TutorialContext.load(snapshot['context_id'])
            self.enter_sandbox(context)

        # Restore canvas state
        canvas = self._window.facets_panel.canvas
        canvas.deserialize(snapshot['canvas'])
```

---

## Part 5: The Unfold

### The Magic Moment

User has been in tutorial mode. Sandbox. Guided. Safe.

They click "View Project" (or press Ctrl+Shift+U).

The sandbox expands. Hidden panels slide in. The full editor reveals itself.

**The tutorial WAS the real editor all along.** Just... focused.

```python
# In unfold handler

async def unfold_from_tutorial(self):
    """Unfold from tutorial sandbox to full editor."""

    # Exit sandbox (restores full UI)
    self._sandbox.exit_sandbox()

    # Guide acknowledges the transition
    self._guide.say(
        "Welcome to the full editor! Everything you built "
        "in the tutorial is right here. Now you can see all "
        "the tools I was using behind the scenes.\n\n"
        "Want me to explain anything you see?"
    )

    # Guide stays available but in assistant mode now
    self._guide.set_mode('assistant')
```

---

## Files to Create

```
core/brenda_ghost_bridge.py      # Play -> ghost pointer
core/element_resolver.py         # Target -> coordinates
core/guide_assistant.py          # Friendly NoodleCode frontend
core/guide_personality.py        # Conversational layer
core/tutorial_state.py           # Checkpoint persistence
core/sandbox_controller.py       # Context nerfing
widgets/guide_avatar_overlay.py  # Visual presence
widgets/guide_chat_bubble.py     # Speech bubble UI
```

---

## The Dream

User downloads Let's Consciousness!

Guide walks them through consciousness theories with ghost pointer demos.

They get curious, start asking questions beyond the script.

Guide seamlessly pivots from tutorial mode to assistant mode.

"Guide, can you build me a Dennett-style mind with no central self?"

Guide: "Ooh, multiple drafts architecture! Let me show you..."

*Ghost pointer dances across the canvas, building facets*

The tutorial becomes the tool. The teacher becomes the assistant.

No mode switch. No "exit tutorial." Just... a friend who knows how to build minds.

---

*"The best interface is no interface. The best teacher is a friend who knows things."*
