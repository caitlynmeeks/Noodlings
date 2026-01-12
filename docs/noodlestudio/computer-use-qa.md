# Computer Use QA System

**Status**: Specification
**Date**: 2026-01-11
**Authors**: Caity + Claude
**Priority**: High (unifies QA testing and Let's tutorial system)

---

## Overview

This spec defines the **unified Computer Use API** for NoodleStudio. The same scriptable actions power:

1. **QA Testing** - "Claude, did I break anything?"
2. **Let's Tutorials** - Ajo Majo demonstrating charm network wiring
3. **User Demos** - Any noodling app showing users how to use it

One API. Multiple consumers. No privileged internal magic.

### Design Principles

1. **Scriptable, Not Hardcoded** - Every action is expressible in YAML/JSON
2. **Same API for Everyone** - Developers, users, noodlings all use identical tools
3. **Ghost Cursor Native** - Visual feedback is built-in, not bolted on
4. **AI-Native Verification** - Claude can look at screenshots and verify state
5. **Composable** - Simple actions combine into complex workflows

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONSUMERS                                 │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   QA Tests      │  Let's Plays    │  User App Demos             │
│   (YAML)        │  (.play.yaml)   │  (YAML/Script)              │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              ComputerUseScript (this spec)                      │
│   - Action execution                                            │
│   - Target resolution                                           │
│   - Ghost cursor integration                                    │
│   - Screenshot capture                                          │
│   - Assertion checking                                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              ComputerUseController (existing)                    │
│   - Low-level screenshot/click/type                             │
│   - Qt event injection                                          │
│   - Thread safety                                               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              GhostCursorController (existing)                    │
│   - Visual feedback                                             │
│   - Bezier movement                                             │
│   - Click ripples                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: The Scriptable Action Format

Every Computer Use action is expressible as a simple dict/YAML:

### Basic Actions

```yaml
# Click
- action: click
  target: {menu: "File"}

# Click with coordinates (fallback)
- action: click
  target: {x: 150, y: 30}

# Right-click
- action: right_click
  target: {panel: "hierarchy", area: "empty"}

# Double-click
- action: double_click
  target: {item_in_hierarchy: "My Noodling"}

# Type text
- action: type
  text: "Hello world"
  target: {field: "project_name"}  # Optional: click first

# Press key
- action: key
  key: "return"

# Key combo
- action: key
  key: "ctrl+s"

# Drag
- action: drag
  from: {facet_pad: "Perception.output"}
  to: {facet_pad: "Memory.input"}

# Scroll
- action: scroll
  target: {panel: "inspector"}
  direction: "down"
  amount: 200

# Screenshot (for verification)
- action: screenshot
  save_as: "step_3_result.png"  # Optional
```

### Wait Actions

```yaml
# Wait for element to appear
- action: wait_for
  element: {dialog: "New Project"}
  timeout: 5s

# Wait for condition
- action: wait_for
  condition: {facets_loaded: true}
  timeout: 10s

# Fixed delay (avoid when possible)
- action: wait
  duration: 500ms
```

### Assert Actions

```yaml
# Assert element visible
- action: assert
  visible: {panel: "inspector"}

# Assert element has text
- action: assert
  element: {field: "project_name"}
  has_text: "My Project"

# Assert facet exists
- action: assert
  exists: {facet: "Memory"}

# Assert wire connected
- action: assert
  wire: {from: "Perception", to: "Memory"}

# Assert file exists (for save tests)
- action: assert_file
  path: "{{project_dir}}/project.yaml"
```

### Control Flow

```yaml
# Conditional (for branching tutorials)
- action: if
  condition: {dialog_visible: "Welcome"}
  then:
    - action: click
      target: {button: "Skip"}

# Loop (for stress tests)
- action: repeat
  times: 5
  steps:
    - action: click
      target: {button: "Add Facet"}
```

---

## Part 2: Target Resolution

Targets are semantic descriptions that resolve to (x, y) coordinates.

### Target Types

| Target | Example | Resolves To |
|--------|---------|-------------|
| `menu` | `{menu: "File"}` | Menu bar item center |
| `menu_item` | `{menu_item: "New Project"}` | Item in open menu |
| `panel` | `{panel: "inspector"}` | Panel center (or area) |
| `panel_tab` | `{panel_tab: "Facets"}` | Tab in tabbed panel |
| `button` | `{button: "Save"}` | Button center |
| `field` | `{field: "name"}` | Input field center |
| `inspector_field` | `{inspector_field: "Position X"}` | Field in inspector |
| `facet_node` | `{facet_node: "Memory"}` | Facet node in editor |
| `facet_pad` | `{facet_pad: "Memory.input"}` | Connection pad |
| `item_in_hierarchy` | `{item_in_hierarchy: "Toad"}` | Tree item |
| `noodling_in_stage` | `{noodling_in_stage: "Toad"}` | Noodling avatar |
| `chat_input` | `{chat_input: true}` | Chat text field |
| `dialog` | `{dialog: "Settings"}` | Dialog center |
| `element` | `{element: "save_indicator"}` | By object name |
| `x, y` | `{x: 150, y: 300}` | Exact coordinates |

### Area Modifiers

For panels, specify where to click:

```yaml
target: {panel: "hierarchy", area: "center"}   # Default
target: {panel: "hierarchy", area: "top"}
target: {panel: "hierarchy", area: "bottom"}
target: {panel: "hierarchy", area: "empty"}    # Empty space for context menu
```

---

## Part 3: QA Test File Format

Test files live in `tests/ui/` and use `.ui-test.yaml` extension.

### File Structure

```yaml
# tests/ui/smoke/panels.ui-test.yaml

name: "Panel Smoke Test"
description: "Verify all panels open correctly"
tags: [smoke, panels, quick]

# Optional: conditions to skip test
skip_if:
  - {no_project_loaded: true}

# Test phases
phases:
  - name: "Open Each Panel"
    steps:
      - action: click
        target: {menu: "View"}
        comment: "Open View menu"

      - action: click
        target: {menu_item: "Inspector"}
        comment: "Show Inspector"

      - action: assert
        visible: {panel: "inspector"}

      # ... more steps

# What success looks like
success:
  message: "All panels opened correctly"

# What to do on failure
on_failure:
  screenshot: true
  log_ui_state: true
```

### Complete Example: Create Project Test

```yaml
# tests/ui/e2e/create_project.ui-test.yaml

name: "Create New Project"
description: "End-to-end test of project creation workflow"
tags: [e2e, project, critical]

setup:
  - action: wait_for
    element: {splash_complete: true}
    timeout: 30s

phases:
  - name: "Open New Project Dialog"
    steps:
      - action: click
        target: {menu: "File"}

      - action: click
        target: {menu_item: "New Project..."}

      - action: wait_for
        element: {dialog: "New Project"}
        timeout: 5s

  - name: "Fill Project Details"
    steps:
      - action: type
        target: {field: "project_name"}
        text: "QA Test Project"

      - action: type
        target: {field: "author"}
        text: "QA Bot"

  - name: "Create Project"
    steps:
      - action: click
        target: {button: "Create"}

      - action: wait_for
        condition: {project_loaded: true}
        timeout: 10s

      - action: assert
        condition: {project_name: "QA Test Project"}

  - name: "Verify UI State"
    steps:
      - action: assert
        visible: {panel: "hierarchy"}

      - action: assert
        visible: {panel: "stage"}

      - action: screenshot
        save_as: "project_created.png"

cleanup:
  - action: key
    key: "ctrl+w"
    comment: "Close project"

success:
  message: "Project creation workflow complete"
```

---

## Part 4: NoodleCode Integration

### New Tool: `run_ui_test`

Add to `noodle_code_tools.py`:

```python
{
    "name": "run_ui_test",
    "description": """Run a UI test file or test suite.

USE THIS WHEN:
- User asks "did I break anything?" → run smoke tests
- User asks to test specific feature → run targeted test
- Before committing changes → run critical tests

RETURNS:
- Pass/fail status
- Duration
- Any failures with screenshots
- Suggestions for fixes

EXAMPLES:
  # Run all smoke tests
  {"suite": "smoke"}

  # Run specific test
  {"test": "tests/ui/e2e/create_project.ui-test.yaml"}

  # Run tests matching pattern
  {"pattern": "**/facet*.ui-test.yaml"}

  # Run with visual mode (ghost cursor)
  {"suite": "smoke", "visual": true}
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "test": {
                "type": "string",
                "description": "Path to specific test file"
            },
            "suite": {
                "type": "string",
                "enum": ["smoke", "e2e", "panels", "facets", "all"],
                "description": "Test suite to run"
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern for test files"
            },
            "visual": {
                "type": "boolean",
                "description": "Show ghost cursor during tests"
            },
            "stop_on_failure": {
                "type": "boolean",
                "description": "Stop at first failure"
            }
        }
    }
}
```

### New Tool: `ai_verify_ui`

For AI-driven visual verification:

```python
{
    "name": "ai_verify_ui",
    "description": """Take a screenshot and verify UI state using AI vision.

USE THIS WHEN:
- Need to verify something not easily checkable programmatically
- Want to catch visual regressions
- Checking "does this look right?"

The tool takes a screenshot and asks you (Claude) to verify
specific aspects of the UI state.

EXAMPLES:
  # Verify inspector shows correct data
  {"verify": "Inspector panel shows Position X = 100, Y = 200"}

  # Check for visual issues
  {"verify": "No overlapping text or clipped elements visible"}

  # Verify layout
  {"verify": "Facets panel has 3 nodes: Perception, Memory, Response"}
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "verify": {
                "type": "string",
                "description": "What to verify in the screenshot"
            },
            "region": {
                "type": "string",
                "description": "Optional: specific panel/area to focus on"
            }
        },
        "required": ["verify"]
    }
}
```

---

## Part 5: Test Suites

Organize tests into suites:

```
tests/ui/
├── suites.yaml              # Suite definitions
├── smoke/                   # Quick sanity checks
│   ├── panels.ui-test.yaml
│   ├── menus.ui-test.yaml
│   └── basic_click.ui-test.yaml
├── e2e/                     # End-to-end workflows
│   ├── create_project.ui-test.yaml
│   ├── add_noodling.ui-test.yaml
│   └── wire_facets.ui-test.yaml
├── panels/                  # Panel-specific tests
│   ├── inspector.ui-test.yaml
│   ├── facets_editor.ui-test.yaml
│   └── hierarchy.ui-test.yaml
├── facets/                  # Facet system tests
│   ├── add_facet.ui-test.yaml
│   ├── wire_pads.ui-test.yaml
│   └── assembly_load.ui-test.yaml
└── regression/              # Bug regression tests
    └── issue_42_crash.ui-test.yaml
```

### suites.yaml

```yaml
# tests/ui/suites.yaml

suites:
  smoke:
    description: "Quick sanity checks (< 30 seconds)"
    tests:
      - smoke/*.ui-test.yaml

  e2e:
    description: "Full workflow tests"
    tests:
      - e2e/*.ui-test.yaml
    depends_on: smoke  # Run smoke first

  panels:
    description: "Panel-specific tests"
    tests:
      - panels/*.ui-test.yaml

  facets:
    description: "Facet editor tests"
    tests:
      - facets/*.ui-test.yaml

  all:
    description: "All tests"
    tests:
      - "**/*.ui-test.yaml"

  critical:
    description: "Must-pass before commit"
    tests:
      - smoke/*.ui-test.yaml
      - e2e/create_project.ui-test.yaml
      - e2e/wire_facets.ui-test.yaml
```

---

## Part 6: Test Results Format

```yaml
# Output from run_ui_test

result:
  status: "failed"  # passed | failed | error
  suite: "smoke"
  duration: 12.5

  summary:
    total: 8
    passed: 7
    failed: 1
    skipped: 0

  tests:
    - name: "Panel Smoke Test"
      status: "passed"
      duration: 3.2

    - name: "Create New Project"
      status: "failed"
      duration: 5.1
      phase: "Fill Project Details"
      step: 2
      error: "Field not found: project_name"
      screenshot: "failures/create_project_step_2.png"
      suggestion: "The project name field may have been renamed or moved"

  failures:
    - test: "Create New Project"
      phase: "Fill Project Details"
      action: "type"
      target: {field: "project_name"}
      error: "Field not found: project_name"
      screenshot: "failures/create_project_step_2.png"
```

---

## Part 7: Integration with Let's Plays

The same action format works in `.play.yaml` files:

```yaml
# lets_consciousness_intro.play.yaml

beats:
  - id: show_facets_panel
    name: "Showing the Facets Editor"
    on_stage: [ajo]

    direction: |
      Ajo shows the user where facets live.
      Ghost cursor should feel magical.

    ajo:
      speaks: |
        See this panel here? This is where the magic happens.
        Let me show you how neurons connect.

      # SAME ACTION FORMAT as QA tests!
      computer_use:
        - action: click
          target: {panel_tab: "Facets"}
          sync_with_speech: true

        - action: wait
          duration: 500ms

        - action: drag
          from: {facet_pad: "Perception.output"}
          to: {facet_pad: "Memory.input"}
          comment: "Wire the connection"
```

The `computer_use` block in plays uses **identical syntax** to QA tests. No special internal API.

---

## Part 8: Implementation Plan

### Phase 1: Core Script Executor (New)

Create `core/computer_use_script.py`:

```python
class ComputerUseScript:
    """
    Execute scripted Computer Use actions.

    This is the unified API used by:
    - QA tests (via UITestRunner)
    - Let's plays (via PlayExecutor)
    - User demos (via scripting API)
    """

    def __init__(self, main_window, visual_mode: bool = True):
        self.controller = get_computer_use_controller()
        self.ghost = get_ghost_controller()
        self.targets = UITestTargetResolver(main_window)
        self.assertions = UITestAssertions(main_window)
        self.visual_mode = visual_mode

    async def execute(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a single action."""
        action_type = action.get('action')

        if action_type == 'click':
            return await self._do_click(action)
        elif action_type == 'type':
            return await self._do_type(action)
        # ... etc

    async def execute_sequence(self, actions: List[Dict]) -> SequenceResult:
        """Execute a sequence of actions."""
        results = []
        for action in actions:
            result = await self.execute(action)
            results.append(result)
            if not result.success:
                break
        return SequenceResult(results)
```

### Phase 2: Wire to UITestRunner

Update `testing/ui_test_runner.py` to use `ComputerUseScript` instead of duplicating logic.

### Phase 3: Add NoodleCode Tools

Add `run_ui_test` and `ai_verify_ui` tools to `noodle_code_tools.py`.

### Phase 4: Create Initial Test Files

```
tests/ui/smoke/
├── panels.ui-test.yaml      # All panels open
├── menus.ui-test.yaml       # All menus work
└── startup.ui-test.yaml     # App starts cleanly

tests/ui/e2e/
├── create_project.ui-test.yaml
└── wire_facets.ui-test.yaml
```

### Phase 5: Wire to PlayExecutor (Future)

When Let's plays are implemented, they'll use `ComputerUseScript.execute()` for their `computer_use` blocks.

---

## Part 9: Usage Examples

### Developer QA Workflow

```
Developer: "I just changed the Inspector panel. Did I break anything?"

Claude: Let me run the relevant tests.
        [Uses run_ui_test tool with suite="panels"]

        Results:
        ✓ Panel Smoke Test - passed (3.2s)
        ✓ Inspector Field Test - passed (2.1s)
        ✓ Inspector Selection Test - passed (1.8s)
        ✗ Inspector Facet Display - FAILED (4.5s)

        The Inspector Facet Display test failed at step 3:
        Expected to find "PAD Values" section, but it wasn't visible.

        [Screenshot attached]

        Looking at the screenshot, I can see the Inspector panel
        is rendering but the PAD section seems to be collapsed
        or missing. Did you change the facet display layout?
```

### Let's Tutorial

```
Ajo: "Now I'm going to show you how to wire neurons together.
      Watch the glowing cursor..."

      [Ghost cursor sweeps to Perception facet output pad]
      [Click ripple]
      [Cursor drags with trailing afterglow to Memory input pad]
      [Connection wire appears]

Ajo: "See? Just drag from output to input. The connection
      carries information from one facet to another."
```

### User App Demo

```yaml
# In a user's noodling app
demo:
  name: "How to Use the Emotion Dashboard"
  steps:
    - action: click
      target: {button: "Happy"}
      narration: "Click a mood to see how it maps to PAD space"

    - action: wait
      duration: 1s

    - action: screenshot
      narration: "The dashboard now shows positive valence"
```

---

## Success Criteria

- [ ] `ComputerUseScript` class created with unified action execution
- [ ] UITestRunner refactored to use ComputerUseScript
- [ ] `run_ui_test` tool added to NoodleCode
- [ ] `ai_verify_ui` tool added to NoodleCode
- [ ] At least 5 smoke tests created
- [ ] At least 2 e2e tests created
- [ ] Tests can be run from NoodleCode: "run the smoke tests"
- [ ] Ghost cursor works in test visual mode
- [ ] Test results include screenshots on failure
- [ ] Same action format works in .play.yaml files

---

## Notes

- All actions are async to allow for proper timing
- Ghost cursor is optional but default-on for visual feedback
- Targets resolve at execution time (handles dynamic UI)
- Screenshots saved to `tests/ui/screenshots/`
- Failed test screenshots auto-named with timestamp

---

*"The same magic that teaches also tests."*
