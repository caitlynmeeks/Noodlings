# NoodleStudio CLI Commands

Command-line interface for NoodleStudio automation and testing.

**Last Updated:** January 11, 2026

---

## Quick Start

```bash
cd applications/noodlestudio

# Launch normally
./launch_with_log.sh

# Launch and execute a NoodleCode command
./launch_with_log.sh --execute "run the smoke tests"

# Fast launch (no splash) with command
./launch_with_log.sh --no-splash -e "check for issues"
```

---

## CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--execute COMMAND` | `-e` | Execute a NoodleCode command after startup |
| `--project PATH` | `-p` | Open a project on startup |
| `--no-splash` | | Skip the 7-second splash screen |
| `--version` | `-v` | Show version and exit |
| `--help` | `-h` | Show help |

---

## Examples

### Run UI Tests from Command Line

```bash
# Run smoke tests
./launch_with_log.sh --no-splash --execute "run the smoke tests"

# Run all tests
./launch_with_log.sh --no-splash -e "run all UI tests"

# Run specific test file
./launch_with_log.sh --no-splash -e "run_ui_test(test='smoke/panels.ui-test.yaml')"
```

### Open Project and Run Command

```bash
# Open a project
./launch_with_log.sh --project ~/Projects/my-noodling

# Open project and check for issues
./launch_with_log.sh -p ~/Projects/my-noodling -e "check the code for issues"
```

### Automation / CI Usage

```bash
# From Claude Code or CI scripts
cd /Users/thistlequell/git/noodlings_clean/applications/noodlestudio
./launch_with_log.sh --no-splash --execute "run the smoke tests"

# Direct Python invocation
python -m noodlestudio.main --no-splash -e "run the smoke tests"
```

---

## NoodleCode Commands for Testing

When using `--execute`, you can pass any message that NoodleCode understands:

### UI Testing Commands

| Command | What it does |
|---------|--------------|
| `"run the smoke tests"` | Run quick sanity checks |
| `"run all UI tests"` | Run complete test suite |
| `"run_ui_test(suite='smoke')"` | Run smoke test suite |
| `"run_ui_test(suite='e2e')"` | Run end-to-end tests |
| `"run_ui_test(test='smoke/panels.ui-test.yaml')"` | Run specific test file |

### General Commands

| Command | What it does |
|---------|--------------|
| `"check for issues"` | Analyze codebase for problems |
| `"run the unit tests"` | Run pytest unit tests |
| `"take a screenshot"` | Capture current UI state |

---

## Test Suites

Test files are located in `noodlestudio/tests/ui/`:

```
tests/ui/
├── suites.yaml              # Suite definitions
├── smoke/                   # Quick sanity checks (< 30s)
│   ├── panels.ui-test.yaml
│   ├── menus.ui-test.yaml
│   └── startup.ui-test.yaml
└── e2e/                     # Full workflow tests
    └── create_project.ui-test.yaml
```

### Available Suites

| Suite | Description |
|-------|-------------|
| `smoke` | Quick sanity checks |
| `e2e` | Full workflow tests |
| `panels` | Panel-specific tests |
| `facets` | Facet editor tests |
| `all` | All tests |
| `critical` | Must-pass before commit |

---

## How It Works

1. **CLI Parsing:** `main.py` parses arguments before creating the app
2. **Delayed Execution:** Commands execute 2 seconds after window shows (allows NoodleCode engine to initialize)
3. **NoodleCode API:** Uses `NoodleCodePanel.execute_command(message)` to inject commands
4. **Tool Execution:** NoodleCode's `run_ui_test` tool runs the actual tests using Computer Use

### Architecture

```
CLI --execute "run smoke tests"
    |
    v
main.py (QTimer 2s delay)
    |
    v
NoodleCodePanel.execute_command()
    |
    v
NoodleCodeEngine.send_message()
    |
    v
run_ui_test tool -> UITestRunner -> Computer Use
    |
    v
Ghost cursor clicks actual UI elements
```

---

## Writing UI Tests

UI tests are YAML files with phases and steps:

```yaml
name: "My Test"
description: "What this tests"

phases:
  - name: "Verify Something"
    steps:
      - action: click
        target: {menu: "File"}
        comment: "Open File menu"

      - action: wait
        duration: 300ms

      - action: assert
        condition: {dialog_visible: "New Project"}

success:
  message: "Test passed!"
```

### Available Actions

| Action | Description |
|--------|-------------|
| `click` | Left click on target |
| `right_click` | Right click (context menu) |
| `double_click` | Double click |
| `type` | Type text |
| `clear_and_type` | Select all and replace |
| `key` | Press a key |
| `key_combo` | Press key combination |
| `drag` | Drag from one target to another |
| `wait` | Wait fixed duration |
| `wait_for` | Wait for element to appear |
| `assert` | Assert condition is true |
| `screenshot` | Take screenshot |
| `log` | Log a message |

### Target Specifications

```yaml
# Menu bar item
target: {menu: "File"}

# Menu item (in open menu)
target: {menu_item: "New Project..."}

# Panel
target: {panel: "inspector", area: "center"}

# Button by text
target: {button: "Create"}

# Input field
target: {field: "project_name"}

# Dialog
target: {dialog: "New Project"}
```

---

## Troubleshooting

### Command not executing

- Ensure NoodleStudio has time to initialize (2 second delay is built in)
- Check the console output for `[CLI] Executing NoodleCode command: ...`
- Verify NoodleCode panel exists: `window.noodle_code_panel`

### Tests not found

- Tests must be in `noodlestudio/tests/ui/`
- Files must end with `.ui-test.yaml`
- Check `suites.yaml` for suite definitions

### Ghost cursor not visible

- Tests run with `visual: true` by default
- Ghost cursor only appears during test execution
- Check if `GhostCursorController` is initialized

---

## See Also

- [Computer Use QA Spec](/docs/noodlestudio/computer-use-qa.md)
- [Testing Documentation](/docs/noodlestudio/testing.md)
