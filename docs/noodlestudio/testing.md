# Testing

**Regression testing and quality assurance for NoodleStudio**

---

## Overview

NoodleStudio uses pytest with pytest-qt for automated testing. The test suite covers facet execution, Qt signal wiring, Gaussian rendering, and more.

**Current test count:** 128 tests

---

## Quick Start

```bash
cd applications/noodlestudio

# Run all tests
PYTHONPATH=.:../.. pytest

# Verbose output
PYTHONPATH=.:../.. pytest -v

# Single file
PYTHONPATH=.:../.. pytest tests/test_panel_wiring.py

# By name pattern
PYTHONPATH=.:../.. pytest -k "undo"

# Short traceback on failure
PYTHONPATH=.:../.. pytest --tb=short
```

---

## Test Infrastructure

| Component | Location |
|-----------|----------|
| Configuration | `applications/noodlestudio/pytest.ini` |
| Shared fixtures | `applications/noodlestudio/tests/conftest.py` |
| Test files | `applications/noodlestudio/tests/test_*.py` |

### Framework Stack

- **pytest** - Test runner and assertions
- **pytest-qt** - Qt widget testing, signal spying, event simulation
- **PyQt6** - Qt bindings

---

## Test Categories

Use pytest markers to categorize and selectively run tests:

| Marker | Description | When to Run |
|--------|-------------|-------------|
| `@pytest.mark.unit` | Fast tests, no external dependencies | Every commit |
| `@pytest.mark.gui` | Requires Qt event loop | Before merges |
| `@pytest.mark.slow` | Training, rendering, large assets | Manual or nightly |
| `@pytest.mark.integration` | Requires running server | Before release |

### Running by Marker

```bash
# Skip slow tests
PYTHONPATH=.:../.. pytest -m "not slow"

# Only GUI tests
PYTHONPATH=.:../.. pytest -m "gui"

# Unit tests only
PYTHONPATH=.:../.. pytest -m "unit"
```

---

## Development Workflow

### Before Starting Feature Work

Run the test suite to establish baseline:

```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest --tb=short
```

Note any existing failures. Do not fix unrelated issues mid-feature.

### During Development

1. Write tests for new code in `tests/` folder
2. Run affected tests frequently
3. Use fixtures from `conftest.py` rather than reinventing

### Before Committing

```bash
PYTHONPATH=.:../.. pytest -v
```

**All tests must pass** or have documented `@pytest.mark.skip` reasons.

---

## Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `test_agentic_system.py` | 68 | Utility facets, MCP, Player, proxy APIs |
| `test_component_system.py` | 25 | ComponentBase, Registry, Collection, Artbook |
| `test_panel_wiring.py` | 17 | Qt signals, Inspector, Stage View, undo/redo |
| `test_radiance_component.py` | 10 | Gaussian rendering, spatial queries, scene builder |
| `test_clip_queries.py` | 3 | Semantic embedding search |
| `test_gaussian_adapter.py` | 1 | Asset creation |

---

## Writing Tests

### Basic Test Structure

```python
def test_feature_name():
    """Brief description of what's being tested."""
    # Arrange
    setup_data = create_test_data()

    # Act
    result = function_under_test(setup_data)

    # Assert
    assert result.success
    assert result.value == expected_value
```

### Qt/GUI Tests

Use pytest-qt fixtures for widget testing:

```python
def test_signal_emission(main_window, qtbot):
    """Test that selecting entity emits signal."""
    # Arrange
    mock_data = {'id': 'test_001', 'name': 'TestEntity'}

    # Act
    main_window.hierarchy.entitySelected.emit('noodling', mock_data)
    qtbot.wait(50)  # Allow signal propagation

    # Assert
    assert main_window.inspector.current_entity_id == 'test_001'
```

### Radiance/Gaussian Tests

Use the radiance fixtures:

```python
def test_gaussian_query(loaded_radiance_component):
    """Test spatial query on Gaussian data."""
    component = loaded_radiance_component

    # Query nearby Gaussians
    nearby = component.query_radius((0, 1, 0), radius=0.5)

    assert len(nearby) > 0
    assert component.gaussian_count > 0
```

---

## Available Fixtures

Defined in `tests/conftest.py`:

### Qt Fixtures

| Fixture | Description |
|---------|-------------|
| `qapp` | QApplication singleton (session-scoped) |
| `main_window` | Full MainWindow instance |
| `qtbot` | pytest-qt interaction helper |

### Radiance Fixtures

| Fixture | Description |
|---------|-------------|
| `synthetic_radiance_asset` | 1000 random Gaussians |
| `radiance_component` | Component with synthetic data |
| `loaded_radiance_component` | Real asset if available, else synthetic |

### Mock Data Fixtures

| Fixture | Description |
|---------|-------------|
| `mock_noodling_data` | Noodling entity dict |
| `mock_prop_data` | Prop entity dict |
| `mock_zone_data` | Zone entity dict |

### Facet Fixtures

| Fixture | Description |
|---------|-------------|
| `empty_facet_assembly` | Empty FacetAssembly |
| `simple_facet_assembly` | INCOMING -> LLM -> OUTGOING |

### Utility Fixtures

| Fixture | Description |
|---------|-------------|
| `temp_project_dir` | Temporary project directory structure |
| `temp_stage_dir` | Temporary stage with minimal stage.yaml |

---

## Adding New Fixtures

Add shared fixtures to `tests/conftest.py`:

```python
@pytest.fixture
def my_fixture():
    """Description of what this fixture provides."""
    # Setup
    resource = create_resource()

    yield resource

    # Teardown (optional)
    resource.cleanup()
```

For test-file-specific fixtures, define them in the test file itself.

---

## Marking Tests

### Skip a Test

```python
@pytest.mark.skip(reason="Waiting for feature X")
def test_future_feature():
    pass
```

### Skip Conditionally

```python
@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU")
def test_gpu_rendering():
    pass
```

### Mark as Slow

```python
@pytest.mark.slow
def test_full_training_pipeline():
    # Takes several minutes
    pass
```

---

## Debugging Test Failures

### Verbose Output

```bash
PYTHONPATH=.:../.. pytest -v --tb=long
```

### Stop on First Failure

```bash
PYTHONPATH=.:../.. pytest -x
```

### Drop into Debugger

```bash
PYTHONPATH=.:../.. pytest --pdb
```

### Print Statements

```bash
PYTHONPATH=.:../.. pytest -s  # Don't capture stdout
```

---

## Common Issues

### Tokenizers Parallelism Warning

Fixed in `pytest.ini` via environment variable:

```ini
env =
    TOKENIZERS_PARALLELISM=false
```

### Qt Object Deleted

If you see "wrapped C/C++ object has been deleted", ensure widgets are properly parented and not garbage collected during async operations.

### Fixture Not Found

Ensure the fixture is defined in `conftest.py` or the test file, and the parameter name matches exactly.

---

## File Locations

| File | Purpose |
|------|---------|
| `pytest.ini` | pytest configuration, markers, Qt settings |
| `tests/conftest.py` | Shared fixtures |
| `tests/test_*.py` | Test files |

---

## See Also

- [NoodleStudio Overview](overview.md) - IDE introduction
- [Facet System](facets.md) - Cognitive architecture
- [Scripting API](scripting.md) - JavaScript API

---

## Contributing Tests

When adding new features:

1. Add tests in `tests/test_{feature}.py`
2. Use existing fixtures from `conftest.py`
3. Add new fixtures to `conftest.py` if reusable
4. Mark appropriately (`@pytest.mark.slow`, etc.)
5. Run full suite before committing
