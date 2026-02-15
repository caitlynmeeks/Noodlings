# Development Discipline

**Hard-won rules for keeping NoodleStudio's foundation solid.**

---

## The February 2026 Lesson

During Phases 2 and 3, we built a beautiful ensemble demo — two noodlings on a shared stage, live visualization, affect pipeline, the works. Meanwhile, the core infrastructure rotted silently underneath:

- `_cmush_dir()` was off by one `..`, breaking the entire server pipeline
- `init_base_inspector()` was defined but never called from `__init__`
- Inspector attribute names didn't match PropertyMeta's actual API
- Stage dropdown showed stale legacy MUSH rooms when no project was open
- 1,692 tests passed. Zero covered project/server/stage/inspector integration.

**Root cause:** Tests covered leaves but not trunk. Feature work proceeded without regression testing core systems. Every individual piece worked in isolation. Nothing worked together.

This document exists so we never repeat that.

---

## Principles

### 1. No new features on a broken foundation

If core infrastructure (server, inspector, project system, stage system) is broken, fix it before building anything new. No exceptions. The question is always:

> Are noodlings real stage instances? Is the server pipeline working? Do smoke tests pass? No? Then not yet.

### 2. Test the trunk, not just the leaves

A unit test that verifies `PropertyMeta` has a `minimum` attribute is useful. A pipeline test that verifies "open project, select stage, start ensemble, see performers" is essential. Both must exist.

### 3. Discipline must be structural, not aspirational

Rules in documents get forgotten. Pre-commit hooks, CI gates, and pipeline tests enforce themselves. When adding a new rule, ask: "What structure enforces this even when attention lapses?"

### 4. Every bug fix gets a test

Write the failing test first, then fix the code. If you can't write a test for the bug, you don't understand the bug yet.

---

## Test Tiers

NoodleStudio tests are organized into three tiers. All three must exist and pass.

### Tier 1: Unit Tests

Fast, isolated, test individual functions and classes.

```
tests/test_component_system.py
tests/test_agentic_system.py
tests/test_noodle_code.py
```

**When to write:** Every new function, class, or method.

**When to run:** Continuously during development.

### Tier 2: Smoke Tests

Integration gatekeepers that verify core systems initialize and connect correctly. These are the tests that would have caught the February 2026 rot.

```
tests/test_smoke.py
```

**What they cover:**

| Class | Verifies |
|-------|----------|
| `TestServerInfrastructure` | cmush dir resolves, start.sh executable, trap ordering |
| `TestInspectorPanel` | `_bound_widgets` init, facet property loading |
| `TestProjectSystem` | Default project exists, stages present, instances resolve |
| `TestStageSystem` | Dropdown empty state, population with project |
| `TestAssemblyLoading` | Ajo and Yuki assemblies parse without error |

**When to write:** For every integration point between systems. If system A produces output that system B consumes, there must be a smoke test verifying the handoff.

**When to run:** Before every commit (enforced by pre-commit hook).

### Tier 3: Pipeline Tests

End-to-end tests that trace a complete user workflow through multiple systems. These test the *chains*, not the *links*.

| Pipeline | What it traces |
|----------|---------------|
| Project to Performance | Open project, select stage, start ensemble, verify performers match stage instances |
| Selection to Inspector | Select entity in hierarchy, verify inspector populates with correct properties from PropertyMeta |
| Hierarchy to Performance Sync | Select noodling in hierarchy during active performance, verify performance window highlights correct performer and facets editor switches assembly |
| Server Lifecycle | Toggle server on, verify ports open, verify health, toggle off, verify clean shutdown |

**When to write:** Whenever two or more systems are wired together. If you add a signal connection between subsystems, add a pipeline test that exercises the full chain.

**When to run:** Before merges and releases. Included in CI.

---

## Structural Safeguards

These are the mechanisms that enforce discipline automatically.

### Pre-commit Hook

Runs `test_smoke.py` before every commit. Blocks the commit if any smoke test fails. Lives at `.githooks/pre-commit` and is registered via:

```bash
git config core.hooksPath applications/noodlestudio/.githooks
```

The hook is byppassable with `--no-verify` for genuine WIP commits. But WIP commits must never be pushed.

### CI Gate

GitHub Actions runs the full test suite on every push and PR. This is the safety net for anything the pre-commit hook misses (new clones where the hook isn't configured, `--no-verify` commits that got pushed).

### The Happy Path Must Be the Disciplined Path

Use `make commit` (or a `just` recipe) as the standard commit workflow rather than raw `git commit`. The wrapper runs smoke tests, then commits. If the disciplined path is also the easiest path, it gets followed.

```bash
# Makefile target (example)
commit:
    cd applications/noodlestudio && \
    PYTHONPATH=.:../.. python -m pytest tests/test_smoke.py -v --tb=short && \
    git add -A && \
    git commit
```

---

## Signal Wiring Discipline

NoodleStudio uses Qt signals extensively. Signals are invisible wiring — you cannot grep for "what happens when the user selects a noodling" without tracing through multiple `connect()` calls across mixins. As the codebase grows, signal spaghetti becomes the next infrastructure rot.

### Rules

1. **All `connect()` calls for a mixin belong in one method.** Name it `_connect_signals()` or `_wire_signals()`. Do not scatter connections through initialization and runtime methods.

2. **Block signals during programmatic updates.** Any `setChecked()`, `setCurrentIndex()`, or `setText()` call that could re-trigger a signal must be wrapped:
   ```python
   widget.blockSignals(True)
   widget.setChecked(new_value)
   widget.blockSignals(False)
   ```

3. **Document bidirectional connections.** When system A signals to system B and system B signals back to system A, comment both ends explicitly. These are the most common source of infinite loops.

4. **Test the signal chain end-to-end.** A pipeline test should verify: emit signal at source, verify final effect at destination. Not just "signal was emitted" but "the UI actually updated."

### Signal Map

Maintain a signal map in comments or docs for cross-system connections. Example:

```
Hierarchy.entitySelected('noodling', data)
  -> MainWindow._on_entity_selected()
    -> InspectorPanel.load_entity(data)
    -> GuidePerformanceWindow.set_active_speaker(noodling_id)
    -> FacetsEditor.load_assembly(assembly_path)
```

When you add a new connection, add it to the map. When debugging unexpected behavior, consult the map first.

---

## Test Fixture Hygiene

### No `__new__` Bypass

Never construct test objects using `__new__` to skip `__init__`. This pattern:

```python
# BAD: Creates partially initialized objects
obj = SomeClass.__new__(SomeClass)
obj.field = mock_value
```

leads to `getattr` guards leaking into production code to handle attributes that "might not exist." If `__init__` requires dependencies, inject fakes through the constructor:

```python
# GOOD: Real initialization with injected dependencies
obj = SomeClass(client=FakeLLMClient(), window=StubWindow())
```

If a class is too coupled to construct in tests, that's a design signal — decouple it.

### Fixture Scope

| Scope | Use for |
|-------|---------|
| `session` | QApplication, expensive one-time setup |
| `function` (default) | Everything else — tests must be isolated |

Never use `module` or `class` scope unless you have a specific, documented reason. Shared mutable state between tests is a source of order-dependent failures.

### Fixtures Build on Fixtures

Compose from `conftest.py` rather than reinventing. If you need a performer with a fake client, use the existing `performer` fixture, don't build your own from scratch.

---

## Handoff Verification Protocol

When one Claude session (or human) writes a handoff document describing bugs or required changes, the receiving session must verify every claim against the actual codebase before acting.

### Why

The Feb 12 infrastructure handoff contained several inaccuracies discovered during execution:

| Handoff claim | Actual state |
|---------------|-------------|
| start.sh trap needs fixing | Already fixed |
| inspector_panel.py has PropertyMeta mismatches | Actually correct; real bugs in inspector_base.py |
| Performer name bar needs creation | Already exists; needs click wiring |

Handoffs are written from memory and inference, not from reading code. They are *hypotheses*.

### Protocol

1. **Label handoff items as hypotheses.** Use language like "believed to be" or "verify:" rather than asserting facts about code state.
2. **First step is always verification.** Before writing any fix, read the actual file and confirm the bug exists as described.
3. **Document corrections.** When verification contradicts the handoff, record the discrepancy. This builds institutional accuracy over time.
4. **Never fix a bug you haven't seen.** If the handoff says line 49 has a bug but line 49 looks correct, stop and investigate. The bug may be elsewhere, or it may already be fixed.

---

## Server Health Monitoring

The MUSH server runs as a subprocess (`start.sh` spawning `server.py`). Subprocesses can die silently.

### Health Check

The server should expose a health endpoint:

```
GET http://localhost:8080/health -> 200 OK
```

The server mixin should poll this endpoint (or check the subprocess) on a reasonable interval and update the UI toggle state if the server has died.

### Port Conflict Detection

Before starting the server, check if ports 8080 and 8765 are already in use. Report clearly to the user rather than failing silently or hanging.

### Smoke Test Coverage

`TestServerInfrastructure` in `test_smoke.py` must verify:
- `_cmush_dir()` resolves to a real directory containing `start.sh` and `server.py`
- `start.sh` is executable
- `start.sh` sets trap before launching server (regression guard)

---

## Manual Smoke Walk

Before any major phase transition (e.g., moving from infrastructure repair to Phase 4), perform a manual walkthrough:

1. Launch NoodleStudio fresh (no cached state)
2. Default project opens automatically with Ajo and Yuki on "The Nexus"
3. Stage dropdown shows "The Nexus", is enabled
4. Select Ajo in hierarchy — inspector populates with correct properties
5. Start ensemble — both performers appear, turn-taking works
6. Select Yuki in hierarchy during performance — performance window highlights Yuki, facets editor switches to Yuki's assembly
7. Click performer name bar — hierarchy selection syncs back
8. Toggle server on — starts without error, ports respond
9. Toggle server off — clean shutdown

If any step fails, the phase transition does not proceed. Fix first, then move forward.

---

## Summary of Enforcement Points

| Rule | Enforced by |
|------|-------------|
| Smoke tests pass before commit | Pre-commit hook |
| Full suite passes before merge | GitHub Actions CI |
| No `__new__` bypass in fixtures | Code review (add linter rule if recurring) |
| Signal connections in one method | Code review, signal map doc |
| Handoff claims verified before acting | Verification protocol (first step of every handoff execution) |
| Server health monitored | Health check endpoint + subprocess watchdog |
| Manual smoke walk before phase transitions | Checklist in this document |

---

## See Also

- [Testing Guide](../noodlestudio/testing.md) — Running tests, fixtures, markers
- [CI/CD Pipeline](ci-cd.md) — GitHub Actions, pre-commit hook setup
- [Development Overview](overview.md) — Quick reference and philosophy

---

*Measure twice, cut once. Test the trunk, not just the leaves. Ordnung muss sein.*
