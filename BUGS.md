# Noodlings Bug Tracker

**Local bug tracking for quick notes. Use GitHub Issues for public tracking.**

---

## Active Bugs

### Critical (System Unusable)

None currently.

### High Priority (Major Feature Broken)

None currently.

### Medium Priority (Minor Feature Broken, Workaround Exists)

None currently.

### Low Priority (Cosmetic, Edge Cases)

None currently.

---

## Fixed Bugs

### November 20, 2025

- **#001: CollapsibleSection bounce-back** (FIXED in e316f37)
  - **Symptom**: Inspector sections bounced closed after expanding
  - **Cause**: Inspector refresh timer destroyed widgets without state preservation
  - **Fix**: Implemented state save/restore pattern from SceneHierarchy
  - **Files**: inspector_panel.py, collapsible_section.py

- **#002: QGroupBox double-trigger** (FIXED in e316f37)
  - **Symptom**: QGroupBox.toggled signal fired twice per click
  - **Cause**: Qt signal handling + complex parent hierarchy
  - **Fix**: Replaced all QGroupBox with CollapsibleSection using direct mouse events
  - **Files**: inspector_panel.py, scene_hierarchy.py

### November 19, 2025

- **#000: Agent broadcast failure when rumination + speech both generated** (FIXED in 7cbefde)
  - **Symptom**: Speech generated but not broadcast when rumination also present
  - **Cause**: perceive_event() returned only first result
  - **Fix**: Return full list, broadcast all results
  - **File**: agent_bridge.py:1854-1867

---

## Bug Reporting Template

When adding a bug, include:

```markdown
### #XXX: Short description

- **Symptom**: What the user experiences
- **Steps to reproduce**:
  1. Do X
  2. Observe Y
- **Expected**: What should happen
- **Actual**: What actually happens
- **Severity**: Critical/High/Medium/Low
- **Component**: noodleMUSH / NoodleSTUDIO / Temporal Model / etc.
- **Files**: Affected code locations
- **Workaround**: Temporary fix (if any)
```

---

## Known Issues (Not Yet Bugs)

**Performance**:
- Response time varies 2-5 seconds (depends on LLM model)
- Timeline export slow with >1000 events
- Multiple agents (10+) causes LLM inference queuing

**Compatibility**:
- macOS only (MLX requirement)
- Requires M1+ hardware
- Python 3.14+ (for latest features)

**Feature Limitations**:
- No voice input/output yet
- No mobile apps yet
- No cloud deployment yet
- Component marketplace not implemented

---

## Crash Log Locations

**macOS**:
- Python crashes: `~/Library/Logs/DiagnosticReports/Python-*.ips`
- Application logs: `~/Library/Logs/NoodleStudio/`
- Temp logs: `/tmp/noodlestudio_*.log`

**noodleMUSH logs**:
- `applications/cmush/logs/cmush_YYYY-MM-DD.log`

**When reporting crashes**:
- Include full stack trace
- Note exact steps to reproduce
- Check if reproducible (100% vs intermittent)

---

## Issue Lifecycle

```
[Reported] → [Confirmed] → [In Progress] → [Fixed] → [Verified] → [Closed]
```

**Reported**: Bug filed but not yet investigated
**Confirmed**: Reproduced and root cause identified
**In Progress**: Actively being fixed
**Fixed**: Code committed, awaiting verification
**Verified**: Tested and confirmed working
**Closed**: Complete, no further action

---

Last updated: November 20, 2025
