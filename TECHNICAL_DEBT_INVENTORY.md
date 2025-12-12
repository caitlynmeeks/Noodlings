# Technical Debt Inventory

**Philosophy:** "Ordnung muss sein!" (but organically, when friction demands it)

**Last Updated:** December 10, 2025 (Post-Degoosification Backend)

---

## 🔥 HIGH PRIORITY (Blocking or Confusing)

### 1. Model Manager v2 NOT Activated ⚠️

**What:** New multi-provider UI is complete but not active

**Current state:**
- `model_manager_panel_v2.py` exists (850 lines, fully functional)
- Still importing old `model_manager_panel.py`
- Both panels coexist (confusing!)

**Impact:**
- Users can't access multi-provider features
- Anthropic/OpenAI/OpenRouter models not usable in UI
- Wasted 850 lines of code sitting unused

**Fix:**
```python
# In main_window.py, change:
from .panels.model_manager_panel import ModelManagerPanel
# To:
from .panels.model_manager_panel_v2 import ModelManagerPanel
```

**Effort:** 5 minutes

---

### 2. Multi-Provider LLM Execution Missing 🚫

**What:** Can list models from all providers, but can't actually CALL them

**Current state:**
- ProviderManager discovers Anthropic/OpenAI/OpenRouter models ✅
- Inspector shows provider info ✅
- Actual API calls: ONLY Ollama works ❌

**Impact:**
- Setting LARGE → Anthropic does nothing
- LLMFacets still use Ollama even if configured otherwise
- Multi-provider system is "display only"

**What's needed:**
- API call wrappers for each provider
- Request formatting (Anthropic vs OpenAI format differs)
- Response parsing (different JSON structures)
- Error handling per provider
- Rate limiting / retry logic

**Files to create:**
- `noodlestudio/core/llm_client.py` - Unified client
- `noodlestudio/providers/anthropic_client.py`
- `noodlestudio/providers/openai_client.py`
- `noodlestudio/providers/openrouter_client.py`

**Effort:** 2-3 days

---

### 3. DEBUG Console Mode Not Hooked Up 🔌

**What:** UI complete, but server doesn't send debug logs to it

**Current state:**
- Console has DEBUG button and buffer ✅
- `context.log()` in ScriptedFacets collects logs ✅
- Logs never reach NoodleStudio via noodleScope API ❌

**Impact:**
- DEBUG mode shows nothing
- Users think it's broken
- Can't debug ScriptedFacets

**What's needed:**
- noodleScope API endpoint: `POST /debug-log`
- Server sends logs after facet execution
- NoodleStudio receives and displays

**Files to modify:**
- `applications/cmush/server.py` - Send debug logs
- `applications/noodlestudio/noodlestudio/core/noodlescope_client.py` - Receive

**Effort:** 2-3 hours

---

## 🧹 MEDIUM PRIORITY (Cleanup / Polish)

### 4. Backup Files Cluttering Repo 📁

**What:** `.backup` files from development left in working tree

**Files:**
```
?? noodlestudio/core/main_window.py.backup_before_prefs_deletion
?? noodlestudio/core/main_window.py.backup_prefs
```

**Impact:**
- Git status cluttered
- Confusing for other developers
- Not in .gitignore

**Fix:**
```bash
cd applications/noodlestudio
rm noodlestudio/core/main_window.py.backup*
```

**Effort:** 30 seconds

---

### 5. Session Summary Files in Repo 📝

**What:** Session notes meant for personal use, not codebase

**Files:**
```
?? NEURAL_EXPORT_COMPLETE.md
?? PHASE1_COMPLETE.md
?? PHASE2_PLAN.md
?? SCRIPTING_API_INSTALL.md
?? SESSION_SUMMARY_DEC10.md
```

**Impact:**
- Bloats git history
- Not useful for other developers
- Should be in personal notes, not repo

**Options:**
1. **Move to docs folder** if they're useful documentation
2. **Delete** if they're just scratch notes
3. **Add to .gitignore** pattern: `**/SESSION_*.md`

**Effort:** 2 minutes

---

### 6. Test Files in Main Directory 🧪

**What:** Debugging scripts left in app directory

**Files:**
```
?? direct_api_test.py
?? run_api_test_facet.py
?? test_facet_api_example.yaml
?? test_noodle_api.py
?? test_noodle_api_simple.py
```

**Impact:**
- Clutters main directory
- Looks unprofessional
- Not in tests folder

**Fix:**
```bash
mkdir -p tests/scripting_api
mv test_*.py tests/scripting_api/
mv *_test.py tests/scripting_api/
mv test_*.yaml tests/scripting_api/
```

**Effort:** 1 minute

---

### 7. Mysterious `noodlestudio/` Folder 🤔

**What:** Nested `core/noodlestudio/` directory (wat?)

**File:**
```
?? noodlestudio/core/noodlestudio/
```

**Impact:**
- Probably accidental creation
- Confusing import paths
- Could break things if it has code

**Investigation needed:**
```bash
ls -la noodlestudio/core/noodlestudio/
```

**Effort:** 5 minutes to investigate, 1 minute to delete

---

### 8. Agent History State Files Growing 📈

**What:** Agent state files accumulating (should be gitignored)

**Files:**
```
?? cmush/world/agents/.../history/state_465.json
?? cmush/world/agents/.../history/state_466.json
?? cmush/world/agents/.../history/state_467.json
```

**Impact:**
- Git tracks runtime data (shouldn't)
- Repo size grows with every agent action
- Merge conflicts inevitable

**Fix:**
```bash
# Add to .gitignore
echo "applications/cmush/world/agents/*/history/*.json" >> .gitignore
git rm --cached applications/cmush/world/agents/*/history/*.json
```

**Effort:** 2 minutes

---

### 9. Profiler Sessions Tracked 🔍

**What:** Profiler output shouldn't be in git

**File:**
```
?? cmush/profiler_sessions/cmush_session_15589.json
```

**Fix:**
```bash
echo "applications/cmush/profiler_sessions/*.json" >> .gitignore
```

**Effort:** 30 seconds

---

## 🚀 LOW PRIORITY (Future Enhancement)

### 10. Ollama Download Progress Parsing 📊

**What:** Download progress shows "structure" but not live stats

**Current state:**
- Model Manager shows download UI ✅
- Progress tracked in KV ✅
- Live MB/s stats not parsed ❌

**Impact:**
- Minor UX issue
- Users see "downloading..." but not speed/ETA

**What's needed:**
- Parse stdout from `ollama pull`
- Extract MB/s, percentage, ETA
- Update UI in real-time

**Effort:** 3-4 hours

---

### 11. Neural Canvas Training UI 🎨

**What:** Visual training controls not implemented

**Current state:**
- Neural topology editor complete ✅
- MLX code generation works ✅
- Can't train from UI ❌

**What's missing:**
- Training data upload
- Hyperparameter sliders (learning rate, batch size)
- Live loss/accuracy graphs
- Stop/resume training

**Impact:**
- Must train models via code (acceptable for now)
- Less accessible for non-coders

**When to implement:**
- When you start training custom CharmNetworks frequently
- Not urgent if using default topology

**Effort:** 1-2 weeks

---

### 12. Quantum Integration 🔮

**What:** IBM Quantum support infrastructure ready, not connected

**Current state:**
- `TrueRNG` exists ✅
- `QuantumMicrotubuleLayer` implemented ✅
- `IBM_QUANTUM_INTEGRATION_STRATEGY.md` written ✅
- Not wired into facets ❌

**Impact:**
- Cool feature sitting unused
- No quantum "binding experiments" possible

**When to implement:**
- When you say "I want to run a binding experiment NOW"
- Organic friction will trigger it

**Effort:** 2-3 days

---

### 13. Server Tab in Settings ⚙️

**What:** Settings has TODO for Server configuration tab

**From settings_panel.py:500:**
```python
# TODO: Server tab for mush configuration
# server_widget = ServerSettingsWidget()
# self.tabs.addTab(server_widget, "Server")
```

**What it would contain:**
- noodleMUSH server port configuration
- Auto-start on launch toggle (already exists in General)
- Server logs viewer?
- WebSocket settings

**Impact:**
- Minor - server config accessible via files
- Nice-to-have, not critical

**Effort:** 4-6 hours

---

## 🎯 NEXT PRIORITY (From CLAUDE.md)

### 14. Scriptability API Implementation 📝

**Status:** Design complete, not implemented

**What's needed:**
```python
context.noodle.models.set_label("LARGE", "anthropic", "claude-opus-4.5")
context.noodle.neural.get_network(uuid)
context.noodle.agents.get("red-fire-anklebiter")
```

**Files to create:**
- `noodlestudio/scripting/noodle_api.py` (~200 lines)
- `noodlestudio/scripting/models_api.py` (~400 lines)
- `noodlestudio/scripting/neural_api.py` (~400 lines)
- `noodlestudio/scripting/agents_api.py` (~400 lines)
- `noodlestudio/scripting/world_api.py` (~400 lines)
- `noodlestudio/core/uuid_registry.py` (~150 lines)

**Total:** ~850 lines across 8 files

**Effort:** 3-5 days

**Priority:** HIGH (next feature after degoosification)

---

## 🧪 PROCESS DEBT (Running Processes)

### 15. Multiple Wrangler Dev Processes 🤖

**What:** 4 wrangler dev processes still running from testing

**Processes:**
- 385b5b (old)
- b7293e (old)
- f6d46a (old)
- 2f3d0a (current)

**Impact:**
- Wasting CPU/memory
- Port conflicts possible
- Confusing output

**Fix:**
```bash
# Kill all background shells except current
# (Claude can do this with KillShell tool)
```

**Effort:** 1 minute

---

## 📊 Summary by Category

### By Priority:
- 🔥 HIGH: 3 items (v2 activation, LLM execution, DEBUG hookup)
- 🧹 MEDIUM: 6 items (cleanup, gitignore, organization)
- 🚀 LOW: 7 items (polish, future features)

### By Effort:
- **Quick wins** (< 1 hour): 9 items
- **Half-day** (2-6 hours): 4 items
- **Multi-day** (1+ days): 3 items

### By Impact:
- **Blocking features:** Multi-provider execution, DEBUG console
- **Quality of life:** v2 activation, cleanup
- **Nice to have:** Training UI, Quantum, Scriptability API

---

## 🎨 "Timeless Way" Observations

**What's Working:**
- ✅ Features get built quickly and work
- ✅ Exploration reveals what's actually needed
- ✅ No premature optimization
- ✅ Genuine use drives refinement

**Natural Debt Accumulation:**
- 📦 Backup files (natural during big changes)
- 🧪 Test scripts (natural during feature dev)
- 📝 Session notes (natural during iteration)
- 🚧 Half-finished features (waiting for friction)

**Healthy Pattern:**
- Build → Use → Discover friction → Refine
- NOT: Plan everything → Build perfectly → Never change

**Recommendation:**
- Keep this process!
- Just add periodic "Ordnung" sessions (like this one)
- Clean up debris every few weeks
- Don't rush unfinished features

---

## 🗓️ Suggested Cleanup Schedule

### NOW (30 minutes):
1. Delete backup files
2. Add gitignore patterns
3. Kill old wrangler processes
4. Move test files to tests folder

### THIS WEEK (2 hours):
5. Activate Model Manager v2
6. Hook up DEBUG console mode

### THIS MONTH (1 week):
7. Implement Multi-Provider LLM execution
8. Implement Scriptability API

### WHEN FRICTION DEMANDS:
- Neural Canvas training UI
- Quantum integration
- Ollama progress parsing

---

**Ordnung muss sein... but not too much!** 🦆

The "Timeless Way" creates natural debt. That's GOOD - it means you're building based on real use, not speculation. Just clean up periodically when it gets cluttery.

*Christopher Alexander would approve!*
