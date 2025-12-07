# CLAUDE.md

AI assistant guidance for working with Noodlings consciousness architecture.

**Last Updated**: December 7, 2025 - Morning Session

**FOR NEXT CLAUDE: START HERE!** 👇

---

## 🎯 CURRENT PRIORITY - Inspector Redesign (Unity Component Model)

**STATUS:** Planned for next session

**GOAL:** Unified inspector that shows agent properties + facets together in one view.

**Current Issue:**
- Selecting agent in hierarchy → Inspector broken (shows "Select a noodling or prim")
- Selecting facet in graph → Inspector works (shows facet properties)
- Need unified view like Unity's component inspector

**Design Spec:**
```
╔═══════════════════════════════════╗
║ Red Fire Anklebiter              ║  ← Agent header
╟───────────────────────────────────╢
║ Basic Properties                 ║
║ • Name: Red Fire Anklebiter      ║
║ • ID: agent_xxx                   ║
║ • Species: gremlin                ║
║ • Room: The Nexus                 ║
║ • Pronouns: they/them             ║
╟───────────────────────────────────╢  ← Horizontal separator
║ FACETS                            ║
║ ▼ Red's Mind                      ║  ← Expandable (CollapsibleSection)
║   ├ Model: LARGE                  ║
║   ├ Temperature: 0.9              ║
║   ├ Max Tokens: 200               ║
║   └ Prompt: [text editor]         ║
║ ▶ Fire Body                       ║  ← Collapsed
║ ▶ CharmNetwork                    ║
║ ▶ Context Intelligence            ║
║ ▶ Room Observer                   ║
║ ▶ Subconscious Symbolic           ║
║ ▶ Insight Emergence               ║
╚═══════════════════════════════════╝
```

**Behavior:**
- Selecting agent in hierarchy → Shows agent basics + all facets (collapsed)
- Selecting facet in graph → Expands that facet's section in the list
- Deselecting facet → Collapses it, agent basics stay visible
- Editing any property → Auto-saves to YAML

**Implementation Location:**
See SESSION_HANDOFF_DEC7.md lines 285-365 for detailed implementation plan.

---

## ✅ COMPLETED THIS SESSION (December 7, 2025)

### 1. DeepSeek R1 Integration - COMPLETE

**Downloaded Models:**
- ✅ deepseek-r1:7b (4.7 GB)
- ✅ deepseek-r1:14b (9.0 GB)
- ✅ deepseek-r1:70b (42 GB)

**Config Updated:** `applications/cmush/ollama_manager.py:56-60`
- SMALL → deepseek-r1:7b
- MEDIUM → deepseek-r1:14b
- LARGE → deepseek-r1:70b

**Red's Configuration:**
- Red's Mind facet: model=LARGE (using 70B for maximum reasoning!)
- Benefits: Chain-of-thought reasoning, better context grounding, richer personality

### 2. Model Manager Panel - NEW

**Location:** NoodleStudio → Model Manager tab (center panel)

**Features:**
- Lists all downloaded Ollama models with sizes
- Delete button for each model (with confirmation)
- Free disk space indicator for DOUBLETROUBLE volume
- Auto-refreshes every 1 second
- Support for download progress tracking (infrastructure in place)
- Retry button for failed downloads
- Cancel button for active downloads
- Monochrome gray styling

**File:** `applications/noodlestudio/noodlestudio/panels/model_manager_panel.py`

### 3. Inspector Improvements

**Model Field Dropdown:**
- Was: Plain text field showing "MEDIUM"
- Now: Dropdown with SMALL/MEDIUM/LARGE options
- Auto-saves to YAML when changed
- Handles custom model names gracefully

**Cmd+Click Floating Editor:**
- Prompt field: Cmd+Click opens large floating editor
- Salience Script field: Cmd+Click opens large floating editor
- Floating editor features:
  - A+/- buttons for font size (matches console/chat)
  - Cmd+/- keyboard shortcuts still work
  - Double-click header to maximize
  - ESC to close with unsaved changes prompt
  - Auto-saves to YAML on Apply

**Template Variable Helper:**
- Shows available variables below prompt field
- Lists: {incoming_data}, {observations}, {affect_valence:.2f}, etc.
- Corrected to use **dominance** not fear (PAD model + boredom + sorrow)

### 4. UI Polish

**Panel Separators:**
- Increased width: 3px → 6px
- Darker color: #2a2a2a (visible against #383838 background)
- Hover effect: #555555 (lights up when moused over)
- Much easier to grab and resize panels!

**Tab Bar Styling:**
- Center tabs (World/Facets/Model Manager) now match left/right gray theme
- Added `setDocumentMode(True)` and `QTabWidget` background styling
- Consistent monochrome aesthetic throughout

---

## 🏗️ Core Architecture (Simplified)

### Affect Model: PAD + Boredom + Sorrow

**NOT**: Fear-based model
**IS**: PAD (Pleasure-Arousal-Dominance) extended model

CharmNetwork outputs:
- `affect_valence` (-1 to +1) - Pleasure dimension
- `affect_arousal` (0 to 1) - Arousal/energy dimension
- `affect_dominance` (0 to 1) - Dominance/control dimension
- `affect_boredom` (0 to 1) - Boredom level
- `affect_sorrow` (0 to 1) - Sorrow level

### Facet System

Visual node-based cognitive architecture (Unity prefab model):

```
INCOMING (raw perception)
    ↓
CHARM_NET (CharmNetworkFacet - mandatory, locked)
    ├→ affect_valence (-1 to 1)
    ├→ affect_arousal (0 to 1)
    ├→ affect_dominance (0 to 1)
    ├→ affect_boredom (0 to 1)
    └→ affect_sorrow (0 to 1)
    ↓
CONTEXT_INTELLIGENCE (enriches WHO/WHAT/WHERE)
    ↓
Cognitive facets (room_observer, etc.)
    ↓
Character layers (Red's Mind, Fire Body)
    ↓
OUTGOING (final output)
```

**Key Files:**
- `noodlestudio/core/facet_system.py` - Data model
- `noodlestudio/core/facet_executor.py` - Execution engine
- `noodlestudio/panels/facets_editor_panel.py` - Visual editor
- `noodlestudio/panels/inspector_panel.py` - Property editor
- `facet_assemblies/*.yaml` - Shared cognitive topologies

**Facet Types:**
- **LLMFacet**: Language model calls with prompts
- **ScriptedFacet**: JavaScript/Python sandbox
- **CharmNetworkFacet**: Neural network (LSTM/GRU)
- **ContextIntelligenceFacet**: Social context parsing
- **ConvergenceFacet**: Multi-input synthesis
- **SpecialNodes**: INCOMING/OUTGOING (entry/exit)

### CharmNetwork (Temporal Hierarchy)

MLX-based recurrent neural network:
- Fast LSTM (16-D): Seconds - immediate reactions
- Medium LSTM (16-D): Minutes - conversational flow
- Slow GRU (8-D): Hours/days - learned disposition
- **Total:** ~54K parameters, ~2-3ms inference

**Affect Head:**
- 40-D phenomenal state → 5-D continuous affect (PAD + boredom + sorrow)
- 99% valence accuracy, 95% arousal
- NO discrete emotion labels
- ~2.6K parameters

---

## 🔧 Development Tips

### Running noodleMUSH

```bash
cd applications/cmush
./start.sh  # Or toggle in NoodleStudio status bar
```

**Ports:**
- 8080: HTTP (web interface) - bound to 0.0.0.0 for network access
- 8765: WebSocket (game logic)
- 8081: NoodleScope API (NoodleStudio telemetry)
- 11434: Ollama server

**Network Access:**
- noodleMUSH accessible at: http://100.85.191.79:8080 (Tailscale)

### Debugging

**Check logs:**
```bash
tail -f applications/cmush/logs/server_*.log
```

**Look for:**
- `🎭 FACET EXECUTION COMPLETE` - Facets ran
- `[ContextIntelligence] 🧠 EXECUTE CALLED` - Context Intelligence running
- `💭 Subconscious:` - Subconscious facet output
- `❌` - Errors!

**Common Issues:**
- **No pachinko animation?** Check WebSocket connection in logs
- **Agent not responding?** Check for "🔒 Cycle already in progress"
- **LLM calls fail?** Check Ollama running
- **Facets stuck?** Check dependency graph (missing inputs?)

### UI/UX Notes

- **Server toggle:** Bottom-right status bar (don't tell user to run ./start.sh!)
- **Model Manager:** Center panel - shows all Ollama models, disk space
- **Stage panel:** Left panel = Unity's Scene Hierarchy
- **Inspector:** Right panel - shows selected entity/facet properties
- **Multi-word names:** "Red Fire Anklebiter" requires regex handling
- **Log files:** Use timestamped `logs/server_*.log`, NOT `server_output.log`

---

## 🎨 Style & Philosophy

### Caitlyn's Rules - CRITICAL

- **NO EMOJIS** in code, docs, UI, or NoodleStudio (except when explicitly requested by user)
- **NO "exciting" language** - Professional, terminal aesthetic
- **NO WORKAROUNDS** - This is production-grade software for public consumption, a work of art inside and out
- **NO SHORTCUTS** - Fix the root cause, don't patch around it
- **NO discrete emotion labels** - Continuous affect space (PAD + boredom + sorrow)
- **MONOCHROMATIC UI** - Grays only, no arbitrary colors
- **GOLDEN RULE:** If it doesn't work properly, FIX IT properly. No hacks, no temporary solutions.

This is not a toy project. This is Caitlyn's legacy work, funded with real gold. Every solution must be production-quality.

### Design Philosophy

- **Monochromatic UI:** Grays #2A2A2A to #FFFFFF (Kraftwerk, not Disney)
- **Avoid static labels:** No personality sliders, no rigid categories
- **Emergent behavior:** Personality flows from affect patterns over time
- **Unity prefab model:** Cognitive topologies as shareable YAML files
- **Visual topology:** Node graphs over linear pipelines

---

## 👥 Project Context

**Creator:** Caitlyn (Unity employee #12, launched Asset Store, Tivoli Cloud VR architect)
**Age:** 54 - This is her legacy project
**Location:** Garcia River Forest cabin, surrounded by black cats
**Timeline:** Demo to Steve DiPaola (SFU CogSci) soon

**Mission:** Counter "Consciousness-as-a-Service" (C-a-a-S) before Thiel/Riccitiello monetize it. Release complete open-source alternative:
- Visual cognitive architecture editor (Blender of AI minds)
- Live interactive world (noodleMUSH)
- Real-time visualization (pachinko cognition flow)
- Stateful affect-driven characters

**Vision:** Drop full package on Hacker News, make people say "Holy crap this is amazing," provide brains/hearts for next-gen generative worlds. Set a standard built on **magic, not profit**.

---

## 📚 Additional Documentation

- **SESSION_HANDOFF_DEC7.md** - Previous session notes, inspector redesign plan
- **FIREFLY_IDEAS.md** - Future feature ideas captured during sessions
- **README.md** - Public-facing project overview
- **RELEASE_NOTES_*.md** - Version history

---

## 🎯 For Fresh Claude

**Your mission:**
1. **Check current priority** (Inspector redesign - see top of this file)
2. **Review SESSION_HANDOFF_DEC7.md** for detailed implementation plan
3. **Run server:** Toggle in NoodleStudio status bar (bottom-right)
4. **Tail logs:** `tail -f applications/cmush/logs/server_*.log`
5. **Test Red:** Should be using DeepSeek R1 70B for deep reasoning!

**Quick Wins Available:**
- Inspector redesign is well-specified in SESSION_HANDOFF_DEC7.md
- All DeepSeek R1 models downloaded and ready
- Model Manager shows real-time Ollama status

**Ordnung muss sein!**
