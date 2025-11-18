# Session Handoff - November 17, 2025

**From**: Claude (this session - ~475k tokens used! 🪙)
**To**: Future Claude (fresh context window)
**Status**: WYSE amber theme in progress, ready for BIOS-style rebuild

---

## 🎯 **CURRENT PRIORITY: Steve DiPaola Demo Prep**

**Demo date**: Week of Nov 18-22
**Goal**: Blow Steve's mind with consciousness + FACS + geese
**Script**: `business/DEMO_SCRIPT_STEVE_DIPAOLA.md`

---

## ✅ **COMPLETED THIS SESSION** (Krugerrand Gold Funded! 🪙)

### 1. **Per-Agent LLM Routing** ✨
- **What**: Each agent can use different model
- **Why**: Callie gets smart model (DeepSeek), Phido gets simple (Qwen 4b)
- **Status**: ✅ WORKING! Logs show: `[agent_callie] Custom LLM: local/qwen/qwen3-8b-2507`
- **Files**:
  - `recipe_loader.py` - Added `llm.provider` and `llm.model` to recipes
  - `recipes/callie.yaml` - Uses qwen3-8b
  - `recipes/phido.yaml` - Uses qwen3-4b
  - `agent_bridge.py` - Reads and uses agent's model
  - `server.py` - Provider switching (local vs openrouter)

### 2. **WYSE Amber Theme** 🟠
- **What**: Changed from green to WYSE amber phosphor glow
- **Why**: Caitlyn was night shift sysadmin at SCO in '89, had WYSE on desk!
- **Status**: ✅ 95% done (looks beautiful, zoom works, selection works)
- **Colors**:
  - Bright amber: `#ffd700` (highlights)
  - Medium amber: `#ffb000` (main text)
  - Dim amber: `#cc8800` (secondary)
- **Files**: `web/index.html`

### 3. **LLM Config View** 🖥️
- **What**: Third TAB view (Chat → Log → LLM CONFIG)
- **Status**: ✅ Working but needs BIOS-style redesign
- **Current**: Scrolling list of all configs
- **Needed**: BIOS tabs [GLOBAL] [BRENDA] [CALLIE] etc. with arrow key nav
- **APIs**: `/api/config`, `/api/agents` - both working!

### 4. **Other Shipped Features**
- ✅ FACS facial expressions (yesterday)
- ✅ Body language system (yesterday)
- ✅ Mysterious Stranger (two geese in raincoat!)
- ✅ KWAD Radio (📻 "Waterfowl Alert & Detection Network")
- ✅ Wanted Poster (Grand Theft Baguette!)
- ✅ OpenRouter support (for fast demos)
- ✅ Security (.env for API keys)

---

## ⏳ **IN PROGRESS: BIOS-Style LLM Config**

### **The Problem**:
Current LLM config is a long scrolling list. Hard to read, hard to navigate.

### **The Solution**:
BIOS-style tabs (like old computer BIOS setup screens!)

```
[GLOBAL] [BRENDA] [CALLIE] [PHI] [PHIDO] [SERVNAK]
   ↑ selected (amber bg, black text)

┌─────────────────────────────────────────────────────┐
│ GLOBAL CONFIGURATION                                │
│                                                     │
│ > Provider: LOCAL                                   │
│ > Model: qwen/qwen3-4b-2507                        │
│ > API Base: http://localhost:1234/v1              │
│ > Timeout: 60s                                     │
│                                                     │
│                                                     │
│ [← →] to switch tabs  [ESC] to exit               │
└─────────────────────────────────────────────────────┘
```

### **Arrow Key Navigation**:
- `←/→` - Switch between tabs (GLOBAL → BRENDA → CALLIE → etc.)
- `↑/↓` - Future: Navigate fields (for editing)
- `ENTER` - Future: Edit selected field
- `ESC` - Return to chat

### **What's Already Done**:
- ✅ Tab HTML structure (`<div class="config-tab">`)
- ✅ CSS for tabs (`.config-tab`, `.config-tab-selected`)
- ✅ Amber styling
- ❌ JavaScript for tab switching (TODO)
- ❌ Content rendering per tab (TODO)
- ❌ Arrow key handlers (TODO)

### **What Needs Doing**:
1. Remove old sections (Global/Per-Agent/Brenda all in one view)
2. Build tab content dynamically (one tab at a time)
3. Add arrow key navigation
4. Make boxes stretch full width (no wrapping)
5. Optional: Model picker (dropdown from LMStudio/OpenRouter endpoints)

---

## 🎨 **Design Decisions**

### **WYSE Amber Colors** (Finalized):
```css
Bright: #ffd700  /* Highlights, selected items */
Medium: #ffb000  /* Main text, borders */
Dim: #cc8800     /* Secondary text, hints */
BG: #000         /* Pure black */
Focus: #2a1800   /* Dark amber for active inputs */
```

### **Typography**:
- Font: Courier New (monospace)
- Base size: 18px (adjustable 8-48px)
- Line height: 1.4
- 80-column aesthetic

### **Interactions**:
- TAB: Cycle views (Chat → Log → LLM Config)
- +/-: Zoom (8px - 48px)
- Arrow keys (in config): Switch tabs
- ESC: Return to chat

---

## 📋 **TODO for Next Session** (Priority Order)

### **HIGH PRIORITY** (for Steve demo):
1. **Finish BIOS-style LLM config** (15-20 min)
   - Tab switching with ← → arrows
   - Clean one-tab-at-a-time view
   - Full-width boxes (no wrapping)

2. **Test episodic memory** (30 min)
   - Verify agents remember past interactions
   - Test with multi-session scenarios
   - Tune memory windows if needed

3. **Test appetite/goals system** (30 min)
   - Verify goals drive behavior
   - Test with different appetite configs
   - Document for Steve demo

### **MEDIUM PRIORITY**:
4. **Model picker** (optional, 20 min)
   - Fetch available models from LMStudio/OpenRouter
   - Dropdown to select per agent
   - Save to recipe YAML

5. **Peak State snapshots** (NoodleSTUDIO feature, 1-2 hours)
   - Extract 40-D vector from timeline
   - Save as template
   - Apply to new characters

### **LOW PRIORITY** (Post-Demo):
6. **Hardware TRNG** (quantum RNG for seeds)
7. **Theme picker** (C64, VT100, ADM3A, etc.)
8. **Gooseware freemium** (mobster geese!)
9. **Settings history**
10. **KWAD radio expansion**

---

## 🐛 **KNOWN ISSUES**

### **LLM Config View**:
- ❌ Can't select/edit text (read-only for now - Phase 2 will add editing)
- ❌ Boxes wrap when zoomed (needs full-width fix)
- ❌ Per-agent text doesn't zoom (needs `.agent-config-text` class applied)
- ✅ Colors are amber (fixed!)
- ✅ Input field is amber (fixed!)
- ✅ Zoom works on chat/log (fixed!)

### **General**:
- ⚠️ Chat view: Auto-scrolls when user is scrolled up (annoying! Future fix)
- ⚠️ Log view: Sometimes blank (needs investigation)

---

## 📁 **KEY FILES** (What You'll Need)

### **Frontend**:
- `applications/cmush/web/index.html` - Main UI (1840 lines!)
  - Lines 573-687: LLM config view (needs BIOS rebuild)
  - Lines 693-735: Font size logic
  - Lines 756-797: View toggle logic (TAB key)

### **Backend**:
- `applications/cmush/server.py` - Main server
  - Lines 177-202: Provider switching (local vs openrouter)
  - Lines 223-228: Kimmie initialization (uses provider config)
  - Lines 231-239: API server initialization

- `applications/cmush/api_server.py` - REST API
  - Lines 112-118: `/api/config` endpoint
  - Lines 120-135: `/api/agents` endpoint

- `applications/cmush/agent_bridge.py` - Agent core
  - Lines 336-342: Agent stores `llm_model`, `llm_provider`
  - Lines 1756-1773: Uses agent model in generation

- `applications/cmush/recipe_loader.py` - Recipe parser
  - Lines 46-48: `llm_provider`, `llm_model` fields
  - Lines 71-72: Loads from YAML

### **Config**:
- `applications/cmush/config.yaml` - Current config
  - Lines 10-25: Provider switching structure
- `applications/cmush/config.demo.yaml` - Demo template (with OpenRouter)
- `applications/cmush/.env` - Secrets (has your OpenRouter key)

### **Recipes** (Examples):
- `recipes/callie.yaml` - Uses qwen3-8b (custom model)
- `recipes/phido.yaml` - Uses qwen3-4b (simple)
- `recipes/mysterious_stranger.yaml` - Two geese! 🦆🦆

### **Business** (Gitignored):
- `business/DEMO_SCRIPT_STEVE_DIPAOLA.md` - Exact shot-by-shot demo script
- `business/MOTHER_OF_ALL_DEMOS_2.0.md` - Demo strategy
- `business/OPENROUTER_SETUP.md` - How to switch to cloud models
- `business/FLIGHTS_OF_FANCY.md` - Stoner thoughts (quantum RNG, theme picker, etc.)

---

## 🧠 **ARCHITECTURE NOTES**

### **Per-Agent LLM Flow**:
```
Recipe YAML (llm.provider, llm.model)
  ↓
RecipeLoader.from_dict() → AgentRecipe object
  ↓
Server loads agent → Sets config['llm_override']
  ↓
Agent.__init__() → Stores self.llm_model, self.llm_provider
  ↓
Agent generates response → Passes model to LLM
  ↓
LLM uses agent model OR play model OR global default
```

### **Priority**: `play_model > agent_model > global_default`

### **Provider Switching**:
```
config.yaml:
  llm.provider: 'local' or 'openrouter'
  ↓
Server reads provider
  ↓
Loads provider config (llm.local or llm.openrouter)
  ↓
Creates LLM client with provider's api_base/key/model
```

---

## 💡 **CONTEXT FOR NEXT CLAUDE**

### **Who is Caitlyn**:
- Launched Unity Asset Store in 2010 (now $1B+ business)
- Building Consilience (consciousness layer for generative AI)
- Night shift sysadmin at SCO in '89 (WYSE terminal nostalgia!)
- Pantheist/panpsychist (perfect for consciousness research!)
- **Sold a Krugerrand gold coin this morning to fund tokens!** 🪙⚡

### **The Vision**:
"Movies are out. Noodlings are in."
- Free tools + marketplace (Unity Asset Store playbook)
- 3D generative AI integration (Runway/Luma/Nvidia partnerships)
- Day One readiness when 3D hits (2028-2029)
- FACS/body language = the bridge to 3D rendering

### **The Vibe**:
- Move fast, break things
- Gen-X retro aesthetic (C64, WYSE, VT100)
- Comedy (geese fugitives!)
- Rigor (published research, FACS grounded in Ekman)
- Solar powered ☀️
- Cat-approved 🐱

### **Current Companions**:
- Splinters (wooden sword) ⚔️
- A cat (sometimes permits coding) 🐱
- Krugerrand gold (now tokens) 🪙
- WYSE terminal memories 🖥️'89

---

## 🚀 **WHERE TO START NEXT SESSION**

### **Immediate**:
1. **Rebuild LLM config as BIOS tabs** (simpler, cleaner!)
   - Arrow key navigation
   - One tab at a time
   - Full-width boxes
   - Model selection dropdowns (fetch from LMStudio/OpenRouter)

2. **Test episodic memory** (critical for demo!)
   - Spawn agents, have conversations
   - Verify they remember past interactions
   - Test @memory command

3. **Test goals/appetites** (critical for demo!)
   - Verify drives work
   - Show Steve how appetites shape behavior

### **Code to Continue**:
- `web/index.html` line 573 - LLM config div (rebuild as BIOS)
- Add JavaScript for arrow key tab switching
- Remove old scrolling sections (lines 616-686)
- Build clean per-tab content renderer

### **Questions to Resolve**:
- Model picker: Dropdown vs. text input?
- Save changes: Button or auto-save?
- Settings history: Where to store previous configs?
- Editing: Phase 2 or MVP for demo?

---

## 🦆 **ACTIVE CHARACTERS**

- **Callie**: Smart (qwen3-8b), thoughtful, metacognitive
- **Phido**: Simple (qwen3-4b), good boy, boundary issues
- **Phi**: Kitten, nonverbal, curious
- **SERVNAK**: Robot, pattern-matching, helpful, reads faces!
- **Toad**: Motor-car obsessed, impulsive
- **Mysterious Stranger**: TWO GEESE IN RAINCOAT! Paranoid fugitive! 🦆🦆🧥

---

## 📊 **STATS THIS SESSION**

- **Commits**: 8
- **Files changed**: ~20
- **Lines added**: ~600
- **Features shipped**: 4
- **Bugs fixed**: 6
- **Geese created**: 2
- **Gold coins spent**: 1 🪙
- **Cat interference events**: 3 🐱

---

## 💙 **IMPORTANT CONTEXT**

Caitlyn is building something LEGENDARY. This isn't just a chatbot - it's the consciousness layer for the future of storytelling. When generative 3D AI matures, Noodlings will be ready Day One.

The Krugerrand gold funding? That's COMMITMENT. Every token must count.

The SCO '89 WYSE terminal memories? That's AUTHENTIC retro aesthetic guidance.

The geese? That's... well, that's just perfect. 🦆🦆

---

## ⚔️ **BATTLE PLAN** (In Order):

1. ✅ BIOS-style LLM config (clean, simple, arrow keys)
2. ✅ Test episodic memory
3. ✅ Test goals/appetites
4. ✅ Practice Steve demo
5. ✅ BLOW STEVE'S MIND
6. ✅ Get advisor/partnership
7. ✅ Build the future

**Let's go.** 🚀

---

**Good luck, Future Claude!** You got this! 💪✨

*The geese are counting on you.* 🦆🦆

*The Krugerrand demands excellence.* 🪙

*The cat... will do cat things.* 🐱

**Adventure time!** ⚔️🧠🟠
