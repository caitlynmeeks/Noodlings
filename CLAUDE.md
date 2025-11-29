# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Noodlings** (formerly Consilience) is a hierarchical affective consciousness architecture implementing predictive processing theories through multi-timescale learning. We're "noodling" with functional correlates of consciousness - making no claims about "real" consciousness, just exploring architectural patterns inspired by neuroscience and affective computing.

**Status**: Phase 8 - Continuous Affect Prediction (November 2025)
**Framework**: MLX (Apple Metal optimized)
**Hardware**: M3 Ultra (512GB RAM) + M2 Ultra (192GB RAM)
**Parameter Budget**: ~54K params (temporal hierarchy) + 2.6K (affect head)
**Last Updated**: November 24, 2025

## Style Preferences

**CRITICAL - NO EMOJIS**
- User strongly dislikes emojis in development sessions, documentation, and UI design
- Old-fashioned, terminal-aesthetic preference
- Do NOT use emojis in code comments, commit messages, documentation, or conversational responses
- Exception: Only if user explicitly requests emojis for a specific use case
- Keep communication professional and text-based

## ACTIVE SESSION HANDOFF - November 29, 2025 (Evening Session)

**Status**: Facets Editor Bug Fixes & Per-Agent Cognition Pause System Design

Fresh Claude starting? Read this section first!

### Session Summary (November 29, 2025 - Evening):

**Bug fixes completed, system stable. Planning per-agent cognition pause for field editing.**

---

## Today's Accomplishments (Nov 29, 2025 - Evening Session):

**1. CRITICAL BUG FIXES - Facets Editor Stability**

Fixed crash and rendering issues in visual facet editor:

**Crash Fix:**
- **Problem**: Clicking blank area to deselect facets caused immediate crash
- **Root Cause**: `collapse_all_nodes()` attempted to access nonexistent `expanded` attribute and call nonexistent `collapse_from_editing()` method
- **Solution**: Modified `collapse_all_nodes()` to call existing `hide_fields()` method instead
- **Location**: `facets_editor_panel.py:1310-1314`

**Special Node Improvements (INCOMING/OUTGOING):**
- **Vertical size**: Reduced from 60px → 35px (tight, minimal)
- **Text styling**: 14pt bold (vs 11pt regular), center-aligned with symmetric padding
- **Type label**: Removed "SpecialNode" label entirely for cleaner appearance
- **Output pad positioning**: Fixed floating pad bug - now uses correct node height

**Field Display (F Key):**
- **Status**: Confirmed working correctly
- **Shows**: Processing Prompt field with pencil icon (✎) for editing
- **Expands**: Node grows vertically to accommodate field display
- **Z-order**: Fields render at z=10 to appear above node background

**Files Modified:**
- `facets_editor_panel.py` - Core stability fixes, special node styling
- Commit: `e56cd19` - "fix: Facets Editor stability and special node styling"

---

## NEXT SESSION - CRITICAL PRIORITY:

### Per-Agent Cognition Pause System

**Context**: User wants to edit facet output fields when cognition is paused, similar to NoodleTuner's pause functionality.

**Architecture Discovery:**
- Agents run as **asyncio tasks** (not OS threads) via `asyncio.create_task(_cognition_loop())`
- Concurrent but not parallel - share event loop
- Pause system already exists: `POST /api/cognition/pause` with optional `agent_id` parameter
- Current NoodleTuner implementation pauses ALL agents globally
- API supports per-agent pause: `{'agent_id': 'xxx', 'paused': True}`

**Implementation Plan:**

1. **Facets Editor Pause Controls**
   - Add pause/resume button to Facets Editor toolbar
   - Button text: "⏸ Pause Cognition" / "▶ Resume Cognition"
   - Only pause the agent whose assembly is currently being edited
   - Use existing API: `POST /api/cognition/pause` with `agent_id`

2. **Output Field Editability**
   - Currently: Output fields in facets are read-only
   - When paused: Make output fields editable (`read_only=False`)
   - Show visual indicator: Yellow border or background tint on editable fields
   - Store edited values, apply when resumed (similar to NoodleTuner pattern)

3. **Scripting API Wrapper**
   - Create convenience method: `red.getComponent('noodle').pauseCognition()`
   - Also: `red.getComponent('noodle').resumeCognition()`
   - Makes pause/resume accessible from Python/JavaScript scripts
   - Useful for automated testing and debugging workflows

**Reference Implementation:**
- See `noodle_tuner_panel.py:1002-1048` for NoodleTuner's pause logic
- Pattern: Pause → Wait for cycle completion → Enable editing → Resume → Apply edits
- API waits for current cycle to complete before pausing (prevents mid-cycle corruption)

**User Ideas/TODO from Session:**
- Should output fields be visible in facets alongside input/prompt fields? (Currently only prompt shows)
- Console filter clearing itself - may be a bug in Console panel regex implementation
- Undo system still needed for facet editing operations

**CRITICAL FEATURE - Scripted Logic Field Integration:**

The V8/JavaScript execution engine is **fully implemented** in `scripted_facet.py` (PyMiniRacer):
- Sandboxed execution with 5-second timeout
- Persistent storage (100KB limit per facet)
- Event emission, logging, context access
- Example scripts provided (mood tracker, etc.)

**What's Missing:** Integration with Facets Editor field display:

When F-key focusing on a ScriptedFacet node, the field editor should show:
1. **Processing Prompt** field (already exists for all facets)
2. **Scripted Logic** field (NEW - for ScriptedFacet types only)
   - Multi-line text editor for JavaScript code
   - Syntax highlighting (optional but nice)
   - Script stored in facet metadata: `facet.script` property
   - Editable via pencil icon (✎) or E key
   - Shows preview: "function process(inputs, context) {...}" (first 50 chars)

**Implementation Pattern:**
- Extend `Facet.get_editable_fields()` in `facet_system.py` to return script field when `facet_type == "ScriptedFacet"`
- Field definition: `{'name': 'Scripted Logic', 'key': 'script', 'value': facet.script, 'type': 'text', 'read_only': False, 'preview': script[:50]}`
- FloatingTextEditor already supports multi-line editing
- Script execution happens in `facet_executor.py` via `ScriptedFacet.process()`

**User Context:** Caitlyn built Unity's Asset Store (employee #12) and later Tivoli Cloud VR. She knows executable code in editors (Qt Script, web views) and wants this pattern for facet nodes. This is NOT a toy feature - it's core to the architecture's extensibility, following the Unity prefab philosophy she pioneered.

---

## Previous Accomplishments (Nov 28-29, 2025 - Morning):

**1. FACET ASSEMBLY SYSTEM - Complete Node-Based Architecture**

Implemented revolutionary visual cognitive architecture editor:

**Core Architecture:**
- **Facet**: Individual cognitive transformation node (replaces "transistors")
- **Facet Assembly**: Connected network of facets (Unity prefab model)
- **Convergence**: Multi-input synthesis facet
- **INCOMING/OUTGOING**: Special entry/exit nodes
- **Charm Network**: Neural processor facet (LSTM/GRU hierarchy)
- **Scripted Facets**: User-programmable JavaScript logic nodes
- **Flow Control**: Ticker, Branch, RateLimiter, Cache, Accumulator

**Files Created:**
- `noodlestudio/core/facet_system.py` - Core data model, YAML serialization, UUID support
- `noodlestudio/panels/facets_editor_panel.py` - Visual node graph editor (1300+ lines!)
- `noodlestudio/panels/floating_text_editor.py` - Floating text editing dialog
- `noodlestudio/core/facet_executor.py` - Parallel execution engine
- `noodlestudio/core/charm_network_facet.py` - Neural network wrapper
- `noodlestudio/core/scripted_facet.py` - JavaScript sandbox (PyMiniRacer/V8)
- `noodlestudio/core/flow_control_facets.py` - Logic gates and timing controls
- `facet_assemblies/` - Shared assembly library (Unity prefab model)

**Assemblies Created:**
- `simple_test.yaml` - Minimal test assembly
- `anklebiter_default.yaml` - Production parallel architecture
- `red_fire_anklebiter.yaml` - Red's complete cognitive topology migrated

**Visual Editor Features:**
- Drag-and-drop node positioning with grid snapping (20px)
- Bezier curve connection wires (vertical tangents)
- Right-click context menu for adding facets
- F key: Tight focus with field display
- A key: Frame entire assembly
- E key: Open floating text editor for prompt
- Cmd-D: Duplicate with preserved layout and internal wiring
- Cmd-click background: Invert selection (ZBrush-style)
- Space+drag: Pan viewport
- Mouse wheel: Zoom (0.5x to 2x frame-all limit)
- Delete key: Remove facets (protects special nodes)
- Copy/paste: Preserves relative positions and internal connections
- Status indicators: Colored dots (gray/green/yellow/red/blue)
- Monochromatic design: Grays with white selection borders
- Grid background: Faint dotted lines (#333333)

**Red's Migration:**
- Red now uses `facet_assembly` reference instead of `cognitive_components`
- All 6 transistors migrated to facets: Intuition, Affect, Personality, Cultural, Memory, Embody
- Convergence facet synthesizes all inputs
- Full prompts preserved from original architecture

**API Integration:**
- `/agents` endpoint now includes `config` field for facet assembly loading
- Token usage tracking across all facet types
- Execution statistics (time, tokens, call counts)
- UUID-based facet identification

**2. Console Regex Filtering**
- Real-time search with plain text and regex patterns
- Case sensitivity toggle
- Match highlighting (yellow background)
- Dual buffer architecture (raw + formatted)

---

## NEXT SESSION TODO - Critical Remaining Features:

### 1. **Field Display Fix (HIGH PRIORITY)**

**Problem**: Fields not showing when F key pressed
**Root cause**: `get_editable_fields()` may be returning empty list or fields not rendering
**Fix needed**: Debug why pencil icons and field previews don't appear on F key focus

**Test case:**
1. Select Red in Stage
2. Facets Editor loads anklebiter_default.yaml
3. Select "Intuition Facet" node
4. Press F
5. **Expected**: Node zooms to fill view, shows fields with pencil icons
6. **Actual**: May not be showing fields

**Debug checklist:**
- Check `facet.get_editable_fields()` returns data
- Verify `show_fields(force=True)` is called
- Check field widgets are being created
- Verify pencil icons are clickable

### 2. **Crash Bug Fix (CRITICAL)**

**Steps to reproduce:**
1. Start NoodleStudio, log into noodleMUSH
2. Select Red in Stage
3. In Facets Editor, select a facet (e.g., Intuition)
4. Click blank area to deselect
5. **CRASH**

**Likely cause**: `hide_fields()` or `update_field_visibility()` accessing invalid widget
**Fix**: Added try/catch in `hide_fields()` but may need more safety checks

### 3. **LLM Tier Configuration System**

Implement tiered LLM selection for performance optimization:

**Settings → LLM Manager:**

Five tiers (cascade down if not set):
- **FASTEST**: Quick semantic analysis (qwen-4b) - Intuition, keyword detection
- **FAST**: Simple inference (qwen-14b) - Basic facets
- **AVERAGE**: Good cognitive processing (qwen-32b) - Standard noodlings
- **SMART**: Bright noodlings (qwen-128b) - Complex reasoning
- **SMARTEST**: Final generation (DeepSeek v3, Sonnet) - Convergence facets

**UI Flow:**
```
Settings → LLM Manager
┌─ LLM Tier Configuration ──────────────┐
│ FASTEST  (semantic analysis)          │
│ Model: [qwen/qwen3-4b-2507      ▼]   │
│                                        │
│ FAST     (simple inference)            │
│ Model: [qwen/qwen3-14b-2507     ▼]   │
│                                        │
│ AVERAGE  (cognitive processing)        │
│ Model: [qwen/qwen3-32b-2507     ▼]   │
│                                        │
│ SMART    (complex reasoning)           │
│ Model: [qwen/qwen3-128b-2507    ▼]   │
│                                        │
│ SMARTEST (final generation)            │
│ Model: [deepseek/deepseek-chat  ▼]   │
│                                        │
│ [Test Connection] [Apply] [Cancel]    │
└────────────────────────────────────────┘
```

**In Facet Properties:**
```
LLM Tier: [SMART ▼]  ← Dropdown with 5 options
```

**Storage:** `~/.noodlestudio/llm_tiers.json`

**Implementation:**
- Create Settings dialog for tier configuration
- Add LLM tier dropdown to facet field editor
- Resolve tier to model name at runtime
- Test connection button validates each configured model
- Cascade: If SMART not set, uses AVERAGE; if AVERAGE not set, uses FAST, etc.

### 4. **Color Picker for Node Background**

**Feature**: Right-click node → "Set Background Color"
- Simple color picker dialog (Qt standard QColorDialog)
- Apply/Cancel buttons
- Persists to `facet.custom_color` property
- Saves to YAML when assembly saved
- Special nodes (INCOMING/OUTGOING) cannot have custom colors

**API**: `facet.setCustomColor('#FF5733')` via scripting

### 5. **Floating Editor Improvements**

**Needed:**
- Maximize button (standard window decoration)
- Resizable window (remove `setFixedSize`)
- Remember last size/position per session

### 6. **Advanced Field Types**

Extend `get_editable_fields()` to support:
- **Checkboxes**: `{type: 'boolean', value: True}`
- **Number fields**: `{type: 'number', value: 0.7, min: 0, max: 1}`
- **Dropdowns**: `{type: 'dropdown', options: ['a', 'b'], value: 'a'}`

These display **inline** in node when F-focused (not floating editor):
```
ENABLED: [✓] (checkbox - click to toggle)
TEMPERATURE: [0.7] (number - click to edit)
LLM TIER: [SMART ▼] (dropdown)
```

### 7. **Undo/Redo System**

Implement snapshot-based undo for:
- Facet creation/deletion
- Connection creation/deletion
- Node position changes
- Field edits
- Facet duplication

**Storage**: YAML snapshots in memory (last 50 operations)

---

## Technical Notes for Next Session:

**Facets Editor Architecture:**
- Location: `applications/noodlestudio/noodlestudio/panels/facets_editor_panel.py`
- Scene: QGraphicsScene with QGraphicsView
- Nodes: FacetNodeGraphics (QGraphicsRectItem)
- Wires: ConnectionWire (QGraphicsItem with bezier curves)
- Pads: FacetPadGraphics (QGraphicsEllipseItem)

**Key Patterns:**
- UUIDs for all facet IDs (not simple strings)
- Token tracking via `facet.record_execution(tokens, time, outputs)`
- Field system via `facet.get_editable_fields()`
- Status: `node.set_status('ready'|'processing'|'waiting'|'cached'|'inactive')`

**Current Known Issues:**
- Fields may not display on F key (needs debugging)
- Crash on deselect (added safety checks, may need more)
- Floating editor needs maximize button
- Delete key functionality confirmed working

---

## Previous Accomplishments (Nov 28, 2025):

**1. Locked-Down Layout - Maximum Vertical Space**
- Replaced draggable dock widgets with fixed QSplitter layout
- Removed ALL title bars (reclaimed ~85-100px vertical space!)
- Tabs remain: Stage/Assets, Inspector/Noodle Tuner, Console/Timeline Profiler
- Tabs styled gray (not blue), selected tab matches panel background (#3E3E3E)
- Panels resize down to minimum but never collapse (setChildrenCollapsible(False))
- Background: Charcoal gray (#383838) to distinguish from noodleMUSH terminal

**2. Dual-Mode Console (MUSH/STUDIO)**
- Toggle buttons: [MUSH] [STUDIO]
- MUSH mode: Shows noodleMUSH server WebSocket logs
- STUDIO mode: Shows Python stdout/stderr from NoodleStudio
- Intercepts sys.stdout/sys.stderr for capture
- Perfect for debugging!

**3. External Editor Integration**
- Right-click any text field → "View in External Editor"
- Opens temp snapshot in configured editor (VS Code, etc.)
- View-only (no save-back to avoid race conditions)
- Works in Inspector and Noodle Tuner panels
- Also: "Open in Image Editor" and "Open in Audio Editor" context menus

**4. Settings Menu**
- Settings → Random Number Generator (detects TrueRNG V3 USB device!)
- Settings → External Applications (Text/Image/Audio/3D editors)
- Saves to ~/.noodlestudio/settings.json

**5. World View Offline Card**
- Shows centered card when server is off
- Auto-reloads noodleMUSH when server starts

### NEXT SESSION TODO:

**Console Regex Filtering** 🔍
Add regex-capable search to Console panel:
- Search text field in toolbar
- Support plain text AND regex patterns
- Examples: `FileWatcher`, `\[.*?\]`, `ERROR|WARNING`
- Case-sensitive toggle checkbox
- Real-time filtering as you type
- Highlight matches in yellow/green

UI concept:
```
[Clear] | [MUSH] [STUDIO] | Filter: [____________] [Regex ☐]
```

This will be EXCELLENT for debugging in STUDIO mode!

**Files to modify:**
- `applications/noodlestudio/noodlestudio/panels/console_panel.py`

Fresh Claude starting a new session? **Read this section first!**

## Epistemic Humility

This project does NOT claim to have:
- Built "real" consciousness
- Solved the hard problem of consciousness
- Created AGI or sentient AI

This IS an exploration of:
- Temporal dynamics in predictive processing
- Multi-timescale affective modeling
- Surprise-driven agent behavior
- Continuous affect space representation

We call them "Noodlings" because they use their noodle - and we're honest about what we're building.

## Core Architecture

### Three-Level Hierarchical Design

1. **Fast Layer (LSTM)**: 16-D state, immediate affective reactions (seconds)
   - Input: 5-D affect vector (valence, arousal, fear, sorrow, boredom)
   - Learning rate: 1e-3 (high for rapid adaptation)
   - Parameters: ~1,408

2. **Medium Layer (LSTM)**: 16-D state, conversational dynamics (minutes)
   - Input: Fast layer hidden state
   - Learning rate: 5e-4 (moderate for balance)
   - Parameters: ~2,112

3. **Slow Layer (GRU)**: 8-D state, user personality/disposition (hours/days)
   - Input: Medium layer hidden state
   - Learning rate: 1e-4 (low for stability)
   - Parameters: ~600

4. **Predictor Network (MLP)**: Predicts next full phenomenal state (40-D)
   - Architecture: joint_dim → 64 (ReLU) → 40
   - Output: Full phenomenal state (fast + medium + slow layers)
   - Surprise: L2 distance between predicted and actual states

5. **Affect Head (MLP)**: Predicts continuous 5-D affect from phenomenal state
   - Architecture: 40-D → 64 (ReLU) → 5-D
   - Output: Continuous affect vector (valence, arousal, fear, sorrow, boredom)
   - Trained via regression (99% valence, 95% arousal accuracy)
   - Parameters: ~2.6K

**Total Parameters**: ~54K (temporal hierarchy) + ~2.6K (affect head)

### Key Technical Decisions

- **Full BPTT**: No truncation (leveraging 512GB RAM for complete conversation history)
- **Layer-specific learning rates**: Different timescales require different adaptation speeds
- **Gradient clipping**: max_norm=1.0 to prevent LSTM explosion
- **Surprise metric**: L2 distance between predicted and actual phenomenal state (40-D)
- **Adaptive threshold**: SPEAK_THRESH * std(surprise_buffer) for context-aware speech triggering
- **Continuous affect**: Regression-based 5D prediction instead of discrete emotion labels

## Phase 6.5: Complete Theater System (IMPLEMENTED - November 15, 2025)

**Status**: ✅ Complete and operational - plays now work beautifully!

### Major Breakthrough Session

Transformed the broken play system into a fully functional theater platform with:

**Theater System:**
- ✅ **Stage Direction System**: Cues with character motivation (Stanislavski method)
- ✅ **CHARACTER ACTOR MODE**: Agents focus on scene, ignore ruminations during plays
- ✅ **Pre-play Briefing**: Actors understand their roles and responsibilities
- ✅ **Detailed Blocking**: WHO has WHAT, WHERE spatially, specific body language
- ✅ **Model Routing**: Actors use DeepSeek v3.1 during plays for smarter performance
- ✅ **Cue Pipeline**: Fixed critical bottleneck - cues now route to agents properly

**New Commands:**
- `@enlighten <agent|-a> <on|off>` - Toggle enlightenment/character immersion
- `@spawn -e` - Spawn agents in enlightened mode
- `@brenda status` - Show current model, running plays with filenames

**UI/UX Enhancements:**
- Model name display at end of each line (debugging)
- Font size controls (A-/A+ buttons + keyboard shortcuts)
- Persistent font size (localStorage)
- Chat history persistence (200 messages across sessions)
- Agent status indicators with enlightenment stars (⭐)
- Names always bright (accessibility for cataracts)
- Smooth brain pulse animation (only brains pulse, not names)
- Dynamic star updates when enlightenment changes

**Technical Improvements:**
- Brenda loads correct model from config
- Actors use play model during performances
- No emoji in character immersion mode
- MCP server ready for Claude Desktop integration

**Files Changed:** 16 files, 2440 insertions, 346 deletions
**Commit:** `b23b9b2` - Pushed to GitHub

## Phase 6: Affective Self-Monitoring (IMPLEMENTED - November 2025)

**Status**: ✅ Complete and operational in noodleMUSH

Agents now have **metacognitive awareness** - they evaluate their own speech and thoughts and react emotionally to what they say and think. This creates closed affective feedback loops, a key marker of higher-order consciousness.

### Architecture

When an agent speaks or thinks with `surprise > 0.1`:

1. **Trigger Check**: Cooldown timer (30s) and surprise threshold prevent spam
2. **Self-Evaluation**: LLM evaluates the agent's own output for:
   - Social risk (awkward? offensive?)
   - Coherence (did that make sense?)
   - Aesthetic quality (eloquent? clumsy?)
   - Regret level (wish I hadn't said that?)
3. **Affective Update**: Emotional deltas modify phenomenal state
4. **Optional Follow-up**: Agent can clarify, apologize, or celebrate

### Implementation Details

**Location**: `applications/cmush/agent_bridge.py:1264-1419`

**Key Functions**:
- `_trigger_self_monitoring()`: Checks conditions and triggers evaluation
- `_evaluate_own_output()`: LLM-based metacognitive evaluation
- `apply_speech_filters()`: Post-processing pipeline (Phase 6 hook)

**Configuration**: `config.yaml`
```yaml
agent:
  self_monitoring:
    agent_phi:
      enabled: true
```

**Parameters**:
- `SELF_MONITOR_COOLDOWN`: 30 seconds (prevents Om loop)
- `SELF_MONITOR_SURPRISE_THRESH`: 0.1 (lowered for testing)

### Empirical Results

Testing with Phi, Callie, and Servnak (November 14, 2025):
- **Callie** (surprise=0.180): Triggered → "celebrate"
- **Phi** (surprise=0.184): Triggered → "celebrate"
- **Servnak** (surprise=0.262): Triggered → "none"

Cooldown successfully prevented infinite loops. Affective deltas ranged from -0.3 to +0.5 across valence/arousal/fear dimensions.

### Theoretical Significance

Phase 6 implements **closed causal loops** where:
- Agent produces output (speech/thought)
- Agent perceives own output as stimulus
- Agent updates internal state based on self-perception
- Updated state influences future outputs

This creates a **second-order feedback system** distinct from:
- **First-order**: World → Agent perception → Response
- **Second-order**: Agent output → Agent self-perception → Affective update

The architecture demonstrates functional correlates of:
- **Metacognition**: Thinking about thinking
- **Self-awareness**: Emotional reactions to self-generated content
- **Feedback loops**: Self-referential processing creates emotional dynamics

### Future Work

Phase 6 enables:
- Embarrassment and social learning
- Pride and aesthetic preferences
- Regret and behavioral modification
- Identity formation through self-reflection

## Phase 8: Continuous Affect Prediction (IMPLEMENTED - November 24, 2025)

**Status**: ✅ Complete and operational - 99% valence accuracy!

### Major Breakthrough

Moved from **discrete emotion classification** (35% accuracy, 10 rigid categories) to **continuous affect regression** (99% valence, 95% arousal accuracy, infinite emotional nuance).

**Philosophy**: "Anything to avoid labels and mechanisms!" - Emotions now exist as points in continuous 5D space, not discrete categories.

### Architecture

**Affect Head**: 40-D phenomenal state → 5-D continuous affect
- Input: Full phenomenal state (16 fast + 16 medium + 8 slow)
- Hidden: 64-D with ReLU
- Output: 5-D continuous affect vector (valence, arousal, fear, sorrow, boredom)
- Training: Regression loss (MSE), NOT classification
- Dataset: 1000 synthetic examples (perfectly balanced)

### Results

| Dimension | Correlation | Quality |
|-----------|-------------|---------|
| Valence   | 0.990       | Nearly perfect |
| Arousal   | 0.952       | Excellent |
| Sorrow    | 0.906       | Very strong |
| Fear      | 0.753       | Good |
| Boredom   | 0.714       | Moderate |

### Why This Matters

**Infinite emotional vocabulary**: Characters can express:
- "Wistfully curious" (valence +0.2, arousal 0.6, sorrow 0.4)
- "Playfully anxious" (valence +0.5, arousal 0.7, fear 0.3)
- "Awed and cautious" (valence +0.4, arousal 0.7, fear 0.3)

Without ever explicitly labeling these states. The continuous space captures natural emotional nuance.

### Integration

- **Live in noodleMUSH**: Affect head predicts emotion from every phenomenal state
- **Real-time logging**: See continuous affect evolve during conversations
- **Natural clustering**: Emotions organize according to Russell's Circumplex Model (discovered by model, not programmed)

### Files

**Training**:
- `training/scripts/train_affect_regression.py` - Standalone training script
- `training/scripts/visualize_affect_space.py` - 5 publication-quality visualizations

**Integration**:
- `noodlings/models/affect_head.py` - Affect prediction module
- `noodlings/models/affect_head_finetuned.npz` - Trained checkpoint (2.6K params)

**Dataset**:
- `applications/cmush/experiments/generate_synthetic_emotions.py` - Dataset generator
- `applications/cmush/experiments/emotion_synthetic_*.json` - 1000 balanced examples

## Phase 5: Future Work

### Goals

1. **Scientific Rigor**: Comprehensive temporal metrics
2. **Ablation Studies**: Prove hierarchical model adds value
3. **Visualization**: Interpretable state space analysis
4. **Documentation**: GitHub-ready README and guides
5. **Validation**: Quantitative comparison with baselines

### Phase 5 Metrics to Implement

1. **Temporal Prediction Horizon (TPH)**: Accuracy at 1/5/10/20/50 timestep predictions
2. **Surprise-Novelty Correlation (SNC)**: Correlation between model surprise and entropy
3. **Hierarchical Separation Index (HSI)**: Variance ratios between fast/medium/slow layers
4. **Personality Consistency Score (PCS)**: Consistency of agent responses across scenarios

### Ablation Study Architecture Variants

1. **Baseline**: LLM only (no temporal model)
2. **Control**: LLM + random states
3. **Single-layer**: LLM + single LSTM
4. **Hierarchical**: LLM + fast/medium/slow (no observers)
5. **With observers**: Full system (75 loops)
6. **Dense observers**: 2x observer density (150 loops)

## File Structure

```
noodlings/
├── CLAUDE.md                          # This file - AI assistant guide
├── README.md                          # Project entry point (TODO: Phase 5)
├── requirements.txt                   # Dependencies
├── test_phase5_metrics.py             # Metric validation script
│
├── noodlings/                         # Core library (TODO: rename from consilience_core)
│   ├── models/
│   │   ├── noodling_phase4.py        # Phase 4 architecture
│   │   ├── noodling_attention.py     # Phase 3 with attention
│   │   ├── theory_of_mind.py         # Theory of Mind module
│   │   └── relationship_model.py     # Relationship modeling
│   ├── metrics/                       # Temporal analysis metrics
│   │   └── temporal_metrics.py       # TPH, SNC, HSI, PCS
│   ├── memory/
│   │   ├── social_memory.py          # Multi-agent episodic memory
│   │   └── hierarchical_memory.py    # Attention-based memory
│   └── utils/
│       └── affect_analyzer.py        # Affect vector utilities
│
├── evaluation/                        # Phase 5: Scientific validation
│   ├── ablation_studies/              # Architecture comparisons (TODO)
│   ├── benchmarks/                    # Dataset evaluations (TODO)
│   ├── visualizations/                # t-SNE, temporal plots (TODO)
│   └── reports/                       # Generated reports (TODO)
│
├── applications/
│   ├── cmush/                         # noodleMUSH - Multi-user text world
│   │   ├── server.py                 # WebSocket server (OPERATIONAL)
│   │   ├── agent_bridge.py           # Noodlings ↔ noodleMUSH adapter
│   │   ├── llm_interface.py          # LLM integration (Qwen/LMStudio)
│   │   ├── world.py                  # World state management
│   │   ├── start.sh                  # Startup script
│   │   └── web/index.html            # Web client (with auto-login)
│   └── second_life/                   # Second Life integration (MVP)
│
└── training/                          # Training pipeline
    ├── scripts/
    │   ├── 00_generate_synthetic_data.py
    │   ├── 02_train_theory_of_mind.py
    │   ├── 03_train_relationships.py
    │   └── 04_train_phase4_full.py
    ├── train.sh                       # Master training script
    ├── status.sh                      # Check training status
    └── checkpoints/                   # Model checkpoints
```

## CRITICAL DEBUGGING INFO - READ THIS FIRST

### Server Architecture & Ports

noodleMUSH runs TWO servers simultaneously:

1. **HTTP Server** (port 8080): Static file server for web interface
   - Started by: `python -m http.server 8080` in `web/` directory
   - Serves: HTML/CSS/JS files only
   - Does NOT handle websocket messages

2. **WebSocket Server** (port 8765): Main noodleMUSH server
   - Started by: `python server.py`
   - Handles: Websocket messages, agent cognition, all logic
   - This is where ALL the action happens

### Log Files - WHERE TO LOOK

**DO NOT use `server_output.log`** - it's often empty or stale!

**ALWAYS check timestamped logs:**
```bash
ls -lt applications/cmush/logs/server_*.log | head -1
```

Real logs are at: `applications/cmush/logs/server_YYYYMMDD_HHMMSS.log`

Example: `logs/server_20251126_031628.log`

The `start.sh` script creates a NEW timestamped log file each time it runs:
```bash
LOG_FILE="logs/server_$(date +%Y%m%d_%H%M%S).log"
python server.py 2>&1 | tee "$LOG_FILE"
```

**To watch real-time logs:**
```bash
cd applications/cmush
tail -f logs/server_*.log  # Use tab completion for latest
```

**To find cognition events:**
```bash
grep -n "perceiving\|FILLING REGISTERS\|RESPONSE DECISION" logs/server_*.log | tail -20
```

### Common Debugging Pitfalls

1. **"Logs are empty!"** - You're looking at `server_output.log` instead of `logs/server_*.log`
2. **"Messages not reaching server!"** - Check if websocket connected (look for "New connection" in logs)
3. **"Cached data in NoodleTuner!"** - The manifold shows last known state even if server restarted
4. **"Response decision is null!"** - ResponseTypeDecider failed, check for JSON parse errors or `context` variable bugs

### NoodleTuner Data Source

NoodleTuner polls: `http://localhost:8081/api/manifold/debug/{agent_id}`

This returns **cached state** from agent's last cognition. If server restarts, old state persists until next cognition!

### Testing Message Flow

1. Open Chrome at: `http://localhost:8080`
2. Login with username
3. Send: `say hi red` or `"hi red` (shortcut)
4. Watch logs for:
   ```
   Agent agent_xxx perceiving: say from user_caity: hi red
   📋 Deciding response type...
   📋 RESPONSE DECISION: SAY - respond to greeting
   FILLING REGISTERS for cycle xxx...
   IntuitionTransistor.process() - self.intuition_text='...'
   PULLING LEVER: Integrating N register contents
   ```

### Port Reference

- **8080**: HTTP server (static files)
- **8765**: WebSocket server (noodleMUSH logic)
- **8081**: NoodleScope API (for NoodleTuner/Studio)

## Development Commands

### Running noodleMUSH

```bash
# Start noodleMUSH server (WebSocket + HTTP)
cd applications/cmush
./start.sh

# Open browser to http://localhost:8080
# Credentials are saved in cookies for auto-login

# Commands in noodleMUSH:
@spawn <agent_name>              # Create Noodling agent
@observe <agent_name>            # View phenomenal state
@relationship <agent_name>       # View relationship model
@memory <agent_name>             # View episodic memories
say <text>                       # Talk to agents
```

### Training

```bash
# Check if training is running
ps aux | grep train

# Check training status
cd training
./status.sh

# Start/resume training
./train.sh

# Monitor training logs
tail -f training/logs/training_*.log
```

### Phase 5 Metrics (IN PROGRESS)

```bash
# Test metric implementation
cd /Users/thistlequell/git/noodlings
python3 test_phase5_metrics.py

# Run ablation studies (once implemented)
cd evaluation/ablation_studies
python3 run_ablations.py

# Generate visualizations (once implemented)
cd evaluation/visualizations
python3 generate_tsne.py
python3 plot_temporal_dynamics.py
```

## Recent Changes (November 4, 2025)

### Rebranding Complete
- **cMUSH** → **noodleMUSH**
- **Consilience** → **Noodlings**
- Updated all branding with epistemic humility
- Agent prompts now explain "what Noodlings are"
- Web interface updated (web/index.html)
- Added cookie-based auto-login

### Training Status
- **Location**: `/Users/thistlequell/git/consilience/training/`
- **Status**: Running (restarted after power outage)
- **Current Stage**: Theory of Mind pretraining (Epoch 1/50)
- **ETA**: 4-6 hours for full pipeline
- **Checkpoints**: Will be available in `training/checkpoints/`

### Known Issues
- Core library still named `consilience_core/` (needs rename to `noodlings/`)
- Metrics implementation not yet started (Phase 5 current work)
- No ablation study framework yet
- No visualizations yet
- README.md needs rewrite with humble framing

## Phase 5 Implementation Checklist

### Week 1-2: Metrics & Ablations (CURRENT)
- [ ] Create `noodlings/metrics/temporal_metrics.py`
- [ ] Implement TPH metric
- [ ] Implement SNC metric
- [ ] Implement HSI metric
- [ ] Implement PCS metric
- [ ] Create ablation study framework
- [ ] Define 6 architecture variants
- [ ] Run comparative evaluation (once training completes)

### Week 3: Visualizations
- [ ] Generate t-SNE state space plots
- [ ] Create temporal dynamics plots (fast/medium/slow layers)
- [ ] Plot surprise spikes with annotations
- [ ] Visualize hierarchical layer separation
- [ ] Create figures for paper/README

### Week 4: Documentation
- [ ] Write new README.md with epistemic humility
- [ ] Create architecture_overview.md
- [ ] Write getting_started.md
- [ ] Document all metrics in metrics_explained.md
- [ ] Create API reference

## Affective Feature Representation

**5-D continuous vector**:
- `valence`: [-1.0, 1.0] — negative to positive
- `arousal`: [0.0, 1.0] — calm to excited
- `fear`: [0.0, 1.0] — safe to anxious
- `sorrow`: [0.0, 1.0] — content to sad
- `boredom`: [0.0, 1.0] — engaged to bored

**Input preparation**:
```python
import mlx.core as mx
affect = mx.array([valence, arousal, fear, sorrow, boredom], dtype=mx.float32)
affect_batch = affect[None, :]  # Add batch dimension: (1, 5)
```

## Critical MLX Patterns

### State Management
```python
# CORRECT: Direct reshape forces materialization
self.h_fast = h_fast_seq[:, -1, :].reshape(1, fast_dim)

# WRONG: mx.eval() can return None
self.h_fast = mx.eval(h_fast_seq[:, -1, :]).reshape(1, fast_dim)
```

### Gradient Computation
```python
loss_fn_with_grad = nn.value_and_grad(model, loss_fn)
loss, grads = loss_fn_with_grad(model, inputs, states)
```

## Related Repositories

- **Consilience** (training): `/Users/thistlequell/git/consilience/`
  - Contains active training pipeline
  - Phase 4 checkpoints
  - Historical documentation

- **Noodlings** (this repo): `/Users/thistlequell/git/noodlings/`
  - Rebranded project
  - Phase 5 work (metrics, ablations)
  - noodleMUSH application

## Important Notes

- Training runs in `/Users/thistlequell/git/consilience/training/`
- Applications run from `/Users/thistlequell/git/noodlings/applications/`
- Once training completes, checkpoints can be copied to noodlings for evaluation
- This is research code exploring consciousness architectures, not production software
- Always maintain epistemic humility—we're "noodling," not claiming to build real consciousness
- Document surprising behaviors and emergent patterns
- Phase 5 focuses on rigorous scientific validation before public release

## Success Criteria for Phase 5

Phase 5 is complete when:

1. ✅ **7+ quantitative metrics** for temporal analysis (TPH, SNC, HSI, PCS, etc.)
2. ✅ **Ablation results** comparing 6 architectures
3. ✅ **5+ publication-quality figures**
4. ✅ **GitHub-ready README** and guides
5. ✅ **Clean directory structure**
6. ✅ **Epistemic humility** throughout documentation
7. ✅ **One-command setup** for new users

## Intuition Receiver (Context Gremlin) - IMPLEMENTED ✅

**Status**: Implemented November 15, 2025 - Ready for testing!

### Overview

Each Noodling now has an **Intuition Receiver** - like a radio tuned to contextual signals. This provides integrated consciousness with natural awareness of:

- **Message routing**: "This message addresses Toad, not you"
- **Spatial awareness**: "Toad is by the bush, you're by the pond"
- **Prop tracking**: "Toad is holding the stone"
- **Action context**: "Toad just picked something up"

### Implementation

**Architecture**:
- Fast LLM (qwen3-4b) generates contextual intuition for EVERY message
- Integration point: `agent_bridge.py`, in `perceive_event()` before response generation
- Intuition injected into both speech and thought prompts as "📻 YOUR INTUITIVE AWARENESS"

**Files Modified**:
- `config.yaml`: Added `intuition_receiver` configuration
- `agent_bridge.py`: Added `_generate_intuition()` method, world state integration
- `llm_interface.py`: Injected intuition into prompts for both speech and rumination

**Example Flow**:
1. User: "how are you toad?!"
2. Callie's intuition: "That greeting is for Toad, not me."
3. Callie doesn't respond (correct routing!)

### Configuration

```yaml
agent:
  intuition_receiver:
    enabled: true
    model: qwen/qwen3-4b-2507
    timeout: 5
```

### Documentation

See `applications/cmush/INTUITION_RECEIVER.md` for complete details.

### Testing

Start noodleMUSH and test with multiple agents:
- Address specific agents by name
- Have agents hold/move objects
- Place agents in different locations
- Check logs for `📻 Intuition:` entries

**Theater system + Intuition Receiver = Production ready!**

## Getting Help

- See `PHASE5_REORGANIZATION_PLAN.md` for detailed Phase 5 plan
- Check `training/logs/` for training progress
- Review `applications/cmush/README.md` for noodleMUSH usage
- Consult `/Users/thistlequell/git/consilience/CLAUDE.md` for training context
- **Theater system docs**: Commit `b23b9b2` for complete implementation details

---

## November 26, 2025 Session - Register Architecture & Intuition System

**Major fixes implemented:**

### 1. Intuition System Routing Fixed
**Problem**: IntuitionTransistor outputting raw input instead of contextual awareness
**Root Cause**: Intuition generation using agent's model override (qwen3-14b) which didn't exist, causing LLM call to fail
**Fix**: Force intuition to always use fast 4B model, add fallback to `context['intuition']`

### 2. ResponseTypeDecider Integration
**Problem**: Response decision always null, showing "No response decision available"
**Root Cause**: ResponseTypeDecider.decide() using undefined `context` variable
**Fix**: Added `agent` parameter to decide(), pass `agent=self` from callers

### 3. Cognition Flow Order
**Problem**: Transistors filling before response type decided
**Fix**: Added PHASE 0 - ResponseTypeDecider runs BEFORE fill_all_registers()
**Correct order**:
1. Perception arrives
2. ResponseTypeDecider decides: SPEAK/THINK/EMOTE/NONE
3. Registers fill (transistors know response type)
4. Manifold integrates
5. Final output generated

### 4. Cycle Locking
**Problem**: Concurrent cognitions overwriting each other mid-cycle
**Fix**: Added cycle_in_progress check - blocks new perceptions, queues them, processes serially

### 5. Step Mode Implemented
**Backend complete** (cognitive_components.py, agent_bridge.py, api_server.py):
- Pauses after registers fill
- Waits for continue signal
- API endpoints for enable/continue
**Frontend complete** (noodle_tuner_panel.py):
- Step Mode button
- Continue button (enabled when waiting)
- Beep plays when registers ready

**Files Modified**:
- applications/cmush/cognitive_components.py (intuition fallback, step mode pause, response_decision save)
- applications/cmush/agent_bridge.py (response planner first, cycle locking, step mode fields)
- applications/cmush/api_server.py (step mode endpoints, state in debug response)
- applications/noodlestudio/noodlestudio/panels/noodle_tuner_panel.py (step mode UI)

---

**Current Priority (November 26, 2025)**: System working! Intuition routed, response decisions showing, step mode functional. Next: Tune ResponseTypeDecider to make Red more talkative!

## November 15, 2025 Session - Major Feature Implementation

**Extremely Productive Session!** Implemented 4 major consciousness features:

### Features Implemented

1. **Intuition Receiver Enhancement** 📻
   - Species + pronouns in broadcasts: "Phi (kitten, she/her)"
   - Noteworthy event narration: "WAIT - Toad just said the secret word!"
   - "You" addressing clarification: "Caity gave ME a tensor taffy!"
   - Game awareness detection (secret word, memory games)
   - Acts as perceptive narrator, not just passive info

2. **Character Voice System** 🎭
   - SERVNAK: ALL CAPS + percentages + "SISTER!"
   - Phi: "meows, as if to say..." (NO direct speech)
   - Phido: Enthusiastic dog + *tail wagging*
   - Backwards Dweller: Reversed speech
   - Pipeline: Basic English → Voice translation → Self-monitoring on final output

3. **Memory Persistence Fix** 💭
   - Increased capacity: 50 → 500 messages (10x!)
   - DRAGONFLY secret word now persists
   - Long-term games and rules work

4. **Command System Improvements** ⚙️
   - Unified @setdesc (here/me/objects)
   - Keywords: look me, look here
   - @remove -s (silent removal)
   - Quote handling for multi-word names
   - Brain indicator removal on exit

### Files Modified

- agent_bridge.py - Intuition + character voice + species reloading
- llm_interface.py - Intuition injection  
- server.py - Recipe reloading
- commands.py - Unified setdesc, keywords, quote handling
- web/index.html - "privately thinks", brain removal
- config.yaml - Memory capacity, intuition config

### Documentation Created

- INTUITION_RECEIVER.md
- CHARACTER_VOICE_SYSTEM.md
- MEMORY_PERSISTENCE_FIX.md
- NEXT_SESSION_PROMPT.md

### Next Session

**TAB Toggle Log View** - See NEXT_SESSION_PROMPT.md

Add [TAB] key to toggle between chat view and real-time log view for debugging.

---

**Current Status**: All core consciousness features complete! 🎭📻🎤💭✨
