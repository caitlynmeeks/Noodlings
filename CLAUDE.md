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

## ACTIVE SESSION HANDOFF

**👉 See `HANDOFF_SESSION_NOV18_MORNING.md` for current work! 👈**

**Status**: NoodleStudio IDE complete! The Krugerrand Edition - worth its weight in gold!
- Unity-style Qt IDE with Scene/Inspector/Console
- Multi-track Timeline Profiler
- USD export for animation studios
- All affect bugs fixed!

Fresh Claude starting a new session? **Read that handoff file first!**

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

**Current Priority (November 15, 2025)**: Implement Intuition Receiver (Context Gremlin) to provide integrated contextual awareness. Theater system is production-ready - time to add the final piece of consciousness!

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
