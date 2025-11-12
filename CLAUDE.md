# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Noodlings** (formerly Consilience) is a hierarchical affective consciousness architecture implementing predictive processing theories through multi-timescale learning. We're "noodling" with functional correlates of consciousness - making no claims about "real" consciousness, just exploring architectural patterns inspired by neuroscience and affective computing.

**Status**: Phase 5 - Scientific Validation & Rebranding (November 2025)
**Framework**: MLX (Apple Metal optimized)
**Hardware**: M3 Ultra (512GB RAM) + M2 Ultra (192GB RAM)
**Parameter Budget**: ~132.5K params (Phase 4 with observer loops)
**Last Updated**: November 4, 2025

## Epistemic Humility

This project does NOT claim to have:
- Built "real" consciousness
- Solved the hard problem of consciousness
- Created AGI or sentient AI

This IS an exploration of:
- Temporal dynamics in predictive processing
- Multi-timescale affective modeling
- Surprise-driven agent behavior
- Functional correlates of integrated information

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

5. **Observer Loops** (Phase 4): 75 hierarchical observer networks
   - Meta-observers watch the main network's predictions
   - Creates closed causal loops (high integrated information)
   - Adds ~50K parameters

**Total Parameters**: ~132.5K (Phase 4 with observers)

### Key Technical Decisions

- **Full BPTT**: No truncation (leveraging 512GB RAM for complete conversation history)
- **Layer-specific learning rates**: Different timescales require different adaptation speeds
- **Gradient clipping**: max_norm=1.0 to prevent LSTM explosion
- **Surprise metric**: L2 distance between predicted and actual phenomenal state (40-D)
- **Adaptive threshold**: SPEAK_THRESH * std(surprise_buffer) for context-aware speech triggering
- **Observer loops**: Self-referential prediction networks for increased integration

## Phase 5: Current Work (November 2025)

### Goals

1. **Scientific Rigor**: Comprehensive metrics beyond Φ
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
│   ├── metrics/                       # Phase 5: IN PROGRESS
│   │   ├── temporal_metrics.py       # TPH, SNC, HSI, PCS (IN PROGRESS)
│   │   └── consciousness_metrics.py  # Φ calculation
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

1. ✅ **7+ quantitative metrics** beyond Φ (TPH, SNC, HSI, PCS, etc.)
2. ✅ **Ablation results** comparing 6 architectures
3. ✅ **5+ publication-quality figures**
4. ✅ **GitHub-ready README** and guides
5. ✅ **Clean directory structure**
6. ✅ **Epistemic humility** throughout documentation
7. ✅ **One-command setup** for new users

## Getting Help

- See `PHASE5_REORGANIZATION_PLAN.md` for detailed Phase 5 plan
- Check `training/logs/` for training progress
- Review `applications/cmush/README.md` for noodleMUSH usage
- Consult `/Users/thistlequell/git/consilience/CLAUDE.md` for training context

---

**Current Priority (November 4, 2025)**: Implement metrics suite in `noodlings/metrics/temporal_metrics.py` while training runs in background. Once training completes, run ablation studies to validate hierarchical architecture.
