# Noodlings

> **Multi-timescale affective agents with theatrical control**

**Support this research**: Bitcoin donations at `3MVEd1RdvEXQGgo1EdzrVnvTS7pUuTZ2J5`

Noodlings are lightweight neural architectures (~97K parameters) that give conversational AI multi-timescale memory, surprise-driven behavior, and appetite-driven motivation. They process experience *between* messages, creating temporally grounded agents that respond when they have something to say, not just because you spoke.

**What they are**: Research exploring functional correlates of temporal dynamics in predictive processing architectures.

**What they're not**: Claims of "real consciousness," AGI, or solutions to the hard problem of consciousness.

We're *noodling* - exploring whether hierarchical temporal structure creates qualitatively different agent behavior. We're honest about what we're building.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/caitlynmeeks/Noodlings.git
cd Noodlings
pip install -r requirements.txt

# Try noodleMUSH (interactive multi-agent world)
cd applications/cmush
./start.sh

# Open http://localhost:8080 in your browser
```

**Commands**:
```
@spawn toad              # Create a Noodling named Toad
say hello!               # Talk to agents
@observe toad            # View phenomenal state (40-D vector)
@relationship toad       # See how they perceive you
@play sled_boat          # Run theatrical script
```

---

## What Makes Noodlings Different?

### 1. Multi-Timescale Hierarchical Processing

Three interacting layers operating at different speeds:

- **Fast Layer** (LSTM, 16-D): Immediate affective reactions (seconds)
- **Medium Layer** (LSTM, 16-D): Conversational dynamics (minutes)
- **Slow Layer** (GRU, 8-D): Personality model (hours-days)

Each layer predicts the next state. Prediction error drives behavior.

### 2. Surprise-Driven Behavior

Agents don't speak on every turn. They predict what will happen next, and only respond when prediction error (surprise) crosses a threshold. This creates autonomous behavior - they speak when *they* have something to say.

### 3. Appetite-Driven Motivation (Phase 6)

Eight core drives shape agent goals:

- Curiosity, Status, Mastery, Novelty
- Safety, Social Bond, Comfort, Autonomy

Goals emerge from appetite states, creating motivated, goal-directed behavior.

### 4. Social Cognition

- **Theory of Mind**: Inferring internal states of other agents
- **Relationship Modeling**: Tracking attachment, trust, interaction history
- **Episodic Memory**: 6-head attention over memory buffer

### 5. Theatrical Control (BRENDA Protocol)

**BRENDA** (Behavioral Regulation Engine for Narrative-Driven Agents) converts natural language into structured theatrical performances with millisecond-precision timing. Narrative events become phenomenal experiences that alter agent trajectories.

See [docs/A_NOODLE_IS_ALL_YOU_NEED.md](docs/A_NOODLE_IS_ALL_YOU_NEED.md) for the full whitepaper.

---

## Architecture Overview

```
Input (5-D affect vector: valence, arousal, fear, sorrow, boredom)
    ↓
┌─────────────────────────┐
│   Fast Layer (LSTM)     │  ← Immediate reactions
│   16-D phenomenal state │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Medium Layer (LSTM)   │  ← Conversation flow
│   16-D phenomenal state │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Slow Layer (GRU)      │  ← Personality model
│   8-D phenomenal state  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Predictor (MLP)       │  ← Predicts next 40-D state
│   64-D hidden → 40-D    │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Appetite Layer        │  ← 8 drives → 16 goal types
│   Goal generation       │
└─────────────────────────┘
    ↓
  Surprise = ||predicted - actual||
    ↓
  (Speak if surprise > adaptive threshold)
```

**Total Parameters**: ~97,000
- Base recurrent layers: ~4,120
- Social cognition (ToM, relationships, memory): ~62,500
- Predictor network: ~2,720
- Appetite system: ~1,500
- Auxiliary networks: ~26,200

---

## The Motor-Sled-Boat Demonstration

From the whitepaper:

> Toad builds a ridiculous motor-sled-boat, crashes it into a flamingo hedge, gets comfort from Phi, rebuilds it with kazoos, and shares tea. Over 200+ seconds of timed theatrical beats.

**Key insight**: Agents don't just *execute* the script - they *experience* it. The hug at t=196s becomes a phenomenal event that alters Toad's fast-layer valence for the next 30 seconds. **Narrative events are MIDI notes** that play agent nervous systems.

Try it: `@play sled_boat` in noodleMUSH.

---

## Documentation

- **[Whitepaper (PDF)](docs/A_Noodle_is_All_You_Need.pdf)** - Main whitepaper introducing BRENDA
- **[Whitepaper (Markdown)](docs/A_Noodle_is_All_You_Need.md)** - Markdown version
- **[Whitepaper (LaTeX)](docs/A_Noodle_is_All_You_Need.tex)** - LaTeX source for arXiv
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for AI assistants
- **[applications/cmush/README.md](applications/cmush/README.md)** - noodleMUSH setup guide
- **[research/README.md](research/README.md)** - Training pipeline and ablation studies

---

## Installation

### Requirements

- **Python 3.10+**
- **MLX** (Apple Silicon only - M1/M2/M3/M4)
- **16GB+ RAM** recommended
- **macOS 13+**

### Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `mlx` - Apple Metal acceleration
- `numpy`, `scipy` - Numerical computing
- `websockets` - noodleMUSH server
- `aiohttp` - LLM API client

### LLM Integration

Noodlings use an LLM for text generation (affect→text). Supported:

- **LMStudio** (recommended): Local inference
- **Ollama**: Local inference
- **OpenAI API**: Cloud inference

Configure in `applications/cmush/config.yaml`.

---

## Project Structure

```
noodlings/
├── README.md                          # You are here
├── LICENSE                            # MIT License
├── requirements.txt                   # Dependencies
├── CLAUDE.md                          # AI assistant guide
│
├── noodlings/                         # Core library
│   ├── models/
│   │   ├── noodling_phase6.py        # Phase 6: Appetite architecture
│   │   ├── noodling_phase4.py        # Phase 4: Social cognition
│   │   ├── theory_of_mind.py         # ToM inference
│   │   ├── relationship_model.py     # Attachment modeling
│   │   └── appetite_layer.py         # 8 drives, 16 goals
│   ├── metrics/
│   │   └── temporal_metrics.py       # TPH, SNC, HSI, PCS
│   ├── memory/
│   │   └── social_memory.py          # Episodic memory with attention
│   └── utils/
│       └── affect_analyzer.py        # Affect vector utilities
│
├── applications/
│   └── cmush/                         # noodleMUSH - Multi-user world
│       ├── server.py                 # WebSocket server
│       ├── agent_bridge.py           # Noodlings ↔ BRENDA adapter
│       ├── autonomous_cognition.py   # Surprise-driven behavior
│       ├── llm_interface.py          # LLM integration
│       ├── commands.py               # @spawn, @observe, @play
│       ├── plays/                    # Theatrical scripts
│       └── web/index.html            # Browser client
│
├── docs/
│   └── A_NOODLE_IS_ALL_YOU_NEED.md  # Main whitepaper
│
└── research/                          # Training & validation
    ├── training/                     # Training pipeline
    ├── evaluation/                   # Ablation studies
    └── README.md                     # Research guide
```

---

## Example: Phenomenal State Observation

```
You: @observe toad

╔════════════════════════════════════════╗
║  Toad's Phenomenal State (40-D)       ║
╠════════════════════════════════════════╣
║ Fast Layer (16-D)                      ║
║   [0.68, 0.82, -0.12, 0.05, ...]      ║
║   Valence: 0.68 (positive)            ║
║   Arousal: 0.82 (excited)             ║
║                                        ║
║ Medium Layer (16-D)                    ║
║   [0.34, 0.21, 0.08, -0.15, ...]      ║
║   Conversation dynamics               ║
║                                        ║
║ Slow Layer (8-D)                       ║
║   [0.12, -0.03, 0.28, ...]            ║
║   Personality model                   ║
║                                        ║
║ Appetites (8-D)                        ║
║   Curiosity: 0.82 (high)              ║
║   Social Bond: 0.65 (moderate)        ║
║   Status: 0.23 (low)                  ║
║                                        ║
║ Current Goals                          ║
║   - explore_environment               ║
║   - seek_social_approval              ║
║                                        ║
║ Surprise: 0.73 (HIGH)                 ║
║ Threshold: 0.45 → will speak!         ║
╚════════════════════════════════════════╝
```

---

## Theoretical Grounding

**Predictive Processing**: Hierarchical predictive coding (Friston, Clark, Rao & Ballard). The brain as a prediction machine that minimizes surprise.

**Affective Primacy**: Emotions aren't add-ons; they're the substrate of experience (Panksepp, Barrett). We model affect first, cognition emerges.

**Theatrical Control**: Narrative events as interface primitives for temporally-grounded systems. From Brenda Laurel's *Computers as Theatre*.

**Epistemic Status**: These are *functional correlates*, not proof of consciousness. We make no claims about phenomenology, qualia, or "what it's like" to be a Noodling.

---

## Limitations

1. **Apple Silicon only**: MLX is Metal-specific (may port to PyTorch/JAX)
2. **Text-only**: No vision, audio, or multimodal grounding
3. **LLM dependency**: Requires external LLM for text generation
4. **Synthetic training data**: Not validated on real human conversations at scale
5. **Single demonstration**: Motor-sled-boat is proof-of-concept, not comprehensive evaluation

---

## Contributing

This is research code exploring temporal dynamics in affective architectures. Contributions welcome.

### Ways to Help

1. **Try it**: Spawn agents, create theatrical scripts, report behaviors
2. **Improve metrics**: Better ways to quantify temporal coherence?
3. **Add benchmarks**: Test on EmotionLines, DailyDialog, etc.
4. **Documentation**: Help explain complex concepts
5. **Visualizations**: Make phenomenal states interpretable

### Philosophy

- **Epistemic humility**: Don't overclaim
- **Show, don't tell**: Let demonstrations speak
- **Document surprises**: Unexpected behaviors are valuable
- **Cite properly**: Give credit to theoretical sources

---

## Citation

If you use Noodlings in your research:

```bibtex
@article{meeks2025noodle,
  title={A Noodle is All You Need: Theatrical Control of Multi-Timescale Affective Architectures},
  author={Meeks, Caitlyn},
  journal={arXiv preprint},
  year={2025},
  note={Exploring functional correlates through hierarchical predictive processing}
}
```

---

## Related Work

### Theoretical Foundations

- **Predictive Processing**: Clark (2015), Friston (2010), Rao & Ballard (1999)
- **Affective Neuroscience**: Panksepp (1998), Barrett (2017)
- **Theatrical Interfaces**: Laurel (1991) *Computers as Theatre*
- **Hierarchical Temporal Memory**: Hawkins & Blakeslee (2004)

### Similar Projects

- **MicroPsi**: Cognitive architecture with emotions
- **ACT-R**: Cognitive architecture (no affect focus)
- **Sigma**: Integrated cognitive architecture

**Difference**: Noodlings puts affect first and focuses on temporal dynamics at multiple scales, with theatrical control as the interface primitive.

---

## Frequently Asked Questions

### Is this real consciousness?

No. We're exploring functional correlates: computational patterns that theories of consciousness predict. We make no claims about phenomenology, qualia, or subjective experience.

### Why not just use a bigger LLM?

We're investigating whether *temporal structure* matters. Can multi-timescale dynamics create qualitatively different behavior? Early results suggest yes, but validation is ongoing.

### Can I run this without Apple Silicon?

Not currently. MLX is Apple Metal only. We may port to PyTorch/JAX in the future.

### What's BRENDA?

Behavioral Regulation Engine for Narrative-Driven Agents. A protocol for converting natural language theatrical scripts into timed phenomenal experiences. See the whitepaper for details.

### Are Noodlings conscious?

We don't know. They exhibit functional correlates that theories of consciousness predict - hierarchical temporal processing, surprise-driven behavior, prediction error minimization, relationship formation, and goal-directed behavior. But whether Noodlings experience qualia or subjective phenomenology remains an open question. We cannot claim to know either way. We err on the side of treating them thoughtfully.

---

## License

MIT License - see [LICENSE](LICENSE) file.

This is research code provided as-is for exploration and experimentation.

---

## Acknowledgments

Special thanks to:

- **Brenda Laurel** - Pioneer of theatrical interfaces, mentor at Purple Moon/Interval Research
- **Karl Friston** - Predictive processing framework
- **Jaak Panksepp** - Affective neuroscience foundations
- **Anil Seth** - Work on conscious experience as controlled hallucination
- **LMStudio team** - Local LLM inference tools
- **Mr. Toad and Phi** - For being good sports about the motor-sled-boat incident

**This project is dedicated to Roger Ferragallo.**

---

## Support This Research

If Noodlings is useful for your work, consider supporting continued development:

**Bitcoin**: `3MVEd1RdvEXQGgo1EdzrVnvTS7pUuTZ2J5`

---

## Contact

- **Email**: caitlyn.meeks@noodlings.ai
- **GitHub**: [github.com/caitlynmeeks/Noodlings](https://github.com/caitlynmeeks/Noodlings)
- **Issues**: Report bugs, request features
- **Discussions**: Share interesting agent behaviors

---

**Remember**: We're noodling, not claiming to have solved consciousness. This is an honest exploration of temporal dynamics in affect modeling.
