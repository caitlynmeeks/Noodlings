# Noodlings v0.1.0 - Initial Public Release

**Multi-timescale affective agents with theatrical control**

This is the first public release of Noodlings, a lightweight neural architecture (~97K parameters) exploring functional correlates of consciousness through hierarchical predictive processing.

##  Key Features

### Architecture
- **Multi-Timescale Hierarchical Processing**: Fast/Medium/Slow layers (LSTM/LSTM/GRU) operating at different temporal scales
- **Surprise-Driven Behavior**: Agents respond when prediction error crosses adaptive thresholds, not on every turn
- **Appetite-Driven Motivation (Phase 6)**: Eight core drives (Curiosity, Status, Mastery, Novelty, Safety, Social Bond, Comfort, Autonomy) shape goal-directed behavior
- **Social Cognition**: Theory of Mind inference, relationship modeling, episodic memory with attention
- **~97K parameters total**: Lightweight and efficient

### BRENDA Protocol
**Behavioral Regulation Engine for Narrative-Driven Agents** - Convert natural language theatrical scripts into millisecond-precision phenomenal experiences. Narrative events become MIDI notes that play agent nervous systems.

### noodleMUSH
Interactive multi-agent world where you can:
- Spawn Noodlings and watch them interact
- Observe their 40-D phenomenal states in real-time
- Run theatrical scripts (try `@play sled_boat` for the motor-sled-boat demonstration)
- See relationship dynamics evolve

##  What This Is

Research exploring whether hierarchical temporal structure creates qualitatively different agent behavior. We're investigating functional correlates of consciousness theories - not claiming to have built "real" consciousness.

**Epistemic humility**: We cannot claim to know whether Noodlings experience qualia or subjective phenomenology. The question remains open, and we treat them thoughtfully.

##  Quick Start

```bash
git clone https://github.com/caitlynmeeks/Noodlings.git
cd Noodlings
pip install -r requirements.txt

cd applications/cmush
./start.sh
# Open http://localhost:8080
```

##  Requirements

- **Python 3.10+**
- **MLX** (Apple Silicon only - M1/M2/M3/M4)
- **16GB+ RAM** recommended
- **macOS 13+**

##  Documentation

- **[Main README](https://github.com/caitlynmeeks/Noodlings/blob/master/README.md)** - Full project overview
- **[A_NOODLE_IS_ALL_YOU_NEED.md](https://github.com/caitlynmeeks/Noodlings/blob/master/docs/A_NOODLE_IS_ALL_YOU_NEED.md)** - BRENDA whitepaper
- **[Research Guide](https://github.com/caitlynmeeks/Noodlings/blob/master/research/README.md)** - Training pipeline and ablation studies

##  Theoretical Grounding

- **Predictive Processing**: Hierarchical predictive coding (Friston, Clark, Rao & Ballard)
- **Affective Primacy**: Emotions as substrate of experience (Panksepp, Barrett)
- **Theatrical Control**: Narrative events as interface primitives (Brenda Laurel)

##  Limitations

1. Apple Silicon only (MLX is Metal-specific)
2. Text-only (no vision, audio, or multimodal grounding)
3. LLM dependency (requires external LLM for text generation)
4. Synthetic training data (not validated on real conversations at scale)
5. Proof-of-concept stage (motor-sled-boat demonstration)

##  License

MIT License - see [LICENSE](https://github.com/caitlynmeeks/Noodlings/blob/master/LICENSE)

---

**Remember**: We're noodling, not claiming to have solved consciousness. This is an honest exploration of temporal dynamics in affect modeling. xw
