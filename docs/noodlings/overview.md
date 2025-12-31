# Noodlings

AI characters with continuous affect, temporal memory, and visual cognitive architectures.

---

## What is a Noodling?

A Noodling is an AI character defined by:

1. **Recipe** - Personality, appearance, baseline affect
2. **Facet Assembly** - Visual cognitive architecture (how they think)
3. **Radiance** - Optional Gaussian splat avatar

Unlike stateless chatbots, Noodlings have:
- Continuous internal state that evolves over time
- Multi-timescale memory (seconds, minutes, hours)
- Affect-driven behavior (not just prompt-driven)
- Perception-filtered world knowledge

## The Phenomenal Vector (PV)

Every Noodling has a 40-dimensional phenomenal state:

| Layer | Dimensions | Timescale | Purpose |
|-------|------------|-----------|---------|
| Fast | 16-D | Seconds | Immediate reactions |
| Medium | 16-D | Minutes | Conversational flow |
| Slow | 8-D | Hours/Days | Personality drift |

This is a **data structure**, not metaphysics. You can observe it, edit it, save it.

## Affect Model

5-dimensional continuous affect (no discrete emotion labels):

| Dimension | Range | Meaning |
|-----------|-------|---------|
| Valence | -1 to +1 | Pleasure/displeasure |
| Arousal | 0 to 1 | Energy level |
| Dominance | 0 to 1 | Sense of control |
| Boredom | 0 to 1 | Need for novelty |
| Sorrow | 0 to 1 | Grief/loss state |

## Next

- [Recipe Format](recipe-format.md) - Define a character
- [Affect Model](affect-model.md) - Deep dive on affect
- [Facet Assemblies](facet-assemblies.md) - Cognitive architecture
