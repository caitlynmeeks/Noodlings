# Engine Integrations

NoodleStudio characters can be deployed in third-party game engines and platforms.

**Design in NoodleStudio. Deploy anywhere.**

---

## Available Integrations

| Engine | Status | Export Format | Documentation |
|--------|--------|---------------|---------------|
| **Unity** | Implemented | `.noodling` package | [Unity Integration](unity.md) |
| Unreal | Planned | TBD | - |
| Godot | Planned | TBD | - |
| Web (Three.js) | Planned | TBD | - |

---

## Export Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DESIGN TIME (NoodleStudio)                   │
├─────────────────────────────────────────────────────────────────┤
│  1. Design character personality and motivation                 │
│  2. Build facet assembly (cognition architecture)               │
│  3. Test dialogue and emotional responses                       │
│  4. Tune PAD baseline and dynamics                              │
│  5. File > Export > Export to Unity Package...                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     .noodling Package                           │
├─────────────────────────────────────────────────────────────────┤
│  manifest.json      - Package metadata                          │
│  character.json     - Personality, motivation, initial PAD      │
│  assembly.json      - Facet configuration                       │
│  expressions.json   - PAD → FACS → VRM blendshape mapping       │
│  plays/             - Optional narrative beats                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RUNTIME (Game Engine)                        │
├─────────────────────────────────────────────────────────────────┤
│  Engine loads .noodling package                                 │
│  Character behavior runs via LLM calls                          │
│  PAD state drives facial expressions on avatar                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Affect Model Export

NoodleStudio uses a 5-dimensional internal affect model:

| Dimension | Range | Description |
|-----------|-------|-------------|
| Valence | -1 to +1 | Pleasure/displeasure |
| Arousal | 0 to 1 | Activation level |
| Dominance | 0 to 1 | Control/submission |
| Boredom | 0 to 1 | Engagement level |
| Sorrow | 0 to 1 | Background grief |

**Export mapping (5D to 3D PAD):**
- `valence` exports as `pleasure`
- `arousal` exports as `arousal`
- `dominance` exports as `dominance`
- `boredom` and `sorrow` are internal only (inform PAD but not exported)

This aligns with Mehrabian and Russell's PAD model used widely in affective computing.

---

## Expression Mapping Chain

The exported `expressions.json` contains a complete mapping chain:

```
PAD State → Emotion Weights → FACS Action Units → VRM Blendshapes
```

1. **PAD to Emotions**: Mehrabian/Russell weights map PAD to emotion intensities
2. **Emotions to FACS**: Each emotion activates specific Action Units
3. **FACS to VRM**: Action Units drive VRM-compatible blendshapes

This allows any VRM avatar to display appropriate expressions based on the character's emotional state.

---

## LLM Provider Support

Exported characters can use various LLM backends at runtime:

| Provider | Description |
|----------|-------------|
| OpenAI | Direct API (GPT-4o, GPT-4o-mini) |
| NoodleROUTER | Managed API via api.noodlings.ai |
| Azure OpenAI | Enterprise deployment |
| Local | Ollama or other local inference |

The `assembly.json` specifies model preferences (SMALL, MEDIUM, LARGE labels) which the runtime maps to specific models.

---

## See Also

- [Unity Integration](unity.md) - Full Unity plugin documentation
- [Package Format Specification](package-format.md) - Detailed JSON schemas
- [Build Settings](/docs/noodlestudio/build-settings.md) - Standalone app builds
