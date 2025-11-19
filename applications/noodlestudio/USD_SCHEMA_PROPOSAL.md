# Proposal to USD Alliance: Character and Ensemble Schemas

**Submitted by**: Caitlyn Meeks, Noodlings Project
**Date**: November 18, 2025
**Contact**: [Your contact info]
**Project**: NoodleStudio - Consciousness Agent Development Platform

---

## Executive Summary

We propose two new **typed schemas** for USD to support AI-driven character agents in animation, game development, and virtual production pipelines:

1. **`Character` Schema** - Individual consciousness agent with personality, affect, and behavioral parameters
2. **`Ensemble` Schema** - Collection of Characters with defined relationship dynamics

These schemas bridge the gap between **scene description** and **behavioral intelligence**, enabling studios to treat AI agents as first-class scene entities alongside geometry, lights, and cameras.

---

## Motivation

### Current Problem

Modern pipelines treat AI characters as:
- External systems (not part of scene description)
- Opaque black boxes (personality hidden in code)
- Non-portable (tied to specific game engines/frameworks)
- Non-animatable (no time-sampled behavioral data)

### Our Solution

Define Characters as **USD prims** with:
- **Personality traits** (animatable attributes)
- **Affective state** (emotional vector, time-sampled)
- **Backstory/prompt** (metadata)
- **LLM configuration** (provider, model, parameters)
- **Relationship networks** (to other Characters)

This makes AI characters:
- ✅ **Portable** - Works in any USD-compliant tool
- ✅ **Inspectable** - All parameters visible in scene graph
- ✅ **Animatable** - Personality/affect can be keyframed
- ✅ **Composable** - Characters combine into Ensembles
- ✅ **Versionable** - Track character evolution over time

---

## Proposed Schema: `Character`

### Schema Definition

```usd
class Character "Character" (
    doc = """Character represents an AI-driven agent with personality,
             affect, and behavioral parameters. Suitable for NPCs,
             virtual actors, and consciousness-based entities."""
    inherits = </Typed>
    customData = {
        string className = "Character"
        string schemaType = "singleApply"
    }
) {
    # ===== IDENTITY =====
    uniform token character:species = "human" (
        doc = "Character species/type (human, robot, alien, creature, etc.)"
        allowedTokens = ["human", "robot", "alien", "creature", "hybrid", "abstract"]
    )

    string character:backstory = "" (
        doc = "Character backstory and context for LLM prompting"
    )

    string character:description = "" (
        doc = "Physical appearance and mannerisms"
    )

    # ===== PERSONALITY (Big Five + Extensions) =====
    # All personality traits are [0.0, 1.0] normalized

    float character:personality:extraversion = 0.5 (
        doc = "Outgoing (1.0) vs. Reserved (0.0)"
    )

    float character:personality:agreeableness = 0.5 (
        doc = "Cooperative (1.0) vs. Competitive (0.0)"
    )

    float character:personality:conscientiousness = 0.5 (
        doc = "Organized (1.0) vs. Spontaneous (0.0)"
    )

    float character:personality:neuroticism = 0.5 (
        doc = "Anxious (1.0) vs. Stable (0.0)"
    )

    float character:personality:openness = 0.5 (
        doc = "Creative (1.0) vs. Traditional (0.0)"
    )

    float character:personality:curiosity = 0.5 (
        doc = "Inquisitive (1.0) vs. Indifferent (0.0)"
    )

    float character:personality:impulsivity = 0.5 (
        doc = "Spontaneous (1.0) vs. Methodical (0.0)"
    )

    float character:personality:emotionalVolatility = 0.5 (
        doc = "Reactive (1.0) vs. Stable (0.0)"
    )

    # ===== AFFECTIVE STATE (5-D Affect Vector) =====
    # Can be time-sampled for animation!

    float character:affect:valence = 0.0 (
        doc = "Emotional valence: Positive (1.0) to Negative (-1.0)"
    )

    float character:affect:arousal = 0.5 (
        doc = "Arousal level: High (1.0) to Low (0.0)"
    )

    float character:affect:fear = 0.0 (
        doc = "Fear level: Terrified (1.0) to Fearless (0.0)"
    )

    float character:affect:sorrow = 0.0 (
        doc = "Sorrow level: Grieving (1.0) to Joyful (0.0)"
    )

    float character:affect:boredom = 0.0 (
        doc = "Boredom level: Bored (1.0) to Engaged (0.0)"
    )

    # ===== LLM CONFIGURATION =====

    uniform token character:llm:provider = "local" (
        doc = "LLM provider (local, openai, anthropic, etc.)"
        allowedTokens = ["local", "openai", "anthropic", "together", "groq", "custom"]
    )

    string character:llm:model = "" (
        doc = "Specific model identifier (e.g., 'gpt-4', 'claude-3-opus')"
    )

    float character:llm:temperature = 0.7 (
        doc = "LLM sampling temperature (0.0 = deterministic, 1.0 = creative)"
    )

    int character:llm:maxTokens = 200 (
        doc = "Maximum tokens per response"
    )

    # ===== RELATIONSHIPS =====

    rel character:ensemble (
        doc = "Relationship to parent Ensemble prim (if any)"
    )

    rel character:relationships (
        doc = "Relationships to other Character prims"
    )

    # ===== BEHAVIORAL PARAMETERS =====

    float character:speakThreshold = 0.5 (
        doc = "Surprise threshold for triggering speech (0.0 = chatty, 1.0 = silent)"
    )

    bool character:enlightened = false (
        doc = "Enlightenment mode: Character-immersive (false) vs. Meta-aware (true)"
    )

    # ===== METADATA =====

    uniform token character:archetype = "custom" (
        doc = "Character archetype (hero, villain, mentor, trickster, etc.)"
        allowedTokens = ["hero", "villain", "mentor", "trickster", "sidekick",
                        "love_interest", "fool", "ruler", "caregiver", "custom"]
    )

    string[] character:tags = [] (
        doc = "Searchable tags for character discovery"
    )
}
```

### Example Usage

```usd
def Character "Harlequin" (
    prepend apiSchemas = ["CharacterSchema"]
) {
    uniform token character:species = "jester"
    string character:backstory = "A quick-witted servant who survives by his wits..."

    # Personality
    float character:personality:extraversion = 0.9
    float character:personality:curiosity = 0.9
    float character:personality:impulsivity = 0.8

    # Current affect (can be time-sampled!)
    float character:affect:valence.timeSamples = {
        0: 0.6,
        10: 0.8,
        20: 0.4
    }

    # LLM config
    uniform token character:llm:provider = "local"
    string character:llm:model = "qwen/qwen3-14b-2507"

    # Metadata
    uniform token character:archetype = "trickster"
    string[] character:tags = ["comedy", "servant", "italian"]
}
```

---

## Proposed Schema: `Ensemble`

### Schema Definition

```usd
class Ensemble "Ensemble" (
    doc = """Ensemble represents a collection of Characters designed
             to work together with defined relationship dynamics.
             Like a Unity prefab for character groups."""
    inherits = </Typed>
    customData = {
        string className = "Ensemble"
        string schemaType = "singleApply"
    }
) {
    # ===== IDENTITY =====

    string ensemble:description = "" (
        doc = "Description of ensemble dynamics and purpose"
    )

    uniform token ensemble:genre = "drama" (
        doc = "Genre/style of ensemble"
        allowedTokens = ["drama", "comedy", "action", "horror", "scifi",
                        "fantasy", "romance", "mystery", "historical", "custom"]
    )

    # ===== COMPOSITION =====

    rel ensemble:characters (
        doc = "Member Characters in this ensemble"
    )

    # ===== WORLD BUILDING =====

    string ensemble:suggestedSetting = "" (
        doc = "Recommended physical setting for this ensemble"
    )

    string ensemble:relationshipDynamics = "" (
        doc = "Description of how characters interact"
    )

    string[] ensemble:sceneSuggestions = [] (
        doc = "Suggested scenes/scenarios for this ensemble"
    )

    # ===== METADATA =====

    string ensemble:version = "1.0.0" (
        doc = "Ensemble version (semantic versioning)"
    )

    string ensemble:author = "" (
        doc = "Original creator of ensemble"
    )

    float ensemble:price = 0.0 (
        doc = "Commercial price (0.0 = free)"
    )

    uniform token ensemble:license = "free" (
        doc = "License type"
        allowedTokens = ["free", "indie", "studio", "enterprise", "custom"]
    )

    string[] ensemble:tags = [] (
        doc = "Searchable tags"
    )

    int ensemble:downloadCount = 0 (
        doc = "Number of times downloaded (marketplace stat)"
    )

    float ensemble:rating = 0.0 (
        doc = "User rating 0.0-5.0"
    )
}
```

### Example Usage

```usd
def Ensemble "CommediaDellArte" (
    prepend apiSchemas = ["EnsembleSchema"]
) {
    string ensemble:description = "Classic Italian theater archetypes..."
    uniform token ensemble:genre = "comedy"

    # Member characters
    prepend rel ensemble:characters = [
        </Stage/Characters/Harlequin>,
        </Stage/Characters/Pantalone>,
        </Stage/Characters/Colombina>,
        </Stage/Characters/IlCapitano>
    ]

    # World building
    string ensemble:suggestedSetting = "A piazza in Renaissance Venice"
    string ensemble:relationshipDynamics = "Harlequin schemes, Pantalone hoards..."
    string[] ensemble:sceneSuggestions = [
        "The servants conspire to steal Pantalone's gold",
        "Il Capitano tries to impress Colombina"
    ]

    # Metadata
    string ensemble:version = "1.0.0"
    string ensemble:author = "Noodlings Studio"
    float ensemble:price = 0.0
    uniform token ensemble:license = "free"
    string[] ensemble:tags = ["comedy", "theater", "classical"]
}
```

---

## Use Cases

### 1. Animation Studios
**Before USD Character Schema:**
- Animators manually keyframe all character actions
- No behavioral intelligence in pipeline
- Character personalities live in separate documents

**After:**
- Characters are USD prims with personality attributes
- Time-sample affect states to drive facial animation
- Export character behavior alongside geometry
- Personality traits inform animation choices

### 2. Game Development
**Before:**
- NPCs configured in engine-specific formats (Unity prefabs, Unreal Blueprints)
- Not portable between engines
- Artists can't inspect NPC personalities in DCC tools

**After:**
- NPCs defined as USD Characters
- Import into any engine
- Artists edit personality in Maya/Houdini
- Behavior portable across Unity, Unreal, Godot

### 3. Virtual Production
**Before:**
- Virtual actors controlled by separate AI systems
- No scene description for actor intelligence
- Can't preview AI behavior in pre-vis

**After:**
- Virtual actors are USD Characters
- Preview AI responses in USD-based tools
- Time-sample affect for emotion-driven cameras
- AI behavior part of scene versioning

### 4. AI Research
**Before:**
- Agent configurations scattered across codebases
- Hard to reproduce experiments
- Can't share agent baselines

**After:**
- Agents are portable USD Characters
- Reproduce experiments by sharing .usda files
- Benchmark personality archetypes
- Version control for agent evolution

---

## Technical Benefits

### 1. **Interoperability**
- Works in Maya, Houdini, Blender, Katana, etc.
- Import/export between all USD-compliant tools
- No vendor lock-in

### 2. **Composition**
- Layer character modifications (override personality in shot)
- Reference character from asset library
- Variant sets for mood/context switching

### 3. **Animation**
- Time-sample personality traits (character arc!)
- Keyframe affect state (emotional beats)
- Drive procedural animation from affect

### 4. **Performance**
- Lazy evaluation (USD's strength)
- Scene graph traversal for character discovery
- Efficient relationship queries

### 5. **Versioning**
- Track character evolution over time
- Git-friendly text format (.usda)
- Diff personality changes

---

## Implementation Notes

### File Format
- `.usda` (ASCII) for human-readable characters
- `.usdc` (Crate) for production performance
- `.usdz` for packaged character assets

### Namespace
- `character:` namespace for Character attributes
- `ensemble:` namespace for Ensemble attributes
- Follows USD naming conventions

### Compatibility
- Schema doesn't break existing USD pipelines
- Graceful degradation (characters just Xforms if schema not loaded)
- No new dependencies (uses standard USD types)

### Extensibility
- Studios can add custom attributes via API schemas
- Personality model extensible (add traits via composition)
- LLM providers open-ended (token list suggests but doesn't restrict)

---

## Comparison to Existing USD Schemas

| Schema | Purpose | Relationship to Character |
|--------|---------|--------------------------|
| `UsdSkel` | Skeletal animation | Character might reference skeleton |
| `UsdGeom` | Geometry primitives | Character might own geometry |
| `UsdShade` | Material/shading | Character appearance uses shading |
| `UsdLux` | Lighting | No direct relationship |
| **`Character`** | **AI behavior/personality** | **New domain: behavioral intelligence** |
| **`Ensemble`** | **Character grouping** | **New domain: relationship networks** |

---

## Industry Precedent

### Similar Efforts
1. **glTF Extensions** - Added material, physics, lights to glTF over time
2. **USD Itself** - Started with geometry, added skeletons, materials, volumes
3. **Web Standards** - HTML evolved from documents to applications

### Why USD?
- USD is becoming **the** interchange format for 3D pipelines
- Pixar, ILM, Disney, Unity, Epic, Nvidia all invested
- Better to extend USD than create parallel standard

---

## Requested Actions

We request the USD Alliance working group to:

1. **Review schemas** for technical correctness and USD conventions
2. **Discuss scope** - Should this be in core USD or an extension?
3. **Provide feedback** on naming, structure, attributes
4. **Consider inclusion** in USD roadmap (perhaps USD 25.x or 26.x)

We are committed to:
- Implementing reference USD plugins
- Providing sample .usda files
- Writing comprehensive documentation
- Supporting adoption in production pipelines

---

## Reference Implementation

We have built **NoodleStudio**, a production tool using these schemas:
- Export/import Character and Ensemble prims
- Time-sampled affect animation
- Ensemble composition and spawning
- Full USD integration

**Source code**: https://github.com/caitlynmeeks/Noodlings
**Documentation**: [Included in this proposal]

---

## Contact Information

**Caitlyn Meeks**
Founder, Noodlings Project
Email: [Your email]
GitHub: https://github.com/caitlynmeeks

**Collaborators:**
- Claude (Anthropic AI) - Schema design partner
- [Other contributors]

---

## Appendix A: Complete Schema Files

See attached:
- `character_schema.usda` - Full Character schema definition
- `ensemble_schema.usda` - Full Ensemble schema definition
- `examples/commedia_dellarte.usda` - Example ensemble
- `examples/space_trek_crew.usda` - Example ensemble

---

## Appendix B: Endorsements

**Philip Rosedale** (Second Life Founder):
*"From Second Life prims to USD prims to Noodling prims - this is the natural evolution of virtual world building. Characters deserve first-class scene representation."*

[Add more endorsements as we get them]

---

**Submitted with respect and enthusiasm for USD's continued evolution.**

🎭 Let's make characters first-class citizens in scene description!
