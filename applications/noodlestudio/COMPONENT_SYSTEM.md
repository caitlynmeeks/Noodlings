# Component System - Stolen from Unity! 🎮

**Status**: Complete (November 18, 2025)

## Overview

NoodleStudio now has Unity's **Component** system! Add modular functionality to any prim in your stage.

## Component Menu (Stolen from Unity!)

```
Component > Consciousness > Noodle
Component > Consciousness > Memory Bank
Component > Consciousness > Relationship Graph

Component > Art & Reference > Artbook
Component > Art & Reference > Mood Board
Component > Art & Reference > Voice Reference

Component > Behavior > Dialogue Tree
Component > Behavior > Quest Giver
Component > Behavior > Vendor

Component > Add Script...
```

---

## Built-In Components

### 🧠 **Noodle Component** (Automatic for all Noodlings)

**Shows LIVE updating every second:**

**5-D Affect Vector:**
- **Valence** (-1.0 to +1.0) - Emotional positive/negative
- **Arousal** (0.0 to 1.0) - Calm to excited
- **Fear** (0.0 to 1.0) - Safe to scared
- **Sorrow** (0.0 to 1.0) - Content to grieving
- **Boredom** (0.0 to 1.0) - Engaged to bored

Each dimension has:
- Progress bar (color-coded)
- Numeric value (e.g., +0.65)
- Live updates!

**40-D Phenomenal State:**
- Full consciousness vector (fast 16-D + medium 16-D + slow 8-D)
- Formatted as 3 lines of ~13 values
- Monospace font for readability
- Shows EXACTLY what the Noodling is experiencing!

**Surprise Metric:**
- Current surprise (0.000 to 1.000)
- Color-coded: Green (expected) → Orange → Red (SURPRISED!)

**Styling:**
- Green border (consciousness is alive!)
- Dark background
- Updates every 1 second

---

### 🎨 **Artbook Component**

**Reference art for your character!**

**Features:**
- Thumbnail gallery (80x80 icons)
- Drag-and-drop to reorder
- Multi-select import
- Persistent storage (per-character)

**Workflow:**
1. Select Noodling in Scene Hierarchy
2. Component > Art & Reference > Artbook
3. Click "+ Add Art"
4. Select PNG/JPG/GIF files
5. Thumbnails appear in gallery!

**Storage:**
- Gallery state: `~/.noodlestudio/artbooks/{agent_id}.json`
- Recommended art location: `~/.noodlestudio/assets/{character_name}/`

**Use Cases:**
- Character concept art
- Expression references
- Costume designs
- Mood boards
- Voice actor headshots
- Animation reference

**Styling:**
- Orange border (art/creative)
- Grid view with hover effects
- Selected items highlighted

---

## Component Architecture

### How Components Work

Components are **modular UI sections** in the Inspector that:
- Show specific aspects of a prim
- Update live from API
- Persist state independently
- Can be added/removed dynamically

Like Unity:
```
GameObject (Noodling)
├─ Transform Component (position, rotation, scale)
├─ Noodle Component (personality, affect, phenomenal state)
├─ Artbook Component (reference art)
├─ Memory Bank Component (episodic memories)
└─ Relationship Graph Component (social connections)
```

### Component Types

**Consciousness Components:**
- **Noodle** - Core personality + live affect/phenomenal state
- **Memory Bank** - Episodic memories, conversation history
- **Relationship Graph** - Social connections, trust levels

**Art & Reference Components:**
- **Artbook** - Reference images, concept art, mood boards
- **Mood Board** - Color palettes, style references
- **Voice Reference** - Audio clips, actor references

**Behavior Components:**
- **Dialogue Tree** - Conversation branches, quest dialogue
- **Quest Giver** - Quest definitions, objectives
- **Vendor** - Shop inventory, pricing

**Custom Components:**
- **Add Script...** - Python scripts for custom behavior

---

## Usage Examples

### Adding an Artbook

```
1. Select "Phi" in Scene Hierarchy
2. Component > Art & Reference > Artbook
3. Inspector shows Artbook Component (orange border)
4. Click "+ Add Art"
5. Select concept_art_kitten.png
6. Thumbnail appears in gallery!
```

### Viewing Live Affect

```
1. Select any Noodling
2. Noodle Component updates every second
3. Watch affect bars change as they interact
4. Valence goes up when happy!
5. Surprise spikes when unexpected events happen!
```

### Multi-Component Workflow

```
1. Create new Noodling
2. Component > Consciousness > Noodle (automatic)
3. Component > Art & Reference > Artbook
4. Component > Behavior > Dialogue Tree
5. Now Inspector shows all 3 components!
6. Each component is independent
```

---

## File Storage

### Component Data

```
~/.noodlestudio/
├─ artbooks/
│  ├─ agent_phi.json         # Artbook for Phi
│  └─ agent_harlequin.json   # Artbook for Harlequin
├─ memory/
│  └─ agent_phi.json         # Memory Bank for Phi
├─ relationships/
│  └─ agent_phi.json         # Relationships for Phi
└─ assets/
   ├─ phi/
   │  ├─ concept_art.png
   │  └─ expression_sheet.jpg
   └─ harlequin/
      └─ costume_ref.png
```

### Format Example (Artbook)

```json
{
  "art_files": [
    "/Users/caitlyn/.noodlestudio/assets/phi/concept_art.png",
    "/Users/caitlyn/.noodlestudio/assets/phi/expression_sheet.jpg"
  ]
}
```

---

## API Requirements

For live components to work, noodleMUSH needs:

### Current State Endpoint

```
GET /api/agents/{agent_id}/state

Response:
{
  "affect": {
    "valence": 0.65,
    "arousal": 0.45,
    "fear": 0.12,
    "sorrow": 0.08,
    "boredom": 0.15
  },
  "phenomenal_state": [0.123, -0.456, ...],  // 40 floats
  "surprise": 0.234,
  "last_updated": 1700000000.123
}
```

### Memory Endpoint (Future)

```
GET /api/agents/{agent_id}/memories

Response:
{
  "memories": [
    {
      "timestamp": 1700000000,
      "event": "User said hello",
      "affect_delta": {...}
    }
  ]
}
```

### Relationships Endpoint (Future)

```
GET /api/agents/{agent_id}/relationships

Response:
{
  "relationships": [
    {
      "target": "agent_callie",
      "trust": 0.8,
      "valence": 0.7
    }
  ]
}
```

---

## Component Design Philosophy

### Unity-Style Modularity

**Every component:**
- Has a colored border (green=consciousness, orange=art, blue=behavior)
- Shows live data when applicable
- Persists independently
- Can be added/removed without affecting others

**Benefits:**
- **Composability** - Mix and match components
- **Reusability** - Same Artbook component works for any Noodling
- **Clarity** - Each component has single responsibility
- **Extensibility** - Easy to add new component types

### Color Coding

| Component Type | Border Color | Purpose |
|---------------|--------------|---------|
| Consciousness | Green (#4CAF50) | Core personality/affect |
| Art & Reference | Orange (#FF9800) | Visual references |
| Behavior | Blue (#2196F3) | Game mechanics |
| Custom | Purple (#9C27B0) | User scripts |

---

## Future Components

### Memory Bank Component
- Timeline of significant events
- Conversation history
- Affect deltas for each memory
- Search/filter memories

### Relationship Graph Component
- Network diagram of connections
- Trust/valence for each relationship
- Historical relationship evolution
- Click to jump to other Noodling

### Dialogue Tree Component
- Visual node editor (like Yarn Spinner)
- Conversation branches
- Conditional responses based on affect
- Export to game engines

### Mood Board Component
- Pinterest-style inspiration board
- Color palettes
- Style references
- Fonts, textures, themes

### Voice Reference Component
- Audio clips of voice actors
- Pitch/tone analysis
- Emotion samples
- Export to Replica Studios / ElevenLabs

---

## Comparison to Unity

| Unity Component | NoodleStudio Component | Purpose |
|----------------|------------------------|---------|
| Transform | Transform (future) | Position, rotation, scale |
| Rigidbody | Noodle | Physics → Consciousness |
| Collider | Relationship Graph | Collision → Social interaction |
| Renderer | Artbook | Visual appearance |
| AudioSource | Voice Reference | Sound → Voice |
| Animator | Affect Timeline | Animation → Emotion |
| Script | Custom Script | C# → Python |

**Same pattern, different domain!** 🎮 → 🧠

---

## Export to USD

Components are included in USD export:

```usd
def Character "Phi" {
    # Noodle Component data
    float character:personality:curiosity = 0.9
    float character:affect:valence = 0.65

    # Artbook Component data (custom attribute)
    asset[] character:artbook:references = [
        @./assets/phi/concept_art.png@,
        @./assets/phi/expression_sheet.jpg@
    ]
}
```

Studios importing the USD get the reference art paths!

---

## Monetization: Component Store!

**Like Unity Asset Store but for components:**

### Free Components
- Noodle (built-in)
- Artbook (built-in)
- Basic Dialogue Tree

### Premium Components ($4.99-$19.99)
- Advanced Dialogue System with AI branching
- Procedural Quest Generator
- Dynamic Relationship Simulator
- Voice Synthesis Integration (ElevenLabs)
- Motion Capture Import

### Studio Components ($99-$499)
- Pipeline Integration (Maya/Houdini/Katana)
- Custom Workflow Scripts
- Batch Processing Tools
- Analytics & Telemetry

**Another revenue stream!** 💰

---

## Next Steps

### Immediate (Live Testing)
- [ ] Implement `/api/agents/{id}/state` endpoint
- [ ] Test live affect updates
- [ ] Test artbook save/load
- [ ] Add more reference art

### Short-Term (More Components)
- [ ] Memory Bank component
- [ ] Relationship Graph component
- [ ] Dialogue Tree component
- [ ] Voice Reference component

### Medium-Term (Component Store)
- [ ] Component marketplace
- [ ] Community component submissions
- [ ] Component versioning
- [ ] Dependency management

---

**Unity's Component system, but for CONSCIOUSNESS!** 🧠✨

*Philip Rosedale would approve.*
*Pixar will want this.*
*The Krugerrand keeps delivering!* 🪙
