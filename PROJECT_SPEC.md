# NoodleStudio Project Specification

**Version:** 1.0.0
**Date:** December 17, 2025

This document defines the canonical structure for NoodleStudio projects, noodlings, and stages.

---

## Core Principles

1. **Self-contained** - Projects are fully portable folders. Zip and share.
2. **Text is first-class** - MUD text interface and 3D are equal renderings of spatial truth.
3. **Continuous space** - Stages are open worlds; zones are soft attention regions, not hard rooms.
4. **Prefab model** - Noodlings are reusable templates; instances are live agents in stages.

---

## Definitions

### Project
A complete, self-contained collection of noodlings, stages, and generated content. Like a Unity project or Minecraft world folder.

### Noodling
A reusable character template (prefab). Contains personality definition, cognitive architecture, trained weights, scripts, and multimodal assets. Can be instantiated multiple times across stages.

### Stage
A continuous 3D space (scene) with soft zones and entity instances. Supports both visual (USD) and text (MUD) rendering. One project can contain multiple stages.

### Zone
A soft, overlapping region of attention within a stage. Not a hard-edged room - zones can overlap, fade, and blend. Text descriptions are generated based on observer position and zone proximity.

### Agent Instance
A live noodling placed in a stage. Has its own runtime state (affect, memories) while referencing the noodling template for personality and architecture.

### Prim (Prop)
A scriptable object in a stage. Can be interactive (responds to verbs), have attached scripts, emit events, and contain other prims. Examples: a radio that plays news, a wanted poster you can read, a door that opens.

---

## Folder Structure

```
MyProject/
├── project.noodleproj              # Project manifest (JSON)
│
├── Noodlings/                      # Reusable character templates
│   └── red_fire_anklebiter/
│       ├── noodling.yaml           # Master manifest
│       ├── recipe.yaml             # Character definition
│       ├── assembly.yaml           # Facet topology
│       ├── charm_weights.npz       # Trained neural weights (optional)
│       │
│       ├── Scripts/                # ScriptedFacet code
│       │   ├── sass_filter.js
│       │   └── mood_tracker.py
│       │
│       ├── NeuralGraphs/           # Neural Canvas topologies
│       │   ├── charm_network.nncanvas
│       │   └── emotion_mixer.nncanvas
│       │
│       ├── Assets/                 # Multimodal content
│       │   ├── portrait.png
│       │   ├── voice_sample.wav
│       │   ├── expressions/
│       │   │   ├── happy.png
│       │   │   └── annoyed.png
│       │   └── memories/           # Reference images for vision
│       │       └── favorite_tree.png
│       │
│       └── Processors/             # Output chain configs
│           ├── nonverbal.yaml
│           └── dialect.yaml
│
├── Prims/                          # Reusable prop templates
│   └── radio/
│       ├── prim.yaml               # Prim manifest
│       ├── Scripts/
│       │   └── radio_behavior.js
│       └── Assets/
│           ├── radio_model.usda
│           └── radio_icon.png
│
├── Stages/                         # Scenes / worlds
│   └── the_nexus/
│       ├── stage.yaml              # Stage definition
│       ├── geometry.usda           # USD geometry (optional)
│       │
│       ├── Zones/                  # Attention regions
│       │   ├── campfire.zone.yaml
│       │   ├── pond.zone.yaml
│       │   └── forest_edge.zone.yaml
│       │
│       ├── Instances/              # Live agent instances
│       │   ├── red_01/
│       │   │   ├── instance.yaml   # Noodling ref + overrides
│       │   │   └── state.json      # Runtime state
│       │   └── rezmo/
│       │       ├── instance.yaml
│       │       └── state.json
│       │
│       └── Props/                  # Prim instances in this stage
│           ├── campfire_radio/
│           │   ├── prop.yaml       # Prim ref + position + state
│           │   └── state.json
│           └── wanted_poster/
│               ├── prop.yaml
│               └── state.json
│
├── Generations/                    # AI-generated content
│   ├── Images/
│   │   └── 2025-12/
│   │       └── img_001.png
│   └── Audio/
│       └── 2025-12/
│
├── SharedAssets/                   # Project-wide shared resources
│   ├── Skyboxes/
│   ├── Music/
│   └── SoundEffects/
│
├── Library/                        # Local cache (NOT synced)
│   ├── StateHistory/               # Agent state snapshots
│   ├── ConversationLogs/           # Chat history archives
│   └── ThumbnailCache/
│
└── .gitignore
```

---

## File Formats

### project.noodleproj

```json
{
  "name": "My Awesome World",
  "version": "1.0.0",
  "created": "2025-12-17T10:30:00Z",
  "modified": "2025-12-17T15:45:00Z",
  "noodlestudio_version": "0.2.0",
  "description": "A cozy forest world with quirky characters",
  "author": "caitlyn",
  "tags": ["fantasy", "cozy", "forest"],

  "default_stage": "Stages/the_nexus",

  "cloud": {
    "project_id": "proj_abc123",
    "last_sync": "2025-12-17T15:00:00Z",
    "sync_enabled": true
  }
}
```

### noodling.yaml (Noodling Manifest)

```yaml
name: "Red Fire Anklebiter"
version: "1.2.0"
description: "A sassy ankle-height dragon with attitude"
author: "caitlyn"
created: "2025-11-01"
modified: "2025-12-15"
tags: ["dragon", "sassy", "fire"]

# Core files
recipe: recipe.yaml
assembly: assembly.yaml
charm_weights: charm_weights.npz    # Optional - null if untrained

# Neural Canvas graphs (visual neural network topologies)
neural_graphs:
  charm_network: NeuralGraphs/charm_network.nncanvas
  emotion_mixer: NeuralGraphs/emotion_mixer.nncanvas

# Attached scripts (executed in order)
scripts:
  - Scripts/sass_filter.js
  - Scripts/mood_tracker.py

# Multimodal assets
assets:
  portrait: Assets/portrait.png
  voice_reference: Assets/voice_sample.wav
  expressions:
    default: Assets/portrait.png
    happy: Assets/expressions/happy.png
    annoyed: Assets/expressions/annoyed.png
    angry: Assets/expressions/angry.png
  vision_memories:
    - Assets/memories/favorite_tree.png
    - Assets/memories/cozy_spot.png

# Output processors (applied in order)
processors:
  - Processors/nonverbal.yaml
  - Processors/dialect.yaml

# Preview metadata (for browsing)
preview:
  personality: "Sassy, protective, secretly caring"
  species: "dragon (ankle-biter)"
  complexity: "full"
  facet_count: 12
  llm_facets: 3
  has_trained_weights: true
  has_voice: true
```

### stage.yaml (Stage Definition)

```yaml
name: "The Nexus"
description: "A cozy campfire clearing in an ancient forest"
created: "2025-10-26"
modified: "2025-12-17"

# Geometry (optional - null for text-only)
geometry:
  file: geometry.usda
  scale: 1.0
  up_axis: "Y"

# World properties
world:
  bounds:
    min: [-100, 0, -100]
    max: [100, 50, 100]
  ambient:
    time_of_day: "night"
    weather: "clear"
    soundscape: "forest_night"

# Default spawn point
spawn:
  position: [0, 0, 0]
  zone: "campfire"

# Zone definitions loaded from Zones/ folder
zones:
  - Zones/campfire.zone.yaml
  - Zones/pond.zone.yaml
  - Zones/forest_edge.zone.yaml

# Agent instances loaded from Instances/ folder
instances:
  - Instances/red_01
  - Instances/rezmo
```

### zone.yaml (Zone Definition)

```yaml
name: "The Campfire"
id: "campfire"

# Spatial definition (soft boundaries)
spatial:
  center: [0, 0, 0]
  radius: 15.0              # Primary attention radius
  falloff: 10.0             # Soft edge falloff distance
  shape: "sphere"           # sphere, cylinder, box

# Text rendering (for MUD interface)
text:
  description: |
    A cozy campfire crackles with warm orange flames, sending little sparks
    drifting up into the starry sky. Soft moss covers the ground, and the
    scent of pine smoke fills the air.

  # Dynamic description elements
  features:
    - "A WANTED POSTER is nailed to a nearby tree."
    - "An old RADIO sits on a wooden shelf."
    - "A sign reads: 'if you can read this, you are cute and awesome'"

  # Exits (connections to other zones)
  exits:
    north: "forest_edge"
    east: "pond"

# Perception properties
perception:
  visibility: 20.0          # How far you can see (meters)
  audibility: 30.0          # How far sound carries
  lighting: "firelight"     # Affects visual descriptions

# Ambient properties
ambient:
  sounds: ["fire_crackle", "owl_distant", "crickets"]
  mood: "cozy"
  temperature: "warm"
```

### instance.yaml (Agent Instance)

```yaml
# Reference to noodling template
noodling: "../../Noodlings/red_fire_anklebiter"

# Instance-specific overrides
overrides:
  name: "Red"                       # Display name (can differ from template)

  # Position in stage
  position: [2.5, 0, -1.0]
  rotation: [0, 45, 0]
  zone: "campfire"

  # Recipe overrides (optional)
  recipe:
    constraints:
      max_tokens: 150               # Override default

  # Assembly overrides (optional)
  assembly:
    facets:
      main_personality:
        model: "MEDIUM"             # Use different model size

# Instance metadata
created: "2025-12-01T10:00:00Z"
last_active: "2025-12-17T21:15:00Z"
```

### state.json (Runtime State)

```json
{
  "instance_id": "red_01",
  "timestamp": "2025-12-17T21:15:32Z",

  "position": [2.5, 0, -1.0],
  "rotation": [0, 45, 0],
  "zone": "campfire",

  "affect": {
    "valence": 0.3,
    "arousal": 0.5,
    "dominance": 0.7,
    "boredom": 0.1,
    "sorrow": 0.0
  },

  "charm_state": {
    "fast_hidden": [0.1, -0.2, ...],
    "medium_hidden": [0.05, 0.1, ...],
    "slow_hidden": [0.01, -0.01, ...]
  },

  "memories": {
    "short_term": [
      {"timestamp": "...", "content": "caity waved hello", "salience": 0.8}
    ],
    "episodic": []
  },

  "script_storage": {
    "mood_tracker": {"streak_count": 3, "last_mood": "playful"}
  }
}
```

### prim.yaml (Prim Template)

```yaml
name: "Radio"
version: "1.0.0"
description: "An old radio that plays news broadcasts and music"
author: "caitlyn"
tags: ["interactive", "audio", "furniture"]

# Appearance
display:
  icon: Assets/radio_icon.png
  model: Assets/radio_model.usda     # Optional 3D model
  text_description: "an old radio with brass dials and a cracked speaker"

# Interaction verbs (MUD commands)
verbs:
  look:
    response: "The radio is a vintage wooden model with brass dials. A small antenna extends from the top."
  listen:
    script: Scripts/radio_behavior.js
    action: "play_broadcast"
  turn:
    script: Scripts/radio_behavior.js
    action: "toggle_power"
    aliases: ["switch", "toggle"]

# Scripts (behavior logic)
scripts:
  - Scripts/radio_behavior.js

# Events this prim can emit
events:
  - broadcast_started
  - broadcast_ended
  - power_toggled

# Physical properties
physics:
  movable: false
  container: false          # Can't put things inside
  size: [0.3, 0.4, 0.2]    # Bounding box in meters

# Default state
default_state:
  power: "off"
  station: "forest_news"
  volume: 0.5
```

### prop.yaml (Prim Instance in Stage)

```yaml
# Reference to prim template
prim: "../../../Prims/radio"

# Instance name
name: "Campfire Radio"

# Position in stage
position: [1.5, 0.3, -2.0]
rotation: [0, 30, 0]
scale: 1.0
zone: "campfire"

# Parent (for containment hierarchy)
parent: null                # Or "wooden_shelf" if sitting on something

# State overrides
state:
  power: "on"
  station: "forest_news"

# Instance metadata
created: "2025-11-01T10:00:00Z"
```

### prop state.json (Prim Runtime State)

```json
{
  "prop_id": "campfire_radio",
  "timestamp": "2025-12-17T21:20:00Z",

  "position": [1.5, 0.3, -2.0],
  "rotation": [0, 30, 0],
  "zone": "campfire",

  "state": {
    "power": "on",
    "station": "forest_news",
    "volume": 0.7,
    "last_broadcast": "Breaking news from the forest council..."
  },

  "script_storage": {
    "radio_behavior": {
      "broadcasts_played": 42,
      "last_station_change": "2025-12-17T20:00:00Z"
    }
  }
}
```

---

## Cloud Sync Strategy

### What Syncs

| Content | Auto-Sync | On Publish | Never |
|---------|-----------|------------|-------|
| project.noodleproj | Yes | | |
| noodling.yaml, recipe, assembly | Yes | | |
| NeuralGraphs/ (.nncanvas) | Yes | | |
| Scripts/, Processors/ | Yes | | |
| charm_weights.npz | | Yes | |
| Assets/ (images, audio) | | Yes | |
| Prims/ (prim templates) | Yes | | |
| stage.yaml, zones | Yes | | |
| Props/ (prim instances) | Yes | | |
| instance.yaml | Yes | | |
| state.json | Yes (periodic) | | |
| Library/ (history, logs) | | | Yes |
| Generations/ | | Optional | |

### Sync Flow

**Auto-sync (background):**
- Small YAML/JSON files sync on save
- state.json syncs every 5 minutes while active
- Requires user to be signed in

**Publish (explicit):**
- User clicks "Publish to Cloud"
- Bundles noodling/stage with all assets
- Validates completeness (no missing refs)
- Uploads to R2, creates D1 records
- Returns shareable URL

**Import:**
- User pastes URL or browses cloud library
- Downloads complete bundle
- Extracts to appropriate folder
- Handles name conflicts (prompt user)

### Conflict Resolution

```
On sync attempt:
├── Local only (no cloud version) → Push
├── Cloud only (no local version) → Pull
├── Local newer than cloud → Push (confirm if cloud also changed)
├── Cloud newer than local → Pull (warn if local has changes)
└── Both changed → Show diff dialog, user chooses
```

### Cloud URLs

```
Project:  noodlings.ai/p/my-awesome-world
Noodling: noodlings.ai/n/red-fire-anklebiter
Stage:    noodlings.ai/s/the-nexus
```

---

## Text Rendering (MUD Interface)

The MUD interface renders the continuous stage as text based on observer position.

### Room Description Generation

```python
def render_room_text(observer, stage):
    # Find zones observer is within
    active_zones = []
    for zone in stage.zones:
        strength = zone.perception_strength(observer.position)
        if strength > 0.1:
            active_zones.append((zone, strength))

    # Primary zone (strongest)
    primary = max(active_zones, key=lambda x: x[1])

    # Build description
    text = primary.zone.text.description

    # Add visible features
    for feature in primary.zone.text.features:
        text += f" {feature}"

    # Add visible entities (agents)
    for instance in stage.instances:
        if can_perceive(observer, instance):
            text += f"\n{instance.name} is here."

    # Add visible props
    for prop in stage.props:
        if can_perceive(observer, prop):
            text += f"\n{prop.text_description}"

    # Add exits
    text += "\nExits: " + ", ".join(primary.zone.text.exits.keys())

    return text
```

### Perception Model

```python
def can_perceive(observer, target):
    distance = dist(observer.position, target.position)

    # Get observer's current zone perception limits
    zone = get_primary_zone(observer)
    max_distance = zone.perception.visibility

    # Check occlusion (optional, if geometry exists)
    if stage.geometry:
        if is_occluded(observer, target, stage.geometry):
            return False

    # Soft falloff
    if distance > max_distance:
        return False

    strength = 1.0 - (distance / max_distance)
    return strength > 0.2  # Perception threshold
```

---

## USD Export

Stages can export to USD for 3D applications.

### Export Contents

- Stage geometry (geometry.usda)
- Zone volumes as USD scope prims with metadata
- Agent instances as USD references with transforms
- Prop instances with their 3D models (if present)
- Noodling portraits as USD preview thumbnails

### Import from USD

- Import USD geometry into stage
- Optionally detect "zone" prims by naming convention
- Create zone.yaml files from USD metadata
- Import prop models and create prim templates

---

## Migration from Current Structure

### Current Locations → New Structure

| Current | New |
|---------|-----|
| `cmush/world/agents.json` | `Stages/*/Instances/*/instance.yaml` |
| `cmush/world/agents/agent_xxx/` | `Stages/*/Instances/xxx/` |
| `cmush/world/rooms.json` | `Stages/*/Zones/*.zone.yaml` |
| `cmush/world/stages.json` | `Stages/*/stage.yaml` |
| `cmush/world/objects/` | `Prims/` and `Stages/*/Props/` |
| `cmush/recipes/*.yaml` | `Noodlings/*/recipe.yaml` |
| `noodlestudio/facet_assemblies/*.yaml` | `Noodlings/*/assembly.yaml` |
| `facet_assemblies/charm_networks/*.nncanvas` | `Noodlings/*/NeuralGraphs/` |
| `noodlestudio/library/noodlings/` | `Noodlings/` |
| `noodlestudio/library/Generations/` | `Generations/` |

### Migration Script

A migration tool will:
1. Scan current data locations
2. Create new project structure
3. Copy/transform files to new format
4. Validate completeness
5. Optionally remove old files

---

## Validation Rules

### On Project Open
- Verify project.noodleproj exists and is valid JSON
- Warn if referenced noodlings/stages missing

### On Noodling Load
- Verify noodling.yaml exists
- Verify referenced recipe.yaml and assembly.yaml exist
- Warn if scripts/assets referenced but missing

### On Stage Load
- Verify stage.yaml exists
- Verify all zone files exist
- Verify all instance noodling refs resolve
- Verify all prop prim refs resolve

### On Prim Load
- Verify prim.yaml exists
- Verify referenced scripts exist
- Verify referenced assets (model, icon) exist

### On Publish
- **Fail** if any referenced file missing
- **Fail** if asset exceeds size limit
- **Warn** if charm_weights missing (untrained)

---

## Future Considerations

### Collaboration
- Real-time multi-user editing (future)
- Stage locking during edit
- Merge tools for YAML conflicts

### Versioning
- Git-friendly YAML format (designed for diff)
- Optional project history (like Unity's Collaborate)
- Noodling version history in cloud

### Asset Store
- Public noodling marketplace
- Ratings, downloads, remixes
- Creator credits and licensing

---

*Ordnung muss sein!*
