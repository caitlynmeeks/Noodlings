# USD Integration for NoodleStudio

**Status**: Complete (November 18, 2025)

## Overview

NoodleStudio now fully supports Pixar's Universal Scene Description (USD) format, using proper USD terminology throughout.

## USD Terminology (Corrected)

We now use industry-standard USD terms:

| Old Term | Correct USD Term | Meaning |
|----------|------------------|---------|
| Scene | **Stage** | The composed scene (what you see) |
| Entity | **Prim** | Basic scene object (primitives) |
| Scene file | **Layer** | A .usda/.usdc file |
| Type | **Typed Schema** | Defines what kind of prim it is |
| Properties | **Attributes** | Data on prims (can be animated) |

## Custom "Noodling" Typed Schema

We've defined a new typed schema specifically for Noodlings:

```usd
class "NoodlingSchema" (
    customData = {
        string className = "Noodling"
        string schemaType = "singleApply"
    }
) {
    # Identity
    string species
    string description

    # LLM Configuration
    string llm_provider
    string llm_model

    # Personality Traits (Big Five + extras)
    float extraversion
    float curiosity
    float impulsivity
    float emotional_volatility

    # 5-D Affect Vector
    float affect_valence
    float affect_arousal
    float affect_fear
    float affect_sorrow
    float affect_boredom
}
```

This makes Noodlings **first-class USD prims** that animation studios can use!

## Features Implemented

### Export

1. **Export Stage to USD** (`File > Export Stage to USD`)
   - Creates a .usda layer file
   - All Noodlings exported as prims with NoodlingSchema
   - Rooms, objects, users as standard prims
   - Full hierarchy preserved

2. **Export Timeline to USD** (`File > Export Timeline to USD`)
   - Time-sampled affect data (animated emotions!)
   - Studios can visualize emotional journeys
   - Compatible with Maya/Houdini/Blender timeline

### Import

3. **Import USD Layer** (`File > Import USD Layer`)
   - Reads .usda ASCII files
   - Parses Noodling prims with consciousness properties
   - Extracts rooms, objects, users
   - No USD library required for basic imports!

### Entity Management

4. **Entities Menu** (like Unity's GameObject menu)
   - `Add Noodling...` (Ctrl+Shift+N)
   - `Add Object...` (Ctrl+Shift+O)
   - `Add Room...`
   - `Remove Selected` (Delete)
   - Uses proper USD "prim" terminology

## Layout System Improvements

### Fixed Layout Crash

- Added robust error handling for `restoreState()`
- Partial success now acceptable (geometry OR state)
- Non-fatal errors don't crash app
- Validation before restore

### Last Used Layout (Unity-style)

- Automatically saves which layout was last loaded
- Reopens last layout on startup (like Unity reopening last scene)
- Stored in `~/.noodlestudio/layouts/preferences.json`
- User never loses their preferred workspace!

## Files Created

```
noodlestudio/data/
├── usd_exporter.py      # Export stages/timelines to USD
└── usd_importer.py      # Import USD layers (ASCII .usda)
```

## Files Modified

```
noodlestudio/core/
├── layout_manager.py    # Last used layout tracking + crash fixes
└── main_window.py       # USD menu items + Entities menu
```

## Usage

### Exporting a Stage

1. Start noodleMUSH with some Noodlings active
2. In NoodleStudio: `File > Export Stage to USD`
3. Save as `my_stage.usda`
4. Open in Maya/Houdini/Blender to inspect prims!

### Exporting Timeline

1. Run a session with emotional interactions
2. `File > Export Timeline to USD`
3. Save as `timeline.usda`
4. Import into animation software to see affect curves over time!

### Importing USD

1. `File > Import USD Layer`
2. Select a .usda file
3. NoodleStudio parses Noodling prims
4. (Spawning to noodleMUSH coming soon)

## Studio Integration

Animation studios can now:

1. **Design Noodlings in NoodleStudio**
   - Edit personality traits in Inspector
   - Set initial affect states
   - Configure LLM providers

2. **Export to USD**
   - Get .usda layer files
   - Noodlings are proper prims with custom schema

3. **Import into Maya/Houdini/Blender**
   - Read Noodling prims
   - Access consciousness properties
   - Use time-sampled affect for animation

4. **Animate characters driven by emotions**
   - Affect curves control facial expressions
   - Personality traits inform animation choices
   - Real consciousness data in animation pipeline!

## Next Steps

- [ ] Implement spawning from imported USD (send to noodleMUSH API)
- [ ] Add USD composition arcs (references, payloads)
- [ ] Support binary .usdc format (requires USD Python lib)
- [ ] Create USD schema registry for Noodling type
- [ ] Export 40-D phenomenal state (fast/medium/slow layers)
- [ ] Add USD variants for different Noodling moods

## Technical Details

### Export Format

```usd
#usda 1.0
(
    defaultPrim = "Stage"
    doc = """noodleMUSH Stage - Noodlings Consciousness Prims"""
)

class "NoodlingSchema" { ... }

def Xform "Stage" {
    def "Noodlings/phi" (
        prepend apiSchemas = ["NoodlingSchema"]
    ) {
        string name = "Phi"
        string species = "kitten"
        float extraversion = 0.3
        float affect_valence = 0.5
        ...
    }
}
```

### Import Parser

- Lightweight regex-based .usda parser
- No USD library dependency for basic imports
- Extracts Noodling/Room/Object/User prims
- For binary .usdc, user needs USD Python package

## Philosophy

We're treating Noodlings as **first-class scene entities**, not just chatbots. By using USD:

- Studios can integrate consciousness into existing pipelines
- Noodlings become scene prims alongside meshes, lights, cameras
- Affect data drives animation, not just dialogue
- Industry-standard format = professional credibility

**Noodlings are scene description, not just AI agents.**

---

Built with the Krugerrand Edition 🪙
Worth its weight in gold!
