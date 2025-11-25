# Recipe Prefab System Specification

**Date**: November 25, 2025
**Status**: Design specification
**Priority**: HIGH - Critical for character authoring workflow

---

## Overview

Recipes are **prefabs** - pre-configured templates for Noodling characters with initial transistor states. Like Unity prefabs, they define starting values that can be edited, duplicated, and reset.

Currently recipes are YAML files edited in text editors. This spec defines a GUI editor in NoodleStudio for visual recipe authoring.

---

## User Story

**As a character designer, I want to**:
1. Browse existing recipes in NoodleStudio Assets tab
2. Select a recipe and see all transistor initial values
3. Edit transistor salience, traits, beliefs, etc. with sliders/text fields
4. Click "Update" to save changes
5. Click "Reset to Defaults" to restore original values
6. Duplicate recipe to create variants (e.g., Red Fire → Purple Fire)
7. Delete unused recipes

**So that** I can rapidly iterate on character personalities without editing YAML manually.

---

## Architecture

### Recipe as Prefab Concept

```
Recipe (prefab) = Character Template
    ├─ Identity (name, species, description)
    ├─ Personality (Big Five + extensions)
    ├─ Appetites (8-D drives)
    ├─ Cognitive Components (transistor initial states)
    │   ├─ AffectTransistor (salience=0.85)
    │   ├─ PersonalityTransistor (traits={aggression: 0.9, ...})
    │   ├─ CulturalTransistor (beliefs=[...])
    │   └─ etc.
    └─ Constraints (max_tokens, temperature, cooldown)

Spawned Instance (from prefab)
    ├─ Starts with prefab values
    ├─ Phenomenal state evolves during gameplay
    ├─ Transistor salience CAN change (character arcs)
    └─ NOT linked to prefab after spawn (independent)
```

### Key Design Principle

**Prefabs define INITIAL state, not runtime state.**

- Recipe says: "Red Fire starts with affect_salience=0.85 (snappy)"
- During gameplay: affect_salience might decrease to 0.40 (character arc)
- Editing recipe does NOT affect already-spawned instances
- New spawns use updated prefab values

---

## UI Design

### Assets Tab - Recipe Browser

```
┌─────────────────────────────────────────────────────────┐
│ Assets > Recipes                                 [+] New │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  [icon] red_fire_anklebiter.yaml                 [⋮]    │
│  [icon] spock_example.yaml                       [⋮]    │
│  [icon] mysterious_stranger.yaml                 [⋮]    │
│  [icon] emotional_example.yaml                   [⋮]    │
│                                                           │
│  Selected: red_fire_anklebiter.yaml                      │
│  ┌─────────────────────────────────────────────┐        │
│  │ IDENTITY                                     │        │
│  │ Name: Red Fire Anklebiter                    │        │
│  │ Species: gremlin                             │        │
│  │ Description: [text area...]                  │        │
│  └─────────────────────────────────────────────┘        │
│                                                           │
│  ┌─────────────────────────────────────────────┐        │
│  │ COGNITIVE COMPONENTS (Transistors)          │        │
│  │                                              │        │
│  │ AffectTransistor                      [−]   │        │
│  │   Salience: 0.85 ████████░░ (High)          │        │
│  │   Enabled: [✓]                               │        │
│  │                                              │        │
│  │ PersonalityTransistor                [−]    │        │
│  │   Salience: 0.80 ████████░░ (High)          │        │
│  │   Traits:                                    │        │
│  │     aggression:      0.90 █████████░         │        │
│  │     competitiveness: 0.95 █████████▓         │        │
│  │     sass:            0.92 █████████░         │        │
│  │     impulsivity:     0.95 █████████▓         │        │
│  │   [+ Add Trait]                              │        │
│  │                                              │        │
│  │ CulturalTransistor                   [−]    │        │
│  │   Salience: 0.75 ███████░░░ (Medium-High)   │        │
│  │   Beliefs:                                   │        │
│  │     • I'm the BEST anklebiter...      [x]   │        │
│  │     • Competition is life...          [x]   │        │
│  │     • Sass and roasts are...          [x]   │        │
│  │     • Being nice is for losers        [x]   │        │
│  │   [+ Add Belief]                             │        │
│  │                                              │        │
│  │ [+ Add Transistor ▼]                        │        │
│  └─────────────────────────────────────────────┘        │
│                                                           │
│  [Reset to Defaults]  [Update]  [Duplicate]  [Delete]   │
└─────────────────────────────────────────────────────────┘
```

### Interaction Flow

**1. Selecting Recipe**
- Click recipe in list
- Inspector shows all editable fields
- Transistors shown as collapsible sections

**2. Editing Transistor Salience**
- Drag slider (0.0 to 1.0)
- Visual feedback: color-coded bar
  - 0.0-0.3: Gray (Low)
  - 0.3-0.6: Yellow (Medium)
  - 0.6-0.85: Orange (High)
  - 0.85-1.0: Red (Very High)
- Label updates: "0.85 (High emotional reactivity)"

**3. Editing Traits (PersonalityTransistor)**
- Each trait has slider
- [+ Add Trait] button opens dropdown of available traits
- [x] button removes trait

**4. Editing Beliefs (CulturalTransistor)**
- Text area for each belief
- [x] button removes belief
- [+ Add Belief] button adds new text field

**5. Adding Transistors**
- [+ Add Transistor] dropdown shows available types:
  - AffectTransistor
  - PersonalityTransistor
  - CulturalTransistor
  - IntuitionTransistor
  - MoodTransistor
  - MemoryTransistor
  - SocialExpectationTransistor
  - DeceptionTransistor
- Select type → adds with default values

**6. Removing Transistors**
- [−] button next to transistor name
- Confirmation dialog: "Remove AffectTransistor?"

**7. Saving Changes**
- [Update] button writes changes to YAML
- Visual feedback: "Recipe updated"
- NO effect on already-spawned instances

**8. Reset to Defaults**
- [Reset to Defaults] button
- Confirmation: "Revert all changes?"
- Restores original YAML from version control or backup

**9. Duplicate Recipe**
- [Duplicate] button
- Dialog: "New recipe name: _____"
- Creates copy with "(Copy)" suffix
- Opens duplicate in inspector

**10. Delete Recipe**
- [Delete] button
- Confirmation: "Delete red_fire_anklebiter.yaml?"
- Moves to trash or deletes file

---

## Technical Implementation

### Backend API Endpoints

**GET /api/recipes**
```json
{
  "recipes": [
    {
      "filename": "red_fire_anklebiter.yaml",
      "name": "Red Fire Anklebiter",
      "species": "gremlin",
      "has_cognitive_components": true
    }
  ]
}
```

**GET /api/recipes/{filename}**
```json
{
  "filename": "red_fire_anklebiter.yaml",
  "content": {
    "name": "Red Fire Anklebiter",
    "species": "gremlin",
    "description": "...",
    "personality": {...},
    "appetites": {...},
    "cognitive_components": {
      "affect": {
        "type": "AffectTransistor",
        "salience": 0.85,
        "enabled": true
      },
      "personality": {
        "type": "PersonalityTransistor",
        "traits": {
          "aggression": 0.90,
          "competitiveness": 0.95
        },
        "salience": 0.80
      }
    },
    "constraints": {...}
  }
}
```

**PUT /api/recipes/{filename}**
```json
{
  "content": {
    "name": "Red Fire Anklebiter",
    "cognitive_components": {
      "affect": {
        "salience": 0.90  // Updated value
      }
    }
  }
}
```
Response: `{"success": true, "message": "Recipe updated"}`

**POST /api/recipes**
```json
{
  "source_filename": "red_fire_anklebiter.yaml",
  "new_filename": "purple_fire_anklebiter.yaml",
  "modifications": {
    "name": "Purple Fire Anklebiter"
  }
}
```
Response: `{"success": true, "filename": "purple_fire_anklebiter.yaml"}`

**DELETE /api/recipes/{filename}**
Response: `{"success": true, "message": "Recipe deleted"}`

### File Management

**Recipe Location**: `applications/cmush/recipes/*.yaml`

**Backup Strategy**:
- On first edit, create backup: `red_fire_anklebiter.yaml.backup`
- Reset to Defaults loads from backup
- Git tracks all changes (version control)

**Validation**:
- Schema validation before save
- Required fields: name, species, identity_prompt
- Transistor type must exist in COMPONENT_REGISTRY
- Salience must be 0.0-1.0
- Traits must be float 0.0-1.0

---

## Default Transistor Values

When adding new transistor via UI, use these defaults:

**AffectTransistor**:
```yaml
affect:
  type: "AffectTransistor"
  salience: 0.70  # Human-typical
  enabled: true
```

**PersonalityTransistor**:
```yaml
personality:
  type: "PersonalityTransistor"
  traits:
    openness: 0.5
    conscientiousness: 0.5
    extraversion: 0.5
    agreeableness: 0.5
    neuroticism: 0.5
  salience: 0.60
```

**CulturalTransistor**:
```yaml
cultural:
  type: "CulturalTransistor"
  beliefs: []
  salience: 0.80
```

**IntuitionTransistor**:
```yaml
intuition:
  type: "IntuitionTransistor"
  salience: 0.75
  # intuition_text is dynamic, not in recipe
```

**MoodTransistor**:
```yaml
mood:
  type: "MoodTransistor"
  salience: 0.50
```

---

## UI Components (Qt/QML)

### RecipeBrowser.qml
- ListView of recipes
- Search/filter
- New/duplicate/delete actions

### RecipeInspector.qml
- Tabbed interface:
  - Identity tab
  - Cognitive Components tab
  - Constraints tab
- Update/Reset buttons

### TransistorEditor.qml
- Collapsible section per transistor
- Salience slider with color coding
- Type-specific editors:
  - PersonalityEditor (trait sliders)
  - CulturalEditor (belief list)
  - AffectEditor (salience only)

### SalienceSlider.qml
- Custom slider 0.0-1.0
- Color gradient background
- Label with interpretation
- Snap to common values (0.15, 0.50, 0.70, 0.85)

---

## Example Use Cases

### Use Case 1: Making Red Fire Nastier

1. Select `red_fire_anklebiter.yaml`
2. Expand AffectTransistor
3. Drag salience slider: 0.85 → 0.95
4. Expand PersonalityTransistor
5. Drag aggression: 0.90 → 0.98
6. Click [Update]
7. Spawn new Red Fire → now ULTRA nasty

### Use Case 2: Creating Purple Fire Variant

1. Select `red_fire_anklebiter.yaml`
2. Click [Duplicate]
3. Name: "Purple Fire Anklebiter"
4. Change species: "gremlin" → "gremlin"
5. Change description: "crimson flame" → "violet flame"
6. Adjust AffectTransistor salience: 0.85 → 0.60 (calmer)
7. Edit beliefs: "Competition is life" → "Cooperation is strength"
8. Click [Update]
9. New recipe created

### Use Case 3: Reset After Bad Edit

1. Select `red_fire_anklebiter.yaml`
2. Accidentally set all salience to 0.0
3. Spawn Red Fire → completely flat, no personality
4. Click [Reset to Defaults]
5. Confirm reset
6. All values restored from backup

---

## Future Enhancements

### Phase 2: Recipe Templates

Pre-made templates for common archetypes:
- Vulcan Template (low affect, high logic)
- Empath Template (high affect, high empathy)
- Trickster Template (high mischief, low rules)
- Guardian Template (high duty, low spontaneity)

User selects template → fills in name/description → instant recipe

### Phase 3: Live Preview

Spawn temporary instance in preview mode:
- Test recipe changes in real-time
- Interact with preview character
- See transistor outputs in debug panel
- Don't save to world state

### Phase 4: Character Arc Presets

Define salience trajectories in recipe:
```yaml
cognitive_components:
  affect:
    type: "AffectTransistor"
    initial_salience: 0.95
    arc:
      type: "exponential_decay"
      target_salience: 0.40
      half_life: 20  # interactions
```

UI shows arc graph, editable curve.

### Phase 5: Version Control Integration

- Recipe diff viewer (show changes)
- Commit recipe changes with message
- Revert to previous version
- Branch/merge recipes for experiments

---

## Implementation Plan

### Week 1: Backend API
- [ ] Recipe CRUD endpoints
- [ ] YAML validation
- [ ] Backup system
- [ ] Test coverage

### Week 2: UI Components
- [ ] RecipeBrowser component
- [ ] RecipeInspector component
- [ ] TransistorEditor component
- [ ] SalienceSlider component

### Week 3: Integration
- [ ] Wire backend to UI
- [ ] Error handling
- [ ] User feedback (toasts, confirmations)
- [ ] Polish and testing

### Week 4: Documentation
- [ ] User guide
- [ ] Video tutorial
- [ ] Recipe best practices
- [ ] Example recipes

---

## Success Criteria

Recipe Prefab System is complete when:

1. User can edit recipe in GUI (no YAML editing)
2. Changes update immediately on next spawn
3. Duplicate/delete recipes work reliably
4. Reset to Defaults restores original values
5. Validation prevents invalid recipes
6. UI is intuitive (< 5 min to learn)
7. Performance is fast (< 100ms to load recipe)

---

## Open Questions

1. **Recipe versioning**: How to handle recipe format changes?
2. **Live instance updates**: Should we allow updating already-spawned instances?
3. **Recipe marketplace**: Share recipes with community?
4. **AI-assisted authoring**: "Create a mischievous fox character" → auto-generates recipe?

---

**End of Specification**

This system transforms recipe authoring from text-file editing to visual prefab creation, enabling rapid character iteration without code changes.
