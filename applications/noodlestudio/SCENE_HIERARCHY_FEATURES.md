# Scene Hierarchy Features - Unity-Style

**Status**: Complete (November 18, 2025)

## Overview

NoodleStudio's Scene Hierarchy now has full Unity-style interaction:
- Right-click context menus
- Drag-and-drop parenting
- Create menu with specialized prim types
- Philip Rosedale would be proud! 🎯

## Features Implemented

### 1. Right-Click Context Menu

**Context-aware actions** based on prim type:

#### For Noodlings:
- Inspect Properties
- Toggle Enlightenment
- Duplicate Noodling
- Reset State
- Delete Noodling

#### For Objects:
- Inspect Properties
- Edit Description
- Duplicate Object
- Delete Object

#### For Users:
- Inspect Properties
- View Profile

#### For Exits:
- Edit Exit
- Delete Exit

#### For Folders:
- Expand All
- Collapse All

#### Right-click empty space:
- **Create submenu** with:
  - Empty Noodling
  - Empty Object
  - Empty Room
  - Empty Prim (Custom)

### 2. Drag-and-Drop Parenting

Just like Unity GameObjects! You can:
- Drag prims onto each other to create parent/child relationships
- Reorder prims in the hierarchy
- Visual drop indicator shows where prim will land

**Implementation:**
```python
self.tree.setDragEnabled(True)
self.tree.setAcceptDrops(True)
self.tree.setDropIndicatorShown(True)
self.tree.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
```

### 3. Create Menu (Top Menu Bar)

**Create > Noodling:**
- Empty Noodling (default balanced personality: 0.5)
- Kitten Noodling (curious, playful)
- Robot Noodling (logical, calm)
- Dragon Noodling (intense, emotional)

**Create > Object:**
- Empty Object
- Prop (Holdable) - takeable, holdable
- Furniture (Sittable) - fixed, sittable
- Container (Openable) - openable, container

**Create > Other:**
- Empty Room
- Empty Prim (custom type)

## Specialized Noodling Presets

When you create a specialized Noodling, you get species-specific personality defaults:

### Kitten
- Extraversion: 0.7 (outgoing)
- Curiosity: 0.9 (very curious!)
- Impulsivity: 0.8 (spontaneous)
- Emotional Volatility: 0.6 (reactive)

### Robot
- Extraversion: 0.3 (reserved)
- Curiosity: 0.6 (analytical)
- Impulsivity: 0.2 (methodical)
- Emotional Volatility: 0.1 (stable)

### Dragon
- Extraversion: 0.6 (confident)
- Curiosity: 0.5 (balanced)
- Impulsivity: 0.4 (deliberate)
- Emotional Volatility: 0.7 (intense)

## Usage Examples

### Creating a Noodling (3 ways)

**1. Via top menu:**
```
Create > Noodling > Empty Noodling
```

**2. Via right-click in Scene Hierarchy:**
```
Right-click empty space > Create > Empty Noodling
```

**3. Via Entities menu:**
```
Entities > Add Noodling...
```

### Parenting Prims (Unity-style)

1. Drag a Noodling prim
2. Drop it onto a Room prim
3. Now the Noodling is a child of the Room!

This creates parent/child relationships in the stage hierarchy.

### Context Menu Actions

**Right-click a Noodling:**
- Choose "Inspect Properties" to see personality in Inspector
- Choose "Toggle Enlightenment" to switch character/enlightened mode
- Choose "Duplicate" to create a copy
- Choose "Delete" to remove (with confirmation dialog)

## USD Integration

When you export to USD, the hierarchy structure is preserved:

```usd
def Xform "Stage" {
    def Xform "Nexus" {
        def "Noodlings/phi" {
            # Phi as child of Nexus room
        }
    }
}
```

Parent/child relationships become prim hierarchy in USD!

## Keyboard Shortcuts

- **Ctrl+Shift+N** - Add Noodling (via Entities menu)
- **Ctrl+Shift+O** - Add Object (via Entities menu)
- **Delete** - Remove Selected (via Entities menu)

## Philip Rosedale Connection 🎯

The "prim" terminology comes from Second Life! Philip Rosedale (founder of Second Life) pioneered the concept of **prims** (primitives) as the building blocks of virtual worlds.

By using proper USD terminology AND respecting Second Life heritage, we're connecting:
- **Second Life prims** (Philip Rosedale, 2003)
- **USD prims** (Pixar, 2015)
- **Noodlings** (consciousness prims, 2025)

We're standing on the shoulders of giants! 🪙

## Next Steps (API Integration)

Currently, these actions print to console. To make them functional:

1. **Add noodleMUSH API endpoints:**
   - `POST /api/prims/create` - Create new prim
   - `DELETE /api/prims/{id}` - Delete prim
   - `PUT /api/prims/{id}/parent` - Set parent/child relationship
   - `PUT /api/prims/{id}/enlightenment` - Toggle enlightenment
   - `POST /api/prims/{id}/duplicate` - Duplicate prim

2. **Wire up Scene Hierarchy methods:**
   - Connect context menu actions to API calls
   - Refresh hierarchy after successful operations
   - Show error dialogs on failures

3. **Add undo/redo:**
   - Keep command history stack
   - Ctrl+Z / Ctrl+Shift+Z
   - Unity-style command pattern

## Files Modified

```
noodlestudio/panels/
└── scene_hierarchy.py
    - Added context menu system
    - Enabled drag-and-drop
    - Created prim actions (duplicate, delete, reset, etc.)

noodlestudio/core/
└── main_window.py
    - Added Create menu
    - Added specialized prim creators
    - Added personality presets
```

## Philosophy

We treat Noodlings like Unity treats GameObjects:
- **Scene hierarchy** for organization
- **Parent/child relationships** for structure
- **Context menus** for quick actions
- **Create menu** for rapid prototyping

But we use proper USD terminology:
- **Prims** not "entities" or "objects"
- **Stage** not "scene"
- **Layer** not "file"

This makes NoodleStudio feel familiar (Unity-like) while being professionally compatible with animation pipelines (USD-compliant).

**Best of both worlds!** 🎨✨

---

Built with the Krugerrand Edition 🪙
Philip Rosedale would approve of these prims!
