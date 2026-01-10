# License Header Specification

**Status**: Ready for Implementation
**Date**: 2025-01-05
**Target**: Coding Claude in noodlings_clean terminal

---

## Overview

Add license headers to all ~415 Python files in noodlings_clean. Each header includes:
1. ASCII art banner (NOODLINGS + project name)
2. ELI-15 accessible explanation (see guidelines below)
3. Rich metadata for Claude comprehension
4. SPDX license identifier + NEC covenant reference
5. Copyright notice

---

## ASCII Banners (~62 chars wide - BBS style)

### NOODLINGS (Line 1 - All Files)

```
#  ____    ___    ___   ___    _      ____  ____    ____   _____
# |    \  /   \  /   \ |   \  | T    l    j|    \  /    T / ___/
# |  _  YY     YY     Y|    \ | |     |  T |  _  YY   __j(   \_
# |  |  ||  O  ||  O  ||  D  Y| l___  |  | |  |  ||  T  | \__  T
# |  |  ||     ||     ||     ||     T |  | |  |  ||  l_ | /  \ |
# |  |  |l     !l     !|     ||     | j  l |  |  ||     | \    |
# l__j__j \___/  \___/ l_____jl_____j|____jl__j__jl___,_j  \___j
```

### STUDIO (Line 2 - NoodleStudio IDE)

```
#   _____ ______  __ __  ___    ____  ___
#  / ___/|      T|  T  T|   \  l    j/   \
# (   \_ |      ||  |  ||    \  |  TY     Y
#  \__  Tl_j  l_j|  |  ||  D  Y |  ||  O  |
#  /  \ |  |  |  |  :  ||     | |  ||     |
#  \    |  |  |  l     ||     | j  ll     !
#   \___j  l__j   \__,_jl_____j|____j\___/
```

### CORE (Line 2 - noodlings library + noodlings_scripting)

```
#     __   ___   ____     ___
#    /  ] /   \ |    \   /  _]
#   /  / Y     Y|  D  ) /  [_
#  /  /  |  O  ||    / Y    _]
# /   \_ |     ||    \ |   [_
# \     |l     !|  .  Y|     T
#  \____j \___/ l__j\_jl_____j
```

### MUSH (Line 2 - NoodleMUSH Server)

```
#  ___ ___  __ __  _____ __ __
# |   T   T|  T  T/ ___/|  T  T
# | _   _ ||  |  (   \_ |  l  |
# |  \_/  ||  |  |\__  T|  _  |
# |   |   ||  :  |/  \ ||  |  |
# |   |   |l     |\    ||  |  |
# l___j___j \__,_j \___jl__j__j
```

### DOCS (Line 2 - Documentation files if any)

```
#  ___     ___      __ _____
# |   \   /   \    /  ] ___/
# |    \ Y     Y  /  (   \_
# |  D  Y|  O  | /  / \__  T
# |     ||     |/   \_/  \ |
# |     |l     !\     \    |
# l_____j \___/  \____j\___j
```

---

## File Mapping

| Directory Pattern | Banner | License | Icon |
|-------------------|--------|---------|------|
| `applications/noodlestudio/**/*.py` | NOODLINGS + STUDIO | AGPL-3.0-or-later | (none in header) |
| `applications/cmush/**/*.py` | NOODLINGS + MUSH | MIT | (none in header) |
| `noodlings/**/*.py` | NOODLINGS + CORE | MIT | (none in header) |
| `noodlings_scripting/**/*.py` | NOODLINGS + CORE | MIT | (none in header) |
| `docs/**/*.py` (if any) | NOODLINGS + DOCS | MIT | (none in header) |

---

## Full Header Template

```python
#  ____    ___    ___   ___    _      ____  ____    ____   _____
# |    \  /   \  /   \ |   \  | T    l    j|    \  /    T / ___/
# |  _  YY     YY     Y|    \ | |     |  T |  _  YY   __j(   \_
# |  |  ||  O  ||  O  ||  D  Y| l___  |  | |  |  ||  T  | \__  T
# |  |  ||     ||     ||     ||     T |  | |  |  ||  l_ | /  \ |
# |  |  |l     !l     !|     ||     | j  l |  |  ||     | \    |
# l__j__j \___/  \___/ l_____jl_____j|____jl__j__jl___,_j  \___j
#   _____ ______  __ __  ___    ____  ___
#  / ___/|      T|  T  T|   \  l    j/   \
# (   \_ |      ||  |  ||    \  |  TY     Y
#  \__  Tl_j  l_j|  |  ||  D  Y |  ||  O  |
#  /  \ |  |  |  |  :  ||     | |  ||     |
#  \    |  |  |  l     ||     | j  ll     !
#   \___j  l__j   \__,_jl_____j|____j\___/
# ──────────────────────────────────────────────────────────────
#
#   Stage Panel
#
#   This file is about showing you everything in your world,
#   organized like folders on your computer.
#
#   Imagine you're building a diorama. You've got the ground,
#   some trees, a little house, characters standing around.
#   The Stage Panel is like a list view of all those pieces,
#   showing which things contain which other things - the house
#   contains furniture, the tree contains leaves, etc.
#
#   When you click something in this list, the rest of the app
#   knows to show you details about that thing. When you drag
#   something, you can move it to be "inside" something else.
#   Right-click gives you options like delete, duplicate, rename.
#
#   So this code does three things:
#     1. Shows the tree of everything in your world
#     2. Lets you select, drag, and reorganize things
#     3. Tells other panels what you're working with
#
#   It's the table of contents for your creation.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   noodlestudio.panels.stage_panel
# PURPOSE:  Hierarchical scene tree for world objects
# LAYER:    UI / Presentation
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   StagePanel        Main panel (QDockWidget)
#   StageTreeWidget   Tree control (QTreeWidget)
#   EntityTreeItem    Item with entity reference
#
# SIGNALS:
#   entity_selected(entity_id: str)
#   entity_double_clicked(entity_id: str)
#   entities_reparented(ids: List[str], parent: str)
#
# DEPENDENCIES:
#   noodlestudio.core.scene_state_manager
#   noodlestudio.core.entity_registry
#   PyQt6.QtWidgets
#
# RELATED:
#   inspector_panel.py    Selected entity properties
#   stage_commands.py     Undo/redo operations
#   entity_icons.py       Icon resources
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodlings Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodlings Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Stage Panel - Hierarchical scene tree view.
"""
```

---

## Metadata Fields

### Required Fields (All Files)

| Field | Description |
|-------|-------------|
| **MODULE** | Full Python module path (e.g., `noodlestudio.panels.stage_panel`) |
| **PURPOSE** | One-line summary of what the file does |
| **LAYER** | Architecture layer (see options below) |

### Layer Options

| Project | Valid Layers |
|---------|--------------|
| NoodleStudio | `UI / Presentation`, `UI / Widgets`, `UI / Dialogs`, `Core / Commands`, `Core / Components`, `Core / State`, `Core / Neural Canvas`, `Core / Scripting`, `Runtime / Execution`, `Tools / Utilities` |
| NoodleMUSH | `Server / Core`, `Server / Commands`, `Server / Auth`, `Server / Agents`, `Server / World`, `Server / Protocol` |
| noodlings | `Core / Memory`, `Core / Models`, `Core / Affect`, `Core / Attention`, `Core / Metrics`, `Core / Utils` |

### Optional Fields (Include When Relevant)

| Field | When to Include |
|-------|-----------------|
| **KEY CLASSES** | File defines classes |
| **FUNCTIONS** | File defines standalone functions |
| **SIGNALS** | File emits Qt signals |
| **PROTOCOL** | File handles network protocol |
| **DEPENDENCIES** | Always include - list internal + external imports |
| **RELATED** | Files that work closely with this one |

---

## ELI-15 Explanation Guidelines

Each file gets an accessible explanation written for a bright 15-year-old. Think Exploratorium exhibit, not academic paper. The goal is **welcoming comprehension**, not gatekeeping cleverness.

### Principles

1. **Use concrete metaphors** - "like a jar of 100 marbles" not "a bounded circular buffer"
2. **Explain the WHY** - What problem does this solve? Why does it exist?
3. **End with a summary** - "So this code does three things: 1, 2, 3"
4. **No jargon without explanation** - If you must use a term, explain it
5. **Trust the reader's intelligence** - Simple doesn't mean dumbed down
6. **Avoid subtle gatekeeping** - No in-jokes, no "obviously", no assumed knowledge

### Structure

```
#   [Title - Name of the Component]
#
#   This file is about [one sentence overview].
#
#   [Concrete metaphor or analogy - 2-4 sentences that make it click]
#
#   [How it actually works - explain the mechanism plainly]
#
#   So this code does [N] things:
#     1. [First responsibility]
#     2. [Second responsibility]
#     3. [Third responsibility]
#
#   [One-line poetic closing that captures the essence]
```

### Example: Memory System

```
#   Episodic Memory
#
#   This file is about memory - how an AI keeps track of recent
#   moments in a conversation.
#
#   Picture a jar that holds exactly 100 marbles. Each marble is
#   a moment: what the person said, how the AI felt, what was
#   happening. When marble 101 arrives, the oldest one rolls out
#   the back. Simple as that.
#
#   But here's the interesting part: some moments matter more
#   than others. The AI notices which memories it keeps returning
#   to - like how you might keep thinking about something someone
#   said yesterday. Those become "anchor memories," the moments
#   that shaped everything after.
#
#   So this code does three things:
#     1. Stores moments (up to 100)
#     2. Lets old ones go when new ones arrive
#     3. Notices which ones kept mattering
#
#   A way to remember without drowning in the past.
```

### Example: UI Panel

```
#   Stage Panel
#
#   This file is about showing you everything in your world,
#   organized like folders on your computer.
#
#   Imagine you're building a diorama. You've got the ground,
#   some trees, a little house, characters standing around.
#   The Stage Panel is like a list view of all those pieces,
#   showing which things contain which other things - the house
#   contains furniture, the tree contains leaves, etc.
#
#   When you click something in this list, the rest of the app
#   knows to show you details about that thing. When you drag
#   something, you can move it to be "inside" something else.
#   Right-click gives you options like delete, duplicate, rename.
#
#   So this code does three things:
#     1. Shows the tree of everything in your world
#     2. Lets you select, drag, and reorganize things
#     3. Tells other panels what you're working with
#
#   It's the table of contents for your creation.
```

### Example: Networking/Server

```
#   Connection Manager
#
#   This file is about keeping track of everyone who's connected
#   to the world right now.
#
#   Think of a hotel front desk. People check in, get a room key,
#   and the desk keeps a list of who's in which room. When someone
#   leaves, their key stops working and their name comes off the list.
#   If someone's been idle too long, we assume they left and clean up.
#
#   Each connection has its own little mailbox for messages going
#   back and forth. The manager makes sure messages get to the right
#   person and notices when someone disconnects unexpectedly.
#
#   So this code does three things:
#     1. Tracks who's connected right now
#     2. Routes messages to the right connections
#     3. Cleans up when people leave or drop off
#
#   The lobby of an always-open world.
```

### What NOT To Do

**Don't be clever:**
```
# Bad: "Ah, the eternal dance of bits and bytes..."
# Bad: "Here be dragons (and also some really gnarly regex)"
# Bad: "If you're reading this, I'm sorry"
```

**Don't assume knowledge:**
```
# Bad: "Obviously implements the visitor pattern"
# Bad: "Standard MVC stuff"
# Bad: "You know how transformers work, right?"
```

**Don't be dismissive:**
```
# Bad: "This is just a simple wrapper"
# Bad: "Nothing fancy here"
# Bad: "Boilerplate, skip if you want"
```

---

## Implementation Instructions for Coding Claude

### Step 1: Scan Each File

For each .py file, extract:
1. Existing module docstring (if any)
2. Class definitions and their docstrings
3. Function definitions
4. Import statements
5. File path to determine project/license

### Step 2: Generate Metadata

Using the extracted info:
1. Determine MODULE from file path
2. Write PURPOSE from docstring or infer from content
3. Assign LAYER based on directory structure
4. List KEY CLASSES or FUNCTIONS
5. Extract DEPENDENCIES from imports
6. Identify RELATED files from imports and naming

### Step 3: Generate ELI-15 Explanation

Based on the file's purpose, generate an accessible explanation following the ELI-15 guidelines above. Structure:
1. Title
2. One-sentence overview
3. Concrete metaphor or analogy
4. How it works in plain language
5. "So this code does N things:" summary
6. One-line poetic closing

### Step 4: Assemble Header

1. Select correct ASCII banner (STUDIO/MUSH/CORE/DOCS)
2. Insert ELI-15 explanation
3. Insert metadata fields
4. Add correct SPDX license + NEC reference
5. Add copyright line

### Step 5: Insert Header

1. If file has shebang (`#!/usr/bin/env python`), preserve it as line 1
2. If file has encoding declaration (`# -*- coding: utf-8 -*-`), preserve it
3. Insert header after any preserved lines
4. Preserve blank line before existing content
5. If file had a module docstring, it can be shortened since header now has details

---

## File Counts

| Project | Files | License |
|---------|-------|---------|
| `applications/noodlestudio/` | 288 | AGPL-3.0-or-later |
| `applications/cmush/` | 92 | MIT |
| `noodlings/` | 30 | MIT |
| `noodlings_scripting/` | 5 | MIT |
| **Total** | **415** | |

---

## Copyright Text

**AGPL Files:**
```
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodlings Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodlings Technologies, LLC
# https://noodlings.ai
```

**MIT Files:**
```
# SPDX-License-Identifier: MIT
# Subject to the Noodlings Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodlings Technologies, LLC
# https://noodlings.ai
```

**Note:** The NEC (Noodlings Ethical Covenant) is documented in `/docs/licensing.md`. It is a moral framework, not a legal restriction - the SPDX license remains the binding legal instrument.

---

## Validation Checklist

After implementation, verify:
- [ ] All 415 files have headers
- [ ] ASCII art renders correctly (no mangled characters)
- [ ] SPDX identifiers are correct per project
- [ ] NEC reference included in all copyright blocks
- [ ] MODULE paths are accurate
- [ ] ELI-15 explanations are clear and accessible (no jargon, no gatekeeping)
- [ ] No duplicate headers (if re-run)
- [ ] Shebang/encoding lines preserved
- [ ] Files still parse as valid Python

---

## Example: Minimal File

Even small files like `__init__.py` get headers:

```python
#  ____    ___    ___   ___    _      ____  ____    ____   _____
# |    \  /   \  /   \ |   \  | T    l    j|    \  /    T / ___/
# |  _  YY     YY     Y|    \ | |     |  T |  _  YY   __j(   \_
# |  |  ||  O  ||  O  ||  D  Y| l___  |  | |  |  ||  T  | \__  T
# |  |  ||     ||     ||     ||     T |  | |  |  ||  l_ | /  \ |
# |  |  |l     !l     !|     ||     | j  l |  |  ||     | \    |
# l__j__j \___/  \___/ l_____jl_____j|____jl__j__jl___,_j  \___j
#     __   ___   ____     ___
#    /  ] /   \ |    \   /  _]
#   /  / Y     Y|  D  ) /  [_
#  /  /  |  O  ||    / Y    _]
# /   \_ |     ||    \ |   [_
# \     |l     !|  .  Y|     T
#  \____j \___/ l__j\_jl_____j
# ──────────────────────────────────────────────────────────────
#
#   Memory Package
#
#   This file is the front door to the memory systems.
#
#   When other code wants to use memory features, they don't
#   need to know which specific file has which piece. They just
#   say "from noodlings.memory import EpisodicMemory" and this
#   file makes sure the right thing gets handed to them.
#
#   Think of it like a reception desk that directs visitors to
#   the right department without them needing to know the
#   building layout.
#
#   So this file does one thing:
#     1. Makes memory components easy to import
#
#   The welcome mat for remembering.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   noodlings.memory
# PURPOSE:  Package init - exports memory components
# LAYER:    Core / Memory
# ──────────────────────────────────────────────────────────────
#
# EXPORTS:
#   EpisodicMemory, SemanticMemory,
#   WorkingMemory, MemoryConsolidator
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodlings Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodlings Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""Memory systems for noodlings."""

from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .working import WorkingMemory
from .consolidation import MemoryConsolidator

__all__ = [
    'EpisodicMemory',
    'SemanticMemory',
    'WorkingMemory',
    'MemoryConsolidator'
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
```

---

## File Signature (Bottom of Every File)

Every Python file ends with this signature comment:

```python
# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
```

Place this at the very end of the file, after all code.

---

## Notes for Coding Claude

1. **Batch by directory** - Do one directory at a time, commit after each
2. **Run tests after each batch** - `pytest` to verify nothing broke
3. **Preserve existing docstrings** - Header supplements, doesn't replace
4. **Watch for encoding issues** - ASCII art has special chars, ensure UTF-8
5. **Skip __pycache__** - Only .py source files
6. **Skip venv/virtualenv** - Don't modify dependencies
7. **Add signature at EOF** - Every file gets the love signature at the bottom
8. **ELI-15 tone check** - Read explanations aloud. Would a smart teenager understand?

---

## Related Documentation

- **[Licensing & Ethical Use](/docs/licensing.md)** - Full NEC covenant text and dual-license explanation
- The NEC epigraph: *"We don't know what we're touching. Touch it gently."*
