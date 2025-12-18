# Spatial Visualization Design Document

**Version:** 1.0.0
**Date:** December 18, 2025

This document describes the Qt Quick 3D spatial visualization system for NoodleStudio.

---

## Core Insight

**Text, 2D maps, 3D views, and detailed geometry are all projections of the same semantic truth.**

The zone YAML files are canonical. Everything else - MUD text, 2D mini-maps, 3D bubble views, full USD renders - is a view of that data.

---

## Levels of Detail

The spatial visualization supports progressive levels of detail:

| Level | Name | Description |
|-------|------|-------------|
| 1 | Graph View | Metro-map style topology diagram (nodes and edges) |
| 2 | Bubble View | Colored spheres at zone positions with connection lines |
| 3 | Iconic View | Simple 3D shapes representing zone types (hamburger, tree, etc.) |
| 4 | Full 3D | Complete USD/glTF geometry with lighting and materials |

The current implementation provides **Level 2: Bubble View**.

---

## Zone Data Model

Each zone has:

```yaml
spatial:
  center: [x, y, z]      # Position in 3D space (meters)
  radius: 15.0           # Primary attention radius
  falloff: 10.0          # Soft edge falloff distance
  shape: "sphere"        # sphere, cylinder, box

text:
  exits:                 # Connections to other zones
    north: "forest_edge"
    east: "pond"
```

Zones connect via the `exits` field in their YAML, or via the `zone_graph` in `stage.yaml`.

---

## Qt Quick 3D Implementation

### Architecture

```
SpatialViewPanel (Python QWidget)
    |
    +-- ZoneModel (QObject - data bridge)
    |       |
    |       +-- load_stage(path) -> parses zone YAMLs
    |       +-- getZones() -> list of zone dicts for QML
    |       +-- getConnections() -> list of connection dicts
    |       +-- selectZone(id) -> handles click selection
    |
    +-- QQuickWidget
            |
            +-- SpatialView.qml
                    |
                    +-- View3D (3D scene)
                    |       +-- PerspectiveCamera with orbit controls
                    |       +-- DirectionalLight (key + fill)
                    |       +-- Ground grid (Rectangle)
                    |       +-- Repeater3D with Sphere delegates
                    |
                    +-- Canvas (2D connection line overlay)
                    +-- MouseArea (orbit, pan, zoom, click)
```

### Controls

| Action | Mouse | Result |
|--------|-------|--------|
| Orbit | Right-drag | Rotate camera around center |
| Pan | Middle-drag | Move camera laterally |
| Zoom | Scroll wheel | Move camera closer/further |
| Select | Click | Select zone, emit signal |

---

## File Locations

| File | Purpose |
|------|---------|
| `panels/spatial_view_panel.py` | Python panel class and ZoneModel |
| `qml/SpatialView.qml` | Qt Quick 3D scene (auto-generated) |

---

## Integration

The SpatialViewPanel:

1. Receives project_manager reference from MainWindow
2. Populates stage selector from project_manager.list_stages()
3. Loads zones via zone_model.load_stage(stage_path)
4. Emits zoneSelected(zone_id, zone_data) when user clicks a zone
5. MainWindow can route this to Inspector for zone editing

---

## Future Enhancements

### Level 3: Iconic View
- Load iconic 3D shapes per zone type
- Map zone.shape to different primitives
- Support custom icons via zone.icon field

### Level 4: Full 3D
- Load geometry.usda from stage folder
- Position USD prims at zone.center
- Integrate with existing USD pipeline

### Additional Features
- Zone labels (billboard text above spheres)
- Agent instance markers (smaller spheres for noodlings)
- Prop markers (cube icons)
- Real-time position updates via WebSocket
- Camera bookmarks
- First-person walkthrough mode

---

## Color Palette

The default zone palette uses warm earth tones and cool accents:

| Index | Color | Name |
|-------|-------|------|
| 0 | #4A90A4 | Teal |
| 1 | #E07B53 | Coral |
| 2 | #8B7355 | Brown |
| 3 | #6B8E4E | Olive |
| 4 | #9B6B9E | Purple |
| 5 | #C4A35A | Gold |
| 6 | #5B8BA0 | Steel Blue |
| 7 | #A0522D | Sienna |
| 8 | #708090 | Slate |
| 9 | #BC8F8F | Rosy Brown |
| 10 | #5F9EA0 | Cadet Blue |
| 11 | #D2691E | Chocolate |

Colors are assigned in order of zone loading (alphabetical by filename).
