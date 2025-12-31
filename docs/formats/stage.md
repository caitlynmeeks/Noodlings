# Stage Format

YAML format for world scenes.

---

## Overview

`stage.yaml` defines a world scene: zones (rooms), their connections,
props, and spawned Noodlings.

## Location

```
Stages/
└── my_stage/
    ├── stage.yaml      # This file
    ├── hierarchy.yaml  # Scene hierarchy (auto-generated)
    └── zones/
        ├── nexus.yaml
        └── garden.yaml
```

## Stage Schema

```yaml
name: "The Nexus"
description: "A hub connecting multiple realms"

# Default spawn point
spawn_zone: nexus

# Zones in this stage
zones:
  - nexus
  - garden
  - library

# Global stage settings
settings:
  ambient_light: 0.3
  time_scale: 1.0
```

## Zone Schema

`zones/nexus.yaml`:

```yaml
id: nexus
name: "The Nexus"
description: |
  A shimmering crossroads where paths converge.
  Portals flicker with distant destinations.

# Connections to other zones
exits:
  north: garden
  east: library
  portal: otherworld  # Can be any name

# Props in this zone
props:
  - id: fountain_1
    prim: prims/fountain.yaml
    position: [0, 0, 0]
    rotation: [0, 0, 0]

# Noodlings that spawn here by default
residents:
  - red
  - yuki
```

## Hierarchy File

`hierarchy.yaml` stores the user's organizational structure in NoodleStudio
(folders, custom ordering). Auto-generated, don't edit manually.

```yaml
nodes:
  - id: root
    type: folder
    name: Stage
    children:
      - id: zones_folder
        type: folder
        name: Zones
        children:
          - id: nexus
            type: zone
          - id: garden
            type: zone
```

## Loading in Code

```python
import yaml

with open("Stages/my_stage/stage.yaml") as f:
    stage = yaml.safe_load(f)

for zone_name in stage["zones"]:
    with open(f"Stages/my_stage/zones/{zone_name}.yaml") as f:
        zone = yaml.safe_load(f)
        print(f"{zone['name']}: {zone['description'][:50]}...")
```
