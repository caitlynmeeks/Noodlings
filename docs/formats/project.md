# Project Format

YAML format for NoodleStudio projects.

---

## Overview

`project.yaml` is the root file for a NoodleStudio project. It contains
project metadata and references to stages, settings, and assets.

## Location

```
MyProject/
├── project.yaml       # This file
├── Noodlings/
├── Stages/
├── Prims/
├── Scripts/
└── Assets/
```

## Schema

```yaml
name: "My Project"
version: "0.1.0"
description: "A world with interesting characters"

# Author info
author: "Your Name"
email: "you@example.com"

# Default stage to load
default_stage: "Stages/nexus"

# Project-wide settings
settings:
  # Server settings
  auto_start_mush: true
  mush_port: 8765

  # Default LLM configuration
  default_provider: ollama
  model_labels:
    thinking: ollama/llama3.2
    speaking: ollama/llama3.2
    perception: ollama/llama3.2

# Asset paths (relative to project root)
paths:
  noodlings: Noodlings/
  stages: Stages/
  prims: Prims/
  scripts: Scripts/
  assets: Assets/

# Creation metadata
created: "2025-12-30T12:00:00Z"
modified: "2025-12-30T12:00:00Z"
```

## Minimal Project

```yaml
name: "Minimal"
default_stage: "Stages/default"
```

## Creating a Project

In NoodleStudio: File > New Project

Or manually:

```bash
mkdir MyProject
cd MyProject
mkdir Noodlings Stages Prims Scripts Assets

cat > project.yaml << 'EOF'
name: "MyProject"
default_stage: "Stages/default"
EOF

mkdir -p Stages/default/zones
cat > Stages/default/stage.yaml << 'EOF'
name: "Default Stage"
spawn_zone: start
zones:
  - start
EOF

cat > Stages/default/zones/start.yaml << 'EOF'
id: start
name: "Starting Area"
description: "You are here."
exits: {}
props: []
EOF
```

## Loading in Code

```python
import yaml
from pathlib import Path

project_root = Path("MyProject")
with open(project_root / "project.yaml") as f:
    project = yaml.safe_load(f)

print(f"Project: {project['name']}")
print(f"Default stage: {project['default_stage']}")
```
