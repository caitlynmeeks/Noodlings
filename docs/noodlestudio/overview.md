# NoodleStudio

Desktop IDE for designing cognitive architectures and building worlds.

---

## What is NoodleStudio?

NoodleStudio is a PyQt6 desktop application for:

- **Designing Noodlings** - Visual cognitive architecture editing
- **Building Worlds** - Stages, zones, props, spatial relationships
- **Managing Assets** - Gaussian splats, VRM avatars, scripts
- **Running Servers** - Integrated NoodleMUSH control

Think of it as Unity or Blender, but for cognition.

## Panels

| Panel | Purpose |
|-------|---------|
| **Stage View** | Scene hierarchy (zones, noodlings, props) |
| **Assets** | Project file browser |
| **Inspector** | Properties of selected entity |
| **Facets Editor** | Cognitive architecture node graph |
| **Neural Canvas** | Advanced visual programming |
| **Chat** | Talk to the world |
| **Gaussian Viewer** | 3D radiance preview |

## Requirements

- macOS 13+ (Apple Silicon recommended)
- Python 3.10+
- PyQt6

## Launch

```bash
cd applications/noodlestudio
./launch_with_log.sh
```

Or from NoodleStudio settings, enable "Auto-start MUSH server" for integrated operation.

## Projects

NoodleStudio organizes work into **projects**:

```
MyProject/
├── project.yaml
├── Noodlings/
├── Stages/
├── Prims/
└── Assets/
```

## Next

- [Panels Reference](panels.md) - All panels explained
- [Scripting](scripting.md) - JavaScript API
- [Neural Canvas](neural-canvas.md) - Visual programming
