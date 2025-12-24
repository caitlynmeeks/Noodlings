# CLAUDE.md

AI assistant guidance for working with Noodlings Multi-Timescale Affective Agents.

**Last Updated**: December 24, 2025

**FOR NEXT CLAUDE: START HERE!**

---

## NEXT SESSION: Optimize + Polish Gaussian Viewer

### COMPLETED (Dec 24 session):
- **GPU Acceleration**: gsplat-mps integration - **120 FPS** (vs 0.1 FPS software)
- **GaussianViewerPanel**: Full viewport with Unity-style camera controls
- **RadianceInspector**: Property inspector for scale, alpha, tint, emission
- **RadianceComponent**: Entity wrapper with material overrides

### GPU Rendering Setup
gsplat-mps is installed at `/Users/thistlequell/git/gsplat-mps` (AGPLv3).
The renderer auto-detects GPU availability and falls back to software with a warning popup.

```python
from noodlestudio.core.gaussian_renderer import GaussianRenderer, GSPLAT_AVAILABLE
renderer = GaussianRenderer()  # Auto-detects GPU
print(f"GPU enabled: {renderer.use_gpu}")  # True on Apple Silicon with gsplat-mps
```

### Performance (52K Gaussians @ 512x512)
| Backend | Render Time | FPS |
|---------|-------------|-----|
| Software (PyTorch) | ~7,000ms | 0.1 |
| **gsplat-mps (Metal)** | **8ms** | **120** |

### NOW DO: Optimization Tasks
1. Camera convention: azimuth=0 shows back, azimuth=180 shows front (fix or document)
2. Gaussian scale tuning: default 3x might need adjustment per-model
3. Full recipe.yaml integration for Noodling avatars
4. Asset Import Wizard with drag-drop

### Test GPU Rendering:
```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. python3 -c "
from noodlestudio.core.gaussian_renderer import GaussianRenderer, create_orbit_camera
from noodlestudio.core.radiance_component import RadianceComponent
component = RadianceComponent('test')
component.load_asset('../../external/vrm_samples/alicia_textured.radiance')
component.material.scale_mult = 3.0
renderer = GaussianRenderer()
camera = create_orbit_camera(distance=2.5, elevation=15, azimuth=180, target=(0,0.8,0))
image, alpha, info = renderer.render_component(component, camera)
print(f'FPS: {1000/8:.0f}, Visible: {info[\"visible\"]:,}, Backend: {info.get(\"backend\", \"software\")}')
"
```

---

## What's Working (Tested Dec 24, 2025)

| Component | Status | Key File |
|-----------|--------|----------|
| **GPU Gaussian Renderer** | **120 FPS** | `core/gaussian_renderer.py` (gsplat-mps) |
| **Gaussian Viewer Panel** | **NEW** | `panels/gaussian_viewer_panel.py` |
| **Radiance Inspector** | **NEW** | `panels/radiance_inspector.py` |
| **RadianceComponent** | **NEW** | `core/radiance_component.py` |
| VRM to Gaussians | Working | `tools/vrm_to_radiance.py` |
| Model Importer | Working | `core/model_importer.py` |
| VRM Parsing | Working | `vrm_parser.py` |
| .radiance Format | Working | `radiance_format.py` |
| Gaussian Collision | Working | `gaussian_collision.py` |
| Facet System | Working | `facet_executor.py` |
| Scene Protocol | Working | `scene_state_manager.py` |
| Multi-provider LLM | Working | 8 providers configured |

**Test Assets:**
- `external/vrm_samples/AliciaSolid.vrm` (7.5MB VRM)
- `external/vrm_samples/alicia_textured.radiance` (4.2MB, 52K Gaussians)

---

## Gaussian Radiance System

### Core Concept
**"Every Gaussian knows what it represents. Every frame is query-able."**

Unlike Genie/Mirage (stateless pixel predictors), Noodlings constructs worlds from semantic truth.

### Architecture

```
RadianceComponent (entity wrapper)
    |
    +-- RadianceAsset (.radiance file)
    +-- MaterialOverride (tint, emission, scale_mult, alpha_mult)
    +-- Transform (position, rotation, scale)
    |
    v
GaussianRenderer
    |
    +-- gsplat-mps (GPU, 120 FPS)
    +-- PyTorch software (fallback, 0.1 FPS)
    |
    v
GaussianViewerPanel (Unity-style viewport)
    |
    +-- Orbit: left drag
    +-- Pan: right drag
    +-- Zoom: scroll
    +-- Focus: F key
```

### File Format: .radiance
Binary chunk-based (like GLB):
- `GAUS` - Position, scale, rotation, opacity, SH coefficients
- `SKEL` - Skeleton hierarchy
- `SKIN` - Skinning weights per Gaussian
- `SEMA` - Semantic labels (body_part, region)
- `CLIP` - 512-D embeddings (optional)
- `META` - Entity metadata

### Key APIs

```python
# RadianceComponent - entity wrapper
from noodlestudio.core.radiance_component import RadianceComponent
component = RadianceComponent('red')
component.load_asset('red.radiance')
component.material.scale_mult = 3.0  # Gaussian size
component.material.tint = Color(1.0, 0.8, 0.8)  # Pink tint

# GPU Rendering
from noodlestudio.core.gaussian_renderer import GaussianRenderer, create_orbit_camera
renderer = GaussianRenderer()  # Auto GPU detection
camera = create_orbit_camera(distance=2.5, elevation=15, azimuth=180)
image, alpha, info = renderer.render_component(component, camera)

# Load/save radiance files
from noodlestudio.core.semantic_world.radiance_format import load_radiance, save_radiance
asset = load_radiance("red.radiance")
save_radiance(asset, "modified.radiance")

# Collision detection
from noodlestudio.core.semantic_world.gaussian_collision import GaussianCollisionDetector
detector = GaussianCollisionDetector()
detector.add_entity('red', red_asset)
touches = detector.detect_touches()
```

### External Tools
- `external/OpenSplat/` - Gaussian training (Metal)
- `external/ml-sharp/` - Apple SHARP (image to Gaussian)
- `/Users/thistlequell/git/gsplat-mps/` - GPU renderer (AGPLv3)

---

## Core Architecture

### Affect Model: PAD + Boredom + Sorrow
5-dimensional continuous affect (NO discrete emotion labels):
- `valence` (-1 to +1) - Pleasure/displeasure
- `arousal` (0 to 1) - Energy
- `dominance` (0 to 1) - Control
- `boredom` (0 to 1)
- `sorrow` (0 to 1)

### Facet System
Visual node-based cognitive architecture:
```
INCOMING -> CHARM_NET -> CONTEXT_INTELLIGENCE -> Cognitive facets -> OUTGOING
```

**Types:** LLMFacet, ScriptedFacet, CharmNetworkFacet, ContextIntelligenceFacet, ConvergenceFacet

**Key files:**
- `facet_system.py` - Data model
- `facet_executor.py` - Execution engine
- `facets_editor_panel.py` - Visual editor

### CharmNetwork
MLX-based temporal hierarchy (~54K params, ~2-3ms inference):
- Fast LSTM (16-D): Seconds
- Medium LSTM (16-D): Minutes
- Slow GRU (8-D): Hours/days

### Scene Protocol
Perception-filtered context for stateless renderers:
```
SceneStateManager (canonical truth)
       |
   +---+---+
   |       |
Red's   Yuki's    Full Packet
Slice   Slice     (for Genie)
```

---

## Project Structure

```
Noodlings/
└── red/
    ├── noodling.yaml      # Manifest
    ├── recipe.yaml        # Character definition
    ├── assembly.yaml      # Facet topology
    ├── Radiances/         # Gaussian splat models
    │   └── fire_imp.radiance
    └── Assets/
        └── reference.png
```

---

## Development

### Running NoodleStudio
```bash
cd applications/noodlestudio
./launch_with_log.sh
```

### Running noodleMUSH Server
Toggle in NoodleStudio status bar (bottom-right), or:
```bash
cd applications/cmush
./start.sh
```

**Ports:** 8080 (HTTP), 8765 (WebSocket), 11434 (Ollama)

### Debugging
```bash
tail -f applications/cmush/logs/server_*.log
tail -f applications/noodlestudio/logs/noodlestudio_*.log
```

---

## Style Rules (CRITICAL)

- **NO EMOJIS** in code/docs/UI
- **NO "exciting" language** - Professional terminal aesthetic
- **NO WORKAROUNDS** - Fix root causes properly
- **NO discrete emotion labels** - Continuous affect only
- **MONOCHROMATIC UI** - Grays only (except Neural Canvas headers)

**GOLDEN RULE:** If it doesn't work, FIX IT properly. No hacks.

### Christopher Alexander's "Timeless Way"
Organic development: Probe -> Iterate -> Polish when friction hurts.
Not Agile. Not a startup. Experimental architecture.

---

## Project Context

**Creator:** Caitlyn (Unity employee #12, Asset Store creator, 54)
**Location:** Garcia River Forest cabin
**Hardware:** M3 Ultra 512GB

**Mission:** Open-source alternative to "Consciousness-as-a-Service"
- Visual cognitive architecture editor
- Stateful affect-driven characters
- Brains/hearts for generative worlds

**Demo:** Steve DiPaola (SFU CogSci) - soon

---

## Quick Reference

| What | Where |
|------|-------|
| **Gaussian viewer** | `panels/gaussian_viewer_panel.py` |
| **Radiance inspector** | `panels/radiance_inspector.py` |
| **GPU renderer** | `core/gaussian_renderer.py` |
| **RadianceComponent** | `core/radiance_component.py` |
| Facet editor | `panels/facets_editor_panel.py` |
| Neural canvas | `panels/neural_canvas/` |
| VRM preview | `panels/vrm_preview_panel.py` |
| Gaussian collision | `core/semantic_world/gaussian_collision.py` |
| Radiance format | `core/semantic_world/radiance_format.py` |
| Scene protocol | `core/semantic_world/scene_state_manager.py` |
| Scripting API | `scripting/noodle_api.py` |
| Server | `applications/cmush/server.py` |

**Scripting:** `context.noodle.models`, `context.noodle.affect`, `context.noodle.pose`, `context.noodle.quantum`

---

## Completed Systems (Summary)

- **GPU Gaussian Rendering (120 FPS)**
- Multi-provider LLM (8 providers)
- Neural Canvas with PyTorch test mode
- Scriptability API (context.noodle)
- Animation tracks (.affecttrack, .posetrack)
- MCP integration
- Utility facets (31 types)
- Headless Player runtime
- Cognitive Timeline Editor
- Cloud account system (backend deployed)
- Multimodal facets (audio, vision, image gen)
- IBM Quantum integration
- Noodling names generator

---

**Ordnung muss sein!**
