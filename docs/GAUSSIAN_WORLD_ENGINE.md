# Gaussian World Engine

**Noodlings Generative 3D Engine for Real-Time Interactive Worlds**

**Created:** December 20, 2025
**Hardware:** Apple M3 Ultra (32 cores, 512GB unified memory)
**Status:** Implementation Beginning

---

## Executive Summary

Build a local generative 3D world engine that renders interactive environments from the Noodlings Scene Protocol (NSP). Unlike Genie/Mirage which are stateless pixel predictors, this engine maintains explicit semantic understanding through language-embedded Gaussian splats.

**Core Differentiator:** While Genie hallucinates worlds from pixels, Noodlings *constructs* them from semantic truth. Every Gaussian knows what it represents. Every frame is query-able. Every character maintains identity across forms.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    NSP SCENE STATE MANAGER                       │
│              (canonical semantic truth - EXISTS)                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GAUSSIAN SCENE BUILDER                        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ DreamGaussian│  │   4D-GS      │  │     LangSplat        │  │
│  │ (characters) │  │ (dynamics)   │  │ (semantic embedding) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │                 │                    │                │
│         └─────────────────┴────────────────────┘                │
│                           │                                      │
│                           ▼                                      │
│              COMPOSITE GAUSSIAN SCENE                            │
│              (positions, dynamics, semantics)                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ RENDERER │   │  QUERY   │   │ PERCEIVE │
        │ (pixels) │   │ (search) │   │ (slices) │
        └──────────┘   └──────────┘   └──────────┘
              │               │               │
              ▼               ▼               ▼
         Final Frame    "Find Red"    Red's POV
         (450+ FPS)      → [3,0,2]    (filtered)
```

---

## Technology Stack

### Core Components

| Component | Technology | Purpose | Status |
|-----------|------------|---------|--------|
| Scene Reconstruction | OpenSplat | Train Gaussians from images | To Install |
| Character Generation | DreamGaussian | Image/text → 3D Gaussian | To Install |
| Dynamics | 4D Gaussian Splatting | Animation/motion | Phase 3 |
| Semantic Embedding | LangSplat | Language queries in 3D | Phase 4 |
| Rendering | MetalSplatter / gsplat-mps | Real-time on Apple Silicon | To Install |
| Scene Management | NSP (existing) | Semantic truth | EXISTS |

### Hardware Capabilities (M3 Ultra 512GB)

```
Component               Specification           Implication
─────────────────────────────────────────────────────────────
Chip                    Apple M3 Ultra          Latest Apple Silicon
CPU Cores               32 (24P + 8E)           Parallel training
GPU Cores               80 (estimated)          Real-time rendering
Unified Memory          512 GB                  Hold entire worlds
Memory Bandwidth        ~800 GB/s               Fast tensor ops
Neural Engine           32-core                 Accelerated inference
```

**Capacity Estimates:**
- Single Gaussian scene: ~100MB - 2GB
- Character model: ~50-500MB
- Full stage with 10 characters: ~5-10GB
- **Headroom:** Can hold 50+ scenes simultaneously

---

## Implementation Phases

### Phase 1: Foundation (Days 1-3)

**Goal:** Verify Gaussian splatting works on M3 Ultra

#### 1.1 Install OpenSplat
```bash
# Dependencies
brew install cmake libomp opencv eigen

# Clone and build
git clone https://github.com/pierotofy/OpenSplat.git
cd OpenSplat
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j32
```

#### 1.2 Test with Sample Scene
- Use provided garden/bicycle scenes
- Verify training completes (~12 min on M2, faster on M3)
- Export .ply file

#### 1.3 Install MetalSplatter for Viewing
```bash
git clone https://github.com/scier/MetalSplatter.git
# Build with Xcode or swift build
```

#### 1.4 Validation Criteria
- [ ] OpenSplat compiles without errors
- [ ] Training produces valid .ply
- [ ] MetalSplatter renders at 60+ FPS
- [ ] Memory usage reasonable (<50GB for sample)

### Phase 2: Character Pipeline (Days 4-7)

**Goal:** Generate noodling characters as Gaussian splats

#### 2.1 Install DreamGaussian
```bash
git clone https://github.com/dreamgaussian/dreamgaussian.git
cd dreamgaussian
pip install -r requirements.txt
# May need PyTorch MPS backend configuration
```

#### 2.2 Prepare Reference Art
```
library/
├── Noodlings/
│   └── red/
│       └── Assets/
│           ├── reference_front.png    # Fire imp front view
│           ├── reference_side.png     # Fire imp side view
│           └── reference_poses/       # Multiple poses
```

#### 2.3 Generate Character Gaussians
```python
# Test: Single image to 3D
python main.py --config configs/image.yaml input=red_reference.png

# Expected: ~2 minutes to textured mesh
# Output: .ply + .obj + textures
```

#### 2.4 NSP Character Integration
Create adapter that:
1. Reads `VisualForm` from NSP scene packet
2. Loads corresponding Gaussian model
3. Positions at entity coordinates
4. Applies rotation/scale from transform

#### 2.5 Validation Criteria
- [ ] DreamGaussian runs on MPS backend
- [ ] Reference art → 3D in <5 minutes
- [ ] Character recognizable from multiple angles
- [ ] Multiple forms work (fire imp variants)

### Phase 3: Dynamics (Days 8-12)

**Goal:** Characters can move and animate

#### 3.1 Install 4D Gaussian Splatting
```bash
git clone https://github.com/hustvl/4DGaussians.git
cd 4DGaussians
pip install -r requirements.txt
```

#### 3.2 Action-to-Deformation Mapping
```python
# NSP actions map to Gaussian deformations
ACTION_DEFORMATIONS = {
    "walk": {"type": "translation", "velocity": [0.1, 0, 0]},
    "jump": {"type": "arc", "height": 1.0, "duration": 0.5},
    "turn": {"type": "rotation", "axis": "y", "speed": 90},
    "emote_wave": {"type": "bone_animation", "clip": "wave"},
}
```

#### 3.3 Temporal Consistency
- Maintain Gaussian identity across frames
- Interpolate positions smoothly
- Handle form transitions (fire imp → different mood)

#### 3.4 Validation Criteria
- [ ] Character moves through space smoothly
- [ ] Actions trigger appropriate deformations
- [ ] No temporal flickering/artifacts
- [ ] 30+ FPS during animation

### Phase 4: Semantic Integration (Days 13-18)

**Goal:** Query-able 3D world with language embeddings

#### 4.1 Install LangSplat
```bash
git clone https://github.com/minghanqin/LangSplat.git
cd LangSplat
pip install -r requirements.txt
# Requires CLIP model download
```

#### 4.2 NSP → CLIP Embedding Pipeline
```python
def embed_entity(entity: Noodling) -> np.ndarray:
    """Convert NSP entity to CLIP embedding."""
    description = f"{entity.name}, a {entity.species}"
    if entity.current_form:
        description += f" in {entity.current_form.name} form"
    if entity.visible_affect:
        description += f", looking {affect_to_text(entity.visible_affect)}"
    return clip_model.encode_text(description)
```

#### 4.3 Bidirectional Queries
```python
# Render → Query
scene.query("where is the fire imp?")  # Returns [3.0, 0.0, 2.0]

# Query → Render
scene.highlight("anything on fire")    # Returns mask of fire regions
```

#### 4.4 Perception Slice Generation
```python
def perception_from_gaussians(viewer_pos, fov_cone):
    """Generate perception slice from Gaussian scene."""
    visible_gaussians = scene.query_frustum(viewer_pos, fov_cone)
    entities = []
    for g in visible_gaussians:
        entity_id = scene.query_semantic(g.position, "what entity is this?")
        if entity_id:
            entities.append(entity_id)
    return PerceptionSlice(viewer=viewer_id, visible_entities=entities)
```

#### 4.5 Validation Criteria
- [ ] LangSplat trains on custom scenes
- [ ] Queries return accurate locations
- [ ] Perception slices match NSP ground truth
- [ ] Query latency <100ms

### Phase 5: Full Pipeline Integration (Days 19-25)

**Goal:** NSP Scene Packet → Real-time Gaussian World

#### 5.1 Adapter Architecture
```python
# noodlestudio/core/gaussian_world/
├── __init__.py
├── gaussian_scene.py          # Composite scene management
├── character_loader.py        # DreamGaussian integration
├── dynamics_engine.py         # 4D-GS animation
├── semantic_layer.py          # LangSplat queries
├── nsp_adapter.py             # Scene packet → Gaussians
└── renderer.py                # Metal/MPS rendering
```

#### 5.2 NSP Adapter Interface
```python
class GaussianWorldAdapter:
    """Converts NSP scene packets to Gaussian scenes."""

    def __init__(self, character_cache_path: str):
        self.character_cache = {}  # entity_id → GaussianModel
        self.scene = CompositeGaussianScene()

    def apply_scene_packet(self, packet: ScenePacket):
        """Update Gaussian scene from NSP packet."""
        # Update character positions
        for noodling in packet.noodlings:
            if noodling.id not in self.character_cache:
                self.character_cache[noodling.id] = self._load_character(noodling)
            self._update_character_transform(noodling)
            self._update_character_dynamics(noodling)

        # Update camera
        if packet.camera:
            self.scene.set_camera(packet.camera.to_gaussian_camera())

        # Embed semantics
        self._update_semantic_embeddings(packet)

    def render(self) -> np.ndarray:
        """Render current scene to pixels."""
        return self.scene.render()

    def query(self, text: str) -> List[Tuple[str, Vec3]]:
        """Query scene with natural language."""
        return self.scene.semantic_query(text)
```

#### 5.3 Server Integration
```python
# In server.py - add Gaussian world endpoint
@app.route('/api/gaussian/frame', methods=['GET'])
async def get_gaussian_frame():
    """Get current rendered frame from Gaussian engine."""
    packet = scene_state_manager.get_current_packet()
    gaussian_adapter.apply_scene_packet(packet)
    frame = gaussian_adapter.render()
    return send_file(frame_to_jpeg(frame), mimetype='image/jpeg')

# WebSocket for real-time streaming
@sio.on('subscribe_gaussian_stream')
async def subscribe_gaussian(sid):
    """Stream Gaussian frames at 30 FPS."""
    while True:
        frame = gaussian_adapter.render()
        await sio.emit('gaussian_frame', frame_to_base64(frame), room=sid)
        await asyncio.sleep(1/30)
```

#### 5.4 Validation Criteria
- [ ] Scene packet → rendered frame in <100ms
- [ ] Streaming at 30+ FPS sustained
- [ ] Character positions match NSP truth
- [ ] Queries return correct entities
- [ ] Memory stable over long sessions

### Phase 6: Polish and Demo (Days 26-30)

**Goal:** Demonstrable system for Steve DiPaola meeting

#### 6.1 Demo Scene: "Red Explores the Nexus"
- Stage: The Nexus (central hub)
- Character: Red (fire imp)
- Actions: Walk, look around, emote
- Queries: "where is Red?", "what's glowing?"

#### 6.2 Side-by-Side Comparison
```
┌─────────────────────┬─────────────────────┐
│    TEXT (MUD)       │   GAUSSIAN 3D       │
├─────────────────────┼─────────────────────┤
│ The Nexus           │ [rendered 3D view]  │
│ A swirling hub...   │                     │
│                     │                     │
│ Red is here.        │ [Red visible]       │
│ She looks curious.  │ [curious pose]      │
└─────────────────────┴─────────────────────┘
```

#### 6.3 Performance Targets
| Metric | Target | Stretch |
|--------|--------|---------|
| Frame rate | 30 FPS | 60 FPS |
| Latency (packet → frame) | <100ms | <50ms |
| Character load time | <5s | <2s |
| Query response | <100ms | <50ms |
| Memory (10 characters) | <20GB | <10GB |

---

## File Structure

```
noodlestudio/
├── core/
│   ├── semantic_world/           # EXISTS - NSP implementation
│   │   ├── scene_packet.py
│   │   ├── perception.py
│   │   ├── scene_state_manager.py
│   │   └── scene_emitter.py
│   │
│   └── gaussian_world/           # NEW - Gaussian engine
│       ├── __init__.py
│       ├── gaussian_scene.py     # Scene composition
│       ├── character_loader.py   # DreamGaussian integration
│       ├── dynamics_engine.py    # 4D-GS animation
│       ├── semantic_layer.py     # LangSplat queries
│       ├── nsp_adapter.py        # NSP → Gaussians
│       └── renderer.py           # Metal rendering
│
├── panels/
│   └── gaussian_view_panel.py    # NEW - 3D preview in NoodleStudio
│
└── external/                     # NEW - External tool integrations
    ├── opensplat/                # Submodule or scripts
    ├── dreamgaussian/
    ├── 4d_gaussian_splatting/
    └── langsplat/
```

---

## Dependencies

### System (Homebrew)
```bash
brew install cmake libomp opencv eigen
brew install python@3.11  # If needed
```

### Python
```
torch>=2.0.0
torchvision
numpy
opencv-python
pillow
plyfile
tqdm
transformers  # For CLIP
open_clip_torch
trimesh
pyglet
```

### External Repositories
- OpenSplat: https://github.com/pierotofy/OpenSplat
- DreamGaussian: https://github.com/dreamgaussian/dreamgaussian
- 4DGaussians: https://github.com/hustvl/4DGaussians
- LangSplat: https://github.com/minghanqin/LangSplat
- MetalSplatter: https://github.com/scier/MetalSplatter
- gsplat-mps: https://github.com/iffyloop/gsplat-mps

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MPS backend issues | Medium | High | Fall back to CPU, report bugs |
| Memory pressure | Low | Medium | Monitor, optimize batch sizes |
| Character quality | Medium | Medium | Iterate on reference art, try GaussianDreamer |
| Training time | Low | Low | M3 Ultra is fast, can parallelize |
| Integration complexity | Medium | High | Start simple, add features incrementally |

---

## Success Metrics

### Phase 1 Complete When:
- OpenSplat trains a scene on M3 Ultra
- Render visible in MetalSplatter at 60+ FPS

### Phase 2 Complete When:
- Red (fire imp) generated as Gaussian model
- Recognizable from 8 cardinal directions

### Phase 3 Complete When:
- Red walks smoothly through space
- Action input → visible response

### Phase 4 Complete When:
- "Where is Red?" returns correct position
- Perception slice matches ground truth

### Phase 5 Complete When:
- NSP packet → rendered frame pipeline works
- Real-time streaming at 30+ FPS

### Demo Ready When:
- 5-minute walkthrough runs without crashes
- Side-by-side text/3D comparison works
- At least 3 queries work correctly

---

## References

### Papers
- [3D Gaussian Splatting (SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [DreamGaussian (ICLR 2024 Oral)](https://arxiv.org/abs/2309.16653)
- [4D Gaussian Splatting (CVPR 2024)](https://arxiv.org/abs/2310.08528)
- [LangSplat (CVPR 2024 Highlight)](https://arxiv.org/abs/2312.16084)
- [HUGS: Human Gaussian Splats (Apple)](https://machinelearning.apple.com/research/hugs)

### Code Repositories
- OpenSplat: https://github.com/pierotofy/OpenSplat
- DreamGaussian: https://github.com/dreamgaussian/dreamgaussian
- 4DGaussians: https://github.com/hustvl/4DGaussians
- LangSplat: https://github.com/minghanqin/LangSplat
- MetalSplatter: https://github.com/scier/MetalSplatter

### Noodlings Internal
- `docs/SCENE_PROTOCOL_SPEC.md` - NSP specification
- `noodlestudio/core/semantic_world/` - NSP implementation
- `CLAUDE.md` - Project context

---

## Changelog

### December 23, 2025 - PHASE 4 COMPLETE (CLIP Queries)

**IMPLEMENTATION DIVERGENCE:** We took a different path than originally planned but achieved the same goals faster.

**What We Built (with NinaK):**

| Original Plan | Actual Implementation |
|--------------|----------------------|
| DreamGaussian (image→3D) | VRM mesh→Gaussians (`vrm_to_radiance.py`) |
| MetalSplatter/gsplat-mps | Pure PyTorch/MPS renderer (`gaussian_renderer.py`) |
| LangSplat integration | Native CLIP queries (`semantic_query.py`) |
| External tool orchestration | Unified `.radiance` format with all data |

**Phase Status Update:**

| Phase | Original | Actual Status | Notes |
|-------|----------|---------------|-------|
| 1: Foundation | OpenSplat | COMPLETE (different) | VRM pipeline instead |
| 2: Characters | DreamGaussian | COMPLETE (different) | Mesh→Gaussian conversion |
| 3: Dynamics | 4D-GS | PENDING | Need skeletal animation |
| 4: Semantics | LangSplat | **COMPLETE** | Native CLIP in semantic_query.py |
| 5: Integration | NSP adapter | **COMPLETE** | scene_protocol_integration.py |
| 6: Polish | Demo | IN PROGRESS | Asset Import Wizard next |

**New Files Created:**

```
semantic_world/
├── semantic_query.py     # CLIP queries - CLIPEmbeddingGenerator
├── radiance_format.py    # .radiance binary format
├── gaussian_renderer.py  # Pure PyTorch/MPS rendering
├── gaussian_adapter.py   # Scene compositor
└── gaussian_collision.py # Touch detection + affect mapping

tools/
└── vrm_to_radiance.py    # VRM → Gaussian conversion

cmush/
└── scene_protocol_integration.py  # Server wiring (updated)
```

**Key Achievement:** Natural language queries work!
```python
result = query_scene_semantic("left hand", top_k=3)
# Returns: [{"body_part": "left Hand", "similarity": 1.0, "position": [...]}]
```

**Server Endpoints Added:**
- `semantic_query` - Natural language search
- `semantic_raycast` - Click-to-inspect
- `get_visible_body_parts` - FOV-based visibility

**Why We Diverged:**
1. VRM files already exist (VRChat ecosystem) - no training needed
2. Mesh→Gaussian is deterministic (vs generative uncertainty)
3. Native CLIP is simpler than LangSplat training pipeline
4. `.radiance` format bundles everything (Gaussians + semantics + skeleton)

**What's Still Needed:**
- Phase 3: Skeletal animation (bone transforms → Gaussian deformation)
- Asset Import Wizard (drag-drop VRM/GLTF)
- Collision → Affect impulses (partially done in gaussian_collision.py)

### December 20, 2025
- Initial specification created
- Hardware verified: M3 Ultra 512GB
- Implementation plan drafted
- Phase 1 beginning

---

## For Fresh Claude Context

**Key Files to Read First:**
1. `CLAUDE.md` - Project overview and current focus
2. `docs/ARCHITECTURE.md` - Full system architecture
3. `semantic_world/semantic_query.py` - CLIP query system
4. `semantic_world/radiance_format.py` - Asset format
5. `cmush/scene_protocol_integration.py` - Server integration

**What's Working:**
- VRM → Gaussians → .radiance (tested with AliciaSolid.vrm)
- CLIP embeddings auto-generated from semantic labels
- Natural language queries on Gaussian scenes
- Server WebSocket endpoints for queries
- FOV-based body part visibility

**What's Next:**
- Asset Import Wizard (panels/asset_import_wizard.py)
- Skeletal animation (bone transforms)
- Touch → Affect (gaussian_collision.py has foundation)

**Test Commands:**
```bash
# Test CLIP queries
cd applications/cmush
PYTHONPATH=../noodlestudio:../.. ../../venv/bin/python3 test_scene_protocol.py

# Test VRM conversion
cd applications/noodlestudio
PYTHONPATH=.:../.. python -m noodlestudio.tools.vrm_to_radiance \
  ../../external/vrm_samples/AliciaSolid.vrm -o alicia.radiance
```

---

**Let's build the thing.**
