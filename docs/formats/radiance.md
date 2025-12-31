# Radiance Format

Binary format for semantic Gaussian splat models.

---

## Overview

`.radiance` files store 3D Gaussian splat data with semantic annotations,
skeletal rigging, and CLIP embeddings. Unlike raw PLY files, radiance files
know what each Gaussian represents.

## File Structure

Binary chunk-based format (similar to GLB):

```
RADI [magic]
VERSION [u32]
CHUNK_COUNT [u32]

CHUNK_HEADER { type: [4 bytes], size: [u32] }
CHUNK_DATA [size bytes]
...
```

## Chunk Types

### GAUS - Gaussian Data
Core splat data per Gaussian:
- Position (3 floats)
- Scale (3 floats)
- Rotation (4 floats, quaternion)
- Opacity (1 float)
- SH coefficients (48 floats for degree 3)

### SKEL - Skeleton
Bone hierarchy:
- Bone count
- Per bone: name, parent index, bind pose

### SKIN - Skinning Weights
Per-Gaussian bone weights:
- Bone indices (4 per Gaussian)
- Weights (4 per Gaussian, sum to 1)

### SEMA - Semantic Labels
Per-Gaussian annotations:
- Body part (head, torso, arm_l, arm_r, leg_l, leg_r)
- Region (face, hair, clothing, skin)

### CLIP - Embeddings (optional)
512-D CLIP vectors for semantic queries.

### META - Metadata
JSON blob with:
- Display name
- Source file
- Creation date
- Gaussian count

## Creating Radiance Files

### From VRM

```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. python3 -m noodlestudio.tools.vrm_to_radiance \
    input.vrm -o output.radiance -v
```

### From Trained PLY

```bash
PYTHONPATH=.:../.. python3 -m noodlestudio.tools.vrm_to_radiance \
    trained.ply -o output.radiance -v
```

## Loading in Code

```python
from noodlestudio.core.semantic_world.radiance_format import load_radiance

asset = load_radiance("avatar.radiance")
print(f"Gaussians: {asset.gaussian_count}")
print(f"Bones: {len(asset.skeleton.bones) if asset.skeleton else 0}")
```

## Rendering

Radiance files render via `GaussianRenderer`:

```python
from noodlestudio.core.gaussian_renderer import GaussianRenderer
from noodlestudio.core.radiance_component import RadianceComponent

component = RadianceComponent("avatar")
component.load_asset("avatar.radiance")

renderer = GaussianRenderer()
image, alpha, info = renderer.render_component(component, camera)
```

GPU rendering (gsplat-mps) achieves 120 FPS on Apple Silicon.
