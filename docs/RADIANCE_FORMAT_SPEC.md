# Radiance Format Specification v1.0

**Semantic Gaussian Splat Format for Noodlings**

Every Gaussian knows what it represents. Every frame is query-able.

---

## Overview

`.radiance` is an extended Gaussian Splat format that embeds semantic metadata, skeletal binding, and entity association into every splat. Unlike standard PLY files which are purely geometric, Radiance files are *knowable* - they can answer questions like "what did I click on?" and "is Red touching Yuki?"

### Design Principles

1. **Semantic Truth** - Each Gaussian carries meaning, not just color
2. **Animation-Ready** - Skeletal binding enables LBS deformation
3. **Query-able** - CLIP embeddings enable natural language queries
4. **Collision-Aware** - Overlap detection for physics and social touch
5. **Portable** - Single file contains everything needed

### Naming Convention

| Asset Type | Example |
|------------|---------|
| Character | `red.radiance` |
| Prop | `magic_radio.radiance` |
| Environment | `the_nexus.radiance` |

Why "Radiance": Gaussians literally radiate color via spherical harmonics. "Red's Radiance" sounds good. Evokes Neural Radiance Fields (NeRF) heritage.

---

## File Structure

### Binary Format (Recommended)

```
.radiance file structure:
┌─────────────────────────────────────┐
│ Header (32 bytes)                   │
├─────────────────────────────────────┤
│ Chunk: GAUS - Gaussian Parameters   │
│ Chunk: SKEL - Skeleton Definition   │
│ Chunk: SKIN - Per-Gaussian Skinning │
│ Chunk: SEMA - Semantic Labels       │
│ Chunk: CLIP - CLIP Embeddings       │  ← Optional
│ Chunk: SPRG - Spring Bone Physics   │  ← Optional
│ Chunk: META - Entity Metadata       │
└─────────────────────────────────────┘
```

### Header (32 bytes)

```c
struct RadianceHeader {
    char     magic[4];        // "RADI"
    uint32_t version;         // 0x00010000 for v1.0
    uint32_t gaussian_count;  // Number of Gaussians
    uint32_t bone_count;      // Number of bones (0 if no skeleton)
    uint32_t chunk_count;     // Number of chunks following header
    uint32_t flags;           // Feature flags (see below)
    uint8_t  reserved[8];     // Future use
};

// Feature flags
#define RADIANCE_HAS_SKELETON   0x0001
#define RADIANCE_HAS_SKINNING   0x0002
#define RADIANCE_HAS_SEMANTICS  0x0004
#define RADIANCE_HAS_CLIP       0x0008
#define RADIANCE_HAS_SPRINGS    0x0010
#define RADIANCE_ANIMATED       0x0020  // 4D Gaussian (multiple frames)
```

### Chunk Format

Each chunk follows this structure:

```c
struct ChunkHeader {
    char     type[4];      // Chunk type ID
    uint32_t size;         // Size of chunk data (not including header)
    uint32_t flags;        // Chunk-specific flags
    uint32_t reserved;
};
// Followed by `size` bytes of chunk data
```

---

## Chunk Definitions

### GAUS - Gaussian Parameters

Core Gaussian splat data. Compatible with standard 3DGS format.

```c
struct GaussianData {
    // Position (12 bytes)
    float x, y, z;

    // Scale (12 bytes) - log scale for numerical stability
    float scale_x, scale_y, scale_z;

    // Rotation (16 bytes) - quaternion (x, y, z, w)
    float rot_x, rot_y, rot_z, rot_w;

    // Opacity (4 bytes) - sigmoid-encoded
    float opacity;

    // Spherical Harmonics (variable)
    // DC term: 3 floats (RGB)
    // Higher bands: 45 floats for degree 3
    float sh_dc[3];
    float sh_rest[45];  // Optional, depends on SH degree
};
// Total: 92 bytes per Gaussian (degree 3 SH)
// Or 56 bytes per Gaussian (DC only)
```

**Chunk flags:**
- Bits 0-3: SH degree (0=DC only, 1-3=higher bands)
- Bit 4: Compressed (future)

### SKEL - Skeleton Definition

Bone hierarchy for animation.

```c
struct SkeletonChunk {
    uint32_t bone_count;
    uint32_t humanoid_map_count;
    // Followed by:
    // - bone_count * BoneData
    // - humanoid_map_count * HumanoidMapping
};

struct BoneData {
    char     name[32];       // Null-terminated bone name
    int32_t  parent_index;   // -1 for root
    float    position[3];    // Local position
    float    rotation[4];    // Local rotation (quaternion)
    float    scale[3];       // Local scale
};

struct HumanoidMapping {
    char     humanoid_name[32];  // e.g., "leftUpperArm"
    uint32_t bone_index;
};
```

### SKIN - Per-Gaussian Skinning

Bone weights for Linear Blend Skinning (LBS).

```c
struct SkinningData {
    // Per Gaussian (4 bone influences max)
    uint16_t bone_indices[4];  // Bone indices
    float    bone_weights[4];  // Weights (sum to 1.0)
};
// 24 bytes per Gaussian
```

### SEMA - Semantic Labels

Per-Gaussian semantic metadata.

```c
struct SemanticData {
    uint8_t  body_region;      // Enum: HEAD, TORSO, LEFT_ARM, etc.
    uint8_t  semantic_flags;   // Bit flags for properties
    uint16_t label_offset;     // Offset into string table
};
// 4 bytes per Gaussian

// Followed by string table:
struct StringTable {
    uint32_t count;
    uint32_t offsets[count];   // Offsets into string data
    char     strings[];        // Null-terminated strings
};

// Body region enum
enum BodyRegion {
    REGION_OTHER = 0,
    REGION_HEAD = 1,
    REGION_TORSO = 2,
    REGION_LEFT_ARM = 3,
    REGION_RIGHT_ARM = 4,
    REGION_LEFT_LEG = 5,
    REGION_RIGHT_LEG = 6,
    REGION_LEFT_HAND = 7,
    REGION_RIGHT_HAND = 8,
    REGION_TAIL = 9,
    REGION_ACCESSORY = 10,
};
```

### CLIP - CLIP Embeddings (Optional)

For LangSplat-style natural language queries.

```c
struct CLIPChunk {
    uint32_t embedding_dim;    // 512 or 768
    uint32_t quantization;     // 0=float32, 1=float16, 2=int8
    // Followed by gaussian_count * embedding_dim values
};
```

### SPRG - Spring Bone Physics (Optional)

Physics parameters for hair, cloth, tails.

```c
struct SpringChunk {
    uint32_t chain_count;
    uint32_t collider_count;
    // Followed by chain and collider data
};

struct SpringChain {
    char     name[32];
    uint32_t bone_count;
    uint32_t bone_indices[];   // bone_count indices
    float    stiffness;
    float    gravity_power;
    float    gravity_dir[3];
    float    drag_force;
    float    hit_radius;
};

struct SpringCollider {
    uint32_t bone_index;
    float    offset[3];
    float    radius;
};
```

### META - Entity Metadata

High-level information about the asset.

```c
struct MetadataChunk {
    char     entity_type[16];   // "noodling", "prim", "environment"
    char     entity_id[64];     // UUID or name
    char     display_name[64];
    char     author[64];
    char     created[32];       // ISO 8601 timestamp
    float    bounds_min[3];
    float    bounds_max[3];
    float    center[3];
    uint32_t tag_count;
    // Followed by tag_count null-terminated strings
};
```

---

## Collision Detection

### Gaussian Overlap Integral

Two Gaussians G1 and G2 overlap when their probability densities multiply to a significant value. The closed-form solution:

```
overlap(G1, G2) = sqrt(det(Σ1) * det(Σ2) / det(Σ1 + Σ2)) * exp(-0.5 * d_mahal²)
```

Where:
- `Σ1, Σ2` are the 3x3 covariance matrices
- `d_mahal` is the Mahalanobis distance between centers

### Covariance from Scale + Rotation

```python
def covariance_matrix(scale, rotation_quat):
    """Build 3x3 covariance from scale and rotation."""
    # Scale matrix (diagonal)
    S = np.diag(scale ** 2)

    # Rotation matrix from quaternion
    R = quaternion_to_matrix(rotation_quat)

    # Covariance: R @ S @ R.T
    return R @ S @ R.T
```

### Touch Event Generation

```python
@dataclass
class TouchEvent:
    entity_a: str
    body_part_a: str
    entity_b: str
    body_part_b: str
    position: Tuple[float, float, float]
    intensity: float  # 0-1, based on overlap integral
    timestamp: float
```

### Physics → Affect Pipeline

```
Touch Detected → TouchEvent
       ↓
  Affect Impulse Generator
       ↓
  CharmNetwork.inject_state({
      'arousal': f(intensity, body_part),
      'valence': f(relationship, touch_type)
  })
```

---

## Directory Structure

For noodling packages:

```
Noodlings/
└── red/
    ├── noodling.yaml           # Manifest
    ├── recipe.yaml             # Character definition
    ├── assembly.yaml           # Facet topology
    ├── Radiances/
    │   ├── fire_imp.radiance   # Default form
    │   └── angry_fire_imp.radiance  # Variant
    └── Assets/
        └── reference_front.png
```

---

## API Examples

### Loading a Radiance File

```python
from noodlestudio.core.semantic_world.radiance_format import RadianceAsset

# Load
asset = RadianceAsset.load("red.radiance")

print(f"Gaussians: {asset.gaussian_count}")
print(f"Bones: {asset.bone_count}")
print(f"Entity: {asset.metadata.display_name}")

# Access Gaussian data
positions = asset.positions  # (N, 3) array
scales = asset.scales        # (N, 3) array
rotations = asset.rotations  # (N, 4) array

# Access semantic data
for i in range(asset.gaussian_count):
    label = asset.get_semantic_label(i)
    region = asset.get_body_region(i)
    print(f"Gaussian {i}: {label} ({region})")
```

### Collision Detection

```python
from noodlestudio.core.semantic_world.gaussian_collision import (
    GaussianCollisionDetector,
    TouchEvent
)

detector = GaussianCollisionDetector()
detector.add_entity("red", red_asset)
detector.add_entity("yuki", yuki_asset)

# Detect touches
touches = detector.detect_touches(threshold=0.1)
for touch in touches:
    print(f"{touch.entity_a}'s {touch.body_part_a} touched "
          f"{touch.entity_b}'s {touch.body_part_b}")
```

### Animation with Skinning

```python
from noodlestudio.core.pose_track import PoseTrack, PoseRetargeter

# Load pose
track = PoseTrack.load_yaml("wave.posetrack")
pose = track.sample(t=1.5)

# Retarget to bone rotations
retargeter = PoseRetargeter()
bone_rotations = retargeter.apply_pose(pose)

# Apply LBS deformation
deformed_positions = asset.apply_skinning(bone_rotations)
```

---

## Compatibility

### Import from PLY

Standard Gaussian PLY files can be imported and enriched:

```python
asset = RadianceAsset.from_ply("model.ply")
asset.metadata.entity_id = "red"
asset.metadata.display_name = "Red"

# Add skeleton from VRM
asset.import_skeleton_from_vrm("red.vrm")

# Compute skinning weights (nearest vertex)
asset.compute_skinning_from_vrm("red.vrm")

# Save as radiance
asset.save("red.radiance")
```

### Export to PLY

For compatibility with external viewers:

```python
asset.export_ply("red.ply")  # Standard PLY, loses semantic data
```

---

## Version History

- **v1.0** (December 2025) - Initial specification
  - Core Gaussian parameters
  - Skeleton and skinning
  - Semantic labels
  - CLIP embeddings (optional)
  - Spring bone physics (optional)

---

*Every Gaussian knows what it represents. Every touch is meaningful.*
