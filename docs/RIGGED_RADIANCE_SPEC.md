# Rigged Radiance Specification

Runtime skeletal animation for Gaussian splat avatars.

**Status:** Design Complete, Implementation Pending
**Author:** Caitlyn + Claude
**Date:** December 24, 2025

---

## Overview

Rigged Radiance enables real-time skeletal animation of Gaussian splat characters. Unlike mesh skinning where vertices move, Gaussian skinning transforms position, rotation, and optionally scale of each splat.

### Design Goals

1. **Fast and fluid** - 60 FPS with 50K+ Gaussians
2. **Rig-agnostic** - Same animation plays on any humanoid avatar
3. **Affect-driven** - Pose responds to emotional state
4. **Scriptable** - Full control from facet scripts
5. **Cross-platform** - CPU fallback, GPU acceleration optional

### Key Insight

Our muscle-based animation system (inspired by Unity Mecanim) provides rig-agnostic poses. Combined with VRM's standardized humanoid skeleton, we can animate any avatar with the same pose data.

---

## Architecture

### System Diagram

```
                         ANIMATION SOURCES
                    ┌──────────┬──────────────┐
                    │          │              │
              ┌─────v────┐ ┌───v───┐ ┌────────v────────┐
              │PoseTrack │ │Script │ │ Procedural      │
              │(.posetrack)│ │API   │ │ (breathing,IK) │
              └─────┬────┘ └───┬───┘ └────────┬────────┘
                    │          │              │
                    └──────────┼──────────────┘
                               │
                    ┌──────────v──────────┐
                    │   MUSCLE VALUES     │
                    │  Dict[str, float]   │
                    │  47 humanoid muscles│
                    │  normalized [-1, 1] │
                    └──────────┬──────────┘
                               │
                    ┌──────────v──────────┐
                    │  MuscleRetargeter   │
                    │  muscle -> bone     │
                    │  local quaternions  │
                    └──────────┬──────────┘
                               │
                    ┌──────────v──────────┐
                    │   SkeletonPoser     │
                    │  local -> world     │
                    │  (B, 4, 4) matrices │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────v─────────┐ ┌────v─────┐ ┌───────v───────┐
    │  GaussianSkinner  │ │SpringSim │ │ BlendShapes   │
    │  LBS/DQS for      │ │ hair,    │ │ facial morph  │
    │  positions+rots   │ │ cloth    │ │ targets       │
    └─────────┬─────────┘ └────┬─────┘ └───────┬───────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────v──────────┐
                    │  POSED GAUSSIANS    │
                    │  positions (N, 3)   │
                    │  rotations (N, 4)   │
                    │  [scales (N, 3)]    │
                    └──────────┬──────────┘
                               │
                    ┌──────────v──────────┐
                    │  GaussianRenderer   │
                    │  gsplat-mps (GPU)   │
                    │  or software        │
                    └─────────────────────┘
```

### File Locations

```
noodlestudio/core/
├── skinning/
│   ├── __init__.py
│   ├── muscle_retargeter.py    # Muscle -> bone rotation
│   ├── skeleton_poser.py       # Hierarchy -> world matrices
│   ├── gaussian_skinner.py     # LBS/DQS implementation
│   └── spring_bone_sim.py      # VRM spring physics
├── pose_track.py               # (existing) Animation curves
├── radiance_component.py       # (extend) Add pose state
└── gaussian_renderer.py        # (extend) Accept posed data
```

---

## Data Structures

### Existing (in .radiance files)

```python
# Per-Gaussian skinning (from VRM conversion)
skin_bone_indices: np.ndarray   # (N, 4) uint16 - up to 4 bone influences
skin_bone_weights: np.ndarray   # (N, 4) float32 - blend weights, sum to 1.0

# Skeleton hierarchy
skeleton: RadianceSkeleton
  bones: List[RadianceBone]     # Rest pose transforms
  humanoid_map: Dict[str, int]  # 'leftArm' -> bone index

# Each bone
RadianceBone:
  name: str
  parent_index: int             # -1 for root
  position: (x, y, z)           # Local position (rest)
  rotation: (x, y, z, w)        # Local rotation quaternion (rest)
  scale: (x, y, z)              # Local scale (rest)

# Secondary animation
spring_chains: List[SpringChain]
spring_colliders: List[SpringCollider]
```

### New Runtime State

```python
@dataclass
class PoseState:
    """Current pose of a RadianceComponent."""

    # Input: muscle values
    muscles: Dict[str, float]           # 47 humanoid muscles, [-1, 1]
    root_position: np.ndarray           # (3,) world position offset
    root_rotation: np.ndarray           # (4,) world rotation quaternion

    # Computed: bone transforms
    bone_local_rotations: np.ndarray    # (B, 4) local quaternions
    bone_world_matrices: np.ndarray     # (B, 4, 4) skinning matrices

    # Computed: posed Gaussians
    posed_positions: np.ndarray         # (N, 3) deformed positions
    posed_rotations: np.ndarray         # (N, 4) deformed rotations

    # Cache invalidation
    dirty: bool = True
```

---

## Components

### 1. MuscleRetargeter

Converts normalized muscle values to per-bone local rotations.

```python
class MuscleRetargeter:
    """
    Map humanoid muscles to bone rotations.

    Muscles are normalized [-1, 1] values that map to bone rotation ranges.
    This abstraction enables rig-agnostic animation - the same muscle values
    produce equivalent poses on different skeletons.
    """

    def __init__(self, skeleton: RadianceSkeleton):
        self.skeleton = skeleton
        self.bone_name_to_index = {b.name: i for i, b in enumerate(skeleton.bones)}
        self.humanoid_to_index = skeleton.humanoid_map

    def retarget(self, muscles: Dict[str, float]) -> np.ndarray:
        """
        Convert muscle values to bone local rotations.

        Args:
            muscles: {'LeftArm.FrontBack': 0.5, 'Head.TurnLeftRight': -0.3, ...}

        Returns:
            bone_rotations: (num_bones, 4) quaternions (xyzw)
        """
        bone_rotations = np.zeros((len(self.skeleton.bones), 4), dtype=np.float32)
        bone_rotations[:, 3] = 1.0  # Identity quaternions

        for muscle_name, value in muscles.items():
            # Parse muscle name: 'LeftArm.FrontBack' -> bone='leftArm', axis='FrontBack'
            parts = muscle_name.split('.')
            if len(parts) != 2:
                continue

            humanoid_bone = self._muscle_to_humanoid_bone(parts[0])
            axis_name = parts[1]

            bone_idx = self.humanoid_to_index.get(humanoid_bone, -1)
            if bone_idx < 0:
                continue

            # Get muscle definition (min/max degrees, axis)
            muscle_def = MUSCLE_DEFINITIONS.get(muscle_name)
            if not muscle_def:
                continue

            # Map [-1, 1] to [min_degrees, max_degrees]
            min_deg, max_deg = muscle_def['min'], muscle_def['max']
            degrees = min_deg + (value + 1) * 0.5 * (max_deg - min_deg)

            # Create rotation quaternion around axis
            axis = self._axis_name_to_vector(muscle_def['axis'])
            rotation = quaternion_from_axis_angle(axis, np.radians(degrees))

            # Compose with existing rotation for this bone
            bone_rotations[bone_idx] = quaternion_multiply(
                bone_rotations[bone_idx], rotation
            )

        return bone_rotations
```

### 2. SkeletonPoser

Computes world-space skinning matrices from local bone rotations.

```python
class SkeletonPoser:
    """
    Compute skinning matrices from bone hierarchy.

    Walks the skeleton hierarchy to compute world-space transforms,
    then multiplies by inverse bind matrices for skinning.
    """

    def __init__(self, skeleton: RadianceSkeleton):
        self.skeleton = skeleton
        self.num_bones = len(skeleton.bones)

        # Precompute rest pose matrices
        self.rest_local_matrices = self._compute_rest_local()
        self.inverse_bind_matrices = self._compute_inverse_bind()

    def _compute_rest_local(self) -> np.ndarray:
        """Compute local transform matrix for each bone at rest."""
        matrices = np.zeros((self.num_bones, 4, 4), dtype=np.float32)
        for i, bone in enumerate(self.skeleton.bones):
            matrices[i] = compose_matrix(
                bone.position, bone.rotation, bone.scale
            )
        return matrices

    def _compute_inverse_bind(self) -> np.ndarray:
        """Compute inverse bind matrices (rest pose world -> local)."""
        # First compute world matrices at rest
        world_matrices = self._compute_world_matrices(
            np.zeros((self.num_bones, 4), dtype=np.float32)
        )
        world_matrices[:, :, 3] = 1.0  # Identity rotations initially

        # ... actually compute from hierarchy ...

        # Inverse for skinning
        inverse = np.zeros_like(world_matrices)
        for i in range(self.num_bones):
            inverse[i] = np.linalg.inv(world_matrices[i])
        return inverse

    def compute_skinning_matrices(
        self,
        local_rotations: np.ndarray,  # (B, 4) quaternions
        root_position: np.ndarray = None,
        root_rotation: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute world-space skinning matrices.

        Args:
            local_rotations: Per-bone local rotation quaternions
            root_position: Optional root translation
            root_rotation: Optional root rotation

        Returns:
            skinning_matrices: (B, 4, 4) matrices to transform rest->posed
        """
        world_matrices = np.zeros((self.num_bones, 4, 4), dtype=np.float32)

        for i, bone in enumerate(self.skeleton.bones):
            # Local transform = rest transform * animation rotation
            local = self.rest_local_matrices[i].copy()
            anim_rot = quaternion_to_matrix(local_rotations[i])
            local[:3, :3] = local[:3, :3] @ anim_rot

            # World = parent_world * local
            if bone.parent_index < 0:
                # Root bone
                world_matrices[i] = local
                if root_position is not None:
                    world_matrices[i][:3, 3] += root_position
                if root_rotation is not None:
                    root_mat = quaternion_to_matrix(root_rotation)
                    world_matrices[i][:3, :3] = root_mat @ world_matrices[i][:3, :3]
            else:
                world_matrices[i] = world_matrices[bone.parent_index] @ local

        # Skinning matrix = world * inverse_bind
        skinning_matrices = np.zeros_like(world_matrices)
        for i in range(self.num_bones):
            skinning_matrices[i] = world_matrices[i] @ self.inverse_bind_matrices[i]

        return skinning_matrices
```

### 3. GaussianSkinner

Applies skinning deformation to Gaussian positions and rotations.

```python
class GaussianSkinner:
    """
    Deform Gaussians using skeletal skinning.

    Supports two methods:
    - LBS (Linear Blend Skinning): Fast, may have artifacts at joints
    - DQS (Dual Quaternion Skinning): Better quality, more expensive
    """

    def __init__(self, method: str = 'lbs'):
        assert method in ('lbs', 'dqs')
        self.method = method

    def skin(
        self,
        positions: np.ndarray,      # (N, 3) rest positions
        rotations: np.ndarray,      # (N, 4) rest rotations (xyzw)
        bone_indices: np.ndarray,   # (N, 4) bone references
        bone_weights: np.ndarray,   # (N, 4) blend weights
        skinning_matrices: np.ndarray,  # (B, 4, 4) from SkeletonPoser
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply skinning to Gaussians.

        Returns:
            posed_positions: (N, 3)
            posed_rotations: (N, 4)
        """
        if self.method == 'lbs':
            return self._lbs(positions, rotations, bone_indices,
                           bone_weights, skinning_matrices)
        else:
            return self._dqs(positions, rotations, bone_indices,
                           bone_weights, skinning_matrices)

    def _lbs(self, positions, rotations, bone_indices, bone_weights, matrices):
        """Linear Blend Skinning."""
        n = len(positions)
        posed_positions = np.zeros((n, 3), dtype=np.float32)
        posed_rotations = np.zeros((n, 4), dtype=np.float32)

        for i in range(n):
            # Blend transformation matrices
            blended_matrix = np.zeros((4, 4), dtype=np.float32)
            for j in range(4):
                bone_idx = bone_indices[i, j]
                weight = bone_weights[i, j]
                if weight > 0:
                    blended_matrix += weight * matrices[bone_idx]

            # Transform position
            pos_h = np.array([*positions[i], 1.0])
            posed_positions[i] = (blended_matrix @ pos_h)[:3]

            # Extract rotation from blended matrix
            rot_matrix = blended_matrix[:3, :3]
            # Orthonormalize (LBS can produce non-orthogonal matrices)
            u, _, vh = np.linalg.svd(rot_matrix)
            rot_matrix = u @ vh

            # Compose with Gaussian's rest rotation
            bone_quat = matrix_to_quaternion(rot_matrix)
            posed_rotations[i] = quaternion_multiply(bone_quat, rotations[i])

        return posed_positions, posed_rotations

    def _dqs(self, positions, rotations, bone_indices, bone_weights, matrices):
        """Dual Quaternion Skinning (better quality, preserves volume)."""
        # Convert matrices to dual quaternions
        # Blend in dual quaternion space
        # Convert back to position + rotation
        # ... implementation ...
        pass
```

### 4. SpringBoneSimulator

Secondary physics for hair, cloth, and accessories.

```python
class SpringBoneSimulator:
    """
    VRM-compatible spring bone physics.

    Simulates secondary motion (hair, clothing, tails) using
    spring dynamics with collision detection.
    """

    def __init__(
        self,
        chains: List[SpringChain],
        colliders: List[SpringCollider],
        skeleton: RadianceSkeleton,
    ):
        self.chains = chains
        self.colliders = colliders
        self.skeleton = skeleton

        # Per-bone physics state
        self.velocities: Dict[int, np.ndarray] = {}
        self.current_rotations: Dict[int, np.ndarray] = {}

    def step(
        self,
        dt: float,
        bone_world_matrices: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Simulate one physics step.

        Args:
            dt: Time delta in seconds
            bone_world_matrices: Current world matrices (before spring)

        Returns:
            spring_rotations: {bone_index: quaternion} for spring bones
        """
        spring_rotations = {}

        for chain in self.chains:
            for bone_idx in chain.bone_indices:
                # Get parent transform
                parent_idx = self.skeleton.bones[bone_idx].parent_index
                parent_matrix = bone_world_matrices[parent_idx]

                # Current tail position
                current_pos = self.current_rotations.get(bone_idx,
                    self._get_rest_position(bone_idx, parent_matrix))

                # Target position (where bone wants to be)
                target_pos = self._get_rest_position(bone_idx, parent_matrix)

                # Spring force toward target
                spring_force = (target_pos - current_pos) * chain.stiffness

                # Gravity
                gravity = np.array([0, -1, 0]) * chain.gravity_power

                # Velocity integration
                velocity = self.velocities.get(bone_idx, np.zeros(3))
                velocity += (spring_force + gravity) * dt
                velocity *= (1.0 - chain.drag_force)  # Damping

                # Position integration
                new_pos = current_pos + velocity * dt

                # Collider response
                for collider in self.colliders:
                    new_pos = self._resolve_collision(new_pos, collider,
                                                      bone_world_matrices)

                # Store state
                self.velocities[bone_idx] = velocity
                self.current_rotations[bone_idx] = new_pos

                # Convert position to rotation
                spring_rotations[bone_idx] = self._position_to_rotation(
                    new_pos, parent_matrix, bone_idx
                )

        return spring_rotations
```

---

## Integration

### RadianceComponent Extensions

```python
class RadianceComponent:
    """Extended with pose state."""

    def __init__(self, entity_id: str):
        # ... existing ...

        # Skinning components (lazy init)
        self._retargeter: MuscleRetargeter = None
        self._poser: SkeletonPoser = None
        self._skinner: GaussianSkinner = None
        self._spring_sim: SpringBoneSimulator = None

        # Pose state
        self._pose_state: PoseState = None
        self._pose_dirty: bool = True

    def _init_skinning(self):
        """Initialize skinning components on first use."""
        if self._retargeter is None and self.asset.skeleton:
            self._retargeter = MuscleRetargeter(self.asset.skeleton)
            self._poser = SkeletonPoser(self.asset.skeleton)
            self._skinner = GaussianSkinner(method='lbs')
            if self.asset.spring_chains:
                self._spring_sim = SpringBoneSimulator(
                    self.asset.spring_chains,
                    self.asset.spring_colliders,
                    self.asset.skeleton,
                )
            self._pose_state = PoseState()

    # === Public API ===

    def set_muscle(self, muscle: str, value: float):
        """Set single muscle value [-1, 1]."""
        self._init_skinning()
        self._pose_state.muscles[muscle] = np.clip(value, -1, 1)
        self._pose_dirty = True

    def set_muscles(self, muscles: Dict[str, float]):
        """Set multiple muscles at once (more efficient)."""
        self._init_skinning()
        for k, v in muscles.items():
            self._pose_state.muscles[k] = np.clip(v, -1, 1)
        self._pose_dirty = True

    def reset_pose(self):
        """Reset to rest pose."""
        self._init_skinning()
        self._pose_state.muscles.clear()
        self._pose_dirty = True

    def get_posed_gaussians(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get posed Gaussian data for rendering.

        Returns:
            positions, rotations, scales (all posed)
        """
        if self._pose_dirty:
            self._recompute_pose()

        return (
            self._pose_state.posed_positions,
            self._pose_state.posed_rotations,
            self.asset.scales,  # Scales unchanged for now
        )

    def _recompute_pose(self):
        """Recompute posed Gaussians from current muscle state."""
        self._init_skinning()

        # Muscles -> bone rotations
        bone_rotations = self._retargeter.retarget(self._pose_state.muscles)

        # Bone rotations -> world matrices
        skinning_matrices = self._poser.compute_skinning_matrices(
            bone_rotations,
            self._pose_state.root_position,
            self._pose_state.root_rotation,
        )

        # Optional: spring bone simulation
        if self._spring_sim:
            spring_rots = self._spring_sim.step(1/60, skinning_matrices)
            # Merge spring rotations into bone_rotations
            # Recompute affected matrices

        # Apply skinning to Gaussians
        posed_pos, posed_rot = self._skinner.skin(
            self.asset.positions,
            self.asset.rotations,
            self.asset.skin_bone_indices,
            self.asset.skin_bone_weights,
            skinning_matrices,
        )

        self._pose_state.posed_positions = posed_pos
        self._pose_state.posed_rotations = posed_rot
        self._pose_dirty = False
```

### GaussianRenderer Extensions

```python
class GaussianRenderer:
    def render_component(
        self,
        component: RadianceComponent,
        camera: CameraParams,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Render with optional pose."""

        # Check if component has pose
        if component._pose_state and component._pose_state.muscles:
            positions, rotations, scales = component.get_posed_gaussians()
        else:
            positions = component.asset.positions
            rotations = component.asset.rotations
            scales = component.asset.scales

        # Continue with rendering...
```

---

## Scripting API

### pose_api.py Extensions

```python
class PoseAPI:
    """Scripting interface for skeletal animation."""

    def __init__(self, runtime):
        self._runtime = runtime

    # === Muscle Control ===

    def setMuscle(self, entity_id: str, muscle: str, value: float):
        """
        Set single muscle value.

        Args:
            entity_id: Target entity ('red', 'yuki', etc.)
            muscle: Muscle name ('LeftArm.FrontBack', 'Head.TurnLeftRight')
            value: Normalized value [-1, 1]

        JavaScript:
            context.noodle.pose.setMuscle('red', 'Head.TurnLeftRight', 0.5);
        """
        component = self._runtime.get_component(entity_id)
        if component:
            component.set_muscle(muscle, value)

    def setMuscles(self, entity_id: str, muscles: Dict[str, float]):
        """
        Set multiple muscles at once.

        JavaScript:
            context.noodle.pose.setMuscles('red', {
                'Head.TurnLeftRight': 0.5,
                'Spine.FrontBack': -0.2,
                'LeftArm.DownUp': 0.8
            });
        """
        component = self._runtime.get_component(entity_id)
        if component:
            component.set_muscles(muscles)

    def getMuscles(self, entity_id: str) -> Dict[str, float]:
        """Get current muscle values."""
        component = self._runtime.get_component(entity_id)
        if component and component._pose_state:
            return dict(component._pose_state.muscles)
        return {}

    def resetPose(self, entity_id: str):
        """Reset to rest pose."""
        component = self._runtime.get_component(entity_id)
        if component:
            component.reset_pose()

    # === Track Playback ===

    def loadTrack(self, path: str) -> PoseTrackProxy:
        """Load a pose track from file."""
        # ... existing implementation ...

    def applyTrack(self, entity_id: str, track: PoseTrackProxy, time: float):
        """Apply pose track at specific time."""
        muscles = track._player.sample(time)
        self.setMuscles(entity_id, muscles)

    # === Convenience ===

    def lookAt(self, entity_id: str, target: List[float]):
        """
        Orient head/eyes toward target position.

        JavaScript:
            context.noodle.pose.lookAt('red', [0, 1.5, 2]);
        """
        # Compute head/eye muscles to look at target
        # Uses IK solver internally
        pass

    def breathe(self, entity_id: str, intensity: float = 1.0):
        """
        Apply subtle breathing motion.

        JavaScript:
            context.noodle.pose.breathe('red', 0.5);
        """
        t = time.time()
        breath = math.sin(t * 0.5) * 0.1 * intensity
        self.setMuscles(entity_id, {
            'Chest.FrontBack': breath,
            'Spine.FrontBack': breath * 0.5,
        })
```

---

## Performance

### Benchmarks (Target)

| Gaussians | CPU Skinning | GPU Skinning | Notes |
|-----------|--------------|--------------|-------|
| 10K | 2ms | 0.2ms | Simple avatar |
| 50K | 10ms | 0.5ms | Detailed avatar |
| 100K | 20ms | 1ms | High-detail |
| 200K | 40ms | 2ms | Extreme (LOD recommended) |

### Optimization Strategies

1. **Lazy Computation**
   - Only recompute when muscles change
   - Cache bone matrices between frames if pose unchanged

2. **LOD Integration**
   - Reduce Gaussian count at distance
   - Skip spring bone sim for distant characters

3. **GPU Skinning (Future)**
   ```glsl
   // Compute shader approach
   layout(local_size_x = 256) in;

   buffer RestPositions { vec4 rest_pos[]; };
   buffer PosedPositions { vec4 posed_pos[]; };
   buffer BoneIndices { uvec4 bone_idx[]; };
   buffer BoneWeights { vec4 bone_wgt[]; };
   uniform mat4 bone_matrices[MAX_BONES];

   void main() {
       uint i = gl_GlobalInvocationID.x;
       vec4 pos = vec4(0);
       for (int j = 0; j < 4; j++) {
           uint bone = bone_idx[i][j];
           float weight = bone_wgt[i][j];
           pos += weight * (bone_matrices[bone] * rest_pos[i]);
       }
       posed_pos[i] = pos;
   }
   ```

4. **SIMD/Vectorization**
   - Use numpy vectorized operations
   - Avoid Python loops for skinning

---

## Implementation Phases

### Phase 1: Core Skinning (Week 1)
- [ ] `muscle_retargeter.py` - muscle to bone rotation
- [ ] `skeleton_poser.py` - hierarchy traversal, matrix computation
- [ ] `gaussian_skinner.py` - LBS implementation
- [ ] Unit tests with known poses
- [ ] Manual testing with hardcoded muscle values

### Phase 2: Integration (Week 2)
- [ ] `RadianceComponent` pose state and API
- [ ] `GaussianRenderer` integration
- [ ] `pose_api.py` scripting extensions
- [ ] Viewer panel: pose sliders for debugging

### Phase 3: Animation Playback (Week 3)
- [ ] `PoseTrackPlayer` integration with components
- [ ] Affect + Pose synchronization (`.noodletrack`)
- [ ] Timeline editor integration
- [ ] Demo: breathing, head tracking, simple gestures

### Phase 4: Polish (Week 4)
- [ ] `SpringBoneSimulator` for hair/cloth
- [ ] DQS option for better joint quality
- [ ] GPU skinning (if CPU proves too slow)
- [ ] LOD integration
- [ ] Performance profiling and optimization

---

## Testing

### Unit Tests

```python
def test_muscle_retargeter_identity():
    """Empty muscles should produce identity rotations."""
    skeleton = create_test_skeleton()
    retargeter = MuscleRetargeter(skeleton)

    rotations = retargeter.retarget({})

    for i in range(len(skeleton.bones)):
        assert np.allclose(rotations[i], [0, 0, 0, 1])  # Identity

def test_skeleton_poser_rest_pose():
    """Rest pose should produce identity skinning matrices."""
    skeleton = create_test_skeleton()
    poser = SkeletonPoser(skeleton)

    identity_rots = np.zeros((len(skeleton.bones), 4))
    identity_rots[:, 3] = 1.0

    matrices = poser.compute_skinning_matrices(identity_rots)

    for i in range(len(skeleton.bones)):
        assert np.allclose(matrices[i], np.eye(4), atol=1e-5)

def test_gaussian_skinner_identity():
    """Identity skinning should not move Gaussians."""
    skinner = GaussianSkinner(method='lbs')

    positions = np.random.randn(100, 3).astype(np.float32)
    rotations = np.zeros((100, 4), dtype=np.float32)
    rotations[:, 3] = 1.0

    bone_indices = np.zeros((100, 4), dtype=np.uint16)
    bone_weights = np.zeros((100, 4), dtype=np.float32)
    bone_weights[:, 0] = 1.0

    identity_matrices = np.eye(4, dtype=np.float32)[np.newaxis].repeat(10, axis=0)

    posed_pos, posed_rot = skinner.skin(
        positions, rotations, bone_indices, bone_weights, identity_matrices
    )

    assert np.allclose(posed_pos, positions)
    assert np.allclose(posed_rot, rotations)
```

### Visual Tests

1. **T-Pose Verification**: Load VRM, convert to radiance, render - should match original
2. **Single Muscle**: Set one muscle to extremes, verify correct bone moves
3. **Full Body Pose**: Apply known pose track frame, compare to reference
4. **Spring Bones**: Enable physics, verify hair/cloth moves naturally

---

## References

- Unity Mecanim: https://docs.unity3d.com/Manual/MecanimHumanoids.html
- VRM Specification: https://vrm.dev/en/vrm/vrm_about/
- Dual Quaternion Skinning: https://www.cs.utah.edu/~ladislav/dq/dqs.pdf
- Linear Blend Skinning: Classic CG technique, see any rigging textbook

---

**Ordnung muss sein!**
