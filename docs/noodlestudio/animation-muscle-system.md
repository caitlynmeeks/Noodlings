# Animation Muscle System

**Status**: Production
**Last Updated**: 2026-01-08
**Authors**: Caity + Claude
**Inspiration**: Unity Mecanim

---

## Overview

NoodleStudio uses a **muscle-based humanoid animation system** inspired by Unity's Mecanim. Instead of animating bone rotations directly (which are rig-specific), we animate in **muscle space** - a normalized, rig-agnostic representation that can be applied to any humanoid skeleton.

### Why Muscles Instead of Bones?

| Bone Rotations | Muscle Values |
|----------------|---------------|
| Rig-specific (bone names vary) | Universal (47 standard muscles) |
| Euler angles (axis order matters) | Normalized [-1, 1] range |
| Can't share between rigs | One animation fits all humanoids |
| Requires manual retargeting | Automatic retargeting built-in |

**Example**: A "wave" animation created for a Mixamo rig can be applied to any VRM avatar without modification - the muscle system handles the translation.

---

## Core Concepts

### Muscle Space

Every humanoid pose is represented as 47 muscle values, each in the range [-1, 1]:
- **-1**: Minimum rotation (e.g., arm fully down)
- **0**: Rest/T-pose position
- **+1**: Maximum rotation (e.g., arm fully up)

The actual rotation degrees depend on the muscle definition:
```python
'LeftArm.DownUp': {
    'axis': 'Z',        # Which bone axis
    'min_deg': -60,     # Degrees at muscle value -1
    'max_deg': 170,     # Degrees at muscle value +1
    'default': 0        # Rest pose value
}
```

### Retargeting Flow

```
┌─────────────────────────────────────┐
│  Source Animation (Mixamo FBX)       │
│  - Bone rotations per frame          │
│  - Mixamo-specific bone names        │
└──────────────┬──────────────────────┘
               │
        [RetargetFrom]
        AnimationRetargeter
               │
               ↓
┌─────────────────────────────────────┐
│  PoseTrack (Muscle Space)            │
│  - 47 muscle channels                │
│  - Normalized [-1, 1] values         │
│  - Rig-agnostic, portable            │
└──────────────┬──────────────────────┘
               │
        [RetargetTo]
        PoseRetargeter
               │
               ↓
┌─────────────────────────────────────┐
│  Target Avatar (VRM)                 │
│  - Avatar-specific bone rotations    │
│  - Applied via skeleton binding      │
└─────────────────────────────────────┘
```

---

## Standard Humanoid Muscles (47)

### Body / Spine (9 muscles)

| Muscle | Description | Range |
|--------|-------------|-------|
| `Spine.FrontBack` | Bend forward/backward | -40° to +40° |
| `Spine.LeftRight` | Bend left/right | -40° to +40° |
| `Spine.TwistLeftRight` | Twist left/right | -40° to +40° |
| `Chest.FrontBack` | Chest bend forward/back | -40° to +40° |
| `Chest.LeftRight` | Chest bend left/right | -40° to +40° |
| `Chest.TwistLeftRight` | Chest twist | -40° to +40° |
| `UpperChest.FrontBack` | Upper chest bend | -20° to +20° |
| `UpperChest.LeftRight` | Upper chest side bend | -20° to +20° |
| `UpperChest.TwistLeftRight` | Upper chest twist | -20° to +20° |

### Head / Neck (6 muscles)

| Muscle | Description | Range |
|--------|-------------|-------|
| `Neck.NodDownUp` | Nod down/up | -40° to +40° |
| `Neck.TiltLeftRight` | Tilt head left/right | -40° to +40° |
| `Neck.TurnLeftRight` | Turn head left/right | -40° to +40° |
| `Head.NodDownUp` | Additional head nod | -40° to +40° |
| `Head.TiltLeftRight` | Additional head tilt | -40° to +40° |
| `Head.TurnLeftRight` | Additional head turn | -40° to +40° |

### Eyes / Jaw (5 muscles)

| Muscle | Description | Range |
|--------|-------------|-------|
| `LeftEye.DownUp` | Left eye vertical | -15° to +12° |
| `LeftEye.InOut` | Left eye horizontal | -20° to +20° |
| `RightEye.DownUp` | Right eye vertical | -15° to +12° |
| `RightEye.InOut` | Right eye horizontal | -20° to +20° |
| `Jaw.Close` | Jaw open/close | 0° to +30° |

### Left Arm (9 muscles)

| Muscle | Description | Range |
|--------|-------------|-------|
| `LeftShoulder.DownUp` | Shoulder shrug | -15° to +30° |
| `LeftShoulder.FrontBack` | Shoulder forward/back | -15° to +15° |
| `LeftArm.DownUp` | Arm raise/lower | -60° to +170° |
| `LeftArm.FrontBack` | Arm forward/back | -60° to +100° |
| `LeftArm.TwistInOut` | Upper arm twist | -90° to +90° |
| `LeftForeArm.Stretch` | Elbow bend | 0° to +150° |
| `LeftForeArm.TwistInOut` | Forearm twist | -90° to +90° |
| `LeftHand.DownUp` | Wrist flex/extend | -80° to +80° |
| `LeftHand.InOut` | Wrist deviation | -40° to +40° |

### Right Arm (9 muscles)

Mirror of Left Arm: `RightShoulder.DownUp`, `RightArm.DownUp`, etc.

### Left Leg (8 muscles)

| Muscle | Description | Range |
|--------|-------------|-------|
| `LeftUpperLeg.FrontBack` | Thigh forward/back | -50° to +120° |
| `LeftUpperLeg.InOut` | Thigh spread | -60° to +40° |
| `LeftUpperLeg.TwistInOut` | Thigh twist | -60° to +60° |
| `LeftLowerLeg.Stretch` | Knee bend | 0° to +150° |
| `LeftLowerLeg.TwistInOut` | Shin twist | -30° to +30° |
| `LeftFoot.UpDown` | Ankle flex | -50° to +50° |
| `LeftFoot.TwistInOut` | Ankle twist | -30° to +30° |
| `LeftToes.UpDown` | Toe curl | -50° to +50° |

### Right Leg (8 muscles)

Mirror of Left Leg: `RightUpperLeg.FrontBack`, `RightLowerLeg.Stretch`, etc.

---

## PoseTrack File Format

Animations are stored as `.posetrack` YAML files:

```yaml
name: Wave Hello
duration: 2.5
fps: 30
author: Caity
created: 2026-01-08
archetype: greeting
tags: [friendly, casual, wave]

# Optional root motion
root_motion:
  position:
    - {time: 0.0, value: [0, 0, 0]}
    - {time: 2.5, value: [0, 0, 0.5]}
  rotation:
    - {time: 0.0, value: [0, 0, 0, 1]}

# Muscle channels
muscles:
  RightArm.DownUp:
    - {time: 0.0, value: 0.0}
    - {time: 0.5, value: 0.8, easing: ease_out}
    - {time: 1.5, value: 0.8}
    - {time: 2.0, value: 0.0, easing: ease_in}

  RightArm.FrontBack:
    - {time: 0.0, value: 0.0}
    - {time: 0.5, value: 0.3}
    - {time: 2.0, value: 0.0}

  RightForeArm.Stretch:
    - {time: 0.0, value: 0.2}
    - {time: 0.5, value: 0.1}
    - {time: 1.0, value: 0.3}
    - {time: 1.5, value: 0.1}
    - {time: 2.0, value: 0.2}

  # ... other muscles

# Blend shapes (facial)
blend_shapes:
  happy:
    - {time: 0.0, value: 0.0}
    - {time: 0.5, value: 0.6}
    - {time: 2.0, value: 0.6}
    - {time: 2.5, value: 0.0}

# Sync markers
markers:
  - {time: 0.5, name: wave_start}
  - {time: 1.5, name: wave_peak}
  - {time: 2.0, name: wave_end}
```

### Keyframe Easing

Supported easing functions:
- `linear` (default)
- `ease_in` - Slow start
- `ease_out` - Slow end
- `ease_in_out` - Slow start and end
- `bezier` - Custom bezier curve (requires `control_points`)

---

## Key Classes

### PoseTrack

Container for a complete animation in muscle space.

```python
from noodlestudio.core.pose_track import PoseTrack

# Load from file
track = PoseTrack.load_yaml('wave.posetrack')

# Access metadata
print(track.name)      # "Wave Hello"
print(track.duration)  # 2.5
print(track.fps)       # 30

# Access channels
for muscle, channel in track.muscle_channels.items():
    print(f"{muscle}: {len(channel.keyframes)} keyframes")

# Sample at time
pose = track.sample(0.75)  # Returns PoseState
```

### PoseState

Snapshot of a complete body pose at a moment in time.

```python
from noodlestudio.core.pose_track import PoseState

pose = track.sample(1.0)

# Access muscle values (normalized -1 to 1)
print(pose.muscles['RightArm.DownUp'])  # e.g., 0.8
print(pose.muscles['Spine.FrontBack'])  # e.g., 0.0

# Access blend shapes
print(pose.blend_shapes.get('happy', 0))  # e.g., 0.6

# Access root motion
print(pose.root_position)   # [0, 0, 0.2]
print(pose.root_rotation)   # [0, 0, 0, 1]
```

### PoseTrackPlayer

Playback controller with time management.

```python
from noodlestudio.core.pose_track import PoseTrackPlayer

player = PoseTrackPlayer(track)
player.speed = 1.0
player.is_looping = True

player.play()

# In update loop:
pose = player.update()  # Returns current PoseState
if pose:
    apply_to_avatar(pose)

# Controls
player.pause()
player.stop()
player.seek(1.5)  # Jump to time

# Callbacks
player.on_marker('wave_peak', lambda: print("Peak!"))
player.on_complete(lambda: print("Done!"))
```

### PoseRetargeter

Converts muscle values to bone rotations for a specific avatar.

```python
from noodlestudio.core.pose_track import PoseRetargeter

# Create retargeter for avatar
retargeter = PoseRetargeter(avatar_config)

# Convert pose to bone rotations
bone_rotations = retargeter.apply_pose(pose)
# Returns: Dict[bone_name, Tuple[euler_x, euler_y, euler_z]]

# Apply to skeleton
for bone_name, rotation in bone_rotations.items():
    skeleton.set_bone_rotation(bone_name, rotation)
```

---

## Scripting API

The `context.noodle.pose` API exposes muscle control to ScriptedFacets.

### Loading and Playing

```javascript
// Load a pose track
const track = await context.noodle.pose.loadTrack('wave.posetrack');

// Play with options
context.noodle.pose.play({
    speed: 1.0,
    loop: false,
    startTime: 0.0
});

// Basic controls
context.noodle.pose.pause();
context.noodle.pose.stop();
context.noodle.pose.seek(1.5);
```

### Direct Muscle Control

```javascript
// Set individual muscle values (procedural animation)
context.noodle.pose.setMuscle('Head.TurnLeftRight', 0.3);
context.noodle.pose.setMuscle('RightArm.DownUp', 0.5);

// Get current muscle values
const muscles = context.noodle.pose.getMuscles();
console.log(muscles['Spine.FrontBack']);

// Get muscle definition (for range info)
const def = context.noodle.pose.getMuscleDefinition('LeftArm.DownUp');
// { axis: 'Z', min_deg: -60, max_deg: 170, default: 0 }
```

### Blend Shapes

```javascript
// Get current blend shapes
const shapes = context.noodle.pose.getBlendShapes();
console.log(shapes.happy);  // 0.6

// Blend shapes are typically animated via PoseTrack
// but can be set directly through affect system
```

### Markers and Callbacks

```javascript
// React to animation markers
context.noodle.pose.onMarker('wave_start', () => {
    console.log('Wave starting!');
    context.noodle.affect.nudge({ valence: 0.1 });
});

context.noodle.pose.onComplete(() => {
    console.log('Animation finished');
});
```

### Momentum (Physics Decay)

When transitioning out of a pose, momentum provides natural decay:

```javascript
// After animation ends, add momentum for smooth settle
context.noodle.pose.setMomentum(
    ['RightArm.DownUp', 'RightForeArm.Stretch'],
    {
        decay: 0.95,      // Per-frame multiplier
        duration: 0.5,    // Seconds to settle
        targetPose: null  // null = return to rest
    }
);
```

---

## FBX Import (Mixamo, etc.)

Convert bone-based animations to muscle space:

```python
from noodlestudio.core.fbx_importer import import_fbx_animation

# Import Mixamo animation
track = import_fbx_animation(
    'dancing.fbx',
    source_type='mixamo'  # or 'unity', 'custom'
)

# Save as portable PoseTrack
track.save_yaml('dancing.posetrack')

# Now usable on ANY humanoid avatar
```

### Supported Source Rigs

| Source | Bone Mapping |
|--------|--------------|
| `mixamo` | Mixamo auto-rigged characters |
| `unity` | Unity Humanoid rig |
| `custom` | Provide custom bone map |

### Custom Bone Mapping

```python
custom_map = {
    'MyHead': 'Head',
    'MyNeck': 'Neck',
    'MySpine1': 'Spine',
    'MySpine2': 'Chest',
    # ...
}

track = import_fbx_animation(
    'custom_rig.fbx',
    source_type='custom',
    bone_map=custom_map
)
```

---

## Avatar Binding

Before muscle values can be applied, an avatar needs a muscle binding:

```python
from noodlestudio.core.model_importer import ModelImporter

importer = ModelImporter()
model = importer.import_model('avatar.vrm')

# VRM avatars automatically get muscle binding via humanoid_map
binding = importer.create_muscle_binding(model)

# Binding contains:
# - bone_to_humanoid: Maps model bone index → HumanoidBone enum
# - rest_rotations: T-pose rotations for each bone
# - muscle_scales: Per-avatar adjustments (if needed)
```

### VRM Integration

VRM files include a `humanoid` mapping that directly corresponds to our muscle system:

```python
# VRM humanoid bones → NoodleStudio HumanoidBone
vrm_map = {
    'hips': HumanoidBone.HIPS,
    'spine': HumanoidBone.SPINE,
    'chest': HumanoidBone.CHEST,
    'upperChest': HumanoidBone.UPPER_CHEST,
    'neck': HumanoidBone.NECK,
    'head': HumanoidBone.HEAD,
    'leftShoulder': HumanoidBone.LEFT_SHOULDER,
    'leftUpperArm': HumanoidBone.LEFT_UPPER_ARM,
    # ... etc
}
```

---

## Integration with Facets

### PoseTrackFacet

A facet that plays pose tracks in cognitive assemblies:

```yaml
# In assembly.yaml
nodes:
  - type: PoseTrackFacet
    name: wave_player
    properties:
      track: wave.posetrack
      trigger_on_input: true
      blend_time: 0.3

connections:
  - from: INCOMING
    to: wave_player.trigger
  - from: wave_player.complete
    to: OUTGOING
```

### Affect-Driven Animation

Connect affect states to pose selection:

```yaml
nodes:
  - type: AffectToAnimFacet
    name: affect_anim
    properties:
      mappings:
        high_valence: happy_idle.posetrack
        low_valence: sad_idle.posetrack
        high_arousal: excited_idle.posetrack
```

---

## File Locations

| Component | Path |
|-----------|------|
| Core System | `noodlestudio/core/pose_track.py` |
| Scripting API | `noodlestudio/scripting/pose_api.py` |
| FBX Importer | `noodlestudio/core/fbx_importer.py` |
| Model Importer | `noodlestudio/core/model_importer.py` |
| VRM Parser | `noodlestudio/core/semantic_world/vrm_parser.py` |
| Track Editor UI | `noodlestudio/panels/animation_track_editor.py` |
| VRM Preview | `noodlestudio/panels/vrm_preview_panel.py` |

---

## Examples

### Simple Wave Animation

```python
from noodlestudio.core.pose_track import (
    PoseTrack, MuscleChannel, Keyframe
)

track = PoseTrack(name="Simple Wave", duration=2.0, fps=30)

# Right arm up
track.add_channel('RightArm.DownUp', MuscleChannel([
    Keyframe(0.0, 0.0),
    Keyframe(0.5, 0.8, easing='ease_out'),
    Keyframe(1.5, 0.8),
    Keyframe(2.0, 0.0, easing='ease_in'),
]))

# Elbow bend for wave
track.add_channel('RightForeArm.Stretch', MuscleChannel([
    Keyframe(0.0, 0.2),
    Keyframe(0.7, 0.1),
    Keyframe(1.0, 0.4),
    Keyframe(1.3, 0.1),
    Keyframe(2.0, 0.2),
]))

track.save_yaml('wave.posetrack')
```

### Procedural Head Tracking

```javascript
// In ScriptedFacet - make character look at target
function updateHeadTracking(targetPosition) {
    const noodling = context.noodle.scene.getNoodling();
    const headPos = noodling.getBoneWorldPosition('Head');

    // Calculate direction to target
    const dir = normalize(subtract(targetPosition, headPos));

    // Convert to muscle values
    const yaw = Math.atan2(dir.x, dir.z) * (180 / Math.PI);
    const pitch = Math.asin(dir.y) * (180 / Math.PI);

    // Normalize to muscle range and apply
    context.noodle.pose.setMuscle('Head.TurnLeftRight', yaw / 80);
    context.noodle.pose.setMuscle('Head.NodDownUp', pitch / 40);
}
```

### Blend Between Poses

```javascript
// Blend between two poses based on affect
const valence = context.noodle.affect.current.valence;

// Get muscle values from both poses
const happyMuscles = happyPose.muscles;
const sadMuscles = sadPose.muscles;

// Blend based on valence (0 = sad, 1 = happy)
const t = (valence + 1) / 2;  // Map [-1,1] to [0,1]

for (const muscle of Object.keys(happyMuscles)) {
    const blended = lerp(sadMuscles[muscle], happyMuscles[muscle], t);
    context.noodle.pose.setMuscle(muscle, blended);
}
```

---

## See Also

- [Facet System](facets.md) - Cognitive assembly pipelines
- [Affect Model](affect-model.md) - Emotional state system
- [VRM Parser](vrm-parser.md) - Avatar loading
- [Scripting API](scripting.md) - JavaScript API reference

---

*"Muscles, not bones. Portability, not lock-in."*
