# Noodle Animation System

A unified, rig-agnostic animation format for affect and body pose.

## The Core Insight

**Mecanim's muscle space is like our affect space** - both are:
- Normalized representations (-1 to +1 or 0 to 1)
- Interpolatable between keyframes (scalar values, not quaternions)
- Abstracted from underlying implementation (rig-agnostic)
- Directly mappable to different targets

This means we can use the **same track system** for both:
- **Affect Animation** (PAD+BS channels): valence, arousal, dominance, boredom, sorrow
- **Pose Animation** (muscle channels): LeftArm.Stretch, Head.NodDown, LeftEye.LookIn, etc.

## Architecture

```
                    NoodleTrack (base)
                          |
          +---------------+---------------+
          |                               |
     AffectTrack                     PoseTrack
   (PAD+BS channels)            (muscle channels)
          |                               |
          v                               v
    CharmNetwork                   PoseRetargeter
  (affect → behavior)            (muscles → bones)
          |                               |
          v                               v
    Facet Assembly                    Avatar
   (dialogue, actions)            (VRM, skeleton)
```

## Channel Namespaces

### Affect Channels (default PAD+BS)

```yaml
affect:
  valence: 0.0      # -1 (negative) to +1 (positive)
  arousal: 0.5      # 0 (calm) to 1 (excited)
  dominance: 0.5    # 0 (submissive) to 1 (dominant)
  boredom: 0.0      # 0 (engaged) to 1 (bored)
  sorrow: 0.0       # 0 (content) to 1 (sad)
  # Extensible for alien intelligences:
  # resonance: 0.0
  # crystallinity: 0.5
```

### Muscle Channels (~47 standard, inspired by Mecanim)

```yaml
muscles:
  # Body
  Spine.FrontBack: 0.0        # -1 (back) to 1 (front)
  Spine.LeftRight: 0.0
  Spine.TwistLeftRight: 0.0
  Chest.FrontBack: 0.0
  Chest.LeftRight: 0.0
  Chest.TwistLeftRight: 0.0
  UpperChest.FrontBack: 0.0
  UpperChest.LeftRight: 0.0
  UpperChest.TwistLeftRight: 0.0
  Neck.NodDownUp: 0.0         # -1 (down) to 1 (up)
  Neck.TiltLeftRight: 0.0
  Neck.TurnLeftRight: 0.0
  Head.NodDownUp: 0.0
  Head.TiltLeftRight: 0.0
  Head.TurnLeftRight: 0.0

  # Eyes
  LeftEye.DownUp: 0.0
  LeftEye.InOut: 0.0
  RightEye.DownUp: 0.0
  RightEye.InOut: 0.0
  Jaw.Close: 0.0

  # Left Arm
  LeftShoulder.DownUp: 0.0
  LeftShoulder.FrontBack: 0.0
  LeftArm.DownUp: 0.0
  LeftArm.FrontBack: 0.0
  LeftArm.TwistInOut: 0.0
  LeftForeArm.Stretch: 0.0    # 0 (bent) to 1 (straight)
  LeftForeArm.TwistInOut: 0.0
  LeftHand.DownUp: 0.0
  LeftHand.InOut: 0.0

  # Right Arm (mirror of left)
  RightShoulder.DownUp: 0.0
  RightShoulder.FrontBack: 0.0
  RightArm.DownUp: 0.0
  RightArm.FrontBack: 0.0
  RightArm.TwistInOut: 0.0
  RightForeArm.Stretch: 0.0
  RightForeArm.TwistInOut: 0.0
  RightHand.DownUp: 0.0
  RightHand.InOut: 0.0

  # Left Leg
  LeftUpperLeg.FrontBack: 0.0
  LeftUpperLeg.InOut: 0.0
  LeftUpperLeg.TwistInOut: 0.0
  LeftLowerLeg.Stretch: 0.0
  LeftLowerLeg.TwistInOut: 0.0
  LeftFoot.UpDown: 0.0
  LeftFoot.TwistInOut: 0.0
  LeftToes.UpDown: 0.0

  # Right Leg (mirror of left)
  RightUpperLeg.FrontBack: 0.0
  RightUpperLeg.InOut: 0.0
  RightUpperLeg.TwistInOut: 0.0
  RightLowerLeg.Stretch: 0.0
  RightLowerLeg.TwistInOut: 0.0
  RightFoot.UpDown: 0.0
  RightFoot.TwistInOut: 0.0
  RightToes.UpDown: 0.0

  # Fingers (optional, 30 more values)
  # LeftThumb.1.Spread, LeftThumb.1.Stretched, etc.
```

### Blend Shape Channels (face)

```yaml
blendshapes:
  # VRM standard expressions
  happy: 0.0
  angry: 0.0
  sad: 0.0
  relaxed: 0.0
  surprised: 0.0

  # Visemes
  aa: 0.0
  ih: 0.0
  ou: 0.0
  ee: 0.0
  oh: 0.0

  # Blink
  blink: 0.0
  blinkLeft: 0.0
  blinkRight: 0.0

  # Look direction (can also use eye muscles)
  lookUp: 0.0
  lookDown: 0.0
  lookLeft: 0.0
  lookRight: 0.0
```

## File Format: .noodletrack

```yaml
# Unified track format for affect and pose
version: 1.0
type: combined                    # affect, pose, blendshape, or combined
fps: 30
duration: 5.0

# Root motion (world space)
root:
  position:
    - time: 0.0
      value: [0, 0, 0]
      interpolation: linear
    - time: 5.0
      value: [1, 0, 0]
  rotation:
    - time: 0.0
      value: [0, 0, 0, 1]         # quaternion

# Affect channels
affect:
  valence:
    - time: 0.0
      value: 0.0
      interpolation: bezier
      handles: [0.1, 0.0, 0.2, 0.5]
    - time: 2.0
      value: 0.8
    - time: 5.0
      value: 0.2
  arousal:
    - time: 0.0
      value: 0.3
    - time: 2.0
      value: 0.9

# Muscle channels
muscles:
  Head.NodDownUp:
    - time: 0.0
      value: 0.0
    - time: 1.0
      value: -0.3
    - time: 2.0
      value: 0.2
  LeftArm.FrontBack:
    - time: 0.0
      value: 0.0
    - time: 2.5
      value: 0.7

# Blend shape channels
blendshapes:
  happy:
    - time: 0.0
      value: 0.0
    - time: 2.0
      value: 0.8

# Markers for scripted events
markers:
  - time: 2.0
    name: peak_emotion
    data: { trigger: "tears_start" }
  - time: 4.5
    name: transition
    data: { next_track: "calm_down.noodletrack" }
```

## Retargeting Pipeline

### Import (RetargetFrom)

```
Source Animation          Noodle Track
   (FBX, BVH)     -->    (normalized)
       |                      |
  Bone rotations    Muscle values [-1, 1]
  + World position    + Root position/rotation
  + Blend shapes      + Blend shape values
```

1. Parse source animation (bone keyframes)
2. For each bone, compute muscle values using inverse mapping
3. Store as normalized channel values in .noodletrack

### Runtime (RetargetTo)

```
Noodle Track           Target Avatar
(normalized)    -->    (VRM, skeleton)
     |                      |
Muscle values      Bone rotations
     |                      |
Read avatar's      Apply per-bone
muscle ranges      rotation limits
```

1. Load avatar's muscle definitions (min/max per axis)
2. For each muscle value, compute bone rotation
3. Apply IK correction for hand/foot placement (optional)

### Avatar Muscle Definition

Each avatar defines how muscles map to bones:

```yaml
# avatar_muscles.yaml
humanoid: true
bones:
  Head:
    node: "head_bone"           # Bone name in skeleton
    axes:
      NodDownUp:
        axis: [1, 0, 0]         # Local rotation axis
        min: -40                # Degrees
        max: 60
      TiltLeftRight:
        axis: [0, 0, 1]
        min: -40
        max: 40
      TurnLeftRight:
        axis: [0, 1, 0]
        min: -70
        max: 70

  LeftArm:
    node: "arm_l"
    axes:
      DownUp:
        axis: [0, 0, 1]
        min: -40
        max: 100
      FrontBack:
        axis: [1, 0, 0]
        min: -60
        max: 100
      TwistInOut:
        axis: [0, 1, 0]
        min: -90
        max: 50
```

## Integration with Existing Systems

### Affect Track Facet

The existing `AffectTrackFacet` handles affect channels:

```javascript
// In ScriptedFacet
var track = context.noodle.affect.loadTrack("grief.noodletrack");
track.play();

// Sample affect at current time
var state = context.noodle.affect.getState();
context.log("Valence: " + state.valence);
```

### Pose Track API (new)

```javascript
// In ScriptedFacet
var pose = context.noodle.pose.loadTrack("wave.noodletrack");
pose.play();

// Apply to avatar
context.noodle.pose.setAvatar("yuki");
context.noodle.pose.apply();  // Retargets to avatar bones

// Sample muscles at current time
var muscles = context.noodle.pose.getMuscles();
context.log("Head nod: " + muscles["Head.NodDownUp"]);

// Direct muscle control
context.noodle.pose.setMuscle("LeftArm.FrontBack", 0.5);
```

### Combined Playback

```javascript
// Load combined track (affect + pose + blendshapes)
var anim = context.noodle.animation.load("greet_happily.noodletrack");
anim.play();

// Access different aspects
var affect = anim.affect;      // AffectTrackProxy
var pose = anim.pose;          // PoseTrackProxy
var face = anim.blendshapes;   // BlendShapeTrackProxy

// Blend with live CharmNetwork
context.noodle.affect.setBlendMode("weighted", {
  track: 0.3,      // 30% from animation
  live: 0.7        // 70% from CharmNetwork
});
```

## Momentum Handoff (Donald Duck Problem)

When a track ends, hand off final state to live systems:

### Affect Momentum

```javascript
// Track ending callback
track.onComplete = function(finalState) {
  // Inject final affect into CharmNetwork
  // Network's temporal dynamics will let it decay naturally
  context.noodle.affect.inject({
    valence: finalState.valence,
    arousal: finalState.arousal,
    dominance: finalState.dominance
  }, "natural");  // Use network's natural decay
};
```

### Pose Momentum

```javascript
pose.onComplete = function(finalMuscles) {
  // Hand off to procedural animation system
  context.noodle.pose.setMomentum(finalMuscles, {
    decay: "spring",        // Spring back to neutral
    stiffness: 0.5,
    damping: 0.3
  });
};
```

## Non-Humanoid Support

For non-humanoid characters (quadrupeds, centaurs, aliens):

```yaml
# Non-standard muscle definition
humanoid: false
archetype: quadruped

bones:
  # Custom bone mappings
  FrontLeftLeg:
    node: "leg_FL"
    axes:
      FrontBack: { axis: [1,0,0], min: -60, max: 60 }
      Spread: { axis: [0,0,1], min: -30, max: 30 }

  Tail:
    node: "tail_base"
    axes:
      UpDown: { axis: [1,0,0], min: -45, max: 45 }
      LeftRight: { axis: [0,0,1], min: -60, max: 60 }

  # Additional tail segments
  Tail.Mid: { ... }
  Tail.Tip: { ... }
```

## Implementation Files

```
noodlestudio/core/
  affect_track.py           # AffectTrack, AffectTrackFacet (DONE)
  pose_track.py             # PoseTrack, PoseTrackFacet (TODO)
  animation_track.py        # Unified NoodleTrack base (TODO)
  pose_retargeter.py        # Muscle → bone mapping (TODO)

noodlestudio/scripting/
  affect_api.py             # context.noodle.affect (DONE)
  pose_api.py               # context.noodle.pose (TODO)
  animation_api.py          # context.noodle.animation (TODO)

noodlestudio/core/semantic_world/
  vrm_parser.py             # VRM → skeleton (DONE)
  avatar_muscle_mapper.py   # VRM → muscle definitions (TODO)
```

## Sources

- [Unity Mecanim Humanoids Blog](https://unity.com/blog/engine-platform/mecanim-humanoids)
- [Unity Retargeting Manual](https://docs.unity3d.com/Manual/Retargeting.html)
- [Unity HumanPoseHandler API](https://docs.unity3d.com/ScriptReference/HumanPoseHandler.html)
- [Unity HumanTrait API](https://docs.unity3d.com/ScriptReference/HumanTrait.html)
- [Unity Muscle & Settings](https://docs.unity3d.com/Manual/MuscleDefinitions.html)

---

**Author**: Caitlyn + Claude
**Date**: December 21, 2025
