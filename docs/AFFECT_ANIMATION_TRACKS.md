# Affect Animation Tracks

**What Maya did for motion, we do for emotion.**

Keyframeable affect curves that let animators embed feeling directly into character performances. Like FBX animation tracks, but for the soul.

**Created:** December 21, 2025 (Highway 101 idea)
**Status:** Design concept

---

## The Insight

Animators already keyframe position, rotation, scale, blend shapes. Why not keyframe *how the character feels*?

```
Time:     0s      1s      2s      3s      4s      5s
          |       |       |       |       |       |
Valence:  ────────╱╲──────────────╲───────────────
          0.2    0.8     0.6      -0.3    -0.2

Arousal:  ──────────────╱╲────────────────────────
          0.3    0.3    0.9      0.7     0.4

Dominance: ╲──────────────────────╱───────────────
           0.7    0.5    0.4     0.6     0.7
```

The character's expressions, body language, voice tone, and decision-making all flow from these curves. The animator sculpts the emotional journey; the system expresses it.

---

## Use Cases

### 1. Cinematic Sequences
Pre-authored emotional beats for cutscenes:
- Character receives bad news (valence drops, arousal spikes)
- Slowly processes grief (arousal decays, sorrow rises)
- Finds resolve (dominance climbs, valence stabilizes)

### 2. Scripted Performances
NPCs with authored emotional arcs:
- Shopkeeper who gets progressively more annoyed
- Guide who grows more excited as you approach the destination
- Villain whose confidence wavers during boss fight

### 3. Hybrid Live + Authored
Blend pre-authored affect with live CharmNetwork:
- Base emotional arc from track
- Real-time modulation from player interaction
- Weighted blend: 70% track, 30% live

### 4. Motion Capture Enhancement
Capture actor's performance, but author the internal state:
- Mocap gives you the body
- Affect track gives you the soul
- Together: complete performance

---

## File Format: .affecttrack

### Basic Structure

```yaml
# character_reaction.affecttrack
format: "affect-track"
version: "1.0"

metadata:
  name: "Receiving Bad News"
  duration: 8.5          # seconds
  fps: 30                # sample rate for export
  author: "Caitlyn"
  created: "2025-12-21"
  tags: ["dramatic", "grief", "reaction"]

# The five affect dimensions
channels:
  valence:
    interpolation: "bezier"    # bezier, linear, step, hermite
    keyframes:
      - time: 0.0
        value: 0.6
        in_tangent: [0, 0]
        out_tangent: [0.2, 0]

      - time: 1.2
        value: 0.1             # shock
        in_tangent: [-0.3, 0.5]
        out_tangent: [0.1, -0.2]

      - time: 3.5
        value: -0.4            # grief settles
        in_tangent: [-0.2, 0]
        out_tangent: [0.5, 0.1]

      - time: 8.5
        value: -0.2            # numb acceptance
        in_tangent: [-0.1, 0]
        out_tangent: [0, 0]

  arousal:
    interpolation: "bezier"
    keyframes:
      - time: 0.0
        value: 0.4

      - time: 1.0
        value: 0.9             # spike on news

      - time: 2.5
        value: 0.7             # still processing

      - time: 8.5
        value: 0.3             # exhausted

  dominance:
    interpolation: "bezier"
    keyframes:
      - time: 0.0
        value: 0.7             # confident before

      - time: 1.5
        value: 0.2             # lost control

      - time: 6.0
        value: 0.5             # regaining composure

      - time: 8.5
        value: 0.6

  boredom:
    interpolation: "linear"
    keyframes:
      - time: 0.0
        value: 0.0             # fully engaged throughout
      - time: 8.5
        value: 0.0

  sorrow:
    interpolation: "bezier"
    keyframes:
      - time: 0.0
        value: 0.0

      - time: 2.0
        value: 0.3             # sorrow emerging

      - time: 5.0
        value: 0.7             # peak grief

      - time: 8.5
        value: 0.5             # lingering

# Optional: markers for sync points
markers:
  - time: 1.0
    name: "news_delivered"

  - time: 3.5
    name: "tears_start"

  - time: 6.0
    name: "composure_begins"

# Optional: blend regions with live affect
blend_regions:
  - start: 4.0
    end: 6.0
    live_weight: 0.3           # Allow 30% live affect influence

# Optional: trigger events at keyframes
events:
  - time: 1.0
    event: "play_sound"
    data: { clip: "gasp.ogg" }

  - time: 3.5
    event: "start_tears"
    data: { intensity: 0.6 }
```

### Compact Binary Format (.affectbin)

For runtime efficiency:

```
Header (32 bytes):
  magic: "AFFT"           (4 bytes)
  version: uint16         (2 bytes)
  flags: uint16           (2 bytes)
  duration: float32       (4 bytes)
  fps: uint16             (2 bytes)
  num_channels: uint8     (1 byte)
  num_markers: uint16     (2 bytes)
  reserved: 15 bytes

Per Channel:
  channel_id: uint8       (0=valence, 1=arousal, 2=dominance, 3=boredom, 4=sorrow)
  interpolation: uint8    (0=linear, 1=bezier, 2=step, 3=hermite)
  num_keyframes: uint16

  Per Keyframe:
    time: float32
    value: float32
    in_tangent: float32[2]   (for bezier/hermite)
    out_tangent: float32[2]
```

---

## Curve Editor UI

### Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Affect Curve Editor                          [▶ Play] [⟲] [💾 Save]    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Channel Toggles:                                                        │
│ [■] Valence (pink)  [■] Arousal (orange)  [■] Dominance (blue)         │
│ [■] Boredom (gray)  [■] Sorrow (purple)                                │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ 1.0 ┤                                                                   │
│     │                    ●━━━━●                                         │
│ 0.5 ┤    ●━━━━━━━━━━━━━━╱      ╲━━━━━━━━━●                             │
│     │   ╱                                  ╲                            │
│ 0.0 ┤━━●                                    ╲━━━━━━━━━●                 │
│     │                                                                   │
│-0.5 ┤                                         ━━━━━━━━━━━━━●           │
│     │                                                                   │
│     └───┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬──── │
│         0s      1s      2s      3s      4s      5s      6s      7s     │
│                                                                         │
│ ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ▲ Playhead: 2.34s                                                       │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Selected Keyframe:                                                      │
│   Channel: Valence    Time: [2.0    ]s    Value: [0.65   ]             │
│   In Tangent: [-0.3, 0.2]    Out Tangent: [0.4, -0.1]                  │
│   Interpolation: [Bezier ▼]                                             │
│                                                                         │
│ [+ Add Key]  [- Delete]  [Flatten]  [Auto-Tangent]  [Copy] [Paste]     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Interactions

**Navigation:**
- Pan: Middle-drag or Alt+drag
- Zoom: Scroll wheel (vertical = value, horizontal = time with Shift)
- Frame all: F
- Frame selection: Shift+F

**Keyframe Editing:**
- Add keyframe: Double-click on curve, or Ctrl+click
- Select: Click keyframe
- Multi-select: Shift+click, or box select
- Move: Drag keyframe
- Delete: Delete key or Backspace

**Tangent Editing:**
- Show tangent handles: Click keyframe
- Adjust: Drag handle endpoints
- Break tangent: Alt+drag (independent in/out)
- Unify tangent: Shift+drag (mirror in/out)

**Curve Presets:**
- Right-click curve area → Apply Preset:
  - "Ease In"
  - "Ease Out"
  - "Ease In-Out"
  - "Linear"
  - "Step"
  - "Overshoot"
  - "Bounce"

### Expression Mapping Presets

Pre-built affect → expression mappings:

```yaml
# expression_mappings/realistic_human.yaml
mappings:
  # High valence + low arousal = gentle smile
  - condition:
      valence: [0.3, 1.0]
      arousal: [0.0, 0.4]
    expression:
      mouth_smile: 0.5
      eye_squint: 0.2
      brow_raise: 0.1

  # Low valence + high arousal = distress
  - condition:
      valence: [-1.0, -0.3]
      arousal: [0.6, 1.0]
    expression:
      brow_furrow: 0.8
      mouth_frown: 0.6
      eye_wide: 0.4
      nostril_flare: 0.3

  # High sorrow = tears + drooping
  - condition:
      sorrow: [0.5, 1.0]
    expression:
      tear_flow: { map: "sorrow", scale: 0.8 }
      eye_droop: 0.4
      mouth_downturn: 0.5
```

---

## Integration with NoodleStudio

### Profiler Timeline Integration

The existing Cognitive Timeline can display affect tracks:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Cognitive Timeline                           [REC] [PAUSE] [CLEAR]      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ CHARM_NET   ██░░░░██░░░░██░░░░██░░░░██░░░░██░░░░                       │
│                                                                         │
│ AFFECT TRACK "grief_reaction.affecttrack"                              │
│ Valence:    ────────╲________╱─────                                    │
│ Arousal:    ────────╱╲___________──                                    │
│ Sorrow:     ________╱────────────╲─                                    │
│             ▲ playing                                                   │
│                                                                         │
│ BLENDED OUTPUT                                                          │
│ Valence:    ────────╲____~~~~╱─────  (track + live)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Facet Integration

New facet type: **AffectTrackFacet**

```yaml
facets:
  - id: "grief_performance"
    type: "AFFECT_TRACK"
    name: "Grief Reaction"
    position: [500, 200]
    config:
      track: "tracks/grief_reaction.affecttrack"
      trigger: "on_bad_news"           # when to start
      blend_mode: "override"           # override, blend, additive
      blend_weight: 0.8                # if blending
      loop: false
      speed: 1.0
```

### Scripting API

```javascript
// Load and play affect track
var track = context.noodle.affect.loadTrack("grief_reaction.affecttrack");
track.play();

// Control playback
track.pause();
track.resume();
track.seek(3.5);  // jump to 3.5 seconds
track.speed = 0.5;  // half speed

// Query current values
var current = track.sample(track.currentTime);
// { valence: -0.3, arousal: 0.6, dominance: 0.3, boredom: 0.0, sorrow: 0.5 }

// Blend with live affect
context.noodle.affect.setBlendMode("weighted", {
    track: 0.7,
    live: 0.3
});

// Listen for markers
track.onMarker("tears_start", function() {
    context.noodle.particles.emit("tears", character.eyePosition);
});

// Create track programmatically
var newTrack = context.noodle.affect.createTrack();
newTrack.addKeyframe("valence", 0.0, 0.5);
newTrack.addKeyframe("valence", 2.0, -0.8, {
    interpolation: "bezier",
    outTangent: [0.5, -0.3]
});
newTrack.save("custom_reaction.affecttrack");
```

---

## Blend Modes

How authored tracks combine with live CharmNetwork output:

### Override
Track completely replaces live affect:
```
output = track
```

### Weighted Blend
Linear interpolation:
```
output = track * weight + live * (1 - weight)
```

### Additive
Track offsets live values:
```
output = clamp(live + (track - 0.5) * strength, -1, 1)
```

### Multiplicative
Track scales live values:
```
output = live * track
```

### Maximum
Take more extreme value:
```
output = sign(track) * max(abs(track), abs(live))
```

### Envelope
Track defines allowed range:
```
output = clamp(live, track_low, track_high)
```

---

## Animation Binding

### Blend Shape Mapping

Affect channels drive blend shapes automatically:

```yaml
# In noodling recipe or stage config
affect_bindings:
  blend_shapes:
    # Direct mappings
    - affect: "valence"
      positive:
        smile: { weight: 0.8 }
        brow_raise: { weight: 0.3 }
      negative:
        frown: { weight: 0.7 }
        brow_furrow: { weight: 0.5 }

    - affect: "arousal"
      high:
        eye_wide: { weight: 0.6 }
        nostril_flare: { weight: 0.3 }
      low:
        eye_droop: { weight: 0.4 }

    - affect: "sorrow"
      high:
        tear_flow: { weight: 1.0, threshold: 0.3 }
        eye_red: { weight: 0.5 }
```

### Body Language Mapping

Affect influences pose/animation selection:

```yaml
affect_bindings:
  posture:
    - condition: { dominance: [0.7, 1.0] }
      pose: "confident_stance"

    - condition: { dominance: [0.0, 0.3], valence: [-1.0, -0.3] }
      pose: "defeated_slump"

    - condition: { arousal: [0.7, 1.0] }
      animation_speed: 1.3  # faster movements

  gestures:
    - condition: { valence: [0.5, 1.0], arousal: [0.5, 1.0] }
      gesture_frequency: 1.5  # more animated

    - condition: { boredom: [0.6, 1.0] }
      idle_variance: 0.8  # fidgeting
```

### Voice Mapping

Affect modulates TTS parameters:

```yaml
affect_bindings:
  voice:
    pitch:
      base: 1.0
      valence_influence: 0.1      # happier = slightly higher
      arousal_influence: 0.15     # excited = higher

    speed:
      base: 1.0
      arousal_influence: 0.2      # excited = faster
      boredom_influence: -0.15    # bored = slower

    volume:
      base: 1.0
      arousal_influence: 0.3      # excited = louder
      dominance_influence: 0.2    # confident = louder
```

---

## Export/Import

### Export to Standard Formats

**To FBX (as custom property tracks):**
```
affect_valence    → Custom float track
affect_arousal    → Custom float track
affect_dominance  → Custom float track
affect_boredom    → Custom float track
affect_sorrow     → Custom float track
```

**To JSON (for web/runtime):**
```json
{
  "duration": 8.5,
  "channels": {
    "valence": {
      "times": [0, 1.2, 3.5, 8.5],
      "values": [0.6, 0.1, -0.4, -0.2],
      "interpolation": "bezier",
      "tangents": [...]
    }
  }
}
```

### Import from Motion Capture

Parse emotion recognition data from mocap sessions:

```python
# Hypothetical workflow
mocap_session = load_mocap("actor_performance.c3d")
emotion_data = analyze_facial_expressions(mocap_session)

affect_track = convert_to_affect_track(emotion_data, {
    "smile_intensity": "valence",
    "brow_furrow": lambda v: -v * 0.5,  # maps to negative valence
    "movement_energy": "arousal",
    # etc.
})

affect_track.save("captured_performance.affecttrack")
```

---

## Implementation Priority

### Phase 1: Core Format
- [ ] .affecttrack YAML parser/writer
- [ ] Runtime affect sampler (interpolation)
- [ ] Basic AffectTrackFacet

### Phase 2: Curve Editor
- [ ] Dockable curve editor panel
- [ ] Keyframe manipulation
- [ ] Tangent editing
- [ ] Playback controls

### Phase 3: Integration
- [ ] Blend with live CharmNetwork
- [ ] Expression mapping system
- [ ] Scripting API

### Phase 4: Advanced
- [ ] Binary format for runtime
- [ ] FBX export
- [ ] Mocap emotion import
- [ ] Multi-character synchronization

---

## Open Questions

1. **Sampling rate:** Fixed FPS or event-driven keyframes only?

2. **Compression:** For long tracks, support curve simplification?

3. **Layers:** Multiple tracks stacked with blend weights? (like animation layers)

4. **Retargeting:** Can one affect track be applied to different characters with different expression mappings?

5. **Procedural generation:** Tools to auto-generate affect tracks from:
   - Dialogue sentiment analysis
   - Music/audio energy
   - Story beat annotations

---

## Emotional Momentum (The Donald Duck Problem)

### The Insight

Animator creates a sequence where Donald Duck gets angry and loses his temper. The affect track shows valence plummeting, arousal spiking, dominance swinging wildly.

**The animation ends. What happens to the affect?**

Traditional approach: Snap back to neutral. Donald is suddenly calm. Feels wrong.

**Our approach: Emotional Momentum.**

When the affect track finishes, the final affect state becomes the *initial condition* for the live CharmNetwork. Donald's anger doesn't disappear - it **persists and naturally decays** through the temporal dynamics of his neural affect system.

```
ANIMATION PLAYING                    ANIMATION ENDS → LIVE CHARMNET

Valence:  ──╲______╱╲___[-0.7]  →   [-0.7]────────────────[0.0]
              authored track              natural decay (minutes)

Arousal:  ────╱╲____╱──[0.8]    →   [0.8]─────────────────[0.4]
              authored track              natural decay (seconds)
```

Donald finishes his tantrum animation, then walks around STILL GRUMPY. He snaps at other characters. His responses are colored by residual anger. Over time (governed by CharmNetwork decay rates), he calms down.

### Implementation

```yaml
# In AffectTrackFacet config
affect_track:
  path: "donald_tantrum.affecttrack"

  # What happens when track ends
  on_complete: "momentum"   # momentum | snap_neutral | hold | loop

  momentum_config:
    # Transfer final track values to CharmNetwork state
    transfer_to_live: true

    # Optional: scale the transfer (don't want full intensity?)
    transfer_scale: 0.9

    # Optional: blend time from track to live
    crossfade_duration: 0.5
```

### Affect State Handoff

```python
class AffectTrackFacet:
    def on_track_complete(self):
        if self.on_complete == "momentum":
            # Get final affect values from track
            final_affect = self.track.sample(self.track.duration)

            # Inject into CharmNetwork as current state
            charm_net = self.get_facet("charm_network")
            charm_net.inject_state(
                valence=final_affect.valence * self.transfer_scale,
                arousal=final_affect.arousal * self.transfer_scale,
                dominance=final_affect.dominance * self.transfer_scale,
                boredom=final_affect.boredom * self.transfer_scale,
                sorrow=final_affect.sorrow * self.transfer_scale,
                crossfade=self.crossfade_duration
            )

            # CharmNetwork now owns the state
            # Natural decay takes over
```

### Scripting API

```javascript
// Play track with momentum
var track = context.noodle.affect.loadTrack("tantrum.affecttrack");
track.play({
    onComplete: "momentum",
    transferScale: 0.9
});

// Query: is this residual from a track or purely live?
var state = context.noodle.affect.getState();
console.log(state.source);  // "live", "track", or "momentum_decay"
console.log(state.momentum_remaining);  // 0.0-1.0, how much track influence remains

// Manually inject affect (same as momentum handoff)
context.noodle.affect.inject({
    valence: -0.5,
    arousal: 0.8,
    decay: "natural"  // let CharmNetwork decay it
});
```

### Design Implications

**CharmNetwork decay rates matter more:**
- Fast layer: seconds (immediate reactions)
- Medium layer: minutes (mood)
- Slow layer: hours/days (temperament)

An angry outburst leaves Donald grumpy for MINUTES because the medium/slow layers absorbed the state. This is realistic - emotions have inertia.

**Animators control the setup, CharmNetwork controls the follow-through.**

---

## Universal Noodling Rig (The Mecanim Problem)

### The Challenge

Unity's Mecanim solved a huge problem: animations that work across ANY humanoid character regardless of skeleton differences. You can buy an animation pack and apply it to any character.

**VRChat depends entirely on this.** Thousands of avatars, millions of animations, all interoperable.

For Gaussian splat noodlings, we need the same thing:
- Animations authored once
- Work on any noodling body
- No per-character retargeting

### Mecanim's Solution (for reference)

Mecanim uses "muscle space" - instead of bone rotations, animations are stored as muscle activations (0-1 values for each degree of freedom). The runtime maps muscles to whatever skeleton is present.

```
Mecanim Animation         Any Humanoid Rig

LeftArmStretch: 0.7  →    Maps to actual shoulder/elbow bones
SpineForward: 0.3    →    Maps to actual spine bones
HeadTilt: -0.2       →    Maps to actual neck/head bones
```

### Our Solution: Semantic Pose Space

Instead of muscles (which assume bones), we use **semantic pose descriptors** that map to whatever representation the noodling uses - bones, blend shapes, Gaussian deformations, or procedural systems.

```yaml
# Universal pose descriptor
pose:
  # Body regions (hierarchical)
  torso:
    lean_forward: 0.3      # -1 to 1
    twist: 0.0
    bend_side: -0.1

  head:
    tilt: -0.2
    turn: 0.4
    nod: 0.1

  left_arm:
    reach_forward: 0.7
    reach_side: 0.2
    reach_up: 0.0
    bend: 0.5              # elbow bend
    twist: 0.0             # forearm rotation

  right_arm:
    reach_forward: 0.3
    reach_side: -0.1
    reach_up: 0.0
    bend: 0.2
    twist: 0.0

  left_hand:
    grip: 0.8              # fist
    spread: 0.0            # fingers apart
    point_index: 0.0       # pointing

  # For quadrupeds (like Yuki!)
  quadruped:
    front_left_leg:
      reach: 0.5
      lift: 0.3
    tail:
      curl: 0.4
      wag_phase: 0.7       # for cyclic wagging
      wag_amplitude: 0.3
    ears:
      left_perk: 0.8
      right_perk: 0.6

  # Face (universal across body types)
  face:
    # Use Apple ARKit standard or similar
    jaw_open: 0.2
    mouth_smile_left: 0.5
    mouth_smile_right: 0.5
    brow_inner_up: 0.0
    brow_outer_up_left: 0.3
    eye_blink_left: 0.0
    eye_blink_right: 0.0
    eye_look_up: 0.1
    eye_look_left: 0.0
    # ... full ARKit-style blend shape set
```

### Rig Definition

Each noodling defines how semantic poses map to their representation:

```yaml
# yuki_cyberfox/rig_definition.yaml
rig_type: "quadruped_fox"
version: "1.0"

# What body regions this rig supports
supported_regions:
  - torso
  - head
  - quadruped.front_left_leg
  - quadruped.front_right_leg
  - quadruped.back_left_leg
  - quadruped.back_right_leg
  - quadruped.tail
  - quadruped.ears
  - face

# How to apply poses to this specific rig
mappings:
  # Torso - maps to spine Gaussians
  torso.lean_forward:
    type: "gaussian_deform"
    affected_region: "spine_gaussians"
    transform: "rotate_x"
    scale: 30  # degrees per unit

  # Head - maps to head bone + Gaussians
  head.turn:
    type: "bone_rotation"
    bone: "head"
    axis: "y"
    scale: 80  # degrees

  # Tail - procedural wag
  quadruped.tail.wag_phase:
    type: "procedural"
    generator: "sine_wave"
    apply_to: "tail_chain"

  # Face - direct blend shape mapping
  face.mouth_smile_left:
    type: "blend_shape"
    target: "smile_L"
    scale: 1.0

  # Ears - spring bone influence
  quadruped.ears.left_perk:
    type: "spring_bone_target"
    chain: "ear_left"
    target_rotation: [30, 0, 0]
```

### Animation Format (.noodleanim)

```yaml
# sit_and_wag.noodleanim
format: "noodling-animation"
version: "1.0"

metadata:
  name: "Sit and Wag"
  duration: 2.0
  fps: 30
  rig_type: "quadruped"    # Compatible rig types
  author: "Caitlyn"

  # Tags for animation browser
  tags: ["idle", "happy", "sitting"]

  # What this animation requires
  requires:
    - quadruped.tail       # Won't work without a tail
    - face                 # Needs facial expressions

  # Optional elements (graceful degradation)
  optional:
    - quadruped.ears       # Use if available

# Keyframed pose data
tracks:
  # Body pose
  torso.lean_forward:
    keyframes:
      - time: 0.0
        value: 0.0
      - time: 2.0
        value: 0.0

  # Tail wagging (cyclic)
  quadruped.tail.wag_phase:
    keyframes:
      - time: 0.0
        value: 0.0
      - time: 0.5
        value: 1.0
      - time: 1.0
        value: 0.0
      - time: 1.5
        value: 1.0
      - time: 2.0
        value: 0.0
    interpolation: "linear"  # for smooth cycling

  quadruped.tail.wag_amplitude:
    keyframes:
      - time: 0.0
        value: 0.3
      - time: 2.0
        value: 0.3

  # Happy face
  face.mouth_smile_left:
    keyframes:
      - time: 0.0
        value: 0.0
      - time: 0.3
        value: 0.6
      - time: 2.0
        value: 0.6

  face.mouth_smile_right:
    keyframes:
      - time: 0.0
        value: 0.0
      - time: 0.3
        value: 0.6
      - time: 2.0
        value: 0.6

# Affect track (embedded)
affect:
  valence:
    keyframes:
      - time: 0.0
        value: 0.5
      - time: 2.0
        value: 0.6
  arousal:
    keyframes:
      - time: 0.0
        value: 0.4
      - time: 2.0
        value: 0.4
```

### Retargeting at Runtime

```python
class AnimationRetargeter:
    def __init__(self, animation, target_rig):
        self.animation = animation
        self.target_rig = target_rig
        self.mappings = self._build_mappings()

    def _build_mappings(self):
        mappings = {}
        for track_name in self.animation.tracks:
            if track_name in self.target_rig.supported_regions:
                # Direct mapping available
                mappings[track_name] = self.target_rig.mappings[track_name]
            elif self._has_fallback(track_name):
                # Use fallback (e.g., ignore optional tracks)
                mappings[track_name] = None
            else:
                # Required but missing - animation won't work
                raise IncompatibleRigError(f"Rig missing required: {track_name}")
        return mappings

    def sample(self, time):
        pose = {}
        for track_name, mapping in self.mappings.items():
            if mapping is None:
                continue
            value = self.animation.sample(track_name, time)
            pose[track_name] = mapping.apply(value)
        return pose
```

### Humanoid vs Quadruped vs Custom

Pre-defined rig types:

```yaml
# rig_types/humanoid.yaml
rig_type: "humanoid"
regions:
  - torso
  - head
  - left_arm
  - right_arm
  - left_hand
  - right_hand
  - left_leg
  - right_leg
  - left_foot
  - right_foot
  - face

# rig_types/quadruped.yaml
rig_type: "quadruped"
regions:
  - torso
  - head
  - quadruped.front_left_leg
  - quadruped.front_right_leg
  - quadruped.back_left_leg
  - quadruped.back_right_leg
  - quadruped.tail
  - face

# rig_types/serpentine.yaml
rig_type: "serpentine"
regions:
  - head
  - serpentine.body_segments  # array of segments
  - face

# Custom rigs extend these
rig_type: "humanoid_winged"
extends: "humanoid"
additional_regions:
  - wings.left
  - wings.right
```

### VRChat Compatibility Layer

For importing VRChat animations:

```python
class VRChatAnimationImporter:
    """Import Unity/Mecanim humanoid animations."""

    def import_fbx(self, path):
        # Read FBX animation
        fbx_anim = load_fbx_animation(path)

        # Convert Mecanim muscles to semantic poses
        noodle_anim = NoodleAnimation()

        for frame in fbx_anim.frames:
            # Map Mecanim muscle values to our pose descriptors
            pose = self.mecanim_to_semantic(frame)
            noodle_anim.add_frame(frame.time, pose)

        return noodle_anim

    def mecanim_to_semantic(self, frame):
        return {
            "left_arm.reach_forward": frame.LeftArm.Stretch,
            "left_arm.reach_side": frame.LeftArm.Spread,
            "left_arm.bend": frame.LeftForeArm.Stretch,
            # ... mapping for all Mecanim muscles
            "face.jaw_open": frame.Jaw.Open,
            "face.mouth_smile_left": frame.Mouth.Smile.Left,
            # ... etc
        }
```

---

## The Full Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ANIMATION AUTHORING                               │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   Pose      │    │   Affect    │    │   Events    │                  │
│  │   Curves    │    │   Curves    │    │   Markers   │                  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                  │
│         │                  │                  │                          │
│         └────────────┬─────┴─────────────────┘                          │
│                      │                                                   │
│                      ▼                                                   │
│              .noodleanim file                                           │
│         (universal, rig-agnostic)                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RUNTIME                                         │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Animation Retargeter                          │   │
│  │                                                                  │   │
│  │   .noodleanim  +  rig_definition.yaml  →  Applied Pose          │   │
│  │   (universal)     (character-specific)    (bones/shapes/splats)  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Affect System                                 │   │
│  │                                                                  │   │
│  │   Affect Track  →  [MOMENTUM HANDOFF]  →  CharmNetwork          │   │
│  │   (authored)       (when track ends)      (live, decaying)       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Expression Mapping                            │   │
│  │                                                                  │   │
│  │   Blended Affect  →  Blend Shapes  →  Gaussian Deformations     │   │
│  │   (track + live)     Voice Params     Body Language              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**"Animation is about timing. Affect animation is about *feeling* the timing. Emotional momentum is about feeling the *aftermath*."**
