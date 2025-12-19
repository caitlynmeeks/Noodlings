# Noodlings Scene Protocol (NSP)

**Version:** 0.1.0 (Draft)
**Date:** December 18, 2025

A protocol for providing semantic scene truth to stateless generative rendering engines (Google Genie, Mirage, etc.).

---

## Core Premise

**Genie is stateless. Noodlings is stateful.**

Generative 3D engines like Genie and Mirage render frames without memory. They need:
- Spatial context (where is everything?)
- Visual references (what does each character look like?)
- Narrative context (what's happening?)
- Camera direction (how to frame it?)

Noodlings provides all of this as a **Scene Packet** - a complete snapshot of semantic truth that can be rendered to any output modality:
- Text (MUD descriptions)
- 2D illustrated maps (NoodleStudio editor)
- 3D generative video (Genie/Mirage)
- Traditional 3D render (USD pipeline)

**Text, 2D maps, 3D renders are all projections of the same semantic truth.**

---

## Design Principles

1. **Rich over minimal** - Provide maximum context; let the renderer use what it needs
2. **Semantic over aesthetic** - We describe *what*, renderer decides *how*
3. **Stateful persistence** - Characters maintain identity, memory, relationships across frames
4. **Reference-driven consistency** - Visual references ensure character coherence
5. **Decomposable to text** - Everything can be flattened to LLM-readable descriptions

---

## Scene Packet Structure

A Scene Packet is emitted on every significant state change (utterance, action, movement, etc.) or at a regular tick rate for continuous rendering.

```
SCENE_PACKET
├── header
├── spatial_truth
├── entities
├── reference_bundle
├── narrative_context
└── camera_directive
```

---

## Header

```yaml
header:
  protocol_version: "0.1.0"
  packet_id: "pkt_1734567890123"
  timestamp: 1734567890.123
  timestamp_iso: "2025-12-18T21:15:30.123Z"

  stage:
    id: "lemondrops_forest"
    name: "Lemondrops Forest"

  # Packet type for renderer optimization
  packet_type: "full"          # full | delta | camera_only

  # Optional: reference previous packet for delta mode
  previous_packet_id: null
```

**Packet Types:**
- `full` - Complete scene state (use on scene enter, major changes)
- `delta` - Only what changed since previous packet (optimization)
- `camera_only` - Only camera directive changed (for smooth camera moves)

---

## Spatial Truth

The canonical representation of space. Zones are soft attention regions, not hard rooms.

```yaml
spatial_truth:
  # World coordinate system
  coordinate_system:
    units: "meters"
    up_axis: "Y"
    handedness: "right"

  # World bounds
  bounds:
    min: [-100, 0, -100]
    max: [100, 50, 100]

  # Ambient world state
  ambient:
    time_of_day: "dusk"           # dawn | morning | midday | afternoon | dusk | night
    time_precise: "18:42"         # optional HH:MM
    weather: "light_fog"          # clear | cloudy | light_fog | heavy_fog | rain | storm | snow
    season: "autumn"
    temperature: "cool"           # cold | cool | mild | warm | hot
    lighting_mood: "golden_hour"
    soundscape: ["crickets", "distant_owl", "leaves_rustling"]

  # Zone definitions
  zones:
    - id: "campfire"
      name: "The Campfire"
      center: [0, 0, 0]
      radius: 15.0
      falloff: 10.0
      shape: "sphere"             # sphere | cylinder | box

      # Visual description (for LLMs and renderer hints)
      description: |
        A cozy campfire crackles with warm orange flames, sending sparks
        drifting into the starry sky. Soft moss covers the ground.

      # Key features in this zone
      features:
        - "crackling campfire in center"
        - "moss-covered ground"
        - "ring of smooth sitting stones"
        - "old wooden shelf with radio"

      # Mood hints for renderer
      mood: "cozy"
      lighting: "firelight"

      # Connections to other zones
      exits:
        north: "forest_edge"
        east: "pond"

    - id: "pond"
      name: "The Quiet Pond"
      center: [30, -2, 0]
      radius: 20.0
      # ... etc

  # Zone graph (explicit connections if not using exits)
  zone_connections:
    - from: "campfire"
      to: "forest_edge"
      direction: "north"
      traversal: "walk"           # walk | climb | jump | swim | teleport
      description: "A worn path leads into darker trees"

    - from: "campfire"
      to: "pond"
      direction: "east"
      traversal: "walk"
      description: "Fireflies mark the way to water"
```

---

## Entities

All dynamic objects in the scene: noodlings (AI characters), players (humans), and prims (interactive objects).

### Noodlings (AI Characters)

```yaml
entities:
  noodlings:
    red:
      # Identity
      id: "red_fire_anklebiter"
      display_name: "Red"
      species: "dragon (ankle-biter)"

      # Spatial
      position: [2.5, 0, -1.0]
      rotation: [0, 45, 0]        # euler degrees
      facing: [0.7, 0, 0.7]       # unit vector (derived)
      zone: "campfire"

      # Physical
      height: 0.3                 # meters (ankle-height!)
      scale: 1.0

      # CURRENT VISUAL STATE
      visual_state: "default"
      posture: "sitting"          # standing | sitting | lying | crouching | flying
      current_action: "speaking"  # idle | speaking | listening | moving | emoting
      expression: "mischievous"
      gaze_target: "player"       # entity_id or position

      # PERCEPTION (what this entity sees)
      perception:
        fov_horizontal: 120       # degrees
        fov_vertical: 60
        range: 15.0               # meters

        # Computed: what's in view
        sees: ["player", "campfire", "radio"]
        attention_focus: "player"
        attention_strength: 0.9   # how focused (0-1)

      # AFFECT STATE (from CharmNetwork)
      affect:
        valence: 0.4              # -1 to 1 (pleasure)
        arousal: 0.6              # 0 to 1 (energy)
        dominance: 0.7            # 0 to 1 (control)
        boredom: 0.1              # 0 to 1
        sorrow: 0.0               # 0 to 1

      # Derived mood hint for renderer
      mood_hint: "playfully_sassy"

      # VISUAL FORMS (multi-state characters)
      visual_forms:
        default:
          description: |
            A tiny dragon about ankle-height, with rust-red scales,
            small bat-like wings, and expressive golden eyes.
            Perpetually looks slightly annoyed but secretly caring.
          reference_images:
            neutral: "noodlings://red/portrait.png"
            happy: "noodlings://red/expressions/happy.png"
            annoyed: "noodlings://red/expressions/annoyed.png"
            angry: "noodlings://red/expressions/angry.png"
          style_hints:
            scale_texture: "metallic_rust"
            eye_glow: "subtle_gold"
            smoke_wisps: "occasional"

      # Voice (for audio rendering)
      voice:
        reference: "noodlings://red/voice_sample.wav"
        description: "Raspy, small but fierce, slight growl undertone"
        pitch: "high"
        pace: "quick"

    yuki:
      id: "yuki_cyberfox"
      display_name: "Yuki"
      species: "cyberfox (shapeshifter)"

      position: [10.2, 0, 5.0]
      rotation: [0, -30, 0]
      zone: "forest_edge"

      # MULTI-STATE CHARACTER
      visual_state: "humanoid_fox"    # current active form

      visual_forms:
        ghostly_fox:
          description: |
            Translucent spectral fox form. Pale blue ethereal glow,
            wisps of energy trailing from fur. Can pass through objects.
            Eyes are bright cyan points of light.
          reference_images:
            front: "noodlings://yuki/ghostly_front.png"
            side: "noodlings://yuki/ghostly_side.png"
            moving: "noodlings://yuki/ghostly_run.png"
          style_hints:
            opacity: 0.7
            glow_color: "#88CCFF"
            particles: "ethereal_motes"
            shadow: false

        normal_fox:
          description: |
            Beautiful silver-white fox with cyan circuit-like markings
            glowing softly along fur. Fluffy tail, alert triangular ears.
            About the size of a large cat.
          reference_images:
            neutral: "noodlings://yuki/fox_front.png"
            alert: "noodlings://yuki/fox_alert.png"
            running: "noodlings://yuki/fox_running.png"
            sleeping: "noodlings://yuki/fox_curled.png"
          style_hints:
            fur_quality: "fluffy_iridescent"
            marking_glow: "#00FFFF"
            eye_color: "#00FFFF"

        humanoid_fox:
          description: |
            Anthropomorphic fox girl. Silver-white hair with fox ears,
            fluffy tail, cyan eyes. Wears cyber-aesthetic clothing with
            soft glow accents. Graceful and alert posture.
          reference_images:
            neutral: "noodlings://yuki/humanoid_neutral.png"
            happy: "noodlings://yuki/humanoid_happy.png"
            concerned: "noodlings://yuki/humanoid_concerned.png"
            action: "noodlings://yuki/humanoid_action.png"
          style_hints:
            clothing: "cyber_jacket_shorts"
            accessories: ["ear_piercings", "holo_bracelet", "tail_ring"]
            marking_glow: "#00FFFF"

      # Expression overlay (works across forms where applicable)
      expression: "curious"
      posture: "standing_alert"
      current_action: "ears_perked_listening"

      affect:
        valence: 0.2
        arousal: 0.7
        dominance: 0.5
        boredom: 0.0
        sorrow: 0.1

      perception:
        fov_horizontal: 180       # foxes have wide vision
        range: 25.0
        sees: ["strange_noise_source", "trees", "moonlight"]
        attention_focus: "strange_noise"
        attention_strength: 0.95
```

### Players (Humans)

```yaml
  players:
    caity:
      id: "player_caity"
      display_name: "Caity"

      position: [0, 0, 2.0]
      rotation: [0, 180, 0]
      facing: [0, 0, -1]
      zone: "campfire"

      height: 1.65

      # Player avatar (if defined)
      avatar:
        description: "A human with curious eyes and paint-stained fingers"
        reference_image: "noodlings://players/caity_avatar.png"
        # Or null for invisible/implied player

      posture: "sitting"
      current_action: "listening"
      gaze_target: "red"

      perception:
        fov_horizontal: 90
        range: 20.0
        sees: ["red", "campfire", "radio", "yuki_distant"]
        attention_focus: "red"
```

### Prims (Interactive Objects)

```yaml
  prims:
    campfire:
      id: "campfire_main"
      type: "fire"
      position: [0, 0, 0]
      zone: "campfire"

      description: "A crackling campfire with dancing orange flames"

      state:
        intensity: 0.8            # 0-1
        color: "orange_yellow"

      # Dynamic visual properties
      visual_dynamics:
        flicker: true
        particles: "sparks_upward"
        light_radius: 8.0
        light_color: "#FF6622"

      affordances: []             # not directly interactable

    campfire_radio:
      id: "radio_01"
      type: "radio"
      position: [1.5, 0.3, -2.0]
      rotation: [0, 30, 0]
      zone: "campfire"
      parent: "wooden_shelf"      # sitting on something

      description: "An old radio with brass dials and a cracked speaker"
      reference_image: "noodlings://prims/radio.png"

      state:
        power: "on"
        station: "forest_news"
        volume: 0.5
        currently_playing: "soft jazz with occasional static"

      visual_dynamics:
        dial_glow: true
        dial_color: "#FFAA00"

      # What you can do with it (for LLM and interaction)
      affordances:
        - verb: "listen"
          description: "Listen to the broadcast"
        - verb: "turn"
          description: "Toggle power or change station"
          aliases: ["switch", "toggle"]
        - verb: "adjust"
          description: "Change volume"

    wanted_poster:
      id: "poster_wanted_01"
      type: "poster"
      position: [3.0, 1.5, -4.0]
      rotation: [0, 15, 0]
      zone: "campfire"
      parent: "oak_tree_01"

      description: "A weathered WANTED poster nailed to a tree"
      reference_image: "noodlings://prims/wanted_poster.png"

      state:
        readable_text: |
          WANTED: The Notorious Noodle Thief
          Reward: 50 Gold Acorns
          Last seen: Stealing dreams from sleeping foxes

      affordances:
        - verb: "read"
          description: "Read the poster text"
        - verb: "take"
          description: "Remove from tree"
          requires: "permission"
```

---

## Reference Bundle

Pre-packaged visual references for the current frame. Renderers can fetch these or use inline base64.

```yaml
reference_bundle:
  # Character references (current form + expression)
  characters:
    red:
      form: "default"
      primary_ref: "noodlings://red/portrait.png"
      expression_ref: "noodlings://red/expressions/mischievous.png"
      # Or inline:
      # primary_ref_base64: "data:image/png;base64,..."

      # Text description (for text-to-3D fallback)
      description: |
        Tiny rust-red dragon, ankle-height, bat-like wings folded,
        golden eyes with mischievous glint, sitting on haunches,
        small smoke wisp from nostrils.

    yuki:
      form: "humanoid_fox"
      primary_ref: "noodlings://yuki/humanoid_neutral.png"
      expression_ref: "noodlings://yuki/expressions/curious.png"

      description: |
        Anthropomorphic silver-white fox girl, alert stance, ears perked
        forward, cyan eyes wide, fluffy tail raised, wearing dark cyber
        jacket with cyan glow trim.

  # Prim references
  prims:
    campfire_radio:
      ref: "noodlings://prims/radio.png"
      description: "Vintage wooden radio, brass dials, warm dial glow"

  # Environment references (optional)
  environment:
    skybox_hint: "night_stars_light_fog"
    ground_hint: "mossy_forest_floor"
    foliage_ref: "noodlings://env/forest_trees.png"
```

---

## Narrative Context

What's happening in the story. Provides context for mood, tension, and appropriate rendering style.

```yaml
narrative_context:
  # Recent dialogue (most recent first)
  recent_dialogue:
    - speaker: "red"
      text: "You're not fooling anyone with that innocent look."
      timestamp: 1734567888.0
      seconds_ago: 2.3
      tone: "teasing"

    - speaker: "caity"
      text: "Who, me?"
      timestamp: 1734567885.0
      seconds_ago: 5.1
      tone: "playful_innocent"

    - speaker: "red"
      text: "*narrows eyes*"
      timestamp: 1734567883.0
      seconds_ago: 7.3
      type: "emote"

  # Recent actions/events
  recent_events:
    - actor: "red"
      action: "raised_eyebrow"
      timestamp: 1734567888.5
      seconds_ago: 1.8

    - actor: "yuki"
      action: "ears_swiveled_toward_noise"
      timestamp: 1734567880.0
      seconds_ago: 10.3

    - event: "radio_changed_song"
      timestamp: 1734567870.0
      seconds_ago: 20.3

  # Scene-level narrative state
  scene_state:
    tension: 0.2                  # 0-1 (low = peaceful, high = conflict)
    energy: 0.5                   # 0-1 (low = calm, high = active)
    intimacy: 0.7                 # 0-1 (low = formal, high = close)
    humor: 0.6                    # 0-1
    mystery: 0.3                  # 0-1

    current_beat: "playful_banter"
    # Examples: "tense_standoff", "emotional_revelation", "action_sequence",
    #           "quiet_moment", "comedic_relief", "mysterious_discovery"

  # Longer context summary (for LLM-based renderers)
  context_summary: |
    Caity and Red are having a playful exchange by the campfire. Red is
    pretending to be suspicious of Caity's intentions, but the mood is
    warm and teasing. In the background, Yuki has noticed something in
    the forest and is alert but not alarmed. The radio plays soft jazz.
```

---

## Camera Directive

High-level cinematography instructions. The renderer interprets these into actual camera parameters.

```yaml
camera_directive:
  # Shot type
  mode: "FOCUS_ON"

  # Available modes:
  # - POV(subject)           - First person through subject's eyes
  # - SHOW_WHAT_SEES(subject) - Subject's POV looking at their attention target
  # - FOCUS_ON(subject)      - Camera on subject (framing varies)
  # - TWO_SHOT(a, b)         - Frame two subjects in conversation
  # - GROUP_SHOT(subjects)   - Frame multiple subjects
  # - ESTABLISH(zone)        - Wide shot establishing location
  # - FOLLOW(subject)        - Third person following
  # - FREE(position, look_at) - Specific camera placement
  # - CINEMATIC(style)       - Let renderer choose dramatic framing

  # Primary subject
  subject: "red"

  # Framing
  framing: "medium"
  # Options: extreme_closeup | closeup | medium_closeup | medium |
  #          medium_wide | wide | extreme_wide

  # Angle
  angle: "eye_level"
  # Options: worms_eye | low | eye_level | high | birds_eye | dutch

  # Other subjects to keep in frame
  include_in_frame:
    - entity: "caity"
      importance: 0.7           # how important to keep visible
    - entity: "campfire"
      importance: 0.3

  # Camera movement
  movement: "gentle_drift"
  # Options: static | gentle_drift | slow_push | slow_pull |
  #          tracking | handheld | crane | orbit

  # Style hints
  style:
    # Lens
    focal_length: 50            # mm equivalent (24=wide, 50=normal, 85=portrait, 135=telephoto)

    # Depth of field
    dof_mode: "shallow"         # none | shallow | medium | deep
    dof_focus: "subject"        # subject | position | auto

    # Color/mood
    color_temperature: "warm"   # cool | neutral | warm
    color_grade: "firelight"    # natural | cinematic | dreamy | noir | etc.

    # Post
    film_grain: 0.1             # 0-1
    vignette: 0.2               # 0-1
    bloom: 0.3                  # 0-1

  # Transition (if changing from previous shot)
  transition:
    type: "cut"                 # cut | dissolve | fade | wipe
    duration: 0.0               # seconds (0 for cut)
```

### Camera Directive Examples

```yaml
# Intimate conversation closeup
camera_directive:
  mode: "FOCUS_ON"
  subject: "red"
  framing: "closeup"
  angle: "eye_level"
  include_in_frame:
    - entity: "caity"
      importance: 0.3
  style:
    focal_length: 85
    dof_mode: "shallow"
    color_grade: "firelight"

# Two-shot for dialogue
camera_directive:
  mode: "TWO_SHOT"
  subjects: ["red", "caity"]
  framing: "medium"
  angle: "eye_level"
  style:
    focal_length: 35
    dof_mode: "medium"

# Establishing shot
camera_directive:
  mode: "ESTABLISH"
  zone: "campfire"
  framing: "wide"
  angle: "high"
  movement: "slow_crane_down"
  style:
    focal_length: 24
    dof_mode: "deep"

# POV shot - see through Yuki's eyes
camera_directive:
  mode: "POV"
  subject: "yuki"
  style:
    focal_length: 35
    # Yuki's perception affects rendering
    # (her fox vision might see different spectrum, etc.)

# Show what Yuki is looking at
camera_directive:
  mode: "SHOW_WHAT_SEES"
  subject: "yuki"
  framing: "medium"
  # Camera positioned at Yuki's POV, focused on her attention_focus

# Dramatic cinematic (let renderer decide)
camera_directive:
  mode: "CINEMATIC"
  style: "mysterious"
  subjects: ["yuki", "strange_noise_source"]
  hints:
    - "tension building"
    - "something in the shadows"
```

---

## Complete Scene Packet Example

```yaml
# NOODLINGS SCENE PACKET
# Protocol Version 0.1.0

header:
  protocol_version: "0.1.0"
  packet_id: "pkt_20251218_211530_001"
  timestamp: 1734567890.123
  timestamp_iso: "2025-12-18T21:15:30.123Z"
  stage:
    id: "lemondrops_forest"
    name: "Lemondrops Forest"
  packet_type: "full"

spatial_truth:
  coordinate_system:
    units: "meters"
    up_axis: "Y"
  bounds:
    min: [-100, 0, -100]
    max: [100, 50, 100]
  ambient:
    time_of_day: "night"
    time_precise: "21:15"
    weather: "clear"
    lighting_mood: "moonlit_firelight"
    soundscape: ["fire_crackle", "crickets", "owl_distant"]
  zones:
    - id: "campfire"
      name: "The Campfire"
      center: [0, 0, 0]
      radius: 15.0
      description: "A cozy campfire clearing"
      mood: "cozy"
      exits: {north: "forest_edge", east: "pond"}

entities:
  noodlings:
    red:
      id: "red_fire_anklebiter"
      display_name: "Red"
      position: [2.5, 0, -1.0]
      facing: [0.7, 0, 0.7]
      zone: "campfire"
      visual_state: "default"
      posture: "sitting"
      expression: "mischievous"
      gaze_target: "caity"
      affect: {valence: 0.4, arousal: 0.6, dominance: 0.7, boredom: 0.1, sorrow: 0.0}
      perception:
        sees: ["caity", "campfire", "radio"]
        attention_focus: "caity"

  players:
    caity:
      id: "player_caity"
      position: [0, 0, 2.0]
      facing: [0, 0, -1]
      zone: "campfire"
      posture: "sitting"
      gaze_target: "red"

  prims:
    campfire:
      position: [0, 0, 0]
      state: {intensity: 0.8}
    radio:
      position: [1.5, 0.3, -2.0]
      state: {power: "on", playing: "soft jazz"}

reference_bundle:
  characters:
    red:
      form: "default"
      primary_ref: "noodlings://red/portrait.png"
      expression_ref: "noodlings://red/expressions/mischievous.png"
      description: "Tiny rust-red dragon, golden eyes, mischievous glint"

narrative_context:
  recent_dialogue:
    - speaker: "red"
      text: "You're not fooling anyone."
      seconds_ago: 2.3
  scene_state:
    tension: 0.2
    energy: 0.5
    intimacy: 0.7
    current_beat: "playful_banter"

camera_directive:
  mode: "FOCUS_ON"
  subject: "red"
  framing: "medium_closeup"
  angle: "eye_level"
  include_in_frame:
    - entity: "caity"
      importance: 0.5
  movement: "gentle_drift"
  style:
    focal_length: 65
    dof_mode: "shallow"
    color_grade: "firelight"
```

---

## Transport & Encoding

### JSON Encoding

Primary format. Scene packets are UTF-8 JSON.

```json
{
  "header": {
    "protocol_version": "0.1.0",
    "packet_id": "pkt_20251218_211530_001",
    ...
  },
  "spatial_truth": { ... },
  "entities": { ... },
  "reference_bundle": { ... },
  "narrative_context": { ... },
  "camera_directive": { ... }
}
```

### Binary Encoding (Future)

For high-frequency streaming, a binary format (MessagePack, FlatBuffers, or custom) may be defined.

### WebSocket Streaming

```
ws://noodlings-server/scene-stream

→ { "subscribe": { "stage": "lemondrops_forest", "mode": "full" } }
← { "packet_type": "full", ... }
← { "packet_type": "delta", "changes": [...] }
← { "packet_type": "camera_only", "camera_directive": {...} }
```

### REST Endpoint

```
GET /api/scene/{stage_id}/current
Returns: Current scene packet (full)

GET /api/scene/{stage_id}/stream
Returns: Server-sent events stream of packets
```

---

## Reference Asset Protocol

### URI Scheme

```
noodlings://red/portrait.png
noodlings://yuki/humanoid_neutral.png
noodlings://prims/radio.png
noodlings://env/skybox_night.hdr
```

### Resolution

1. Check local project: `{project}/Noodlings/red/Assets/portrait.png`
2. Check cloud cache: `~/.noodlings/cache/red/portrait.png`
3. Fetch from cloud: `https://assets.noodlings.ai/red/portrait.png`

### Inline Base64

For self-contained packets:

```yaml
reference_bundle:
  characters:
    red:
      primary_ref_inline: "data:image/png;base64,iVBORw0KGgo..."
```

---

## Integration with Genie/Mirage

### Adapter Layer

Each generative engine needs an adapter that:
1. Receives NSP Scene Packets
2. Transforms to engine-specific format
3. Submits reference images for consistency
4. Returns rendered frames/video

```python
class GenieAdapter:
    def submit_packet(self, packet: ScenePacket) -> GenieJob:
        # Transform NSP to Genie's input format
        genie_input = {
            "scene_description": self.flatten_to_text(packet),
            "reference_images": self.extract_refs(packet),
            "camera": self.map_camera(packet.camera_directive),
            # ...
        }
        return genie.submit(genie_input)

    def flatten_to_text(self, packet) -> str:
        # Generate text description from semantic data
        # This is what Genie's underlying LLM sees
        pass
```

### Text Flattening

For LLM-based renderers, the scene packet flattens to:

```
SCENE: The Campfire at Lemondrops Forest
TIME: Night, 9:15 PM, clear sky with moonlight and firelight

LOCATION: A cozy campfire clearing. Crackling fire in center, moss-covered
ground, ring of sitting stones. An old radio plays soft jazz on a wooden shelf.

CHARACTERS PRESENT:
- Red (tiny rust-red dragon, ankle-height): Sitting near fire, looking at Caity
  with mischievous expression. Golden eyes glinting. Mood: playful/sassy.
- Caity (human): Sitting across fire, looking at Red with innocent expression.

JUST HAPPENED:
- Red said "You're not fooling anyone." (teasing tone)

MOOD: Cozy, playful banter, low tension, intimate

CAMERA: Medium closeup on Red, Caity visible in background. Warm firelight
color grade, shallow depth of field, gentle camera drift.
```

---

## Perception Slices

A **Perception Slice** is a filtered Scene Packet representing what a specific entity perceives. Used as cognitive input for Noodling facet assemblies.

### Why Perception Slices?

Noodlings should only reason about what they can perceive:
- Can't see entities behind them
- Can't hear whispered conversations across the room
- Don't know other entities' internal states (unless expressed)
- Only remember events they witnessed

This creates realistic information asymmetry and emergent social dynamics.

### Slice Generation

```python
def generate_perception_slice(
    full_packet: ScenePacket,
    perceiver_id: str
) -> PerceptionSlice:
    """
    Filter full scene to what perceiver can see/hear/know.
    """
    perceiver = full_packet.entities.get(perceiver_id)
    cone = perceiver.perception

    slice = PerceptionSlice(
        perceiver=perceiver_id,
        timestamp=full_packet.header.timestamp,

        # SELF - full access to own state
        self_state=perceiver.copy(),

        # PERCEIVED ENTITIES - filtered by cone
        perceived_entities={},

        # PERCEIVED EVENTS - only witnessed
        perceived_events=[],

        # SPATIAL CONTEXT - zones in range
        spatial_context={},
    )

    # Filter entities by perception cone
    for entity_id, entity in full_packet.entities.all():
        if entity_id == perceiver_id:
            continue

        visibility = compute_visibility(
            perceiver_pos=perceiver.position,
            perceiver_facing=perceiver.facing,
            perceiver_fov=cone.fov_horizontal,
            perceiver_range=cone.range,
            target_pos=entity.position,
            occlusion=full_packet.spatial_truth.geometry
        )

        if visibility > 0.1:  # perception threshold
            # Include entity but NOT their internal state
            slice.perceived_entities[entity_id] = {
                "id": entity.id,
                "display_name": entity.display_name,
                "position": entity.position,
                "distance": dist(perceiver.position, entity.position),
                "direction": relative_direction(perceiver, entity),
                "visibility": visibility,

                # Observable externals only
                "posture": entity.posture,
                "current_action": entity.current_action,
                "expression": entity.expression,  # what face they're making
                "gaze_target": entity.gaze_target,

                # NOT included: affect, memories, attention_focus details
                # (unless perceiver has special abilities)
            }

    # Filter events by witness
    for event in full_packet.narrative_context.recent_events:
        if was_witnessed(perceiver, event, full_packet):
            slice.perceived_events.append(event)

    # Filter dialogue by audibility
    for dialogue in full_packet.narrative_context.recent_dialogue:
        if could_hear(perceiver, dialogue, full_packet):
            slice.perceived_events.append({
                "type": "heard_speech",
                "speaker": dialogue.speaker,
                "text": dialogue.text,
                "tone": dialogue.tone,
                "seconds_ago": dialogue.seconds_ago,
            })

    return slice
```

### Perception Slice Schema

```yaml
perception_slice:
  # Who is perceiving
  perceiver: "red"
  timestamp: 1734567890.123

  # SELF - full internal access
  self_state:
    position: [2.5, 0, -1.0]
    facing: [0.7, 0, 0.7]
    zone: "campfire"
    affect:
      valence: 0.4
      arousal: 0.6
      dominance: 0.7
      boredom: 0.1
      sorrow: 0.0
    # Full internal state available to own cognition

  # PERCEIVED ENTITIES - observable externals only
  perceived_entities:
    caity:
      display_name: "Caity"
      position: [0, 0, 2.0]
      distance: 3.5
      direction: "in_front"       # relative to perceiver
      visibility: 0.95
      posture: "sitting"
      expression: "innocent"
      gaze_target: "self"         # they're looking at me
      current_action: "listening"
      # NO access to: caity's thoughts, memories, true intentions

    campfire:
      type: "prim"
      position: [0, 0, 0]
      distance: 2.8
      direction: "front_left"
      state: {intensity: 0.8}     # observable state

  # NOT PERCEIVED - outside cone or occluded
  # (Yuki is behind Red, so not in this slice)

  # PERCEIVED EVENTS - only what was witnessed
  perceived_events:
    - type: "heard_speech"
      speaker: "caity"
      text: "Who, me?"
      tone: "playful_innocent"
      seconds_ago: 5.1

    - type: "observed_action"
      actor: "caity"
      action: "shifted_position"
      seconds_ago: 8.0

  # SPATIAL AWARENESS - zones perceiver knows about
  spatial_context:
    current_zone: "campfire"
    known_exits:
      north: "forest_edge"
      east: "pond"
    ambient:
      lighting: "firelight"
      sounds: ["fire_crackle", "soft_jazz_from_radio"]
```

### Perception Modifiers

Different entities perceive differently:

```yaml
# Fox has wide peripheral vision
yuki:
  perception:
    fov_horizontal: 180
    fov_vertical: 90
    range: 25
    night_vision: true      # sees in low light
    motion_sensitivity: 0.9  # notices movement easily

# Dragon has heat sensing
red:
  perception:
    fov_horizontal: 120
    range: 15
    heat_sense: true        # perceives warm bodies even if occluded
    smoke_sense: true       # detects fire/smoke at distance

# Ghost form can perceive through walls
yuki_ghostly:
  perception:
    fov_horizontal: 360     # all-around awareness
    occlusion_ignore: true  # sees through objects
    range: 30
```

### Pipeline Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    SCENE STATE MANAGER                      │
│  (canonical truth - positions, states, events)              │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ PERCEPTION      │  │ PERCEPTION      │  │ SCENE PACKET    │
│ SLICE GENERATOR │  │ SLICE GENERATOR │  │ EMITTER         │
│ (for Red)       │  │ (for Yuki)      │  │ (for Genie)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Red's Context   │  │ Yuki's Context  │  │ Full Scene      │
│ (facet input)   │  │ (facet input)   │  │ Packet (JSON)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ FACET EXECUTOR  │  │ FACET EXECUTOR  │  │ GENIE ADAPTER   │
│ (Red thinks)    │  │ (Yuki thinks)   │  │ (render frame)  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Context Assembly for Facets

The perception slice feeds into the facet assembly's context:

```python
def build_facet_context(noodling_id: str, scene_state: SceneState) -> dict:
    """
    Build the context dict passed to facet execution.
    """
    # Generate what this noodling perceives
    slice = generate_perception_slice(scene_state.to_packet(), noodling_id)

    return {
        # Self
        "self": slice.self_state,
        "affect": slice.self_state.affect,
        "position": slice.self_state.position,
        "zone": slice.self_state.zone,

        # World (filtered by perception)
        "perceived_entities": slice.perceived_entities,
        "perceived_events": slice.perceived_events,
        "spatial_context": slice.spatial_context,

        # Who I'm talking to (if in conversation)
        "conversation_partner": determine_conversation_partner(slice),

        # Recent input directed at me
        "last_input": get_last_input_to(noodling_id),

        # Memories (internal, not perception-filtered)
        "memories": get_memories(noodling_id),
    }
```

---

## Future Extensions

### Animation Hints

```yaml
entities:
  noodlings:
    red:
      animation_hint:
        gesture: "dismissive_wave"
        intensity: 0.7
        timing: "on_dialogue"
```

### Physics State

```yaml
entities:
  prims:
    falling_acorn:
      physics:
        velocity: [0, -2.5, 0]
        angular_velocity: [0.1, 0.5, 0.2]
        in_flight: true
```

### Audio Cues

```yaml
audio_context:
  ambient:
    - source: "campfire"
      sound: "fire_crackle"
      volume: 0.7
  dialogue:
    - speaker: "red"
      audio_ref: "noodlings://tts/red_line_001.wav"
      # Or TTS directive:
      tts:
        text: "You're not fooling anyone."
        voice: "red"
        emotion: "teasing"
```

### Multi-Frame Sequences

```yaml
sequence:
  id: "red_eyeroll_sequence"
  frames:
    - duration: 0.5
      red: { expression: "skeptical" }
      camera: { movement: "slow_push" }
    - duration: 0.3
      red: { expression: "eyeroll", action: "looks_away" }
    - duration: 0.4
      red: { expression: "smirk", gaze_target: "caity" }
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-18 | Initial draft |

---

*Ordnung muss sein!*
