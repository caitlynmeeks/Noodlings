# Networking and Audio Architecture

**Goal:** Multi-user social presence with spatial audio, voice chat, and object-attached sound.

---

## 1. WHAT'S HOSTED WHERE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARCHITECTURE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

CLIENT (NoodleStudio / Web Browser)          SERVER (cMUSH)
┌────────────────────────────────┐          ┌────────────────────────────────┐
│                                │          │                                │
│  Gaussian Renderer             │          │  Entity State (authoritative)  │
│  - Scene composition           │          │  - Positions, rotations        │
│  - Mirror/portal rendering     │          │  - Animations, expressions     │
│  - Particle systems            │◄────────►│  - Audio emitter states        │
│                                │ WebSocket│                                │
│  Spatial Audio Mixer           │  (state) │  AI Cognition                  │
│  - 3D positioning              │          │  - Facet execution             │
│  - Distance attenuation        │          │  - NPC behavior                │
│  - HRTF processing             │          │                                │
│                                │          │  Room/Zone Management          │
│  Voice Capture                 │          │  - Who's where                 │
│  - Microphone input            │          │  - Visibility                  │
│  - Noise suppression           │          │  - Perception filtering        │
│                                │          │                                │
│  Input Handling                │          │  Chat/Dialogue                 │
│  - Movement                    │          │  - Text messages               │
│  - Interactions                │          │  - Emotes                      │
│                                │          │                                │
│  Interpolation/Prediction      │          └────────────────────────────────┘
│  - Smooth remote entities      │                       │
│  - Hide network latency        │                       │
│                                │                       │
└────────────────────────────────┘                       │
         │                                               │
         │ WebRTC                                        │
         │ (voice)                                       │
         ▼                                               ▼
┌────────────────────────────────┐          ┌────────────────────────────────┐
│  VOICE SERVER (SFU)            │          │  CLOUDFLARE WORKERS            │
│                                │          │                                │
│  Options:                      │          │  Authentication                │
│  - LiveKit (recommended)       │          │  - OAuth (Google, GitHub)      │
│  - mediasoup                   │          │  - Session tokens              │
│  - Jitsi                       │          │                                │
│  - Daily.co (hosted)           │          │  Asset Storage (R2)            │
│                                │          │  - Gaussian splats (.ply)      │
│  Responsibilities:             │          │  - Audio clips                 │
│  - WebRTC signaling            │          │  - Textures                    │
│  - Stream forwarding           │          │                                │
│  - Room management             │          │  WebRTC Signaling (optional)   │
│  - NOT audio mixing            │          │  - ICE candidates              │
│  (clients do spatial mix)      │          │  - Session descriptions        │
│                                │          │                                │
└────────────────────────────────┘          └────────────────────────────────┘
```

---

## 2. NETWORKING PROTOCOL

### State Synchronization (WebSocket)

**Server → Client: Entity Update**
```json
{
  "type": "entity_update",
  "timestamp": 1703184000.123,
  "entities": [
    {
      "id": "noodling_red",
      "position": [1.5, 0, 3.2],
      "rotation": [0, 0.707, 0, 0.707],
      "velocity": [0.5, 0, 0.1],
      "animation": {
        "state": "walking",
        "blend_shapes": {"happy": 0.3}
      },
      "audio_emitters": [
        {
          "id": "footsteps",
          "playing": true,
          "volume": 0.5
        }
      ]
    }
  ]
}
```

**Client → Server: Player Input**
```json
{
  "type": "player_input",
  "timestamp": 1703184000.456,
  "position": [2.0, 0, 4.0],
  "rotation": [0, 0, 0, 1],
  "actions": ["interact_prim_radio"]
}
```

### Update Rates

| Data Type | Rate | Transport |
|-----------|------|-----------|
| Position/rotation | 20 Hz | WebSocket |
| Animation state | 10 Hz | WebSocket |
| Blend shapes | 30 Hz | WebSocket |
| Audio emitter state | On change | WebSocket |
| Voice audio | 50 packets/sec | WebRTC |
| Chat messages | On send | WebSocket |

### Client-Side Interpolation

```python
class RemoteEntity:
    """Smoothly interpolate remote entity positions."""

    def __init__(self):
        self.buffer = []  # [(timestamp, position, rotation)]
        self.interp_delay = 0.1  # 100ms buffer

    def add_snapshot(self, timestamp, position, rotation):
        self.buffer.append((timestamp, position, rotation))
        # Keep last 1 second of snapshots
        cutoff = timestamp - 1.0
        self.buffer = [(t, p, r) for t, p, r in self.buffer if t > cutoff]

    def get_interpolated(self, render_time):
        """Get position at render_time (current_time - interp_delay)."""
        target_time = render_time - self.interp_delay

        # Find surrounding snapshots
        before = after = None
        for i, (t, p, r) in enumerate(self.buffer):
            if t <= target_time:
                before = (t, p, r)
            if t >= target_time and after is None:
                after = (t, p, r)
                break

        if before is None:
            return after[1], after[2] if after else (None, None)
        if after is None:
            # Extrapolate (risky but necessary)
            return before[1], before[2]

        # Interpolate
        t = (target_time - before[0]) / (after[0] - before[0])
        position = before[1] * (1-t) + after[1] * t
        rotation = slerp(before[2], after[2], t)

        return position, rotation
```

---

## 3. VOICE CHAT ARCHITECTURE

### Why SFU (Selective Forwarding Unit)?

| Approach | Pros | Cons |
|----------|------|------|
| **Peer-to-Peer** | Low latency, no server | O(n²) connections, NAT issues |
| **MCU (mixing)** | Single stream to clients | Server CPU intensive |
| **SFU (forwarding)** | Server just forwards | Clients do mixing, scales well |

**SFU wins** because:
- Clients need individual streams for spatial positioning anyway
- Server doesn't decode audio = low CPU
- Scales to many users per room

### LiveKit Integration (Recommended)

```python
# Server-side: Create room token
import livekit.api as lk

async def create_voice_room_token(user_id: str, room_name: str) -> str:
    """Generate token for user to join voice room."""
    token = lk.AccessToken(
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )
    token.add_grant(lk.VideoGrant(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
    ))
    token.identity = user_id
    return token.to_jwt()
```

```javascript
// Client-side: Connect and spatialize
import { Room, RoomEvent } from 'livekit-client';

const room = new Room();
await room.connect(LIVEKIT_URL, token);

// When someone speaks, spatialize based on their position
room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
    if (track.kind === 'audio') {
        const audioElement = track.attach();

        // Get participant's world position from entity state
        const entityPos = getEntityPosition(participant.identity);

        // Apply spatial audio
        spatialAudio.attachVoice(participant.identity, audioElement, entityPos);
    }
});
```

### Voice + Spatial Audio Flow

```
Speaker's Mic                    Listener's Ears
     │                                 ▲
     ▼                                 │
[Capture & Encode]              [Spatial Positioning]
     │                                 │
     ▼                                 │
[WebRTC Send]──────►[SFU]──────►[WebRTC Receive]
                                       │
                                       ▼
                              [Get Speaker Position]
                                       │
                                       ▼
                              [Apply HRTF + Distance]
                                       │
                                       ▼
                              [Mix with Scene Audio]
```

---

## 4. AUDIO EMITTERS ON OBJECTS

### Component Model

Every entity (Noodling, Prim, Zone) can have audio emitters attached:

```python
@dataclass
class AudioEmitterComponent:
    """Audio source attached to an entity."""
    emitter_id: str
    clip_url: str

    # Playback
    autoplay: bool = False
    loop: bool = False
    playing: bool = False

    # Spatial
    spatial: bool = True
    offset: Tuple[float, float, float] = (0, 0, 0)  # Local offset from entity

    # Attenuation
    volume: float = 1.0
    ref_distance: float = 1.0
    max_distance: float = 50.0
    rolloff: float = 1.0

    # Advanced
    cone_inner_angle: float = 360.0
    cone_outer_angle: float = 360.0
    cone_outer_gain: float = 0.0
```

### Entity Integration

```python
# In scene_packet.py

@dataclass
class Noodling:
    id: str
    name: str
    position: Vector3
    # ... existing fields ...

    # NEW: Audio emitters
    audio_emitters: List[AudioEmitterComponent] = field(default_factory=list)


@dataclass
class Prim:
    id: str
    name: str
    position: Vector3
    # ... existing fields ...

    # NEW: Audio emitters
    audio_emitters: List[AudioEmitterComponent] = field(default_factory=list)
```

### YAML Definition

```yaml
# In a prim definition (e.g., radio.yaml)
prim:
  name: "Old Radio"
  mesh: "props/radio.glb"

  audio_emitters:
    - id: "music"
      clip: "audio/jazz_loop.ogg"
      loop: true
      autoplay: false
      volume: 0.7
      max_distance: 15.0

    - id: "static"
      clip: "audio/radio_static.ogg"
      loop: true
      autoplay: true
      volume: 0.2
      max_distance: 5.0
```

```yaml
# In a noodling recipe
noodling:
  name: "Red"
  species: "fire_imp"

  audio_emitters:
    - id: "voice"
      clip: null  # Filled by TTS
      spatial: true
      ref_distance: 1.0
      max_distance: 20.0

    - id: "footsteps"
      clip: "audio/imp_footstep.ogg"
      loop: false
      volume: 0.4
      max_distance: 10.0
```

---

## 5. SCRIPTING API

### Audio API for ScriptedFacets

```javascript
// context.noodle.audio API

// === Emitter Management ===

// Attach a new audio source to this entity
context.noodle.audio.attachSource("door_creak", {
    clip: "sounds/door_creak.ogg",
    volume: 0.8,
    spatial: true,
    maxDistance: 20,
    loop: false
});

// Remove an emitter
context.noodle.audio.removeSource("door_creak");

// === Playback Control ===

// Play a sound
context.noodle.audio.play("footsteps");

// Play at specific position (one-shot, not attached to entity)
context.noodle.audio.playAt("explosion", [10, 0, 5], {
    volume: 1.0,
    maxDistance: 100
});

// Stop a sound
context.noodle.audio.stop("music");

// Pause/resume
context.noodle.audio.pause("music");
context.noodle.audio.resume("music");

// === Properties ===

// Set volume
context.noodle.audio.setVolume("music", 0.5);

// Set pitch (for doppler-like effects)
context.noodle.audio.setPitch("engine", 1.2);

// Check if playing
if (context.noodle.audio.isPlaying("alarm")) {
    // ...
}

// === Events ===

// Listen for audio events
context.noodle.audio.on("ended", "music", () => {
    console.log("Music finished!");
});

// === Voice (for noodlings) ===

// Speak with TTS (if configured)
await context.noodle.audio.speak("Hello, traveler!", {
    voice: "nova",
    emotion: "friendly"
});

// Check if currently speaking
if (context.noodle.audio.isSpeaking()) {
    // Wait for speech to finish
}
```

### Example: Interactive Radio

```javascript
// radio_script.js - Attached to a radio prim

const STATIONS = [
    { name: "Jazz FM", clip: "audio/jazz_loop.ogg" },
    { name: "Rock Radio", clip: "audio/rock_loop.ogg" },
    { name: "Classical", clip: "audio/classical_loop.ogg" },
];

let currentStation = 0;
let isOn = false;

function onInteract(context, verb) {
    if (verb === "turn_on" || verb === "use") {
        if (!isOn) {
            turnOn(context);
        } else {
            nextStation(context);
        }
    } else if (verb === "turn_off") {
        turnOff(context);
    }
}

function turnOn(context) {
    isOn = true;
    const station = STATIONS[currentStation];

    context.noodle.audio.attachSource("music", {
        clip: station.clip,
        loop: true,
        volume: 0.7,
        maxDistance: 15
    });
    context.noodle.audio.play("music");

    // Announce
    context.noodle.world.emit("radio_on", {
        station: station.name,
        position: context.entity.position
    });
}

function nextStation(context) {
    currentStation = (currentStation + 1) % STATIONS.length;
    const station = STATIONS[currentStation];

    // Swap clip
    context.noodle.audio.stop("music");
    context.noodle.audio.attachSource("music", {
        clip: station.clip,
        loop: true,
        volume: 0.7,
        maxDistance: 15
    });
    context.noodle.audio.play("music");
}

function turnOff(context) {
    isOn = false;
    context.noodle.audio.stop("music");
    context.noodle.audio.removeSource("music");
}
```

### Example: Noodling Footsteps

```javascript
// footstep_facet.js - Attached to a noodling

function onAnimationFrame(context, animState) {
    // Play footstep on specific animation frames
    const footstepFrames = [5, 15];  // Left foot, right foot

    if (animState.name === "walking" && footstepFrames.includes(animState.frame)) {
        // Vary pitch slightly for realism
        const pitch = 0.9 + Math.random() * 0.2;
        context.noodle.audio.setPitch("footsteps", pitch);
        context.noodle.audio.play("footsteps");
    }
}

function onGroundChange(context, groundType) {
    // Swap footstep sound based on surface
    const clips = {
        "grass": "audio/footstep_grass.ogg",
        "stone": "audio/footstep_stone.ogg",
        "wood": "audio/footstep_wood.ogg",
        "water": "audio/footstep_splash.ogg",
    };

    const clip = clips[groundType] || clips["stone"];

    context.noodle.audio.attachSource("footsteps", {
        clip: clip,
        volume: 0.4,
        maxDistance: 10,
        spatial: true
    });
}
```

---

## 6. IMPLEMENTATION PLAN

### Phase 1: Audio Emitters on Entities
1. Add `audio_emitters` field to Noodling/Prim in scene_packet.py
2. Extend spatial_audio.py to track entity-attached sources
3. Add AudioEmitterComponent to entity YAML format
4. Wire position updates: entity moves → audio source moves

### Phase 2: Scripting API
1. Create `audio_api.py` in scripting module
2. Wire to ScriptedFacet context
3. Add event system (on("ended"), on("started"))
4. Test with interactive prims

### Phase 3: Networking
1. Add audio emitter state to entity sync packets
2. Client receives "emitter X started playing" → plays locally
3. Server is authoritative on WHEN sounds play
4. Client handles HOW they're positioned/mixed

### Phase 4: Voice Chat
1. Deploy LiveKit (or use Daily.co for quick start)
2. Add voice room management to cMUSH
3. Wire voice streams to spatial audio positioning
4. Add push-to-talk / voice activation

---

## 7. BANDWIDTH CONSIDERATIONS

### Per-User Bandwidth

| Stream | Direction | Bandwidth |
|--------|-----------|-----------|
| State sync | Server → Client | ~5 KB/s per entity |
| Player input | Client → Server | ~1 KB/s |
| Voice (Opus) | P2P via SFU | ~32 kbps per stream |
| Asset loading | CDN → Client | Burst on load |

### Scaling

- **10 users in room:** ~500 KB/s state + 320 kbps voice
- **50 users in room:** ~2.5 MB/s state + 1.6 Mbps voice
- **Interest management:** Only sync entities within perception range

### Optimizations

1. **Delta compression** - Only send changed fields
2. **Interest management** - Don't sync distant entities
3. **LOD for updates** - Lower update rate for far entities
4. **Voice culling** - Don't forward voice from users > 50m away
