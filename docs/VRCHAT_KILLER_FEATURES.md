# VRChat Killer Features - Drinking Their Milkshake

**Goal:** Build the social VR features that make VRChat addictive, but on our Gaussian World Engine.

**Key Insight:** Gaussians make mirrors and portals TRIVIAL. Traditional rendering requires expensive scene re-renders and stencil tricks. Gaussians are just point clouds - render from any camera position instantly.

---

## 1. MIRRORS (The VRChat Obsession)

VRChat users congregate around mirrors because:
- Self-expression validation (seeing your avatar)
- Social gathering spot (watch yourself + others)
- Performance optimization (mirrors are intentional perf sinks)

### Implementation

```
Mirror Surface
     |
     v
[Reflect Camera] --> [Gaussian Renderer] --> [Render Texture]
     |                                              |
     v                                              v
Camera.position = reflect(main_cam, mirror_plane)   Display on mirror quad
Camera.forward = reflect(main_cam.forward, normal)
```

**With Gaussians:**
1. Compute reflected camera position across mirror plane
2. Render Gaussians from reflected viewpoint to texture
3. Display texture on mirror surface
4. Done! No stencil buffers, no scene graph traversal

**Mirror Types:**
- **Flat mirror** - Simple plane reflection
- **Curved mirror** - Multiple sample points, blend
- **Portal mirror** - Shows different location (see Portals)
- **Time-delayed mirror** - Buffer last N frames, show past self

### Code Structure
```python
class MirrorSurface:
    plane_normal: Vector3
    plane_point: Vector3
    render_texture: Texture
    resolution: Tuple[int, int]

    def get_reflected_camera(self, main_camera: Camera) -> Camera:
        # Reflect position across plane
        d = dot(main_camera.position - self.plane_point, self.plane_normal)
        reflected_pos = main_camera.position - 2 * d * self.plane_normal

        # Reflect forward direction
        reflected_forward = reflect(main_camera.forward, self.plane_normal)

        return Camera(position=reflected_pos, forward=reflected_forward)
```

---

## 2. PORTALS (Portal-Game Style)

See-through windows to other locations. Cross the threshold, teleport there.

### Implementation

```
Portal A (in Room 1)          Portal B (in Room 2)
     |                              |
     v                              v
[Camera at B's position]      [Camera at A's position]
     |                              |
     v                              v
[Render Room 2 Gaussians]     [Render Room 1 Gaussians]
     |                              |
     v                              v
[Display on Portal A]         [Display on Portal B]
```

**Key Features:**
- **Linked pairs** - Portal A shows Portal B's view, and vice versa
- **Recursive rendering** - Portal in portal in portal (depth limited)
- **Seamless crossing** - Walk through, teleport to other side
- **Momentum preservation** - Exit velocity matches entry velocity (relative to portal orientation)

### Portal Rendering
```python
class Portal:
    position: Vector3
    rotation: Quaternion
    linked_portal: 'Portal'
    render_depth: int = 2  # Recursion limit

    def get_destination_camera(self, viewer_camera: Camera) -> Camera:
        # Transform viewer position relative to this portal
        local_pos = self.world_to_local(viewer_camera.position)
        local_forward = self.world_to_local_direction(viewer_camera.forward)

        # Transform to linked portal's space (flipped 180 on Y)
        dest_pos = self.linked_portal.local_to_world(local_pos * Vector3(1, 1, -1))
        dest_forward = self.linked_portal.local_to_world_direction(local_forward * Vector3(1, 1, -1))

        return Camera(position=dest_pos, forward=dest_forward)
```

### Recursive Portals
For portal-in-portal rendering:
1. Render deepest portals first (no recursion)
2. Use their textures when rendering parent portals
3. Depth limit prevents infinite recursion
4. Each level halves resolution for performance

---

## 3. MULTI-USER NETWORKING

Real-time state synchronization for social presence.

### Architecture

```
Client A          Server           Client B
   |                |                  |
   |--[position]--->|                  |
   |                |--[A's position]->|
   |                |<-[position]------|
   |<-[B's position]|                  |
```

### State Sync Protocol

**Entity State Packet:**
```json
{
  "entity_id": "uuid",
  "timestamp": 1703184000.123,
  "transform": {
    "position": [x, y, z],
    "rotation": [qx, qy, qz, qw],
    "scale": [sx, sy, sz]
  },
  "animation": {
    "blend_shapes": {"happy": 0.5, "blink": 0.0},
    "spring_bones": true  // Let client simulate locally
  },
  "voice": {
    "speaking": true,
    "volume": 0.8
  }
}
```

**Network Topology Options:**
1. **Client-Server** (simple, authoritative)
2. **Peer-to-Peer** (lower latency, WebRTC)
3. **Hybrid** - Server for state, P2P for voice

### Interpolation & Prediction
- **Interpolation buffer** - 100ms delay, smooth playback
- **Extrapolation** - Predict position when packets late
- **Snap threshold** - Teleport if error > 2m

### We Already Have:
- WebSocket infrastructure in cMUSH server
- Entity state in SceneStateManager
- Scene packets with transforms

**New Needed:**
- Client-side interpolation
- Voice channel integration
- Lobby/instance management

---

## 4. SPATIAL AUDIO

Positional sound for immersion.

### Architecture

```
Audio Source (position, audio data)
       |
       v
[Web Audio API PannerNode]
       |
       +-- Distance attenuation (1/r^2)
       +-- HRTF spatialization (left/right ear delay)
       +-- Occlusion (walls block sound)
       |
       v
[Listener position/orientation]
       |
       v
[Stereo output]
```

### Sound Types

1. **Point sources** - Objects, footsteps, ambient
2. **Voice chat** - Other users, spatially positioned
3. **Ambient zones** - Background audio per area
4. **UI sounds** - Non-spatial, always centered

### Voice Chat Integration

```
Microphone -> WebRTC -> [Other clients]
                              |
                              v
                        [Spatial positioning]
                              |
                              v
                        [3D audio output]
```

**WebRTC for voice:**
- Peer-to-peer (low latency)
- Opus codec (optimized for voice)
- Spatial positioning based on avatar position

### Occlusion
- Raycast from listener to source
- If blocked by geometry, reduce high frequencies
- "Muffled" effect for sound through walls

---

## 5. PARTICLES

GPU-accelerated particle systems.

### Gaussian Particles!

**Key Insight:** Gaussians ARE soft particles. Use small Gaussians for:
- Fire (orange/yellow, rising)
- Smoke (gray, expanding)
- Sparkles (white, random motion)
- Snow (white, falling)
- Magic effects (colored, swirling)

### Particle System
```python
class GaussianParticleSystem:
    max_particles: int = 10000
    emission_rate: float  # particles/second
    lifetime: Tuple[float, float]  # min, max seconds

    # Initial state
    spawn_position: Vector3
    spawn_velocity: Vector3
    spawn_spread: float

    # Physics
    gravity: Vector3
    drag: float

    # Appearance
    color_over_lifetime: Gradient
    size_over_lifetime: Curve
    opacity_over_lifetime: Curve

    # Each particle is a small Gaussian
    particles: List[GaussianParticle]
```

### GPU Compute (WebGPU)
```wgsl
@compute @workgroup_size(256)
fn update_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= particle_count) { return; }

    // Update velocity
    particles[i].velocity += gravity * dt;
    particles[i].velocity *= (1.0 - drag * dt);

    // Update position
    particles[i].position += particles[i].velocity * dt;

    // Update lifetime
    particles[i].age += dt;
    if (particles[i].age > particles[i].lifetime) {
        // Respawn at emitter
        respawn_particle(i);
    }
}
```

---

## 6. CUSTOM SHADERS / PROGRAMMABLE SURFACES

In-world shader surfaces for custom effects.

### Options

**A. Post-process effects on Gaussian render:**
- Bloom, color grading, vignette
- Applied after Gaussian rasterization
- Global or per-portal

**B. Custom Gaussian appearance:**
- Modify SH coefficients for color effects
- Scale/opacity animation
- "Hologram" effect, "ghost" transparency

**C. Screen-space effects on surfaces:**
- Render Gaussians to texture
- Apply shader (water ripples, CRT scanlines, etc.)
- Display on in-world surface

**D. ScriptedFacet material control:**
```javascript
// In a ScriptedFacet
function update(context) {
    const t = context.time;

    // Animate material color based on time
    const hue = (t * 0.1) % 1.0;
    context.noodle.world.setMaterialColor(
        "disco_ball",
        hslToRgb(hue, 1.0, 0.5)
    );
}
```

### Water Shader Example
```
[Gaussian Scene] --> [Render to texture]
                            |
                            v
                    [Water surface shader]
                    - Sample scene texture with UV distortion
                    - Add caustics
                    - Fresnel reflection blend
                            |
                            v
                    [Display on water plane]
```

---

## 7. IMPLEMENTATION PRIORITY

### Phase 1: Core Social (Week 1-2)
1. **Mirrors** - High impact, relatively simple
2. **Basic networking** - Position sync over WebSocket
3. **Spatial audio** - Web Audio API foundation

### Phase 2: Advanced Features (Week 3-4)
4. **Portals** - Linked portal pairs
5. **Voice chat** - WebRTC integration
6. **Basic particles** - Gaussian particle emitters

### Phase 3: Polish (Week 5+)
7. **Recursive portals** - Portal in portal
8. **Custom shaders** - Post-process pipeline
9. **Advanced particles** - GPU compute, collision

---

## 8. ARCHITECTURE

### New Files Needed

```
noodlestudio/core/social/
    mirror_renderer.py      # Mirror camera + render
    portal_system.py        # Linked portals
    network_sync.py         # Entity state sync
    spatial_audio.py        # 3D audio positioning
    voice_chat.py           # WebRTC voice
    particle_system.py      # Gaussian particles
    shader_surfaces.py      # Custom surface effects
```

### Integration Points

- **GaussianRenderer** - Add multi-camera support for mirrors/portals
- **SceneStateManager** - Network state sync hooks
- **FacetExecutor** - Particle/shader ScriptedFacets
- **WebSocket server** - Voice signaling, state broadcast

---

## WHY GAUSSIANS WIN

| Feature | Traditional Rendering | Gaussian Splatting |
|---------|----------------------|-------------------|
| Mirrors | Re-render entire scene | Just change camera |
| Portals | Stencil buffer tricks | Just change camera |
| Particles | Billboard quads | Native soft blending |
| LOD | Manual mesh levels | Automatic (fewer splats) |
| Occlusion | Complex culling | Depth sorting free |

**Gaussians are inherently multi-view.** The same splat data renders from any angle.
This is our unfair advantage over polygon-based VRChat.

---

## DRINKING THE MILKSHAKE

VRChat's moat:
- Network effects (users attract users)
- Content ecosystem (avatar creators)
- Brand recognition

Our advantages:
- **Better rendering** (Gaussians > polygons for organic content)
- **AI-native** (Noodlings cognition built-in)
- **Open source** (no platform lock-in)
- **VRM import** (leverage existing avatar ecosystem)
- **Text-first** (MUD compatibility, accessibility)

**Strategy:** Make it trivially easy to import VRChat-compatible content (VRM avatars, VRCSDK worlds) but render them BETTER and make them SMARTER.

The mirror feature alone will get VRChat refugees excited. "OMG the mirrors are so smooth!"
