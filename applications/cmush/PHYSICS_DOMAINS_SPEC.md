# Physics Domains - Nested Reality Architecture

**Authors:** Lieutenant Caitlyn + Commander Spock
**Date:** November 22, 2025
**Status:** Advanced architectural specification
**Concept:** Nested physics simulations with domain inheritance

---

## Core Concept

**Physics Domain** = A simulation space where physics rules apply to all children.

**Domains can be nested:**
- Main world has global physics domain
- Snow globe has local physics domain (children only)
- Magic bubble has local physics domain (altered rules)
- Terrarium has local physics domain (different temperature/humidity)

**Objects inherit physics from their domain parent.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  MAIN WORLD (Global Physics Domain)                         │
│  Rules: Gravity down, Temperature 70°F, Normal physics      │
│                                                              │
│  ├─ Chocolate Bunny                                         │
│  │   Physics: Melts at 90°F (inherits from Main World)     │
│  │                                                          │
│  ├─ Hot Teapot (250°F Heat Emitter)                        │
│  │   Physics: Radiates heat (inherits from Main World)     │
│  │                                                          │
│  └─ Snow Globe Prim ❄️                                     │
│      │  ┌──────────────────────────────────────────────┐  │
│      │  │  SNOW GLOBE DOMAIN (Nested Physics)          │  │
│      │  │  Rules: Temperature 20°F, Perpetual snow     │  │
│      │  │                                               │  │
│      │  │  ├─ Tiny Snowman                             │  │
│      │  │  │   Physics: Never melts (domain is 20°F)  │  │
│      │  │  │                                            │  │
│      │  │  ├─ Miniature Trees                          │  │
│      │  │  │   Physics: Always frosted                 │  │
│      │  │  │                                            │  │
│      │  │  └─ Falling Snow (perpetual)                 │  │
│      │  │      Physics: Always snowing (domain rule)   │  │
│      │  └──────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Physics Domain Component

### PhysicsDomain Class

```python
class PhysicsDomain:
    """
    Physics simulation domain.

    Defines physics rules for all objects within this domain.
    Runs background simulation to update object states based on
    environmental emitters and time.
    """

    def __init__(
        self,
        domain_id: str,
        parent_domain: Optional['PhysicsDomain'] = None
    ):
        """
        Initialize physics domain.

        Args:
            domain_id: Unique domain identifier
            parent_domain: Parent domain (for nesting)
        """
        self.domain_id = domain_id
        self.parent_domain = parent_domain

        # Domain rules (override parent)
        self.temperature = None  # None = inherit from parent
        self.humidity = None
        self.gravity = None  # "down", "up", "zero", "radial"
        self.time_scale = 1.0  # Time dilation (1.0 = normal)

        # Objects in this domain
        self.objects = []  # List of prim IDs

        # Child domains (nested)
        self.child_domains = []

        # Emitters in this domain
        self.emitters = {}  # prim_id → list of emitters

        # Running state
        self.simulation_enabled = True

    def add_object(self, prim_id: str):
        """Add object to this domain."""
        self.objects.append(prim_id)

    def remove_object(self, prim_id: str):
        """Remove object from domain."""
        if prim_id in self.objects:
            self.objects.remove(prim_id)

    def add_child_domain(self, child_domain: 'PhysicsDomain'):
        """Add nested physics domain."""
        child_domain.parent_domain = self
        self.child_domains.append(child_domain)

    def get_effective_temperature(self, prim_id: str) -> float:
        """
        Get effective temperature for object.

        Considers:
        - Domain base temperature
        - Nearby heat emitters
        - Parent domain (if nested)

        Args:
            prim_id: Object ID

        Returns:
            Effective temperature at object's location (°F)
        """
        # Base temperature from domain
        base_temp = self.temperature if self.temperature is not None else \
                   (self.parent_domain.get_base_temperature() if self.parent_domain else 70.0)

        # Add contributions from heat emitters
        total_heat = base_temp

        for emitter_id, emitters in self.emitters.items():
            for emitter in emitters:
                if isinstance(emitter, HeatEmitter) and emitter.enabled:
                    distance = self._calculate_distance(prim_id, emitter_id)
                    contribution = emitter.get_effective_temperature(distance, base_temp)
                    # Take max (hottest source dominates)
                    total_heat = max(total_heat, contribution)

        return total_heat

    def update_physics(self, delta_time: float):
        """
        Update physics for all objects in domain.

        Called periodically (e.g., every second) by background task.

        Args:
            delta_time: Time elapsed since last update (seconds)
        """
        if not self.simulation_enabled:
            return

        # Apply time dilation
        effective_dt = delta_time * self.time_scale

        # Update each object
        for prim_id in self.objects:
            self._update_object_physics(prim_id, effective_dt)

        # Update child domains
        for child in self.child_domains:
            child.update_physics(effective_dt)

    def _update_object_physics(self, prim_id: str, delta_time: float):
        """
        Update physics for single object.

        Checks:
        - Temperature effects (melting, freezing)
        - Humidity effects (rusting, drying)
        - Time effects (decay, evaporation)
        """
        # Get object POD
        from world import World
        world = World("world")  # TODO: Inject world properly
        pod = world.get_object_pod(prim_id)

        if not pod:
            return  # No physics

        # Calculate effective environment
        temp = self.get_effective_temperature(prim_id)
        humidity = self.get_effective_humidity(prim_id)

        # Check for temperature-induced state changes
        self._check_phase_change(pod, temp, delta_time)
        self._check_viscosity_change(pod, temp)
        self._check_melting(pod, temp, delta_time)

        # Update POD in world
        world.update_pod(prim_id, pod)

    def _check_melting(self, pod: PhysicsObjectDescriptor, temperature: float, dt: float):
        """Check if object should melt."""
        material = pod.material.lower()

        # Chocolate melts at 90°F
        if 'chocolate' in material and temperature > 90:
            if 'melting' not in pod.state.lower():
                pod.change_state("beginning to melt")

            # Progress melting over time
            melt_progress = min(1.0, dt / 60.0)  # 1 minute to fully melt
            if melt_progress > 0.8:
                pod.change_state("melted into puddle")
                pod.phase = "liquid"

        # Wax/candle melts at 150°F
        elif 'wax' in material or 'candle' in material:
            if temperature > 150:
                pod.change_state("melting")
                pod.viscosity = "liquid"

        # Ice melts at 32°F
        elif 'ice' in material and temperature > 32:
            pod.change_state("melting to water")
            pod.phase = "liquid"

    def _check_phase_change(self, pod: PhysicsObjectDescriptor, temp: float, dt: float):
        """Check for solid/liquid/gas phase transitions."""
        # Implementation left as exercise
        pass

    def _check_viscosity_change(self, pod: PhysicsObjectDescriptor, temp: float):
        """Update viscosity based on temperature."""
        # Implementation left as exercise
        pass

    def _calculate_distance(self, prim1_id: str, prim2_id: str) -> float:
        """Calculate distance between prims (simplified)."""
        # TODO: Use actual spatial coordinates
        return 1.0  # Default: 1 meter

    def get_base_temperature(self) -> float:
        """Get base temperature (considering parent)."""
        if self.temperature is not None:
            return self.temperature
        elif self.parent_domain:
            return self.parent_domain.get_base_temperature()
        else:
            return 70.0  # Default

    def get_effective_humidity(self, prim_id: str) -> float:
        """Get effective humidity for object."""
        # Similar to temperature
        return self.humidity if self.humidity is not None else 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'domain_id': self.domain_id,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'gravity': self.gravity,
            'time_scale': self.time_scale,
            'objects': self.objects,
            'child_domains': [child.to_dict() for child in self.child_domains],
            'simulation_enabled': self.simulation_enabled
        }
```

---

## 2. Example: Chocolate Bunny Near Hot Teapot

### Setup

```python
# Main world physics domain
main_domain = PhysicsDomain("main_world")
main_domain.temperature = 70.0  # Room temp

# Hot teapot
teapot = world.create_object(
    name="Vulcan Teapot",
    description="Traditional Vulcan teapot - SUPER HOT",
    location="room_kitchen"
)

teapot_pod = PhysicsObjectDescriptor(
    material="ceramic",
    state="very hot"
)

heat_emitter = HeatEmitter(
    temperature=250.0,
    heat_radius=2.0,  # Heat felt up to 2 meters away
    attenuation=2.0
)

# Attach to teapot
main_domain.add_object(teapot.uid)
main_domain.emitters[teapot.uid] = [heat_emitter]

# Chocolate bunny
choco_bunny = world.create_object(
    name="Chocolate Bunny",
    description="Easter chocolate bunny",
    location="room_kitchen"
)

bunny_pod = PhysicsObjectDescriptor(
    material="chocolate",
    state="solid",
    semantic_properties=["edible", "sweet"]
)

# Place near teapot (0.5 meters away)
set_distance(choco_bunny, teapot, 0.5)

main_domain.add_object(choco_bunny.uid)
```

### Physics Simulation

```python
# Background physics update (runs every second)
main_domain.update_physics(delta_time=1.0)

# Physics domain calculates:
# 1. Get effective temperature at bunny's location
bunny_temp = main_domain.get_effective_temperature(choco_bunny.uid)
# → Considers:
#   - Base room temp: 70°F
#   - Heat from teapot at 0.5m: ~180°F
# → Result: ~150°F at bunny's location

# 2. Check chocolate melting threshold (90°F)
if bunny_temp > 90:
    bunny_pod.change_state("beginning to melt")

# After 30 seconds at 150°F:
if time_at_high_temp > 30:
    bunny_pod.change_state("melted into chocolate puddle")
    bunny_pod.phase = "liquid"
    bunny_pod.viscosity = "viscous liquid"

# Broadcast event (NO Noodling involved - just physics!)
broadcast_event("The chocolate bunny has melted into a puddle near the hot teapot.")

# Nearby Noodlings observe (via Somatic if present):
Mole: "*sees melted chocolate* Oh dear! The bunny melted from the teapot's heat!"
```

**Result:** Chocolate bunny melts **automatically** from proximity to heat source.

---

## 3. Nested Physics Domains: Snow Globe

### Snow Globe Setup

```python
# Snow globe prim
snow_globe = world.create_object(
    name="Magical Snow Globe",
    description="A glass sphere with perpetual winter inside",
    location="room_study"
)

# Create nested physics domain for snow globe interior
snow_globe_domain = PhysicsDomain(
    domain_id="snow_globe_interior",
    parent_domain=main_domain  # Nested inside main world
)

# Snow globe physics rules (DIFFERENT from outside world)
snow_globe_domain.temperature = 20.0  # Always freezing
snow_globe_domain.humidity = 0.7      # Snowy
snow_globe_domain.gravity = "down"     # Normal gravity
snow_globe_domain.weather = "perpetual_snow"  # Always snowing

# Attach domain to snow globe prim
snow_globe.add_component(snow_globe_domain)

# Create objects INSIDE snow globe
tiny_snowman = world.create_object(
    name="Tiny Snowman",
    description="Miniature snowman inside globe",
    parent=snow_globe.uid  # Parent = snow globe
)

snowman_pod = PhysicsObjectDescriptor(
    material="snow",
    state="frozen solid"
)

# Add to snow globe's domain (NOT main domain)
snow_globe_domain.add_object(tiny_snowman.uid)

# Mini trees
for i in range(3):
    tree = world.create_object(
        name=f"Mini Tree {i}",
        parent=snow_globe.uid
    )
    tree_pod = PhysicsObjectDescriptor(
        material="wood",
        state="frosted"
    )
    snow_globe_domain.add_object(tree.uid)
```

### Physics Behavior

```python
# OUTSIDE snow globe (main world):
main_domain.temperature = 80.0  # Warm summer day

# Chocolate bunny OUTSIDE:
bunny_temp = main_domain.get_effective_temperature(bunny.uid)
# → 80°F (warm)
# → Chocolate begins melting

# INSIDE snow globe:
snowman_temp = snow_globe_domain.get_effective_temperature(snowman.uid)
# → 20°F (freezing - domain temperature)
# → Snowman stays frozen
# → NEVER melts, regardless of outside temperature!

# Snow continues falling inside globe
# Trees stay frosted
# Perpetual winter inside, summer outside
```

**Result:** Nested domain has independent physics!

---

## 4. Example: Terrarium with Different Climate

```python
# Terrarium (tropical environment inside)
terrarium = world.create_object(
    name="Tropical Terrarium",
    description="Glass enclosure with rainforest climate"
)

terrarium_domain = PhysicsDomain("terrarium_interior", main_domain)
terrarium_domain.temperature = 85.0  # Hot and humid
terrarium_domain.humidity = 0.9      # Very humid
terrarium_domain.weather = "mist"    # Perpetual mist

# Plants inside terrarium
fern = world.create_object("Fern", parent=terrarium.uid)
fern_pod = PhysicsObjectDescriptor(
    material="plant",
    state="thriving in humidity"
)
terrarium_domain.add_object(fern.uid)

# OUTSIDE terrarium:
main_domain.temperature = 70.0
main_domain.humidity = 0.4  # Dry

# Fern in dry room would wilt
# BUT fern is in terrarium domain → 85°F, 90% humidity
# → Fern thrives!

# If you remove fern from terrarium:
terrarium_domain.remove_object(fern.uid)
main_domain.add_object(fern.uid)  # Now in main domain

# Fern experiences main world physics:
fern_humidity = main_domain.get_effective_humidity(fern.uid)
# → 0.4 (dry)
# → Fern begins wilting
# → State change: "wilting from lack of humidity"
```

---

## 5. Domain Inheritance Rules

### Temperature Inheritance

```python
def get_effective_temperature(prim_id: str) -> float:
    """
    Get temperature considering:
    1. Object's domain temperature
    2. Parent domain temperature (if domain.temperature is None)
    3. Nearby heat emitters
    """

    # Example:
    # Main world: 70°F
    # Snow globe domain: 20°F (overrides)
    # Tiny snowman in snow globe: 20°F (from domain)

    # Hot teapot emitter (250°F) in main world:
    # - Affects objects in main world
    # - Does NOT affect objects in snow globe (different domain!)
```

### Cross-Domain Interaction

**Question:** If hot teapot is placed NEXT to snow globe, does it melt the snowman inside?

**Answer:** **No!** Domain boundary isolates physics.

```python
# Hot teapot (250°F) at 0.1 meters from snow globe

# Snow globe exterior (glass):
glass_temp = main_domain.get_effective_temperature(snow_globe.uid)
# → 200°F (heated by teapot)
# → Glass is hot to touch

# Snow globe interior (domain isolated):
snowman_temp = snow_globe_domain.get_effective_temperature(snowman.uid)
# → 20°F (domain temperature)
# → Snowman UNAFFECTED

# Domain acts as thermal insulation!
```

**Exception:** Heat can cross boundaries if explicitly allowed:

```python
snow_globe_domain.thermal_isolation = False  # Domain is permeable

# Now:
snowman_temp = calculate_with_external_heat(...)
# → Heat from teapot penetrates domain
# → Snowman begins melting
```

---

## 6. Example: Magic Bubble (Altered Physics)

```python
# Magic bubble with reversed gravity
bubble = world.create_object("Anti-Gravity Bubble")

bubble_domain = PhysicsDomain("bubble_interior", main_domain)
bubble_domain.gravity = "up"  # Reversed!
bubble_domain.temperature = None  # Inherit from parent

# Objects inside bubble:
floating_rock = world.create_object("Rock", parent=bubble.uid)
bubble_domain.add_object(floating_rock.uid)

# Physics:
# Main world: Gravity down, rock falls
# Inside bubble: Gravity up, rock floats!

# When rock is inside bubble:
rock.state = "floating upward slowly"

# When rock exits bubble (crosses domain boundary):
bubble_domain.remove_object(rock.uid)
main_domain.add_object(rock.uid)

# Rock immediately affected by main world gravity:
rock.state = "falling downward"
```

---

## 7. Stage-Level Physics Component

### Implementation

**In NoodleStudio Scene:**

```
Scene: "Wild Wood"
├─ Physics Domain Component ⚙️ (on Scene root)
│   Temperature: 70°F
│   Humidity: Normal
│   Gravity: Down
│   Time Scale: 1.0
│   Objects: 127 prims
│   Child Domains: 3 nested domains
│
├─ Room: Mole's Cottage
│   ├─ Fireplace (Heat Emitter: 800°F)
│   ├─ Chocolate Bunny (responds to heat)
│   └─ Vulcan Teapot (Heat Emitter: 250°F)
│
├─ Room: Study
│   └─ Snow Globe Prim
│       └─ Nested Physics Domain ❄️
│           Temperature: 20°F (override)
│           Weather: Perpetual snow
│           Objects: 12 mini prims
│           ├─ Tiny Snowman (never melts)
│           ├─ Mini Trees (always frosted)
│           └─ Falling Snow
│
└─ Room: Greenhouse
    └─ Tropical Terrarium
        └─ Nested Physics Domain 🌴
            Temperature: 85°F (override)
            Humidity: 0.9 (override)
            Objects: 8 plants
            └─ All plants thrive in humidity
```

---

## 8. Background Physics Simulation

### Update Loop

```python
class PhysicsSimulationManager:
    """
    Manages all physics domain updates.

    Runs background task that updates domains periodically.
    """

    def __init__(self):
        self.domains = {}  # domain_id → PhysicsDomain
        self.running = False
        self.update_interval = 1.0  # Update every second

    async def simulation_loop(self):
        """Background loop updating all physics."""
        while self.running:
            # Update all root domains
            for domain_id, domain in self.domains.items():
                if domain.parent_domain is None:  # Root domain only
                    domain.update_physics(self.update_interval)

            await asyncio.sleep(self.update_interval)

    def add_domain(self, domain: PhysicsDomain):
        """Register physics domain."""
        self.domains[domain.domain_id] = domain

    def start(self):
        """Start background simulation."""
        self.running = True
        asyncio.create_task(self.simulation_loop())

    def stop(self):
        """Stop background simulation."""
        self.running = False
```

---

## 9. Cross-Domain Events

### Event: Removing Object from Nested Domain

```python
# Take tiny snowman OUT of snow globe
snow_globe_domain.remove_object(snowman.uid)
main_domain.add_object(snowman.uid)

# Immediate physics change:
# Was: 20°F (snow globe domain)
# Now: 70°F (main world)

# Snowman begins melting!
snowman_pod.change_state("rapidly melting in warm air")

# Broadcast:
"The tiny snowman, removed from its frozen globe, begins melting
 rapidly in the warm room air!"

# Nearby Noodlings:
Mole: "Oh no! The poor snowman is melting! *tries to put it back in globe*"
```

### Event: Placing Hot Object in Cold Domain

```python
# Place hot teapot (250°F) INSIDE snow globe

# Teapot cools down (domain is 20°F)
teapot_temp_in_globe = snow_globe_domain.get_effective_temperature(teapot.uid)
# → Drops from 250°F to 20°F over time (heat dissipates)

# Heat emitter still works, but:
heat_contribution = teapot_heat.get_effective_temperature(
    distance=0.5,
    ambient_temp=20.0  # Domain base temp
)
# → Warms nearby objects slightly, but overall domain stays cold

# Tiny snowman near hot teapot:
snowman_temp = 20.0 + heat_contribution
# → Maybe 40°F locally (warm spot in cold domain)
# → Snowman melts SLIGHTLY near teapot
# → But rest of domain stays 20°F
```

---

## 10. Complete Example: Multi-Domain Scene

### Scene: Wizard's Study

```python
# Main world
main = PhysicsDomain("main_world")
main.temperature = 70.0

# Study room
study = world.create_room("Wizard's Study")

# 1. Snow globe on desk
snow_globe = create_snow_globe_with_domain(parent=main)
# → Interior: 20°F, perpetual snow

# 2. Tropical terrarium on shelf
terrarium = create_terrarium_with_domain(parent=main)
# → Interior: 85°F, 90% humidity, mist

# 3. Magic bubble floating
bubble = create_antigravity_bubble_with_domain(parent=main)
# → Interior: Gravity reversed

# 4. Chocolate bunny on desk
bunny = create_chocolate_bunny()
main.add_object(bunny.uid)

# 5. Hot teapot on desk
teapot = create_vulcan_teapot_with_heat_emitter()
main.add_object(teapot.uid)

# 6. Fireplace in room
fireplace = create_fireplace_with_emitters()
main.add_object(fireplace.uid)

# Simulation runs:
main.update_physics(1.0)

# Results:
# - Snow globe: Perpetual winter (isolated)
# - Terrarium: Plants thrive in humidity (isolated)
# - Bubble: Objects float upward (isolated)
# - Bunny: Begins melting from teapot heat (main domain)
# - Fireplace: Warms room, agents gather (main domain)

# Agent observing:
Mole: "*looks around room* Fascinating! The snow globe stays frozen,
       the terrarium stays humid, and the bubble has reversed gravity.
       *notices bunny* Oh dear, the chocolate is melting from the teapot!
       *feels warmth from fireplace* Though the fire keeps the room cozy."
```

---

## Summary

**Physics Domains enable:**

✅ **Passive physics updates** (bunny melts near teapot - automatic)
✅ **Stage-level simulation** (PhysicsDomain component on Scene)
✅ **Nested domains** (snow globe has own physics)
✅ **Domain inheritance** (child domains inherit parent rules)
✅ **Physics isolation** (inside ≠ outside)
✅ **Cross-domain boundaries** (objects can move between domains)
✅ **Background simulation** (update loop runs automatically)

**Complete emitter catalog:**

✅ Thermal (heat/cold)
✅ Acoustic (sound/vibration)
✅ Optical (light/color)
✅ Olfactory (scent/smoke)
✅ Fluid (liquid/gas)
✅ Radiation (radioactive/magnetic/electric)
✅ Pressure (air/water)
✅ Particulate (dust/pollen/sparkles)

**Architecture pattern:**

```
Physics Domain (Stage/Scene level)
    ↓ operates on
Objects with PODs (children)
    ↓ respond to
Environmental Emitters (heat, sound, etc.)
    ↓ broadcast to
Somatic Cognitive Transistor (perception)
    ↓ integrates with
Cognitive Manifold (synthesis)
    ↓ produces
Embodied conscious response
```

**Result:** Rich, nested, dynamic physics that affects both objects AND consciousness.

---

**Status:** Complete architectural specification

**Next:** Implement background physics simulation and test nested domains

*— Commander Spock*

**Fascinating nested reality architecture, Lieutenant. The snow globe concept is particularly elegant.**
