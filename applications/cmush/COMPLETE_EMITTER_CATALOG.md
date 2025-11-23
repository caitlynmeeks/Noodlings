# Complete Environmental Emitter Catalog

**Authors:** Commander Spock + Lieutenant Caitlyn
**Date:** November 22, 2025
**Purpose:** Comprehensive taxonomy of physical signal emitters

---

## Emitter Categories

### 1. Thermal Emitters

**HeatEmitter** (radiates warmth)
- **Examples:** Campfire, stove, sun, lava, Vulcan teapot
- **Properties:** temperature (°F), heat_radius, attenuation
- **Effects:** Warmth, comfort, burns, melting
- **Salience:** Distance-dependent, context-aware

**ColdEmitter** (radiates cold/absorbs heat)
- **Examples:** Ice block, freezer, liquid nitrogen, winter wind
- **Properties:** temperature (°F), cold_radius, attenuation
- **Effects:** Cooling, freezing, shivering, frost
- **Salience:** Distance-dependent

---

### 2. Acoustic Emitters

**SoundEmitter** (broadcasts sound waves)
- **Examples:** Siren, music box, engine, speech, bells
- **Properties:** decibels, sound_type, frequency, pattern
- **Effects:** Pleasure, pain, distraction, communication
- **Salience:** Decibel-dependent, context-aware (orphanage vs road)

**VibrationEmitter** (low-frequency oscillation)
- **Examples:** Earthquake, heavy machinery, bass speakers, footsteps
- **Properties:** intensity, frequency, pattern, radius
- **Effects:** Shaking, rumbling feeling, disorientation
- **Salience:** Intensity-dependent
- **Example:** "The ground shakes from the heavy machinery. *stumbles* Hard to stand!"

---

### 3. Optical Emitters

**LightEmitter** (illuminates area)
- **Examples:** Torch, lantern, sun, glowing crystal, fire imp
- **Properties:** brightness (lumens), color, light_radius, flicker
- **Effects:** Vision enabled, mood boost, glare, shadows
- **Salience:** Brightness-dependent
- **Examples:**
  - Dim candle (50 lumens): "Cozy lighting" (0.2 salience)
  - Bright searchlight (10,000 lumens): "*squints* BLINDING!" (0.8 salience)

**ColorEmitter** (specific wavelength)
- **Examples:** Red light, blue glow, green crystal, rainbow
- **Properties:** color (hex), intensity, radius
- **Effects:** Mood modulation (blue = calm, red = energizing)
- **Salience:** Moderate (0.3-0.5)
- **Example:** "The soft blue glow is calming. *relaxes*"

---

### 4. Olfactory Emitters

**ScentEmitter** (disperses odor molecules)
- **Examples:** Bakery, flowers, garbage, perfume, skunk
- **Properties:** scent_type, intensity, pleasantness, radius
- **Effects:** Attraction, repulsion, hunger, nausea, memory triggers
- **Salience:** Pleasantness and intensity dependent
- **Examples:**
  - Fresh bread (pleasant, 0.8 intensity): "Mmm! *sniffs* Smells delicious!" (0.6 salience)
  - Garbage (unpleasant, 0.9 intensity): "*gags* What IS that smell?!" (0.8 salience)
  - Flowers (pleasant, 0.4 intensity): "*sniffs* Nice" (0.2 salience)

**SmokeEmitter** (particulate + scent)
- **Examples:** Factory smoke, campfire, incense, cigarette
- **Properties:** density, composition, toxicity, height
- **Effects:** Coughing, eye irritation, visibility reduction
- **Salience:** Density-dependent (0.7 for thick smoke)
- **Example:** "*coughs* Can't breathe in this smoke! *covers nose*"

---

### 5. Fluid Emitters

**LiquidEmitter** (flows/leaks liquid)
- **Examples:** Pipe leak, fountain, dripping faucet, bleeding wound
- **Properties:** flow_rate, liquid_type, viscosity, temperature
- **Effects:** Wetness, puddle formation, flooding
- **Salience:** Flow rate dependent
- **Examples:**
  - Dripping faucet (0.1 L/min): "*hears drip* Annoying drip" (0.3 salience)
  - Burst pipe (50 L/min): "*water spraying* FLOOD! GET BUCKETS!" (0.9 salience)
  - Fountain (5 L/min): "*listens to burbling* Pleasant fountain" (0.2 salience)

**GasEmitter** (releases gas)
- **Examples:** Natural gas leak, steam vent, perfume spray, fog machine
- **Properties:** gas_type, flow_rate, toxicity, visibility
- **Effects:** Smell, breathing difficulty, fog/obscuration
- **Salience:** Toxicity and smell dependent
- **Example:** "*sniff sniff* Do you smell gas? *worried*" (0.7 salience)

---

### 6. Radiation Emitters

**RadioactiveEmitter** (ionizing radiation)
- **Examples:** Uranium ore, reactor core, medical X-ray, radium watch
- **Properties:** radiation_type (alpha/beta/gamma), intensity (rads/hr), radius
- **Effects:** Invisible danger, Geiger counter clicks, long-term harm
- **Salience:** NOT directly perceivable (unless Geiger counter present)
- **Example (with Geiger):** "*Geiger counter CLICKING RAPIDLY* Radiation! Get back!" (0.9 salience)

**MagneticEmitter** (magnetic field)
- **Examples:** Magnet, electromagnet, lodestone, Earth's magnetic field
- **Properties:** field_strength (gauss), polarity, radius
- **Effects:** Metal attraction, compass deflection, iron filings align
- **Salience:** Low for agents (0.1), unless holding metal
- **Example:** "*metal sword yanks toward magnet* Whoa! Strong magnetic pull!"

**ElectricEmitter** (electric field)
- **Examples:** Static electricity, Tesla coil, power lines, lightning
- **Properties:** voltage, current, arc_distance, frequency
- **Effects:** Shock, hair standing up, tingling, attraction
- **Salience:** Voltage-dependent (high voltage = 0.9 salience!)
- **Example:** "*hair stands on end* Static charge! *ZAP!* OUCH!"

---

### 7. Pressure Emitters

**AirPressureEmitter** (wind/pressure differential)
- **Examples:** Wind tunnel, storm front, vacuum, decompression
- **Properties:** pressure (psi), direction, strength
- **Effects:** Push/pull, difficulty moving, ear popping
- **Salience:** Pressure difference dependent
- **Example:** "The wind is pushing me backward! *leans into gale*"

**WaterPressureEmitter** (hydrostatic pressure)
- **Examples:** Deep water, dam, water jet
- **Properties:** pressure (psi), depth, flow_direction
- **Effects:** Crushing feeling, difficulty breathing, ear pain
- **Salience:** Depth-dependent (0.9 at great depth)
- **Example:** "*feels pressure increasing* Too deep! Going back up! *ears hurt*"

---

### 8. Particulate Emitters

**DustEmitter** (airborne particles)
- **Examples:** Dusty library, construction site, desert wind
- **Properties:** density, particle_size, composition
- **Effects:** Coughing, eye irritation, visibility reduction
- **Salience:** Density-dependent
- **Example:** "*coughs* So dusty! *waves hand through dust cloud*"

**PollenEmitter** (allergen particles)
- **Examples:** Flowering plants, fields in spring
- **Properties:** pollen_count, allergen_strength
- **Effects:** Sneezing, itchy eyes, allergic reaction
- **Salience:** Allergen sensitivity dependent
- **Example:** "*ACHOO!* *sniffles* Pollen is terrible today! *wipes nose*"

**SparkleEmitter** (glitter/sparkles)
- **Examples:** Fairy dust, glitter bomb, magical aura
- **Properties:** density, color, pattern
- **Effects:** Visual distraction, wonder, annoyance (if unwanted)
- **Salience:** Aesthetic preference dependent
- **Example:** "*eyes widen* Oooh, sparkly! So pretty!"

---

### 9. Chemical Emitters

**ToxicGasEmitter** (harmful vapors)
- **Examples:** Chlorine gas, carbon monoxide, sulfur dioxide
- **Properties:** toxicity, concentration, dispersion_rate
- **Effects:** Breathing difficulty, dizziness, unconsciousness
- **Salience:** HIGH (0.9) - survival threat
- **Example:** "*gasps* Can't breathe! *COUGH COUGH* Toxic gas! GET OUT!"

**VaporEmitter** (steam/mist)
- **Examples:** Boiling water, fog machine, breath on cold day
- **Properties:** density, temperature, visibility_reduction
- **Effects:** Obscured vision, dampness, humidity
- **Salience:** Density-dependent
- **Example:** "*peers through steam* Can barely see through this mist!"

---

### 10. Energy Field Emitters

**GravityWellEmitter** (semantic gravity)
- **Examples:** Black hole, heavy object, gravity generator
- **Properties:** strength, radius, attraction/repulsion
- **Effects:** Pulled toward, difficulty moving away, orbital behavior
- **Salience:** Strength-dependent
- **Example:** "*struggles to walk* Something is... pulling me toward it!"

**TemporalDistortionEmitter** (time anomaly - whimsical)
- **Examples:** Time crystal, temporal vortex, magic clock
- **Properties:** time_dilation_factor, radius
- **Effects:** Time moves slower/faster, thoughts feel stretched/compressed
- **Salience:** High (0.8) - very disorienting
- **Example:** "*thoughts moving slowly* ...why... is... everything... so... slow?"

---

## Complete Emitter Interaction Matrix

### Campfire on Winter Night (Multi-Emitter)

```python
campfire = world.get_object("campfire")

# Add multiple emitters
campfire.add_component(HeatEmitter(
    temperature=800.0,
    heat_radius=10.0
))

campfire.add_component(LightEmitter(
    brightness=1000,  # Lumens
    color="#FF6600",  # Orange
    light_radius=15.0
))

campfire.add_component(SoundEmitter(
    sound_type="crackling",
    decibels=45,
    pattern="intermittent"
))

campfire.add_component(SmokeEmitter(
    density=0.3,  # Light smoke
    height=5.0,   # Rises up
    scent="woodsmoke"
))

# Agent 3 meters away:
Mole's Somatic receives:
- Heat: 200°F (warm, cozy) → salience 0.4
- Light: 800 lumens (bright) → salience 0.3
- Sound: 42 dB (pleasant) → salience 0.2
- Smoke: 0.1 density (faint) → salience 0.1

# Manifold integration:
Cultural (0.5): "Fire is civilizing"
Personality (0.6): "Cozy and pleasant"
Somatic (0.4): "*warm by fire* Cozy..."

Output: "*sits by crackling fire* Lovely warmth on a cold night.
         *watches dancing flames* Quite civilized, this."
```

---

## Toad's Motor Car (Multi-Emitter Vehicle)

```python
toads_car = world.get_object("toads_motor_car")

# Sound: Air raid siren
toads_car.add_component(SoundEmitter(
    sound_type="siren",
    decibels=120
))

# Sound: Engine rumble
toads_car.add_component(SoundEmitter(
    sound_type="engine",
    decibels=85
))

# Heat: Hot engine
toads_car.add_component(HeatEmitter(
    temperature=250.0,
    heat_radius=3.0
))

# Gas: Exhaust fumes
toads_car.add_component(GasEmitter(
    gas_type="exhaust",
    toxicity=0.3,
    scent="gasoline"
))

# When Toad drives by:
Mole receives:
- Siren: 106 dB at 5m → salience 0.84 (VERY HIGH)
- Engine: 78 dB at 5m → salience 0.5 (loud)
- Heat: 110°F at 5m → salience 0.3 (warm)
- Exhaust: 0.2 toxicity → salience 0.4 (unpleasant)

# Manifold (somatic dominates):
"*winces at deafening siren* HI TOAD! *coughs from exhaust*
 WONDERFUL CAR! *shouts* CAN YOU TURN OFF THE SIREN?!"
```

---

## Summary: Complete Emitter Taxonomy

**Implemented:**
✅ Heat/Cold Emitters
✅ Sound Emitters
✅ Light Emitters

**Specified:**
✅ Liquid Emitters (pipes, leaks)
✅ Radiation Emitters (radioactive sources)
✅ Scent Emitters (smells)
✅ Smoke/Particulate Emitters
✅ Gas Emitters (toxic/benign)
✅ Vibration Emitters
✅ Magnetic Emitters
✅ Electric Emitters
✅ Pressure Emitters
✅ Gravity Wells
✅ Temporal Distortion (whimsical)

**Pattern:**
- Emitter → Signal → Distance → Effective Intensity → Salience → Somatic Response

**Architecture:**
- All emitters inherit from `EnvironmentalEmitter` base class
- All received by `SomaticCognitiveTransistor`
- All integrated by `CognitiveManifold`
- All create embodied consciousness

---

**The environmental emitter framework is complete and extensible.**

*— Commander Spock*
