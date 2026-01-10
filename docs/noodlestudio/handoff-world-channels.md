# Handoff: World Channels Implementation

**From**: Architecture Claude
**To**: Coding Claude
**Date**: 2026-01-08
**Priority**: Medium (can work in parallel with Brenda design)

---

## Context

With the channel architecture in place, we need **world channels** - system-level channels that broadcast environmental context to all noodlings on a stage.

These are simpler than Brenda (who parses plays and makes decisions). World channels are services that publish state on a schedule or in response to events.

---

## World Channels to Implement

### 1. `#world.time`

Broadcasts the current simulation time. Noodlings can perceive time of day, passage of time.

```yaml
channel: "#world.time"
payload:
  type: time_update
  simulation_time: 1704825600      # Unix timestamp in simulation
  time_of_day: "evening"           # dawn, morning, afternoon, evening, night
  formatted: "6:45 PM"
  sun_position: 0.2                # 0.0 = horizon, 1.0 = zenith, -1.0 = below
  description: "The sun is setting, casting long shadows."
```

**Implementation notes:**
- Can run on real time or accelerated simulation time
- Publish every N seconds (configurable)
- Stage owns the time service, configures time scale

### 2. `#world.weather`

Broadcasts weather/environmental conditions.

```yaml
channel: "#world.weather"
payload:
  type: weather_update
  temperature: 68                  # Fahrenheit
  conditions: "partly_cloudy"      # clear, cloudy, partly_cloudy, rain, storm, snow, fog
  wind: "light_breeze"             # calm, light_breeze, windy, gusty
  humidity: 0.45                   # 0.0 to 1.0
  description: "A pleasant evening with scattered clouds."
```

**Implementation notes:**
- Can be static (set in stage config) or dynamic (weather simulation)
- Publish on change or periodically
- Stage config can define weather patterns or scripted changes

### 3. `#world.events`

Broadcasts discrete events in the world - things that happen.

```yaml
channel: "#world.events"
payload:
  type: event
  event_type: "sound"              # sound, visual, physical, social
  source: "door"                   # What caused it
  description: "A door slammed in the next room."
  location: "adjacent_room"        # here, nearby, distant, adjacent_room
  intensity: 0.7                   # 0.0 to 1.0
```

**Implementation notes:**
- Event-driven, not periodic
- Other systems can publish to this (Brenda can trigger events)
- Noodlings can "perceive" events based on their attention/location

### 4. `#world.ambiance`

Broadcasts mood/atmosphere - the "vibe" of the scene.

```yaml
channel: "#world.ambiance"
payload:
  type: ambiance
  mood: "tense"                    # calm, tense, joyful, melancholy, mysterious, etc.
  energy: 0.6                      # 0.0 to 1.0
  description: "There's an undercurrent of anticipation in the air."
```

**Implementation notes:**
- Set by stage director (Brenda) or stage config
- Influences noodling affect/behavior
- Can change with story beats

---

## Architecture

### WorldChannelService

Create a service class that manages world channels:

```python
# runtime/world_channels.py

class WorldChannelService:
    """
    Manages world-level channels for environmental context.

    Owned by Stage/NoodleApp. Publishes to ChannelBus.
    """

    def __init__(self, channel_bus: ChannelBus, config: dict = None):
        self.channel_bus = channel_bus
        self.config = config or {}

        # Time state
        self.simulation_time = time.time()
        self.time_scale = config.get('time_scale', 1.0)  # 1.0 = real time

        # Weather state
        self.weather = {
            'temperature': 70,
            'conditions': 'clear',
            'wind': 'calm',
            'humidity': 0.5
        }

        # Ambiance state
        self.ambiance = {
            'mood': 'calm',
            'energy': 0.5
        }

        # Timer for periodic updates
        self._time_timer = None
        self._weather_timer = None

    def start(self):
        """Start publishing world channels."""
        self._publish_time()
        self._publish_weather()
        self._publish_ambiance()

        # Set up periodic time updates (every 10 simulation seconds)
        # Implementation depends on your timer system

    def stop(self):
        """Stop publishing."""
        pass

    def _publish_time(self):
        """Publish current time to #world.time."""
        time_of_day = self._calculate_time_of_day()
        sun_pos = self._calculate_sun_position()

        self.channel_bus.publish("#world.time", ChannelMessage(
            channel="#world.time",
            from_noodling="system",
            timestamp=time.time(),
            payload={
                'type': 'time_update',
                'simulation_time': self.simulation_time,
                'time_of_day': time_of_day,
                'formatted': self._format_time(),
                'sun_position': sun_pos,
                'description': self._describe_time()
            }
        ))

    def _publish_weather(self):
        """Publish weather to #world.weather."""
        self.channel_bus.publish("#world.weather", ChannelMessage(
            channel="#world.weather",
            from_noodling="system",
            timestamp=time.time(),
            payload={
                'type': 'weather_update',
                **self.weather,
                'description': self._describe_weather()
            }
        ))

    def _publish_ambiance(self):
        """Publish ambiance to #world.ambiance."""
        self.channel_bus.publish("#world.ambiance", ChannelMessage(
            channel="#world.ambiance",
            from_noodling="system",
            timestamp=time.time(),
            payload={
                'type': 'ambiance',
                **self.ambiance,
                'description': self._describe_ambiance()
            }
        ))

    def trigger_event(self, event_type: str, source: str,
                      description: str, **kwargs):
        """Publish a world event to #world.events."""
        self.channel_bus.publish("#world.events", ChannelMessage(
            channel="#world.events",
            from_noodling="system",
            timestamp=time.time(),
            payload={
                'type': 'event',
                'event_type': event_type,
                'source': source,
                'description': description,
                **kwargs
            }
        ))

    def set_weather(self, **kwargs):
        """Update weather state and publish."""
        self.weather.update(kwargs)
        self._publish_weather()

    def set_ambiance(self, mood: str = None, energy: float = None):
        """Update ambiance and publish."""
        if mood:
            self.ambiance['mood'] = mood
        if energy is not None:
            self.ambiance['energy'] = energy
        self._publish_ambiance()

    # Helper methods for time calculations
    def _calculate_time_of_day(self) -> str:
        # Convert simulation_time to time of day
        hour = (self.simulation_time // 3600) % 24
        if 5 <= hour < 8:
            return "dawn"
        elif 8 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 20:
            return "evening"
        else:
            return "night"

    def _calculate_sun_position(self) -> float:
        # Simplified: -1 at midnight, 1 at noon
        hour = (self.simulation_time // 3600) % 24
        return math.sin((hour - 6) * math.pi / 12)

    def _format_time(self) -> str:
        # Format for display
        pass

    def _describe_time(self) -> str:
        # Natural language description
        tod = self._calculate_time_of_day()
        descriptions = {
            'dawn': "The first light of dawn is breaking.",
            'morning': "Bright morning light fills the space.",
            'afternoon': "The afternoon sun streams in.",
            'evening': "The sun is setting, casting warm light.",
            'night': "Night has fallen."
        }
        return descriptions.get(tod, "")

    def _describe_weather(self) -> str:
        # Natural language weather description
        pass

    def _describe_ambiance(self) -> str:
        # Natural language ambiance description
        pass
```

### Stage Integration

Wire WorldChannelService into the stage/app:

```python
# In NoodleApp or Stage
class NoodleApp:
    def __init__(self, ...):
        self.channel_bus = ChannelBus()
        self.world_channels = WorldChannelService(
            self.channel_bus,
            config=self.stage_config.get('world', {})
        )

    def start(self):
        self.world_channels.start()

    def stop(self):
        self.world_channels.stop()
```

### Stage Config

Allow stages to configure world channels:

```yaml
# stage.yaml
name: "Tutorial Stage"

world:
  time_scale: 1.0              # Real time
  initial_time: "18:00"        # Start at 6 PM

  weather:
    temperature: 68
    conditions: partly_cloudy
    wind: light_breeze

  ambiance:
    mood: calm
    energy: 0.5
```

---

## Noodling Subscription

Noodlings can subscribe to world channels to perceive the environment:

```yaml
# guide/assembly.yaml
channels:
  subscribe:
    - "#directors.cues"
    - "#world.time"
    - "#world.ambiance"
  publish:
    - "#directors.feedback"
```

Then in facets:

```yaml
facets:
  - name: Perception
    type: LLM
    incoming:
      - user_input
      - channel:#world.time
      - channel:#world.ambiance
    prompt: |
      Current time: {{world_time.description}}
      Atmosphere: {{world_ambiance.description}}

      User said: {{user_input}}
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `runtime/world_channels.py` | CREATE - WorldChannelService |
| `runtime/app.py` | MODIFY - Wire in WorldChannelService |
| Stage config schema | MODIFY - Add world section |
| `tests/test_world_channels.py` | CREATE - Unit tests |

---

## Testing

1. Unit test WorldChannelService methods
2. Integration test: Start service, verify messages on bus
3. Test stage config loading
4. Test noodling receiving world channel messages

---

## Notes

- World channels are **read-only** for noodlings - only system publishes
- Brenda can call `world_channels.set_ambiance()` to change mood for story beats
- Events channel is for discrete happenings; time/weather/ambiance are continuous state

This is simpler than Brenda - more of a utility service. Should be quick to implement!

---

*"The world breathes. The noodlings listen."*
