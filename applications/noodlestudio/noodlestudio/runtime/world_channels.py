# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   World Channels - Environmental Context Broadcasting
#
#   System-level channels that broadcast environmental context
#   to all noodlings on a stage: time, weather, events, ambiance.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.world_channels
# PURPOSE:  World Channels
# LAYER:    Studio / Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   WorldChannelService
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .channels import ChannelBus, ChannelMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Channel Names
# =============================================================================

CHANNEL_TIME = "#world.time"
CHANNEL_WEATHER = "#world.weather"
CHANNEL_EVENTS = "#world.events"
CHANNEL_AMBIANCE = "#world.ambiance"


# =============================================================================
# World Configuration
# =============================================================================

@dataclass
class WorldConfig:
    """
    Configuration for world channel service.

    Loaded from stage.yaml world section:
        world:
          time_scale: 1.0
          initial_time: "18:00"
          weather:
            temperature: 68
            conditions: partly_cloudy
          ambiance:
            mood: calm
            energy: 0.5
    """

    # Time settings
    time_scale: float = 1.0  # 1.0 = real time, 60.0 = 1 min per second
    initial_time: Optional[str] = None  # "HH:MM" or None for current time
    time_update_interval: float = 10.0  # Seconds between time broadcasts

    # Weather settings
    weather_temperature: float = 70.0  # Fahrenheit
    weather_conditions: str = "clear"  # clear, cloudy, partly_cloudy, rain, storm, snow, fog
    weather_wind: str = "calm"  # calm, light_breeze, windy, gusty
    weather_humidity: float = 0.5  # 0.0 to 1.0
    weather_update_interval: float = 60.0  # Seconds between weather broadcasts

    # Ambiance settings
    ambiance_mood: str = "calm"  # calm, tense, joyful, melancholy, mysterious, etc.
    ambiance_energy: float = 0.5  # 0.0 to 1.0

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'WorldConfig':
        """Load from stage world config dict."""
        config = WorldConfig()

        if not data:
            return config

        # Time settings
        config.time_scale = data.get('time_scale', config.time_scale)
        config.initial_time = data.get('initial_time', config.initial_time)
        config.time_update_interval = data.get('time_update_interval', config.time_update_interval)

        # Weather settings (nested or flat)
        weather = data.get('weather', {})
        if isinstance(weather, dict):
            config.weather_temperature = weather.get('temperature', config.weather_temperature)
            config.weather_conditions = weather.get('conditions', config.weather_conditions)
            config.weather_wind = weather.get('wind', config.weather_wind)
            config.weather_humidity = weather.get('humidity', config.weather_humidity)
        config.weather_update_interval = data.get('weather_update_interval', config.weather_update_interval)

        # Ambiance settings (nested or flat)
        ambiance = data.get('ambiance', {})
        if isinstance(ambiance, dict):
            config.ambiance_mood = ambiance.get('mood', config.ambiance_mood)
            config.ambiance_energy = ambiance.get('energy', config.ambiance_energy)

        return config


# =============================================================================
# World Channel Service
# =============================================================================

class WorldChannelService:
    """
    Manages world-level channels for environmental context.

    Owned by Stage/NoodleApp. Publishes to ChannelBus:
        #world.time     - Current simulation time
        #world.weather  - Weather/environmental conditions
        #world.events   - Discrete world events
        #world.ambiance - Mood/atmosphere

    Usage:
        bus = ChannelBus()
        world = WorldChannelService(bus, WorldConfig.from_dict(stage_config))
        world.start()

        # Later...
        world.set_weather(conditions="rain")
        world.set_ambiance(mood="tense")
        world.trigger_event("sound", "door", "A door slammed in the next room.")

        world.stop()
    """

    def __init__(
        self,
        channel_bus: ChannelBus,
        config: Optional[WorldConfig] = None
    ):
        """
        Initialize world channel service.

        Args:
            channel_bus: The channel bus to publish to
            config: World configuration (uses defaults if None)
        """
        self.channel_bus = channel_bus
        self.config = config or WorldConfig()

        # Time state
        self._simulation_start = time.time()
        self._simulation_time = self._parse_initial_time()
        self._last_time_broadcast = 0.0

        # Weather state
        self._weather = {
            'temperature': self.config.weather_temperature,
            'conditions': self.config.weather_conditions,
            'wind': self.config.weather_wind,
            'humidity': self.config.weather_humidity
        }
        self._last_weather_broadcast = 0.0

        # Ambiance state
        self._ambiance = {
            'mood': self.config.ambiance_mood,
            'energy': self.config.ambiance_energy
        }

        # Timer callback (set by start())
        self._timer_callback: Optional[Callable[[], None]] = None
        self._running = False

        logger.debug("WorldChannelService initialized")

    def _parse_initial_time(self) -> float:
        """
        Parse initial_time config to simulation timestamp.

        Returns:
            Simulation time in seconds since midnight
        """
        if self.config.initial_time:
            try:
                parts = self.config.initial_time.split(":")
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
                return (hours * 3600) + (minutes * 60)
            except (ValueError, IndexError):
                logger.warning(f"Invalid initial_time format: {self.config.initial_time}")

        # Default: use current time of day
        now = datetime.now()
        return (now.hour * 3600) + (now.minute * 60) + now.second

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """
        Start publishing world channels.

        Immediately publishes current state for all channels.
        """
        self._running = True
        self._simulation_start = time.time()

        # Publish initial state
        self._publish_time()
        self._publish_weather()
        self._publish_ambiance()

        logger.info("WorldChannelService started")

    def stop(self):
        """Stop publishing world channels."""
        self._running = False
        logger.info("WorldChannelService stopped")

    def tick(self):
        """
        Called periodically to check if broadcasts are needed.

        Should be called from app's main loop or timer.
        """
        if not self._running:
            return

        now = time.time()

        # Update simulation time
        elapsed_real = now - self._simulation_start
        self._simulation_time += elapsed_real * self.config.time_scale
        self._simulation_time %= 86400  # Wrap at midnight
        self._simulation_start = now

        # Check if time broadcast needed
        if now - self._last_time_broadcast >= self.config.time_update_interval:
            self._publish_time()
            self._last_time_broadcast = now

        # Check if weather broadcast needed (less frequent)
        if now - self._last_weather_broadcast >= self.config.weather_update_interval:
            self._publish_weather()
            self._last_weather_broadcast = now

    # =========================================================================
    # Time Channel
    # =========================================================================

    def _calculate_time_of_day(self) -> str:
        """Calculate time of day string from simulation time."""
        hour = int(self._simulation_time // 3600) % 24

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
        """
        Calculate sun position (-1 to 1).

        Returns:
            -1.0 at midnight, 1.0 at noon
        """
        hour = (self._simulation_time / 3600) % 24
        # Sin wave: 0 at 6am, 1 at noon, 0 at 6pm, -1 at midnight
        return math.sin((hour - 6) * math.pi / 12)

    def _format_time(self) -> str:
        """Format simulation time for display (e.g., '6:45 PM')."""
        hours = int(self._simulation_time // 3600) % 24
        minutes = int((self._simulation_time % 3600) // 60)

        if hours == 0:
            return f"12:{minutes:02d} AM"
        elif hours < 12:
            return f"{hours}:{minutes:02d} AM"
        elif hours == 12:
            return f"12:{minutes:02d} PM"
        else:
            return f"{hours - 12}:{minutes:02d} PM"

    def _describe_time(self) -> str:
        """Generate natural language time description."""
        tod = self._calculate_time_of_day()
        sun_pos = self._calculate_sun_position()

        descriptions = {
            'dawn': "The first light of dawn is breaking over the horizon.",
            'morning': "Bright morning light fills the space.",
            'afternoon': "The afternoon sun streams in warmly.",
            'evening': "The sun is setting, casting long golden shadows.",
            'night': "Night has fallen, stars beginning to appear."
        }

        base = descriptions.get(tod, "")

        # Add sun position detail
        if sun_pos > 0.9:
            base = "The sun is high overhead. " + base
        elif sun_pos < -0.8:
            base = "Darkness blankets everything. " + base

        return base

    def _publish_time(self):
        """Publish current time to #world.time."""
        payload = {
            'type': 'time_update',
            'simulation_time': self._simulation_time,
            'time_of_day': self._calculate_time_of_day(),
            'formatted': self._format_time(),
            'sun_position': round(self._calculate_sun_position(), 2),
            'description': self._describe_time()
        }

        self.channel_bus.publish(
            CHANNEL_TIME,
            ChannelMessage.create(CHANNEL_TIME, payload, "system")
        )

        logger.debug(f"Published time: {payload['formatted']} ({payload['time_of_day']})")

    def get_time(self) -> Dict[str, Any]:
        """Get current time state."""
        return {
            'simulation_time': self._simulation_time,
            'time_of_day': self._calculate_time_of_day(),
            'formatted': self._format_time(),
            'sun_position': round(self._calculate_sun_position(), 2),
            'description': self._describe_time()
        }

    def set_time(self, time_str: str):
        """
        Set simulation time.

        Args:
            time_str: Time in "HH:MM" format
        """
        try:
            parts = time_str.split(":")
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            self._simulation_time = (hours * 3600) + (minutes * 60)
            self._simulation_start = time.time()
            self._publish_time()
        except (ValueError, IndexError):
            logger.error(f"Invalid time format: {time_str}")

    # =========================================================================
    # Weather Channel
    # =========================================================================

    def _describe_weather(self) -> str:
        """Generate natural language weather description."""
        temp = self._weather['temperature']
        conditions = self._weather['conditions']
        wind = self._weather['wind']

        # Temperature description
        if temp < 32:
            temp_desc = "freezing"
        elif temp < 50:
            temp_desc = "cold"
        elif temp < 65:
            temp_desc = "cool"
        elif temp < 80:
            temp_desc = "pleasant"
        elif temp < 90:
            temp_desc = "warm"
        else:
            temp_desc = "hot"

        # Conditions description
        conditions_map = {
            'clear': "clear skies",
            'cloudy': "overcast skies",
            'partly_cloudy': "scattered clouds",
            'rain': "rain falling steadily",
            'storm': "a thunderstorm raging",
            'snow': "snow drifting down",
            'fog': "thick fog obscuring visibility"
        }
        cond_desc = conditions_map.get(conditions, conditions)

        # Wind description
        wind_map = {
            'calm': "",
            'light_breeze': "A light breeze stirs the air.",
            'windy': "Wind gusts occasionally.",
            'gusty': "Strong gusts of wind sweep through."
        }
        wind_desc = wind_map.get(wind, "")

        base = f"A {temp_desc} day with {cond_desc}."
        if wind_desc:
            base += " " + wind_desc

        return base

    def _publish_weather(self):
        """Publish weather to #world.weather."""
        payload = {
            'type': 'weather_update',
            **self._weather,
            'description': self._describe_weather()
        }

        self.channel_bus.publish(
            CHANNEL_WEATHER,
            ChannelMessage.create(CHANNEL_WEATHER, payload, "system")
        )

        logger.debug(f"Published weather: {payload['conditions']}, {payload['temperature']}F")

    def get_weather(self) -> Dict[str, Any]:
        """Get current weather state."""
        return {
            **self._weather,
            'description': self._describe_weather()
        }

    def set_weather(
        self,
        temperature: Optional[float] = None,
        conditions: Optional[str] = None,
        wind: Optional[str] = None,
        humidity: Optional[float] = None
    ):
        """
        Update weather state and publish.

        Args:
            temperature: Temperature in Fahrenheit
            conditions: clear, cloudy, partly_cloudy, rain, storm, snow, fog
            wind: calm, light_breeze, windy, gusty
            humidity: 0.0 to 1.0
        """
        if temperature is not None:
            self._weather['temperature'] = temperature
        if conditions is not None:
            self._weather['conditions'] = conditions
        if wind is not None:
            self._weather['wind'] = wind
        if humidity is not None:
            self._weather['humidity'] = max(0.0, min(1.0, humidity))

        self._publish_weather()

    # =========================================================================
    # Ambiance Channel
    # =========================================================================

    def _describe_ambiance(self) -> str:
        """Generate natural language ambiance description."""
        mood = self._ambiance['mood']
        energy = self._ambiance['energy']

        mood_descriptions = {
            'calm': "A sense of peace pervades the space.",
            'tense': "There's an undercurrent of tension in the air.",
            'joyful': "A feeling of joy and celebration fills the atmosphere.",
            'melancholy': "A wistful, contemplative mood hangs in the air.",
            'mysterious': "An air of mystery and intrigue permeates everything.",
            'anxious': "A nervous energy crackles through the space.",
            'hopeful': "A sense of hope and possibility lingers.",
            'ominous': "A foreboding presence seems to lurk nearby."
        }

        base = mood_descriptions.get(mood, f"The mood feels {mood}.")

        # Energy modifier
        if energy > 0.8:
            base = "The atmosphere is charged with intensity. " + base
        elif energy < 0.2:
            base = "Everything feels subdued and quiet. " + base

        return base

    def _publish_ambiance(self):
        """Publish ambiance to #world.ambiance."""
        payload = {
            'type': 'ambiance',
            **self._ambiance,
            'description': self._describe_ambiance()
        }

        self.channel_bus.publish(
            CHANNEL_AMBIANCE,
            ChannelMessage.create(CHANNEL_AMBIANCE, payload, "system")
        )

        logger.debug(f"Published ambiance: {payload['mood']}, energy={payload['energy']}")

    def get_ambiance(self) -> Dict[str, Any]:
        """Get current ambiance state."""
        return {
            **self._ambiance,
            'description': self._describe_ambiance()
        }

    def set_ambiance(
        self,
        mood: Optional[str] = None,
        energy: Optional[float] = None
    ):
        """
        Update ambiance and publish.

        Args:
            mood: calm, tense, joyful, melancholy, mysterious, etc.
            energy: 0.0 to 1.0
        """
        if mood is not None:
            self._ambiance['mood'] = mood
        if energy is not None:
            self._ambiance['energy'] = max(0.0, min(1.0, energy))

        self._publish_ambiance()

    # =========================================================================
    # Events Channel
    # =========================================================================

    def trigger_event(
        self,
        event_type: str,
        source: str,
        description: str,
        location: str = "here",
        intensity: float = 0.5,
        **kwargs
    ):
        """
        Publish a world event to #world.events.

        Args:
            event_type: sound, visual, physical, social
            source: What caused the event (e.g., "door", "explosion")
            description: Human-readable description
            location: here, nearby, distant, adjacent_room
            intensity: 0.0 to 1.0
            **kwargs: Additional event-specific data
        """
        payload = {
            'type': 'event',
            'event_type': event_type,
            'source': source,
            'description': description,
            'location': location,
            'intensity': max(0.0, min(1.0, intensity)),
            **kwargs
        }

        self.channel_bus.publish(
            CHANNEL_EVENTS,
            ChannelMessage.create(CHANNEL_EVENTS, payload, "system")
        )

        logger.debug(f"Published event: {event_type} from {source} - {description}")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_full_context(self) -> Dict[str, Any]:
        """
        Get complete world context for noodling perception.

        Returns dict with time, weather, and ambiance suitable for
        injection into facet context.
        """
        return {
            'time': self.get_time(),
            'weather': self.get_weather(),
            'ambiance': self.get_ambiance()
        }


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
