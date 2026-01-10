# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
# World Channels Tests
# ──────────────────────────────────────────────────────────────
"""
Tests for WorldChannelService: #world.time, #world.weather, #world.events, #world.ambiance.
"""

import pytest
import time
from typing import List

from noodlestudio.runtime.channels import ChannelBus, ChannelMessage
from noodlestudio.runtime.world_channels import (
    WorldChannelService,
    WorldConfig,
    CHANNEL_TIME,
    CHANNEL_WEATHER,
    CHANNEL_EVENTS,
    CHANNEL_AMBIANCE
)


# =============================================================================
# WorldConfig Tests
# =============================================================================

class TestWorldConfig:
    """Tests for WorldConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorldConfig()

        assert config.time_scale == 1.0
        assert config.initial_time is None
        assert config.weather_temperature == 70.0
        assert config.weather_conditions == "clear"
        assert config.weather_wind == "calm"
        assert config.weather_humidity == 0.5
        assert config.ambiance_mood == "calm"
        assert config.ambiance_energy == 0.5

    def test_from_dict_empty(self):
        """Test from_dict with empty/None data."""
        config = WorldConfig.from_dict({})
        assert config.time_scale == 1.0

        config = WorldConfig.from_dict(None)
        assert config.time_scale == 1.0

    def test_from_dict_with_values(self):
        """Test from_dict with stage config."""
        data = {
            'time_scale': 60.0,
            'initial_time': '18:00',
            'weather': {
                'temperature': 68,
                'conditions': 'partly_cloudy',
                'wind': 'light_breeze',
                'humidity': 0.45
            },
            'ambiance': {
                'mood': 'tense',
                'energy': 0.8
            }
        }

        config = WorldConfig.from_dict(data)
        assert config.time_scale == 60.0
        assert config.initial_time == '18:00'
        assert config.weather_temperature == 68
        assert config.weather_conditions == 'partly_cloudy'
        assert config.weather_wind == 'light_breeze'
        assert config.weather_humidity == 0.45
        assert config.ambiance_mood == 'tense'
        assert config.ambiance_energy == 0.8


# =============================================================================
# WorldChannelService Basic Tests
# =============================================================================

class TestWorldChannelService:
    """Tests for WorldChannelService core functionality."""

    def test_create_service(self):
        """Test creating a service."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        assert service.channel_bus is bus
        assert service.config is not None

    def test_create_service_with_config(self):
        """Test creating service with custom config."""
        bus = ChannelBus()
        config = WorldConfig(time_scale=2.0)
        service = WorldChannelService(bus, config)

        assert service.config.time_scale == 2.0

    def test_start_publishes_initial_state(self):
        """Test start() publishes initial messages to all channels."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        # All channels should have initial messages
        assert bus.get_latest(CHANNEL_TIME) is not None
        assert bus.get_latest(CHANNEL_WEATHER) is not None
        assert bus.get_latest(CHANNEL_AMBIANCE) is not None

    def test_stop(self):
        """Test stop() marks service as not running."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        service.start()
        assert service._running

        service.stop()
        assert not service._running


# =============================================================================
# Time Channel Tests
# =============================================================================

class TestTimeChannel:
    """Tests for #world.time channel."""

    def test_time_message_structure(self):
        """Test time message has correct structure."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        msg = bus.get_latest(CHANNEL_TIME)
        payload = msg.payload

        assert 'type' in payload
        assert payload['type'] == 'time_update'
        assert 'simulation_time' in payload
        assert 'time_of_day' in payload
        assert 'formatted' in payload
        assert 'sun_position' in payload
        assert 'description' in payload

    def test_time_of_day_dawn(self):
        """Test time_of_day returns 'dawn' for 5-8 AM."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="06:00")
        service = WorldChannelService(bus, config)

        assert service._calculate_time_of_day() == "dawn"

    def test_time_of_day_morning(self):
        """Test time_of_day returns 'morning' for 8-12."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="10:00")
        service = WorldChannelService(bus, config)

        assert service._calculate_time_of_day() == "morning"

    def test_time_of_day_afternoon(self):
        """Test time_of_day returns 'afternoon' for 12-17."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="14:00")
        service = WorldChannelService(bus, config)

        assert service._calculate_time_of_day() == "afternoon"

    def test_time_of_day_evening(self):
        """Test time_of_day returns 'evening' for 17-20."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="18:00")
        service = WorldChannelService(bus, config)

        assert service._calculate_time_of_day() == "evening"

    def test_time_of_day_night(self):
        """Test time_of_day returns 'night' for 20-5."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="23:00")
        service = WorldChannelService(bus, config)

        assert service._calculate_time_of_day() == "night"

    def test_sun_position_noon(self):
        """Test sun_position is ~1 at noon."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="12:00")
        service = WorldChannelService(bus, config)

        sun_pos = service._calculate_sun_position()
        assert sun_pos > 0.99  # Should be ~1.0

    def test_sun_position_midnight(self):
        """Test sun_position is ~-1 at midnight."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="00:00")
        service = WorldChannelService(bus, config)

        sun_pos = service._calculate_sun_position()
        assert sun_pos < -0.99  # Should be ~-1.0

    def test_format_time_am(self):
        """Test time formatting for AM."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="09:30")
        service = WorldChannelService(bus, config)

        formatted = service._format_time()
        assert formatted == "9:30 AM"

    def test_format_time_pm(self):
        """Test time formatting for PM."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="18:45")
        service = WorldChannelService(bus, config)

        formatted = service._format_time()
        assert formatted == "6:45 PM"

    def test_format_time_noon(self):
        """Test time formatting for noon."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="12:00")
        service = WorldChannelService(bus, config)

        formatted = service._format_time()
        assert formatted == "12:00 PM"

    def test_format_time_midnight(self):
        """Test time formatting for midnight."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="00:00")
        service = WorldChannelService(bus, config)

        formatted = service._format_time()
        assert formatted == "12:00 AM"

    def test_set_time(self):
        """Test set_time updates simulation time."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        service.set_time("15:30")
        msg = bus.get_latest(CHANNEL_TIME)

        assert msg.payload['formatted'] == "3:30 PM"
        assert msg.payload['time_of_day'] == "afternoon"

    def test_get_time(self):
        """Test get_time returns current time state."""
        bus = ChannelBus()
        config = WorldConfig(initial_time="09:00")
        service = WorldChannelService(bus, config)

        time_state = service.get_time()
        assert time_state['formatted'] == "9:00 AM"
        assert time_state['time_of_day'] == "morning"


# =============================================================================
# Weather Channel Tests
# =============================================================================

class TestWeatherChannel:
    """Tests for #world.weather channel."""

    def test_weather_message_structure(self):
        """Test weather message has correct structure."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        msg = bus.get_latest(CHANNEL_WEATHER)
        payload = msg.payload

        assert payload['type'] == 'weather_update'
        assert 'temperature' in payload
        assert 'conditions' in payload
        assert 'wind' in payload
        assert 'humidity' in payload
        assert 'description' in payload

    def test_set_weather_temperature(self):
        """Test set_weather updates temperature."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        service.set_weather(temperature=85)
        msg = bus.get_latest(CHANNEL_WEATHER)

        assert msg.payload['temperature'] == 85
        assert "warm" in msg.payload['description'].lower()

    def test_set_weather_conditions(self):
        """Test set_weather updates conditions."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        service.set_weather(conditions="storm")
        msg = bus.get_latest(CHANNEL_WEATHER)

        assert msg.payload['conditions'] == "storm"
        assert "storm" in msg.payload['description'].lower()

    def test_set_weather_wind(self):
        """Test set_weather updates wind."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        service.set_weather(wind="gusty")
        msg = bus.get_latest(CHANNEL_WEATHER)

        assert msg.payload['wind'] == "gusty"

    def test_set_weather_humidity_clamped(self):
        """Test humidity is clamped to 0-1."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        service.set_weather(humidity=1.5)
        msg = bus.get_latest(CHANNEL_WEATHER)
        assert msg.payload['humidity'] == 1.0

        service.set_weather(humidity=-0.5)
        msg = bus.get_latest(CHANNEL_WEATHER)
        assert msg.payload['humidity'] == 0.0

    def test_get_weather(self):
        """Test get_weather returns current weather state."""
        bus = ChannelBus()
        config = WorldConfig(
            weather_temperature=55,
            weather_conditions="rain"
        )
        service = WorldChannelService(bus, config)

        weather = service.get_weather()
        assert weather['temperature'] == 55
        assert weather['conditions'] == "rain"
        assert 'description' in weather


# =============================================================================
# Ambiance Channel Tests
# =============================================================================

class TestAmbianceChannel:
    """Tests for #world.ambiance channel."""

    def test_ambiance_message_structure(self):
        """Test ambiance message has correct structure."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        msg = bus.get_latest(CHANNEL_AMBIANCE)
        payload = msg.payload

        assert payload['type'] == 'ambiance'
        assert 'mood' in payload
        assert 'energy' in payload
        assert 'description' in payload

    def test_set_ambiance_mood(self):
        """Test set_ambiance updates mood."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        service.set_ambiance(mood="mysterious")
        msg = bus.get_latest(CHANNEL_AMBIANCE)

        assert msg.payload['mood'] == "mysterious"
        assert "mystery" in msg.payload['description'].lower() or "intrigue" in msg.payload['description'].lower()

    def test_set_ambiance_energy(self):
        """Test set_ambiance updates energy."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        service.set_ambiance(energy=0.9)
        msg = bus.get_latest(CHANNEL_AMBIANCE)

        assert msg.payload['energy'] == 0.9
        assert "intensity" in msg.payload['description'].lower() or "charged" in msg.payload['description'].lower()

    def test_set_ambiance_energy_clamped(self):
        """Test energy is clamped to 0-1."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        service.set_ambiance(energy=1.5)
        msg = bus.get_latest(CHANNEL_AMBIANCE)
        assert msg.payload['energy'] == 1.0

        service.set_ambiance(energy=-0.5)
        msg = bus.get_latest(CHANNEL_AMBIANCE)
        assert msg.payload['energy'] == 0.0

    def test_get_ambiance(self):
        """Test get_ambiance returns current ambiance state."""
        bus = ChannelBus()
        config = WorldConfig(ambiance_mood="joyful", ambiance_energy=0.7)
        service = WorldChannelService(bus, config)

        ambiance = service.get_ambiance()
        assert ambiance['mood'] == "joyful"
        assert ambiance['energy'] == 0.7


# =============================================================================
# Events Channel Tests
# =============================================================================

class TestEventsChannel:
    """Tests for #world.events channel."""

    def test_trigger_event(self):
        """Test trigger_event publishes to events channel."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        service.trigger_event(
            event_type="sound",
            source="door",
            description="A door slammed in the next room."
        )

        msg = bus.get_latest(CHANNEL_EVENTS)
        payload = msg.payload

        assert payload['type'] == 'event'
        assert payload['event_type'] == "sound"
        assert payload['source'] == "door"
        assert payload['description'] == "A door slammed in the next room."

    def test_trigger_event_with_location(self):
        """Test trigger_event with location parameter."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        service.trigger_event(
            event_type="visual",
            source="light",
            description="A bright light flashed.",
            location="nearby"
        )

        msg = bus.get_latest(CHANNEL_EVENTS)
        assert msg.payload['location'] == "nearby"

    def test_trigger_event_with_intensity(self):
        """Test trigger_event with intensity parameter."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        service.trigger_event(
            event_type="physical",
            source="earthquake",
            description="The ground shook.",
            intensity=0.8
        )

        msg = bus.get_latest(CHANNEL_EVENTS)
        assert msg.payload['intensity'] == 0.8

    def test_trigger_event_intensity_clamped(self):
        """Test intensity is clamped to 0-1."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        service.trigger_event(
            event_type="sound",
            source="explosion",
            description="BOOM!",
            intensity=2.0
        )

        msg = bus.get_latest(CHANNEL_EVENTS)
        assert msg.payload['intensity'] == 1.0

    def test_trigger_event_with_extra_data(self):
        """Test trigger_event with additional kwargs."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        service.trigger_event(
            event_type="social",
            source="npc",
            description="Someone entered the room.",
            character_name="Alice",
            entering=True
        )

        msg = bus.get_latest(CHANNEL_EVENTS)
        assert msg.payload['character_name'] == "Alice"
        assert msg.payload['entering'] is True


# =============================================================================
# Full Context Tests
# =============================================================================

class TestFullContext:
    """Tests for get_full_context method."""

    def test_get_full_context(self):
        """Test get_full_context returns all world state."""
        bus = ChannelBus()
        config = WorldConfig(
            initial_time="14:00",
            weather_temperature=75,
            weather_conditions="partly_cloudy",
            ambiance_mood="calm",
            ambiance_energy=0.5
        )
        service = WorldChannelService(bus, config)

        context = service.get_full_context()

        # Time
        assert 'time' in context
        assert context['time']['time_of_day'] == "afternoon"

        # Weather
        assert 'weather' in context
        assert context['weather']['temperature'] == 75

        # Ambiance
        assert 'ambiance' in context
        assert context['ambiance']['mood'] == "calm"


# =============================================================================
# Tick Tests
# =============================================================================

class TestTick:
    """Tests for tick() periodic updates."""

    def test_tick_advances_time(self):
        """Test tick() advances simulation time."""
        bus = ChannelBus()
        config = WorldConfig(
            initial_time="12:00",
            time_scale=60.0,  # 1 real second = 1 simulated minute
            time_update_interval=0.01  # Trigger broadcast quickly
        )
        service = WorldChannelService(bus, config)
        service.start()

        initial_time = service._simulation_time

        # Wait a bit and tick
        time.sleep(0.05)
        service.tick()

        # Time should have advanced significantly
        assert service._simulation_time > initial_time

    def test_tick_when_not_running(self):
        """Test tick() does nothing when not running."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        # Don't start the service
        initial_time = service._simulation_time
        service.tick()

        # Nothing should change
        assert service._simulation_time == initial_time


# =============================================================================
# Subscriber Integration Tests
# =============================================================================

class TestSubscriberIntegration:
    """Tests for subscribers receiving world channel updates."""

    def test_subscriber_receives_time_updates(self):
        """Test subscriber receives time channel messages."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        received: List[ChannelMessage] = []
        bus.subscribe(CHANNEL_TIME, lambda m: received.append(m))

        service.start()

        assert len(received) == 1
        assert received[0].payload['type'] == 'time_update'

    def test_subscriber_receives_weather_updates(self):
        """Test subscriber receives weather channel messages."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        received: List[ChannelMessage] = []
        bus.subscribe(CHANNEL_WEATHER, lambda m: received.append(m))

        service.start()

        assert len(received) == 1
        assert received[0].payload['type'] == 'weather_update'

        # Now update weather
        service.set_weather(temperature=50)

        assert len(received) == 2
        assert received[1].payload['temperature'] == 50

    def test_subscriber_receives_events(self):
        """Test subscriber receives event channel messages."""
        bus = ChannelBus()
        service = WorldChannelService(bus)

        received: List[ChannelMessage] = []
        bus.subscribe(CHANNEL_EVENTS, lambda m: received.append(m))

        service.trigger_event("sound", "bell", "A bell rang.")

        assert len(received) == 1
        assert received[0].payload['source'] == "bell"

    def test_message_from_noodling_is_system(self):
        """Test all world channel messages are from 'system'."""
        bus = ChannelBus()
        service = WorldChannelService(bus)
        service.start()

        assert bus.get_latest(CHANNEL_TIME).from_noodling == "system"
        assert bus.get_latest(CHANNEL_WEATHER).from_noodling == "system"
        assert bus.get_latest(CHANNEL_AMBIANCE).from_noodling == "system"

        service.trigger_event("test", "test", "test")
        assert bus.get_latest(CHANNEL_EVENTS).from_noodling == "system"


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
