# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
# Channel Architecture Tests
# ──────────────────────────────────────────────────────────────
"""
Tests for the channel architecture: ChannelBus, ChannelMessage, ChannelsConfig.
"""

import pytest
import time
from typing import List

from noodlestudio.runtime.channels import (
    ChannelBus,
    ChannelMessage,
    ChannelsConfig,
    get_global_channel_bus
)
from noodlestudio.core.facet_system import FacetAssembly


# =============================================================================
# ChannelMessage Tests
# =============================================================================

class TestChannelMessage:
    """Tests for ChannelMessage dataclass."""

    def test_create_message(self):
        """Test creating a message with factory method."""
        msg = ChannelMessage.create(
            channel="#test.channel",
            payload={"greeting": "Hello"},
            from_noodling="guide"
        )

        assert msg.channel == "#test.channel"
        assert msg.from_noodling == "guide"
        assert msg.payload == {"greeting": "Hello"}
        assert msg.timestamp > 0

    def test_message_serialization(self):
        """Test message to_dict and from_dict."""
        msg = ChannelMessage(
            channel="#directors.cues",
            from_noodling="brenda",
            timestamp=1234567890.0,
            payload={"type": "cue", "direction": "Look surprised"}
        )

        d = msg.to_dict()
        assert d['channel'] == "#directors.cues"
        assert d['from'] == "brenda"
        assert d['payload']['type'] == "cue"

        msg2 = ChannelMessage.from_dict(d)
        assert msg2.channel == msg.channel
        assert msg2.from_noodling == msg.from_noodling
        assert msg2.payload == msg.payload

    def test_default_from_noodling(self):
        """Test default from_noodling is 'system'."""
        msg = ChannelMessage.create("#world.weather", {"temp": 72})
        assert msg.from_noodling == "system"


# =============================================================================
# ChannelsConfig Tests
# =============================================================================

class TestChannelsConfig:
    """Tests for ChannelsConfig dataclass."""

    def test_empty_config(self):
        """Test empty channels config."""
        config = ChannelsConfig()
        assert config.subscribe == []
        assert config.publish == []

    def test_config_with_channels(self):
        """Test config with subscribe/publish channels."""
        config = ChannelsConfig(
            subscribe=["#directors.cues", "#world.context"],
            publish=["#directors.feedback"]
        )

        assert "#directors.cues" in config.subscribe
        assert "#world.context" in config.subscribe
        assert "#directors.feedback" in config.publish

    def test_config_serialization(self):
        """Test config to_dict and from_dict."""
        config = ChannelsConfig(
            subscribe=["#a", "#b"],
            publish=["#c"]
        )

        d = config.to_dict()
        assert d['subscribe'] == ["#a", "#b"]
        assert d['publish'] == ["#c"]

        config2 = ChannelsConfig.from_dict(d)
        assert config2.subscribe == config.subscribe
        assert config2.publish == config.publish

    def test_empty_config_to_dict(self):
        """Test empty config serializes to empty dict."""
        config = ChannelsConfig()
        d = config.to_dict()
        assert d == {}

    def test_from_dict_with_none(self):
        """Test from_dict with None returns empty config."""
        config = ChannelsConfig.from_dict(None)
        assert config.subscribe == []
        assert config.publish == []


# =============================================================================
# ChannelBus Tests
# =============================================================================

class TestChannelBus:
    """Tests for ChannelBus pub/sub functionality."""

    def test_subscribe_and_publish(self):
        """Test basic subscribe and publish."""
        bus = ChannelBus()
        received: List[ChannelMessage] = []

        def callback(msg: ChannelMessage):
            received.append(msg)

        bus.subscribe("#test", callback)
        bus.publish_simple("#test", {"value": 42}, "sender")

        assert len(received) == 1
        assert received[0].payload == {"value": 42}
        assert received[0].from_noodling == "sender"

    def test_multiple_subscribers(self):
        """Test multiple subscribers receive messages."""
        bus = ChannelBus()
        received1: List[ChannelMessage] = []
        received2: List[ChannelMessage] = []

        bus.subscribe("#channel", lambda m: received1.append(m))
        bus.subscribe("#channel", lambda m: received2.append(m))

        bus.publish_simple("#channel", {"data": "test"})

        assert len(received1) == 1
        assert len(received2) == 1
        assert received1[0].payload == received2[0].payload

    def test_unsubscribe(self):
        """Test unsubscribe stops receiving messages."""
        bus = ChannelBus()
        received: List[ChannelMessage] = []

        def callback(msg):
            received.append(msg)

        bus.subscribe("#test", callback)
        bus.publish_simple("#test", {"first": True})

        assert len(received) == 1

        bus.unsubscribe("#test", callback)
        bus.publish_simple("#test", {"second": True})

        assert len(received) == 1  # Still 1, didn't receive second

    def test_unsubscribe_all(self):
        """Test unsubscribe_all removes callback from all channels."""
        bus = ChannelBus()
        received: List[ChannelMessage] = []

        def callback(msg):
            received.append(msg)

        bus.subscribe("#channel1", callback)
        bus.subscribe("#channel2", callback)

        bus.unsubscribe_all(callback)

        bus.publish_simple("#channel1", {})
        bus.publish_simple("#channel2", {})

        assert len(received) == 0

    def test_get_latest(self):
        """Test get_latest returns most recent message."""
        bus = ChannelBus()

        bus.publish_simple("#weather", {"temp": 70})
        bus.publish_simple("#weather", {"temp": 72})
        bus.publish_simple("#weather", {"temp": 75})

        latest = bus.get_latest("#weather")
        assert latest is not None
        assert latest.payload == {"temp": 75}

    def test_get_latest_empty_channel(self):
        """Test get_latest returns None for empty channel."""
        bus = ChannelBus()
        assert bus.get_latest("#nonexistent") is None

    def test_get_history(self):
        """Test get_history returns messages in order."""
        bus = ChannelBus()

        for i in range(5):
            bus.publish_simple("#events", {"index": i})

        history = bus.get_history("#events")
        assert len(history) == 5
        assert history[0].payload == {"index": 0}
        assert history[4].payload == {"index": 4}

    def test_get_history_with_limit(self):
        """Test get_history respects limit."""
        bus = ChannelBus()

        for i in range(10):
            bus.publish_simple("#events", {"index": i})

        history = bus.get_history("#events", limit=3)
        assert len(history) == 3
        assert history[0].payload == {"index": 7}  # Last 3

    def test_history_limit(self):
        """Test history is trimmed when limit exceeded."""
        bus = ChannelBus(history_limit=5)

        for i in range(10):
            bus.publish_simple("#events", {"index": i})

        history = bus.get_history("#events")
        assert len(history) == 5
        assert history[0].payload == {"index": 5}

    def test_clear_history(self):
        """Test clear_history removes messages."""
        bus = ChannelBus()

        bus.publish_simple("#a", {"value": 1})
        bus.publish_simple("#b", {"value": 2})

        bus.clear_history("#a")
        assert bus.get_latest("#a") is None
        assert bus.get_latest("#b") is not None

        bus.clear_history()
        assert bus.get_latest("#b") is None

    def test_publish_returns_subscriber_count(self):
        """Test publish returns number of subscribers notified."""
        bus = ChannelBus()

        bus.subscribe("#test", lambda m: None)
        bus.subscribe("#test", lambda m: None)

        count = bus.publish_simple("#test", {})
        assert count == 2

    def test_no_subscribers(self):
        """Test publishing to channel with no subscribers."""
        bus = ChannelBus()
        count = bus.publish_simple("#empty", {"data": "test"})

        assert count == 0
        # Message should still be in history
        assert bus.get_latest("#empty") is not None

    def test_callback_error_doesnt_stop_others(self):
        """Test that error in one callback doesn't stop other callbacks."""
        bus = ChannelBus()
        received: List[ChannelMessage] = []

        def bad_callback(msg):
            raise ValueError("Intentional error")

        def good_callback(msg):
            received.append(msg)

        bus.subscribe("#test", bad_callback)
        bus.subscribe("#test", good_callback)

        bus.publish_simple("#test", {"value": 1})

        assert len(received) == 1  # Good callback still received

    def test_get_stats(self):
        """Test get_stats returns useful info."""
        bus = ChannelBus()

        bus.subscribe("#a", lambda m: None)
        bus.subscribe("#b", lambda m: None)
        bus.publish_simple("#a", {})

        stats = bus.get_stats()
        assert stats['channels_with_subscribers'] == 2
        assert stats['channels_with_messages'] == 1
        assert stats['total_subscribers'] == 2
        assert stats['total_messages'] == 1


# =============================================================================
# FacetAssembly Channel Integration Tests
# =============================================================================

class TestFacetAssemblyChannels:
    """Tests for FacetAssembly channel integration."""

    def test_assembly_with_no_channels(self):
        """Test assembly with no channels config."""
        assembly = FacetAssembly(name="No Channels")

        assert assembly.get_subscribe_channels() == []
        assert assembly.get_publish_channels() == []
        assert not assembly.can_publish_to("#any.channel")
        assert not assembly.subscribes_to("#any.channel")

    def test_assembly_with_channels_config(self):
        """Test assembly with ChannelsConfig."""
        config = ChannelsConfig(
            subscribe=["#directors.cues", "#world.weather"],
            publish=["#directors.feedback"]
        )
        assembly = FacetAssembly(name="Guide", channels=config)

        assert assembly.get_subscribe_channels() == ["#directors.cues", "#world.weather"]
        assert assembly.get_publish_channels() == ["#directors.feedback"]
        assert assembly.subscribes_to("#directors.cues")
        assert not assembly.subscribes_to("#other")
        assert assembly.can_publish_to("#directors.feedback")
        assert not assembly.can_publish_to("#directors.cues")

    def test_assembly_with_dict_channels(self):
        """Test assembly with raw dict channels (fallback mode)."""
        assembly = FacetAssembly(name="Dict Channels")
        assembly.channels = {
            'subscribe': ['#a', '#b'],
            'publish': ['#c']
        }

        assert assembly.get_subscribe_channels() == ['#a', '#b']
        assert assembly.get_publish_channels() == ['#c']

    def test_assembly_serialization_with_channels(self):
        """Test assembly serialization includes channels."""
        config = ChannelsConfig(
            subscribe=["#in"],
            publish=["#out"]
        )
        assembly = FacetAssembly(name="Test", channels=config)

        d = assembly.to_dict()
        assert 'channels' in d
        assert d['channels']['subscribe'] == ["#in"]
        assert d['channels']['publish'] == ["#out"]

    def test_assembly_deserialization_with_channels(self):
        """Test assembly deserialization loads channels."""
        data = {
            'name': 'Test Assembly',
            'channels': {
                'subscribe': ['#directors.cues'],
                'publish': ['#directors.feedback']
            },
            'facets': [],
            'connections': []
        }

        assembly = FacetAssembly.from_dict(data)
        assert assembly.subscribes_to("#directors.cues")
        assert assembly.can_publish_to("#directors.feedback")


# =============================================================================
# Global Bus Tests
# =============================================================================

class TestGlobalChannelBus:
    """Tests for global channel bus singleton."""

    def test_get_global_bus(self):
        """Test get_global_channel_bus returns same instance."""
        bus1 = get_global_channel_bus()
        bus2 = get_global_channel_bus()
        assert bus1 is bus2


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
