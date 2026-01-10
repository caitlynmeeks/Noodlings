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
#   Channel Architecture
#
#   Named message buses for inter-noodling communication.
#   Enables pub/sub patterns beyond direct speech: stage direction,
#   environmental context, group communication, private messaging.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.channels
# PURPOSE:  Channel Architecture
# LAYER:    Studio / Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ChannelMessage, ChannelBus, ChannelsConfig
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Channel Message
# =============================================================================

@dataclass
class ChannelMessage:
    """
    A message published to a channel.

    Attributes:
        channel: The channel name (e.g., "#directors.cues")
        from_noodling: Sender ID or "system"
        timestamp: Unix timestamp when published
        payload: The message content (dict)
    """
    channel: str
    from_noodling: str
    timestamp: float
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'channel': self.channel,
            'from': self.from_noodling,
            'timestamp': self.timestamp,
            'payload': self.payload
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ChannelMessage':
        """Deserialize from dictionary."""
        return ChannelMessage(
            channel=data.get('channel', ''),
            from_noodling=data.get('from', 'system'),
            timestamp=data.get('timestamp', 0.0),
            payload=data.get('payload', {})
        )

    @staticmethod
    def create(
        channel: str,
        payload: Dict[str, Any],
        from_noodling: str = "system"
    ) -> 'ChannelMessage':
        """Factory method to create a new message with current timestamp."""
        return ChannelMessage(
            channel=channel,
            from_noodling=from_noodling,
            timestamp=time.time(),
            payload=payload
        )


# =============================================================================
# Channels Configuration (for assembly.yaml)
# =============================================================================

@dataclass
class ChannelsConfig:
    """
    Channel subscription/publish configuration for an assembly.

    Loaded from assembly.yaml:
        channels:
          subscribe:
            - "#directors.cues"
            - "#world.context"
          publish:
            - "#directors.feedback"
    """
    subscribe: List[str] = field(default_factory=list)
    publish: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        if self.subscribe:
            result['subscribe'] = self.subscribe
        if self.publish:
            result['publish'] = self.publish
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ChannelsConfig':
        """Deserialize from dictionary."""
        if not data:
            return ChannelsConfig()
        return ChannelsConfig(
            subscribe=data.get('subscribe', []),
            publish=data.get('publish', [])
        )


# =============================================================================
# Channel Callback Type
# =============================================================================

# Callback signature: (message: ChannelMessage) -> None
ChannelCallback = Callable[[ChannelMessage], None]


# =============================================================================
# Channel Bus
# =============================================================================

class ChannelBus:
    """
    Named message bus for inter-noodling communication.

    Channels follow a naming convention:
        #world.*       - Public environmental (weather, time, events)
        #directors.*   - Stage management (cues, feedback)
        #dm.*          - Direct messages (private)
        #<scope>.*     - Scoped group channels

    Usage:
        bus = ChannelBus()

        # Subscribe to a channel
        def on_cue(msg: ChannelMessage):
            print(f"Got cue: {msg.payload}")

        bus.subscribe("#directors.cues", on_cue)

        # Publish to a channel
        bus.publish("#directors.cues", ChannelMessage.create(
            channel="#directors.cues",
            from_noodling="brenda",
            payload={"type": "cue", "direction": "Walk through the menu"}
        ))

        # Get latest message on a channel
        latest = bus.get_latest("#world.weather")
    """

    def __init__(self, history_limit: int = 100):
        """
        Initialize the channel bus.

        Args:
            history_limit: Maximum messages to retain per channel
        """
        self._subscribers: Dict[str, List[ChannelCallback]] = {}
        self._history: Dict[str, List[ChannelMessage]] = {}
        self._history_limit = history_limit

        logger.debug("ChannelBus initialized")

    def subscribe(self, channel: str, callback: ChannelCallback) -> None:
        """
        Subscribe to a channel.

        Args:
            channel: Channel name (e.g., "#directors.cues")
            callback: Function to call when message arrives
        """
        if channel not in self._subscribers:
            self._subscribers[channel] = []

        if callback not in self._subscribers[channel]:
            self._subscribers[channel].append(callback)
            logger.debug(f"Subscribed to channel: {channel}")

    def unsubscribe(self, channel: str, callback: ChannelCallback) -> None:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel name
            callback: The callback to remove
        """
        if channel in self._subscribers:
            try:
                self._subscribers[channel].remove(callback)
                logger.debug(f"Unsubscribed from channel: {channel}")
            except ValueError:
                pass  # Callback wasn't subscribed

    def unsubscribe_all(self, callback: ChannelCallback) -> None:
        """
        Unsubscribe a callback from all channels.

        Args:
            callback: The callback to remove from all channels
        """
        for channel in self._subscribers:
            try:
                self._subscribers[channel].remove(callback)
            except ValueError:
                pass

    def publish(self, channel: str, message: ChannelMessage) -> int:
        """
        Publish a message to all subscribers.

        Args:
            channel: Channel name (uses message.channel if not matching)
            message: The message to publish

        Returns:
            Number of subscribers notified
        """
        # Ensure channel matches
        if message.channel != channel:
            message = ChannelMessage(
                channel=channel,
                from_noodling=message.from_noodling,
                timestamp=message.timestamp,
                payload=message.payload
            )

        # Store in history
        if channel not in self._history:
            self._history[channel] = []
        self._history[channel].append(message)

        # Trim history if needed
        if len(self._history[channel]) > self._history_limit:
            self._history[channel] = self._history[channel][-self._history_limit:]

        # Notify subscribers
        subscribers = self._subscribers.get(channel, [])
        notified = 0

        for callback in subscribers:
            try:
                callback(message)
                notified += 1
            except Exception as e:
                logger.error(f"Error in channel callback for {channel}: {e}")

        logger.debug(
            f"Published to {channel}: {len(message.payload)} payload keys, "
            f"{notified} subscribers notified"
        )

        return notified

    def publish_simple(
        self,
        channel: str,
        payload: Dict[str, Any],
        from_noodling: str = "system"
    ) -> int:
        """
        Convenience method to publish a message with auto-generated timestamp.

        Args:
            channel: Channel name
            payload: Message payload
            from_noodling: Sender ID

        Returns:
            Number of subscribers notified
        """
        message = ChannelMessage.create(channel, payload, from_noodling)
        return self.publish(channel, message)

    def get_latest(self, channel: str) -> Optional[ChannelMessage]:
        """
        Get the most recent message on a channel.

        Args:
            channel: Channel name

        Returns:
            Most recent message, or None if no messages
        """
        history = self._history.get(channel, [])
        return history[-1] if history else None

    def get_history(
        self,
        channel: str,
        limit: Optional[int] = None
    ) -> List[ChannelMessage]:
        """
        Get message history for a channel.

        Args:
            channel: Channel name
            limit: Maximum messages to return (None = all)

        Returns:
            List of messages, oldest first
        """
        history = self._history.get(channel, [])
        if limit:
            return history[-limit:]
        return list(history)

    def get_channels_with_subscribers(self) -> List[str]:
        """Get list of channels that have at least one subscriber."""
        return [ch for ch, subs in self._subscribers.items() if subs]

    def get_channels_with_messages(self) -> List[str]:
        """Get list of channels that have at least one message."""
        return [ch for ch, msgs in self._history.items() if msgs]

    def get_subscriber_count(self, channel: str) -> int:
        """Get number of subscribers for a channel."""
        return len(self._subscribers.get(channel, []))

    def clear_history(self, channel: Optional[str] = None) -> None:
        """
        Clear message history.

        Args:
            channel: Specific channel to clear, or None for all
        """
        if channel:
            self._history[channel] = []
        else:
            self._history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the bus."""
        return {
            'channels_with_subscribers': len(self.get_channels_with_subscribers()),
            'channels_with_messages': len(self.get_channels_with_messages()),
            'total_subscribers': sum(
                len(subs) for subs in self._subscribers.values()
            ),
            'total_messages': sum(
                len(msgs) for msgs in self._history.values()
            ),
        }


# =============================================================================
# Singleton Instance (optional - can use per-stage instances instead)
# =============================================================================

_global_bus: Optional[ChannelBus] = None


def get_global_channel_bus() -> ChannelBus:
    """
    Get the global ChannelBus instance.

    For most use cases, prefer stage-owned ChannelBus instances.
    This global is useful for cross-stage communication.
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = ChannelBus()
    return _global_bus


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
