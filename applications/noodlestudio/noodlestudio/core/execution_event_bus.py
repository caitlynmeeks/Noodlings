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
#   Execution Event Bus - Central nervous system for facet execution events
#
#   Provides unified event distribution for: - FacetExecutor ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.execution_event_bus
# PURPOSE:  Execution Event Bus
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   EventChannel, EventPriority, Event, EventListener, ExecutionEventBus
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import time
import logging
from typing import Dict, Any, List, Callable, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class EventChannel(Enum):
    """Event channels for routing and filtering."""
    EXECUTION = "execution"      # Facet execution events
    WORLD = "world"             # World state changes (agents, objects, rooms)
    COGNITION = "cognition"     # Agent cognition events
    SCRIPT = "script"           # Script-generated events
    VISUALIZATION = "visualization"  # Visualization hints


class EventPriority(Enum):
    """Event priority for ordering."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """
    Universal event structure.

    All events flow through the bus with this structure, regardless of source.
    """
    # Core identification
    type: str  # Event type (e.g., "facet_execution", "world_change")
    subtype: str  # Event subtype (e.g., "facet_start", "agent_speak")

    # Timing
    timestamp: float
    cycle: Optional[int] = None

    # Source
    source_id: Optional[str] = None  # Facet ID, agent ID, etc.
    source_name: Optional[str] = None

    # Channel and priority
    channel: EventChannel = EventChannel.EXECUTION
    priority: EventPriority = EventPriority.NORMAL

    # Payload (event-specific data)
    data: Dict[str, Any] = field(default_factory=dict)

    # Routing metadata
    target_agents: Optional[List[str]] = None  # Specific agents to notify
    broadcast: bool = True  # Broadcast to all listeners?


class EventListener:
    """
    Event listener registration.

    Wraps a callback function with filtering and routing logic.
    """

    def __init__(
        self,
        callback: Callable,
        event_type: Optional[str] = None,
        event_subtype: Optional[str] = None,
        channel: Optional[EventChannel] = None,
        source_filter: Optional[str] = None,
        min_priority: EventPriority = EventPriority.LOW
    ):
        """
        Register event listener with optional filtering.

        Args:
            callback: Async function to call when event matches
                     Signature: async def callback(event: Event) -> None
            event_type: Only listen to this event type (None = all)
            event_subtype: Only listen to this subtype (None = all)
            channel: Only listen to this channel (None = all)
            source_filter: Only listen to events from this source ID (None = all)
            min_priority: Only listen to events at or above this priority
        """
        self.callback = callback
        self.event_type = event_type
        self.event_subtype = event_subtype
        self.channel = channel
        self.source_filter = source_filter
        self.min_priority = min_priority

        # Stats
        self.events_received = 0
        self.events_processed = 0
        self.last_event_time = 0.0

    def matches(self, event: Event) -> bool:
        """Check if event matches this listener's filters."""

        # Priority check
        if event.priority.value < self.min_priority.value:
            return False

        # Type check
        if self.event_type and event.type != self.event_type:
            return False

        # Subtype check
        if self.event_subtype and event.subtype != self.event_subtype:
            return False

        # Channel check
        if self.channel and event.channel != self.channel:
            return False

        # Source filter
        if self.source_filter and event.source_id != self.source_filter:
            return False

        return True

    async def notify(self, event: Event):
        """Notify listener of event."""
        self.events_received += 1

        try:
            await self.callback(event)
            self.events_processed += 1
            self.last_event_time = time.time()
        except Exception as e:
            logger.error(f"Event listener callback failed: {e}", exc_info=True)


class ExecutionEventBus:
    """
    Central event bus for facet execution and world events.

    Single instance per application. All components emit and listen here.
    """

    def __init__(self):
        """Initialize event bus."""

        # Listeners by channel (for fast routing)
        self.listeners: Dict[EventChannel, List[EventListener]] = defaultdict(list)
        self.global_listeners: List[EventListener] = []  # Listen to all channels

        # Event queue (priority-ordered)
        self.event_queue: asyncio.Queue = asyncio.Queue()

        # Event history (for debugging and replay)
        self.event_history: List[Event] = []
        self.max_history_size = 1000

        # Stats
        self.total_events_emitted = 0
        self.total_events_processed = 0
        self.events_by_type: Dict[str, int] = defaultdict(int)

        # Processing task
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("ExecutionEventBus initialized")

    def start(self):
        """Start event processing loop."""
        if not self.running:
            self.running = True
            self.processing_task = asyncio.create_task(self._process_events())
            logger.info("ExecutionEventBus started")

    async def stop(self):
        """Stop event processing loop."""
        if self.running:
            self.running = False
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            logger.info("ExecutionEventBus stopped")

    async def _process_events(self):
        """Event processing loop (runs continuously)."""
        while self.running:
            try:
                # Get next event (blocks until available)
                event = await self.event_queue.get()

                # Record in history
                self.event_history.append(event)
                if len(self.event_history) > self.max_history_size:
                    self.event_history.pop(0)

                # Update stats
                self.total_events_processed += 1
                self.events_by_type[event.type] += 1

                # Notify listeners
                await self._notify_listeners(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}", exc_info=True)

    async def _notify_listeners(self, event: Event):
        """Notify all matching listeners of event."""

        # Collect matching listeners
        matching_listeners = []

        # Global listeners (all channels)
        matching_listeners.extend([
            listener for listener in self.global_listeners
            if listener.matches(event)
        ])

        # Channel-specific listeners
        channel_listeners = self.listeners.get(event.channel, [])
        matching_listeners.extend([
            listener for listener in channel_listeners
            if listener.matches(event)
        ])

        # Notify all in parallel
        if matching_listeners:
            await asyncio.gather(*[
                listener.notify(event)
                for listener in matching_listeners
            ], return_exceptions=True)

    async def emit(
        self,
        event_type: str,
        event_subtype: str,
        channel: EventChannel = EventChannel.EXECUTION,
        priority: EventPriority = EventPriority.NORMAL,
        source_id: Optional[str] = None,
        source_name: Optional[str] = None,
        cycle: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
        target_agents: Optional[List[str]] = None,
        broadcast: bool = True
    ):
        """
        Emit event to bus.

        Args:
            event_type: Event type (e.g., "facet_execution")
            event_subtype: Event subtype (e.g., "facet_start")
            channel: Event channel for routing
            priority: Event priority
            source_id: ID of event source (facet, agent, etc.)
            source_name: Human-readable source name
            cycle: Cognition cycle number (if applicable)
            data: Event-specific payload
            target_agents: Specific agents to notify (None = all)
            broadcast: Broadcast to all listeners?
        """

        event = Event(
            type=event_type,
            subtype=event_subtype,
            timestamp=time.time(),
            cycle=cycle,
            source_id=source_id,
            source_name=source_name,
            channel=channel,
            priority=priority,
            data=data or {},
            target_agents=target_agents,
            broadcast=broadcast
        )

        # Add to queue
        await self.event_queue.put(event)
        self.total_events_emitted += 1

    def register_listener(
        self,
        callback: Callable,
        event_type: Optional[str] = None,
        event_subtype: Optional[str] = None,
        channel: Optional[EventChannel] = None,
        source_filter: Optional[str] = None,
        min_priority: EventPriority = EventPriority.LOW,
        global_listener: bool = False
    ) -> EventListener:
        """
        Register event listener.

        Args:
            callback: Async callback function
            event_type: Filter by event type (None = all)
            event_subtype: Filter by subtype (None = all)
            channel: Filter by channel (None = all)
            source_filter: Filter by source ID (None = all)
            min_priority: Minimum priority to receive
            global_listener: Listen to all channels?

        Returns:
            EventListener object (can be used to unregister)
        """

        listener = EventListener(
            callback=callback,
            event_type=event_type,
            event_subtype=event_subtype,
            channel=channel,
            source_filter=source_filter,
            min_priority=min_priority
        )

        if global_listener:
            self.global_listeners.append(listener)
        else:
            target_channel = channel or EventChannel.EXECUTION
            self.listeners[target_channel].append(listener)

        logger.debug(f"Registered listener: type={event_type}, subtype={event_subtype}, channel={channel}")

        return listener

    def unregister_listener(self, listener: EventListener):
        """Unregister event listener."""

        # Remove from global listeners
        if listener in self.global_listeners:
            self.global_listeners.remove(listener)

        # Remove from channel listeners
        for channel_listeners in self.listeners.values():
            if listener in channel_listeners:
                channel_listeners.remove(listener)

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""

        return {
            'total_emitted': self.total_events_emitted,
            'total_processed': self.total_events_processed,
            'queue_size': self.event_queue.qsize(),
            'history_size': len(self.event_history),
            'listeners': {
                'global': len(self.global_listeners),
                **{
                    channel.value: len(listeners)
                    for channel, listeners in self.listeners.items()
                }
            },
            'events_by_type': dict(self.events_by_type)
        }

    def get_recent_events(
        self,
        count: int = 50,
        event_type: Optional[str] = None,
        channel: Optional[EventChannel] = None
    ) -> List[Event]:
        """
        Get recent events from history.

        Args:
            count: Number of events to return
            event_type: Filter by type (None = all)
            channel: Filter by channel (None = all)

        Returns:
            List of recent events (newest first)
        """

        filtered = self.event_history

        if event_type:
            filtered = [e for e in filtered if e.type == event_type]

        if channel:
            filtered = [e for e in filtered if e.channel == channel]

        return list(reversed(filtered[-count:]))


# Global singleton instance
_global_event_bus: Optional[ExecutionEventBus] = None


def get_event_bus() -> ExecutionEventBus:
    """
    Get global event bus instance (singleton).

    Returns:
        Global ExecutionEventBus instance
    """
    global _global_event_bus

    if _global_event_bus is None:
        _global_event_bus = ExecutionEventBus()
        _global_event_bus.start()

    return _global_event_bus


def reset_event_bus():
    """Reset global event bus (for testing)."""
    global _global_event_bus

    if _global_event_bus:
        asyncio.create_task(_global_event_bus.stop())

    _global_event_bus = None


if __name__ == "__main__":
    """Test event bus."""

    print("=== Testing ExecutionEventBus ===\n")

    async def test_event_bus():
        # Create bus
        bus = ExecutionEventBus()
        bus.start()

        # Register test listeners
        execution_events = []
        async def on_execution_event(event: Event):
            execution_events.append(event)
            print(f"Execution event: {event.subtype} from {event.source_name}")

        world_events = []
        async def on_world_event(event: Event):
            world_events.append(event)
            print(f"World event: {event.subtype}")

        bus.register_listener(
            on_execution_event,
            channel=EventChannel.EXECUTION
        )

        bus.register_listener(
            on_world_event,
            channel=EventChannel.WORLD
        )

        # Emit test events
        await bus.emit(
            "facet_execution",
            "facet_start",
            channel=EventChannel.EXECUTION,
            source_id="facet_123",
            source_name="Intuition Facet",
            cycle=1,
            data={'input': 'test'}
        )

        await bus.emit(
            "facet_execution",
            "facet_complete",
            channel=EventChannel.EXECUTION,
            source_id="facet_123",
            source_name="Intuition Facet",
            cycle=1,
            data={'output': 'result', 'execution_time': 0.5}
        )

        await bus.emit(
            "world_change",
            "agent_speak",
            channel=EventChannel.WORLD,
            source_id="agent_red",
            source_name="Red Fire Anklebiter",
            data={'text': 'Hello world!'}
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Check results
        print(f"\nExecution events received: {len(execution_events)}")
        print(f"World events received: {len(world_events)}")

        # Stats
        stats = bus.get_stats()
        print(f"\nBus stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Recent events
        recent = bus.get_recent_events(count=10)
        print(f"\nRecent events: {len(recent)}")
        for event in recent:
            print(f"  {event.type}/{event.subtype} from {event.source_name}")

        await bus.stop()
        print("\n=== Test complete ===")

    asyncio.run(test_event_bus())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
