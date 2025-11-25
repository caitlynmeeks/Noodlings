"""
Unity-style Event System for Noodlings.

Provides component event messaging (OnSpeak, OnFACSChange, etc.)
for reactive programming and script integration.
"""

from typing import Callable, List, Any, Dict
import logging

logger = logging.getLogger(__name__)


class Event:
    """
    Unity-style event that components can fire and scripts can subscribe to.

    Example:
        event = Event()
        event.add_listener(lambda data: print(data))
        event.invoke({'message': 'Hello!'})
    """

    def __init__(self, name: str = "UnnamedEvent"):
        """
        Initialize event.

        Args:
            name: Event name for debugging
        """
        self.name = name
        self.listeners: List[Callable] = []
        self.one_time_listeners: List[Callable] = []

    def add_listener(self, callback: Callable):
        """
        Add persistent listener to event.

        Args:
            callback: Function to call when event fires
        """
        if callback not in self.listeners:
            self.listeners.append(callback)
            logger.debug(f"[{self.name}] Added listener: {callback.__name__}")

    def add_listener_once(self, callback: Callable):
        """
        Add one-time listener (auto-removes after first fire).

        Args:
            callback: Function to call once
        """
        if callback not in self.one_time_listeners:
            self.one_time_listeners.append(callback)
            logger.debug(f"[{self.name}] Added one-time listener: {callback.__name__}")

    def remove_listener(self, callback: Callable):
        """
        Remove specific listener.

        Args:
            callback: Listener to remove
        """
        if callback in self.listeners:
            self.listeners.remove(callback)
            logger.debug(f"[{self.name}] Removed listener: {callback.__name__}")

    def remove_all_listeners(self):
        """Remove all listeners (persistent and one-time)."""
        count = len(self.listeners) + len(self.one_time_listeners)
        self.listeners.clear()
        self.one_time_listeners.clear()
        logger.debug(f"[{self.name}] Removed all {count} listeners")

    def invoke(self, data: Any = None):
        """
        Fire event, calling all listeners.

        Args:
            data: Event data passed to listeners
        """
        # Call persistent listeners
        for listener in self.listeners:
            try:
                listener(data)
            except Exception as e:
                logger.error(f"[{self.name}] Listener {listener.__name__} failed: {e}")

        # Call one-time listeners
        for listener in self.one_time_listeners:
            try:
                listener(data)
            except Exception as e:
                logger.error(f"[{self.name}] One-time listener {listener.__name__} failed: {e}")

        # Clear one-time listeners
        self.one_time_listeners.clear()

    def has_listeners(self) -> bool:
        """Check if event has any listeners."""
        return len(self.listeners) > 0 or len(self.one_time_listeners) > 0

    def listener_count(self) -> int:
        """Get total listener count."""
        return len(self.listeners) + len(self.one_time_listeners)


class EventBus:
    """
    Global event bus for system-wide events.

    Singleton pattern for world-level events like agent spawn/removal.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.events: Dict[str, Event] = {}
        self._initialized = True

    def get_event(self, event_name: str) -> Event:
        """
        Get or create event by name.

        Args:
            event_name: Event identifier

        Returns:
            Event instance
        """
        if event_name not in self.events:
            self.events[event_name] = Event(event_name)

        return self.events[event_name]

    def fire(self, event_name: str, data: Any = None):
        """
        Fire event by name.

        Args:
            event_name: Event to fire
            data: Event data
        """
        event = self.get_event(event_name)
        event.invoke(data)


# Global event bus instance
event_bus = EventBus()


# Example usage
if __name__ == '__main__':
    # Create event
    on_speak = Event("OnSpeak")

    # Add listener
    def print_speech(data):
        print(f"Agent said: {data['text']}")

    on_speak.add_listener(print_speech)

    # Fire event
    on_speak.invoke({'text': 'Hello world!', 'timestamp': 12345})

    # Output: "Agent said: Hello world!"
