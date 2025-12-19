"""
Event Store - The Append-Only Log of Reality

The event store IS the world. Events are immutable facts - once something
happened, it cannot unhappen. All world state is projected from this log.

This is event sourcing for existence itself.

Features:
    - Append-only: Events cannot be modified or deleted
    - Queryable: Filter by time, stage, entity, type
    - Subscribable: Register callbacks for real-time event streams
    - Persistent: Write-ahead log to disk, load on startup

Author: Caitlyn + Claude
Date: December 2025
"""

import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Iterator
from collections import defaultdict

from .event import Event, EventType


# Type alias for event callbacks
EventCallback = Callable[[Event], None]


class EventStore:
    """
    The append-only log of all happenings.

    This store maintains the complete history of events and provides
    efficient querying and real-time subscriptions.

    Usage:
        store = EventStore(persist_path="world/events")

        # Append events (the only write operation)
        store.append(event)

        # Query events
        recent = store.since(minutes=10)
        in_nexus = store.in_stage("the_nexus")
        red_events = store.involving("red")

        # Subscribe to new events
        store.subscribe(my_callback)
        store.subscribe(my_callback, stage="the_nexus")  # Filtered
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the event store.

        Args:
            persist_path: Directory for event persistence. If None, in-memory only.
        """
        # The log - ordered list of all events
        self._events: List[Event] = []

        # Indexes for efficient querying
        self._by_stage: Dict[str, List[int]] = defaultdict(list)  # stage_id -> event indices
        self._by_entity: Dict[str, List[int]] = defaultdict(list)  # entity_id -> event indices
        self._by_type: Dict[EventType, List[int]] = defaultdict(list)  # type -> event indices

        # Subscribers for real-time updates
        self._subscribers: List[tuple[EventCallback, Optional[str], Optional[str]]] = []
        # Each subscriber is (callback, stage_filter, entity_filter)

        # Persistence
        self._persist_path = Path(persist_path) if persist_path else None
        self._write_lock = threading.Lock()

        # Load existing events if persistence enabled
        if self._persist_path:
            self._load_from_disk()

    # ─────────────────────────────────────────────────────────────────────────
    # Core Operations
    # ─────────────────────────────────────────────────────────────────────────

    def append(self, event: Event) -> Event:
        """
        Append an event to the store.

        This is the ONLY write operation. Events are immutable once appended.

        Args:
            event: The event to append

        Returns:
            The appended event (with any normalization applied)
        """
        with self._write_lock:
            # Get the index this event will have
            idx = len(self._events)

            # Add to main log
            self._events.append(event)

            # Update indexes
            if event.spatial and event.spatial.stage_id:
                self._by_stage[event.spatial.stage_id].append(idx)

            # Index by actor
            if event.actor:
                self._by_entity[event.actor].append(idx)

            # Index by object (target entity)
            if event.object:
                self._by_entity[event.object].append(idx)

            # Index by all witnesses
            for witness in event.witnesses:
                self._by_entity[witness.entity_id].append(idx)

            # Index by type
            self._by_type[event.type].append(idx)

            # Persist if enabled
            if self._persist_path:
                self._persist_event(event)

            # Notify subscribers
            self._notify_subscribers(event)

        return event

    def __len__(self) -> int:
        """Return total number of events."""
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        """Iterate over all events in chronological order."""
        return iter(self._events)

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Queries
    # ─────────────────────────────────────────────────────────────────────────

    def since(
        self,
        timestamp: Optional[datetime] = None,
        minutes: Optional[float] = None,
        seconds: Optional[float] = None
    ) -> List[Event]:
        """
        Get events since a given time.

        Args:
            timestamp: Absolute timestamp cutoff
            minutes: Get events from last N minutes
            seconds: Get events from last N seconds

        Returns:
            List of events since the cutoff, chronologically ordered
        """
        if timestamp is None:
            if minutes is not None:
                timestamp = datetime.utcnow() - timedelta(minutes=minutes)
            elif seconds is not None:
                timestamp = datetime.utcnow() - timedelta(seconds=seconds)
            else:
                return list(self._events)  # Return all

        return [e for e in self._events if e.timestamp >= timestamp]

    def between(self, start: datetime, end: datetime) -> List[Event]:
        """Get events between two timestamps."""
        return [e for e in self._events if start <= e.timestamp <= end]

    def last(self, n: int) -> List[Event]:
        """Get the last N events."""
        return self._events[-n:] if n < len(self._events) else list(self._events)

    # ─────────────────────────────────────────────────────────────────────────
    # Spatial Queries
    # ─────────────────────────────────────────────────────────────────────────

    def in_stage(self, stage_id: str, since: Optional[datetime] = None) -> List[Event]:
        """
        Get events that occurred in a specific stage.

        Args:
            stage_id: The stage to filter by
            since: Optional timestamp cutoff

        Returns:
            Events in that stage, chronologically ordered
        """
        indices = self._by_stage.get(stage_id, [])
        events = [self._events[i] for i in indices]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events

    def in_zone(self, stage_id: str, zone_id: str, since: Optional[datetime] = None) -> List[Event]:
        """Get events in a specific zone within a stage."""
        stage_events = self.in_stage(stage_id, since)
        return [e for e in stage_events
                if e.spatial and e.spatial.zone == zone_id]

    # ─────────────────────────────────────────────────────────────────────────
    # Entity Queries
    # ─────────────────────────────────────────────────────────────────────────

    def involving(self, entity_id: str, since: Optional[datetime] = None) -> List[Event]:
        """
        Get events involving an entity (as actor, target, or witness).

        Args:
            entity_id: The entity to find events for
            since: Optional timestamp cutoff

        Returns:
            Events involving that entity
        """
        indices = self._by_entity.get(entity_id, [])
        events = [self._events[i] for i in indices]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events

    def witnessed_by(self, entity_id: str, since: Optional[datetime] = None) -> List[Event]:
        """Get events that were witnessed by an entity."""
        events = self.involving(entity_id, since)
        return [e for e in events if e.witnessed_by(entity_id)]

    def acted_by(self, entity_id: str, since: Optional[datetime] = None) -> List[Event]:
        """Get events where an entity was the actor."""
        events = self.involving(entity_id, since)
        return [e for e in events if e.actor == entity_id]

    # ─────────────────────────────────────────────────────────────────────────
    # Type Queries
    # ─────────────────────────────────────────────────────────────────────────

    def of_type(self, event_type: EventType, since: Optional[datetime] = None) -> List[Event]:
        """Get events of a specific type."""
        indices = self._by_type.get(event_type, [])
        events = [self._events[i] for i in indices]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events

    def speech(self, since: Optional[datetime] = None) -> List[Event]:
        """Get all speech events."""
        return self.of_type(EventType.SPEECH, since)

    def movements(self, since: Optional[datetime] = None) -> List[Event]:
        """Get all movement events."""
        return self.of_type(EventType.MOVEMENT, since)

    # ─────────────────────────────────────────────────────────────────────────
    # Combined Queries
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        stage: Optional[str] = None,
        entity: Optional[str] = None,
        event_type: Optional[EventType] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """
        Combined query with multiple filters.

        All provided filters must match (AND logic).
        """
        # Start with all events or filtered set
        if stage:
            events = self.in_stage(stage, since)
        elif entity:
            events = self.involving(entity, since)
        elif event_type:
            events = self.of_type(event_type, since)
        else:
            events = self.since(since) if since else list(self._events)

        # Apply additional filters
        if stage and entity:
            events = [e for e in events if e.actor == entity or
                      e.object == entity or
                      e.witnessed_by(entity)]
        if event_type and (stage or entity):
            events = [e for e in events if e.type == event_type]

        # Apply limit
        if limit:
            events = events[-limit:]

        return events

    # ─────────────────────────────────────────────────────────────────────────
    # Subscriptions
    # ─────────────────────────────────────────────────────────────────────────

    def subscribe(
        self,
        callback: EventCallback,
        stage: Optional[str] = None,
        entity: Optional[str] = None
    ) -> Callable[[], None]:
        """
        Subscribe to new events.

        Args:
            callback: Function to call with each new event
            stage: Only receive events in this stage
            entity: Only receive events involving this entity

        Returns:
            Unsubscribe function
        """
        subscription = (callback, stage, entity)
        self._subscribers.append(subscription)

        # Return unsubscribe function
        def unsubscribe():
            if subscription in self._subscribers:
                self._subscribers.remove(subscription)

        return unsubscribe

    def _notify_subscribers(self, event: Event):
        """Notify all matching subscribers of a new event."""
        for callback, stage_filter, entity_filter in self._subscribers:
            # Check stage filter
            if stage_filter:
                if not event.spatial or event.spatial.stage_id != stage_filter:
                    continue

            # Check entity filter
            if entity_filter:
                if not (event.actor == entity_filter or
                        event.object == entity_filter or
                        event.witnessed_by(entity_filter)):
                    continue

            # Callback matches filters
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event subscriber: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _persist_event(self, event: Event):
        """Write event to disk."""
        if not self._persist_path:
            return

        # Ensure directory exists
        self._persist_path.mkdir(parents=True, exist_ok=True)

        # Append to daily log file
        date_str = event.timestamp.strftime("%Y-%m-%d")
        log_file = self._persist_path / f"events_{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(event.to_json().replace("\n", " ") + "\n")

    def _load_from_disk(self):
        """Load all events from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return

        # Find all event log files
        log_files = sorted(self._persist_path.glob("events_*.jsonl"))

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            event = Event.from_json(line)
                            # Add to store without re-persisting
                            idx = len(self._events)
                            self._events.append(event)

                            # Update indexes
                            if event.spatial and event.spatial.stage_id:
                                self._by_stage[event.spatial.stage_id].append(idx)
                            if event.actor:
                                self._by_entity[event.actor].append(idx)
                            if event.object:
                                self._by_entity[event.object].append(idx)
                            for witness in event.witnesses:
                                self._by_entity[witness.entity_id].append(idx)
                            self._by_type[event.type].append(idx)

            except Exception as e:
                print(f"Error loading {log_file}: {e}")

        print(f"Loaded {len(self._events)} events from disk")

    def save_snapshot(self, path: str):
        """Save complete snapshot of all events."""
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_count": len(self._events),
            "events": [e.to_dict() for e in self._events]
        }

        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)

    @classmethod
    def load_snapshot(cls, path: str) -> 'EventStore':
        """Load store from a snapshot file."""
        with open(path, "r") as f:
            snapshot = json.load(f)

        store = cls()
        for event_dict in snapshot["events"]:
            event = Event.from_dict(event_dict)
            store._events.append(event)

            # Update indexes
            idx = len(store._events) - 1
            if event.spatial and event.spatial.stage_id:
                store._by_stage[event.spatial.stage_id].append(idx)
            if event.actor:
                store._by_entity[event.actor].append(idx)
            if event.object:
                store._by_entity[event.object].append(idx)
            for witness in event.witnesses:
                store._by_entity[witness.entity_id].append(idx)
            store._by_type[event.type].append(idx)

        return store

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def clear(self):
        """
        Clear all events. USE WITH CAUTION.

        This violates the append-only principle but may be needed for testing.
        """
        with self._write_lock:
            self._events.clear()
            self._by_stage.clear()
            self._by_entity.clear()
            self._by_type.clear()

    def stats(self) -> Dict[str, Any]:
        """Get statistics about the event store."""
        return {
            "total_events": len(self._events),
            "stages": len(self._by_stage),
            "entities": len(self._by_entity),
            "events_by_type": {
                t.value: len(indices)
                for t, indices in self._by_type.items()
            },
            "subscribers": len(self._subscribers),
            "oldest": self._events[0].timestamp.isoformat() if self._events else None,
            "newest": self._events[-1].timestamp.isoformat() if self._events else None
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Global Event Store Instance
# ═══════════════════════════════════════════════════════════════════════════════

_global_store: Optional[EventStore] = None


def get_event_store() -> EventStore:
    """Get the global event store instance."""
    global _global_store
    if _global_store is None:
        _global_store = EventStore()
    return _global_store


def init_event_store(persist_path: Optional[str] = None) -> EventStore:
    """Initialize the global event store with optional persistence."""
    global _global_store
    _global_store = EventStore(persist_path=persist_path)
    return _global_store


__all__ = [
    "EventStore",
    "get_event_store",
    "init_event_store",
]
