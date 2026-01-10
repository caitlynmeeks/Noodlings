# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Semantic Integration - Event-Based Context Building
#
#   Instead of agents reading raw world state ("room has 3 chairs"),
#   they experience events ("Alice sat down in the red chair").
#   This module bridges the semantic_world event system with cmush,
#   recording speech, movement, actions as typed events. When an
#   agent needs context, it queries recent events relevant to them
#   and builds a narrative description. Events, not snapshots.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.semantic_integration
# PURPOSE:  Bridge semantic_world events to cmush agents
# LAYER:    Backend / Integration
# ──────────────────────────────────────────────────────────────
#
# KEY FUNCTIONS:
#   init_semantic_world()      Initialize event system
#   log_speech()               Record speech event
#   log_movement()             Record movement event
#   log_arrival()              Record arrival event
#   log_departure()            Record departure event
#   get_semantic_context()     Build narrative context for agent
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Semantic World Integration for noodleMUSH

This module bridges the semantic_world event system with noodleMUSH,
enabling rich narrative context for agents based on events rather than
raw world state.

Usage in server.py:
    from semantic_integration import get_semantic_context, log_event

    # When someone speaks
    log_event('speech', speaker_id, stage_id, content=message)

    # When building agent context
    context = get_semantic_context(agent_id, stage_id)
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

# Add noodlestudio to path for semantic_world imports
noodlestudio_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "../noodlestudio/noodlestudio/core"
))
if noodlestudio_path not in sys.path:
    sys.path.insert(0, noodlestudio_path)

from semantic_world import (
    Event,
    EventType,
    EventStore,
    ContextBuilder,
    StageDefinition,
    SpatialContext,
    speech_event,
    movement_event,
    arrival_event,
    departure_event,
    action_event,
    perception_event,
    social_event,
    environmental_event,
)

# Global instances
_event_store: Optional[EventStore] = None
_context_builder: Optional[ContextBuilder] = None
_stage_definitions: Dict[str, StageDefinition] = {}


def init_semantic_world(persist_path: Optional[str] = None, stages_path: Optional[str] = None):
    """
    Initialize the semantic world system.

    Call this once at server startup.

    Args:
        persist_path: Directory for event persistence (optional)
        stages_path: Directory containing stage YAML definitions
    """
    global _event_store, _context_builder

    _event_store = EventStore(persist_path=persist_path)
    _context_builder = ContextBuilder(_event_store, stages_path=stages_path)

    print(f"[SemanticWorld] Initialized. Events: {len(_event_store)}")


def get_event_store() -> EventStore:
    """Get the global event store."""
    global _event_store
    if _event_store is None:
        init_semantic_world()
    return _event_store


def get_context_builder() -> ContextBuilder:
    """Get the global context builder."""
    global _context_builder
    if _context_builder is None:
        init_semantic_world()
    return _context_builder


# ═══════════════════════════════════════════════════════════════════════════════
# Event Logging Functions
# ═══════════════════════════════════════════════════════════════════════════════

def log_speech(
    speaker_id: str,
    stage_id: str,
    content: str,
    manner: Optional[str] = None,
    witnesses: Optional[List[str]] = None
) -> Event:
    """
    Log a speech event.

    Args:
        speaker_id: Who spoke (e.g., "agent_red", "user_caity")
        stage_id: Stage where speech occurred
        content: What was said
        manner: How it was said (optional)
        witnesses: List of entity IDs who heard it

    Returns:
        The logged Event
    """
    store = get_event_store()

    event = speech_event(
        speaker=speaker_id,
        content=content,
        stage_id=stage_id,
        manner=manner,
        witnesses=witnesses or []
    )

    return store.append(event)


def log_movement(
    mover_id: str,
    stage_id: str,
    destination: str,
    origin: Optional[str] = None,
    manner: Optional[str] = None,
    witnesses: Optional[List[str]] = None
) -> Event:
    """
    Log a movement event (moving within a stage).

    Args:
        mover_id: Who moved
        stage_id: Stage where movement occurred
        destination: Where they moved to (anchor name)
        origin: Where they moved from (optional)
        manner: How they moved (optional)
        witnesses: Who saw it

    Returns:
        The logged Event
    """
    store = get_event_store()

    event = movement_event(
        mover=mover_id,
        destination=destination,
        stage_id=stage_id,
        origin=origin,
        manner=manner
    )

    # Add witnesses
    for w in (witnesses or []):
        event.add_witness(w)

    return store.append(event)


def log_arrival(
    arriver_id: str,
    stage_id: str,
    from_stage: Optional[str] = None,
    manner: Optional[str] = None,
    witnesses: Optional[List[str]] = None
) -> Event:
    """
    Log an arrival event (entering a stage).

    Args:
        arriver_id: Who arrived
        stage_id: Stage they arrived at
        from_stage: Stage they came from (optional)
        manner: How they arrived
        witnesses: Who saw the arrival

    Returns:
        The logged Event
    """
    store = get_event_store()

    event = arrival_event(
        arriver=arriver_id,
        stage_id=stage_id,
        from_stage=from_stage,
        manner=manner
    )

    for w in (witnesses or []):
        event.add_witness(w)

    return store.append(event)


def log_departure(
    departer_id: str,
    stage_id: str,
    to_stage: Optional[str] = None,
    manner: Optional[str] = None,
    witnesses: Optional[List[str]] = None
) -> Event:
    """
    Log a departure event (leaving a stage).

    Args:
        departer_id: Who left
        stage_id: Stage they left from
        to_stage: Stage they went to (optional)
        manner: How they left
        witnesses: Who saw the departure

    Returns:
        The logged Event
    """
    store = get_event_store()

    event = departure_event(
        departer=departer_id,
        stage_id=stage_id,
        to_stage=to_stage,
        manner=manner
    )

    for w in (witnesses or []):
        event.add_witness(w)

    return store.append(event)


def log_action(
    actor_id: str,
    verb: str,
    stage_id: str,
    target: Optional[str] = None,
    manner: Optional[str] = None,
    detail: Optional[str] = None,
    witnesses: Optional[List[str]] = None
) -> Event:
    """
    Log a general action event.

    Args:
        actor_id: Who did the action
        verb: What they did (e.g., "picked up", "examined", "sat down")
        stage_id: Where it happened
        target: Target of action (optional)
        manner: How they did it
        detail: Additional detail
        witnesses: Who saw it

    Returns:
        The logged Event
    """
    store = get_event_store()

    event = action_event(
        actor=actor_id,
        verb=verb,
        target=target,
        stage_id=stage_id,
        manner=manner,
        detail=detail
    )

    for w in (witnesses or []):
        event.add_witness(w)

    return store.append(event)


def log_emote(
    actor_id: str,
    stage_id: str,
    emote_text: str,
    witnesses: Optional[List[str]] = None
) -> Event:
    """
    Log an emote (action without speech).

    Args:
        actor_id: Who emoted
        stage_id: Where
        emote_text: The emote action (e.g., "waves hello", "giggles")
        witnesses: Who saw it

    Returns:
        The logged Event
    """
    store = get_event_store()

    # Parse emote into verb and detail
    words = emote_text.split()
    verb = words[0] if words else "emoted"
    detail = " ".join(words[1:]) if len(words) > 1 else None

    event = Event(
        type=EventType.EMOTE,
        actor=actor_id,
        verb=verb,
        detail=detail,
        spatial=SpatialContext(stage_id=stage_id)
    )

    for w in (witnesses or []):
        event.add_witness(w)

    return store.append(event)


# ═══════════════════════════════════════════════════════════════════════════════
# Context Building Functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_semantic_context(
    entity_id: str,
    stage_id: str,
    window_minutes: int = 10,
    max_events: int = 10
) -> str:
    """
    Get rich semantic context for an agent.

    This is the main function to call when building agent context.
    It returns a narrative text describing the entity's experience
    of recent events.

    Args:
        entity_id: The agent requesting context
        stage_id: The stage they're in
        window_minutes: How far back to look for events
        max_events: Maximum events to include

    Returns:
        Narrative context text suitable for LLM prompt
    """
    builder = get_context_builder()
    context = builder.build_context(
        entity_id=entity_id,
        stage_id=stage_id,
        window=timedelta(minutes=window_minutes),
        max_events=max_events
    )
    return context.narrative


def get_full_context(
    entity_id: str,
    stage_id: str,
    window_minutes: int = 10
):
    """
    Get the full AgentContext object with structured data.

    Use this when you need more than just the narrative text.

    Returns:
        AgentContext with narrative, tension_level, others_present, etc.
    """
    builder = get_context_builder()
    return builder.build_context(
        entity_id=entity_id,
        stage_id=stage_id,
        window=timedelta(minutes=window_minutes)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage Definition Management
# ═══════════════════════════════════════════════════════════════════════════════

def register_stage(stage_id: str, definition: StageDefinition):
    """Register a stage definition for context building."""
    builder = get_context_builder()
    builder._stage_cache[stage_id] = definition


def register_stage_from_room(room_id: str, room_data: Dict[str, Any]):
    """
    Create and register a stage definition from legacy room data.

    This bridges the old rooms.json format to semantic stages.
    """
    name = room_data.get('name', room_id)
    description = room_data.get('description', '')

    # Create minimal stage definition
    stage_def = StageDefinition(
        id=room_id,
        name=name,
        essence=description,
        anchors={},
        features={},
        zones={}
    )

    register_stage(room_id, stage_def)


# ═══════════════════════════════════════════════════════════════════════════════
# Subscription for Real-time Updates
# ═══════════════════════════════════════════════════════════════════════════════

def subscribe_to_events(callback, stage: Optional[str] = None, entity: Optional[str] = None):
    """
    Subscribe to new events.

    Args:
        callback: Function(Event) to call for each new event
        stage: Filter to only events in this stage
        entity: Filter to only events involving this entity

    Returns:
        Unsubscribe function
    """
    store = get_event_store()
    return store.subscribe(callback, stage=stage, entity=entity)


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics and Debugging
# ═══════════════════════════════════════════════════════════════════════════════

def get_stats() -> Dict[str, Any]:
    """Get event store statistics."""
    store = get_event_store()
    return store.stats()


def get_recent_events(n: int = 10) -> List[Event]:
    """Get the N most recent events."""
    store = get_event_store()
    return store.last(n)


__all__ = [
    # Initialization
    "init_semantic_world",
    "get_event_store",
    "get_context_builder",

    # Event logging
    "log_speech",
    "log_movement",
    "log_arrival",
    "log_departure",
    "log_action",
    "log_emote",

    # Context building
    "get_semantic_context",
    "get_full_context",

    # Stage management
    "register_stage",
    "register_stage_from_room",

    # Subscriptions
    "subscribe_to_events",

    # Debugging
    "get_stats",
    "get_recent_events",
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
