"""
Projections - Views Derived from the Event Log

The event log IS reality. Everything else is a projection - a view derived
from the accumulated happenings.

Three fundamental projections:
    - Stage: The container (accumulated creation/destruction events)
    - Situation: What's happening NOW (recent events + current states)
    - Experience: Events filtered through a perspective (for one entity)

These are computed, never stored as source of truth.

Author: Caitlyn + Claude
Date: December 2025
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict

from .event import Event, EventType, Witness
from .event_store import EventStore


# ═══════════════════════════════════════════════════════════════════════════════
# Presence - An entity's current state in a stage
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Presence:
    """
    An entity's presence in a stage at this moment.

    Derived from their most recent events.
    """

    entity_id: str
    stage_id: str

    # Where within the stage
    anchor: Optional[str] = None  # Named location (hearth, threshold, etc)
    zone: Optional[str] = None

    # Current state (from most recent events)
    posture: Optional[str] = None  # standing, sitting, hunched over
    facing: Optional[str] = None   # What they're oriented toward
    doing: Optional[str] = None    # Current activity
    state: Optional[str] = None    # Emotional/mental state

    # When they arrived
    arrived_at: Optional[datetime] = None

    # Most recent action
    last_action: Optional[str] = None
    last_action_at: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Spatial Relation - How two entities relate in space
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpatialRelation:
    """Spatial relationship between two entities."""

    subject: str      # Who
    object: str       # Relative to whom/what
    relation: str     # The relationship ("near", "across from", "behind")
    quality: Optional[str] = None  # Additional meaning ("maintaining distance")


# ═══════════════════════════════════════════════════════════════════════════════
# Tension - Narrative potential in the current moment
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tension:
    """A point of narrative tension or potential."""

    description: str
    involves: List[str]  # Entity IDs involved
    intensity: float = 0.5  # 0-1
    source_event: Optional[str] = None  # Event ID that created this tension


# ═══════════════════════════════════════════════════════════════════════════════
# Situation - The dynamic present moment
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Situation:
    """
    The current situation in a stage.

    Projected from recent events - who's here, what they're doing,
    what just happened, what tensions exist.
    """

    stage_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Current atmosphere
    atmosphere: Dict[str, str] = field(default_factory=dict)
    # Keys: mood, light, sound, etc.

    # Who's present and their current state
    presences: Dict[str, Presence] = field(default_factory=dict)
    # Key: entity_id

    # Spatial relationships
    relations: List[SpatialRelation] = field(default_factory=list)

    # Recent events (narrative context)
    recent_events: List[Event] = field(default_factory=list)

    # Narrative tensions
    tensions: List[Tension] = field(default_factory=list)

    def get_presence(self, entity_id: str) -> Optional[Presence]:
        """Get an entity's presence if they're in this situation."""
        return self.presences.get(entity_id)

    def entities_at(self, anchor: str) -> List[str]:
        """Get entities at a specific anchor point."""
        return [
            p.entity_id for p in self.presences.values()
            if p.anchor == anchor
        ]

    def relation_between(self, a: str, b: str) -> Optional[SpatialRelation]:
        """Get the spatial relation between two entities."""
        for r in self.relations:
            if (r.subject == a and r.object == b) or \
               (r.subject == b and r.object == a):
                return r
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Experience - Reality filtered through a perspective
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Experience:
    """
    One entity's experience of recent events.

    This is the phenomenological view - reality as experienced by a specific
    entity, filtered through what they witnessed, colored by their
    interpretations and emotional responses.
    """

    entity_id: str
    stage_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Events as this entity experienced them
    # Each event is from their perspective (using their witness data)
    witnessed_events: List[Event] = field(default_factory=list)

    # Their current state
    current_anchor: Optional[str] = None
    current_state: Optional[str] = None
    current_doing: Optional[str] = None

    # Others they're aware of (and how)
    awareness_of: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Key: entity_id, Value: {relation, state, last_observed, interpretation}

    # Recent internal events (their own thoughts/feelings)
    internal_events: List[Event] = field(default_factory=list)

    def narrate(self) -> str:
        """Generate a narrative of this experience."""
        lines = []

        # Where am I?
        if self.current_anchor:
            lines.append(f"You are at the {self.current_anchor}.")

        # What am I doing?
        if self.current_doing:
            lines.append(f"You are {self.current_doing}.")

        # What have I witnessed?
        if self.witnessed_events:
            lines.append("")
            lines.append("Recently:")
            for event in self.witnessed_events[-5:]:  # Last 5
                narration = event.narrate(perspective=self.entity_id)
                if narration:
                    lines.append(f"- {narration}")

        # Who else am I aware of?
        if self.awareness_of:
            lines.append("")
            for other_id, awareness in self.awareness_of.items():
                relation = awareness.get("relation", "nearby")
                state = awareness.get("state", "")
                line = f"{other_id.title()} is {relation}"
                if state:
                    line += f", {state}"
                lines.append(line + ".")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Projection Functions
# ═══════════════════════════════════════════════════════════════════════════════

def project_situation(
    store: EventStore,
    stage_id: str,
    window: timedelta = timedelta(minutes=10)
) -> Situation:
    """
    Project the current situation in a stage from the event log.

    Args:
        store: The event store to project from
        stage_id: The stage to project
        window: How far back to look for "recent" events

    Returns:
        The current Situation
    """
    situation = Situation(stage_id=stage_id)

    # Get recent events in this stage
    cutoff = datetime.utcnow() - window
    events = store.in_stage(stage_id, since=cutoff)
    situation.recent_events = events

    # Track presences from arrivals/departures/movements
    for event in events:
        _update_presences(situation, event)

    # Infer spatial relations
    _infer_relations(situation)

    # Detect narrative tensions
    _detect_tensions(situation, events)

    # Build atmosphere from environmental events and presences
    _build_atmosphere(situation, events)

    return situation


def project_experience(
    store: EventStore,
    entity_id: str,
    stage_id: str,
    window: timedelta = timedelta(minutes=10)
) -> Experience:
    """
    Project an entity's experience from the event log.

    Args:
        store: The event store
        entity_id: Whose experience to project
        stage_id: The stage context
        window: How far back to look

    Returns:
        The entity's Experience
    """
    experience = Experience(entity_id=entity_id, stage_id=stage_id)

    cutoff = datetime.utcnow() - window

    # Get events this entity witnessed
    events = store.witnessed_by(entity_id, since=cutoff)

    # Filter to this stage and deduplicate by event ID
    seen_ids = set()
    unique_events = []
    for e in events:
        if e.spatial and e.spatial.stage_id == stage_id:
            if e.id not in seen_ids:
                seen_ids.add(e.id)
                unique_events.append(e)

    events = unique_events
    experience.witnessed_events = events

    # Find internal events
    experience.internal_events = [
        e for e in events
        if e.type == EventType.INTERNAL and e.subject == entity_id
    ]

    # Determine current state from most recent events
    actor_events = [e for e in events if e.actor == entity_id]
    if actor_events:
        latest = actor_events[-1]
        if latest.spatial and latest.spatial.anchor:
            experience.current_anchor = latest.spatial.anchor
        if latest.manner:
            experience.current_state = latest.manner

    # Build awareness of others
    for event in events:
        if event.actor != entity_id and event.actor != "environment":
            _update_awareness(experience, event, entity_id)
        if event.object and event.object != entity_id:
            _update_awareness_of_object(experience, event, entity_id)

    return experience


def project_narrative(
    store: EventStore,
    stage_id: str,
    perspective: Optional[str] = None,
    window: timedelta = timedelta(minutes=5)
) -> str:
    """
    Project a narrative text from recent events.

    Args:
        store: The event store
        stage_id: The stage to narrate
        perspective: Entity perspective (or None for objective)
        window: Time window

    Returns:
        Narrative text
    """
    cutoff = datetime.utcnow() - window

    if perspective:
        events = store.witnessed_by(perspective, since=cutoff)
        # Filter and deduplicate
        seen_ids = set()
        unique_events = []
        for e in events:
            if e.spatial and e.spatial.stage_id == stage_id:
                if e.id not in seen_ids:
                    seen_ids.add(e.id)
                    unique_events.append(e)
        events = unique_events
    else:
        events = store.in_stage(stage_id, since=cutoff)

    lines = []
    for event in events:
        narration = event.narrate(perspective=perspective)
        if narration:
            lines.append(narration)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _update_presences(situation: Situation, event: Event):
    """Update presences based on an event."""
    actor = event.actor
    if not actor or actor == "environment":
        return

    # Ensure presence exists
    if actor not in situation.presences:
        situation.presences[actor] = Presence(
            entity_id=actor,
            stage_id=situation.stage_id
        )

    presence = situation.presences[actor]

    # Update based on event type
    if event.type == EventType.ARRIVAL:
        presence.arrived_at = event.timestamp

    if event.type == EventType.DEPARTURE:
        # Remove from presences
        del situation.presences[actor]
        return

    if event.type == EventType.MOVEMENT:
        if event.destination:
            presence.anchor = event.destination
        if event.spatial and event.spatial.anchor:
            presence.anchor = event.spatial.anchor

    # Update last action
    presence.last_action = event.verb
    presence.last_action_at = event.timestamp

    # Extract state from manner
    if event.manner:
        presence.state = event.manner

    # Extract doing from verb + object
    if event.verb and event.object:
        presence.doing = f"{event.verb} {event.object}"
    elif event.verb:
        presence.doing = event.verb


def _infer_relations(situation: Situation):
    """Infer spatial relations between presences."""
    presences = list(situation.presences.values())

    for i, p1 in enumerate(presences):
        for p2 in presences[i + 1:]:
            relation = _compute_relation(p1, p2)
            if relation:
                situation.relations.append(relation)


def _compute_relation(p1: Presence, p2: Presence) -> Optional[SpatialRelation]:
    """Compute the spatial relation between two presences."""
    # Same anchor = near each other
    if p1.anchor and p1.anchor == p2.anchor:
        return SpatialRelation(
            subject=p1.entity_id,
            object=p2.entity_id,
            relation="near",
            quality="at the same place"
        )

    # Different anchors = across/distant
    if p1.anchor and p2.anchor:
        return SpatialRelation(
            subject=p1.entity_id,
            object=p2.entity_id,
            relation="across the room from",
            quality="maintaining distance"
        )

    return None


def _detect_tensions(situation: Situation, events: List[Event]):
    """Detect narrative tensions from events."""
    # Look for unresolved social events
    for event in events:
        if event.type == EventType.SOCIAL:
            situation.tensions.append(Tension(
                description="Recent interaction still resonates",
                involves=[event.actor, event.object] if event.object else [event.actor],
                intensity=0.5,
                source_event=event.id
            ))

    # Look for prolonged silence after arrival
    arrivals = [e for e in events if e.type == EventType.ARRIVAL]
    speech = [e for e in events if e.type == EventType.SPEECH]

    for arrival in arrivals:
        # If someone arrived and hasn't spoken
        speaker_ids = {s.actor for s in speech if s.timestamp > arrival.timestamp}
        if arrival.actor not in speaker_ids:
            situation.tensions.append(Tension(
                description=f"{arrival.actor.title()} has been silent since arriving",
                involves=[arrival.actor],
                intensity=0.3,
                source_event=arrival.id
            ))

    # Look for frustrated states
    for presence in situation.presences.values():
        if presence.state and "frustrat" in presence.state.lower():
            situation.tensions.append(Tension(
                description=f"{presence.entity_id.title()}'s frustration is visible",
                involves=[presence.entity_id],
                intensity=0.6
            ))


def _build_atmosphere(situation: Situation, events: List[Event]):
    """Build atmosphere from events."""
    # Start with defaults
    situation.atmosphere = {
        "mood": "neutral",
        "energy": "calm"
    }

    # Environmental events modify atmosphere
    for event in events:
        if event.type == EventType.ENVIRONMENTAL:
            if event.detail:
                situation.atmosphere["recent_change"] = event.detail

        if event.type == EventType.ATMOSPHERIC:
            if event.detail:
                situation.atmosphere["mood"] = event.detail

    # Presences modify atmosphere
    if situation.tensions:
        situation.atmosphere["mood"] = "tense"

    presence_count = len(situation.presences)
    if presence_count == 0:
        situation.atmosphere["energy"] = "still, empty"
    elif presence_count == 1:
        situation.atmosphere["energy"] = "solitary"
    elif presence_count > 3:
        situation.atmosphere["energy"] = "busy"


def _update_awareness(experience: Experience, event: Event, perspective: str):
    """Update experience's awareness of another entity from an event."""
    other = event.actor

    if other not in experience.awareness_of:
        experience.awareness_of[other] = {}

    awareness = experience.awareness_of[other]

    # Get witness data for this perspective
    witness = event.get_witness(perspective)
    if witness:
        if witness.interpretation:
            awareness["interpretation"] = witness.interpretation

    # Update what we know about them
    if event.spatial and event.spatial.anchor:
        awareness["location"] = event.spatial.anchor
    if event.manner:
        awareness["state"] = event.manner

    awareness["last_observed"] = event.timestamp.isoformat()
    awareness["last_action"] = event.verb


def _update_awareness_of_object(experience: Experience, event: Event, perspective: str):
    """Update awareness when entity is the object of an event."""
    target = event.object
    if not target or target == perspective:
        return

    # Don't add stage_id as an entity awareness
    if event.spatial and target == event.spatial.stage_id:
        return

    if target not in experience.awareness_of:
        experience.awareness_of[target] = {}

    # The target was involved in something
    experience.awareness_of[target]["involved_in"] = event.verb
    experience.awareness_of[target]["last_observed"] = event.timestamp.isoformat()


__all__ = [
    # Data classes
    "Presence",
    "SpatialRelation",
    "Tension",
    "Situation",
    "Experience",

    # Projection functions
    "project_situation",
    "project_experience",
    "project_narrative",
]
