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
#   Semantic World Event System
#
#   Events are the atomic unit of reality. The universe is no...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.semantic_world.event
# PURPOSE:  Semantic World Event System
# LAYER:    Studio / Semantic World
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   EventType, Witness, Effect, SpatialContext, Event
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from uuid import uuid4
import json


class EventType(Enum):
    """
    The fundamental types of happenings in the world.

    Everything that occurs falls into one of these categories.
    """

    # Communication
    SPEECH = "speech"           # "red said 'hello'"
    EMOTE = "emote"             # "red smiled warmly"

    # Physical
    ACTION = "action"           # "servnak slammed the mechanism"
    MOVEMENT = "movement"       # "red drifted toward the fire"
    GESTURE = "gesture"         # "red waved dismissively"

    # Mental/Perceptual
    PERCEPTION = "perception"   # "red noticed servnak's frustration"
    INTERNAL = "internal"       # "red felt a pang of guilt"
    ATTENTION = "attention"     # "red's focus shifted to the door"

    # Environmental
    ENVIRONMENTAL = "environmental"  # "the fire crackled louder"
    ATMOSPHERIC = "atmospheric"      # "shadows deepened in the corners"
    TEMPORAL = "temporal"            # "an hour passed in silence"

    # Existential
    CREATION = "creation"       # "servnak placed a crystal on the bench"
    DESTRUCTION = "destruction" # "the mechanism shattered"
    ARRIVAL = "arrival"         # "red entered the nexus"
    DEPARTURE = "departure"     # "servnak left without a word"

    # Relational
    SOCIAL = "social"           # "red and servnak made eye contact"
    RELATIONSHIP = "relationship"  # "trust deepened between them"


@dataclass
class Witness:
    """
    How an entity perceived an event.

    Not all witnesses perceive events equally. Attention, interpretation,
    and emotional response vary by the perceiver.
    """

    entity_id: str

    # Did they notice at all?
    noticed: bool = True

    # How much attention did they give it? (0.0 = peripheral, 1.0 = full focus)
    attention: float = 0.5

    # Their interpretation of what happened (may differ from "truth")
    interpretation: Optional[str] = None

    # Emotional response evoked
    emotional_response: Optional[str] = None

    # Did this event change their state?
    state_change: Optional[str] = None


@dataclass
class Effect:
    """
    A change in the world caused by an event.

    Events ripple outward, changing presences, relationships,
    atmosphere, and the stage itself.
    """

    # What kind of change
    type: str  # "presence", "atmosphere", "relationship", "stage", "state"

    # What entity/aspect is affected
    target: str

    # The change itself
    change: str

    # Previous value (for potential rollback/history)
    previous: Optional[str] = None


@dataclass
class SpatialContext:
    """
    Where an event happened, in semantic-relational terms.

    Not coordinates. Meaning.
    """

    # The stage/room where this occurred
    stage_id: str

    # The anchor point (named meaningful location)
    anchor: Optional[str] = None

    # Zone the event occurred in
    zone: Optional[str] = None

    # Relation to other entities ("near the fireplace", "across from servnak")
    relation: Optional[str] = None


@dataclass
class Event:
    """
    The atomic unit of reality.

    An event is a happening - something that occurred at a moment in time,
    involving an actor, witnessed by others, rippling effects outward.

    Events are immutable once created. They are facts of history.
    The event log is the source of truth from which all state is projected.

    Example:
        Event(
            type=EventType.MOVEMENT,
            actor="red",
            verb="entered",
            object="the_nexus",
            origin="garden",
            manner="slowly, hesitantly",
            detail="paused at threshold, hand trailing on doorframe",
            subtext="returning from solitude, not ready for company",
            emotional_color="reluctant, seeking warmth"
        )
    """

    # Identity
    id: str = field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # The happening itself
    type: EventType = EventType.ACTION

    # Who initiated this event (entity_id, or "environment" for environmental events)
    actor: str = ""

    # What happened (active voice verb)
    verb: str = ""

    # Target/destination/recipient (optional)
    object: Optional[str] = None

    # With what instrument/means (optional)
    instrument: Optional[str] = None

    # From where (for movement/arrival)
    origin: Optional[str] = None

    # To where (for movement/departure)
    destination: Optional[str] = None

    # The texture - HOW it happened
    manner: Optional[str] = None  # "slowly", "angrily", "with hesitation"
    detail: Optional[str] = None  # Specific observable details

    # The meaning - WHY / deeper significance
    subtext: Optional[str] = None  # The unspoken meaning
    emotional_color: Optional[str] = None  # The emotional tone

    # Where it happened
    spatial: Optional[SpatialContext] = None

    # Who perceived it and how
    witnesses: List[Witness] = field(default_factory=list)

    # What changed as a result
    effects: List[Effect] = field(default_factory=list)

    # For speech events - the actual words
    content: Optional[str] = None

    # For internal events - whose internal state
    subject: Optional[str] = None

    # Arbitrary metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize the event."""
        # Ensure type is EventType enum
        if isinstance(self.type, str):
            self.type = EventType(self.type)

        # Actor always witnesses their own event (unless environmental)
        if self.actor and self.actor != "environment":
            if not any(w.entity_id == self.actor for w in self.witnesses):
                self.witnesses.insert(0, Witness(
                    entity_id=self.actor,
                    noticed=True,
                    attention=1.0,  # Full attention to your own actions
                    interpretation=None,  # Actor knows what they did
                    emotional_response=None  # Set separately if needed
                ))

    # ─────────────────────────────────────────────────────────────
    # Narrative rendering
    # ─────────────────────────────────────────────────────────────

    def narrate(self, perspective: Optional[str] = None) -> str:
        """
        Render this event as natural language.

        If perspective is provided, narrate from that entity's point of view.
        Otherwise, narrate objectively.
        """
        if perspective:
            return self._narrate_perspectival(perspective)
        return self._narrate_objective()

    def _narrate_objective(self) -> str:
        """Third-person objective narration."""
        parts = []

        # Actor + verb
        actor_name = self.actor.replace("_", " ").title()
        parts.append(f"{actor_name} {self.verb}")

        # Object if present
        if self.object:
            obj_name = self.object.replace("_", " ")
            parts.append(obj_name)

        # Origin/destination for movement
        if self.origin:
            parts.append(f"from {self.origin.replace('_', ' ')}")
        if self.destination:
            parts.append(f"to {self.destination.replace('_', ' ')}")

        base = " ".join(parts)

        # Add manner
        if self.manner:
            base = f"{base}, {self.manner}"

        # Add detail as separate sentence
        if self.detail:
            base = f"{base}. {self.detail.capitalize()}"

        return base + "."

    def _narrate_perspectival(self, perspective: str) -> str:
        """First/second person narration from a specific perspective."""
        # Find this entity's witness record
        witness = next(
            (w for w in self.witnesses if w.entity_id == perspective),
            None
        )

        # If they didn't witness it, they can't narrate it
        if not witness or not witness.noticed:
            return ""

        # Determine if actor or observer
        if self.actor == perspective:
            return self._narrate_as_actor()
        else:
            return self._narrate_as_observer(witness)

    def _narrate_as_actor(self) -> str:
        """Narrate from the perspective of the one who did it."""
        parts = []

        # Second person for the actor
        parts.append(f"You {self.verb}")

        if self.object:
            obj_name = self.object.replace("_", " ")
            parts.append(obj_name)

        if self.origin:
            parts.append(f"from {self.origin.replace('_', ' ')}")
        if self.destination:
            parts.append(f"to {self.destination.replace('_', ' ')}")

        base = " ".join(parts)

        if self.manner:
            base = f"{base}, {self.manner}"

        # Include subtext for actor (they know why)
        if self.subtext:
            base = f"{base}. {self.subtext.capitalize()}"

        return base + "."

    def _narrate_as_observer(self, witness: Witness) -> str:
        """Narrate from the perspective of someone who observed."""
        parts = []

        actor_name = self.actor.replace("_", " ").title()
        parts.append(f"{actor_name} {self.verb}")

        if self.object:
            obj_name = self.object.replace("_", " ")
            parts.append(obj_name)

        base = " ".join(parts)

        if self.manner:
            base = f"{base}, {self.manner}"

        # Use witness's interpretation instead of objective detail
        if witness.interpretation:
            base = f"{base}. {witness.interpretation.capitalize()}"
        elif self.detail:
            base = f"{base}. {self.detail.capitalize()}"

        # Add emotional response if strong
        if witness.emotional_response and witness.attention > 0.5:
            base = f"{base} ({witness.emotional_response})"

        return base + "."

    # ─────────────────────────────────────────────────────────────
    # Witness management
    # ─────────────────────────────────────────────────────────────

    def add_witness(
        self,
        entity_id: str,
        noticed: bool = True,
        attention: float = 0.5,
        interpretation: Optional[str] = None,
        emotional_response: Optional[str] = None
    ) -> 'Event':
        """Add a witness to this event. Returns self for chaining."""
        self.witnesses.append(Witness(
            entity_id=entity_id,
            noticed=noticed,
            attention=attention,
            interpretation=interpretation,
            emotional_response=emotional_response
        ))
        return self

    def witnessed_by(self, entity_id: str) -> bool:
        """Check if an entity witnessed this event."""
        return any(w.entity_id == entity_id and w.noticed for w in self.witnesses)

    def get_witness(self, entity_id: str) -> Optional[Witness]:
        """Get a specific witness's perception of this event."""
        return next((w for w in self.witnesses if w.entity_id == entity_id), None)

    # ─────────────────────────────────────────────────────────────
    # Effect management
    # ─────────────────────────────────────────────────────────────

    def add_effect(
        self,
        effect_type: str,
        target: str,
        change: str,
        previous: Optional[str] = None
    ) -> 'Event':
        """Add an effect caused by this event. Returns self for chaining."""
        self.effects.append(Effect(
            type=effect_type,
            target=target,
            change=change,
            previous=previous
        ))
        return self

    # ─────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type.value,
            "actor": self.actor,
            "verb": self.verb,
            "object": self.object,
            "instrument": self.instrument,
            "origin": self.origin,
            "destination": self.destination,
            "manner": self.manner,
            "detail": self.detail,
            "subtext": self.subtext,
            "emotional_color": self.emotional_color,
            "spatial": {
                "stage_id": self.spatial.stage_id,
                "anchor": self.spatial.anchor,
                "zone": self.spatial.zone,
                "relation": self.spatial.relation
            } if self.spatial else None,
            "witnesses": [
                {
                    "entity_id": w.entity_id,
                    "noticed": w.noticed,
                    "attention": w.attention,
                    "interpretation": w.interpretation,
                    "emotional_response": w.emotional_response,
                    "state_change": w.state_change
                }
                for w in self.witnesses
            ],
            "effects": [
                {
                    "type": e.type,
                    "target": e.target,
                    "change": e.change,
                    "previous": e.previous
                }
                for e in self.effects
            ],
            "content": self.content,
            "subject": self.subject,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Deserialize from dictionary."""
        spatial = None
        if data.get("spatial"):
            spatial = SpatialContext(**data["spatial"])

        witnesses = [
            Witness(**w) for w in data.get("witnesses", [])
        ]

        effects = [
            Effect(**e) for e in data.get("effects", [])
        ]

        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            type=EventType(data["type"]),
            actor=data.get("actor", ""),
            verb=data.get("verb", ""),
            object=data.get("object"),
            instrument=data.get("instrument"),
            origin=data.get("origin"),
            destination=data.get("destination"),
            manner=data.get("manner"),
            detail=data.get("detail"),
            subtext=data.get("subtext"),
            emotional_color=data.get("emotional_color"),
            spatial=spatial,
            witnesses=witnesses,
            effects=effects,
            content=data.get("content"),
            subject=data.get("subject"),
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ═══════════════════════════════════════════════════════════════════════════════
# Event Factory Functions - Convenient constructors for common event types
# ═══════════════════════════════════════════════════════════════════════════════

def speech_event(
    speaker: str,
    content: str,
    stage_id: str,
    manner: Optional[str] = None,
    witnesses: Optional[List[str]] = None
) -> Event:
    """Create a speech event."""
    event = Event(
        type=EventType.SPEECH,
        actor=speaker,
        verb="said",
        content=content,
        manner=manner,
        spatial=SpatialContext(stage_id=stage_id)
    )

    # Add witnesses
    for w in (witnesses or []):
        event.add_witness(w, attention=0.8)

    return event


def movement_event(
    mover: str,
    destination: str,
    stage_id: str,
    origin: Optional[str] = None,
    manner: Optional[str] = None,
    detail: Optional[str] = None,
    subtext: Optional[str] = None
) -> Event:
    """Create a movement event."""
    return Event(
        type=EventType.MOVEMENT,
        actor=mover,
        verb="moved",
        destination=destination,
        origin=origin,
        manner=manner,
        detail=detail,
        subtext=subtext,
        spatial=SpatialContext(stage_id=stage_id, anchor=destination)
    )


def arrival_event(
    arriver: str,
    stage_id: str,
    from_stage: Optional[str] = None,
    manner: Optional[str] = None,
    detail: Optional[str] = None
) -> Event:
    """Create an arrival event (entering a stage)."""
    return Event(
        type=EventType.ARRIVAL,
        actor=arriver,
        verb="arrived",
        object=stage_id,
        origin=from_stage,
        manner=manner,
        detail=detail,
        spatial=SpatialContext(stage_id=stage_id)
    )


def departure_event(
    departer: str,
    stage_id: str,
    to_stage: Optional[str] = None,
    manner: Optional[str] = None
) -> Event:
    """Create a departure event (leaving a stage)."""
    return Event(
        type=EventType.DEPARTURE,
        actor=departer,
        verb="departed",
        object=stage_id,
        destination=to_stage,
        manner=manner,
        spatial=SpatialContext(stage_id=stage_id)
    )


def action_event(
    actor: str,
    verb: str,
    target: Optional[str],
    stage_id: str,
    manner: Optional[str] = None,
    detail: Optional[str] = None,
    instrument: Optional[str] = None
) -> Event:
    """Create a general action event."""
    return Event(
        type=EventType.ACTION,
        actor=actor,
        verb=verb,
        object=target,
        instrument=instrument,
        manner=manner,
        detail=detail,
        spatial=SpatialContext(stage_id=stage_id)
    )


def perception_event(
    perceiver: str,
    perceived: str,
    observation: str,
    stage_id: str,
    interpretation: Optional[str] = None
) -> Event:
    """Create a perception event (noticing something)."""
    return Event(
        type=EventType.PERCEPTION,
        actor=perceiver,
        verb="noticed",
        object=perceived,
        detail=observation,
        subtext=interpretation,
        spatial=SpatialContext(stage_id=stage_id)
    )


def internal_event(
    subject: str,
    feeling: str,
    stage_id: str,
    trigger: Optional[str] = None
) -> Event:
    """Create an internal event (thought, feeling)."""
    return Event(
        type=EventType.INTERNAL,
        actor=subject,
        subject=subject,
        verb="felt",
        content=feeling,
        object=trigger,
        spatial=SpatialContext(stage_id=stage_id)
    )


def environmental_event(
    description: str,
    stage_id: str,
    anchor: Optional[str] = None
) -> Event:
    """Create an environmental event (something in the world changes)."""
    return Event(
        type=EventType.ENVIRONMENTAL,
        actor="environment",
        verb="shifted",
        detail=description,
        spatial=SpatialContext(stage_id=stage_id, anchor=anchor)
    )


def social_event(
    participants: List[str],
    interaction: str,
    stage_id: str,
    detail: Optional[str] = None,
    emotional_color: Optional[str] = None
) -> Event:
    """Create a social event (interaction between entities)."""
    return Event(
        type=EventType.SOCIAL,
        actor=participants[0] if participants else "unknown",
        verb=interaction,
        object=participants[1] if len(participants) > 1 else None,
        detail=detail,
        emotional_color=emotional_color,
        spatial=SpatialContext(stage_id=stage_id),
        metadata={"participants": participants}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Module initialization
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core classes
    "Event",
    "EventType",
    "Witness",
    "Effect",
    "SpatialContext",

    # Factory functions
    "speech_event",
    "movement_event",
    "arrival_event",
    "departure_event",
    "action_event",
    "perception_event",
    "internal_event",
    "environmental_event",
    "social_event",
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
