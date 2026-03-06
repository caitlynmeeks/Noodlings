# ------------------------------------------------------------------
#   Set Dressing - Stage environment and blocking marks
#
#   Defines the physical space noodlings inhabit. A StageSet describes
#   the environment (objects, atmosphere). BlockingMarks assign each
#   noodling a position with a unique perspective and visibility list.
#
#   Together they give every noodling a sense of place: same cafe,
#   different views.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.core.set_dressing
# PURPOSE:  Set Dressing
# LAYER:    Studio / Core
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Optional
import os
import yaml


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class SetObject:
    """A named object in the stage environment.

    Each object has an id (for machine reference), a human-readable
    name, and a prose description that gets injected into LLM context.
    """
    id: str
    name: str
    description: str


@dataclass
class OpeningBeat:
    """A single beat in the opening scene.

    Three beat types:
    - 'cue': An improv direction for a specific noodling. The noodling
      runs its assembly with the cue as brenda_direction.
    - 'narration': Authored text displayed without speaker attribution.
    - 'pause': A timed pause between beats.
    """
    beat_type: str = 'cue'
    noodling: str = ''
    cue: str = ''
    text: str = ''
    duration: float = 1.0

    def to_dict(self) -> dict:
        """Serialize to a plain dict for YAML output."""
        if self.beat_type == 'cue':
            d = {'noodling': self.noodling, 'cue': self.cue}
        elif self.beat_type == 'narration':
            d = {'type': 'narration', 'text': self.text}
        elif self.beat_type == 'pause':
            d = {'type': 'pause', 'duration': self.duration}
        else:
            d = {'type': self.beat_type}
        return d

    @staticmethod
    def from_dict(data: dict) -> 'OpeningBeat':
        """Deserialize from a plain dict (YAML input).

        Detection logic:
        - Has 'noodling' key -> cue beat
        - Has type: narration -> narration beat
        - Has type: pause -> pause beat
        """
        if 'noodling' in data:
            return OpeningBeat(
                beat_type='cue',
                noodling=data.get('noodling', ''),
                cue=data.get('cue', ''),
            )
        beat_type = data.get('type', '')
        if beat_type == 'narration':
            return OpeningBeat(
                beat_type='narration',
                text=data.get('text', ''),
            )
        if beat_type == 'pause':
            return OpeningBeat(
                beat_type='pause',
                duration=float(data.get('duration', 1.0)),
            )
        return OpeningBeat(beat_type=beat_type or 'cue')


@dataclass
class OpeningScene:
    """Opening scene configuration on a StageSet.

    Three modes:
    - 'silent': No opening (current default behavior).
    - 'live': Noodlings improvise from cue beats sequentially.
    - 'narrated': Authored narration text displayed before interaction.
    """
    mode: str = 'silent'
    narration: str = ''
    beats: List[OpeningBeat] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for YAML output."""
        d = {'mode': self.mode}
        if self.narration:
            d['narration'] = self.narration
        if self.beats:
            d['beats'] = [b.to_dict() for b in self.beats]
        return d

    @staticmethod
    def from_dict(data: dict) -> 'OpeningScene':
        """Deserialize from a plain dict (YAML input)."""
        beats = [
            OpeningBeat.from_dict(b) for b in data.get('beats', [])
        ]
        return OpeningScene(
            mode=data.get('mode', 'silent'),
            narration=data.get('narration', ''),
            beats=beats,
        )


@dataclass
class StageSet:
    """The environment definition for a stage.

    Contains a description of the overall space, a list of scene
    objects that noodlings can perceive, and an optional opening
    scene that plays before user interaction begins.
    """
    name: str
    description: str
    objects: List[SetObject] = field(default_factory=list)
    opening: Optional[OpeningScene] = None

    def get_object(self, obj_id: str) -> Optional[SetObject]:
        """Look up a scene object by id."""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def to_dict(self) -> dict:
        """Serialize to a plain dict for YAML output."""
        d = {
            'name': self.name,
            'description': self.description,
            'objects': [
                {'id': o.id, 'name': o.name, 'description': o.description}
                for o in self.objects
            ],
        }
        if self.opening is not None:
            d['opening'] = self.opening.to_dict()
        return d

    @staticmethod
    def from_dict(data: dict) -> 'StageSet':
        """Deserialize from a plain dict (YAML input)."""
        objects = [
            SetObject(
                id=o['id'],
                name=o['name'],
                description=o.get('description', ''),
            )
            for o in data.get('objects', [])
        ]
        opening = None
        if 'opening' in data:
            opening = OpeningScene.from_dict(data['opening'])
        return StageSet(
            name=data.get('name', ''),
            description=data.get('description', ''),
            objects=objects,
            opening=opening,
        )


@dataclass
class BlockingMark:
    """A named position in the stage with a unique perspective.

    Each noodling is assigned to a mark. The mark defines what they
    see (can_see list of object IDs) and their first-person perspective
    prose. The optional activity describes what the noodling is doing
    at this mark (used in opening scene context).
    """
    id: str
    name: str
    perspective: str
    can_see: List[str] = field(default_factory=list)
    activity: str = ''

    def to_dict(self) -> dict:
        """Serialize to a plain dict for YAML output."""
        d = {
            'id': self.id,
            'name': self.name,
            'perspective': self.perspective,
            'can_see': list(self.can_see),
        }
        if self.activity:
            d['activity'] = self.activity
        return d

    @staticmethod
    def from_dict(data: dict) -> 'BlockingMark':
        """Deserialize from a plain dict (YAML input)."""
        return BlockingMark(
            id=data.get('id', ''),
            name=data.get('name', ''),
            perspective=data.get('perspective', ''),
            can_see=data.get('can_see', []),
            activity=data.get('activity', ''),
        )


# =====================================================================
# YAML I/O
# =====================================================================

def load_set(stage_path: str) -> Optional[StageSet]:
    """Load set.yaml from a stage directory.

    Returns None if set.yaml does not exist (backward-compatible
    with stages that have no set dressing).
    """
    set_yaml = os.path.join(stage_path, 'set.yaml')
    if not os.path.exists(set_yaml):
        return None

    with open(set_yaml, 'r') as f:
        data = yaml.safe_load(f) or {}

    return StageSet.from_dict(data)


def save_set(stage_path: str, stage_set: StageSet) -> None:
    """Write set.yaml to a stage directory."""
    set_yaml = os.path.join(stage_path, 'set.yaml')
    with open(set_yaml, 'w') as f:
        yaml.dump(stage_set.to_dict(), f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


def load_marks(stage_path: str) -> List[BlockingMark]:
    """Load all blocking marks from Marks/*.mark.yaml.

    Returns an empty list if the Marks directory does not exist.
    """
    marks_dir = os.path.join(stage_path, 'Marks')
    if not os.path.isdir(marks_dir):
        return []

    marks = []
    for filename in sorted(os.listdir(marks_dir)):
        if not filename.endswith('.mark.yaml'):
            continue
        mark = load_mark(os.path.join(marks_dir, filename))
        marks.append(mark)

    return marks


def load_mark(path: str) -> BlockingMark:
    """Load a single blocking mark from a .mark.yaml file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return BlockingMark.from_dict(data)


def save_mark(path: str, mark: BlockingMark) -> None:
    """Write a single blocking mark to a .mark.yaml file."""
    with open(path, 'w') as f:
        yaml.dump(mark.to_dict(), f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


# =====================================================================
# Context Builder
# =====================================================================

def build_scene_context(
    stage_set: Optional[StageSet],
    mark: Optional[BlockingMark],
    other_noodlings: Optional[List[dict]] = None,
) -> str:
    """Assemble formatted scene context for LLM injection.

    Args:
        stage_set: The stage's environment definition (None = no set)
        mark: The noodling's assigned blocking mark (None = no assignment)
        other_noodlings: List of dicts with 'name' and optional 'mark_name'

    Returns:
        Formatted context string. Empty string if stage_set is None.

    Output format (with mark):
        THE SPACE:
        <set description>

        WHERE YOU ARE:
        <mark perspective>

        WHAT YOU CAN SEE:
        - <object name>: <object description>
        ...

        WHO IS HERE:
        - <name> is at <mark_name>.
        ...

    Output format (without mark):
        THE SPACE:
        <set description>

        AROUND YOU:
        - <object name>: <object description>
        ...

        WHO IS HERE:
        - <name> is at <mark_name>.
        ...
    """
    if stage_set is None:
        return ''

    lines = []

    # THE SPACE
    lines.append('THE SPACE:')
    lines.append(stage_set.description)

    if mark:
        # WHERE YOU ARE
        lines.append('')
        lines.append('WHERE YOU ARE:')
        lines.append(mark.perspective)

        # WHAT YOU CAN SEE (only objects in can_see)
        visible_objects = []
        for obj_id in mark.can_see:
            obj = stage_set.get_object(obj_id)
            if obj:
                visible_objects.append(obj)

        if visible_objects:
            lines.append('')
            lines.append('WHAT YOU CAN SEE:')
            for obj in visible_objects:
                lines.append(f'- {obj.name}: {obj.description}')
    else:
        # No mark -- show all objects
        if stage_set.objects:
            lines.append('')
            lines.append('AROUND YOU:')
            for obj in stage_set.objects:
                lines.append(f'- {obj.name}: {obj.description}')

    # WHO IS HERE
    if other_noodlings:
        lines.append('')
        lines.append('WHO IS HERE:')
        for other in other_noodlings:
            name = other.get('name', 'Someone')
            mark_name = other.get('mark_name', '')
            if mark_name:
                lines.append(f'- {name} is at {mark_name}.')
            else:
                lines.append(f'- {name} is here.')

    return '\n'.join(lines)
