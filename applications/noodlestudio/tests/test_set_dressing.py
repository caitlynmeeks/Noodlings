# ------------------------------------------------------------------
#   Set Dressing Tests
#
#   Tests for stage environment data model, YAML I/O, and context
#   builder. Uses real tempdir + real YAML files (no mocks).
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_set_dressing
# PURPOSE:  Set Dressing Tests
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from noodlestudio.core.set_dressing import (
    SetObject, StageSet, BlockingMark,
    OpeningBeat, OpeningScene,
    load_set, save_set, load_marks, load_mark, save_mark,
    build_scene_context,
)
from noodlestudio.core.scene_node import SceneNodeType


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_set():
    """A small StageSet for testing."""
    return StageSet(
        name='Hearthwood Cafe',
        description='A small cafe with stone walls and warm lighting.',
        objects=[
            SetObject(id='fireplace', name='Stone Fireplace',
                      description='A large fireplace crackling with warmth.'),
            SetObject(id='counter', name='Counter',
                      description='Worn volcanic stone countertop.'),
            SetObject(id='bookshelf', name='Bookshelf',
                      description='Overstuffed with well-loved books.'),
        ],
    )


@pytest.fixture
def sample_mark():
    """A BlockingMark for testing."""
    return BlockingMark(
        id='behind_counter',
        name='Behind the Counter',
        perspective="You're behind the counter, polishing a glass.",
        can_see=['counter', 'fireplace'],
    )


@pytest.fixture
def stage_dir(sample_set, sample_mark):
    """A temp directory with set.yaml and Marks/ populated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write set.yaml
        save_set(tmpdir, sample_set)

        # Create Marks/ with two marks
        marks_dir = os.path.join(tmpdir, 'Marks')
        os.makedirs(marks_dir)

        save_mark(os.path.join(marks_dir, 'behind_counter.mark.yaml'), sample_mark)

        window_mark = BlockingMark(
            id='window_seat',
            name='Window Seat',
            perspective="You're by the window, watching the forest.",
            can_see=['bookshelf'],
        )
        save_mark(os.path.join(marks_dir, 'window_seat.mark.yaml'), window_mark)

        yield tmpdir


# =====================================================================
# Dataclass Construction
# =====================================================================

class TestDataclasses:
    """Verify dataclass construction and field access."""

    def test_set_object_fields(self):
        obj = SetObject(id='lamp', name='Table Lamp', description='A warm glow.')
        assert obj.id == 'lamp'
        assert obj.name == 'Table Lamp'
        assert obj.description == 'A warm glow.'

    def test_stage_set_fields(self, sample_set):
        assert sample_set.name == 'Hearthwood Cafe'
        assert len(sample_set.objects) == 3

    def test_blocking_mark_fields(self, sample_mark):
        assert sample_mark.id == 'behind_counter'
        assert sample_mark.name == 'Behind the Counter'
        assert len(sample_mark.can_see) == 2

    def test_blocking_mark_default_can_see(self):
        mark = BlockingMark(id='x', name='X', perspective='Here.')
        assert mark.can_see == []

    def test_blocking_mark_activity_field(self):
        mark = BlockingMark(
            id='x', name='X', perspective='Here.',
            activity='polishing a glass',
        )
        assert mark.activity == 'polishing a glass'

    def test_blocking_mark_default_activity_is_empty(self):
        mark = BlockingMark(id='x', name='X', perspective='Here.')
        assert mark.activity == ''


# =====================================================================
# Round-trip Serialization
# =====================================================================

class TestSerialization:
    """to_dict / from_dict round-trips."""

    def test_stage_set_round_trip(self, sample_set):
        d = sample_set.to_dict()
        restored = StageSet.from_dict(d)
        assert restored.name == sample_set.name
        assert restored.description == sample_set.description
        assert len(restored.objects) == len(sample_set.objects)
        for orig, rest in zip(sample_set.objects, restored.objects):
            assert orig.id == rest.id
            assert orig.name == rest.name
            assert orig.description == rest.description

    def test_blocking_mark_round_trip(self, sample_mark):
        d = sample_mark.to_dict()
        restored = BlockingMark.from_dict(d)
        assert restored.id == sample_mark.id
        assert restored.name == sample_mark.name
        assert restored.perspective == sample_mark.perspective
        assert restored.can_see == sample_mark.can_see

    def test_blocking_mark_activity_round_trip(self):
        mark = BlockingMark(
            id='x', name='X', perspective='Here.',
            activity='wiping down the counter',
        )
        d = mark.to_dict()
        assert d['activity'] == 'wiping down the counter'
        restored = BlockingMark.from_dict(d)
        assert restored.activity == 'wiping down the counter'

    def test_blocking_mark_empty_activity_omitted_from_dict(self, sample_mark):
        d = sample_mark.to_dict()
        assert 'activity' not in d

    def test_stage_set_get_object(self, sample_set):
        obj = sample_set.get_object('fireplace')
        assert obj is not None
        assert obj.name == 'Stone Fireplace'

    def test_stage_set_get_object_missing(self, sample_set):
        assert sample_set.get_object('nonexistent') is None


# =====================================================================
# YAML I/O
# =====================================================================

class TestYAMLIO:
    """Real file I/O with tempdir."""

    def test_load_save_set_round_trip(self, sample_set):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_set(tmpdir, sample_set)
            loaded = load_set(tmpdir)
            assert loaded is not None
            assert loaded.name == sample_set.name
            assert len(loaded.objects) == 3

    def test_load_set_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_set(tmpdir) is None

    def test_load_marks_round_trip(self, stage_dir):
        marks = load_marks(stage_dir)
        assert len(marks) == 2
        ids = {m.id for m in marks}
        assert 'behind_counter' in ids
        assert 'window_seat' in ids

    def test_load_marks_missing_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_marks(tmpdir) == []

    def test_save_load_single_mark(self, sample_mark):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.mark.yaml')
            save_mark(path, sample_mark)
            loaded = load_mark(path)
            assert loaded.id == sample_mark.id
            assert loaded.perspective == sample_mark.perspective
            assert loaded.can_see == sample_mark.can_see

    def test_saved_yaml_is_readable(self, sample_set):
        """Verify the YAML output is valid and human-readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_set(tmpdir, sample_set)
            with open(os.path.join(tmpdir, 'set.yaml')) as f:
                raw = yaml.safe_load(f)
            assert raw['name'] == 'Hearthwood Cafe'
            assert len(raw['objects']) == 3


# =====================================================================
# Context Builder
# =====================================================================

class TestBuildSceneContext:
    """build_scene_context() output formatting."""

    def test_full_context_with_mark(self, sample_set, sample_mark):
        others = [
            {'name': 'Juanita', 'mark_name': 'Window Seat'},
            {'name': 'Krampus', 'mark_name': 'By the Fire'},
        ]
        ctx = build_scene_context(sample_set, sample_mark, others)

        assert 'THE SPACE:' in ctx
        assert 'A small cafe' in ctx
        assert 'WHERE YOU ARE:' in ctx
        assert 'polishing a glass' in ctx
        assert 'WHAT YOU CAN SEE:' in ctx
        assert 'Counter:' in ctx
        assert 'Stone Fireplace:' in ctx
        # Bookshelf NOT in can_see
        assert 'Bookshelf:' not in ctx
        assert 'WHO IS HERE:' in ctx
        assert 'Juanita is at Window Seat.' in ctx

    def test_context_no_mark_shows_all_objects(self, sample_set):
        ctx = build_scene_context(sample_set, None)
        assert 'AROUND YOU:' in ctx
        assert 'Stone Fireplace:' in ctx
        assert 'Counter:' in ctx
        assert 'Bookshelf:' in ctx
        assert 'WHERE YOU ARE:' not in ctx

    def test_context_no_set_returns_empty(self):
        ctx = build_scene_context(None, None)
        assert ctx == ''

    def test_context_empty_can_see(self, sample_set):
        mark = BlockingMark(id='empty', name='Empty', perspective='Nowhere.',
                            can_see=[])
        ctx = build_scene_context(sample_set, mark)
        assert 'WHERE YOU ARE:' in ctx
        assert 'WHAT YOU CAN SEE:' not in ctx

    def test_context_no_others(self, sample_set, sample_mark):
        ctx = build_scene_context(sample_set, sample_mark, None)
        assert 'WHO IS HERE:' not in ctx

    def test_context_other_without_mark_name(self, sample_set, sample_mark):
        others = [{'name': 'Ghost'}]
        ctx = build_scene_context(sample_set, sample_mark, others)
        assert 'Ghost is here.' in ctx


# =====================================================================
# Opening Scene Data Model
# =====================================================================

class TestOpeningBeat:
    """OpeningBeat construction and round-trip."""

    def test_cue_beat_construction(self):
        beat = OpeningBeat(beat_type='cue', noodling='ajo', cue='Look up')
        assert beat.beat_type == 'cue'
        assert beat.noodling == 'ajo'
        assert beat.cue == 'Look up'

    def test_cue_beat_round_trip(self):
        beat = OpeningBeat(beat_type='cue', noodling='ajo', cue='Greet them')
        d = beat.to_dict()
        assert d == {'noodling': 'ajo', 'cue': 'Greet them'}
        restored = OpeningBeat.from_dict(d)
        assert restored.beat_type == 'cue'
        assert restored.noodling == 'ajo'
        assert restored.cue == 'Greet them'

    def test_narration_beat_round_trip(self):
        beat = OpeningBeat(beat_type='narration', text='The bells ring.')
        d = beat.to_dict()
        assert d == {'type': 'narration', 'text': 'The bells ring.'}
        restored = OpeningBeat.from_dict(d)
        assert restored.beat_type == 'narration'
        assert restored.text == 'The bells ring.'

    def test_pause_beat_round_trip(self):
        beat = OpeningBeat(beat_type='pause', duration=2.5)
        d = beat.to_dict()
        assert d == {'type': 'pause', 'duration': 2.5}
        restored = OpeningBeat.from_dict(d)
        assert restored.beat_type == 'pause'
        assert restored.duration == 2.5

    def test_default_beat_type_is_cue(self):
        beat = OpeningBeat()
        assert beat.beat_type == 'cue'


class TestOpeningScene:
    """OpeningScene construction and round-trip."""

    def test_default_mode_is_silent(self):
        scene = OpeningScene()
        assert scene.mode == 'silent'
        assert scene.beats == []
        assert scene.narration == ''

    def test_live_mode_round_trip(self):
        scene = OpeningScene(
            mode='live',
            beats=[
                OpeningBeat(beat_type='cue', noodling='ajo', cue='Greet'),
                OpeningBeat(beat_type='pause', duration=1.5),
                OpeningBeat(beat_type='narration', text='Bells ring.'),
            ],
        )
        d = scene.to_dict()
        assert d['mode'] == 'live'
        assert len(d['beats']) == 3
        restored = OpeningScene.from_dict(d)
        assert restored.mode == 'live'
        assert len(restored.beats) == 3
        assert restored.beats[0].beat_type == 'cue'
        assert restored.beats[1].beat_type == 'pause'
        assert restored.beats[2].beat_type == 'narration'

    def test_narrated_mode_round_trip(self):
        scene = OpeningScene(mode='narrated', narration='Once upon a time.')
        d = scene.to_dict()
        restored = OpeningScene.from_dict(d)
        assert restored.mode == 'narrated'
        assert restored.narration == 'Once upon a time.'

    def test_stage_set_with_opening_round_trip(self, sample_set):
        opening = OpeningScene(
            mode='live',
            beats=[
                OpeningBeat(beat_type='cue', noodling='ajo', cue='Wave'),
            ],
        )
        sample_set.opening = opening
        d = sample_set.to_dict()
        assert 'opening' in d
        restored = StageSet.from_dict(d)
        assert restored.opening is not None
        assert restored.opening.mode == 'live'
        assert len(restored.opening.beats) == 1

    def test_stage_set_without_opening_loads_as_none(self, sample_set):
        """Backward compat: sets without opening key load as None."""
        d = sample_set.to_dict()
        assert 'opening' not in d
        restored = StageSet.from_dict(d)
        assert restored.opening is None

    def test_stage_set_with_opening_yaml_round_trip(self, sample_set):
        """Opening survives save_set/load_set cycle."""
        opening = OpeningScene(
            mode='live',
            beats=[
                OpeningBeat(beat_type='cue', noodling='ajo', cue='Nod'),
                OpeningBeat(beat_type='pause', duration=1.0),
            ],
        )
        sample_set.opening = opening
        with tempfile.TemporaryDirectory() as tmpdir:
            save_set(tmpdir, sample_set)
            loaded = load_set(tmpdir)
            assert loaded.opening is not None
            assert loaded.opening.mode == 'live'
            assert len(loaded.opening.beats) == 2
            assert loaded.opening.beats[0].noodling == 'ajo'
            assert loaded.opening.beats[1].duration == 1.0


# =====================================================================
# SceneNodeType Enum
# =====================================================================

class TestSceneNodeType:
    """Verify SET and BLOCKING_MARK exist in the enum."""

    def test_set_type_exists(self):
        assert SceneNodeType.SET.value == 'set'

    def test_blocking_mark_type_exists(self):
        assert SceneNodeType.BLOCKING_MARK.value == 'blocking_mark'


# =====================================================================
# Template Verification (Hearthwood Cafe)
# =====================================================================

TEMPLATE_STAGE = os.path.join(
    os.path.dirname(__file__), '..', 'library', 'templates',
    'Getting Started', 'Stages', 'the_nexus',
)


class TestHearthwoodCafeTemplate:
    """Verify the default Getting Started template has valid set dressing."""

    def test_set_loads_with_7_objects(self):
        stage_set = load_set(TEMPLATE_STAGE)
        assert stage_set is not None
        assert stage_set.name == 'Hearthwood Cafe'
        assert len(stage_set.objects) == 7

    def test_all_object_ids_unique(self):
        stage_set = load_set(TEMPLATE_STAGE)
        ids = [o.id for o in stage_set.objects]
        assert len(ids) == len(set(ids))

    def test_three_marks_exist(self):
        marks = load_marks(TEMPLATE_STAGE)
        assert len(marks) == 3
        ids = {m.id for m in marks}
        assert ids == {'behind_counter', 'window_seat', 'by_the_fire'}

    def test_mark_can_see_refs_are_valid(self):
        stage_set = load_set(TEMPLATE_STAGE)
        marks = load_marks(TEMPLATE_STAGE)
        valid_ids = {o.id for o in stage_set.objects}
        for mark in marks:
            for obj_id in mark.can_see:
                assert obj_id in valid_ids, \
                    f"Mark '{mark.id}' references unknown object '{obj_id}'"

    def test_instance_mark_refs_are_valid(self):
        marks = load_marks(TEMPLATE_STAGE)
        valid_mark_ids = {m.id for m in marks}
        instances_dir = os.path.join(TEMPLATE_STAGE, 'Instances')
        for name in ('ajo', 'juanita', 'krampus'):
            inst_yaml = os.path.join(instances_dir, name, 'instance.yaml')
            with open(inst_yaml) as f:
                data = yaml.safe_load(f)
            mark = data.get('overrides', {}).get('mark', '')
            assert mark in valid_mark_ids, \
                f"Instance '{name}' has invalid mark '{mark}'"

    def test_ajo_sees_everything(self):
        """Ajo (barista) should see all 7 objects."""
        marks = load_marks(TEMPLATE_STAGE)
        ajo_mark = next(m for m in marks if m.id == 'behind_counter')
        assert len(ajo_mark.can_see) == 7

    def test_default_opening_loads_with_correct_mode(self):
        """Default set.yaml opening loads as 'live' mode."""
        stage_set = load_set(TEMPLATE_STAGE)
        assert stage_set.opening is not None
        assert stage_set.opening.mode == 'live'
        assert len(stage_set.opening.beats) == 8

    def test_opening_beat_noodling_refs_are_valid_instances(self):
        """Beat noodling references must match valid instance IDs."""
        stage_set = load_set(TEMPLATE_STAGE)
        instances_dir = os.path.join(TEMPLATE_STAGE, 'Instances')
        valid_ids = set()
        if os.path.isdir(instances_dir):
            for name in os.listdir(instances_dir):
                if os.path.isdir(os.path.join(instances_dir, name)):
                    valid_ids.add(name)
        for beat in stage_set.opening.beats:
            if beat.beat_type == 'cue':
                assert beat.noodling in valid_ids, \
                    f"Beat references unknown noodling '{beat.noodling}'"

    def test_mark_activities_are_nonempty(self):
        """All marks should have non-empty activity strings."""
        marks = load_marks(TEMPLATE_STAGE)
        for mark in marks:
            assert mark.activity.strip(), \
                f"Mark '{mark.id}' has empty activity"
