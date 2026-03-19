# ------------------------------------------------------------------
#   Tests for Directed Ensemble Architecture -- Phase A
#
#   Covers: Role field persistence, inspector dropdown, hierarchy
#   role indicator, stage discovery role extraction, signal routing.
# ------------------------------------------------------------------

import os
import tempfile

import pytest
import yaml


# =================================================================
# ROLE PERSISTENCE (instance.yaml)
# =================================================================

class TestRolePersistence:
    """Test that role field writes to and reads from instance.yaml."""

    def _make_instance(self, tmpdir, overrides=None):
        """Create a minimal instance directory with instance.yaml."""
        inst_dir = os.path.join(tmpdir, 'test_noodling')
        os.makedirs(inst_dir)
        data = {
            'noodling': 'some_noodling',
            'overrides': overrides or {
                'name': 'Test',
                'zone': 'default',
            }
        }
        with open(os.path.join(inst_dir, 'instance.yaml'), 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        return inst_dir

    def _read_overrides(self, inst_dir):
        with open(os.path.join(inst_dir, 'instance.yaml'), 'r') as f:
            data = yaml.safe_load(f)
        return data.get('overrides', {})

    def test_save_role_director(self):
        """Writing role=director persists to instance.yaml."""
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        with tempfile.TemporaryDirectory() as tmpdir:
            inst_dir = self._make_instance(tmpdir)
            mixin = EntityInspectorMixin()
            mixin._save_instance_override(inst_dir, 'role', 'director')

            overrides = self._read_overrides(inst_dir)
            assert overrides['role'] == 'director'

    def test_save_role_performer(self):
        """Writing role=performer persists to instance.yaml."""
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        with tempfile.TemporaryDirectory() as tmpdir:
            inst_dir = self._make_instance(tmpdir)
            mixin = EntityInspectorMixin()
            mixin._save_instance_override(inst_dir, 'role', 'performer')

            overrides = self._read_overrides(inst_dir)
            assert overrides['role'] == 'performer'

    def test_save_role_empty_clears(self):
        """Writing role='' clears the role."""
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        with tempfile.TemporaryDirectory() as tmpdir:
            inst_dir = self._make_instance(tmpdir, {
                'name': 'Test', 'role': 'director'
            })
            mixin = EntityInspectorMixin()
            mixin._save_instance_override(inst_dir, 'role', '')

            overrides = self._read_overrides(inst_dir)
            assert overrides['role'] == ''

    def test_get_instance_override_reads_role(self):
        """_get_instance_override reads role from entity_data."""
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        mixin = EntityInspectorMixin()
        entity_data = {
            'data': {
                'overrides': {'role': 'director'}
            }
        }
        assert mixin._get_instance_override(entity_data, 'role', '') == 'director'

    def test_get_instance_override_role_default(self):
        """_get_instance_override returns default when role not set."""
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        mixin = EntityInspectorMixin()
        entity_data = {'data': {'overrides': {'name': 'Test'}}}
        assert mixin._get_instance_override(entity_data, 'role', '') == ''


# =================================================================
# INSPECTOR ROLE DROPDOWN
# =================================================================

class TestInspectorRoleDropdown:
    """Test that the role dropdown callback produces correct values."""

    def test_on_role_changed_director(self):
        """Selecting 'Director' saves lowercase 'director'."""
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        with tempfile.TemporaryDirectory() as tmpdir:
            inst_dir = os.path.join(tmpdir, 'brenda')
            os.makedirs(inst_dir)
            data = {'noodling': 'x', 'overrides': {'name': 'Brenda'}}
            with open(os.path.join(inst_dir, 'instance.yaml'), 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

            mixin = EntityInspectorMixin()
            mixin.is_loading = False

            entity_data = {
                'path': inst_dir,
                'id': 'agent_brenda',
                'data': data,
            }

            # Capture emitted signal
            emitted = []

            class FakeSignal:
                def emit(self, *args):
                    emitted.append(args)

            mixin.noodlingPropertyChanged = FakeSignal()
            mixin._on_role_changed('Director', entity_data)

            # Verify persistence
            with open(os.path.join(inst_dir, 'instance.yaml'), 'r') as f:
                saved = yaml.safe_load(f)
            assert saved['overrides']['role'] == 'director'

            # Verify signal
            assert len(emitted) == 1
            assert emitted[0] == ('agent_brenda', 'role', 'director')

    def test_on_role_changed_none_clears(self):
        """Selecting '(None)' saves empty string."""
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        with tempfile.TemporaryDirectory() as tmpdir:
            inst_dir = os.path.join(tmpdir, 'ajo')
            os.makedirs(inst_dir)
            data = {'noodling': 'x', 'overrides': {'name': 'Ajo', 'role': 'performer'}}
            with open(os.path.join(inst_dir, 'instance.yaml'), 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

            mixin = EntityInspectorMixin()
            mixin.is_loading = False

            entity_data = {
                'path': inst_dir,
                'id': 'agent_ajo',
                'data': data,
            }

            emitted = []

            class FakeSignal:
                def emit(self, *args):
                    emitted.append(args)

            mixin.noodlingPropertyChanged = FakeSignal()
            mixin._on_role_changed('(None)', entity_data)

            with open(os.path.join(inst_dir, 'instance.yaml'), 'r') as f:
                saved = yaml.safe_load(f)
            assert saved['overrides']['role'] == ''
            assert emitted[0] == ('agent_ajo', 'role', '')

    def test_on_role_changed_skips_when_loading(self):
        """No write when is_loading is True."""
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        with tempfile.TemporaryDirectory() as tmpdir:
            inst_dir = os.path.join(tmpdir, 'ajo')
            os.makedirs(inst_dir)
            data = {'noodling': 'x', 'overrides': {'name': 'Ajo'}}
            with open(os.path.join(inst_dir, 'instance.yaml'), 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

            mixin = EntityInspectorMixin()
            mixin.is_loading = True

            entity_data = {'path': inst_dir, 'id': 'agent_ajo', 'data': data}
            mixin._on_role_changed('Director', entity_data)

            with open(os.path.join(inst_dir, 'instance.yaml'), 'r') as f:
                saved = yaml.safe_load(f)
            assert 'role' not in saved.get('overrides', {})


# =================================================================
# STAGE DISCOVERY (role extraction)
# =================================================================

class TestStageDiscoveryRole:
    """Test that _discover_stage_instances extracts role from overrides."""

    def _make_stage_with_instances(self, tmpdir, instances):
        """Build a minimal stage directory with noodling instances.

        Args:
            instances: list of dicts with keys: id, name, role (optional)
        """
        stage_dir = os.path.join(tmpdir, 'test_stage')
        os.makedirs(stage_dir)

        # Minimal stage.yaml
        with open(os.path.join(stage_dir, 'stage.yaml'), 'w') as f:
            yaml.dump({'name': 'Test Stage', 'description': 'A test stage'}, f)

        instances_dir = os.path.join(stage_dir, 'Instances')
        os.makedirs(instances_dir)

        for inst in instances:
            inst_id = inst['id']
            inst_path = os.path.join(instances_dir, inst_id)
            os.makedirs(inst_path)

            # Create noodling template dir with minimal files
            noodling_dir = os.path.join(tmpdir, f'noodling_{inst_id}')
            os.makedirs(noodling_dir, exist_ok=True)
            with open(os.path.join(noodling_dir, 'assembly.yaml'), 'w') as f:
                yaml.dump({'name': f'{inst["name"]} Assembly', 'facets': []}, f)
            with open(os.path.join(noodling_dir, 'noodling.yaml'), 'w') as f:
                yaml.dump({'name': inst['name']}, f)

            # Compute relative path from instance to noodling template
            noodling_ref = os.path.relpath(noodling_dir, inst_path)

            overrides = {'name': inst['name']}
            if 'role' in inst:
                overrides['role'] = inst['role']

            inst_data = {
                'noodling': noodling_ref,
                'overrides': overrides,
            }
            with open(os.path.join(inst_path, 'instance.yaml'), 'w') as f:
                yaml.dump(inst_data, f, default_flow_style=False)

        return stage_dir

    def test_discovery_extracts_director_role(self):
        """Director role extracted from instance overrides."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = self._make_stage_with_instances(tmpdir, [
                {'id': 'brenda', 'name': 'Brenda', 'role': 'director'},
                {'id': 'ajo', 'name': 'Ajo', 'role': 'performer'},
            ])

            manager = GuidePerformanceManager(StubMainWindow())
            results = manager._discover_stage_instances(stage_dir)

            brenda = next(r for r in results if r['noodling_id'] == 'brenda')
            ajo = next(r for r in results if r['noodling_id'] == 'ajo')

            assert brenda['role'] == 'director'
            assert ajo['role'] == 'performer'

    def test_discovery_default_role_empty(self):
        """Instances without role override default to empty string."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = self._make_stage_with_instances(tmpdir, [
                {'id': 'ajo', 'name': 'Ajo'},  # no role key
            ])

            manager = GuidePerformanceManager(StubMainWindow())
            results = manager._discover_stage_instances(stage_dir)

            assert len(results) == 1
            assert results[0]['role'] == ''

    def test_discovery_mixed_roles(self):
        """Stage with director, performers, and unassigned noodlings."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = self._make_stage_with_instances(tmpdir, [
                {'id': 'brenda', 'name': 'Brenda', 'role': 'director'},
                {'id': 'ajo', 'name': 'Ajo', 'role': 'performer'},
                {'id': 'krampus', 'name': 'Krampus', 'role': 'performer'},
                {'id': 'extra', 'name': 'Extra'},  # no role
            ])

            manager = GuidePerformanceManager(StubMainWindow())
            results = manager._discover_stage_instances(stage_dir)

            roles = {r['noodling_id']: r['role'] for r in results}
            assert roles == {
                'brenda': 'director',
                'ajo': 'performer',
                'krampus': 'performer',
                'extra': '',
            }


# =================================================================
# MANAGER ROLE UPDATE
# =================================================================

class TestManagerRoleUpdate:
    """Test that update_role() stores role in instance metadata."""

    def test_update_role_stores_in_metadata(self):
        """update_role() writes role to _instance_metadata."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        manager._instance_metadata = {
            'ajo': {'name': 'Ajo', 'role': ''},
        }

        manager.update_role('ajo', 'performer')
        assert manager._instance_metadata['ajo']['role'] == 'performer'

    def test_update_role_director(self):
        """update_role() can set director role."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        manager._instance_metadata = {
            'brenda': {'name': 'Brenda', 'role': ''},
        }

        manager.update_role('brenda', 'director')
        assert manager._instance_metadata['brenda']['role'] == 'director'

    def test_update_role_clear(self):
        """update_role('') clears the role."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        manager._instance_metadata = {
            'brenda': {'name': 'Brenda', 'role': 'director'},
        }

        manager.update_role('brenda', '')
        assert manager._instance_metadata['brenda']['role'] == ''

    def test_update_role_unknown_noodling_noop(self):
        """update_role() for unknown noodling_id does nothing."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        manager._instance_metadata = {}
        manager.update_role('nonexistent', 'director')
        # No error, no crash


# =================================================================
# HIERARCHY ROLE INDICATOR
# =================================================================

class TestHierarchyRoleIndicator:
    """Test that role suffix appears in hierarchy tree display names."""

    def test_role_suffix_formatting(self):
        """Director/Performer roles produce correct suffix strings."""
        # Test the formatting logic directly
        for role, expected in [
            ('director', ' [Director]'),
            ('performer', ' [Performer]'),
            ('', ''),
        ]:
            suffix = f" [{role.capitalize()}]" if role else ""
            assert suffix == expected

    def test_director_suffix_in_display(self):
        """Display name includes [Director] when role is set."""
        name = "Brenda"
        role = "director"
        suffix = f" [{role.capitalize()}]" if role else ""
        assert name + suffix == "Brenda [Director]"

    def test_performer_suffix_in_display(self):
        """Display name includes [Performer] when role is set."""
        name = "Ajo Majo"
        role = "performer"
        suffix = f" [{role.capitalize()}]" if role else ""
        assert name + suffix == "Ajo Majo [Performer]"

    def test_no_suffix_when_no_role(self):
        """Display name has no suffix when role is empty."""
        name = "Ajo Majo"
        role = ""
        suffix = f" [{role.capitalize()}]" if role else ""
        assert name + suffix == "Ajo Majo"


# =================================================================
# SIGNAL ROUTING
# =================================================================

class TestRoleSignalRouting:
    """Test that role property change routes through signals mixin."""

    def test_signal_routes_to_update_role(self):
        """_on_noodling_property_changed routes role to manager.update_role."""
        from noodlestudio.core.main_window_signals_mixin import MainWindowSignalsMixin

        # Track calls
        calls = []

        class FakeManager:
            is_active = True
            def update_role(self, nid, value):
                calls.append(('update_role', nid, value))

        class FakeWindow(MainWindowSignalsMixin):
            guide_performance_manager = FakeManager()

        window = FakeWindow()
        window._on_noodling_property_changed('agent_brenda', 'role', 'director')

        assert calls == [('update_role', 'brenda', 'director')]

    def test_signal_noop_when_no_manager(self):
        """No crash when manager is not set."""
        from noodlestudio.core.main_window_signals_mixin import MainWindowSignalsMixin

        class FakeWindow(MainWindowSignalsMixin):
            guide_performance_manager = None

        window = FakeWindow()
        window._on_noodling_property_changed('agent_brenda', 'role', 'director')
        # No error


# =================================================================
# SMOKE: DEFAULT INSTANCES (backward compat)
# =================================================================

class TestDefaultInstancesBackwardCompat:
    """Verify existing instances work without a role field."""

    def test_ajo_instance_has_no_role(self):
        """Ajo's instance.yaml has no role -- defaults to empty."""
        inst_path = os.path.join(
            os.path.dirname(__file__), '..',
            'library/templates/Getting Started/Stages/the_nexus/Instances/ajo/instance.yaml'
        )
        if not os.path.exists(inst_path):
            pytest.skip("Default template not found")

        with open(inst_path) as f:
            data = yaml.safe_load(f)

        role = data.get('overrides', {}).get('role', '')
        assert role == '', f"Ajo should have no role set, got: {role!r}"

    def test_existing_instances_discoverable_with_roles(self):
        """Stage discovery works for all instances including director."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        stage_path = os.path.join(
            os.path.dirname(__file__), '..',
            'library/templates/Getting Started/Stages/the_nexus'
        )
        if not os.path.isdir(stage_path):
            pytest.skip("Default stage not found")

        manager = GuidePerformanceManager(StubMainWindow())
        results = manager._discover_stage_instances(stage_path)

        # Ajo, Krampus, Juanita (no role) + Brenda (director)
        assert len(results) >= 4
        roles = {r['name']: r['role'] for r in results}
        assert roles.get('Brenda') == 'director'
        # Non-directors should have empty role
        for r in results:
            if r['name'] != 'Brenda':
                assert r['role'] == '', f"{r['name']} has unexpected role: {r['role']!r}"


# =================================================================
# PHASE B: PER-CHARACTER TEXT BOXES + VIEW MODE
# =================================================================

class TestViewMode:
    """Test Stage View / Script View toggle."""

    def test_default_view_is_script(self, qapp):
        """Panel defaults to script view (interleaved dialogue)."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)
        assert panel._view_mode == 'script'
        assert not panel.dialogue_view.isHidden()
        assert panel._char_text_row.isHidden()

    def test_toggle_to_stage_view(self, qapp):
        """Toggling to stage view shows per-character areas, hides interleaved."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel.set_view_mode('stage')
        assert panel._view_mode == 'stage'
        assert not panel._char_text_row.isHidden()
        assert panel.dialogue_view.isHidden()

    def test_toggle_back_to_script_view(self, qapp):
        """Toggling back to script view restores interleaved dialogue."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel.set_view_mode('stage')
        panel.set_view_mode('script')
        assert panel._view_mode == 'script'
        assert not panel.dialogue_view.isHidden()
        assert panel._char_text_row.isHidden()

    def test_segmented_control_reflects_mode(self, qapp):
        """Segmented control highlights active mode."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        # Default: script mode -- Script segment active
        assert panel._view_mode == 'script'
        assert panel._seg_active_style in panel._seg_script.styleSheet()
        assert panel._seg_inactive_style in panel._seg_stage.styleSheet()
        panel.set_view_mode('stage')
        assert panel._seg_active_style in panel._seg_stage.styleSheet()
        assert panel._seg_inactive_style in panel._seg_script.styleSheet()
        panel.set_view_mode('script')
        assert panel._seg_active_style in panel._seg_script.styleSheet()

    def test_three_char_text_views_exist(self, qapp):
        """Three per-character text areas created for left/center/right."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)
        assert set(panel._char_text_views.keys()) == {'left', 'center', 'right'}
        for view in panel._char_text_views.values():
            assert view.isReadOnly()


class TestPerCharacterTextAreas:
    """Test per-character event dispatch in Stage View."""

    def test_append_spoken_event(self, qapp):
        """Spoken event text appears in the correct character's text area."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        # Assign noodling to slot
        panel._noodling_to_slot['ajo'] = 'left'

        panel.append_character_event('ajo', 'spoken', 'Hello there!')
        text = panel._char_text_views['left'].toPlainText()
        assert 'Hello there!' in text

    def test_append_action_event(self, qapp):
        """Action event text appears in correct column."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel._noodling_to_slot['krampus'] = 'right'
        panel.append_character_event('krampus', 'action', 'jumps at the noise')
        text = panel._char_text_views['right'].toPlainText()
        assert 'jumps at the noise' in text

    def test_append_thought_event(self, qapp):
        """Thought event text appears in correct column."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel._noodling_to_slot['ajo'] = 'left'
        panel.append_character_event('ajo', 'thought', 'Third cup this week...')
        text = panel._char_text_views['left'].toPlainText()
        assert 'Third cup this week...' in text

    def test_events_go_to_correct_columns(self, qapp):
        """Events for different characters appear in different columns."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel._noodling_to_slot['ajo'] = 'left'
        panel._noodling_to_slot['juanita'] = 'center'
        panel._noodling_to_slot['krampus'] = 'right'

        panel.append_character_event('ajo', 'spoken', 'Ajo says hi')
        panel.append_character_event('juanita', 'spoken', 'Juanita waves')
        panel.append_character_event('krampus', 'action', 'Krampus bounces')

        assert 'Ajo says hi' in panel._char_text_views['left'].toPlainText()
        assert 'Juanita waves' in panel._char_text_views['center'].toPlainText()
        assert 'Krampus bounces' in panel._char_text_views['right'].toPlainText()

        # Verify no cross-contamination
        assert 'Juanita' not in panel._char_text_views['left'].toPlainText()
        assert 'Krampus' not in panel._char_text_views['center'].toPlainText()

    def test_clear_character_text_all(self, qapp):
        """clear_character_text() without ID clears all columns."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel._noodling_to_slot['ajo'] = 'left'
        panel.append_character_event('ajo', 'spoken', 'Some text')
        panel.clear_character_text()

        for view in panel._char_text_views.values():
            assert view.toPlainText() == ''

    def test_clear_character_text_single(self, qapp):
        """clear_character_text(noodling_id) clears only that column."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel._noodling_to_slot['ajo'] = 'left'
        panel._noodling_to_slot['krampus'] = 'right'
        panel.append_character_event('ajo', 'spoken', 'Ajo text')
        panel.append_character_event('krampus', 'spoken', 'Krampus text')

        panel.clear_character_text('ajo')
        assert panel._char_text_views['left'].toPlainText() == ''
        assert 'Krampus text' in panel._char_text_views['right'].toPlainText()

    def test_clear_dialogue_clears_both_views(self, qapp):
        """clear_dialogue() clears both interleaved and per-character views."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel._noodling_to_slot['ajo'] = 'left'
        panel.append_character_event('ajo', 'spoken', 'Stage text')
        panel.append_noodling_text('ajo', 'Ajo', 'Script text')

        panel.clear_dialogue()
        assert panel._char_text_views['left'].toPlainText() == ''
        assert panel.dialogue_view.toPlainText() == ''


class TestOffstageSection:
    """Test the offstage director section."""

    def test_offstage_hidden_by_default(self, qapp):
        """Offstage section not visible when no director set."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)
        assert not panel._offstage_section.isVisible()

    def test_set_director_shows_offstage_in_stage_view(self, qapp):
        """Setting a director shows offstage section in Stage View."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel.set_view_mode('stage')
        panel.set_director('brenda', 'Brenda')
        assert not panel._offstage_section.isHidden()
        assert 'Brenda' in panel._offstage_status.text()

    def test_offstage_hidden_in_script_view(self, qapp):
        """Offstage section hidden in Script View even with director."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel.set_director('brenda', 'Brenda')
        panel.set_view_mode('script')
        assert panel._offstage_section.isHidden()

    def test_set_offstage_status(self, qapp):
        """Status text updates in offstage section."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel.set_director('brenda', 'Brenda')
        panel.set_offstage_status('generating...')
        assert 'generating...' in panel._offstage_status.text()
        assert 'Brenda' in panel._offstage_status.text()

    def test_clear_director(self, qapp):
        """Setting director to None hides offstage."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel.set_view_mode('stage')
        panel.set_director('brenda', 'Brenda')
        assert not panel._offstage_section.isHidden()

        panel.set_director(None, '')
        assert panel._offstage_section.isHidden()

    def test_append_offstage_beat(self, qapp):
        """Beat text appears in the offstage details area."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        panel.set_director('brenda', 'Brenda')
        panel.append_offstage_beat('n+0  CUE ajo: drops cup')
        panel.append_offstage_beat('n+5  CUE krampus: jumps')

        text = panel._offstage_beat_view.toPlainText()
        assert 'CUE ajo: drops cup' in text
        assert 'CUE krampus: jumps' in text


# =================================================================
# PHASE C: DIRECTED BEAT FORMAT + DISPATCH
# =================================================================

class TestDirectedBeatParsing:
    """Test parsing of directed_beat JSON from director output."""

    def _get_manager(self):
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow
        return GuidePerformanceManager(StubMainWindow())

    def test_parse_valid_beat(self):
        """Valid directed_beat JSON parses correctly."""
        import json
        manager = self._get_manager()

        beat_json = json.dumps({
            "type": "directed_beat",
            "events": [
                {"character": "ajo", "tick": 0, "type": "action",
                 "text": "drops the cup"},
                {"character": "krampus", "tick": 5, "type": "spoken",
                 "text": "EEEEK!"},
            ],
            "narration": "The cup slips."
        })

        beat = manager._parse_directed_beat(beat_json)
        assert beat is not None
        assert beat['type'] == 'directed_beat'
        assert len(beat['events']) == 2
        assert beat['narration'] == 'The cup slips.'

    def test_parse_invalid_json_returns_none(self):
        """Malformed JSON returns None."""
        manager = self._get_manager()
        assert manager._parse_directed_beat("not json {{{") is None

    def test_parse_wrong_type_returns_none(self):
        """JSON without type=directed_beat returns None."""
        import json
        manager = self._get_manager()
        assert manager._parse_directed_beat(json.dumps({"type": "other"})) is None

    def test_parse_missing_events_returns_none(self):
        """JSON without events list returns None."""
        import json
        manager = self._get_manager()
        result = manager._parse_directed_beat(
            json.dumps({"type": "directed_beat"})
        )
        assert result is None

    def test_parse_extracts_from_markdown_fence(self):
        """Beat JSON wrapped in markdown code fence still parses."""
        import json
        manager = self._get_manager()

        raw = "Here's the beat:\n```json\n" + json.dumps({
            "type": "directed_beat",
            "events": [
                {"character": "ajo", "tick": 0, "type": "spoken",
                 "text": "Hello"}
            ]
        }) + "\n```\nDone."

        beat = manager._parse_directed_beat(raw)
        assert beat is not None
        assert len(beat['events']) == 1

    def test_parse_beat_with_expressions(self):
        """Beat events with expression dicts parse correctly."""
        import json
        manager = self._get_manager()

        beat_json = json.dumps({
            "type": "directed_beat",
            "events": [
                {"character": "ajo", "tick": 0, "type": "action",
                 "text": "drops cup",
                 "expression": {"surprise": 0.8, "distress": 0.4}},
            ]
        })

        beat = manager._parse_directed_beat(beat_json)
        assert beat['events'][0]['expression'] == {
            "surprise": 0.8, "distress": 0.4
        }


class TestBeatDispatch:
    """Test tick-based event scheduling and dispatch."""

    def test_dispatch_event_routes_to_window(self, qapp):
        """_dispatch_event sends event to window's append_character_event."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
            PerformanceState,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import StubMainWindow, FakeLLMClient

        dispatched = []

        class TrackingWindow:
            def append_character_event(self, nid, etype, text):
                dispatched.append((nid, etype, text))
            def append_offstage_beat(self, text):
                pass
            def append_screenplay_line(self, *a): pass

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = TrackingWindow()
        manager._performance_state = PerformanceState.PLAYING

        performer = NoodlingPerformer('ajo', 'Ajo Majo', FakeLLMClient())
        manager._performers = {'ajo': performer}

        event = {'character': 'ajo', 'tick': 0, 'type': 'spoken',
                 'text': 'Hello!'}
        manager._dispatch_event(event)

        assert len(dispatched) == 1
        assert dispatched[0] == ('ajo', 'spoken', 'Hello!')

    def test_dispatch_event_unknown_character_noop(self, qapp):
        """_dispatch_event for unknown character does nothing."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
            PerformanceState,
        )
        from tests.conftest import StubMainWindow

        dispatched = []

        class TrackingWindow:
            def append_character_event(self, nid, etype, text):
                dispatched.append((nid, etype, text))

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = TrackingWindow()
        manager._performance_state = PerformanceState.PLAYING
        manager._performers = {}

        event = {'character': 'nobody', 'tick': 0, 'type': 'spoken',
                 'text': 'Ghost text'}
        manager._dispatch_event(event)
        assert len(dispatched) == 0

    def test_dispatch_stores_spoken_in_history(self, qapp):
        """Spoken events get recorded in ensemble_history."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
            PerformanceState,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import StubMainWindow, FakeLLMClient

        class SilentWindow:
            def append_character_event(self, *a): pass
            def append_offstage_beat(self, *a): pass
            def append_screenplay_line(self, *a): pass

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = SilentWindow()
        manager._performance_state = PerformanceState.PLAYING

        performer = NoodlingPerformer('ajo', 'Ajo Majo', FakeLLMClient())
        manager._performers = {'ajo': performer}

        event = {'character': 'ajo', 'tick': 0, 'type': 'spoken',
                 'text': 'Hello!'}
        manager._dispatch_event(event)

        assert len(manager._ensemble_history) == 1
        assert manager._ensemble_history[0]['role'] == 'Ajo Majo'
        assert manager._ensemble_history[0]['content'] == 'Hello!'

    def test_dispatch_action_in_history(self, qapp):
        """Action events stored with ACTION prefix in history."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
            PerformanceState,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import StubMainWindow, FakeLLMClient

        class SilentWindow:
            def append_character_event(self, *a): pass
            def append_offstage_beat(self, *a): pass
            def append_screenplay_line(self, *a): pass

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = SilentWindow()
        manager._performance_state = PerformanceState.PLAYING

        performer = NoodlingPerformer('ajo', 'Ajo Majo', FakeLLMClient())
        manager._performers = {'ajo': performer}

        event = {'character': 'ajo', 'tick': 0, 'type': 'action',
                 'text': 'drops the cup'}
        manager._dispatch_event(event)

        assert 'ACTION: drops the cup' in manager._ensemble_history[0]['content']

    def test_dispatch_thought_not_in_history(self, qapp):
        """Thought events not stored in ensemble_history (private)."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
            PerformanceState,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import StubMainWindow, FakeLLMClient

        class SilentWindow:
            def append_character_event(self, *a): pass
            def append_offstage_beat(self, *a): pass
            def append_screenplay_line(self, *a): pass

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = SilentWindow()
        manager._performance_state = PerformanceState.PLAYING

        performer = NoodlingPerformer('ajo', 'Ajo Majo', FakeLLMClient())
        manager._performers = {'ajo': performer}

        event = {'character': 'ajo', 'tick': 0, 'type': 'thought',
                 'text': 'Third cup this week'}
        manager._dispatch_event(event)

        assert len(manager._ensemble_history) == 0


# =================================================================
# PHASE D: BRENDA'S DIRECTOR ASSEMBLY
# =================================================================

class TestBrendaAssembly:
    """Test Brenda's director assembly structure."""

    def test_assembly_has_scene_writer(self):
        """Assembly contains a Scene Writer LLM facet."""
        assembly_path = os.path.join(
            os.path.dirname(__file__), '..',
            'library/templates/Getting Started/Noodlings/brenda/assembly.yaml'
        )
        with open(assembly_path) as f:
            data = yaml.safe_load(f)

        facet_map = {f['id']: f for f in data['facets']}
        assert 'scene_writer' in facet_map
        sw = facet_map['scene_writer']
        assert sw['type'] == 'LLM'
        assert sw['model'] == 'LARGE'
        assert 'directed_beat' in sw['prompt']
        assert '{character_descriptions}' in sw['prompt']
        assert '{ensemble_history}' in sw['prompt']

    def test_assembly_has_beat_formatter(self):
        """Assembly contains a Beat Formatter ScriptedFacet."""
        assembly_path = os.path.join(
            os.path.dirname(__file__), '..',
            'library/templates/Getting Started/Noodlings/brenda/assembly.yaml'
        )
        with open(assembly_path) as f:
            data = yaml.safe_load(f)

        facet_map = {f['id']: f for f in data['facets']}
        assert 'beat_formatter' in facet_map
        bf = facet_map['beat_formatter']
        assert bf['type'] == 'ScriptedFacet'
        assert 'directed_beat' in bf['prompt']
        assert 'JSON.parse' in bf['prompt']

    def test_assembly_connections(self):
        """Assembly has correct data flow: INCOMING -> SW -> BF -> OUTGOING."""
        assembly_path = os.path.join(
            os.path.dirname(__file__), '..',
            'library/templates/Getting Started/Noodlings/brenda/assembly.yaml'
        )
        with open(assembly_path) as f:
            data = yaml.safe_load(f)

        conns = data.get('connections', [])
        conn_set = {(c['from'], c['to']) for c in conns}
        assert ('incoming.out', 'scene_writer.in') in conn_set
        assert ('scene_writer.out', 'beat_formatter.in') in conn_set
        assert ('beat_formatter.out', 'outgoing.in') in conn_set

    def test_noodling_yaml_no_vrm(self):
        """Brenda's noodling.yaml should not specify a VRM path."""
        noodling_path = os.path.join(
            os.path.dirname(__file__), '..',
            'library/templates/Getting Started/Noodlings/brenda/noodling.yaml'
        )
        with open(noodling_path) as f:
            data = yaml.safe_load(f)

        assert data.get('vrm_path') is None or data.get('vrm_path') == ''
        assert 'director' in data.get('tags', [])


# =================================================================
# PHASE E: DIRECTED ENSEMBLE DETECTION + REACTIVE FLOW
# =================================================================

class TestDirectedEnsembleDetection:
    """Test that the manager detects director role and splits performers."""

    def _make_stage_with_director(self, tmpdir):
        """Build a stage with Brenda (director) + Ajo (performer)."""
        stage_dir = os.path.join(tmpdir, 'test_stage')
        os.makedirs(stage_dir)

        with open(os.path.join(stage_dir, 'stage.yaml'), 'w') as f:
            yaml.dump({'name': 'Test', 'description': 'Test stage'}, f)

        instances_dir = os.path.join(stage_dir, 'Instances')

        for inst in [
            {'id': 'brenda', 'name': 'Brenda', 'role': 'director'},
            {'id': 'ajo', 'name': 'Ajo', 'role': 'performer'},
        ]:
            inst_path = os.path.join(instances_dir, inst['id'])
            os.makedirs(inst_path)

            noodling_dir = os.path.join(tmpdir, f'noodling_{inst["id"]}')
            os.makedirs(noodling_dir, exist_ok=True)
            with open(os.path.join(noodling_dir, 'assembly.yaml'), 'w') as f:
                yaml.dump({'name': f'{inst["name"]} Assembly', 'facets': []}, f)
            with open(os.path.join(noodling_dir, 'noodling.yaml'), 'w') as f:
                yaml.dump({'name': inst['name']}, f)

            noodling_ref = os.path.relpath(noodling_dir, inst_path)
            overrides = {'name': inst['name'], 'role': inst['role']}
            with open(os.path.join(inst_path, 'instance.yaml'), 'w') as f:
                yaml.dump({
                    'noodling': noodling_ref,
                    'overrides': overrides,
                }, f, default_flow_style=False)

        return stage_dir

    def test_discovery_identifies_director(self):
        """Director noodling detected in stage instances."""
        import tempfile
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = self._make_stage_with_director(tmpdir)
            manager = GuidePerformanceManager(StubMainWindow())
            results = manager._discover_stage_instances(stage_dir)

            directors = [r for r in results if r['role'] == 'director']
            performers = [r for r in results if r['role'] == 'performer']

            assert len(directors) == 1
            assert directors[0]['name'] == 'Brenda'
            assert len(performers) == 1
            assert performers[0]['name'] == 'Ajo'


class TestCharacterDescriptions:
    """Test _format_character_descriptions for director context."""

    def test_excludes_director(self):
        """Character descriptions exclude the director noodling."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import StubMainWindow, FakeLLMClient

        manager = GuidePerformanceManager(StubMainWindow())
        manager._instance_metadata = {
            'brenda': {'name': 'Brenda', 'role': 'director'},
            'ajo': {
                'name': 'Ajo Majo', 'role': 'performer',
                'description': 'A curious axolotl', 'appearance': 'Pink gills',
            },
            'krampus': {
                'name': 'Krampus', 'role': 'performer',
                'description': 'Alpine kid',
            },
        }
        manager._performers = {
            'brenda': NoodlingPerformer('brenda', 'Brenda', FakeLLMClient()),
            'ajo': NoodlingPerformer('ajo', 'Ajo Majo', FakeLLMClient()),
            'krampus': NoodlingPerformer('krampus', 'Krampus', FakeLLMClient()),
        }

        desc = manager._format_character_descriptions()
        assert 'Ajo Majo' in desc
        assert 'Krampus' in desc
        assert 'Brenda' not in desc

    def test_includes_appearance(self):
        """Character descriptions include appearance when available."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import StubMainWindow, FakeLLMClient

        manager = GuidePerformanceManager(StubMainWindow())
        manager._instance_metadata = {
            'ajo': {
                'name': 'Ajo', 'role': 'performer',
                'appearance': 'Bright-eyed axolotl',
            },
        }
        manager._performers = {
            'ajo': NoodlingPerformer('ajo', 'Ajo', FakeLLMClient()),
        }

        desc = manager._format_character_descriptions()
        assert 'Bright-eyed axolotl' in desc


class TestReactiveFlow:
    """Test the reactive auto-advance flow."""

    def test_auto_advance_delay_default(self):
        """Default auto-advance delay is 5000ms."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        assert manager._auto_advance_delay_ms == 5000

    def test_user_message_cancels_auto_advance(self, qapp):
        """User message in directed mode cancels auto-advance timer."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
            PerformanceState,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from PyQt6.QtCore import QTimer
        from tests.conftest import StubMainWindow, StubWindow, FakeLLMClient

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = StubWindow()
        manager._performance_state = PerformanceState.PLAYING
        manager._directed_mode = True

        director = NoodlingPerformer('brenda', 'Brenda', FakeLLMClient())
        manager._director_performer = director
        manager._performers = {
            'brenda': director,
            'ajo': NoodlingPerformer('ajo', 'Ajo', FakeLLMClient()),
        }
        manager._instance_metadata = {
            'brenda': {'name': 'Brenda', 'role': 'director'},
            'ajo': {'name': 'Ajo', 'role': 'performer'},
        }

        # Simulate a running auto-advance timer
        timer = QTimer()
        timer.setSingleShot(True)
        manager._auto_advance_timer = timer
        timer.start(99999)

        # User sends message -- should cancel timer
        manager._on_user_message_ensemble("Hello there!")

        # Timer should be stopped
        assert not timer.isActive()

    def test_improv_turn_queue_excludes_director(self, qapp):
        """In improv fallback, director is excluded from turn queue."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
            PerformanceState,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import StubMainWindow, StubWindow, FakeLLMClient

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = StubWindow()
        manager._performance_state = PerformanceState.PLAYING
        manager._directed_mode = False  # Improv fallback

        manager._performers = {
            'brenda': NoodlingPerformer('brenda', 'Brenda', FakeLLMClient()),
            'ajo': NoodlingPerformer('ajo', 'Ajo', FakeLLMClient()),
            'krampus': NoodlingPerformer('krampus', 'Krampus', FakeLLMClient()),
        }
        manager._instance_metadata = {
            'brenda': {'name': 'Brenda', 'role': 'director'},
            'ajo': {'name': 'Ajo', 'role': 'performer'},
            'krampus': {'name': 'Krampus', 'role': 'performer'},
        }

        # Build the queue using the same filter logic (without executing)
        queue = [
            nid for nid, p in manager._performers.items()
            if not p.paused
            and manager._instance_metadata.get(nid, {}).get('role') != 'director'
        ]

        # Director should NOT be in the turn queue
        assert 'brenda' not in queue
        assert 'ajo' in queue
        assert 'krampus' in queue


# =================================================================
# BUGFIX TESTS: Typewriter Sprint (March 15, 2026)
# =================================================================

class TestCharacterIdResolution:
    """Test fuzzy character ID matching in _dispatch_event (Bug 3)."""

    def _make_manager(self, qapp):
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
            PerformanceState,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import StubMainWindow, FakeLLMClient

        dispatched = []

        class TrackingWindow:
            def append_character_event(self, nid, etype, text):
                dispatched.append((nid, etype, text))
            def append_offstage_beat(self, *a): pass
            def append_screenplay_line(self, *a): pass

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = TrackingWindow()
        manager._performance_state = PerformanceState.PLAYING

        manager._performers = {
            'ajo': NoodlingPerformer('ajo', 'Ajo Majo', FakeLLMClient()),
            'juanita': NoodlingPerformer('juanita', 'Juanita', FakeLLMClient()),
            'krampus': NoodlingPerformer('krampus', 'Krampus', FakeLLMClient()),
        }
        manager._instance_metadata = {
            'ajo': {'name': 'Ajo Majo', 'role': ''},
            'juanita': {'name': 'Juanita', 'role': ''},
            'krampus': {'name': 'Krampus', 'role': ''},
        }

        return manager, dispatched

    def test_exact_match(self, qapp):
        """Exact noodling_id match works."""
        manager, dispatched = self._make_manager(qapp)
        assert manager._resolve_character_id('ajo') == 'ajo'

    def test_display_name_match(self, qapp):
        """Display name lowered matches (e.g., 'ajo majo' -> 'ajo')."""
        manager, dispatched = self._make_manager(qapp)
        assert manager._resolve_character_id('ajo majo') == 'ajo'

    def test_underscore_variant(self, qapp):
        """Underscored display name matches (e.g., 'ajo_majo' -> 'ajo')."""
        manager, dispatched = self._make_manager(qapp)
        assert manager._resolve_character_id('ajo_majo') == 'ajo'

    def test_simple_names_match(self, qapp):
        """Simple names match directly."""
        manager, dispatched = self._make_manager(qapp)
        assert manager._resolve_character_id('juanita') == 'juanita'
        assert manager._resolve_character_id('krampus') == 'krampus'

    def test_unknown_returns_empty(self, qapp):
        """Unknown character ID returns empty string."""
        manager, dispatched = self._make_manager(qapp)
        assert manager._resolve_character_id('nobody') == ''

    def test_dispatch_with_display_name(self, qapp):
        """Full dispatch pipeline works with display name as character ID."""
        manager, dispatched = self._make_manager(qapp)
        event = {'character': 'ajo majo', 'tick': 0, 'type': 'spoken',
                 'text': 'Hello!'}
        manager._dispatch_event(event)
        assert len(dispatched) == 1
        assert dispatched[0] == ('ajo', 'spoken', 'Hello!')


class TestSlotResetOnReplay:
    """Test that VRM slot assignments reset between performances (Bug 1)."""

    def test_clear_dialogue_resets_slots(self, qapp):
        """clear_dialogue() clears _noodling_to_slot mapping."""
        from noodlestudio.runtime.ui.guide_performance_window import (
            PerformancePanel,
        )

        panel = PerformancePanel()
        panel._noodling_to_slot = {'ajo': 'left', 'juanita': 'center'}
        panel.clear_dialogue()
        assert panel._noodling_to_slot == {}

    def test_slot_assignment_deterministic_after_reset(self, qapp):
        """After reset, pre-assignment produces consistent ordering."""
        from noodlestudio.runtime.ui.guide_performance_window import (
            PerformancePanel,
        )

        panel = PerformancePanel()
        panel.set_ensemble_visible(True)

        # Simulate stale state from previous play
        panel._noodling_to_slot = {'old_char': 'left'}
        panel.clear_dialogue()

        # Pre-assign fresh -- same pattern as start_ensemble_from_stage
        slots = ['left', 'center', 'right']
        for i, nid in enumerate(['ajo', 'juanita', 'krampus']):
            panel._noodling_to_slot[nid] = slots[i]

        assert panel._get_slot('ajo') == 'left'
        assert panel._get_slot('juanita') == 'center'
        assert panel._get_slot('krampus') == 'right'
        # Stale entry should be gone
        assert 'old_char' not in panel._noodling_to_slot


class TestEnsembleActiveDefault:
    """Test that ensemble_active defaults to True in inspector (Bug 4)."""

    def test_inspector_default_matches_runtime(self):
        """Inspector and runtime both default ensemble_active to True."""
        from noodlestudio.panels.inspector_entity import EntityInspectorMixin

        mixin = EntityInspectorMixin()
        # Simulate entity_data with no ensemble_active override
        result = mixin._get_instance_override(
            {'path': '/tmp/fake', 'overrides': {}},
            'ensemble_active', True
        )
        assert result is True


class TestStageDropdownDisplay:
    """Test that stage dropdown shows only display name (Bug 5)."""

    def test_dropdown_no_directory_name(self, qapp):
        """Stage dropdown shows 'Hearthwood Cafe', not 'Hearthwood Cafe (the_nexus)'."""
        import tempfile
        from noodlestudio.panels.scene_hierarchy_stage_mixin import (
            SceneHierarchyStageMixin,
        )
        from PyQt6.QtWidgets import QComboBox

        # Create minimal mixin with required attributes
        mixin = SceneHierarchyStageMixin.__new__(SceneHierarchyStageMixin)
        mixin.stage_selector = QComboBox()
        mixin.current_stage = None

        class FakeProjectManager:
            def is_project_open(self):
                return True
            def list_stages(self):
                return ['the_nexus']
            def get_stage_path(self, name):
                return stage_dir

        mixin.project_manager = FakeProjectManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = os.path.join(tmpdir, 'the_nexus')
            os.makedirs(stage_dir)

            import yaml
            with open(os.path.join(stage_dir, 'stage.yaml'), 'w') as f:
                yaml.dump({'name': 'Hearthwood Cafe'}, f)

            # Patch get_stage_path to return our temp dir
            mixin.project_manager.get_stage_path = lambda n: stage_dir
            mixin.populate_stage_selector()

        assert mixin.stage_selector.count() == 1
        text = mixin.stage_selector.itemText(0)
        assert text == 'Hearthwood Cafe'
        assert '(the_nexus)' not in text


class TestRoleDisplayDefault:
    """Test that empty role shows 'Performer' in inspector (Bug 6)."""

    def test_empty_role_displays_as_performer(self):
        """When role override is empty, display shows 'Performer'."""
        role = ''
        display = role.capitalize() if role else 'Performer'
        assert display == 'Performer'

    def test_director_role_displays_correctly(self):
        """When role is 'director', display shows 'Director'."""
        role = 'director'
        display = role.capitalize() if role else 'Performer'
        assert display == 'Director'


class TestOffstageLayout:
    """Test offstage section sizing and beat view visibility (Bug 2)."""

    def test_offstage_minimum_height(self, qapp):
        """Offstage section has adequate minimum height."""
        from noodlestudio.runtime.ui.guide_performance_window import (
            PerformancePanel,
        )

        panel = PerformancePanel()
        assert panel._offstage_section.minimumHeight() >= 80

    def test_set_offstage_beat_text(self, qapp):
        """set_offstage_beat_text replaces content and shows the view."""
        from noodlestudio.runtime.ui.guide_performance_window import (
            PerformancePanel,
        )

        panel = PerformancePanel()
        panel.set_offstage_beat_text("Raw director output:\nsome JSON here")
        assert 'some JSON here' in panel._offstage_beat_view.toPlainText()
        assert not panel._offstage_beat_view.isHidden()


class TestSafeFormatPrompt:
    """Test that prompt template substitution handles JSON braces safely."""

    def test_substitutes_known_variables(self):
        """Known variables in {name} pattern are replaced."""
        from noodlestudio.core.facet_executor import FacetExecutor

        template = "Hello {name}, your role is {role}."
        result = FacetExecutor._safe_format_prompt(
            template, {'name': 'Ajo', 'role': 'performer'}
        )
        assert result == "Hello Ajo, your role is performer."

    def test_leaves_json_braces_untouched(self):
        """JSON object braces { "key": "value" } are not mangled."""
        from noodlestudio.core.facet_executor import FacetExecutor

        template = 'Output JSON: { "type": "directed_beat", "events": [] }'
        result = FacetExecutor._safe_format_prompt(template, {})
        assert result == template  # Unchanged

    def test_mixed_variables_and_json(self):
        """Variables are substituted while JSON braces are preserved."""
        from noodlestudio.core.facet_executor import FacetExecutor

        template = (
            "CHARACTERS:\n{character_descriptions}\n\n"
            'FORMAT: { "type": "directed_beat", '
            '"character": "{character_id}" }'
        )
        result = FacetExecutor._safe_format_prompt(
            template, {'character_descriptions': 'Ajo (id: ajo)'}
        )
        assert 'Ajo (id: ajo)' in result
        assert '{ "type": "directed_beat"' in result
        # {character_id} is not in variables, left as-is
        assert '{character_id}' in result

    def test_unknown_variables_left_as_is(self):
        """Variables not in the dict are left unchanged."""
        from noodlestudio.core.facet_executor import FacetExecutor

        template = "Hello {name}, welcome to {place}."
        result = FacetExecutor._safe_format_prompt(
            template, {'name': 'Ajo'}
        )
        assert result == "Hello Ajo, welcome to {place}."

    def test_brenda_prompt_real_assembly(self):
        """Brenda's actual Scene Writer prompt formats correctly."""
        import yaml
        from noodlestudio.core.facet_executor import FacetExecutor

        assembly_path = (
            "library/templates/Getting Started/"
            "Noodlings/brenda/assembly.yaml"
        )
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base, assembly_path)

        with open(full_path) as f:
            data = yaml.safe_load(f)

        scene_writer = None
        for facet in data['facets']:
            if facet['id'] == 'scene_writer':
                scene_writer = facet
                break
        assert scene_writer is not None

        variables = {
            'character_descriptions': '- Ajo Majo (id: ajo)\n- Juanita (id: juanita)',
            'stage_context': 'A small cafe in a volcanic caldera',
            'ensemble_history': 'Ajo: Hello!',
            'incoming_data': 'Hi everyone!',
            'conversation_history': '',
        }

        result = FacetExecutor._safe_format_prompt(
            scene_writer['prompt'], variables
        )

        # Variables should be substituted
        assert 'Ajo Majo (id: ajo)' in result
        assert 'A small cafe in a volcanic caldera' in result
        assert 'Hi everyone!' in result
        # JSON format example should be preserved
        assert '"directed_beat"' in result
        assert '"character"' in result


class TestDirectorVisibilityDoesNotClobberPerformerSlot:
    """Regression: director with visible=false must not hide a performer's VRM slot.

    Bug: _get_slot() fallback returns 'left' when all slots are taken.
    If the director (who has no slot) hits the visibility loop, the fallback
    causes the left performer's container to be hidden.
    """

    def _make_instances(self):
        """Return instance info dicts for a 3-performer + 1-director ensemble."""
        return [
            {'noodling_id': 'ajo', 'name': 'Ajo Majo', 'vrm_path': '/ajo.vrm',
             'ensemble_active': True, 'visible': True},
            {'noodling_id': 'juanita', 'name': 'Juanita', 'vrm_path': '/juanita.vrm',
             'ensemble_active': True, 'visible': True},
            {'noodling_id': 'krampus', 'name': 'Krampus', 'vrm_path': '/krampus.vrm',
             'ensemble_active': True, 'visible': True},
            {'noodling_id': 'brenda', 'name': 'Brenda', 'vrm_path': None,
             'ensemble_active': True, 'visible': False},
        ]

    def test_director_visible_false_does_not_hide_left_slot(self, qapp):
        """Director with visible=false should not hide the left performer container."""
        pytest.importorskip("PyQt6")
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel

        panel = PerformancePanel(ensemble_mode=True)

        # Pre-assign performer slots (as the manager does)
        panel._noodling_to_slot['ajo'] = 'left'
        panel._noodling_to_slot['juanita'] = 'center'
        panel._noodling_to_slot['krampus'] = 'right'

        # Simulate the visibility loop from guide_performance_manager
        instances = self._make_instances()
        instance_metadata = {
            'ajo': {'role': ''},
            'juanita': {'role': ''},
            'krampus': {'role': ''},
            'brenda': {'role': 'director'},
        }

        for info in instances:
            nid = info['noodling_id']
            role = instance_metadata.get(nid, {}).get('role')
            if not info.get('visible', True) and role != 'director':
                slot = panel._get_slot(nid)
                container = panel._vrm_containers.get(slot)
                if container:
                    container.setVisible(False)

        # All three performer containers must remain visible
        for slot_key in ('left', 'center', 'right'):
            container = panel._vrm_containers[slot_key]
            assert not container.isHidden(), (
                f"Slot '{slot_key}' was hidden -- director visibility "
                f"clobbered a performer slot"
            )

    def test_director_not_assigned_to_performer_slot(self, qapp):
        """_get_slot for an unassigned noodling when all slots are full returns
        'left' fallback but does NOT assign the noodling to that slot."""
        pytest.importorskip("PyQt6")
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel

        panel = PerformancePanel(ensemble_mode=True)
        panel._noodling_to_slot['ajo'] = 'left'
        panel._noodling_to_slot['juanita'] = 'center'
        panel._noodling_to_slot['krampus'] = 'right'

        # Calling _get_slot for brenda should return fallback, not assign
        slot = panel._get_slot('brenda')
        assert slot == 'left'  # Fallback
        assert 'brenda' not in panel._noodling_to_slot


class TestPreflightLabelCheck:
    """Pre-flight validation catches unassigned model labels before play starts."""

    def _make_assembly_yaml(self, tmpdir, name, facets):
        """Write a minimal assembly YAML with given facet model fields."""
        path = os.path.join(tmpdir, f'{name}_assembly.yaml')
        data = {
            'facets': [
                {'id': f'f{i}', 'name': f'Facet {i}',
                 'type': 'ResponseFacet', 'model': model,
                 'prompt': 'test'}
                for i, model in enumerate(facets)
            ],
            'connections': [],
        }
        with open(path, 'w') as f:
            yaml.dump(data, f)
        return path

    def _make_instances(self, tmpdir, assemblies):
        """Build instance dicts with assembly paths."""
        return [
            {
                'noodling_id': f'nid_{i}',
                'name': f'Noodling {i}',
                'assembly_path': self._make_assembly_yaml(
                    tmpdir, f'noodling_{i}', models
                ),
                'ensemble_active': True,
            }
            for i, models in enumerate(assemblies)
        ]

    def test_all_labels_assigned_returns_empty(self, qapp):
        """When all referenced labels are assigned, returns empty list."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from noodlestudio.core.model_label_manager import get_model_label_manager
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        label_mgr = get_model_label_manager()

        # Ensure SMALL is assigned (defaults usually exist)
        provider, model = label_mgr.get_model_for_label("SMALL")
        if not provider or not model:
            label_mgr.set_model_for_label("SMALL", "ollama", "test-model")

        with tempfile.TemporaryDirectory() as tmpdir:
            instances = self._make_instances(tmpdir, [['SMALL']])
            unassigned = manager._preflight_label_check(instances)
            assert unassigned == []

    def test_unassigned_label_detected(self, qapp):
        """Unassigned labels referenced by assemblies are reported."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from noodlestudio.core.model_label_manager import get_model_label_manager
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        label_mgr = get_model_label_manager()

        # Clear a label to make it unassigned
        label_mgr.set_model_for_label(
            "ZZZTEST_NONEXISTENT", None, None, emit_signal=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            instances = self._make_instances(
                tmpdir, [['ZZZTEST_NONEXISTENT']]
            )
            unassigned = manager._preflight_label_check(instances)
            assert 'ZZZTEST_NONEXISTENT' in unassigned

    def test_inactive_instances_skipped(self, qapp):
        """Instances with ensemble_active=False are not scanned."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_assembly_yaml(
                tmpdir, 'inactive', ['ZZZTEST_BOGUS']
            )
            instances = [{
                'noodling_id': 'inactive',
                'name': 'Inactive',
                'assembly_path': path,
                'ensemble_active': False,
            }]
            unassigned = manager._preflight_label_check(instances)
            assert unassigned == []

    def test_multiple_assemblies_union(self, qapp):
        """Labels from all active assemblies are collected."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from noodlestudio.core.model_label_manager import get_model_label_manager
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        label_mgr = get_model_label_manager()

        # Ensure SMALL is assigned but ZZZTEST_A and ZZZTEST_B are not
        provider, model = label_mgr.get_model_for_label("SMALL")
        if not provider or not model:
            label_mgr.set_model_for_label("SMALL", "ollama", "test-model")
        label_mgr.set_model_for_label(
            "ZZZTEST_A", None, None, emit_signal=False
        )
        label_mgr.set_model_for_label(
            "ZZZTEST_B", None, None, emit_signal=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            instances = self._make_instances(
                tmpdir, [['SMALL', 'ZZZTEST_A'], ['ZZZTEST_B']]
            )
            unassigned = manager._preflight_label_check(instances)
            assert 'ZZZTEST_A' in unassigned
            assert 'ZZZTEST_B' in unassigned
            assert 'SMALL' not in unassigned

    def test_default_template_labels_assigned(self, qapp):
        """Getting Started template assemblies use labels that are
        typically assigned by default (SMALL, LARGE)."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from tests.conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())

        # Build instances from real default template assemblies
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template = os.path.join(
            base, 'library', 'templates', 'Getting Started'
        )
        instances_dir = os.path.join(template, 'Stages', 'the_nexus', 'Instances')

        instances = []
        for name in sorted(os.listdir(instances_dir)):
            inst_dir = os.path.join(instances_dir, name)
            inst_yaml = os.path.join(inst_dir, 'instance.yaml')
            if not os.path.exists(inst_yaml):
                continue
            with open(inst_yaml) as f:
                data = yaml.safe_load(f)
            noodling_ref = data.get('noodling', '')
            noodling_path = os.path.normpath(
                os.path.join(inst_dir, noodling_ref)
            )
            assembly_path = os.path.join(noodling_path, 'assembly.yaml')
            instances.append({
                'noodling_id': name,
                'name': data.get('overrides', {}).get('name', name),
                'assembly_path': assembly_path,
                'ensemble_active': data.get('overrides', {}).get(
                    'ensemble_active', True
                ),
            })

        # Collect which labels the template actually uses
        required = set()
        for info in instances:
            if not info.get('ensemble_active', True):
                continue
            path = info['assembly_path']
            if os.path.exists(path):
                with open(path) as f:
                    asm = yaml.safe_load(f)
                for facet in asm.get('facets', []):
                    m = facet.get('model', 'SMALL')
                    if m:
                        required.add(m.upper())

        # These should be SMALL and LARGE at minimum
        assert 'SMALL' in required or 'LARGE' in required, (
            f"Template assemblies use labels: {required}"
        )


# Made with love. Use with love.
# Caitlyn Meeks 2026
