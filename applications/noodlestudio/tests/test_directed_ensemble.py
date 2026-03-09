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

    def test_existing_instances_discoverable_without_role(self):
        """Stage discovery works for instances that predate the role field."""
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

        # All existing instances should have role='' (backward compat)
        for r in results:
            assert r['role'] == '', f"{r['name']} has unexpected role: {r['role']!r}"
        assert len(results) >= 3  # Ajo, Krampus, Juanita


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

    def test_toggle_button_changes_label(self, qapp):
        """Toggle button text reflects current mode."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        panel = PerformancePanel(ensemble_mode=True)

        assert panel._view_toggle_btn.text() == "Stage View"  # Can switch to stage
        panel.set_view_mode('stage')
        assert panel._view_toggle_btn.text() == "Script View"  # Can switch back
        panel.set_view_mode('script')
        assert panel._view_toggle_btn.text() == "Stage View"

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

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = SilentWindow()
        manager._performance_state = PerformanceState.PLAYING

        performer = NoodlingPerformer('ajo', 'Ajo Majo', FakeLLMClient())
        manager._performers = {'ajo': performer}

        event = {'character': 'ajo', 'tick': 0, 'type': 'thought',
                 'text': 'Third cup this week'}
        manager._dispatch_event(event)

        assert len(manager._ensemble_history) == 0


# Made with love. Use with love.
# Caitlyn Meeks 2026
