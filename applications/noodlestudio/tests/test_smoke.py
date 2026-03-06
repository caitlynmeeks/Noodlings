# ------------------------------------------------------------------
#   Smoke Test Suite
#
#   Integration gatekeeper for NoodleStudio. These tests verify
#   that the core infrastructure works: server path, inspector,
#   project system, stage system, and assembly loading.
#
#   Run before every commit to catch regressions early.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_smoke
# PURPOSE:  Smoke Test Suite
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

LIBRARY_DIR = os.path.join(os.path.dirname(__file__), '..', 'library')


# =====================================================================
# Server Infrastructure
# =====================================================================

class TestServerInfrastructure:
    """cmush directory, start.sh, and server toggle must be correct."""

    def test_cmush_dir_resolves_to_existing_directory(self):
        """_cmush_dir() path must resolve to a real directory."""
        from noodlestudio.core.main_window_server_mixin import MainWindowServerMixin

        path = MainWindowServerMixin._cmush_dir()
        assert os.path.isdir(path), f"cmush dir does not exist: {path}"

    def test_start_sh_exists_and_is_executable(self):
        """start.sh must exist in the cmush directory and be executable."""
        from noodlestudio.core.main_window_server_mixin import MainWindowServerMixin

        path = MainWindowServerMixin._cmush_dir()
        start_sh = os.path.join(path, 'start.sh')
        assert os.path.isfile(start_sh), f"start.sh not found: {start_sh}"
        mode = os.stat(start_sh).st_mode
        assert mode & stat.S_IXUSR, "start.sh is not executable"


# =====================================================================
# Inspector Panel
# =====================================================================

class TestInspectorPanel:
    """Inspector must initialize base attributes and load facets."""

    def test_inspector_has_bound_widgets_after_init(self, qapp):
        """_bound_widgets must exist and be a dict after __init__."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        panel = InspectorPanel(None)
        assert hasattr(panel, '_bound_widgets')
        assert isinstance(panel._bound_widgets, dict)

    def test_inspector_has_all_base_attributes(self, qapp):
        """All base attributes from init_base_inspector() must be present."""
        from noodlestudio.panels.inspector_panel import InspectorPanel
        panel = InspectorPanel(None)
        for attr in ('property_fields', 'component_widgets',
                     'collapsible_states', 'is_loading', '_bound_widgets'):
            assert hasattr(panel, attr), f"Missing attribute: {attr}"

    def test_property_meta_attributes(self):
        """PropertyMeta must have minimum, maximum, prop_type, hidden."""
        from noodlestudio.core.property_binding import PropertyMeta
        meta = PropertyMeta(name="test", prop_type=float)
        assert hasattr(meta, 'minimum')
        assert hasattr(meta, 'maximum')
        assert hasattr(meta, 'prop_type')
        assert hasattr(meta, 'hidden')


# =====================================================================
# Project System
# =====================================================================

class TestProjectSystem:
    """Default project must exist with valid structure."""

    def test_default_project_exists(self):
        """Default project directory must exist."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        assert os.path.isdir(project_dir), f"Default project not found: {project_dir}"
        assert os.path.isfile(os.path.join(project_dir, 'project.noodleproj'))

    def test_default_project_has_noodlings(self):
        """Ajo, Krampus, and Juanita noodling templates must exist."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        for noodling in ('ajo_majo', 'krampus', 'juanita'):
            noodling_dir = os.path.join(project_dir, 'Noodlings', noodling)
            assert os.path.isdir(noodling_dir), f"Noodling not found: {noodling_dir}"
            assert os.path.isfile(os.path.join(noodling_dir, 'assembly.yaml'))
            assert os.path.isfile(os.path.join(noodling_dir, 'recipe.yaml'))
            assert os.path.isfile(os.path.join(noodling_dir, 'noodling.yaml'))

    def test_default_project_has_stage_instances(self):
        """Stage must have Ajo, Krampus, and Juanita instances."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        instances_dir = os.path.join(
            project_dir, 'Stages', 'the_nexus', 'Instances'
        )
        assert os.path.isdir(instances_dir)
        assert os.path.isdir(os.path.join(instances_dir, 'ajo'))
        assert os.path.isdir(os.path.join(instances_dir, 'krampus'))
        assert os.path.isdir(os.path.join(instances_dir, 'juanita'))

    def test_instance_refs_resolve(self):
        """Instance noodling references must resolve to real directories."""
        project_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        instances_dir = os.path.join(
            project_dir, 'Stages', 'the_nexus', 'Instances'
        )
        for instance_name in ('ajo', 'krampus', 'juanita'):
            instance_path = os.path.join(
                instances_dir, instance_name, 'instance.yaml'
            )
            with open(instance_path) as f:
                data = yaml.safe_load(f)
            ref = data['noodling']
            resolved = os.path.normpath(
                os.path.join(os.path.dirname(instance_path), ref)
            )
            assert os.path.isdir(resolved), \
                f"Noodling ref not found: {resolved} (from {instance_name})"


# =====================================================================
# Stage System
# =====================================================================

class TestStageSystem:
    """Stage dropdown must handle empty state gracefully."""

    def test_stage_dropdown_empty_without_project(self, qapp):
        """With no project open, stage dropdown shows placeholder."""
        from noodlestudio.panels.scene_hierarchy import SceneHierarchy
        from noodlestudio.core.project_manager import ProjectManager

        hierarchy = SceneHierarchy()
        hierarchy.set_project_manager(ProjectManager())

        assert hierarchy.stage_selector.count() >= 1
        text = hierarchy.stage_selector.itemText(0)
        assert "No project" in text
        assert hierarchy.stage_selector.isEnabled() is False


# =====================================================================
# Launch Flow (Template Discovery + Project Chooser)
# =====================================================================

class TestLaunchFlow:
    """Template system and project chooser must work for first-run experience."""

    def test_templates_directory_exists(self):
        """library/templates/ must exist and contain valid templates."""
        from noodlestudio.dialogs.project_chooser_dialog import _templates_dir
        tdir = _templates_dir()
        assert tdir.is_dir(), f"Templates directory not found: {tdir}"

    def test_template_discovery_finds_templates(self):
        """Template discovery must find at least Getting Started and Empty Project."""
        from noodlestudio.dialogs.project_chooser_dialog import _discover_templates
        templates = _discover_templates()
        names = {t['name'] for t in templates}
        assert 'Getting Started' in names, f"Missing Getting Started: {names}"
        assert 'Empty Project' in names, f"Missing Empty Project: {names}"

    def test_ajo_vrm_inside_template(self):
        """Ajo's VRM must be inside the template, not an external reference."""
        template_dir = os.path.join(LIBRARY_DIR, 'templates', 'Getting Started')
        vrm = os.path.join(
            template_dir, 'Noodlings', 'ajo_majo', 'Radiances', 'AjoMajo.vrm'
        )
        assert os.path.isfile(vrm), f"VRM not found in template: {vrm}"

    def test_project_chooser_dialog_constructs(self, qapp):
        """ProjectChooserDialog must construct without error."""
        from noodlestudio.dialogs.project_chooser_dialog import ProjectChooserDialog
        dialog = ProjectChooserDialog()
        assert dialog.windowTitle() == "Choose a Project"
        assert len(dialog._templates) >= 2


# =====================================================================
# Assembly Loading
# =====================================================================

class TestAssemblyLoading:
    """Ajo, Krampus, and Juanita assemblies must parse without error."""

    def test_ajo_assembly_parses(self):
        """Ajo's assembly.yaml must parse and have required fields."""
        assembly_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Noodlings', 'ajo_majo', 'assembly.yaml'
        )
        with open(assembly_path) as f:
            data = yaml.safe_load(f)
        assert 'name' in data
        assert 'facets' in data
        assert len(data['facets']) >= 3

    def test_krampus_assembly_parses(self):
        """Krampus assembly.yaml must parse and have required fields."""
        assembly_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Noodlings', 'krampus', 'assembly.yaml'
        )
        with open(assembly_path) as f:
            data = yaml.safe_load(f)
        assert 'name' in data
        assert 'facets' in data
        assert len(data['facets']) >= 3

    def test_juanita_assembly_parses(self):
        """Juanita assembly.yaml must parse and have required fields."""
        assembly_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Noodlings', 'juanita', 'assembly.yaml'
        )
        with open(assembly_path) as f:
            data = yaml.safe_load(f)
        assert 'name' in data
        assert 'facets' in data
        assert len(data['facets']) >= 3

    def test_set_yaml_exists_in_default_template(self):
        """Default stage must have a set.yaml with scene objects."""
        set_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Stages', 'the_nexus', 'set.yaml'
        )
        assert os.path.isfile(set_path), f"set.yaml not found: {set_path}"
        with open(set_path) as f:
            data = yaml.safe_load(f)
        assert 'objects' in data
        assert len(data['objects']) == 7

    def test_marks_directory_has_marks(self):
        """Default stage must have 3 blocking marks."""
        marks_dir = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Stages', 'the_nexus', 'Marks'
        )
        assert os.path.isdir(marks_dir)
        marks = [f for f in os.listdir(marks_dir) if f.endswith('.mark.yaml')]
        assert len(marks) == 3
        mark_names = {f.replace('.mark.yaml', '') for f in marks}
        assert mark_names == {'behind_counter', 'window_seat', 'by_the_fire'}

    def test_instances_have_mark_overrides(self):
        """Each noodling instance must have a mark override."""
        instances_dir = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Stages', 'the_nexus', 'Instances'
        )
        expected = {
            'ajo': 'behind_counter',
            'juanita': 'window_seat',
            'krampus': 'by_the_fire',
        }
        for instance_name, expected_mark in expected.items():
            instance_path = os.path.join(instances_dir, instance_name, 'instance.yaml')
            with open(instance_path) as f:
                data = yaml.safe_load(f)
            mark = data.get('overrides', {}).get('mark', '')
            assert mark == expected_mark, \
                f"{instance_name}: expected mark '{expected_mark}', got '{mark}'"

    def test_stage_instance_discovery(self):
        """_discover_stage_instances must find all three noodlings."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        stage_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started', 'Stages', 'the_nexus'
        )
        instances = manager._discover_stage_instances(stage_path)
        assert len(instances) == 3
        ids = {i['noodling_id'] for i in instances}
        assert ids == {'ajo', 'krampus', 'juanita'}

    def test_set_yaml_has_opening_beats(self):
        """Default set.yaml must have an opening scene with 8 beats."""
        set_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Stages', 'the_nexus', 'set.yaml'
        )
        with open(set_path) as f:
            data = yaml.safe_load(f)
        opening = data.get('opening')
        assert opening is not None, "set.yaml missing opening section"
        assert opening['mode'] == 'live'
        assert len(opening['beats']) == 8

    def test_marks_have_activities(self):
        """All 3 blocking marks must have non-empty activity fields."""
        from noodlestudio.core.set_dressing import load_marks
        marks_dir = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Stages', 'the_nexus'
        )
        marks = load_marks(marks_dir)
        assert len(marks) == 3
        for mark in marks:
            assert mark.activity.strip(), \
                f"Mark '{mark.id}' has empty activity"


# =====================================================================
# MVP Surface (Phase A verification)
# =====================================================================

class TestMVPSurface:
    """Verify the app launches with only MVP panels and menus.

    These tests lock down the Phase A renovation: deferred panels are
    None, deferred menus are gone, status bar is clean, and the default
    center tab is Facets Editor.
    """

    @pytest.fixture(autouse=True, scope='class')
    def _create_window(self, qapp):
        """Create MainWindow once for all MVP surface tests."""
        from noodlestudio.core.main_window import MainWindow
        cls = type(self)
        cls._window = MainWindow()
        yield
        cls._window.close()

    @property
    def window(self):
        return type(self)._window

    def test_mvp_panels_instantiated(self):
        """All MVP panels exist as non-None attributes."""
        for attr in ('hierarchy', 'assets', 'unified_editor',
                     'settings_panel', 'noodle_code_panel', 'inspector',
                     'console', 'profiler_panel', 'cognitive_cycles'):
            assert getattr(self.window, attr, None) is not None, \
                f"MVP panel missing: {attr}"

    def test_deferred_panels_are_none(self):
        """Deferred panels are None, not instantiated."""
        for attr in ('spatial_view', 'gaussian_viewer', 'ui_canvas_editor',
                     'web_view', 'world_view'):
            assert getattr(self.window, attr) is None, \
                f"Deferred panel should be None: {attr}"

    def test_no_account_menu(self):
        """Account menu does not exist in the menu bar."""
        menu_titles = [a.text() for a in self.window.menuBar().actions()]
        assert '&Account' not in menu_titles, \
            f"Account menu should not exist: {menu_titles}"

    def test_no_component_menu(self):
        """Component menu does not exist in the menu bar."""
        menu_titles = [a.text() for a in self.window.menuBar().actions()]
        assert '&Component' not in menu_titles, \
            f"Component menu should not exist: {menu_titles}"

    def test_no_auth_in_status_bar(self):
        """Status bar has no auth/avatar/enter-world widgets."""
        assert not hasattr(self.window, 'account_status_widget')
        assert not hasattr(self.window, 'avatar_dropdown')
        assert not hasattr(self.window, 'enter_world_btn')

    def test_default_center_tab_is_assembly(self):
        """Default center tab is Assembly (Unified Editor)."""
        tabs = self.window.center_tabs
        current = tabs.tabText(tabs.currentIndex())
        assert current == "Assembly", \
            f"Default tab should be 'Assembly', got '{current}'"

    def test_no_build_menu_items(self):
        """File menu has no Build Settings or Build Application items."""
        file_menu = None
        for action in self.window.menuBar().actions():
            if action.text() == '&File':
                file_menu = action.menu()
                break
        assert file_menu is not None, "File menu not found"
        item_texts = [a.text() for a in file_menu.actions()]
        assert 'Build Settings...' not in item_texts
        assert 'Build Application...' not in item_texts
