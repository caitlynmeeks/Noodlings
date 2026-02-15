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

        class Stub(MainWindowServerMixin):
            pass

        stub = object.__new__(Stub)
        path = stub._cmush_dir()
        assert os.path.isdir(path), f"cmush dir does not exist: {path}"

    def test_start_sh_exists_and_is_executable(self):
        """start.sh must exist in the cmush directory and be executable."""
        from noodlestudio.core.main_window_server_mixin import MainWindowServerMixin

        class Stub(MainWindowServerMixin):
            pass

        stub = object.__new__(Stub)
        path = stub._cmush_dir()
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
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
        assert os.path.isdir(project_dir), f"Default project not found: {project_dir}"
        assert os.path.isfile(os.path.join(project_dir, 'project.noodleproj'))

    def test_default_project_has_noodlings(self):
        """Ajo and Yuki noodling templates must exist."""
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
        for noodling in ('ajo_majo', 'yuki_cyberfox'):
            noodling_dir = os.path.join(project_dir, 'Noodlings', noodling)
            assert os.path.isdir(noodling_dir), f"Noodling not found: {noodling_dir}"
            assert os.path.isfile(os.path.join(noodling_dir, 'assembly.yaml'))
            assert os.path.isfile(os.path.join(noodling_dir, 'recipe.yaml'))
            assert os.path.isfile(os.path.join(noodling_dir, 'noodling.yaml'))

    def test_default_project_has_stage_instances(self):
        """Stage must have Ajo and Yuki instances."""
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
        instances_dir = os.path.join(
            project_dir, 'Stages', 'the_nexus', 'Instances'
        )
        assert os.path.isdir(instances_dir)
        assert os.path.isdir(os.path.join(instances_dir, 'ajo'))
        assert os.path.isdir(os.path.join(instances_dir, 'yuki'))

    def test_instance_refs_resolve(self):
        """Instance noodling references must resolve to real directories."""
        project_dir = os.path.join(LIBRARY_DIR, 'Welcome to NoodleStudio')
        instances_dir = os.path.join(
            project_dir, 'Stages', 'the_nexus', 'Instances'
        )
        for instance_name in ('ajo', 'yuki'):
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
# Assembly Loading
# =====================================================================

class TestAssemblyLoading:
    """Ajo and Yuki assemblies must parse without error."""

    def test_ajo_assembly_parses(self):
        """Ajo's assembly.yaml must parse and have required fields."""
        assembly_path = os.path.join(
            LIBRARY_DIR, 'Welcome to NoodleStudio',
            'Noodlings', 'ajo_majo', 'assembly.yaml'
        )
        with open(assembly_path) as f:
            data = yaml.safe_load(f)
        assert 'name' in data
        assert 'facets' in data
        assert len(data['facets']) >= 3

    def test_yuki_assembly_parses(self):
        """Yuki's assembly.yaml must parse and have required fields."""
        assembly_path = os.path.join(
            LIBRARY_DIR, 'Welcome to NoodleStudio',
            'Noodlings', 'yuki_cyberfox', 'assembly.yaml'
        )
        with open(assembly_path) as f:
            data = yaml.safe_load(f)
        assert 'name' in data
        assert 'facets' in data
        assert len(data['facets']) >= 3

    def test_stage_instance_discovery(self):
        """_discover_stage_instances must find both noodlings."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        stage_path = os.path.join(
            LIBRARY_DIR, 'Welcome to NoodleStudio', 'Stages', 'the_nexus'
        )
        instances = manager._discover_stage_instances(stage_path)
        assert len(instances) == 2
        ids = {i['noodling_id'] for i in instances}
        assert ids == {'ajo', 'yuki'}
