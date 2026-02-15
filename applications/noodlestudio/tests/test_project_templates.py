# ------------------------------------------------------------------
#   Project Templates Tests
#
#   Verifies: template discovery, template structure, project
#   creation from templates, self-contained asset paths.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_project_templates
# PURPOSE:  Project Templates Tests
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
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestTemplateDiscovery:
    """Template discovery must find all valid templates."""

    def test_templates_dir_exists(self):
        """Templates directory must exist at library/templates/."""
        from noodlestudio.dialogs.project_chooser_dialog import _templates_dir
        tdir = _templates_dir()
        assert tdir.is_dir(), f"Templates directory not found: {tdir}"

    def test_discover_finds_getting_started(self):
        """Template discovery must find the Getting Started template."""
        from noodlestudio.dialogs.project_chooser_dialog import _discover_templates
        templates = _discover_templates()
        names = [t['name'] for t in templates]
        assert 'Getting Started' in names, \
            f"Getting Started not found in templates: {names}"

    def test_discover_finds_empty_project(self):
        """Template discovery must find the Empty Project template."""
        from noodlestudio.dialogs.project_chooser_dialog import _discover_templates
        templates = _discover_templates()
        names = [t['name'] for t in templates]
        assert 'Empty Project' in names, \
            f"Empty Project not found in templates: {names}"

    def test_all_templates_have_project_file(self):
        """Every discovered template must have a project.noodleproj."""
        from noodlestudio.dialogs.project_chooser_dialog import _discover_templates
        templates = _discover_templates()
        for tmpl in templates:
            proj_file = os.path.join(tmpl['path'], 'project.noodleproj')
            assert os.path.isfile(proj_file), \
                f"Missing project.noodleproj in template: {tmpl['name']}"


class TestGettingStartedTemplate:
    """Getting Started template must be self-contained with correct structure."""

    @pytest.fixture
    def template_path(self):
        from noodlestudio.dialogs.project_chooser_dialog import _templates_dir
        return str(_templates_dir() / 'Getting Started')

    def test_noodlings_exist(self, template_path):
        """Ajo and Yuki noodling templates must exist with required files."""
        for noodling in ('ajo_majo', 'yuki_cyberfox'):
            noodling_dir = os.path.join(template_path, 'Noodlings', noodling)
            assert os.path.isdir(noodling_dir), f"Missing noodling: {noodling}"
            for filename in ('assembly.yaml', 'recipe.yaml', 'noodling.yaml'):
                assert os.path.isfile(os.path.join(noodling_dir, filename)), \
                    f"Missing {filename} in {noodling}"

    def test_ajo_vrm_is_local(self, template_path):
        """Ajo's VRM must be inside the template, not an external reference."""
        noodling_yaml = os.path.join(
            template_path, 'Noodlings', 'ajo_majo', 'noodling.yaml'
        )
        with open(noodling_yaml) as f:
            data = yaml.safe_load(f)

        vrm_ref = data.get('vrm_path', '')
        assert '..' not in vrm_ref, \
            f"VRM path must not escape the noodling directory: {vrm_ref}"

        # The VRM file must actually exist at the referenced path
        vrm_abs = os.path.join(template_path, 'Noodlings', 'ajo_majo', vrm_ref)
        assert os.path.isfile(vrm_abs), \
            f"VRM file not found at: {vrm_abs}"

    def test_instance_refs_resolve_within_template(self, template_path):
        """Instance noodling references must resolve to paths inside the template."""
        instances_dir = os.path.join(
            template_path, 'Stages', 'the_nexus', 'Instances'
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
                f"Instance ref does not resolve: {resolved} (from {instance_name})"
            # Must resolve within the template directory
            assert resolved.startswith(template_path), \
                f"Instance ref escapes template: {resolved}"

    def test_stage_yaml_valid(self, template_path):
        """Stage YAML must parse and contain expected fields."""
        stage_yaml = os.path.join(
            template_path, 'Stages', 'the_nexus', 'stage.yaml'
        )
        with open(stage_yaml) as f:
            data = yaml.safe_load(f)
        assert data['name'] == 'The Nexus'
        assert 'geometry' in data


class TestCreateFromTemplate:
    """create_project_from_template must deep-copy and rename correctly."""

    def test_creates_project_directory(self, tmp_path):
        """Created project must exist with project.noodleproj."""
        from noodlestudio.dialogs.project_chooser_dialog import (
            _templates_dir, create_project_from_template,
        )
        template = str(_templates_dir() / 'Getting Started')
        dest = create_project_from_template(
            template, 'TestProject', str(tmp_path)
        )
        assert dest is not None
        assert os.path.isdir(dest)
        assert os.path.isfile(os.path.join(dest, 'project.noodleproj'))

    def test_project_name_updated_in_noodleproj(self, tmp_path):
        """project.noodleproj must have the user-chosen name, not the template name."""
        from noodlestudio.dialogs.project_chooser_dialog import (
            _templates_dir, create_project_from_template,
        )
        template = str(_templates_dir() / 'Getting Started')
        dest = create_project_from_template(
            template, 'My Custom Name', str(tmp_path)
        )
        proj_file = os.path.join(dest, 'project.noodleproj')
        with open(proj_file) as f:
            data = yaml.safe_load(f)
        assert data['name'] == 'My Custom Name'

    def test_noodlings_copied_into_project(self, tmp_path):
        """Noodling templates must be copied into the new project."""
        from noodlestudio.dialogs.project_chooser_dialog import (
            _templates_dir, create_project_from_template,
        )
        template = str(_templates_dir() / 'Getting Started')
        dest = create_project_from_template(
            template, 'CopyTest', str(tmp_path)
        )
        for noodling in ('ajo_majo', 'yuki_cyberfox'):
            assert os.path.isdir(os.path.join(dest, 'Noodlings', noodling))

    def test_vrm_copied_into_project(self, tmp_path):
        """VRM file must be inside the copied project."""
        from noodlestudio.dialogs.project_chooser_dialog import (
            _templates_dir, create_project_from_template,
        )
        template = str(_templates_dir() / 'Getting Started')
        dest = create_project_from_template(
            template, 'VRMCopyTest', str(tmp_path)
        )
        vrm_path = os.path.join(
            dest, 'Noodlings', 'ajo_majo', 'Radiances', 'AjoMajo.vrm'
        )
        assert os.path.isfile(vrm_path), \
            f"VRM not copied into project: {vrm_path}"

    def test_refuses_duplicate_name(self, tmp_path):
        """Must return None if a project with the same name already exists."""
        from noodlestudio.dialogs.project_chooser_dialog import (
            _templates_dir, create_project_from_template,
        )
        template = str(_templates_dir() / 'Empty Project')
        # Create once
        create_project_from_template(template, 'DupeTest', str(tmp_path))
        # Try again
        result = create_project_from_template(template, 'DupeTest', str(tmp_path))
        assert result is None

    def test_empty_template_creates_minimal_structure(self, tmp_path):
        """Empty Project template must create a project with Stages directory."""
        from noodlestudio.dialogs.project_chooser_dialog import (
            _templates_dir, create_project_from_template,
        )
        template = str(_templates_dir() / 'Empty Project')
        dest = create_project_from_template(
            template, 'EmptyTest', str(tmp_path)
        )
        assert os.path.isdir(dest)
        assert os.path.isfile(os.path.join(dest, 'project.noodleproj'))
        assert os.path.isdir(os.path.join(dest, 'Stages'))


class TestNoStaleHierarchy:
    """Templates must not contain hierarchy.yaml (it has absolute paths)."""

    def test_templates_have_no_hierarchy_yaml(self):
        """No template should ship with a hierarchy.yaml file."""
        from noodlestudio.dialogs.project_chooser_dialog import _templates_dir
        tdir = _templates_dir()
        for entry in sorted(tdir.iterdir()):
            if not entry.is_dir():
                continue
            for dirpath, _dirs, filenames in os.walk(entry):
                assert 'hierarchy.yaml' not in filenames, (
                    f"Template '{entry.name}' contains hierarchy.yaml at "
                    f"{dirpath} — this file stores absolute paths and must "
                    f"not be shipped in templates"
                )

    def test_create_from_template_strips_hierarchy(self, tmp_path):
        """create_project_from_template must remove any hierarchy.yaml files."""
        from noodlestudio.dialogs.project_chooser_dialog import (
            _templates_dir, create_project_from_template,
        )
        template = str(_templates_dir() / 'Getting Started')
        dest = create_project_from_template(
            template, 'HierarchyTest', str(tmp_path)
        )
        # Walk the created project and verify no hierarchy.yaml exists
        for dirpath, _dirs, filenames in os.walk(dest):
            assert 'hierarchy.yaml' not in filenames, (
                f"Created project contains hierarchy.yaml at {dirpath}"
            )


class TestNoodleprojIsYaml:
    """All .noodleproj files must be valid YAML (not JSON)."""

    def test_template_noodleproj_files_are_yaml(self):
        """Every template project.noodleproj must round-trip through YAML cleanly."""
        from noodlestudio.dialogs.project_chooser_dialog import _templates_dir
        tdir = _templates_dir()
        for entry in sorted(tdir.iterdir()):
            proj_file = entry / 'project.noodleproj'
            if not proj_file.exists():
                continue
            with open(proj_file) as f:
                raw = f.read()
            # Must parse as YAML
            data = yaml.safe_load(raw)
            assert isinstance(data, dict), \
                f"{proj_file.name}: parsed to {type(data)}, expected dict"
            assert 'name' in data, \
                f"{entry.name}/project.noodleproj missing 'name' field"
            # Must NOT be JSON (no leading brace)
            assert not raw.lstrip().startswith('{'), \
                f"{entry.name}/project.noodleproj is JSON, should be YAML"

    def test_created_project_noodleproj_is_yaml(self, tmp_path):
        """ProjectManager.create_project must write YAML, not JSON."""
        from noodlestudio.core.project_manager import ProjectManager
        pm = ProjectManager()
        pm.create_project(str(tmp_path), 'yaml_test')
        proj_file = tmp_path / 'yaml_test' / 'project.noodleproj'
        with open(proj_file) as f:
            raw = f.read()
        assert not raw.lstrip().startswith('{'), \
            "create_project wrote JSON, expected YAML"
        data = yaml.safe_load(raw)
        assert data['name'] == 'yaml_test'

    def test_saved_project_noodleproj_stays_yaml(self, tmp_path):
        """After save_project, .noodleproj must still be YAML."""
        from noodlestudio.core.project_manager import ProjectManager
        pm = ProjectManager()
        pm.create_project(str(tmp_path), 'save_test')
        pm.save_project()
        proj_file = tmp_path / 'save_test' / 'project.noodleproj'
        with open(proj_file) as f:
            raw = f.read()
        assert not raw.lstrip().startswith('{'), \
            "save_project wrote JSON, expected YAML"

    def test_reopened_project_noodleproj_stays_yaml(self, tmp_path):
        """After open_project (which writes last_opened), file must stay YAML."""
        from noodlestudio.core.project_manager import ProjectManager
        pm = ProjectManager()
        pm.create_project(str(tmp_path), 'reopen_test')
        project_path = str(tmp_path / 'reopen_test')
        pm.close_project()
        pm.open_project(project_path)
        proj_file = tmp_path / 'reopen_test' / 'project.noodleproj'
        with open(proj_file) as f:
            raw = f.read()
        assert not raw.lstrip().startswith('{'), \
            "open_project wrote JSON on last_opened update, expected YAML"
        data = yaml.safe_load(raw)
        assert 'last_opened' in data
