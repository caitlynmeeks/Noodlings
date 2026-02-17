# ------------------------------------------------------------------
#   Ensemble From Stage Tests
#
#   Verifies: instance discovery from stage directories, assembly
#   path resolution, performer creation from stage instances.
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_ensemble_from_stage
# PURPOSE:  Ensemble From Stage Tests
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

LIBRARY_DIR = os.path.join(os.path.dirname(__file__), '..', 'library')
STUDIO_DIR = os.path.join(os.path.dirname(__file__), '..')


class TestStageInstanceDiscovery:
    """_discover_stage_instances must find noodling instances from stage dirs."""

    def _make_manager(self):
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from conftest import StubMainWindow
        return GuidePerformanceManager(StubMainWindow())

    def test_discovers_three_instances_from_default_project(self):
        """Default project stage must yield Ajo, Krampus, and Juanita instances."""
        manager = self._make_manager()
        stage_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started', 'Stages', 'the_nexus'
        )
        instances = manager._discover_stage_instances(stage_path)

        assert len(instances) == 3
        ids = {i['noodling_id'] for i in instances}
        assert ids == {'ajo', 'krampus', 'juanita'}

    def test_instance_names_from_overrides(self):
        """Instance names must come from instance.yaml overrides (3 noodlings)."""
        manager = self._make_manager()
        stage_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started', 'Stages', 'the_nexus'
        )
        instances = manager._discover_stage_instances(stage_path)
        names = {i['noodling_id']: i['name'] for i in instances}

        assert names['ajo'] == 'Ajo Majo'
        assert names['krampus'] == 'Krampus'
        assert names['juanita'] == 'Juanita'

    def test_assembly_paths_resolve_to_real_files(self):
        """Each instance's assembly_path must point to a real, parseable file."""
        manager = self._make_manager()
        stage_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started', 'Stages', 'the_nexus'
        )
        instances = manager._discover_stage_instances(stage_path)

        for info in instances:
            assert os.path.isfile(info['assembly_path']), \
                f"Assembly not found: {info['assembly_path']}"
            with open(info['assembly_path']) as f:
                data = yaml.safe_load(f)
            assert 'name' in data, \
                f"Assembly missing 'name' for {info['noodling_id']}"
            assert 'facets' in data, \
                f"Assembly missing 'facets' for {info['noodling_id']}"

    def test_ajo_vrm_path_discovered(self):
        """Ajo's noodling.yaml has vrm_path; it must be resolved."""
        manager = self._make_manager()
        stage_path = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started', 'Stages', 'the_nexus'
        )
        instances = manager._discover_stage_instances(stage_path)
        ajo = next(i for i in instances if i['noodling_id'] == 'ajo')

        # vrm_path may be None if the VRM file doesn't exist on disk,
        # but the field must be present in the result dict
        assert 'vrm_path' in ajo

    def test_empty_stage_returns_no_instances(self, tmp_path):
        """A stage with no Instances/ directory must return empty list."""
        manager = self._make_manager()
        stage_dir = tmp_path / 'empty_stage'
        stage_dir.mkdir()
        instances = manager._discover_stage_instances(str(stage_dir))
        assert instances == []

    def test_nonexistent_stage_returns_no_instances(self, tmp_path):
        """A nonexistent stage path must return empty list."""
        manager = self._make_manager()
        instances = manager._discover_stage_instances(
            str(tmp_path / 'does_not_exist')
        )
        assert instances == []


class TestStageInstanceWithTmpProject:
    """Test instance discovery with a constructed temp project."""

    @pytest.fixture
    def tmp_project_with_instances(self, tmp_path):
        """Build a temp project with instances pointing to real assemblies."""
        # Find real assembly files
        ajo_assembly = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Noodlings', 'ajo_majo', 'assembly.yaml'
        )
        krampus_assembly = os.path.join(
            LIBRARY_DIR, 'templates', 'Getting Started',
            'Noodlings', 'krampus', 'assembly.yaml'
        )

        # Create noodling template dirs with symlinked assemblies
        noodlings_dir = tmp_path / 'Noodlings'
        for name, src_assembly in [('alpha', ajo_assembly), ('beta', krampus_assembly)]:
            noodling_dir = noodlings_dir / name
            noodling_dir.mkdir(parents=True)
            # Copy the assembly (not symlink, for portability)
            with open(src_assembly) as f:
                (noodling_dir / 'assembly.yaml').write_text(f.read())

        # Create stage with instances
        stage_dir = tmp_path / 'Stages' / 'test_stage'
        instances_dir = stage_dir / 'Instances'

        for inst_name, noodling_name, display_name in [
            ('alpha_inst', 'alpha', 'Alpha Noodling'),
            ('beta_inst', 'beta', 'Beta Noodling'),
        ]:
            inst_dir = instances_dir / inst_name
            inst_dir.mkdir(parents=True)
            instance_data = {
                'noodling': f'../../../../Noodlings/{noodling_name}',
                'overrides': {'name': display_name},
            }
            with open(inst_dir / 'instance.yaml', 'w') as f:
                yaml.dump(instance_data, f)

        return str(stage_dir)

    def test_discovers_custom_instances(self, tmp_project_with_instances):
        """Instance discovery must work with arbitrary project structures."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        instances = manager._discover_stage_instances(
            tmp_project_with_instances
        )

        assert len(instances) == 2
        ids = {i['noodling_id'] for i in instances}
        assert ids == {'alpha_inst', 'beta_inst'}

    def test_custom_instance_names(self, tmp_project_with_instances):
        """Custom instance override names must be used."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from conftest import StubMainWindow

        manager = GuidePerformanceManager(StubMainWindow())
        instances = manager._discover_stage_instances(
            tmp_project_with_instances
        )
        names = {i['noodling_id']: i['name'] for i in instances}
        assert names['alpha_inst'] == 'Alpha Noodling'
        assert names['beta_inst'] == 'Beta Noodling'
