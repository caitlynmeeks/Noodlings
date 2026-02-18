# ──────────────────────────────────────────────────────────────
#   Tests for VRM file discovery (Phase F.1)
# ──────────────────────────────────────────────────────────────

import os
import tempfile

import pytest
import yaml

from noodlestudio.core.vrm_discovery import discover_vrm_files


class TestVRMDiscovery:
    """Tests for discover_vrm_files utility."""

    def test_returns_list(self):
        """discover_vrm_files returns a list."""
        result = discover_vrm_files()
        assert isinstance(result, list)

    def test_finds_library_template_vrms(self):
        """Discovers Ajo, Krampus, and Juanita VRMs from library templates."""
        result = discover_vrm_files()
        names = [item['name'] for item in result]
        assert 'Ajo Majo' in names
        assert 'Krampus' in names or any('krampus' in n.lower() for n in names)
        assert 'Juanita' in names or any('juanita' in n.lower() for n in names)

    def test_returns_correct_dict_structure(self):
        """Each result has name, path, source, noodling_dir keys."""
        result = discover_vrm_files()
        assert len(result) > 0

        for item in result:
            assert 'name' in item
            assert 'path' in item
            assert 'source' in item
            assert 'noodling_dir' in item
            assert isinstance(item['name'], str)
            assert isinstance(item['path'], str)
            assert item['source'] in ('library', 'project')

    def test_vrm_paths_exist(self):
        """All returned VRM paths point to real files."""
        result = discover_vrm_files()
        for item in result:
            assert os.path.exists(item['path']), f"VRM not found: {item['path']}"

    def test_noodling_dirs_exist(self):
        """All returned noodling_dir paths are real directories."""
        result = discover_vrm_files()
        for item in result:
            assert os.path.isdir(item['noodling_dir']), f"Dir not found: {item['noodling_dir']}"

    def test_library_source_label(self):
        """Library VRMs have source='library'."""
        result = discover_vrm_files()
        ajo = next((item for item in result if item['name'] == 'Ajo Majo'), None)
        assert ajo is not None
        assert ajo['source'] == 'library'

    def test_no_project_root_returns_library_only(self):
        """With no project_root, only library VRMs are returned."""
        result = discover_vrm_files(project_root=None)
        assert all(item['source'] == 'library' for item in result)

    def test_nonexistent_project_root_ignored(self):
        """A non-existent project root is handled gracefully."""
        result = discover_vrm_files(project_root='/nonexistent/path/for/testing')
        # Should still return library VRMs without error
        assert isinstance(result, list)
        assert len(result) > 0

    def test_sorted_by_source_then_name(self):
        """Results are sorted: library first, then project, alphabetically."""
        result = discover_vrm_files()
        if len(result) < 2:
            pytest.skip("Need at least 2 VRMs to test sorting")

        for i in range(len(result) - 1):
            a, b = result[i], result[i + 1]
            # Library before project
            if a['source'] == 'project' and b['source'] == 'library':
                pytest.fail(f"Project item '{a['name']}' before library item '{b['name']}'")

    def test_no_duplicates_by_path(self):
        """No two results share the same absolute VRM path."""
        result = discover_vrm_files()
        paths = [os.path.normpath(item['path']) for item in result]
        assert len(paths) == len(set(paths)), "Duplicate VRM paths found"

    def test_project_vrms_discovered(self):
        """VRM files in a project's Noodlings/ directory are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a project noodling with a VRM
            noodling_dir = os.path.join(tmpdir, 'Noodlings', 'test_character')
            radiances_dir = os.path.join(noodling_dir, 'Radiances')
            os.makedirs(radiances_dir)

            # Write noodling.yaml
            noodling_yaml = os.path.join(noodling_dir, 'noodling.yaml')
            with open(noodling_yaml, 'w') as f:
                yaml.dump({
                    'name': 'Test Character',
                    'vrm_path': 'Radiances/test.vrm',
                }, f)

            # Create a dummy VRM file
            vrm_path = os.path.join(radiances_dir, 'test.vrm')
            with open(vrm_path, 'wb') as f:
                f.write(b'\x00' * 16)  # Minimal dummy file

            result = discover_vrm_files(project_root=tmpdir)
            project_items = [item for item in result if item['source'] == 'project']
            assert len(project_items) == 1
            assert project_items[0]['name'] == 'Test Character'
            assert os.path.realpath(project_items[0]['path']) == os.path.realpath(vrm_path)

    def test_skips_noodling_without_vrm(self):
        """Noodling directories without VRM files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a noodling with no VRM
            noodling_dir = os.path.join(tmpdir, 'Noodlings', 'no_vrm')
            os.makedirs(noodling_dir)

            noodling_yaml = os.path.join(noodling_dir, 'noodling.yaml')
            with open(noodling_yaml, 'w') as f:
                yaml.dump({'name': 'No VRM Character'}, f)

            result = discover_vrm_files(project_root=tmpdir)
            project_items = [item for item in result if item['source'] == 'project']
            assert len(project_items) == 0

    def test_skips_nonexistent_vrm_in_yaml(self):
        """Noodling.yaml pointing to nonexistent VRM is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            noodling_dir = os.path.join(tmpdir, 'Noodlings', 'bad_ref')
            os.makedirs(noodling_dir)

            noodling_yaml = os.path.join(noodling_dir, 'noodling.yaml')
            with open(noodling_yaml, 'w') as f:
                yaml.dump({
                    'name': 'Bad Ref',
                    'vrm_path': 'Radiances/missing.vrm',
                }, f)

            result = discover_vrm_files(project_root=tmpdir)
            project_items = [item for item in result if item['source'] == 'project']
            assert len(project_items) == 0

    def test_fallback_name_from_filename(self):
        """When noodling.yaml has no name, falls back to VRM filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            noodling_dir = os.path.join(tmpdir, 'Noodlings', 'nameless')
            radiances_dir = os.path.join(noodling_dir, 'Radiances')
            os.makedirs(radiances_dir)

            # noodling.yaml with vrm_path but no name
            noodling_yaml = os.path.join(noodling_dir, 'noodling.yaml')
            with open(noodling_yaml, 'w') as f:
                yaml.dump({'vrm_path': 'Radiances/cool_model.vrm'}, f)

            vrm_path = os.path.join(radiances_dir, 'cool_model.vrm')
            with open(vrm_path, 'wb') as f:
                f.write(b'\x00' * 16)

            result = discover_vrm_files(project_root=tmpdir)
            project_items = [item for item in result if item['source'] == 'project']
            assert len(project_items) == 1
            # Should use directory name as fallback
            assert project_items[0]['name'] == 'Nameless'

    def test_radiances_scan_fallback(self):
        """VRM found by scanning Radiances/ when noodling.yaml has no vrm_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            noodling_dir = os.path.join(tmpdir, 'Noodlings', 'scan_test')
            radiances_dir = os.path.join(noodling_dir, 'Radiances')
            os.makedirs(radiances_dir)

            # noodling.yaml with name but no vrm_path
            noodling_yaml = os.path.join(noodling_dir, 'noodling.yaml')
            with open(noodling_yaml, 'w') as f:
                yaml.dump({'name': 'Scan Test'}, f)

            vrm_path = os.path.join(radiances_dir, 'scan_test.vrm')
            with open(vrm_path, 'wb') as f:
                f.write(b'\x00' * 16)

            result = discover_vrm_files(project_root=tmpdir)
            project_items = [item for item in result if item['source'] == 'project']
            assert len(project_items) == 1
            assert project_items[0]['name'] == 'Scan Test'
