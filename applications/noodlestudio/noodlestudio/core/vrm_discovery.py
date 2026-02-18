# ──────────────────────────────────────────────────────────────
#
#   VRM Discovery - Enumerate available VRM files from library and project
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.vrm_discovery
# PURPOSE:  VRM File Discovery
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY FUNCTIONS:
#   discover_vrm_files
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
from pathlib import Path

import yaml


def discover_vrm_files(project_root: str = None) -> list:
    """
    Enumerate available VRM files from library and project.

    Searches three directories:
    1. Library templates (shipped with NoodleStudio)
    2. Library noodlings (standalone library characters)
    3. Project noodlings (user's own characters)

    Args:
        project_root: Absolute path to the current project root.
                      If None, only library VRMs are returned.

    Returns:
        List of dicts sorted by (source, name):
        [
            {
                'name': 'Ajo Majo',
                'path': '/abs/path/to/AjoMajo.vrm',
                'source': 'library' | 'project',
                'noodling_dir': '/abs/path/to/noodling/template',
            },
            ...
        ]
    """
    results = []
    seen_paths = set()

    # Resolve the library root relative to this module
    # __file__ = .../noodlestudio/core/vrm_discovery.py
    # core/ -> noodlestudio/ (package) -> noodlestudio/ (app dir with library/)
    module_dir = Path(__file__).resolve().parent       # core/
    package_dir = module_dir.parent                    # noodlestudio/ (package)
    app_dir = package_dir.parent                       # applications/noodlestudio/
    library_dir = app_dir / 'library'

    # 1. Library templates
    templates_noodlings = library_dir / 'templates' / 'Getting Started' / 'Noodlings'
    if templates_noodlings.is_dir():
        _scan_noodlings_dir(templates_noodlings, 'library', results, seen_paths)

    # 2. Library noodlings (standalone)
    library_noodlings = library_dir / 'noodlings'
    if library_noodlings.is_dir():
        _scan_noodlings_dir(library_noodlings, 'library', results, seen_paths)

    # 3. Project noodlings
    if project_root:
        project_noodlings = Path(project_root) / 'Noodlings'
        if project_noodlings.is_dir():
            _scan_noodlings_dir(project_noodlings, 'project', results, seen_paths)

    # Sort by source (library first) then name
    results.sort(key=lambda item: (0 if item['source'] == 'library' else 1, item['name'].lower()))

    return results


def _scan_noodlings_dir(noodlings_dir: Path, source: str,
                        results: list, seen_paths: set):
    """
    Scan a noodlings directory for VRM files.

    Reads each subdirectory's noodling.yaml (if present) to find
    the VRM path and display name. Falls back to filename stem
    as display name if noodling.yaml is absent.

    Args:
        noodlings_dir: Directory containing noodling subdirectories
        source: 'library' or 'project'
        results: List to append results to (mutated in place)
        seen_paths: Set of absolute VRM paths already seen (dedup)
    """
    for entry in sorted(noodlings_dir.iterdir()):
        if not entry.is_dir():
            continue

        # Try reading noodling.yaml for name and vrm_path
        noodling_yaml = entry / 'noodling.yaml'
        name = None
        vrm_path = None

        if noodling_yaml.exists():
            try:
                with open(noodling_yaml) as f:
                    data = yaml.safe_load(f) or {}
                name = data.get('name')
                vrm_ref = data.get('vrm_path', '')
                if vrm_ref:
                    resolved = (entry / vrm_ref).resolve()
                    if resolved.exists():
                        vrm_path = str(resolved)
            except Exception:
                pass

        # If no VRM from noodling.yaml, scan Radiances/ for any .vrm files
        if not vrm_path:
            radiances_dir = entry / 'Radiances'
            if radiances_dir.is_dir():
                for vrm_file in sorted(radiances_dir.iterdir()):
                    if vrm_file.suffix.lower() == '.vrm' and vrm_file.is_file():
                        vrm_path = str(vrm_file.resolve())
                        if not name:
                            name = vrm_file.stem.replace('_', ' ').title()
                        break

        if not vrm_path:
            continue

        # Deduplicate by absolute path
        abs_path = os.path.normpath(vrm_path)
        if abs_path in seen_paths:
            continue
        seen_paths.add(abs_path)

        # Fall back to directory name for display
        if not name:
            name = entry.name.replace('_', ' ').title()

        results.append({
            'name': name,
            'path': abs_path,
            'source': source,
            'noodling_dir': str(entry.resolve()),
        })
