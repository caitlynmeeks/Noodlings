# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Hot Reload System - Safe module reloading for live development
#
#   Provides controlled hot-reloading of Python modules while...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.hot_reload
# PURPOSE:  Hot Reload
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ReloadResult, HotReloadManager, get_hot_reload_manager(), hot_reload()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import importlib
import sys
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class ReloadResult:
    """Result of a module reload attempt."""
    success: bool
    module_name: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


class HotReloadManager(QObject):
    """
    Manages safe hot-reloading of Python modules.

    Tracks which modules can be safely reloaded and provides
    hooks for components to re-initialize after reload.
    """

    # Signal emitted after successful reload
    module_reloaded = pyqtSignal(str)  # module_name

    # Singleton
    _instance: Optional['HotReloadManager'] = None

    # Modules that are SAFE to reload
    SAFE_MODULES = {
        # Noodle Code tools
        'noodlestudio.core.noodle_code_tools',

        # Facet implementations (stateless)
        'noodlestudio.core.utility_facets',
        'noodlestudio.core.flow_control_facets',
        'noodlestudio.core.scripted_facet',
        'noodlestudio.core.charm_network_facet',
        'noodlestudio.core.mcp_facet',

        # Data models (no running instances depend on class identity)
        'noodlestudio.core.facet_system',

        # Scripting APIs (recreated on each script run)
        'noodlestudio.scripting.noodle_api',
        'noodlestudio.scripting.models_api',
        'noodlestudio.scripting.affect_api',
        'noodlestudio.scripting.pose_api',
        'noodlestudio.scripting.quantum_api',
        'noodlestudio.scripting.audio_api',
        'noodlestudio.scripting.vision_api',
        'noodlestudio.scripting.training_api',
        'noodlestudio.scripting.cloud_api',

        # Gaussian/radiance tools
        'noodlestudio.tools.vrm_to_radiance',
        'noodlestudio.tools.auto_rigger',
        'noodlestudio.tools.face_detail_camera',
        'noodlestudio.tools.face_detail_training',
    }

    # Modules that should NEVER be reloaded
    UNSAFE_MODULES = {
        # Core app infrastructure
        'noodlestudio.main',
        'noodlestudio.core.main_window',

        # All MainWindow mixins
        'noodlestudio.core.main_window_panels_mixin',
        'noodlestudio.core.main_window_project_mixin',
        'noodlestudio.core.main_window_settings_mixin',
        'noodlestudio.core.main_window_statusbar_mixin',
        'noodlestudio.core.main_window_entities_mixin',
        'noodlestudio.core.main_window_account_mixin',

        # Singletons with state
        'noodlestudio.core.project_manager',
        'noodlestudio.core.provider_manager',
        'noodlestudio.core.model_label_manager',
        'noodlestudio.core.account_manager',
        'noodlestudio.core.generations_manager',
        'noodlestudio.core.mcp_manager',

        # All panels (instantiated as widgets)
        'noodlestudio.panels.inspector_panel',
        'noodlestudio.panels.scene_hierarchy',
        'noodlestudio.panels.assets_panel',
        'noodlestudio.panels.console_panel',
        'noodlestudio.panels.facets_editor_panel',
        'noodlestudio.panels.gaussian_viewer_panel',
        'noodlestudio.panels.noodle_code_panel',
        'noodlestudio.panels.settings_panel',

        # This module itself
        'noodlestudio.core.hot_reload',
    }

    def __init__(self):
        super().__init__()
        self._reload_hooks: Dict[str, List[Callable]] = {}
        self._reload_history: List[ReloadResult] = []

    @classmethod
    def instance(cls) -> 'HotReloadManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = HotReloadManager()
        return cls._instance

    def can_reload(self, module_name: str) -> tuple[bool, str]:
        """
        Check if a module can be safely reloaded.

        Returns (can_reload, reason).
        """
        # Check explicit unsafe list
        if module_name in self.UNSAFE_MODULES:
            return False, "Module is in UNSAFE list (would break running app)"

        # Check explicit safe list
        if module_name in self.SAFE_MODULES:
            return True, "Module is in SAFE list"

        # Unknown module - be conservative
        # Check if it's a panel (unsafe)
        if '.panels.' in module_name:
            return False, "Panel modules cannot be reloaded (already instantiated)"

        # Check if it's a mixin (unsafe)
        if 'mixin' in module_name.lower():
            return False, "Mixin modules cannot be reloaded (already mixed in)"

        # Default: allow with warning
        return True, "Module not in explicit lists - proceeding with caution"

    def reload_module(self, module_name: str, force: bool = False) -> ReloadResult:
        """
        Reload a Python module.

        Args:
            module_name: Full module path (e.g., 'noodlestudio.core.utility_facets')
            force: If True, ignore safety checks (dangerous!)

        Returns:
            ReloadResult with success status and details.
        """
        # Safety check
        if not force:
            can_reload, reason = self.can_reload(module_name)
            if not can_reload:
                result = ReloadResult(
                    success=False,
                    module_name=module_name,
                    message=f"Cannot reload: {reason}",
                )
                self._reload_history.append(result)
                return result

        # Check if module is loaded
        if module_name not in sys.modules:
            result = ReloadResult(
                success=False,
                module_name=module_name,
                message="Module not loaded",
            )
            self._reload_history.append(result)
            return result

        try:
            # Get the module
            module = sys.modules[module_name]

            # Reload it
            importlib.reload(module)

            # Run any registered hooks
            self._run_reload_hooks(module_name)

            # Emit signal
            self.module_reloaded.emit(module_name)

            result = ReloadResult(
                success=True,
                module_name=module_name,
                message=f"Successfully reloaded {module_name}",
            )
            self._reload_history.append(result)

            print(f"[HotReload] Reloaded: {module_name}")
            return result

        except Exception as e:
            result = ReloadResult(
                success=False,
                module_name=module_name,
                message=f"Reload failed: {e}",
                error=str(e),
            )
            self._reload_history.append(result)
            print(f"[HotReload] FAILED to reload {module_name}: {e}")
            return result

    def reload_file(self, file_path: Path) -> ReloadResult:
        """
        Reload the module corresponding to a file path.

        Converts file path to module name and reloads.
        """
        # Convert path to module name
        module_name = self._path_to_module(file_path)

        if not module_name:
            return ReloadResult(
                success=False,
                module_name=str(file_path),
                message="Could not determine module name from path",
            )

        return self.reload_module(module_name)

    def _path_to_module(self, file_path: Path) -> Optional[str]:
        """Convert file path to module name."""
        try:
            path = Path(file_path).resolve()

            # Find the noodlestudio package
            parts = path.parts
            if 'noodlestudio' not in parts:
                return None

            # Get path from noodlestudio onwards
            idx = parts.index('noodlestudio')
            module_parts = list(parts[idx:])

            # Remove .py extension
            if module_parts[-1].endswith('.py'):
                module_parts[-1] = module_parts[-1][:-3]

            # Handle __init__.py
            if module_parts[-1] == '__init__':
                module_parts = module_parts[:-1]

            return '.'.join(module_parts)

        except Exception as e:
            print(f"[HotReload] Error converting path to module: {e}")
            return None

    def register_reload_hook(self, module_name: str, callback: Callable):
        """
        Register a callback to run after a module is reloaded.

        Useful for re-initializing components that depend on the module.
        """
        if module_name not in self._reload_hooks:
            self._reload_hooks[module_name] = []
        self._reload_hooks[module_name].append(callback)

    def unregister_reload_hook(self, module_name: str, callback: Callable):
        """Unregister a reload hook."""
        if module_name in self._reload_hooks:
            try:
                self._reload_hooks[module_name].remove(callback)
            except ValueError:
                pass

    def _run_reload_hooks(self, module_name: str):
        """Run all registered hooks for a module."""
        hooks = self._reload_hooks.get(module_name, [])
        for hook in hooks:
            try:
                hook()
            except Exception as e:
                print(f"[HotReload] Hook error for {module_name}: {e}")

    def get_safe_modules(self) -> Set[str]:
        """Get set of modules known to be safe for reload."""
        return self.SAFE_MODULES.copy()

    def get_reload_history(self, count: int = 20) -> List[ReloadResult]:
        """Get recent reload attempts."""
        return self._reload_history[-count:]

    def reload_all_safe(self) -> List[ReloadResult]:
        """Reload all currently-loaded safe modules."""
        results = []
        for module_name in self.SAFE_MODULES:
            if module_name in sys.modules:
                result = self.reload_module(module_name)
                results.append(result)
        return results

    def reload_noodle_code_tools(self) -> ReloadResult:
        """
        Convenience method to reload Noodle Code tools.

        Call this after editing noodle_code_tools.py.
        """
        return self.reload_module('noodlestudio.core.noodle_code_tools')

    def reload_facets(self) -> List[ReloadResult]:
        """
        Convenience method to reload all facet implementations.

        Call this after editing any facet module.
        """
        facet_modules = [
            'noodlestudio.core.utility_facets',
            'noodlestudio.core.flow_control_facets',
            'noodlestudio.core.scripted_facet',
            'noodlestudio.core.charm_network_facet',
            'noodlestudio.core.mcp_facet',
        ]

        results = []
        for module_name in facet_modules:
            if module_name in sys.modules:
                result = self.reload_module(module_name)
                results.append(result)
        return results


# Singleton accessor
def get_hot_reload_manager() -> HotReloadManager:
    """Get the global hot reload manager instance."""
    return HotReloadManager.instance()


# Convenience function for Noodle Code
def hot_reload(module_name: str, force: bool = False) -> ReloadResult:
    """
    Hot-reload a module by name.

    This is the main entry point for Noodle Code to use.

    Example:
        result = hot_reload('noodlestudio.core.utility_facets')
        if result.success:
            print("Reloaded!")
    """
    return get_hot_reload_manager().reload_module(module_name, force)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
