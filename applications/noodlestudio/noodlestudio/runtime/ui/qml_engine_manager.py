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
#   QML Engine Manager
#
#   Singleton that manages a shared QML engine for all QML widgets.
#   Provides component caching and type registration.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.qml_engine_manager
# PURPOSE:  QML Engine Manager
# LAYER:    Studio / UI Runtime / QML
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   QMLEngineManager
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Check for QML availability
QML_AVAILABLE = False
QQmlEngine = None
QQmlComponent = None

try:
    from PyQt6.QtQml import QQmlEngine as _QQmlEngine, QQmlComponent as _QQmlComponent
    from PyQt6.QtCore import QUrl
    QQmlEngine = _QQmlEngine
    QQmlComponent = _QQmlComponent
    QML_AVAILABLE = True
except ImportError:
    logger.debug("PyQt6.QtQml not available - QML widgets disabled")


class QMLEngineManager:
    """
    Manages a shared QML engine for all QML widgets.

    Benefits:
    - Shared component cache
    - Single JavaScript engine
    - Reduced memory footprint
    - Centralized import path management

    Usage:
        engine = QMLEngineManager.instance()
        component = engine.create_component(Path("widget.qml"))
    """

    _instance: Optional['QMLEngineManager'] = None

    @classmethod
    def instance(cls) -> 'QMLEngineManager':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        if cls._instance is not None and isinstance(cls._instance, QMLEngineManager):
            cls._instance._cleanup()
        cls._instance = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if QML support is available."""
        return QML_AVAILABLE

    def __init__(self):
        if not QML_AVAILABLE:
            logger.warning("QML engine not available - PyQt6.QtQml required")
            self._engine = None
            return

        self._engine = QQmlEngine()

        # Add import paths for common QML modules
        self._add_import_paths()

        # Component cache: path -> QQmlComponent
        self._component_cache: Dict[str, Any] = {}

    def _add_import_paths(self) -> None:
        """Add standard import paths for QML modules."""
        if not self._engine:
            return

        # Add path relative to this module for bundled QML
        module_dir = Path(__file__).parent
        qml_modules_path = module_dir / "qml_modules"
        if qml_modules_path.exists():
            self._engine.addImportPath(str(qml_modules_path))

        # Add path for project-local QML
        project_qml_path = module_dir.parent.parent.parent / "resources" / "qml"
        if project_qml_path.exists():
            self._engine.addImportPath(str(project_qml_path))

    def add_import_path(self, path: str) -> None:
        """Add an additional import path for QML modules."""
        if self._engine:
            self._engine.addImportPath(path)

    def create_component(self, qml_path: Path) -> Optional[Any]:
        """
        Create a QML component from file.

        Uses caching to avoid recompiling the same QML file.

        Args:
            qml_path: Path to the .qml file

        Returns:
            QQmlComponent if successful, None if error
        """
        if not self._engine:
            logger.error("Cannot create QML component - engine not available")
            return None

        # Check cache
        path_str = str(qml_path.resolve())
        if path_str in self._component_cache:
            return self._component_cache[path_str]

        # Create new component
        component = QQmlComponent(self._engine, QUrl.fromLocalFile(path_str))

        if component.status() == QQmlComponent.Status.Error:
            for error in component.errors():
                logger.error(f"QML Error in {qml_path.name}: {error.toString()}")
            return None

        # Cache successful components
        self._component_cache[path_str] = component
        return component

    def clear_cache(self) -> None:
        """Clear the component cache."""
        self._component_cache.clear()

    def clear_cache_for(self, qml_path: Path) -> None:
        """Remove a specific file from the cache (for hot reload)."""
        path_str = str(qml_path.resolve())
        self._component_cache.pop(path_str, None)

    @property
    def engine(self) -> Optional[Any]:
        """Get the underlying QQmlEngine."""
        return self._engine

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._component_cache.clear()
        # Note: QQmlEngine cleanup is handled by Qt


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
