"""
NoodleStudio Build System - Create standalone applications from projects

This module provides the build infrastructure for packaging NoodleStudio
projects into standalone executables that can run without the editor.

Core Components:
    Builder - Main orchestrator for the build process
    BuildConfig - Configuration loaded from build.yaml
    Packager - Asset collection and filtering
    MacOSBundler - Creates macOS .app bundles

Usage:
    from noodlestudio.appbuilder import Builder, BuildConfig

    config = BuildConfig.load("/path/to/project")
    builder = Builder(config)
    result = await builder.build("/path/to/output.app")

Author: Caitlyn + Claude
Date: January 3, 2026
"""

from .builder import Builder, BuildConfig, BuildResult

__all__ = [
    'Builder',
    'BuildConfig',
    'BuildResult',
]

__version__ = '1.0.0'
