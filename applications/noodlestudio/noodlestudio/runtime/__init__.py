"""
NoodleStudio Runtime - Headless execution of NoodleStudio projects

This module provides the runtime foundation for executing NoodleStudio
projects without the editor GUI. It's used for:
- Built standalone applications (via py2app)
- Command-line execution
- Integration into other systems
- Testing and automation

Core Components:
    NoodleApp - Main runtime class that loads and executes projects
    NoodleAppConfig - Configuration for the runtime
    HeadlessLLMClient - LLM client without Qt dependencies
    LLMConfig - Configuration for LLM providers

Usage:
    # As a module
    from noodlestudio.runtime import NoodleApp

    app = NoodleApp()
    app.load_project("/path/to/project")
    result = await app.run("Hello, world!")

    # From command line
    python -m noodlestudio.runtime /path/to/project --interactive

Author: Caitlyn + Claude
Date: January 3, 2026
"""

from .app import NoodleApp, NoodleAppConfig, ProjectConfig
from .llm_client import HeadlessLLMClient, LLMConfig, create_llm_client_from_env

__all__ = [
    'NoodleApp',
    'NoodleAppConfig',
    'ProjectConfig',
    'HeadlessLLMClient',
    'LLMConfig',
    'create_llm_client_from_env',
]

__version__ = '1.0.0'
