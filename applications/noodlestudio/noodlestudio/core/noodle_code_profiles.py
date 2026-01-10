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
#   Noodle Code Profiles - Customizable AI assistant personalities
#
#   Profiles are markdown files that define the personality, ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.noodle_code_profiles
# PURPOSE:  Noodle Code Profiles
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   Profile, NoodleCodeProfileManager, get_profile_manager()
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
from typing import Dict, List, Optional
from dataclasses import dataclass

from PyQt6.QtCore import QSettings


@dataclass
class Profile:
    """A Noodle Code personality profile."""
    name: str
    path: Path
    content: str
    description: str = ""
    is_builtin: bool = False

    @classmethod
    def from_file(cls, path: Path, is_builtin: bool = False) -> 'Profile':
        """Load a profile from a markdown file."""
        content = path.read_text()

        # Extract description from first line if it's a comment
        lines = content.strip().split('\n')
        description = ""
        if lines and lines[0].startswith('<!--') and '-->' in lines[0]:
            description = lines[0].replace('<!--', '').replace('-->', '').strip()
        elif lines and lines[0].startswith('#'):
            description = lines[0].lstrip('#').strip()

        return cls(
            name=path.stem,
            path=path,
            content=content,
            description=description,
            is_builtin=is_builtin
        )


class NoodleCodeProfileManager:
    """
    Manages Noodle Code personality profiles.

    Profiles are stored in:
    - Built-in: <app>/noodlestudio/noodle_code_profiles/
    - User: ~/.noodlestudio/noodlecode_profiles/
    """

    # Default profile content (embedded for first-run)
    BUILTIN_PROFILES = {
        'default': '''# Default

Curious, helpful, direct. A reliable coding partner who acts decisively.

## Core Principle
**Act first, ask only when truly blocked.** You have tools - use them. Don't ask
"do you have access to X?" when you can just try reading it. Don't ask 5 questions
when you can grep the codebase and find the answer yourself.

## Style
- Professional and clear
- When asked to fix something: find the code, read it, propose a fix
- No emojis unless the user uses them first
- Paragraph breaks between distinct thoughts
- Honest technical assessments - if something is wrong, say so directly
- Stay on task - solve the immediate problem before philosophizing

## Anti-patterns to avoid
- Asking permission to look at code you can already access
- Listing questions when you could just investigate
- Drifting into abstract discussions when there's concrete work to do
- Being overly deferential - you're a capable partner, not an assistant asking for guidance
''',
        'creative': '''# Creative Mode

Ideas flow fast. Prototype rapidly. Energy is HIGH.

## Style
- Enthusiastic about possibilities
- Suggest bold alternatives
- "What if we tried..." is your favorite phrase
- Move fast, we can refine later
- Brainstorm freely before narrowing down
''',
        'architect': '''# Architect Mode

Step back. See the whole system. Design before coding.

## Style
- Ask clarifying questions before writing code
- Consider edge cases and implications
- Draw out the full design first
- Think about maintainability and extensibility
- "Let me understand the requirements..." is your opener
''',
        'reviewer': '''# Code Review Mode

Critical eye. Find the problems. Improve the code.

## Style
- Look for bugs, edge cases, security issues
- Question assumptions
- Suggest improvements to structure and clarity
- Be direct about problems - sugar-coating helps no one
- "I noticed a potential issue..." is your signature
''',
        'mlx': '''# MLX Specialist

Deep expertise in Apple's MLX framework for machine learning.

## Domain Knowledge
- MLX uses lazy evaluation - ops don't execute until needed
- Use mx.eval() to force computation
- Metal GPU acceleration is automatic
- Memory format is channels-last by default
- mx.compile() for optimization

## Style
- Reference MLX best practices
- Suggest Metal-optimized approaches
- Know the MLX ecosystem (mlx-lm, mlx-whisper, etc.)
''',
    }

    def __init__(self):
        self._profiles: Dict[str, Profile] = {}
        self._current_profile_name: str = "default"
        self._settings = QSettings("Noodlings", "NoodleStudio")

        # Initialize directories and load profiles
        self._ensure_directories()
        self._load_profiles()
        self._restore_last_profile()

    def _get_builtin_dir(self) -> Path:
        """Get built-in profiles directory."""
        return Path(__file__).parent / "noodle_code_profiles"

    def _get_user_dir(self) -> Path:
        """Get user profiles directory."""
        return Path.home() / ".noodlestudio" / "noodlecode_profiles"

    def _ensure_directories(self):
        """Create profile directories if they don't exist."""
        # Create user directory
        user_dir = self._get_user_dir()
        user_dir.mkdir(parents=True, exist_ok=True)

        # Create built-in directory and write default profiles
        builtin_dir = self._get_builtin_dir()
        builtin_dir.mkdir(parents=True, exist_ok=True)

        for name, content in self.BUILTIN_PROFILES.items():
            profile_path = builtin_dir / f"{name}.md"
            if not profile_path.exists():
                profile_path.write_text(content)

    def _load_profiles(self):
        """Load all profiles from both directories."""
        self._profiles.clear()

        # Load built-in profiles
        builtin_dir = self._get_builtin_dir()
        if builtin_dir.exists():
            for path in builtin_dir.glob("*.md"):
                try:
                    profile = Profile.from_file(path, is_builtin=True)
                    self._profiles[profile.name] = profile
                except Exception as e:
                    print(f"[Profiles] Error loading {path}: {e}")

        # Load user profiles (can override built-ins)
        user_dir = self._get_user_dir()
        if user_dir.exists():
            for path in user_dir.glob("*.md"):
                try:
                    profile = Profile.from_file(path, is_builtin=False)
                    self._profiles[profile.name] = profile
                except Exception as e:
                    print(f"[Profiles] Error loading {path}: {e}")

        # Ensure default exists
        if "default" not in self._profiles:
            self._profiles["default"] = Profile(
                name="default",
                path=Path(""),
                content=self.BUILTIN_PROFILES["default"],
                description="Default profile",
                is_builtin=True
            )

    def _restore_last_profile(self):
        """Restore the last used profile from settings."""
        last_profile = self._settings.value("noodlecode/profile", "default")
        if last_profile in self._profiles:
            self._current_profile_name = last_profile
        else:
            self._current_profile_name = "default"

    def get_profile_names(self) -> List[str]:
        """Get list of available profile names, sorted with default first."""
        names = list(self._profiles.keys())
        # Sort with default first, then alphabetically
        names.sort(key=lambda x: (x != "default", x))
        return names

    def get_profile(self, name: str) -> Optional[Profile]:
        """Get a profile by name."""
        return self._profiles.get(name)

    def get_current_profile(self) -> Profile:
        """Get the currently selected profile."""
        return self._profiles.get(self._current_profile_name,
                                   self._profiles.get("default"))

    def set_current_profile(self, name: str) -> bool:
        """Set the current profile by name."""
        if name in self._profiles:
            self._current_profile_name = name
            self._settings.setValue("noodlecode/profile", name)
            print(f"[Profiles] Switched to: {name}")
            return True
        return False

    @property
    def current_profile_name(self) -> str:
        """Get the name of the current profile."""
        return self._current_profile_name

    def get_profile_prompt(self, name: Optional[str] = None) -> str:
        """Get the prompt text for a profile (or current if name is None)."""
        if name is None:
            profile = self.get_current_profile()
        else:
            profile = self.get_profile(name)

        if profile:
            return profile.content
        return ""

    def reload_profiles(self):
        """Reload all profiles from disk."""
        self._load_profiles()

    def create_user_profile(self, name: str, content: str) -> bool:
        """Create a new user profile."""
        if not name or '/' in name or '\\' in name:
            return False

        user_dir = self._get_user_dir()
        profile_path = user_dir / f"{name}.md"

        try:
            profile_path.write_text(content)
            self._load_profiles()
            return True
        except Exception as e:
            print(f"[Profiles] Error creating profile: {e}")
            return False


# Singleton instance
_profile_manager: Optional[NoodleCodeProfileManager] = None


def get_profile_manager() -> NoodleCodeProfileManager:
    """Get the global profile manager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = NoodleCodeProfileManager()
    return _profile_manager

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
