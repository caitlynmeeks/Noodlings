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
#   Noodle Code Tools - File and project operations for AI assistant
#
#   Provides Claude Code-style tools for reading, writing, an...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.noodle_code_tools
# PURPOSE:  Noodle Code Tools
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ToolResult, NoodleCodeTools
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import re
import subprocess
import fnmatch
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    # For computer_use screenshots
    image_base64: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NoodleCodeTools:
    """
    Tools available to Noodle Code.

    All file operations are scoped to the project directory for security.
    """

    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path

    def set_project_path(self, path: Path):
        """Set the project path (called when project opens)."""
        self.project_path = path

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return tool schemas for LLM."""
        return [
            {
                "name": "read_file",
                "description": "Read the contents of a file. Returns line-numbered output.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (relative to project or absolute)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start from (1-based). Default: 1"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum lines to read. Default: 500"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "Create or overwrite a file with new content.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (relative to project or absolute)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "edit_file",
                "description": "Replace a specific string in a file. The old_string must match exactly.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (relative to project or absolute)"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Exact string to find and replace"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "String to replace it with"
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            },
            {
                "name": "glob",
                "description": "Find files matching a glob pattern (e.g., '**/*.py', 'components/*.yaml')",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match files"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in (default: project root)"
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "grep",
                "description": "Search for a pattern in files. Returns matching lines with context.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for"
                        },
                        "path": {
                            "type": "string",
                            "description": "File or directory to search in (default: project root)"
                        },
                        "glob_pattern": {
                            "type": "string",
                            "description": "Only search files matching this glob (e.g., '*.py')"
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Lines of context before/after match. Default: 2"
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "bash",
                "description": "Execute a shell command. Use for git, npm, pip, etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds. Default: 30"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "list_directory",
                "description": "List contents of a directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path (default: project root)"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "List recursively. Default: false"
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max depth for recursive listing. Default: 3"
                        }
                    }
                }
            },
            {
                "name": "hot_reload",
                "description": "Hot-reload a Python module to apply code changes without restarting. Only works for safe modules (tools, facets, APIs).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "module_name": {
                            "type": "string",
                            "description": "Full module path (e.g., 'noodlestudio.core.utility_facets')"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Alternative: path to .py file to reload"
                        }
                    }
                }
            },
            {
                "name": "soft_restart",
                "description": "Restart NoodleStudio with state preservation. Use this when you've edited panel classes, mixins, or core singletons that can't be hot-reloaded. Current project, tabs, and selection will be restored.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation of why restart is needed"
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Set to true to actually trigger restart. If false or omitted, returns info about what would happen."
                        }
                    }
                }
            },
            {
                "name": "computer_use",
                "description": """Control NoodleStudio's UI - take screenshots, click buttons, type text, drag elements.

USE THIS TOOL WHEN:
- User asks you to interact with the UI (click, type, navigate)
- You need to see what's currently on screen
- You need to demonstrate how to use a feature
- Debugging UI issues visually

WORKFLOW:
1. First take a screenshot: action="screenshot"
2. The screenshot returns a UI ELEMENT MAP with EXACT coordinates for all clickable elements
3. Use the provided coordinates directly - DO NOT try to read coordinates from the image
4. Click using: action="left_click", coordinate=[x, y]
5. Type text if needed: action="type", text="hello"

UI ELEMENT MAP (IMPORTANT):
- Screenshots include a list of all clickable UI elements (tabs, buttons, inputs)
- Each element has EXACT coordinates from the Qt widget tree
- Example: "Tab: Noodle Code -> (257, 11)" means click at [257, 11]
- Trust these coordinates - they come directly from Qt, not image analysis

COORDINATES: [x, y] relative to NoodleStudio window.""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["screenshot", "ui_elements", "mouse_move", "left_click", "right_click", "double_click", "middle_click", "type", "key", "scroll", "drag"],
                            "description": "The action to perform. Use 'ui_elements' to get just the clickable element list without a screenshot."
                        },
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "[x, y] coordinates for mouse actions (relative to window)"
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to type (for 'type' action) or key combo (for 'key' action, e.g. 'ctrl+s', 'enter', 'tab')"
                        },
                        "scroll_direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Direction to scroll (for 'scroll' action)"
                        },
                        "scroll_amount": {
                            "type": "integer",
                            "description": "Pixels to scroll. Default: 120"
                        },
                        "start_coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "[x, y] start position for drag"
                        },
                        "end_coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "[x, y] end position for drag"
                        }
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "github",
                "description": """GitHub CLI (gh) for issues, PRs, and repository operations.

COMMON OPERATIONS:
- List issues: command="issue list"
- View issue: command="issue view 42"
- Create issue: command="issue create --title 'Bug: X' --label bug --body 'Description'"
- List PRs: command="pr list"
- View PR: command="pr view 123"
- Create PR: command="pr create --title 'Feature: X' --body 'Description'"
- Search: command="search issues 'crash' --repo owner/repo"
- Repo info: command="repo view"

The repository context is auto-detected from the project's git config.""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The gh subcommand to run (e.g., 'issue list', 'pr view 42')"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "run_ui_test",
                "description": """Run UI tests using Computer Use to actually click the UI.

USE THIS WHEN:
- User asks "did I break anything?" -> run smoke tests
- User asks to test specific feature -> run targeted test
- Before committing changes -> run critical tests
- User asks to run smoke/e2e/panel tests

RETURNS:
- Pass/fail status per test
- Duration
- Any failures with screenshots
- Suggestions for fixes

EXAMPLES:
  # Run all smoke tests
  {"suite": "smoke"}

  # Run specific test file
  {"test": "tests/ui/smoke/panels.ui-test.yaml"}

  # Run tests matching pattern
  {"pattern": "**/facet*.ui-test.yaml"}

  # Run with visual mode (ghost cursor)
  {"suite": "smoke", "visual": true}""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "test": {
                            "type": "string",
                            "description": "Path to specific test file"
                        },
                        "suite": {
                            "type": "string",
                            "enum": ["smoke", "e2e", "panels", "facets", "all"],
                            "description": "Test suite to run"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern for test files"
                        },
                        "visual": {
                            "type": "boolean",
                            "description": "Show ghost cursor during tests (default: true)"
                        },
                        "stop_on_failure": {
                            "type": "boolean",
                            "description": "Stop at first failure (default: true)"
                        }
                    }
                }
            },
            {
                "name": "ai_verify_ui",
                "description": """Take a screenshot and verify UI state using AI vision.

USE THIS WHEN:
- Need to verify something not easily checkable programmatically
- Want to catch visual regressions
- Checking "does this look right?"
- Verifying UI matches expected state

Takes a screenshot and you (Claude) verify specific aspects of the UI state.

EXAMPLES:
  # Verify inspector shows correct data
  {"verify": "Inspector panel shows Position X = 100, Y = 200"}

  # Check for visual issues
  {"verify": "No overlapping text or clipped elements visible"}

  # Verify layout
  {"verify": "Facets panel has 3 nodes: Perception, Memory, Response"}

  # Check specific region
  {"verify": "The chat input field is empty", "region": "chat_panel"}""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "verify": {
                            "type": "string",
                            "description": "What to verify in the screenshot"
                        },
                        "region": {
                            "type": "string",
                            "description": "Optional: specific panel/area to focus on"
                        }
                    },
                    "required": ["verify"]
                }
            },
            {
                "name": "capture_visual_baseline",
                "description": """Capture current UI as a visual baseline for testing.

USE THIS WHEN:
- Setting up visual tests for the first time
- UI has intentionally changed and baseline needs updating
- Human verifies "this looks correct" and wants to save it

RETURNS:
- Path to saved baseline image
- Screenshot of the captured baseline

EXAMPLES:
  # After importing Ajo and verifying it looks correct
  {"name": "ajo_imported_textured"}

  # After opening File menu
  {"name": "file_menu_open"}

  # After creating a new character
  {"name": "new_character_dialog"}""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the baseline (e.g., 'ajo_imported', 'file_menu_open')"
                        }
                    },
                    "required": ["name"]
                }
            }
        ]

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to project, with security checks."""
        p = Path(path)

        # If absolute, use as-is but verify it's within allowed areas
        if p.is_absolute():
            resolved = p.resolve()
        else:
            # Relative to project
            if self.project_path:
                resolved = (self.project_path / path).resolve()
            else:
                resolved = Path(path).resolve()

        # Security: prevent escaping project directory for relative paths
        if self.project_path and not str(path).startswith('/'):
            try:
                resolved.relative_to(self.project_path.resolve())
            except ValueError:
                raise PermissionError(f"Path escapes project directory: {path}")

        return resolved

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool and return result."""
        handler = getattr(self, f"tool_{tool_name}", None)
        if not handler:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}"
            )

        try:
            return await handler(**args)
        except PermissionError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")

    # ========== FILE TOOLS ==========

    async def tool_read_file(
        self,
        path: str,
        offset: int = 1,
        limit: int = 500
    ) -> ToolResult:
        """Read file contents with line numbers."""
        try:
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}"
                )

            if full_path.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Path is a directory: {path}"
                )

            content = full_path.read_text(errors='replace')
            lines = content.splitlines()

            # Convert to 0-based index
            start_idx = max(0, offset - 1)
            end_idx = start_idx + limit
            selected = lines[start_idx:end_idx]

            # Format with line numbers
            result = []
            for i, line in enumerate(selected, start=start_idx + 1):
                result.append(f"{i:5d} | {line}")

            output = "\n".join(result)

            # Add truncation notice if applicable
            if end_idx < len(lines):
                output += f"\n\n... ({len(lines) - end_idx} more lines)"

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def tool_write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        try:
            full_path = self._resolve_path(path)

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            full_path.write_text(content)

            line_count = len(content.splitlines())
            return ToolResult(
                success=True,
                output=f"Wrote {line_count} lines to {path}"
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def tool_edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str
    ) -> ToolResult:
        """Replace a string in a file."""
        try:
            full_path = self._resolve_path(path)

            if not full_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}"
                )

            content = full_path.read_text()

            # Check if old_string exists
            if old_string not in content:
                # Show similar content to help debug
                return ToolResult(
                    success=False,
                    output="",
                    error=f"String not found in file. Make sure it matches exactly including whitespace."
                )

            # Count occurrences
            count = content.count(old_string)
            if count > 1:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"String found {count} times. Please provide more context to make it unique."
                )

            # Perform replacement
            new_content = content.replace(old_string, new_string, 1)
            full_path.write_text(new_content)

            return ToolResult(
                success=True,
                output=f"Edited {path}: replaced 1 occurrence"
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    # ========== SEARCH TOOLS ==========

    async def tool_glob(
        self,
        pattern: str,
        path: Optional[str] = None
    ) -> ToolResult:
        """Find files matching glob pattern."""
        try:
            if path:
                base = self._resolve_path(path)
            elif self.project_path:
                base = self.project_path
            else:
                base = Path.cwd()

            if not base.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a directory: {path}"
                )

            # Find matching files
            matches = list(base.glob(pattern))

            # Sort by modification time (newest first)
            matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            # Limit results
            max_results = 100
            truncated = len(matches) > max_results
            matches = matches[:max_results]

            # Format output
            result = []
            for p in matches:
                try:
                    rel = p.relative_to(base)
                except ValueError:
                    rel = p
                result.append(str(rel))

            output = "\n".join(result)
            if truncated:
                output += f"\n\n... (truncated, {len(matches)} total matches)"
            elif not result:
                output = "No files found matching pattern"

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def tool_grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob_pattern: Optional[str] = None,
        context_lines: int = 2
    ) -> ToolResult:
        """Search for pattern in files."""
        try:
            if path:
                base = self._resolve_path(path)
            elif self.project_path:
                base = self.project_path
            else:
                base = Path.cwd()

            regex = re.compile(pattern, re.IGNORECASE)
            results = []
            files_searched = 0
            max_results = 50

            # Determine files to search
            if base.is_file():
                files = [base]
            else:
                if glob_pattern:
                    files = list(base.rglob(glob_pattern))
                else:
                    # Default: search common text files
                    files = []
                    for ext in ['*.py', '*.yaml', '*.yml', '*.json', '*.md', '*.txt', '*.js']:
                        files.extend(base.rglob(ext))

            # Skip binary/large files
            max_file_size = 1024 * 1024  # 1MB

            for file_path in files:
                if len(results) >= max_results:
                    break

                if not file_path.is_file():
                    continue

                if file_path.stat().st_size > max_file_size:
                    continue

                files_searched += 1

                try:
                    content = file_path.read_text(errors='replace')
                    lines = content.splitlines()

                    for i, line in enumerate(lines):
                        if regex.search(line):
                            # Get context
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)

                            try:
                                rel_path = file_path.relative_to(base)
                            except ValueError:
                                rel_path = file_path

                            context_block = []
                            for j in range(start, end):
                                prefix = ">" if j == i else " "
                                context_block.append(f"{prefix} {j+1:4d} | {lines[j]}")

                            results.append(f"{rel_path}:\n" + "\n".join(context_block))

                            if len(results) >= max_results:
                                break

                except Exception:
                    continue  # Skip files that can't be read

            if not results:
                output = f"No matches found (searched {files_searched} files)"
            else:
                output = f"Found {len(results)} matches:\n\n" + "\n\n".join(results)
                if len(results) >= max_results:
                    output += f"\n\n... (truncated at {max_results} results)"

            return ToolResult(success=True, output=output)

        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regex: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    # ========== EXECUTION TOOLS ==========

    async def tool_bash(
        self,
        command: str,
        timeout: int = 30
    ) -> ToolResult:
        """Execute a shell command."""
        # Security: block obviously dangerous commands
        dangerous_patterns = [
            r'rm\s+-rf\s+/',
            r'sudo\s+rm',
            r'mkfs',
            r'dd\s+if=',
            r':\(\)\{',  # Fork bomb
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                return ToolResult(
                    success=False,
                    output="",
                    error="Command blocked for safety"
                )

        try:
            # Run in project directory if available
            cwd = str(self.project_path) if self.project_path else None

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )

            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate if too long
            max_output = 10000
            if len(output) > max_output:
                output = output[:max_output] + f"\n\n... (truncated, {len(output)} total chars)"

            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    output=output,
                    error=f"Command exited with code {result.returncode}"
                )

            return ToolResult(success=True, output=output)

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s"
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def tool_list_directory(
        self,
        path: Optional[str] = None,
        recursive: bool = False,
        max_depth: int = 3
    ) -> ToolResult:
        """List directory contents."""
        try:
            if path:
                base = self._resolve_path(path)
            elif self.project_path:
                base = self.project_path
            else:
                base = Path.cwd()

            if not base.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a directory: {path}"
                )

            def list_dir(dir_path: Path, depth: int = 0) -> List[str]:
                if depth > max_depth:
                    return []

                entries = []
                try:
                    items = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                except PermissionError:
                    return [f"{'  ' * depth}[permission denied]"]

                for item in items:
                    # Skip hidden files and common ignored dirs
                    if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules', '.git']:
                        continue

                    prefix = "  " * depth
                    if item.is_dir():
                        entries.append(f"{prefix}{item.name}/")
                        if recursive:
                            entries.extend(list_dir(item, depth + 1))
                    else:
                        entries.append(f"{prefix}{item.name}")

                return entries

            entries = list_dir(base)

            if not entries:
                output = "(empty directory)"
            else:
                output = "\n".join(entries[:500])  # Limit output
                if len(entries) > 500:
                    output += f"\n\n... ({len(entries) - 500} more entries)"

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    # ========== DEVELOPMENT TOOLS ==========

    async def tool_hot_reload(
        self,
        module_name: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> ToolResult:
        """
        Hot-reload a Python module to apply code changes.

        Only works for safe modules (tools, facets, APIs).
        Panel classes and core singletons require app restart.
        """
        try:
            from .hot_reload import get_hot_reload_manager

            manager = get_hot_reload_manager()

            # Determine module to reload
            if module_name:
                target = module_name
            elif file_path:
                # Convert file path to module
                result = manager.reload_file(Path(file_path))
                if result.success:
                    return ToolResult(
                        success=True,
                        output=f"Hot-reloaded: {result.module_name}\n{result.message}"
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=result.message,
                        error=result.error
                    )
            else:
                # List available safe modules
                safe = manager.get_safe_modules()
                output = "Available safe modules for hot-reload:\n\n"
                for mod in sorted(safe):
                    output += f"  - {mod}\n"
                output += "\nPass module_name or file_path to reload."
                return ToolResult(success=True, output=output)

            # Check if safe
            can_reload, reason = manager.can_reload(target)
            if not can_reload:
                return ToolResult(
                    success=False,
                    output=f"Cannot hot-reload {target}: {reason}\n\nThis module requires an app restart to apply changes.",
                    error=reason
                )

            # Reload
            result = manager.reload_module(target)

            if result.success:
                return ToolResult(
                    success=True,
                    output=f"Hot-reloaded: {target}\n\nChanges are now live. No restart required."
                )
            else:
                return ToolResult(
                    success=False,
                    output=result.message,
                    error=result.error
                )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def tool_soft_restart(
        self,
        reason: str = "Code changes applied",
        confirm: bool = False
    ) -> ToolResult:
        """
        Trigger a soft restart of NoodleStudio.

        Use this when you've edited files that can't be hot-reloaded
        (panel classes, mixins, core singletons).
        """
        try:
            # Get list of unsafe modules for context
            from .hot_reload import get_hot_reload_manager
            manager = get_hot_reload_manager()

            if not confirm:
                # Just return info about what would happen
                unsafe_examples = list(manager.UNSAFE_MODULES)[:5]
                return ToolResult(
                    success=True,
                    output=(
                        f"Soft Restart would:\n"
                        f"1. Save current state (project, tabs, selection)\n"
                        f"2. Restart NoodleStudio\n"
                        f"3. Restore saved state\n\n"
                        f"Use this when you've edited modules like:\n"
                        f"  - {chr(10).join('  - ' + m for m in unsafe_examples)}\n"
                        f"  ... and {len(manager.UNSAFE_MODULES) - 5} more\n\n"
                        f"To trigger restart, call soft_restart with confirm=true"
                    )
                )

            # Actually trigger restart
            # We need to get the main window reference
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if not app:
                return ToolResult(
                    success=False,
                    output="",
                    error="No QApplication instance found"
                )

            # Find main window
            main_window = None
            for widget in app.topLevelWidgets():
                if widget.__class__.__name__ == 'MainWindow':
                    main_window = widget
                    break

            if not main_window:
                return ToolResult(
                    success=False,
                    output="",
                    error="MainWindow not found"
                )

            # Schedule the restart (can't do it synchronously from async)
            from PyQt6.QtCore import QTimer
            from .soft_restart import perform_soft_restart

            def do_restart():
                perform_soft_restart(main_window, reason)

            QTimer.singleShot(100, do_restart)

            return ToolResult(
                success=True,
                output=f"Soft restart initiated.\nReason: {reason}\n\nNoodleStudio will restart momentarily..."
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def tool_computer_use(
        self,
        action: str,
        coordinate: list = None,
        text: str = None,
        duration: int = None,
        scroll_direction: str = None,
        scroll_amount: int = None,
        start_coordinate: list = None,
        button: str = "left"
    ) -> ToolResult:
        """
        Claude Computer Use - See and interact with NoodleStudio's UI.

        This tool enables you to control NoodleStudio like a human user would:
        take screenshots to see the UI, then click, type, and navigate.

        Actions:
            screenshot: Capture current window state (returns base64 PNG)
            mouse_move: Move cursor to coordinate [x, y]
            left_click: Click left button at coordinate [x, y]
            right_click: Click right button at coordinate [x, y]
            double_click: Double-click at coordinate [x, y]
            middle_click: Click middle button at coordinate [x, y]
            type: Type text string into focused widget
            key: Press key combination (e.g., "return", "ctrl+s", "escape")
            scroll: Scroll at coordinate (scroll_direction: "up"/"down"/"left"/"right")
            drag: Drag from start_coordinate to coordinate

        Coordinates are relative to NoodleStudio window (0,0 = top-left).

        Workflow:
            1. Take screenshot to see current state
            2. Analyze UI to find target element
            3. Perform action (click, type, etc.)
            4. Take another screenshot to verify result

        Examples:
            # See the current UI
            {"action": "screenshot"}

            # Click File menu (assuming it's at x=45, y=12)
            {"action": "left_click", "coordinate": [45, 12]}

            # Type text
            {"action": "type", "text": "Hello World"}

            # Press Enter
            {"action": "key", "text": "return"}

            # Save with Ctrl+S
            {"action": "key", "text": "ctrl+s"}

            # Scroll down
            {"action": "scroll", "coordinate": [400, 300], "scroll_direction": "down"}

            # Drag and drop
            {"action": "drag", "start_coordinate": [100, 100], "coordinate": [200, 200]}
        """
        try:
            # Get the computer use controller - do NOT reload, keep the singleton
            from .computer_use_controller import get_computer_use_controller
            controller = get_computer_use_controller()
            
            # If controller doesn't have main_window, auto-discover from running app
            debug_info = ["v2"]  # Version marker to verify code is running
            if not controller.main_window:
                from PyQt6.QtWidgets import QApplication, QMainWindow
                app = QApplication.instance()
                if app:
                    debug_info.append(f"App: {app}")
                    widgets = app.topLevelWidgets()
                    debug_info.append(f"Widgets: {len(widgets)}")
                    for widget in widgets:
                        winfo = f"{widget.__class__.__name__} vis={widget.isVisible()} mw={isinstance(widget, QMainWindow)}"
                        debug_info.append(winfo)
                        # Look for any QMainWindow instance  
                        if isinstance(widget, QMainWindow) and widget.isVisible():
                            controller.set_main_window(widget)
                            debug_info.append(f"SET OK: {widget}")
                            break
                else:
                    debug_info.append("No app!")

            if not controller.main_window:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Computer Use not initialized. MainWindow not set. Debug: {'; '.join(debug_info)}"
                )

            action = action.lower()

            # Screenshot
            if action == "screenshot":
                # Take screenshot WITHOUT rulers (they confused the model)
                b64_data, width, height = controller.screenshot(add_rulers=False)

                # Get actual UI element positions from Qt widget tree
                ui_summary = controller.get_ui_summary()

                return ToolResult(
                    success=True,
                    output=(
                        f"Screenshot captured: {width}x{height} pixels.\n\n"
                        f"{ui_summary}\n\n"
                        f"Use the coordinates above to click elements. "
                        f"The coordinates are EXACT - no need to estimate from the image."
                    ),
                    image_base64=b64_data,
                    metadata={"width": width, "height": height}
                )

            # Just get UI elements without screenshot
            elif action == "ui_elements":
                ui_summary = controller.get_ui_summary()
                return ToolResult(
                    success=True,
                    output=(
                        f"Current UI element positions:\n\n{ui_summary}\n\n"
                        f"Use these coordinates directly for clicking."
                    )
                )

            # Calibration screenshot with crosshairs at known coordinates
            elif action == "calibrate":
                b64_data, width, height, cal_points = controller.screenshot_with_calibration()
                points_str = ", ".join([f"{label}=({x},{y})" for x, y, label in cal_points])
                return ToolResult(
                    success=True,
                    output=(
                        f"CALIBRATION MODE: {width}x{height} pixels\n\n"
                        f"Red crosshairs with yellow centers are drawn at these EXACT coordinates:\n"
                        f"{points_str}\n\n"
                        f"Each crosshair is labeled like 'E(640,400)' meaning point E is at x=640, y=400.\n"
                        f"Compare what you SEE in the image to these known coordinates.\n"
                        f"If they match, your coordinate reading is accurate.\n"
                        f"If they differ, note the offset pattern."
                    ),
                    image_base64=b64_data,
                    metadata={"width": width, "height": height, "calibration_points": cal_points}
                )

            # Mouse move
            elif action == "mouse_move":
                if not coordinate or len(coordinate) < 2:
                    return ToolResult(
                        success=False,
                        output="",
                        error="mouse_move requires coordinate: [x, y]"
                    )
                success = controller.mouse_move(coordinate[0], coordinate[1])
                return ToolResult(
                    success=success,
                    output=f"Mouse moved to ({coordinate[0]}, {coordinate[1]})" if success else "",
                    error=None if success else "Mouse move failed"
                )

            # Clicks
            elif action in ("left_click", "click"):
                if not coordinate or len(coordinate) < 2:
                    return ToolResult(
                        success=False,
                        output="",
                        error="left_click requires coordinate: [x, y]"
                    )
                success = controller.click(coordinate[0], coordinate[1], "left")
                return ToolResult(
                    success=success,
                    output=f"Left click at ({coordinate[0]}, {coordinate[1]})" if success else "",
                    error=None if success else "Click failed"
                )

            elif action == "right_click":
                if not coordinate or len(coordinate) < 2:
                    return ToolResult(
                        success=False,
                        output="",
                        error="right_click requires coordinate: [x, y]"
                    )
                success = controller.click(coordinate[0], coordinate[1], "right")
                return ToolResult(
                    success=success,
                    output=f"Right click at ({coordinate[0]}, {coordinate[1]})" if success else "",
                    error=None if success else "Click failed"
                )

            elif action == "middle_click":
                if not coordinate or len(coordinate) < 2:
                    return ToolResult(
                        success=False,
                        output="",
                        error="middle_click requires coordinate: [x, y]"
                    )
                success = controller.click(coordinate[0], coordinate[1], "middle")
                return ToolResult(
                    success=success,
                    output=f"Middle click at ({coordinate[0]}, {coordinate[1]})" if success else "",
                    error=None if success else "Click failed"
                )

            elif action == "double_click":
                if not coordinate or len(coordinate) < 2:
                    return ToolResult(
                        success=False,
                        output="",
                        error="double_click requires coordinate: [x, y]"
                    )
                success = controller.double_click(coordinate[0], coordinate[1], button)
                return ToolResult(
                    success=success,
                    output=f"Double click at ({coordinate[0]}, {coordinate[1]})" if success else "",
                    error=None if success else "Double click failed"
                )

            # Type text
            elif action == "type":
                if not text:
                    return ToolResult(
                        success=False,
                        output="",
                        error="type requires text parameter"
                    )
                success = controller.type_text(text)
                display = text[:50] + "..." if len(text) > 50 else text
                return ToolResult(
                    success=success,
                    output=f"Typed: '{display}'" if success else "",
                    error=None if success else "Type failed"
                )

            # Key press
            elif action == "key":
                if not text:
                    return ToolResult(
                        success=False,
                        output="",
                        error="key requires text parameter (e.g., 'return', 'ctrl+s')"
                    )
                success = controller.key(text)
                return ToolResult(
                    success=success,
                    output=f"Key pressed: {text}" if success else "",
                    error=None if success else "Key press failed"
                )

            # Scroll
            elif action == "scroll":
                if not coordinate or len(coordinate) < 2:
                    return ToolResult(
                        success=False,
                        output="",
                        error="scroll requires coordinate: [x, y]"
                    )
                direction = scroll_direction or "down"
                amount = scroll_amount or 120

                # Convert direction to deltas
                delta_x, delta_y = 0, 0
                if direction == "up":
                    delta_y = amount
                elif direction == "down":
                    delta_y = -amount
                elif direction == "left":
                    delta_x = amount
                elif direction == "right":
                    delta_x = -amount

                success = controller.scroll(coordinate[0], coordinate[1], delta_x, delta_y)
                return ToolResult(
                    success=success,
                    output=f"Scrolled {direction} at ({coordinate[0]}, {coordinate[1]})" if success else "",
                    error=None if success else "Scroll failed"
                )

            # Drag
            elif action == "drag":
                if not start_coordinate or len(start_coordinate) < 2:
                    return ToolResult(
                        success=False,
                        output="",
                        error="drag requires start_coordinate: [x, y]"
                    )
                if not coordinate or len(coordinate) < 2:
                    return ToolResult(
                        success=False,
                        output="",
                        error="drag requires coordinate (end position): [x, y]"
                    )
                success = controller.drag(
                    start_coordinate[0], start_coordinate[1],
                    coordinate[0], coordinate[1],
                    button
                )
                return ToolResult(
                    success=success,
                    output=f"Dragged from ({start_coordinate[0]}, {start_coordinate[1]}) to ({coordinate[0]}, {coordinate[1]})" if success else "",
                    error=None if success else "Drag failed"
                )

            # Get window info
            elif action == "window_info":
                width, height = controller.get_window_size()
                history = controller.get_action_history()[-5:]
                return ToolResult(
                    success=True,
                    output=(
                        f"NoodleStudio Window: {width}x{height} pixels\n"
                        f"Recent actions: {len(history)}\n" +
                        "\n".join(f"  - {a['action']}" + (f" at {a['coordinate']}" if a['coordinate'] else "") for a in history)
                    )
                )

            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown action: {action}. Valid: screenshot, mouse_move, left_click, right_click, middle_click, double_click, type, key, scroll, drag, window_info"
                )

        except Exception as e:
            import traceback
            return ToolResult(
                success=False,
                output="",
                error=f"Computer Use error: {e}\n{traceback.format_exc()}"
            )

    async def tool_github(self, command: str) -> ToolResult:
        """
        GitHub CLI (gh) for repository operations.

        Runs `gh <command>` in the project directory.

        Examples:
            command="issue list"
            command="issue view 42"
            command="issue create --title 'Bug' --label bug"
            command="pr list"
            command="pr view 123"
            command="pr create --title 'Feature'"
            command="repo view"
            command="search issues 'crash'"

        Returns:
            ToolResult with gh output or error
        """
        import shutil

        # Check if gh is installed
        gh_path = shutil.which("gh")
        if not gh_path:
            return ToolResult(
                success=False,
                output="",
                error="GitHub CLI (gh) is not installed. Install with: brew install gh"
            )

        # Security: prevent shell injection
        # Only allow alphanumeric, spaces, dashes, underscores, quotes, colons, slashes
        import re
        if re.search(r'[;&|`$(){}]', command):
            return ToolResult(
                success=False,
                output="",
                error="Invalid characters in command. Shell operators not allowed."
            )

        try:
            # Run gh command
            full_command = f"gh {command}"
            cwd = str(self.project_path) if self.project_path else None

            result = subprocess.run(
                full_command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "GH_FORCE_TTY": "0"}  # Disable TTY formatting
            )

            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}" if output else result.stderr

            if result.returncode == 0:
                return ToolResult(success=True, output=output.strip())
            else:
                # Check for common auth issue
                if "gh auth login" in output or "not logged in" in output.lower():
                    return ToolResult(
                        success=False,
                        output=output.strip(),
                        error="Not authenticated. Run 'gh auth login' in terminal first."
                    )
                return ToolResult(
                    success=False,
                    output=output.strip(),
                    error=f"gh command failed (exit {result.returncode})"
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Command timed out after 30 seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"GitHub CLI error: {e}"
            )

    # ========== UI TESTING TOOLS ==========

    async def tool_run_ui_test(
        self,
        test: str = None,
        suite: str = None,
        pattern: str = None,
        visual: bool = True,
        stop_on_failure: bool = True
    ) -> ToolResult:
        """
        Run UI tests using Computer Use.

        Runs actual UI tests that click, type, and verify the UI.
        Ghost cursor visualizes what the test is doing.
        """
        try:
            # Import test runner
            from ..testing.ui_test_runner import UITestRunner

            # Get main window
            from PyQt6.QtWidgets import QApplication, QMainWindow
            from PyQt6.QtCore import QThread, QMetaObject, Qt, Q_ARG
            import threading

            app = QApplication.instance()
            if not app:
                return ToolResult(
                    success=False,
                    output="",
                    error="No QApplication instance found"
                )

            main_window = None
            for widget in app.topLevelWidgets():
                if isinstance(widget, QMainWindow) and widget.isVisible():
                    main_window = widget
                    break

            if not main_window:
                return ToolResult(
                    success=False,
                    output="",
                    error="MainWindow not found"
                )

            # Check if we're on the main thread - if not, we need to run tests differently
            on_main_thread = QThread.currentThread() == app.thread()
            print(f"[run_ui_test] On main thread: {on_main_thread}")

            # If on background thread, disable visual mode (ghost cursor can't be created)
            # and run assertions synchronously using QMetaObject
            if not on_main_thread:
                visual = False
                print("[run_ui_test] Running on background thread, disabling visual mode")

            # Create runner
            runner = UITestRunner(main_window, visual_mode=visual)

            # Determine what to run
            test_files = []
            noodlestudio_path = Path(__file__).parent.parent
            tests_dir = noodlestudio_path / "tests" / "ui"

            if test:
                # Single test file
                test_path = Path(test)
                if not test_path.is_absolute():
                    test_path = tests_dir / test
                if test_path.exists():
                    test_files = [test_path]
                else:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Test file not found: {test}"
                    )

            elif suite:
                # Load suite from suites.yaml
                suites_file = tests_dir / "suites.yaml"
                if suites_file.exists():
                    import yaml
                    with open(suites_file) as f:
                        suites_config = yaml.safe_load(f) or {}
                    suite_def = suites_config.get("suites", {}).get(suite)
                    if suite_def:
                        for test_pattern in suite_def.get("tests", []):
                            test_files.extend(tests_dir.glob(test_pattern))
                    else:
                        # Fallback: suite name as directory
                        suite_dir = tests_dir / suite
                        if suite_dir.is_dir():
                            test_files = list(suite_dir.glob("*.ui-test.yaml"))
                else:
                    # No suites.yaml, use suite as directory name
                    suite_dir = tests_dir / suite
                    if suite_dir.is_dir():
                        test_files = list(suite_dir.glob("*.ui-test.yaml"))

            elif pattern:
                # Glob pattern
                test_files = list(tests_dir.glob(pattern))

            else:
                # Default: run smoke tests
                smoke_dir = tests_dir / "smoke"
                if smoke_dir.is_dir():
                    test_files = list(smoke_dir.glob("*.ui-test.yaml"))

            if not test_files:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"No test files found. Tests should be in: {tests_dir}"
                )

            # Run tests
            results = []
            total_passed = 0
            total_failed = 0
            total_duration = 0

            for test_file in sorted(test_files):
                result = await runner.run_test_file(str(test_file))
                results.append(result)
                total_duration += result.duration

                if result.success:
                    total_passed += 1
                else:
                    total_failed += 1
                    if stop_on_failure:
                        break

            # Format output
            output_lines = [
                f"UI Test Results",
                f"===============",
                f"",
                f"Passed: {total_passed}",
                f"Failed: {total_failed}",
                f"Duration: {total_duration:.2f}s",
                f"",
            ]

            for result in results:
                status = "PASS" if result.success else "FAIL"
                output_lines.append(f"[{status}] {result.name} ({result.duration:.2f}s)")
                if not result.success and result.error:
                    output_lines.append(f"       Error: {result.error}")

            overall_success = total_failed == 0
            output = "\n".join(output_lines)

            return ToolResult(
                success=overall_success,
                output=output,
                error=None if overall_success else f"{total_failed} test(s) failed"
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"UI testing module not available: {e}"
            )
        except Exception as e:
            import traceback
            return ToolResult(
                success=False,
                output="",
                error=f"UI test error: {e}\n{traceback.format_exc()}"
            )

    async def tool_ai_verify_ui(
        self,
        verify: str,
        region: str = None
    ) -> ToolResult:
        """
        Take a screenshot for AI visual verification.

        Returns the screenshot with the verification prompt so Claude
        can visually verify the UI state.
        """
        try:
            # Get computer use controller
            from .computer_use_controller import get_computer_use_controller
            controller = get_computer_use_controller()

            # Ensure main window is set
            if not controller.main_window:
                from PyQt6.QtWidgets import QApplication, QMainWindow
                app = QApplication.instance()
                if app:
                    for widget in app.topLevelWidgets():
                        if isinstance(widget, QMainWindow) and widget.isVisible():
                            controller.set_main_window(widget)
                            break

            if not controller.main_window:
                return ToolResult(
                    success=False,
                    output="",
                    error="MainWindow not found for screenshot"
                )

            # Take screenshot
            b64_data, width, height = controller.screenshot(add_rulers=False)

            # Get UI element summary for context
            ui_summary = controller.get_ui_summary()

            # Build verification context
            output_lines = [
                f"AI Visual Verification Request",
                f"===============================",
                f"",
                f"Screenshot: {width}x{height} pixels",
            ]

            if region:
                output_lines.append(f"Focus Region: {region}")

            output_lines.extend([
                f"",
                f"VERIFY: {verify}",
                f"",
                f"Please examine the screenshot and verify the above condition.",
                f"Report PASS if the condition is met, FAIL if not, with explanation.",
                f"",
                f"Current UI Elements:",
                ui_summary,
            ])

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                image_base64=b64_data,
                metadata={"width": width, "height": height, "verify": verify, "region": region}
            )

        except Exception as e:
            import traceback
            return ToolResult(
                success=False,
                output="",
                error=f"AI verify error: {e}\n{traceback.format_exc()}"
            )

    async def tool_capture_visual_baseline(self, name: str) -> ToolResult:
        """
        Capture current UI state as a visual baseline.

        Takes a screenshot and saves it to tests/ui/baselines/{name}.png
        for use with assert_visual in UI tests.
        """
        try:
            # Get computer use controller
            from .computer_use_controller import get_computer_use_controller
            controller = get_computer_use_controller()

            # Ensure main window is set
            if not controller.main_window:
                from PyQt6.QtWidgets import QApplication, QMainWindow
                app = QApplication.instance()
                if app:
                    for widget in app.topLevelWidgets():
                        if isinstance(widget, QMainWindow) and widget.isVisible():
                            controller.set_main_window(widget)
                            break

            if not controller.main_window:
                return ToolResult(
                    success=False,
                    output="",
                    error="MainWindow not found for screenshot"
                )

            # Take screenshot (without rulers for clean baseline)
            b64_data, width, height = controller.screenshot(add_rulers=False)

            # Save as baseline
            from ..testing.visual_verifier import VisualVerifier
            verifier = VisualVerifier()
            path = verifier.capture_baseline(name, b64_data)

            # List existing baselines for context
            baselines = verifier.list_baselines()

            return ToolResult(
                success=True,
                output=(
                    f"Visual baseline captured successfully.\n\n"
                    f"Name: {name}\n"
                    f"Path: {path}\n"
                    f"Dimensions: {width}x{height}\n\n"
                    f"Use in UI tests:\n"
                    f"  - action: assert_visual\n"
                    f"    baseline: \"{name}\"\n"
                    f"    threshold: 0.95\n\n"
                    f"Available baselines ({len(baselines)}):\n"
                    f"  {', '.join(baselines[:10])}"
                    f"{'...' if len(baselines) > 10 else ''}"
                ),
                image_base64=b64_data,
                metadata={"path": str(path), "width": width, "height": height}
            )

        except Exception as e:
            import traceback
            return ToolResult(
                success=False,
                output="",
                error=f"Baseline capture error: {e}\n{traceback.format_exc()}"
            )


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
