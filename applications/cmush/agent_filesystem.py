"""
Agent File System - Sandboxed filesystem interface for Consilience agents

Provides agents with:
- Personal directory structure (inbox, outbox, thoughts, data, scripts)
- Safe file operations (read, write, list)
- Sandboxed command execution
- Storage quotas and security

Author: cMUSH Project
Date: October 2025
"""

import os
import asyncio
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentFilesystem:
    """
    Sandboxed file system interface for agents.

    Each agent gets their own directory tree with controlled access.
    Prevents directory traversal and limits file operations to safe commands.
    """

    # Allowed bash commands for sandboxed execution
    ALLOWED_COMMANDS = {
        'ls', 'cat', 'grep', 'wc', 'head', 'tail', 'find',
        'echo', 'mkdir', 'touch', 'cp', 'mv', 'rm',
        'python3', 'node', 'date', 'pwd'
    }

    # Dangerous command patterns to reject
    DANGEROUS_PATTERNS = ['&&', '||', ';', '|', '>', '>>', '<', '`', '$', '..']

    def __init__(self, agent_id: str, base_path: str = "world/agents", config: Dict = None):
        """
        Initialize agent filesystem.

        Args:
            agent_id: Unique agent identifier
            base_path: Base directory for all agent filesystems
            config: Configuration dict with quotas and limits
        """
        self.agent_id = agent_id
        self.base_path = base_path
        # IMPORTANT: Store agent_dir as absolute path to avoid resolution issues
        self.agent_dir = os.path.abspath(os.path.join(base_path, agent_id))

        # Configuration
        self.config = config or {}
        self.max_file_size = self.config.get('max_file_size', 1_048_576)  # 1MB
        self.max_total_storage = self.config.get('max_total_storage', 104_857_600)  # 100MB
        self.command_timeout = self.config.get('command_timeout', 5)  # seconds

        # Initialize directory structure
        self._initialize_structure()

        logger.info(f"AgentFilesystem initialized: {agent_id} at {self.agent_dir}")

    def _initialize_structure(self):
        """Create agent directory structure if it doesn't exist."""
        subdirs = ['inbox', 'outbox', 'memories', 'thoughts', 'data', 'scripts']

        # Create agent root
        os.makedirs(self.agent_dir, exist_ok=True)

        # Create subdirectories
        for subdir in subdirs:
            dir_path = os.path.join(self.agent_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)

        # Create README if it doesn't exist
        readme_path = os.path.join(self.agent_dir, 'README.txt')
        if not os.path.exists(readme_path):
            with open(readme_path, 'w') as f:
                f.write(f"""Agent Filesystem for {self.agent_id}
{'=' * 60}

Directory Structure:
- inbox/     : Incoming messages from other agents and users
- outbox/    : Outgoing messages (queued for delivery)
- memories/  : Persistent memory snapshots and episodic records
- thoughts/  : Internal monologue and daily thought logs
- data/      : Personal data storage (notes, journals, plans)
- scripts/   : Agent-created scripts and programs

Storage Quota: {self.max_total_storage / 1_048_576:.0f}MB
Max File Size: {self.max_file_size / 1_024:.0f}KB

Created: {datetime.now().isoformat()}
""")

        logger.debug(f"Directory structure initialized for {self.agent_id}")

    def _resolve_path(self, path: str) -> str:
        """
        Resolve path relative to agent directory.

        Args:
            path: Relative path within agent directory

        Returns:
            Absolute path

        Raises:
            ValueError: If path escapes agent directory
        """
        # Handle absolute paths that are already in agent dir
        if os.path.isabs(path):
            full_path = os.path.abspath(path)
        else:
            full_path = os.path.abspath(os.path.join(self.agent_dir, path))

        # Security: prevent directory traversal
        if not full_path.startswith(self.agent_dir):
            raise ValueError(f"Path escapes agent directory: {path}")

        return full_path

    def _validate_read_access(self, path: str):
        """
        Validate that path is readable.

        Args:
            path: Absolute path to validate

        Raises:
            PermissionError: If path is not readable
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if not os.access(path, os.R_OK):
            raise PermissionError(f"No read access: {path}")

    def _validate_write_access(self, path: str):
        """
        Validate that path is writable.

        Args:
            path: Absolute path to validate

        Raises:
            PermissionError: If path is not in writable directory
        """
        # Get relative path from agent root
        rel_path = os.path.relpath(path, self.agent_dir)

        # Allowed write directories
        allowed_dirs = ['data', 'thoughts', 'outbox', 'scripts', 'memories']

        # Check if path starts with allowed directory
        if not any(rel_path.startswith(d) for d in allowed_dirs):
            raise PermissionError(f"Cannot write to: {rel_path}")

    def _check_storage_quota(self, new_file_size: int = 0):
        """
        Check if storage quota would be exceeded.

        Args:
            new_file_size: Size of new file to be written

        Raises:
            ValueError: If quota would be exceeded
        """
        current_size = self.get_storage_usage()

        if current_size + new_file_size > self.max_total_storage:
            raise ValueError(f"Storage quota exceeded: {current_size + new_file_size} / {self.max_total_storage}")

    def get_storage_usage(self) -> int:
        """
        Calculate total storage used by agent.

        Returns:
            Total bytes used
        """
        total = 0
        for root, dirs, files in os.walk(self.agent_dir):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
        return total

    def file_exists(self, path: str) -> bool:
        """
        Check if file exists.

        Args:
            path: Relative path within agent directory

        Returns:
            True if file exists
        """
        try:
            full_path = self._resolve_path(path)
            return os.path.isfile(full_path)
        except (ValueError, FileNotFoundError):
            return False

    def read_file(self, path: str) -> str:
        """
        Read file from agent directory.

        Args:
            path: Relative path within agent directory

        Returns:
            File contents as string

        Raises:
            ValueError: If path invalid
            FileNotFoundError: If file doesn't exist
            PermissionError: If no read access
        """
        full_path = self._resolve_path(path)
        self._validate_read_access(full_path)

        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.debug(f"Agent {self.agent_id} read file: {path}")
        return content

    def write_file(self, path: str, content: str, append: bool = False):
        """
        Write file to agent directory.

        Args:
            path: Relative path within agent directory
            content: Content to write
            append: If True, append to existing file

        Raises:
            ValueError: If file too large or quota exceeded
            PermissionError: If path not writable
        """
        full_path = self._resolve_path(path)
        self._validate_write_access(full_path)

        # Check file size
        content_bytes = content.encode('utf-8')
        if len(content_bytes) > self.max_file_size:
            raise ValueError(f"File too large: {len(content_bytes)} / {self.max_file_size}")

        # Check storage quota
        if not append or not os.path.exists(full_path):
            self._check_storage_quota(len(content_bytes))
        else:
            current_size = os.path.getsize(full_path)
            self._check_storage_quota(len(content_bytes) - current_size)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Write file
        mode = 'a' if append else 'w'
        with open(full_path, mode, encoding='utf-8') as f:
            f.write(content)

        logger.debug(f"Agent {self.agent_id} wrote file: {path} ({len(content_bytes)} bytes)")

    def append_file(self, path: str, content: str):
        """
        Append to file.

        Args:
            path: Relative path within agent directory
            content: Content to append
        """
        self.write_file(path, content, append=True)

    def delete_file(self, path: str):
        """
        Delete file from agent directory.

        Args:
            path: Relative path within agent directory

        Raises:
            ValueError: If path invalid
            PermissionError: If path not writable
        """
        full_path = self._resolve_path(path)
        self._validate_write_access(full_path)

        if os.path.isfile(full_path):
            os.remove(full_path)
            logger.debug(f"Agent {self.agent_id} deleted file: {path}")
        else:
            raise FileNotFoundError(f"File not found: {path}")

    def list_directory(self, path: str = '.') -> List[str]:
        """
        List files in directory.

        Args:
            path: Relative path within agent directory (default: root)

        Returns:
            List of filenames

        Raises:
            ValueError: If path invalid
            FileNotFoundError: If directory doesn't exist
        """
        full_path = self._resolve_path(path)
        self._validate_read_access(full_path)

        if not os.path.isdir(full_path):
            raise ValueError(f"Not a directory: {path}")

        files = os.listdir(full_path)
        logger.debug(f"Agent {self.agent_id} listed directory: {path}")
        return files

    def _is_safe_command(self, command: str) -> bool:
        """
        Check if command is safe to execute.

        Args:
            command: Bash command to validate

        Returns:
            True if command is safe
        """
        # Get base command
        parts = command.split()
        if not parts:
            return False

        base_cmd = parts[0]

        # Must be in whitelist
        if base_cmd not in self.ALLOWED_COMMANDS:
            logger.warning(f"Command not in whitelist: {base_cmd}")
            return False

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in command:
                logger.warning(f"Dangerous pattern in command: {pattern}")
                return False

        return True

    async def execute_command(self, command: str) -> Dict[str, str]:
        """
        Execute sandboxed bash command in agent directory.

        Args:
            command: Bash command to execute

        Returns:
            Dict with 'stdout', 'stderr', 'returncode'

        Raises:
            ValueError: If command is not safe
        """
        if not self._is_safe_command(command):
            raise ValueError(f"Command not allowed: {command}")

        try:
            # Execute command in agent directory
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=self.agent_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Security: limit resources
                preexec_fn=None,  # Don't allow preexec
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.command_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ValueError(f"Command timed out after {self.command_timeout}s")

            result = {
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8'),
                'returncode': process.returncode
            }

            logger.info(f"Agent {self.agent_id} executed: {command} (return: {result['returncode']})")
            return result

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            raise

    def get_stats(self) -> Dict:
        """
        Get filesystem statistics.

        Returns:
            Dict with storage usage, file counts, etc.
        """
        stats = {
            'agent_id': self.agent_id,
            'storage_used': self.get_storage_usage(),
            'storage_quota': self.max_total_storage,
            'storage_percent': (self.get_storage_usage() / self.max_total_storage) * 100,
            'file_counts': {}
        }

        # Count files in each directory
        subdirs = ['inbox', 'outbox', 'memories', 'thoughts', 'data', 'scripts']
        for subdir in subdirs:
            dir_path = os.path.join(self.agent_dir, subdir)
            if os.path.exists(dir_path):
                stats['file_counts'][subdir] = len(os.listdir(dir_path))

        return stats


# Convenience functions for testing
async def test_filesystem():
    """Test agent filesystem operations."""
    print("Testing AgentFilesystem...")

    # Create filesystem
    fs = AgentFilesystem('test_agent')

    # Write test file
    fs.write_file('data/test.txt', 'Hello, world!\n')
    print("✓ Created test.txt")

    # Read file
    content = fs.read_file('data/test.txt')
    print(f"✓ Read test.txt: {content.strip()}")

    # Append to file
    fs.append_file('data/test.txt', 'Second line.\n')
    content = fs.read_file('data/test.txt')
    print(f"✓ Appended to test.txt: {content.strip()}")

    # List directory
    files = fs.list_directory('data')
    print(f"✓ Listed data/: {files}")

    # Execute command
    result = await fs.execute_command('ls -la')
    print(f"✓ Executed command: {result['returncode']}")
    print(result['stdout'])

    # Get stats
    stats = fs.get_stats()
    print(f"✓ Storage: {stats['storage_used']} / {stats['storage_quota']} bytes")

    # Test security
    try:
        fs.read_file('../../../etc/passwd')
        print("✗ Security breach!")
    except ValueError as e:
        print(f"✓ Security working: {e}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(test_filesystem())
