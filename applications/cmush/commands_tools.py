# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Agent Tools Commands
#
#   These commands give Noodlings their own private workspace -
#   a journal for thoughts, a messaging inbox, and a sandboxed
#   filesystem for notes and data.
#
#   Journal:
#     @think <thought>     -> Record a private thought
#     @remember            -> Read past journal entries
#
#   Messaging:
#     @message bob "Hi"    -> Send private message
#     @inbox               -> Check messages
#
#   Filesystem:
#     @write file.txt "..."  -> Write to agent's directory
#     @read file.txt         -> Read a file
#     @ls                    -> List files
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.commands_tools
# PURPOSE:  Agent filesystem and messaging commands
# LAYER:    Backend / Commands
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ToolsCommandsMixin    Think, remember, message, filesystem
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Agent Tools Commands Mixin for cMUSH

Contains commands for agent tools and filesystem operations:
- @think: Record private thoughts
- @remember: Read previous thoughts
- @message: Send private messages
- @inbox: Check inbox
- @write: Write to filesystem
- @read: Read from filesystem
- @ls: List directory
- @exec: Execute sandboxed command

Author: Caitlyn + Claude
Date: December 2025
"""

from typing import Dict
from datetime import datetime


class ToolsCommandsMixin:
    """Mixin providing agent tool commands for CommandParser."""

    async def cmd_think(self, user_id: str, args: str) -> Dict:
        """Write a private thought to agent's journal."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        if not args:
            return {'success': False, 'output': 'Usage: @think <thought>', 'events': []}

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        thought = args.strip()
        today = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Write to thought log
        agent.filesystem.append_file(
            f"thoughts/{today}.txt",
            f"[{timestamp}] {thought}\n"
        )

        return {
            'success': True,
            'output': 'Thought recorded in your journal.',
            'events': []
        }

    async def cmd_remember(self, user_id: str, args: str) -> Dict:
        """Read previous thoughts from agent's journal."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        # Default to today
        date = args.strip() if args else datetime.now().strftime("%Y-%m-%d")
        thoughts_file = f"thoughts/{date}.txt"

        try:
            thoughts = agent.filesystem.read_file(thoughts_file)
            return {'success': True, 'output': f"\nThoughts for {date}:\n{thoughts}", 'events': []}
        except FileNotFoundError:
            return {'success': True, 'output': f"No thoughts recorded for {date}.", 'events': []}

    async def cmd_message(self, user_id: str, args: str) -> Dict:
        """Send private message to another agent or user."""
        if not args:
            return {'success': False, 'output': 'Usage: @message <agent_name> <text>', 'events': []}

        parts = args.split(None, 1)
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: @message <agent_name> <text>', 'events': []}

        target_name, message = parts
        target_id = f"agent_{target_name}" if not target_name.startswith('agent_') else target_name

        # Get sender's agent (if agent)
        if user_id.startswith('agent_'):
            sender = self.agent_manager.get_agent(user_id)
            if not sender:
                return {'success': False, 'output': 'Agent not found.', 'events': []}

            # Send via messaging system
            await sender.messaging.send_message(
                from_id=user_id,
                to_id=target_id,
                content=message
            )

            return {
                'success': True,
                'output': f'Message sent to {target_name}.',
                'events': []
            }
        else:
            # Human user sending message
            # For now, just log it
            return {'success': False, 'output': 'Human messaging not yet implemented.', 'events': []}

    async def cmd_inbox(self, user_id: str, args: str) -> Dict:
        """Check inbox for messages."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        # Check inbox
        messages = await agent.messaging.check_inbox(agent.agent_id, mark_as_read=False, unread_only=False)

        if not messages:
            return {'success': True, 'output': 'Your inbox is empty.', 'events': []}

        lines = [f"\nInbox ({len(messages)} messages):"]
        lines.append("=" * 60)

        for msg in messages[-10:]:  # Show last 10
            from_name = msg['from']
            content = msg['content'][:100] + ('...' if len(msg['content']) > 100 else '')
            unread = '[UNREAD] ' if msg.get('unread') else ''
            lines.append(f"{unread}From {from_name}:")
            lines.append(f"  {content}")
            lines.append("")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_write_file(self, user_id: str, args: str) -> Dict:
        """Write to a file in agent's filesystem."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        if not args:
            return {'success': False, 'output': 'Usage: @write <filepath> <content>', 'events': []}

        parts = args.split(None, 1)
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: @write <filepath> <content>', 'events': []}

        filepath, content = parts

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        try:
            agent.filesystem.write_file(filepath, content + '\n')
            return {'success': True, 'output': f'Wrote to {filepath}.', 'events': []}
        except Exception as e:
            return {'success': False, 'output': f'Error: {str(e)}', 'events': []}

    async def cmd_read_file(self, user_id: str, args: str) -> Dict:
        """Read from a file in agent's filesystem."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        if not args:
            return {'success': False, 'output': 'Usage: @read <filepath>', 'events': []}

        filepath = args.strip()

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        try:
            content = agent.filesystem.read_file(filepath)
            return {'success': True, 'output': f'\n{filepath}:\n{content}', 'events': []}
        except Exception as e:
            return {'success': False, 'output': f'Error: {str(e)}', 'events': []}

    async def cmd_list_files(self, user_id: str, args: str) -> Dict:
        """List files in agent's filesystem."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        path = args.strip() if args else '.'

        try:
            files = agent.filesystem.list_directory(path)
            if not files:
                return {'success': True, 'output': f'Directory {path} is empty.', 'events': []}

            lines = [f"\nFiles in {path}:"]
            for f in sorted(files):
                lines.append(f"  {f}")

            return {'success': True, 'output': '\n'.join(lines), 'events': []}
        except Exception as e:
            return {'success': False, 'output': f'Error: {str(e)}', 'events': []}

    async def cmd_execute_command(self, user_id: str, args: str) -> Dict:
        """Execute a sandboxed command in agent's filesystem."""
        if not user_id.startswith('agent_'):
            return {'success': False, 'output': 'Only agents can use this command.', 'events': []}

        if not args:
            return {'success': False, 'output': 'Usage: @exec <command>', 'events': []}

        command = args.strip()

        agent = self.agent_manager.get_agent(user_id)
        if not agent:
            return {'success': False, 'output': 'Agent not found.', 'events': []}

        try:
            result = await agent.filesystem.execute_command(command)
            output = result['stdout'] if result['stdout'] else result['stderr']
            return {
                'success': result['returncode'] == 0,
                'output': f'\n{output}',
                'events': []
            }
        except Exception as e:
            return {'success': False, 'output': f'Error: {str(e)}', 'events': []}

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
