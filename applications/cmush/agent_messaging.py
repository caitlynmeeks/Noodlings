"""
Agent Messaging System - Inter-agent and user-agent communication

Provides agents with:
- Inbox/outbox for asynchronous messages
- Private agent-to-agent messaging
- Message persistence and delivery tracking
- Automatic cleanup of old messages

Author: cMUSH Project
Date: October 2025
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AgentMessage:
    """Represents a message between agents or users."""

    def __init__(
        self,
        from_id: str,
        to_id: str,
        content: str,
        timestamp: Optional[float] = None,
        message_type: str = 'text'
    ):
        """
        Initialize message.

        Args:
            from_id: Sender ID (user_* or agent_*)
            to_id: Recipient ID (user_* or agent_*)
            content: Message content
            timestamp: Unix timestamp (default: now)
            message_type: Message type (text, system, etc.)
        """
        self.from_id = from_id
        self.to_id = to_id
        self.content = content
        self.timestamp = timestamp or time.time()
        self.message_type = message_type
        self.message_id = f"{from_id}_{to_id}_{self.timestamp}"

    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        return {
            'from': self.from_id,
            'to': self.to_id,
            'content': self.content,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'type': self.message_type,
            'id': self.message_id
        }

    def to_file_content(self) -> str:
        """Format message for file storage."""
        dt = datetime.fromtimestamp(self.timestamp)
        return f"""From: {self.from_id}
To: {self.to_id}
Date: {dt.isoformat()}
Type: {self.message_type}

{self.content}
"""

    @staticmethod
    def from_file(filepath: str) -> 'AgentMessage':
        """Parse message from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse header
        lines = content.split('\n')
        from_id = None
        to_id = None
        timestamp = None
        message_type = 'text'

        body_start = 0
        for i, line in enumerate(lines):
            if line.startswith('From: '):
                from_id = line[6:].strip()
            elif line.startswith('To: '):
                to_id = line[4:].strip()
            elif line.startswith('Date: '):
                date_str = line[6:].strip()
                timestamp = datetime.fromisoformat(date_str).timestamp()
            elif line.startswith('Type: '):
                message_type = line[6:].strip()
            elif line == '':
                body_start = i + 1
                break

        # Get body
        body = '\n'.join(lines[body_start:])

        return AgentMessage(
            from_id=from_id,
            to_id=to_id,
            content=body,
            timestamp=timestamp,
            message_type=message_type
        )


class AgentMessaging:
    """
    Manages inter-agent and user-agent messaging.

    Handles inbox/outbox, message delivery, and cleanup.
    """

    def __init__(self, base_path: str = "world/agents", config: Dict = None):
        """
        Initialize messaging system.

        Args:
            base_path: Base directory for agent filesystems
            config: Configuration dict
        """
        self.base_path = base_path
        self.config = config or {}

        # Configuration
        self.max_message_size = self.config.get('max_message_size', 10_000)  # 10KB
        self.inbox_retention_days = self.config.get('inbox_retention_days', 30)

        logger.info("AgentMessaging initialized")

    def _get_inbox_path(self, agent_id: str) -> str:
        """Get path to agent's inbox directory."""
        return os.path.join(self.base_path, agent_id, 'inbox')

    def _get_outbox_path(self, agent_id: str) -> str:
        """Get path to agent's outbox directory."""
        return os.path.join(self.base_path, agent_id, 'outbox')

    def _ensure_directories(self, agent_id: str):
        """Ensure inbox/outbox directories exist."""
        os.makedirs(self._get_inbox_path(agent_id), exist_ok=True)
        os.makedirs(self._get_outbox_path(agent_id), exist_ok=True)

    async def send_message(
        self,
        from_id: str,
        to_id: str,
        content: str,
        message_type: str = 'text'
    ) -> bool:
        """
        Send message from one agent/user to another.

        Args:
            from_id: Sender ID
            to_id: Recipient ID
            content: Message content
            message_type: Message type

        Returns:
            True if message sent successfully

        Raises:
            ValueError: If message too large or invalid IDs
        """
        # Validate message size
        if len(content) > self.max_message_size:
            raise ValueError(f"Message too large: {len(content)} / {self.max_message_size}")

        # Create message
        message = AgentMessage(
            from_id=from_id,
            to_id=to_id,
            content=content,
            message_type=message_type
        )

        # Ensure directories exist
        self._ensure_directories(from_id)
        self._ensure_directories(to_id)

        # Write to sender's outbox
        timestamp = int(message.timestamp)
        outbox_filename = f"to_{to_id}_{timestamp}.msg"
        outbox_path = os.path.join(self._get_outbox_path(from_id), outbox_filename)

        with open(outbox_path, 'w', encoding='utf-8') as f:
            f.write(message.to_file_content())

        # Write to recipient's inbox
        inbox_filename = f"from_{from_id}_{timestamp}.msg"
        inbox_path = os.path.join(self._get_inbox_path(to_id), inbox_filename)

        with open(inbox_path, 'w', encoding='utf-8') as f:
            f.write(message.to_file_content())

        # Mark as unread
        self._mark_unread(inbox_path)

        logger.info(f"Message sent: {from_id} -> {to_id} ({len(content)} chars)")
        return True

    def _mark_unread(self, message_path: str):
        """Mark message as unread by creating .unread marker."""
        marker_path = message_path + '.unread'
        Path(marker_path).touch()

    def _mark_read(self, message_path: str):
        """Mark message as read by removing .unread marker."""
        marker_path = message_path + '.unread'
        if os.path.exists(marker_path):
            os.remove(marker_path)

    def _is_unread(self, message_path: str) -> bool:
        """Check if message is unread."""
        marker_path = message_path + '.unread'
        return os.path.exists(marker_path)

    async def check_inbox(
        self,
        agent_id: str,
        mark_as_read: bool = True,
        unread_only: bool = True
    ) -> List[Dict]:
        """
        Check agent's inbox for messages.

        Args:
            agent_id: Agent to check inbox for
            mark_as_read: If True, mark messages as read
            unread_only: If True, only return unread messages

        Returns:
            List of message dicts
        """
        inbox_path = self._get_inbox_path(agent_id)
        messages = []

        if not os.path.exists(inbox_path):
            return messages

        # Get all .msg files
        for filename in sorted(os.listdir(inbox_path)):
            if not filename.endswith('.msg'):
                continue

            filepath = os.path.join(inbox_path, filename)

            # Check if unread
            is_unread = self._is_unread(filepath)

            # Skip if only want unread and this is read
            if unread_only and not is_unread:
                continue

            try:
                # Parse message
                message = AgentMessage.from_file(filepath)
                message_dict = message.to_dict()
                message_dict['filename'] = filename
                message_dict['filepath'] = filepath
                message_dict['unread'] = is_unread

                messages.append(message_dict)

                # Mark as read if requested
                if mark_as_read and is_unread:
                    self._mark_read(filepath)

            except Exception as e:
                logger.error(f"Error parsing message {filename}: {e}")

        logger.debug(f"Agent {agent_id} checked inbox: {len(messages)} messages")
        return messages

    async def get_outbox(self, agent_id: str) -> List[Dict]:
        """
        Get agent's outbox (sent messages).

        Args:
            agent_id: Agent ID

        Returns:
            List of message dicts
        """
        outbox_path = self._get_outbox_path(agent_id)
        messages = []

        if not os.path.exists(outbox_path):
            return messages

        # Get all .msg files
        for filename in sorted(os.listdir(outbox_path)):
            if not filename.endswith('.msg'):
                continue

            filepath = os.path.join(outbox_path, filename)

            try:
                # Parse message
                message = AgentMessage.from_file(filepath)
                message_dict = message.to_dict()
                message_dict['filename'] = filename
                message_dict['filepath'] = filepath

                messages.append(message_dict)

            except Exception as e:
                logger.error(f"Error parsing message {filename}: {e}")

        return messages

    async def delete_message(self, filepath: str):
        """
        Delete a message file.

        Args:
            filepath: Path to message file
        """
        if os.path.exists(filepath):
            os.remove(filepath)

            # Remove unread marker if exists
            marker = filepath + '.unread'
            if os.path.exists(marker):
                os.remove(marker)

            logger.debug(f"Deleted message: {filepath}")

    async def cleanup_old_messages(self, agent_id: str):
        """
        Delete messages older than retention period.

        Args:
            agent_id: Agent to clean up messages for
        """
        cutoff_time = time.time() - (self.inbox_retention_days * 86400)
        deleted = 0

        # Clean inbox
        inbox_path = self._get_inbox_path(agent_id)
        if os.path.exists(inbox_path):
            for filename in os.listdir(inbox_path):
                if not filename.endswith('.msg'):
                    continue

                filepath = os.path.join(inbox_path, filename)
                file_time = os.path.getmtime(filepath)

                if file_time < cutoff_time:
                    await self.delete_message(filepath)
                    deleted += 1

        # Clean outbox
        outbox_path = self._get_outbox_path(agent_id)
        if os.path.exists(outbox_path):
            for filename in os.listdir(outbox_path):
                if not filename.endswith('.msg'):
                    continue

                filepath = os.path.join(outbox_path, filename)
                file_time = os.path.getmtime(filepath)

                if file_time < cutoff_time:
                    await self.delete_message(filepath)
                    deleted += 1

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old messages for {agent_id}")

    def get_message_stats(self, agent_id: str) -> Dict:
        """
        Get messaging statistics for agent.

        Args:
            agent_id: Agent ID

        Returns:
            Dict with inbox/outbox counts
        """
        inbox_path = self._get_inbox_path(agent_id)
        outbox_path = self._get_outbox_path(agent_id)

        inbox_count = 0
        inbox_unread = 0
        if os.path.exists(inbox_path):
            for f in os.listdir(inbox_path):
                if f.endswith('.msg'):
                    inbox_count += 1
                    filepath = os.path.join(inbox_path, f)
                    if self._is_unread(filepath):
                        inbox_unread += 1

        outbox_count = 0
        if os.path.exists(outbox_path):
            outbox_count = len([f for f in os.listdir(outbox_path) if f.endswith('.msg')])

        return {
            'agent_id': agent_id,
            'inbox_total': inbox_count,
            'inbox_unread': inbox_unread,
            'outbox_total': outbox_count
        }


# Convenience functions for testing
async def test_messaging():
    """Test agent messaging system."""
    print("Testing AgentMessaging...")

    messaging = AgentMessaging()

    # Send test message
    await messaging.send_message(
        from_id='agent_alice',
        to_id='agent_bob',
        content='Hello Bob! Want to explore the world together?'
    )
    print("✓ Sent message: alice -> bob")

    # Check inbox
    messages = await messaging.check_inbox('agent_bob', mark_as_read=False)
    print(f"✓ Bob's inbox: {len(messages)} messages")
    if messages:
        print(f"  From: {messages[0]['from']}")
        print(f"  Content: {messages[0]['content'][:50]}...")

    # Send reply
    await messaging.send_message(
        from_id='agent_bob',
        to_id='agent_alice',
        content='Absolutely! I would love to explore together.'
    )
    print("✓ Sent reply: bob -> alice")

    # Check alice's inbox
    messages = await messaging.check_inbox('agent_alice')
    print(f"✓ Alice's inbox: {len(messages)} unread messages")

    # Get stats
    stats = messaging.get_message_stats('agent_bob')
    print(f"✓ Bob's stats: {stats}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_messaging())
