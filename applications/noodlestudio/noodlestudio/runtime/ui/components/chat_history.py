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
#   ChatHistory Component
#
#   A scrolling message list displaying conversation history....
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.components.chat_history
# PURPOSE:  ChatHistory Component
# LAYER:    Studio / UI Components
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MessageRole, ChatMessage, ChatHistory
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import datetime

from ..component import UIComponent, register_component


class MessageRole(Enum):
    """Role of the message sender."""
    USER = "user"
    NOODLING = "noodling"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """A single message in the chat history."""
    role: MessageRole
    content: str
    sender_name: str = ""
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "role": self.role.value,
            "content": self.content,
            "sender_name": self.sender_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessage':
        """Deserialize from dict."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            sender_name=data.get("sender_name", ""),
            timestamp=timestamp
        )


@register_component
class ChatHistory(UIComponent):
    """
    Scrolling chat message history.

    Displays messages in a scrollable container with different
    styling for user messages (right-aligned) and noodling
    messages (left-aligned).

    Properties:
        messages: List of ChatMessage objects
        user_bubble_color: Background for user messages
        noodling_bubble_color: Background for noodling messages
        system_color: Color for system messages
        show_timestamps: Whether to display timestamps
        show_sender_names: Whether to display sender names
        font_size: Base font size for messages
        bubble_radius: Border radius for message bubbles
        bubble_padding: Padding inside bubbles
        message_spacing: Vertical space between messages
    """

    component_type: str = "ChatHistory"

    def __init__(self, name: str = ""):
        super().__init__(name)

        # Message storage
        self._messages: List[ChatMessage] = []

        # Styling - user messages
        self.user_bubble_color: str = "#3b82f6"  # Blue
        self.user_text_color: str = "#ffffff"

        # Styling - noodling messages
        self.noodling_bubble_color: str = "#374151"  # Dark gray
        self.noodling_text_color: str = "#ffffff"

        # Styling - system messages
        self.system_color: str = "#6b7280"  # Medium gray

        # Display options
        self.show_timestamps: bool = False
        self.show_sender_names: bool = True
        self.font_size: int = 13
        self.bubble_radius: int = 12
        self.bubble_padding: int = 10
        self.message_spacing: int = 8

        # Background
        self.background: str = "#1a1a1a"

        # Scrolling behavior
        self.auto_scroll: bool = True  # Auto-scroll to bottom on new messages

    @property
    def messages(self) -> List[ChatMessage]:
        """Get all messages."""
        return self._messages

    def add_message(self, role: MessageRole, content: str,
                    sender_name: str = "", timestamp: Optional[datetime] = None) -> ChatMessage:
        """
        Add a new message to the history.

        Args:
            role: USER, NOODLING, or SYSTEM
            content: Message text
            sender_name: Display name of sender
            timestamp: Message time (defaults to now)

        Returns:
            The created ChatMessage
        """
        msg = ChatMessage(
            role=role,
            content=content,
            sender_name=sender_name,
            timestamp=timestamp or datetime.now()
        )
        self._messages.append(msg)

        # Notify widget to update if rendered
        if hasattr(self, '_widget') and self._widget:
            self._widget.add_message(msg)

        return msg

    def clear_messages(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        if hasattr(self, '_widget') and self._widget:
            self._widget.clear()

    def to_dict(self) -> dict:
        """Serialize to dict."""
        data = super().to_dict()
        data.update({
            "messages": [m.to_dict() for m in self._messages],
            "user_bubble_color": self.user_bubble_color,
            "user_text_color": self.user_text_color,
            "noodling_bubble_color": self.noodling_bubble_color,
            "noodling_text_color": self.noodling_text_color,
            "system_color": self.system_color,
            "show_timestamps": self.show_timestamps,
            "show_sender_names": self.show_sender_names,
            "font_size": self.font_size,
            "bubble_radius": self.bubble_radius,
            "bubble_padding": self.bubble_padding,
            "message_spacing": self.message_spacing,
            "background": self.background,
            "auto_scroll": self.auto_scroll
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatHistory':
        """Deserialize from dict."""
        component = cls(name=data.get("name", ""))
        component._apply_base_properties(data)

        # Load messages
        for msg_data in data.get("messages", []):
            component._messages.append(ChatMessage.from_dict(msg_data))

        # Load styling
        component.user_bubble_color = data.get("user_bubble_color", component.user_bubble_color)
        component.user_text_color = data.get("user_text_color", component.user_text_color)
        component.noodling_bubble_color = data.get("noodling_bubble_color", component.noodling_bubble_color)
        component.noodling_text_color = data.get("noodling_text_color", component.noodling_text_color)
        component.system_color = data.get("system_color", component.system_color)
        component.show_timestamps = data.get("show_timestamps", component.show_timestamps)
        component.show_sender_names = data.get("show_sender_names", component.show_sender_names)
        component.font_size = data.get("font_size", component.font_size)
        component.bubble_radius = data.get("bubble_radius", component.bubble_radius)
        component.bubble_padding = data.get("bubble_padding", component.bubble_padding)
        component.message_spacing = data.get("message_spacing", component.message_spacing)
        component.background = data.get("background", component.background)
        component.auto_scroll = data.get("auto_scroll", component.auto_scroll)

        return component

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
