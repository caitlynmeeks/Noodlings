"""
Authentication for cMUSH

Simple username/password authentication with SHA256 hashing.
No email required - lightweight auth for personal/Tailscale use.

Author: cMUSH Project
Date: October 2025
"""

import hashlib
import os
import secrets
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash password with SHA256 and salt.

    Args:
        password: Plain text password
        salt: Optional salt (will generate if not provided)

    Returns:
        Tuple of (password_hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)

    # Hash password with salt
    hash_input = f"{password}{salt}".encode('utf-8')
    password_hash = hashlib.sha256(hash_input).hexdigest()

    return password_hash, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """
    Verify password against stored hash.

    Args:
        password: Plain text password to check
        stored_hash: Stored hash to compare against
        salt: Salt used for hashing

    Returns:
        True if password matches
    """
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == stored_hash


class AuthManager:
    """
    Manages user authentication.

    Works with World state manager for user storage.
    """

    def __init__(self, world):
        """
        Initialize auth manager.

        Args:
            world: World state manager
        """
        self.world = world
        self.sessions = {}  # session_token -> user_id

    def create_user(
        self,
        username: str,
        password: str,
        spawn_room: str = "room_000"
    ) -> Tuple[bool, str]:
        """
        Create a new user account.

        Args:
            username: Desired username
            password: Plain text password
            spawn_room: Initial room

        Returns:
            Tuple of (success, message)
        """
        # Validate username
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters."

        if len(username) > 20:
            return False, "Username must be at most 20 characters."

        if not username.isalnum():
            return False, "Username must be alphanumeric."

        # Check if username exists
        if self.world.user_exists(username):
            return False, "Username already taken."

        # Validate password
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters."

        # Hash password
        password_hash, salt = hash_password(password)

        # Store full hash with salt
        stored_hash = f"{password_hash}:{salt}"

        # Create user
        try:
            user_id = self.world.create_user(
                username=username,
                password_hash=stored_hash,
                spawn_room=spawn_room
            )

            logger.info(f"User created: {username} ({user_id})")
            return True, f"Account created: {username}"

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, f"Error creating account: {str(e)}"

    def authenticate(
        self,
        username: str,
        password: str
    ) -> Tuple[bool, Optional[str], str]:
        """
        Authenticate user credentials.

        Args:
            username: Username
            password: Plain text password

        Returns:
            Tuple of (success, user_id, message)
        """
        user_id = f"user_{username}"
        user = self.world.get_user(user_id)

        if not user:
            return False, None, "Invalid username or password."

        # Parse stored hash
        stored_data = user['password_hash']
        if ':' not in stored_data:
            logger.error(f"Invalid password hash format for {username}")
            return False, None, "Authentication error."

        stored_hash, salt = stored_data.split(':', 1)

        # Verify password
        if not verify_password(password, stored_hash, salt):
            return False, None, "Invalid username or password."

        logger.info(f"User authenticated: {username}")
        return True, user_id, "Authentication successful."

    def create_session(self, user_id: str) -> str:
        """
        Create session token for user.

        Args:
            user_id: User ID

        Returns:
            Session token
        """
        session_token = secrets.token_urlsafe(32)
        self.sessions[session_token] = user_id

        logger.debug(f"Session created for {user_id}")
        return session_token

    def verify_session(self, session_token: str) -> Optional[str]:
        """
        Verify session token.

        Args:
            session_token: Token to verify

        Returns:
            User ID if valid, None otherwise
        """
        return self.sessions.get(session_token)

    def end_session(self, session_token: str):
        """
        End user session.

        Args:
            session_token: Token to invalidate
        """
        if session_token in self.sessions:
            user_id = self.sessions[session_token]
            del self.sessions[session_token]
            logger.debug(f"Session ended for {user_id}")

    def get_active_sessions(self) -> int:
        """Get count of active sessions."""
        return len(self.sessions)
