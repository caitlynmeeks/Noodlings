"""
Authentication for cMUSH

Supports two authentication modes:
1. Local: username/password with SHA256 hashing (for personal/Tailscale use)
2. Cloud: NoodleStudio session token validation (for unified auth)

Author: cMUSH Project
Date: October 2025
"""

import hashlib
import os
import json
import secrets
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict, Any
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
        user = self.world.get_user_by_username(username)

        if not user:
            return False, None, "Invalid username or password."

        user_id = user['uid']

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

    # -------------------------------------------------------------------------
    # Cloud Authentication (NoodleStudio unified auth)
    # -------------------------------------------------------------------------

    def authenticate_with_cloud_token(
        self,
        token: str,
        avatar_id: Optional[str] = None,
        api_base: str = "https://noodlings-api.caitsters.workers.dev"
    ) -> Tuple[bool, Optional[str], str, Optional[Dict[str, Any]]]:
        """
        Authenticate using a NoodleStudio cloud session token.

        Validates the token against the backend API, then finds or creates
        a local MUSH user linked to the cloud account.

        Args:
            token: NoodleStudio session token (from AccountManager)
            avatar_id: Optional avatar ID to use (from user's avatar list)
            api_base: Backend API base URL

        Returns:
            Tuple of (success, user_id, message, user_profile)
            user_profile contains: id, email, display_name, avatar info
        """
        # Validate token with backend
        try:
            req = urllib.request.Request(
                f'{api_base}/auth/me',
                headers={'Authorization': f'Bearer {token}'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                user_data = data.get('user', {})

                cloud_user_id = user_data.get('id', '')
                email = user_data.get('email', '')
                display_name = user_data.get('display_name', '')

                if not cloud_user_id:
                    return False, None, "Invalid token: no user ID", None

                logger.info(f"Cloud token validated for {email} ({cloud_user_id})")

        except urllib.error.HTTPError as e:
            if e.code == 401:
                return False, None, "Session expired. Please log in again.", None
            logger.error(f"Cloud auth HTTP error: {e}")
            return False, None, f"Authentication failed: {e}", None
        except Exception as e:
            logger.error(f"Cloud auth error: {e}")
            return False, None, f"Authentication error: {e}", None

        # Find or create local user linked to cloud account
        local_user_id = self._get_or_create_cloud_linked_user(
            cloud_user_id=cloud_user_id,
            email=email,
            display_name=display_name
        )

        if not local_user_id:
            return False, None, "Failed to create local user", None

        # Build user profile for response
        user_profile = {
            'cloud_id': cloud_user_id,
            'email': email,
            'display_name': display_name,
            'avatar_id': avatar_id
        }

        logger.info(f"Cloud user authenticated: {display_name or email} -> {local_user_id}")
        return True, local_user_id, "Cloud authentication successful.", user_profile

    def _get_or_create_cloud_linked_user(
        self,
        cloud_user_id: str,
        email: str,
        display_name: str
    ) -> Optional[str]:
        """
        Find existing user linked to cloud account, or create one.

        Args:
            cloud_user_id: Cloud account ID
            email: User's email
            display_name: User's display name

        Returns:
            Local user ID, or None on failure
        """
        # Look for existing user with this cloud_user_id
        for user_id, user_data in self.world.users.items():
            if user_data.get('cloud_user_id') == cloud_user_id:
                logger.debug(f"Found existing cloud-linked user: {user_id}")
                return user_id

        # No existing user - create one
        # Use display_name or email prefix as username
        base_username = display_name or email.split('@')[0]
        # Sanitize for MUSH username (alphanumeric only)
        username = ''.join(c for c in base_username if c.isalnum())[:20]

        # Ensure unique
        original_username = username
        counter = 1
        while self.world.user_exists(username):
            username = f"{original_username[:17]}{counter}"
            counter += 1

        try:
            # Find a valid spawn room - use first available room or create default
            spawn_room = "room_000"
            if self.world.rooms:
                # Use first available room
                spawn_room = next(iter(self.world.rooms.keys()))
            elif not self.world.get_room("room_000"):
                # No rooms exist - create a default one
                spawn_room = self.world.create_room(
                    name="The Nexus",
                    description="A cozy campfire with crackling logs. Welcome to the world!"
                )

            # Create user without password (cloud-only auth)
            user_id = self.world.create_user(
                username=username,
                password_hash="CLOUD_AUTH",  # Marker for cloud-only users
                spawn_room=spawn_room
            )

            # Store cloud link
            user = self.world.get_user(user_id)
            if user:
                user['cloud_user_id'] = cloud_user_id
                user['email'] = email
                user['cloud_display_name'] = display_name
                self.world.save_all()

            logger.info(f"Created cloud-linked user: {username} ({user_id}) -> {cloud_user_id}")
            return user_id

        except Exception as e:
            logger.error(f"Failed to create cloud-linked user: {e}")
            return None

    def is_cloud_user(self, user_id: str) -> bool:
        """Check if a user is linked to a cloud account."""
        user = self.world.get_user(user_id)
        if user:
            return user.get('password_hash') == "CLOUD_AUTH"
        return False
