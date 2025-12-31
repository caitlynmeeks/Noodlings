"""
Account Manager for NoodleStudio
================================

Handles user authentication, session storage, and cloud API state.
Sessions are stored securely in the OS keychain where available,
falling back to encrypted file storage.
"""

import os
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from PyQt6.QtCore import QObject, pyqtSignal, QSettings

logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = os.environ.get('NOODLINGS_API_URL', 'https://noodlings-api.caitsters.workers.dev')


@dataclass
class UserProfile:
    """User profile data from cloud API."""
    id: str
    email: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    credits_balance: int
    providers: list  # OAuth providers linked to account


class AvatarAssetLocation(Enum):
    """Where the avatar's visual asset lives."""
    LOCAL = "local"           # Local .radiance file path
    CLOUD = "cloud"           # Uploaded to backend (URL or asset_id)
    LIBRARY = "library"       # Built-in library avatar


@dataclass
class AvatarMetadata:
    """
    A user's in-world persona/presentation.

    One user can have multiple avatars (like Second Life).
    This is SEPARATE from UserProfile (account-level data).

    The display_name and description are PUBLIC - visible to other users
    in shared worlds. Think of it like an SL profile.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    owner_id: str = ""  # UserProfile.id

    # Presentation (PUBLIC - visible to other users)
    display_name: str = ""              # "Red Fire Anklebiter"
    description: str = ""               # "A mischievous fire imp who loves chaos"
    pronouns: Optional[str] = None      # "they/them", "she/her", etc.
    tags: List[str] = field(default_factory=list)  # ["fire", "imp", "chaotic"]

    # Visual asset reference
    asset_location: AvatarAssetLocation = AvatarAssetLocation.LOCAL
    asset_ref: str = ""                 # Path, URL, or asset_id depending on location
    thumbnail_url: Optional[str] = None # Preview image for avatar picker

    # State
    is_default: bool = False            # Use this avatar when entering worlds
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Future: runtime customization (doesn't require re-uploading asset)
    # tint: Optional[Tuple[float, float, float]] = None
    # scale: float = 1.0

    def to_dict(self) -> dict:
        """Serialize for API/storage."""
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "display_name": self.display_name,
            "description": self.description,
            "pronouns": self.pronouns,
            "tags": self.tags,
            "asset_location": self.asset_location.value,
            "asset_ref": self.asset_ref,
            "thumbnail_url": self.thumbnail_url,
            "is_default": self.is_default,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AvatarMetadata':
        """Deserialize from API/storage."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            owner_id=data.get("owner_id", ""),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            pronouns=data.get("pronouns"),
            tags=data.get("tags", []),
            asset_location=AvatarAssetLocation(data.get("asset_location", "local")),
            asset_ref=data.get("asset_ref", ""),
            thumbnail_url=data.get("thumbnail_url"),
            is_default=data.get("is_default", False),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )


class AccountManager(QObject):
    """
    Manages user authentication state for NoodleStudio.

    Signals:
        logged_in: Emitted when user successfully logs in
        logged_out: Emitted when user logs out
        credits_updated: Emitted when credit balance changes
        login_failed: Emitted when login fails (str: error message)
        avatars_changed: Emitted when avatar list changes
    """

    logged_in = pyqtSignal(object)  # UserProfile
    logged_out = pyqtSignal()
    credits_updated = pyqtSignal(int)  # new balance
    login_failed = pyqtSignal(str)  # error message
    avatars_changed = pyqtSignal()  # avatar list modified

    # Singleton instance
    _instance: Optional['AccountManager'] = None

    @classmethod
    def instance(cls) -> 'AccountManager':
        """Get the singleton AccountManager instance."""
        if cls._instance is None:
            cls._instance = AccountManager()
        return cls._instance

    def __init__(self):
        super().__init__()
        self._session_token: Optional[str] = None
        self._user: Optional[UserProfile] = None
        self._avatars: List[AvatarMetadata] = []
        self._settings = QSettings('Noodlings', 'NoodleStudio')

        # Try to restore session on init
        self._restore_session()
        self._load_avatars()

    @property
    def is_logged_in(self) -> bool:
        """Check if user is currently logged in."""
        return self._session_token is not None and self._user is not None

    @property
    def user(self) -> Optional[UserProfile]:
        """Get current user profile."""
        return self._user

    @property
    def session_token(self) -> Optional[str]:
        """Get current session token."""
        return self._session_token

    @property
    def credits_balance(self) -> int:
        """Get current credit balance."""
        return self._user.credits_balance if self._user else 0

    @property
    def avatars(self) -> List[AvatarMetadata]:
        """Get list of user's avatars."""
        return self._avatars.copy()

    @property
    def default_avatar(self) -> Optional[AvatarMetadata]:
        """Get the default avatar, or first avatar if none marked default."""
        for avatar in self._avatars:
            if avatar.is_default:
                return avatar
        return self._avatars[0] if self._avatars else None

    # -------------------------------------------------------------------------
    # Avatar Management
    # -------------------------------------------------------------------------

    def add_avatar(self, avatar: AvatarMetadata) -> None:
        """Add a new avatar."""
        avatar.owner_id = self._user.id if self._user else ""
        avatar.updated_at = datetime.utcnow().isoformat()

        # If this is the first avatar, make it default
        if not self._avatars:
            avatar.is_default = True

        self._avatars.append(avatar)
        self._save_avatars()
        self.avatars_changed.emit()
        logger.info(f"Added avatar: {avatar.display_name} ({avatar.id})")

    def update_avatar(self, avatar: AvatarMetadata) -> None:
        """Update an existing avatar."""
        for i, existing in enumerate(self._avatars):
            if existing.id == avatar.id:
                avatar.updated_at = datetime.utcnow().isoformat()
                self._avatars[i] = avatar
                self._save_avatars()
                self.avatars_changed.emit()
                logger.info(f"Updated avatar: {avatar.display_name}")
                return
        logger.warning(f"Avatar not found for update: {avatar.id}")

    def delete_avatar(self, avatar_id: str) -> bool:
        """Delete an avatar by ID. Returns True if deleted."""
        for i, avatar in enumerate(self._avatars):
            if avatar.id == avatar_id:
                was_default = avatar.is_default
                del self._avatars[i]

                # If we deleted the default, make first remaining avatar default
                if was_default and self._avatars:
                    self._avatars[0].is_default = True

                self._save_avatars()
                self.avatars_changed.emit()
                logger.info(f"Deleted avatar: {avatar_id}")
                return True
        return False

    def set_default_avatar(self, avatar_id: str) -> None:
        """Set an avatar as the default."""
        for avatar in self._avatars:
            avatar.is_default = (avatar.id == avatar_id)
        self._save_avatars()
        self.avatars_changed.emit()

    def get_avatar(self, avatar_id: str) -> Optional[AvatarMetadata]:
        """Get avatar by ID."""
        for avatar in self._avatars:
            if avatar.id == avatar_id:
                return avatar
        return None

    def _avatars_file_path(self) -> Path:
        """Get path to local avatars storage file."""
        # Store in user's app data directory
        app_data = Path.home() / '.noodlings'
        app_data.mkdir(exist_ok=True)
        return app_data / 'avatars.json'

    def _load_avatars(self) -> None:
        """Load avatars from local storage."""
        path = self._avatars_file_path()
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self._avatars = [AvatarMetadata.from_dict(a) for a in data]
                logger.debug(f"Loaded {len(self._avatars)} avatars from {path}")
            except Exception as e:
                logger.error(f"Failed to load avatars: {e}")
                self._avatars = []
        else:
            self._avatars = []

    def _save_avatars(self) -> None:
        """Save avatars to local storage."""
        path = self._avatars_file_path()
        try:
            with open(path, 'w') as f:
                json.dump([a.to_dict() for a in self._avatars], f, indent=2)
            logger.debug(f"Saved {len(self._avatars)} avatars to {path}")
        except Exception as e:
            logger.error(f"Failed to save avatars: {e}")

    # TODO: Cloud sync methods (future)
    # def sync_avatars_to_cloud(self) -> None:
    # def sync_avatars_from_cloud(self) -> None:
    # def upload_avatar_asset(self, avatar_id: str) -> str:  # Returns cloud asset_id

    def get_login_url(self, provider: str) -> str:
        """Get OAuth login URL for a provider."""
        # The return URL should open the app's auth handler
        # For desktop apps, we use a localhost callback
        return_url = 'http://localhost:19847/auth/callback'
        return f'{API_BASE_URL}/auth/login/{provider}?return_url={return_url}'

    def set_session(self, token: str):
        """
        Set session token and fetch user profile.
        Called after OAuth callback with token.
        """
        self._session_token = token
        self._save_session()

        # Fetch user profile
        self._fetch_user_profile()

    def logout(self):
        """Log out current user."""
        if self._session_token:
            # Try to invalidate session on server (fire and forget)
            try:
                import urllib.request
                req = urllib.request.Request(
                    f'{API_BASE_URL}/auth/logout',
                    method='POST',
                    headers={'Authorization': f'Bearer {self._session_token}'}
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                logger.warning(f"Failed to invalidate session on server: {e}")

        self._session_token = None
        self._user = None
        self._clear_session()
        self.logged_out.emit()

    def refresh_user(self):
        """Refresh user profile from server."""
        if self._session_token:
            self._fetch_user_profile()

    def refresh_credits(self):
        """Refresh just the credit balance."""
        if not self._session_token:
            return

        try:
            import urllib.request
            req = urllib.request.Request(
                f'{API_BASE_URL}/credits/balance',
                headers={'Authorization': f'Bearer {self._session_token}'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                if self._user:
                    self._user.credits_balance = data.get('balance', 0)
                    self.credits_updated.emit(self._user.credits_balance)
        except Exception as e:
            logger.warning(f"Failed to refresh credits: {e}")

    def _fetch_user_profile(self):
        """Fetch user profile from API."""
        if not self._session_token:
            return

        try:
            import urllib.request
            req = urllib.request.Request(
                f'{API_BASE_URL}/auth/me',
                headers={'Authorization': f'Bearer {self._session_token}'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                user_data = data.get('user', {})

                self._user = UserProfile(
                    id=user_data.get('id', ''),
                    email=user_data.get('email', ''),
                    display_name=user_data.get('display_name'),
                    avatar_url=user_data.get('avatar_url'),
                    credits_balance=user_data.get('credits_balance', 0),
                    providers=data.get('providers', [])
                )

                self.logged_in.emit(self._user)
                logger.info(f"Logged in as {self._user.email}")

        except urllib.error.HTTPError as e:
            if e.code == 401:
                # Session expired
                logger.info("Session expired, logging out")
                self._session_token = None
                self._user = None
                self._clear_session()
                self.login_failed.emit("Session expired. Please log in again.")
            else:
                logger.error(f"Failed to fetch user profile: {e}")
                self.login_failed.emit(f"Failed to fetch profile: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch user profile: {e}")
            self.login_failed.emit(str(e))

    def _save_session(self):
        """Save session token securely."""
        if not self._session_token:
            return

        # Try keychain first (macOS)
        if self._save_to_keychain(self._session_token):
            logger.debug("Session saved to keychain")
            return

        # Fall back to QSettings (encrypted on some platforms)
        self._settings.setValue('session_token', self._session_token)
        logger.debug("Session saved to settings")

    def _restore_session(self):
        """Restore session from storage if auto-login is enabled."""
        # Check auto-login setting
        auto_login = self._settings.value('auto_login', False, type=bool)
        if not auto_login:
            logger.debug("Auto-login disabled, skipping session restore")
            return

        # Try keychain first
        token = self._load_from_keychain()

        if not token:
            # Fall back to QSettings
            token = self._settings.value('session_token')

        if token:
            self._session_token = token
            logger.debug("Session restored, validating...")
            # Validate by fetching profile
            self._fetch_user_profile()

    def _clear_session(self):
        """Clear stored session."""
        self._delete_from_keychain()
        self._settings.remove('session_token')

    # --- Keychain integration (macOS) ---

    def _save_to_keychain(self, token: str) -> bool:
        """Save token to macOS keychain."""
        try:
            import subprocess
            result = subprocess.run([
                'security', 'add-generic-password',
                '-a', 'NoodleStudio',
                '-s', 'noodlings.ai',
                '-w', token,
                '-U'  # Update if exists
            ], capture_output=True)
            return result.returncode == 0
        except Exception:
            return False

    def _load_from_keychain(self) -> Optional[str]:
        """Load token from macOS keychain."""
        try:
            import subprocess
            result = subprocess.run([
                'security', 'find-generic-password',
                '-a', 'NoodleStudio',
                '-s', 'noodlings.ai',
                '-w'
            ], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _delete_from_keychain(self):
        """Delete token from macOS keychain."""
        try:
            import subprocess
            subprocess.run([
                'security', 'delete-generic-password',
                '-a', 'NoodleStudio',
                '-s', 'noodlings.ai'
            ], capture_output=True)
        except Exception:
            pass


class OAuthCallbackServer:
    """
    Local HTTP server to receive OAuth callbacks.

    After OAuth provider redirects to localhost:19847/auth/callback#token=xxx,
    this server captures the token and notifies the account manager.
    """

    def __init__(self, account_manager: AccountManager):
        self.account_manager = account_manager
        self._server = None
        self._thread = None

    def start(self):
        """Start the callback server."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading

        account_manager = self.account_manager

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Serve the callback page that extracts token from URL fragment
                if self.path.startswith('/auth/callback'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()

                    # JavaScript extracts token from hash and sends to /auth/token
                    html = '''<!DOCTYPE html>
<html>
<head>
    <title>NoodleStudio Login</title>
    <style>
        body {
            background: #2a2a2a;
            color: #d2d2d2;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            text-align: center;
            padding: 40px;
            background: #383838;
            border-radius: 8px;
        }
        h1 { color: #76AF6A; }
        p { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Logged In!</h1>
        <p>You can close this window and return to NoodleStudio.</p>
    </div>
    <script>
        // Extract token from URL fragment
        const hash = window.location.hash.substring(1);
        const params = new URLSearchParams(hash);
        const token = params.get('token');

        if (token) {
            // Send token to local server
            fetch('/auth/token', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({token: token})
            }).then(() => {
                // Close window after short delay
                setTimeout(() => window.close(), 1500);
            });
        }
    </script>
</body>
</html>'''
                    self.wfile.write(html.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/auth/token':
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode())
                    token = data.get('token')

                    if token:
                        # Set session in account manager
                        account_manager.set_session(token)

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"success": true}')
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress HTTP logs
                pass

        self._server = HTTPServer(('localhost', 19847), CallbackHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("OAuth callback server started on localhost:19847")

    def stop(self):
        """Stop the callback server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None


# Global callback server instance
_callback_server: Optional[OAuthCallbackServer] = None


def start_oauth_callback_server():
    """Start the OAuth callback server if not already running."""
    global _callback_server
    if _callback_server is None:
        _callback_server = OAuthCallbackServer(AccountManager.instance())
        _callback_server.start()


def stop_oauth_callback_server():
    """Stop the OAuth callback server."""
    global _callback_server
    if _callback_server:
        _callback_server.stop()
        _callback_server = None
