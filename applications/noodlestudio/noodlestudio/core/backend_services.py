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
#   Backend Services for NoodleStudio
#
#   ================================= Client-side wrappers fo...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.backend_services
# PURPOSE:  Backend Services for NoodleStudio
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AssetType, OnlineStatus, FriendStatus, TeleportStatus, InventoryItem
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import json
import asyncio
import aiohttp
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = os.environ.get('NOODLINGS_API_URL', 'https://noodlings-api.caitsters.workers.dev')


# =============================================================================
# Data Classes
# =============================================================================

class AssetType(Enum):
    """Types of assets that can be owned."""
    AVATAR = "avatar"
    PROP = "prop"
    STAGE = "stage"
    GAUSSIAN = "gaussian"
    AUDIO = "audio"
    SCRIPT = "script"


class OnlineStatus(Enum):
    """User online status."""
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"
    INVISIBLE = "invisible"


class FriendStatus(Enum):
    """Friend request status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"


class TeleportStatus(Enum):
    """Teleport invitation status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"


@dataclass
class InventoryItem:
    """An item in the user's inventory."""
    id: str
    asset_type: AssetType
    name: str
    description: str = ""
    thumbnail_url: Optional[str] = None
    asset_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acquired_at: Optional[datetime] = None
    is_equipped: bool = False
    is_favorite: bool = False


@dataclass
class Friend:
    """A friend in the social graph."""
    user_id: str
    display_name: str
    avatar_url: Optional[str] = None
    status: FriendStatus = FriendStatus.ACCEPTED
    online_status: OnlineStatus = OnlineStatus.OFFLINE
    current_location: Optional[str] = None  # Stage ID if online
    last_seen: Optional[datetime] = None
    added_at: Optional[datetime] = None
    can_teleport_to: bool = True  # Permission to TP to them
    can_see_location: bool = True  # Permission to see where they are


@dataclass
class PublicStage:
    """A stage listed in the world directory."""
    id: str
    name: str
    description: str = ""
    owner_id: str = ""
    owner_name: str = ""
    thumbnail_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    population: int = 0
    max_population: int = 32
    is_featured: bool = False
    rating: float = 0.0
    created_at: Optional[datetime] = None


@dataclass
class TeleportInvitation:
    """An invitation to teleport."""
    id: str
    from_user_id: str
    from_user_name: str
    to_user_id: str
    destination_stage_id: str
    destination_stage_name: str
    destination_position: List[float] = field(default_factory=lambda: [0, 0, 0])
    message: str = ""
    status: TeleportStatus = TeleportStatus.PENDING
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


@dataclass
class Achievement:
    """An achievement/milestone."""
    id: str
    name: str
    description: str
    icon_url: Optional[str] = None
    category: str = "general"
    points: int = 0
    is_unlocked: bool = False
    unlocked_at: Optional[datetime] = None
    progress: float = 0.0  # 0-1 for progress-based achievements
    progress_max: int = 1


# =============================================================================
# Base Service Class
# =============================================================================

class BaseService:
    """Base class for all backend services."""

    def __init__(self, get_token: Callable[[], Optional[str]]):
        """
        Initialize service.

        Args:
            get_token: Callback to get current session token
        """
        self._get_token = get_token
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def token(self) -> Optional[str]:
        """Get current session token."""
        return self._get_token()

    @property
    def headers(self) -> Dict[str, str]:
        """Get headers with auth token."""
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an API request."""
        session = await self._get_session()
        url = f"{API_BASE_URL}{endpoint}"

        try:
            async with session.request(
                method,
                url,
                json=data,
                params=params,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 401:
                    raise AuthenticationError("Session expired")
                elif response.status == 403:
                    raise PermissionError("Access denied")
                elif response.status == 404:
                    raise NotFoundError(f"Resource not found: {endpoint}")
                elif response.status >= 400:
                    error_text = await response.text()
                    raise APIError(f"API error {response.status}: {error_text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to API: {e}")

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# Custom Exceptions
# =============================================================================

class APIError(Exception):
    """Base API error."""
    pass


class AuthenticationError(APIError):
    """Authentication failed."""
    pass


class NotFoundError(APIError):
    """Resource not found."""
    pass


# =============================================================================
# Inventory Service
# =============================================================================

class InventoryService(BaseService):
    """
    Manages user's inventory of assets.

    Endpoints:
        GET  /inventory - List all inventory items
        GET  /inventory/:id - Get specific item
        POST /inventory - Add item (from purchase/gift)
        PUT  /inventory/:id - Update item (equip, favorite)
        DELETE /inventory/:id - Remove item
    """

    async def list_items(
        self,
        asset_type: Optional[AssetType] = None,
        equipped_only: bool = False,
        favorites_only: bool = False
    ) -> List[InventoryItem]:
        """List inventory items with optional filters."""
        params = {}
        if asset_type:
            params['type'] = asset_type.value
        if equipped_only:
            params['equipped'] = 'true'
        if favorites_only:
            params['favorites'] = 'true'

        data = await self._request('GET', '/inventory', params=params)

        return [
            InventoryItem(
                id=item['id'],
                asset_type=AssetType(item['type']),
                name=item['name'],
                description=item.get('description', ''),
                thumbnail_url=item.get('thumbnail_url'),
                asset_url=item.get('asset_url'),
                metadata=item.get('metadata', {}),
                acquired_at=datetime.fromisoformat(item['acquired_at']) if item.get('acquired_at') else None,
                is_equipped=item.get('is_equipped', False),
                is_favorite=item.get('is_favorite', False),
            )
            for item in data.get('items', [])
        ]

    async def get_item(self, item_id: str) -> InventoryItem:
        """Get a specific inventory item."""
        data = await self._request('GET', f'/inventory/{item_id}')
        item = data['item']

        return InventoryItem(
            id=item['id'],
            asset_type=AssetType(item['type']),
            name=item['name'],
            description=item.get('description', ''),
            thumbnail_url=item.get('thumbnail_url'),
            asset_url=item.get('asset_url'),
            metadata=item.get('metadata', {}),
            acquired_at=datetime.fromisoformat(item['acquired_at']) if item.get('acquired_at') else None,
            is_equipped=item.get('is_equipped', False),
            is_favorite=item.get('is_favorite', False),
        )

    async def equip_item(self, item_id: str) -> bool:
        """Equip an item (avatar, etc.)."""
        await self._request('PUT', f'/inventory/{item_id}', data={'is_equipped': True})
        return True

    async def unequip_item(self, item_id: str) -> bool:
        """Unequip an item."""
        await self._request('PUT', f'/inventory/{item_id}', data={'is_equipped': False})
        return True

    async def set_favorite(self, item_id: str, is_favorite: bool) -> bool:
        """Set/unset item as favorite."""
        await self._request('PUT', f'/inventory/{item_id}', data={'is_favorite': is_favorite})
        return True

    async def get_equipped_avatar(self) -> Optional[InventoryItem]:
        """Get the currently equipped avatar."""
        items = await self.list_items(asset_type=AssetType.AVATAR, equipped_only=True)
        return items[0] if items else None


# =============================================================================
# Friend Service
# =============================================================================

class FriendService(BaseService):
    """
    Manages social connections.

    Endpoints:
        GET  /friends - List friends
        POST /friends/request - Send friend request
        POST /friends/accept/:id - Accept request
        POST /friends/decline/:id - Decline request
        POST /friends/block/:id - Block user
        DELETE /friends/:id - Remove friend
        PUT /friends/:id/permissions - Update permissions
        GET /friends/online - List online friends
    """

    async def list_friends(self, include_pending: bool = False) -> List[Friend]:
        """List all friends."""
        params = {'include_pending': 'true'} if include_pending else {}
        data = await self._request('GET', '/friends', params=params)

        return [
            Friend(
                user_id=f['user_id'],
                display_name=f['display_name'],
                avatar_url=f.get('avatar_url'),
                status=FriendStatus(f.get('status', 'accepted')),
                online_status=OnlineStatus(f.get('online_status', 'offline')),
                current_location=f.get('current_location'),
                last_seen=datetime.fromisoformat(f['last_seen']) if f.get('last_seen') else None,
                added_at=datetime.fromisoformat(f['added_at']) if f.get('added_at') else None,
                can_teleport_to=f.get('can_teleport_to', True),
                can_see_location=f.get('can_see_location', True),
            )
            for f in data.get('friends', [])
        ]

    async def get_online_friends(self) -> List[Friend]:
        """Get list of online friends."""
        data = await self._request('GET', '/friends/online')

        return [
            Friend(
                user_id=f['user_id'],
                display_name=f['display_name'],
                avatar_url=f.get('avatar_url'),
                status=FriendStatus.ACCEPTED,
                online_status=OnlineStatus(f.get('online_status', 'online')),
                current_location=f.get('current_location'),
                can_teleport_to=f.get('can_teleport_to', True),
                can_see_location=f.get('can_see_location', True),
            )
            for f in data.get('friends', [])
        ]

    async def send_request(self, user_id: str, message: str = "") -> bool:
        """Send a friend request."""
        await self._request('POST', '/friends/request', data={
            'user_id': user_id,
            'message': message
        })
        return True

    async def accept_request(self, user_id: str) -> bool:
        """Accept a friend request."""
        await self._request('POST', f'/friends/accept/{user_id}')
        return True

    async def decline_request(self, user_id: str) -> bool:
        """Decline a friend request."""
        await self._request('POST', f'/friends/decline/{user_id}')
        return True

    async def remove_friend(self, user_id: str) -> bool:
        """Remove a friend."""
        await self._request('DELETE', f'/friends/{user_id}')
        return True

    async def block_user(self, user_id: str) -> bool:
        """Block a user."""
        await self._request('POST', f'/friends/block/{user_id}')
        return True

    async def update_permissions(
        self,
        user_id: str,
        can_teleport_to: Optional[bool] = None,
        can_see_location: Optional[bool] = None
    ) -> bool:
        """Update friend permissions."""
        data = {}
        if can_teleport_to is not None:
            data['can_teleport_to'] = can_teleport_to
        if can_see_location is not None:
            data['can_see_location'] = can_see_location

        await self._request('PUT', f'/friends/{user_id}/permissions', data=data)
        return True

    async def set_online_status(self, status: OnlineStatus) -> bool:
        """Set your online status."""
        await self._request('PUT', '/me/status', data={'status': status.value})
        return True


# =============================================================================
# World Directory Service
# =============================================================================

class WorldDirectoryService(BaseService):
    """
    Discovers public stages.

    Endpoints:
        GET /worlds - List public stages
        GET /worlds/featured - Get featured stages
        GET /worlds/popular - Get popular stages
        GET /worlds/:id - Get stage details
        POST /worlds - Register a stage as public
        PUT /worlds/:id - Update stage listing
        DELETE /worlds/:id - Remove from directory
    """

    async def list_stages(
        self,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        sort_by: str = "population",
        limit: int = 50,
        offset: int = 0
    ) -> List[PublicStage]:
        """List public stages."""
        params = {
            'sort': sort_by,
            'limit': str(limit),
            'offset': str(offset)
        }
        if tags:
            params['tags'] = ','.join(tags)
        if search:
            params['search'] = search

        data = await self._request('GET', '/worlds', params=params)

        return [
            PublicStage(
                id=s['id'],
                name=s['name'],
                description=s.get('description', ''),
                owner_id=s.get('owner_id', ''),
                owner_name=s.get('owner_name', ''),
                thumbnail_url=s.get('thumbnail_url'),
                tags=s.get('tags', []),
                population=s.get('population', 0),
                max_population=s.get('max_population', 32),
                is_featured=s.get('is_featured', False),
                rating=s.get('rating', 0.0),
                created_at=datetime.fromisoformat(s['created_at']) if s.get('created_at') else None,
            )
            for s in data.get('stages', [])
        ]

    async def get_featured(self) -> List[PublicStage]:
        """Get featured stages."""
        data = await self._request('GET', '/worlds/featured')
        return [
            PublicStage(
                id=s['id'],
                name=s['name'],
                description=s.get('description', ''),
                thumbnail_url=s.get('thumbnail_url'),
                tags=s.get('tags', []),
                population=s.get('population', 0),
                is_featured=True,
            )
            for s in data.get('stages', [])
        ]

    async def get_popular(self, limit: int = 10) -> List[PublicStage]:
        """Get most populated stages."""
        data = await self._request('GET', '/worlds/popular', params={'limit': str(limit)})
        return [
            PublicStage(
                id=s['id'],
                name=s['name'],
                population=s.get('population', 0),
                thumbnail_url=s.get('thumbnail_url'),
            )
            for s in data.get('stages', [])
        ]

    async def get_stage(self, stage_id: str) -> PublicStage:
        """Get stage details."""
        data = await self._request('GET', f'/worlds/{stage_id}')
        s = data['stage']

        return PublicStage(
            id=s['id'],
            name=s['name'],
            description=s.get('description', ''),
            owner_id=s.get('owner_id', ''),
            owner_name=s.get('owner_name', ''),
            thumbnail_url=s.get('thumbnail_url'),
            tags=s.get('tags', []),
            population=s.get('population', 0),
            max_population=s.get('max_population', 32),
            is_featured=s.get('is_featured', False),
            rating=s.get('rating', 0.0),
            created_at=datetime.fromisoformat(s['created_at']) if s.get('created_at') else None,
        )

    async def register_stage(
        self,
        stage_id: str,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        thumbnail_url: Optional[str] = None,
        max_population: int = 32
    ) -> bool:
        """Register a stage in the public directory."""
        await self._request('POST', '/worlds', data={
            'stage_id': stage_id,
            'name': name,
            'description': description,
            'tags': tags or [],
            'thumbnail_url': thumbnail_url,
            'max_population': max_population,
        })
        return True

    async def update_population(self, stage_id: str, population: int) -> bool:
        """Update stage population count (called by MUSH server)."""
        await self._request('PUT', f'/worlds/{stage_id}/population', data={
            'population': population
        })
        return True


# =============================================================================
# Teleport Service
# =============================================================================

class TeleportService(BaseService):
    """
    Handles teleport invitations.

    Flow:
        1. User A sends invitation to User B
        2. User B sees invitation in their queue
        3. User B accepts/declines
        4. If accepted, MUSH server handles the actual teleport

    Endpoints:
        GET /teleport/invitations - List pending invitations
        POST /teleport/invite - Send invitation
        POST /teleport/accept/:id - Accept invitation
        POST /teleport/decline/:id - Decline invitation
        GET /teleport/destinations - List saved destinations
        POST /teleport/destinations - Save a destination
    """

    async def get_invitations(self) -> List[TeleportInvitation]:
        """Get pending teleport invitations."""
        data = await self._request('GET', '/teleport/invitations')

        return [
            TeleportInvitation(
                id=inv['id'],
                from_user_id=inv['from_user_id'],
                from_user_name=inv['from_user_name'],
                to_user_id=inv['to_user_id'],
                destination_stage_id=inv['destination_stage_id'],
                destination_stage_name=inv.get('destination_stage_name', ''),
                destination_position=inv.get('destination_position', [0, 0, 0]),
                message=inv.get('message', ''),
                status=TeleportStatus(inv.get('status', 'pending')),
                created_at=datetime.fromisoformat(inv['created_at']) if inv.get('created_at') else None,
                expires_at=datetime.fromisoformat(inv['expires_at']) if inv.get('expires_at') else None,
            )
            for inv in data.get('invitations', [])
        ]

    async def send_invitation(
        self,
        to_user_id: str,
        destination_stage_id: str,
        destination_stage_name: str = "",
        destination_position: Optional[List[float]] = None,
        message: str = ""
    ) -> str:
        """Send a teleport invitation. Returns invitation ID."""
        data = await self._request('POST', '/teleport/invite', data={
            'to_user_id': to_user_id,
            'destination_stage_id': destination_stage_id,
            'destination_stage_name': destination_stage_name,
            'destination_position': destination_position or [0, 0, 0],
            'message': message,
        })
        return data['invitation_id']

    async def accept_invitation(self, invitation_id: str) -> Dict[str, Any]:
        """Accept a teleport invitation. Returns destination info."""
        data = await self._request('POST', f'/teleport/accept/{invitation_id}')
        return {
            'stage_id': data['stage_id'],
            'stage_name': data.get('stage_name', ''),
            'position': data.get('position', [0, 0, 0]),
            'server_url': data.get('server_url'),  # WebSocket URL for MUSH
        }

    async def decline_invitation(self, invitation_id: str) -> bool:
        """Decline a teleport invitation."""
        await self._request('POST', f'/teleport/decline/{invitation_id}')
        return True

    async def get_saved_destinations(self) -> List[Dict[str, Any]]:
        """Get user's saved destinations (bookmarks)."""
        data = await self._request('GET', '/teleport/destinations')
        return data.get('destinations', [])

    async def save_destination(
        self,
        name: str,
        stage_id: str,
        stage_name: str,
        position: List[float]
    ) -> bool:
        """Save a destination bookmark."""
        await self._request('POST', '/teleport/destinations', data={
            'name': name,
            'stage_id': stage_id,
            'stage_name': stage_name,
            'position': position,
        })
        return True


# =============================================================================
# Achievement Service
# =============================================================================

class AchievementService(BaseService):
    """
    Tracks achievements and milestones.

    Endpoints:
        GET /achievements - List all achievements
        GET /achievements/unlocked - List unlocked achievements
        POST /achievements/unlock/:id - Unlock an achievement (server-side validation)
        POST /achievements/progress/:id - Update progress
    """

    async def list_achievements(self, unlocked_only: bool = False) -> List[Achievement]:
        """List all achievements."""
        endpoint = '/achievements/unlocked' if unlocked_only else '/achievements'
        data = await self._request('GET', endpoint)

        return [
            Achievement(
                id=a['id'],
                name=a['name'],
                description=a.get('description', ''),
                icon_url=a.get('icon_url'),
                category=a.get('category', 'general'),
                points=a.get('points', 0),
                is_unlocked=a.get('is_unlocked', False),
                unlocked_at=datetime.fromisoformat(a['unlocked_at']) if a.get('unlocked_at') else None,
                progress=a.get('progress', 0.0),
                progress_max=a.get('progress_max', 1),
            )
            for a in data.get('achievements', [])
        ]

    async def get_progress(self, achievement_id: str) -> Dict[str, Any]:
        """Get progress for a specific achievement."""
        data = await self._request('GET', f'/achievements/{achievement_id}')
        return {
            'progress': data.get('progress', 0),
            'progress_max': data.get('progress_max', 1),
            'is_unlocked': data.get('is_unlocked', False),
        }

    async def report_progress(self, achievement_id: str, progress: float) -> bool:
        """Report progress toward an achievement."""
        await self._request('POST', f'/achievements/progress/{achievement_id}', data={
            'progress': progress
        })
        return True

    async def get_total_points(self) -> int:
        """Get total achievement points."""
        data = await self._request('GET', '/achievements/points')
        return data.get('total_points', 0)


# =============================================================================
# Asset Storage Service
# =============================================================================

class AssetStorageService(BaseService):
    """
    Upload and download assets from R2 storage.

    Endpoints:
        POST /assets/upload - Get presigned upload URL
        GET /assets/:id - Get asset download URL
        DELETE /assets/:id - Delete an asset
    """

    async def get_upload_url(
        self,
        filename: str,
        content_type: str,
        asset_type: AssetType
    ) -> Dict[str, str]:
        """Get a presigned URL for uploading an asset."""
        data = await self._request('POST', '/assets/upload', data={
            'filename': filename,
            'content_type': content_type,
            'asset_type': asset_type.value,
        })
        return {
            'upload_url': data['upload_url'],
            'asset_id': data['asset_id'],
            'final_url': data['final_url'],
        }

    async def upload_file(
        self,
        file_path: str,
        asset_type: AssetType,
        content_type: Optional[str] = None
    ) -> str:
        """Upload a file and return its URL."""
        path = Path(file_path)

        if not content_type:
            # Guess content type
            ext = path.suffix.lower()
            content_types = {
                '.ply': 'application/octet-stream',
                '.glb': 'model/gltf-binary',
                '.gltf': 'model/gltf+json',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.ogg': 'audio/ogg',
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
            }
            content_type = content_types.get(ext, 'application/octet-stream')

        # Get upload URL
        urls = await self.get_upload_url(path.name, content_type, asset_type)

        # Upload the file
        session = await self._get_session()
        with open(file_path, 'rb') as f:
            async with session.put(
                urls['upload_url'],
                data=f.read(),
                headers={'Content-Type': content_type}
            ) as response:
                if response.status != 200:
                    raise APIError(f"Upload failed: {response.status}")

        return urls['final_url']

    async def get_download_url(self, asset_id: str) -> str:
        """Get a download URL for an asset."""
        data = await self._request('GET', f'/assets/{asset_id}')
        return data['url']

    async def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset."""
        await self._request('DELETE', f'/assets/{asset_id}')
        return True


# =============================================================================
# Unified Backend Client
# =============================================================================

class BackendClient:
    """
    Unified client for all backend services.

    Usage:
        client = BackendClient(lambda: account_manager.session_token)
        await client.initialize()

        # Use services
        avatar = await client.inventory.get_equipped_avatar()
        friends = await client.friends.get_online_friends()
        stages = await client.worlds.get_popular()
    """

    def __init__(self, get_token: Callable[[], Optional[str]]):
        """
        Initialize backend client.

        Args:
            get_token: Callback to get current session token
        """
        self._get_token = get_token

        # Initialize services
        self.inventory = InventoryService(get_token)
        self.friends = FriendService(get_token)
        self.worlds = WorldDirectoryService(get_token)
        self.teleport = TeleportService(get_token)
        self.achievements = AchievementService(get_token)
        self.assets = AssetStorageService(get_token)

        self._initialized = False

    async def initialize(self):
        """Initialize the client (verify connection)."""
        # Could do a health check here
        self._initialized = True
        logger.info("[BackendClient] Initialized")

    async def close(self):
        """Close all service connections."""
        await self.inventory.close()
        await self.friends.close()
        await self.worlds.close()
        await self.teleport.close()
        await self.achievements.close()
        await self.assets.close()
        logger.info("[BackendClient] Closed")

    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self._get_token() is not None


# =============================================================================
# Global Singleton
# =============================================================================

_backend_client: Optional[BackendClient] = None


def get_backend_client() -> BackendClient:
    """Get the global BackendClient singleton."""
    global _backend_client
    if _backend_client is None:
        from .account_manager import AccountManager
        _backend_client = BackendClient(lambda: AccountManager.instance().session_token)
    return _backend_client


async def initialize_backend():
    """Initialize the global backend client."""
    client = get_backend_client()
    await client.initialize()
    return client


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Backend Services Test")
    print("=" * 60)

    # Test data classes
    item = InventoryItem(
        id="avatar_001",
        asset_type=AssetType.AVATAR,
        name="Fire Imp Avatar",
        description="A mischievous fire imp",
        is_equipped=True,
    )
    print(f"\nInventory item: {item.name} ({item.asset_type.value})")

    friend = Friend(
        user_id="user_123",
        display_name="TestFriend",
        online_status=OnlineStatus.ONLINE,
        current_location="the_nexus",
    )
    print(f"Friend: {friend.display_name} is {friend.online_status.value} at {friend.current_location}")

    stage = PublicStage(
        id="stage_nexus",
        name="The Nexus",
        description="Central hub for all noodlings",
        population=42,
        is_featured=True,
    )
    print(f"Stage: {stage.name} - {stage.population} online")

    # Test client creation
    client = BackendClient(lambda: "test_token")
    print(f"\nBackend client created")
    print(f"  - Inventory service: {type(client.inventory).__name__}")
    print(f"  - Friends service: {type(client.friends).__name__}")
    print(f"  - Worlds service: {type(client.worlds).__name__}")
    print(f"  - Teleport service: {type(client.teleport).__name__}")
    print(f"  - Achievements service: {type(client.achievements).__name__}")
    print(f"  - Assets service: {type(client.assets).__name__}")

    print("\n" + "=" * 60)
    print("Test complete!")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
