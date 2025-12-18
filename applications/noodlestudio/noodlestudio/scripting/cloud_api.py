"""
Cloud API for NoodleStudio Scripting
====================================

Provides context.noodle.cloud interface for interacting with
the Noodlings cloud backend.

Usage in ScriptedFacet:
    // Get current user
    const user = await context.noodle.cloud.getUser();

    // Check credits
    const balance = await context.noodle.cloud.getCredits();

    // Save noodling to cloud
    await context.noodle.cloud.saveNoodling(noodlingData);

    // Use routed LLM (charges credits)
    const response = await context.noodle.cloud.generate({
        model: 'anthropic/claude-3-sonnet',
        messages: [{ role: 'user', content: 'Hello!' }]
    });
"""

import os
import json
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

# API base URL
API_BASE_URL = os.environ.get('NOODLINGS_API_URL', 'https://noodlings-api.caitsters.workers.dev')


@dataclass
class User:
    id: str
    email: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    credits_balance: int


@dataclass
class Noodling:
    id: str
    name: str
    display_name: Optional[str]
    description: Optional[str]
    version: int
    is_public: bool
    created_at: int
    updated_at: int


@dataclass
class LLMResponse:
    id: str
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    credits_charged: int


class CloudAPIError(Exception):
    """Error from cloud API"""
    def __init__(self, message: str, status_code: int = 0, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class CloudAPI:
    """
    Client for Noodlings Cloud API.

    Handles authentication, noodling storage, and routed LLM calls.
    """

    def __init__(self, session_token: Optional[str] = None):
        self._session_token = session_token
        self._base_url = API_BASE_URL
        self._http_session: Optional[aiohttp.ClientSession] = None

    @property
    def is_authenticated(self) -> bool:
        return self._session_token is not None

    def set_session_token(self, token: str):
        """Set the session token for authenticated requests"""
        self._session_token = token

    def clear_session(self):
        """Clear the current session"""
        self._session_token = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def close(self):
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    def _headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self._session_token:
            headers['Authorization'] = f'Bearer {self._session_token}'
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        session = await self._get_session()
        url = f'{self._base_url}{path}'

        async with session.request(
            method,
            url,
            json=data,
            params=params,
            headers=self._headers()
        ) as response:
            result = await response.json()

            if not response.ok:
                raise CloudAPIError(
                    result.get('error', 'Request failed'),
                    response.status,
                    result.get('details')
                )

            return result

    # --- Authentication ---

    def get_login_url(self, provider: str, return_url: str = '/') -> str:
        """Get OAuth login URL for a provider"""
        from urllib.parse import urlencode
        params = urlencode({'return_url': return_url})
        return f'{self._base_url}/auth/login/{provider}?{params}'

    async def get_user(self) -> User:
        """Get current authenticated user"""
        result = await self._request('GET', '/auth/me')
        user_data = result['user']
        return User(
            id=user_data['id'],
            email=user_data['email'],
            display_name=user_data.get('display_name'),
            avatar_url=user_data.get('avatar_url'),
            credits_balance=user_data.get('credits_balance', 0)
        )

    async def logout(self):
        """Logout current session"""
        await self._request('POST', '/auth/logout')
        self.clear_session()

    # --- Credits ---

    async def get_credits(self) -> int:
        """Get current credit balance"""
        result = await self._request('GET', '/credits/balance')
        return result['balance']

    async def get_credit_tiers(self) -> List[Dict]:
        """Get available credit purchase tiers"""
        result = await self._request('GET', '/credits/tiers')
        return result['tiers']

    async def get_usage(self, period: str = 'month') -> Dict:
        """Get usage statistics for a period"""
        result = await self._request('GET', '/credits/usage', params={'period': period})
        return result

    async def purchase_credits(
        self,
        tier_index: int,
        success_url: str,
        cancel_url: str
    ) -> str:
        """Create Stripe checkout session for credits. Returns checkout URL."""
        result = await self._request('POST', '/credits/purchase', {
            'tier_index': tier_index,
            'success_url': success_url,
            'cancel_url': cancel_url
        })
        return result['checkout_url']

    # --- Noodlings ---

    async def list_noodlings(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[Noodling]:
        """List user's noodlings"""
        result = await self._request('GET', '/noodlings', params={
            'limit': limit,
            'offset': offset
        })
        return [
            Noodling(
                id=n['id'],
                name=n['name'],
                display_name=n.get('display_name'),
                description=n.get('description'),
                version=n.get('version', 1),
                is_public=bool(n.get('is_public')),
                created_at=n['created_at'],
                updated_at=n['updated_at']
            )
            for n in result['noodlings']
        ]

    async def create_noodling(
        self,
        name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Noodling:
        """Create a new noodling"""
        result = await self._request('POST', '/noodlings', {
            'name': name,
            'display_name': display_name,
            'description': description
        })
        n = result['noodling']
        return Noodling(
            id=n['id'],
            name=n['name'],
            display_name=n.get('display_name'),
            description=n.get('description'),
            version=n.get('version', 1),
            is_public=bool(n.get('is_public')),
            created_at=n['created_at'],
            updated_at=n['updated_at']
        )

    async def get_noodling(self, noodling_id: str) -> Dict:
        """Get noodling with all details"""
        return await self._request('GET', f'/noodlings/{noodling_id}')

    async def update_noodling(
        self,
        noodling_id: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        is_public: Optional[bool] = None
    ) -> Noodling:
        """Update noodling metadata"""
        data = {}
        if display_name is not None:
            data['display_name'] = display_name
        if description is not None:
            data['description'] = description
        if is_public is not None:
            data['is_public'] = is_public

        result = await self._request('PATCH', f'/noodlings/{noodling_id}', data)
        n = result['noodling']
        return Noodling(
            id=n['id'],
            name=n['name'],
            display_name=n.get('display_name'),
            description=n.get('description'),
            version=n.get('version', 1),
            is_public=bool(n.get('is_public')),
            created_at=n['created_at'],
            updated_at=n['updated_at']
        )

    async def delete_noodling(self, noodling_id: str):
        """Delete a noodling and all its assets"""
        await self._request('DELETE', f'/noodlings/{noodling_id}')

    # --- File uploads ---

    async def upload_recipe(self, noodling_id: str, content: str):
        """Upload recipe YAML to a noodling"""
        session = await self._get_session()
        url = f'{self._base_url}/noodlings/{noodling_id}/recipe'

        async with session.put(
            url,
            data=content,
            headers={
                'Content-Type': 'text/yaml',
                'Authorization': f'Bearer {self._session_token}'
            }
        ) as response:
            if not response.ok:
                result = await response.json()
                raise CloudAPIError(result.get('error'), response.status)

    async def download_recipe(self, noodling_id: str) -> str:
        """Download recipe YAML from a noodling"""
        session = await self._get_session()
        url = f'{self._base_url}/noodlings/{noodling_id}/recipe'

        async with session.get(url, headers=self._headers()) as response:
            if not response.ok:
                result = await response.json()
                raise CloudAPIError(result.get('error'), response.status)
            return await response.text()

    async def upload_facet_assembly(self, noodling_id: str, content: str):
        """Upload facet assembly YAML to a noodling"""
        session = await self._get_session()
        url = f'{self._base_url}/noodlings/{noodling_id}/facet-assembly'

        async with session.put(
            url,
            data=content,
            headers={
                'Content-Type': 'text/yaml',
                'Authorization': f'Bearer {self._session_token}'
            }
        ) as response:
            if not response.ok:
                result = await response.json()
                raise CloudAPIError(result.get('error'), response.status)

    async def download_facet_assembly(self, noodling_id: str) -> str:
        """Download facet assembly YAML from a noodling"""
        session = await self._get_session()
        url = f'{self._base_url}/noodlings/{noodling_id}/facet-assembly'

        async with session.get(url, headers=self._headers()) as response:
            if not response.ok:
                result = await response.json()
                raise CloudAPIError(result.get('error'), response.status)
            return await response.text()

    async def upload_charm_network(self, noodling_id: str, data: bytes):
        """Upload charm network weights to a noodling"""
        session = await self._get_session()
        url = f'{self._base_url}/noodlings/{noodling_id}/charm-network'

        async with session.put(
            url,
            data=data,
            headers={
                'Content-Type': 'application/octet-stream',
                'Authorization': f'Bearer {self._session_token}'
            }
        ) as response:
            if not response.ok:
                result = await response.json()
                raise CloudAPIError(result.get('error'), response.status)

    async def download_charm_network(self, noodling_id: str) -> bytes:
        """Download charm network weights from a noodling"""
        session = await self._get_session()
        url = f'{self._base_url}/noodlings/{noodling_id}/charm-network'

        async with session.get(url, headers=self._headers()) as response:
            if not response.ok:
                result = await response.json()
                raise CloudAPIError(result.get('error'), response.status)
            return await response.read()

    async def upload_reference(
        self,
        noodling_id: str,
        file_path: str,
        asset_type: str,
        purpose: Optional[str] = None
    ) -> Dict:
        """Upload a reference asset (image, audio, 3D, video)"""
        session = await self._get_session()
        url = f'{self._base_url}/noodlings/{noodling_id}/references'

        path = Path(file_path)

        data = aiohttp.FormData()
        data.add_field('file', open(path, 'rb'), filename=path.name)
        data.add_field('asset_type', asset_type)
        if purpose:
            data.add_field('purpose', purpose)

        async with session.post(
            url,
            data=data,
            headers={'Authorization': f'Bearer {self._session_token}'}
        ) as response:
            result = await response.json()
            if not response.ok:
                raise CloudAPIError(result.get('error'), response.status)
            return result['asset']

    # --- LLM Generation ---

    async def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        noodling_id: Optional[str] = None,
        facet_name: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate LLM completion via routed API.
        Credits are automatically deducted based on token usage.
        """
        result = await self._request('POST', '/llm/generate', {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'noodling_id': noodling_id,
            'facet_name': facet_name
        })

        return LLMResponse(
            id=result['id'],
            content=result['content'],
            model=result['model'],
            input_tokens=result['usage']['input_tokens'],
            output_tokens=result['usage']['output_tokens'],
            total_tokens=result['usage']['total_tokens'],
            credits_charged=result['credits_charged']
        )

    async def estimate_cost(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000
    ) -> Dict:
        """Estimate credit cost for a generation"""
        return await self._request('POST', '/llm/estimate', {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens
        })

    async def list_models(self) -> List[Dict]:
        """List available LLM models with pricing"""
        result = await self._request('GET', '/llm/models')
        return result['models']

    # --- Asset Store ---

    async def browse_store(
        self,
        limit: int = 20,
        offset: int = 0,
        tag: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Dict]:
        """Browse public noodlings in the store"""
        params = {'limit': limit, 'offset': offset}
        if tag:
            params['tag'] = tag
        if search:
            params['q'] = search

        result = await self._request('GET', '/store/browse', params=params)
        return result['noodlings']


# Singleton instance for scripting API
_cloud_api: Optional[CloudAPI] = None


def get_cloud_api() -> CloudAPI:
    """Get the shared CloudAPI instance"""
    global _cloud_api
    if _cloud_api is None:
        _cloud_api = CloudAPI()
    return _cloud_api


def set_session_token(token: str):
    """Set session token on the shared instance"""
    get_cloud_api().set_session_token(token)


# JavaScript API wrapper for ScriptedFacet
class CloudAPIJS:
    """
    JavaScript-compatible wrapper for CloudAPI.
    Exposed as context.noodle.cloud in ScriptedFacet.
    """

    def __init__(self, api: CloudAPI, noodling_id: Optional[str] = None):
        self._api = api
        self._noodling_id = noodling_id

    def _run(self, coro):
        """Run async code synchronously for JS compatibility"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

    # Expose methods that can be called from JavaScript

    def isAuthenticated(self) -> bool:
        return self._api.is_authenticated

    def getUser(self) -> Dict:
        user = self._run(self._api.get_user())
        return {
            'id': user.id,
            'email': user.email,
            'displayName': user.display_name,
            'avatarUrl': user.avatar_url,
            'creditsBalance': user.credits_balance
        }

    def getCredits(self) -> int:
        return self._run(self._api.get_credits())

    def getUsage(self, period: str = 'month') -> Dict:
        return self._run(self._api.get_usage(period))

    def listNoodlings(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        noodlings = self._run(self._api.list_noodlings(limit, offset))
        return [
            {
                'id': n.id,
                'name': n.name,
                'displayName': n.display_name,
                'description': n.description,
                'version': n.version,
                'isPublic': n.is_public
            }
            for n in noodlings
        ]

    def saveNoodling(self, data: Dict) -> Dict:
        """Save noodling to cloud (create or update)"""
        noodling_id = data.get('id')

        if noodling_id:
            # Update existing
            noodling = self._run(self._api.update_noodling(
                noodling_id,
                display_name=data.get('displayName'),
                description=data.get('description'),
                is_public=data.get('isPublic')
            ))
        else:
            # Create new
            noodling = self._run(self._api.create_noodling(
                name=data['name'],
                display_name=data.get('displayName'),
                description=data.get('description')
            ))

        # Upload files if provided
        if 'recipe' in data:
            self._run(self._api.upload_recipe(noodling.id, data['recipe']))
        if 'facetAssembly' in data:
            self._run(self._api.upload_facet_assembly(noodling.id, data['facetAssembly']))

        return {'id': noodling.id, 'version': noodling.version}

    def loadNoodling(self, noodling_id: str) -> Dict:
        """Load noodling from cloud"""
        return self._run(self._api.get_noodling(noodling_id))

    def generate(self, request: Dict) -> Dict:
        """Generate LLM completion"""
        response = self._run(self._api.generate(
            model=request['model'],
            messages=request['messages'],
            max_tokens=request.get('maxTokens', 1000),
            temperature=request.get('temperature', 0.7),
            noodling_id=request.get('noodlingId') or self._noodling_id,
            facet_name=request.get('facetName')
        ))

        return {
            'id': response.id,
            'content': response.content,
            'model': response.model,
            'usage': {
                'inputTokens': response.input_tokens,
                'outputTokens': response.output_tokens,
                'totalTokens': response.total_tokens
            },
            'creditsCharged': response.credits_charged
        }

    def estimateCost(self, request: Dict) -> Dict:
        """Estimate generation cost"""
        return self._run(self._api.estimate_cost(
            model=request['model'],
            messages=request['messages'],
            max_tokens=request.get('maxTokens', 1000)
        ))

    def listModels(self) -> List[Dict]:
        """List available models"""
        return self._run(self._api.list_models())

    def browseStore(
        self,
        limit: int = 20,
        offset: int = 0,
        tag: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Dict]:
        """Browse asset store"""
        return self._run(self._api.browse_store(limit, offset, tag, search))
