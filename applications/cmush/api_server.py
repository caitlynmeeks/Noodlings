"""
NoodleScope 2.0 API Server

HTTP REST API for:
- Session profiler data endpoints
- @Kimmie interpretation service
- Static file serving for noodlescope2.html

Runs alongside WebSocket server on port 8081.

Author: noodleMUSH Project
Date: November 2025
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from aiohttp import web
import aiohttp_cors

from session_profiler import SessionProfiler
from kimmie_character import KimmieCharacter
from performance_tracker import get_tracker

logger = logging.getLogger(__name__)


class NoodleScopeAPI:
    """
    REST API server for NoodleScope 2.0.

    Provides endpoints for session data retrieval and @Kimmie interpretations.
    """

    def __init__(
        self,
        session_profiler: Optional[SessionProfiler] = None,
        kimmie: Optional[KimmieCharacter] = None,
        host: str = '0.0.0.0',
        port: int = 8081
    ):
        """
        Initialize NoodleScope API server.

        Args:
            session_profiler: Active session profiler instance
            kimmie: @Kimmie character instance
            host: Server host
            port: Server port
        """
        self.session_profiler = session_profiler
        self.kimmie = kimmie
        self.host = host
        self.port = port

        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()

        logger.info(f"NoodleScope API initialized on {host}:{port}")

    def setup_routes(self):
        """Setup HTTP routes."""
        # Session profiler endpoints
        self.app.router.add_get('/api/profiler/sessions', self.list_sessions)
        self.app.router.add_get('/api/profiler/session/{session_id}', self.get_session)
        self.app.router.add_get('/api/profiler/live-session', self.get_live_session)
        self.app.router.add_get('/api/profiler/realtime/{agent_id}', self.get_realtime_feed)
        self.app.router.add_get('/api/profiler/operations/{agent_id}', self.get_operations)

        # @Kimmie interpretation endpoint
        self.app.router.add_post('/api/kimmie/interpret', self.kimmie_interpret)

        # Health check
        self.app.router.add_get('/api/health', self.health_check)

        # Static file serving (noodlescope2.html)
        web_dir = Path(__file__).parent / 'web'
        self.app.router.add_static('/web/', path=web_dir, name='web')
        self.app.router.add_get('/noodlescope', self.serve_noodlescope)

        logger.info("API routes configured")

    def setup_cors(self):
        """Setup CORS for frontend access."""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })

        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            if not isinstance(route.resource, web.StaticResource):
                cors.add(route)

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            'status': 'ok',
            'profiler_active': self.session_profiler is not None,
            'kimmie_active': self.kimmie is not None
        })

    async def list_sessions(self, request: web.Request) -> web.Response:
        """
        List available profiler sessions.

        GET /api/profiler/sessions

        Returns:
            [{
                "id": "cmush_session_1234567890",
                "agents": ["agent_desobelle", "agent_callie"]
            }]
        """
        try:
            profiler_dir = Path('profiler_sessions')
            if not profiler_dir.exists():
                return web.json_response([])

            sessions = []
            for session_file in profiler_dir.glob('*.json'):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)

                    sessions.append({
                        'id': session_data['metadata']['session_id'],
                        'agents': session_data['metadata']['agents'],
                        'start_time': session_data['metadata']['start_time'],
                        'duration': session_data.get('duration', 0)
                    })
                except Exception as e:
                    logger.error(f"Error loading session {session_file}: {e}")
                    continue

            # Sort by start time (most recent first)
            sessions.sort(key=lambda s: s['start_time'], reverse=True)

            return web.json_response(sessions)

        except Exception as e:
            logger.error(f"Error listing sessions: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def get_session(self, request: web.Request) -> web.Response:
        """
        Get complete session data.

        GET /api/profiler/session/{session_id}

        Returns:
            {
                "metadata": {...},
                "duration": 123.45,
                "timelines": {
                    "agent_desobelle": [
                        {
                            "timestamp": 0.0,
                            "affect": {...},
                            "surprise": 0.15,
                            "hsi": {...},
                            ...
                        }
                    ]
                }
            }
        """
        try:
            session_id = request.match_info['session_id']
            session_file = Path('profiler_sessions') / f'{session_id}.json'

            if not session_file.exists():
                return web.json_response({'error': 'Session not found'}, status=404)

            with open(session_file, 'r') as f:
                session_data = json.load(f)

            return web.json_response(session_data)

        except Exception as e:
            logger.error(f"Error getting session: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def get_live_session(self, request: web.Request) -> web.Response:
        """
        Get current live session data from memory.

        GET /api/profiler/live-session

        Returns:
            {
                "metadata": {...},
                "duration": 123.45,
                "timelines": {...}
            }
        """
        try:
            if not self.session_profiler:
                return web.json_response({'error': 'No active session profiler'}, status=503)

            import time
            session_data = {
                "metadata": self.session_profiler.session_metadata,
                "duration": time.time() - self.session_profiler.session_start,
                "timelines": {}
            }

            # Convert timeline data with phenomenal_state properly formatted
            for agent_id, timeline in self.session_profiler.agent_timelines.items():
                session_data["timelines"][agent_id] = []
                for record in timeline:
                    # Flatten phenomenal_state structure for frontend
                    if 'phenomenal_state' in record and isinstance(record['phenomenal_state'], dict):
                        # If it's nested (fast/medium/slow), flatten to single array
                        if 'full' in record['phenomenal_state']:
                            phenomenal_state = record['phenomenal_state']['full']
                        else:
                            # Combine fast/medium/slow
                            phenomenal_state = (
                                record['phenomenal_state'].get('fast', []) +
                                record['phenomenal_state'].get('medium', []) +
                                record['phenomenal_state'].get('slow', [])
                            )
                    else:
                        phenomenal_state = record.get('phenomenal_state', [])

                    session_data["timelines"][agent_id].append({
                        "timestamp": record.get("timestamp", 0),
                        "phenomenal_state": phenomenal_state,
                        "affect": record.get("affect", {}),
                        "surprise": record.get("surprise", 0),
                        "hsi": record.get("hsi", {}),
                        "speech": record.get("speech", {}),
                        "event_context": record.get("event_context", "")
                    })

            return web.json_response(session_data)

        except Exception as e:
            logger.error(f"Error getting live session: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def get_realtime_feed(self, request: web.Request) -> web.Response:
        """
        Get recent timeline data for real-time display.

        GET /api/profiler/realtime/{agent_id}?last_n=100

        Returns:
            [
                {
                    "timestamp": 0.0,
                    "affect": {...},
                    "surprise": 0.15,
                    ...
                }
            ]
        """
        try:
            agent_id = request.match_info['agent_id']
            last_n = int(request.query.get('last_n', '100'))

            if not self.session_profiler:
                return web.json_response({'error': 'No active session profiler'}, status=503)

            timeline = self.session_profiler.get_realtime_feed(agent_id, last_n=last_n)

            return web.json_response(timeline)

        except Exception as e:
            logger.error(f"Error getting realtime feed: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def get_operations(self, request: web.Request) -> web.Response:
        """
        Get recent operations for an agent (for timeline console view).

        GET /api/profiler/operations/{agent_id}?last_n=50

        Returns:
            [
                {
                    "id": 0,
                    "agent_id": "agent_callie",
                    "type": "llm_generate_response",
                    "timestamp": "2025-11-15T01:23:45.123456",
                    "duration_ms": 1247.82,
                    "status": "success",
                    "details": {}
                },
                ...
            ]
        """
        try:
            agent_id = request.match_info['agent_id']
            last_n = int(request.query.get('last_n', 50))

            # Get tracker instance
            tracker = get_tracker()

            # Get operations for this agent
            operations = tracker.get_recent_operations(agent_id, last_n)

            return web.json_response(operations)

        except Exception as e:
            logger.error(f"Error getting operations: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def kimmie_interpret(self, request: web.Request) -> web.Response:
        """
        Get @Kimmie's interpretation of a timeline segment.

        POST /api/kimmie/interpret
        Body:
            {
                "agent_id": "agent_desobelle",
                "start_time": 10.5,
                "end_time": 25.3,
                "user_message": "What happened here?"
            }

        Returns:
            {
                "interpretation": "Okay, see this spike at 15.3 seconds? ..."
            }
        """
        try:
            if not self.kimmie:
                return web.json_response({'error': '@Kimmie not available'}, status=503)

            data = await request.json()
            agent_id = data.get('agent_id')
            start_time = data.get('start_time')
            end_time = data.get('end_time')
            user_message = data.get('user_message', 'What happened during this time?')

            if not agent_id:
                return web.json_response({'error': 'agent_id required'}, status=400)

            # Get interpretation from @Kimmie
            interpretation = await self.kimmie.interpret(
                user_message=user_message,
                agent_id=agent_id,
                start_time=start_time,
                end_time=end_time
            )

            return web.json_response({
                'interpretation': interpretation
            })

        except Exception as e:
            logger.error(f"Error getting @Kimmie interpretation: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def serve_noodlescope(self, request: web.Request) -> web.Response:
        """Serve noodlescope2.html."""
        html_path = Path(__file__).parent / 'web' / 'noodlescope2.html'

        if not html_path.exists():
            return web.Response(text='NoodleScope 2.0 not found', status=404)

        with open(html_path, 'r') as f:
            html_content = f.read()

        return web.Response(text=html_content, content_type='text/html')

    def set_session_profiler(self, profiler: SessionProfiler):
        """Update session profiler reference."""
        self.session_profiler = profiler
        if self.kimmie:
            self.kimmie.set_session_profiler(profiler)

    def set_kimmie(self, kimmie: KimmieCharacter):
        """Update @Kimmie character reference."""
        self.kimmie = kimmie
        if self.session_profiler:
            kimmie.set_session_profiler(self.session_profiler)

    async def start(self):
        """Start the API server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"NoodleScope API server running on http://{self.host}:{self.port}")
        logger.info(f"NoodleScope 2.0 UI available at http://{self.host}:{self.port}/noodlescope")

        return runner


async def standalone_server():
    """Run API server standalone (for testing)."""
    api = NoodleScopeAPI()
    runner = await api.start()

    try:
        print(f"🧠 NoodleScope API server running on http://localhost:8081")
        print(f"📊 Open http://localhost:8081/noodlescope")
        await asyncio.Event().wait()  # Run forever
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await runner.cleanup()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
    )

    asyncio.run(standalone_server())
