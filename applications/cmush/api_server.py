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
        config: Optional[Dict] = None,
        agent_manager = None,
        server = None,
        host: str = '0.0.0.0',
        port: int = 8081
    ):
        """
        Initialize NoodleScope API server.

        Args:
            session_profiler: Active session profiler instance
            kimmie: @Kimmie character instance
            config: Server configuration dict
            agent_manager: Agent manager instance
            server: CMUSHServer instance for shutdown control
            host: Server host
            port: Server port
        """
        self.session_profiler = session_profiler
        self.kimmie = kimmie
        self.config = config or {}
        self.server = server
        self.agent_manager = agent_manager
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

        # LLM configuration endpoints (for UI)
        self.app.router.add_get('/api/config', self.get_config)
        self.app.router.add_post('/api/config/save', self.save_config)
        self.app.router.add_get('/api/agents', self.get_agents)
        self.app.router.add_get('/api/agents/{agent_id}/state', self.get_agent_state)
        self.app.router.add_post('/api/agents/{agent_id}/update', self.update_agent)
        self.app.router.add_delete('/api/agents/{agent_id}', self.delete_agent)
        self.app.router.add_post('/api/agents', self.create_agent)

        # Cognitive Components (for Inspector)
        self.app.router.add_get('/api/agents/{agent_id}/components', self.get_agent_components)
        self.app.router.add_get('/api/agents/{agent_id}/components/{component_id}', self.get_component)
        self.app.router.add_post('/api/agents/{agent_id}/components/{component_id}/update', self.update_component)

        # Objects and rooms
        self.app.router.add_post('/api/objects', self.create_object)
        self.app.router.add_post('/api/objects/{object_id}/update', self.update_object)
        self.app.router.add_post('/api/rooms', self.create_room)
        self.app.router.add_post('/api/rooms/{room_id}/update', self.update_room)

        # Server control
        self.app.router.add_post('/api/shutdown', self.shutdown_server)

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

    async def get_config(self, request: web.Request) -> web.Response:
        """Get server configuration (for LLM config UI)."""
        return web.json_response({
            'llm': self.config.get('llm', {}),
            'brenda': self.config.get('brenda', {}),
            'agent': self.config.get('agent', {})
        })

    async def save_config(self, request: web.Request) -> web.Response:
        """Save config field to YAML file."""
        try:
            data = await request.json()
            field_path = data.get('field', '')
            value = data.get('value', '')

            # Parse field path (e.g., "llm.provider", "brenda.model", "recipes.callie.llm.model")
            parts = field_path.split('.')

            # Handle recipe saves (save to individual recipe file)
            if parts[0] == 'recipes':
                import yaml
                agent_name = parts[1]
                recipe_path = Path(f'recipes/{agent_name}.yaml')

                with open(recipe_path, 'r') as f:
                    recipe = yaml.safe_load(f)

                # Navigate to field (e.g., llm.model)
                current = recipe
                for part in parts[2:-1]:  # Skip 'recipes' and agent_name
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                current[parts[-1]] = value

                with open(recipe_path, 'w') as f:
                    yaml.dump(recipe, f, default_flow_style=False, sort_keys=False)

                return web.json_response({'success': True, 'message': f'Recipe for {agent_name} saved'})

            # Otherwise save to config.yaml
            import yaml
            config_path = Path('config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Navigate to the field and update it
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value (handle timeout special case - remove 's' suffix)
            final_key = parts[-1]
            if final_key == 'timeout' and value.endswith('s'):
                current[final_key] = int(value[:-1])
            else:
                current[final_key] = value.lower() if final_key == 'provider' else value

            # Save back to YAML
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Reload config in memory
            self.config = config

            return web.json_response({'success': True, 'message': 'Config saved'})

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_agents(self, request: web.Request) -> web.Response:
        """Get list of agents and their LLM configurations."""
        if not self.agent_manager:
            return web.json_response({'agents': []})

        agents_data = []
        for agent_id, agent in self.agent_manager.agents.items():
            # Get location from agent's current_room attribute
            location = getattr(agent, 'current_room', None)

            # Get description and personality from world.agents if available
            description = None
            personality_traits = None
            if hasattr(self.agent_manager, 'world') and self.agent_manager.world:
                if agent_id in self.agent_manager.world.agents:
                    description = self.agent_manager.world.agents[agent_id].get('description')
                    personality_traits = self.agent_manager.world.agents[agent_id].get('personality_traits')

            agents_data.append({
                'id': agent_id,
                'name': agent.agent_name,
                'species': agent.species,
                'description': description,
                'personality_traits': personality_traits or agent.personality,  # Use world data or agent object
                'llm_provider': agent.llm_provider,
                'llm_model': agent.llm_model,
                'current_room': location,
                'location': location  # Alias for compatibility
            })

        return web.json_response({'agents': agents_data})

    async def get_agent_state(self, request: web.Request) -> web.Response:
        """Get live state for a specific agent (for Inspector panel)."""
        agent_id = request.match_info['agent_id']

        if not self.agent_manager or agent_id not in self.agent_manager.agents:
            return web.json_response({'error': 'Agent not found'}, status=404)

        agent = self.agent_manager.agents[agent_id]

        # Get current phenomenal state
        state = agent.get_phenomenal_state()

        # Build affect vector from phenomenal state (first 5 dimensions)
        phenomenal = state.get('phenomenal_state', [])
        if len(phenomenal) >= 5:
            affect = {
                'valence': float(phenomenal[0]),
                'arousal': float(phenomenal[1]),
                'fear': float(phenomenal[2]),
                'sorrow': float(phenomenal[3]),
                'boredom': float(phenomenal[4])
            }
        else:
            affect = {
                'valence': 0.0,
                'arousal': 0.0,
                'fear': 0.0,
                'sorrow': 0.0,
                'boredom': 0.0
            }

        return web.json_response({
            'agent_id': agent_id,
            'affect': affect,
            'phenomenal_state': [float(x) for x in phenomenal],
            'surprise': float(state.get('surprise', 0.0)),
            'fast_state': [float(x) for x in state.get('fast_state', [])] if state.get('fast_state') is not None else [],
            'medium_state': [float(x) for x in state.get('medium_state', [])] if state.get('medium_state') is not None else [],
            'slow_state': [float(x) for x in state.get('slow_state', [])] if state.get('slow_state') is not None else []
        })

    async def update_agent(self, request: web.Request) -> web.Response:
        """Update agent properties from Inspector panel."""
        agent_id = request.match_info['agent_id']

        try:
            data = await request.json()
        except:
            return web.json_response({'error': 'Invalid JSON'}, status=400)

        # Load agents.json
        import json
        from pathlib import Path
        agents_path = Path('world/agents.json')

        try:
            with open(agents_path, 'r') as f:
                agents = json.load(f)
        except:
            return web.json_response({'error': 'Could not load agents'}, status=500)

        if agent_id not in agents:
            return web.json_response({'error': 'Agent not found'}, status=404)

        agent = agents[agent_id]

        # Update fields
        if 'name' in data:
            agent['name'] = data['name']
        if 'species' in data:
            agent['species'] = data['species']
        if 'description' in data:
            agent['description'] = data['description']

        # Update LLM config
        if 'llm_provider' in data or 'llm_model' in data:
            if 'config' not in agent:
                agent['config'] = {}
            if 'llm_override' not in agent['config']:
                agent['config']['llm_override'] = {}

            if 'llm_provider' in data:
                agent['config']['llm_override']['provider'] = data['llm_provider']
            if 'llm_model' in data:
                agent['config']['llm_override']['model'] = data['llm_model']

        # Update personality traits
        if 'personality' in data:
            if 'personality_traits' not in agent:
                agent['personality_traits'] = {}
            agent['personality_traits'].update(data['personality'])

        # Save back to agents.json
        with open(agents_path, 'w') as f:
            json.dump(agents, f, indent=2)

        # Update in-memory agent data (if agent is loaded)
        if self.agent_manager and agent_id in self.agent_manager.agents:
            agent_obj = self.agent_manager.agents[agent_id]
            # Update agent metadata
            if 'name' in data:
                agent_obj.agent_name = data['name']
            if 'species' in data:
                agent_obj.species = data['species']
            if 'description' in data:
                # Set agent object description attribute (for 'look' command)
                agent_obj.agent_description = data['description']
            if 'personality' in data:
                # Update agent object personality dict
                agent_obj.personality.update(data['personality'])
            # Also update world.agents dict
            if hasattr(self.agent_manager, 'world') and self.agent_manager.world:
                if agent_id in self.agent_manager.world.agents:
                    if 'description' in data:
                        self.agent_manager.world.agents[agent_id]['description'] = data['description']
                    if 'personality' in data:
                        if 'personality_traits' not in self.agent_manager.world.agents[agent_id]:
                            self.agent_manager.world.agents[agent_id]['personality_traits'] = {}
                        self.agent_manager.world.agents[agent_id]['personality_traits'].update(data['personality'])
            logger.info(f"Updated {agent_id} in-memory agent data")

        return web.json_response({'success': True, 'message': 'Agent updated'})

    async def delete_agent(self, request: web.Request) -> web.Response:
        """Delete/derez an agent (Studio operation - no auth required)."""
        agent_id = request.match_info['agent_id']

        # Remove from running agent manager
        if self.agent_manager and agent_id in self.agent_manager.agents:
            await self.agent_manager.remove_agent(agent_id, delete_state=False)

            # CRITICAL: Also remove from world's in-memory agents dict
            if hasattr(self.agent_manager, 'world') and self.agent_manager.world:
                if agent_id in self.agent_manager.world.agents:
                    del self.agent_manager.world.agents[agent_id]
                    logger.info(f"Removed {agent_id} from world.agents in-memory dict")

        # Remove from agents.json file
        import json
        from pathlib import Path
        agents_path = Path('world/agents.json')

        try:
            with open(agents_path, 'r') as f:
                agents = json.load(f)

            if agent_id in agents:
                del agents[agent_id]
                with open(agents_path, 'w') as f:
                    json.dump(agents, f, indent=2)

            return web.json_response({'success': True, 'message': f'Agent {agent_id} derezzed'})
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def create_agent(self, request: web.Request) -> web.Response:
        """Create a new agent/noodling (Studio operation - no auth required)."""
        try:
            data = await request.json()
            agent_name = data.get('name', 'NewNoodling')
            species = data.get('species', 'unknown')
            pronouns = data.get('pronouns', 'they/them')

            # Generate agent ID
            agent_id = f"agent_{agent_name.lower().replace(' ', '_')}"

            # Create minimal agent structure
            new_agent = {
                'name': agent_name,
                'species': species,
                'pronouns': pronouns,
                'location': 'pond',  # Default starting location
                'description': f'A newly rezzed {species}.',
                'personality_traits': {
                    'openness': 0.5,
                    'conscientiousness': 0.5,
                    'extraversion': 0.5,
                    'agreeableness': 0.5,
                    'neuroticism': 0.5
                }
            }

            # Add to agents.json
            import json
            from pathlib import Path
            agents_path = Path('world/agents.json')

            agents = {}
            if agents_path.exists():
                with open(agents_path, 'r') as f:
                    agents = json.load(f)

            agents[agent_id] = new_agent

            with open(agents_path, 'w') as f:
                json.dump(agents, f, indent=2)

            # Create agent directory and checkpoint path
            import os
            agent_dir = f'world/agents/{agent_id}'
            os.makedirs(agent_dir, exist_ok=True)

            # Create subdirectories for agent state
            for subdir in ['data', 'inbox', 'outbox', 'memories', 'history']:
                os.makedirs(os.path.join(agent_dir, subdir), exist_ok=True)

            # Set checkpoint path (file may not exist - agent will init with random weights)
            checkpoint_path = f'{agent_dir}/checkpoint.npz'
            new_agent['checkpoint_path'] = checkpoint_path
            new_agent['current_room'] = 'room_000'  # Default starting room

            # Add to world's in-memory dict
            if self.agent_manager and hasattr(self.agent_manager, 'world') and self.agent_manager.world:
                self.agent_manager.world.agents[agent_id] = new_agent
                logger.info(f"Added {agent_id} to world.agents in-memory dict")

                # Add agent to room occupants
                room = self.agent_manager.world.get_room('room_000')
                if room:
                    if 'occupants' not in room:
                        room['occupants'] = []
                    if agent_id not in room['occupants']:
                        room['occupants'].append(agent_id)
                        logger.info(f"Added {agent_id} to room_000 occupants")

            # Save updated agents.json with checkpoint_path
            with open(agents_path, 'w') as f:
                json.dump(agents, f, indent=2)

            # Spawn in agent_manager using create_agent
            if self.agent_manager:
                try:
                    await self.agent_manager.create_agent(
                        agent_id=agent_id,
                        checkpoint_path=checkpoint_path,
                        spawn_room='room_000',
                        agent_name=agent_name,
                        agent_description=new_agent['description']  # Pass description from JSON
                    )
                    logger.info(f"Spawned agent {agent_id} in world")
                except Exception as spawn_error:
                    logger.error(f"Failed to spawn agent {agent_id}: {spawn_error}", exc_info=True)
                    print(f"[ERROR] Failed to spawn agent {agent_id}: {spawn_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue anyway - agent was saved to JSON
            else:
                logger.warning(f"No agent_manager available to spawn {agent_id}")

            return web.json_response({'success': True, 'agent_id': agent_id, 'message': f'Agent {agent_name} created'})
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_agent_components(self, request: web.Request) -> web.Response:
        """
        Get all cognitive components for an agent.

        Returns list of components with metadata for Inspector display.
        """
        agent_id = request.match_info['agent_id']

        if not self.agent_manager or agent_id not in self.agent_manager.agents:
            return web.json_response({'error': 'Agent not found'}, status=404)

        agent = self.agent_manager.agents[agent_id]

        # Get component registry from agent
        if not hasattr(agent, 'components'):
            return web.json_response({'error': 'Agent does not have component system'}, status=500)

        components_data = agent.components.to_dict()
        return web.json_response(components_data)

    async def get_component(self, request: web.Request) -> web.Response:
        """
        Get detailed information about a specific component.

        Returns component prompt template, parameters, and metadata.
        """
        agent_id = request.match_info['agent_id']
        component_id = request.match_info['component_id']

        if not self.agent_manager or agent_id not in self.agent_manager.agents:
            return web.json_response({'error': 'Agent not found'}, status=404)

        agent = self.agent_manager.agents[agent_id]

        if not hasattr(agent, 'components'):
            return web.json_response({'error': 'Agent does not have component system'}, status=500)

        component = agent.components.get_component(component_id)
        if not component:
            return web.json_response({'error': f'Component {component_id} not found'}, status=404)

        return web.json_response(component.to_dict())

    async def update_component(self, request: web.Request) -> web.Response:
        """
        Update component parameters (called from Inspector).

        Allows hot-reloading of prompts and parameters without restart.
        """
        agent_id = request.match_info['agent_id']
        component_id = request.match_info['component_id']

        if not self.agent_manager or agent_id not in self.agent_manager.agents:
            return web.json_response({'error': 'Agent not found'}, status=404)

        agent = self.agent_manager.agents[agent_id]

        if not hasattr(agent, 'components'):
            return web.json_response({'error': 'Agent does not have component system'}, status=500)

        component = agent.components.get_component(component_id)
        if not component:
            return web.json_response({'error': f'Component {component_id} not found'}, status=404)

        try:
            data = await request.json()
            parameters = data.get('parameters', {})

            # Update component parameters
            component.update_parameters(parameters)

            logger.info(f"Updated component {component_id} for {agent_id}: {parameters}")

            return web.json_response({
                'success': True,
                'message': f'Component {component_id} updated',
                'component': component.to_dict()
            })

        except Exception as e:
            logger.error(f"Error updating component {component_id}: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def create_object(self, request: web.Request) -> web.Response:
        """Create a new prim/object (Studio operation - no auth required)."""
        try:
            data = await request.json()
            obj_name = data.get('name', 'NewPrim')
            location = data.get('location', 'room_000')  # Use passed location or default to Nexus

            # Generate object ID
            obj_id = f"obj_{obj_name.lower().replace(' ', '_')}"

            # Create minimal object structure
            new_obj = {
                'name': obj_name,
                'description': 'A newly created object.',
                'location': location,
                'properties': {}
            }

            # Add to objects.json
            import json
            from pathlib import Path
            objects_path = Path('world/objects.json')

            objects = {}
            if objects_path.exists():
                with open(objects_path, 'r') as f:
                    objects = json.load(f)

            objects[obj_id] = new_obj

            with open(objects_path, 'w') as f:
                json.dump(objects, f, indent=2)

            # Add to world's in-memory dict
            if self.agent_manager and hasattr(self.agent_manager, 'world') and self.agent_manager.world:
                self.agent_manager.world.objects[obj_id] = new_obj
                logger.info(f"Added {obj_id} to world.objects in-memory dict")

                # Add to room's objects array (bidirectional link)
                if location in self.agent_manager.world.rooms:
                    if 'objects' not in self.agent_manager.world.rooms[location]:
                        self.agent_manager.world.rooms[location]['objects'] = []
                    if obj_id not in self.agent_manager.world.rooms[location]['objects']:
                        self.agent_manager.world.rooms[location]['objects'].append(obj_id)
                        # Persist room update to file
                        rooms_path = Path('world/rooms.json')
                        with open(rooms_path, 'r') as f:
                            rooms = json.load(f)
                        if location in rooms:
                            if 'objects' not in rooms[location]:
                                rooms[location]['objects'] = []
                            rooms[location]['objects'].append(obj_id)
                            with open(rooms_path, 'w') as f:
                                json.dump(rooms, f, indent=2)
                        logger.info(f"Added {obj_id} to room {location} objects array")

            return web.json_response({'success': True, 'object_id': obj_id, 'message': f'Object {obj_name} created'})
        except Exception as e:
            logger.error(f"Error creating object: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def update_object(self, request: web.Request) -> web.Response:
        """Update object/prim properties (Studio operation - no auth required)."""
        try:
            object_id = request.match_info['object_id']
            data = await request.json()

            # Load objects.json
            import json
            from pathlib import Path
            objects_path = Path('world/objects.json')

            if not objects_path.exists():
                return web.json_response({'error': 'objects.json not found'}, status=404)

            with open(objects_path, 'r') as f:
                objects = json.load(f)

            if object_id not in objects:
                return web.json_response({'error': f'Object {object_id} not found'}, status=404)

            # Update fields
            if 'name' in data:
                objects[object_id]['name'] = data['name']
            if 'description' in data:
                objects[object_id]['description'] = data['description']

            # Save to file
            with open(objects_path, 'w') as f:
                json.dump(objects, f, indent=2)

            # Update in-memory dict
            if self.agent_manager and hasattr(self.agent_manager, 'world') and self.agent_manager.world:
                if object_id in self.agent_manager.world.objects:
                    if 'name' in data:
                        self.agent_manager.world.objects[object_id]['name'] = data['name']
                    if 'description' in data:
                        self.agent_manager.world.objects[object_id]['description'] = data['description']
                    logger.info(f"Updated {object_id} in world.objects in-memory dict")

            return web.json_response({'success': True, 'message': f'Object {object_id} updated'})
        except Exception as e:
            logger.error(f"Error updating object: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def create_room(self, request: web.Request) -> web.Response:
        """Create a new room (Studio operation - no auth required)."""
        try:
            data = await request.json()
            room_name = data.get('name', 'NewRoom')

            # Generate room ID
            room_id = room_name.lower().replace(' ', '_')

            # Create minimal room structure
            new_room = {
                'uid': room_id,
                'name': room_name,
                'description': 'A newly created room.',
                'exits': {},
                'occupants': [],
                'objects': [],
                'owner': 'user_admin',
                'created': '2025-11-19T00:00:00'
            }

            # Add to rooms.json
            import json
            from pathlib import Path
            rooms_path = Path('world/rooms.json')

            rooms = {}
            if rooms_path.exists():
                with open(rooms_path, 'r') as f:
                    rooms = json.load(f)

            rooms[room_id] = new_room

            with open(rooms_path, 'w') as f:
                json.dump(rooms, f, indent=2)

            # Add to world's in-memory dict
            if self.agent_manager and hasattr(self.agent_manager, 'world') and self.agent_manager.world:
                self.agent_manager.world.rooms[room_id] = new_room
                logger.info(f"Added {room_id} to world.rooms in-memory dict")

            return web.json_response({'success': True, 'room_id': room_id, 'message': f'Room {room_name} created'})
        except Exception as e:
            logger.error(f"Error creating room: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def update_room(self, request: web.Request) -> web.Response:
        """Update room properties (Studio operation - no auth required)."""
        try:
            room_id = request.match_info['room_id']
            data = await request.json()

            # Load rooms.json
            import json
            from pathlib import Path
            rooms_path = Path('world/rooms.json')

            if not rooms_path.exists():
                return web.json_response({'error': 'rooms.json not found'}, status=404)

            with open(rooms_path, 'r') as f:
                rooms = json.load(f)

            if room_id not in rooms:
                return web.json_response({'error': f'Room {room_id} not found'}, status=404)

            # Update fields (only description for now, can extend later)
            if 'description' in data:
                rooms[room_id]['description'] = data['description']

            # Save to file
            with open(rooms_path, 'w') as f:
                json.dump(rooms, f, indent=2)

            # Update in-memory dict
            if self.agent_manager and hasattr(self.agent_manager, 'world') and self.agent_manager.world:
                if room_id in self.agent_manager.world.rooms:
                    if 'description' in data:
                        self.agent_manager.world.rooms[room_id]['description'] = data['description']
                    logger.info(f"Updated {room_id} in world.rooms in-memory dict")

            return web.json_response({'success': True, 'message': f'Room {room_id} updated'})
        except Exception as e:
            logger.error(f"Error updating room: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            'status': 'ok',
            'profiler_active': self.session_profiler is not None,
            'kimmie_active': self.kimmie is not None
        })

    async def shutdown_server(self, request: web.Request) -> web.Response:
        """
        Gracefully shutdown the server.

        POST /api/shutdown
        Body: {"delay": 5}  # optional, defaults to 5 seconds

        Returns:
            {"success": true, "message": "Shutdown initiated"}
        """
        if not self.server:
            return web.json_response({
                'success': False,
                'error': 'Server instance not available'
            }, status=500)

        try:
            data = await request.json()
            delay = data.get('delay', 5)
        except Exception:
            delay = 5

        # Trigger graceful shutdown
        import asyncio
        asyncio.create_task(self.server.graceful_shutdown(delay))

        return web.json_response({
            'success': True,
            'message': f'Server shutdown initiated. Shutting down in {delay} seconds.'
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
