"""
WebSocket Server for cMUSH

Handles:
- WebSocket connections
- Authentication
- Command routing
- Event broadcasting
- Agent lifecycle management

Author: cMUSH Project
Date: October 2025
"""

import asyncio
import websockets
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Set
import sys
import os

# Add consilience_core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from world import World
from auth import AuthManager
from commands import CommandParser
from agent_bridge import AgentManager
from llm_interface import OpenAICompatibleLLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/cmush_{datetime.now().strftime("%Y-%m-%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class CMUSHServer:
    """
    cMUSH WebSocket server.

    Manages connections, routes commands, broadcasts events,
    and coordinates between users and Consilience agents.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize cMUSH server.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_path = config_path  # Store for later use
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info("cMUSH Server starting...")
        logger.info(f"Config loaded from {config_path}")

        # Initialize world
        world_dir = self.config['paths']['world_dir']
        self.world = World(world_dir=world_dir)

        # Initialize auth
        self.auth = AuthManager(self.world)

        # Initialize LLM (will be created in async context)
        self.llm = None

        # Initialize agent manager (after LLM)
        self.agent_manager = None

        # Initialize command parser (after agent manager)
        self.command_parser = None

        # Active connections: websocket -> user_id
        self.connections: Dict = {}

        # Auto-save timer
        self.save_interval = self.config['world'].get('auto_save_interval', 300)
        self.save_task = None

        # Autonomous event polling
        self.autonomous_poll_interval = self.config.get('agent', {}).get('autonomous_poll_interval', 10)
        self.autonomous_poll_task = None

    async def initialize_async_components(self):
        """Initialize async components (LLM, agents)."""
        # Initialize LLM
        llm_config = self.config['llm']
        self.llm = OpenAICompatibleLLM(
            api_base=llm_config['api_base'],
            api_key=llm_config.get('api_key', 'not-needed'),
            model=llm_config['model'],
            timeout=llm_config.get('timeout', 30),
            max_concurrent=5,  # Match number of LMStudio instances
            use_model_instances=True  # Enable model:N pattern for parallel inference
        )
        await self.llm.__aenter__()

        # Initialize agent manager (pass global config for personality traits)
        self.agent_manager = AgentManager(self.llm, self.world, global_config=self.config)

        # Load existing agents
        await self.load_agents()

        # Initialize command parser (with config for persistence)
        self.command_parser = CommandParser(
            self.world,
            self.agent_manager,
            server=self,
            config=self.config,
            config_path=self.config_path
        )

        logger.info("Async components initialized")

    async def load_agents(self):
        """Load all agents from world state."""
        for agent_id, agent_data in self.world.get_all_agents().items():
            checkpoint_path = agent_data['checkpoint_path']
            current_room = agent_data['current_room']
            config = agent_data.get('config', {})

            # Phase 6: Inject self-monitoring config from global config.yaml
            # This ensures saved agents get the latest self-monitoring settings
            config['self_monitoring'] = self.config['agent'].get('self_monitoring', {})
            print(f"[DEBUG LOAD] agent_id={agent_id}, injecting self_monitoring config: {config['self_monitoring']}", flush=True)

            try:
                await self.agent_manager.create_agent(
                    agent_id=agent_id,
                    checkpoint_path=checkpoint_path,
                    spawn_room=current_room,
                    config=config
                )
                logger.info(f"Loaded agent: {agent_id}")
            except Exception as e:
                logger.error(f"Error loading agent {agent_id}: {e}")

    async def handle_connection(self, websocket, path=None):
        """
        Handle WebSocket connection.

        Args:
            websocket: WebSocket connection
            path: Request path
        """
        user_id = None
        session_token = None

        try:
            logger.info(f"New connection from {websocket.remote_address}")

            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')

                    # Authentication messages
                    if msg_type == 'register':
                        response = await self.handle_register(data)
                        await websocket.send(json.dumps(response))

                    elif msg_type == 'login':
                        response = await self.handle_login(data)
                        if response['success']:
                            user_id = response['user_id']
                            session_token = response['session_token']
                            self.connections[websocket] = user_id

                            # Send welcome message
                            await self.send_to_user(websocket, {
                                'type': 'system',
                                'text': f"Welcome, {data['username']}!"
                            })

                            # Generate enter event so agents notice the user's arrival
                            user = self.world.get_user(user_id)
                            room = self.world.get_user_room(user_id)
                            if user and room:
                                username = user.get('username', user_id)
                                description = user.get('description', '')

                                # Create enter event with description
                                enter_text = f"{username} appears"
                                if description:
                                    enter_text += f". {description}"
                                else:
                                    enter_text += "."

                                enter_event = {
                                    'type': 'enter',
                                    'user': user_id,
                                    'room': room['uid'],
                                    'text': enter_text
                                }

                                # Broadcast to other users immediately
                                await self.broadcast_event(enter_event)

                                # Let agents perceive the arrival (non-blocking)
                                # User shouldn't wait for agents to think/respond
                                asyncio.create_task(self._handle_agent_entrance(enter_event))

                            # Show current room
                            look_result = await self.command_parser.cmd_look(user_id, '')
                            await self.send_to_user(websocket, {
                                'type': 'output',
                                'text': look_result['output']
                            })

                        await websocket.send(json.dumps(response))

                    # Command messages (require authentication)
                    elif msg_type == 'command':
                        if websocket not in self.connections:
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'text': 'Not authenticated'
                            }))
                            continue

                        user_id = self.connections[websocket]
                        command_text = data.get('command', '')

                        # Execute command
                        result = await self.command_parser.parse_and_execute(
                            user_id=user_id,
                            command_text=command_text
                        )

                        # Send output to user
                        if result['output']:
                            await self.send_to_user(websocket, {
                                'type': 'output',
                                'text': result['output']
                            })

                        # Broadcast events
                        for event in result.get('events', []):
                            await self.broadcast_event(event)

                            # Handle special yeet event - forcibly disconnect user
                            if event['type'] == 'yeet':
                                target_id = event['user']
                                # Find websocket for target user
                                target_ws = None
                                for ws, ws_user_id in self.connections.items():
                                    if ws_user_id == target_id:
                                        target_ws = ws
                                        break

                                if target_ws:
                                    # Send goodbye message
                                    await self.send_to_user(target_ws, {
                                        'type': 'output',
                                        'text': '\n👋 You have been disconnected by an administrator.\n'
                                    })
                                    # Close connection
                                    await target_ws.close(code=1000, reason="Disconnected by admin")
                                    logger.info(f"User {target_id} was yeeted by {user_id}")

                            # Let agents perceive the event (including enter for spawns, exit for following, think for ruminations)
                            if event['type'] in ['say', 'emote', 'enter', 'exit', 'think']:
                                agent_responses = await self.agent_manager.broadcast_event(event)

                                # Broadcast agent responses
                                for agent_response in agent_responses:
                                    agent_id = agent_response['agent_id']
                                    agent_data = self.world.get_user(agent_id)

                                    # Get agent name from agent manager (respects @setname changes)
                                    agent_obj = self.agent_manager.get_agent(agent_id)
                                    agent_name = agent_obj.agent_name if agent_obj else agent_data.get('name', agent_id)

                                    # Handle follow command - move the agent
                                    if agent_response['command'] == 'follow':
                                        direction = agent_response.get('direction', 'north')
                                        current_room = self.world.get_room(agent_data['current_room'])

                                        if current_room and direction in current_room['exits']:
                                            new_room_id = current_room['exits'][direction]
                                            # Move agent
                                            self.world.move_user(agent_id, new_room_id)

                                            # Broadcast exit event
                                            exit_event = {
                                                'type': 'exit',
                                                'user': agent_id,
                                                'username': agent_name,
                                                'room': agent_data['current_room'],
                                                'text': f"{agent_name} {agent_response['text']}"
                                            }
                                            await self.broadcast_event(exit_event)

                                            # Broadcast enter event in new room
                                            enter_event = {
                                                'type': 'enter',
                                                'user': agent_id,
                                                'username': agent_name,
                                                'room': new_room_id,
                                                'text': f"{agent_name} arrives."
                                            }
                                            await self.broadcast_event(enter_event)
                                            continue  # Don't broadcast as normal agent response

                                    # Create event for agent response (say/emote)
                                    agent_event = {
                                        'type': agent_response['command'],
                                        'user': agent_id,
                                        'username': agent_name,
                                        'room': agent_data['current_room'],
                                        'text': agent_response['text']
                                    }

                                    # Broadcast to websocket clients (humans)
                                    await self.broadcast_event(agent_event)

                                    # Let OTHER agents perceive this agent's response
                                    other_agent_responses = await self.agent_manager.broadcast_event(agent_event)

                                    # If other agents respond to this agent, broadcast those too
                                    for other_response in other_agent_responses:
                                        other_agent_id = other_response['agent_id']
                                        other_agent_data = self.world.get_user(other_agent_id)
                                        other_agent_obj = self.agent_manager.get_agent(other_agent_id)
                                        other_agent_name = other_agent_obj.agent_name if other_agent_obj else other_agent_data.get('name', other_agent_id)

                                        other_agent_event = {
                                            'type': other_response['command'],
                                            'user': other_agent_id,
                                            'username': other_agent_name,
                                            'room': other_agent_data['current_room'],
                                            'text': other_response['text']
                                        }

                                        await self.broadcast_event(other_agent_event)

                    # Ping/pong for keepalive
                    elif msg_type == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {websocket.remote_address}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'text': f'Error: {str(e)}'
                    }))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {websocket.remote_address}")
        finally:
            # Clean up connection
            if websocket in self.connections:
                user_id = self.connections[websocket]
                del self.connections[websocket]
                logger.info(f"User disconnected: {user_id}")

            if session_token:
                self.auth.end_session(session_token)

    async def handle_register(self, data: Dict) -> Dict:
        """
        Handle user registration.

        Args:
            data: Registration data

        Returns:
            Response dict
        """
        username = data.get('username', '')
        password = data.get('password', '')

        success, message = self.auth.create_user(username, password)

        return {
            'type': 'register_response',
            'success': success,
            'message': message
        }

    async def handle_login(self, data: Dict) -> Dict:
        """
        Handle user login.

        Args:
            data: Login data

        Returns:
            Response dict
        """
        username = data.get('username', '')
        password = data.get('password', '')

        success, user_id, message = self.auth.authenticate(username, password)

        if success:
            session_token = self.auth.create_session(user_id)

            return {
                'type': 'login_response',
                'success': True,
                'user_id': user_id,
                'session_token': session_token,
                'message': message
            }
        else:
            return {
                'type': 'login_response',
                'success': False,
                'message': message
            }

    async def send_to_user(self, websocket, message: Dict):
        """
        Send message to specific user.

        Args:
            websocket: Target websocket
            message: Message dict
        """
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending to user: {e}")

    async def _handle_agent_entrance(self, enter_event: Dict):
        """
        Handle agent perception of user entrance (async background task).

        This runs in the background so login isn't blocked by agent LLM calls.

        Args:
            enter_event: The enter event to broadcast to agents
        """
        try:
            # Let agents perceive the arrival
            agent_responses = await self.agent_manager.broadcast_event(enter_event)

            # Broadcast any agent responses
            for agent_response in agent_responses:
                agent_id = agent_response['agent_id']
                agent_data = self.world.get_user(agent_id)
                agent_obj = self.agent_manager.get_agent(agent_id)
                agent_name = agent_obj.agent_name if agent_obj else agent_data.get('name', agent_id)

                if agent_response['command'] == 'say':
                    agent_event = {
                        'type': 'say',
                        'user': agent_id,
                        'username': agent_name,
                        'room': agent_data['current_room'],
                        'text': agent_response['text']
                    }
                    await self.broadcast_event(agent_event)
                elif agent_response['command'] == 'emote':
                    agent_event = {
                        'type': 'emote',
                        'user': agent_id,
                        'username': agent_name,
                        'room': agent_data['current_room'],
                        'text': agent_response['text']
                    }
                    await self.broadcast_event(agent_event)
        except Exception as e:
            logger.error(f"Error handling agent entrance: {e}", exc_info=True)

    async def broadcast_event(self, event: Dict):
        """
        Broadcast event to all users in the same room.

        Args:
            event: Event to broadcast
        """
        room_id = event.get('room')
        if not room_id:
            return

        event_type = event.get('type')
        user_id = event.get('user')
        username = event.get('username', user_id)
        text = event.get('text', '')

        # Format message based on event type
        if event_type == 'say':
            formatted_text = f'{username} says, "{text}"'
        elif event_type == 'emote':
            formatted_text = f'{username} {text}'
        elif event_type == 'think':
            formatted_text = f'{username} thinks, {text}'
        elif event_type == 'thought':
            # Autonomous cognition ruminations (strikethrough in client)
            formatted_text = f'{username} thinks, {text}'
        elif event_type == 'enter':
            formatted_text = text
        elif event_type == 'exit':
            formatted_text = text
        else:
            formatted_text = text

        # Find users in room
        room_occupants = self.world.get_room_occupants(room_id)

        # Send to connected users
        for ws, connected_user_id in self.connections.items():
            if connected_user_id in room_occupants:
                # Don't echo back to sender for say/emote/think/thought
                if event_type in ['say', 'emote', 'think', 'thought'] and connected_user_id == user_id:
                    continue

                await self.send_to_user(ws, {
                    'type': 'event',
                    'event_type': event_type,
                    'text': formatted_text
                })

    async def auto_save_loop(self):
        """Periodically save world and agent state."""
        while True:
            await asyncio.sleep(self.save_interval)
            logger.info("Auto-saving world and agent states...")
            self.world.save_all()
            await self.agent_manager.save_all_agents()
            logger.info("Auto-save complete")

    async def autonomous_event_loop(self):
        """Periodically check for and broadcast autonomous agent events."""
        while True:
            await asyncio.sleep(self.autonomous_poll_interval)

            try:
                # Check for autonomous events
                events = await self.agent_manager.check_autonomous_events()

                # Broadcast each event
                for event in events:
                    await self.broadcast_event(event)
                    logger.debug(f"Broadcast autonomous event from {event.get('user')}")

            except Exception as e:
                logger.error(f"Error in autonomous event loop: {e}", exc_info=True)

    async def start(self):
        """Start the cMUSH server."""
        # Initialize async components
        await self.initialize_async_components()

        # Start auto-save task
        self.save_task = asyncio.create_task(self.auto_save_loop())

        # Start autonomous event polling task
        self.autonomous_poll_task = asyncio.create_task(self.autonomous_event_loop())

        # Start WebSocket server
        host = self.config['server']['host']
        port = self.config['server']['port']

        logger.info(f"Starting WebSocket server on {host}:{port}")

        async with websockets.serve(self.handle_connection, host, port):
            logger.info("cMUSH server ready!")
            logger.info(f"World: {self.world.get_stats()}")
            logger.info(f"Agents: {len(self.agent_manager.agents)}")
            await asyncio.Future()  # Run forever

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down cMUSH server...")

        # Cancel background tasks
        if self.save_task:
            self.save_task.cancel()
        if self.autonomous_poll_task:
            self.autonomous_poll_task.cancel()

        # Save everything (stop cognition on shutdown)
        self.world.save_all()
        await self.agent_manager.save_all_agents(stop_cognition=True)

        # Close LLM session
        if self.llm:
            await self.llm.close()

        logger.info("Shutdown complete")


async def main():
    """Main entry point."""
    # Ensure directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('world', exist_ok=True)

    # Create server
    server = CMUSHServer(config_path='config.yaml')

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
