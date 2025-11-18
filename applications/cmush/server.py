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
from typing import Dict, Set, List
from pathlib import Path
import sys
import os

# Add consilience_core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from world import World
from auth import AuthManager
from commands import CommandParser
from agent_bridge import AgentManager
from llm_interface import OpenAICompatibleLLM
from session_profiler import SessionProfiler
from kimmie_character import KimmieCharacter
from api_server import NoodleScopeAPI
from recipe_loader import RecipeLoader

# Setup logging
os.makedirs('logs', exist_ok=True)
log_filename = f'logs/cmush_{datetime.now().strftime("%Y-%m-%d")}.log'

# Configure logging with explicit file handler
file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'))

# Get root logger and configure
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized - writing to {log_filename}")


class WebSocketLogHandler(logging.Handler):
    """
    Custom logging handler that broadcasts log messages to subscribed WebSocket clients.
    """

    def __init__(self, server=None):
        super().__init__()
        self.server = server
        self.log_buffer = []  # Recent logs for new subscribers
        self.max_buffer = 100  # Keep last 100 log entries

    def emit(self, record):
        """Emit a log record to subscribed WebSocket clients."""
        try:
            log_entry = self.format(record)

            # Add to buffer
            self.log_buffer.append({
                'level': record.levelname,
                'name': record.name,
                'message': record.getMessage(),
                'timestamp': record.created
            })

            # Trim buffer if needed
            if len(self.log_buffer) > self.max_buffer:
                self.log_buffer = self.log_buffer[-self.max_buffer:]

            # Broadcast to subscribed clients (if server is set)
            if self.server:
                asyncio.create_task(self.server.broadcast_log({
                    'type': 'log',
                    'level': record.levelname,
                    'name': record.name,
                    'message': record.getMessage(),
                    'timestamp': record.created
                }))
        except Exception:
            self.handleError(record)


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

        # Initialize recipe loader (for reloading species on agent load)
        self.recipe_loader = RecipeLoader("recipes")

        # Initialize LLM (will be created in async context)
        self.llm = None

        # Initialize agent manager (after LLM)
        self.agent_manager = None

        # Initialize command parser (after agent manager)
        self.command_parser = None

        # Active connections: websocket -> user_id
        self.connections: Dict = {}

        # Log subscribers: websockets that want to receive log streams
        self.log_subscribers: Set = set()

        # Chat history for session continuity
        self.chat_history = []
        self.history_file = Path('world/chat_history.json')
        self.max_history = 200  # Keep last 200 messages
        self._load_chat_history()

        # Setup WebSocket log handler
        self.ws_log_handler = WebSocketLogHandler(server=self)
        self.ws_log_handler.setLevel(logging.INFO)
        self.ws_log_handler.setFormatter(logging.Formatter('[%(levelname)s] [%(name)s] %(message)s'))
        root_logger.addHandler(self.ws_log_handler)

        # Auto-save timer
        self.save_interval = self.config['world'].get('auto_save_interval', 300)
        self.save_task = None

        # Autonomous event polling
        self.autonomous_poll_interval = self.config.get('agent', {}).get('autonomous_poll_interval', 10)
        self.autonomous_poll_task = None

        # NoodleScope 2.0 components
        self.session_profiler = None
        self.kimmie = None
        self.api_server = None
        self.api_runner = None

    async def initialize_async_components(self):
        """Initialize async components (LLM, agents)."""
        # Initialize LLM with provider switching
        llm_config = self.config['llm']

        # Determine which provider to use
        provider = llm_config.get('provider', 'local')
        logger.info(f"LLM provider: {provider}")

        # Get provider-specific config
        if provider == 'openrouter':
            provider_config = llm_config.get('openrouter', {})
            logger.info(f"🌐 Using OpenRouter with model: {provider_config.get('model')}")
        else:  # default to 'local'
            provider_config = llm_config.get('local', llm_config)  # Fallback to root llm config for backward compat
            logger.info(f"💻 Using local LMStudio with model: {provider_config.get('model')}")

        self.llm = OpenAICompatibleLLM(
            api_base=provider_config.get('api_base', 'http://localhost:1234/v1'),
            api_key=provider_config.get('api_key', 'not-needed'),
            model=provider_config.get('model', 'qwen/qwen3-4b-2507'),
            timeout=provider_config.get('timeout', 30),
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

        # Initialize NoodleScope 2.0 components
        session_id = f"cmush_session_{int(asyncio.get_event_loop().time())}"
        self.session_profiler = SessionProfiler(session_id=session_id)

        # Initialize @Kimmie character (use provider config from above)
        self.kimmie = KimmieCharacter(
            llm_base_url=provider_config.get('api_base', 'http://localhost:1234/v1'),
            llm_model=provider_config.get('model', 'qwen/qwen3-4b-2507'),
            session_profiler=self.session_profiler
        )

        # Initialize NoodleScope API server
        self.api_server = NoodleScopeAPI(
            session_profiler=self.session_profiler,
            kimmie=self.kimmie,
            config=self.config,  # Pass config for LLM config UI
            agent_manager=self.agent_manager,  # Pass agent manager for agent list
            host='0.0.0.0',
            port=8081
        )

        # Wire profiler into agent manager
        self.agent_manager.set_session_profiler(self.session_profiler)

        logger.info("Async components initialized")
        logger.info(f"Session profiler active: {session_id}")

    def _load_chat_history(self):
        """
        Load chat history from disk.

        Loads the last 200 messages from the history file if it exists.
        """
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                    # Ensure we only keep the last max_history messages
                    if len(self.chat_history) > self.max_history:
                        self.chat_history = self.chat_history[-self.max_history:]
                    logger.info(f"Loaded {len(self.chat_history)} messages from chat history")
            else:
                self.chat_history = []
                logger.info("No existing chat history found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading chat history: {e}", exc_info=True)
            self.chat_history = []

    def _save_chat_history(self):
        """
        Save chat history to disk.

        Saves the last max_history messages to the history file.
        """
        try:
            # Ensure world directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)

            # Keep only the last max_history messages
            history_to_save = self.chat_history[-self.max_history:]

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(history_to_save)} messages to chat history")
        except Exception as e:
            logger.error(f"Error saving chat history: {e}", exc_info=True)

    async def load_agents(self):
        """Load all agents from world state."""
        for agent_id, agent_data in self.world.get_all_agents().items():
            checkpoint_path = agent_data['checkpoint_path']
            current_room = agent_data['current_room']
            config = agent_data.get('config', {})

            # Phase 6: Inject self-monitoring config from global config.yaml
            # This ensures saved agents get the latest self-monitoring settings
            config['self_monitoring'] = self.config['agent'].get('self_monitoring', {})
            logger.debug(f"[LOAD] agent_id={agent_id}, injecting self_monitoring config: {config['self_monitoring']}")

            # Reload recipe to get species and other critical parameters
            # This ensures character voice translation works for loaded agents
            agent_name = agent_data.get('name', agent_id.replace('agent_', ''))
            recipe = self.recipe_loader.load_recipe(agent_name)
            if recipe:
                # Reload species from recipe (critical for character voice!)
                config['species'] = recipe.species

                # Per-agent LLM configuration (if specified in recipe)
                if recipe.llm_provider or recipe.llm_model:
                    config['llm_override'] = {
                        'provider': recipe.llm_provider,
                        'model': recipe.llm_model
                    }
                    logger.info(f"[LOAD] {agent_id} will use custom LLM: {recipe.llm_provider}/{recipe.llm_model}")

                # Also reload identity_prompt if not already in config
                if 'identity_prompt' not in config:
                    config['identity_prompt'] = recipe.identity_prompt
                logger.info(f"[LOAD] Reloaded recipe for {agent_id}: species={recipe.species}")

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

                            # Send chat history first
                            for history_entry in self.chat_history:
                                await self.send_to_user(websocket, {
                                    'type': 'history',
                                    'text': history_entry['text'],
                                    'timestamp': history_entry['timestamp']
                                })

                            # Send welcome message
                            await self.send_to_user(websocket, {
                                'type': 'system',
                                'text': f"Welcome, {data['username']}!"
                            })

                            # Send agent list with enlightenment status
                            agent_list = []
                            for agent_id, agent in self.agent_manager.agents.items():
                                agent_list.append({
                                    'id': agent_id,
                                    'name': agent.agent_name,
                                    'enlightened': agent.config.get('enlightenment', False)
                                })

                            await self.send_to_user(websocket, {
                                'type': 'agents',
                                'agents': agent_list
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

                    # Tab completion request (require authentication)
                    elif msg_type == 'complete':
                        if websocket not in self.connections:
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'text': 'Not authenticated'
                            }))
                            continue

                        user_id = self.connections[websocket]
                        command = data.get('command', '')
                        partial = data.get('partial', '')
                        msg_id = data.get('id')

                        # Get completions
                        matches = await self.get_completions(user_id, command, partial)

                        # Send response
                        await websocket.send(json.dumps({
                            'type': 'completions',
                            'id': msg_id,
                            'matches': matches
                        }))

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

                    # Log subscription
                    elif msg_type == 'subscribe_logs':
                        if websocket not in self.connections:
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'text': 'Not authenticated'
                            }))
                            continue

                        self.log_subscribers.add(websocket)
                        logger.info(f"Client subscribed to logs: {self.connections[websocket]}")

                        # Send recent log buffer
                        for log_entry in self.ws_log_handler.log_buffer:
                            await websocket.send(json.dumps({
                                'type': 'log',
                                **log_entry
                            }))

                        await websocket.send(json.dumps({
                            'type': 'subscribed',
                            'message': 'Log streaming enabled'
                        }))

                    elif msg_type == 'unsubscribe_logs':
                        self.log_subscribers.discard(websocket)
                        logger.info(f"Client unsubscribed from logs: {self.connections.get(websocket, 'unknown')}")

                        await websocket.send(json.dumps({
                            'type': 'unsubscribed',
                            'message': 'Log streaming disabled'
                        }))

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
        metadata = event.get('metadata', {})

        # Extract model name from metadata (for debugging model routing)
        model_used = metadata.get('model_used', '')
        model_suffix = f' [{model_used}]' if model_used else ''

        # Format message based on event type
        if event_type == 'say':
            formatted_text = f'{username} says, "{text}"{model_suffix}'
        elif event_type == 'emote':
            formatted_text = f'{username} {text}{model_suffix}'
        elif event_type == 'think':
            formatted_text = f'{username} thinks, {text}{model_suffix}'
        elif event_type == 'thought':
            # Autonomous cognition ruminations (strikethrough in client)
            formatted_text = f'{username} thinks, {text}{model_suffix}'
        elif event_type == 'enter':
            formatted_text = text
        elif event_type == 'exit':
            formatted_text = text
        else:
            formatted_text = text

        # Append to chat history with timestamp
        timestamp = datetime.now().isoformat()
        self.chat_history.append({
            'text': formatted_text,
            'timestamp': timestamp
        })

        # Trim history to max_history messages
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

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

    async def broadcast_log(self, log_entry: Dict):
        """
        Broadcast log entry to subscribed WebSocket clients.

        Args:
            log_entry: Log entry dictionary with type, level, name, message, timestamp
        """
        for ws in list(self.log_subscribers):
            try:
                await ws.send(json.dumps(log_entry))
            except Exception as e:
                # Remove disconnected websocket
                self.log_subscribers.discard(ws)
                logger.debug(f"Removed disconnected log subscriber: {e}")

    async def get_completions(self, user_id: str, command: str, partial: str) -> List[str]:
        """
        Get tab completion matches for a command.

        Args:
            user_id: User requesting completions
            command: Command being completed (e.g., '@setdesc', 'take')
            partial: Partial name to match

        Returns:
            List of matching names (sorted alphabetically)
        """
        matches = []
        partial_lower = partial.lower()

        # Get user's current room
        user = self.world.get_user(user_id)
        if not user:
            return []

        room_id = user['current_room']
        room = self.world.get_room(room_id)
        if not room:
            return []

        # Commands that complete objects
        object_commands = {'@setdesc', '@describe', 'take', 'get', 'drop', 'look', 'examine'}
        # Commands that complete agents
        agent_commands = {'@observe', '@relationship', '@memory', '@me'}
        # Commands that complete rooms
        room_commands = {'@teleport', '@goto'}

        # Get matching objects in current room
        if command in object_commands:
            for obj_id in room.get('contents', []):
                obj = self.world.get_object(obj_id)
                if obj:
                    name = obj.get('name', '')
                    if name.lower().startswith(partial_lower):
                        matches.append(name)

        # Get matching agents
        if command in agent_commands or command in {'look', 'examine'}:
            for occupant_id in room.get('occupants', []):
                occupant = self.world.get_user(occupant_id)
                if occupant and occupant.get('type') == 'agent':
                    # Get agent name from agent manager (respects @setname)
                    agent_obj = self.agent_manager.get_agent(occupant_id)
                    name = agent_obj.agent_name if agent_obj else occupant.get('name', occupant_id)
                    if name.lower().startswith(partial_lower):
                        matches.append(name)

        # Get matching rooms (all rooms, not just connected ones)
        if command in room_commands:
            for room_id, room_data in self.world.rooms.items():
                name = room_data.get('name', '')
                if name.lower().startswith(partial_lower):
                    matches.append(name)

        # Sort alphabetically and remove duplicates
        matches = sorted(set(matches))

        return matches

    async def auto_save_loop(self):
        """Periodically save world and agent state."""
        while True:
            await asyncio.sleep(self.save_interval)
            logger.info("Auto-saving world and agent states...")
            self._save_chat_history()
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

        # Start NoodleScope API server
        if self.api_server:
            self.api_runner = await self.api_server.start()
            logger.info("NoodleScope 2.0 API server started on port 8081")

        # Start WebSocket server
        host = self.config['server']['host']
        port = self.config['server']['port']

        logger.info(f"Starting WebSocket server on {host}:{port}")

        async with websockets.serve(self.handle_connection, host, port):
            logger.info("cMUSH server ready!")
            logger.info(f"World: {self.world.get_stats()}")
            logger.info(f"Agents: {len(self.agent_manager.agents)}")
            logger.info("📊 NoodleScope 2.0 UI: http://localhost:8081/noodlescope")
            await asyncio.Future()  # Run forever

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down cMUSH server...")

        # Cancel background tasks
        if self.save_task:
            self.save_task.cancel()
        if self.autonomous_poll_task:
            self.autonomous_poll_task.cancel()

        # Export session profiler data
        if self.session_profiler:
            try:
                session_file = self.session_profiler.export_session()
                logger.info(f"Session data exported: {session_file}")
            except Exception as e:
                logger.error(f"Error exporting session: {e}")

        # Save chat history before saving world
        self._save_chat_history()

        # Save everything (stop cognition on shutdown)
        self.world.save_all()
        await self.agent_manager.save_all_agents(stop_cognition=True)

        # Cleanup NoodleScope API server
        if self.api_runner:
            await self.api_runner.cleanup()
            logger.info("NoodleScope API server stopped")

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
