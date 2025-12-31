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

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from world import World
from auth import AuthManager
from commands import CommandParser
from agent_bridge import AgentManager
from llm_interface import OpenAICompatibleLLM
from session_profiler import SessionProfiler
from kimmie_character import KimmieCharacter
from recipe_loader import RecipeLoader
from script_manager import ScriptManager
from entropy_service import initialize_entropy_service

# Semantic World integration
from semantic_integration import (
    init_semantic_world,
    log_speech,
    log_arrival,
    log_departure,
    log_emote,
    log_action,
    register_stage_from_room,
    get_stats as get_semantic_stats
)

# Scene Protocol integration (Noodlings Scene Protocol for renderers)
from scene_protocol_integration import (
    SCENE_PROTOCOL_AVAILABLE,
    GAUSSIAN_ADAPTER_AVAILABLE,
    SEMANTIC_QUERY_AVAILABLE,
    init_scene_state_manager,
    sync_room_to_zone,
    sync_agent_to_noodling,
    sync_player_to_scene,
    record_dialogue as scene_record_dialogue,
    get_scene_packet_json,
    # Gaussian Scene Composition
    init_gaussian_scene_integration,
    compose_gaussian_scene,
    get_gaussian_scene_json,
    # Semantic Query (CLIP natural language)
    init_semantic_query_engine,
    query_scene_semantic,
    raycast_scene,
    get_entity_visible_body_parts,
    register_entity_radiance,
)

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

        # Initialize entropy service (quantum randomness)
        entropy_config = self.config.get('entropy', {})
        use_hardware = entropy_config.get('use_hardware', False)
        device_path = entropy_config.get('device_path', None)
        self.use_hardware_rng = use_hardware
        self.rng_device_path = device_path
        initialize_entropy_service(use_hardware=use_hardware, device_path=device_path)
        logger.info(f"Entropy service initialized: hardware={use_hardware}, device={device_path}")

        # Initialize world - check for PROJECT_PATH first
        self.project_path = None
        project_path = os.environ.get("PROJECT_PATH")

        if project_path and os.path.exists(project_path):
            # New project mode - use ProjectBridge
            try:
                from project_bridge import ProjectBridge, setup_world_from_project
                logger.info(f"Loading from project: {project_path}")
                self.project_path = project_path

                # Create legacy-compatible world in project's Library folder
                world_dir = setup_world_from_project(project_path)
                logger.info(f"Project world cache created at: {world_dir}")

                # Set recipes path to project's Noodlings folder
                self.recipes_path = os.path.join(project_path, "Noodlings")
            except Exception as e:
                logger.error(f"Failed to load project, falling back to legacy: {e}")
                import traceback
                traceback.print_exc()
                world_dir = self.config['paths']['world_dir']
                self.recipes_path = "recipes"
        else:
            # Legacy mode - use config world_dir
            world_dir = self.config['paths']['world_dir']
            self.recipes_path = "recipes"
            logger.info(f"Using legacy world directory: {world_dir}")

        self.world = World(world_dir=world_dir)

        # Initialize auth
        self.auth = AuthManager(self.world)

        # Initialize recipe loader
        self.recipe_loader = RecipeLoader(self.recipes_path)

        # Initialize LLM (will be created in async context)
        self.llm = None

        # Initialize agent manager (after LLM)
        self.agent_manager = None

        # Initialize command parser (after agent manager)
        self.command_parser = None

        # Initialize script manager (after agent manager)
        self.script_manager = None

        # Active connections: websocket -> user_id
        self.connections: Dict = {}

        # Log subscribers: websockets that want to receive log streams
        self.log_subscribers: Set = set()

        # Chat history for session continuity (per-project if PROJECT_PATH set)
        self.chat_history = []
        project_path = os.environ.get('PROJECT_PATH')
        if project_path:
            # Use project-specific chat history
            self.history_file = Path(project_path) / 'Library' / 'chat_history.json'
            # Ensure Library folder exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using project chat history: {self.history_file}")
        else:
            # Fall back to legacy global history
            self.history_file = Path('world/chat_history.json')
            logger.info("Using legacy global chat history")
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

        # Affect state broadcasting (for brain indicator UI)
        self.affect_broadcast_interval = 2.0  # Broadcast every 2 seconds
        self.affect_broadcast_task = None

        # Profiler and interpretation components
        self.session_profiler = None
        self.kimmie = None

    async def initialize_async_components(self):
        """Initialize async components (LLM, agents)."""
        # Initialize LLM with provider switching
        llm_config = self.config['llm']

        # Determine which provider to use
        provider = llm_config.get('provider', 'local')
        logger.info(f"LLM provider: {provider}")

        # Get provider-specific config
        provider_config = None  # Initialize for Kimmie later

        if provider == 'ollama':
            # Use Ollama with full observability
            from ollama_manager import OllamaManager, OllamaConfig

            # Load NoodleStudio preferences for Ollama settings
            from pathlib import Path
            import json
            prefs_file = Path.home() / ".noodlestudio" / "settings.json"
            ollama_prefs = {}
            if prefs_file.exists():
                try:
                    with open(prefs_file, 'r') as f:
                        settings = json.load(f)
                        ollama_prefs = settings.get('ollama', {})
                except:
                    pass

            # Merge config with preferences (prefs override config)
            # NOTE: Model assignments now managed by ModelLabelManager, not config
            ollama_config_dict = llm_config.get('ollama', {})
            merged_config = {
                'host': ollama_prefs.get('host') or ollama_config_dict.get('host', 'http://localhost:11434'),
                'models_directory': ollama_prefs.get('models_directory') or ollama_config_dict.get('models_directory', '/Volumes/DOUBLETROUBLE/models'),
                'default_timeout': ollama_config_dict.get('default_timeout', 120),
                'load_timeout': ollama_config_dict.get('load_timeout', 300),
            }

            ollama_config = OllamaConfig(**merged_config)
            self.llm = OllamaManager(config=ollama_config)
            await self.llm.__aenter__()

            logger.info(f"🦙 Using Ollama:")
            logger.info(f"  SMALL:  {ollama_config.get_model_for_tier('SMALL')}")
            logger.info(f"  MEDIUM: {ollama_config.get_model_for_tier('MEDIUM')}")
            logger.info(f"  LARGE:  {ollama_config.get_model_for_tier('LARGE')}")
            logger.info(f"  Host:   {ollama_config.host}")

            # Set provider_config for Kimmie (use Ollama's host, but in OpenAI-compatible format)
            provider_config = {
                'api_base': ollama_config.host,
                'model': ollama_config.get_model_for_tier('MEDIUM')  # Kimmie uses MEDIUM tier
            }

        elif provider == 'openrouter':
            provider_config = llm_config.get('openrouter', {})
            logger.info(f"🌐 Using OpenRouter with model: {provider_config.get('model')}")

            self.llm = OpenAICompatibleLLM(
                api_base=provider_config.get('api_base', 'https://openrouter.ai/api/v1'),
                api_key=provider_config.get('api_key', 'not-needed'),
                model=provider_config.get('model', 'SMALL'),
                timeout=provider_config.get('timeout', 30),
                max_concurrent=20,
                use_model_instances=False
            )
            await self.llm.__aenter__()

        else:  # default to 'local' (Ollama)
            provider_config = llm_config.get('local', llm_config)  # Fallback to root llm config for backward compat
            logger.info(f"💻 Using local Ollama with model: {provider_config.get('model')}")

            self.llm = OpenAICompatibleLLM(
                api_base=provider_config.get('api_base', 'http://localhost:11434/v1'),
                api_key=provider_config.get('api_key', 'not-needed'),
                model=provider_config.get('model', 'SMALL'),
                timeout=provider_config.get('timeout', 30),
                max_concurrent=20,
                use_model_instances=False
            )
            await self.llm.__aenter__()

        # Initialize agent manager (pass global config for personality traits)
        self.agent_manager = AgentManager(self.llm, self.world, global_config=self.config)

        # Load existing agents
        await self.load_agents()

        # Initialize script manager (server-authoritative scripting)
        self.script_manager = ScriptManager(self.world, self.agent_manager)
        logger.info(" ScriptManager initialized")

        # Initialize command parser (with config for persistence)
        self.command_parser = CommandParser(
            self.world,
            self.agent_manager,
            server=self,
            config=self.config,
            config_path=self.config_path,
            script_manager=self.script_manager  # Pass script_manager
        )

        # Initialize session profiler
        session_id = f"cmush_session_{int(asyncio.get_event_loop().time())}"
        self.session_profiler = SessionProfiler(session_id=session_id)

        # Initialize @Kimmie character (use provider config from above)
        self.kimmie = KimmieCharacter(
            llm_base_url=provider_config.get('api_base', 'http://localhost:11434/v1'),
            llm_model=provider_config.get('model', 'SMALL'),
            session_profiler=self.session_profiler
        )

        # Wire profiler into agent manager
        self.agent_manager.set_session_profiler(self.session_profiler)

        # Initialize Semantic World system
        try:
            # Get stages path from world_dir
            world_dir = self.config['paths']['world_dir']
            stages_path = os.path.join(world_dir, 'stages')
            events_path = os.path.join(world_dir, 'events')

            init_semantic_world(
                persist_path=events_path,
                stages_path=stages_path
            )

            # Register existing rooms as stages
            for room_id, room_data in self.world.rooms.items():
                register_stage_from_room(room_id, room_data)

            semantic_stats = get_semantic_stats()
            logger.info(f"[SemanticWorld] Initialized: {semantic_stats['total_events']} events loaded")
        except Exception as e:
            logger.warning(f"[SemanticWorld] Failed to initialize (non-fatal): {e}")

        # Initialize Scene Protocol (Noodlings Scene Protocol for renderers)
        if SCENE_PROTOCOL_AVAILABLE:
            try:
                # Get default stage name from first room or config
                default_room = next(iter(self.world.rooms.values()), {})
                stage_name = default_room.get('name', 'The World')
                stage_id = next(iter(self.world.rooms.keys()), 'default')

                init_scene_state_manager(
                    stage_id=stage_id,
                    stage_name=stage_name
                )

                # Sync rooms to zones
                for room_id, room_data in self.world.rooms.items():
                    sync_room_to_zone(room_data, room_id)

                logger.info(f"[SceneProtocol] Initialized with {len(self.world.rooms)} zones")
            except Exception as e:
                logger.warning(f"[SceneProtocol] Failed to initialize (non-fatal): {e}")
                import traceback
                traceback.print_exc()

        # Initialize Gaussian Scene Integration (for 3D Gaussian Splatting rendering)
        if GAUSSIAN_ADAPTER_AVAILABLE:
            try:
                # Use PROJECT_PATH env var if set, otherwise default to project root
                project_path = os.environ.get('PROJECT_PATH', str(Path(__file__).parent.parent.parent))
                if init_gaussian_scene_integration(project_path):
                    logger.info(f"[Gaussian] Scene integration initialized")
            except Exception as e:
                logger.warning(f"[Gaussian] Failed to initialize (non-fatal): {e}")

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
            # Get checkpoint path - use default if not specified
            checkpoint_path = agent_data.get('checkpoint_path', '../../models/checkpoints/best_checkpoint.npz')
            current_room = agent_data.get('current_room', 'room_000')
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
                # Use recipe name if available (e.g., "Red Fire Anklebiter")
                if recipe.name:
                    agent_name = recipe.name
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

                # Phase 7: Reload affective_reinforcement from recipe
                # Critical: Ensures agents get latest reinforcement config on server restart
                if recipe.affective_reinforcement:
                    config['affective_reinforcement'] = recipe.affective_reinforcement
                    logger.info(f"[LOAD] Loaded affective_reinforcement for {agent_id}: {recipe.affective_reinforcement}")

                logger.info(f"[LOAD] Reloaded recipe for {agent_id}: species={recipe.species}")

            # NEW: Inject ensemble context if agent is part of an ensemble
            ensemble_data = agent_data.get('ensemble')
            if ensemble_data:
                ensemble_name = ensemble_data.get('name', 'Unknown Ensemble')
                ensemble_mission = ensemble_data.get('mission', '')
                agent_role = ensemble_data.get('role', 'member')
                ensemble_dynamics = ensemble_data.get('dynamics', {})
                ensemble_knowledge = ensemble_data.get('knowledge', {})

                # Build ensemble context addition
                ensemble_context = f"\n\n**ENSEMBLE CONTEXT**\n"
                ensemble_context += f"You are part of the {ensemble_name}.\n"
                ensemble_context += f"Your role: {agent_role}\n"
                ensemble_context += f"Shared mission: {ensemble_mission}\n"

                if ensemble_knowledge:
                    ensemble_context += f"\nShared knowledge:\n"
                    for key, value in ensemble_knowledge.items():
                        if isinstance(value, list):
                            ensemble_context += f"  - {key}: {', '.join(str(v) for v in value)}\n"
                        else:
                            ensemble_context += f"  - {key}: {value}\n"

                # Append to identity_prompt
                current_identity = config.get('identity_prompt', '')
                config['identity_prompt'] = current_identity + ensemble_context
                logger.info(f"[LOAD] Injected {ensemble_name} context for {agent_id} (role: {agent_role})")

            try:
                await self.agent_manager.create_agent(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    checkpoint_path=checkpoint_path,
                    spawn_room=current_room,
                    config=config
                )
                logger.info(f"Loaded agent: {agent_id}")

                # Scene Protocol: sync agent to noodling for perception system
                if SCENE_PROTOCOL_AVAILABLE:
                    sync_agent_to_noodling(
                        agent_data={
                            'name': agent_name,
                            'species': config.get('species', 'unknown'),
                        },
                        agent_id=agent_id,
                        room_id=current_room
                    )
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
                        logger.info(f"[AUTH] Registration attempt: username={data.get('username')}")
                        response = await self.handle_register(data)
                        logger.info(f"[AUTH] Registration response: {response}")
                        await websocket.send(json.dumps(response))

                    elif msg_type == 'token_auth':
                        # Cloud token authentication (NoodleStudio unified auth)
                        logger.info(f"[AUTH] Token auth attempt")
                        response = await self.handle_token_auth(data)
                        logger.info(f"[AUTH] Token auth response: success={response.get('success')}")
                        if response['success']:
                            user_id = response['user_id']
                            session_token = response['session_token']
                            self.connections[websocket] = user_id

                            # Get avatar info for welcome message
                            avatar_name = response.get('avatar_name', response.get('display_name', 'Traveler'))

                            # Scene Protocol: sync player to scene
                            if SCENE_PROTOCOL_AVAILABLE:
                                user = self.world.get_user(user_id)
                                if user:
                                    room_id = user.get('current_room', 'room_000')
                                    sync_player_to_scene(
                                        player_id=user_id,
                                        player_name=avatar_name,
                                        room_id=room_id
                                    )

                            # Send chat history first
                            for history_entry in self.chat_history:
                                await self.send_to_user(websocket, {
                                    'type': 'history',
                                    'text': history_entry['text'],
                                    'timestamp': history_entry['timestamp']
                                })

                            # Send welcome message
                            pid = os.getpid()
                            ws_port = self.config.get('server', {}).get('port', 8765)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            rng_status = self._get_rng_status()

                            banner = (
                                ":::.    :::.    ...         ...    :::::::-.   :::    .,::::::      .        :    ...    ::: .::::::.   ::   .:\n"
                                "`;;;;,  `;;; .;;;;;;;.   .;;;;;;;.  ;;,   `';, ;;;    ;;;;''''      ;;,.    ;;;   ;;     ;;;;;;`    `  ,;;   ;;,\n"
                                "  [[[[[. '[[,[[     \\[[,,[[     \\[[,`[[     [[ [[[     [[cccc       [[[[, ,[[[[, [['     [[\'[==/[[[[,,[[[,,,[[[[\n"
                                "  $$$ \"Y$c$$$$$,     $$$$$$,     $$$ $$,    $$ $$'     $$\"\"\"\"       $$$$$$$$\"$$$ $$      $$$  '''    $\"$$$\"\"\"$$$\n"
                                "  888    Y88\"888,_ _,88P\"888,_ _,88P 888_,o8P'o88oo,.__888oo,__     888 Y88\" 888o88    .d888 88b    dP 888   \"88o\n"
                                "  MMM     YM  \"YMMMMMP\"   \"YMMMMMP\"  MMMMP\"`  \"\"\"\"YUMMM\"\"\"\"YUMMM    MMM  M'  \"MMM \"YmmMMMM\"\"  \"YMmMY\"  MMM    YMM\n"
                                "\n"
                                f"Welcome, {avatar_name}!\n"
                                "Noodlings Multi-User Shared Hallucination\n"
                                f"(Authenticated via NoodleStudio)\n"
                                "\n"
                                f"Server: PID {pid} | ws://localhost:{ws_port} | {timestamp}\n"
                                f"RNG: {rng_status}\n"
                            )

                            await self.send_to_user(websocket, {
                                'type': 'tui-green',
                                'text': banner
                            })

                            # Send agent list
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

                            # Generate enter event
                            user = self.world.get_user(user_id)
                            room = self.world.get_user_room(user_id)
                            if user and room:
                                room_id = user.get('current_room', 'room_000')
                                enter_text = f"{avatar_name} materializes."

                                enter_event = {
                                    'type': 'enter',
                                    'user': user_id,
                                    'room': room_id,
                                    'text': enter_text
                                }

                                await self.broadcast_event(enter_event)
                                asyncio.create_task(self._handle_agent_entrance(enter_event))

                            # Show current room
                            look_result = await self.command_parser.cmd_look(user_id, '')
                            await self.send_to_user(websocket, {
                                'type': 'output',
                                'text': look_result['output']
                            })

                        await websocket.send(json.dumps(response))

                    elif msg_type == 'login':
                        logger.info(f"[AUTH] Login attempt: username={data.get('username')}")
                        response = await self.handle_login(data)
                        logger.info(f"[AUTH] Login response: success={response.get('success')}, user_id={response.get('user_id')}")
                        if response['success']:
                            user_id = response['user_id']
                            session_token = response['session_token']
                            self.connections[websocket] = user_id

                            # Scene Protocol: sync player to scene
                            if SCENE_PROTOCOL_AVAILABLE:
                                user = self.world.get_user(user_id)
                                if user:
                                    room_id = user.get('current_room', 'room_000')
                                    sync_player_to_scene(
                                        player_id=user_id,
                                        player_name=data['username'],
                                        room_id=room_id
                                    )

                            # Send chat history first
                            for history_entry in self.chat_history:
                                await self.send_to_user(websocket, {
                                    'type': 'history',
                                    'text': history_entry['text'],
                                    'timestamp': history_entry['timestamp']
                                })

                            # Send welcome message with ASCII banner and server diagnostics
                            # (os and datetime already imported at module level)

                            pid = os.getpid()
                            ws_port = self.config.get('server', {}).get('port', 8765)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # Check RNG status
                            rng_status = self._get_rng_status()

                            banner = (
                                ":::.    :::.    ...         ...    :::::::-.   :::    .,::::::      .        :    ...    ::: .::::::.   ::   .:\n"
                                "`;;;;,  `;;; .;;;;;;;.   .;;;;;;;.  ;;,   `';, ;;;    ;;;;''''      ;;,.    ;;;   ;;     ;;;;;;`    `  ,;;   ;;,\n"
                                "  [[[[[. '[[,[[     \\[[,,[[     \\[[,`[[     [[ [[[     [[cccc       [[[[, ,[[[[, [['     [[\'[==/[[[[,,[[[,,,[[[[\n"
                                "  $$$ \"Y$c$$$$$,     $$$$$$,     $$$ $$,    $$ $$'     $$\"\"\"\"       $$$$$$$$\"$$$ $$      $$$  '''    $\"$$$\"\"\"$$$\n"
                                "  888    Y88\"888,_ _,88P\"888,_ _,88P 888_,o8P'o88oo,.__888oo,__     888 Y88\" 888o88    .d888 88b    dP 888   \"88o\n"
                                "  MMM     YM  \"YMMMMMP\"   \"YMMMMMP\"  MMMMP\"`  \"\"\"\"YUMMM\"\"\"\"YUMMM    MMM  M'  \"MMM \"YmmMMMM\"\"  \"YMmMY\"  MMM    YMM\n"
                                "\n"
                                f"Welcome, {data['username']}!\n"
                                "Noodlings Multi-User Shared Hallucination\n"
                                "\n"
                                f"Server: PID {pid} | ws://localhost:{ws_port} | {timestamp}\n"
                                f"RNG: {rng_status}\n"
                            )

                            await self.send_to_user(websocket, {
                                'type': 'tui-green',  # Use TUI green for banner
                                'text': banner
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
                                room_id = user.get('current_room', 'room_000')

                                # Create enter event with description
                                enter_text = f"{username} appears"
                                if description:
                                    enter_text += f". {description}"
                                else:
                                    enter_text += "."

                                enter_event = {
                                    'type': 'enter',
                                    'user': user_id,
                                    'room': room_id,
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

                        # Lab mode interception: Check if lab test is active for this user
                        lab_session = None
                        if hasattr(self, 'lab_sessions'):
                            lab_session = self.lab_sessions.get(user_id)

                        logger.info(f"[LAB] user={user_id}, session={lab_session is not None}, command={command_text[:50]}")

                        if lab_session and not command_text.startswith('@lab'):
                            # Lab mode active - intercept messages for dual cognition
                            # Only intercept say/emote commands (not admin commands)
                            if not command_text.startswith('@'):
                                # Extract message text
                                if command_text.startswith('"'):
                                    # say shortcut
                                    message = command_text[1:]
                                elif command_text.startswith(':'):
                                    # emote shortcut
                                    message = command_text[1:]
                                elif command_text.startswith('say '):
                                    message = command_text[4:]
                                elif command_text.startswith('emote '):
                                    message = command_text[6:]
                                else:
                                    message = None

                                if message:
                                    # Get target agent (try all agents in room until we find a loaded one)
                                    user = self.world.get_user(user_id)
                                    room = self.world.get_room(user['current_room'])
                                    agents_in_room = [occ for occ in room.get('occupants', []) if occ.startswith('agent_')]

                                    logger.info(f"[LAB] Message extracted: '{message}', agents_in_room: {agents_in_room}")

                                    target_agent = None
                                    for agent_id in agents_in_room:
                                        agent_obj = self.agent_manager.get_agent(agent_id)
                                        if agent_obj:
                                            target_agent = agent_obj
                                            logger.info(f"[LAB] Selected target agent: {agent_id}")
                                            break

                                    if target_agent:
                                            # Intercept and run dual cognition
                                            logger.info(f"[LAB] About to call intercept_message...")
                                            async def broadcast_to_user_func(text):
                                                await self.send_to_user(websocket, {
                                                    'type': 'output',
                                                    'text': text
                                                })

                                            intercepted = await lab_session.intercept_message(
                                                message=message,
                                                agent=target_agent,
                                                world=self.world,
                                                broadcast_fn=broadcast_to_user_func
                                            )

                                            logger.info(f"[LAB] Intercepted: {intercepted}")

                                            if intercepted:
                                                # Message was intercepted - don't execute normal command
                                                logger.info(f"[LAB] Skipping normal command execution")
                                                continue

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

                            # Handle graceful shutdown event
                            if event['type'] == 'shutdown':
                                delay = event.get('delay', 5)
                                logger.info(f"Shutdown initiated by {user_id}, delay={delay}s")

                                # Schedule shutdown in background
                                asyncio.create_task(self.graceful_shutdown(delay))

                            # Let agents perceive the event (say, emote, enter, exit)
                            # NOTE: 'think' events are PRIVATE - not broadcast to agents!
                            if event['type'] in ['say', 'emote', 'enter', 'exit']:
                                agent_responses = await self.agent_manager.broadcast_event(event)

                                # Broadcast agent responses
                                for agent_response in agent_responses:
                                    agent_id = agent_response['agent_id']
                                    agent_data = self.world.get_user(agent_id)

                                    # Skip if agent no longer exists
                                    if not agent_data:
                                        continue

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

                                    # Create event for agent response (say/emote/think)
                                    agent_event = {
                                        'type': agent_response['command'],
                                        'user': agent_id,
                                        'username': agent_name,
                                        'room': agent_data['current_room'],
                                        'text': agent_response['text']
                                    }

                                    # Broadcast to websocket clients (humans) - ALL types including 'think'
                                    await self.broadcast_event(agent_event)

                                    # CRITICAL FIX: 'think' events are PRIVATE - don't broadcast to other agents!
                                    # Only 'say' and 'emote' events should trigger other agents' cognition
                                    if agent_response['command'] in ['say', 'emote']:
                                        # Let OTHER agents perceive this agent's response
                                        other_agent_responses = await self.agent_manager.broadcast_event(agent_event)
                                    else:
                                        # Skip agent perception for 'think' commands (private thoughts)
                                        other_agent_responses = []

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

                    # Log subscription (Studio operation - no auth required)
                    elif msg_type == 'subscribe_logs':
                        self.log_subscribers.add(websocket)
                        client_id = self.connections.get(websocket, 'studio_console')
                        logger.info(f"Client subscribed to logs: {client_id}")

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

                    # Scene Protocol: Get scene packet (for LLM renderers like Genie)
                    elif msg_type == 'get_scene_packet':
                        scene_json = get_scene_packet_json()
                        await websocket.send(json.dumps({
                            'type': 'scene_packet',
                            'data': json.loads(scene_json) if scene_json else None,
                            'available': SCENE_PROTOCOL_AVAILABLE
                        }))

                    # Gaussian Scene: Get composed Gaussian scene (for 3DGS renderers)
                    elif msg_type == 'get_gaussian_scene':
                        gaussian_json = get_gaussian_scene_json()
                        await websocket.send(json.dumps({
                            'type': 'gaussian_scene',
                            'data': json.loads(gaussian_json) if gaussian_json else None,
                            'available': GAUSSIAN_ADAPTER_AVAILABLE
                        }))

                    # Semantic Query: Natural language query on Gaussian scene
                    elif msg_type == 'semantic_query':
                        query = data.get('query', '')
                        top_k = data.get('top_k', 5)
                        result = query_scene_semantic(query, top_k=top_k)
                        await websocket.send(json.dumps({
                            'type': 'semantic_query_result',
                            'data': result,
                            'available': SEMANTIC_QUERY_AVAILABLE
                        }))

                    # Semantic Query: Raycast for click-to-inspect
                    elif msg_type == 'semantic_raycast':
                        origin = data.get('origin', [0, 0, 0])
                        direction = data.get('direction', [0, 0, 1])
                        result = raycast_scene(origin, direction)
                        await websocket.send(json.dumps({
                            'type': 'semantic_raycast_result',
                            'data': result,
                            'available': SEMANTIC_QUERY_AVAILABLE
                        }))

                    # Semantic Query: Get visible body parts
                    elif msg_type == 'get_visible_body_parts':
                        perceiver_id = data.get('perceiver_id', '')
                        target_id = data.get('target_id', '')
                        visible_parts = get_entity_visible_body_parts(perceiver_id, target_id)
                        await websocket.send(json.dumps({
                            'type': 'visible_body_parts_result',
                            'perceiver_id': perceiver_id,
                            'target_id': target_id,
                            'visible_parts': visible_parts,
                            'available': SEMANTIC_QUERY_AVAILABLE
                        }))

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
            data: Login data (includes optional 'invisible' flag for admin)

        Returns:
            Response dict
        """
        username = data.get('username', '')
        password = data.get('password', '')
        invisible = data.get('invisible', False)  # Admin invisible mode

        success, user_id, message = self.auth.authenticate(username, password)

        if success:
            session_token = self.auth.create_session(user_id)

            # Set invisible mode for this user if requested
            user = self.world.get_user(user_id)
            if user:
                user['invisible'] = invisible
                self.world.save_all()

                if invisible:
                    logger.info(f"Admin {username} logged in (INVISIBLE MODE)")
                else:
                    logger.info(f"User {username} logged in")

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

    async def handle_token_auth(self, data: Dict) -> Dict:
        """
        Handle cloud token authentication (NoodleStudio unified auth).

        Args:
            data: Token auth data containing:
                - token: NoodleStudio session token
                - avatar_id: Optional avatar ID to use
                - avatar: Optional avatar metadata dict

        Returns:
            Response dict with user_id, session_token, avatar info
        """
        token = data.get('token', '')
        avatar_id = data.get('avatar_id')
        avatar_data = data.get('avatar', {})

        if not token:
            return {
                'type': 'token_auth_response',
                'success': False,
                'message': 'No token provided'
            }

        # Authenticate with cloud
        success, user_id, message, profile = self.auth.authenticate_with_cloud_token(
            token=token,
            avatar_id=avatar_id
        )

        if success:
            session_token = self.auth.create_session(user_id)

            # Determine display name (avatar > cloud profile > generic)
            avatar_name = avatar_data.get('display_name') or profile.get('display_name') or 'Traveler'

            # Update user's current avatar info
            user = self.world.get_user(user_id)
            if user:
                user['current_avatar_id'] = avatar_id
                user['current_avatar_name'] = avatar_name
                if avatar_data.get('description'):
                    user['description'] = avatar_data['description']

                # Verify user's current_room exists - fix if not
                current_room = user.get('current_room')
                if not current_room or not self.world.get_room(current_room):
                    # Room doesn't exist - find a valid one
                    if self.world.rooms:
                        valid_room = next(iter(self.world.rooms.keys()))
                        user['current_room'] = valid_room
                        logger.info(f"Fixed user {user_id} room: {current_room} -> {valid_room}")
                    else:
                        # No rooms exist - create a default
                        new_room = self.world.create_room(
                            name="The Nexus",
                            description="A cozy campfire with crackling logs. Welcome to the world!"
                        )
                        user['current_room'] = new_room
                        logger.info(f"Created default room for user {user_id}: {new_room}")

                self.world.save_all()

            logger.info(f"Token auth success: {avatar_name} ({user_id})")

            return {
                'type': 'token_auth_response',
                'success': True,
                'user_id': user_id,
                'session_token': session_token,
                'display_name': profile.get('display_name', ''),
                'avatar_name': avatar_name,
                'avatar_id': avatar_id,
                'message': message
            }
        else:
            return {
                'type': 'token_auth_response',
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

    async def broadcast_to_user(self, user_id: str, text: str):
        """
        Broadcast text message to specific user by user_id.

        Used by lab system to send comparison results.

        Args:
            user_id: Target user ID
            text: Text to send
        """
        # Find websocket for user
        for ws, ws_user_id in self.connections.items():
            if ws_user_id == user_id:
                await self.send_to_user(ws, {
                    'type': 'output',
                    'text': text
                })
                break

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

        # Log to Semantic World event store
        try:
            # Get room occupants as witnesses (excluding the actor)
            room_occupants = self.world.get_room_occupants(room_id) if room_id else []
            witnesses = [occ for occ in room_occupants if occ != user_id]

            if event_type == 'say':
                log_speech(
                    speaker_id=user_id,
                    stage_id=room_id,
                    content=text,
                    witnesses=witnesses
                )
            elif event_type == 'emote':
                log_emote(
                    actor_id=user_id,
                    stage_id=room_id,
                    emote_text=text,
                    witnesses=witnesses
                )
            elif event_type == 'enter':
                log_arrival(
                    arriver_id=user_id,
                    stage_id=room_id,
                    witnesses=witnesses
                )
            elif event_type == 'exit':
                log_departure(
                    departer_id=user_id,
                    stage_id=room_id,
                    witnesses=witnesses
                )
        except Exception as e:
            logger.debug(f"[SemanticWorld] Event logging failed (non-fatal): {e}")

        # Scene Protocol: sync events to SceneStateManager
        if SCENE_PROTOCOL_AVAILABLE:
            try:
                if event_type == 'say':
                    # Record dialogue for perception slices
                    scene_record_dialogue(user_id, text, tone="neutral")
                elif event_type == 'enter':
                    # Player/agent entered room - update their zone
                    if user_id.startswith('agent_'):
                        agent_data = self.world.get_user(user_id)
                        if agent_data:
                            sync_agent_to_noodling(
                                agent_data={'name': username, 'species': agent_data.get('species', 'unknown')},
                                agent_id=user_id,
                                room_id=room_id
                            )
                    else:
                        sync_player_to_scene(
                            player_id=user_id,
                            player_name=username,
                            room_id=room_id
                        )
            except Exception as e:
                logger.debug(f"[SceneProtocol] Event sync failed (non-fatal): {e}")

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

    def _get_rng_status(self) -> str:
        """Get RNG status message for MOTD."""
        if self.use_hardware_rng:
            device = self.rng_device_path or "Hardware device"
            return f"Hardware RNG active ({device}) - Quantum non-determinism enabled"
        else:
            return "Internal RNG (deterministic pseudorandom). Consider avalanche effect RNG for quantum randomness"

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

    async def affect_broadcast_loop(self):
        """
        Periodically broadcast agent affect states to all connected clients.
        This updates the brain indicator UI with real-time 5-D affect vectors.
        """
        while True:
            await asyncio.sleep(self.affect_broadcast_interval)

            try:
                # Get all active agents
                for agent_id, agent in self.agent_manager.agents.items():
                    try:
                        # Get phenomenal state
                        state = agent.get_phenomenal_state()

                        # Extract 5-D affect vector from fast state
                        fast_state = state.get('fast')  # Correct key: 'fast' not 'fast_state'
                        if fast_state is not None and len(fast_state) >= 5:
                            # fast_state is 16-D, first 5 are affect: [valence, arousal, fear, sorrow, boredom, ...]
                            affect_vector = fast_state[:5].tolist() if hasattr(fast_state, 'tolist') else list(fast_state[:5])

                            # Broadcast to all connected clients
                            state_message = {
                                'type': 'agent_state',
                                'agent_id': agent_id,
                                'affect': affect_vector
                            }

                            for ws in list(self.connections.keys()):
                                try:
                                    await ws.send(json.dumps(state_message))
                                except Exception:
                                    pass  # Client disconnected, will be cleaned up elsewhere

                    except Exception as e:
                        logger.debug(f"Error broadcasting state for {agent_id}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error in affect broadcast loop: {e}", exc_info=True)

    async def start(self):
        """Start the cMUSH server."""
        # Initialize async components
        await self.initialize_async_components()

        # Start auto-save task
        self.save_task = asyncio.create_task(self.auto_save_loop())

        # Start autonomous event polling task
        self.autonomous_poll_task = asyncio.create_task(self.autonomous_event_loop())

        # Start affect state broadcasting task
        self.affect_broadcast_task = asyncio.create_task(self.affect_broadcast_loop())

        # Start WebSocket server
        host = self.config['server']['host']
        port = self.config['server']['port']

        logger.info(f"Starting WebSocket server on {host}:{port}")

        async with websockets.serve(self.handle_connection, host, port):
            logger.info("cMUSH server ready!")
            logger.info(f"World: {self.world.get_stats()}")
            logger.info(f"Agents: {len(self.agent_manager.agents)}")
            await asyncio.Future()  # Run forever

    async def graceful_shutdown(self, delay: int = 5):
        """
        Gracefully shutdown server with warning period.

        Args:
            delay: Seconds to wait before shutting down
        """
        # Broadcast shutdown warning to all connected clients
        warning_message = {
            'type': 'system_message',
            'text': f'\n  SERVER SHUTDOWN IN {delay} SECONDS \n\nSaving world state and disconnecting all users...\n'
        }

        for websocket in list(self.connections.keys()):
            try:
                await self.send_to_user(websocket, warning_message)
            except Exception as e:
                logger.error(f"Error sending shutdown warning: {e}")

        # Wait for delay period
        if delay > 0:
            await asyncio.sleep(delay)

        # Call existing shutdown method
        await self.shutdown()

        # Terminate process
        logger.info("Exiting process...")
        sys.exit(0)

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down cMUSH server...")

        # Cancel background tasks
        if self.save_task:
            self.save_task.cancel()
        if self.autonomous_poll_task:
            self.autonomous_poll_task.cancel()
        if self.affect_broadcast_task:
            self.affect_broadcast_task.cancel()

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
