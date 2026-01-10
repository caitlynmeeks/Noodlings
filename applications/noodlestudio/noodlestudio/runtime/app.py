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
#   NoodleApp - Headless runtime for NoodleStudio projects
#
#   This is the core runtime that executes NoodleStudio proje...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.app
# PURPOSE:  App
# LAYER:    Studio / Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ProjectConfig, NoodleAppConfig, NoodleApp
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from .llm_client import HeadlessLLMClient, LLMConfig, create_llm_client_from_env
from .channels import ChannelBus, ChannelMessage
from .world_channels import WorldChannelService, WorldConfig
from .brenda import BrendaDirector

logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    """Configuration loaded from build.yaml or project.yaml."""

    name: str = "Untitled Project"
    version: str = "1.0.0"
    main_stage: str = ""  # Path to main stage (e.g., "Stages/the_nexus")
    entry_assembly: str = ""  # Direct assembly path (alternative to stage)

    # Build settings
    include_renderer: bool = True
    window_size: tuple = (1280, 720)

    @staticmethod
    def load(project_path: Path) -> 'ProjectConfig':
        """
        Load project configuration from build.yaml or project.yaml.

        Args:
            project_path: Path to project directory

        Returns:
            Loaded ProjectConfig
        """
        config = ProjectConfig()

        # Try build.yaml first, then project.yaml
        for config_name in ["build.yaml", "project.yaml"]:
            config_file = project_path / config_name
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f) or {}

                config.name = data.get('name', config.name)
                config.version = data.get('version', config.version)
                config.main_stage = data.get('main_stage', config.main_stage)
                config.entry_assembly = data.get('entry_assembly', config.entry_assembly)

                # Settings block
                settings = data.get('settings', {})
                config.include_renderer = settings.get('include_renderer', True)
                window = settings.get('window_size', [1280, 720])
                config.window_size = tuple(window)

                logger.info(f"Loaded project config from {config_file}")
                break

        return config


@dataclass
class NoodleAppConfig:
    """Configuration for NoodleApp runtime."""

    # Project settings
    project_path: str = ""  # Path to project directory

    # LLM settings (if not using LLMConfig directly)
    llm_provider: str = "ollama"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""

    # Execution settings
    max_iterations: int = 10
    timeout: float = 120.0
    verbose: bool = False

    # Model label mapping
    model_labels: Dict[str, str] = field(default_factory=dict)


class NoodleApp:
    """
    Headless runtime for NoodleStudio projects.

    Loads a project, resolves the main stage/noodling, and executes
    the associated facet assembly.

    Usage:
        app = NoodleApp()
        app.load_project("/path/to/project")
        result = await app.run("Hello, world!")

    Or with assembly directly:
        app = NoodleApp()
        app.load_assembly("/path/to/assembly.yaml")
        result = await app.run("Hello, world!")
    """

    def __init__(self, config: Optional[NoodleAppConfig] = None):
        """
        Initialize NoodleApp runtime.

        Args:
            config: Runtime configuration. Uses defaults if not provided.
        """
        self.config = config or NoodleAppConfig()
        self.project_config: Optional[ProjectConfig] = None
        self.project_path: Optional[Path] = None

        self.assembly = None
        self.executor = None
        self.llm_client: Optional[HeadlessLLMClient] = None

        # Channel bus for inter-noodling communication
        self.channel_bus = ChannelBus()

        # World channels service (initialized when project loads)
        self.world_channels: Optional[WorldChannelService] = None
        self._world_config: Optional[Dict[str, Any]] = None

        # Stage director (Brenda)
        self.director: Optional[BrendaDirector] = None

        # Event handlers
        self._on_output: Optional[Callable[[str], None]] = None
        self._on_event: Optional[Callable[[Dict], None]] = None

    def _create_llm_client(self) -> HeadlessLLMClient:
        """Create LLM client from config or environment."""
        # First try environment variables
        if os.environ.get("NOODLE_LLM_PROVIDER"):
            return create_llm_client_from_env()

        # Use config
        llm_config = LLMConfig(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model_labels=self.config.model_labels
        )

        return HeadlessLLMClient(llm_config)

    async def _ensure_executor(self):
        """Initialize facet executor if not already done."""
        if self.executor is not None:
            return

        if self.llm_client is None:
            self.llm_client = self._create_llm_client()

        # Initialize world channels service
        if self.world_channels is None:
            world_config = WorldConfig.from_dict(self._world_config or {})
            self.world_channels = WorldChannelService(self.channel_bus, world_config)
            self.world_channels.start()
            if self.config.verbose:
                logger.info("[NoodleApp] Started world channels service")

        # Import executor (lazy to avoid Qt import at module load)
        from ..core.facet_executor import FacetExecutor

        self.executor = FacetExecutor(
            llm_client=self.llm_client,
            use_event_bus=False,  # No event bus in headless mode
            concurrency_mode='serial',  # Simpler for headless
            channel_bus=self.channel_bus  # For inter-noodling communication
        )

        if self.config.verbose:
            logger.info("[NoodleApp] Initialized facet executor")

    def load_project(self, project_path: str) -> bool:
        """
        Load a NoodleStudio project.

        Loads project configuration and resolves the main stage/assembly.

        Args:
            project_path: Path to project directory

        Returns:
            True if loaded successfully
        """
        self.project_path = Path(project_path)

        if not self.project_path.exists():
            logger.error(f"Project not found: {project_path}")
            return False

        if not self.project_path.is_dir():
            logger.error(f"Project path is not a directory: {project_path}")
            return False

        # Load project configuration
        self.project_config = ProjectConfig.load(self.project_path)

        if self.config.verbose:
            logger.info(f"[NoodleApp] Loaded project: {self.project_config.name}")

        # Find and load the entry assembly
        return self._resolve_entry_assembly()

    def _resolve_entry_assembly(self) -> bool:
        """
        Resolve the entry assembly from project config.

        Priority:
        1. entry_assembly (direct path)
        2. main_stage -> stage.yaml -> noodling -> assembly.yaml
        """
        if not self.project_path or not self.project_config:
            return False

        # Option 1: Direct assembly path
        if self.project_config.entry_assembly:
            assembly_path = self.project_path / self.project_config.entry_assembly
            return self.load_assembly(str(assembly_path))

        # Option 2: Load from main stage
        if self.project_config.main_stage:
            return self._load_assembly_from_stage(self.project_config.main_stage)

        # Option 3: Look for any assembly in facet_assemblies/
        assemblies_dir = self.project_path / "facet_assemblies"
        if assemblies_dir.exists():
            for yaml_file in assemblies_dir.glob("*.yaml"):
                logger.info(f"[NoodleApp] Using first found assembly: {yaml_file}")
                return self.load_assembly(str(yaml_file))

        logger.error("[NoodleApp] No entry assembly found in project")
        return False

    def _load_assembly_from_stage(self, stage_ref: str) -> bool:
        """
        Load assembly from a stage reference.

        Stages contain noodlings, which reference assemblies.

        Args:
            stage_ref: Stage reference (e.g., "Stages/the_nexus")

        Returns:
            True if assembly loaded
        """
        stage_path = self.project_path / stage_ref

        # Look for stage.yaml
        stage_yaml = stage_path / "stage.yaml"
        if not stage_yaml.exists():
            # Try as direct path
            if stage_path.suffix == ".yaml" and stage_path.exists():
                stage_yaml = stage_path
            else:
                logger.error(f"[NoodleApp] Stage not found: {stage_ref}")
                return False

        # Load stage
        with open(stage_yaml, 'r') as f:
            stage_data = yaml.safe_load(f) or {}

        # Extract world config if present
        self._world_config = stage_data.get('world', {})

        # Find noodlings in stage
        noodlings = stage_data.get('noodlings', [])
        if not noodlings:
            logger.warning(f"[NoodleApp] Stage has no noodlings: {stage_ref}")
            # Fall back to looking for assembly in stage directory
            for yaml_file in stage_path.glob("*.yaml"):
                if yaml_file.name != "stage.yaml":
                    return self.load_assembly(str(yaml_file))
            return False

        # Load first noodling's assembly
        first_noodling = noodlings[0]
        noodling_ref = first_noodling.get('ref', first_noodling.get('path', ''))

        return self._load_assembly_from_noodling(noodling_ref)

    def _load_assembly_from_noodling(self, noodling_ref: str) -> bool:
        """
        Load assembly from a noodling reference.

        Args:
            noodling_ref: Noodling reference (e.g., "Noodlings/red")

        Returns:
            True if assembly loaded
        """
        noodling_path = self.project_path / noodling_ref

        # Look for recipe.yaml which contains assembly reference
        recipe_yaml = noodling_path / "recipe.yaml"
        if not recipe_yaml.exists():
            logger.error(f"[NoodleApp] Noodling recipe not found: {noodling_ref}")
            return False

        with open(recipe_yaml, 'r') as f:
            recipe_data = yaml.safe_load(f) or {}

        # Get assembly reference
        assembly_ref = recipe_data.get('assembly', recipe_data.get('facet_assembly', ''))

        if assembly_ref:
            # Resolve relative to noodling or project
            assembly_path = noodling_path / assembly_ref
            if not assembly_path.exists():
                assembly_path = self.project_path / assembly_ref

            if assembly_path.exists():
                return self.load_assembly(str(assembly_path))

        # Look for assembly.yaml in noodling directory
        assembly_yaml = noodling_path / "assembly.yaml"
        if assembly_yaml.exists():
            return self.load_assembly(str(assembly_yaml))

        logger.error(f"[NoodleApp] No assembly found for noodling: {noodling_ref}")
        return False

    def load_assembly(self, assembly_path: str) -> bool:
        """
        Load a facet assembly directly.

        Args:
            assembly_path: Path to assembly YAML file

        Returns:
            True if loaded successfully
        """
        from ..core.facet_system import FacetAssembly

        path = Path(assembly_path)
        if not path.exists():
            logger.error(f"[NoodleApp] Assembly not found: {assembly_path}")
            return False

        try:
            self.assembly = FacetAssembly.load_yaml(str(path))
            if self.config.verbose:
                logger.info(f"[NoodleApp] Loaded assembly: {self.assembly.name}")
                logger.info(f"[NoodleApp]   Facets: {len(self.assembly.facets)}")
                logger.info(f"[NoodleApp]   Connections: {len(self.assembly.connections)}")
            return True
        except Exception as e:
            logger.error(f"[NoodleApp] Failed to load assembly: {e}")
            return False

    async def run(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the loaded assembly with given input.

        Args:
            input_data: Input to pass to INCOMING node
            context: Optional execution context

        Returns:
            Dict with:
                - response: Final output from OUTGOING node
                - outputs: All facet outputs
                - stats: Execution statistics
                - error: Error message if failed
        """
        if self.assembly is None:
            return {
                'response': None,
                'error': 'No assembly loaded',
                'outputs': {},
                'stats': {}
            }

        await self._ensure_executor()

        # Build context
        exec_context = {
            'agent_id': 'runtime',
            'agent_name': self.project_config.name if self.project_config else 'NoodleApp',
            'agent_species': 'runtime',
        }
        if context:
            exec_context.update(context)

        try:
            result = await asyncio.wait_for(
                self.executor.execute(
                    self.assembly,
                    incoming_data=input_data,
                    context=exec_context
                ),
                timeout=self.config.timeout
            )

            # Emit output if handler registered
            if self._on_output and result.response:
                self._on_output(result.response)

            return {
                'response': result.response,
                'outputs': result.facet_outputs,
                'stats': {
                    'total_time': result.total_time,
                    'total_tokens': result.total_tokens,
                    'facets_executed': result.facets_executed,
                    'execution_id': result.execution_id
                },
                'error': None
            }

        except asyncio.TimeoutError:
            return {
                'response': None,
                'error': f'Execution timed out after {self.config.timeout}s',
                'outputs': {},
                'stats': {}
            }
        except Exception as e:
            logger.exception(f"[NoodleApp] Execution error: {e}")
            return {
                'response': None,
                'error': str(e),
                'outputs': {},
                'stats': {}
            }

    async def run_interactive(
        self,
        input_func: Callable[[], str] = input,
        output_func: Callable[[str], None] = print
    ):
        """
        Run in interactive REPL mode.

        Args:
            input_func: Function to get user input
            output_func: Function to display output
        """
        if self.assembly is None:
            output_func("[NoodleApp] No assembly loaded")
            return

        name = self.project_config.name if self.project_config else self.assembly.name
        output_func(f"[NoodleApp] Interactive mode with '{name}'")
        output_func("[NoodleApp] Type 'quit' or 'exit' to stop\n")

        while True:
            try:
                user_input = input_func("> ")

                if user_input.lower() in ('quit', 'exit', 'q'):
                    break

                if not user_input.strip():
                    continue

                result = await self.run(user_input)

                if result['error']:
                    output_func(f"[Error] {result['error']}")
                elif result['response']:
                    output_func(result['response'])
                else:
                    output_func("[No response]")

                output_func("")  # Blank line

            except KeyboardInterrupt:
                output_func("\n[NoodleApp] Interrupted")
                break
            except EOFError:
                break

    def on_output(self, callback: Callable[[str], None]):
        """Register callback for output events."""
        self._on_output = callback

    def on_event(self, callback: Callable[[Dict], None]):
        """Register callback for execution events."""
        self._on_event = callback

    def tick(self):
        """
        Main tick - call periodically from main loop.

        Updates world channels and stage director.
        """
        if self.world_channels:
            self.world_channels.tick()
        if self.director:
            self.director.tick()

    def tick_world(self):
        """
        Update world channels (call periodically from main loop).

        This advances simulation time and broadcasts updates if needed.
        Deprecated: Use tick() instead which also updates the director.
        """
        if self.world_channels:
            self.world_channels.tick()

    def set_world_time(self, time_str: str):
        """Set simulation time (e.g., '18:00')."""
        if self.world_channels:
            self.world_channels.set_time(time_str)

    def set_world_weather(self, **kwargs):
        """
        Set weather conditions.

        Args:
            temperature: Temperature in Fahrenheit
            conditions: clear, cloudy, partly_cloudy, rain, storm, snow, fog
            wind: calm, light_breeze, windy, gusty
            humidity: 0.0 to 1.0
        """
        if self.world_channels:
            self.world_channels.set_weather(**kwargs)

    def set_world_ambiance(self, mood: Optional[str] = None, energy: Optional[float] = None):
        """
        Set scene ambiance.

        Args:
            mood: calm, tense, joyful, melancholy, mysterious, etc.
            energy: 0.0 to 1.0
        """
        if self.world_channels:
            self.world_channels.set_ambiance(mood=mood, energy=energy)

    def trigger_world_event(
        self,
        event_type: str,
        source: str,
        description: str,
        **kwargs
    ):
        """
        Trigger a world event.

        Args:
            event_type: sound, visual, physical, social
            source: What caused the event
            description: Human-readable description
            **kwargs: Additional data (location, intensity, etc.)
        """
        if self.world_channels:
            self.world_channels.trigger_event(event_type, source, description, **kwargs)

    def get_world_context(self) -> Dict[str, Any]:
        """Get full world context (time, weather, ambiance)."""
        if self.world_channels:
            return self.world_channels.get_full_context()
        return {}

    # =========================================================================
    # Stage Director (Brenda)
    # =========================================================================

    def load_director(self, play_path: str) -> bool:
        """
        Load Brenda with a play script.

        Args:
            play_path: Path to .play.yaml file

        Returns:
            True if loaded successfully
        """
        self.director = BrendaDirector(self.channel_bus)
        return self.director.load_play(play_path)

    def start_performance(self):
        """Start the directed performance."""
        if self.director:
            self.director.start()

    def stop_performance(self):
        """Stop the directed performance."""
        if self.director:
            self.director.stop()

    def publish_user_input(self, text: str):
        """
        Publish user input to #user.input channel.

        This allows Brenda and other systems to react to user messages.

        Args:
            text: The user's message text
        """
        import time
        self.channel_bus.publish(
            "#user.input",
            ChannelMessage(
                channel="#user.input",
                from_noodling="user",
                timestamp=time.time(),
                payload={'text': text}
            )
        )

    def get_director_state(self) -> Dict[str, Any]:
        """Get current director/performance state."""
        if self.director:
            return self.director.get_state()
        return {}

    def get_play_info(self) -> Dict[str, Any]:
        """Get info about the loaded play."""
        if self.director:
            return self.director.get_play_info()
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get cumulative execution statistics."""
        if self.executor:
            return self.executor.get_statistics()
        return {}

    async def cleanup(self):
        """Clean up resources."""
        if self.director:
            self.director.stop()
            self.director = None

        if self.world_channels:
            self.world_channels.stop()
            self.world_channels = None

        if self.llm_client:
            await self.llm_client.close()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
