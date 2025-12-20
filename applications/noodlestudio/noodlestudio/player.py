"""
NoodleStudio Player - Headless facet assembly execution runtime

Allows running facet assemblies without the GUI for:
- Command-line tools (like Toy Claude Code)
- Batch processing
- Integration into other applications
- Testing and automation

Usage:
    # As library:
    from noodlestudio.player import Player

    player = Player()
    player.load_assembly("my_agent.yaml")
    result = await player.run("Hello, world!")

    # From command line:
    python -m noodlestudio.player --assembly my_agent.yaml --input "Hello"

Author: Caitlyn + Claude
Date: December 20, 2025
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass


@dataclass
class PlayerConfig:
    """Configuration for Player runtime."""
    # LLM settings
    llm_provider: str = "ollama"  # ollama, anthropic, openai, openrouter
    llm_model: str = ""  # Model name/label (empty = use assembly defaults)
    llm_api_key: str = ""  # API key (if needed)
    llm_base_url: str = ""  # Custom base URL (for local servers)

    # Execution settings
    max_iterations: int = 10  # Max ReAct-style iterations
    timeout: float = 120.0  # Execution timeout in seconds
    verbose: bool = False  # Print execution details

    # MCP settings
    mcp_config_path: str = ""  # Path to MCP servers config


class Player:
    """
    Headless runtime for executing facet assemblies.

    This is a lightweight alternative to running NoodleStudio for
    executing cognitive architectures in production or from CLI.
    """

    def __init__(self, config: Optional[PlayerConfig] = None):
        """
        Initialize Player runtime.

        Args:
            config: Optional configuration. Defaults will be used if not provided.
        """
        self.config = config or PlayerConfig()
        self.assembly = None
        self.executor = None
        self.llm_client = None

        # Event handlers
        self._on_output: Optional[Callable[[str], None]] = None
        self._on_event: Optional[Callable[[Dict], None]] = None

    async def _ensure_llm_client(self):
        """Initialize LLM client if not already done."""
        if self.llm_client is not None:
            return

        # Import LLM router (lazy import to avoid Qt dependency at import time)
        try:
            from .core.llm_client_router import LLMClientRouter, ProviderConfig
        except ImportError:
            # Fallback for standalone use
            from noodlestudio.core.llm_client_router import LLMClientRouter, ProviderConfig

        # Configure provider
        provider_config = ProviderConfig(
            provider=self.config.llm_provider,
            api_key=self.config.llm_api_key or None,
            base_url=self.config.llm_base_url or None
        )

        self.llm_client = LLMClientRouter(provider_config)

        if self.config.verbose:
            print(f"[Player] Initialized LLM client: {self.config.llm_provider}")

    async def _ensure_executor(self):
        """Initialize facet executor if not already done."""
        if self.executor is not None:
            return

        await self._ensure_llm_client()

        # Import executor (lazy import)
        try:
            from .core.facet_executor import FacetExecutor
        except ImportError:
            from noodlestudio.core.facet_executor import FacetExecutor

        # Create executor with LLM client
        self.executor = FacetExecutor(
            llm_client=self.llm_client,
            use_event_bus=False,  # No event bus in headless mode
            concurrency_mode='serial'  # Simpler for headless
        )

        if self.config.verbose:
            print(f"[Player] Initialized facet executor")

    def load_assembly(self, path: str) -> bool:
        """
        Load a facet assembly from YAML file.

        Args:
            path: Path to assembly YAML file

        Returns:
            True if loaded successfully
        """
        try:
            from .core.facet_system import FacetAssembly
        except ImportError:
            from noodlestudio.core.facet_system import FacetAssembly

        assembly_path = Path(path)
        if not assembly_path.exists():
            print(f"[Player] Assembly not found: {path}")
            return False

        try:
            self.assembly = FacetAssembly.load_yaml(str(assembly_path))
            if self.config.verbose:
                print(f"[Player] Loaded assembly: {self.assembly.name}")
                print(f"[Player]   Facets: {len(self.assembly.facets)}")
                print(f"[Player]   Connections: {len(self.assembly.connections)}")
            return True
        except Exception as e:
            print(f"[Player] Failed to load assembly: {e}")
            return False

    def load_assembly_from_dict(self, data: Dict[str, Any]) -> bool:
        """
        Load a facet assembly from dictionary.

        Args:
            data: Assembly data dictionary

        Returns:
            True if loaded successfully
        """
        try:
            from .core.facet_system import FacetAssembly
        except ImportError:
            from noodlestudio.core.facet_system import FacetAssembly

        try:
            self.assembly = FacetAssembly.from_dict(data)
            if self.config.verbose:
                print(f"[Player] Loaded assembly: {self.assembly.name}")
            return True
        except Exception as e:
            print(f"[Player] Failed to load assembly: {e}")
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
            context: Optional execution context (agent info, etc.)

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
            'agent_id': 'player',
            'agent_name': 'Player',
            'agent_species': 'player',
        }
        if context:
            exec_context.update(context)

        try:
            # Execute assembly
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
            input_func: Function to get user input (default: input())
            output_func: Function to display output (default: print())
        """
        if self.assembly is None:
            output_func("[Player] No assembly loaded")
            return

        output_func(f"[Player] Interactive mode with '{self.assembly.name}'")
        output_func("[Player] Type 'quit' or 'exit' to stop\n")

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
                output_func("\n[Player] Interrupted")
                break
            except EOFError:
                break

    def on_output(self, callback: Callable[[str], None]):
        """Register callback for output events."""
        self._on_output = callback

    def on_event(self, callback: Callable[[Dict], None]):
        """Register callback for execution events."""
        self._on_event = callback

    def get_statistics(self) -> Dict[str, Any]:
        """Get cumulative execution statistics."""
        if self.executor:
            return self.executor.get_statistics()
        return {}


async def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="NoodleStudio Player - Execute facet assemblies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run assembly with input
  python -m noodlestudio.player --assembly agent.yaml --input "Hello"

  # Interactive mode
  python -m noodlestudio.player --assembly agent.yaml --interactive

  # Use specific LLM provider
  python -m noodlestudio.player --assembly agent.yaml \\
      --provider anthropic --api-key $ANTHROPIC_API_KEY

  # Pipe input/output
  echo "Hello" | python -m noodlestudio.player --assembly agent.yaml
        """
    )

    parser.add_argument(
        '--assembly', '-a',
        required=True,
        help='Path to facet assembly YAML file'
    )
    parser.add_argument(
        '--input', '-i',
        help='Input text (reads from stdin if not provided and not interactive)'
    )
    parser.add_argument(
        '--interactive', '-I',
        action='store_true',
        help='Run in interactive REPL mode'
    )
    parser.add_argument(
        '--provider', '-p',
        default='ollama',
        help='LLM provider: ollama, anthropic, openai, openrouter (default: ollama)'
    )
    parser.add_argument(
        '--model', '-m',
        default='',
        help='Model name/label (uses assembly defaults if not specified)'
    )
    parser.add_argument(
        '--api-key', '-k',
        default='',
        help='API key for LLM provider'
    )
    parser.add_argument(
        '--base-url', '-u',
        default='',
        help='Custom base URL for LLM provider'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=120.0,
        help='Execution timeout in seconds (default: 120)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose execution details'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output result as JSON'
    )

    args = parser.parse_args()

    # Create config
    config = PlayerConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        llm_api_key=args.api_key,
        llm_base_url=args.base_url,
        timeout=args.timeout,
        verbose=args.verbose
    )

    # Create player
    player = Player(config)

    # Load assembly
    if not player.load_assembly(args.assembly):
        sys.exit(1)

    # Run in appropriate mode
    if args.interactive:
        await player.run_interactive()
    else:
        # Get input
        if args.input:
            input_text = args.input
        elif not sys.stdin.isatty():
            input_text = sys.stdin.read().strip()
        else:
            print("Error: No input provided. Use --input or pipe stdin, or use --interactive")
            sys.exit(1)

        # Execute
        result = await player.run(input_text)

        # Output
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result['error']:
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)
            elif result['response']:
                print(result['response'])
            else:
                print("[No response]")


if __name__ == '__main__':
    asyncio.run(main())
