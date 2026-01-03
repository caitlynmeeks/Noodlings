"""
NoodleStudio Runtime CLI - Command-line interface for running projects

Usage:
    python -m noodlestudio.runtime path/to/project
    python -m noodlestudio.runtime --assembly path/to/assembly.yaml
    python -m noodlestudio.runtime path/to/project --interactive

Author: Caitlyn + Claude
Date: January 3, 2026
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .app import NoodleApp, NoodleAppConfig


def setup_logging(verbose: bool = False):
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog='noodlestudio.runtime',
        description="NoodleStudio Runtime - Execute NoodleStudio projects headlessly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run project interactively
  python -m noodlestudio.runtime path/to/project --interactive

  # Run assembly with single input
  python -m noodlestudio.runtime --assembly agent.yaml --input "Hello"

  # Run project with custom LLM provider
  python -m noodlestudio.runtime path/to/project \\
      --provider anthropic --api-key $ANTHROPIC_API_KEY

  # Pipe input/output
  echo "Hello" | python -m noodlestudio.runtime --assembly agent.yaml

Environment Variables:
  NOODLE_LLM_PROVIDER    LLM provider (ollama, anthropic, openai, openrouter)
  NOODLE_LLM_MODEL       Default model name
  NOODLE_LLM_BASE_URL    Custom API base URL
  ANTHROPIC_API_KEY      Anthropic API key
  OPENAI_API_KEY         OpenAI API key
  OPENROUTER_API_KEY     OpenRouter API key
        """
    )

    # Positional argument for project path
    parser.add_argument(
        'project',
        nargs='?',
        help='Path to NoodleStudio project directory'
    )

    # Alternative: direct assembly path
    parser.add_argument(
        '--assembly', '-a',
        help='Path to facet assembly YAML file (alternative to project)'
    )

    # Input handling
    parser.add_argument(
        '--input', '-i',
        help='Input text (reads from stdin if not provided and not interactive)'
    )

    parser.add_argument(
        '--interactive', '-I',
        action='store_true',
        help='Run in interactive REPL mode'
    )

    # LLM configuration
    parser.add_argument(
        '--provider', '-p',
        default='ollama',
        choices=['ollama', 'anthropic', 'openai', 'openrouter'],
        help='LLM provider (default: ollama)'
    )

    parser.add_argument(
        '--model', '-m',
        default='',
        help='Model name (uses project/assembly defaults if not specified)'
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

    # Execution settings
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=120.0,
        help='Execution timeout in seconds (default: 120)'
    )

    # Output options
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

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output'
    )

    return parser


async def run_cli(args: argparse.Namespace) -> int:
    """
    Run the CLI with parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    # Validate arguments
    if not args.project and not args.assembly:
        print("Error: Either project path or --assembly must be provided", file=sys.stderr)
        return 1

    # Create config
    config = NoodleAppConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        llm_api_key=args.api_key,
        llm_base_url=args.base_url,
        timeout=args.timeout,
        verbose=args.verbose
    )

    # Create app
    app = NoodleApp(config)

    try:
        # Load project or assembly
        if args.assembly:
            if not app.load_assembly(args.assembly):
                print(f"Error: Failed to load assembly: {args.assembly}", file=sys.stderr)
                return 1
            if not args.quiet:
                print(f"Loaded assembly: {app.assembly.name}", file=sys.stderr)
        else:
            if not app.load_project(args.project):
                print(f"Error: Failed to load project: {args.project}", file=sys.stderr)
                return 1
            if not args.quiet:
                print(f"Loaded project: {app.project_config.name}", file=sys.stderr)

        # Run in appropriate mode
        if args.interactive:
            await app.run_interactive()
            return 0

        # Get input
        if args.input:
            input_text = args.input
        elif not sys.stdin.isatty():
            input_text = sys.stdin.read().strip()
        else:
            print("Error: No input provided. Use --input, pipe stdin, or use --interactive",
                  file=sys.stderr)
            return 1

        # Execute
        result = await app.run(input_text)

        # Output
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            if result['error']:
                print(f"Error: {result['error']}", file=sys.stderr)
                return 1
            elif result['response']:
                print(result['response'])
            else:
                if not args.quiet:
                    print("[No response]")

        # Print stats if verbose
        if args.verbose and result.get('stats'):
            stats = result['stats']
            print(f"\n--- Execution Stats ---", file=sys.stderr)
            print(f"Time: {stats.get('total_time', 0):.2f}s", file=sys.stderr)
            print(f"Tokens: {stats.get('total_tokens', 0)}", file=sys.stderr)
            print(f"Facets: {stats.get('facets_executed', 0)}", file=sys.stderr)

        return 0

    finally:
        await app.cleanup()


def main():
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        logging.disable(logging.CRITICAL)
    else:
        setup_logging(args.verbose)

    # Run async main
    try:
        exit_code = asyncio.run(run_cli(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
