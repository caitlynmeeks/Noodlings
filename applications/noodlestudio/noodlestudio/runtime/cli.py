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
#   NoodleStudio Runtime CLI - Command-line interface for running projects
#
#   Usage: python -m noodlestudio.runtime path/to/project pyt...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.cli
# PURPOSE:  Cli
# LAYER:    Studio / Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   setup_logging(), create_parser(), run_gui()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

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

  # Run project with GUI window
  python -m noodlestudio.runtime path/to/project --gui

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

    parser.add_argument(
        '--gui', '-g',
        action='store_true',
        help='Run with GUI window (renders ui.yaml canvas)'
    )

    parser.add_argument(
        '--ui',
        default='',
        help='Path to ui.yaml file (default: project/ui.yaml)'
    )

    parser.add_argument(
        '--play',
        default='',
        help='Path to .play.yaml file for Brenda stage direction'
    )

    parser.add_argument(
        '--window-size', '-w',
        default='1024x768',
        help='Window size as WxH (default: 1024x768)'
    )

    # LLM configuration
    parser.add_argument(
        '--provider', '-p',
        default='ollama',
        choices=['ollama', 'anthropic', 'openai', 'openrouter', 'noodlerouter'],
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


def _build_config_to_splash_config(build_config, project_path: Path) -> dict:
    """
    Convert BuildConfig.splash to SplashScreen config dict.

    Args:
        build_config: BuildConfig instance
        project_path: Path to project directory for resolving relative paths

    Returns:
        Config dict for SplashScreen
    """
    splash = build_config.splash
    identity = build_config.identity

    # Resolve image path
    image_path = None
    if splash.image:
        img = project_path / splash.image
        if img.exists():
            image_path = str(img)

    # Map attribution position from build config to splash screen format
    # build_config uses: bottom_right, bottom_left, bottom_center
    # SplashScreen uses: bottom-right, bottom-left, bottom-center
    attr_pos = splash.attribution_position.replace('_', '-')

    return {
        'title': identity.name,
        'image': image_path,
        'background': splash.background,
        'duration': splash.duration,
        'fade_in': splash.fade_in,
        'fade_out': splash.fade_out,
        'click_to_skip': splash.click_to_dismiss,
        'show_loading': True,
        'loading_style': 'dots',
        'attribution': {
            'position': attr_pos,
            'style': 'badge',
            'show_nec_link': True,  # Always required
        }
    }


def _apply_build_config_llm_settings(args: argparse.Namespace, build_config) -> None:
    """
    Apply LLM settings from build.yaml to args if CLI didn't override.

    build.yaml llm.provider values:
      - noodlerouter: Use NoodleROUTER API
      - user_keys: User provides own Anthropic/OpenAI keys
      - ollama: Local Ollama models
      - bundled: Use bundled API key (not recommended)

    Args:
        args: Parsed command-line arguments (modified in place)
        build_config: BuildConfig loaded from build.yaml
    """
    import os

    if build_config is None or not hasattr(build_config, 'llm'):
        return

    llm = build_config.llm

    # Map build.yaml provider to runtime provider
    provider_map = {
        'noodlerouter': 'noodlerouter',
        'user_keys': None,  # Will detect from env
        'ollama': 'ollama',
        'bundled': 'noodlerouter',  # Bundled key uses noodlerouter
    }

    build_provider = llm.provider if hasattr(llm, 'provider') else 'noodlerouter'

    # Only apply if user didn't explicitly set --provider (check if it's default)
    # We detect "explicit" by checking if it's the parser default 'ollama'
    # and build.yaml specifies something different
    if args.provider == 'ollama' and build_provider != 'ollama':
        mapped_provider = provider_map.get(build_provider)

        if build_provider == 'user_keys':
            # Detect which provider the user has keys for
            if os.environ.get('ANTHROPIC_API_KEY'):
                args.provider = 'anthropic'
            elif os.environ.get('OPENAI_API_KEY'):
                args.provider = 'openai'
            elif os.environ.get('OPENROUTER_API_KEY'):
                args.provider = 'openrouter'
            else:
                # No keys found - will need to prompt user or fail gracefully
                print("Warning: user_keys mode but no API keys found in environment", file=sys.stderr)
                print("Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY", file=sys.stderr)
        elif mapped_provider:
            args.provider = mapped_provider

        # Handle bundled key
        if build_provider == 'bundled' and hasattr(llm, 'bundled_key') and llm.bundled_key:
            if not args.api_key:  # Only if CLI didn't provide key
                args.api_key = llm.bundled_key


def run_gui(args: argparse.Namespace) -> int:
    """
    Run the GUI application.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow
        from PyQt6.QtCore import Qt
    except ImportError:
        print("Error: PyQt6 is required for GUI mode", file=sys.stderr)
        return 1

    from .ui import (
        load_ui, create_default_ui, QtWidgetRenderer,
        AnchoredWidget, UIEventDispatcher
    )
    from .ui.overlay import CharacterOverlayWindow
    from ..widgets.splash_screen import SplashScreen

    # Parse window size
    try:
        width, height = map(int, args.window_size.lower().split('x'))
    except ValueError:
        print(f"Error: Invalid window size format: {args.window_size}", file=sys.stderr)
        print("Expected format: WIDTHxHEIGHT (e.g., 1024x768)", file=sys.stderr)
        return 1

    # Determine UI file path
    if args.ui:
        ui_path = Path(args.ui)
    elif args.project:
        ui_path = Path(args.project) / 'ui.yaml'
    else:
        ui_path = None

    # Load or create UI
    if ui_path and ui_path.exists():
        try:
            root = load_ui(ui_path)
            if not args.quiet:
                print(f"Loaded UI: {ui_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading UI: {e}", file=sys.stderr)
            return 1
    else:
        root = create_default_ui()
        if not args.quiet:
            print("Using default UI", file=sys.stderr)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("NoodleStudio Runtime")

    # Create main window
    window = QMainWindow()
    window.setWindowTitle("NoodleStudio Runtime")
    window.resize(width, height)

    # Render UI components
    renderer = QtWidgetRenderer()

    # Create event dispatcher and wire to renderer
    dispatcher = UIEventDispatcher(renderer)
    renderer.set_event_dispatcher(dispatcher.dispatch)
    dispatcher.root_component = root  # For FacetAssembly lookups

    # Derive project path from ui.yaml location if --project not provided
    # This allows running with just --ui path/to/ui.yaml
    if args.ui and not args.project:
        args.project = str(Path(args.ui).resolve().parent)
        if not args.quiet:
            print(f"Derived project path from UI: {args.project}", file=sys.stderr)

    # Set project path for relative path resolution in assemblies and assets
    if args.project:
        project_path = str(Path(args.project).resolve())
        dispatcher.set_project_path(project_path)
        renderer.set_project_path(project_path)
        if not args.quiet:
            print(f"Project path: {args.project}", file=sys.stderr)

    # Load build.yaml early to configure LLM settings before executor init
    build_config = None
    project_path_obj = Path(args.project).resolve() if args.project else None

    if project_path_obj:
        build_yaml_path = project_path_obj / 'build.yaml'
        if build_yaml_path.exists():
            try:
                from ..core.build_config import BuildConfig
                build_config = BuildConfig.from_yaml(build_yaml_path)
                if not args.quiet:
                    print(f"Loaded build config: {build_yaml_path}", file=sys.stderr)

                # Apply LLM settings from build.yaml to args
                _apply_build_config_llm_settings(args, build_config)
                if not args.quiet and build_config.llm.provider != 'ollama':
                    print(f"LLM provider from build.yaml: {build_config.llm.provider}", file=sys.stderr)

            except Exception as e:
                print(f"Warning: Failed to load build.yaml: {e}", file=sys.stderr)

    # Set up facet executor for run_assembly actions
    if args.project:
        try:
            from .app import NoodleApp, NoodleAppConfig
            import asyncio
            import threading

            config = NoodleAppConfig(
                llm_provider=args.provider,
                llm_model=args.model,
                llm_api_key=args.api_key,
                llm_base_url=args.base_url,
                verbose=args.verbose
            )
            noodle_app = NoodleApp(config)
            noodle_app.project_path = Path(args.project).resolve()

            # Initialize executor in a separate thread with its own event loop
            init_error = None

            def init_in_thread():
                nonlocal init_error
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(noodle_app._ensure_executor())
                except Exception as e:
                    init_error = e
                finally:
                    loop.close()

            thread = threading.Thread(target=init_in_thread)
            thread.start()
            thread.join(timeout=10)  # Wait up to 10 seconds

            if init_error:
                raise init_error

            if noodle_app.executor:
                dispatcher.set_facet_executor(noodle_app.executor)
                if not args.quiet:
                    print("Facet executor initialized", file=sys.stderr)
            else:
                print("Warning: Facet executor not initialized", file=sys.stderr)

            # Wire app to dispatcher for Brenda direction injection
            dispatcher.set_app(noodle_app)

            # Load play script if provided
            if args.play:
                play_path = Path(args.play)
                if not play_path.is_absolute():
                    play_path = Path(args.project).resolve() / args.play
                if play_path.exists():
                    if noodle_app.load_director(str(play_path)):
                        noodle_app.start_performance()
                        if not args.quiet:
                            play_info = noodle_app.get_play_info()
                            print(f"Loaded play: {play_info.get('title', 'Unknown')}", file=sys.stderr)
                            print(f"  Beats: {play_info.get('beat_count', 0)}", file=sys.stderr)
                            print(f"  Starting beat: {noodle_app.get_director_state().get('current_beat_id')}", file=sys.stderr)
                    else:
                        print(f"Warning: Failed to load play: {play_path}", file=sys.stderr)
                else:
                    print(f"Warning: Play file not found: {play_path}", file=sys.stderr)

        except Exception as e:
            print(f"Warning: Failed to initialize facet executor: {e}", file=sys.stderr)

    # Use AnchoredWidget for proper resize handling
    anchored_widget = AnchoredWidget(root, renderer)
    window.setCentralWidget(anchored_widget)

    # Splash screen setup (build_config already loaded above for LLM settings)
    splash_screen = None

    # Determine if splash screen should be shown
    show_splash = (
        build_config is not None and
        build_config.splash.enabled and
        project_path_obj is not None
    )

    # Prepare character overlay config (will be created after splash or directly)
    character_overlay = None
    overlay_config_data = None
    if ui_path and ui_path.exists():
        try:
            import yaml
            with open(ui_path, 'r') as f:
                ui_data = yaml.safe_load(f)
            overlay_config_data = ui_data.get('overlay', {})
        except Exception as e:
            print(f"Warning: Failed to read overlay config: {e}", file=sys.stderr)

    def show_main_window():
        """Show main window and character overlay."""
        nonlocal character_overlay
        window.show()

        if not args.quiet:
            print(f"Window: {width}x{height}", file=sys.stderr)

        # Create character overlay if configured
        if overlay_config_data and overlay_config_data.get('enabled', False):
            try:
                vrm_path = overlay_config_data.get('vrm_path', '')

                # Resolve relative path from ui.yaml location
                if vrm_path and not Path(vrm_path).is_absolute():
                    vrm_path = str((ui_path.parent / vrm_path).resolve())

                if vrm_path and Path(vrm_path).exists():
                    size = tuple(overlay_config_data.get('size', [300, 400]))
                    offset = tuple(overlay_config_data.get('offset', [20, 100]))
                    anchor = overlay_config_data.get('anchor', 'right')

                    character_overlay = CharacterOverlayWindow(
                        parent_window=window,
                        vrm_path=vrm_path,
                        size=size,
                        offset=offset,
                        anchor=anchor
                    )
                    character_overlay.show()

                    if not args.quiet:
                        print(f"Character overlay: {vrm_path}", file=sys.stderr)
                else:
                    if not args.quiet:
                        print(f"Warning: VRM file not found: {vrm_path}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to create character overlay: {e}", file=sys.stderr)

    if show_splash:
        # Create and show splash screen
        splash_config = _build_config_to_splash_config(build_config, project_path_obj)
        splash_screen = SplashScreen(splash_config)

        if not args.quiet:
            image_info = f" (image: {build_config.splash.image})" if build_config.splash.image else ""
            print(f"Showing splash screen{image_info}", file=sys.stderr)

        # Show splash, then show main window when complete
        splash_screen.show_splash(on_complete=show_main_window)
    else:
        # No splash - show window directly
        show_main_window()

    # Run event loop
    return app.exec()


async def run_cli(args: argparse.Namespace) -> int:
    """
    Run the CLI with parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    # Handle GUI mode
    if args.gui:
        return run_gui(args)

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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
