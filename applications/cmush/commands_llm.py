# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   LLM Control Commands
#
#   These commands let you configure which AI model powers
#   the Noodlings' language abilities. You can switch between
#   local Ollama models and cloud providers.
#
#   Commands:
#     @model               -> See what model is being used
#     @model qwen3-32b     -> Switch to a different model
#     @models              -> List all available models
#     @maxservers 4        -> Run up to 4 LLM queries in parallel
#
#   This affects all Noodlings in the world. Different models
#   have different personalities and capabilities.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.commands_llm
# PURPOSE:  LLM model configuration commands
# LAYER:    Backend / Commands
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   LLMCommandsMixin    Model, models, maxservers
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
LLM Control Commands Mixin for cMUSH

Contains commands for managing LLM configuration:
- @model: Change or view current LLM model
- @models: List available models
- @maxservers: Configure parallel LLM instances

Author: Caitlyn + Claude
Date: December 2025
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)


class LLMCommandsMixin:
    """Mixin providing LLM control commands for CommandParser."""

    async def cmd_set_model(self, user_id: str, args: str) -> Dict:
        """Change the LLM model being used by all agents."""
        if not args:
            # Show current model
            current_model = self.agent_manager.llm.get_model()
            return {
                'success': True,
                'output': (
                    f"Current LLM model: {current_model}\n\n"
                    "Usage: @model <model_name>\n"
                    "Example: @model qwen3-32b-128k@q8_0\n\n"
                    "To see available models: @models"
                ),
                'events': []
            }

        new_model = args.strip()

        # Change model for all agents
        self.agent_manager.llm.set_model(new_model)

        # Save to config.yaml for persistence across sessions
        if self.config and self.config_path:
            self.config['llm']['model'] = new_model
            self._save_config()
            persistence_msg = "\nModel saved to config.yaml (will persist across sessions)"
        else:
            persistence_msg = "\nWarning: Model not saved to config (will reset on restart)"

        return {
            'success': True,
            'output': (
                f"LLM model changed to: {new_model}\n\n"
                "This affects all agents immediately.\n"
                f"Previous conversations remain in memory.{persistence_msg}"
            ),
            'events': []
        }

    async def cmd_list_models(self, user_id: str, args: str) -> Dict:
        """List available models from LMStudio."""
        try:
            models = await self.agent_manager.llm.list_models()

            if not models:
                return {
                    'success': True,
                    'output': (
                        "No models found or LMStudio didn't respond.\n\n"
                        "Make sure LMStudio is running and has models loaded."
                    ),
                    'events': []
                }

            current_model = self.agent_manager.llm.get_model()

            lines = ["\nAvailable LLM Models"]
            lines.append("=" * 50)
            lines.append("")

            for model in models:
                if model == current_model:
                    lines.append(f"  > {model} (current)")
                else:
                    lines.append(f"    {model}")

            lines.append("")
            lines.append(f"Total models: {len(models)}")
            lines.append("")
            lines.append("To switch models: @model <model_name>")

            return {
                'success': True,
                'output': '\n'.join(lines),
                'events': []
            }

        except Exception as e:
            logger.error(f"Error listing models: {e}", exc_info=True)
            return {
                'success': False,
                'output': f"Error listing models: {str(e)}\n\nMake sure LMStudio is running.",
                'events': []
            }

    async def cmd_set_maxservers(self, user_id: str, args: str) -> Dict:
        """Configure the maximum number of parallel LLM instances (model:0, model:1, etc.)."""
        if not args:
            # Show current settings
            current = self.agent_manager.llm.get_maxservers()
            instance_mode = self.agent_manager.llm.get_use_model_instances()

            mode_str = "ENABLED (using model:N pattern)" if instance_mode else "DISABLED (single model)"

            return {
                'success': True,
                'output': (
                    "Current parallel instance settings:\n\n"
                    f"  Max concurrent instances: {current}\n"
                    f"  Model instance mode: {mode_str}\n\n"
                    "Usage: @maxservers <number>\n"
                    "Example: @maxservers 1  (legacy single-instance mode)\n"
                    "Example: @maxservers 5  (default: 5 parallel instances)\n\n"
                    "Note: Requires LMStudio with 'Enable JIT model loading' enabled\n"
                    f"      Uses round-robin: model:0, model:1, ..., model:{current-1}"
                ),
                'events': []
            }

        try:
            new_max = int(args.strip())
            if new_max < 1:
                return {
                    'success': False,
                    'output': "Error: max_concurrent must be at least 1",
                    'events': []
                }

            # Update LLM client settings
            old_max = self.agent_manager.llm.get_maxservers()
            self.agent_manager.llm.set_maxservers(new_max)

            # Save to config.yaml for persistence
            if self.config and self.config_path:
                self.config['llm']['max_concurrent'] = new_max
                self._save_config()
                persistence_msg = "\nSetting saved to config.yaml (will persist across sessions)"
            else:
                persistence_msg = "\nWarning: Setting not saved to config (will reset on restart)"

            mode_str = (
                "model:0, model:1, ..., model:" + str(new_max - 1)
                if new_max > 1 else "single model (legacy mode)"
            )

            return {
                'success': True,
                'output': (
                    f"Max concurrent instances changed: {old_max} -> {new_max}\n\n"
                    f"Pattern: {mode_str}\n"
                    f"This affects all agents immediately.{persistence_msg}"
                ),
                'events': []
            }

        except ValueError:
            return {
                'success': False,
                'output': f"Error: '{args}' is not a valid number\n\nUsage: @maxservers <number>",
                'events': []
            }

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
