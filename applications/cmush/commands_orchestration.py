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
#   Orchestration Commands (Phase 6)
#
#   These are BRENDA's tools for directing Noodlings like a
#   theater director. You can adjust their appetites (internal
#   drives) and set goals to make stories happen.
#
#   Appetite commands:
#     @stoke chester curiosity 0.8  -> Make Chester curious
#     @sate chester hunger 0.0      -> Chester's not hungry
#     @appetites chester            -> See all appetites
#
#   Goal commands:
#     @override chester "Find the treasure"
#     @bias chester "explore" 0.7
#     @goals chester               -> List Chester's goals
#
#   This is the storyteller's toolkit for emergent narrative.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.commands_orchestration
# PURPOSE:  Phase 6 appetite and goal control
# LAYER:    Backend / Commands
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   OrchestrationCommandsMixin    Stoke, sate, override, bias
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Orchestration Commands Mixin for cMUSH

Contains Phase 6 appetite and goal orchestration commands:
- @stoke/@sate/@appetites: Appetite control
- @override/@bias/@goals: Goal orchestration
- @reset_goals/@clear_bias: Clear orchestration

Author: Caitlyn + Claude
Date: December 2025
"""

from typing import Dict


class OrchestrationCommandsMixin:
    """Mixin providing Phase 6 orchestration commands for CommandParser."""

    async def cmd_stoke_appetite(self, user_id: str, args: str) -> Dict:
        """Increase an agent's appetite (Phase 6 feature)."""
        if not args:
            return {
                'success': True,
                'output': (
                    "Brenda's Appetite Orchestration - @stoke\n"
                    "=" * 50 + "\n\n"
                    "Increase an agent's internal drive/appetite.\n\n"
                    "Usage: @stoke <agent_name> <appetite> <amount>\n\n"
                    "Example: @stoke Toad novelty 0.3\n\n"
                    "Available Appetites:\n"
                    "  curiosity, status, mastery, novelty,\n"
                    "  safety, social_bond, comfort, autonomy\n\n"
                    "Amount: 0.0-1.0"
                ),
                'events': []
            }

        parts = args.split()
        if len(parts) < 3:
            return {
                'success': False,
                'output': "Usage: @stoke <agent_name> <appetite> <amount>",
                'events': []
            }

        agent_name, appetite_name = parts[0], parts[1].lower()

        try:
            amount = float(parts[2])
            if not 0 <= amount <= 1.0:
                return {'success': False, 'output': "Amount must be between 0.0 and 1.0", 'events': []}
        except ValueError:
            return {'success': False, 'output': f"Invalid amount: {parts[2]}", 'events': []}

        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not hasattr(agent, 'stoke_appetite'):
            return {
                'success': True,
                'output': f"Phase 6 not available for {agent_name}. Uses Phase 4.",
                'events': []
            }

        agent.stoke_appetite(appetite_name, amount)
        return {
            'success': True,
            'output': f"{agent_name}'s {appetite_name} appetite increased by {amount:.2f}",
            'events': [{'type': 'appetite_change', 'agent': agent_name, 'appetite': appetite_name, 'change': amount}]
        }

    async def cmd_sate_appetite(self, user_id: str, args: str) -> Dict:
        """Satisfy/decrease an agent's appetite (Phase 6 feature)."""
        if not args:
            return {
                'success': True,
                'output': (
                    "Brenda's Appetite Orchestration - @sate\n"
                    "=" * 50 + "\n\n"
                    "Satisfy/decrease an agent's internal drive.\n\n"
                    "Usage: @sate <agent_name> <appetite> <amount>\n\n"
                    "See @stoke for list of appetites."
                ),
                'events': []
            }

        parts = args.split()
        if len(parts) < 3:
            return {'success': False, 'output': "Usage: @sate <agent_name> <appetite> <amount>", 'events': []}

        agent_name, appetite_name = parts[0], parts[1].lower()

        try:
            amount = float(parts[2])
            if not 0 <= amount <= 1.0:
                return {'success': False, 'output': "Amount must be between 0.0 and 1.0", 'events': []}
        except ValueError:
            return {'success': False, 'output': f"Invalid amount: {parts[2]}", 'events': []}

        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not hasattr(agent, 'sate_appetite'):
            return {'success': True, 'output': f"Phase 6 not available for {agent_name}.", 'events': []}

        agent.sate_appetite(appetite_name, amount)
        return {
            'success': True,
            'output': f"{agent_name}'s {appetite_name} appetite decreased by {amount:.2f}",
            'events': [{'type': 'appetite_change', 'agent': agent_name, 'appetite': appetite_name, 'change': -amount}]
        }

    async def cmd_show_appetites(self, user_id: str, args: str) -> Dict:
        """Show an agent's current appetite levels (Phase 6 feature)."""
        if not args:
            return {'success': False, 'output': "Usage: @appetites <agent_name>", 'events': []}

        agent_name = args.strip()
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not hasattr(agent, 'get_appetites'):
            return {'success': True, 'output': f"Phase 6 not available for {agent_name}.", 'events': []}

        appetites = agent.get_appetites()
        lines = [f"{agent_name}'s Appetites (Phase 6)", "=" * 50, ""]

        for name in ["curiosity", "status", "mastery", "novelty", "safety", "social_bond", "comfort", "autonomy"]:
            value = appetites.get(name, 0.0)
            bar = "#" * int(value * 20) + "-" * (20 - int(value * 20))
            lines.append(f"  {name:12s} [{bar}] {value:.2f}")

        return {'success': True, 'output': '\n'.join(lines), 'events': []}

    async def cmd_override_goal(self, user_id: str, args: str) -> Dict:
        """Override an agent's goal activation (Phase 6 feature)."""
        if not args:
            return {
                'success': True,
                'output': (
                    "Brenda's Goal Orchestration - @override\n"
                    "=" * 50 + "\n\n"
                    "Directly override an agent's goal activation.\n\n"
                    "Usage: @override <agent_name> <goal> <strength>\n\n"
                    "Strength: 0.0-1.0"
                ),
                'events': []
            }

        parts = args.split()
        if len(parts) < 3:
            return {'success': False, 'output': "Usage: @override <agent_name> <goal> <strength>", 'events': []}

        agent_name, goal_name = parts[0], parts[1].lower()

        try:
            strength = float(parts[2])
            if not 0 <= strength <= 1.0:
                return {'success': False, 'output': "Strength must be between 0.0 and 1.0", 'events': []}
        except ValueError:
            return {'success': False, 'output': f"Invalid strength: {parts[2]}", 'events': []}

        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not hasattr(agent, 'override_goal'):
            return {'success': True, 'output': f"Phase 6 not available for {agent_name}.", 'events': []}

        agent.override_goal(goal_name, strength)
        return {
            'success': True,
            'output': f"{agent_name}'s goal '{goal_name}' overridden to {strength:.2f}",
            'events': [{'type': 'goal_override', 'agent': agent_name, 'goal': goal_name, 'strength': strength}]
        }

    async def cmd_set_goal_bias(self, user_id: str, args: str) -> Dict:
        """Add a persistent bias to an agent's goal generation (Phase 6 feature)."""
        if not args:
            return {
                'success': True,
                'output': (
                    "Brenda's Goal Orchestration - @bias\n"
                    "=" * 50 + "\n\n"
                    "Add a subtle, persistent bias to goal generation.\n\n"
                    "Usage: @bias <agent_name> <goal> <bias>\n\n"
                    "Bias: -1.0 to 1.0"
                ),
                'events': []
            }

        parts = args.split()
        if len(parts) < 3:
            return {'success': False, 'output': "Usage: @bias <agent_name> <goal> <bias>", 'events': []}

        agent_name, goal_name = parts[0], parts[1].lower()

        try:
            bias = float(parts[2])
            if not -1.0 <= bias <= 1.0:
                return {'success': False, 'output': "Bias must be between -1.0 and 1.0", 'events': []}
        except ValueError:
            return {'success': False, 'output': f"Invalid bias: {parts[2]}", 'events': []}

        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not hasattr(agent, 'set_goal_bias'):
            return {'success': True, 'output': f"Phase 6 not available for {agent_name}.", 'events': []}

        agent.set_goal_bias(goal_name, bias)
        return {
            'success': True,
            'output': f"{agent_name}'s '{goal_name}' bias set to {bias:+.2f}",
            'events': [{'type': 'goal_bias', 'agent': agent_name, 'goal': goal_name, 'bias': bias}]
        }

    async def cmd_reset_goals(self, user_id: str, args: str) -> Dict:
        """Clear goal overrides for an agent (Phase 6 feature)."""
        if not args:
            return {'success': False, 'output': "Usage: @reset_goals <agent_name> [goal]", 'events': []}

        parts = args.split()
        agent_name = parts[0]
        goal_name = parts[1].lower() if len(parts) > 1 else None

        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not hasattr(agent, 'clear_goal_overrides'):
            return {'success': True, 'output': f"Phase 6 not available for {agent_name}.", 'events': []}

        agent.clear_goal_overrides(goal_name)
        if goal_name:
            output = f"Cleared goal override for '{goal_name}' on {agent_name}."
        else:
            output = f"Cleared all goal overrides for {agent_name}."

        return {'success': True, 'output': output, 'events': [{'type': 'goal_reset', 'agent': agent_name, 'goal': goal_name}]}

    async def cmd_clear_bias(self, user_id: str, args: str) -> Dict:
        """Clear goal biases for an agent (Phase 6 feature)."""
        if not args:
            return {'success': False, 'output': "Usage: @clear_bias <agent_name> [goal]", 'events': []}

        parts = args.split()
        agent_name = parts[0]
        goal_name = parts[1].lower() if len(parts) > 1 else None

        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not hasattr(agent, 'clear_goal_biases'):
            return {'success': True, 'output': f"Phase 6 not available for {agent_name}.", 'events': []}

        agent.clear_goal_biases(goal_name)
        if goal_name:
            output = f"Cleared goal bias for '{goal_name}' on {agent_name}."
        else:
            output = f"Cleared all goal biases for {agent_name}."

        return {'success': True, 'output': output, 'events': [{'type': 'bias_reset', 'agent': agent_name, 'goal': goal_name}]}

    async def cmd_show_goals(self, user_id: str, args: str) -> Dict:
        """Show an agent's current goal activations, overrides, and biases (Phase 6 feature)."""
        if not args:
            return {'success': False, 'output': "Usage: @goals <agent_name>", 'events': []}

        agent_name = args.strip()
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not hasattr(agent, 'get_goal_overrides'):
            return {'success': True, 'output': f"Phase 6 not available for {agent_name}.", 'events': []}

        overrides = agent.get_goal_overrides()
        biases = agent.get_goal_biases()

        lines = [f"{agent_name}'s Goal State (Phase 6)", "=" * 50, ""]

        if overrides:
            lines.append("Active Overrides:")
            for goal, strength in overrides.items():
                bar = "#" * int(strength * 20) + "-" * (20 - int(strength * 20))
                lines.append(f"  {goal:25s} [{bar}] {strength:.2f}")
            lines.append("")
        else:
            lines.append("Active Overrides: None\n")

        if biases:
            lines.append("Active Biases:")
            for goal, bias in biases.items():
                lines.append(f"  {goal:25s} {bias:+.2f}")
        else:
            lines.append("Active Biases: None")

        return {'success': True, 'output': '\n'.join(lines), 'events': []}

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
