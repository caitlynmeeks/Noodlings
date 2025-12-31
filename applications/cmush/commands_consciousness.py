"""
Consciousness Commands Mixin for cMUSH

Contains commands for agent consciousness control:
- @enlighten: Toggle enlightenment mode
- @status: Show comprehensive agent status

Author: cMUSH Project
Date: December 2025
"""

from typing import Dict


class ConsciousnessCommandsMixin:
    """Mixin providing consciousness commands for CommandParser."""

    async def cmd_enlighten(self, user_id: str, args: str) -> Dict:
        """Toggle enlightenment mode for an agent (allow meta-discussion of phenomenal states)."""
        if not args:
            return {
                'success': False,
                'output': (
                    "Usage: @enlighten <agent_name> <on|off>\n"
                    "       @enlighten -a <on|off>\n\n"
                    "Examples:\n"
                    "  @enlighten Callie on   - Enlighten specific agent\n"
                    "  @enlighten -a on       - Enlighten all agents\n"
                    "  @enlighten -a off      - Return all agents to character immersion"
                ),
                'events': []
            }

        parts = args.strip().split()
        if len(parts) != 2:
            return {
                'success': False,
                'output': "Usage: @enlighten <agent_name> <on|off>",
                'events': []
            }

        target, mode = parts
        mode = mode.lower()

        if mode not in ['on', 'off']:
            return {'success': False, 'output': "Mode must be 'on' or 'off'", 'events': []}

        new_state = (mode == 'on')

        # Handle -a flag for all agents
        if target == '-a':
            agents_dict = self.agent_manager.agents
            if not agents_dict:
                return {'success': False, 'output': "No agents currently active.", 'events': []}

            updated_agents = []
            for agent_id, agent in agents_dict.items():
                agent.config['enlightenment'] = new_state
                agent_name = agent_id.replace('agent_', '')
                updated_agents.append(agent_name)

            await self._broadcast_agent_list_to_all()

            action = "Enlightened" if new_state else "Returned to immersion"
            agent_list = ", ".join(updated_agents)
            return {
                'success': True,
                'output': f"{action} all agents: {agent_list}",
                'events': []
            }

        # Handle single agent
        agent_name = target
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        agent.config['enlightenment'] = new_state
        await self._broadcast_agent_list_to_all()

        if new_state:
            output = f"{agent_name} is now enlightened (can discuss phenomenal states meta-cognitively)"
        else:
            output = f"{agent_name} returned to character immersion"

        return {'success': True, 'output': output, 'events': []}

    async def _broadcast_agent_list_to_all(self):
        """Broadcast updated agent list to all connected clients (for star updates)."""
        agent_list = []
        for agent_id, agent in self.agent_manager.agents.items():
            agent_list.append({
                'id': agent_id,
                'name': agent.agent_name,
                'enlightened': agent.config.get('enlightenment', False)
            })

        for ws in self.server.connections.keys():
            await self.server.send_to_user(ws, {
                'type': 'agents',
                'agents': agent_list
            })

    async def cmd_comprehensive_status(self, user_id: str, args: str) -> Dict:
        """Show comprehensive status for an agent including enlightenment, appetites, goals, and more."""
        if not args:
            return {
                'success': False,
                'output': "Usage: @status <agent_name>\nExample: @status Callie",
                'events': []
            }

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        lines = []
        lines.append(f"COMPREHENSIVE STATUS: {agent.agent_name.upper()}")
        lines.append("=" * 60)
        lines.append("")

        # Identity & Enlightenment
        enlightenment = agent.config.get('enlightenment', False)
        enlightenment_status = "ENLIGHTENED" if enlightenment else "IMMERSED"
        lines.append(f"Name: {agent.agent_name}")
        lines.append(f"ID: {agent.agent_id}")
        lines.append(f"Enlightenment: {enlightenment_status}")
        lines.append(f"Current Room: {agent.current_room}")
        lines.append("")

        # Appetites (Phase 6)
        if hasattr(agent, 'get_appetites'):
            try:
                appetites = agent.get_appetites()
                lines.append("APPETITES:")
                for name, value in appetites.items():
                    bar = "#" * int(value * 20) + "-" * (20 - int(value * 20))
                    lines.append(f"  {name:14s}: [{bar}] {value:.2f}")
                lines.append("")
            except Exception:
                lines.append("APPETITES: Error loading")
        else:
            lines.append("APPETITES: Phase 6 not available")
        lines.append("")

        # Goals (Phase 6)
        if hasattr(agent, 'get_goal_overrides'):
            try:
                overrides = agent.get_goal_overrides()
                biases = agent.get_goal_biases()
                if overrides or biases:
                    lines.append("GOAL ORCHESTRATION:")
                    if overrides:
                        lines.append("  Overrides:")
                        for goal, strength in overrides.items():
                            lines.append(f"    {goal}: {strength:.2f}")
                    if biases:
                        lines.append("  Biases:")
                        for goal, bias in biases.items():
                            lines.append(f"    {goal}: {bias:+.2f}")
                else:
                    lines.append("GOAL ORCHESTRATION: None active")
            except Exception:
                lines.append("GOAL ORCHESTRATION: Error loading")
        else:
            lines.append("GOAL ORCHESTRATION: Phase 6 not available")
        lines.append("")

        # Cognition engine stats
        if agent.cognition_engine:
            stats = agent.cognition_engine.get_stats()
            lines.append("COGNITION ENGINE:")
            lines.append(f"  Running: {'Yes' if stats['running'] else 'No'}")
            lines.append(f"  Cognitive Pressure: {stats['cognitive_pressure']:.2f}")
            lines.append(f"  Speech Urgency: {stats['speech_urgency']:.2f}")
        lines.append("")

        # Self-protection
        if hasattr(agent, 'withdrawn_users') and agent.withdrawn_users:
            lines.append("SELF-PROTECTION:")
            lines.append(f"  Withdrawn from {len(agent.withdrawn_users)} user(s)")
        else:
            lines.append("SELF-PROTECTION: No withdrawals")

        return {'success': True, 'output': '\n'.join(lines), 'events': []}
