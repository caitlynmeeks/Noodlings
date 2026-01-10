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
#   Cognition Control Commands
#
#   These commands let you peek into and tune how Noodlings
#   think. Each Noodling has a cognition engine that runs in
#   the background, processing thoughts and deciding when
#   to speak or act.
#
#   Commands:
#     @cognition chester   -> See Chester's thinking stats
#     @set_frequency ...   -> How often does he ruminate?
#     @ruminate chester    -> Force him to think right now
#
#   This is useful for debugging why a Noodling is too chatty,
#   too quiet, or seems to be thinking about the wrong things.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.commands_cognition
# PURPOSE:  Cognition engine control commands
# LAYER:    Backend / Commands
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   CognitionCommandsMixin    Stats, frequency, rumination
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Cognition Control Commands Mixin for cMUSH

Contains commands for managing agent cognitive processes:
- @cognition: Show cognition engine statistics
- @set_frequency: Set rumination interval
- @ruminate: Force immediate rumination

Author: Caitlyn + Claude
Date: December 2025
"""

from typing import Dict


class CognitionCommandsMixin:
    """Mixin providing cognition control commands for CommandParser."""

    async def cmd_cognition_stats(self, user_id: str, args: str) -> Dict:
        """Show agent's autonomous cognition statistics."""
        if not args:
            return {'success': False, 'output': 'Usage: @cognition <agent_name>', 'events': []}

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not agent.cognition_engine:
            return {'success': True, 'output': f"Agent '{agent_name}' has no cognition engine.", 'events': []}

        stats = agent.cognition_engine.get_stats()

        lines = [f"\nCognition Stats for {agent_name}:"]
        lines.append("=" * 60)
        lines.append(f"Running: {'Yes' if stats['running'] else 'No'}")
        lines.append(f"Thoughts Buffered: {stats['thoughts_buffered']}")
        lines.append(f"Cognitive Pressure: {stats['cognitive_pressure']:.2f} / 1.0")
        lines.append(f"Time Since Last Speech: {stats['time_since_speech']:.0f}s")
        lines.append(f"Speech Urgency: {stats['speech_urgency']:.2f} (threshold: {agent.cognition_engine.speech_urgency_threshold:.2f})")
        lines.append(f"Wake Interval: {agent.cognition_engine.wake_interval}s")
        lines.append(f"Min Speech Interval: {agent.cognition_engine.min_speech_interval}s")

        # Show personality traits (cognition engine personality, not recipe personality)
        personality = agent.cognition_engine.personality
        lines.append("\nCognition Personality Traits:")
        lines.append(f"  Extraversion: {personality['extraversion']:.2f} (affects chattiness)")
        lines.append(f"  Emotional Sensitivity: {personality['emotional_sensitivity']:.2f}")
        lines.append(f"  Curiosity: {personality['curiosity']:.2f}")
        lines.append(f"  Spontaneity: {personality['spontaneity']:.2f}")
        lines.append(f"  Reflection Depth: {personality['reflection_depth']:.2f}")
        lines.append(f"  Social Orientation: {personality['social_orientation']:.2f}")

        # Predict when next speech might occur
        if stats['speech_urgency'] >= agent.cognition_engine.speech_urgency_threshold:
            if stats['time_since_speech'] >= agent.cognition_engine.min_speech_interval:
                lines.append("\nAgent is ready to speak spontaneously!")
            else:
                time_until = agent.cognition_engine.min_speech_interval - stats['time_since_speech']
                lines.append(f"\nReady to speak in ~{time_until:.0f}s")
        else:
            pressure_needed = agent.cognition_engine.speech_urgency_threshold - stats['speech_urgency']
            lines.append(f"\nBuilding pressure... ({pressure_needed:.2f} more needed)")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_set_frequency(self, user_id: str, args: str) -> Dict:
        """Set agent's rumination frequency for this session."""
        if not args:
            return {'success': False, 'output': 'Usage: @set_frequency <agent_name> <seconds>', 'events': []}

        parts = args.split()
        if len(parts) < 2:
            return {'success': False, 'output': 'Usage: @set_frequency <agent_name> <seconds>', 'events': []}

        agent_name, freq = parts[0], parts[1]
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        try:
            frequency = int(freq)
            if frequency < 5:
                return {'success': False, 'output': 'Frequency must be at least 5 seconds.', 'events': []}
            if frequency > 600:
                return {'success': False, 'output': 'Frequency must be at most 600 seconds (10 minutes).', 'events': []}
        except ValueError:
            return {'success': False, 'output': 'Frequency must be a number.', 'events': []}

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not agent.cognition_engine:
            return {'success': True, 'output': f"Agent '{agent_name}' has no cognition engine.", 'events': []}

        # Update frequency
        old_freq = agent.cognition_engine.wake_interval
        agent.cognition_engine.wake_interval = frequency

        return {
            'success': True,
            'output': f"Updated {agent_name}'s rumination frequency: {old_freq}s -> {frequency}s\n" +
                     f"Agent will now think every {frequency} seconds.",
            'events': []
        }

    async def cmd_force_rumination(self, user_id: str, args: str) -> Dict:
        """Force an agent to ruminate immediately and broadcast the result."""
        if not args:
            return {'success': False, 'output': 'Usage: @ruminate <agent_name>', 'events': []}

        agent_name = args.strip()
        agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{agent_name}' not found.", 'events': []}

        if not agent.cognition_engine:
            return {'success': True, 'output': f"Agent '{agent_name}' has no cognition engine.", 'events': []}

        # Broadcast rumination indicator
        room = agent.current_room
        rumination_event = {
            'type': 'emote',
            'user': agent_id,
            'username': agent.agent_name,
            'room': room,
            'text': f"* {agent.agent_name} closes their eyes, lost in thought... *",
            'metadata': {'rumination': True}
        }

        # Force rumination
        try:
            thoughts = await agent.cognition_engine._ruminate()

            # Show results
            if thoughts:
                thought_text = "\n  - ".join(thoughts)
                output = f"{agent_name} ruminated and thought:\n  - {thought_text}"
            else:
                output = f"{agent_name} ruminated but generated no thoughts."

            return {
                'success': True,
                'output': output,
                'events': [rumination_event]
            }
        except Exception as e:
            return {
                'success': False,
                'output': f"Error during rumination: {str(e)}",
                'events': [rumination_event]
            }

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
