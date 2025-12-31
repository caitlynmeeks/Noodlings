"""
Utility Commands Mixin for cMUSH

Contains general utility commands:
- help: Show command reference
- quit/logout: Disconnect
- @shutdown: Gracefully stop server (admin)
- @yeet: Force disconnect user (admin)
- @withdrawn/@reengage: Agent self-protection status
- @lab: Double-blind affect testing

Author: cMUSH Project
Date: December 2025
"""

from typing import Dict
import time


class UtilityCommandsMixin:
    """Mixin providing utility commands for CommandParser."""

    async def cmd_help(self, user_id: str, args: str) -> Dict:
        """Show help."""
        lines = [
            "\ncMUSH Commands:",
            "=" * 40,
            "Movement: north, south, east, west, up, down (or n/s/e/w/u/d)",
            "Communication: say <text>, emote <action>, tell <user> <message>",
            "Shortcuts: \"<text> (say), :<action> (emote)",
            "Observation: look, inventory, who",
            "Manipulation: take <object>, drop <object>",
            "Building: @create <room|object> <name>, @describe <text>, @dig <dir> <name>",
            "Object: @setdesc <object> <desc>, @destroy <object> (use quotes for multi-word)",
            "Agent (users): @rez <name> [desc], @observe <name>, @me, @relationship <name>, @memory <name>, @agents",
            "Agent (self): @whoami, @setname <name>, @setdesc <description>",
            "Agent (admin): @remove <name>, @tpinvite <name>, @reset confirm, @yeet <user>",
            "Agent Tools: @think <thought>, @remember [date], @message <agent> <text>, @inbox",
            "Filesystem: @write <file> <content>, @read <file>, @ls [dir], @exec <command>",
            "Cognition: @cognition <agent>, @set_frequency <agent> <seconds>, @ruminate <agent>",
            "Consciousness: @enlighten <agent|-a> <on|off>, @status <agent>",
            "LLM: @model [model_name], @models, @maxservers [number]",
            "BRENDA: @brenda make <agent> <adjective>, @brenda write play <story>, @brenda plays list/start/stop/next",
            "Utility: help, quit"
        ]

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_quit(self, user_id: str, args: str) -> Dict:
        """Quit/logout."""
        return {
            'success': True,
            'output': 'Goodbye!',
            'events': [{'type': 'quit', 'user': user_id}]
        }

    async def cmd_shutdown(self, user_id: str, args: str) -> Dict:
        """Shutdown the noodleMUSH server."""
        if not self.server:
            return {
                'success': False,
                'output': 'ERROR: Server instance not available for shutdown.',
                'events': []
            }

        # Confirmation check
        if args.strip().lower() != 'confirm':
            return {
                'success': False,
                'output': (
                    'WARNING: This will shut down the entire noodleMUSH server!\n'
                    'All agents will be saved and stopped.\n'
                    'All users will be disconnected.\n\n'
                    'Type: @shutdown confirm'
                ),
                'events': []
            }

        # Trigger graceful shutdown
        import asyncio
        asyncio.create_task(self.server.shutdown())

        return {
            'success': True,
            'output': (
                'Initiating graceful shutdown...\n'
                'Saving all agent states...\n'
                'Server will shut down momentarily.'
            ),
            'events': [{
                'type': 'system',
                'text': 'Server is shutting down. All agents are being saved.'
            }]
        }

    async def cmd_yeet(self, user_id: str, args: str) -> Dict:
        """Forcibly disconnect a user (admin command)."""
        if not args:
            return {'success': False, 'output': 'Usage: @yeet <username>', 'events': []}

        target_name = args.strip().lower()

        # Find user by username
        target_id = None
        target_user = None
        for uid, user in self.world.users.items():
            if user.get('username', '').lower() == target_name:
                target_id = uid
                target_user = user
                break

        if not target_id:
            return {'success': False, 'output': f"User '{args.strip()}' not found.", 'events': []}

        # Remove user from all room occupants lists
        for room_id, room in self.world.rooms.items():
            if target_id in room.get('occupants', []):
                room['occupants'].remove(target_id)

        # Remove user from users dict
        if target_id in self.world.users:
            del self.world.users[target_id]

        # Persist changes
        self.world.save_all()

        # Create yeet event - server will handle disconnection
        return {
            'success': True,
            'output': f"Yeeting {target_user.get('username', target_id)} from the server...",
            'events': [{'type': 'yeet', 'user': target_id, 'username': target_user.get('username', target_id)}]
        }

    async def cmd_check_withdrawn(self, user_id: str, args: str) -> Dict:
        """
        Check if an agent has withdrawn from interacting with you or others.

        Usage: @withdrawn [agent_name]
        Examples:
          @withdrawn            - Check all withdrawn statuses
          @withdrawn Callie     - Check if Callie has withdrawn from anyone
        """
        if not args:
            # Show all withdrawn states
            output_lines = ["Agent Withdrawal Status\n"]

            found_any = False
            for agent_id in self.agent_manager.agents:
                agent = self.agent_manager.get_agent(agent_id)
                if agent and hasattr(agent, 'withdrawn_users') and agent.withdrawn_users:
                    found_any = True
                    agent_name = agent.agent_name
                    output_lines.append(f"\n{agent_name}:")

                    for withdrawn_user_id, timestamp in agent.withdrawn_users.items():
                        time_elapsed = time.time() - timestamp
                        minutes_ago = int(time_elapsed / 60)

                        user_name = withdrawn_user_id.replace('user_', '').replace('agent_', '').title()
                        output_lines.append(f"  - Withdrawn from {user_name} ({minutes_ago}m ago)")

            if not found_any:
                output_lines.append("No agents have withdrawn from any interactions.")

            output_lines.append("\n\nNote: Agents automatically re-engage after 5 minutes cooling off period.")
            output_lines.append("Use @reengage <agent_name> to manually reset an agent's withdrawn state.")

            return {
                'success': True,
                'output': '\n'.join(output_lines),
                'events': []
            }

        # Check specific agent
        agent_name = args.strip()
        agent = self.agent_manager.get_agent(agent_name)

        if not agent:
            return {
                'success': False,
                'output': f"ERROR: Agent '{agent_name}' not found",
                'events': []
            }

        output_lines = [f"{agent.agent_name}'s Withdrawal Status\n"]

        if not hasattr(agent, 'withdrawn_users') or not agent.withdrawn_users:
            output_lines.append(f"{agent.agent_name} is currently engaging with everyone.")
        else:
            output_lines.append(f"{agent.agent_name} has withdrawn from:")
            for withdrawn_user_id, timestamp in agent.withdrawn_users.items():
                time_elapsed = time.time() - timestamp
                minutes_ago = int(time_elapsed / 60)
                time_remaining = max(0, 5 - minutes_ago)

                user_name = withdrawn_user_id.replace('user_', '').replace('agent_', '').title()

                if time_remaining > 0:
                    output_lines.append(f"  - {user_name} (re-engages in {time_remaining}m)")
                else:
                    output_lines.append(f"  - {user_name} (ready to re-engage)")

        output_lines.append(f"\nThis is {agent.agent_name}'s self-protective boundary setting.")
        output_lines.append("It happens when they experience distress (negative affect).")

        return {
            'success': True,
            'output': '\n'.join(output_lines),
            'events': []
        }

    async def cmd_reengage(self, user_id: str, args: str) -> Dict:
        """
        Manually reset an agent's withdrawn state, allowing them to re-engage.

        Usage: @reengage <agent_name>
        Example: @reengage Callie
        """
        if not args:
            return {
                'success': False,
                'output': "Usage: @reengage <agent_name>\nExample: @reengage Callie",
                'events': []
            }

        agent_name = args.strip()
        agent = self.agent_manager.get_agent(agent_name)

        if not agent:
            return {
                'success': False,
                'output': f"ERROR: Agent '{agent_name}' not found",
                'events': []
            }

        if not hasattr(agent, 'withdrawn_users') or not agent.withdrawn_users:
            return {
                'success': True,
                'output': f"{agent.agent_name} has not withdrawn from anyone - no action needed.",
                'events': []
            }

        # Clear all withdrawn users
        withdrawn_count = len(agent.withdrawn_users)
        withdrawn_names = [uid.replace('user_', '').replace('agent_', '').title()
                          for uid in agent.withdrawn_users.keys()]

        agent.withdrawn_users.clear()

        return {
            'success': True,
            'output': (
                f"{agent.agent_name}'s boundaries have been reset\n\n"
                f"Cleared withdrawal from {withdrawn_count} user(s):\n"
                f"  {', '.join(withdrawn_names)}\n\n"
                f"{agent.agent_name} is now open to re-engagement.\n"
                f"Please treat them with kindness and respect."
            ),
            'events': [{'type': 'reengage', 'agent': agent_name, 'cleared_count': withdrawn_count}]
        }

    async def cmd_lab(self, user_id: str, args: str) -> Dict:
        """
        Lab Mode - Double-blind affect testing.

        Commands:
          @lab start [trials]    - Start test session (default 50 trials)
          @lab status            - Show current test status
          @lab choose <A|B|equal> - Record your choice
          @lab stop              - Stop test and save results
          @lab results           - Show final results

        Usage:
          @lab start 50          - Start 50-trial test
          say hello              - System runs dual cognition
          @lab choose B          - Choose response B
          (repeat...)
        """
        if not self.server:
            return {
                'success': False,
                'output': 'ERROR: Server instance not available.',
                'events': []
            }

        # Parse command
        parts = args.strip().split(maxsplit=1)
        if not parts:
            subcommand = 'help'
            subargs = ''
        else:
            subcommand = parts[0].lower()
            subargs = parts[1] if len(parts) > 1 else ''

        # Get or create lab session
        if not hasattr(self.server, 'lab_sessions'):
            self.server.lab_sessions = {}

        lab_session = self.server.lab_sessions.get(user_id)

        # Handle subcommands
        if subcommand == 'start':
            # Start new lab session
            if lab_session:
                return {
                    'success': False,
                    'output': 'Lab test already active. Use @lab stop to end current test.',
                    'events': []
                }

            # Parse trial count
            try:
                trials = int(subargs) if subargs else 50
                if trials < 1 or trials > 500:
                    raise ValueError("Trial count must be between 1 and 500")
            except ValueError as e:
                return {
                    'success': False,
                    'output': f'Invalid trial count: {e}\nUsage: @lab start [trials]',
                    'events': []
                }

            # Create lab session
            from lab_system import LabTestSession
            lab_session = LabTestSession(
                player_id=user_id,
                target_trials=trials,
                experiment_name='affect_test'
            )
            self.server.lab_sessions[user_id] = lab_session

            # Get server diagnostics
            import os
            pid = os.getpid()

            # Get WebSocket port from config
            ws_port = self.config.get('server', {}).get('port', 8765) if self.config else 8765

            output = f"""
Lab Mode ACTIVE: Double-blind affect testing [v2.0 - EMOTE ENABLED]
Server: PID {pid} on ws://localhost:{ws_port}
Target: {trials} trials | Current: 0/{trials}

How it works:
1. Send messages to agents as normal
2. System runs dual cognition (real vs random affect)
3. Choose which response is better (A or B)
4. Repeat until {trials} trials complete

Your messages will be intercepted for testing.
Use @lab stop to end early.
"""
            return {
                'success': True,
                'output': output,
                'events': []
            }

        elif subcommand == 'status':
            # Show status
            if not lab_session:
                return {
                    'success': False,
                    'output': 'No lab test active. Use @lab start [trials] to begin.',
                    'events': []
                }

            status = lab_session.get_status()
            output = f"""
Lab Test Status
Session ID: {status['session_id']}
Progress: {status['trials_completed']}/{status['target_trials']} trials
Win Rate: {status['win_rate']:.1f}%

Results:
  Real affect preferred:   {status['real_preferred']}
  Random affect preferred: {status['random_preferred']}
  No preference:           {status['equal']}

{('Awaiting your choice...' if status['awaiting_choice'] else 'Ready for next trial')}
"""
            return {
                'success': True,
                'output': output,
                'events': []
            }

        elif subcommand == 'choose':
            # Record choice
            if not lab_session:
                return {
                    'success': False,
                    'output': 'No lab test active. Use @lab start [trials] to begin.',
                    'events': []
                }

            if not subargs:
                return {
                    'success': False,
                    'output': 'Please specify choice: @lab choose A, @lab choose B, or @lab choose equal',
                    'events': []
                }

            choice = subargs.strip().upper()
            if choice not in ['A', 'B', 'EQUAL']:
                return {
                    'success': False,
                    'output': 'Invalid choice. Use: @lab choose A, @lab choose B, or @lab choose equal',
                    'events': []
                }

            # Record choice (async)
            async def broadcast_to_user(message):
                """Helper to broadcast messages to user."""
                if hasattr(self.server, 'broadcast_to_user'):
                    await self.server.broadcast_to_user(user_id, message)

            await lab_session.record_choice(choice, self.world, broadcast_to_user)

            # Check if test complete
            if lab_session.trials_completed >= lab_session.target_trials:
                # Clean up session
                del self.server.lab_sessions[user_id]

            return {
                'success': True,
                'output': '',  # Output already sent by record_choice
                'events': []
            }

        elif subcommand == 'stop':
            # Stop test
            if not lab_session:
                return {
                    'success': False,
                    'output': 'No lab test active.',
                    'events': []
                }

            # Save partial results
            status = lab_session.get_status()
            output_path = lab_session._save_results(time.time() - lab_session.start_time)

            output = f"""
Lab test stopped.
Completed {status['trials_completed']}/{lab_session.target_trials} trials.
Win rate: {status['win_rate']:.1f}%

Results saved to: {output_path}
"""

            # Clean up session
            del self.server.lab_sessions[user_id]

            return {
                'success': True,
                'output': output,
                'events': []
            }

        elif subcommand == 'results':
            # Show results
            if not lab_session:
                return {
                    'success': False,
                    'output': 'No lab test active.',
                    'events': []
                }

            status = lab_session.get_status()
            trials_completed = status['trials_completed']
            real_pct = (status['real_preferred'] / trials_completed * 100) if trials_completed > 0 else 0
            random_pct = (status['random_preferred'] / trials_completed * 100) if trials_completed > 0 else 0

            output = f"""
Lab Test Results (In Progress)
Trials: {trials_completed}/{lab_session.target_trials}
Win Rate: {status['win_rate']:.1f}%

Real Affect Preferred:   {status['real_preferred']} ({real_pct:.1f}%)
Random Affect Preferred: {status['random_preferred']} ({random_pct:.1f}%)
No Preference:           {status['equal']}
"""
            return {
                'success': True,
                'output': output,
                'events': []
            }

        else:
            # Help
            output = """
Lab Mode - Double-Blind Affect Testing

Commands:
  @lab start [trials]     - Start test session (default 50)
  @lab status             - Show current test status
  @lab choose <A|B|equal> - Record your choice
  @lab stop               - Stop test and save results
  @lab results            - Show results summary

How it works:
1. Start a test: @lab start 50
2. Send messages to agents normally
3. System presents two responses (A and B)
4. Choose which is better: @lab choose A (or B or equal)
5. Repeat until test complete

The system tests whether real affect prediction improves
agent responses compared to random affect vectors.
"""
            return {
                'success': True,
                'output': output,
                'events': []
            }
