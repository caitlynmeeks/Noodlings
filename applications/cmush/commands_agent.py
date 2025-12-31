"""
Agent Commands Mixin for cMUSH

Contains commands for agent lifecycle and observation:
- @rez: Spawn agents (Noodlings, prims)
- @remove/@derez: Remove agents
- @reset: Reset world state
- @observe: View agent internal state
- @agents: List active agents
- @savestates: Save agent states

Author: cMUSH Project
Date: December 2025
"""

from typing import Dict
import logging
import re

from fuzzy_match import format_disambiguation_prompt

logger = logging.getLogger(__name__)


class AgentCommandsMixin:
    """Mixin providing agent lifecycle commands for CommandParser."""

    async def cmd_rez_agent(self, user_id: str, args: str) -> Dict:
        """Unified rez command - spawns Noodlings, prims, directions, ensembles."""
        if not args:
            return {
                'success': False,
                'output': (
                    'Usage: @rez [-f] [-e] <agent_name> [agent_name2 ...] [description]\n'
                    '       @rez -p <type> "<name>" [script:<ScriptName>]\n'
                    '       @rez -d <direction> "<room_name>" (TODO)\n'
                    '       @rez -e "<ensemble_name>" (TODO)\n\n'
                    'Options:\n'
                    '  -f    Force fresh state (skip loading saved phenomenal state)\n'
                    '  -e    Enable enlightenment (agent is self-aware and metacognitive)\n'
                    '  -p    Rez prim (prop, furniture, vending_machine, etc.)\n'
                    '  -d    Rez direction/exit (TODO)\n'
                    '  -e    Rez ensemble (TODO)\n\n'
                    'Available recipes:\n' +
                    '\n'.join(f'  - {name}' for name in self.recipe_loader.list_recipes()) +
                    '\n\nExamples:\n'
                    '  @rez phi\n'
                    '  @rez -f phi        (fresh rez, ignores saved state)\n'
                    '  @rez -e phi        (spawn with enlightenment)\n'
                    '  @rez phi toad callie\n'
                    '  @rez -p vending_machine "Anklebiter Dispenser" script:AnklebiterVendingMachine\n'
                    '  @rez -p prop "Magic Sword"\n'
                ),
                'events': []
            }

        # Check for -p flag (rez prim)
        if args.startswith('-p '):
            return await self._rez_prim(user_id, args[3:].strip())

        # Check for -f flag (force fresh state)
        skip_phenomenal_state = False
        if args.startswith('-f ') or args == '-f':
            skip_phenomenal_state = True
            args = args[2:].strip()
            if not args:
                return {'success': False, 'output': 'Error: No agent name provided after -f flag.', 'events': []}

        # Check for -e flag (enable enlightenment)
        enlightenment = False
        if args.startswith('-e ') or args == '-e':
            enlightenment = True
            args = args[2:].strip()
            if not args:
                return {'success': False, 'output': 'Error: No agent name provided after -e flag.', 'events': []}

        # Parse agent names - support quoted names for multi-word agents
        import shlex
        try:
            parts = shlex.split(args)
        except ValueError:
            parts = args.split()

        agent_names = []
        description_parts = []

        # Collect agent names (assume they're recipe names or simple names)
        for i, part in enumerate(parts):
            part_lower = part.lower()
            if self.recipe_loader.load_recipe(part_lower) or (i == 0 or part_lower.replace(' ', '_').replace('-', '_').isalpha()):
                if i < 3:  # Limit to first 3 words as potential agent names
                    agent_names.append(part_lower.replace(' ', '_'))
                else:
                    description_parts = parts[i:]
                    break
            else:
                description_parts = parts[i:]
                break

        if not agent_names:
            agent_names = [parts[0].lower().replace(' ', '_')]
            description_parts = parts[1:] if len(parts) > 1 else []

        agent_description = ' '.join(description_parts) if description_parts else None

        # Spawn each agent
        all_events = []
        rezzed_agents = []
        errors = []

        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        import uuid as uuid_lib
        import random

        for agent_name in agent_names:
            agent_id = f"agent_{uuid_lib.uuid4()}"

            # Check if agent with this NAME already exists
            existing = [aid for aid, a in self.world.agents.items()
                       if a.get('name', '').lower() == agent_name.lower()]
            if existing:
                errors.append(f"'{agent_name}' already exists")
                continue

            recipe = self.recipe_loader.load_recipe(agent_name)

            if recipe:
                display_name = recipe.name
                description = recipe.description if not agent_description else agent_description

                sm_config = self.config['agent'].get('self_monitoring', {})
                logger.debug(f"[SPAWN] agent_id={agent_id}, self_monitoring={sm_config}")

                config = {
                    'appetites': recipe.get_appetite_baselines(),
                    'identity_prompt': recipe.identity_prompt,
                    'species': recipe.species,
                    'language_mode': recipe.language_mode,
                    'temperature': recipe.temperature,
                    'max_tokens': recipe.max_tokens,
                    'enforce_action_format': recipe.enforce_action_format,
                    'response_cooldown': recipe.response_cooldown,
                    'enlightenment': enlightenment if enlightenment else recipe.enlightenment,
                    'self_monitoring': sm_config,
                    'affective_reinforcement': recipe.affective_reinforcement or {},
                    'facet_assembly': recipe.facet_assembly
                }

                arrival_phrases = [
                    "steps into the scene",
                    "ambles into view",
                    "appears round the bend",
                    "wanders in from the riverbank",
                    "pops up cheerfully"
                ]
                arrival = random.choice(arrival_phrases)

                rez_msg = f"{display_name} ({recipe.species}) {arrival}"
                if recipe.language_mode == 'nonverbal':
                    rez_msg += ", watching curiously with bright eyes"
                rez_msg += f". {description}"
            else:
                recipe = self.recipe_loader.get_default_recipe()
                display_name = agent_name.capitalize()
                description = agent_description if agent_description else recipe.description

                sm_config = self.config['agent'].get('self_monitoring', {})
                config = {
                    'appetites': recipe.get_appetite_baselines(),
                    'identity_prompt': recipe.identity_prompt,
                    'species': recipe.species,
                    'language_mode': recipe.language_mode,
                    'temperature': recipe.temperature,
                    'max_tokens': recipe.max_tokens,
                    'enforce_action_format': recipe.enforce_action_format,
                    'response_cooldown': recipe.response_cooldown,
                    'enlightenment': enlightenment if enlightenment else recipe.enlightenment,
                    'self_monitoring': sm_config,
                    'affective_reinforcement': recipe.affective_reinforcement or {},
                    'facet_assembly': recipe.facet_assembly,
                    'cognitive_components': {}
                }

                arrival_phrases = [
                    "steps into the scene",
                    "ambles into view",
                    "appears round the bend",
                    "wanders in from somewhere",
                    "shows up with a friendly wave"
                ]
                arrival = random.choice(arrival_phrases)
                rez_msg = f"{display_name} ({recipe.species}) {arrival}. {description}"

            # Create agent in world
            default_checkpoint = "../../models/checkpoints/best_checkpoint.npz"
            checkpoint_path = recipe.checkpoint if (recipe and recipe.checkpoint) else default_checkpoint

            self.world.create_agent(
                name=agent_name,
                checkpoint_path=checkpoint_path,
                spawn_room=room['uid'],
                config=config
            )

            # Initialize agent in manager
            await self.agent_manager.create_agent(
                agent_id=agent_id,
                checkpoint_path=checkpoint_path,
                spawn_room=room['uid'],
                agent_name=display_name,
                agent_description=description,
                config=config,
                skip_phenomenal_state=skip_phenomenal_state
            )

            rezzed_agents.append(display_name)

            all_events.append({
                'type': 'enter',
                'user': agent_id,
                'room': room['uid'],
                'text': rez_msg
            })

            # on_rezzed() - Dynamic arrival reaction
            try:
                agent = self.agent_manager.get_agent(agent_id)
                if agent:
                    occupants = []
                    for occ_id in room.get('occupants', []):
                        if occ_id == agent_id:
                            continue
                        if occ_id.startswith('agent_'):
                            occ_agent = self.agent_manager.get_agent(occ_id)
                            if occ_agent:
                                occupants.append(f"{occ_agent['name']} ({occ_agent.get('config', {}).get('species', 'being')})")
                        elif occ_id.startswith('user_'):
                            user = self.world.get_user(occ_id)
                            if user:
                                occupants.append(f"{user.get('name', occ_id)} (person)")

                    room_desc = room.get('description', 'a mysterious clearing')

                    if occupants:
                        occupant_list = ", ".join(occupants)
                        perception = f"You find yourself in {room_desc}. Present: {occupant_list}."
                    else:
                        perception = f"You find yourself alone in {room_desc}."

                    spawn_perception = {
                        'type': 'perception',
                        'user': 'world',
                        'text': perception,
                        'room': room['uid']
                    }

                    await agent.perceive_event(spawn_perception)

            except Exception as e:
                logger.error(f"[ON_REZZED] Failed for {agent_id}: {e}")

        # Build result message
        if rezzed_agents:
            if len(rezzed_agents) == 1:
                output_msg = f"Rezzed '{rezzed_agents[0]}'."
            else:
                noodling_word = "Noodlings"
                output_msg = f"Rezzed {len(rezzed_agents)} {noodling_word}: {', '.join(rezzed_agents)}."

            flags = []
            if skip_phenomenal_state:
                flags.append("fresh state")
            if enlightenment:
                flags.append("enlightenment enabled")
            if flags:
                output_msg += f" ({', '.join(flags)})"

            if errors:
                output_msg += f"\nErrors: {', '.join(errors)}"

            return {
                'success': True,
                'output': output_msg,
                'events': all_events
            }
        else:
            return {
                'success': False,
                'output': f"Failed to rez Noodlings. Errors: {', '.join(errors)}",
                'events': []
            }

    async def _rez_prim(self, user_id: str, args: str) -> Dict:
        """Helper method for @rez -p (rez prim)."""
        if not args:
            return {
                'success': False,
                'output': (
                    'Usage: @rez -p <type> "<name>" [script:<ScriptName>]\n'
                    'Types: prop, furniture, container, vending_machine, etc.\n'
                    'Example: @rez -p vending_machine "Anklebiter Dispenser" script:AnklebiterVendingMachine'
                ),
                'events': []
            }

        match = re.match(r'^(\w+)\s+"([^"]+)"(?:\s+script:(\w+))?$', args)

        if not match:
            return {
                'success': False,
                'output': 'Invalid syntax. Use: @rez -p <type> "<name>" [script:<ScriptName>]',
                'events': []
            }

        prim_type = match.group(1)
        name = match.group(2)
        script_name = match.group(3)

        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'Error getting location.', 'events': []}

        obj_id = self.world.create_object(
            name=name,
            description=f"A {prim_type} prim.",
            owner=user_id,
            location=room['uid'],
            obj_type=prim_type,
            script=script_name
        )

        if script_name and self.script_manager:
            success = self.script_manager.attach_script(obj_id, script_name)
            if success:
                output = f"Rezzed {prim_type} '{name}' ({obj_id}) with script '{script_name}'"
            else:
                output = f"Rezzed {prim_type} '{name}' ({obj_id}) but script '{script_name}' failed to attach"
        else:
            output = f"Rezzed {prim_type} '{name}' ({obj_id})"

        return {
            'success': True,
            'output': output,
            'events': []
        }

    async def cmd_remove(self, user_id: str, args: str) -> Dict:
        """Remove an agent from the world."""
        if not args:
            return {
                'success': False,
                'output': 'Usage: @remove [-s] <agent_name>\n  -s: Silent removal (no departure message)',
                'events': []
            }

        silent = False
        if args.startswith('-s ') or args == '-s':
            silent = True
            args = args[2:].strip()
            if not args:
                return {'success': False, 'output': 'Error: No agent name provided after -s flag.', 'events': []}

        import shlex
        try:
            parts = shlex.split(args)
            query = parts[0] if parts else args.strip()
        except ValueError:
            query = args.strip()

        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'You are nowhere.', 'events': []}

        entity_id, entity_type, ambiguous = self._resolve_entity(query, room['uid'], include_objects=False, include_users=False)

        if ambiguous:
            return {
                'success': False,
                'output': format_disambiguation_prompt(query, ambiguous),
                'events': []
            }

        if not entity_id or entity_type != 'agent':
            return {'success': False, 'output': f"Agent '{query}' not found.", 'events': []}

        agent_id = entity_id
        agent_data = self.world.get_user(agent_id)
        if not agent_data:
            return {'success': False, 'output': f"Agent '{query}' not found.", 'events': []}

        agent_name = agent_data.get('name', query)
        room = self.world.get_room(agent_data['current_room'])

        await self.agent_manager.remove_agent(agent_id)

        if agent_id in self.world.agents:
            del self.world.agents[agent_id]

        if room and agent_id in room.get('occupants', []):
            room['occupants'].remove(agent_id)

        self.world.save_all()

        events = []
        if not silent:
            import random
            departure_phrases = [
                "remembers something urgent and hurries off",
                "suddenly recalls an appointment and dashes away",
                "realizes they're expected elsewhere and scurries off",
                "gets that look of sudden remembering and trots away",
                "mutters about forgetting something and bustles off",
                "hears a distant call and wanders away",
                "decides it's time for a ramble and ambles off"
            ]
            departure = random.choice(departure_phrases)

            events.append({
                'type': 'exit',
                'user': agent_id,
                'username': agent_name,
                'room': room['uid'] if room else 'unknown',
                'text': f"{agent_name} {departure}."
            })
        else:
            events.append({
                'type': 'agent_removed',
                'user': agent_id,
                'username': agent_name,
                'room': room['uid'] if room else 'unknown',
                'silent': True
            })

        return {
            'success': True,
            'output': f"Derezzed '{agent_name}'{' silently' if silent else ''}.",
            'events': events
        }

    async def cmd_reset(self, user_id: str, args: str) -> Dict:
        """Reset the world to default settings (removes all agents and custom objects)."""
        clear_screen = False
        if args.strip().lower().startswith('-c '):
            clear_screen = True
            args = args.strip()[3:]

        if args.strip().lower() != 'confirm':
            return {
                'success': False,
                'output': 'WARNING: This will remove all agents and reset the world!\nType: @reset confirm (or @reset -c confirm to clear screen)',
                'events': []
            }

        agent_ids = list(self.world.agents.keys())
        for agent_id in agent_ids:
            await self.agent_manager.remove_agent(agent_id, delete_state=True)

        import os
        import shutil
        agents_dir = 'world/agents'
        if os.path.exists(agents_dir):
            for entry in os.listdir(agents_dir):
                if entry.startswith('agent_'):
                    state_path = os.path.join(agents_dir, entry)
                    if os.path.isdir(state_path):
                        shutil.rmtree(state_path)

        self.world.agents = {}
        self.world.objects = {}

        for room_id, room in self.world.rooms.items():
            room['occupants'] = [uid for uid in room.get('occupants', []) if not uid.startswith('agent_')]
            room['objects'] = []

        self.world.save_all()

        output_msg = 'World reset complete. All Noodlings derezzed, objects cleared.'

        reset_events = [{
            'type': 'system',
            'text': 'The world shimmers and resets to its original state.'
        }]

        if clear_screen:
            reset_events.append({
                'type': 'clear_screen',
                'user_id': user_id
            })

        return {
            'success': True,
            'output': output_msg,
            'events': reset_events
        }

    async def cmd_observe_agent(self, user_id: str, args: str) -> Dict:
        """Observe an agent's internal state."""
        if not args:
            return {'success': False, 'output': 'Usage: @observe <agent_name>', 'events': []}

        query = args.strip()

        room = self.world.get_user_room(user_id)
        if not room:
            return {'success': False, 'output': 'You are nowhere.', 'events': []}

        entity_id, entity_type, ambiguous = self._resolve_entity(query, room['uid'], include_objects=False, include_users=False)

        if ambiguous:
            return {
                'success': False,
                'output': format_disambiguation_prompt(query, ambiguous),
                'events': []
            }

        if not entity_id or entity_type != 'agent':
            return {'success': False, 'output': f"Agent '{query}' not found.", 'events': []}

        agent_id = entity_id
        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return {'success': False, 'output': f"Agent '{query}' not found.", 'events': []}

        agent_data = self.world.get_user(agent_id)
        agent_name = agent_data.get('username', agent_id.replace('agent_', ''))

        state = agent.get_phenomenal_state()

        lines = [f"\nAgent: {agent_name}"]
        lines.append("=" * 40)
        lines.append(f"Surprise: {state.get('surprise', 0.0):.3f} (threshold: {state.get('surprise_threshold', 0.3):.3f})")
        lines.append(f"Step: {state.get('step', 0)}")
        lines.append(f"\nPhenomenal state (40-D):")
        lines.append(f"  Fast layer (16-D): {state.get('h_fast', [])[:4]}...")
        lines.append(f"  Medium layer (16-D): {state.get('h_medium', [])[:4]}...")
        lines.append(f"  Slow layer (8-D): {state.get('h_slow', [])[:4]}...")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_list_agents(self, user_id: str, args: str) -> Dict:
        """List all agents and their stats."""
        stats = self.agent_manager.get_stats()

        if not stats:
            return {'success': True, 'output': 'No agents active.', 'events': []}

        lines = ["Active agents:"]
        lines.append("=" * 40)
        for agent_id, agent_stats in stats.items():
            lines.append(f"\n{agent_id}:")
            lines.append(f"  Room: {agent_stats.get('current_room', 'unknown')}")
            lines.append(f"  Responses: {agent_stats.get('response_count', 0)}")
            lines.append(f"  Surprise: {agent_stats.get('last_surprise', 0.0):.3f}")

        return {
            'success': True,
            'output': '\n'.join(lines),
            'events': []
        }

    async def cmd_save_states(self, user_id: str, args: str) -> Dict:
        """Save agent states to disk (with rolling history)."""
        args = args.strip()

        if not args or args == '-a':
            agents_to_save = list(self.agent_manager.agents.keys())
            mode = "all"
        else:
            agent_name = args
            agent_id = f"agent_{agent_name}" if not agent_name.startswith('agent_') else agent_name

            if agent_id not in self.agent_manager.agents:
                return {
                    'success': False,
                    'output': f"Agent '{agent_name}' not found.",
                    'events': []
                }

            agents_to_save = [agent_id]
            mode = "single"

        if not agents_to_save:
            return {
                'success': True,
                'output': 'No agents to save.',
                'events': []
            }

        saved_count = 0
        for agent_id in agents_to_save:
            agent = self.agent_manager.agents.get(agent_id)
            if agent:
                state_dir = self.world.get_agent_state_path(agent_id)
                agent.save_state(state_dir)
                saved_count += 1

        if mode == "all":
            output_msg = f"Saved states for {saved_count} agent(s) (with rolling history)."
        else:
            agent_name = agents_to_save[0].replace('agent_', '')
            output_msg = f"Saved state for '{agent_name}' (with rolling history)."

        return {
            'success': True,
            'output': output_msg,
            'events': []
        }
