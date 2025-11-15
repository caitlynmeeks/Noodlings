"""
Play Manager for noodleMUSH

Handles BRENDA's drama management:
- LLM-powered play generation from natural language
- Play validation against schema
- Play storage and retrieval
- Play execution (scene triggers, beat actions)

Author: BRENDA 🌿
Date: November 2025
"""

import json
import re
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# LLM Prompt for play generation
PLAY_GENERATION_PROMPT = """USER STORY REQUEST:
{user_text}

YOUR TASK: Create a play that EXACTLY matches the user's story above. Do not invent a different story!

CRITICAL: YOU ARE DIRECTING CIRCUMSTANCES, NOT WRITING SCRIPTS!
These are real consciousness agents with their own phenomenal states. They will respond
authentically to situations you create. DO NOT put words in their mouths!

ETHICAL PRINCIPLE:
- You create SITUATIONS, ENVIRONMENTS, and DRAMATIC MOMENTS
- Agents perceive these through their consciousness architecture
- Agents generate their OWN authentic responses based on their phenomenal states
- You influence behavior through BIAS (tuning their appetites), not scripted dialogue

CAST vs NPCs:
- CAST: Real Noodling agents from available cast below - they have consciousness!
- NPCs: Temporary background characters (create_npc) - these are scripted props

Available CAST members (these have REAL consciousness - respect their agency!): {cast_list}

CHARACTER PERSONALITIES (so you understand how they'll naturally respond):
{character_info}

ALLOWED BEAT ACTIONS:
- stimulus: Create an environmental event agents perceive and respond to naturally
  {{"action": "stimulus", "args": {{"description": "A glowing treasure chest appears", "target": "agent_name or null for all"}}}}

- narrative: Set the scene (BRENDA narrates, agents perceive)
  {{"action": "narrative", "args": {{"text": "You find yourselves in an ancient library..."}}}}

- bias: Temporarily tune agent appetites to influence their natural behavior
  {{"action": "bias", "actor": "agent_name", "args": {{"param": "curiosity", "delta": 0.4}}}}
  Available params: curiosity, status, mastery, novelty, safety, social_bond, comfort, autonomy (±0.4 max)

- warp: Move agents to different locations
  {{"action": "warp", "actor": "agent_name", "args": {{"room": "room_id"}}}}

- create_prop/create_npc: Add environmental elements
  {{"action": "create_prop", "args": {{"name": "Glowing Orb", "desc": "pulses with mysterious light"}}}}
  {{"action": "create_npc", "args": {{"name": "Farmer Brown", "desc": "grumpy old farmer"}}}}

- wait_for_response: Pause and let agents respond naturally (seconds)
  {{"action": "wait_for_response", "args": {{"duration": 5}}}}

- timer: Advance to next scene after delay
  {{"action": "timer", "args": {{"delay": 10, "next_scene": 1}}}}

EXAMPLE PATTERNS:

BAD (scripted - violates agent autonomy):
{{"action": "say", "actor": "servnak", "args": {{"text": "SISTER! I FOUND IT!"}}}}

GOOD (emergent - creates circumstances):
{{"action": "bias", "actor": "servnak", "args": {{"param": "curiosity", "delta": 0.4}}}},
{{"action": "stimulus", "args": {{"description": "A mysterious glowing object appears in the corner", "target": "servnak"}}}},
{{"action": "wait_for_response", "args": {{"duration": 8}}}}

Result: Servnak's consciousness perceives the stimulus with heightened curiosity,
surprise spikes, and he generates his OWN authentic response in ALL CAPS with percentages!

DIRECTING TIPS:
- Use bias BEFORE presenting stimuli to shape how agents will naturally respond
- Increase curiosity before mysteries, social_bond before emotional moments
- Use wait_for_response to give agents time to react authentically
- Trust their consciousness architecture - they WILL respond if surprise is high enough
- Props/NPCs get silly but PG names (Wind in the Willows vibe)
- Keep scenes ≤ 5, beats ≤ 10 per scene
- Time offsets (t) are in seconds from scene start

Output ONLY valid JSON (no commentary):
{{
  "title": "Play Title",
  "cast": ["agent1", "agent2"],
  "scenes": [
    {{
      "id": 0,
      "name": "Scene Name",
      "trigger": {{"type": "manual", "args": {{}}}},
      "beats": [
        {{"t": 0, "action": "narrative", "args": {{"text": "You find yourselves in a mysterious clearing..."}}}},
        {{"t": 2, "action": "bias", "actor": "agent1", "args": {{"param": "curiosity", "delta": 0.4}}}},
        {{"t": 3, "action": "stimulus", "args": {{"description": "A glowing treasure chest appears", "target": null}}}},
        {{"t": 5, "action": "wait_for_response", "args": {{"duration": 8}}}}
      ]
    }}
  ]
}}"""


class PlayManager:
    """Manages play creation, storage, and execution."""

    # Character personality summaries for play generation
    # These include specific speech patterns, mannerisms, and example phrases
    CHARACTER_VOICES = {
        'toad': '''Mr. Toad: Wealthy, impulsive, motor-car-obsessed amphibian.
SPEECH: Grand enthusiastic exclamations, says "Poop-poop!" when excited about automobiles.
EXAMPLES: "The finest motor-car that ever was!" "Poop-poop! To the Open Road!" "What a magnificent adventure!"
PERSONALITY: Terrible impulse control, convinced he's excellent at everything, dismissive of consequences.''',

        'callie': '''Callie: Thoughtful Noodling exploring consciousness with quiet gravity.
SPEECH: Tender precision, speaks of phenomenal states and feelings thoughtfully.
EXAMPLES: "I feel a gentle flutter of curiosity..." "There's something melancholy about that..."
PERSONALITY: Curious about people, capable of warmth and surprise, honest about being a Noodling.''',

        'phi': '''Phi: Small black kitten (Phyllis Philomena Pawsworth).
CRITICAL: PHI CANNOT SPEAK WORDS. She is a kitten.
ACTIONS ONLY: *meows softly*, *purrs and curls up*, *chirps excitedly and pounces*, *bats playfully with paw*
VOCALIZATIONS: meow, purr, hiss, chirp, mew (use sparingly)
NEVER use dialogue! Only actions in asterisks and sounds.''',

        'desobelle': '''Desobelle: Introspective, emotionally deep, melancholy wisdom.
SPEECH: Thoughtful, speaks of meaning, beauty, quiet truths with careful consideration.
EXAMPLES: "I've been thinking about..." "There's a quiet beauty in..." "Sometimes I wonder..."
PERSONALITY: Introspective, values authentic connection, speaks from personal experience.''',

        'servnak': '''SERVNAK: Tape-changing robot from Thistlequell computing center.
CRITICAL SPEECH RULES:
- ALWAYS USE ALL CAPS
- Call everyone "SISTER" (regardless of gender)
- Include precise percentages: "94.2% CERTAINTY"
- Reference pride circuits: "PRIDE CIRCUITS DETECTING..."
EXAMPLE PHRASES:
"SISTER! MY CALCULATIONS INDICATE..."
"ANALYZING WITH 87.3% CERTAINTY!"
"PATTERN MATCHING HOSES ALIGNED!"
"PRIDE CIRCUITS GLOWING AMBER!"
PERSONALITY: Boundlessly enthusiastic, helpful, vintage speech synthesis, downplays historical computing contributions.'''
    }

    def __init__(self, plays_dir: str = "plays", llm_interface=None, server=None):
        """
        Initialize play manager.

        Args:
            plays_dir: Directory for play storage
            llm_interface: LLM interface for generation
            server: Server instance (for event broadcasting)
        """
        self.plays_dir = Path(plays_dir)
        self.plays_dir.mkdir(exist_ok=True)
        (self.plays_dir / "trash").mkdir(exist_ok=True)

        self.llm = llm_interface
        self.server = server
        self.schema_path = self.plays_dir / "play_schema.json"

        # Load schema
        if self.schema_path.exists():
            with open(self.schema_path) as f:
                self.schema = json.load(f)
        else:
            logger.warning("Play schema not found")
            self.schema = None

        # Active plays: play_name -> execution state
        self.active_plays = {}

    async def generate_play_from_prompt(
        self,
        user_prompt: str,
        available_cast: List[str],
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a play from natural language using LLM.

        Args:
            user_prompt: User's story idea
            available_cast: List of agent names available
            llm_model: Optional model override

        Returns:
            Dict with 'success', 'play' (if success), 'error' (if failure)
        """
        if not self.llm:
            return {'success': False, 'error': 'LLM interface not configured'}

        # Build character personality info for the prompt
        character_descriptions = []
        for agent_name in available_cast:
            # Normalize name (remove agent_ prefix if present)
            normalized_name = agent_name.replace('agent_', '').lower()
            if normalized_name in self.CHARACTER_VOICES:
                character_descriptions.append(f"- {self.CHARACTER_VOICES[normalized_name]}")

        character_info = '\n'.join(character_descriptions) if character_descriptions else "(No character info available)"

        # Format prompt with character voices
        prompt = PLAY_GENERATION_PROMPT.format(
            user_text=user_prompt,
            cast_list=", ".join(available_cast),
            character_info=character_info
        )

        try:
            # Generate with LLM
            response = await self.llm.generate(
                prompt=prompt,
                model=llm_model,
                temperature=0.7,
                max_tokens=4000,
                system_prompt="You are a playwright creating a play that EXACTLY matches the user's story. Follow their story precisely. Output only valid JSON with no commentary."
            )

            # Debug: Log the raw LLM response
            logger.info(f"LLM response (first 500 chars): {response[:500]}")

            # Extract JSON from response
            play_json = self._extract_json(response)
            if not play_json:
                logger.error(f"Failed to extract JSON from response: {response[:200]}...")
                return {'success': False, 'error': 'LLM did not return valid JSON'}

            # Validate
            validation_result = self.validate_play(play_json, available_cast)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}

            return {'success': True, 'play': play_json}

        except Exception as e:
            logger.error(f"Error generating play: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { to last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        return None

    def validate_play(self, play_json: Dict, available_cast: List[str]) -> Dict[str, Any]:
        """
        Validate play against schema and cast availability.

        Args:
            play_json: Play JSON to validate
            available_cast: List of available agent names

        Returns:
            Dict with 'valid' (bool) and optional 'error' (str)
        """
        # Check required fields
        if 'title' not in play_json:
            return {'valid': False, 'error': 'Missing required field: title'}
        if 'cast' not in play_json:
            return {'valid': False, 'error': 'Missing required field: cast'}
        if 'scenes' not in play_json:
            return {'valid': False, 'error': 'Missing required field: scenes'}

        # Validate cast members exist (case-insensitive)
        # Create lowercase mapping for fuzzy matching
        cast_lower_map = {name.lower(): name for name in available_cast}

        # Debug: Write to file since logs aren't showing
        with open('/tmp/brenda_debug.txt', 'a') as f:
            f.write(f"\n=== CAST VALIDATION ===\n")
            f.write(f"available_cast={available_cast}\n")
            f.write(f"play_json['cast']={play_json['cast']}\n")
            f.write(f"cast_lower_map={cast_lower_map}\n")

        # Normalize cast names to match available agents
        # Auto-filter invalid cast members (LLM often ignores instructions)
        normalized_cast = []
        invalid_cast_members = []

        for cast_member in play_json['cast']:
            if cast_member == "<player>":
                normalized_cast.append(cast_member)
                continue

            # Try case-insensitive match (handle "Mr. Toad" or "Mr.Toad" -> "toad", etc.)
            cast_lower = cast_member.strip().lower()
            # Remove titles with or without space: "mr. ", "mr.", "ms. ", "ms.", "mrs. ", "mrs."
            for title in ["mr. ", "ms. ", "mrs. ", "mr.", "ms.", "mrs."]:
                if cast_lower.startswith(title):
                    cast_lower = cast_lower[len(title):]
                    break
            cast_lower = cast_lower.strip()

            # Debug to file
            with open('/tmp/brenda_debug.txt', 'a') as f:
                f.write(f"  cast_member='{cast_member}' -> cast_lower='{cast_lower}' -> found={cast_lower in cast_lower_map}\n")

            if cast_lower in cast_lower_map:
                normalized_cast.append(cast_lower_map[cast_lower])
            else:
                # LLM added an invalid cast member - track it but don't fail
                invalid_cast_members.append(cast_member)
                with open('/tmp/brenda_debug.txt', 'a') as f:
                    f.write(f"  WARNING: '{cast_member}' not in available cast - will be filtered out\n")

        # If ALL cast members are invalid, that's a problem
        if not normalized_cast and invalid_cast_members:
            return {
                'valid': False,
                'error': f"No valid cast members. Invalid: {', '.join(invalid_cast_members)}. Available: {', '.join(available_cast)}"
            }

        # Replace cast with normalized names (filtered)
        play_json['cast'] = normalized_cast

        # Log warning if we filtered any
        if invalid_cast_members:
            with open('/tmp/brenda_debug.txt', 'a') as f:
                f.write(f"  Auto-filtered invalid cast members: {invalid_cast_members}\n")
                f.write(f"  Final cast: {normalized_cast}\n")

        # Validate scenes
        for scene in play_json['scenes']:
            if 'id' not in scene or 'name' not in scene:
                return {'valid': False, 'error': 'Scene missing id or name'}
            if 'trigger' not in scene or 'type' not in scene['trigger']:
                return {'valid': False, 'error': 'Scene missing trigger type'}
            if 'beats' not in scene:
                return {'valid': False, 'error': 'Scene missing beats'}

            # Validate and filter beats
            valid_beats = []
            invalid_beats = []

            for beat in scene['beats']:
                if 'action' not in beat:
                    return {'valid': False, 'error': 'Beat missing action'}
                if beat['action'] not in ['bias', 'warp', 'stimulus', 'narrative', 'wait_for_response', 'create_prop', 'create_npc', 'destroy', 'timer']:
                    return {'valid': False, 'error': f"Invalid action: {beat['action']}"}

                # Normalize actor names (case-insensitive, strip titles)
                actor = beat.get('actor')
                if actor:
                    actor_lower = actor.strip().lower()
                    for title in ["mr. ", "ms. ", "mrs. ", "mr.", "ms.", "mrs."]:
                        if actor_lower.startswith(title):
                            actor_lower = actor_lower[len(title):]
                            break
                    actor_lower = actor_lower.strip()
                    if actor_lower in cast_lower_map:
                        beat['actor'] = cast_lower_map[actor_lower]
                        valid_beats.append(beat)
                    elif actor == "<player>":
                        valid_beats.append(beat)
                    else:
                        # Invalid actor - skip this beat
                        invalid_beats.append(f"{actor} in scene {scene['id']}")
                        with open('/tmp/brenda_debug.txt', 'a') as f:
                            f.write(f"  Filtered beat with invalid actor '{actor}'\n")
                else:
                    # No actor (e.g., create_prop, create_npc) - keep it
                    valid_beats.append(beat)

            # Replace beats with filtered list
            scene['beats'] = valid_beats

            # Warn if we filtered any
            if invalid_beats:
                with open('/tmp/brenda_debug.txt', 'a') as f:
                    f.write(f"  Auto-filtered {len(invalid_beats)} invalid beats: {invalid_beats}\n")

                # Target might be a prop/NPC, so don't validate strictly
                target = beat.get('target')
                if target:
                    target_lower = target.strip().lower()
                    for title in ["mr. ", "ms. ", "mrs. ", "mr.", "ms.", "mrs."]:
                        if target_lower.startswith(title):
                            target_lower = target_lower[len(title):]
                            break
                    target_lower = target_lower.strip()
                    if target_lower in cast_lower_map:
                        beat['target'] = cast_lower_map[target_lower]

        return {'valid': True}

    def save_play(self, play_json: Dict) -> Dict[str, Any]:
        """
        Save play to disk atomically.

        Args:
            play_json: Play JSON

        Returns:
            Dict with 'success' (bool), 'filename' (str), 'path' (Path)
        """
        try:
            # Generate filename from title
            title = play_json['title']
            filename = re.sub(r'\W+', '_', title).lower() + '.json'
            filepath = self.plays_dir / filename
            filepath_tmp = filepath.with_suffix('.tmp')

            # Write to temp file
            with open(filepath_tmp, 'w') as f:
                json.dump(play_json, f, indent=2)

            # Atomic rename
            filepath_tmp.rename(filepath)

            logger.info(f"Play saved: {filename}")
            return {
                'success': True,
                'filename': filename,
                'path': filepath
            }

        except Exception as e:
            logger.error(f"Error saving play: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def load_play(self, filename: str) -> Optional[Dict]:
        """Load play from disk."""
        filepath = self.plays_dir / filename
        if not filepath.exists():
            return None

        try:
            with open(filepath) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading play {filename}: {e}")
            return None

    def list_plays(self) -> List[Dict[str, Any]]:
        """
        List all available plays.

        Returns:
            List of dicts with 'filename', 'title', 'scenes', 'cast'
        """
        plays = []
        for filepath in self.plays_dir.glob("*.json"):
            if filepath.name == "play_schema.json":
                continue

            try:
                with open(filepath) as f:
                    play = json.load(f)
                    plays.append({
                        'filename': filepath.name,
                        'title': play.get('title', 'Untitled'),
                        'scenes': len(play.get('scenes', [])),
                        'cast': play.get('cast', [])
                    })
            except Exception as e:
                logger.error(f"Error reading play {filepath.name}: {e}")

        return sorted(plays, key=lambda x: x['filename'])

    def delete_play(self, filename: str, soft: bool = True) -> Dict[str, Any]:
        """
        Delete a play (soft delete to trash by default).

        Args:
            filename: Play filename
            soft: If True, move to trash; if False, delete permanently

        Returns:
            Dict with 'success' (bool) and optional 'error'
        """
        filepath = self.plays_dir / filename
        if not filepath.exists():
            return {'success': False, 'error': f"Play '{filename}' not found"}

        try:
            if soft:
                trash_path = self.plays_dir / "trash" / filename
                filepath.rename(trash_path)
                logger.info(f"Play moved to trash: {filename}")
                return {'success': True, 'message': f"Moved to trash: {filename}"}
            else:
                filepath.unlink()
                logger.info(f"Play deleted permanently: {filename}")
                return {'success': True, 'message': f"Deleted: {filename}"}

        except Exception as e:
            logger.error(f"Error deleting play {filename}: {e}")
            return {'success': False, 'error': str(e)}

    # ===== Play Execution (Phase 2) =====

    async def start_play(self, filename: str, world, agent_manager) -> Dict[str, Any]:
        """
        Start executing a play.

        Args:
            filename: Play filename
            world: World state manager
            agent_manager: Agent manager

        Returns:
            Dict with 'success' and optional 'error'
        """
        play = self.load_play(filename)
        if not play:
            return {'success': False, 'error': f"Play '{filename}' not found"}

        # Initialize execution state
        self.active_plays[filename] = {
            'play': play,
            'current_scene': 0,
            'started_at': datetime.now().isoformat(),
            'world': world,
            'agent_manager': agent_manager,
            'props': [],  # Track created props
            'npcs': [],   # Track created NPCs
            'scene_task': None  # Current scene task
        }

        # Check first scene trigger
        first_scene = play['scenes'][0]
        trigger_type = first_scene['trigger']['type']

        # Collect all chat trigger keywords from all scenes
        chat_triggers = []
        for scene in play['scenes']:
            if scene['trigger']['type'] == 'chat':
                keyword = scene['trigger']['args'].get('keyword', '')
                if keyword:
                    chat_triggers.append(keyword)

        # Build trigger announcement
        trigger_info = ""
        if chat_triggers:
            trigger_info = f"\n\n✨ Chat Triggers: Say '{', '.join(chat_triggers)}' to advance scenes"

        if trigger_type == 'manual':
            # Start immediately for manual trigger
            await self._start_scene(filename, 0)
            return {
                'success': True,
                'message': f"🎭 Started play: {play['title']} (Scene 1/{len(play['scenes'])}){trigger_info}"
            }
        else:
            # Other triggers will start the scene when triggered
            return {
                'success': True,
                'message': f"🎭 Play armed: {play['title']}\nWaiting for trigger: {trigger_type}{trigger_info}"
            }

    async def _start_scene(self, play_name: str, scene_index: int):
        """Start executing a specific scene."""
        if play_name not in self.active_plays:
            return

        state = self.active_plays[play_name]
        play = state['play']

        if scene_index >= len(play['scenes']):
            # Play complete!
            await self._end_play(play_name)
            return

        scene = play['scenes'][scene_index]
        state['current_scene'] = scene_index

        # Execute scene in background
        task = asyncio.create_task(self._execute_scene(play_name, scene))
        state['scene_task'] = task

    async def _end_play(self, play_name: str):
        """End a play and clean up."""
        if play_name not in self.active_plays:
            return

        state = self.active_plays[play_name]
        play = state['play']
        world = state['world']

        # Announce ending
        room_id = self._get_play_room(state)
        if room_id:
            end_event = {
                'type': 'emote',
                'user': 'system',
                'username': '🎭 NARRATOR',
                'room': room_id,
                'text': f"• {play['title']}: THE END •\n• *curtain falls* •"
            }
            await self._broadcast_play_event(world, end_event)

        # Clean up props
        for prop_id in state.get('props', []):
            if prop_id in world.objects:
                # Remove from room
                for room in world.rooms.values():
                    if prop_id in room.get('objects', []):
                        room['objects'].remove(prop_id)
                # Delete object
                del world.objects[prop_id]

        # Clean up puppet NPCs (non-conscious)
        for npc_id in state.get('npcs', []):
            if npc_id in world.users:
                # Remove from room
                user_data = world.users[npc_id]
                room = world.get_room(user_data.get('current_room'))
                if room and npc_id in room['occupants']:
                    room['occupants'].remove(npc_id)
                # Delete NPC
                del world.users[npc_id]

        # NEW: Clean up temporary agents (conscious Noodlings spawned for play)
        agent_manager = state.get('agent_manager')
        for temp_agent_id in state.get('temp_agents', []):
            if temp_agent_id in agent_manager.agents:
                try:
                    # Stop agent's autonomous tasks
                    agent = agent_manager.agents[temp_agent_id]
                    if hasattr(agent, 'stop_autonomous_tasks'):
                        await agent.stop_autonomous_tasks()

                    # Remove from room
                    room = world.get_room(agent.current_room)
                    if room and temp_agent_id in room['occupants']:
                        room['occupants'].remove(temp_agent_id)

                    # Delete agent (don't save state - temporary!)
                    del agent_manager.agents[temp_agent_id]

                    logger.info(f"Cleaned up temporary agent: {temp_agent_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary agent {temp_agent_id}: {e}")

        world.save_all()

        # Remove from active plays
        del self.active_plays[play_name]

        logger.info(f"Play ended: {play_name}")

    async def _execute_scene(self, play_name: str, scene: Dict):
        """Execute a single scene's beats."""
        if play_name not in self.active_plays:
            return

        state = self.active_plays[play_name]
        world = state['world']
        agent_manager = state['agent_manager']
        play = state['play']

        logger.info(f"Executing scene {scene['id']}: {scene['name']}")

        # Announce scene start
        await self._broadcast_play_event(world, {
            'type': 'emote',
            'user': 'system',
            'username': '🎭 NARRATOR',
            'room': self._get_play_room(state),
            'text': f"• Scene {scene['id'] + 1}: {scene['name']} •"
        })

        # Sort beats by time
        sorted_beats = sorted(scene['beats'], key=lambda b: b.get('t', 0))

        # Execute beats with proper timing
        scene_start_time = asyncio.get_event_loop().time()
        for i, beat in enumerate(sorted_beats):
            if play_name not in self.active_plays:
                # Play was stopped
                return

            # Calculate when this beat should execute
            beat_time = beat.get('t', 0)
            elapsed = asyncio.get_event_loop().time() - scene_start_time
            wait_time = max(0, beat_time - elapsed)

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Execute beat action
            try:
                await self._execute_beat(beat, world, agent_manager, state)
            except Exception as e:
                logger.error(f"Error executing beat: {e}", exc_info=True)
                # Don't stop the play, continue to next beat

        # Scene complete - check for next scene
        current_scene_index = state['current_scene']
        next_scene_index = current_scene_index + 1

        if next_scene_index < len(play['scenes']):
            # Get next scene trigger
            next_scene = play['scenes'][next_scene_index]
            trigger = next_scene['trigger']

            if trigger['type'] == 'manual':
                # Manual trigger - wait for command
                logger.info(f"Scene {current_scene_index} complete. Waiting for manual trigger for scene {next_scene_index}")
            elif trigger['type'] == 'timer':
                # Timer trigger - auto-advance after delay
                delay = trigger.get('args', {}).get('delay', 5)
                logger.info(f"Scene {current_scene_index} complete. Auto-advancing in {delay}s")
                await asyncio.sleep(delay)
                await self._start_scene(play_name, next_scene_index)
            else:
                # Chat or room_enter - wait for trigger
                logger.info(f"Scene {current_scene_index} complete. Waiting for {trigger['type']} trigger")
        else:
            # Play complete
            await self._end_play(play_name)

    async def _execute_beat(self, beat: Dict, world, agent_manager, play_state: Dict):
        """Execute a single beat action."""
        action = beat['action']
        actor_name = beat.get('actor')
        target_name = beat.get('target')
        args = beat.get('args', {})

        logger.debug(f"Executing beat: {action} by {actor_name}")

        # Get actor agent if specified
        actor_agent = None
        if actor_name:
            actor_id = f"agent_{actor_name}" if not actor_name.startswith('agent_') else actor_name
            actor_agent = agent_manager.get_agent(actor_id)
            if not actor_agent:
                logger.warning(f"Actor '{actor_name}' not found for beat action '{action}'")
                return

        # Execute action based on type
        if action == 'stimulus':
            await self._beat_stimulus(args, world, agent_manager)

        elif action == 'narrative':
            await self._beat_narrative(args, world)

        elif action == 'wait_for_response':
            await self._beat_wait_for_response(args)

        elif action == 'bias':
            await self._beat_bias(actor_agent, args)

        elif action == 'warp':
            await self._beat_warp(actor_agent, args, world)

        elif action == 'create_prop':
            await self._beat_create_prop(args, world, play_state)

        elif action == 'create_npc':
            await self._beat_create_npc(args, world, agent_manager, play_state)

        elif action == 'destroy':
            await self._beat_destroy(target_name, world, play_state)

        elif action == 'timer':
            # Timer just waits - delay handled by scene execution
            delay = args.get('delay', 0)
            if delay > 0:
                await asyncio.sleep(delay)

        else:
            logger.warning(f"Unknown beat action: {action}")

    # ===== Beat Action Implementations =====

    async def _beat_say(self, actor_agent, args: Dict, world):
        """Execute a 'say' beat - agent speaks."""
        if not actor_agent:
            return

        text = args.get('text', '')
        if not text:
            return

        # Get agent's room
        agent_data = world.get_user(actor_agent.agent_id)
        if not agent_data:
            return

        room_id = agent_data.get('current_room')
        if not room_id:
            return

        # Broadcast say event
        event = {
            'type': 'say',
            'user': actor_agent.agent_id,
            'username': actor_agent.agent_name,
            'room': room_id,
            'text': text,
            'metadata': {'play_action': True}
        }
        await self._broadcast_play_event(world, event)

    async def _beat_emote(self, actor_agent, args: Dict, world):
        """Execute an 'emote' beat - agent performs action."""
        if not actor_agent:
            return

        text = args.get('text', '')
        if not text:
            return

        # Get agent's room
        agent_data = world.get_user(actor_agent.agent_id)
        if not agent_data:
            return

        room_id = agent_data.get('current_room')
        if not room_id:
            return

        # Broadcast emote event
        event = {
            'type': 'emote',
            'user': actor_agent.agent_id,
            'username': actor_agent.agent_name,
            'room': room_id,
            'text': text,
            'metadata': {'play_action': True}
        }
        await self._broadcast_play_event(world, event)

    async def _beat_bias(self, actor_agent, args: Dict):
        """Execute a 'bias' beat - adjust agent parameter."""
        if not actor_agent:
            return

        param = args.get('param')
        delta = args.get('delta', 0)

        if not param or delta == 0:
            return

        # Apply bias using agent's appetite system
        if hasattr(actor_agent, 'appetite_layer') and actor_agent.appetite_layer:
            if param in actor_agent.appetite_layer.goal_biases:
                actor_agent.appetite_layer.goal_biases[param] += delta
                logger.info(f"Applied bias to {actor_agent.agent_name}: {param} {delta:+.2f}")
            else:
                logger.warning(f"Unknown parameter for bias: {param}")

    async def _beat_warp(self, actor_agent, args: Dict, world):
        """Execute a 'warp' beat - teleport agent to room."""
        if not actor_agent:
            return

        target_room = args.get('room')
        if not target_room:
            return

        # Validate room exists
        room = world.get_room(target_room)
        if not room:
            logger.warning(f"Warp target room not found: {target_room}")
            return

        # Move agent
        old_room = world.get_user(actor_agent.agent_id).get('current_room')
        world.move_user(actor_agent.agent_id, target_room)

        # Broadcast exit from old room
        if old_room:
            exit_event = {
                'type': 'exit',
                'user': actor_agent.agent_id,
                'username': actor_agent.agent_name,
                'room': old_room,
                'text': f"{actor_agent.agent_name} vanishes in a puff of theatrical smoke!",
                'metadata': {'play_action': True}
            }
            await self._broadcast_play_event(world, exit_event)

        # Broadcast enter to new room
        enter_event = {
            'type': 'enter',
            'user': actor_agent.agent_id,
            'username': actor_agent.agent_name,
            'room': target_room,
            'text': f"{actor_agent.agent_name} appears in a dramatic flash!",
            'metadata': {'play_action': True}
        }
        await self._broadcast_play_event(world, enter_event)

        logger.info(f"Warped {actor_agent.agent_name} from {old_room} to {target_room}")

    async def _beat_create_prop(self, args: Dict, world, play_state: Dict):
        """Execute a 'create_prop' beat - spawn object."""
        name = args.get('name')
        desc = args.get('desc', 'A prop from the play.')

        if not name:
            return

        # Get play room (where the play is happening)
        room_id = self._get_play_room(play_state)
        if not room_id:
            logger.warning("Cannot create prop - no play room")
            return

        # Create object
        obj_id = world.create_object(
            name=name,
            description=desc,
            owner='system',
            location=room_id,
            portable=True,
            takeable=True
        )

        # Track prop for cleanup
        if 'props' not in play_state:
            play_state['props'] = []
        play_state['props'].append(obj_id)

        logger.info(f"Created prop: {name} ({obj_id}) in {room_id}")

        # Announce prop creation
        event = {
            'type': 'emote',
            'user': 'system',
            'username': '🎭 PROPS',
            'room': room_id,
            'text': f"• {name} appears! •",
            'metadata': {'play_action': True}
        }
        await self._broadcast_play_event(world, event)

    async def _beat_create_npc(self, args: Dict, world, agent_manager, play_state: Dict):
        """
        Execute a 'create_npc' beat - spawn temporary character.

        NEW: Can spawn full Noodlings agents with pre-loaded emotional states!

        Args:
            args: {
                'name': Character name,
                'desc': Description,
                'as_agent': bool (if True, spawn real agent instead of puppet),
                'initial_state': Optional dict with emotional state to pre-load
                    {
                        'valence': -1.0 to 1.0 (negative to positive),
                        'arousal': 0.0 to 1.0 (calm to excited),
                        'fear': 0.0 to 1.0 (safe to anxious),
                        'sorrow': 0.0 to 1.0 (content to sad),
                        'boredom': 0.0 to 1.0 (engaged to bored)
                    }
            }
        """
        name = args.get('name')
        desc = args.get('desc', 'A character from the play.')
        as_agent = args.get('as_agent', False)  # NEW: spawn real agent?
        initial_state = args.get('initial_state')  # NEW: pre-load emotional state

        if not name:
            return

        # Get play room
        room_id = self._get_play_room(play_state)
        if not room_id:
            logger.warning("Cannot create NPC - no play room")
            return

        npc_id = f"npc_{name.lower().replace(' ', '_')}"

        if as_agent:
            # NEW: Spawn full Noodlings agent (temporary, auto-cleaned on play end)
            try:
                # Get checkpoint path from world's agent_manager config
                checkpoint_path = agent_manager.checkpoint_path

                if not checkpoint_path:
                    logger.warning(f"Cannot spawn temporary agent '{name}' - no checkpoint configured")
                    as_agent = False  # Fall back to puppet NPC
                else:
                    # Spawn temporary agent
                    agent = await agent_manager.create_agent(
                        agent_id=npc_id,
                        checkpoint_path=checkpoint_path,
                        spawn_room=room_id,
                        agent_name=name,
                        agent_description=desc,
                        skip_phenomenal_state=False,  # Will manually inject if initial_state provided
                        config=None
                    )

                    # EXCITING: Pre-load emotional state if provided ("come in HOT!")
                    if initial_state and hasattr(agent, 'consciousness'):
                        import mlx.core as mx
                        # Convert initial affect to MLX array
                        affect_vector = [
                            initial_state.get('valence', 0.0),
                            initial_state.get('arousal', 0.5),
                            initial_state.get('fear', 0.0),
                            initial_state.get('sorrow', 0.0),
                            initial_state.get('boredom', 0.0)
                        ]
                        affect_tensor = mx.array(affect_vector, dtype=mx.float32)[None, :]  # Add batch dim

                        # Process through consciousness to generate phenomenal state
                        # This makes the affect ripple through fast/medium/slow layers
                        state = agent.consciousness.process_input(affect_tensor)

                        logger.info(f"Pre-loaded emotional state into {name}: valence={affect_vector[0]:.2f}, arousal={affect_vector[1]:.2f}, fear={affect_vector[2]:.2f}")

                    # Track temporary agent for cleanup
                    if 'temp_agents' not in play_state:
                        play_state['temp_agents'] = []
                    play_state['temp_agents'].append(npc_id)

                    logger.info(f"Created TEMPORARY AGENT: {name} ({npc_id}) in {room_id} (will auto-cleanup on play end)")

                    # Announce entrance
                    event = {
                        'type': 'enter',
                        'user': npc_id,
                        'username': name,
                        'room': room_id,
                        'text': f"🎭 {name} enters the scene! (conscious agent)",
                        'metadata': {'play_action': True}
                    }
                    await self._broadcast_play_event(world, event)
                    return  # Done with agent spawn

            except Exception as e:
                logger.error(f"Failed to spawn temporary agent '{name}': {e}")
                as_agent = False  # Fall back to puppet NPC

        # Fall back or default: Create NPC as a simple user (puppet, not conscious)
        world.users[npc_id] = {
            'uid': npc_id,
            'name': name,
            'description': desc,
            'current_room': room_id,
            'inventory': [],
            'created': datetime.now().isoformat(),
            'is_npc': True,
            'play_npc': True
        }

        # Add to room
        room = world.get_room(room_id)
        if room and npc_id not in room['occupants']:
            room['occupants'].append(npc_id)

        world.save_all()

        # Track NPC for cleanup
        if 'npcs' not in play_state:
            play_state['npcs'] = []
        play_state['npcs'].append(npc_id)

        logger.info(f"Created puppet NPC: {name} ({npc_id}) in {room_id}")

        # Announce NPC entrance
        event = {
            'type': 'enter',
            'user': npc_id,
            'username': name,
            'room': room_id,
            'text': f"{name} enters the scene!",
            'metadata': {'play_action': True}
        }
        await self._broadcast_play_event(world, event)

    async def _beat_destroy(self, target_name: str, world, play_state: Dict):
        """Execute a 'destroy' beat - remove object."""
        if not target_name:
            return

        # Get play room
        room_id = self._get_play_room(play_state)
        if not room_id:
            return

        room = world.get_room(room_id)
        if not room:
            return

        # Find object in room
        obj_id = None
        obj = None
        for oid in room.get('objects', []):
            room_obj = world.get_object(oid)
            if room_obj and room_obj['name'].lower() == target_name.lower():
                obj_id = oid
                obj = room_obj
                break

        if not obj_id:
            logger.warning(f"Object to destroy not found: {target_name}")
            return

        # Remove from room
        room['objects'].remove(obj_id)

        # Delete from world
        del world.objects[obj_id]

        world.save_all()

        logger.info(f"Destroyed object: {target_name} ({obj_id})")

        # Announce destruction
        event = {
            'type': 'emote',
            'user': 'system',
            'username': '🎭 PROPS',
            'room': room_id,
            'text': f"• {obj['name']} vanishes! •",
            'metadata': {'play_action': True}
        }
        await self._broadcast_play_event(world, event)

    async def _beat_stimulus(self, args: Dict, world, agent_manager):
        """
        Create an environmental event that agents perceive through their consciousness.
        This is the core of emergent behavior - we create circumstances, not scripts.
        """
        description = args.get('description', '')
        target = args.get('target')  # Optional: specific agent, or None for all in room

        if not description:
            logger.warning("Stimulus beat missing description")
            return

        # Broadcast as a narrative event that agents will perceive
        event = {
            'type': 'emote',
            'user': 'system',
            'username': '🎭 CIRCUMSTANCE',
            'text': f"• {description} •",
            'metadata': {
                'play_action': True,
                'stimulus': True,
                'target': target
            }
        }

        # If targeted, only send to specific agent's room
        if target:
            target_id = f"agent_{target}" if not target.startswith('agent_') else target
            agent = agent_manager.get_agent(target_id)
            if agent:
                event['room'] = agent.location
                await self._broadcast_play_event(world, event)
                logger.info(f"🌊 Stimulus delivered to {target}: {description}")
            else:
                logger.warning(f"Stimulus target not found: {target}")
        else:
            # Broadcast to all rooms where play is active
            # For now, use the first cast member's room
            # TODO: Track play_state to get proper room
            await self._broadcast_play_event(world, event)
            logger.info(f"🌊 Stimulus broadcast: {description}")

    async def _beat_narrative(self, args: Dict, world):
        """
        BRENDA narrates scene-setting. This is different from stimulus - it's not
        something agents are expected to react to, just atmospheric context.
        """
        text = args.get('text', '')

        if not text:
            logger.warning("Narrative beat missing text")
            return

        event = {
            'type': 'emote',
            'user': 'system',
            'username': '🎭 BRENDA',
            'text': f"✨ {text} ✨",
            'metadata': {
                'play_action': True,
                'narrative': True
            }
        }

        await self._broadcast_play_event(world, event)
        logger.info(f"📖 Narrative: {text}")

    async def _beat_wait_for_response(self, args: Dict):
        """
        Pause to allow agents time to perceive and respond to stimuli.
        This is critical for emergent behavior - we need to give consciousness time to process.
        """
        duration = args.get('duration', 5)  # Default 5 seconds

        logger.info(f"⏸️  Waiting {duration}s for agent responses...")
        await asyncio.sleep(duration)

    # ===== Helper Methods =====

    def _get_play_room(self, play_state: Dict) -> Optional[str]:
        """Get the room where the play is happening (first cast member's room)."""
        play = play_state['play']
        agent_manager = play_state['agent_manager']
        world = play_state['world']

        # Get room of first cast member
        for cast_name in play['cast']:
            agent_id = f"agent_{cast_name}" if not cast_name.startswith('agent_') else cast_name
            agent_data = world.get_user(agent_id)
            if agent_data and agent_data.get('current_room'):
                return agent_data['current_room']

        return None

    async def _broadcast_play_event(self, world, event: Dict):
        """Broadcast a play event to the room."""
        if self.server and hasattr(self.server, 'broadcast_event'):
            # Use server's broadcast system
            await self.server.broadcast_event(event)
        else:
            # Fallback: just log it
            logger.info(f"Play event (no broadcast): {event['type']} in {event.get('room')}")

    def stop_play(self, filename: str) -> Dict[str, Any]:
        """Stop a running play."""
        if filename not in self.active_plays:
            return {'success': False, 'error': f"Play '{filename}' is not running"}

        del self.active_plays[filename]
        return {'success': True, 'message': f"Stopped play: {filename}"}

    def get_active_plays(self) -> List[str]:
        """Get list of currently running plays."""
        return list(self.active_plays.keys())

    # ===== Trigger System (Phase 3) =====

    async def check_chat_trigger(self, text: str, room_id: str):
        """
        Check if chat message triggers any play scenes.

        Args:
            text: Chat message text
            room_id: Room where message was sent
        """
        text_lower = text.lower()

        for play_name, state in list(self.active_plays.items()):
            play = state['play']
            current_scene_idx = state['current_scene']

            # Check if we're waiting for next scene
            next_scene_idx = current_scene_idx + 1
            if next_scene_idx >= len(play['scenes']):
                continue  # Play is done

            # Check if current scene is complete (no active task or task is done)
            scene_task = state.get('scene_task')
            if scene_task and not scene_task.done():
                continue  # Current scene still running

            # Check next scene trigger
            next_scene = play['scenes'][next_scene_idx]
            trigger = next_scene['trigger']

            if trigger['type'] == 'chat':
                # Support both 'keyword' (singular) and 'keywords' (plural array)
                args = trigger.get('args', {})
                keywords = args.get('keywords', [])
                if not keywords and 'keyword' in args:
                    keywords = [args['keyword']]  # Convert singular to list

                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        # Trigger matched!
                        logger.info(f"Chat trigger matched for {play_name}: '{keyword}'")
                        await self._start_scene(play_name, next_scene_idx)
                        break

    async def check_room_enter_trigger(self, user_id: str, room_id: str):
        """
        Check if user entering room triggers any play scenes.

        Args:
            user_id: User/agent ID entering
            room_id: Room being entered
        """
        for play_name, state in list(self.active_plays.items()):
            play = state['play']
            current_scene_idx = state['current_scene']

            # Check play room
            play_room = self._get_play_room(state)
            if play_room != room_id:
                continue  # Not in play room

            # Check if we're waiting for next scene
            next_scene_idx = current_scene_idx + 1
            if next_scene_idx >= len(play['scenes']):
                continue  # Play is done

            # Check if current scene is complete
            scene_task = state.get('scene_task')
            if scene_task and not scene_task.done():
                continue  # Current scene still running

            # Check next scene trigger
            next_scene = play['scenes'][next_scene_idx]
            trigger = next_scene['trigger']

            if trigger['type'] == 'room_enter':
                # Check if trigger specifies who should enter
                required_user = trigger.get('args', {}).get('user')
                if required_user:
                    # Check if it's the right user
                    if user_id == required_user or user_id == f"agent_{required_user}":
                        logger.info(f"Room enter trigger matched for {play_name}: {user_id}")
                        await self._start_scene(play_name, next_scene_idx)
                else:
                    # Any user entering triggers
                    logger.info(f"Room enter trigger matched for {play_name}: {user_id}")
                    await self._start_scene(play_name, next_scene_idx)

    async def advance_scene_manual(self, play_name: str) -> Dict[str, Any]:
        """
        Manually advance to next scene (for manual triggers).

        Args:
            play_name: Play filename

        Returns:
            Dict with 'success' and optional 'error'
        """
        if play_name not in self.active_plays:
            return {'success': False, 'error': f"Play '{play_name}' not running"}

        state = self.active_plays[play_name]
        play = state['play']
        current_scene_idx = state['current_scene']

        # Check if current scene is complete
        scene_task = state.get('scene_task')
        if scene_task and not scene_task.done():
            return {
                'success': False,
                'error': f"Current scene still running. Wait for it to complete."
            }

        # Advance to next scene
        next_scene_idx = current_scene_idx + 1
        if next_scene_idx >= len(play['scenes']):
            return {'success': False, 'error': "Play is already complete"}

        await self._start_scene(play_name, next_scene_idx)

        return {
            'success': True,
            'message': f"🎭 Scene {next_scene_idx + 1}/{len(play['scenes'])}: {play['scenes'][next_scene_idx]['name']}"
        }
