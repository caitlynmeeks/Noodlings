"""
Scene Protocol Integration - Bridges cmush World with Noodlings Scene Protocol

Provides:
- SceneStateManager singleton for canonical world truth
- Syncs cmush World state to SceneStateManager
- Generates PerceptionSlices per-agent for WorldAPI
- Processes WorldAPI pending commands

This module enables ScriptedFacets to access world state via:
    context.noodle.world.perceivedEntities
    context.noodle.world.canSee("yuki")
    context.noodle.world.speak("Hello!", "friendly")

Author: Commander Spock + Cadet Caity
Date: December 18, 2025
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Scene Protocol imports
try:
    from noodlestudio.core.semantic_world import (
        SceneStateManager,
        Vector3,
        Zone,
        Noodling,
        Player,
        Prim,
        PerceptionCone,
        Affect,
        VisualForm,
        generate_perception_slice,
    )
    from noodlestudio.scripting.world_api import WorldAPI, get_world_api
    SCENE_PROTOCOL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Scene Protocol not available: {e}")
    SCENE_PROTOCOL_AVAILABLE = False
    SceneStateManager = None
    WorldAPI = None


# =============================================================================
# Global State
# =============================================================================

_scene_state_manager: Optional['SceneStateManager'] = None
_world_apis: Dict[str, 'WorldAPI'] = {}


def get_scene_state_manager() -> Optional['SceneStateManager']:
    """Get global SceneStateManager instance."""
    return _scene_state_manager


def init_scene_state_manager(stage_id: str = "default", stage_name: str = "The World") -> Optional['SceneStateManager']:
    """
    Initialize the global SceneStateManager.

    Args:
        stage_id: ID for the current stage
        stage_name: Display name for the stage

    Returns:
        SceneStateManager instance or None if not available
    """
    global _scene_state_manager

    if not SCENE_PROTOCOL_AVAILABLE:
        logger.warning("Scene Protocol not available - SceneStateManager not initialized")
        return None

    _scene_state_manager = SceneStateManager(
        stage_id=stage_id,
        stage_name=stage_name
    )
    logger.info(f"[SceneProtocol] SceneStateManager initialized: {stage_name} ({stage_id})")
    return _scene_state_manager


# =============================================================================
# World Sync Functions
# =============================================================================

def sync_room_to_zone(room_data: Dict[str, Any], room_id: str):
    """
    Sync a cmush room to a Scene Protocol Zone.

    Args:
        room_data: Room data from cmush World
        room_id: Room identifier
    """
    if not _scene_state_manager:
        return

    # Create zone from room
    zone = Zone(
        id=room_id,
        name=room_data.get('name', room_id),
        center=Vector3(0, 0, 0),  # Default center
        radius=20.0,  # Default radius
        falloff=10.0,
        description=room_data.get('description', ''),
        features=room_data.get('features', []),
        mood=room_data.get('mood', 'neutral'),
        lighting=room_data.get('lighting', 'ambient'),
        exits=room_data.get('exits', {})
    )

    _scene_state_manager.add_zone(zone)
    logger.debug(f"[SceneProtocol] Synced room to zone: {room_id}")


def sync_agent_to_noodling(agent_data: Dict[str, Any], agent_id: str, room_id: str):
    """
    Sync a cmush agent to a Scene Protocol Noodling.

    Args:
        agent_data: Agent data including recipe info
        agent_id: Agent identifier
        room_id: Current room/zone
    """
    if not _scene_state_manager:
        return

    # Get or create noodling
    if agent_id not in _scene_state_manager.noodlings:
        noodling = _scene_state_manager.add_noodling(
            noodling_id=agent_id,
            display_name=agent_data.get('name', agent_id),
            position=[0, 0, 0],  # Default position
            species=agent_data.get('species', 'unknown'),
            height=agent_data.get('height', 1.0),
        )
    else:
        noodling = _scene_state_manager.noodlings[agent_id]

    # Update state
    noodling.zone = room_id

    # Update affect if available
    if 'affect' in agent_data:
        affect_data = agent_data['affect']
        if isinstance(affect_data, list):
            noodling.affect = Affect(
                valence=affect_data[0] if len(affect_data) > 0 else 0.0,
                arousal=affect_data[1] if len(affect_data) > 1 else 0.5,
                dominance=affect_data[2] if len(affect_data) > 2 else 0.5,
                sorrow=affect_data[3] if len(affect_data) > 3 else 0.0,
                boredom=affect_data[4] if len(affect_data) > 4 else 0.0,
            )
        elif isinstance(affect_data, dict):
            noodling.affect = Affect(
                valence=affect_data.get('valence', 0.0),
                arousal=affect_data.get('arousal', 0.5),
                dominance=affect_data.get('dominance', 0.5),
                sorrow=affect_data.get('sorrow', 0.0),
                boredom=affect_data.get('boredom', 0.0),
            )

    # Update perception cone from config
    if 'perception' in agent_data:
        perc = agent_data['perception']
        noodling.perception = PerceptionCone(
            fov_horizontal=perc.get('fov_horizontal', 120),
            fov_vertical=perc.get('fov_vertical', 90),
            range=perc.get('range', 15.0),
            heat_sense=perc.get('heat_sense', False),
            night_vision=perc.get('night_vision', False),
        )

    # Update visual state if multi-form
    if 'visual_state' in agent_data:
        noodling.visual_state = agent_data['visual_state']

    logger.debug(f"[SceneProtocol] Synced agent to noodling: {agent_id} in {room_id}")


def sync_player_to_scene(player_id: str, player_name: str, room_id: str):
    """
    Sync a cmush player to Scene Protocol.

    Args:
        player_id: Player identifier
        player_name: Display name
        room_id: Current room/zone
    """
    if not _scene_state_manager:
        return

    if player_id not in _scene_state_manager.players:
        player = _scene_state_manager.add_player(
            player_id=player_id,
            display_name=player_name,
            position=[0, 0, 3.0],  # Default in front
        )
    else:
        player = _scene_state_manager.players[player_id]

    player.zone = room_id
    logger.debug(f"[SceneProtocol] Synced player: {player_id} in {room_id}")


def record_dialogue(speaker_id: str, text: str, tone: str = "neutral"):
    """
    Record dialogue in Scene Protocol.

    Args:
        speaker_id: Who spoke
        text: What was said
        tone: Tone of voice
    """
    if not _scene_state_manager:
        return

    _scene_state_manager.record_dialogue(speaker_id, text, tone)
    logger.debug(f"[SceneProtocol] Recorded dialogue: {speaker_id} ({tone})")


# =============================================================================
# Perception Integration
# =============================================================================

def get_agent_world_api(agent_id: str) -> Optional['WorldAPI']:
    """
    Get or create WorldAPI for an agent.

    Each agent gets their own perception-filtered view.

    Args:
        agent_id: Agent identifier

    Returns:
        WorldAPI instance or None if not available
    """
    global _world_apis

    if not SCENE_PROTOCOL_AVAILABLE:
        return None

    if agent_id not in _world_apis:
        world_api = get_world_api(agent_id)
        world_api.set_scene_state_manager(_scene_state_manager)
        _world_apis[agent_id] = world_api

    return _world_apis[agent_id]


def update_world_api_perception(agent_id: str) -> bool:
    """
    Update an agent's WorldAPI with fresh perception slice.

    Call this before facet execution to give the agent
    current perception-filtered world state.

    Args:
        agent_id: Agent identifier

    Returns:
        True if successful
    """
    if not _scene_state_manager:
        return False

    world_api = get_agent_world_api(agent_id)
    if not world_api:
        return False

    # Generate perception slice for this agent
    try:
        perception_slice = _scene_state_manager.generate_perception_slice(agent_id)
        world_api.update_from_perception_slice(perception_slice)
        logger.debug(f"[SceneProtocol] Updated perception for {agent_id}: {len(perception_slice.perceived_entities)} entities")
        return True
    except Exception as e:
        logger.warning(f"[SceneProtocol] Failed to update perception for {agent_id}: {e}")
        return False


def process_world_api_commands(agent_id: str) -> Dict[str, Any]:
    """
    Process pending commands from WorldAPI after facet execution.

    Call this after facet execution to apply any changes
    the agent requested (expression, gaze, speak, etc.).

    Args:
        agent_id: Agent identifier

    Returns:
        Dict of commands that were applied
    """
    if not _scene_state_manager:
        return {}

    world_api = get_agent_world_api(agent_id)
    if not world_api:
        return {}

    commands = world_api.get_pending_commands()

    if not commands:
        return {}

    logger.debug(f"[SceneProtocol] Processing {len(commands)} commands for {agent_id}")

    # Apply commands to SceneStateManager
    noodling = _scene_state_manager.noodlings.get(agent_id)
    if not noodling:
        logger.warning(f"[SceneProtocol] No noodling found for {agent_id}")
        return commands

    # Apply state changes
    if 'expression' in commands:
        noodling.expression = commands['expression']
        logger.debug(f"[SceneProtocol] {agent_id} expression -> {commands['expression']}")

    if 'posture' in commands:
        noodling.posture = commands['posture']
        logger.debug(f"[SceneProtocol] {agent_id} posture -> {commands['posture']}")

    if 'gaze' in commands:
        noodling.gaze_target = commands['gaze']
        logger.debug(f"[SceneProtocol] {agent_id} gaze -> {commands['gaze']}")

    if 'action' in commands:
        noodling.current_action = commands['action']
        logger.debug(f"[SceneProtocol] {agent_id} action -> {commands['action']}")

    if 'move_to' in commands:
        pos = commands['move_to']
        noodling.position = Vector3(pos[0], pos[1], pos[2])
        logger.debug(f"[SceneProtocol] {agent_id} position -> {pos}")

    if 'speak' in commands:
        speak_data = commands['speak']
        _scene_state_manager.record_dialogue(
            agent_id,
            speak_data['text'],
            speak_data.get('tone', 'neutral')
        )
        logger.debug(f"[SceneProtocol] {agent_id} spoke: {speak_data['text'][:50]}...")

    # Camera commands (if enabled)
    if 'camera_focus' in commands:
        focus = commands['camera_focus']
        _scene_state_manager.set_camera_focus(
            focus['subject'],
            framing=focus.get('framing', 'medium'),
            mode='FOCUS_ON'
        )
        logger.debug(f"[SceneProtocol] Camera focus -> {focus['subject']}")

    if 'camera_two_shot' in commands:
        two_shot = commands['camera_two_shot']
        _scene_state_manager.set_camera_focus(
            two_shot['subjects'][0],
            framing=two_shot.get('framing', 'medium'),
            mode='TWO_SHOT'
        )
        logger.debug(f"[SceneProtocol] Camera two-shot -> {two_shot['subjects']}")

    if 'camera_pov' in commands:
        _scene_state_manager.set_camera_focus(
            commands['camera_pov'],
            mode='POV'
        )
        logger.debug(f"[SceneProtocol] Camera POV -> {commands['camera_pov']}")

    return commands


# =============================================================================
# Facet Executor Integration
# =============================================================================

def prepare_facet_context(agent_id: str, exec_vars: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare context for facet execution with WorldAPI.

    Call this before facet_executor.execute() to inject WorldAPI.

    Args:
        agent_id: Agent identifier
        exec_vars: Existing execution variables dict

    Returns:
        Updated exec_vars with WorldAPI injected
    """
    if not SCENE_PROTOCOL_AVAILABLE:
        return exec_vars

    # Update perception before execution
    update_world_api_perception(agent_id)

    # Get WorldAPI for this agent
    world_api = get_agent_world_api(agent_id)
    if world_api:
        # The NoodleAPI will access WorldAPI via context.noodle.world
        # We need to set it on the noodle API instance
        if '_noodle_api' in exec_vars:
            exec_vars['_noodle_api'].set_world_api(world_api)
        else:
            # Store WorldAPI directly if NoodleAPI not present
            exec_vars['_world_api'] = world_api

        logger.debug(f"[SceneProtocol] Injected WorldAPI for {agent_id}")

    return exec_vars


def finalize_facet_context(agent_id: str) -> Dict[str, Any]:
    """
    Finalize context after facet execution.

    Call this after facet_executor.execute() to process WorldAPI commands.

    Args:
        agent_id: Agent identifier

    Returns:
        Commands that were applied
    """
    return process_world_api_commands(agent_id)


# =============================================================================
# Scene Packet Export
# =============================================================================

def get_scene_packet_json(indent: int = 2) -> Optional[str]:
    """
    Get current scene state as JSON.

    Useful for debugging or sending to external renderers.

    Args:
        indent: JSON indent level

    Returns:
        JSON string or None
    """
    if not _scene_state_manager:
        return None

    packet = _scene_state_manager.generate_scene_packet()
    return packet.to_json(indent=indent)


def get_scene_packet_text() -> Optional[str]:
    """
    Get current scene state as flattened text.

    Useful for LLM-based renderers like Genie.

    Returns:
        Text description or None
    """
    if not _scene_state_manager:
        return None

    packet = _scene_state_manager.generate_scene_packet()
    return packet.flatten_to_text()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Availability check
    "SCENE_PROTOCOL_AVAILABLE",

    # Global state
    "get_scene_state_manager",
    "init_scene_state_manager",

    # World sync
    "sync_room_to_zone",
    "sync_agent_to_noodling",
    "sync_player_to_scene",
    "record_dialogue",

    # Perception
    "get_agent_world_api",
    "update_world_api_perception",
    "process_world_api_commands",

    # Facet integration
    "prepare_facet_context",
    "finalize_facet_context",

    # Export
    "get_scene_packet_json",
    "get_scene_packet_text",
]
