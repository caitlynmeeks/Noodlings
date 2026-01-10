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
#   Scene Protocol Integration - World State Bridge
#
#   This is the connector between the server's world state and the
#   NoodleStudio scene system. It syncs rooms to zones, agents to
#   Noodlings, and enables perception-filtered views. When a facet
#   script calls context.noodle.world.canSee("yuki"), this module
#   answers that question. It also handles Gaussian scene composition
#   and semantic queries (asking "where is Red's hand?" via CLIP).
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.scene_protocol_integration
# PURPOSE:  Bridge server world state to Studio scene system
# LAYER:    Backend / Integration
# ──────────────────────────────────────────────────────────────
#
# KEY FUNCTIONS:
#   sync_room_to_zone()         Sync room to Scene Protocol Zone
#   sync_agent_to_noodling()    Sync agent to Scene Protocol Noodling
#   get_agent_world_api()       Get agent's perception-filtered view
#   compose_gaussian_scene()    Create Gaussian scene from world state
#   query_scene_semantic()      Natural language scene queries
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

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
        ZoneBounds,
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

# Gaussian Adapter imports
try:
    from noodlestudio.core.semantic_world import (
        GaussianAssetManager,
        GaussianSceneCompositor,
        GaussianGenerator,
        GaussianScene,
        compose_scene_from_packet,
        init_gaussian_adapter,
        get_asset_manager,
        get_compositor,
        get_generator,
    )
    GAUSSIAN_ADAPTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Gaussian Adapter not available: {e}")
    GAUSSIAN_ADAPTER_AVAILABLE = False
    GaussianAssetManager = None
    GaussianSceneCompositor = None

# Semantic Query imports (CLIP natural language queries on Gaussians)
try:
    from noodlestudio.core.semantic_world.semantic_query import (
        SemanticQueryEngine,
        CLIPEmbeddingGenerator,
        populate_asset_embeddings,
        SemanticSearchResult,
        SplatHitInfo,
    )
    from noodlestudio.core.semantic_world.radiance_format import (
        RadianceAsset,
        load_radiance,
    )
    SEMANTIC_QUERY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Semantic Query not available: {e}")
    SEMANTIC_QUERY_AVAILABLE = False
    SemanticQueryEngine = None
    RadianceAsset = None


# =============================================================================
# Global State
# =============================================================================

_scene_state_manager: Optional['SceneStateManager'] = None
_world_apis: Dict[str, 'WorldAPI'] = {}
_gaussian_project_path: Optional[str] = None
_semantic_query_engine: Optional['SemanticQueryEngine'] = None
_entity_radiance_assets: Dict[str, 'RadianceAsset'] = {}  # entity_id -> RadianceAsset


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

    # Create zone from room with new Zone structure
    bounds = ZoneBounds(
        shape='circle',
        radius=20.0,
        perception_radius=25.0,
        perception_falloff=10.0,
    )
    zone = Zone(
        id=room_id,
        name=room_data.get('name', room_id),
        world_position=Vector3(0, 0, 0),  # Default position
        bounds=bounds,
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
# Gaussian Scene Composition
# =============================================================================

def init_gaussian_scene_integration(project_path: str) -> bool:
    """
    Initialize the Gaussian adapter for scene composition.

    Args:
        project_path: Path to the NoodleStudio project

    Returns:
        True if initialization succeeded
    """
    global _gaussian_project_path

    if not GAUSSIAN_ADAPTER_AVAILABLE:
        logger.warning("[Gaussian] Adapter not available - skipping initialization")
        return False

    try:
        _gaussian_project_path = project_path
        init_gaussian_adapter(project_path)
        asset_manager = get_asset_manager()
        if asset_manager:
            logger.info(f"[Gaussian] Initialized with {len(asset_manager.assets)} assets at {project_path}")
            return True
        else:
            logger.warning("[Gaussian] Asset manager not created")
            return False
    except Exception as e:
        logger.error(f"[Gaussian] Failed to initialize: {e}")
        return False


def compose_gaussian_scene() -> Optional['GaussianScene']:
    """
    Compose a Gaussian scene from the current SceneStateManager state.

    This is the core wiring: ScenePacket -> GaussianScene.
    Call this when you need the renderable Gaussian representation.

    Returns:
        GaussianScene or None if not available
    """
    if not _scene_state_manager:
        logger.debug("[Gaussian] No SceneStateManager - cannot compose scene")
        return None

    if not GAUSSIAN_ADAPTER_AVAILABLE:
        logger.debug("[Gaussian] Adapter not available - cannot compose scene")
        return None

    try:
        # Generate current scene packet
        packet = _scene_state_manager.generate_scene_packet()

        # Compose Gaussian scene
        scene = compose_scene_from_packet(packet)

        if scene:
            logger.debug(f"[Gaussian] Composed scene with {len(scene.instances)} instances")
        return scene
    except Exception as e:
        logger.error(f"[Gaussian] Failed to compose scene: {e}")
        return None


def get_gaussian_scene_json(indent: int = 2) -> Optional[str]:
    """
    Get current Gaussian scene as JSON.

    Useful for sending to external renderers.

    Args:
        indent: JSON indent level

    Returns:
        JSON string or None
    """
    import json
    from dataclasses import asdict

    scene = compose_gaussian_scene()
    if not scene:
        return None

    try:
        # Convert to serializable dict
        scene_dict = {
            "scene_id": scene.scene_id,
            "stage_id": scene.stage_id,
            "stage_name": scene.stage_name,
            "instances": {},
            "assets": {},
            "camera": {
                "position": [scene.camera_position.x, scene.camera_position.y, scene.camera_position.z],
                "target": [scene.camera_target.x, scene.camera_target.y, scene.camera_target.z],
                "fov": scene.camera_fov,
            },
            "lighting": {
                "key_direction": [scene.key_light_direction.x, scene.key_light_direction.y, scene.key_light_direction.z],
                "key_color": scene.key_light_color,
                "key_intensity": scene.key_light_intensity,
                "ambient_color": scene.ambient_color,
            },
            "environment_asset_id": scene.environment_asset_id,
            "skybox_path": scene.skybox_path,
        }

        # Add instances
        for inst_id, inst in scene.instances.items():
            scene_dict["instances"][inst_id] = {
                "instance_id": inst.instance_id,
                "asset_id": inst.asset_id,
                "transform": {
                    "position": [inst.transform.position.x, inst.transform.position.y, inst.transform.position.z],
                    "rotation": [inst.transform.rotation.x, inst.transform.rotation.y, inst.transform.rotation.z],
                    "scale": [inst.transform.scale.x, inst.transform.scale.y, inst.transform.scale.z],
                },
                "zone_id": inst.zone_id,
                "visible": inst.visible,
                "opacity": inst.opacity,
                "tint_color": inst.tint_color,
                "entity_type": inst.entity_type,
                "entity_id": inst.entity_id,
            }

        # Add asset metadata
        for asset_id, asset in scene.assets.items():
            scene_dict["assets"][asset_id] = {
                "id": asset.id,
                "name": asset.name,
                "asset_type": asset.asset_type,
                "ply_path": asset.ply_path,
                "gaussian_count": asset.gaussian_count,
                "file_size_mb": asset.file_size_mb,
                "semantic_tags": asset.semantic_tags,
                "noodling_id": asset.noodling_id,
                "visual_form": asset.visual_form,
            }

        return json.dumps(scene_dict, indent=indent)
    except Exception as e:
        logger.error(f"[Gaussian] Failed to serialize scene: {e}")
        return None


def get_gaussian_asset_manager() -> Optional['GaussianAssetManager']:
    """Get the Gaussian asset manager for direct asset operations."""
    if GAUSSIAN_ADAPTER_AVAILABLE:
        return get_asset_manager()
    return None


def get_gaussian_compositor() -> Optional['GaussianSceneCompositor']:
    """Get the Gaussian scene compositor for custom composition."""
    if GAUSSIAN_ADAPTER_AVAILABLE:
        return get_compositor()
    return None


def get_gaussian_generator() -> Optional['GaussianGenerator']:
    """Get the Gaussian generator for asset generation."""
    if GAUSSIAN_ADAPTER_AVAILABLE:
        return get_generator()
    return None


# =============================================================================
# Semantic Query Integration (CLIP natural language queries)
# =============================================================================

def init_semantic_query_engine() -> bool:
    """
    Initialize the semantic query engine for CLIP-based queries.

    Call this after loading radiance assets to enable natural language
    queries like "where is Red's left hand?"

    Returns:
        True if initialization succeeded
    """
    global _semantic_query_engine

    if not SEMANTIC_QUERY_AVAILABLE:
        logger.warning("[Semantic] Query engine not available - missing dependencies")
        return False

    try:
        _semantic_query_engine = SemanticQueryEngine(auto_generate_embeddings=True)
        logger.info("[Semantic] Query engine initialized")
        return True
    except Exception as e:
        logger.error(f"[Semantic] Failed to initialize query engine: {e}")
        return False


def get_semantic_query_engine() -> Optional['SemanticQueryEngine']:
    """Get the semantic query engine."""
    return _semantic_query_engine


def register_entity_radiance(
    entity_id: str,
    radiance_path: str,
    display_name: str = "",
    entity_type: str = "noodling"
) -> bool:
    """
    Register a radiance asset for an entity.

    This loads the .radiance file and registers it with the semantic
    query engine, enabling natural language queries on its body parts.

    Args:
        entity_id: Entity identifier (e.g., "red_fire_anklebiter")
        radiance_path: Path to .radiance file
        display_name: Human-readable name
        entity_type: "noodling", "prim", "environment"

    Returns:
        True if registration succeeded
    """
    global _entity_radiance_assets

    if not SEMANTIC_QUERY_AVAILABLE:
        return False

    if not _semantic_query_engine:
        init_semantic_query_engine()

    if not _semantic_query_engine:
        return False

    try:
        # Load radiance asset
        asset = load_radiance(radiance_path)
        _entity_radiance_assets[entity_id] = asset

        # Register with query engine (auto-generates CLIP embeddings)
        _semantic_query_engine.register_entity(
            entity_id,
            asset,
            display_name=display_name or entity_id,
            entity_type=entity_type
        )

        logger.info(f"[Semantic] Registered entity: {entity_id} ({asset.gaussian_count} Gaussians)")
        return True

    except Exception as e:
        logger.error(f"[Semantic] Failed to register {entity_id}: {e}")
        return False


def query_scene_semantic(query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """
    Query the scene using natural language.

    Examples:
        - "Red's left hand" -> finds Red's left hand Gaussians
        - "the chair" -> finds chair prims
        - "head" -> finds all head body parts

    Args:
        query: Natural language query
        top_k: Number of results to return

    Returns:
        Dict with matches: [{"entity_id", "body_part", "similarity", "position"}, ...]
    """
    if not _semantic_query_engine:
        return None

    try:
        result = _semantic_query_engine.query_text(query, top_k=top_k)

        return {
            "query": result.query,
            "search_time_ms": result.search_time_ms,
            "matches": [
                {
                    "entity_id": m.entity_id,
                    "body_part": m.body_part,
                    "similarity": m.similarity,
                    "position": list(m.position) if m.position else None,
                }
                for m in result.matches
            ]
        }
    except Exception as e:
        logger.error(f"[Semantic] Query failed: {e}")
        return None


def raycast_scene(
    ray_origin: List[float],
    ray_direction: List[float]
) -> Optional[Dict[str, Any]]:
    """
    Raycast into the scene and return what was hit.

    Used for click-to-inspect in UI.

    Args:
        ray_origin: [x, y, z] ray start
        ray_direction: [x, y, z] ray direction (normalized)

    Returns:
        Dict with hit info: {"hit", "entity_id", "body_part", "position", ...}
    """
    if not _semantic_query_engine:
        return None

    try:
        import numpy as np
        origin = np.array(ray_origin, dtype=np.float32)
        direction = np.array(ray_direction, dtype=np.float32)

        hit = _semantic_query_engine.raycast(origin, direction)

        if hit and hit.hit:
            return {
                "hit": True,
                "entity_id": hit.entity_id,
                "body_part": hit.body_part,
                "body_region": hit.body_region,
                "position": list(hit.position),
                "distance": hit.distance,
                "gaussian_index": hit.gaussian_index,
            }
        else:
            return {"hit": False}

    except Exception as e:
        logger.error(f"[Semantic] Raycast failed: {e}")
        return {"hit": False, "error": str(e)}


def get_entity_visible_body_parts(
    perceiver_id: str,
    target_id: str,
    perceiver_pos: Optional[List[float]] = None,
    perceiver_facing: Optional[List[float]] = None,
    fov: float = 120.0
) -> List[str]:
    """
    Get which body parts of target_id are visible to perceiver_id.

    Uses Gaussian positions and perceiver FOV to determine visibility.

    Args:
        perceiver_id: Who is looking
        target_id: Who is being looked at
        perceiver_pos: Override position (uses scene state if None)
        perceiver_facing: Override facing (uses scene state if None)
        fov: Field of view in degrees

    Returns:
        List of visible body part labels
    """
    if target_id not in _entity_radiance_assets:
        return []

    if not _scene_state_manager:
        return []

    try:
        import numpy as np
        import math

        # Get perceiver position/facing from scene state
        if perceiver_pos is None or perceiver_facing is None:
            perceiver = _scene_state_manager.noodlings.get(perceiver_id)
            if not perceiver:
                perceiver = _scene_state_manager.players.get(perceiver_id)
            if not perceiver:
                return []

            perceiver_pos = [perceiver.position.x, perceiver.position.y, perceiver.position.z]
            perceiver_facing = [perceiver.facing.x, perceiver.facing.y, perceiver.facing.z]

        perceiver_pos = np.array(perceiver_pos, dtype=np.float32)
        perceiver_facing = np.array(perceiver_facing, dtype=np.float32)
        perceiver_facing = perceiver_facing / (np.linalg.norm(perceiver_facing) + 1e-8)

        # Get target radiance asset
        asset = _entity_radiance_assets[target_id]
        if asset.positions is None or not asset.semantic_labels:
            return []

        # Get unique visible labels
        visible_labels = set()
        half_fov_rad = math.radians(fov / 2)

        for i in range(asset.gaussian_count):
            pos = asset.positions[i]
            label = asset.semantic_labels[i] if i < len(asset.semantic_labels) else ""

            if not label:
                continue

            # Direction to Gaussian
            to_gaussian = pos - perceiver_pos
            distance = np.linalg.norm(to_gaussian)
            if distance < 0.01:
                continue

            to_gaussian_norm = to_gaussian / distance

            # Angle check
            dot = np.dot(perceiver_facing, to_gaussian_norm)
            angle = math.acos(max(-1, min(1, dot)))

            if angle <= half_fov_rad:
                visible_labels.add(label)

        return list(visible_labels)

    except Exception as e:
        logger.error(f"[Semantic] Failed to get visible body parts: {e}")
        return []


def get_entity_radiance_asset(entity_id: str) -> Optional['RadianceAsset']:
    """Get the RadianceAsset for an entity if registered."""
    return _entity_radiance_assets.get(entity_id)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Availability check
    "SCENE_PROTOCOL_AVAILABLE",
    "GAUSSIAN_ADAPTER_AVAILABLE",
    "SEMANTIC_QUERY_AVAILABLE",

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

    # Gaussian Scene Composition
    "init_gaussian_scene_integration",
    "compose_gaussian_scene",
    "get_gaussian_scene_json",
    "get_gaussian_asset_manager",
    "get_gaussian_compositor",
    "get_gaussian_generator",

    # Semantic Query (CLIP natural language)
    "init_semantic_query_engine",
    "get_semantic_query_engine",
    "register_entity_radiance",
    "query_scene_semantic",
    "raycast_scene",
    "get_entity_visible_body_parts",
    "get_entity_radiance_asset",
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
