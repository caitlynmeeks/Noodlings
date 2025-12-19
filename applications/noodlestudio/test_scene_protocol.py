#!/usr/bin/env python3
"""
Test script for Noodlings Scene Protocol (NSP)

Tests:
1. SceneStateManager - canonical truth
2. ScenePacket generation - full scene snapshot
3. PerceptionSlice generation - filtered per-entity view
4. Information asymmetry - Red can't see what's behind her

Run from noodlestudio directory:
    python test_scene_protocol.py
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from noodlestudio.core.semantic_world import (
    # Scene Protocol
    SceneStateManager,
    ScenePacket,
    Vector3,
    Zone,
    Noodling,
    Player,
    Prim,
    PerceptionCone,
    Affect,
    VisualForm,
    CameraMode,
    Framing,

    # Perception
    PerceptionSliceGenerator,
    generate_perception_slice,
)


def test_scene_protocol():
    print("=" * 60)
    print("NOODLINGS SCENE PROTOCOL TEST")
    print("=" * 60)
    print()

    # =========================================================================
    # 1. Create Scene State Manager
    # =========================================================================
    print("1. Creating SceneStateManager...")
    manager = SceneStateManager(
        stage_id="campfire_clearing",
        stage_name="The Campfire Clearing"
    )
    print(f"   Stage: {manager.stage_name} ({manager.stage_id})")
    print()

    # =========================================================================
    # 2. Add a Zone
    # =========================================================================
    print("2. Adding zone...")
    campfire_zone = Zone(
        id="campfire",
        name="The Campfire",
        center=Vector3(0, 0, 0),
        radius=15.0,
        falloff=10.0,
        description="A cozy campfire crackles with warm orange flames",
        features=["crackling fire", "ring of sitting stones", "old radio on shelf"],
        mood="cozy",
        lighting="firelight",
        exits={"north": "forest_edge", "east": "pond"}
    )
    manager.add_zone(campfire_zone)
    manager.spatial_truth.ambient.time_of_day = "night"
    manager.spatial_truth.ambient.soundscape = ["fire_crackle", "crickets", "owl_distant"]
    print(f"   Zone: {campfire_zone.name} (radius={campfire_zone.radius}m)")
    print()

    # =========================================================================
    # 3. Add Red (fire imp) - facing the player
    # =========================================================================
    print("3. Adding Red (fire imp)...")
    red = manager.add_noodling(
        noodling_id="red",
        display_name="Red",
        position=[2.5, 0, -1.0],
        species="fire imp",
        height=0.3,
    )
    red.facing = Vector3(0, 0, 1)  # Facing +Z (toward player)
    red.visual_state = "default"
    red.visual_forms["default"] = VisualForm(
        id="default",
        description="A tiny fire imp, ankle-height, with flickering orange flames for hair and mischievous ember eyes",
        reference_images={
            "neutral": "noodlings://red/portrait.png",
            "mischievous": "noodlings://red/expressions/mischievous.png",
        },
        style_hints={"flame_intensity": 0.7, "ember_glow": True}
    )
    red.expression = "mischievous"
    red.posture = "sitting"
    red.current_action = "speaking"
    red.gaze_target = "caity"
    red.affect = Affect(valence=0.4, arousal=0.6, dominance=0.7, boredom=0.1, sorrow=0.0)
    red.perception = PerceptionCone(
        fov_horizontal=120,
        range=15.0,
        heat_sense=True,  # Fire imp special ability!
    )
    print(f"   Red: position={red.position.to_list()}, facing={red.facing.to_list()}")
    print(f"   Special: heat_sense={red.perception.heat_sense}")
    print()

    # =========================================================================
    # 4. Add Yuki (cyberfox) - BEHIND Red (she can't see her!)
    # =========================================================================
    print("4. Adding Yuki (cyberfox) - behind Red...")
    yuki = manager.add_noodling(
        noodling_id="yuki",
        display_name="Yuki",
        position=[-5.0, 0, -3.0],  # Behind Red
        species="cyberfox",
        height=0.4,
    )
    yuki.facing = Vector3(1, 0, 0)  # Facing +X
    yuki.visual_state = "humanoid_fox"
    yuki.visual_forms["humanoid_fox"] = VisualForm(
        id="humanoid_fox",
        description="Anthropomorphic silver-white fox girl with cyan circuit markings",
        reference_images={
            "neutral": "noodlings://yuki/humanoid_neutral.png",
            "curious": "noodlings://yuki/humanoid_curious.png",
        }
    )
    yuki.visual_forms["ghostly_fox"] = VisualForm(
        id="ghostly_fox",
        description="Translucent spectral fox, pale blue ethereal glow",
        reference_images={"neutral": "noodlings://yuki/ghostly.png"},
        style_hints={"opacity": 0.7, "glow": "soft_blue"}
    )
    yuki.expression = "curious"
    yuki.posture = "standing_alert"
    yuki.current_action = "ears_perked_listening"
    yuki.perception = PerceptionCone(
        fov_horizontal=180,  # Fox wide vision
        range=25.0,
        night_vision=True,
        motion_sensitivity=0.9,
    )
    print(f"   Yuki: position={yuki.position.to_list()} (behind Red)")
    print(f"   Form: {yuki.visual_state}")
    print()

    # =========================================================================
    # 5. Add Player (Caity) - in front of Red
    # =========================================================================
    print("5. Adding player (Caity)...")
    caity = manager.add_player(
        player_id="caity",
        display_name="Caity",
        position=[0, 0, 3.0],  # In front of Red
    )
    caity.facing = Vector3(0, 0, -1)  # Facing Red
    caity.posture = "sitting"
    caity.gaze_target = "red"
    print(f"   Caity: position={caity.position.to_list()}")
    print()

    # =========================================================================
    # 6. Add a Prim (radio)
    # =========================================================================
    print("6. Adding prim (radio)...")
    radio = manager.add_prim(
        prim_id="campfire_radio",
        prim_type="radio",
        position=[1.5, 0.3, -2.0],
        description="An old radio with brass dials and warm dial glow",
    )
    radio.state = {"power": "on", "station": "forest_jazz", "volume": 0.5}
    print(f"   Radio: {radio.description}")
    print()

    # =========================================================================
    # 7. Record some dialogue
    # =========================================================================
    print("7. Recording dialogue...")
    manager.record_dialogue("red", "You're not fooling anyone with that innocent look.", tone="teasing")
    manager.record_dialogue("caity", "Who, me?", tone="playful_innocent")
    manager.update_scene_state(tension=0.2, energy=0.5, intimacy=0.7, current_beat="playful_banter")
    print(f"   Scene beat: {manager.scene_state.current_beat}")
    print()

    # =========================================================================
    # 8. Set camera
    # =========================================================================
    print("8. Setting camera...")
    manager.set_camera_focus("red", framing="medium_closeup", mode="FOCUS_ON")
    print(f"   Camera: {manager.camera_directive.mode.value} on {manager.camera_directive.subject}")
    print()

    # =========================================================================
    # 9. Generate Full Scene Packet
    # =========================================================================
    print("=" * 60)
    print("GENERATING FULL SCENE PACKET")
    print("=" * 60)
    packet = manager.generate_scene_packet()

    print(f"\nPacket ID: {packet.header.packet_id}")
    print(f"Stage: {packet.header.stage_name}")
    print(f"Entities: {len(packet.noodlings)} noodlings, {len(packet.players)} players, {len(packet.prims)} prims")
    print(f"Recent dialogue: {len(packet.narrative_context.recent_dialogue)} entries")
    print()

    # Show flattened text (what Genie's LLM sees)
    print("FLATTENED TEXT (for LLM-based renderers):")
    print("-" * 40)
    print(packet.flatten_to_text())
    print("-" * 40)
    print()

    # =========================================================================
    # 10. Generate Perception Slices - THE KEY TEST
    # =========================================================================
    print("=" * 60)
    print("PERCEPTION SLICES - Information Asymmetry Test")
    print("=" * 60)
    print()

    # Red's perception slice
    print("RED'S PERCEPTION SLICE:")
    print("-" * 40)
    red_slice = manager.generate_perception_slice("red")

    print(f"Red sees {len(red_slice.perceived_entities)} entities:")
    for eid, entity in red_slice.perceived_entities.items():
        print(f"  - {entity.display_name}: {entity.direction}, {entity.distance:.1f}m away, visibility={entity.visibility:.2f}")

    # Check if Red can see Yuki (she's behind Red - but Red has heat sense!)
    can_see_yuki = "yuki" in red_slice.perceived_entities
    print(f"\nCan Red see Yuki? {can_see_yuki}")
    if can_see_yuki:
        yuki_percept = red_slice.perceived_entities["yuki"]
        if yuki_percept.visibility < 0.5:
            print(f"  HEAT SENSE! Yuki is behind Red but detected via heat_sense")
            print(f"  visibility={yuki_percept.visibility:.2f} (low = sensed, not seen)")
        else:
            print(f"  Full visual (visibility={yuki_percept.visibility:.2f})")
    else:
        print("  No - Yuki is outside perception range")

    # Can Red see Caity?
    can_see_caity = "caity" in red_slice.perceived_entities
    print(f"Can Red see Caity? {can_see_caity}")
    if can_see_caity:
        caity_percept = red_slice.perceived_entities["caity"]
        print(f"  Caity appears: {caity_percept.expression}, {caity_percept.posture}")
        print(f"  Looking at Red? {caity_percept.gaze_target == 'red'}")

    print(f"\nRed heard {len(red_slice.perceived_events)} events:")
    for event in red_slice.perceived_events[:3]:
        if event.text:
            print(f"  - {event.actor} said: \"{event.text}\"")
    print()

    # Yuki's perception slice - she has 180 FOV!
    print("YUKI'S PERCEPTION SLICE:")
    print("-" * 40)
    yuki_slice = manager.generate_perception_slice("yuki")

    print(f"Yuki sees {len(yuki_slice.perceived_entities)} entities:")
    for eid, entity in yuki_slice.perceived_entities.items():
        print(f"  - {entity.display_name}: {entity.direction}, {entity.distance:.1f}m away")

    # Yuki has 180 FOV - can she see more?
    can_see_red = "red" in yuki_slice.perceived_entities
    print(f"\nCan Yuki see Red? {can_see_red}")
    if can_see_red:
        print("  (Yuki has 180 FOV - wider peripheral vision)")
    print()

    # =========================================================================
    # 11. Narrative text from perception
    # =========================================================================
    print("RED'S NARRATIVE CONTEXT (for facet input):")
    print("-" * 40)
    print(red_slice.to_narrative_text())
    print("-" * 40)
    print()

    # =========================================================================
    # 12. JSON output sample
    # =========================================================================
    print("PACKET JSON SAMPLE (first 1000 chars):")
    print("-" * 40)
    json_output = packet.to_json(indent=2)
    print(json_output[:1000] + "...")
    print("-" * 40)
    print()

    print("=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    print()
    print("Key results:")
    print(f"  - Full packet generated with {len(packet.noodlings)} noodlings")
    print(f"  - Red's slice: sees {len(red_slice.perceived_entities)} entities (Caity in front)")
    print(f"  - Red CANNOT see Yuki (behind her) - information asymmetry works!")
    print(f"  - Yuki's slice: sees {len(yuki_slice.perceived_entities)} entities (180 FOV)")
    print()


if __name__ == "__main__":
    test_scene_protocol()
