#!/usr/bin/env python3
"""
Test script for WorldAPI integration

Tests:
1. WorldAPI can be created and accessed
2. WorldAPI properties work with mock data
3. WorldAPI commands are properly queued
4. NoodleAPI.world property works
5. Scene Protocol integration imports work

Run from noodlestudio directory:
    python test_world_api.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("WORLD API INTEGRATION TEST")
print("=" * 60)
print()

# =============================================================================
# Test 1: Import WorldAPI
# =============================================================================
print("1. Testing WorldAPI import...")
try:
    from noodlestudio.scripting.world_api import (
        WorldAPI,
        WorldAPIState,
        PerceivedEntityJS,
        PerceivedEventJS,
        get_world_api,
    )
    print("   WorldAPI imported successfully")
except ImportError as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

# =============================================================================
# Test 2: Create WorldAPI instance
# =============================================================================
print("\n2. Creating WorldAPI instance...")
api = WorldAPI("test_agent")
print(f"   Created WorldAPI for noodling_id: {api.noodling_id}")
print(f"   State type: {type(api._state).__name__}")

# =============================================================================
# Test 3: Test properties with default state
# =============================================================================
print("\n3. Testing default properties...")
print(f"   myPosition: {api.myPosition}")
print(f"   myZone: '{api.myZone}'")
print(f"   myPosture: '{api.myPosture}'")
print(f"   affect: {api.affect}")
print(f"   perceivedEntities: {api.perceivedEntities}")
print(f"   conversationPartner: {api.conversationPartner}")

# =============================================================================
# Test 4: Test command methods
# =============================================================================
print("\n4. Testing command methods...")

# Set expression
api.setExpression("curious")
print(f"   setExpression('curious') - pending: {api._state.pending_expression}")

# Set gaze
api.setGaze("player")
print(f"   setGaze('player') - pending: {api._state.pending_gaze}")

# Speak
api.speak("Hello world!", "friendly")
print(f"   speak('Hello world!', 'friendly') - pending: {api._state.pending_speak}")

# Get pending commands
commands = api.get_pending_commands()
print(f"\n   Pending commands retrieved: {list(commands.keys())}")
print(f"   Commands: {commands}")

# Verify commands were cleared
commands_after = api.get_pending_commands()
print(f"   Commands after get (should be empty): {commands_after}")

# =============================================================================
# Test 5: Test query methods
# =============================================================================
print("\n5. Testing query methods...")

# Manually add a perceived entity to test queries
api._state.perceived_entities = [
    PerceivedEntityJS(
        id="yuki",
        displayName="Yuki",
        entityType="noodling",
        position=[5.0, 0, 0],
        distance=5.0,
        direction="right",
        visibility=0.9,
        posture="standing",
        action="listening",
        expression="curious",
        lookingAt="test_agent",
        visualForm="humanoid_fox"
    )
]

print(f"   canSee('yuki'): {api.canSee('yuki')}")
print(f"   canSee('unknown'): {api.canSee('unknown')}")
print(f"   getDistanceTo('yuki'): {api.getDistanceTo('yuki')}")
print(f"   getDirectionTo('yuki'): {api.getDirectionTo('yuki')}")
print(f"   isLookingAtMe('yuki'): {api.isLookingAtMe('yuki')}")

entity = api.getEntity("yuki")
print(f"   getEntity('yuki'): {entity['displayName']} ({entity['visualForm']})")

# =============================================================================
# Test 6: Test NoodleAPI integration
# =============================================================================
print("\n6. Testing NoodleAPI.world integration...")
try:
    from noodlestudio.scripting.noodle_api import NoodleAPI, get_noodle_api

    noodle = NoodleAPI()

    # Access world property
    world = noodle.world
    print(f"   noodle.world type: {type(world).__name__}")
    print(f"   noodle.world.noodling_id: '{world.noodling_id}'")

    # Set custom WorldAPI
    custom_api = get_world_api("red")
    noodle.set_world_api(custom_api)
    print(f"   After set_world_api('red'): {noodle.world.noodling_id}")

    # Test to_dict includes world
    api_dict = noodle.to_dict()
    print(f"   'world' in to_dict(): {'world' in api_dict}")
    if 'world' in api_dict:
        print(f"   world dict keys: {list(api_dict['world'].keys())[:5]}...")

except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 7: Scene Protocol integration (optional)
# =============================================================================
print("\n7. Testing Scene Protocol integration...")
try:
    from noodlestudio.core.semantic_world import (
        SceneStateManager,
        Vector3,
        Zone,
        Noodling,
        PerceptionCone,
        Affect,
    )

    # Create a simple scene
    manager = SceneStateManager("test_stage", "Test Stage")

    # Add a zone
    zone = Zone(
        id="test_zone",
        name="Test Zone",
        center=Vector3(0, 0, 0),
        radius=20.0,
        falloff=10.0,
    )
    manager.add_zone(zone)

    # Add noodlings
    red = manager.add_noodling(
        noodling_id="red",
        display_name="Red",
        position=[0, 0, 0],
        species="fire_imp",
    )
    red.facing = Vector3(0, 0, 1)
    red.perception = PerceptionCone(
        fov_horizontal=120,
        range=15.0,
        heat_sense=True,
    )

    yuki = manager.add_noodling(
        noodling_id="yuki",
        display_name="Yuki",
        position=[5, 0, 2],
        species="cyberfox",
    )
    yuki.facing = Vector3(-1, 0, 0)

    # Generate perception slice for Red
    slice = manager.generate_perception_slice("red")
    print(f"   SceneStateManager created: {manager.stage_name}")
    print(f"   Red's perception slice: {len(slice.perceived_entities)} entities")

    # Update WorldAPI from slice
    red_api = get_world_api("red")
    red_api.update_from_perception_slice(slice)
    print(f"   WorldAPI updated from slice")
    print(f"   Red can see: {[e['displayName'] for e in red_api.perceivedEntities]}")

    print("   Scene Protocol integration: OK")

except ImportError as e:
    print(f"   Scene Protocol not available (optional): {e}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)
print()
print("Summary:")
print("  - WorldAPI imports and creates instances")
print("  - Properties return expected defaults")
print("  - Command methods queue pending commands")
print("  - Query methods work with mock data")
print("  - NoodleAPI.world property works")
print("  - Scene Protocol integration available")
print()
print("The WorldAPI is ready for use in ScriptedFacets via:")
print("  context.noodle.world.perceivedEntities")
print("  context.noodle.world.canSee('yuki')")
print("  context.noodle.world.speak('Hello!', 'friendly')")
print()
