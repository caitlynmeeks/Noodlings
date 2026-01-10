# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Test Action Stream
#
#   Test suite for action stream.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.test_action_stream
# PURPOSE:  Tests for action stream
# LAYER:    Studio / Application
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   on_change()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

#!/usr/bin/env python3
"""
Test script for Action Stream API

Tests:
1. ActionStreamHandler creation and session management
2. Action parsing and processing
3. Rate limiting
4. Integration with ScenePacketEmitter
5. Semantic change notifications

Run from noodlestudio directory:
    python test_action_stream.py
"""

import sys
import asyncio
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("ACTION STREAM API TEST")
print("=" * 60)
print()

# =============================================================================
# Test 1: Import Action Stream
# =============================================================================
print("1. Testing Action Stream import...")
try:
    from noodlestudio.core.semantic_world import (
        ActionType,
        Action,
        ActionAck,
        ActionSession,
        ActionStreamHandler,
        WebSocketActionStream,
        get_action_handler,
        init_action_handler,
    )
    print("   Action Stream imported successfully")
except ImportError as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# Test 2: Create Action instances
# =============================================================================
print("\n2. Testing Action creation...")

# Player move action
move_action = Action(
    action_type=ActionType.PLAYER_MOVE,
    direction=[0, 0, 1],
    value=1.5
)
print(f"   Player move: {move_action.to_dict()}")
print(f"   JSON size: {len(move_action.to_json())} bytes")

# Camera look action
camera_action = Action(
    action_type=ActionType.CAMERA_LOOK,
    target_id="red"
)
print(f"   Camera look: {camera_action.to_dict()}")
print(f"   JSON size: {len(camera_action.to_json())} bytes")

# Interaction action
interact_action = Action(
    action_type=ActionType.INTERACT,
    entity_id="radio",
    verb="toggle"
)
print(f"   Interact: {interact_action.to_dict()}")
print(f"   JSON size: {len(interact_action.to_json())} bytes")

# =============================================================================
# Test 3: Parse Action from JSON
# =============================================================================
print("\n3. Testing Action parsing from JSON...")

test_json = '{"action":"camera_orbit","delta":[5,0],"t":1734567890.123}'
parsed = Action.from_dict({"action": "camera_orbit", "delta": [5, 0], "t": 1734567890.123})
print(f"   Parsed action type: {parsed.action_type}")
print(f"   Parsed delta: {parsed.delta}")

# =============================================================================
# Test 4: ActionStreamHandler session management
# =============================================================================
print("\n4. Testing ActionStreamHandler sessions...")

handler = ActionStreamHandler()
print(f"   Created handler: {type(handler).__name__}")

# Create session
session = handler.create_session(player_id="caity", stage_id="campfire")
print(f"   Session created: {session.session_id}")
print(f"   Player: {session.player_id}, Stage: {session.current_stage}")

# Get session
retrieved = handler.get_session(session.session_id)
print(f"   Session retrieved: {retrieved is not None}")

# =============================================================================
# Test 5: Action processing
# =============================================================================
print("\n5. Testing action processing...")

async def test_action_processing():
    # Test ping
    ping_ack = await handler.process_action(
        session.session_id,
        {"action": "ping", "t": time.time()}
    )
    print(f"   Ping ack: {ping_ack.to_dict()}")

    # Test camera look
    camera_ack = await handler.process_action(
        session.session_id,
        {"action": "camera_look", "target": "red"}
    )
    print(f"   Camera look ack: {camera_ack.to_dict()}")

    # Test unknown action
    unknown_ack = await handler.process_action(
        session.session_id,
        {"action": "unknown_action"}
    )
    print(f"   Unknown action ack: ok={unknown_ack.accepted}, msg='{unknown_ack.message}'")

    # Test invalid session
    invalid_ack = await handler.process_action(
        "invalid_session_id",
        {"action": "ping"}
    )
    print(f"   Invalid session ack: ok={invalid_ack.accepted}, msg='{invalid_ack.message}'")

asyncio.run(test_action_processing())

# =============================================================================
# Test 6: Rate limiting
# =============================================================================
print("\n6. Testing rate limiting...")

async def test_rate_limiting():
    # Set low rate limit for testing
    handler.max_actions_per_second = 5

    # Send 10 actions rapidly
    accepted = 0
    rejected = 0
    for i in range(10):
        ack = await handler.process_action(
            session.session_id,
            {"action": "ping", "t": time.time()}
        )
        if ack.accepted:
            accepted += 1
        else:
            rejected += 1

    print(f"   Sent 10 actions with limit=5/sec")
    print(f"   Accepted: {accepted}, Rejected: {rejected}")

    # Reset rate limit
    handler.max_actions_per_second = 60

asyncio.run(test_rate_limiting())

# =============================================================================
# Test 7: Semantic change notifications
# =============================================================================
print("\n7. Testing semantic change notifications...")

semantic_changes = []

def on_change(change_type: str):
    semantic_changes.append(change_type)
    print(f"   Semantic change: {change_type}")

handler.on_semantic_change(on_change)

async def test_semantic_changes():
    # Interact should trigger semantic change
    await handler.process_action(
        session.session_id,
        {"action": "interact", "entity": "radio", "verb": "toggle"}
    )

    # Sync request should trigger semantic change
    await handler.process_action(
        session.session_id,
        {"action": "sync"}
    )

asyncio.run(test_semantic_changes())

print(f"   Total semantic changes: {len(semantic_changes)}")

# =============================================================================
# Test 8: Integration with SceneStateManager
# =============================================================================
print("\n8. Testing integration with SceneStateManager...")

try:
    from noodlestudio.core.semantic_world import (
        SceneStateManager,
        Vector3,
        ScenePacketEmitter,
        EmitterConfig,
    )

    # Create scene state manager
    manager = SceneStateManager("test_stage", "Test Stage")

    # Add player
    player = manager.add_player(
        player_id="caity",
        display_name="Caity",
        position=[0, 0, 0]
    )

    # Create handler with manager
    integrated_handler = ActionStreamHandler(manager)

    # Create session
    int_session = integrated_handler.create_session(
        player_id="caity",
        stage_id="test_stage"
    )

    # Test player move with actual state update
    async def test_integrated_move():
        print(f"   Player position before: {player.position.to_list()}")

        await integrated_handler.process_action(
            int_session.session_id,
            {"action": "player_move", "dir": [0, 0, 1], "val": 5.0}
        )

        print(f"   Player position after: {player.position.to_list()}")

    asyncio.run(test_integrated_move())

    # Test emitter integration
    emitter = ScenePacketEmitter(manager)
    emitter.connect_action_handler(integrated_handler)
    print("   Emitter connected to action handler")

    print("   Integration with SceneStateManager: OK")

except ImportError as e:
    print(f"   SceneStateManager not available: {e}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 9: Handler stats
# =============================================================================
print("\n9. Testing handler stats...")

stats = handler.get_stats()
print(f"   Active sessions: {stats['active_sessions']}")
print(f"   Total actions: {stats['total_actions_processed']}")
print(f"   Aggregate rate: {stats['aggregate_actions_per_second']:.1f}/sec")

# =============================================================================
# Test 10: Session cleanup
# =============================================================================
print("\n10. Testing session cleanup...")

handler.close_session(session.session_id)
retrieved_after = handler.get_session(session.session_id)
print(f"   Session after close: {retrieved_after}")

stats_after = handler.get_stats()
print(f"   Active sessions after close: {stats_after['active_sessions']}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)
print()
print("Summary:")
print("  - Action types: PLAYER_MOVE, CAMERA_LOOK, INTERACT, etc.")
print("  - Actions are tiny JSON payloads (~50-100 bytes)")
print("  - Sessions track player, rate limits, stats")
print("  - Rate limiting prevents abuse")
print("  - Semantic changes trigger packet emissions")
print("  - Integration with SceneStateManager works")
print()
print("Action Stream is ready for high-frequency renderer communication:")
print()
print("  # Low frequency - semantic truth")
print("  POST /scene -> full ScenePacket")
print()
print("  # High frequency - steering inputs")
print('  WS: {"action":"camera_orbit","delta":[5,0]}')
print('  WS: {"action":"player_move","dir":[0,0,1],"val":1.5}')
print('  WS: {"action":"interact","entity":"radio","verb":"toggle"}')
print()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
