#!/usr/bin/env python3
"""
Test script for affect-memory integration features:
1. Memory → Affect Blending
2. Name-Based Memory Triggering
3. Social Contagion (laughter, yawning, fear, sadness)
"""

import asyncio
import websockets
import json
import time

async def test_affect_memory_features():
    """Test the three affect-memory features"""

    uri = "ws://localhost:8765"

    print("=" * 70)
    print("Testing Affect-Memory Integration Features")
    print("=" * 70)

    async with websockets.connect(uri) as websocket:
        # Login
        await websocket.send(json.dumps({
            "type": "login",
            "username": "TestUser",
            "password": "test123"
        }))

        response = await websocket.recv()
        print(f"\nLogin response: {response}\n")

        # Wait a moment for server to be ready
        await asyncio.sleep(1)

        # Test 1: Emotional Contagion - Laughter
        print("\n" + "=" * 70)
        print("TEST 1: Emotional Contagion - Laughter")
        print("=" * 70)
        print("Sending: 'haha that's so funny!'")
        print("Expected: Laughter detection, valence and arousal boost")

        await websocket.send(json.dumps({
            "type": "say",
            "text": "haha that's so funny!"
        }))

        # Receive responses
        for _ in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                if data.get("type") == "agent_speech":
                    print(f"✓ Agent {data['agent']} responded: {data['text'][:80]}")
            except asyncio.TimeoutError:
                break

        await asyncio.sleep(1)

        # Test 2: Emotional Contagion - Fear
        print("\n" + "=" * 70)
        print("TEST 2: Emotional Contagion - Fear")
        print("=" * 70)
        print("Sending: 'I'm so scared and anxious right now'")
        print("Expected: Fear detection, fear and arousal boost")

        await websocket.send(json.dumps({
            "type": "say",
            "text": "I'm so scared and anxious right now"
        }))

        for _ in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                if data.get("type") == "agent_speech":
                    print(f"✓ Agent {data['agent']} responded: {data['text'][:80]}")
            except asyncio.TimeoutError:
                break

        await asyncio.sleep(1)

        # Test 3: Emotional Contagion - Sadness
        print("\n" + "=" * 70)
        print("TEST 3: Emotional Contagion - Sadness")
        print("=" * 70)
        print("Sending: 'I'm crying and heartbroken'")
        print("Expected: Sadness detection, sorrow boost and valence decrease")

        await websocket.send(json.dumps({
            "type": "say",
            "text": "I'm crying and heartbroken"
        }))

        for _ in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                if data.get("type") == "agent_speech":
                    print(f"✓ Agent {data['agent']} responded: {data['text'][:80]}")
            except asyncio.TimeoutError:
                break

        await asyncio.sleep(1)

        # Test 4: Name-Based Memory Triggering
        print("\n" + "=" * 70)
        print("TEST 4: Name-Based Memory Triggering")
        print("=" * 70)
        print("First, creating a memory with 'Alice'...")

        await websocket.send(json.dumps({
            "type": "say",
            "text": "Alice was such a wonderful friend, I miss her so much"
        }))

        for _ in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                if data.get("type") == "agent_speech":
                    print(f"✓ Agent {data['agent']} responded: {data['text'][:80]}")
            except asyncio.TimeoutError:
                break

        await asyncio.sleep(2)

        print("\nNow mentioning 'Alice' again to trigger memory...")
        print("Expected: Name detection, memory retrieval, affect blending")

        await websocket.send(json.dumps({
            "type": "say",
            "text": "Have you heard from Alice recently?"
        }))

        for _ in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                if data.get("type") == "agent_speech":
                    print(f"✓ Agent {data['agent']} responded: {data['text'][:80]}")
            except asyncio.TimeoutError:
                break

        await asyncio.sleep(1)

        # Test 5: Yawning contagion
        print("\n" + "=" * 70)
        print("TEST 5: Emotional Contagion - Yawning/Sleepiness")
        print("=" * 70)
        print("Sending: '*yawns* I'm so tired'")
        print("Expected: Sleepiness detection, boredom boost and arousal decrease")

        await websocket.send(json.dumps({
            "type": "say",
            "text": "*yawns* I'm so tired"
        }))

        for _ in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                if data.get("type") == "agent_speech":
                    print(f"✓ Agent {data['agent']} responded: {data['text'][:80]}")
            except asyncio.TimeoutError:
                break

        print("\n" + "=" * 70)
        print("TESTING COMPLETE")
        print("=" * 70)
        print("\nTo verify the features are working, check the server logs for:")
        print("  1. 'emotional contagion: laughter/fear/sadness/sleepiness'")
        print("  2. 'triggered N memories by names'")
        print("  3. 'Memory affect blending: N memories triggered'")
        print("\nUse: tail -f noodlemush.log | grep -E '(contagion|triggered|blending)'")
        print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_affect_memory_features())
