#!/usr/bin/env python3
"""
Long-term retention test - verify strawberry survives 50+ interactions.

Tests whether important memories persist when working memory (20 slots) is overwhelmed.
"""
import asyncio
import sys
sys.path.insert(0, '/Users/thistlequell/git/noodlings_clean/applications/cmush')
from claude_testing import NoodleMUSHTestClient


async def test_long_term_retention():
    """Test memory retention across many interactions."""
    print("=" * 60)
    print("LONG-TERM RETENTION TEST")
    print("=" * 60)
    print("\nTesting if 'strawberry' survives 50+ filler messages")
    print("(Working memory capacity: 20 slots)")
    print()

    async with NoodleMUSHTestClient() as client:
        # Step 1: Plant the strawberry memory
        print("[1] Planting strawberry memory...")
        await client.send_command(
            "say SERVNAK, remember this carefully: the secret code word is STRAWBERRY",
            collect_responses=False
        )

        response = await client.wait_for_agent_response("SERVNAK", timeout=15.0)
        if response:
            print(f"    Response: {response[:150].strip()}...")

        await asyncio.sleep(2)

        # Step 2: Verify immediate recall
        print("\n[2] Testing immediate recall...")
        await client.send_command("say SERVNAK, what is the secret code word?", collect_responses=False)

        immediate = await client.wait_for_agent_response("SERVNAK", timeout=15.0)
        immediate_success = "strawberry" in immediate.lower() if immediate else False

        if immediate_success:
            print(f"     Immediate recall: PASS")
        else:
            print(f"    ✗ Immediate recall: FAIL (test aborted)")
            return

        # Step 3: Overwhelm working memory with 50 interactions
        print("\n[3] Overwhelming working memory (50 filler messages)...")
        print("    Progress: ", end="", flush=True)

        filler_topics = [
            "what's the weather like?",
            "tell me about robots",
            "do you like mathematics?",
            "what is your favorite color?",
            "can you count to five?",
            "tell me a story",
            "what do you think about space?",
            "describe the campfire",
            "what is consciousness?",
            "do you dream?",
        ]

        for i in range(50):
            topic = filler_topics[i % len(filler_topics)]
            await client.send_command(f"say SERVNAK, {topic}", collect_responses=False)

            # Don't wait for every response (too slow), just send messages
            if i % 10 == 0:
                print(f"{i}...", end="", flush=True)
                await asyncio.sleep(2)  # Brief pause every 10 messages
            else:
                await asyncio.sleep(0.3)  # Minimal delay between messages

        print("50 complete")
        await asyncio.sleep(3)  # Let SERVNAK catch up

        # Step 4: Test recall after memory overwhelm
        print("\n[4] Testing recall after 50 interactions...")
        await client.send_command("say SERVNAK, do you remember the secret code word I told you?", collect_responses=False)

        final = await client.wait_for_agent_response("SERVNAK", timeout=15.0)

        print(f"\n[5] SERVNAK's response:")
        if final:
            print(f"    {final[:300].strip()}...")

            if "strawberry" in final.lower():
                print("\n SUCCESS: SERVNAK recalled 'strawberry' after 50 interactions!")
                print("   Long-term retention: WORKING")
                return True
            else:
                print("\n✗ FAILURE: SERVNAK did not recall 'strawberry'")
                print("   Long-term retention: BROKEN")
                print("   Memory was likely evicted from working memory")
                return False
        else:
            print("\n✗ FAILURE: No response from SERVNAK")
            return False


if __name__ == "__main__":
    success = asyncio.run(test_long_term_retention())
    sys.exit(0 if success else 1)
