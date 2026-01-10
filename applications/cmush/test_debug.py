#!/usr/bin/env python3
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
# MODULE:   applications.cmush.test_debug
# PURPOSE:  Debug test with extensive message flow logging
# LAYER:    Backend / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Debug test - verify message flow with extensive logging.
"""
import asyncio
import os
import sys

# Add cmush directory to path
_cmush_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _cmush_dir)

from claude_testing import NoodleMUSHTestClient


async def test_debug():
    """Debug test with logging."""
    print("=" * 60)
    print("DEBUG TEST - Message Flow Verification")
    print("=" * 60)

    client = NoodleMUSHTestClient()
    await client.connect()
    print(f" Connected")

    # Wait for login to complete
    await asyncio.sleep(2)

    # Clear any queued messages
    cleared_count = 0
    while not client.message_queue.empty():
        client.message_queue.get_nowait()
        cleared_count += 1
    print(f" Cleared {cleared_count} queued messages")

    # Send a simple message
    print("\nSending: 'say hello SERVNAK!'")
    await client.send_command("say hello SERVNAK!", collect_responses=False)

    # Now actively wait and log ALL incoming messages
    print("\nWaiting for messages (15 seconds)...")
    print("-" * 60)

    deadline = asyncio.get_event_loop().time() + 15.0
    message_count = 0

    while asyncio.get_event_loop().time() < deadline:
        try:
            msg = await asyncio.wait_for(
                client.message_queue.get(),
                timeout=1.0
            )
            message_count += 1
            msg_type = msg.get('type')
            text = msg.get('text', '')[:80]  # First 80 chars
            print(f"[{message_count}] Type: {msg_type}, Text: {text}")

            # Check if it's from SERVNAK
            if 'SERVNAK' in text.upper():
                print(f"    ^^^ SERVNAK MESSAGE DETECTED!")

        except asyncio.TimeoutError:
            print(".", end="", flush=True)
            continue

    print(f"\n\nTotal messages received: {message_count}")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(test_debug())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
