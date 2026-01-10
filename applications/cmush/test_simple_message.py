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
# MODULE:   applications.cmush.test_simple_message
# PURPOSE:  Simple diagnostic test for message flow
# LAYER:    Backend / Tests
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Simple diagnostic test - send one message and see ALL responses.
"""
import asyncio
import os
import sys

# Add cmush directory to path
_cmush_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _cmush_dir)

from claude_testing import NoodleMUSHTestClient


async def test_simple():
    """Send a simple message and dump all responses."""
    print("Sending 'say hello SERVNAK' and dumping ALL responses...")
    print("=" * 60)

    async with NoodleMUSHTestClient() as client:
        # Send message
        responses = await client.send_command("say hello SERVNAK", wait_for_response=5.0)

        print(f"\nReceived {len(responses)} messages:")
        print("=" * 60)

        for i, msg in enumerate(responses):
            print(f"\n[Message {i+1}]")
            print(f"Type: {msg.get('type')}")
            print(f"Full content: {msg}")
            print("-" * 40)


if __name__ == "__main__":
    asyncio.run(test_simple())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
