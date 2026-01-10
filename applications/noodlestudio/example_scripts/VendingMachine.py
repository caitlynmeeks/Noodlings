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
#   Vending Machine Example Script
#
#   This example shows how to create an interactive prop that
#   responds to the "use" action and produces random results.
#
#   The vending machine has an inventory of silly items. When
#   someone uses it, a random item gets spawned into the world.
#   After 20 uses, the machine breaks down.
#
#   This demonstrates:
#   - Handling OnUse events
#   - Random selection from a list
#   - Spawning props with RezPrim()
#   - Usage limits and degradation
#   - Achievement/easter egg triggers
#
#   A simple economy in a virtual snack.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.example_scripts.VendingMachine
# PURPOSE:  Example of use-to-dispense item spawner
# LAYER:    Scripting / Examples
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   VendingMachine    NoodleScript that dispenses random items
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Vending Machine Example - Buy items with... whatever currency you want!

A vending machine that dispenses random items when used.

Usage:
1. Create prim "Vending Machine"
2. Attach this script
3. @use vending machine
4. Random item spawns!
"""

from noodlestudio.scripting import NoodleScript, Noodlings, Debug
import random


class VendingMachine(NoodleScript):
    """Dispenses random items when used."""

    def Start(self):
        """Initialize vending machine inventory."""
        self.items = [
            "Tensor Taffy",
            "Atomic Fireball",
            "Noodling Plushie",
            "Mysterious Potion",
            "Glowing Crystal",
            "Ancient Scroll",
            "Rubber Duck",
            "Portal Gun (Broken)"
        ]

        self.price = 5  # Gold coins? Krugerrands? You decide!
        self.uses = 0

        Debug.Log("Vending machine stocked and ready!")

    def OnUse(self, user):
        """Someone used the vending machine!"""
        self.uses += 1

        # Pick random item
        item = random.choice(self.items)

        Debug.Log(f"{user} bought {item} from vending machine")

        # Rez the item
        rezzed = Noodlings.RezPrim("prop", item, room=self.prim.room)

        # Give feedback
        Noodlings.SendMessage(
            user,
            f"*CLUNK* The vending machine dispenses a {item}!"
        )

        # Easter egg after many uses
        if self.uses == 10:
            Debug.Log("Vending machine achievement unlocked!")
            Noodlings.SendMessage(
                user,
                "The vending machine displays: 'FREQUENT BUYER ACHIEVEMENT UNLOCKED!'"
            )

        # Machine breaks after 20 uses
        if self.uses >= 20:
            Debug.LogWarning("Vending machine is OUT OF ORDER")
            Noodlings.SendMessage(user, "The machine sparks and displays: OUT OF ORDER")
            self.enabled = False  # Disable script

    def OnClick(self, clicker):
        """Clicking shows price."""
        Noodlings.SendMessage(
            clicker,
            f"Vending Machine - {self.price} coins per item\n"
            f"Items available: {len(self.items)}\n"
            f"Uses remaining: {20 - self.uses}"
        )

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
