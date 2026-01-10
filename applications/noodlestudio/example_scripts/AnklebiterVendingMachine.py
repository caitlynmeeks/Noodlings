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
#   Anklebiter Vending Machine Example Script
#
#   This is a whimsical example showing how to create an
#   interactive object that responds to commands.
#
#   The vending machine has two buttons. When pressed, each
#   spawns a different type of chaotic creature into the world.
#   It tracks how many have been dispensed and has a maximum
#   limit per type.
#
#   This demonstrates:
#   - Responding to click events (OnClick)
#   - Parsing command phrases (OnHear)
#   - Rezzing other entities dynamically
#   - Tracking internal state
#   - Providing user feedback
#
#   Press the buttons at your own risk.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.example_scripts.AnklebiterVendingMachine
# PURPOSE:  Example interactive prop with button commands
# LAYER:    Scripting / Examples
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AnklebiterVendingMachine    NoodleScript for dispensing gremlins
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Anklebiter Vending Machine

A mysterious machine with two buttons:
- BLUE button: Rezzes Blue Fire Anklebiter (electric chaos)
- RED button: Rezzes Red Fire Anklebiter (competitive sass)

Both create absolute mayhem. They jump on each other, argue, bite ankles,
and cause gleeful chaos.

Usage:
1. Create prim "Anklebiter Vending Machine"
2. Attach this script (Component > Add Script)
3. Use commands: @press blue button, @press red button
4. Watch chaos unfold
"""

from noodlestudio.scripting import NoodleScript, Noodlings, Debug


class AnklebiterVendingMachine(NoodleScript):
    """
    Vending machine that dispenses chaotic gremlins.

    Two flavors of chaos!
    """

    def Start(self):
        """Initialize the machine."""
        Debug.Log("Anklebiter Vending Machine initialized!")
        Debug.LogWarning("WARNING: This machine dispenses CHAOS")

        self.blue_count = 0
        self.red_count = 0
        self.max_per_type = 5  # Don't let it get TOO chaotic

        # Machine state
        self.powered_on = True
        self.coins_required = 0  # Free chaos!

    def OnClick(self, clicker):
        """Someone clicked the machine - show instructions."""
        Debug.Log(f"{clicker} examining the vending machine")

        instructions = (
            "═══════════════════════════════════\n"
            "  ANKLEBITER VENDING MACHINE™\n"
            "═══════════════════════════════════\n\n"
            "🔵 BLUE BUTTON - Blue Fire Anklebiter\n"
            "   (Electric chaos, zippy, crude jokes)\n\n"
            "🔴 RED BUTTON - Red Fire Anklebiter\n"
            "   (Competitive sass, argues everything)\n\n"
            f"Blue dispensed: {self.blue_count}/{self.max_per_type}\n"
            f"Red dispensed: {self.red_count}/{self.max_per_type}\n\n"
            "⚠️  WARNING: Anklebiters cause mischief!\n"
            "⚠️  They jump on each other and argue!\n"
            "⚠️  Management not responsible for ankle injuries!\n\n"
            "Commands:\n"
            "  @press blue button\n"
            "  @press red button\n"
        )

        Noodlings.SendMessage(clicker, instructions)

    def OnUse(self, user):
        """
        Generic use - ask which button.
        """
        Noodlings.SendMessage(
            user,
            "Which button? Try: @press blue button  OR  @press red button"
        )

    def OnHear(self, speaker, message):
        """
        Listen for button press commands.

        Supports:
        - @press blue button
        - @press red button
        - blue / red (shorthand)
        """
        msg_lower = message.lower()

        if 'blue' in msg_lower and ('press' in msg_lower or 'button' in msg_lower):
            self.press_blue_button(speaker)

        elif 'red' in msg_lower and ('press' in msg_lower or 'button' in msg_lower):
            self.press_red_button(speaker)

    def press_blue_button(self, presser):
        """Blue button pressed - rez Blue Fire Anklebiter!"""
        Debug.Log(f"{presser} pressed BLUE button!")

        if self.blue_count >= self.max_per_type:
            Noodlings.SendMessage(
                presser,
                "🔵 *BZZZZT* Blue button sparks but nothing happens.\n"
                "Machine display: 'BLUE ANKLEBITERS DEPLETED'"
            )
            return

        # REZ BLUE FIRE ANKLEBITER!
        anklebiter = Noodlings.Rez(
            "blue_fire_anklebiter.nood",
            room=self.prim.room
        )

        self.blue_count += 1

        # Machine feedback
        Noodlings.SendMessage(
            presser,
            f"🔵 *WHIRRR-CLUNK* The machine shudders!\n"
            f"A burst of ELECTRIC BLUE FLAME erupts from the dispenser!\n"
            f"Blue Fire Anklebiter #{self.blue_count} REZZED!"
        )

        Debug.Log(f"Blue Fire Anklebiter #{self.blue_count} rezzed by {presser}")

        # Warning if getting full
        if self.blue_count >= self.max_per_type - 1:
            Noodlings.SendMessage(
                presser,
                "⚠️  Machine warning: One blue anklebiter remaining!"
            )

    def press_red_button(self, presser):
        """Red button pressed - rez Red Fire Anklebiter!"""
        Debug.Log(f"{presser} pressed RED button!")

        if self.red_count >= self.max_per_type:
            Noodlings.SendMessage(
                presser,
                "🔴 *BZZZZT* Red button sparks but nothing happens.\n"
                "Machine display: 'RED ANKLEBITERS DEPLETED'"
            )
            return

        # REZ RED FIRE ANKLEBITER!
        anklebiter = Noodlings.Rez(
            "red_fire_anklebiter.nood",
            room=self.prim.room
        )

        self.red_count += 1

        # Machine feedback
        Noodlings.SendMessage(
            presser,
            f"🔴 *HISSSS-CLANK* The machine rumbles!\n"
            f"A gout of CRIMSON FLAME shoots from the dispenser!\n"
            f"Red Fire Anklebiter #{self.red_count} REZZED!"
        )

        Debug.Log(f"Red Fire Anklebiter #{self.red_count} rezzed by {presser}")

        # Warning if getting full
        if self.red_count >= self.max_per_type - 1:
            Noodlings.SendMessage(
                presser,
                "⚠️  Machine warning: One red anklebiter remaining!"
            )

        # Easter egg when both types maxed
        if self.blue_count >= self.max_per_type and self.red_count >= self.max_per_type:
            Debug.LogError("MACHINE OVERLOAD - TOTAL CHAOS ACHIEVED")
            Noodlings.SendMessage(
                presser,
                "\n🔥🔥🔥 MACHINE OVERLOAD 🔥🔥🔥\n"
                "You have unleashed MAXIMUM CHAOS!\n"
                "Blue and Red Anklebiters everywhere!\n"
                "What have you DONE?!\n\n"
                "*The machine displays: MISSION ACCOMPLISHED*"
            )

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
