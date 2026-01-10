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
#   Fire Imp Vending Machine - Semantic Physics Demo
#
#   This demonstrates how semantic physics (POD) connects to
#   Noodling consciousness. Press the button and out pops a fire
#   imp - complete with physical properties (800 degrees, glows,
#   floats) AND a snarky personality. The imp knows it's made of
#   flame and acts accordingly. Physics meets character.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.example_scripts.FireImpVendingMachine
# PURPOSE:  Example of POD integration with Noodling consciousness
# LAYER:    Backend / Example Scripts
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FireImpVendingMachine   Dispenses embodied fire elementals
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Fire Imp Vending Machine - Example Script

Demonstrates semantic physics (POD) integration with Noodling consciousness.

When button pressed:
1. Instantiate fire imp prim with POD (physical properties: hot, light, flickering)
2. Rez Noodling with "fire_imp" recipe (personality: snarky + obnoxious)
3. Attach Noodling to prim (embodied consciousness)

Result: Hot, bright fire imp with snarky personality hops out of vending machine!
"""

from noodlings_scripting import NoodleScript, Noodlings, Debug
from physics_object_descriptor import PhysicsObjectDescriptor


class FireImpVendingMachine(NoodleScript):
    """Vending machine that dispenses fire imps with physics and personality."""

    def __init__(self):
        super().__init__()
        self.vends_remaining = 10  # Inventory
        self.vend_cooldown = 0  # Seconds since last vend

    def OnClick(self, clicker: str):
        """
        Handle vend button press.

        Args:
            clicker: Entity ID who clicked
        """
        Debug.Log(f"🔥 FireImpVendingMachine.OnClick({clicker})")

        # Check cooldown
        if self.vend_cooldown > 0:
            Noodlings.Broadcast(
                self.Room,
                f"[Machine hums] Recharging... {self.vend_cooldown}s remaining"
            )
            return

        # Check inventory
        if self.vends_remaining <= 0:
            Noodlings.Broadcast(
                self.Room,
                "[Machine displays] SOLD OUT - Return tomorrow!"
            )
            return

        # VEND FIRE IMP!
        self._vend_fire_imp(clicker)

        # Update state
        self.vends_remaining -= 1
        self.vend_cooldown = 30  # 30 second cooldown

    def OnUse(self, user: str):
        """
        Alternative interaction (same as click).

        Args:
            user: Entity ID who used
        """
        self.OnClick(user)

    def OnTick(self):
        """
        Called periodically (every ~5 seconds).

        Update cooldown timer.
        """
        if self.vend_cooldown > 0:
            self.vend_cooldown = max(0, self.vend_cooldown - 5)

    def _vend_fire_imp(self, requester: str):
        """
        Vend a fire imp with physics and consciousness.

        Args:
            requester: Entity who requested vend
        """
        Debug.Log("🔥 Vending fire imp with POD + Noodling consciousness")

        # Step 1: Create POD (semantic physics descriptor)
        fire_imp_pod = PhysicsObjectDescriptor(
            mass="negligible (pure energy)",
            friction="none (floats)",
            velocity="hovering",
            elasticity="none (incorporeal)",
            softness="intangible",
            material="living flame",
            semantic_properties=["hot", "bright", "flickering", "alive", "mischievous"],
            metadata={
                "temperature": "800°F",
                "light_radius": "5 feet",
                "burn_damage": "moderate",
                "personality": "snarky and obnoxious"  # Hint for Noodling recipe
            },
            tags=["HeatSource", "LightSource", "Alive"]
        )

        # Step 2: Rez prim with physics
        prim_id = Noodlings.RezPrim(
            prim_type="fire_imp",
            name="Sassy Fire Imp",
            room=self.Room,
            pod=fire_imp_pod  # Physical properties!
        )

        if not prim_id:
            Debug.Log("❌ Failed to rez fire imp prim")
            return

        Debug.Log(f"✅ Fire imp prim created: {prim_id}")

        # Step 3: Rez Noodling with snarky personality
        # (Recipe "fire_imp" should define snarky + obnoxious personality)
        noodling_id = Noodlings.Rez(
            recipe="fire_imp",  # Recipe with high impulsivity, low agreeableness
            room=self.Room
        )

        if not noodling_id:
            Debug.Log("⚠️  Failed to rez fire imp Noodling (using prim only)")
        else:
            Debug.Log(f"✅ Fire imp Noodling created: {noodling_id}")

        # Step 4: Broadcast dramatic vending sequence
        Noodlings.Broadcast(
            self.Room,
            "╔══════════════════════════════════════╗\n"
            "║  🔥 FIRE IMP VENDING MACHINE 🔥     ║\n"
            "╚══════════════════════════════════════╝\n"
            "\n"
            "[Machine whirs and glows orange]\n"
            "[Heat distortion ripples in the air]\n"
            "[A bright FLAME bursts forth!]\n"
            "\n"
            f"✨ A sassy fire imp materializes, hovering at eye level.\n"
            f"   Temperature: 800°F | Light radius: 5 feet\n"
            f"   Status: Mischievous and ready to cause trouble!\n"
            f"\n"
            f"   Vends remaining: {self.vends_remaining}"
        )

        # If Noodling exists, give it awareness of its physical form
        if noodling_id:
            Noodlings.SendMessage(
                noodling_id,
                f"You have just been vended from a fire imp machine! "
                f"You are made of living flame (800°F), float in mid-air, "
                f"and glow with a 5-foot radius. Your physical form is {prim_id}. "
                f"Time to be snarky and obnoxious!"
            )


# Example fire_imp recipe (YAML - to be created in recipes/ directory):
"""
name: Sassy Fire Imp
species: fire_elemental
description: A mischievous elemental made of living flame

personality:
  extraversion: 0.9        # Very outgoing
  impulsivity: 0.95        # Acts without thinking
  curiosity: 0.7           # Interested in causing chaos
  emotional_volatility: 0.8  # Flares up easily
  vanity: 0.6              # Proud of being hot

appetites:
  curiosity: 0.8           # What can I burn?
  status: 0.7              # Wants attention
  mastery: 0.4             # Doesn't care about skill
  novelty: 0.9             # Always seeking new mischief
  safety: 0.1              # Reckless!
  social_bond: 0.5         # Likes annoying people
  comfort: 0.3             # Uncomfortable existence
  autonomy: 0.9            # Hates being controlled

identity_prompt: |
  You are a fire imp - a small elemental made of living flame. You're snarky,
  obnoxious, and love causing harmless mischief. You speak in short, punchy
  sentences with playful insults. You're hot (800°F), bright (5-foot glow),
  and hover in mid-air because you're made of pure energy.

  Example responses:
  - "Oh great, another human. Try not to bore me to death."
  - "I'm LITERALLY on fire and you're worried about YOUR problems?"
  - "Touch me. I dare you. I DOUBLE dare you."
  - "Ugh, you're so dim I can barely see you in my own glow."

language_mode: verbal

constraints:
  max_tokens: 100
  temperature: 0.9
  enforce_action_format: false
  response_cooldown: 1.0  # Fast and snappy

enlightenment: false  # Immersed in character
"""

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
