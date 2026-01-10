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
#   Schrodinger's Box - Quantum Cat Experiment
#
#   A fun demonstration of quantum superposition using Noodlings.
#   The box contains a cat in superposition - both alive AND ghost
#   until observed. Press the COLLAPSE button and the wavefunction
#   collapses: 50% chance you get a happy live cat, 50% chance you
#   get an adorable ghost cat. Both are delightful outcomes. Real
#   quantum physics, but nobody gets hurt (not even virtually).
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.example_scripts.SchrodingersBox
# PURPOSE:  Quantum superposition demonstration with Noodlings
# LAYER:    Backend / Example Scripts
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SchrodingersBox   Quantum measurement -> Noodling spawner
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Schrodinger's Box - Quantum Cat Experiment

A mysterious box containing a cat in quantum superposition.
Press the big red COLLAPSE button to observe the outcome!

The result is determined by actual quantum mechanics simulation:
- |0> state = Cat is ALIVE (cartoon cat emerges)
- |1> state = Cat is GHOST (adorable ghost-cat emerges)

Until observed, the cat exists in BOTH states simultaneously.
This is real quantum physics made tangible in a virtual world.
"""

import random
import time
from noodlings_scripting import NoodleScript, Noodlings, Debug


class SchrodingersBox(NoodleScript):
    """
    The Schrodinger's Box - a quantum superposition demonstration.

    Contains a cat in superposition until observed.
    """

    def __init__(self):
        super().__init__()
        self.has_been_observed = False
        self.observation_count = 0
        self.last_outcome = None  # 'alive' or 'ghost'
        self.cooldown = 0

    def OnClick(self, clicker: str):
        """
        Handle the COLLAPSE button press.

        Args:
            clicker: Entity ID who clicked
        """
        Debug.Log(f"[Schrodinger] Box clicked by {clicker}")

        # Check cooldown
        if self.cooldown > 0:
            Noodlings.Broadcast(
                self.Room,
                f"[The box hums quietly] Preparing new quantum state... {self.cooldown}s"
            )
            return

        # COLLAPSE THE WAVEFUNCTION!
        self._perform_quantum_measurement(clicker)

    def OnUse(self, user: str):
        """Alternative interaction - same as click."""
        self.OnClick(user)

    def OnTick(self):
        """Update cooldown timer."""
        if self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - 5)

    def _perform_quantum_measurement(self, observer: str):
        """
        Collapse the quantum superposition and spawn the cat.

        Uses simulated quantum mechanics:
        - Qubit starts in superposition: |psi> = (|0> + |1>) / sqrt(2)
        - Measurement collapses to |0> (alive) or |1> (ghost) with 50/50 probability
        - True quantum randomness simulated with high-entropy seeding

        Args:
            observer: The entity causing the collapse
        """
        Debug.Log("[Schrodinger] Performing quantum measurement...")

        # Dramatic buildup
        Noodlings.Broadcast(
            self.Room,
            "\n"
            "  +================================+\n"
            "  |   SCHRODINGER'S BOX           |\n"
            "  |   [COLLAPSE] button pressed   |\n"
            "  +================================+\n"
            "\n"
            "  The box begins to vibrate...\n"
            "  Quantum superposition destabilizing...\n"
            "  The wavefunction is collapsing...\n"
        )

        # QUANTUM MEASUREMENT
        # Use high-entropy seed for "true" randomness
        random.seed(int(time.time_ns()) ^ id(self) ^ hash(observer))

        # 50/50 collapse - this is the quantum moment!
        measurement = random.random()
        is_alive = measurement < 0.5

        self.observation_count += 1
        self.last_outcome = 'alive' if is_alive else 'ghost'

        Debug.Log(f"[Schrodinger] Measurement result: {self.last_outcome} (value: {measurement:.4f})")

        if is_alive:
            self._spawn_alive_cat(observer, measurement)
        else:
            self._spawn_ghost_cat(observer, measurement)

        # Set cooldown for next observation
        self.cooldown = 60  # 60 seconds before next cat

    def _spawn_alive_cat(self, observer: str, measurement: float):
        """Spawn the alive cartoon cat."""
        Debug.Log("[Schrodinger] Spawning ALIVE cat!")

        # Rez the alive cat noodling
        noodling_id = Noodlings.Rez(
            recipe="schrodinger_alive_cat",
            room=self.Room
        )

        Noodlings.Broadcast(
            self.Room,
            "\n"
            "  *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*\n"
            "  *                                       *\n"
            "  *   MEASUREMENT RESULT: |0>             *\n"
            "  *                                       *\n"
            "  *   THE CAT IS ALIVE!                   *\n"
            "  *                                       *\n"
            "  *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*\n"
            "\n"
            "  The box door swings open!\n"
            "  A joyful cartoon cat bounds out, meowing excitedly!\n"
            "\n"
            f"  Quantum value: {measurement:.6f}\n"
            f"  Observation #: {self.observation_count}\n"
            f"  Collapsed to: |0> (ALIVE)\n"
            "\n"
            "  The cat stretches and looks around happily.\n"
            "  It seems thrilled to have beaten the quantum odds!\n"
        )

        if noodling_id:
            # Send awareness to the cat
            Noodlings.SendMessage(
                noodling_id,
                f"You have just emerged from Schrodinger's Box! The quantum measurement "
                f"collapsed to |0> and you are ALIVE! {observer} observed you into existence. "
                f"You're so happy to be here! This is observation #{self.observation_count}."
            )

    def _spawn_ghost_cat(self, observer: str, measurement: float):
        """Spawn the adorable ghost cat."""
        Debug.Log("[Schrodinger] Spawning GHOST cat!")

        # Rez the ghost cat noodling
        noodling_id = Noodlings.Rez(
            recipe="schrodinger_ghost_cat",
            room=self.Room
        )

        Noodlings.Broadcast(
            self.Room,
            "\n"
            "  ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~\n"
            "  ~                                       ~\n"
            "  ~   MEASUREMENT RESULT: |1>             ~\n"
            "  ~                                       ~\n"
            "  ~   THE CAT IS... A GHOST!              ~\n"
            "  ~                                       ~\n"
            "  ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~\n"
            "\n"
            "  The box door creaks open slowly...\n"
            "  A softly glowing, translucent cat floats out!\n"
            "\n"
            f"  Quantum value: {measurement:.6f}\n"
            f"  Observation #: {self.observation_count}\n"
            f"  Collapsed to: |1> (GHOST)\n"
            "\n"
            "  The ghost cat giggles ethereally.\n"
            "  It seems perfectly content with this outcome!\n"
            "  'Being a ghost is actually pretty fun!' it seems to say.\n"
        )

        if noodling_id:
            # Send awareness to the ghost cat
            Noodlings.SendMessage(
                noodling_id,
                f"You have just emerged from Schrodinger's Box! The quantum measurement "
                f"collapsed to |1> and you are a GHOST! {observer} observed you into "
                f"existence. You're cheerfully spooky! This is observation #{self.observation_count}. "
                f"Boo! Hehe!"
            )


# Recipe files needed in recipes/ directory:
# - schrodinger_alive_cat.yaml (uses schrodinger_alive_cat facet assembly)
# - schrodinger_ghost_cat.yaml (uses schrodinger_ghost_cat facet assembly)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
