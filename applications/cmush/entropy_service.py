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
#   Entropy Service
#
#   The randomness source for all AI decisions in noodleMUSH.
#   When a TrueRNG V3 hardware device is connected, it provides
#   genuine quantum-derived random numbers. Otherwise, it falls
#   back to Python's built-in random generator. All uncertainty
#   in agent behavior flows through this single service. Also
#   includes an "avalanche RNG" that mimics quantum collapse
#   effects with rare extreme values.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.entropy_service
# PURPOSE:  Unified random number generation (hardware or PRNG)
# LAYER:    Backend / Core Infrastructure
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   EntropyService   Global singleton for all randomness
#   EntropyPool      Thread-safe buffer for hardware RNG
#   AvalancheRNG     Heavy-tailed distribution generator
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Entropy Service - Hardware True Random Number Generation

Provides a unified interface for all randomness in noodleMUSH.
Routes to either TrueRNG V3 hardware or Python's PRNG.

Philosophy: Every decision point gets genuine quantum entropy.

Author: Caitlyn + Claude
"""

import random as stdlib_random
import numpy as np
from typing import Optional, List
import serial
import struct
import threading
from queue import Queue
import logging

logger = logging.getLogger(__name__)


class AvalancheRNG:
    """
    Avalanche effect random number generator with heavy-tailed distribution.

    Mimics electron avalanche breakdown in quantum devices.
    Uses power-law distribution to create rare large spikes (quantum collapse events).

    Based on recent microtubule consciousness research (2025) suggesting
    quantum effects in biological systems follow avalanche statistics.
    """

    def __init__(self, entropy_service: 'EntropyService', beta: float = 2.0):
        """
        Initialize avalanche RNG.

        Args:
            entropy_service: Underlying entropy source (hardware or PRNG)
            beta: Power law exponent (2.0 = heavy tails, 3.0 = lighter tails)
        """
        self.entropy_service = entropy_service
        self.beta = beta

    def generate(self, shape: tuple = (1,)) -> np.ndarray:
        """
        Generate avalanche-distributed random numbers.

        Returns array in range [-1, 1] with heavy tails.
        Most values near 0, rare extreme events near ±1.

        Args:
            shape: Output shape (e.g., (1,) or (16,))

        Returns:
            numpy array with avalanche distribution
        """
        # Generate uniform random values
        size = np.prod(shape) if isinstance(shape, tuple) else shape
        uniform_vals = np.array([self.entropy_service.random() for _ in range(size)])

        # Power law transform: creates heavy tails
        # -log(u) is exponential, raising to 1/beta creates power law
        avalanche = np.power(-np.log(uniform_vals + 1e-10), 1.0 / self.beta)

        # Normalize to [-1, 1] with tanh
        normalized = np.tanh(avalanche - 1.0)

        return normalized.reshape(shape) if isinstance(shape, tuple) else normalized

    def generate_positive(self, shape: tuple = (1,)) -> np.ndarray:
        """
        Generate positive-only avalanche values in [0, 1].

        Useful for probability thresholds and magnitudes.
        """
        vals = self.generate(shape)
        return (vals + 1.0) / 2.0  # Map [-1,1] to [0,1]


class EntropyPool:
    """
    Thread-safe entropy pool that prefetches from TrueRNG.

    Avoids USB latency by maintaining a buffer of random bytes.
    Background thread continuously refills the pool.
    """

    def __init__(self, device_path: str, pool_size: int = 4096):
        self.device_path = device_path
        self.pool = Queue(maxsize=pool_size)
        self.device = None
        self.running = False
        self.thread = None

    def start(self):
        """Start entropy collection thread."""
        try:
            self.device = serial.Serial(self.device_path, timeout=1)
            self.running = True
            self.thread = threading.Thread(target=self._fill_pool, daemon=True)
            self.thread.start()
            logger.info(f"TrueRNG entropy pool started: {self.device_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to start TrueRNG: {e}")
            return False

    def stop(self):
        """Stop entropy collection."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.device:
            self.device.close()

    def _fill_pool(self):
        """Background thread: continuously fill entropy pool."""
        while self.running:
            try:
                if not self.pool.full():
                    # Read 32 bytes at a time
                    entropy_bytes = self.device.read(32)
                    if len(entropy_bytes) == 32:
                        self.pool.put(entropy_bytes)
            except Exception as e:
                logger.error(f"Entropy pool error: {e}")
                break

    def get_bytes(self, n: int) -> Optional[bytes]:
        """Get n random bytes from pool (non-blocking)."""
        try:
            result = b''
            while len(result) < n:
                chunk = self.pool.get(timeout=0.1)
                result += chunk[:n - len(result)]
            return result
        except:
            return None


class EntropyService:
    """
    Global entropy service for noodleMUSH.

    Usage:
        entropy = get_entropy_service()
        value = entropy.uniform(0.0, 1.0)
        choice = entropy.choice(['a', 'b', 'c'])
    """

    def __init__(self, use_hardware: bool = False, device_path: Optional[str] = None):
        self.use_hardware = use_hardware
        self.device_path = device_path
        self.pool = None

        if use_hardware and device_path:
            self.pool = EntropyPool(device_path)
            if not self.pool.start():
                logger.warning("Falling back to PRNG")
                self.use_hardware = False
                self.pool = None

    def _get_random_bytes(self, n: int) -> bytes:
        """Get n random bytes from hardware or PRNG."""
        if self.use_hardware and self.pool:
            entropy_bytes = self.pool.get_bytes(n)
            if entropy_bytes:
                return entropy_bytes
            # Fallback if pool fails
            logger.warning("Entropy pool empty, falling back to PRNG")

        # PRNG fallback
        return bytes([stdlib_random.randint(0, 255) for _ in range(n)])

    def uniform(self, a: float, b: float) -> float:
        """Generate uniform random float in [a, b)."""
        # Get 4 bytes, interpret as uint32, normalize to [0, 1)
        rand_bytes = self._get_random_bytes(4)
        rand_uint = struct.unpack('I', rand_bytes)[0]
        normalized = rand_uint / (2**32)
        return a + (b - a) * normalized

    def randint(self, a: int, b: int) -> int:
        """Generate random integer in [a, b]."""
        range_size = b - a + 1
        rand_bytes = self._get_random_bytes(4)
        rand_uint = struct.unpack('I', rand_bytes)[0]
        return a + (rand_uint % range_size)

    def choice(self, seq: List) -> any:
        """Choose random element from sequence."""
        if not seq:
            raise IndexError("Cannot choose from empty sequence")
        idx = self.randint(0, len(seq) - 1)
        return seq[idx]

    def random(self) -> float:
        """Generate random float in [0, 1)."""
        return self.uniform(0.0, 1.0)

    def expovariate(self, lambd: float) -> float:
        """Generate exponentially distributed random variable."""
        u = self.random()
        return -np.log(1 - u) / lambd

    def shuffle(self, seq: List) -> None:
        """Shuffle sequence in place (Fisher-Yates algorithm)."""
        for i in range(len(seq) - 1, 0, -1):
            j = self.randint(0, i)
            seq[i], seq[j] = seq[j], seq[i]

    def create_avalanche_rng(self, beta: float = 2.0) -> AvalancheRNG:
        """
        Create an avalanche RNG using this entropy service.

        Args:
            beta: Power law exponent (2.0 = heavy tails)

        Returns:
            AvalancheRNG instance
        """
        return AvalancheRNG(entropy_service=self, beta=beta)

    def get_config(self) -> dict:
        """Return current configuration."""
        return {
            'use_hardware': self.use_hardware,
            'device_path': self.device_path,
            'active': self.pool is not None
        }

    def shutdown(self):
        """Clean shutdown of entropy service."""
        if self.pool:
            self.pool.stop()


# Global singleton instance
_entropy_service: Optional[EntropyService] = None


def initialize_entropy_service(use_hardware: bool = False, device_path: Optional[str] = None):
    """Initialize global entropy service."""
    global _entropy_service
    _entropy_service = EntropyService(use_hardware, device_path)
    logger.info(f"Entropy service initialized: hardware={use_hardware}, device={device_path}")


def get_entropy_service() -> EntropyService:
    """Get global entropy service instance."""
    global _entropy_service
    if _entropy_service is None:
        # Auto-initialize with PRNG fallback
        initialize_entropy_service(use_hardware=False)
    return _entropy_service

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
