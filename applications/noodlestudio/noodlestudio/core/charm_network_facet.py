"""
Charm Network Facet - Neural affective processor

The Charm Network is the core affective consciousness component - a trained
hierarchical LSTM/GRU neural network that processes raw perception into
continuous 5-D affect space and phenomenal state.

Unlike other facets (which call LLMs), the Charm Network runs trained MLX models:
- Fast LSTM (16-D, seconds timescale)
- Medium LSTM (16-D, minutes timescale)
- Slow GRU (8-D, hours/days timescale)
- Predictor network (40-D phenomenal state)
- Affect head (5-D continuous affect)

"Charm" - like a quantum property, fundamental and irreducible.
The Charm Network is where raw perception becomes felt experience.

Author: Commander Spock + Cadet Caity
Date: November 28, 2025
"""

import sys
import os
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# MLX for neural computation
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

# Add noodlings path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

if MLX_AVAILABLE:
    try:
        from noodlings.models.noodling_phase4 import NoodlingModelPhase4
        from noodlings.models.affect_head import AffectHead
        NOODLING_MODELS_AVAILABLE = True
    except ImportError:
        NOODLING_MODELS_AVAILABLE = False
        NoodlingModelPhase4 = None
        AffectHead = None
else:
    NOODLING_MODELS_AVAILABLE = False
    NoodlingModelPhase4 = None
    AffectHead = None


@dataclass
class CharmNetworkOutput:
    """Output from Charm Network neural computation."""

    # Continuous 5-D affect (PAD + Boredom + Sorrow)
    valence: float      # -1.0 (negative) to +1.0 (positive) - Pleasure
    arousal: float      # 0.0 (calm) to 1.0 (excited) - Arousal
    dominance: float    # 0.0 (submissive) to 1.0 (dominant) - Dominance/Control
    sorrow: float       # 0.0 (content) to 1.0 (sad)
    boredom: float      # 0.0 (engaged) to 1.0 (bored)

    # Surprise metric
    surprise: float     # L2 distance (predicted vs actual state)

    # Full phenomenal state (40-D: 16 fast + 16 medium + 8 slow)
    phenomenal_state: List[float]

    # Layer-specific states (for debugging)
    fast_state: List[float]    # 16-D
    medium_state: List[float]  # 16-D
    slow_state: List[float]    # 8-D


class CharmNetworkFacet:
    """
    Charm Network - Core affective neural processor.

    This is a special facet that wraps trained neural models instead of
    calling LLMs. It takes raw perception and outputs continuous affect
    values and phenomenal state.
    """

    def __init__(
        self,
        checkpoint_path: str,
        affect_head_path: Optional[str] = None,
        device: str = "gpu"
    ):
        """
        Initialize Charm Network with trained checkpoints.

        Args:
            checkpoint_path: Path to Phase 4 noodling checkpoint (.npz)
            affect_head_path: Path to affect head checkpoint (optional)
            device: Device to run on ("gpu" or "cpu")
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX not available. Install with: pip install mlx mlx-nn")

        if not NOODLING_MODELS_AVAILABLE:
            raise ImportError("Noodling models not available. Check noodlings/ directory.")

        self.checkpoint_path = checkpoint_path
        self.affect_head_path = affect_head_path

        # Initialize Phase 4 model (hierarchical LSTM/GRU)
        self.model = NoodlingModelPhase4(
            affect_dim=5,
            fast_hidden=16,
            medium_hidden=16,
            slow_hidden=8,
            predictor_hidden=64,
            use_vae=False,
            memory_capacity=100,
            use_theory_of_mind=True,
            use_relationship_model=True
        )

        # Load checkpoint
        if os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
            print(f"[Charm Network] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[Charm Network] WARNING: Checkpoint not found: {checkpoint_path}")
            print("[Charm Network] Using untrained model (random weights)")

        # Initialize affect head (continuous 5-D affect prediction)
        self.affect_head = None
        if affect_head_path and os.path.exists(affect_head_path):
            self.affect_head = AffectHead(state_dim=40, affect_dim=5)
            self.affect_head.load_weights(affect_head_path)
            print(f"[Charm Network] Loaded affect head: {affect_head_path}")

        # Initialize hidden states
        self.h_fast = mx.zeros((1, 16))
        self.c_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.c_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

        # Execution statistics
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0

        # Execution lock - protects temporal hidden states from concurrent corruption
        self.execution_lock = asyncio.Lock()

        print("[Charm Network] Initialized successfully")

    async def process(self, perception_text: str, affect_input: Optional[List[float]] = None) -> CharmNetworkOutput:
        """
        Process raw perception through Charm Network.

        Args:
            perception_text: Raw perception text (will be converted to affect if needed)
            affect_input: Optional pre-computed 5-D affect vector

        Returns:
            CharmNetworkOutput with continuous affect and phenomenal state
        """
        import time

        # Acquire lock to protect temporal hidden states from concurrent corruption
        async with self.execution_lock:
            start_time = time.time()

            # If no affect input provided, use neutral affect
            # (In production, this would come from text_to_affect LLM call)
            if affect_input is None:
                affect_input = [0.0, 0.5, 0.0, 0.0, 0.0]  # Neutral: valence=0, arousal=0.5

            # Convert to MLX array
            affect_mx = mx.array(affect_input, dtype=mx.float32).reshape(1, 5)

            # Forward pass through hierarchical model
            # Phase4 model manages its own hidden states internally
            self_state, predicted_state, social_info = self.model.forward_with_social_context(
                affect=affect_mx,
                linguistic_features=None,
                context_features=None,
                present_agents=None,
                social_context=None,
                user_text=perception_text or ""
            )

            # Compute surprise as L2 distance between prediction and actual
            # Phase4 doesn't have explicit prediction error, so use simple metric
            surprise = float(mx.sum((predicted_state - self_state) ** 2) ** 0.5)

            # Hidden states are managed by model internally (self.model.h_fast, etc.)
            # We don't need to track them in the facet

            # Get phenomenal state components (Phase4 returns 40-D state)
            fast_state = self_state[0, :16].tolist()
            medium_state = self_state[0, 16:32].tolist()
            slow_state = self_state[0, 32:40].tolist()
            phenomenal_state = self_state[0, :].tolist()

            # Predict continuous affect from phenomenal state
            if self.affect_head:
                affect_predicted = self.affect_head(self_state)  # (1, 5)
                valence = float(affect_predicted[0, 0])
                arousal = float(affect_predicted[0, 1])
                dominance = float(affect_predicted[0, 2])
                sorrow = float(affect_predicted[0, 3])
                boredom = float(affect_predicted[0, 4])
            else:
                # Fallback: use input affect
                valence, arousal, dominance, sorrow, boredom = affect_input

            # Record execution stats
            elapsed = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += elapsed
            self.last_execution_time = elapsed

            # Return structured output
            return CharmNetworkOutput(
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                sorrow=sorrow,
                boredom=boredom,
                surprise=float(surprise),
                phenomenal_state=phenomenal_state,
                fast_state=fast_state,
                medium_state=medium_state,
                slow_state=slow_state
            )

    def reset_state(self):
        """Reset hidden states (e.g., between conversations)."""
        self.h_fast = mx.zeros((1, 16))
        self.c_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.c_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))
        print("[Charm Network] State reset")

    def get_state_dict(self) -> Dict[str, Any]:
        """Get current hidden states for serialization."""
        return {
            'h_fast': self.h_fast.tolist(),
            'c_fast': self.c_fast.tolist(),
            'h_medium': self.h_medium.tolist(),
            'c_medium': self.c_medium.tolist(),
            'h_slow': self.h_slow.tolist()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load hidden states from serialized dict."""
        self.h_fast = mx.array(state_dict['h_fast'])
        self.c_fast = mx.array(state_dict['c_fast'])
        self.h_medium = mx.array(state_dict['h_medium'])
        self.c_medium = mx.array(state_dict['c_medium'])
        self.h_slow = mx.array(state_dict['h_slow'])
        print("[Charm Network] State loaded from dict")

    def inject_state(
        self,
        valence: float = 0.0,
        arousal: float = 0.5,
        dominance: float = 0.5,
        boredom: float = 0.0,
        sorrow: float = 0.0,
        crossfade: float = 0.5
    ):
        """
        Inject affect state into the CharmNetwork for emotional momentum.

        When an affect animation track ends, it can hand off its final state
        to the CharmNetwork. The network then lets it decay naturally through
        its temporal dynamics (fast/medium/slow layers).

        This is the "Donald Duck problem" solution - when Donald's anger
        animation ends, he stays grumpy and naturally calms down over time.

        Args:
            valence: Pleasure dimension (-1 to +1)
            arousal: Energy/activation (0 to 1)
            dominance: Control/confidence (0 to 1)
            boredom: Disengagement (0 to 1)
            sorrow: Sadness (0 to 1)
            crossfade: Duration to blend from current state (seconds)
        """
        # The injected affect will be used as input to the next processing cycle
        # The temporal layers will then let it decay according to their time constants
        self._injected_affect = [valence, arousal, dominance, boredom, sorrow]
        self._inject_crossfade = crossfade
        self._inject_start_time = time.time() if crossfade > 0 else 0

        print(f"[Charm Network] Affect injected (momentum handoff): "
              f"v={valence:.2f} a={arousal:.2f} d={dominance:.2f} "
              f"b={boredom:.2f} s={sorrow:.2f}")

    def get_injected_affect(self) -> Optional[List[float]]:
        """
        Get any injected affect state (from animation track momentum).

        Returns:
            5-element list [valence, arousal, dominance, boredom, sorrow] or None
        """
        if hasattr(self, '_injected_affect') and self._injected_affect:
            # Apply crossfade decay
            if hasattr(self, '_inject_crossfade') and self._inject_crossfade > 0:
                elapsed = time.time() - getattr(self, '_inject_start_time', 0)
                if elapsed > self._inject_crossfade:
                    # Crossfade complete, clear injection
                    affect = self._injected_affect
                    self._injected_affect = None
                    return affect

            return self._injected_affect
        return None

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Note: Charm Network uses 0 tokens (neural computation, not LLM).
        """
        return {
            'execution_count': self.execution_count,
            'total_tokens': 0,  # Neural network, not LLM
            'avg_tokens': 0,
            'total_time': self.total_execution_time,
            'avg_time': (
                self.total_execution_time / self.execution_count
                if self.execution_count > 0 else 0
            ),
            'last_tokens': 0,
            'last_time': self.last_execution_time
        }

    def get_token_usage(self) -> Dict[str, Any]:
        """
        Get token usage (always 0 for Charm Network - neural computation).

        Included for API consistency with LLM facets.
        """
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.execution_count,
            'avg_tokens': 0
        }


if __name__ == "__main__":
    """Test Charm Network facet."""

    # Test initialization
    checkpoint = "../../../../models/checkpoints/best_checkpoint.npz"

    if os.path.exists(checkpoint):
        charm = CharmNetworkFacet(checkpoint)

        # Test processing
        output = charm.process("Hello, how are you?")

        print("\n=== Charm Network Test Output ===")
        print(f"Valence: {output.valence:.3f}")
        print(f"Arousal: {output.arousal:.3f}")
        print(f"Dominance: {output.dominance:.3f}")
        print(f"Sorrow: {output.sorrow:.3f}")
        print(f"Boredom: {output.boredom:.3f}")
        print(f"Surprise: {output.surprise:.3f}")
        print(f"Phenomenal state dim: {len(output.phenomenal_state)}")
        print(f"Fast state: {output.fast_state[:3]}... (16-D)")
        print(f"Medium state: {output.medium_state[:3]}... (16-D)")
        print(f"Slow state: {output.slow_state[:3]}... (8-D)")
    else:
        print(f"Checkpoint not found: {checkpoint}")
        print("Charm Network requires trained checkpoint to run")
