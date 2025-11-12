"""
Episodic Memory Buffer for Consilience Phase 3

Maintains a sliding window of recent conscious moments with circular buffer storage.
Each memory entry stores:
- Phenomenal state (40-D hierarchical representation)
- Affect vector (5-D emotional state)
- Key vector (64-D learned representation for retrieval)
- Metadata (timestamp, surprise, user text, attention weights)

Author: Consilience Project
Date: October 2025
"""

from collections import deque
from typing import Dict, List, Tuple, Optional
import mlx.core as mx
import numpy as np
from datetime import datetime


class EpisodicMemory:
    """
    Circular buffer storing recent phenomenal states for attention-based retrieval.

    Design Philosophy:
    - Working memory capacity ~100 moments (Baddeley's phonological loop analogy)
    - Efficient key-value caching for batch attention operations
    - Tracks which memories are "important" via cumulative attention weights
    """

    def __init__(self, capacity: int = 100, state_dim: int = 40, key_dim: int = 64):
        """
        Initialize episodic memory buffer.

        Args:
            capacity: Maximum number of moments to store (default 100)
            state_dim: Dimensionality of phenomenal state (default 40)
            key_dim: Dimensionality of retrieval keys (default 64)
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.key_dim = key_dim

        # Circular buffer using deque (efficient push/pop)
        self.buffer = deque(maxlen=capacity)

        # Pre-allocated arrays for batch operations
        self.keys_cache: Optional[mx.array] = None
        self.values_cache: Optional[mx.array] = None
        self.cache_valid = False

        # Statistics
        self.total_moments_added = 0

    def add(self, moment: Dict) -> None:
        """
        Add new moment to episodic memory.

        Args:
            moment: Dictionary containing:
                - step (int): Timestep number
                - timestamp (datetime): When this moment occurred
                - phenomenal_state (mx.array): [40-D] hierarchical state
                - affect (mx.array): [5-D] emotional state
                - key_vector (mx.array): [64-D] retrieval key
                - user_text (str): User input text
                - surprise (float): Prediction error magnitude
                - attention_weights (mx.array or None): [N] how this moment attended to past
        """
        # Validate structure
        required_fields = ['step', 'phenomenal_state', 'affect', 'key_vector', 'user_text']
        for field in required_fields:
            if field not in moment:
                raise ValueError(f"Memory moment missing required field: {field}")

        # Validate dimensions
        if moment['phenomenal_state'].shape[-1] != self.state_dim:
            raise ValueError(f"Phenomenal state has wrong dimension: "
                           f"{moment['phenomenal_state'].shape[-1]} != {self.state_dim}")

        if moment['key_vector'].shape[-1] != self.key_dim:
            raise ValueError(f"Key vector has wrong dimension: "
                           f"{moment['key_vector'].shape[-1]} != {self.key_dim}")

        # Add to buffer (automatically evicts oldest if at capacity)
        self.buffer.append(moment)
        self.total_moments_added += 1

        # Invalidate cache (needs rebuild)
        self.cache_valid = False

    def get_keys(self) -> mx.array:
        """
        Return all key vectors as a batch matrix.

        Returns:
            mx.array: [N, 64] where N = current buffer size
        """
        if len(self.buffer) == 0:
            return mx.zeros((0, self.key_dim))

        if not self.cache_valid or self.keys_cache is None:
            # Rebuild cache
            keys_list = [m['key_vector'] for m in self.buffer]
            self.keys_cache = mx.stack(keys_list, axis=0)

        return self.keys_cache

    def get_values(self) -> mx.array:
        """
        Return all phenomenal states as a batch matrix.

        Returns:
            mx.array: [N, 40] where N = current buffer size
        """
        if len(self.buffer) == 0:
            return mx.zeros((0, self.state_dim))

        if not self.cache_valid or self.values_cache is None:
            # Rebuild cache
            values_list = [m['phenomenal_state'] for m in self.buffer]
            self.values_cache = mx.stack(values_list, axis=0)
            self.cache_valid = True  # Mark as valid after building both caches

        return self.values_cache

    def get_affects(self) -> mx.array:
        """
        Return all affect vectors as a batch matrix.

        Returns:
            mx.array: [N, 5] affect vectors
        """
        if len(self.buffer) == 0:
            return mx.zeros((0, 5))

        affects_list = [m['affect'] for m in self.buffer]
        return mx.stack(affects_list, axis=0)

    def get_texts(self) -> List[str]:
        """
        Return user text for all moments (for interpretation/debugging).

        Returns:
            List[str]: User inputs corresponding to each moment
        """
        return [m['user_text'] for m in self.buffer]

    def get_surprises(self) -> List[float]:
        """
        Return surprise values for all moments.

        Returns:
            List[float]: Surprise (prediction error) for each moment
        """
        return [m.get('surprise', 0.0) for m in self.buffer]

    def get_most_attended(self, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Find moments that received most attention across time.

        These are "anchor memories" - formative moments in the relationship.

        Args:
            top_k: Number of top moments to return

        Returns:
            List of (step, importance_score, user_text) tuples
        """
        if len(self.buffer) == 0:
            return []

        # Accumulate attention weights received by each moment
        importance = np.zeros(len(self.buffer))

        for i, moment in enumerate(self.buffer):
            attn_weights = moment.get('attention_weights')

            if attn_weights is not None:
                # This moment received attention from future moments
                # Accumulate the attention weights where this moment was attended to
                attn_array = np.array(attn_weights)
                if len(attn_array) > i:
                    importance[i] += float(attn_array[i])

        # Find top-k moments
        if importance.sum() == 0:
            # No attention data yet, return most recent
            indices = list(range(max(0, len(self.buffer) - top_k), len(self.buffer)))
            return [(self.buffer[i]['step'], 0.0, self.buffer[i]['user_text'])
                    for i in indices]

        top_indices = importance.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.buffer):
                moment = self.buffer[idx]
                results.append((
                    moment['step'],
                    float(importance[idx]),
                    moment['user_text']
                ))

        return results

    def get_recent(self, k: int = 10) -> List[Dict]:
        """
        Get the k most recent moments.

        Args:
            k: Number of recent moments to return

        Returns:
            List of moment dictionaries (most recent last)
        """
        if len(self.buffer) == 0:
            return []

        start_idx = max(0, len(self.buffer) - k)
        return list(self.buffer)[start_idx:]

    def get_moment(self, step: int) -> Optional[Dict]:
        """
        Retrieve a specific moment by step number.

        Args:
            step: Step number to retrieve

        Returns:
            Moment dictionary if found, None otherwise
        """
        for moment in self.buffer:
            if moment['step'] == step:
                return moment
        return None

    def clear(self) -> None:
        """
        Clear all memories and reset buffer.

        Use when starting a new conversation or resetting the agent.
        """
        self.buffer.clear()
        self.keys_cache = None
        self.values_cache = None
        self.cache_valid = False
        self.total_moments_added = 0

    def __len__(self) -> int:
        """Return current number of moments in buffer."""
        return len(self.buffer)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"EpisodicMemory(capacity={self.capacity}, "
                f"current_size={len(self.buffer)}, "
                f"total_added={self.total_moments_added})")

    def get_statistics(self) -> Dict:
        """
        Get memory buffer statistics.

        Returns:
            Dictionary with statistics about the memory buffer
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'total_added': self.total_moments_added
            }

        surprises = self.get_surprises()

        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_added': self.total_moments_added,
            'avg_surprise': np.mean(surprises) if surprises else 0.0,
            'max_surprise': np.max(surprises) if surprises else 0.0,
            'oldest_step': self.buffer[0]['step'],
            'newest_step': self.buffer[-1]['step']
        }
