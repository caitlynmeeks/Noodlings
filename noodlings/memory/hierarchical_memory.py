"""
Hierarchical Memory System for Consilience

Three-tier architecture:
1. Working Memory: Recent interactions (20 slots, full detail)
2. Episodic Memory: Important moments (200 slots, high surprise/emotion)
3. Semantic Memory: Compressed patterns (stored in slow layer state)

Inspired by cognitive neuroscience models of human memory.

Author: Consilience Project
Date: October 2025
"""

import mlx.core as mx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    timestamp: float
    step: int
    user_id: str
    user_text: str
    affect: mx.array  # 5-D
    phenomenal_state: Dict  # fast/medium/slow
    surprise: float
    response: Optional[str]
    importance: float  # Computed importance score


class HierarchicalMemory:
    """
    Three-tier memory system with automatic consolidation.

    Mimics human memory:
    - Working memory: What just happened (seconds to minutes)
    - Episodic memory: Significant events (hours to days)
    - Semantic memory: Learned patterns (permanent)
    """

    def __init__(
        self,
        working_capacity: int = 20,
        episodic_capacity: int = 200,
        surprise_threshold: float = 0.5,
        importance_decay: float = 0.95
    ):
        """
        Initialize hierarchical memory.

        Args:
            working_capacity: Working memory size
            episodic_capacity: Episodic memory size
            surprise_threshold: Min surprise for episodic storage
            importance_decay: Decay factor for importance scores
        """
        self.working_capacity = working_capacity
        self.episodic_capacity = episodic_capacity
        self.surprise_threshold = surprise_threshold
        self.importance_decay = importance_decay

        # Working memory: FIFO deque
        self.working_memory: deque = deque(maxlen=working_capacity)

        # Episodic memory: Priority-based storage
        self.episodic_memory: List[MemoryEntry] = []

        # Statistics
        self.total_memories = 0
        self.consolidations = 0
        self.evictions = 0

        logger.info(
            f"Hierarchical memory initialized: "
            f"working={working_capacity}, episodic={episodic_capacity}"
        )

    def add(
        self,
        timestamp: float,
        step: int,
        user_id: str,
        user_text: str,
        affect: mx.array,
        phenomenal_state: Dict,
        surprise: float,
        response: Optional[str] = None
    ):
        """
        Add new memory entry.

        Automatically:
        1. Adds to working memory
        2. Consolidates to episodic if important
        3. Evicts old low-importance episodic memories

        Args:
            timestamp: Unix timestamp
            step: Interaction step number
            user_id: User identifier
            user_text: User input text
            affect: 5-D affect vector
            phenomenal_state: Full phenomenal state dict
            surprise: Prediction error
            response: Agent response (if any)
        """
        # Compute importance score
        importance = self._compute_importance(
            surprise=surprise,
            affect=affect,
            response=response
        )

        entry = MemoryEntry(
            timestamp=timestamp,
            step=step,
            user_id=user_id,
            user_text=user_text,
            affect=affect,
            phenomenal_state=phenomenal_state,
            surprise=surprise,
            response=response,
            importance=importance
        )

        # Add to working memory (automatic FIFO)
        self.working_memory.append(entry)
        self.total_memories += 1

        # Consolidate to episodic if important
        if importance > self.surprise_threshold:
            self._consolidate_to_episodic(entry)

        # Decay importance of old episodic memories
        self._decay_episodic_importance()

        # Evict low-importance episodic memories if at capacity
        if len(self.episodic_memory) >= self.episodic_capacity:
            self._evict_lowest_importance()

    def _compute_importance(
        self,
        surprise: float,
        affect: mx.array,
        response: Optional[str]
    ) -> float:
        """
        Compute importance score for memory consolidation.

        Factors:
        - Surprise (prediction error)
        - Emotional intensity (arousal + abs(valence))
        - Whether agent responded

        Returns:
            Importance score [0, 1]
        """
        # Surprise component (0-1)
        surprise_score = min(surprise, 1.0)

        # Emotional intensity (0-1)
        # Handle MLX arrays - need to squeeze batch dimension first
        import mlx.core as mx
        if isinstance(affect, mx.array):
            # Squeeze batch dimension: (1, 5) -> (5,)
            affect_squeezed = affect.squeeze() if len(affect.shape) > 1 else affect
            valence = float(affect_squeezed[0].item())
            arousal = float(affect_squeezed[1].item())
        else:
            # Regular Python list/array
            valence = float(affect[0])
            arousal = float(affect[1])

        emotion_score = min(abs(valence) + arousal, 1.0) / 2.0

        # Response component (binary)
        response_score = 1.0 if response else 0.0

        # Weighted combination
        importance = (
            0.5 * surprise_score +
            0.3 * emotion_score +
            0.2 * response_score
        )

        return importance

    def _consolidate_to_episodic(self, entry: MemoryEntry):
        """
        Consolidate entry from working to episodic memory.

        Args:
            entry: Memory entry to consolidate
        """
        self.episodic_memory.append(entry)
        self.consolidations += 1

        logger.debug(
            f"Consolidated to episodic: surprise={entry.surprise:.2f}, "
            f"importance={entry.importance:.2f}"
        )

    def _decay_episodic_importance(self):
        """
        Decay importance of episodic memories over time.

        Older memories become less important unless reinforced.
        """
        for entry in self.episodic_memory:
            entry.importance *= self.importance_decay

    def _evict_lowest_importance(self):
        """
        Evict lowest-importance episodic memory.

        Makes room for new important memories.
        """
        if not self.episodic_memory:
            return

        # Find minimum importance
        min_idx = min(
            range(len(self.episodic_memory)),
            key=lambda i: self.episodic_memory[i].importance
        )

        evicted = self.episodic_memory.pop(min_idx)
        self.evictions += 1

        logger.debug(
            f"Evicted episodic memory: step={evicted.step}, "
            f"importance={evicted.importance:.2f}"
        )

    def retrieve_working(self, last_n: Optional[int] = None) -> List[MemoryEntry]:
        """
        Retrieve from working memory.

        Args:
            last_n: Number of recent entries (None = all)

        Returns:
            List of memory entries
        """
        if last_n is None:
            return list(self.working_memory)
        else:
            return list(self.working_memory)[-last_n:]

    def retrieve_episodic(
        self,
        user_id: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Retrieve from episodic memory.

        Args:
            user_id: Filter by user (None = all users)
            min_importance: Minimum importance threshold
            limit: Maximum number to retrieve

        Returns:
            List of memory entries, sorted by importance
        """
        filtered = self.episodic_memory

        # Filter by user
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]

        # Filter by importance
        filtered = [e for e in filtered if e.importance >= min_importance]

        # Sort by importance (descending)
        filtered = sorted(filtered, key=lambda e: e.importance, reverse=True)

        # Limit
        return filtered[:limit]

    def retrieve_context(
        self,
        user_id: str,
        context_size: int = 10
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant context for user interaction.

        Combines:
        - Recent working memory (last 5)
        - Relevant episodic memories (user-specific, important)

        Args:
            user_id: User to retrieve context for
            context_size: Total context size

        Returns:
            Combined context memories
        """
        # Get recent working memory
        working = self.retrieve_working(last_n=5)

        # Get user-specific episodic memories
        episodic = self.retrieve_episodic(
            user_id=user_id,
            min_importance=0.3,
            limit=context_size - len(working)
        )

        # Combine (working first, then episodic by importance)
        context = working + episodic

        return context[:context_size]

    def get_user_history(self, user_id: str) -> Dict:
        """
        Get full history for specific user.

        Returns:
            Dict with statistics and memories
        """
        # Count interactions
        working_count = sum(1 for e in self.working_memory if e.user_id == user_id)
        episodic_count = sum(1 for e in self.episodic_memory if e.user_id == user_id)

        # Get episodic memories
        user_episodic = self.retrieve_episodic(user_id=user_id, limit=100)

        # Compute statistics
        if user_episodic:
            avg_surprise = np.mean([e.surprise for e in user_episodic])
            avg_importance = np.mean([e.importance for e in user_episodic])
        else:
            avg_surprise = 0.0
            avg_importance = 0.0

        return {
            'user_id': user_id,
            'working_memory_count': working_count,
            'episodic_memory_count': episodic_count,
            'total_interactions': working_count + episodic_count,
            'avg_surprise': float(avg_surprise),
            'avg_importance': float(avg_importance),
            'important_moments': [
                {
                    'step': e.step,
                    'text': e.user_text,
                    'surprise': e.surprise,
                    'importance': e.importance
                }
                for e in user_episodic[:10]
            ]
        }

    def get_stats(self) -> Dict:
        """Get memory system statistics."""
        return {
            'working_memory': {
                'capacity': self.working_capacity,
                'current': len(self.working_memory),
                'utilization': len(self.working_memory) / self.working_capacity
            },
            'episodic_memory': {
                'capacity': self.episodic_capacity,
                'current': len(self.episodic_memory),
                'utilization': len(self.episodic_memory) / self.episodic_capacity,
                'avg_importance': np.mean([e.importance for e in self.episodic_memory])
                if self.episodic_memory else 0.0
            },
            'total_memories': self.total_memories,
            'consolidations': self.consolidations,
            'evictions': self.evictions,
            'consolidation_rate': self.consolidations / max(self.total_memories, 1)
        }

    def save_state(self, filepath: str):
        """
        Save memory state to disk.

        Args:
            filepath: Path to save file
        """
        import pickle

        state = {
            'working_memory': list(self.working_memory),
            'episodic_memory': self.episodic_memory,
            'total_memories': self.total_memories,
            'consolidations': self.consolidations,
            'evictions': self.evictions
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Memory state saved: {filepath}")

    def load_state(self, filepath: str):
        """
        Load memory state from disk.

        Args:
            filepath: Path to load from
        """
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.working_memory = deque(state['working_memory'], maxlen=self.working_capacity)
        self.episodic_memory = state['episodic_memory']
        self.total_memories = state['total_memories']
        self.consolidations = state['consolidations']
        self.evictions = state['evictions']

        logger.info(f"Memory state loaded: {filepath}")
