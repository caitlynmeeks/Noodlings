"""
Social Episodic Memory: Multi-Agent Memory System for Phase 4

Extends Phase 3's episodic memory to track:
- Self phenomenal states
- Other agents' inferred states
- Relationship dynamics
- Social context

This implements Theory of Mind computational substrate:
- Track multiple agents simultaneously (up to 10)
- Infer others' mental states from observations
- Model relationship patterns over time

Author: Consilience Project
Date: October 2025
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import datetime


class AgentState:
    """
    Represents the inferred state of another agent.

    Attributes:
        name: Agent identifier (e.g., "Alice", "Bob")
        inferred_state: 40-D phenomenal state (what they might be feeling)
        confidence: How certain we are about this inference (0-1)
        last_interaction: Step number of last interaction
        last_mentioned: Step number when last mentioned
        is_present: Whether agent is currently present in conversation
    """

    def __init__(
        self,
        name: str,
        inferred_state: mx.array,
        confidence: float = 0.5,
        last_interaction: int = -1,
        last_mentioned: int = -1,
        is_present: bool = False
    ):
        self.name = name
        self.inferred_state = inferred_state  # [40] phenomenal state
        self.confidence = confidence
        self.last_interaction = last_interaction
        self.last_mentioned = last_mentioned
        self.is_present = is_present

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'inferred_state': self.inferred_state.tolist() if isinstance(self.inferred_state, mx.array) else self.inferred_state,
            'confidence': float(self.confidence),
            'last_interaction': int(self.last_interaction),
            'last_mentioned': int(self.last_mentioned),
            'is_present': bool(self.is_present)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentState':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            inferred_state=mx.array(data['inferred_state']),
            confidence=data['confidence'],
            last_interaction=data['last_interaction'],
            last_mentioned=data['last_mentioned'],
            is_present=data['is_present']
        )


class SocialContext:
    """
    Represents the social context of a moment.

    Tracks:
    - Who is present
    - Conversation topic/theme
    - Group emotional valence
    - Social dynamics (conflict, harmony, etc.)
    """

    def __init__(
        self,
        present_agents: List[str],
        topic: str = "",
        group_valence: float = 0.0,
        group_arousal: float = 0.0,
        interaction_type: str = "conversation"  # conversation, conflict, celebration, etc.
    ):
        self.present_agents = present_agents
        self.topic = topic
        self.group_valence = group_valence
        self.group_arousal = group_arousal
        self.interaction_type = interaction_type

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'present_agents': self.present_agents,
            'topic': self.topic,
            'group_valence': float(self.group_valence),
            'group_arousal': float(self.group_arousal),
            'interaction_type': self.interaction_type
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SocialContext':
        """Create from dictionary."""
        return cls(
            present_agents=data['present_agents'],
            topic=data.get('topic', ''),
            group_valence=data.get('group_valence', 0.0),
            group_arousal=data.get('group_arousal', 0.0),
            interaction_type=data.get('interaction_type', 'conversation')
        )


class SocialEpisodicMemory:
    """
    Multi-agent episodic memory system.

    Stores moments containing:
    - Self phenomenal state (40-D)
    - Other agents' inferred states (40-D each)
    - Relationship dynamics
    - Social context
    - Attention weights (from retrieval)

    Extends Phase 3 EpisodicMemory with social awareness.
    """

    def __init__(
        self,
        capacity: int = 100,
        state_dim: int = 40,
        max_agents: int = 10
    ):
        """
        Initialize social episodic memory.

        Args:
            capacity: Maximum number of moments to store
            state_dim: Dimensionality of phenomenal states
            max_agents: Maximum number of other agents to track
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.max_agents = max_agents

        # Memory buffers
        self.memories: List[Dict] = []
        self.current_step = 0

        # Agent tracking
        self.known_agents: Dict[str, AgentState] = {}

    def add_memory(
        self,
        step: int,
        self_state: mx.array,
        affect: mx.array,
        agents: Dict[str, AgentState],
        social_context: SocialContext,
        user_text: str = "",
        key_vector: Optional[mx.array] = None,
        attention_weights: Optional[mx.array] = None
    ):
        """
        Add a new memory moment.

        Args:
            step: Current timestep
            self_state: Self phenomenal state [40]
            affect: Current affect vector [5]
            agents: Dictionary of other agents' states
            social_context: Social context information
            user_text: User's input text
            key_vector: Memory key for attention (computed externally)
            attention_weights: Attention weights if this was retrieved
        """
        memory = {
            'step': step,
            'timestamp': datetime.datetime.now().isoformat(),
            'self_state': self_state,
            'affect': affect,
            'agents': {name: agent.to_dict() for name, agent in agents.items()},
            'social_context': social_context.to_dict(),
            'user_text': user_text,
            'key_vector': key_vector,
            'attention_weights': attention_weights
        }

        # Add to buffer (FIFO if at capacity)
        if len(self.memories) >= self.capacity:
            self.memories.pop(0)

        self.memories.append(memory)

        # Update known agents
        for name, agent in agents.items():
            self.known_agents[name] = agent

        self.current_step = step

    def get_agent(self, name: str) -> Optional[AgentState]:
        """
        Get the current state of a known agent.

        Args:
            name: Agent name

        Returns:
            AgentState if known, None otherwise
        """
        return self.known_agents.get(name)

    def get_all_agents(self) -> Dict[str, AgentState]:
        """Get all known agents."""
        return self.known_agents.copy()

    def get_recent_interactions_with(self, agent_name: str, n: int = 5) -> List[Dict]:
        """
        Get recent interactions involving a specific agent.

        Args:
            agent_name: Name of the agent
            n: Number of recent interactions to retrieve

        Returns:
            List of memory dictionaries involving the agent
        """
        agent_memories = [
            mem for mem in self.memories
            if agent_name in mem['agents'] or agent_name in mem['social_context']['present_agents']
        ]

        return agent_memories[-n:] if len(agent_memories) > n else agent_memories

    def get_relationship_history(
        self,
        agent1: str,
        agent2: str
    ) -> List[Dict]:
        """
        Get interaction history between two agents.

        Args:
            agent1: First agent name
            agent2: Second agent name

        Returns:
            List of memories where both agents were present/mentioned
        """
        relationship_memories = [
            mem for mem in self.memories
            if (agent1 in mem['agents'] or agent1 in mem['social_context']['present_agents']) and
               (agent2 in mem['agents'] or agent2 in mem['social_context']['present_agents'])
        ]

        return relationship_memories

    def get_keys_and_values(self) -> Tuple[mx.array, List[Dict]]:
        """
        Get all memory keys and values for attention.

        Returns:
            keys: [N, key_dim] array of memory keys
            values: List of memory dictionaries
        """
        if not self.memories:
            # Return empty with correct shape
            return mx.zeros((0, self.state_dim * 2 + 16)), []

        keys = []
        values = []

        for mem in self.memories:
            if mem['key_vector'] is not None:
                # Squeeze batch dimension if present: (1, 96) -> (96)
                key = mem['key_vector']
                if len(key.shape) > 1 and key.shape[0] == 1:
                    key = key.squeeze(0)
                keys.append(key)
                values.append(mem)

        if not keys:
            return mx.zeros((0, self.state_dim * 2 + 16)), []

        keys_array = mx.stack(keys)  # Stack to (N, 96)
        return keys_array, values

    def get_memory_count(self) -> int:
        """Get number of stored memories."""
        return len(self.memories)

    def clear(self):
        """Clear all memories and agent tracking."""
        self.memories.clear()
        self.known_agents.clear()
        self.current_step = 0

    def to_dict(self) -> Dict:
        """
        Serialize to dictionary.

        Returns:
            Dictionary containing all memory state
        """
        return {
            'capacity': self.capacity,
            'state_dim': self.state_dim,
            'max_agents': self.max_agents,
            'current_step': self.current_step,
            'known_agents': {name: agent.to_dict() for name, agent in self.known_agents.items()},
            'memories': [
                {
                    'step': mem['step'],
                    'timestamp': mem['timestamp'],
                    'self_state': mem['self_state'].tolist() if isinstance(mem['self_state'], mx.array) else mem['self_state'],
                    'affect': mem['affect'].tolist() if isinstance(mem['affect'], mx.array) else mem['affect'],
                    'agents': mem['agents'],
                    'social_context': mem['social_context'],
                    'user_text': mem['user_text'],
                    'key_vector': mem['key_vector'].tolist() if mem['key_vector'] is not None and isinstance(mem['key_vector'], mx.array) else None,
                    'attention_weights': mem['attention_weights'].tolist() if mem['attention_weights'] is not None and isinstance(mem['attention_weights'], mx.array) else None
                }
                for mem in self.memories
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SocialEpisodicMemory':
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            SocialEpisodicMemory instance
        """
        memory = cls(
            capacity=data['capacity'],
            state_dim=data['state_dim'],
            max_agents=data['max_agents']
        )

        memory.current_step = data['current_step']

        # Restore known agents
        memory.known_agents = {
            name: AgentState.from_dict(agent_data)
            for name, agent_data in data['known_agents'].items()
        }

        # Restore memories
        for mem_data in data['memories']:
            memory.memories.append({
                'step': mem_data['step'],
                'timestamp': mem_data['timestamp'],
                'self_state': mx.array(mem_data['self_state']),
                'affect': mx.array(mem_data['affect']),
                'agents': {
                    name: AgentState.from_dict(agent_data)
                    for name, agent_data in mem_data['agents'].items()
                },
                'social_context': SocialContext.from_dict(mem_data['social_context']),
                'user_text': mem_data['user_text'],
                'key_vector': mx.array(mem_data['key_vector']) if mem_data['key_vector'] is not None else None,
                'attention_weights': mx.array(mem_data['attention_weights']) if mem_data['attention_weights'] is not None else None
            })

        return memory
