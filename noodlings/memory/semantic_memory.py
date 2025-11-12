"""
Semantic Memory System for Consilience

Combines:
1. Implicit semantic memory (slow layer 8-D state) - continuous, learned
2. Explicit semantic memory (structured facts) - discrete, queryable

The slow layer provides dense compressed patterns.
The explicit store provides interpretable facts.

Together they form long-term knowledge about users.

Author: Consilience Project
Date: October 2025
"""

import mlx.core as mx
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticFact:
    """
    A single semantic fact about a user.

    Facts are extracted from episodic memories and stored explicitly.
    """
    fact_type: str  # 'personality', 'preference', 'habit', 'relationship'
    content: str    # Human-readable description
    confidence: float  # [0, 1] - how confident we are
    evidence_count: int  # Number of supporting memories
    first_observed: float  # Timestamp
    last_reinforced: float  # Timestamp

    def decay_confidence(self, decay_rate: float = 0.99):
        """Decay confidence over time (facts need reinforcement)."""
        self.confidence *= decay_rate


@dataclass
class UserSemanticProfile:
    """
    Complete semantic profile for a single user.

    Combines:
    - Implicit representation (slow layer state)
    - Explicit facts (queryable knowledge)
    """
    user_id: str

    # Implicit semantic memory (learned via backprop)
    slow_layer_state: mx.array = field(default_factory=lambda: mx.zeros(8))

    # Explicit semantic facts (extracted from episodic)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    # e.g., {'anxious': 0.7, 'curious': 0.8, 'optimistic': 0.3}

    preferences: List[SemanticFact] = field(default_factory=list)
    # e.g., "Likes discussing AI", "Dislikes small talk"

    habits: List[SemanticFact] = field(default_factory=list)
    # e.g., "Greets in morning", "Often mentions work stress"

    key_facts: List[SemanticFact] = field(default_factory=list)
    # e.g., "Has a dog named Max", "Works as software engineer"

    # Statistics
    total_interactions: int = 0
    first_interaction: Optional[float] = None
    last_interaction: Optional[float] = None

    def add_fact(self, fact: SemanticFact):
        """Add semantic fact to appropriate category."""
        if fact.fact_type == 'personality':
            # Extract trait name and confidence
            # (This would be parsed from fact.content in practice)
            pass
        elif fact.fact_type == 'preference':
            self.preferences.append(fact)
        elif fact.fact_type == 'habit':
            self.habits.append(fact)
        elif fact.fact_type == 'key_fact':
            self.key_facts.append(fact)

    def get_summary(self) -> str:
        """Generate human-readable summary of user."""
        summary_parts = []

        # Personality
        if self.personality_traits:
            traits = [f"{k} ({v:.1f})" for k, v in
                     sorted(self.personality_traits.items(),
                            key=lambda x: x[1], reverse=True)[:3]]
            summary_parts.append(f"Personality: {', '.join(traits)}")

        # Key facts
        if self.key_facts:
            facts = [f.content for f in self.key_facts[:3]]
            summary_parts.append(f"Facts: {'; '.join(facts)}")

        # Stats
        summary_parts.append(
            f"Interactions: {self.total_interactions}"
        )

        return " | ".join(summary_parts)


class SemanticMemorySystem:
    """
    Manages semantic memory for all users.

    Architecture:

    1. Implicit Layer (Continuous):
       - Each user has 8-D slow layer state
       - Updated via backprop during learning
       - Dense, distributed representation

    2. Explicit Layer (Discrete):
       - Facts extracted from episodic memories
       - Queryable, interpretable
       - Sparse representation

    3. Consolidation Process:
       - Periodically extract patterns from episodic
       - Create semantic facts
       - Update personality trait estimates
       - Decay old facts that aren't reinforced
    """

    def __init__(
        self,
        max_users: int = 100,
        fact_extraction_threshold: int = 10,  # Min interactions before extraction
        consolidation_interval: int = 50  # Extract facts every N interactions
    ):
        """
        Initialize semantic memory system.

        Args:
            max_users: Maximum users to track
            fact_extraction_threshold: Min interactions before fact extraction
            consolidation_interval: How often to consolidate episodic → semantic
        """
        self.max_users = max_users
        self.fact_extraction_threshold = fact_extraction_threshold
        self.consolidation_interval = consolidation_interval

        # User profiles (user_id -> profile)
        self.profiles: Dict[str, UserSemanticProfile] = {}

        # Consolidation tracking
        self.interactions_since_consolidation = 0

        logger.info(
            f"Semantic memory initialized: "
            f"max_users={max_users}, "
            f"consolidation_interval={consolidation_interval}"
        )

    def get_or_create_profile(self, user_id: str) -> UserSemanticProfile:
        """
        Get user profile, creating if doesn't exist.

        Args:
            user_id: User identifier

        Returns:
            User's semantic profile
        """
        if user_id not in self.profiles:
            if len(self.profiles) >= self.max_users:
                self._evict_least_recent_user()

            self.profiles[user_id] = UserSemanticProfile(user_id=user_id)
            logger.info(f"Created semantic profile for {user_id}")

        return self.profiles[user_id]

    def update_implicit_state(
        self,
        user_id: str,
        slow_layer_state: mx.array
    ):
        """
        Update user's implicit semantic memory (slow layer state).

        This is called after every forward pass to keep the profile
        synchronized with the neural network state.

        Args:
            user_id: User identifier
            slow_layer_state: 8-D slow layer state from model
        """
        profile = self.get_or_create_profile(user_id)
        profile.slow_layer_state = slow_layer_state
        profile.total_interactions += 1

        import time
        if profile.first_interaction is None:
            profile.first_interaction = time.time()
        profile.last_interaction = time.time()

        self.interactions_since_consolidation += 1

    def consolidate_episodic_memories(
        self,
        user_id: str,
        episodic_memories: List
    ):
        """
        Extract semantic facts from episodic memories.

        This is the key "compression" step:
        - Episodic: "User said X, Y, Z about work stress"
        - Semantic: "User tends to be anxious about work"

        Args:
            user_id: User identifier
            episodic_memories: List of episodic memory entries
        """
        profile = self.get_or_create_profile(user_id)

        if len(episodic_memories) < self.fact_extraction_threshold:
            return  # Not enough data yet

        # Extract personality traits from affect patterns
        self._extract_personality_traits(profile, episodic_memories)

        # Extract habits from temporal patterns
        self._extract_habits(profile, episodic_memories)

        # Extract preferences from repeated topics
        self._extract_preferences(profile, episodic_memories)

        logger.debug(f"Consolidated semantic memory for {user_id}")

    def _extract_personality_traits(
        self,
        profile: UserSemanticProfile,
        memories: List
    ):
        """
        Extract personality traits from affect patterns.

        Example:
        - If valence is often negative → trait: 'pessimistic'
        - If arousal is often high → trait: 'energetic'
        - If fear is often elevated → trait: 'anxious'
        """
        if not memories:
            return

        # Compute average affect across memories
        valences = []
        arousals = []
        fears = []
        sorrows = []

        for mem in memories:
            if hasattr(mem, 'affect'):
                affect = mem.affect
                valences.append(float(affect[0]))
                arousals.append(float(affect[1]))
                fears.append(float(affect[2]))
                sorrows.append(float(affect[3]))

        if not valences:
            return

        # Infer traits
        avg_valence = np.mean(valences)
        avg_arousal = np.mean(arousals)
        avg_fear = np.mean(fears)
        avg_sorrow = np.mean(sorrows)

        # Map to personality traits
        if avg_valence > 0.3:
            profile.personality_traits['optimistic'] = min(avg_valence, 1.0)
        elif avg_valence < -0.3:
            profile.personality_traits['pessimistic'] = min(-avg_valence, 1.0)

        if avg_arousal > 0.6:
            profile.personality_traits['energetic'] = min(avg_arousal, 1.0)
        elif avg_arousal < 0.3:
            profile.personality_traits['calm'] = 1.0 - avg_arousal

        if avg_fear > 0.4:
            profile.personality_traits['anxious'] = min(avg_fear, 1.0)

        if avg_sorrow > 0.4:
            profile.personality_traits['melancholic'] = min(avg_sorrow, 1.0)

    def _extract_habits(
        self,
        profile: UserSemanticProfile,
        memories: List
    ):
        """
        Extract habitual patterns from temporal data.

        Example:
        - If user greets every morning → habit: "morning greeter"
        - If user often discusses work → habit: "work-focused"
        """
        # Count topic frequencies (simplified - would use NLP in practice)
        topic_counts = defaultdict(int)

        for mem in memories:
            if hasattr(mem, 'user_text'):
                text = mem.user_text.lower()

                # Naive topic detection (would use better NLP)
                if 'work' in text or 'job' in text:
                    topic_counts['work'] += 1
                if 'family' in text or 'kids' in text:
                    topic_counts['family'] += 1
                if 'ai' in text or 'technology' in text:
                    topic_counts['technology'] += 1

        # Create habit facts for frequent topics
        total_memories = len(memories)
        for topic, count in topic_counts.items():
            frequency = count / total_memories
            if frequency > 0.2:  # Mentioned in >20% of conversations
                fact = SemanticFact(
                    fact_type='habit',
                    content=f"Often discusses {topic}",
                    confidence=min(frequency * 2, 1.0),
                    evidence_count=count,
                    first_observed=profile.first_interaction or 0,
                    last_reinforced=profile.last_interaction or 0
                )

                # Add if not already present
                if not any(f.content == fact.content for f in profile.habits):
                    profile.habits.append(fact)

    def _extract_preferences(
        self,
        profile: UserSemanticProfile,
        memories: List
    ):
        """
        Extract preferences from positive/negative affect patterns.

        Example:
        - High valence when discussing AI → preference: "interested in AI"
        - Low valence when discussing politics → preference: "dislikes politics"
        """
        # Group memories by topic and compute average affect
        topic_affects = defaultdict(list)

        for mem in memories:
            if hasattr(mem, 'user_text') and hasattr(mem, 'affect'):
                text = mem.user_text.lower()
                valence = float(mem.affect[0])

                # Naive topic detection
                if 'ai' in text or 'consciousness' in text:
                    topic_affects['AI/consciousness'].append(valence)
                if 'dog' in text or 'cat' in text or 'pet' in text:
                    topic_affects['pets'].append(valence)

        # Extract preferences from strong valence patterns
        for topic, valences in topic_affects.items():
            if len(valences) < 3:  # Need multiple instances
                continue

            avg_valence = np.mean(valences)

            if avg_valence > 0.4:  # Positive preference
                fact = SemanticFact(
                    fact_type='preference',
                    content=f"Enjoys discussing {topic}",
                    confidence=min(avg_valence, 1.0),
                    evidence_count=len(valences),
                    first_observed=profile.first_interaction or 0,
                    last_reinforced=profile.last_interaction or 0
                )

                if not any(f.content == fact.content for f in profile.preferences):
                    profile.preferences.append(fact)

            elif avg_valence < -0.4:  # Negative preference
                fact = SemanticFact(
                    fact_type='preference',
                    content=f"Dislikes discussing {topic}",
                    confidence=min(-avg_valence, 1.0),
                    evidence_count=len(valences),
                    first_observed=profile.first_interaction or 0,
                    last_reinforced=profile.last_interaction or 0
                )

                if not any(f.content == fact.content for f in profile.preferences):
                    profile.preferences.append(fact)

    def should_consolidate(self) -> bool:
        """Check if it's time to consolidate episodic → semantic."""
        return self.interactions_since_consolidation >= self.consolidation_interval

    def reset_consolidation_counter(self):
        """Reset consolidation counter."""
        self.interactions_since_consolidation = 0

    def query_user_profile(self, user_id: str) -> Optional[UserSemanticProfile]:
        """
        Get complete semantic profile for user.

        Args:
            user_id: User identifier

        Returns:
            User profile or None if not found
        """
        return self.profiles.get(user_id)

    def get_user_summary(self, user_id: str) -> str:
        """
        Get human-readable summary of user.

        Args:
            user_id: User identifier

        Returns:
            Summary string
        """
        profile = self.profiles.get(user_id)
        if not profile:
            return f"No profile for {user_id}"

        return profile.get_summary()

    def _evict_least_recent_user(self):
        """Evict least recently interacted user to make room."""
        if not self.profiles:
            return

        # Find user with oldest last_interaction
        oldest_user = min(
            self.profiles.keys(),
            key=lambda uid: self.profiles[uid].last_interaction or 0
        )

        del self.profiles[oldest_user]
        logger.info(f"Evicted semantic profile: {oldest_user}")

    def save_state(self, filepath: str):
        """
        Save all semantic profiles to disk.

        Args:
            filepath: Path to save file (JSON)
        """
        # Convert to serializable format
        state = {
            'profiles': {
                user_id: {
                    'user_id': profile.user_id,
                    'slow_layer_state': profile.slow_layer_state.tolist(),
                    'personality_traits': profile.personality_traits,
                    'preferences': [
                        {
                            'fact_type': f.fact_type,
                            'content': f.content,
                            'confidence': f.confidence,
                            'evidence_count': f.evidence_count,
                            'first_observed': f.first_observed,
                            'last_reinforced': f.last_reinforced
                        }
                        for f in profile.preferences
                    ],
                    'habits': [
                        {
                            'fact_type': f.fact_type,
                            'content': f.content,
                            'confidence': f.confidence,
                            'evidence_count': f.evidence_count,
                            'first_observed': f.first_observed,
                            'last_reinforced': f.last_reinforced
                        }
                        for f in profile.habits
                    ],
                    'key_facts': [
                        {
                            'fact_type': f.fact_type,
                            'content': f.content,
                            'confidence': f.confidence,
                            'evidence_count': f.evidence_count,
                            'first_observed': f.first_observed,
                            'last_reinforced': f.last_reinforced
                        }
                        for f in profile.key_facts
                    ],
                    'total_interactions': profile.total_interactions,
                    'first_interaction': profile.first_interaction,
                    'last_interaction': profile.last_interaction
                }
                for user_id, profile in self.profiles.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved semantic memory: {filepath} ({len(self.profiles)} profiles)")

    def load_state(self, filepath: str):
        """
        Load semantic profiles from disk.

        Args:
            filepath: Path to load from
        """
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.profiles = {}

        for user_id, profile_data in state['profiles'].items():
            profile = UserSemanticProfile(
                user_id=profile_data['user_id'],
                slow_layer_state=mx.array(profile_data['slow_layer_state']),
                personality_traits=profile_data['personality_traits'],
                total_interactions=profile_data['total_interactions'],
                first_interaction=profile_data['first_interaction'],
                last_interaction=profile_data['last_interaction']
            )

            # Reconstruct facts
            for fact_data in profile_data['preferences']:
                profile.preferences.append(SemanticFact(**fact_data))

            for fact_data in profile_data['habits']:
                profile.habits.append(SemanticFact(**fact_data))

            for fact_data in profile_data['key_facts']:
                profile.key_facts.append(SemanticFact(**fact_data))

            self.profiles[user_id] = profile

        logger.info(f"Loaded semantic memory: {filepath} ({len(self.profiles)} profiles)")

    def get_stats(self) -> Dict:
        """Get semantic memory statistics."""
        if not self.profiles:
            return {'total_users': 0}

        total_facts = sum(
            len(p.preferences) + len(p.habits) + len(p.key_facts)
            for p in self.profiles.values()
        )

        return {
            'total_users': len(self.profiles),
            'total_facts': total_facts,
            'avg_facts_per_user': total_facts / len(self.profiles),
            'avg_interactions_per_user': np.mean([
                p.total_interactions for p in self.profiles.values()
            ])
        }
