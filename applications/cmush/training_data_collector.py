"""
Training Data Collector for cMUSH

Captures rich interaction data for offline training and evaluation.

Stores:
- Full conversation sequences with multi-agent context
- Affect vectors and phenomenal states
- Relationship evolution trajectories
- Theory of Mind ground truth (when available)
- Surprise patterns and response triggers

Author: Consilience Project
Date: October 2025
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """
    Collects structured training data from cMUSH interactions.

    Data format optimized for:
    1. Phase 4+ fine-tuning (real conversation patterns)
    2. Theory of Mind evaluation
    3. Relationship modeling validation
    4. Multi-agent interaction analysis
    """

    def __init__(self, data_dir: str = "training/data/cmush_real"):
        """
        Initialize training data collector.

        Args:
            data_dir: Directory to store training data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Session tracking
        self.current_session = None
        self.session_data = []

        # Conversation tracking
        self.conversations = {}  # agent_id -> conversation_buffer

        logger.info(f"Training data collector initialized: {data_dir}")

    def start_session(self, session_id: Optional[str] = None):
        """
        Start a new data collection session.

        Args:
            session_id: Optional session identifier
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_session = session_id
        self.session_data = []

        logger.info(f"Started data collection session: {session_id}")

    def log_interaction(
        self,
        agent_id: str,
        user_id: str,
        user_text: str,
        affect_vector: List[float],
        phenomenal_state: Dict,
        surprise: float,
        response: Optional[Dict] = None,
        context: Optional[Dict] = None
    ):
        """
        Log a single agent-user interaction.

        Args:
            agent_id: Agent identifier
            user_id: User identifier
            user_text: User input text
            affect_vector: 5-D affect [valence, arousal, fear, sorrow, boredom]
            phenomenal_state: Full 40-D state (fast/medium/slow)
            surprise: Prediction error magnitude
            response: Agent response (if generated)
            context: Additional context (room, nearby agents, etc.)
        """
        if not self.current_session:
            self.start_session()

        interaction = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session,
            'agent_id': agent_id,
            'user_id': user_id,
            'user_text': user_text,
            'affect': {
                'valence': float(affect_vector[0]),
                'arousal': float(affect_vector[1]),
                'fear': float(affect_vector[2]),
                'sorrow': float(affect_vector[3]),
                'boredom': float(affect_vector[4])
            },
            'phenomenal_state': {
                'fast': phenomenal_state.get('fast', []),
                'medium': phenomenal_state.get('medium', []),
                'slow': phenomenal_state.get('slow', [])
            },
            'surprise': float(surprise),
            'responded': response is not None,
            'response': response,
            'context': context or {}
        }

        # Add to session buffer
        self.session_data.append(interaction)

        # Add to per-agent conversation buffer
        if agent_id not in self.conversations:
            self.conversations[agent_id] = []
        self.conversations[agent_id].append(interaction)

    def log_relationship_snapshot(
        self,
        agent_id: str,
        user_id: str,
        relationship: Dict
    ):
        """
        Log relationship state snapshot.

        Args:
            agent_id: Agent identifier
            user_id: User identifier
            relationship: Relationship model dict
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session,
            'type': 'relationship_snapshot',
            'agent_id': agent_id,
            'user_id': user_id,
            'relationship': relationship
        }

        self.session_data.append(snapshot)

    def log_theory_of_mind_inference(
        self,
        agent_id: str,
        target_user: str,
        inferred_state: Dict,
        ground_truth: Optional[Dict] = None
    ):
        """
        Log Theory of Mind inference.

        Args:
            agent_id: Agent making inference
            target_user: User being modeled
            inferred_state: Agent's inference about user's state
            ground_truth: Actual user state (if available)
        """
        tom_log = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session,
            'type': 'theory_of_mind',
            'agent_id': agent_id,
            'target_user': target_user,
            'inferred_state': inferred_state,
            'ground_truth': ground_truth
        }

        self.session_data.append(tom_log)

    def end_session(self):
        """
        End current session and save data.
        """
        if not self.current_session:
            logger.warning("No active session to end")
            return

        # Save session data
        session_file = self.data_dir / f"session_{self.current_session}.jsonl"

        with open(session_file, 'w') as f:
            for entry in self.session_data:
                f.write(json.dumps(entry) + '\n')

        logger.info(f"Saved {len(self.session_data)} entries to {session_file}")

        # Save per-agent conversation sequences
        for agent_id, conversation in self.conversations.items():
            agent_file = self.data_dir / f"agent_{agent_id}_{self.current_session}.json"

            with open(agent_file, 'w') as f:
                json.dump({
                    'agent_id': agent_id,
                    'session_id': self.current_session,
                    'conversation': conversation
                }, f, indent=2)

        # Reset
        self.current_session = None
        self.session_data = []
        self.conversations = {}

    def create_training_sequences(
        self,
        min_sequence_length: int = 10,
        max_sequence_length: int = 100
    ) -> List[Dict]:
        """
        Convert collected data into training sequences.

        Creates sequences suitable for BPTT training with proper
        temporal structure and context.

        Args:
            min_sequence_length: Minimum turns per sequence
            max_sequence_length: Maximum turns per sequence

        Returns:
            List of training sequences
        """
        sequences = []

        # Process each agent's conversations
        for agent_id, conversation in self.conversations.items():
            if len(conversation) < min_sequence_length:
                continue

            # Split into sequences of max_sequence_length
            for i in range(0, len(conversation), max_sequence_length):
                sequence = conversation[i:i + max_sequence_length]

                if len(sequence) < min_sequence_length:
                    continue

                # Extract affect sequence
                affect_sequence = [
                    [
                        entry['affect']['valence'],
                        entry['affect']['arousal'],
                        entry['affect']['fear'],
                        entry['affect']['sorrow'],
                        entry['affect']['boredom']
                    ]
                    for entry in sequence
                ]

                # Extract phenomenal state sequence
                state_sequence = [
                    entry['phenomenal_state']
                    for entry in sequence
                ]

                sequences.append({
                    'agent_id': agent_id,
                    'sequence_length': len(sequence),
                    'affect_sequence': affect_sequence,
                    'state_sequence': state_sequence,
                    'user_texts': [entry['user_text'] for entry in sequence],
                    'surprises': [entry['surprise'] for entry in sequence],
                    'session_id': sequence[0]['session_id']
                })

        logger.info(f"Created {len(sequences)} training sequences")
        return sequences

    def export_for_training(
        self,
        output_file: str,
        min_sequence_length: int = 10
    ):
        """
        Export all collected data as training dataset.

        Args:
            output_file: Output file path
            min_sequence_length: Minimum sequence length
        """
        sequences = self.create_training_sequences(
            min_sequence_length=min_sequence_length
        )

        with open(output_file, 'w') as f:
            json.dump({
                'dataset_info': {
                    'source': 'cMUSH real conversations',
                    'created': datetime.now().isoformat(),
                    'num_sequences': len(sequences),
                    'total_turns': sum(s['sequence_length'] for s in sequences)
                },
                'sequences': sequences
            }, f, indent=2)

        logger.info(f"Exported training data to {output_file}")
        logger.info(f"  - {len(sequences)} sequences")
        logger.info(f"  - {sum(s['sequence_length'] for s in sequences)} total turns")

    def get_stats(self) -> Dict:
        """Get statistics about collected data."""
        total_interactions = sum(len(conv) for conv in self.conversations.values())

        return {
            'current_session': self.current_session,
            'total_agents': len(self.conversations),
            'total_interactions': total_interactions,
            'session_entries': len(self.session_data)
        }
