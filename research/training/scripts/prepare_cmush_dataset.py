#!/usr/bin/env python3
"""
Prepare cMUSH training data from JSONL session files.

Reads all session JSONL files and consolidates them into
a single training dataset with conversation sequences.

Usage:
    python3 prepare_cmush_dataset.py

Output:
    ../data/cmush_real/exported_dataset.json

Author: Consilience Project
Date: October 31, 2025
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_jsonl_files(data_dir: str):
    """
    Load all JSONL session files.

    Args:
        data_dir: Directory containing session_*.jsonl files

    Returns:
        List of interaction dictionaries
    """
    interactions = []
    jsonl_files = sorted(Path(data_dir).glob('session_*.jsonl'))

    print(f"Found {len(jsonl_files)} session files")

    for jsonl_file in jsonl_files:
        if jsonl_file.stat().st_size == 0:
            continue

        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        interaction = json.loads(line)
                        interactions.append(interaction)
        except Exception as e:
            print(f"Warning: Could not read {jsonl_file}: {e}")
            continue

    print(f"Loaded {len(interactions)} total interactions")
    return interactions


def group_into_sequences(interactions, max_gap_seconds=300):
    """
    Group interactions into conversation sequences.

    Sequences are split when:
    - Agent changes
    - Time gap > max_gap_seconds
    - Session changes

    Args:
        interactions: List of interaction dicts
        max_gap_seconds: Max time between interactions in same sequence

    Returns:
        List of sequence dictionaries
    """
    # Sort by timestamp
    interactions.sort(key=lambda x: x['timestamp'])

    sequences = []
    current_sequence = None

    for interaction in interactions:
        agent_id = interaction['agent_id']
        timestamp = datetime.fromisoformat(interaction['timestamp'])
        session_id = interaction.get('session_id', 'unknown')

        # Check if we should start new sequence
        start_new = False

        if current_sequence is None:
            start_new = True
        elif current_sequence['agent_id'] != agent_id:
            start_new = True
        elif current_sequence['session_id'] != session_id:
            start_new = True
        else:
            # Check time gap
            last_time = datetime.fromisoformat(current_sequence['interactions'][-1]['timestamp'])
            gap = (timestamp - last_time).total_seconds()
            if gap > max_gap_seconds:
                start_new = True

        if start_new:
            if current_sequence is not None and len(current_sequence['interactions']) > 0:
                sequences.append(current_sequence)

            current_sequence = {
                'agent_id': agent_id,
                'session_id': session_id,
                'start_time': timestamp.isoformat(),
                'interactions': []
            }

        current_sequence['interactions'].append(interaction)

    # Add last sequence
    if current_sequence is not None and len(current_sequence['interactions']) > 0:
        sequences.append(current_sequence)

    print(f"Grouped into {len(sequences)} conversation sequences")
    return sequences


def extract_training_sequences(sequences):
    """
    Extract affect sequences and phenomenal states for training.

    Args:
        sequences: List of conversation sequences

    Returns:
        List of training sequence dictionaries
    """
    training_sequences = []

    for seq_idx, seq in enumerate(sequences):
        # Extract affect vectors
        affect_sequence = []
        phenomenal_states = []
        surprises = []

        for interaction in seq['interactions']:
            affect = interaction['affect']
            affect_vector = [
                affect['valence'],
                affect['arousal'],
                affect['fear'],
                affect['sorrow'],
                affect['boredom']
            ]
            affect_sequence.append(affect_vector)

            # Store phenomenal states
            phenom = interaction['phenomenal_state']
            phenomenal_states.append({
                'fast': phenom['fast'],
                'medium': phenom['medium'],
                'slow': phenom['slow']
            })

            surprises.append(interaction['surprise'])

        training_sequences.append({
            'sequence_id': seq_idx,
            'agent_id': seq['agent_id'],
            'session_id': seq['session_id'],
            'start_time': seq['start_time'],
            'length': len(affect_sequence),
            'affect_sequence': affect_sequence,
            'phenomenal_states': phenomenal_states,
            'surprises': surprises,
            'mean_surprise': sum(surprises) / len(surprises) if surprises else 0.0,
            'max_surprise': max(surprises) if surprises else 0.0
        })

    return training_sequences


def main():
    """Main preparation pipeline."""
    # Paths
    project_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = project_dir / 'training' / 'data' / 'cmush_real'
    output_file = data_dir / 'exported_dataset.json'

    print("=" * 70)
    print("CMUSH DATASET PREPARATION")
    print("=" * 70)
    print()

    # 1. Load JSONL files
    print("[1/4] Loading JSONL session files...")
    interactions = load_jsonl_files(data_dir)

    if len(interactions) == 0:
        print("ERROR: No interactions found!")
        return

    # 2. Group into sequences
    print("\n[2/4] Grouping into conversation sequences...")
    sequences = group_into_sequences(interactions)

    # 3. Extract training data
    print("\n[3/4] Extracting training sequences...")
    training_sequences = extract_training_sequences(sequences)

    # 4. Build dataset
    print("\n[4/4] Building dataset...")

    # Calculate statistics
    agent_counts = defaultdict(int)
    total_turns = 0
    sequence_lengths = []

    for seq in training_sequences:
        agent_counts[seq['agent_id']] += 1
        total_turns += seq['length']
        sequence_lengths.append(seq['length'])

    mean_length = sum(sequence_lengths) / len(sequence_lengths)
    max_length = max(sequence_lengths)

    dataset = {
        'dataset_info': {
            'source': 'cMUSH real conversations',
            'created': datetime.now().isoformat(),
            'num_sequences': len(training_sequences),
            'total_turns': total_turns,
            'agents': dict(agent_counts),
            'mean_sequence_length': mean_length,
            'max_sequence_length': max_length
        },
        'sequences': training_sequences
    }

    # Save
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✓ Dataset saved to: {output_file}")
    print()
    print("Dataset Statistics:")
    print(f"  Sequences: {len(training_sequences)}")
    print(f"  Total turns: {total_turns}")
    print(f"  Mean sequence length: {mean_length:.1f}")
    print(f"  Max sequence length: {max_length}")
    print(f"  Agents:")
    for agent_id, count in agent_counts.items():
        print(f"    - {agent_id}: {count} sequences")

    print()
    print("Ready for training! Run:")
    print(f"  python3 training/scripts/05_train_on_cmush_data.py \\")
    print(f"    --data {output_file} \\")
    print(f"    --epochs 20")
    print()


if __name__ == '__main__':
    main()
