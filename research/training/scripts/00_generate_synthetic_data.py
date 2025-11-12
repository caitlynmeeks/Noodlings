#!/usr/bin/env python3
"""
Generate synthetic social conversations for Phase 4 pretraining.

This creates 50,000 examples with known patterns.
Run once, use forever.
"""

import mlx.core as mx
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import random
import sys
import os

# Add project dirs to path (works from any location)
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent.parent
sys.path.insert(0, str(project_dir / 'noodlings'))
sys.path.insert(0, str(project_dir))

from social_memory import SocialContext


def generate_synthetic_dataset(
    num_examples: int = 50000,
    output_dir: str = 'training/data/synthetic'
):
    """
    Generate synthetic social conversation dataset.

    Creates diverse scenarios with known emotional/social patterns.
    """
    print(f"Generating {num_examples:,} synthetic examples...")
    print(f"Output directory: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Scenario templates
    scenarios = [
        # Support scenarios (positive effect)
        {
            'templates': [
                "{agent} helped me with {problem}",
                "{agent} listened when I needed to talk about {problem}",
                "{agent} gave me great advice about {problem}",
                "{agent} was there for me during {problem}"
            ],
            'agent_affect': lambda: [
                random.uniform(0.4, 0.8),   # valence: positive
                random.uniform(0.3, 0.6),   # arousal: moderate
                random.uniform(0.0, 0.2),   # fear: low
                random.uniform(0.0, 0.1),   # sorrow: low
                random.uniform(0.0, 0.1)    # boredom: low
            ],
            'user_affect_change': lambda: random.uniform(0.2, 0.5),  # Feel better
            'relationship_change': lambda: {'trust': random.uniform(0.05, 0.15)},
            'interaction_type': 'support'
        },

        # Conflict scenarios (negative effect)
        {
            'templates': [
                "{agent} and I argued about {topic}",
                "{agent} criticized me for {topic}",
                "{agent} disappointed me by {topic}",
                "I'm upset with {agent} about {topic}"
            ],
            'agent_affect': lambda: [
                random.uniform(-0.6, -0.2),  # valence: negative
                random.uniform(0.5, 0.8),    # arousal: high
                random.uniform(0.2, 0.5),    # fear: moderate
                random.uniform(0.1, 0.3),    # sorrow: some
                random.uniform(0.0, 0.1)     # boredom: low
            ],
            'user_affect_change': lambda: random.uniform(-0.5, -0.2),  # Feel worse
            'relationship_change': lambda: {'trust': random.uniform(-0.2, -0.05)},
            'interaction_type': 'conflict'
        },

        # Empathic worry scenarios
        {
            'templates': [
                "{agent} seems stressed about {problem}",
                "{agent} is worried about {problem}",
                "{agent} told me they're struggling with {problem}",
                "I'm concerned about {agent} and {problem}"
            ],
            'agent_affect': lambda: [
                random.uniform(-0.4, -0.1),  # valence: somewhat negative
                random.uniform(0.4, 0.7),    # arousal: moderate-high
                random.uniform(0.3, 0.6),    # fear: moderate
                random.uniform(0.2, 0.4),    # sorrow: some
                random.uniform(0.0, 0.2)     # boredom: low
            ],
            'user_affect_change': lambda: random.uniform(-0.2, 0.0),  # Empathic distress
            'relationship_change': lambda: {'trust': random.uniform(0.0, 0.08)},  # Caring
            'interaction_type': 'conversation'
        },

        # Shared joy scenarios
        {
            'templates': [
                "{agent} got great news about {topic}!",
                "{agent} is so excited about {topic}",
                "{agent} just achieved {topic}",
                "I'm so happy for {agent} about {topic}"
            ],
            'agent_affect': lambda: [
                random.uniform(0.6, 0.9),    # valence: very positive
                random.uniform(0.6, 0.9),    # arousal: high (excited)
                random.uniform(0.0, 0.1),    # fear: none
                random.uniform(0.0, 0.0),    # sorrow: none
                random.uniform(0.0, 0.1)     # boredom: none
            ],
            'user_affect_change': lambda: random.uniform(0.3, 0.6),  # Shared joy
            'relationship_change': lambda: {'trust': random.uniform(0.02, 0.08)},
            'interaction_type': 'celebration'
        },

        # Neutral conversation scenarios
        {
            'templates': [
                "Talked to {agent} about {topic}",
                "{agent} and I discussed {topic}",
                "Had a conversation with {agent} about {topic}",
                "Caught up with {agent} about {topic}"
            ],
            'agent_affect': lambda: [
                random.uniform(-0.2, 0.3),   # valence: neutral to slightly positive
                random.uniform(0.2, 0.4),    # arousal: low-moderate
                random.uniform(0.0, 0.2),    # fear: low
                random.uniform(0.0, 0.2),    # sorrow: low
                random.uniform(0.1, 0.3)     # boredom: some
            ],
            'user_affect_change': lambda: random.uniform(-0.1, 0.1),  # Minimal change
            'relationship_change': lambda: {'trust': random.uniform(0.0, 0.03)},
            'interaction_type': 'conversation'
        }
    ]

    # Agent pool
    agent_names = [
        'Alice', 'Bob', 'Charlie', 'Diana', 'Eve',
        'Frank', 'Grace', 'Henry', 'Iris', 'Jack',
        'Mom', 'Dad', 'Sister', 'Brother', 'Partner'
    ]

    # Topics/problems
    problems = [
        'work', 'health', 'family issues', 'money problems', 'relationship',
        'career decisions', 'moving', 'loss', 'illness', 'stress'
    ]

    topics = [
        'politics', 'plans', 'the past', 'feelings', 'goals',
        'hobbies', 'travel', 'books', 'movies', 'food'
    ]

    # Generate examples
    dataset = []

    # Track relationship states across examples (for temporal consistency)
    relationship_states = {agent: {'trust': 0.5} for agent in agent_names}

    for i in tqdm(range(num_examples), desc="Generating examples"):
        # Pick scenario
        scenario = random.choice(scenarios)

        # Pick agent
        agent = random.choice(agent_names)

        # Generate text
        template = random.choice(scenario['templates'])
        text = template.format(
            agent=agent,
            problem=random.choice(problems),
            topic=random.choice(topics)
        )

        # Generate affects
        base_user_affect = np.array([
            random.uniform(-0.3, 0.3),   # Base valence
            random.uniform(0.3, 0.6),    # Base arousal
            random.uniform(0.1, 0.3),    # Base fear
            random.uniform(0.1, 0.3),    # Base sorrow
            random.uniform(0.1, 0.3)     # Base boredom
        ])

        agent_affect = np.array(scenario['agent_affect']())

        # Apply scenario effect
        affect_change = scenario['user_affect_change']()
        next_user_affect = base_user_affect.copy()
        next_user_affect[0] += affect_change  # Change valence
        next_user_affect = np.clip(next_user_affect, [-1, 0, 0, 0, 0], [1, 1, 1, 1, 1])

        # Update relationship
        trust_change = scenario['relationship_change']()['trust']
        relationship_states[agent]['trust'] += trust_change
        relationship_states[agent]['trust'] = np.clip(
            relationship_states[agent]['trust'], 0.0, 1.0
        )

        # Create example
        example = {
            'id': i,
            'conversation_text': text,
            'affect_sequence': base_user_affect.tolist(),
            'target_affect': next_user_affect.tolist(),
            'agents_mentioned': [agent],
            'agent_states': {
                agent: {
                    'affect': agent_affect.tolist(),
                    'confidence': random.uniform(0.7, 0.95)
                }
            },
            'relationships': {
                agent: {
                    'trust': relationship_states[agent]['trust'],
                    'attachment': random.choice(['secure', 'anxious', 'avoidant', 'fearful'])
                }
            },
            'social_context': {
                'present_agents': [agent] if random.random() > 0.3 else [],
                'topic': scenario['interaction_type'],
                'interaction_type': scenario['interaction_type']
            }
        }

        dataset.append(example)

    # Split into train/val/test
    random.shuffle(dataset)

    n_train = int(0.8 * len(dataset))
    n_val = int(0.1 * len(dataset))

    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train+n_val]
    test_data = dataset[n_train+n_val:]

    # Save
    print(f"\nSaving dataset...")
    print(f"  Train: {len(train_data):,} examples")
    print(f"  Val:   {len(val_data):,} examples")
    print(f"  Test:  {len(test_data):,} examples")

    with open(output_path / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(output_path / 'val.json', 'w') as f:
        json.dump(val_data, f, indent=2)

    with open(output_path / 'test.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    # Save metadata
    metadata = {
        'num_examples': num_examples,
        'num_agents': len(agent_names),
        'num_scenarios': len(scenarios),
        'splits': {
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data)
        },
        'generated_at': str(np.datetime64('now'))
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Dataset saved to {output_path}")
    print(f"✓ Total size: ~{len(json.dumps(dataset)) / 1024 / 1024:.1f} MB")

    return train_data, val_data, test_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic social conversation data')
    parser.add_argument('--test', action='store_true', help='Generate only 100 examples for testing')
    args = parser.parse_args()

    print("=" * 70)
    print("Consilience Phase 4: Synthetic Data Generation")
    print("=" * 70)

    # Generate dataset
    num_examples = 100 if args.test else 50000
    train, val, test = generate_synthetic_dataset(
        num_examples=num_examples,
        output_dir='training/data/synthetic'
    )

    print("\n" + "=" * 70)
    print("✓ Generation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Check data: cat training/data/synthetic/metadata.json")
    print("  2. Start pretraining: ./training/train.sh")
