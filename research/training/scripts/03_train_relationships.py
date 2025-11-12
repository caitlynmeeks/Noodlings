#!/usr/bin/env python3
"""
Relationship Model Pretraining Script

Trains the Relationship Model to recognize attachment styles, trust levels,
and communication patterns from phenomenal state pairs.

This is Stage 3 of the training pipeline.

Estimated time: 1-2 hours on M3 Ultra
"""

import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add project dirs to path (works from any location)
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent.parent
sys.path.insert(0, str(project_dir / 'noodlings'))
sys.path.insert(0, str(project_dir))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from noodlings.models.theory_of_mind import RelationshipModel


def load_data(data_path: str):
    """Load training data from JSON."""
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data):,} examples")
    return data


def affect_to_state(affect: list) -> mx.array:
    """Convert 5-D affect to 40-D phenomenal state."""
    state = np.zeros(40, dtype=np.float32)
    for i in range(5):
        state[i*8:(i+1)*8] = affect[i] + np.random.normal(0, 0.05, 8)
    return mx.array(state, dtype=mx.float32)


def attachment_to_idx(attachment: str) -> int:
    """Convert attachment style string to index."""
    styles = ['secure', 'anxious', 'avoidant', 'fearful']
    if attachment in styles:
        return styles.index(attachment)
    return 0  # Default to secure


def relationship_loss(
    predicted_trust: mx.array,
    target_trust: mx.array,
    attachment_logits: mx.array,
    target_attachment: mx.array
):
    """
    Loss function for relationship modeling.
    """
    # Trust regression loss
    trust_loss = ((predicted_trust - target_trust) ** 2).mean()

    # Attachment classification loss
    attachment_loss = nn.losses.cross_entropy(
        attachment_logits,
        target_attachment,
        reduction='mean'
    )

    total_loss = trust_loss + attachment_loss

    return total_loss, {
        'trust_loss': float(trust_loss.item()),
        'attachment_loss': float(attachment_loss.item()),
        'total_loss': float(total_loss.item())
    }


def train_epoch(model, optimizer, train_data, epoch, config):
    """Train for one epoch."""
    total_loss = 0.0
    total_trust = 0.0
    total_attach = 0.0
    n_examples = 0

    # Shuffle data
    indices = np.random.permutation(len(train_data))

    pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{config['epochs']}")

    for idx in pbar:
        example = train_data[idx]

        # Skip if no relationships
        if 'relationships' not in example or not example['relationships']:
            continue

        # Get self state
        self_affect = example['affect_sequence']
        if isinstance(self_affect, list) and len(self_affect) > 0:
            if isinstance(self_affect[0], list):
                self_affect = self_affect[0]  # Get first affect in sequence
        self_state = affect_to_state(self_affect)[None, :]

        # Get first agent and their relationship info
        agent_name = list(example['relationships'].keys())[0]
        rel_info = example['relationships'][agent_name]

        # Get agent state
        if 'agent_states' in example and agent_name in example['agent_states']:
            agent_affect = example['agent_states'][agent_name]['affect']
            agent_state = affect_to_state(agent_affect)[None, :]
        else:
            # Skip if no agent state
            continue

        # Target labels
        target_trust = mx.array([[rel_info['trust']]], dtype=mx.float32)
        target_attachment = mx.array([attachment_to_idx(rel_info['attachment'])], dtype=mx.int32)

        # Forward pass with gradient
        def loss_fn():
            rel_vec, attach_logits, trust, comm = model(self_state, agent_state)

            loss, metrics = relationship_loss(
                trust,
                target_trust,
                attach_logits,
                target_attachment
            )

            return loss, metrics

        (loss, metrics), grads = nn.value_and_grad(model, loss_fn)()

        # Clip gradients
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Accumulate metrics
        total_loss += metrics['total_loss']
        total_trust += metrics['trust_loss']
        total_attach += metrics['attachment_loss']
        n_examples += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'trust': f"{metrics['trust_loss']:.4f}",
            'attach': f"{metrics['attachment_loss']:.4f}"
        })

    return {
        'loss': total_loss / n_examples if n_examples > 0 else 0.0,
        'trust_loss': total_trust / n_examples if n_examples > 0 else 0.0,
        'attachment_loss': total_attach / n_examples if n_examples > 0 else 0.0
    }


def validate(model, val_data, config):
    """Validate model."""
    total_loss = 0.0
    total_trust = 0.0
    total_attach = 0.0
    n_valid = 0

    # Track attachment accuracy
    correct_attachments = 0
    total_attachments = 0

    for example in tqdm(val_data, desc="Validating"):
        if 'relationships' not in example or not example['relationships']:
            continue

        # Get states
        self_affect = example['affect_sequence']
        if isinstance(self_affect, list) and len(self_affect) > 0:
            if isinstance(self_affect[0], list):
                self_affect = self_affect[0]
        self_state = affect_to_state(self_affect)[None, :]

        agent_name = list(example['relationships'].keys())[0]
        rel_info = example['relationships'][agent_name]

        if 'agent_states' not in example or agent_name not in example['agent_states']:
            continue

        agent_affect = example['agent_states'][agent_name]['affect']
        agent_state = affect_to_state(agent_affect)[None, :]

        target_trust = mx.array([[rel_info['trust']]], dtype=mx.float32)
        target_attachment = mx.array([attachment_to_idx(rel_info['attachment'])], dtype=mx.int32)

        # Forward pass
        rel_vec, attach_logits, trust, comm = model(self_state, agent_state)

        loss, metrics = relationship_loss(
            trust,
            target_trust,
            attach_logits,
            target_attachment
        )

        total_loss += metrics['total_loss']
        total_trust += metrics['trust_loss']
        total_attach += metrics['attachment_loss']
        n_valid += 1

        # Check attachment accuracy
        predicted_attachment = int(mx.argmax(attach_logits[0]).item())
        target_attach_idx = int(target_attachment[0].item())
        if predicted_attachment == target_attach_idx:
            correct_attachments += 1
        total_attachments += 1

    attachment_accuracy = correct_attachments / total_attachments if total_attachments > 0 else 0.0

    return {
        'loss': total_loss / n_valid if n_valid > 0 else float('inf'),
        'trust_loss': total_trust / n_valid if n_valid > 0 else float('inf'),
        'attachment_loss': total_attach / n_valid if n_valid > 0 else float('inf'),
        'attachment_accuracy': attachment_accuracy
    }


def main():
    print("=" * 70)
    print("Relationship Model Pretraining")
    print("=" * 70)
    print()

    # Configuration
    config = {
        'learning_rate': 1e-3,
        'epochs': 30,
        'checkpoint_dir': 'training/checkpoints/relationships',
        'data_dir': 'training/data/synthetic'
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = load_data(f"{config['data_dir']}/train.json")
    val_data = load_data(f"{config['data_dir']}/val.json")

    # Create model
    print("Creating Relationship Model...")
    model = RelationshipModel(
        state_dim=40,
        relationship_dim=32
    )

    param_count = model.get_parameter_count()
    print(f"  ✓ Model created with {param_count['total']:,} parameters")
    print()

    # Create optimizer
    optimizer = optim.Adam(learning_rate=config['learning_rate'])

    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_trust': [],
        'val_trust': [],
        'attachment_accuracy': []
    }

    print(f"Starting training for {config['epochs']} epochs...")
    print("(This will take 1-2 hours)")
    print()

    start_time = time.time()

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, optimizer, train_data, epoch, config)

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_data, config)
        else:
            val_metrics = {
                'loss': 0.0,
                'trust_loss': 0.0,
                'attachment_loss': 0.0,
                'attachment_accuracy': 0.0
            }

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\nEpoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f} "
              f"(trust: {train_metrics['trust_loss']:.4f}, "
              f"attach: {train_metrics['attachment_loss']:.4f})")

        if (epoch + 1) % 5 == 0:
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Attachment Accuracy: {val_metrics['attachment_accuracy']:.2%}")

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_trust'].append(train_metrics['trust_loss'])
        history['val_trust'].append(val_metrics['trust_loss'])
        history['attachment_accuracy'].append(val_metrics['attachment_accuracy'])

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.npz"
            flat_params = tree_flatten(model.parameters(), destination={})
            mx.savez(str(checkpoint_path), **flat_params)
            print(f"  ✓ Saved checkpoint to {checkpoint_path}")

        # Save best model
        if (epoch + 1) % 5 == 0 and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = checkpoint_dir / "best.npz"
            flat_params = tree_flatten(model.parameters(), destination={})
            mx.savez(str(best_path), **flat_params)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

        print()

    total_time = time.time() - start_time

    # Save final model and history
    final_path = checkpoint_dir / "final.npz"
    flat_params = tree_flatten(model.parameters(), destination={})
    mx.savez(str(final_path), **flat_params)
    print(f"✓ Saved final model to {final_path}")

    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")

    print()
    print("=" * 70)
    print("Relationship Model Training Complete!")
    print("=" * 70)
    print(f"Total training time: {total_time/3600:.1f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final attachment accuracy: {history['attachment_accuracy'][-1]:.2%}")
    print()
    print("Next step: Train Full Phase 4 Model")
    print("  Run: python3 training/scripts/04_train_phase4_full.py")
    print()


if __name__ == '__main__':
    main()
