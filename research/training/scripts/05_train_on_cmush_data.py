#!/usr/bin/env python3
"""
Train Consilience Phase 4 on real cMUSH conversation data.

This script fine-tunes a pre-trained (or randomly initialized) Phase 4 model
on actual human-agent interactions collected from cMUSH.

Usage:
    python3 05_train_on_cmush_data.py \\
        --data ../data/cmush_real/exported_dataset.json \\
        --checkpoint ../checkpoints_phase4/best_checkpoint.npz \\
        --output ../checkpoints_phase4/cmush_finetuned.npz \\
        --epochs 10

Author: Consilience Project
Date: October 2025
"""

import sys
import os
from pathlib import Path

# Add project dirs to path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent.parent
sys.path.insert(0, str(project_dir / 'noodlings'))
sys.path.insert(0, str(project_dir))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
import argparse
from tqdm import tqdm
from datetime import datetime

from consilience_phase4 import ConsilienceModelPhase4


def load_cmush_dataset(data_path: str):
    """
    Load training dataset exported from cMUSH.

    Args:
        data_path: Path to exported_dataset.json

    Returns:
        Dict with dataset info and sequences
    """
    print(f"Loading dataset from {data_path}...")

    with open(data_path, 'r') as f:
        dataset = json.load(f)

    print(f"Dataset info:")
    print(f"  - Source: {dataset['dataset_info']['source']}")
    print(f"  - Created: {dataset['dataset_info']['created']}")
    print(f"  - Sequences: {dataset['dataset_info']['num_sequences']}")
    print(f"  - Total turns: {dataset['dataset_info']['total_turns']}")

    return dataset


def prepare_sequences(sequences, max_seq_len=100):
    """
    Prepare sequences for training.

    Args:
        sequences: List of conversation sequences
        max_seq_len: Maximum sequence length for BPTT

    Returns:
        List of (affect_sequence, agent_id) tuples
    """
    prepared = []

    for seq in sequences:
        affect_seq = np.array(seq['affect_sequence'], dtype=np.float32)

        # Split long sequences
        seq_len = len(affect_seq)
        if seq_len > max_seq_len:
            # Split into overlapping windows
            for start_idx in range(0, seq_len, max_seq_len - 20):  # 20 overlap
                end_idx = min(start_idx + max_seq_len, seq_len)
                window = affect_seq[start_idx:end_idx]

                if len(window) >= 2:  # Min sequence length (need at least 2 for prediction)
                    prepared.append((
                        mx.array(window),
                        seq['agent_id']
                    ))
        else:
            if seq_len >= 2:  # Min 2 timesteps for next-step prediction
                prepared.append((
                    mx.array(affect_seq),
                    seq['agent_id']
                ))

    print(f"Prepared {len(prepared)} training sequences")
    return prepared


def create_loss_fn(model):
    """
    Create loss function for training.

    Predicts next phenomenal state from current affect.

    Args:
        model: ConsiliencePhase4Model

    Returns:
        Loss function
    """
    def loss_fn(affect_sequence, present_agents):
        """
        Compute prediction loss over sequence.

        Args:
            affect_sequence: (seq_len, 5) affect vectors
            present_agents: List of agent IDs

        Returns:
            Mean prediction loss
        """
        seq_len = affect_sequence.shape[0]

        # Initialize states
        model.reset_states()

        total_loss = 0.0

        # Process sequence
        for t in range(seq_len - 1):  # Predict next step
            # Current affect
            affect_t = affect_sequence[t:t+1]  # (1, 5)

            # Forward pass
            result = model(
                affect_t,
                present_agents=present_agents
            )

            # Get next actual state
            affect_next = affect_sequence[t+1:t+2]
            result_next = model(
                affect_next,
                present_agents=present_agents
            )

            h_fast_actual = result_next['h_fast']

            # Prediction from current step
            h_fast_pred = result['h_fast_pred']

            # MSE loss
            loss_t = mx.mean((h_fast_actual - h_fast_pred) ** 2)
            total_loss = total_loss + loss_t

        # Average over sequence
        return total_loss / (seq_len - 1)

    return loss_fn


def train_epoch(model, optimizer, sequences, batch_size=1):
    """
    Train for one epoch.

    Args:
        model: ConsiliencePhase4Model
        optimizer: MLX optimizer
        sequences: List of training sequences
        batch_size: Batch size (currently only supports 1)

    Returns:
        Average loss for epoch
    """
    total_loss = 0.0
    num_batches = 0

    # Shuffle sequences
    indices = np.random.permutation(len(sequences))

    for idx in tqdm(indices, desc="Training"):
        affect_seq, agent_id = sequences[idx]

        present_agents = [agent_id]

        # Create loss and grad function
        loss_and_grad_fn = nn.value_and_grad(model, create_loss_fn(model))

        # Compute loss and gradients
        loss, grads = loss_and_grad_fn(affect_seq, present_agents)

        # Clip gradients
        grads = clip_gradients(grads, max_norm=1.0)

        # Update parameters
        optimizer.update(model, grads)

        # Evaluate updated parameters
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def clip_gradients(grads, max_norm=1.0):
    """
    Clip gradients by global norm.

    Args:
        grads: Gradient dict
        max_norm: Maximum gradient norm

    Returns:
        Clipped gradients
    """
    # Compute global norm
    total_norm = 0.0
    for grad in grads.values():
        if grad is not None:
            total_norm += mx.sum(grad ** 2)

    total_norm = mx.sqrt(total_norm).item()

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        grads = {k: v * clip_coef if v is not None else v
                 for k, v in grads.items()}

    return grads


def evaluate(model, sequences, num_samples=100):
    """
    Evaluate model on validation set.

    Args:
        model: ConsiliencePhase4Model
        sequences: List of validation sequences
        num_samples: Number of sequences to evaluate

    Returns:
        Average loss
    """
    total_loss = 0.0
    num_samples = min(num_samples, len(sequences))

    indices = np.random.choice(len(sequences), num_samples, replace=False)

    loss_fn = create_loss_fn(model)

    for idx in tqdm(indices, desc="Evaluating"):
        affect_seq, agent_id = sequences[idx]
        present_agents = [agent_id]

        loss = loss_fn(affect_seq, present_agents)
        total_loss += loss.item()

    return total_loss / num_samples


def main():
    parser = argparse.ArgumentParser(
        description="Train Consilience on cMUSH conversation data"
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to exported dataset JSON'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to initial checkpoint (optional)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save trained model'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for fine-tuning'
    )

    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation set fraction'
    )

    args = parser.parse_args()

    print("="*60)
    print("Consilience Phase 4 - cMUSH Fine-Tuning")
    print("="*60)

    # Load dataset
    dataset = load_cmush_dataset(args.data)
    sequences = prepare_sequences(dataset['sequences'])

    # Split train/val
    num_val = int(len(sequences) * args.val_split)
    val_sequences = sequences[:num_val]
    train_sequences = sequences[num_val:]

    print(f"\nSplit:")
    print(f"  - Training: {len(train_sequences)} sequences")
    print(f"  - Validation: {len(val_sequences)} sequences")

    # Create model
    print("\nInitializing model...")
    model = ConsilienceModelPhase4()

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_weights(args.checkpoint)
    else:
        print("Training from random initialization")

    # Create optimizer
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    print(f"\nTraining configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Gradient clipping: max_norm=1.0")

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\nStarting training...")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss = train_epoch(
            model,
            optimizer,
            train_sequences,
            batch_size=1
        )

        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        if val_sequences:
            val_loss = evaluate(model, val_sequences, num_samples=min(50, len(val_sequences)))
            val_losses.append(val_loss)
            print(f"Val loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"✓ New best validation loss! Saving checkpoint...")

                # Save model
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                model.save_weights(str(output_path))

                # Save training info
                info_path = output_path.with_suffix('.json')
                with open(info_path, 'w') as f:
                    json.dump({
                        'epoch': epoch,
                        'train_loss': float(train_loss),
                        'val_loss': float(val_loss),
                        'best_val_loss': float(best_val_loss),
                        'learning_rate': args.learning_rate,
                        'trained_on': datetime.now().isoformat(),
                        'dataset': args.data,
                        'num_sequences': len(train_sequences)
                    }, f, indent=2)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {args.output}")

    # Plot training curves (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        plot_path = Path(args.output).with_suffix('.png')
        plt.savefig(plot_path)
        print(f"Training curve saved to: {plot_path}")
    except ImportError:
        print("(matplotlib not available, skipping training curve)")


if __name__ == "__main__":
    main()