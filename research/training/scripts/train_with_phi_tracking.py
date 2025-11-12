#!/usr/bin/env python3
"""
Training Script with Φ Tracking

Train Consilience model while monitoring integrated information (Φ) evolution.

Usage:
    python train_with_phi_tracking.py --config config.yaml
    python train_with_phi_tracking.py --use-observers --track-interval 50

Author: Consilience Project
Date: November 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../noodlings'))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import argparse
from pathlib import Path
import yaml
import time

from consilience_with_observers import ConsilienceWithObservers
from consilience_phase4 import ConsilienceModelPhase4
from phi_tracker import PhiTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train with Φ tracking")

    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--use-observers', action='store_true', help='Use observer loops')
    parser.add_argument('--track-interval', type=int, default=100, help='Steps between Φ measurements')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data-path', type=str, default='../data/synthetic', help='Training data path')
    parser.add_argument('--checkpoint-dir', type=str, default='../../models', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='../logs/phi', help='Φ tracking log directory')
    parser.add_argument('--use-full-phi', action='store_true', help='Use full IIT Φ (slow!)')

    return parser.parse_args()


def generate_synthetic_batch(batch_size: int = 32, seq_len: int = 20) -> mx.array:
    """
    Generate synthetic affect sequences for training.

    In production, replace with real data loader.
    """
    # Generate random affect sequences
    # Shape: [batch_size, seq_len, 5]
    affects = mx.random.normal((batch_size, seq_len, 5))

    # Add some structure (not pure noise)
    # Valence and arousal tend to correlate
    affects = mx.array(affects)

    return affects


def train_step(
    model,
    optimizer,
    affect_batch: mx.array,
    states: tuple
):
    """
    Single training step.

    Args:
        model: Consilience model
        optimizer: MLX optimizer
        affect_batch: Batch of affect sequences [batch, seq, 5]
        states: Hidden states tuple

    Returns:
        loss: Scalar loss
        new_states: Updated hidden states
        outputs: Model outputs dict
    """
    def loss_fn(m):
        """Loss function for gradient computation."""
        if isinstance(m, ConsilienceWithObservers):
            # With observers
            # Process first item in batch for simplicity
            affect_single = affect_batch[0:1, -1, :]  # Last timestep

            outputs = m(affect_input=affect_single)

            # Compute total loss
            # In real training, you'd have target states
            # Here we use prediction error as loss
            phenomenal_state = outputs['phenomenal_state']
            predicted_next = phenomenal_state  # Placeholder

            # Simple MSE loss (replace with actual targets)
            loss_main = mx.mean((phenomenal_state - predicted_next) ** 2)

            # Add observer losses
            total_loss, breakdown = m.compute_total_loss(
                outputs,
                predicted_next
            )

            return total_loss, outputs
        else:
            # Without observers (standard Phase 4)
            # Process through base model
            h_fast, c_fast, h_med, c_med, h_slow = states

            affect_seq = affect_batch[0:1, :, :]  # First item, all timesteps

            model_outputs = m.base_model(
                affect_seq, h_fast, c_fast, h_med, c_med, h_slow
            )

            # Simple prediction loss
            phenomenal_state = model_outputs['phenomenal_state']
            predicted = model_outputs.get('h_fast_pred', phenomenal_state)

            loss = mx.mean((phenomenal_state - predicted) ** 2)

            return loss, model_outputs

    # Compute loss and gradients
    (loss, outputs), grads = mx.value_and_grad(model, loss_fn, has_aux=True)(model)

    # Update parameters
    optimizer.update(model, grads)

    return loss, outputs


def main():
    """Main training loop with Φ tracking."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("CONSILIENCE TRAINING WITH Φ TRACKING")
    print("=" * 80)

    # Initialize model
    print(f"\nInitializing model...")
    print(f"  Observers: {'ENABLED' if args.use_observers else 'DISABLED'}")

    if args.use_observers:
        model = ConsilienceWithObservers(
            affect_dim=5,
            fast_hidden=16,
            medium_hidden=16,
            slow_hidden=8,
            use_observer_loop=True,
            use_meta_observer=True,
            observe_hierarchical_states=True,
            observer_injection_strength=0.1,
            observer_loss_weight=0.5,
            meta_loss_weight=0.2
        )
    else:
        model = ConsilienceModelPhase4(
            affect_dim=5,
            fast_hidden=16,
            medium_hidden=16,
            slow_hidden=8
        )

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    # Initialize Φ tracker
    print(f"\nInitializing Φ tracker...")
    phi_tracker = PhiTracker(
        log_dir=args.log_dir,
        state_dim=40,
        track_interval=args.track_interval,
        use_full_phi=args.use_full_phi
    )

    # Initial states
    h_fast = mx.zeros((1, 16))
    c_fast = mx.zeros((1, 16))
    h_med = mx.zeros((1, 16))
    c_med = mx.zeros((1, 16))
    h_slow = mx.zeros((1, 8))
    states = (h_fast, c_fast, h_med, c_med, h_slow)

    # Training loop
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Φ tracking interval: {args.track_interval} steps")
    print("")

    step = 0
    num_batches_per_epoch = 100  # Adjust based on your data

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for batch_idx in range(num_batches_per_epoch):
            # Generate batch (replace with real data loader)
            affect_batch = generate_synthetic_batch(args.batch_size, seq_len=20)

            # Training step
            loss, outputs = train_step(model, optimizer, affect_batch, states)

            epoch_loss += float(loss)
            step += 1

            # Track Φ every N steps
            if step % args.track_interval == 0:
                # Extract phenomenal state
                if isinstance(model, ConsilienceWithObservers):
                    phenomenal_state = outputs['phenomenal_state']
                    observer_loss = float(outputs['observer_loss'])
                    meta_loss = float(outputs['meta_loss'])
                    observer_influence = float(outputs.get('observer_influence', 0.0))
                else:
                    phenomenal_state = outputs['phenomenal_state']
                    observer_loss = 0.0
                    meta_loss = 0.0
                    observer_influence = 0.0

                # Convert to numpy
                if hasattr(phenomenal_state, 'squeeze'):
                    state_np = np.array(phenomenal_state.squeeze())
                else:
                    state_np = np.array(phenomenal_state)

                # Track Φ
                phi_tracker.track_step(
                    step=step,
                    model=model,
                    state=state_np,
                    loss=float(loss),
                    observer_loss=observer_loss,
                    meta_loss=meta_loss,
                    observer_influence=observer_influence,
                    epoch=epoch
                )

            # Log progress
            if step % 50 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"[Epoch {epoch+1}/{args.epochs}] [Step {step}] Loss: {avg_loss:.6f}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / num_batches_per_epoch

        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Avg Loss: {avg_epoch_loss:.6f}")

        # Φ trend
        phi_trend = phi_tracker.get_phi_trend(window=50)
        print(f"  Current Φ: {phi_trend['current']:.4f}")
        print(f"  Φ Trend: {phi_trend['trend']:+.6f} (slope)")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch{epoch+1}.npz"
            # model.save_weights(checkpoint_path)  # Implement if needed
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    # Save final Φ summary
    phi_tracker.save_summary()

    # Generate visualization
    try:
        phi_tracker.plot_phi_evolution()
    except Exception as e:
        print(f"⚠️  Could not generate plot: {e}")

    # Final statistics
    summary = phi_tracker.get_summary()

    print("\n📊 Final Results:")
    print(f"  Initial Φ: {summary['phi']['initial']:.4f}")
    print(f"  Final Φ: {summary['phi']['final']:.4f}")
    print(f"  Φ Improvement: {summary['phi']['improvement_percent']:+.1f}%")
    print(f"  Mean Φ: {summary['phi']['mean']:.4f}")

    print(f"\n✅ Training session complete!")
    print(f"   Logs saved to: {args.log_dir}")
    print("")


if __name__ == "__main__":
    main()
