#!/usr/bin/env python3
"""
Theory of Mind Pretraining Script

Trains the Theory of Mind module to infer others' mental states from
linguistic cues, context, and history.

This is Stage 2 of the training pipeline.

Estimated time: 2-3 hours on M3 Ultra
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

from noodlings.models.theory_of_mind import TheoryOfMindModule


def load_data(data_path: str):
    """Load training data from JSON."""
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data):,} examples")
    return data


def extract_linguistic_features_simple(text: str, agent_name: str) -> mx.array:
    """
    Simple linguistic feature extraction.

    For now, we'll use a simple approach based on keywords.
    In production, this would use proper NLP.
    """
    features = np.zeros(128, dtype=np.float32)

    text_lower = text.lower()
    agent_lower = agent_name.lower()

    # Sentiment keywords (dimensions 0-19)
    positive_words = ['happy', 'excited', 'great', 'wonderful', 'helped', 'success', 'joy', 'love', 'good', 'better']
    negative_words = ['upset', 'angry', 'sad', 'stressed', 'worried', 'anxious', 'frustrated', 'disappointed', 'bad', 'worse']

    for i, word in enumerate(positive_words[:10]):
        if word in text_lower:
            features[i] = 1.0

    for i, word in enumerate(negative_words[:10]):
        if word in text_lower:
            features[10 + i] = 1.0

    # Emotion keywords (dimensions 20-39)
    emotions = ['stressed', 'worried', 'anxious', 'scared', 'excited', 'happy', 'sad', 'angry', 'calm', 'confident']
    for i, emotion in enumerate(emotions):
        if emotion in text_lower:
            features[20 + i] = 1.0

    # Agent mention intensity (dimensions 40-49)
    agent_count = text_lower.count(agent_lower)
    features[40] = min(agent_count / 3.0, 1.0)  # Normalize

    # Action verbs about agent (dimensions 50-69)
    actions = ['helped', 'said', 'told', 'argued', 'criticized', 'supported', 'listened', 'ignored', 'hurt', 'comforted']
    for i, action in enumerate(actions):
        if action in text_lower and agent_lower in text_lower:
            features[50 + i] = 1.0

    # Add some random noise to remaining dimensions (for now)
    features[70:] = np.random.normal(0, 0.1, 58)

    return mx.array(features, dtype=mx.float32)


def extract_context_features_simple(example: dict) -> mx.array:
    """
    Simple context feature extraction.
    """
    features = np.zeros(64, dtype=np.float32)

    # Social context features
    if 'social_context' in example:
        ctx = example['social_context']

        # Number of agents (dimension 0)
        features[0] = len(ctx.get('present_agents', [])) / 10.0

        # Interaction type (dimensions 1-5, one-hot)
        interaction_types = ['conversation', 'conflict', 'support', 'celebration', 'other']
        interaction_type = ctx.get('interaction_type', 'conversation')
        if interaction_type in interaction_types:
            features[1 + interaction_types.index(interaction_type)] = 1.0

        # Topic encoding (dimensions 6-15)
        topics = ['work', 'family', 'health', 'relationship', 'money', 'other']
        topic = ctx.get('topic', 'other')
        if topic in topics:
            features[6 + topics.index(topic)] = 1.0

    # Add random context features (for now)
    features[16:] = np.random.normal(0, 0.1, 48)

    return mx.array(features, dtype=mx.float32)


def affect_to_state(affect: list) -> mx.array:
    """
    Convert 5-D affect to 40-D phenomenal state.

    This is a simple expansion - in reality, the state would come from
    the Phase 1-3 encoder. For pretraining ToM, we'll use a simple mapping.
    """
    state = np.zeros(40, dtype=np.float32)

    # Expand affect to state
    for i in range(5):
        # Repeat each affect dimension 8 times with slight noise
        state[i*8:(i+1)*8] = affect[i] + np.random.normal(0, 0.05, 8)

    return mx.array(state, dtype=mx.float32)


def theory_of_mind_loss(
    predicted_state: mx.array,
    target_state: mx.array,
    confidence: mx.array,
    mean: mx.array,
    logvar: mx.array,
    beta_kl: float = 0.001
):
    """
    Loss function for Theory of Mind training.
    """
    # State reconstruction loss
    recon_loss = ((predicted_state - target_state) ** 2).mean()

    # KL divergence
    kl_loss = -0.5 * (1 + logvar - mean ** 2 - mx.exp(logvar)).mean()

    # Confidence calibration
    prediction_error = ((predicted_state - target_state) ** 2).mean(axis=-1, keepdims=True)
    confidence_target = mx.exp(-prediction_error)
    confidence_loss = ((confidence - confidence_target) ** 2).mean()

    # Total loss
    total_loss = recon_loss + beta_kl * kl_loss + 0.1 * confidence_loss

    return total_loss, {
        'recon_loss': float(recon_loss.item()),
        'kl_loss': float(kl_loss.item()),
        'confidence_loss': float(confidence_loss.item()),
        'total_loss': float(total_loss.item())
    }


def train_epoch(model, optimizer, train_data, epoch, config):
    """Train for one epoch."""
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_conf = 0.0

    # Shuffle data
    indices = np.random.permutation(len(train_data))

    pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{config['epochs']}")

    for idx in pbar:
        example = train_data[idx]

        # Skip if no agent states
        if 'agent_states' not in example or not example['agent_states']:
            continue

        # Get first agent
        agent_name = list(example['agent_states'].keys())[0]
        agent_info = example['agent_states'][agent_name]

        # Extract features
        ling_feat = extract_linguistic_features_simple(
            example['conversation_text'],
            agent_name
        )[None, :]  # Add batch dimension

        context_feat = extract_context_features_simple(example)[None, :]

        # Target state
        target_affect = agent_info['affect']
        target_state = affect_to_state(target_affect)[None, :]

        # Forward pass with gradient
        def loss_fn():
            inferred_state, confidence, mean, logvar, uncertainty = model(
                ling_feat,
                context_feat,
                None  # No history for first pass
            )

            loss, metrics = theory_of_mind_loss(
                inferred_state,
                target_state,
                confidence,
                mean,
                logvar,
                beta_kl=config['beta_kl']
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
        total_recon += metrics['recon_loss']
        total_kl += metrics['kl_loss']
        total_conf += metrics['confidence_loss']

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'recon': f"{metrics['recon_loss']:.4f}",
            'kl': f"{metrics['kl_loss']:.4f}"
        })

    n = len(train_data)
    return {
        'loss': total_loss / n,
        'recon_loss': total_recon / n,
        'kl_loss': total_kl / n,
        'confidence_loss': total_conf / n
    }


def validate(model, val_data, config):
    """Validate model."""
    total_loss = 0.0
    total_recon = 0.0
    n_valid = 0

    for example in tqdm(val_data, desc="Validating"):
        if 'agent_states' not in example or not example['agent_states']:
            continue

        agent_name = list(example['agent_states'].keys())[0]
        agent_info = example['agent_states'][agent_name]

        ling_feat = extract_linguistic_features_simple(
            example['conversation_text'],
            agent_name
        )[None, :]

        context_feat = extract_context_features_simple(example)[None, :]
        target_state = affect_to_state(agent_info['affect'])[None, :]

        # Forward pass (no gradients)
        inferred_state, confidence, mean, logvar, uncertainty = model(
            ling_feat,
            context_feat,
            None
        )

        loss, metrics = theory_of_mind_loss(
            inferred_state,
            target_state,
            confidence,
            mean,
            logvar,
            beta_kl=config['beta_kl']
        )

        total_loss += metrics['total_loss']
        total_recon += metrics['recon_loss']
        n_valid += 1

    return {
        'loss': total_loss / n_valid if n_valid > 0 else float('inf'),
        'recon_loss': total_recon / n_valid if n_valid > 0 else float('inf')
    }


def main():
    print("=" * 70)
    print("Theory of Mind Pretraining")
    print("=" * 70)
    print()

    # Configuration
    config = {
        'learning_rate': 1e-3,
        'beta_kl': 0.001,
        'epochs': 50,
        'checkpoint_dir': 'training/checkpoints/theory_of_mind',
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
    print("Creating Theory of Mind module...")
    model = TheoryOfMindModule(
        linguistic_dim=128,
        context_dim=64,
        history_dim=40,
        state_dim=40,
        dropout=0.1
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
        'train_recon': [],
        'val_recon': []
    }

    print(f"Starting training for {config['epochs']} epochs...")
    print("(This will take 2-3 hours)")
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
            val_metrics = {'loss': 0.0, 'recon_loss': 0.0}

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\nEpoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f} "
              f"(recon: {train_metrics['recon_loss']:.4f}, "
              f"kl: {train_metrics['kl_loss']:.4f})")

        if (epoch + 1) % 5 == 0:
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_recon'].append(train_metrics['recon_loss'])
        history['val_recon'].append(val_metrics['recon_loss'])

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
    print("Theory of Mind Training Complete!")
    print("=" * 70)
    print(f"Total training time: {total_time/3600:.1f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final training loss: {train_metrics['loss']:.4f}")
    print()
    print("Next step: Train Relationship Model")
    print("  Run: python3 training/scripts/03_train_relationships.py")
    print()


if __name__ == '__main__':
    main()
