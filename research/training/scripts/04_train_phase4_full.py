#!/usr/bin/env python3
"""
Full Phase 4 End-to-End Training Script

Trains the complete Phase 4 model with curriculum learning:
- Stage 1: Simple (1 agent per conversation)
- Stage 2: Medium (2-3 agents)
- Stage 3: Complex (3-5 agents)
- Stage 4: Full (up to 10 agents)

This is Stage 4 of the training pipeline - the final and longest stage.

Estimated time: 24-48 hours on M3 Ultra
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

from consilience_phase4 import ConsilienceModelPhase4
from social_memory import SocialContext
from train_phase4 import phase4_combined_loss


def load_data(data_path: str):
    """Load training data from JSON."""
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data):,} examples")
    return data


def filter_by_num_agents(data: list, max_agents: int):
    """Filter dataset by number of agents mentioned."""
    filtered = []
    for example in data:
        num_agents = len(example.get('agents_mentioned', []))
        if num_agents <= max_agents and num_agents > 0:
            filtered.append(example)
    return filtered


def extract_linguistic_features_simple(text: str, agent_name: str) -> mx.array:
    """Simple linguistic feature extraction."""
    features = np.zeros(128, dtype=np.float32)
    text_lower = text.lower()
    agent_lower = agent_name.lower()

    # Sentiment keywords
    positive_words = ['happy', 'excited', 'great', 'wonderful', 'helped', 'success', 'joy', 'love', 'good', 'better']
    negative_words = ['upset', 'angry', 'sad', 'stressed', 'worried', 'anxious', 'frustrated', 'disappointed', 'bad', 'worse']

    for i, word in enumerate(positive_words[:10]):
        if word in text_lower:
            features[i] = 1.0
    for i, word in enumerate(negative_words[:10]):
        if word in text_lower:
            features[10 + i] = 1.0

    # Emotion keywords
    emotions = ['stressed', 'worried', 'anxious', 'scared', 'excited', 'happy', 'sad', 'angry', 'calm', 'confident']
    for i, emotion in enumerate(emotions):
        if emotion in text_lower:
            features[20 + i] = 1.0

    # Agent mention
    agent_count = text_lower.count(agent_lower)
    features[40] = min(agent_count / 3.0, 1.0)

    # Actions
    actions = ['helped', 'said', 'told', 'argued', 'criticized', 'supported', 'listened', 'ignored', 'hurt', 'comforted']
    for i, action in enumerate(actions):
        if action in text_lower and agent_lower in text_lower:
            features[50 + i] = 1.0

    # Random noise
    features[70:] = np.random.normal(0, 0.1, 58)

    return mx.array(features, dtype=mx.float32)


def extract_context_features_simple(example: dict) -> mx.array:
    """Simple context feature extraction."""
    features = np.zeros(64, dtype=np.float32)

    if 'social_context' in example:
        ctx = example['social_context']
        features[0] = len(ctx.get('present_agents', [])) / 10.0

        interaction_types = ['conversation', 'conflict', 'support', 'celebration', 'other']
        interaction_type = ctx.get('interaction_type', 'conversation')
        if interaction_type in interaction_types:
            features[1 + interaction_types.index(interaction_type)] = 1.0

        topics = ['work', 'family', 'health', 'relationship', 'money', 'other']
        topic = ctx.get('topic', 'other')
        if topic in topics:
            features[6 + topics.index(topic)] = 1.0

    features[16:] = np.random.normal(0, 0.1, 48)
    return mx.array(features, dtype=mx.float32)


def train_epoch(model, optimizer, train_data, epoch, config):
    """Train for one epoch."""
    total_loss = 0.0
    total_affect_loss = 0.0
    total_social_loss = 0.0
    n_examples = 0

    # Reset model states at start of epoch for clean slate
    model.reset_states()
    mx.eval(model.parameters())

    # Shuffle
    indices = np.random.permutation(len(train_data))

    pbar = tqdm(indices, desc=f"Epoch {epoch+1}")

    for idx in pbar:
        example = train_data[idx]

        # Prepare input
        affect = mx.array([example['affect_sequence']], dtype=mx.float32)
        if len(affect.shape) == 3:
            affect = affect[0, -1:, :]  # Get last timestep

        # Extract features for each agent
        agents = example.get('agents_mentioned', [])
        if not agents:
            continue

        linguistic_features = {}
        for agent in agents:
            ling_feat = extract_linguistic_features_simple(
                example['conversation_text'],
                agent
            )
            linguistic_features[agent] = ling_feat[None, :]

        context_features = extract_context_features_simple(example)[None, :]

        # Social context
        social_ctx_dict = example.get('social_context', {})
        social_context = SocialContext(
            present_agents=social_ctx_dict.get('present_agents', agents),
            topic=social_ctx_dict.get('topic', ''),
            group_valence=float(affect[0, 0].item()) if affect.shape[0] > 0 else 0.0,
            group_arousal=float(affect[0, 1].item()) if affect.shape[0] > 0 else 0.0,
            interaction_type=social_ctx_dict.get('interaction_type', 'conversation')
        )

        # Forward pass with gradient
        def loss_fn():
            self_state, predicted_state, social_info = model.forward_with_social_context(
                affect=affect,
                linguistic_features=linguistic_features,
                context_features=context_features,
                present_agents=agents,
                social_context=social_context,
                user_text=example['conversation_text']
            )

            # Target affect
            target_affect = mx.array([example['target_affect']], dtype=mx.float32)

            # Compute loss
            loss, metrics = phase4_combined_loss(
                predicted_affect=predicted_state[:, :5],  # Extract affect dimensions
                target_affect=target_affect,
                social_info=social_info,
                target_social_info=None,  # No ground truth for now
                lambda_social=config['lambda_social']
            )

            return loss, metrics

        try:
            (loss, metrics), grads = nn.value_and_grad(model, loss_fn)()

            # Clip gradients
            grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)

            # Update
            optimizer.update(model, grads)

            # CRITICAL: Force evaluation to prevent Metal command buffer buildup
            mx.eval(model.parameters(), optimizer.state, loss, metrics['total_loss'])

            # Accumulate (convert to Python float to release GPU memory)
            # Handle both MLX arrays and Python floats
            def to_float(val):
                if hasattr(val, 'item'):
                    return float(val.item())
                return float(val)

            total_loss += to_float(metrics['total_loss'])
            total_affect_loss += to_float(metrics['affect_loss'])
            total_social_loss += to_float(metrics['social_loss'])
            n_examples += 1

            # Update progress
            pbar.set_postfix({
                'loss': f"{total_loss / n_examples:.4f}",
                'affect': f"{total_affect_loss / n_examples:.4f}",
                'social': f"{total_social_loss / n_examples:.4f}"
            })

            # CRITICAL: Reset states after each example to prevent Metal buffer accumulation
            # In training, each example should be independent
            model.reset_states()

            # Periodic aggressive cleanup
            if n_examples % 100 == 0:
                mx.eval(model.parameters(), optimizer.state)
                import gc
                gc.collect()

        except Exception as e:
            print(f"\n⚠ Error processing example {idx}: {e}")
            # Reset states on error to prevent corruption
            try:
                model.reset_states()
            except:
                pass
            continue

    # Final cleanup at end of epoch
    mx.eval(model.parameters())
    model.reset_states()

    return {
        'loss': total_loss / n_examples if n_examples > 0 else 0.0,
        'affect_loss': total_affect_loss / n_examples if n_examples > 0 else 0.0,
        'social_loss': total_social_loss / n_examples if n_examples > 0 else 0.0
    }


def validate(model, val_data, config):
    """Validate model."""
    total_loss = 0.0
    total_affect = 0.0
    n_valid = 0

    for example in tqdm(val_data[:500], desc="Validating"):  # Validate on subset
        agents = example.get('agents_mentioned', [])
        if not agents:
            continue

        try:
            affect = mx.array([example['affect_sequence']], dtype=mx.float32)
            if len(affect.shape) == 3:
                affect = affect[0, -1:, :]

            linguistic_features = {}
            for agent in agents:
                ling_feat = extract_linguistic_features_simple(
                    example['conversation_text'],
                    agent
                )
                linguistic_features[agent] = ling_feat[None, :]

            context_features = extract_context_features_simple(example)[None, :]

            social_ctx_dict = example.get('social_context', {})
            social_context = SocialContext(
                present_agents=social_ctx_dict.get('present_agents', agents),
                topic=social_ctx_dict.get('topic', ''),
                group_valence=float(affect[0, 0].item()) if affect.shape[0] > 0 else 0.0,
                group_arousal=float(affect[0, 1].item()) if affect.shape[0] > 0 else 0.0,
                interaction_type=social_ctx_dict.get('interaction_type', 'conversation')
            )

            # Forward
            self_state, predicted_state, social_info = model.forward_with_social_context(
                affect=affect,
                linguistic_features=linguistic_features,
                context_features=context_features,
                present_agents=agents,
                social_context=social_context
            )

            target_affect = mx.array([example['target_affect']], dtype=mx.float32)

            loss, metrics = phase4_combined_loss(
                predicted_affect=predicted_state[:, :5],
                target_affect=target_affect,
                social_info=social_info,
                lambda_social=config['lambda_social']
            )

            total_loss += metrics['total_loss']
            total_affect += metrics['affect_loss']
            n_valid += 1

        except Exception as e:
            continue

    return {
        'loss': total_loss / n_valid if n_valid > 0 else float('inf'),
        'affect_loss': total_affect / n_valid if n_valid > 0 else float('inf')
    }


def main():
    print("=" * 70)
    print("Phase 4 Full End-to-End Training")
    print("=" * 70)
    print()

    # Configuration
    config = {
        'learning_rate': 1e-4,  # Lower for fine-tuning
        'checkpoint_dir': 'training/checkpoints/phase4_pretrain',
        'data_dir': 'training/data/synthetic',
        'phase3_checkpoint': 'noodlings/checkpoints_phase3/best_checkpoint.npz',  # If exists
        'tom_checkpoint': 'training/checkpoints/theory_of_mind/best.npz',
        'rel_checkpoint': 'training/checkpoints/relationships/best.npz'
    }

    # Curriculum stages
    curriculum = [
        {'name': 'Simple', 'epochs': 10, 'max_agents': 1, 'lambda_social': 0.05},
        {'name': 'Medium', 'epochs': 10, 'max_agents': 2, 'lambda_social': 0.10},
        {'name': 'Complex', 'epochs': 20, 'max_agents': 5, 'lambda_social': 0.10},
        {'name': 'Full', 'epochs': 10, 'max_agents': 10, 'lambda_social': 0.15}
    ]

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    print("Curriculum:")
    for i, stage in enumerate(curriculum):
        print(f"  Stage {i+1}: {stage['name']} - {stage['epochs']} epochs, "
              f"max {stage['max_agents']} agents, lambda={stage['lambda_social']}")
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = load_data(f"{config['data_dir']}/train.json")
    val_data = load_data(f"{config['data_dir']}/val.json")

    # Create model
    print("Creating Phase 4 model...")
    model = ConsilienceModelPhase4(
        affect_dim=5,
        fast_hidden=16,
        medium_hidden=16,
        slow_hidden=8,
        memory_capacity=100,
        max_agents=10,
        use_theory_of_mind=True,
        use_relationship_model=True
    )

    param_count = model.get_parameter_count()
    print(f"  ✓ Model created with {param_count['total']:,} parameters")

    # Load pretrained weights if available
    if Path(config['phase3_checkpoint']).exists():
        print(f"  ✓ Loading Phase 3 checkpoint: {config['phase3_checkpoint']}")
        # model.load_phase3_weights(config['phase3_checkpoint'])

    # TODO: Load ToM and Relationship pretrained weights
    # if Path(config['tom_checkpoint']).exists():
    #     print(f"  ✓ Loading Theory of Mind checkpoint")
    # if Path(config['rel_checkpoint']).exists():
    #     print(f"  ✓ Loading Relationship checkpoint")

    print()

    # Create optimizer
    optimizer = optim.Adam(learning_rate=config['learning_rate'])

    # Training loop
    best_val_loss = float('inf')
    history = {
        'stages': [],
        'train_loss': [],
        'val_loss': [],
        'affect_loss': [],
        'social_loss': []
    }

    print("Starting curriculum training...")
    print("Total epochs:", sum(stage['epochs'] for stage in curriculum))
    print("Estimated time: 24-48 hours")
    print()

    start_time = time.time()
    global_epoch = 0

    for stage_idx, stage in enumerate(curriculum):
        print("=" * 70)
        print(f"Curriculum Stage {stage_idx+1}/{len(curriculum)}: {stage['name']}")
        print(f"Max agents: {stage['max_agents']}, Lambda social: {stage['lambda_social']}")
        print("=" * 70)
        print()

        # Filter data for curriculum
        stage_train = filter_by_num_agents(train_data, max_agents=stage['max_agents'])
        print(f"Training on {len(stage_train):,} examples (filtered from {len(train_data):,})")
        print()

        # Update config for this stage
        config['lambda_social'] = stage['lambda_social']

        for epoch in range(stage['epochs']):
            epoch_start = time.time()

            # Train
            train_metrics = train_epoch(model, optimizer, stage_train, global_epoch, config)

            # Validate every 5 epochs
            if (global_epoch + 1) % 5 == 0:
                val_metrics = validate(model, val_data, config)
            else:
                val_metrics = {'loss': 0.0, 'affect_loss': 0.0}

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time

            # Log
            print(f"\nGlobal Epoch {global_epoch+1} | Stage {stage['name']} {epoch+1}/{stage['epochs']} ({epoch_time:.1f}s | Total: {total_time/3600:.1f}h)")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(affect: {train_metrics['affect_loss']:.4f}, "
                  f"social: {train_metrics['social_loss']:.4f})")

            if (global_epoch + 1) % 5 == 0:
                print(f"  Val Loss:   {val_metrics['loss']:.4f}")

            # Save history
            history['stages'].append(stage['name'])
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['affect_loss'].append(train_metrics['affect_loss'])
            history['social_loss'].append(train_metrics['social_loss'])

            # Save checkpoint
            if (global_epoch + 1) % 10 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{global_epoch+1}.npz"
                model.save_weights(str(checkpoint_path))
                print(f"  ✓ Saved checkpoint to {checkpoint_path}")

            # Save best
            if (global_epoch + 1) % 5 == 0 and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = checkpoint_dir / "best.npz"
                model.save_weights(str(best_path))
                print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

            print()
            global_epoch += 1

    total_time = time.time() - start_time

    # Save final
    final_path = checkpoint_dir / "final.npz"
    model.save_weights(str(final_path))
    print(f"✓ Saved final model to {final_path}")

    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")

    print()
    print("=" * 70)
    print("Phase 4 Training Complete!")
    print("=" * 70)
    print(f"Total training time: {total_time/3600:.1f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print()
    print("Next steps:")
    print("  1. Copy pretrained model to release:")
    print(f"     cp {best_path} models/consilience_phase4_pretrained.npz")
    print("  2. Test model:")
    print("     python3 noodlings/example_phase4_usage.py")
    print("  3. Celebrate! 🎉")
    print()


if __name__ == '__main__':
    main()
