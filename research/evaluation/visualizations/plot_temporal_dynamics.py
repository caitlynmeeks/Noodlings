"""
Temporal Dynamics Visualization

Plots fast/medium/slow layer states over conversation time.
Shows hierarchical timescale separation visually.

Author: Noodlings Project
Date: November 2025
"""

import sys
sys.path.insert(0, '../..')

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')


def plot_hierarchical_dynamics(
    model,
    conversation: List[mx.array],
    output_path: str = 'evaluation/visualizations/figures/hierarchical_dynamics.png',
    show_surprise: bool = True
):
    """
    Plot fast/medium/slow layer states over conversation.
    
    Args:
        model: Noodlings model instance
        conversation: Single conversation (affect sequence)
        output_path: Where to save the figure
        show_surprise: Whether to plot surprise on separate axis
    """
    logger.info("Processing conversation through model...")
    
    model.reset_states()
    
    fast_states = []
    medium_states = []
    slow_states = []
    surprises = []
    valences = []
    
    for affect in conversation:
        if affect.ndim == 1:
            affect = affect[None, :]
        
        # Process timestep
        _, surprise = model(affect)
        
        # Record states
        fast_states.append(np.array(model.h_fast).flatten())
        medium_states.append(np.array(model.h_medium).flatten())
        slow_states.append(np.array(model.h_slow).flatten())
        surprises.append(surprise)
        valences.append(float(affect[0, 0]))
    
    fast_states = np.array(fast_states)  # (T, 16)
    medium_states = np.array(medium_states)  # (T, 16)
    slow_states = np.array(slow_states)  # (T, 8)
    surprises = np.array(surprises)
    valences = np.array(valences)
    
    timesteps = np.arange(len(fast_states))
    
    logger.info("Generating plot...")
    
    # Create figure with subplots
    n_plots = 4 if show_surprise else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots), sharex=True)
    
    # Fast layer (high variance - immediate reactions)
    ax = axes[0]
    for i in range(min(5, fast_states.shape[1])):  # Plot first 5 dimensions
        ax.plot(timesteps, fast_states[:, i], alpha=0.7, linewidth=1.5, label=f'Dim {i+1}')
    ax.set_ylabel('Fast Layer\nActivations', fontsize=11, fontweight='bold')
    ax.set_title('Fast Layer: Immediate Reactions (seconds)\nHigh variance, rapid changes', fontsize=12, pad=10)
    ax.legend(ncol=5, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Medium layer (moderate variance - conversational dynamics)
    ax = axes[1]
    for i in range(min(5, medium_states.shape[1])):
        ax.plot(timesteps, medium_states[:, i], alpha=0.7, linewidth=1.5, label=f'Dim {i+1}')
    ax.set_ylabel('Medium Layer\nActivations', fontsize=11, fontweight='bold')
    ax.set_title('Medium Layer: Conversational Dynamics (minutes)\nModerate variance, smoother evolution', fontsize=12, pad=10)
    ax.legend(ncol=5, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Slow layer (low variance - personality model)
    ax = axes[2]
    for i in range(min(5, slow_states.shape[1])):
        ax.plot(timesteps, slow_states[:, i], alpha=0.8, linewidth=2, label=f'Dim {i+1}')
    ax.set_ylabel('Slow Layer\nActivations', fontsize=11, fontweight='bold')
    ax.set_title('Slow Layer: Personality Model (hours/days)\nLow variance, stable over time', fontsize=12, pad=10)
    ax.legend(ncol=5, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Surprise (if requested)
    if show_surprise:
        ax = axes[3]
        ax.plot(timesteps, surprises, color='#e74c3c', linewidth=2, label='Surprise')
        ax.fill_between(timesteps, 0, surprises, alpha=0.3, color='#e74c3c')
        ax.axhline(y=np.mean(surprises), color='#34495e', linestyle='--', linewidth=1.5, label='Mean', alpha=0.7)
        ax.set_ylabel('Surprise', fontsize=11, fontweight='bold')
        ax.set_xlabel('Conversation Timestep', fontsize=11)
        ax.set_title('Surprise Signal: Prediction Error\n(Agent speaks when surprise exceeds threshold)', fontsize=12, pad=10)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel('Conversation Timestep', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    
    # Calculate variance ratios (HSI)
    fast_var = np.var(np.diff(fast_states, axis=0))
    medium_var = np.var(np.diff(medium_states, axis=0))
    slow_var = np.var(np.diff(slow_states, axis=0))
    
    logger.info(f"\nVariance of changes (HSI):")
    logger.info(f"  Fast layer:   {fast_var:.4f}")
    logger.info(f"  Medium layer: {medium_var:.4f}")
    logger.info(f"  Slow layer:   {slow_var:.4f}")
    logger.info(f"  Slow/Fast ratio: {slow_var/fast_var if fast_var > 0 else float('nan'):.3f}")


def plot_layer_comparison(
    model,
    conversation: List[mx.array],
    output_path: str = 'evaluation/visualizations/figures/layer_comparison.png'
):
    """
    Plot mean activation of each layer on same axis for comparison.
    
    Args:
        model: Noodlings model instance
        conversation: Single conversation
        output_path: Where to save the figure
    """
    logger.info("Generating layer comparison plot...")
    
    model.reset_states()
    
    fast_means = []
    medium_means = []
    slow_means = []
    
    for affect in conversation:
        if affect.ndim == 1:
            affect = affect[None, :]
        
        model(affect)
        
        fast_means.append(np.mean(np.abs(model.h_fast)))
        medium_means.append(np.mean(np.abs(model.h_medium)))
        slow_means.append(np.mean(np.abs(model.h_slow)))
    
    timesteps = np.arange(len(fast_means))
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(timesteps, fast_means, label='Fast Layer', linewidth=2.5, alpha=0.8, color='#e74c3c')
    plt.plot(timesteps, medium_means, label='Medium Layer', linewidth=2.5, alpha=0.8, color='#3498db')
    plt.plot(timesteps, slow_means, label='Slow Layer', linewidth=2.5, alpha=0.8, color='#2ecc71')
    
    plt.xlabel('Conversation Timestep', fontsize=12)
    plt.ylabel('Mean Absolute Activation', fontsize=12)
    plt.title('Hierarchical Layer Comparison: Activity Levels Over Time\n(Mean absolute activation per layer)', fontsize=14, pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")


def plot_surprise_spikes(
    model,
    conversation: List[mx.array],
    threshold: float = 0.3,
    output_path: str = 'evaluation/visualizations/figures/surprise_spikes.png'
):
    """
    Plot surprise over time with annotations showing when agent would speak.
    
    Args:
        model: Noodlings model instance
        conversation: Single conversation
        threshold: Surprise threshold for speaking
        output_path: Where to save the figure
    """
    logger.info("Generating surprise spike visualization...")
    
    model.reset_states()
    
    surprises = []
    valences = []
    
    for affect in conversation:
        if affect.ndim == 1:
            affect = affect[None, :]
        
        _, surprise = model(affect)
        surprises.append(surprise)
        valences.append(float(affect[0, 0]))
    
    surprises = np.array(surprises)
    valences = np.array(valences)
    timesteps = np.arange(len(surprises))
    
    # Find speaking points
    speaks = surprises > threshold
    speak_times = timesteps[speaks]
    
    plt.figure(figsize=(14, 7))
    
    # Plot surprise
    plt.plot(timesteps, surprises, linewidth=2, color='#e74c3c', label='Surprise', zorder=2)
    plt.fill_between(timesteps, 0, surprises, alpha=0.2, color='#e74c3c', zorder=1)
    
    # Mark threshold
    plt.axhline(y=threshold, color='#34495e', linestyle='--', linewidth=2, label=f'Threshold ({threshold})', zorder=3)
    
    # Mark speaking points
    plt.scatter(speak_times, surprises[speaks], 
                s=200, marker='*', color='#f39c12', 
                edgecolors='#d35400', linewidth=2,
                label='Agent Speaks', zorder=4)
    
    # Annotate a few speaking points
    for i, t in enumerate(speak_times[:5]):  # Annotate first 5
        plt.annotate(f'Speak #{i+1}', 
                    xy=(t, surprises[int(t)]),
                    xytext=(10, 20), textcoords='offset points',
                    fontsize=9, ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#d35400', lw=1.5))
    
    plt.xlabel('Conversation Timestep', fontsize=12)
    plt.ylabel('Surprise (Prediction Error)', fontsize=12)
    plt.title(f'Surprise-Driven Behavior: When Agent Speaks\n(Agent speaks when surprise > {threshold})', fontsize=14, pad=20)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"Agent would speak {speaks.sum()} times out of {len(surprises)} timesteps ({100*speaks.sum()/len(surprises):.1f}%)")


def main():
    """Generate temporal dynamics visualizations from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate temporal dynamics visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation/visualizations/figures',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # TODO: Load actual model
    logger.info(f"Loading model from {args.checkpoint}...")
    
    # Dummy model
    class DummyModel:
        def __init__(self):
            self.h_fast = mx.zeros((1, 16))
            self.h_medium = mx.zeros((1, 16))
            self.h_slow = mx.zeros((1, 8))
            
        def __call__(self, affect):
            # Fast layer: high sensitivity to affect
            self.h_fast = 0.5 * self.h_fast + 0.5 * mx.random.normal((1, 16)) * (1 + 2*affect[0, 0])
            # Medium layer: moderate dynamics
            self.h_medium = 0.85 * self.h_medium + 0.15 * self.h_fast
            # Slow layer: very stable
            self.h_slow = 0.98 * self.h_slow + 0.02 * self.h_medium[:, :8]
            
            surprise = float(mx.mean(mx.abs(affect))) * float(mx.random.uniform(0.5, 1.5))
            return self.h_fast, surprise
            
        def reset_states(self):
            self.h_fast = mx.zeros((1, 16))
            self.h_medium = mx.zeros((1, 16))
            self.h_slow = mx.zeros((1, 8))
    
    model = DummyModel()
    
    # Generate test conversation with emotional arc
    logger.info("Generating test conversation...")
    conversation = []
    for t in range(50):
        if t < 10:
            valence = 0.0  # Start neutral
        elif t < 20:
            valence = 0.8 * np.sin((t - 10) / 5 * np.pi / 2)  # Rise to positive
        elif t < 30:
            valence = 0.8 * (1 - (t - 20) / 10)  # Decline
        elif t < 40:
            valence = -0.7  # Negative event
        else:
            valence = -0.7 * (1 - (t - 40) / 10)  # Recovery
        
        affect = mx.array([valence, 0.5, 0.2, 0.2, 0.2])
        conversation.append(affect)
    
    # Generate all visualizations
    output_dir = Path(args.output_dir)
    
    plot_hierarchical_dynamics(
        model, conversation,
        output_path=output_dir / 'hierarchical_dynamics.png'
    )
    
    plot_layer_comparison(
        model, conversation,
        output_path=output_dir / 'layer_comparison.png'
    )
    
    plot_surprise_spikes(
        model, conversation,
        output_path=output_dir / 'surprise_spikes.png'
    )
    
    logger.info("\n" + "="*60)
    logger.info("Temporal dynamics visualizations complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
