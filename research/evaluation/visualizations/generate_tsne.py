"""
t-SNE State Space Visualization

Generates 2D projections of phenomen

al states colored by affective valence.
Shows whether the latent space has interpretable structure.

Author: Noodlings Project
Date: November 2025
"""

import sys
sys.path.insert(0, '../..')

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_state_space_visualization(
    model,
    conversations: List[List[mx.array]],
    output_path: str = 'evaluation/visualizations/figures/state_space_tsne.png'
):
    """
    Create t-SNE plot of phenomenal states colored by affect.
    
    Args:
        model: Noodlings model instance
        conversations: List of conversations (affect sequences)
        output_path: Where to save the figure
    """
    logger.info("Collecting phenomenal states from conversations...")
    
    states = []
    valences = []
    labels = []
    
    for conv_idx, conversation in enumerate(conversations):
        model.reset_states()
        
        for t, affect in enumerate(conversation):
            if affect.ndim == 1:
                affect = affect[None, :]
            
            # Process timestep
            model(affect)
            
            # Extract full phenomenal state (40-D)
            h_fast = np.array(model.h_fast).flatten()
            h_medium = np.array(model.h_medium).flatten()
            h_slow = np.array(model.h_slow).flatten()
            
            phenomenal_state = np.concatenate([h_fast, h_medium, h_slow])
            states.append(phenomenal_state)
            
            # Get valence for coloring
            valence = float(affect[0, 0])  # First dimension is valence
            valences.append(valence)
            
            # Label based on valence
            if valence > 0.3:
                labels.append('Positive')
            elif valence < -0.3:
                labels.append('Negative')
            else:
                labels.append('Neutral')
    
    states = np.array(states)
    logger.info(f"Collected {len(states)} states (40-D)")
    
    # t-SNE dimensionality reduction
    logger.info("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    states_2d = tsne.fit_transform(states)
    
    logger.info("Generating plot...")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot each emotion cluster
    colors = {'Positive': '#e74c3c', 'Negative': '#3498db', 'Neutral': '#95a5a6'}
    for label, color in colors.items():
        mask = [l == label for l in labels]
        plt.scatter(
            states_2d[mask, 0],
            states_2d[mask, 1],
            c=color,
            label=label,
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.5
        )
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Phenomenal State Space: Emotional Clustering\n(40-D states projected to 2-D via t-SNE)', fontsize=14, pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")
    
    # Calculate silhouette score
    numeric_labels = [0 if l == 'Positive' else 1 if l == 'Negative' else 2 for l in labels]
    silhouette = silhouette_score(states_2d, numeric_labels)
    logger.info(f"Silhouette score: {silhouette:.3f}")
    logger.info("  (Higher is better; >0.5 indicates good clustering)")
    
    return silhouette


def generate_valence_gradient_plot(
    model,
    conversations: List[List[mx.array]],
    output_path: str = 'evaluation/visualizations/figures/valence_gradient_tsne.png'
):
    """
    t-SNE plot with continuous valence gradient coloring.
    
    Args:
        model: Noodlings model instance
        conversations: List of conversations
        output_path: Where to save the figure
    """
    logger.info("Generating valence gradient t-SNE...")
    
    states = []
    valences = []
    
    for conversation in conversations:
        model.reset_states()
        
        for affect in conversation:
            if affect.ndim == 1:
                affect = affect[None, :]
            
            model(affect)
            
            h_fast = np.array(model.h_fast).flatten()
            h_medium = np.array(model.h_medium).flatten()
            h_slow = np.array(model.h_slow).flatten()
            
            phenomenal_state = np.concatenate([h_fast, h_medium, h_slow])
            states.append(phenomenal_state)
            
            valence = float(affect[0, 0])
            valences.append(valence)
    
    states = np.array(states)
    valences = np.array(valences)
    
    logger.info("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    states_2d = tsne.fit_transform(states)
    
    # Plot with gradient
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        states_2d[:, 0],
        states_2d[:, 1],
        c=valences,
        cmap='RdYlBu_r',  # Red (positive) to Blue (negative)
        alpha=0.7,
        s=30,
        edgecolors='white',
        linewidth=0.5,
        vmin=-1,
        vmax=1
    )
    
    plt.colorbar(scatter, label='Valence (Negative ← → Positive)')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Phenomenal State Space: Valence Gradient\n(Continuous coloring from negative to positive)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved to {output_path}")


def main():
    """Generate t-SNE visualizations from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate t-SNE state space visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, default='../../training/data/synthetic/test.json',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='evaluation/visualizations/figures',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # TODO: Load actual model from checkpoint
    logger.info(f"Loading model from {args.checkpoint}...")
    
    # Dummy model for now
    class DummyModel:
        def __init__(self):
            self.h_fast = mx.zeros((1, 16))
            self.h_medium = mx.zeros((1, 16))
            self.h_slow = mx.zeros((1, 8))
            
        def __call__(self, affect):
            # Simple dynamics based on affect
            self.h_fast = 0.7 * self.h_fast + 0.3 * mx.random.normal((1, 16)) * (1 + affect[0, 0])
            self.h_medium = 0.9 * self.h_medium + 0.1 * self.h_fast
            self.h_slow = 0.95 * self.h_slow + 0.05 * self.h_medium
            return self.h_fast, 0.0
            
        def reset_states(self):
            self.h_fast = mx.zeros((1, 16))
            self.h_medium = mx.zeros((1, 16))
            self.h_slow = mx.zeros((1, 8))
    
    model = DummyModel()
    
    # Generate test data
    logger.info("Generating test conversations...")
    conversations = []
    for _ in range(30):
        conversation = []
        # Create emotional arc (start neutral, move to emotion, return to neutral)
        emotion = np.random.choice([-0.8, 0.8])  # Negative or positive
        for t in range(20):
            if t < 5:
                valence = 0.0  # Start neutral
            elif t < 15:
                valence = emotion * (1 - abs(t - 10) / 5)  # Arc to emotion
            else:
                valence = emotion * (1 - (t - 15) / 5)  # Return to neutral
            
            affect = mx.array([valence, 0.5, 0.2, 0.2, 0.2])
            conversation.append(affect)
        conversations.append(conversation)
    
    # Generate visualizations
    output_dir = Path(args.output_dir)
    
    silhouette = generate_state_space_visualization(
        model, conversations,
        output_path=output_dir / 'state_space_tsne.png'
    )
    
    generate_valence_gradient_plot(
        model, conversations,
        output_path=output_dir / 'valence_gradient_tsne.png'
    )
    
    logger.info("\n" + "="*60)
    logger.info("t-SNE visualizations complete!")
    logger.info(f"Silhouette score: {silhouette:.3f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
