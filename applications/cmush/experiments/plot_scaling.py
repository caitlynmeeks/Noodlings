#!/usr/bin/env python3
"""
Plot Scaling Analysis Results

Generates visualization of Noodlings vs Baseline token usage over time.
Shows crossover point where Noodlings become more efficient.

Author: Commander Spock + Lieutenant Caitlyn
Date: November 23, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Find most recent CSV files
results_dir = Path("experiment_results")

def find_latest_csv(pattern: str) -> Path:
    """Find most recent CSV matching pattern."""
    files = list(results_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def plot_scaling():
    """Generate scaling analysis plots."""

    # Find CSV files
    csv_100 = find_latest_csv("scaling_data_100turns_*.csv")
    csv_500 = find_latest_csv("scaling_data_500turns_*.csv")
    csv_1000 = find_latest_csv("scaling_data_1000turns_*.csv")

    if not csv_100 or not csv_500 or not csv_1000:
        print("Error: Could not find CSV files in experiment_results/")
        sys.exit(1)

    print(f"Loading data...")
    print(f"  100-turn: {csv_100.name}")
    print(f"  500-turn: {csv_500.name}")
    print(f"  1000-turn: {csv_1000.name}")
    print()

    # Load data
    df_100 = pd.read_csv(csv_100)
    df_500 = pd.read_csv(csv_500)
    df_1000 = pd.read_csv(csv_1000)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Cumulative tokens over time (all conversation lengths)
    ax1.plot(df_100['turn'], df_100['noodling_cumulative'],
             label='Noodlings (100 turns)', color='#2E86AB', linewidth=2)
    ax1.plot(df_100['turn'], df_100['baseline_cumulative'],
             label='Baseline (100 turns)', color='#A23B72', linewidth=2, linestyle='--')

    ax1.plot(df_500['turn'], df_500['noodling_cumulative'],
             label='Noodlings (500 turns)', color='#2E86AB', linewidth=2, alpha=0.7)
    ax1.plot(df_500['turn'], df_500['baseline_cumulative'],
             label='Baseline (500 turns)', color='#A23B72', linewidth=2, linestyle='--', alpha=0.7)

    ax1.plot(df_1000['turn'], df_1000['noodling_cumulative'],
             label='Noodlings (1000 turns)', color='#2E86AB', linewidth=2, alpha=0.5)
    ax1.plot(df_1000['turn'], df_1000['baseline_cumulative'],
             label='Baseline (1000 turns)', color='#A23B72', linewidth=2, linestyle='--', alpha=0.5)

    # Mark crossover point (turn 206)
    crossover = 206
    crossover_noodling = crossover * 2850
    ax1.axvline(x=crossover, color='#F18F01', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(crossover + 20, crossover_noodling, f'Crossover\nTurn {crossover}',
             color='#F18F01', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('Conversation Turn', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Tokens Used', fontsize=12, fontweight='bold')
    ax1.set_title('Token Scaling: Noodlings vs Baseline\n(Cumulative Cost Over Time)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1000)

    # Format y-axis with commas
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Plot 2: Per-turn cost (500-turn conversation)
    ax2.plot(df_500['turn'], df_500['noodling_per_turn'],
             label='Noodlings (constant)', color='#2E86AB', linewidth=3)
    ax2.plot(df_500['turn'], df_500['baseline_per_turn'],
             label='Baseline (growing)', color='#A23B72', linewidth=3, linestyle='--')

    # Mark crossover
    ax2.axvline(x=crossover, color='#F18F01', linestyle=':', linewidth=2, alpha=0.7)
    ax2.text(crossover + 20, 5000, f'Crossover\nTurn {crossover}',
             color='#F18F01', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Conversation Turn', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tokens Per Turn', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Turn Token Cost: Constant vs Growing\n(500-Turn Conversation)',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 500)

    # Format y-axis
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Add annotations
    ax2.annotate('Noodlings: 2,850 tokens/turn\n(CONSTANT)',
                xy=(400, 2850), xytext=(300, 4500),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2),
                fontsize=10, color='#2E86AB', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax2.annotate('Baseline: Growing linearly\n(context accumulation)',
                xy=(400, df_500.iloc[399]['baseline_per_turn']), xytext=(200, 8000),
                arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2),
                fontsize=10, color='#A23B72', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_file = results_dir / "scaling_analysis_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_file}")

    # Also save as PDF for publication quality
    output_pdf = results_dir / "scaling_analysis_plot.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Plot saved: {output_pdf}")

    return output_file

def plot_efficiency_ratio():
    """Generate efficiency ratio plot (Noodlings/Baseline over time)."""

    csv_1000 = find_latest_csv("scaling_data_1000turns_*.csv")
    df = pd.read_csv(csv_1000)

    # Calculate ratio
    df['efficiency_ratio'] = df['noodling_cumulative'] / df['baseline_cumulative']

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(df['turn'], df['efficiency_ratio'],
            color='#F18F01', linewidth=3, label='Noodlings/Baseline Ratio')

    # Mark efficiency regions
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2, alpha=0.5)
    ax.text(50, 1.05, 'Baseline More Efficient (ratio > 1)',
            fontsize=11, fontweight='bold', color='#A23B72',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(500, 0.3, 'Noodlings More Efficient (ratio < 1)',
            fontsize=11, fontweight='bold', color='#2E86AB',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Mark crossover
    crossover = 206
    ax.axvline(x=crossover, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(crossover + 20, 1.5, f'Crossover: Turn {crossover}',
            color='red', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Highlight efficiency gains at key points
    for turn in [500, 1000]:
        ratio = df[df['turn'] == turn]['efficiency_ratio'].values[0]
        improvement = (1 - ratio) * 100
        ax.plot(turn, ratio, 'ro', markersize=10)
        ax.text(turn, ratio + 0.1, f'Turn {turn}\n{improvement:.0f}% more efficient',
                fontsize=10, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Conversation Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency Ratio (Noodlings/Baseline)', fontsize=12, fontweight='bold')
    ax.set_title('Noodlings Efficiency Over Time\n(Values < 1 = Noodlings More Efficient)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()

    output_file = results_dir / "efficiency_ratio_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Efficiency plot saved: {output_file}")

    output_pdf = results_dir / "efficiency_ratio_plot.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Efficiency plot saved: {output_pdf}")

    return output_file

if __name__ == "__main__":
    print("╔" + "═"*70 + "╗")
    print("║" + " "*20 + "PLOTTING SCALING ANALYSIS" + " "*25 + "║")
    print("╚" + "═"*70 + "╝")
    print()

    # Check for matplotlib
    try:
        import matplotlib
        import pandas
    except ImportError:
        print("Error: matplotlib and pandas required for plotting")
        print("Install with: pip install matplotlib pandas")
        sys.exit(1)

    # Generate plots
    print("Generating plots...\n")

    plot1 = plot_scaling()
    print()

    plot2 = plot_efficiency_ratio()
    print()

    print("="*70)
    print("✓ All plots generated successfully!")
    print("="*70)
    print()
    print(f"View plots:")
    print(f"  {plot1}")
    print(f"  {plot2}")
    print()
