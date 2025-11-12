#!/usr/bin/env python3
"""
Visualization: Musical Harmonic Hypothesis for Observer Networks

Creates comprehensive visualizations showing whether observer network stability
follows harmonic principles analogous to musical intervals.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import math

# Load results
with open('harmonic_ratios_results.json', 'r') as f:
    data = json.load(f)

results = data['results']
summary = data['summary']

# Extract data
observer_counts = [r['num_observers'] for r in results]
hsi_values = [r['final_hsi'] for r in results]
is_stable = [r['is_stable'] for r in results]

# Classify by interval type
consonant_configs = [50, 60, 80, 90, 100, 120, 150, 200]
dissonant_configs = [70, 75, 99]

# Create figure
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Title
fig.suptitle('Musical Harmonic Hypothesis - Observer Network Stability',
             fontsize=20, fontweight='bold', y=0.98)

# ============================================================================
# Panel 1: HSI by Observer Count (Colored by Interval Type)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])

# Color by interval type
colors = []
for n in observer_counts:
    if n in consonant_configs:
        colors.append('#2ecc71')  # Green = consonant
    elif n in dissonant_configs:
        colors.append('#e74c3c')  # Red = dissonant
    else:
        colors.append('#95a5a6')  # Gray = other

# Plot points
scatter = ax1.scatter(observer_counts, hsi_values, c=colors, s=200,
                     alpha=0.7, edgecolors='black', linewidth=2, zorder=3)

# Stability threshold
ax1.axhline(y=0.3, color='orange', linestyle='--', linewidth=2,
           label='Stability Threshold (HSI < 0.3)', alpha=0.7)

# Highlight harmonic series (50, 100, 150, 200)
harmonic_series = [50, 100, 150, 200]
for i, n in enumerate(harmonic_series, 1):
    if n in observer_counts:
        idx = observer_counts.index(n)
        ax1.annotate(f'{i}×', xy=(n, hsi_values[idx]),
                    xytext=(0, 15), textcoords='offset points',
                    fontsize=12, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax1.set_xlabel('Number of Observers', fontsize=14, fontweight='bold')
ax1.set_ylabel('HSI (Hierarchical Separation Index)', fontsize=14, fontweight='bold')
ax1.set_title('Observer Count vs. Stability (Color = Interval Type)', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Legend
consonant_patch = mpatches.Patch(color='#2ecc71', label='Consonant Intervals')
dissonant_patch = mpatches.Patch(color='#e74c3c', label='Dissonant Intervals')
threshold_line = plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='Stability Threshold')
ax1.legend(handles=[consonant_patch, dissonant_patch, threshold_line],
          loc='upper right', fontsize=12)

# ============================================================================
# Panel 2: Consonant vs. Dissonant Comparison
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])

consonant_stable_rate = summary['consonant_stability_rate'] * 100
dissonant_stable_rate = summary['dissonant_stability_rate'] * 100

bars = ax2.bar(['Consonant\nIntervals', 'Dissonant\nIntervals'],
               [consonant_stable_rate, dissonant_stable_rate],
               color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)

ax2.set_ylabel('Stability Rate (%)', fontsize=14, fontweight='bold')
ax2.set_title('Stability Rate Comparison', fontsize=16, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
for bar, rate in zip(bars, [consonant_stable_rate, dissonant_stable_rate]):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{rate:.1f}%', ha='center', va='bottom',
            fontsize=14, fontweight='bold')

# Add hypothesis test result
if summary['hypothesis_supported']:
    result_text = "✓ HYPOTHESIS SUPPORTED"
    result_color = '#2ecc71'
else:
    result_text = "✗ HYPOTHESIS NOT SUPPORTED"
    result_color = '#e74c3c'

ax2.text(0.5, 0.95, result_text, transform=ax2.transAxes,
        fontsize=12, fontweight='bold', ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=result_color, alpha=0.3))

# ============================================================================
# Panel 3: Musical Interval Pairs
# ============================================================================
ax3 = fig.add_subplot(gs[1, :])

# Define interval pairs
interval_pairs = [
    (50, 100, 'Octave (2:1)'),
    (60, 120, 'Octave (2:1)'),
    (60, 90, 'Perfect Fifth (3:2)'),
    (75, 100, 'Perfect Fourth (4:3)'),
    (80, 100, 'Major Third (5:4)'),
    (70, 99, 'Tritone (√2:1)'),
    (75, 80, 'Minor Second (16:15)'),
]

x_positions = []
pair_labels = []
hsi_pair_values = []
pair_colors = []

for i, (n1, n2, interval_name) in enumerate(interval_pairs):
    # Find HSI values
    try:
        hsi1 = next(r['final_hsi'] for r in results if r['num_observers'] == n1)
        hsi2 = next(r['final_hsi'] for r in results if r['num_observers'] == n2)

        x_positions.extend([i*3, i*3+1])
        pair_labels.extend([f'{n1}', f'{n2}'])
        hsi_pair_values.extend([hsi1, hsi2])

        # Color based on interval type
        if 'Tritone' in interval_name or 'Minor Second' in interval_name:
            pair_colors.extend(['#e74c3c', '#e74c3c'])
        else:
            pair_colors.extend(['#2ecc71', '#2ecc71'])

        # Add interval label
        ax3.text(i*3 + 0.5, max(hsi1, hsi2) + 0.02, interval_name,
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                rotation=0)
    except StopIteration:
        continue

bars = ax3.bar(x_positions, hsi_pair_values, color=pair_colors,
               alpha=0.7, edgecolor='black', linewidth=1.5)

ax3.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_xticks(x_positions)
ax3.set_xticklabels(pair_labels, fontsize=10)
ax3.set_ylabel('HSI', fontsize=14, fontweight='bold')
ax3.set_title('Musical Interval Pairs - HSI Comparison', fontsize=16, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(hsi_pair_values) * 1.15)

# ============================================================================
# Panel 4: Harmonic Series Pattern
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])

harmonic_data = []
for i, n in enumerate([50, 100, 150, 200], 1):
    if n in observer_counts:
        idx = observer_counts.index(n)
        harmonic_data.append((i, n, hsi_values[idx], is_stable[idx]))

if harmonic_data:
    harmonics, counts, hsis, stables = zip(*harmonic_data)

    colors_harm = ['#2ecc71' if s else '#e74c3c' for s in stables]
    bars = ax4.bar(harmonics, hsis, color=colors_harm, alpha=0.7,
                   edgecolor='black', linewidth=2)

    ax4.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Harmonic Multiple', fontsize=14, fontweight='bold')
    ax4.set_ylabel('HSI', fontsize=14, fontweight='bold')
    ax4.set_title('Harmonic Series (Base = 50)', fontsize=16, fontweight='bold')
    ax4.set_xticks(harmonics)
    ax4.set_xticklabels([f'{h}×\n(N={n})' for h, n in zip(harmonics, counts)])
    ax4.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Panel 5: Average HSI by Interval Category
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

consonant_hsi_values = [r['final_hsi'] for r in results if r['num_observers'] in consonant_configs]
dissonant_hsi_values = [r['final_hsi'] for r in results if r['num_observers'] in dissonant_configs]

consonant_avg = np.mean(consonant_hsi_values)
dissonant_avg = np.mean(dissonant_hsi_values)
consonant_std = np.std(consonant_hsi_values)
dissonant_std = np.std(dissonant_hsi_values)

bars = ax5.bar(['Consonant', 'Dissonant'], [consonant_avg, dissonant_avg],
               yerr=[consonant_std, dissonant_std], capsize=10,
               color=['#2ecc71', '#e74c3c'], alpha=0.7,
               edgecolor='black', linewidth=2)

ax5.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax5.set_ylabel('Average HSI', fontsize=14, fontweight='bold')
ax5.set_title('Average HSI by Category', fontsize=16, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, avg in zip(bars, [consonant_avg, dissonant_avg]):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{avg:.4f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold')

# ============================================================================
# Panel 6: Summary Statistics Box
# ============================================================================
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

summary_text = f"""
HYPOTHESIS TEST RESULTS

H0: Consonant and dissonant intervals
    show no difference in stability

H1: Consonant intervals are more stable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Consonant Intervals:
  • Configs tested: {len(consonant_configs)}
  • Stability rate: {consonant_stable_rate:.1f}%
  • Average HSI: {consonant_avg:.4f}
  • Std Dev: {consonant_std:.4f}

Dissonant Intervals:
  • Configs tested: {len(dissonant_configs)}
  • Stability rate: {dissonant_stable_rate:.1f}%
  • Average HSI: {dissonant_avg:.4f}
  • Std Dev: {dissonant_std:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Result: {"✓ H1 SUPPORTED" if summary['hypothesis_supported'] else "✗ H1 NOT SUPPORTED"}

Conclusion:
{'Consonant intervals show higher\nstability, suggesting harmonic\nresonance effects in observer\nnetworks.' if summary['hypothesis_supported'] else 'No clear difference between\nconsonant and dissonant intervals.\nHarmonic hypothesis needs\nrefinement or may be spurious.'}
"""

ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3))

# Save figure
plt.savefig('harmonic_ratios_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'harmonic_ratios_analysis.png'")
print()
