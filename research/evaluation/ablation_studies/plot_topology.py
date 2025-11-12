#!/usr/bin/env python3
"""
Visualize topology ablation results.

Creates publication-quality figures showing how observer network topology
affects hierarchical stability.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_file = Path('results/topology_results.json')

if not results_file.exists():
    print(f"Error: {results_file} not found!")
    print("Run 'python3 test_observer_topology.py' first")
    exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

results = data['results']

print("="*80)
print("TOPOLOGY VISUALIZATION")
print("="*80)
print()

# Extract data
names = [r['topology_name'] for r in results]
depths = [r['hierarchy_depth'] for r in results]
hsis = [r['final_hsi'] for r in results]
losses = [r['final_loss'] for r in results]
topologies = [r['topology'] for r in results]
statuses = [r['status'] for r in results]

# Sort by hierarchy depth for plotting
sorted_indices = np.argsort(depths)
names_sorted = [names[i] for i in sorted_indices]
depths_sorted = [depths[i] for i in sorted_indices]
hsis_sorted = [hsis[i] for i in sorted_indices]
topologies_sorted = [topologies[i] for i in sorted_indices]

# Colors by status
colors = []
for status in [statuses[i] for i in sorted_indices]:
    if '✅' in status:
        colors.append('#2ecc71')
    elif '⚠️' in status:
        colors.append('#f39c12')
    else:
        colors.append('#e74c3c')

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# PLOT 1: HSI vs. Hierarchy Depth
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])

# Scatter plot with topology labels
for i, (depth, hsi, name, color) in enumerate(zip(depths_sorted, hsis_sorted, names_sorted, colors)):
    ax1.scatter(depth, hsi, s=300, c=color, edgecolor='black',
               linewidth=2, zorder=3, alpha=0.8)
    # Label
    ax1.annotate(name.replace('_', ' ').title(),
                xy=(depth, hsi),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

# Thresholds
ax1.axhline(y=0.3, color='green', linestyle='--', linewidth=2,
           label='Stable Threshold', alpha=0.7)
ax1.axhline(y=1.0, color='orange', linestyle='--', linewidth=2,
           label='Unstable Threshold', alpha=0.7)

# Trend line
if len(depths_sorted) > 2:
    z = np.polyfit(depths_sorted, hsis_sorted, 1)
    p = np.poly1d(z)
    depth_range = np.linspace(min(depths_sorted), max(depths_sorted), 100)
    ax1.plot(depth_range, p(depth_range), 'b-', linewidth=2,
            alpha=0.5, label=f'Trend: HSI = {z[0]:.2f}×depth + {z[1]:.2f}')

ax1.set_xlabel('Hierarchy Depth Score', fontsize=14, fontweight='bold')
ax1.set_ylabel('HSI (Hierarchical Separation Index)', fontsize=14, fontweight='bold')
ax1.set_title('Observer Topology Effect on Stability', fontsize=16, fontweight='bold')
ax1.legend(fontsize=11, loc='best')
ax1.grid(alpha=0.3)
ax1.set_ylim(bottom=0)

# ============================================================================
# PLOT 2: Topology stacked bar chart
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])

# Stack L0, L1, L2
x_pos = np.arange(len(names_sorted))
l0_counts = [t[0] for t in topologies_sorted]
l1_counts = [t[1] for t in topologies_sorted]
l2_counts = [t[2] for t in topologies_sorted]

ax2.barh(x_pos, l0_counts, color='#3498db', label='L0', edgecolor='black', linewidth=1)
ax2.barh(x_pos, l1_counts, left=l0_counts, color='#9b59b6',
        label='L1', edgecolor='black', linewidth=1)
ax2.barh(x_pos, l2_counts, left=[l0+l1 for l0, l1 in zip(l0_counts, l1_counts)],
        color='#e74c3c', label='L2', edgecolor='black', linewidth=1)

ax2.set_yticks(x_pos)
ax2.set_yticklabels([n.replace('_', ' ').title() for n in names_sorted], fontsize=10)
ax2.set_xlabel('Observer Count', fontsize=12, fontweight='bold')
ax2.set_title('Topology Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

# ============================================================================
# PLOT 3: HSI comparison bar chart
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

bars = ax3.bar(range(len(names_sorted)), hsis_sorted, color=colors,
              edgecolor='black', linewidth=2, alpha=0.8)

ax3.set_xticks(range(len(names_sorted)))
ax3.set_xticklabels([n.replace('_', '\n') for n in names_sorted],
                    fontsize=9, rotation=0)
ax3.set_ylabel('Final HSI', fontsize=12, fontweight='bold')
ax3.set_title('HSI by Topology', fontsize=13, fontweight='bold')
ax3.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax3.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.5)
ax3.grid(axis='y', alpha=0.3)

# ============================================================================
# PLOT 4: Loss trajectories
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])

for i, (name, result, color) in enumerate(zip(names_sorted,
                                               [results[j] for j in sorted_indices],
                                               colors)):
    loss_history = result['loss_history']
    ax4.plot(loss_history, linewidth=2, label=name.replace('_', ' ').title(),
            color=color, alpha=0.7)

ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax4.set_title('Training Dynamics', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9, ncol=2)
ax4.grid(alpha=0.3)
ax4.set_yscale('log')

# ============================================================================
# PLOT 5: Summary statistics table
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

# Create table data
table_data = []
for name, depth, hsi, status in zip(names_sorted, depths_sorted, hsis_sorted,
                                     [statuses[i] for i in sorted_indices]):
    status_symbol = "✅" if "✅" in status else ("⚠️" if "⚠️" in status else "❌")
    table_data.append([
        name.replace('_', ' ').title(),
        f"{depth:.2f}",
        f"{hsi:.3f}",
        status_symbol
    ])

# Create table
table = ax5.table(cellText=table_data,
                 colLabels=['Topology', 'Depth', 'HSI', 'Status'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color cells
for i in range(len(table_data)):
    if '✅' in statuses[sorted_indices[i]]:
        color = '#d5f4e6'
    elif '⚠️' in statuses[sorted_indices[i]]:
        color = '#fef5e7'
    else:
        color = '#fadbd8'

    for j in range(4):
        table[(i+1, j)].set_facecolor(color)

# Header styling
for j in range(4):
    table[(0, j)].set_facecolor('#34495e')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax5.set_title('Summary Table', fontsize=13, fontweight='bold', pad=20)

# ============================================================================
# Save
# ============================================================================
plt.suptitle('Observer Network Topology Ablation Study',
            fontsize=18, fontweight='bold', y=0.995)

plt.savefig('topology_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: topology_analysis.png")
print()

# ============================================================================
# Print insights
# ============================================================================
print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()

# Best/worst
best_idx = np.argmin(hsis)
worst_idx = np.argmax(hsis)

print(f"1. BEST TOPOLOGY: {names[best_idx]}")
print(f"   HSI: {hsis[best_idx]:.3f}")
print(f"   Distribution: L0={topologies[best_idx][0]}, "
      f"L1={topologies[best_idx][1]}, L2={topologies[best_idx][2]}")
print()

print(f"2. WORST TOPOLOGY: {names[worst_idx]}")
print(f"   HSI: {hsis[worst_idx]:.3f}")
print(f"   Distribution: L0={topologies[worst_idx][0]}, "
      f"L1={topologies[worst_idx][1]}, L2={topologies[worst_idx][2]}")
print()

# Range
hsi_range = max(hsis) - min(hsis)
print(f"3. TOPOLOGY EFFECT SIZE: {hsi_range:.3f}")
if hsi_range > 0.1:
    print("   → LARGE effect: Topology significantly matters!")
elif hsi_range > 0.05:
    print("   → MODERATE effect: Topology matters")
else:
    print("   → SMALL effect: Topology has minimal impact")
print()

# Correlation
if len(depths) > 2:
    corr = np.corrcoef(depths, hsis)[0, 1]
    print(f"4. DEPTH-HSI CORRELATION: r = {corr:.3f}")
    if corr < -0.5:
        print("   → Strong negative: Deeper hierarchy → lower HSI")
    elif corr > 0.5:
        print("   → Strong positive: Deeper hierarchy → HIGHER HSI (!)")
    else:
        print("   → Weak correlation: Depth doesn't determine HSI alone")
    print()

# Flat vs hierarchical
flat_hsi = [h for n, h in zip(names, hsis) if 'flat' in n.lower()]
hier_hsi = [h for n, d, h in zip(names, depths, hsis) if d > 0]

if flat_hsi and hier_hsi:
    flat = flat_hsi[0]
    avg_hier = np.mean(hier_hsi)
    diff = flat - avg_hier

    print(f"5. FLAT vs. HIERARCHICAL:")
    print(f"   Flat HSI: {flat:.3f}")
    print(f"   Hierarchical avg: {avg_hier:.3f}")
    print(f"   Difference: {diff:+.3f}")

    if abs(diff) > 0.05:
        if diff > 0:
            print("   → Hierarchy HELPS! (lowers HSI)")
        else:
            print("   → Hierarchy HURTS! (raises HSI)")
    else:
        print("   → No significant difference")
    print()

print("="*80)
print("SCIENTIFIC INTERPRETATION")
print("="*80)
print()

if hsi_range > 0.1:
    print("🎯 TOPOLOGY IS CRITICAL!")
    print()
    print("The structure of the observer network matters as much as (or more than)")
    print("the total number of observers. This suggests:")
    print()
    print("  • Observer network architecture is a design variable")
    print("  • Optimal topology may be problem-specific")
    print("  • Simple 'more observers = better' is WRONG")
    print()

if abs(corr) < 0.3:
    print("🤔 NON-LINEAR RELATIONSHIP")
    print()
    print("Hierarchy depth alone doesn't determine stability. Other factors:")
    print("  • Specific L0/L1/L2 ratios matter")
    print("  • Might be optimal 'sweet spots'")
    print("  • Interaction effects between levels")
    print()

print("="*80)
print()
