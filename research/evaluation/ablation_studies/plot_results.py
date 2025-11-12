#!/usr/bin/env python3
"""
Visualize Ablation Study Results
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('ablation_results.json', 'r') as f:
    results = json.load(f)

# Extract data
architectures = []
hsi_values = []
tph_1_values = []
tph_10_values = []
snc_values = []
colors = []

color_map = {
    'baseline': '#cccccc',
    'control': '#999999',
    'single_layer': '#ff9999',
    'hierarchical': '#9999ff',
    'phase4': '#ffaa66',
    'dense_observers': '#66ff66'
}

for r in results:
    name = r['name']
    architectures.append(name.replace('_', '\n'))

    # HSI
    hsi = r['evaluation'].get('hsi', {})
    hsi_sf = hsi.get('slow/fast', float('nan'))
    hsi_values.append(hsi_sf)

    # TPH
    tph = r['evaluation'].get('tph', {})
    tph_1 = float(tph.get('1', float('nan')))
    tph_10 = float(tph.get('10', float('nan')))
    tph_1_values.append(tph_1)
    tph_10_values.append(tph_10)

    # SNC
    snc = r['evaluation'].get('snc', float('nan'))
    snc_values.append(snc)

    colors.append(color_map.get(name, '#666666'))

# Create figure
fig = plt.figure(figsize=(16, 10))

# Title
fig.suptitle('Noodlings Ablation Study Results (50 Epochs)\nThe Observer Stabilization Effect',
             fontsize=16, fontweight='bold')

# 1. HSI - Hierarchical Separation Index (MOST IMPORTANT)
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(range(len(architectures)), hsi_values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xticks(range(len(architectures)))
ax1.set_xticklabels(architectures, fontsize=9)
ax1.set_ylabel('HSI (Slow/Fast Variance Ratio)', fontweight='bold')
ax1.set_title('Hierarchical Separation Index\n(Lower = Better)', fontweight='bold')
ax1.axhline(y=0.3, color='green', linestyle='--', linewidth=2, label='Good threshold (<0.3)')
ax1.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='Poor threshold (>1.0)')
ax1.set_ylim([0, 12])
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, hsi_values)):
    if not np.isnan(val):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

# Highlight winner
winner_idx = np.nanargmin(hsi_values)
bars[winner_idx].set_edgecolor('gold')
bars[winner_idx].set_linewidth(4)

# 2. TPH - Temporal Prediction Horizon
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(architectures))
width = 0.35
bars1 = ax2.bar(x - width/2, tph_1_values, width, label='1-step', color=colors,
                edgecolor='black', linewidth=1, alpha=0.7)
bars2 = ax2.bar(x + width/2, tph_10_values, width, label='10-step', color=colors,
                edgecolor='black', linewidth=1, alpha=0.9)
ax2.set_xticks(range(len(architectures)))
ax2.set_xticklabels(architectures, fontsize=9)
ax2.set_ylabel('MSE (Lower = Better)', fontweight='bold')
ax2.set_title('Temporal Prediction Horizon\n(Prediction Accuracy)', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. SNC - Surprise-Novelty Correlation
ax3 = plt.subplot(2, 3, 3)
bars = ax3.bar(range(len(architectures)), snc_values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(architectures)))
ax3.set_xticklabels(architectures, fontsize=9)
ax3.set_ylabel('Pearson r', fontweight='bold')
ax3.set_title('Surprise-Novelty Correlation\n(Higher = Better)', fontweight='bold')
ax3.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='Strong (>0.7)')
ax3.axhline(y=0.4, color='orange', linestyle='--', linewidth=2, label='Moderate (>0.4)')
ax3.set_ylim([-0.1, 1.0])
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# 4. Observer Count vs HSI (The Key Insight!)
ax4 = plt.subplot(2, 3, 4)
observer_counts = [0, 0, 0, 0, 75, 150]  # baseline, control, single, hier, phase4, dense
trainable_archs = [2, 3, 4, 5]  # single, hier, phase4, dense
trainable_observers = [0, 0, 75, 150]
trainable_hsi = [hsi_values[i] for i in trainable_archs]

ax4.plot(trainable_observers, trainable_hsi, 'o-', markersize=12, linewidth=3, color='#4444ff')
ax4.set_xlabel('Number of Observers', fontweight='bold', fontsize=11)
ax4.set_ylabel('HSI (Slow/Fast)', fontweight='bold', fontsize=11)
ax4.set_title('THE OBSERVER STABILIZATION EFFECT\n"The Valley of Death at 75 Observers"',
              fontweight='bold', fontsize=12)
ax4.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (<0.3)')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 12])

# Annotate points
labels = ['Single\n(no obs)', 'Hierarchical\n(no obs)', 'Phase4\n(75 obs)', 'Dense\n(150 obs)']
for i, (x, y, label) in enumerate(zip(trainable_observers, trainable_hsi, labels)):
    ax4.annotate(label, xy=(x, y), xytext=(10, 10 if i != 1 else -30),
                textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow' if i == 3 else 'white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax4.legend(fontsize=8)

# 5. Training Timeline Visualization
ax5 = plt.subplot(2, 3, 5)
epochs = [5, 50]
hierarchical_hsi_timeline = [0.004, 11.423]  # From our experiments!
phase4_hsi_timeline = [1.014, 2.619]
dense_hsi_timeline = [0.285, 0.113]

ax5.plot(epochs, hierarchical_hsi_timeline, 'o-', linewidth=3, markersize=10,
         label='Hierarchical (0 obs)', color='#9999ff')
ax5.plot(epochs, phase4_hsi_timeline, 'o-', linewidth=3, markersize=10,
         label='Phase4 (75 obs)', color='#ffaa66')
ax5.plot(epochs, dense_hsi_timeline, 'o-', linewidth=3, markersize=10,
         label='Dense (150 obs)', color='#66ff66')

ax5.set_xlabel('Training Epochs', fontweight='bold', fontsize=11)
ax5.set_ylabel('HSI (Slow/Fast)', fontweight='bold', fontsize=11)
ax5.set_title('Hierarchy Collapse During Training\n(Without Observers)', fontweight='bold', fontsize=12)
ax5.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good threshold')
ax5.set_yscale('log')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, which='both')

# Annotate the collapse
ax5.annotate('COLLAPSE!', xy=(50, 11.423), xytext=(30, 8),
            fontsize=12, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

ax5.annotate('STABLE!', xy=(50, 0.113), xytext=(30, 0.3),
            fontsize=12, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', linewidth=2))

# 6. Summary box
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = """
KEY FINDINGS:

🏆 WINNER: Dense Observers (150 loops)
   • HSI: 0.113 (ONLY good separation!)
   • TPH: 0.1517 (best prediction)
   • SNC: 0.208 (best surprise)

😱 THE OBSERVER EFFECT:
   • 0 observers → Hierarchy COLLAPSES
     (HSI: 0.004 → 11.423 during training!)

   • 75 observers → "Valley of Death"
     (HSI: 2.619 - not enough stabilization)

   • 150 observers → GOLDILOCKS ZONE
     (HSI: 0.113 - preserved hierarchy!)

💡 HYPOTHESIS:
   Observers act as REGULARIZERS that
   prevent layers from learning the same
   timescale. More observers = more
   constraint = better separation!

📊 All trained models beat baselines on
   prediction (TPH), but ONLY Dense has
   true hierarchical structure!
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('ablation_results_summary.png', dpi=150, bbox_inches='tight')
print("✓ Saved: ablation_results_summary.png")

plt.show()
