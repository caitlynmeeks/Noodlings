#!/usr/bin/env python3
"""
Visualize oscillation mapping results with enhanced analysis.

Creates publication-quality figures showing:
1. HSI oscillation pattern (fine-grained)
2. FFT/frequency analysis
3. Stable zone clustering
4. Modular pattern visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# Load results
results_file = Path('results/oscillation_mapping_results.json')

if not results_file.exists():
    print(f"Error: {results_file} not found!")
    print("Run 'python3 test_oscillation_mapping.py' first")
    exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

results = data['results']
pattern_analysis = data['pattern_analysis']

# Extract data
observer_counts = np.array([r['num_observers'] for r in results])
hsi_values = np.array([r['final_hsi'] for r in results])
statuses = [r['status'] for r in results]

# Filter NaN
valid_mask = ~np.isnan(hsi_values)
observer_counts_valid = observer_counts[valid_mask]
hsi_values_valid = hsi_values[valid_mask]

print("="*80)
print("OSCILLATION MAPPING VISUALIZATION")
print("="*80)
print()
print(f"Data points: {len(observer_counts_valid)}")
print(f"Observer range: {observer_counts_valid.min()}-{observer_counts_valid.max()}")
print()

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Color by status
colors = []
for status in statuses:
    if '✅' in status:
        colors.append('#2ecc71')  # Green - stable
    elif '⚠️' in status:
        colors.append('#f39c12')  # Orange - unstable
    else:
        colors.append('#e74c3c')  # Red - collapsed

# ============================================================================
# PLOT 1: Main oscillation pattern (large, top span)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

# Plot with colored markers
for i, (n, h, c) in enumerate(zip(observer_counts_valid, hsi_values_valid, colors)):
    ax1.scatter(n, h, s=150, c=c, edgecolor='black', linewidth=2, zorder=3, alpha=0.8)

# Connect points to show oscillation
ax1.plot(observer_counts_valid, hsi_values_valid, 'b-', alpha=0.3, linewidth=1.5, zorder=1)

# Thresholds
ax1.axhline(y=0.3, color='green', linestyle='--', linewidth=2, label='Stable Threshold (HSI < 0.3)', alpha=0.7)
ax1.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='Unstable Threshold (HSI > 1.0)', alpha=0.7)

# Shade stability zones
ax1.fill_between(observer_counts_valid, 0, 0.3, alpha=0.1, color='green', label='Stable Zone')
ax1.fill_between(observer_counts_valid, 1.0, max(hsi_values_valid)*1.1, alpha=0.1, color='red', label='Collapsed Zone')

ax1.set_xlabel('Number of Observers', fontsize=14, fontweight='bold')
ax1.set_ylabel('HSI (Hierarchical Separation Index)', fontsize=14, fontweight='bold')
ax1.set_title('Observer Oscillation Pattern - Fine-Grained Mapping', fontsize=16, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(alpha=0.3)
ax1.set_ylim(bottom=0, top=max(hsi_values_valid)*1.1)

# ============================================================================
# PLOT 2: FFT Analysis
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

if len(hsi_values_valid) > 4:
    # Compute FFT
    fft = np.fft.fft(hsi_values_valid)
    freqs = np.fft.fftfreq(len(hsi_values_valid), d=5)  # 5 observer spacing
    power = np.abs(fft[:len(fft)//2])
    freqs_positive = freqs[:len(freqs)//2]

    # Plot power spectrum
    ax2.plot(freqs_positive[1:], power[1:], 'b-', linewidth=2)  # Skip DC
    ax2.fill_between(freqs_positive[1:], 0, power[1:], alpha=0.3)

    # Mark peak
    if len(power[1:]) > 0:
        peak_idx = np.argmax(power[1:])
        peak_freq = freqs_positive[1:][peak_idx]
        period = 1.0 / peak_freq if peak_freq != 0 else np.inf

        ax2.axvline(peak_freq, color='red', linestyle='--', linewidth=2,
                   label=f'Peak: {peak_freq:.3f} (Period ≈ {period:.1f})')
        ax2.scatter([peak_freq], [power[1:][peak_idx]], s=200, c='red',
                   edgecolor='black', linewidth=2, zorder=3, marker='*')

    ax2.set_xlabel('Frequency (cycles per observer)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Power', fontsize=12, fontweight='bold')
    ax2.set_title('Frequency Analysis (FFT)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

# ============================================================================
# PLOT 3: Stability zone distribution
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

stable_zones = pattern_analysis['stable_zones']
unstable_zones = pattern_analysis['unstable_zones']

# Create histogram bins
bins = np.arange(75, 160, 5)
stable_hist, _ = np.histogram([n for n in stable_zones], bins=bins)
unstable_hist, _ = np.histogram([n for n in unstable_zones], bins=bins)

# Stacked bar chart
bin_centers = bins[:-1] + 2.5
width = 4

ax3.bar(bin_centers, stable_hist, width=width, color='#2ecc71',
        label='Stable', alpha=0.8, edgecolor='black', linewidth=1)
ax3.bar(bin_centers, unstable_hist, width=width, bottom=stable_hist,
        color='#f39c12', label='Unstable', alpha=0.8, edgecolor='black', linewidth=1)

ax3.set_xlabel('Observer Count', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Stability Distribution', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)

# ============================================================================
# PLOT 4: Modular pattern (if detected)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 2])

if pattern_analysis['best_modulus']:
    mod = pattern_analysis['best_modulus']
    remainders = observer_counts_valid % mod

    # Group by remainder
    remainder_hsi = {}
    for r, h in zip(remainders, hsi_values_valid):
        if r not in remainder_hsi:
            remainder_hsi[r] = []
        remainder_hsi[r].append(h)

    # Plot box plots
    positions = sorted(remainder_hsi.keys())
    data = [remainder_hsi[r] for r in positions]

    bp = ax4.boxplot(data, positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True)

    # Color boxes by median HSI
    for patch, r in zip(bp['boxes'], positions):
        median_hsi = np.median(remainder_hsi[r])
        if median_hsi < 0.3:
            patch.set_facecolor('#2ecc71')
        elif median_hsi < 1.0:
            patch.set_facecolor('#f39c12')
        else:
            patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)

    ax4.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax4.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.5)

    ax4.set_xlabel(f'Observer Count mod {mod}', fontsize=12, fontweight='bold')
    ax4.set_ylabel('HSI', fontsize=12, fontweight='bold')
    ax4.set_title(f'Modular Pattern (mod {mod})', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3)

# ============================================================================
# PLOT 5: Gap analysis
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])

if 'stable_gaps' in pattern_analysis and len(pattern_analysis['stable_gaps']) > 0:
    gaps = pattern_analysis['stable_gaps']

    # Histogram of gaps
    ax5.hist(gaps, bins=10, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.axvline(pattern_analysis['mean_stable_gap'], color='red',
               linestyle='--', linewidth=3,
               label=f"Mean: {pattern_analysis['mean_stable_gap']:.1f}")

    ax5.set_xlabel('Gap Size (observers)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('Gap Analysis: Distance Between Stable Zones', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(alpha=0.3)

# ============================================================================
# PLOT 6: Clustering visualization
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])

# Plot stable zones as vertical bars
for n in stable_zones:
    ax6.axvline(n, color='green', alpha=0.6, linewidth=3)

for n in unstable_zones:
    ax6.axvline(n, color='orange', alpha=0.3, linewidth=3)

ax6.set_xlim(70, 160)
ax6.set_ylim(0, 1)
ax6.set_xlabel('Observer Count', fontsize=12, fontweight='bold')
ax6.set_yticks([])
ax6.set_title('Clustering Pattern', fontsize=13, fontweight='bold')

# Add clustering score
if 'clustering_score' in pattern_analysis:
    score = pattern_analysis['clustering_score']
    if score > 1.5:
        cluster_text = "CLUSTERED"
    elif score < 0.5:
        cluster_text = "UNIFORM"
    else:
        cluster_text = "MODERATE"

    ax6.text(0.5, 0.5, f"Clustering: {score:.2f}\n({cluster_text})",
            transform=ax6.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================================================
# PLOT 7: Loss trajectory comparison
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])

# Plot loss for a few representative configurations
representative_configs = [75, 95, 105, 125, 135, 150]
colors_rep = ['#e74c3c', '#f39c12', '#2ecc71', '#2ecc71', '#f39c12', '#2ecc71']

for config, color in zip(representative_configs, colors_rep):
    # Find result
    for r in results:
        if r['num_observers'] == config:
            loss_history = r['loss_history']
            ax7.plot(loss_history, linewidth=2, label=f"N={config}", color=color, alpha=0.7)
            break

ax7.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax7.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax7.set_title('Training Dynamics (Representative Configs)', fontsize=13, fontweight='bold')
ax7.legend(fontsize=10, ncol=2)
ax7.grid(alpha=0.3)
ax7.set_yscale('log')

# ============================================================================
# Save figure
# ============================================================================
plt.suptitle('Observer Oscillation Pattern - Complete Analysis',
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('oscillation_mapping_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: oscillation_mapping_analysis.png")
print()

# ============================================================================
# Print insights
# ============================================================================
print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()

print(f"1. STABILITY: {pattern_analysis['stability_ratio']:.1%} of configurations are stable")
print(f"   Stable: {len(stable_zones)} configs")
print(f"   Unstable: {len(unstable_zones)} configs")
print()

if 'dominant_period' in pattern_analysis:
    period = pattern_analysis['dominant_period']
    print(f"2. PERIODICITY: Oscillation period ≈ {period:.1f} observers")
    if period < 30:
        print("   → Strong periodic component detected!")
    print()

if pattern_analysis['best_modulus']:
    print(f"3. MODULAR PATTERN: Best fit at mod {pattern_analysis['best_modulus']}")
    print(f"   Separation strength: {pattern_analysis['modular_separation']:.3f}")
    print()

if 'mean_stable_gap' in pattern_analysis:
    print(f"4. GAP ANALYSIS: Mean gap = {pattern_analysis['mean_stable_gap']:.1f} observers")
    print(f"   Std deviation: {pattern_analysis['gap_std']:.1f}")
    if pattern_analysis['gap_std'] < pattern_analysis['mean_stable_gap'] * 0.3:
        print("   → Consistent spacing!")
    print()

if 'clustering_score' in pattern_analysis:
    score = pattern_analysis['clustering_score']
    print(f"5. CLUSTERING: Score = {score:.2f}")
    if score > 1.5:
        print("   → Stable zones are CLUSTERED (not random!)")
    elif score < 0.5:
        print("   → Stable zones are UNIFORMLY spaced")
    else:
        print("   → Moderate clustering")
    print()

print("="*80)
print("SCIENTIFIC INTERPRETATION")
print("="*80)
print()

# Draw conclusions based on all metrics
if (pattern_analysis.get('dominant_period', float('inf')) < 30 and
    pattern_analysis.get('clustering_score', 0) > 1.5):
    print("🌊 RESONANCE HYPOTHESIS SUPPORTED!")
    print()
    print("Evidence:")
    print("  ✓ Periodic oscillations detected")
    print("  ✓ Stable zones are clustered")
    print("  ✓ Non-monotonic relationship")
    print()
    print("Interpretation:")
    print("  Observer networks may exhibit resonance with main network structure.")
    print("  Certain observer counts align constructively (stable),")
    print("  others destructively (unstable).")
    print()

elif pattern_analysis.get('clustering_score', 0) < 0.5:
    print("📏 UNIFORM STABILITY ZONES")
    print()
    print("Evidence:")
    print("  ✓ Evenly spaced stable configurations")
    print("  ✓ Low clustering score")
    print()
    print("Interpretation:")
    print("  Regular intervals suggest quantized stability levels.")
    print("  May relate to network capacity constraints.")
    print()

else:
    print("🤔 COMPLEX DYNAMICS")
    print()
    print("The pattern is neither purely periodic nor uniform.")
    print("Multiple mechanisms may be at play:")
    print("  • Resonance effects")
    print("  • Capacity constraints")
    print("  • Interference patterns")
    print("  • Emergent nonlinear dynamics")
    print()

print("="*80)
print()
