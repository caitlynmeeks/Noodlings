#!/usr/bin/env python3
"""
Visualize Critical Threshold Results

Creates publication-quality figure showing HSI(N) curve and power law fit.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
with open('threshold_results.json', 'r') as f:
    results = json.load(f)

# Extract valid data
N_vals = []
HSI_vals = []

for r in results:
    hsi = r['hsi'].get('slow/fast', None)
    if hsi is not None and not np.isnan(hsi):
        N_vals.append(r['num_observers'])
        HSI_vals.append(hsi)

N_vals = np.array(N_vals)
HSI_vals = np.array(HSI_vals)

# Sort by N
sort_idx = np.argsort(N_vals)
N_vals = N_vals[sort_idx]
HSI_vals = HSI_vals[sort_idx]

# Fit power law: HSI = k / N^β
log_N = np.log(N_vals)
log_HSI = np.log(HSI_vals)

A = np.vstack([np.ones(len(log_N)), -log_N]).T
coeffs, residuals, _, _ = np.linalg.lstsq(A, log_HSI, rcond=None)

log_k = coeffs[0]
beta = coeffs[1]
k = np.exp(log_k)

# Generate smooth curve
N_smooth = np.linspace(N_vals.min(), N_vals.max(), 100)
HSI_smooth = k / (N_smooth ** beta)

# Calculate R²
HSI_pred = k / (N_vals ** beta)
ss_res = np.sum((HSI_vals - HSI_pred) ** 2)
ss_tot = np.sum((HSI_vals - np.mean(HSI_vals)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Predict N_critical
N_critical = (k / 0.3) ** (1 / beta)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: HSI(N) with power law fit
ax1.scatter(N_vals, HSI_vals, s=100, c='blue', alpha=0.7, edgecolors='black', linewidths=2,
           label='Measured', zorder=3)
ax1.plot(N_smooth, HSI_smooth, 'r-', linewidth=3, alpha=0.7,
        label=f'Power Law Fit: HSI = {k:.1f} / N^{beta:.2f}', zorder=2)

# Threshold lines
ax1.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label='Stable Threshold (HSI < 0.3)')
ax1.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7,
           label='Unstable Threshold (HSI > 1.0)')

# N_critical line
ax1.axvline(x=N_critical, color='purple', linestyle=':', linewidth=2, alpha=0.7,
           label=f'N_critical ≈ {N_critical:.0f}')

# Annotations
for N, HSI in zip(N_vals, HSI_vals):
    if HSI < 0.3:
        status = '✓'
        color = 'green'
    elif HSI < 1.5:
        status = '~'
        color = 'orange'
    else:
        status = '✗'
        color = 'red'
    ax1.annotate(status, (N, HSI), xytext=(5, 5), textcoords='offset points',
                fontsize=14, fontweight='bold', color=color)

ax1.set_xlabel('Number of Observers (N)', fontsize=12, fontweight='bold')
ax1.set_ylabel('HSI (Hierarchical Separation Index)', fontsize=12, fontweight='bold')
ax1.set_title('The Observer Stabilization Law\n' +
             f'HSI(N) = {k:.1f} / N^{beta:.2f}  (R² = {r_squared:.3f})',
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, max(HSI_vals) * 1.1])

# Plot 2: Log-log plot (should be linear if power law is correct)
ax2.scatter(log_N, log_HSI, s=100, c='blue', alpha=0.7, edgecolors='black', linewidths=2,
           label='Measured (log scale)', zorder=3)

log_HSI_fit = log_k - beta * log_N
ax2.plot(log_N, log_HSI_fit, 'r-', linewidth=3, alpha=0.7,
        label=f'Linear Fit: log(HSI) = {log_k:.2f} - {beta:.2f}×log(N)', zorder=2)

ax2.set_xlabel('log(N)', fontsize=12, fontweight='bold')
ax2.set_ylabel('log(HSI)', fontsize=12, fontweight='bold')
ax2.set_title('Log-Log Plot\n(Linear = Power Law Confirmed)',
             fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Add R² annotation
ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}',
        transform=ax2.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: threshold_analysis.png")

# Print summary
print("\n" + "="*80)
print("CRITICAL THRESHOLD ANALYSIS")
print("="*80)
print(f"\nPower Law: HSI(N) = {k:.1f} / N^{beta:.2f}")
print(f"Goodness of fit: R² = {r_squared:.3f}")
print()

if beta > 1.5 and beta < 2.5:
    print(f"✅ β = {beta:.2f} is close to predicted value of 2.0")
    print("   Power law hypothesis CONFIRMED")
else:
    print(f"⚠️ β = {beta:.2f} differs from predicted 2.0")
    print("   Power law hypothesis needs revision")

print()
print(f"Predicted N_critical (HSI < 0.3): {N_critical:.0f} observers")
print()

# Find actual transition
stable = [(N, HSI) for N, HSI in zip(N_vals, HSI_vals) if HSI < 0.3]
unstable = [(N, HSI) for N, HSI in zip(N_vals, HSI_vals) if HSI >= 0.3]

if stable and unstable:
    max_unstable = max(n for n, _ in unstable)
    min_stable = min(n for n, _ in stable)
    print(f"Measured transition: between {max_unstable} and {min_stable} observers")
    print(f"Transition width: {min_stable - max_unstable} observers")
else:
    print("⚠️ Need more data points to precisely locate transition")

print("="*80)

plt.show()
