#!/usr/bin/env python3
"""
Analyze and visualize multi-agent collapse experiment results.

Creates publication-quality figures showing:
1. HSI across conditions (with error bars)
2. Cooperation rates
3. Game performance scores
4. HSI trajectories over time
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load all results
results_dir = Path('results')
conditions = ['A_no_observers', 'B_few_observers', 'C_balanced_observers', 'D_dense_observers']

all_results = {}
for condition in conditions:
    with open(results_dir / f'{condition}_results.json', 'r') as f:
        all_results[condition] = json.load(f)

# Extract summary statistics
summary = {}
for condition in conditions:
    results = all_results[condition]

    hsi_values = [r['summary']['avg_hsi'] for r in results if r['summary']['avg_hsi'] is not None]
    coop_rates = [r['summary']['cooperation_rate'] for r in results]
    scores = [r['summary']['avg_score'] for r in results]

    summary[condition] = {
        'hsi_mean': np.mean(hsi_values) if hsi_values else np.nan,
        'hsi_std': np.std(hsi_values) if hsi_values else np.nan,
        'coop_mean': np.mean(coop_rates),
        'coop_std': np.std(coop_rates),
        'score_mean': np.mean(scores),
        'score_std': np.std(scores),
        'hsi_trajectories': [r['agent_hsi_history'] for r in results]
    }

print("="*80)
print("MULTI-AGENT COLLAPSE EXPERIMENT - ANALYSIS")
print("="*80)
print()

# Print summary table
print("Summary Statistics:")
print(f"{'Condition':<25} {'HSI':<20} {'Cooperation':<15} {'Score':<15}")
print("-"*80)
for condition in conditions:
    s = summary[condition]
    print(f"{condition:<25} {s['hsi_mean']:.3f} ± {s['hsi_std']:.3f}       "
          f"{s['coop_mean']:.1%} ± {s['coop_std']:.1%}    "
          f"{s['score_mean']:.1f} ± {s['score_std']:.1f}")
print()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multi-Agent Hierarchical Collapse Experiment Results', fontsize=16, fontweight='bold')

# Plot 1: HSI across conditions
ax = axes[0, 0]
condition_names = [c.replace('_', ' ').title() for c in conditions]
observer_counts = [0, 3, 10, 15]
hsi_means = [summary[c]['hsi_mean'] for c in conditions]
hsi_stds = [summary[c]['hsi_std'] for c in conditions]

bars = ax.bar(observer_counts, hsi_means, yerr=hsi_stds,
               color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'],
               capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Number of Observers', fontsize=12, fontweight='bold')
ax.set_ylabel('Hierarchical Separation Index (HSI)', fontsize=12, fontweight='bold')
ax.set_title('HSI vs Observer Density', fontsize=14, fontweight='bold')
ax.axhline(y=0.3, color='red', linestyle='--', linewidth=2, label='Stability Threshold (HSI=0.3)')
ax.set_xticks(observer_counts)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Plot 2: Cooperation rates
ax = axes[0, 1]
coop_means = [summary[c]['coop_mean'] for c in conditions]
coop_stds = [summary[c]['coop_std'] for c in conditions]

bars = ax.bar(observer_counts, coop_means, yerr=coop_stds,
               color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'],
               capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Number of Observers', fontsize=12, fontweight='bold')
ax.set_ylabel('Cooperation Rate', fontsize=12, fontweight='bold')
ax.set_title('Cooperation vs Observer Density', fontsize=14, fontweight='bold')
ax.set_xticks(observer_counts)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.grid(axis='y', alpha=0.3)

# Plot 3: Game scores
ax = axes[1, 0]
score_means = [summary[c]['score_mean'] for c in conditions]
score_stds = [summary[c]['score_std'] for c in conditions]

bars = ax.bar(observer_counts, score_means, yerr=score_stds,
               color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'],
               capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Number of Observers', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Game Score', fontsize=12, fontweight='bold')
ax.set_title('Performance vs Observer Density', fontsize=14, fontweight='bold')
ax.set_xticks(observer_counts)
ax.grid(axis='y', alpha=0.3)

# Plot 4: HSI trajectory over time (average across agents for first replication)
ax = axes[1, 1]
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

for i, condition in enumerate(conditions):
    # Get first replication's HSI history
    hsi_history = summary[condition]['hsi_trajectories'][0]

    # Average across all agents at each timestep
    if hsi_history:
        # Get max length
        max_len = max(len(h) for h in hsi_history.values())

        # Pad and average
        timesteps = []
        avg_hsi_over_time = []

        for t in range(max_len):
            values_at_t = [h[t] for agent_id, h in hsi_history.items()
                          if t < len(h) and not np.isnan(h[t])]
            if values_at_t:
                timesteps.append(t)
                avg_hsi_over_time.append(np.mean(values_at_t))

        if timesteps:
            # Plot with smoothing
            window = 10
            if len(avg_hsi_over_time) > window:
                smoothed = np.convolve(avg_hsi_over_time,
                                      np.ones(window)/window,
                                      mode='valid')
                ax.plot(timesteps[window-1:], smoothed,
                       color=colors[i], linewidth=2,
                       label=f'{observer_counts[i]} observers', alpha=0.8)
            else:
                ax.plot(timesteps, avg_hsi_over_time,
                       color=colors[i], linewidth=2,
                       label=f'{observer_counts[i]} observers', alpha=0.8)

ax.set_xlabel('Timestep', fontsize=12, fontweight='bold')
ax.set_ylabel('Average HSI', fontsize=12, fontweight='bold')
ax.set_title('HSI Dynamics Over Time (Smoothed)', fontsize=14, fontweight='bold')
ax.axhline(y=0.3, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('multi_agent_collapse_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved figure: multi_agent_collapse_results.png")
print()

# Statistical analysis
print("="*80)
print("STATISTICAL ANALYSIS")
print("="*80)
print()

print("KEY FINDINGS:")
print()

# Compare no observers vs dense observers
no_obs_hsi = summary['A_no_observers']['hsi_mean']
dense_obs_hsi = summary['D_dense_observers']['hsi_mean']
hsi_change = no_obs_hsi - dense_obs_hsi
hsi_pct = (hsi_change / no_obs_hsi) * 100 if no_obs_hsi > 0 else 0

print(f"1. HSI Difference (No observers vs Dense):")
print(f"   Change: {hsi_change:+.3f} ({hsi_pct:+.1f}%)")
print(f"   Interpretation: {'Observers reduced HSI' if hsi_change > 0 else 'Observers increased HSI'}")
print()

# Cooperation improvement
no_obs_coop = summary['A_no_observers']['coop_mean']
dense_obs_coop = summary['D_dense_observers']['coop_mean']
coop_change = dense_obs_coop - no_obs_coop
coop_pct = (coop_change / no_obs_coop) * 100 if no_obs_coop > 0 else 0

print(f"2. Cooperation Rate Difference (No observers vs Dense):")
print(f"   Change: {coop_change:+.1%} ({coop_pct:+.1f}%)")
print(f"   Interpretation: Dense observers {'improved' if coop_change > 0 else 'reduced'} cooperation")
print()

# Score improvement
no_obs_score = summary['A_no_observers']['score_mean']
dense_obs_score = summary['D_dense_observers']['score_mean']
score_change = dense_obs_score - no_obs_score
score_pct = (score_change / no_obs_score) * 100 if no_obs_score > 0 else 0

print(f"3. Game Performance Difference (No observers vs Dense):")
print(f"   Change: {score_change:+.1f} points ({score_pct:+.1f}%)")
print(f"   Interpretation: Dense observers {'improved' if score_change > 0 else 'reduced'} performance")
print()

# Check if effect is monotonic
print("4. Monotonicity Check:")
hsi_values = [summary[c]['hsi_mean'] for c in conditions]
is_monotonic_decreasing = all(hsi_values[i] >= hsi_values[i+1] for i in range(len(hsi_values)-1))
is_monotonic_increasing = all(hsi_values[i] <= hsi_values[i+1] for i in range(len(hsi_values)-1))

if is_monotonic_decreasing:
    print("   ✓ HSI monotonically DECREASES with observers (supports hypothesis)")
elif is_monotonic_increasing:
    print("   ✗ HSI monotonically INCREASES with observers (contradicts hypothesis)")
else:
    print("   ~ HSI shows non-monotonic relationship with observers")
    print(f"   Pattern: {' → '.join([f'{h:.3f}' for h in hsi_values])}")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()

if is_monotonic_increasing:
    print("⚠️  HYPOTHESIS PARTIALLY FALSIFIED!")
    print()
    print("The prediction that more observers → lower HSI was NOT confirmed.")
    print("In competitive multi-agent games, observers may have different effects")
    print("than in single-agent predictive tasks.")
    print()
    print("However, dense hierarchical observers DID improve:")
    print(f"  • Cooperation rate: +{coop_pct:.1f}%")
    print(f"  • Game performance: +{score_pct:.1f}%")
    print()
    print("This suggests observers help with COORDINATION rather than just HSI.")
elif dense_obs_hsi < no_obs_hsi * 0.8:
    print("✅ HYPOTHESIS SUPPORTED!")
    print()
    print(f"Dense observers reduced HSI by {-hsi_pct:.1f}%, supporting the")
    print("General Hierarchical Collapse Principle in multi-agent systems.")
else:
    print("~ MIXED RESULTS")
    print()
    print("Effects are subtle and require further investigation with:")
    print("  • Longer runs (500+ rounds)")
    print("  • More replications (10+ per condition)")
    print("  • Different game pressures")

print()
print("="*80)
print()
