#!/usr/bin/env python3
"""Quick analysis of threshold results as they arrive."""

import json
import numpy as np
from pathlib import Path

results_file = Path('threshold_results.json')

if not results_file.exists():
    print("⏳ No results yet - experiment still running")
    exit(0)

with open(results_file) as f:
    results = json.load(f)

print("="*80)
print("THRESHOLD EXPERIMENT - INTERIM RESULTS")
print("="*80)
print()

print(f"Completed: {len(results)}/8 configurations")
print()

if results:
    print("Results so far:")
    print("-"*80)
    print(f"{'Observers':<12} {'HSI':<10} {'Status':<15} {'Training (s)':<12}")
    print("-"*80)
    
    for r in results:
        n = r['num_observers']
        hsi = r['hsi'].get('slow/fast', float('nan'))
        time = r['training_time']
        
        if not np.isnan(hsi):
            if hsi < 0.3:
                status = "✅ STABLE"
            elif hsi < 1.5:
                status = "⚠️ UNSTABLE"
            else:
                status = "❌ COLLAPSED"
        else:
            status = "ERROR"
        
        print(f"{n:<12} {hsi:<10.3f} {status:<15} {time:<12.1f}")
    
    print()
    
    # Attempt power law fit if we have enough data
    valid = [(r['num_observers'], r['hsi'].get('slow/fast')) 
             for r in results 
             if r['hsi'].get('slow/fast') and not np.isnan(r['hsi'].get('slow/fast'))]
    
    if len(valid) >= 3:
        N_vals = np.array([n for n, _ in valid])
        HSI_vals = np.array([h for _, h in valid])
        
        log_N = np.log(N_vals)
        log_HSI = np.log(HSI_vals)
        
        A = np.vstack([np.ones(len(log_N)), -log_N]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, log_HSI, rcond=None)
        
        log_k = coeffs[0]
        beta = coeffs[1]
        k = np.exp(log_k)
        
        print("PRELIMINARY POWER LAW FIT:")
        print(f"  HSI(N) = {k:.1f} / N^{beta:.2f}")
        
        if abs(beta - 2.0) < 0.3:
            print(f"  ✅ β ≈ {beta:.2f} (close to predicted 2.0)")
        else:
            print(f"  ⚠️ β = {beta:.2f} (differs from predicted 2.0)")
        
        N_critical = (k / 0.3) ** (1 / beta)
        print(f"  Predicted N_critical: {N_critical:.0f} observers")
        print()
        print("  ⏳ Preliminary - wait for more data for confirmation")
else:
    print("No results yet - first configuration still training")

print()
print("="*80)
