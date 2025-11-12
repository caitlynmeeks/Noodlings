#!/usr/bin/env python3
"""
Real-Time Φ Monitor

Monitor integrated information (Φ) during training in real-time.

Usage:
    python monitor_phi.py --log-dir ../logs/phi
    python monitor_phi.py --watch --interval 5

Author: Consilience Project
Date: November 2025
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Optional


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_number(value: float, decimals: int = 4) -> str:
    """Format number with color based on magnitude."""
    formatted = f"{value:.{decimals}f}"
    return formatted


def format_trend(value: float) -> str:
    """Format trend with arrow indicator."""
    if value > 0.001:
        return f"↗ +{value:.6f}"
    elif value < -0.001:
        return f"↘ {value:.6f}"
    else:
        return f"→ {value:.6f}"


def load_latest_checkpoint(log_dir: Path) -> Optional[Dict]:
    """Load the most recent Φ checkpoint."""
    checkpoints = list(log_dir.glob("phi_checkpoint_*.json"))

    if not checkpoints:
        return None

    # Get most recent
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    with open(latest, 'r') as f:
        data = json.load(f)

    return data


def display_dashboard(checkpoint: Dict):
    """Display Φ monitoring dashboard."""
    clear_screen()

    print("=" * 80)
    print("Φ REAL-TIME MONITOR - Integrated Information Tracking")
    print("=" * 80)

    # Session info
    metadata = checkpoint.get('metadata', {})
    print(f"\nSession: {metadata.get('session_id', 'unknown')}")
    print(f"State Dim: {metadata.get('state_dim', 40)}")
    print(f"Track Interval: {metadata.get('track_interval', 100)} steps")

    # Current metrics
    step = checkpoint.get('step', 0)
    history = checkpoint.get('history', {})

    if not history or not history.get('phi_proxy'):
        print("\n⚠️  No data available yet...")
        return

    # Get latest values
    latest_idx = -1

    phi_proxy = history['phi_proxy'][latest_idx]
    causal_density = history['causal_density'][latest_idx]
    differentiation = history['differentiation'][latest_idx]
    neural_complexity = history['neural_complexity'][latest_idx]
    lempel_ziv = history['lempel_ziv'][latest_idx]
    participation_ratio = history['participation_ratio'][latest_idx]

    loss = history['loss'][latest_idx]
    observer_loss = history['observer_loss'][latest_idx] if history.get('observer_loss') else 0.0
    meta_loss = history['meta_loss'][latest_idx] if history.get('meta_loss') else 0.0
    observer_influence = history['observer_influence'][latest_idx] if history.get('observer_influence') else 0.0

    # Compute trends (last 10 samples)
    window = min(10, len(history['phi_proxy']))
    if window >= 2:
        import numpy as np
        recent_phi = history['phi_proxy'][-window:]
        x = np.arange(len(recent_phi))
        trend_phi = np.polyfit(x, recent_phi, 1)[0]

        recent_loss = history['loss'][-window:]
        trend_loss = np.polyfit(x, recent_loss, 1)[0]
    else:
        trend_phi = 0.0
        trend_loss = 0.0

    # Display
    print(f"\n" + "─" * 80)
    print(f"STEP {step:,}")
    print("─" * 80)

    print(f"\n📊 INTEGRATED INFORMATION (Φ)")
    print(f"  Φ Proxy:             {format_number(phi_proxy)}")
    print(f"  Trend:               {format_trend(trend_phi)}")

    print(f"\n📊 Φ COMPONENTS")
    print(f"  Causal Density:      {format_number(causal_density)}")
    print(f"  Differentiation:     {format_number(differentiation)}")
    print(f"  Neural Complexity:   {format_number(neural_complexity)}")
    print(f"  Lempel-Ziv:          {format_number(lempel_ziv)}")
    print(f"  Participation Ratio: {format_number(participation_ratio)}")

    print(f"\n📊 TRAINING METRICS")
    print(f"  Main Loss:           {format_number(loss, 6)}")
    print(f"  Loss Trend:          {format_trend(trend_loss)}")

    if observer_loss > 0 or observer_influence > 0:
        print(f"\n📊 OBSERVER METRICS")
        print(f"  Observer Loss:       {format_number(observer_loss, 6)}")
        print(f"  Meta Loss:           {format_number(meta_loss, 6)}")
        print(f"  Observer Influence:  {format_number(observer_influence)}")

    # Summary statistics
    summary = checkpoint.get('summary', {})
    if summary and 'phi' in summary:
        phi_stats = summary['phi']

        print(f"\n📊 SESSION STATISTICS")
        print(f"  Φ Initial:           {format_number(phi_stats.get('initial', 0.0))}")
        print(f"  Φ Current:           {format_number(phi_stats.get('final', 0.0))}")
        print(f"  Φ Mean:              {format_number(phi_stats.get('mean', 0.0))}")
        print(f"  Φ Range:             [{format_number(phi_stats.get('min', 0.0))}, {format_number(phi_stats.get('max', 0.0))}]")
        print(f"  Φ Improvement:       {phi_stats.get('improvement_percent', 0.0):+.1f}%")

    # Interpretation
    print(f"\n📊 INTERPRETATION")
    if phi_proxy < 0.5:
        print("  ⚠️  Very low Φ - Minimal integration")
    elif phi_proxy < 1.0:
        print("  ⚡ Low Φ - Some integration present")
    elif phi_proxy < 2.0:
        print("  ✓ Moderate Φ - Consciousness correlates")
    elif phi_proxy < 3.0:
        print("  ✅ High Φ - Strong signature")
    else:
        print("  🌟 Very high Φ - Rich integration!")

    if trend_phi > 0.001:
        print("  ↗ Φ is INCREASING (good!)")
    elif trend_phi < -0.001:
        print("  ↘ Φ is DECREASING (check architecture)")
    else:
        print("  → Φ is STABLE")

    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit")
    print("=" * 80)


def watch_mode(log_dir: Path, interval: int = 5):
    """
    Watch mode: continuously monitor Φ in real-time.

    Args:
        log_dir: Directory with Φ tracking logs
        interval: Update interval in seconds
    """
    print(f"Watching {log_dir} for Φ updates (interval: {interval}s)...")
    print("Press Ctrl+C to exit\n")

    last_step = None

    try:
        while True:
            checkpoint = load_latest_checkpoint(log_dir)

            if checkpoint:
                current_step = checkpoint.get('step', 0)

                # Only update if new data
                if current_step != last_step:
                    display_dashboard(checkpoint)
                    last_step = current_step

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor Φ in real-time")

    parser.add_argument('--log-dir', type=str, default='../logs/phi',
                       help='Φ tracking log directory')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode (continuous updates)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Update interval in seconds (watch mode)')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"❌ Error: Log directory not found: {log_dir}")
        print(f"   Start training first with phi_tracker enabled")
        return 1

    if args.watch:
        # Watch mode
        watch_mode(log_dir, args.interval)
    else:
        # Single snapshot
        checkpoint = load_latest_checkpoint(log_dir)

        if not checkpoint:
            print(f"❌ No checkpoints found in {log_dir}")
            print(f"   Start training first with phi_tracker enabled")
            return 1

        display_dashboard(checkpoint)

    return 0


if __name__ == "__main__":
    sys.exit(main())
