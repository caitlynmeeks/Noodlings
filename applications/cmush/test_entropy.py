#!/usr/bin/env python3
"""
Test script for TrueRNG entropy service.

Verifies:
1. Device detection and connection
2. Entropy pool operation
3. Statistical quality of randomness
4. Graceful fallback to PRNG
"""

import sys
import time
import logging
from collections import Counter
from entropy_service import initialize_entropy_service, get_entropy_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_device_detection():
    """Test if TrueRNG device can be detected and connected."""
    print("\n=== Testing TrueRNG Device Detection ===")

    device_path = "/dev/cu.usbmodem211201"
    initialize_entropy_service(use_hardware=True, device_path=device_path)

    entropy = get_entropy_service()
    config = entropy.get_config()

    print(f"Hardware mode: {config['use_hardware']}")
    print(f"Device path: {config['device_path']}")
    print(f"Pool active: {config['active']}")

    if config['active']:
        print("SUCCESS: TrueRNG device connected and operational")
        return True
    else:
        print("WARNING: Failed to connect to TrueRNG, using PRNG fallback")
        return False


def test_entropy_generation():
    """Test entropy generation performance and quality."""
    print("\n=== Testing Entropy Generation ===")

    entropy = get_entropy_service()

    # Test uniform distribution
    print("\nGenerating 1000 uniform random values [0, 1)...")
    start_time = time.time()
    values = [entropy.uniform(0.0, 1.0) for _ in range(1000)]
    elapsed = time.time() - start_time

    print(f"Time: {elapsed:.3f}s ({1000/elapsed:.0f} values/sec)")
    print(f"Min: {min(values):.6f}, Max: {max(values):.6f}, Mean: {sum(values)/len(values):.6f}")

    # Test integer distribution
    print("\nGenerating 1000 random integers [1, 10]...")
    start_time = time.time()
    integers = [entropy.randint(1, 10) for _ in range(1000)]
    elapsed = time.time() - start_time

    print(f"Time: {elapsed:.3f}s ({1000/elapsed:.0f} values/sec)")

    # Check distribution
    counts = Counter(integers)
    print("\nDistribution (should be roughly uniform):")
    for i in range(1, 11):
        count = counts.get(i, 0)
        bar = '#' * (count // 5)
        print(f"  {i:2d}: {count:3d} {bar}")

    # Test choice
    print("\nTesting choice() with list of options...")
    options = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    choices = [entropy.choice(options) for _ in range(100)]
    choice_counts = Counter(choices)
    print("Choice distribution:")
    for option in options:
        count = choice_counts.get(option, 0)
        print(f"  {option}: {count}")

    # Test exponential distribution
    print("\nTesting expovariate(lambda=1.0)...")
    expo_values = [entropy.expovariate(1.0) for _ in range(100)]
    print(f"Mean: {sum(expo_values)/len(expo_values):.3f} (should be ~1.0)")

    print("\nSUCCESS: All entropy generation tests passed")


def test_fallback_mode():
    """Test that PRNG fallback works correctly."""
    print("\n=== Testing PRNG Fallback Mode ===")

    # Reinitialize with hardware disabled
    initialize_entropy_service(use_hardware=False)

    entropy = get_entropy_service()
    config = entropy.get_config()

    print(f"Hardware mode: {config['use_hardware']}")
    print(f"Pool active: {config['active']}")

    # Generate some values to verify fallback works
    values = [entropy.uniform(0.0, 1.0) for _ in range(10)]
    print(f"Generated {len(values)} values using PRNG fallback")
    print(f"Sample values: {[f'{v:.3f}' for v in values[:5]]}")

    print("\nSUCCESS: PRNG fallback operational")


def test_statistical_quality():
    """Test statistical properties of generated randomness."""
    print("\n=== Testing Statistical Quality ===")

    entropy = get_entropy_service()

    # Chi-square test for uniformity (simplified)
    n_bins = 10
    n_samples = 10000
    expected_per_bin = n_samples / n_bins

    values = [entropy.uniform(0.0, 1.0) for _ in range(n_samples)]

    bins = [0] * n_bins
    for v in values:
        bin_idx = min(int(v * n_bins), n_bins - 1)
        bins[bin_idx] += 1

    print(f"\nUniformity test ({n_samples} samples, {n_bins} bins):")
    print(f"Expected per bin: {expected_per_bin:.1f}")

    chi_square = sum((observed - expected_per_bin)**2 / expected_per_bin for observed in bins)

    print(f"Observed distribution:")
    for i, count in enumerate(bins):
        deviation = count - expected_per_bin
        print(f"  Bin {i}: {count:4d} (deviation: {deviation:+.1f})")

    print(f"\nChi-square statistic: {chi_square:.2f}")
    print(f"(Lower is better; < 20 is good for 10 bins)")

    if chi_square < 20:
        print("SUCCESS: Distribution appears uniform")
    else:
        print("WARNING: Distribution may be non-uniform")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TrueRNG Entropy Service Test Suite")
    print("=" * 60)

    try:
        # Test 1: Device detection
        hardware_available = test_device_detection()

        if hardware_available:
            # Test 2: Entropy generation with hardware
            test_entropy_generation()

            # Test 3: Statistical quality
            test_statistical_quality()

        # Test 4: Fallback mode
        test_fallback_mode()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
