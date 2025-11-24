#!/usr/bin/env python3
"""
Test Experiment 1 with small batch (3 prompts).

Verifies:
- LLM connectivity
- Token tracking
- Timing measurement
- Result collection

Author: Commander Spock
Date: November 23, 2025
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment1_computational_cost import ComputationalCostExperiment


async def main():
    """Run small test batch."""
    print("Testing Experiment 1 with 3 prompts...\n")

    experiment = ComputationalCostExperiment(output_dir="experiment_results")

    # Run with just 3 prompts
    await experiment.run_experiment(num_prompts=3)

    print("\n✓ Test complete! If no errors, ready for full 100-prompt run.")


if __name__ == "__main__":
    asyncio.run(main())
