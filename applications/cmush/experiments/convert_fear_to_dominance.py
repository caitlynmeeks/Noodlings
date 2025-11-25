#!/usr/bin/env python3
"""
Convert existing dataset from fear dimension to dominance dimension.

Fear and dominance are inversely related:
- High fear (0.9) -> Low dominance (0.1)
- Low fear (0.1) -> High dominance (0.9)

dominance = 1.0 - fear
"""

import json
from pathlib import Path

def convert_dataset(input_path, output_path):
    """Convert fear to dominance in dataset."""

    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} examples")

    # Convert each example
    for example in data:
        fear_value = example['affect']['fear']
        # Inverse relationship: fear -> dominance
        dominance_value = 1.0 - fear_value

        # Replace fear with dominance
        del example['affect']['fear']
        example['affect']['dominance'] = dominance_value

    # Save converted dataset
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Saved {len(data)} examples")

    # Show sample
    print("\nSample conversion (first example):")
    sample = data[0]
    print(f"  Emotion: {sample['emotion']}")
    print(f"  Affect vector:")
    for dim, val in sample['affect'].items():
        print(f"    {dim:10s}: {val:+.2f}")

if __name__ == '__main__':
    base_dir = Path(__file__).parent

    print("=" * 70)
    print("CONVERTING FEAR -> DOMINANCE")
    print("=" * 70)
    print()

    # Convert all three files
    for split in ['dataset', 'train', 'val']:
        input_file = base_dir / f'emotion_synthetic_{split}.json'
        output_file = base_dir / f'emotion_synthetic_{split}.json'

        if input_file.exists():
            convert_dataset(input_file, output_file)
            print()

    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
