#!/usr/bin/env python3
"""
Validate synthetic emotion dataset quality.

Checks:
1. Balance across emotion classes
2. Affect vector ranges and consistency
3. Response quality (length, format)
4. No duplicates
"""

import json
from pathlib import Path
import statistics

def validate_dataset(dataset_path):
    """Run validation checks on synthetic dataset."""

    print("=" * 70)
    print("SYNTHETIC DATASET VALIDATION")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")

    with open(dataset_path) as f:
        data = json.load(f)

    print(f"Total examples: {len(data)}")

    # 1. Emotion balance
    print("\n" + "=" * 70)
    print("1. EMOTION CLASS BALANCE")
    print("=" * 70)

    emotion_counts = {}
    for item in data:
        emotion = item['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    emotions = sorted(emotion_counts.keys())
    for emotion in emotions:
        count = emotion_counts[emotion]
        pct = 100 * count / len(data)
        print(f"  {emotion:12s}: {count:4d} ({pct:5.1f}%)")

    # Check balance (should be within 1% of 10%)
    balance_check = all(abs(count/len(data) - 0.1) < 0.01 for count in emotion_counts.values())
    print(f"\n  Balance check: {'PASS' if balance_check else 'FAIL'}")

    # 2. Affect vector validation
    print("\n" + "=" * 70)
    print("2. AFFECT VECTOR VALIDATION")
    print("=" * 70)

    affect_stats = {
        'valence': {'min': [], 'max': [], 'mean': []},
        'arousal': {'min': [], 'max': [], 'mean': []},
        'fear': {'min': [], 'max': [], 'mean': []},
        'sorrow': {'min': [], 'max': [], 'mean': []},
        'boredom': {'min': [], 'max': [], 'mean': []}
    }

    for emotion in emotions:
        emotion_items = [x for x in data if x['emotion'] == emotion]

        for dim in affect_stats.keys():
            values = [item['affect'][dim] for item in emotion_items]
            affect_stats[dim]['min'].append((emotion, min(values)))
            affect_stats[dim]['max'].append((emotion, max(values)))
            affect_stats[dim]['mean'].append((emotion, statistics.mean(values)))

    print("\n  Valence ranges by emotion:")
    for emotion, val in sorted(affect_stats['valence']['mean'], key=lambda x: x[1]):
        print(f"    {emotion:12s}: {val:+.2f} (mean)")

    print("\n  Arousal ranges by emotion:")
    for emotion, val in sorted(affect_stats['arousal']['mean'], key=lambda x: -x[1]):
        print(f"    {emotion:12s}: {val:.2f} (mean)")

    # Check valence ranges make sense
    positive_emotions = ['joy', 'love', 'pride']
    negative_emotions = ['fear', 'sadness', 'anger', 'guilt', 'shame']

    pos_valences = [val for emo, val in affect_stats['valence']['mean'] if emo in positive_emotions]
    neg_valences = [val for emo, val in affect_stats['valence']['mean'] if emo in negative_emotions]

    valence_check = statistics.mean(pos_valences) > 0.5 and statistics.mean(neg_valences) < -0.5
    print(f"\n  Valence polarity check: {'PASS' if valence_check else 'FAIL'}")
    print(f"    Positive emotions mean: {statistics.mean(pos_valences):+.2f}")
    print(f"    Negative emotions mean: {statistics.mean(neg_valences):+.2f}")

    # 3. Response quality
    print("\n" + "=" * 70)
    print("3. RESPONSE QUALITY")
    print("=" * 70)

    response_lengths = [len(item['response']) for item in data]
    print(f"\n  Response length statistics:")
    print(f"    Min:    {min(response_lengths)} chars")
    print(f"    Max:    {max(response_lengths)} chars")
    print(f"    Mean:   {statistics.mean(response_lengths):.1f} chars")
    print(f"    Median: {statistics.median(response_lengths):.1f} chars")

    # Check format (should contain agent action)
    has_says = sum(1 for item in data if 'says,' in item['response'])
    has_thinks = sum(1 for item in data if 'thinks,' in item['response'])
    has_action = has_says + has_thinks

    format_pct = 100 * has_action / len(data)
    print(f"\n  Format check (has 'says' or 'thinks'):")
    print(f"    {has_action}/{len(data)} ({format_pct:.1f}%)")

    format_check = format_pct > 90
    print(f"    Format check: {'PASS' if format_check else 'FAIL'}")

    # 4. Duplicates
    print("\n" + "=" * 70)
    print("4. DUPLICATE CHECK")
    print("=" * 70)

    responses = [item['response'] for item in data]
    unique_responses = set(responses)
    duplicates = len(responses) - len(unique_responses)

    print(f"\n  Total responses: {len(responses)}")
    print(f"  Unique responses: {len(unique_responses)}")
    print(f"  Duplicates: {duplicates}")

    duplicate_check = duplicates == 0
    print(f"  Duplicate check: {'PASS' if duplicate_check else 'FAIL'}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_checks = [balance_check, valence_check, format_check, duplicate_check]
    passed = sum(all_checks)

    print(f"\n  Checks passed: {passed}/{len(all_checks)}")
    print(f"  Overall: {'PASS' if all(all_checks) else 'FAIL'}")
    print()


def main():
    dataset_path = Path(__file__).parent / 'emotion_synthetic_dataset.json'
    validate_dataset(dataset_path)


if __name__ == '__main__':
    main()
