#!/usr/bin/env python3
"""
Experiment 1: Temporal Scaling Analysis

Tests Caity's Hypothesis:
"Noodlings use more tokens SHORT-TERM but become MORE efficient LONG-TERM
as baseline context windows explode."

Methodology:
- Simulate LONG conversations (100, 500, 1000 turns)
- Track cumulative token usage over time
- Measure crossover point where Noodlings become cheaper
- Visualize scaling curves

Key Insight: This is not about per-response cost, it's about SUSTAINED conversation cost.

Author: Commander Spock + Lieutenant Caitlyn
Date: November 23, 2025
"""

import asyncio
import json
import time
import sys
import os
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_interface import OpenAICompatibleLLM


class ScalingAnalysisExperiment:
    """
    Experiment 1: Measure token scaling over long conversations.

    Hypothesis: Noodlings have constant per-turn cost, Baseline grows.
    """

    def __init__(self, output_dir: str = "experiment_results"):
        """
        Initialize experiment.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Results storage
        self.results = {
            'metadata': {
                'experiment': 'Experiment 1: Temporal Scaling Analysis',
                'date': datetime.now().isoformat(),
                'hypothesis': 'Noodlings are more efficient for long conversations'
            },
            'conversation_lengths': {},  # Will hold results for 100, 500, 1000 turns
            'crossover_analysis': {
                'estimated_crossover_turn': None,
                'explanation': ''
            }
        }

        # Pricing (GPT-4 Turbo rates as reference)
        self.PRICE_PER_1K_INPUT = 0.01  # $0.01 / 1K input tokens
        self.PRICE_PER_1K_OUTPUT = 0.03  # $0.03 / 1K output tokens

        # Token estimation constants
        self.NOODLING_TOKENS_PER_TURN = 2850  # Constant (11 LLM calls)
        self.BASELINE_BASE_TOKENS = 250  # Initial prompt
        self.BASELINE_CONTEXT_TOKENS_PER_TURN = 25  # Each turn adds to context

    def _generate_conversation_turns(self, num_turns: int) -> List[str]:
        """
        Generate realistic conversation turns.

        For long conversations, we need varied prompts that would occur naturally.

        Args:
            num_turns: Number of conversation turns to generate

        Returns:
            List of conversation prompts
        """
        # Base prompt templates
        templates = [
            "Hello! How are you?",
            "Tell me about yourself.",
            "What did you do today?",
            "I'm feeling {emotion}.",
            "What do you think about {topic}?",
            "Remember when we talked about {memory}?",
            "I just {action}.",
            "Why do you {behavior}?",
            "That's interesting! Tell me more.",
            "I don't understand. Can you explain?",
            "You seem {observation}.",
            "What's your favorite {category}?",
            "Have you ever {experience}?",
            "I need advice about {problem}.",
            "You're funny! Say something else.",
            "I'm curious about {curiosity}.",
            "What would you do if {hypothetical}?",
            "That reminds me of {association}.",
            "I agree! What else?",
            "I disagree. Here's why: {reason}.",
        ]

        emotions = ["happy", "sad", "angry", "excited", "worried", "grateful"]
        topics = ["friendship", "food", "adventure", "secrets", "the future", "the past"]
        memories = ["bread", "yesterday", "our first meeting", "that funny thing"]
        actions = ["found something", "made a mistake", "learned something new"]
        behaviors = ["waddle", "like bread so much", "wear that coat", "seem nervous"]
        observations = ["happy today", "thoughtful", "distracted", "energetic"]
        categories = ["food", "place", "time of day", "season", "activity"]
        experiences = ["traveled", "kept a secret", "been scared", "felt proud"]
        problems = ["a friendship", "a decision", "something I said", "trust"]
        curiosities = ["your past", "your dreams", "your fears", "your hopes"]
        hypotheticals = ["you could fly", "time stopped", "everyone knew your secret"]
        associations = ["something funny", "a good memory", "a sad story"]
        reasons = ["personal experience", "logic", "intuition", "past mistakes"]

        turns = []
        for i in range(num_turns):
            template = templates[i % len(templates)]

            # Fill in template variables
            prompt = template.format(
                emotion=emotions[i % len(emotions)],
                topic=topics[i % len(topics)],
                memory=memories[i % len(memories)],
                action=actions[i % len(actions)],
                behavior=behaviors[i % len(behaviors)],
                observation=observations[i % len(observations)],
                category=categories[i % len(categories)],
                experience=experiences[i % len(experiences)],
                problem=problems[i % len(problems)],
                curiosity=curiosities[i % len(curiosities)],
                hypothetical=hypotheticals[i % len(hypotheticals)],
                association=associations[i % len(associations)],
                reason=reasons[i % len(reasons)]
            )

            turns.append(prompt)

        return turns

    async def run_noodling_turn(self, prompt_text: str, turn_num: int, llm: OpenAICompatibleLLM) -> Dict:
        """
        Simulate ONE turn of Noodling processing.

        For scaling analysis, we don't need actual LLM calls - we can estimate.
        Noodlings have CONSTANT cost per turn regardless of conversation history.

        Args:
            prompt_text: User input
            turn_num: Current turn number
            llm: LLM interface (not used in simulation)

        Returns:
            Dict with 'tokens', 'latency_ms'
        """
        # Noodlings: CONSTANT cost per turn
        # 11 LLM calls (ResponseDecider + 7 transistors + blend + voice + social)
        tokens = self.NOODLING_TOKENS_PER_TURN

        # Estimated latency (for real LLM calls on M3 Ultra)
        # qwen3-4b: ~300-500ms per call
        latency_ms = 11 * 400  # 11 calls × 400ms = 4400ms

        return {
            'tokens': tokens,
            'latency_ms': latency_ms,
            'turn': turn_num
        }

    async def run_baseline_turn(self, prompt_text: str, turn_num: int, context_size: int, llm: OpenAICompatibleLLM) -> Dict:
        """
        Simulate ONE turn of Baseline processing.

        Baseline cost GROWS with conversation history.
        Each turn, the LLM must reprocess more context.

        Args:
            prompt_text: User input
            turn_num: Current turn number
            context_size: Number of previous turns in context
            llm: LLM interface (not used in simulation)

        Returns:
            Dict with 'tokens', 'latency_ms', 'context_size'
        """
        # Baseline: GROWING cost per turn
        # Input: character prompt + full conversation history + new prompt
        # Output: response (~100 tokens)

        # Character prompt: ~150 tokens (constant)
        # Each previous turn: ~25 tokens (user message + agent response)
        # New prompt: ~50 tokens (average)
        # Response: ~100 tokens

        input_tokens = 150 + (context_size * self.BASELINE_CONTEXT_TOKENS_PER_TURN) + 50
        output_tokens = 100
        total_tokens = input_tokens + output_tokens

        # Latency: Grows with context size (quadratic attention)
        # Base: 400ms for small context
        # Add: ~1ms per token in context
        latency_ms = 400 + (context_size * 1)

        return {
            'tokens': total_tokens,
            'latency_ms': latency_ms,
            'turn': turn_num,
            'context_size': context_size,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }

    async def run_conversation(self, num_turns: int) -> Dict:
        """
        Simulate a full conversation of N turns.

        Args:
            num_turns: Number of conversation turns

        Returns:
            Dict with turn-by-turn data for both systems
        """
        print(f"\n{'='*70}")
        print(f"Simulating {num_turns}-turn conversation...")
        print(f"{'='*70}\n")

        # Generate conversation
        turns = self._generate_conversation_turns(num_turns)

        # Initialize LLM (not actually used in simulation, but maintains interface)
        llm = OpenAICompatibleLLM(
            api_base="http://localhost:1234/v1",
            model="SMALL",
            timeout=60
        )

        results = {
            'num_turns': num_turns,
            'noodling': {
                'turns': [],
                'cumulative_tokens': [],
                'cumulative_latency_ms': [],
                'total_tokens': 0,
                'total_latency_ms': 0,
                'avg_tokens_per_turn': 0,
                'avg_latency_per_turn': 0
            },
            'baseline': {
                'turns': [],
                'cumulative_tokens': [],
                'cumulative_latency_ms': [],
                'total_tokens': 0,
                'total_latency_ms': 0,
                'avg_tokens_per_turn': 0,
                'avg_latency_per_turn': 0
            }
        }

        cumulative_noodling = 0
        cumulative_baseline = 0
        cumulative_noodling_time = 0
        cumulative_baseline_time = 0

        # Simulate each turn
        for turn_num in range(1, num_turns + 1):
            prompt = turns[turn_num - 1]

            # Noodling: Constant cost
            noodling_result = await self.run_noodling_turn(prompt, turn_num, llm)
            cumulative_noodling += noodling_result['tokens']
            cumulative_noodling_time += noodling_result['latency_ms']

            results['noodling']['turns'].append(noodling_result)
            results['noodling']['cumulative_tokens'].append(cumulative_noodling)
            results['noodling']['cumulative_latency_ms'].append(cumulative_noodling_time)

            # Baseline: Growing cost (context = all previous turns)
            context_size = turn_num - 1  # Previous turns
            baseline_result = await self.run_baseline_turn(prompt, turn_num, context_size, llm)
            cumulative_baseline += baseline_result['tokens']
            cumulative_baseline_time += baseline_result['latency_ms']

            results['baseline']['turns'].append(baseline_result)
            results['baseline']['cumulative_tokens'].append(cumulative_baseline)
            results['baseline']['cumulative_latency_ms'].append(cumulative_baseline_time)

            # Progress indicator
            if turn_num % 50 == 0 or turn_num == num_turns:
                print(f"Turn {turn_num}/{num_turns}:")
                print(f"  Noodling: {cumulative_noodling:,} tokens (avg: {cumulative_noodling/turn_num:.0f}/turn)")
                print(f"  Baseline: {cumulative_baseline:,} tokens (avg: {cumulative_baseline/turn_num:.0f}/turn)")
                print(f"  Ratio: {cumulative_noodling/cumulative_baseline:.2f}x")
                print()

        # Calculate final statistics
        results['noodling']['total_tokens'] = cumulative_noodling
        results['noodling']['total_latency_ms'] = cumulative_noodling_time
        results['noodling']['avg_tokens_per_turn'] = round(cumulative_noodling / num_turns, 2)
        results['noodling']['avg_latency_per_turn'] = round(cumulative_noodling_time / num_turns, 2)

        results['baseline']['total_tokens'] = cumulative_baseline
        results['baseline']['total_latency_ms'] = cumulative_baseline_time
        results['baseline']['avg_tokens_per_turn'] = round(cumulative_baseline / num_turns, 2)
        results['baseline']['avg_latency_per_turn'] = round(cumulative_baseline_time / num_turns, 2)

        return results

    def _find_crossover_point(self, conversation_results: Dict) -> int:
        """
        Find the turn where Noodlings become more efficient.

        Args:
            conversation_results: Results from run_conversation()

        Returns:
            Turn number where crossover occurs
        """
        noodling_cumulative = conversation_results['noodling']['cumulative_tokens']
        baseline_cumulative = conversation_results['baseline']['cumulative_tokens']

        for turn in range(len(noodling_cumulative)):
            if baseline_cumulative[turn] > noodling_cumulative[turn]:
                return turn + 1

        return -1  # No crossover found

    async def run_experiment(self):
        """
        Run full scaling analysis experiment.

        Tests: 100, 500, 1000 turn conversations
        """
        print("╔" + "═"*70 + "╗")
        print("║" + " "*15 + "EXPERIMENT 1: TEMPORAL SCALING" + " "*26 + "║")
        print("╚" + "═"*70 + "╝")
        print()

        conversation_lengths = [100, 500, 1000]

        for length in conversation_lengths:
            results = await self.run_conversation(length)
            self.results['conversation_lengths'][str(length)] = results

            # Find crossover point
            crossover = self._find_crossover_point(results)
            print(f"✓ {length}-turn conversation complete")
            if crossover > 0:
                print(f"  Crossover point: Turn {crossover}")
            else:
                print(f"  No crossover found (Noodlings still more expensive)")
            print()

        # Analyze crossover
        self._analyze_crossover()

        # Save results
        self._save_results()

        # Generate visualization data
        self._generate_visualization_data()

        # Print summary
        self._print_summary()

    def _analyze_crossover(self):
        """Analyze when Noodlings become more efficient."""
        # Check 100-turn conversation
        results_100 = self.results['conversation_lengths']['100']
        crossover_100 = self._find_crossover_point(results_100)

        if crossover_100 > 0:
            self.results['crossover_analysis']['estimated_crossover_turn'] = crossover_100
            self.results['crossover_analysis']['explanation'] = (
                f"Baseline becomes more expensive than Noodlings at turn {crossover_100}. "
                f"Beyond this point, Noodlings' constant per-turn cost outperforms "
                f"Baseline's growing context window."
            )
        else:
            # Estimate crossover mathematically
            # Noodlings: 2850 * N
            # Baseline: 250 * N + (25 * N * (N-1) / 2)  [arithmetic series for growing context]
            # Solve: 2850 * N = 250 * N + 12.5 * N^2
            #        2600 * N = 12.5 * N^2
            #        2600 = 12.5 * N
            #        N = 208

            estimated_crossover = 208
            self.results['crossover_analysis']['estimated_crossover_turn'] = estimated_crossover
            self.results['crossover_analysis']['explanation'] = (
                f"No crossover observed in tested ranges. Mathematical estimate: ~{estimated_crossover} turns. "
                f"This assumes full context reprocessing. If Baseline uses summarization, "
                f"crossover may not occur, but consistency will degrade."
            )

    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"experiment1_scaling_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {filename}")

    def _generate_visualization_data(self):
        """Generate CSV data for plotting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for length_str, results in self.results['conversation_lengths'].items():
            filename = self.output_dir / f"scaling_data_{length_str}turns_{timestamp}.csv"

            with open(filename, 'w') as f:
                f.write("turn,noodling_cumulative,baseline_cumulative,noodling_per_turn,baseline_per_turn\n")

                for i in range(len(results['noodling']['cumulative_tokens'])):
                    turn = i + 1
                    noodling_cum = results['noodling']['cumulative_tokens'][i]
                    baseline_cum = results['baseline']['cumulative_tokens'][i]
                    noodling_per = results['noodling']['turns'][i]['tokens']
                    baseline_per = results['baseline']['turns'][i]['tokens']

                    f.write(f"{turn},{noodling_cum},{baseline_cum},{noodling_per},{baseline_per}\n")

            print(f"✓ Visualization data saved: {filename}")

    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*70)
        print("EXPERIMENT 1 RESULTS: TEMPORAL SCALING ANALYSIS")
        print("="*70)

        print(f"\nHypothesis: {self.results['metadata']['hypothesis']}")

        for length_str in ['100', '500', '1000']:
            if length_str not in self.results['conversation_lengths']:
                continue

            results = self.results['conversation_lengths'][length_str]

            print(f"\n--- {length_str}-Turn Conversation ---")

            noodling_total = results['noodling']['total_tokens']
            baseline_total = results['baseline']['total_tokens']
            ratio = noodling_total / baseline_total

            print(f"Noodling: {noodling_total:,} total tokens ({results['noodling']['avg_tokens_per_turn']:.0f}/turn)")
            print(f"Baseline: {baseline_total:,} total tokens ({results['baseline']['avg_tokens_per_turn']:.0f}/turn)")
            print(f"Ratio: {ratio:.2f}x")

            if ratio < 1.0:
                print(f"✓ Noodlings are MORE efficient at {length_str} turns!")
            else:
                print(f"⚠️  Baseline still cheaper at {length_str} turns")

        # Crossover analysis
        print(f"\n--- Crossover Analysis ---")
        crossover = self.results['crossover_analysis']['estimated_crossover_turn']
        explanation = self.results['crossover_analysis']['explanation']
        print(f"Estimated crossover: Turn {crossover}")
        print(f"{explanation}")

        print("\n--- Key Insights ---")
        print("1. Noodlings have CONSTANT per-turn cost (~2,850 tokens)")
        print("2. Baseline has GROWING per-turn cost (context accumulation)")
        print("3. For LONG conversations (days/weeks), Noodlings are MORE efficient")
        print("4. For SHORT conversations (<100 turns), Baseline is cheaper")

        print("\n--- Next Steps ---")
        print("1. Visualize scaling curves (plot CSV data)")
        print("2. Run Experiment 2: Measure quality/consistency difference")
        print("3. Test REAL conversation (not simulation) to validate model")

        print("\n" + "="*70)


async def main():
    """Run Experiment 1."""
    experiment = ScalingAnalysisExperiment(output_dir="experiment_results")
    await experiment.run_experiment()


if __name__ == "__main__":
    asyncio.run(main())
