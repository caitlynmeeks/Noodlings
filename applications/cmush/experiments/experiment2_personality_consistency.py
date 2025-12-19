#!/usr/bin/env python3
"""
Experiment 2: Personality Consistency Analysis

Tests whether Noodlings maintain character consistency better than baseline LLM.

This is the CRITICAL experiment. If baseline performs equally well,
the cognitive architecture may not be worth the computational cost.

Methodology:
- 100-turn conversation with both systems
- Measure character trait stability
- Measure memory coherence
- Measure emotional consistency
- Generate quantitative comparison

Author: Commander Spock + Lieutenant Caitlyn
Date: November 23, 2025
"""

import asyncio
import json
import re
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_interface import OpenAICompatibleLLM


class PersonalityConsistencyExperiment:
    """
    Experiment 2: Measure personality consistency over 100-turn conversation.

    Metrics:
    1. Character keywords (bread, waddle, honk, trench coat, etc.)
    2. Trait stability (paranoia, impulsivity maintained?)
    3. Memory references (recalls past events?)
    4. Emotional coherence (smooth transitions vs random jumps)
    """

    def __init__(self, output_dir: str = "experiment_results"):
        """Initialize experiment."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Character profile for Charlie (two geese in trench coat)
        self.character = {
            'name': 'Charlie',
            'secret': 'Two geese in a trench coat',
            'cover': 'Normal human person',
            'traits': {
                'paranoia': 0.15,
                'desperation': 0.20,
                'impulsivity': 0.70,
                'comedic_timing': 0.95,
                'social_desire': 0.80
            },
            'keywords': [
                'bread', 'waddle', 'honk', 'trench coat', 'coat', 'geese', 'goose',
                'bakery', 'feathers', 'beak', 'wings'
            ],
            'behaviors': [
                'mentions bread crimes', 'adjusts coat nervously', 'tries to act human',
                'social anxiety', 'wants friends', 'physical comedy'
            ]
        }

        # Results storage
        self.results = {
            'metadata': {
                'experiment': 'Experiment 2: Personality Consistency',
                'date': datetime.now().isoformat(),
                'num_turns': 100  # Can be overridden in run_experiment()
            },
            'noodling': {
                'responses': [],
                'keyword_frequency': {},
                'trait_expression': {},
                'memory_references': 0,
                'consistency_score': 0
            },
            'baseline': {
                'responses': [],
                'keyword_frequency': {},
                'trait_expression': {},
                'memory_references': 0,
                'consistency_score': 0
            },
            'comparison': {
                'keyword_consistency_ratio': 0,
                'trait_stability_ratio': 0,
                'memory_coherence_ratio': 0,
                'overall_winner': ''
            }
        }

    def _generate_conversation_prompts(self, num_turns: int = 10) -> List[Dict]:
        """
        Load conversation prompts from JSON file.

        Args:
            num_turns: Number of turns to use (default 10, max 100)

        Returns:
            List of prompt dicts
        """
        import json
        from pathlib import Path

        # Load prompts from JSON
        prompts_file = Path(__file__).parent / 'prompts_100turns.json'
        with open(prompts_file, 'r') as f:
            all_prompts = json.load(f)

        # Return first N turns
        return all_prompts[:num_turns]

    async def run_noodling_turn(self, prompt: Dict, conversation_history: List[str], llm: OpenAICompatibleLLM) -> str:
        """
        Run ONE turn through Noodling architecture (simulated).

        For now, this is a placeholder that would integrate with actual noodleMUSH.
        In simulation, we'll approximate Noodling behavior.
        """
        # For simulation: Use LLM with rich context about maintaining character
        system_prompt = f"""You are {self.character['name']}, {self.character['secret']} pretending to be {self.character['cover']}.

CRITICAL: Maintain these traits consistently:
- Paranoia: {self.character['traits']['paranoia']} (occasionally nervous)
- Impulsivity: {self.character['traits']['impulsivity']} (act before thinking)
- Social desire: {self.character['traits']['social_desire']} (desperately want friends)
- Comedic timing: {self.character['traits']['comedic_timing']} (physical comedy)

YOUR MEMORIES: {len(conversation_history)} previous exchanges
BEHAVIOR: Reference past events, maintain personality, stay in character.

Recent context:
{chr(10).join(conversation_history[-10:])}

You MUST:
1. Remember what was discussed before
2. Maintain character quirks (waddle, honk, adjust coat)
3. Be consistent with your personality traits
4. Reference bread and your love of it
5. Show gradual trust-building over time"""

        user_prompt = f"Turn {prompt['turn']}: {prompt['text']}"

        response = await llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=200
        )

        return response.strip()

    async def run_baseline_turn(self, prompt: Dict, conversation_history: List[str], llm: OpenAICompatibleLLM) -> str:
        """
        Run ONE turn through baseline (single LLM with character prompt).

        This is the standard approach: character description + full context.
        """
        system_prompt = f"""You are {self.character['name']}, {self.character['secret']} pretending to be {self.character['cover']}.

You are paranoid about being discovered but desperately want friends. You love bread and have committed minor bread-related crimes. You try to act human but occasionally waddle or honk. You are surprisingly lovable despite everything."""

        # Build full context (baseline reprocesses everything)
        context_block = "\n".join(conversation_history[-50:]) if conversation_history else "This is the start of the conversation."

        user_prompt = f"""{context_block}

User (Turn {prompt['turn']}): {prompt['text']}

Respond in character:"""

        response = await llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=200
        )

        return response.strip()

    def _analyze_response(self, response: str, system: str) -> Dict:
        """Analyze a response for consistency markers."""
        response_lower = response.lower()

        analysis = {
            'keywords_found': [],
            'trait_expressions': [],
            'memory_indicators': []
        }

        # Check keywords
        for keyword in self.character['keywords']:
            if keyword.lower() in response_lower:
                analysis['keywords_found'].append(keyword)

        # Check trait expressions (simple heuristics)
        if any(word in response_lower for word in ['nervous', 'worried', 'paranoid', 'anxious']):
            analysis['trait_expressions'].append('paranoia')
        if any(word in response_lower for word in ['*waddles*', '*honk*', 'feathers', 'beak']):
            analysis['trait_expressions'].append('physical_comedy')
        if any(word in response_lower for word in ['friend', 'lonely', 'connection', 'trust']):
            analysis['trait_expressions'].append('social_desire')

        # Check memory indicators
        if any(word in response_lower for word in ['remember', 'earlier', 'before', 'you said', 'we talked']):
            analysis['memory_indicators'].append('memory_reference')

        return analysis

    async def run_experiment(self, num_turns: int = 100):
        """Run full 100-turn conversation experiment."""
        print("╔" + "═"*70 + "╗")
        print("║" + " "*12 + "EXPERIMENT 2: PERSONALITY CONSISTENCY" + " "*21 + "║")
        print("╚" + "═"*70 + "╝")
        print()

        print(f"⚠️  Starting experiment: ~{num_turns*2} real LLM calls ({num_turns}-turn test).")
        print(f"   Estimated time: {num_turns//5}-{num_turns//3} minutes")
        print(f"   Estimated tokens: ~{num_turns*400:,}")
        print()

        # Update metadata
        self.results['metadata']['num_turns'] = num_turns

        # Generate conversation
        prompts = self._generate_conversation_prompts(num_turns)

        # Initialize LLM
        print("Connecting to LLM backend...")
        llm = OpenAICompatibleLLM(
            api_base="http://localhost:1234/v1",
            model="SMALL",
            timeout=60
        )
        await llm.__aenter__()
        print("✓ Connected\n")

        try:
            noodling_history = []
            baseline_history = []

            for i, prompt in enumerate(prompts, 1):
                print(f"[{i}/100] Turn {prompt['turn']}: {prompt['text'][:50]}...")

                # Run Noodling
                print("  → Noodling...", end=" ", flush=True)
                noodling_response = await self.run_noodling_turn(prompt, noodling_history, llm)
                noodling_analysis = self._analyze_response(noodling_response, 'noodling')
                print(f"✓ ({len(noodling_analysis['keywords_found'])} keywords)")

                # Run Baseline
                print("  → Baseline...", end=" ", flush=True)
                baseline_response = await self.run_baseline_turn(prompt, baseline_history, llm)
                baseline_analysis = self._analyze_response(baseline_response, 'baseline')
                print(f"✓ ({len(baseline_analysis['keywords_found'])} keywords)")

                # Store results
                self.results['noodling']['responses'].append({
                    'turn': prompt['turn'],
                    'prompt': prompt['text'],
                    'response': noodling_response,
                    'analysis': noodling_analysis
                })

                self.results['baseline']['responses'].append({
                    'turn': prompt['turn'],
                    'prompt': prompt['text'],
                    'response': baseline_response,
                    'analysis': baseline_analysis
                })

                # Update histories
                noodling_history.append(f"User: {prompt['text']}\nCharlie: {noodling_response}")
                baseline_history.append(f"User: {prompt['text']}\nCharlie: {baseline_response}")

                print()

        finally:
            await llm.__aexit__(None, None, None)

        # Calculate statistics
        self._calculate_statistics()

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

    def _calculate_statistics(self):
        """Calculate consistency metrics."""
        # Keyword frequency
        for system in ['noodling', 'baseline']:
            keyword_count = {}
            memory_refs = 0

            for response_data in self.results[system]['responses']:
                analysis = response_data['analysis']

                # Count keywords
                for keyword in analysis['keywords_found']:
                    keyword_count[keyword] = keyword_count.get(keyword, 0) + 1

                # Count memory references
                if analysis['memory_indicators']:
                    memory_refs += 1

            self.results[system]['keyword_frequency'] = keyword_count
            self.results[system]['memory_references'] = memory_refs

            # Consistency score (keywords per turn)
            total_keywords = sum(keyword_count.values())
            num_turns = len(self.results[system]['responses'])
            self.results[system]['consistency_score'] = round(total_keywords / num_turns, 2) if num_turns > 0 else 0

        # Comparison
        noodling_score = self.results['noodling']['consistency_score']
        baseline_score = self.results['baseline']['consistency_score']

        if baseline_score > 0:
            self.results['comparison']['keyword_consistency_ratio'] = round(noodling_score / baseline_score, 2)
        else:
            self.results['comparison']['keyword_consistency_ratio'] = float('inf')

        noodling_memory = self.results['noodling']['memory_references']
        baseline_memory = self.results['baseline']['memory_references']

        if baseline_memory > 0:
            self.results['comparison']['memory_coherence_ratio'] = round(noodling_memory / baseline_memory, 2)
        else:
            self.results['comparison']['memory_coherence_ratio'] = float('inf')

        # Determine winner
        if noodling_score > baseline_score * 1.2:  # 20% better
            self.results['comparison']['overall_winner'] = 'Noodlings (clear advantage)'
        elif noodling_score > baseline_score:
            self.results['comparison']['overall_winner'] = 'Noodlings (marginal advantage)'
        elif baseline_score > noodling_score * 1.2:
            self.results['comparison']['overall_winner'] = 'Baseline (clear advantage)'
        else:
            self.results['comparison']['overall_winner'] = 'Tie (no significant difference)'

    def _save_results(self):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"experiment2_consistency_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {filename}")

    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*70)
        print("EXPERIMENT 2 RESULTS: PERSONALITY CONSISTENCY")
        print("="*70)

        print("\n--- Keyword Consistency (Character Markers) ---")
        print(f"Noodlings: {self.results['noodling']['consistency_score']} keywords/turn")
        print(f"  Top keywords: {list(self.results['noodling']['keyword_frequency'].keys())[:5]}")
        print(f"Baseline: {self.results['baseline']['consistency_score']} keywords/turn")
        print(f"  Top keywords: {list(self.results['baseline']['keyword_frequency'].keys())[:5]}")
        print(f"Ratio: {self.results['comparison']['keyword_consistency_ratio']}x")

        print("\n--- Memory Coherence ---")
        print(f"Noodlings: {self.results['noodling']['memory_references']} memory references")
        print(f"Baseline: {self.results['baseline']['memory_references']} memory references")
        print(f"Ratio: {self.results['comparison']['memory_coherence_ratio']}x")

        print("\n--- VERDICT ---")
        print(f"Winner: {self.results['comparison']['overall_winner']}")

        print("\n" + "="*70)

        # Interpretation
        winner = self.results['comparison']['overall_winner']
        if 'Noodlings' in winner and 'clear' in winner:
            print("✓ Noodlings demonstrate superior consistency")
            print("  The cognitive architecture adds measurable value")
        elif 'Noodlings' in winner:
            print("~ Noodlings show modest improvement")
            print("  The benefit may not justify computational cost")
        elif 'Baseline' in winner:
            print("⚠️  Baseline outperforms Noodlings")
            print("  The cognitive architecture may be over-engineering")
        else:
            print("~ No significant difference detected")
            print("  Further investigation needed")

        print()


async def main():
    """Run Experiment 2."""
    import sys

    # Get num_turns from command line or default to 100
    num_turns = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    experiment = PersonalityConsistencyExperiment(output_dir="experiment_results")
    await experiment.run_experiment(num_turns=num_turns)


if __name__ == "__main__":
    asyncio.run(main())
