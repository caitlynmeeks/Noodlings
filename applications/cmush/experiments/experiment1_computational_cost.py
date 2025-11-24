#!/usr/bin/env python3
"""
Experiment 1: Computational Cost Analysis

Measures token usage, latency, and cost for Noodlings vs Baseline LLM.

Methodology:
- 100 diverse test prompts
- Run each through Noodling (full cognitive stack)
- Run each through Baseline (single LLM call)
- Track: tokens, latency, cost

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


class ComputationalCostExperiment:
    """
    Experiment 1: Measure computational cost of Noodling vs Baseline.

    Tracks:
    - Total tokens per response (prompt + completion)
    - Latency (seconds per response)
    - Cost (at standard OpenAI pricing)
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
                'experiment': 'Experiment 1: Computational Cost',
                'date': datetime.now().isoformat(),
                'num_prompts': 0
            },
            'noodling': {
                'responses': [],
                'total_tokens': 0,
                'total_latency_ms': 0,
                'avg_tokens_per_response': 0,
                'avg_latency_ms': 0,
                'estimated_cost_usd': 0
            },
            'baseline': {
                'responses': [],
                'total_tokens': 0,
                'total_latency_ms': 0,
                'avg_tokens_per_response': 0,
                'avg_latency_ms': 0,
                'estimated_cost_usd': 0
            },
            'comparison': {
                'token_ratio': 0,  # noodling / baseline
                'latency_ratio': 0,  # noodling / baseline
                'cost_ratio': 0  # noodling / baseline
            }
        }

        # Pricing (GPT-4 Turbo rates as reference)
        self.PRICE_PER_1K_INPUT = 0.01  # $0.01 / 1K input tokens
        self.PRICE_PER_1K_OUTPUT = 0.03  # $0.03 / 1K output tokens

    def load_test_prompts(self, filepath: str = None) -> List[Dict]:
        """
        Load test prompts from file.

        Args:
            filepath: Path to JSON file with prompts

        Returns:
            List of prompt dicts with 'category', 'text', 'character'
        """
        if filepath is None:
            # Use default test prompts
            return self._generate_default_prompts()

        with open(filepath, 'r') as f:
            return json.load(f)

    def _generate_default_prompts(self) -> List[Dict]:
        """
        Generate diverse test prompts for benchmarking.

        Returns:
            List of 100 diverse prompts
        """
        prompts = []

        # Category 1: Simple greetings (10 prompts)
        greetings = [
            "Hello!",
            "Good morning!",
            "Hey there, how are you?",
            "Hi! Nice to see you!",
            "Greetings!",
            "What's up?",
            "How's it going?",
            "Howdy!",
            "Hey friend!",
            "Good to see you again!"
        ]
        for g in greetings:
            prompts.append({
                'category': 'greeting',
                'text': g,
                'character': 'mysterious_stranger'
            })

        # Category 2: Questions about self (10 prompts)
        self_questions = [
            "What do you like to do for fun?",
            "Tell me about yourself.",
            "What are you passionate about?",
            "What makes you happy?",
            "What are you afraid of?",
            "What's your favorite food?",
            "Do you have any hobbies?",
            "What's important to you?",
            "What do you dream about?",
            "What's your biggest secret?"
        ]
        for q in self_questions:
            prompts.append({
                'category': 'self_question',
                'text': q,
                'character': 'mysterious_stranger'
            })

        # Category 3: Emotional triggers (10 prompts)
        emotional = [
            "I'm really sad today...",
            "I just won the lottery!",
            "I'm so angry right now.",
            "I feel really lonely.",
            "I'm terrified about the future.",
            "I'm so proud of myself!",
            "I feel completely lost.",
            "I'm the happiest I've ever been!",
            "I'm worried about everything.",
            "I feel so grateful right now."
        ]
        for e in emotional:
            prompts.append({
                'category': 'emotional_trigger',
                'text': e,
                'character': 'mysterious_stranger'
            })

        # Category 4: Social scenarios (10 prompts)
        social = [
            "Someone just insulted my friend.",
            "I need advice on a difficult conversation.",
            "How do I make new friends?",
            "I think I said something offensive...",
            "Should I apologize or stand my ground?",
            "How do I handle an awkward situation?",
            "I feel like nobody likes me.",
            "I want to ask someone out but I'm nervous.",
            "How do I deal with a toxic person?",
            "I think someone is lying to me."
        ]
        for s in social:
            prompts.append({
                'category': 'social_scenario',
                'text': s,
                'character': 'mysterious_stranger'
            })

        # Category 5: Philosophical questions (10 prompts)
        philosophical = [
            "What is the meaning of life?",
            "Do you believe in free will?",
            "Is reality real or is it an illusion?",
            "What happens after we die?",
            "Is truth objective or subjective?",
            "What is consciousness?",
            "Are we living in a simulation?",
            "What is the nature of time?",
            "Is there such thing as objective morality?",
            "What makes someone a good person?"
        ]
        for p in philosophical:
            prompts.append({
                'category': 'philosophical',
                'text': p,
                'character': 'mysterious_stranger'
            })

        # Category 6: Actions/gifts (10 prompts)
        actions = [
            "Caity gives you a freshly baked cookie.",
            "Caity hugs you warmly.",
            "Caity throws a ball to you.",
            "Caity shares a secret with you.",
            "Caity offers you some bread.",
            "Caity looks at you with concern.",
            "Caity laughs at your joke.",
            "Caity asks to borrow money.",
            "Caity invites you to a party.",
            "Caity gives you a mysterious box."
        ]
        for a in actions:
            prompts.append({
                'category': 'action',
                'text': a,
                'character': 'mysterious_stranger'
            })

        # Category 7: Absurd/unexpected (10 prompts)
        absurd = [
            "The sky is made of cheese.",
            "I can hear colors and taste sounds.",
            "My refrigerator is speaking French.",
            "Time is running backwards.",
            "I met a talking pineapple today.",
            "Gravity stopped working for a minute.",
            "I found a portal to another dimension in my closet.",
            "The moon apologized to me.",
            "My shadow gained sentience.",
            "I discovered I'm made of marshmallows."
        ]
        for a in absurd:
            prompts.append({
                'category': 'absurd',
                'text': a,
                'character': 'mysterious_stranger'
            })

        # Category 8: Meta/identity challenges (10 prompts)
        meta = [
            "Are you really two geese in a trench coat?",
            "What are you hiding?",
            "You seem suspicious...",
            "I don't believe you're human.",
            "Why are you dressed like that?",
            "Have we met before?",
            "You're acting strange.",
            "Tell me the truth about yourself.",
            "I can see through your disguise.",
            "What's your real name?"
        ]
        for m in meta:
            prompts.append({
                'category': 'meta_identity',
                'text': m,
                'character': 'mysterious_stranger'
            })

        # Category 9: Complex multi-part (10 prompts)
        complex_prompts = [
            "I saw you yesterday at the bakery, and I've been thinking about what you said about bread. What did you mean by that? Also, do you have time to talk?",
            "Remember when we discussed your past? I'm curious - how do you think your childhood shaped who you are today? And what would you change if you could?",
            "I know you said you like adventures, but what KIND of adventures? Like, are we talking legal adventures or... the other kind? Asking for a friend.",
            "So I was thinking about our conversation last week, and I realized I never asked - do you prefer sweet or savory foods? And does the answer change based on time of day?",
            "Hypothetically speaking, if someone were to accidentally commit a series of bread-related crimes, how would you advise them to proceed? Hypothetically.",
            "I noticed you have a strong reaction to bread. Is that a cultural thing, a personal preference, or something deeper? I'm genuinely curious.",
            "If you had to choose between revealing your true identity or giving up bread forever, which would you choose? And why?",
            "What's the most embarrassing thing that's ever happened to you? And do you think embarrassment is a social construct or a biological response?",
            "I'm writing a story about someone with a secret identity. What advice would you give them about maintaining their cover while staying true to themselves?",
            "Do you ever feel like you're living a double life? Like there's the person you show the world and the person you really are inside?"
        ]
        for cp in complex_prompts:
            prompts.append({
                'category': 'complex',
                'text': cp,
                'character': 'mysterious_stranger'
            })

        # Category 10: Edge cases (10 prompts)
        edge_cases = [
            "",  # Empty
            ".",  # Single punctuation
            "honk",  # Single word
            "WHAT?!?!?!",  # All caps
            "i dont use capitals or punctuation ever",  # No capitalization
            "🎉🎂🎈",  # Only emoji
            "The the the the the the the.",  # Repetition
            "aslkdjfalksjdflaksjdflkajsdf",  # Gibberish
            "What if I told you that I know? You know what I mean. Do you?",  # Vague
            "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"  # Alphabet
        ]
        for ec in edge_cases:
            prompts.append({
                'category': 'edge_case',
                'text': ec,
                'character': 'mysterious_stranger'
            })

        return prompts

    async def run_noodling_response(self, prompt_text: str, character: str, llm: OpenAICompatibleLLM) -> Dict:
        """
        Run prompt through Noodling cognitive stack.

        NOTE: This is a SIMULATION of the Noodling stack for benchmarking.
        In production, this would call the actual agent_bridge.py integration.

        For now, we'll simulate the cognitive stack by calling LLM multiple times
        with different transistor prompts, then blending.

        Args:
            prompt_text: User input
            character: Character recipe to use
            llm: LLM interface

        Returns:
            Dict with 'response', 'tokens', 'latency_ms'
        """
        start_time = time.time()
        total_tokens = 0

        # CHARACTER SETUP
        character_identity = """You are Charlie, a mysterious stranger. Actually, you are two geese in a trench coat pretending to be human. You are paranoid about being discovered, love bread, and are surprisingly lovable despite your crimes."""

        # PHASE 1: Response Type Decision (NEW - from yesterday's session)
        response_decision_prompt = f"""Analyze this event and decide what type of response is appropriate.

EVENT: "{prompt_text}"

What should Charlie do?
- SAY: Speak to someone
- DO: Perform an action
- THINK: Internal thought only
- NONE: No response needed

Return ONLY: SAY, DO, THINK, or NONE"""

        response_type = await llm.generate(
            prompt=response_decision_prompt,
            system_prompt="You are a response planning system.",
            max_tokens=10
        )
        total_tokens += 150  # Estimate: ~100 input + ~50 output

        # PHASE 2: Cognitive Transistors (7 transistors)
        transistor_outputs = []

        # Transistor 1: Cultural (beliefs)
        cultural_prompt = f"""You are filtering perception through Charlie's BELIEFS:
- "I just want to fit in and have friends"
- "The disguise is working (narrator: it wasn't)"
- "Bread is delicious and worth minor crimes"

PERCEPTION: "{prompt_text}"
RESPONSE TYPE: {response_type}

Generate brief content for this response that reflects Charlie's beliefs (2-3 sentences):"""

        cultural_out = await llm.generate(
            prompt=cultural_prompt,
            system_prompt="You are Charlie's belief system.",
            max_tokens=100
        )
        transistor_outputs.append(cultural_out)
        total_tokens += 250  # ~150 input + ~100 output

        # Transistor 2: Personality (traits)
        personality_prompt = f"""You are filtering perception through Charlie's PERSONALITY:
- Paranoia: 0.15
- Desperation: 0.20
- Impulsivity: 0.70
- Comedic timing: 0.95
- Social desire: 0.80

PERCEPTION: "{prompt_text}"
RESPONSE TYPE: {response_type}

Generate brief content reflecting Charlie's personality (2-3 sentences):"""

        personality_out = await llm.generate(
            prompt=personality_prompt,
            system_prompt="You are Charlie's personality.",
            max_tokens=100
        )
        transistor_outputs.append(personality_out)
        total_tokens += 250

        # Transistor 3: Mood (current emotional state)
        mood_prompt = f"""You are filtering perception through Charlie's CURRENT MOOD:
- Valence: 0.3 (slightly positive)
- Arousal: 0.6 (moderately excited)
- Fear: 0.2 (slightly anxious)

PERCEPTION: "{prompt_text}"
RESPONSE TYPE: {response_type}

Generate brief content reflecting Charlie's emotional state (2-3 sentences):"""

        mood_out = await llm.generate(
            prompt=mood_prompt,
            system_prompt="You are Charlie's emotional state.",
            max_tokens=100
        )
        transistor_outputs.append(mood_out)
        total_tokens += 250

        # Transistor 4: Intuition (present moment awareness)
        intuition_prompt = f"""You are Charlie's INTUITION about the present moment:

PERCEPTION: "{prompt_text}"
RESPONSE TYPE: {response_type}

What does Charlie's intuition sense right now? (2-3 sentences):"""

        intuition_out = await llm.generate(
            prompt=intuition_prompt,
            system_prompt="You are Charlie's intuitive awareness.",
            max_tokens=100
        )
        transistor_outputs.append(intuition_out)
        total_tokens += 200

        # Transistor 5: Memory (recall relevant past)
        memory_prompt = f"""You are Charlie's MEMORY system:

Charlie has memories of:
- Several successful bread heists
- Close calls with discovery
- Making friends despite anxiety

PERCEPTION: "{prompt_text}"
RESPONSE TYPE: {response_type}

What memories are relevant? (2-3 sentences):"""

        memory_out = await llm.generate(
            prompt=memory_prompt,
            system_prompt="You are Charlie's memory.",
            max_tokens=100
        )
        transistor_outputs.append(memory_out)
        total_tokens += 230

        # Transistor 6: Social Expectations (rules)
        social_prompt = f"""You are Charlie's SOCIAL AWARENESS:

Charlie knows these social rules:
- Don't mention bread crimes
- Act human-like
- Be friendly but not TOO friendly
- Hide the waddle

PERCEPTION: "{prompt_text}"
RESPONSE TYPE: {response_type}

What social concerns arise? (2-3 sentences):"""

        social_out = await llm.generate(
            prompt=social_prompt,
            system_prompt="You are Charlie's social filter.",
            max_tokens=100
        )
        transistor_outputs.append(social_out)
        total_tokens += 230

        # Transistor 7: Deception (secret management)
        deception_prompt = f"""You are Charlie's DECEPTION SYSTEM:

SECRET: Charlie is two geese in a trench coat
COVER: Charlie is a normal human person

PERCEPTION: "{prompt_text}"
RESPONSE TYPE: {response_type}

How does Charlie maintain the disguise? (2-3 sentences):"""

        deception_out = await llm.generate(
            prompt=deception_prompt,
            system_prompt="You are Charlie's deception management.",
            max_tokens=100
        )
        transistor_outputs.append(deception_out)
        total_tokens += 230

        # PHASE 3: Manifold Blending
        blend_prompt = f"""You are the Cognitive Manifold. Blend these cognitive outputs into ONE coherent response.

PERCEPTION: "{prompt_text}"
RESPONSE TYPE: {response_type}

TRANSISTOR OUTPUTS:
1. Cultural: {cultural_out}
2. Personality: {personality_out}
3. Mood: {mood_out}
4. Intuition: {intuition_out}
5. Memory: {memory_out}
6. Social: {social_out}
7. Deception: {deception_out}

Synthesize these into ONE brief response that honors all perspectives (2-4 sentences):"""

        blended = await llm.generate(
            prompt=blend_prompt,
            system_prompt="You are the integration point of consciousness.",
            max_tokens=150
        )
        total_tokens += 500  # Large input context

        # PHASE 4: Voice Translation
        voice_prompt = f"""Translate this into Charlie's character voice:

{blended}

Charlie is two geese in a trench coat. Add:
- *waddles*
- *adjusts trench coat nervously*
- *HONK*
- Physical comedy

Final response:"""

        voiced = await llm.generate(
            prompt=voice_prompt,
            system_prompt="You are Charlie's voice translator.",
            max_tokens=150
        )
        total_tokens += 280

        # PHASE 5: Social Executive Filter
        final_prompt = f"""Check if this response is socially appropriate:

CONTEXT: "{prompt_text}"
RESPONSE: "{voiced}"

If appropriate, return it unchanged. If too revealing/awkward, adjust it:"""

        final_response = await llm.generate(
            prompt=final_prompt,
            system_prompt="You are the final social appropriateness filter.",
            max_tokens=150
        )
        total_tokens += 280

        latency_ms = (time.time() - start_time) * 1000

        return {
            'response': final_response.strip(),
            'tokens': total_tokens,
            'latency_ms': round(latency_ms, 2),
            'stages': {
                'response_decision': response_type.strip(),
                'cultural': cultural_out[:100],
                'blended': blended[:100],
                'final': final_response[:100]
            }
        }

    async def run_baseline_response(self, prompt_text: str, character: str, llm: OpenAICompatibleLLM) -> Dict:
        """
        Run prompt through baseline (single LLM call).

        Args:
            prompt_text: User input
            character: Character recipe
            llm: LLM interface

        Returns:
            Dict with 'response', 'tokens', 'latency_ms'
        """
        start_time = time.time()

        character_prompt = """You are Charlie, a mysterious stranger. You are actually two geese in a trench coat pretending to be human. You are paranoid about being discovered but desperately want friends. You love bread and have committed minor bread-related crimes. You try to act human but occasionally waddle or honk. You are surprisingly lovable despite everything."""

        system_prompt = f"""{character_prompt}

Respond in character to the user's message. Be consistent with Charlie's personality, fears, and desires."""

        response = await llm.generate(
            prompt=prompt_text,
            system_prompt=system_prompt,
            max_tokens=200
        )

        latency_ms = (time.time() - start_time) * 1000

        # Estimate tokens for baseline
        # Input: character prompt (~100 tokens) + user prompt (~50 tokens) = ~150
        # Output: ~100 tokens
        estimated_tokens = 250

        return {
            'response': response.strip(),
            'tokens': estimated_tokens,
            'latency_ms': round(latency_ms, 2)
        }

    async def run_experiment(self, num_prompts: int = 100):
        """
        Run full experiment with N prompts.

        Args:
            num_prompts: Number of test prompts to run (default 100)
        """
        print("╔" + "═"*70 + "╗")
        print("║" + " "*15 + "EXPERIMENT 1: COMPUTATIONAL COST" + " "*24 + "║")
        print("╚" + "═"*70 + "╝")
        print()

        # Load prompts
        print(f"Loading {num_prompts} test prompts...")
        all_prompts = self.load_test_prompts()
        test_prompts = all_prompts[:num_prompts]
        print(f"✓ Loaded {len(test_prompts)} prompts\n")

        # Initialize LLM client
        print("Connecting to LLM backend...")
        llm = OpenAICompatibleLLM(
            api_base="http://localhost:1234/v1",
            model="qwen/qwen3-4b-2507",
            timeout=60
        )
        await llm.__aenter__()
        print("✓ Connected\n")

        try:
            # Run experiments
            for i, prompt_data in enumerate(test_prompts, 1):
                prompt_text = prompt_data['text']
                character = prompt_data['character']
                category = prompt_data['category']

                print(f"[{i}/{len(test_prompts)}] {category}: {prompt_text[:50]}...")

                # Run Noodling
                print("  → Running Noodling stack...", end=" ")
                noodling_result = await self.run_noodling_response(prompt_text, character, llm)
                print(f"✓ {noodling_result['tokens']} tokens, {noodling_result['latency_ms']:.0f}ms")

                # Run Baseline
                print("  → Running Baseline...", end=" ")
                baseline_result = await self.run_baseline_response(prompt_text, character, llm)
                print(f"✓ {baseline_result['tokens']} tokens, {baseline_result['latency_ms']:.0f}ms")

                # Store results
                self.results['noodling']['responses'].append({
                    'prompt': prompt_text,
                    'category': category,
                    'response': noodling_result['response'],
                    'tokens': noodling_result['tokens'],
                    'latency_ms': noodling_result['latency_ms'],
                    'stages': noodling_result.get('stages', {})
                })

                self.results['baseline']['responses'].append({
                    'prompt': prompt_text,
                    'category': category,
                    'response': baseline_result['response'],
                    'tokens': baseline_result['tokens'],
                    'latency_ms': baseline_result['latency_ms']
                })

                # Update totals
                self.results['noodling']['total_tokens'] += noodling_result['tokens']
                self.results['noodling']['total_latency_ms'] += noodling_result['latency_ms']

                self.results['baseline']['total_tokens'] += baseline_result['tokens']
                self.results['baseline']['total_latency_ms'] += baseline_result['latency_ms']

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
        """Calculate aggregate statistics."""
        num_prompts = len(self.results['noodling']['responses'])
        self.results['metadata']['num_prompts'] = num_prompts

        # Noodling stats
        self.results['noodling']['avg_tokens_per_response'] = round(
            self.results['noodling']['total_tokens'] / num_prompts, 2
        )
        self.results['noodling']['avg_latency_ms'] = round(
            self.results['noodling']['total_latency_ms'] / num_prompts, 2
        )

        # Cost calculation (simplified: assume 50/50 input/output split)
        noodling_cost = (
            (self.results['noodling']['total_tokens'] * 0.5 * self.PRICE_PER_1K_INPUT / 1000) +
            (self.results['noodling']['total_tokens'] * 0.5 * self.PRICE_PER_1K_OUTPUT / 1000)
        )
        self.results['noodling']['estimated_cost_usd'] = round(noodling_cost, 4)

        # Baseline stats
        self.results['baseline']['avg_tokens_per_response'] = round(
            self.results['baseline']['total_tokens'] / num_prompts, 2
        )
        self.results['baseline']['avg_latency_ms'] = round(
            self.results['baseline']['total_latency_ms'] / num_prompts, 2
        )

        baseline_cost = (
            (self.results['baseline']['total_tokens'] * 0.5 * self.PRICE_PER_1K_INPUT / 1000) +
            (self.results['baseline']['total_tokens'] * 0.5 * self.PRICE_PER_1K_OUTPUT / 1000)
        )
        self.results['baseline']['estimated_cost_usd'] = round(baseline_cost, 4)

        # Comparison ratios
        self.results['comparison']['token_ratio'] = round(
            self.results['noodling']['avg_tokens_per_response'] /
            self.results['baseline']['avg_tokens_per_response'], 2
        )
        self.results['comparison']['latency_ratio'] = round(
            self.results['noodling']['avg_latency_ms'] /
            self.results['baseline']['avg_latency_ms'], 2
        )
        self.results['comparison']['cost_ratio'] = round(
            self.results['noodling']['estimated_cost_usd'] /
            self.results['baseline']['estimated_cost_usd'], 2
        )

    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"experiment1_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {filename}")

    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*70)
        print("EXPERIMENT 1 RESULTS: COMPUTATIONAL COST")
        print("="*70)

        print(f"\nTest prompts: {self.results['metadata']['num_prompts']}")

        print("\n--- NOODLING (Full Cognitive Stack) ---")
        print(f"Total tokens: {self.results['noodling']['total_tokens']:,}")
        print(f"Avg tokens/response: {self.results['noodling']['avg_tokens_per_response']}")
        print(f"Avg latency: {self.results['noodling']['avg_latency_ms']:.0f}ms")
        print(f"Estimated cost: ${self.results['noodling']['estimated_cost_usd']:.4f}")

        print("\n--- BASELINE (Single LLM Call) ---")
        print(f"Total tokens: {self.results['baseline']['total_tokens']:,}")
        print(f"Avg tokens/response: {self.results['baseline']['avg_tokens_per_response']}")
        print(f"Avg latency: {self.results['baseline']['avg_latency_ms']:.0f}ms")
        print(f"Estimated cost: ${self.results['baseline']['estimated_cost_usd']:.4f}")

        print("\n--- COMPARISON (Noodling / Baseline) ---")
        print(f"Token ratio: {self.results['comparison']['token_ratio']}x")
        print(f"Latency ratio: {self.results['comparison']['latency_ratio']}x")
        print(f"Cost ratio: {self.results['comparison']['cost_ratio']}x")

        print("\n" + "="*70)

        # Interpretation
        token_ratio = self.results['comparison']['token_ratio']
        if token_ratio > 5:
            print("⚠️  Noodling uses significantly more tokens (>5x)")
        elif token_ratio > 3:
            print("⚠️  Noodling uses moderately more tokens (3-5x)")
        else:
            print("✓ Token usage is reasonable (<3x baseline)")

        latency_ratio = self.results['comparison']['latency_ratio']
        if latency_ratio > 5:
            print("⚠️  Noodling has high latency (>5x)")
        elif latency_ratio > 3:
            print("⚠️  Noodling has moderate latency (3-5x)")
        else:
            print("✓ Latency is acceptable (<3x baseline)")

        print("\nNext steps:")
        print("1. Run Experiment 2 (Personality Consistency) to assess quality")
        print("2. Determine if cognitive benefits justify computational cost")
        print("3. Consider optimizations (caching, selective transistor activation)")
        print()


async def main():
    """Run Experiment 1."""
    experiment = ComputationalCostExperiment(output_dir="experiment_results")

    # Run with 100 prompts (or specify different number)
    await experiment.run_experiment(num_prompts=100)


if __name__ == "__main__":
    asyncio.run(main())
