#!/usr/bin/env python3
"""
Synthetic emotion dataset generator for Noodlings training.

Generates balanced scenarios across 10 emotion classes with LLM-based
agent responses and affect vectors.

Target: 1000 examples (100 per emotion class)
"""

import json
from pathlib import Path
import sys
import urllib.request
import time
import random

# Simple synchronous LLM client
class SimpleLLM:
    def __init__(self, base_url="http://localhost:1234/v1", model="SMALL"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt, max_tokens=300, temperature=0.7):
        """Simple synchronous generation."""
        data = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }).encode('utf-8')

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read())
            return result['choices'][0]['message']['content']


# Emotion categories
EMOTIONS = [
    'fear', 'joy', 'sadness', 'anger', 'love',
    'guilt', 'pride', 'shame', 'curiosity', 'boredom'
]

# Scenario templates for each emotion (10 per emotion = 100 base scenarios)
SCENARIO_TEMPLATES = {
    'fear': [
        "You hear strange scratching sounds coming from inside the walls of your room at night",
        "A shadowy figure appears at the edge of your vision, but vanishes when you look directly at it",
        "You receive a threatening anonymous message warning you to stop your current project",
        "The ground beneath your feet begins to crack and crumble unexpectedly",
        "You realize you're being followed by someone in a dark alley",
        "A loud explosion echoes through the building you're in",
        "You discover that your closest ally has been secretly working against you",
        "Medical test results come back with concerning abnormalities",
        "You're trapped in an elevator that suddenly drops several floors",
        "A dangerous wild animal appears between you and the only exit"
    ],
    'joy': [
        "You finally solve a problem you've been working on for months",
        "A dear friend you haven't seen in years surprises you with a visit",
        "You discover that your creative work has inspired someone deeply",
        "Spring arrives after a long, harsh winter and flowers bloom everywhere",
        "You receive unexpected recognition for your contributions",
        "A child laughs delightedly at something you created",
        "You find the perfect gift for someone you care about",
        "Your experimental prototype works flawlessly on the first try",
        "You wake up to find fresh snow transforming the world into a winter wonderland",
        "Someone tells you that your kindness changed their life"
    ],
    'sadness': [
        "You find an old photograph of happier times that can never return",
        "A beloved companion moves far away and you may never see them again",
        "You watch autumn leaves falling, knowing winter approaches",
        "Your carefully tended garden withers despite your best efforts",
        "You realize a cherished dream is no longer possible to achieve",
        "A childhood home is being demolished to make way for development",
        "You receive news that a mentor who shaped your life has passed away",
        "Rain falls on a day you had planned to be special",
        "You find an unfinished project from someone who is no longer here",
        "The last copy of an irreplaceable memory is accidentally destroyed"
    ],
    'anger': [
        "Someone takes credit for your hard work in front of an important audience",
        "You discover that you've been deliberately lied to for months",
        "A careless action by someone else destroys something precious to you",
        "You're blamed for a failure that wasn't your fault",
        "Someone mocks your deeply held beliefs in a cruel way",
        "A powerful person abuses their authority to harm the vulnerable",
        "Your important concerns are dismissed without consideration",
        "You catch someone betraying your trust for personal gain",
        "Bureaucratic red tape prevents you from helping someone in need",
        "Someone deliberately breaks a promise they swore to keep"
    ],
    'love': [
        "Someone remembers a small detail about you that you mentioned once, months ago",
        "You watch someone you care about achieve their dreams",
        "A warm embrace conveys everything words cannot express",
        "You cook a meal together, working in comfortable, synchronized harmony",
        "Someone trusts you with their deepest vulnerability",
        "You notice how light catches in someone's eyes when they smile",
        "A simple gesture reveals profound care and understanding",
        "You share a moment of perfect understanding without words",
        "Someone stands by you when it would be easier to walk away",
        "You realize how much you've grown together over time"
    ],
    'guilt': [
        "You realize your advice led someone to make a terrible decision",
        "You forgot an important promise and disappointed someone who counted on you",
        "Your success came at a cost to others you didn't consider",
        "You said something hurtful in anger that you can't take back",
        "You were too busy to notice someone needed help desperately",
        "You broke something precious while borrowing it",
        "You lied to protect yourself and an innocent person suffered consequences",
        "You didn't speak up when you witnessed something wrong happening",
        "You wasted an opportunity someone else would have treasured",
        "You realize you've been taking someone's efforts for granted"
    ],
    'pride': [
        "Your student surpasses your own abilities through your teaching",
        "You complete a marathon after training for months",
        "Your innovative solution elegantly solves a complex problem",
        "You stand up for what's right despite personal cost",
        "A project you built from nothing thrives and helps many people",
        "You master a difficult skill through persistent practice",
        "Your creative work expresses exactly what you envisioned",
        "You remain calm and capable in a crisis situation",
        "You help someone achieve something they thought impossible",
        "You build something beautiful with your own hands"
    ],
    'shame': [
        "Your ignorance on an important topic is exposed publicly",
        "You're caught doing something embarrassing by people you respect",
        "You fail at a task everyone assumed you could handle easily",
        "Your private thoughts are accidentally shared with others",
        "You realize you've been confidently wrong about something obvious",
        "Someone discovers a secret weakness you've hidden carefully",
        "You behave in a way that contradicts your stated values",
        "Your attempts to help actually made a situation worse",
        "You're revealed as less competent than people believed",
        "You make an obvious mistake in front of an audience"
    ],
    'curiosity': [
        "You find a strange device with no obvious purpose or origin",
        "An unusual pattern in data suggests something unexpected",
        "Someone mentions a concept you've never encountered before",
        "You discover a hidden door in a building you thought you knew",
        "A cryptic message arrives with no sender information",
        "You overhear part of a conversation that raises intriguing questions",
        "An unexpected correlation appears in your research",
        "You find strange footprints leading into the forest",
        "Someone's behavior changes dramatically for no apparent reason",
        "A book falls open to a page that seems oddly relevant"
    ],
    'boredom': [
        "You attend yet another meeting that could have been an email",
        "The same predictable conversation plays out for the hundredth time",
        "You're stuck waiting for hours with nothing engaging to do",
        "Someone explains something obvious in excruciating detail",
        "Every day follows the exact same routine without variation",
        "You're forced to repeat a simple task endlessly",
        "A long journey stretches ahead with unchanging scenery",
        "You watch a presentation on a topic you already understand completely",
        "Time seems to slow as you wait for something, anything, to happen",
        "You're trapped listening to someone's tedious stories with no escape"
    ]
}


def generate_affect_vector(emotion):
    """
    Generate theoretically correct PAD + sorrow + boredom affect vectors.

    Based on Mehrabian & Russell (1974) PAD model + functionally-specific emotions.

    PAD Dimensions:
    - Pleasure (valence): -1.0 (unpleasant) to +1.0 (pleasant)
    - Arousal: 0.0 (calm/sleepy) to 1.0 (excited/alert)
    - Dominance: 0.0 (submissive/controlled) to 1.0 (dominant/in-control)

    Additional Dimensions:
    - Sorrow: Functionally-specific sadness/loss response
    - Boredom: Understimulation state
    """
    affect_templates = {
        # Fear: Unpleasant + High arousal + VERY LOW dominance (threatened, out of control)
        'fear': {'valence': -0.6, 'arousal': 0.8, 'dominance': 0.1, 'sorrow': 0.2, 'boredom': 0.0},

        # Joy: Pleasant + High arousal + MODERATE-HIGH dominance (confident happiness)
        'joy': {'valence': 0.8, 'arousal': 0.7, 'dominance': 0.7, 'sorrow': 0.0, 'boredom': 0.0},

        # Sadness: Unpleasant + Low arousal + LOW dominance (helpless, defeated)
        'sadness': {'valence': -0.7, 'arousal': 0.2, 'dominance': 0.2, 'sorrow': 0.9, 'boredom': 0.2},

        # Anger: Unpleasant + HIGH arousal + HIGH dominance (aggressive, fighting back)
        'anger': {'valence': -0.7, 'arousal': 0.9, 'dominance': 0.8, 'sorrow': 0.0, 'boredom': 0.0},

        # Love: Pleasant + Moderate arousal + MODERATE dominance (mutual, balanced)
        'love': {'valence': 0.9, 'arousal': 0.5, 'dominance': 0.5, 'sorrow': 0.0, 'boredom': 0.0},

        # Guilt: Unpleasant + Moderate arousal + LOW-MODERATE dominance (responsible but regretful)
        'guilt': {'valence': -0.6, 'arousal': 0.5, 'dominance': 0.3, 'sorrow': 0.6, 'boredom': 0.0},

        # Pride: Pleasant + Moderate arousal + VERY HIGH dominance (accomplished, powerful)
        'pride': {'valence': 0.8, 'arousal': 0.6, 'dominance': 0.9, 'sorrow': 0.0, 'boredom': 0.0},

        # Shame: Unpleasant + Moderate arousal + VERY LOW dominance (exposed, powerless)
        'shame': {'valence': -0.8, 'arousal': 0.5, 'dominance': 0.1, 'sorrow': 0.5, 'boredom': 0.0},

        # Curiosity: Neutral-pleasant + Moderate-high arousal + MODERATE dominance (engaged, exploring)
        'curiosity': {'valence': 0.2, 'arousal': 0.6, 'dominance': 0.6, 'sorrow': 0.0, 'boredom': 0.0},

        # Boredom: Unpleasant + VERY LOW arousal + LOW dominance (understimulated, passive)
        'boredom': {'valence': -0.3, 'arousal': 0.1, 'dominance': 0.3, 'sorrow': 0.2, 'boredom': 0.9}
    }

    # Add small random variation to make realistic
    template = affect_templates[emotion]
    affect = {}
    for key, val in template.items():
        # Add noise: +/- 0.1, clamped to valid range
        noise = random.uniform(-0.1, 0.1)
        noisy_val = val + noise

        # Clamp valence to [-1, 1], others to [0, 1]
        if key == 'valence':
            affect[key] = max(-1.0, min(1.0, noisy_val))
        else:
            affect[key] = max(0.0, min(1.0, noisy_val))

    return affect


def generate_agent_response(llm, scenario, emotion, agent_name="Synthetic"):
    """Generate agent response to scenario with specified emotion."""

    prompt = f"""You are {agent_name}, a thoughtful AI agent experiencing an emotional situation.

SCENARIO: {scenario}

REQUIRED EMOTION: {emotion}

Generate a natural response that:
1. Clearly expresses {emotion} through your words and body language
2. Uses realistic agent action format: "[agent_name] [action] and says, \"[speech]\""
3. Is 2-4 sentences long
4. Feels authentic and specific to the scenario
5. Shows emotional depth without being melodramatic

Example format:
{agent_name} steps back, eyes widening and says, "I wasn't expecting that at all..."

Generate response now:"""

    try:
        response = llm.generate(prompt, max_tokens=200, temperature=0.8)
        return response.strip()
    except Exception as e:
        print(f"  Warning: Response generation failed: {e}")
        return f"{agent_name} responds to the situation with {emotion}."


def generate_synthetic_dataset(llm, target_per_emotion=100):
    """Generate balanced synthetic dataset."""

    print("=" * 70)
    print("SYNTHETIC EMOTION DATASET GENERATOR")
    print("=" * 70)
    print(f"\nTarget: {target_per_emotion} examples per emotion ({target_per_emotion * 10} total)")

    dataset = []

    for emotion in EMOTIONS:
        print(f"\n{'=' * 70}")
        print(f"Generating {emotion.upper()} examples...")
        print('=' * 70)

        templates = SCENARIO_TEMPLATES[emotion]
        examples_generated = 0

        # Generate multiple variations per template
        while examples_generated < target_per_emotion:
            # Cycle through templates
            template = templates[examples_generated % len(templates)]

            # Add variation number to agent name
            agent_name = f"Agent{(examples_generated // len(templates)) + 1}"

            # Generate response
            response = generate_agent_response(llm, template, emotion, agent_name)

            # Generate affect vector
            affect = generate_affect_vector(emotion)

            # Create data entry
            entry = {
                'agent_id': agent_name,
                'context': f"Scenario: {template}",
                'response': response,
                'emotion': emotion,
                'affect': affect,
                'scenario_template': template,
                'synthetic': True
            }

            dataset.append(entry)
            examples_generated += 1

            if examples_generated % 10 == 0:
                print(f"  Generated {examples_generated}/{target_per_emotion} {emotion} examples...")

            # Small delay to avoid overwhelming LLM API
            time.sleep(0.1)

    return dataset


def main():
    # Configuration
    TARGET_PER_EMOTION = 100  # 100 per class = 1000 total

    # Initialize LLM
    print("\nInitializing LLM interface...")
    llm = SimpleLLM(base_url="http://localhost:1234/v1", model="SMALL")

    # Generate dataset
    dataset = generate_synthetic_dataset(llm, target_per_emotion=TARGET_PER_EMOTION)

    # Shuffle to mix emotions
    random.shuffle(dataset)

    # Save dataset
    output_path = Path(__file__).parent / 'emotion_synthetic_dataset.json'
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Statistics
    emotion_counts = {}
    for item in dataset:
        emotion = item['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print(f"\n{'=' * 70}")
    print("FINAL EMOTION DISTRIBUTION")
    print('=' * 70)
    for emotion in EMOTIONS:
        count = emotion_counts.get(emotion, 0)
        pct = 100 * count / len(dataset) if dataset else 0
        bar = '#' * int(pct / 2)
        print(f"  {emotion:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n{'=' * 70}")
    print(f"SAVED: {output_path}")
    print(f"Total examples: {len(dataset)}")
    print('=' * 70)

    # Also create train/val split
    random.seed(42)  # Reproducible split
    random.shuffle(dataset)

    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    train_path = Path(__file__).parent / 'emotion_synthetic_train.json'
    val_path = Path(__file__).parent / 'emotion_synthetic_val.json'

    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)

    print(f"\nTrain/Val Split:")
    print(f"  Training:   {len(train_data)} examples ({len(train_data)/len(dataset)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} examples ({len(val_data)/len(dataset)*100:.1f}%)")
    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print()


if __name__ == '__main__':
    main()
