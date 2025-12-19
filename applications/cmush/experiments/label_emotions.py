#!/usr/bin/env python3
"""
Automatic emotion labeling for noodleMUSH conversation logs.

Reads chat history and labels each agent response with primary emotion.
Uses LLM for zero-shot classification.

Output: emotion_labeled_dataset.json
"""

import json
from pathlib import Path
import sys
import urllib.request
import urllib.parse

# Simple synchronous LLM client for labeling
class SimpleLLM:
    def __init__(self, base_url="http://localhost:1234/v1", model="SMALL"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt, max_tokens=10, temperature=0.0):
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

# Emotion categories (10 basic emotions)
EMOTIONS = [
    'fear', 'joy', 'sadness', 'anger', 'love',
    'guilt', 'pride', 'shame', 'curiosity', 'boredom'
]


def extract_context(chat_history, index, n_turns=5):
    """Get n previous conversation turns for context."""
    start = max(0, index - n_turns)
    context_turns = chat_history[start:index]

    context = []
    for turn in context_turns:
        speaker = turn.get('user', 'unknown')
        text = turn.get('text', '')
        if text:
            context.append(f"{speaker}: {text}")

    return "\n".join(context) if context else "No prior context"


def classify_emotion(llm, context, response):
    """LLM-based emotion classification."""
    prompt = f"""You are labeling emotional responses for machine learning.

Conversation context:
{context}

Agent response:
{response}

Classify the PRIMARY emotion in this response. Choose ONE from this list:
fear, joy, sadness, anger, love, guilt, pride, shame, curiosity, boredom

Return ONLY the emotion label (lowercase, no punctuation).
If multiple emotions present, choose the DOMINANT one.
If neutral/unclear, choose 'curiosity' as default."""

    try:
        result = llm.generate(prompt, max_tokens=10, temperature=0.0)
        label = result.strip().lower()

        # Validate
        if label in EMOTIONS:
            return label

        # Try to map common variations
        if 'happy' in label or 'excited' in label or 'delight' in label:
            return 'joy'
        elif 'scared' in label or 'afraid' in label or 'anxious' in label:
            return 'fear'
        elif 'sad' in label or 'melancholy' in label:
            return 'sadness'
        elif 'angry' in label or 'mad' in label or 'furious' in label:
            return 'anger'
        elif 'loving' in label or 'affection' in label:
            return 'love'
        elif 'guilty' in label:
            return 'guilt'
        elif 'proud' in label:
            return 'pride'
        elif 'ashamed' in label or 'embarrass' in label:
            return 'shame'
        elif 'bored' in label:
            return 'boredom'
        else:
            return 'curiosity'  # Default fallback

    except Exception as e:
        print(f"  Warning: Classification failed: {e}")
        return 'curiosity'


def main():
    print("=" * 70)
    print("EMOTION LABELING PIPELINE")
    print("=" * 70)

    # Load chat history
    chat_path = Path(__file__).parent.parent / 'world' / 'chat_history.json'
    print(f"\nLoading chat history from: {chat_path}")

    if not chat_path.exists():
        print(f"ERROR: Chat history not found at {chat_path}")
        sys.exit(1)

    with open(chat_path, 'r') as f:
        chat_history = json.load(f)

    print(f"Loaded {len(chat_history)} total messages")

    # Initialize LLM (use fast model for labeling)
    print("\nInitializing LLM interface...")
    llm = SimpleLLM(base_url="http://localhost:1234/v1", model="SMALL")

    # Label agent responses
    labeled_data = []
    agent_responses = 0

    print("\nProcessing agent responses...")
    for i, entry in enumerate(chat_history):
        text = entry.get('text', '')

        # Detect agent responses by text pattern (agent_name says/thinks)
        # Skip user messages (username says) and system messages
        is_agent = False
        agent_name = None

        # Common agent patterns
        if ' says, ' in text or ' thinks, ' in text or ' and says, ' in text:
            # Extract potential agent name (first word before action)
            first_word = text.split()[0] if text.split() else ''
            # Check if it's NOT a known user (caity)
            if first_word.lower() not in ['caity', 'thistlequell'] and first_word[0].isupper():
                is_agent = True
                agent_name = first_word

        if not is_agent:
            continue

        agent_responses += 1

        # Skip very short responses
        if len(text.strip()) < 10:
            continue

        # Get context
        context = extract_context(chat_history, i, n_turns=5)

        # Classify emotion
        emotion = classify_emotion(llm, context, text)

        # Extract affect if available
        affect = entry.get('affect', {
            'valence': 0.0,
            'arousal': 0.0,
            'fear': 0.0,
            'sorrow': 0.0,
            'boredom': 0.0
        })

        labeled_data.append({
            'agent_id': agent_name or 'unknown',
            'context': context,
            'response': text,
            'emotion': emotion,
            'affect': affect,
            'timestamp': entry.get('timestamp', ''),
            'index': i
        })

        if len(labeled_data) % 50 == 0:
            print(f"  Labeled {len(labeled_data)} / {agent_responses} agent responses...")

    print(f"\nProcessed {agent_responses} agent responses")
    print(f"Labeled {len(labeled_data)} responses (filtered out very short ones)")

    # Save full dataset
    output_path = Path(__file__).parent / 'emotion_labeled_dataset.json'
    with open(output_path, 'w') as f:
        json.dump(labeled_data, f, indent=2)

    # Statistics
    emotion_counts = {}
    for item in labeled_data:
        emotion = item['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print(f"\n{'=' * 70}")
    print("EMOTION DISTRIBUTION")
    print('=' * 70)
    for emotion in EMOTIONS:
        count = emotion_counts.get(emotion, 0)
        pct = 100 * count / len(labeled_data) if labeled_data else 0
        bar = '#' * int(pct / 2)
        print(f"  {emotion:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n{'=' * 70}")
    print(f"SAVED: {output_path}")
    print(f"Total examples: {len(labeled_data)}")
    print('=' * 70)


if __name__ == '__main__':
    main()
