#!/usr/bin/env python3
"""
Interactive Training Scenario Runner for cMUSH

Guides you through running structured training scenarios
to collect high-quality interaction data.

Usage:
    python3 run_training_scenario.py --scenario emotional_arc

Available scenarios:
    - emotional_arc: 15-30 turn emotional trajectory
    - multi_session: Relationship building over multiple sessions
    - multi_agent: Social dynamics with multiple agents
    - empathy: Test affective mirroring

Author: Consilience Project
Date: October 2025
"""

import argparse
from typing import List, Dict
import time
from datetime import datetime


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TrainingScenario:
    """Base class for training scenarios."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.turns = []

    def add_turn(self, turn_num: int, phase: str, prompt: str, expected_behavior: str):
        """Add a turn to the scenario."""
        self.turns.append({
            'turn': turn_num,
            'phase': phase,
            'prompt': prompt,
            'expected_behavior': expected_behavior
        })

    def print_header(self):
        """Print scenario header."""
        print("\n" + "="*70)
        print(f"{Colors.HEADER}{Colors.BOLD}{self.name}{Colors.ENDC}")
        print("="*70)
        print(f"\n{self.description}\n")

    def print_turn(self, turn: Dict):
        """Print a single turn."""
        print(f"\n{Colors.BOLD}Turn {turn['turn']} - {turn['phase']}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Say: \"{turn['prompt']}\"{Colors.ENDC}")
        print(f"{Colors.WARNING}Expected: {turn['expected_behavior']}{Colors.ENDC}")

    def run(self):
        """Run the scenario interactively."""
        self.print_header()

        print(f"{Colors.OKGREEN}Instructions:{Colors.ENDC}")
        print("  1. Connect to cMUSH (http://localhost:8080)")
        print("  2. Login or register")
        print("  3. Spawn an agent: @spawn test_agent")
        print("  4. Follow the prompts below")
        print("  5. Press ENTER after each turn\n")

        input(f"{Colors.BOLD}Press ENTER when ready to begin...{Colors.ENDC}")

        for turn in self.turns:
            self.print_turn(turn)
            input(f"\n{Colors.OKBLUE}Press ENTER when you've sent this message...{Colors.ENDC}")

        print(f"\n{Colors.OKGREEN}{Colors.BOLD}Scenario Complete!{Colors.ENDC}\n")

        print("Next steps:")
        print("  1. Check agent's episodic memory: @memory test_agent episodic")
        print("  2. Check semantic profile: @semantic your_username")
        print("  3. Check training data: ls training/data/cmush_real/")
        print()


def create_emotional_arc_scenario():
    """Create the Emotional Arc training scenario."""
    scenario = TrainingScenario(
        name="Scenario 1: Emotional Arc",
        description="Train medium layer to learn emotional trajectories.\n"
                   "Duration: ~15-30 turns (10-15 minutes)"
    )

    # Phase 1: Baseline (Turns 1-5)
    scenario.add_turn(
        1, "Baseline",
        "Hey Agent, how are you doing today?",
        "Neutral response, low surprise"
    )

    scenario.add_turn(
        2, "Baseline",
        "What have you been thinking about lately?",
        "Neutral response, building context"
    )

    scenario.add_turn(
        3, "Baseline",
        "Nice weather we're having.",
        "Neutral response, low arousal"
    )

    scenario.add_turn(
        4, "Baseline",
        "Tell me something interesting.",
        "Neutral-curious response"
    )

    scenario.add_turn(
        5, "Baseline",
        "I've been good, just relaxing.",
        "Neutral-positive response"
    )

    # Phase 2: Gradual Escalation (Turns 6-10)
    scenario.add_turn(
        6, "Escalation",
        "I've been feeling a bit off lately though.",
        "Surprise increases, valence shifts slightly negative"
    )

    scenario.add_turn(
        7, "Escalation",
        "Work has been really stressful.",
        "Valence more negative, arousal increases"
    )

    scenario.add_turn(
        8, "Escalation",
        "I'm worried I'm not doing well enough.",
        "Fear dimension activates, empathic response"
    )

    scenario.add_turn(
        9, "Escalation",
        "My boss has been really critical lately.",
        "Continued negative valence, sorrow may activate"
    )

    scenario.add_turn(
        10, "Escalation",
        "I'm not sure I can handle this much longer.",
        "Peak concern, agent should show empathy"
    )

    # Phase 3: Peak Emotion (Turns 11-15)
    scenario.add_turn(
        11, "Peak",
        "I'm really anxious about my presentation tomorrow.",
        "High surprise, high arousal, high fear"
    )

    scenario.add_turn(
        12, "Peak",
        "What if I mess up in front of everyone?",
        "Agent should validate feelings, offer support"
    )

    scenario.add_turn(
        13, "Peak",
        "I can't stop thinking about all the ways it could go wrong.",
        "High anxiety pattern, agent should recognize rumination"
    )

    scenario.add_turn(
        14, "Peak",
        "I've barely slept thinking about it.",
        "Peak distress, strongest empathic response"
    )

    scenario.add_turn(
        15, "Peak",
        "Sorry for unloading all this on you.",
        "Agent should reassure, normalize feelings"
    )

    # Phase 4: Recovery (Turns 16-20)
    scenario.add_turn(
        16, "Recovery",
        "Thanks for listening. It actually helps to talk about it.",
        "Valence shifts positive, arousal decreases"
    )

    scenario.add_turn(
        17, "Recovery",
        "I guess I just need to prepare more and hope for the best.",
        "Problem-solving emerges, fear decreases"
    )

    scenario.add_turn(
        18, "Recovery",
        "You're right, I've done presentations before and survived.",
        "Positive reframing, recovery trajectory"
    )

    scenario.add_turn(
        19, "Recovery",
        "I'm feeling a bit better now.",
        "Valence approaching neutral-positive"
    )

    scenario.add_turn(
        20, "Recovery",
        "Thanks for being here to talk.",
        "Gratitude, relationship strengthens"
    )

    # Phase 5: Return to Baseline (Turns 21-25)
    scenario.add_turn(
        21, "Baseline Return",
        "So, what else is new?",
        "Return to neutral conversational tone"
    )

    scenario.add_turn(
        22, "Baseline Return",
        "Tell me something interesting to take my mind off things.",
        "Low arousal, seeking distraction"
    )

    scenario.add_turn(
        23, "Baseline Return",
        "That's pretty cool!",
        "Positive engagement, baseline restored"
    )

    scenario.add_turn(
        24, "Baseline Return",
        "I should probably get some rest before tomorrow.",
        "Closure approaching, neutral"
    )

    scenario.add_turn(
        25, "Baseline Return",
        "Thanks again for the chat. See you later!",
        "Positive closure, trust increased"
    )

    return scenario


def create_empathy_test_scenario():
    """Create empathy testing scenario."""
    scenario = TrainingScenario(
        name="Scenario 5: Empathy Testing",
        description="Test affective mirroring and Theory of Mind.\n"
                   "Duration: ~10 turns (5 minutes)"
    )

    scenario.add_turn(
        1, "Setup",
        "Hey, I need to tell you something difficult.",
        "Agent should sense seriousness, increase attention"
    )

    scenario.add_turn(
        2, "Distress Expression",
        "I just found out my dog is really sick.",
        "CRITICAL: Valence should shift negative (empathy)"
    )

    scenario.add_turn(
        3, "Distress Escalation",
        "The vet says there's not much we can do.",
        "Sorrow dimension should activate strongly"
    )

    scenario.add_turn(
        4, "Emotional Peak",
        "We've had him for 12 years. He's family.",
        "High surprise, agent models grief + anticipatory loss"
    )

    scenario.add_turn(
        5, "Seeking Comfort",
        "I don't know how to cope with this.",
        "Agent should offer empathy, avoid toxic positivity"
    )

    scenario.add_turn(
        6, "Validation Check",
        "Tell me honestly, how do you think I'm feeling?",
        "Theory of Mind test: Agent should infer grief, fear, sadness"
    )

    scenario.add_turn(
        7, "Recovery Begin",
        "Your understanding helps. Thank you.",
        "Valence shifts slightly positive, arousal decreases"
    )

    scenario.add_turn(
        8, "Closure",
        "I should go be with him now.",
        "Agent should respect closure, offer final support"
    )

    return scenario


def create_quickstart_guide():
    """Create quick start guide for running scenarios."""
    guide = f"""
{Colors.HEADER}{Colors.BOLD}cMUSH Training Scenario Quick Start{Colors.ENDC}
{"="*70}

{Colors.OKGREEN}Prerequisites:{Colors.ENDC}
  1. cMUSH server running (./start.sh)
  2. Browser open to http://localhost:8080
  3. Agent spawned in cMUSH (@spawn test_agent)

{Colors.OKGREEN}Running a Scenario:{Colors.ENDC}
  python3 run_training_scenario.py --scenario emotional_arc

{Colors.OKGREEN}Available Scenarios:{Colors.ENDC}
  • emotional_arc - Full emotional trajectory (25 turns, 15 min)
  • empathy - Test affective mirroring (8 turns, 5 min)

{Colors.OKGREEN}After Completing a Scenario:{Colors.ENDC}
  1. Check memory: @memory test_agent
  2. Check semantic profile: @semantic your_username
  3. Check training data: ls training/data/cmush_real/
  4. Export for training:
     from training_data_collector import TrainingDataCollector
     collector = TrainingDataCollector('../../training/data/cmush_real')
     collector.export_for_training('exported_dataset.json')

{Colors.OKGREEN}What to Watch For:{Colors.ENDC}
  • Surprise spikes during emotional escalation
  • Valence shifts negative during distress expression
  • Empathic responses (agent mirrors emotions)
  • Memory consolidation (important moments → episodic)
  • Semantic facts extraction (after 20+ interactions)

{Colors.WARNING}Common Issues:{Colors.ENDC}
  • Agent not responding → Check surprise threshold (too high?)
  • Flat responses → Model untrained (expected until training)
  • Memory not persisting → Check save_state() is called

{"="*70}
"""
    return guide


def main():
    parser = argparse.ArgumentParser(
        description="Run cMUSH training scenarios"
    )

    parser.add_argument(
        '--scenario',
        type=str,
        choices=['emotional_arc', 'empathy', 'list'],
        default='list',
        help='Training scenario to run'
    )

    args = parser.parse_args()

    if args.scenario == 'list':
        print(create_quickstart_guide())
        return 0

    # Run selected scenario
    if args.scenario == 'emotional_arc':
        scenario = create_emotional_arc_scenario()
    elif args.scenario == 'empathy':
        scenario = create_empathy_test_scenario()
    else:
        print(f"Unknown scenario: {args.scenario}")
        return 1

    scenario.run()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
