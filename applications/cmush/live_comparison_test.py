#!/usr/bin/env python3
# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Live Comparison Test
#
#   This is a benchmarking tool that watches a live noodleMUSH
#   conversation and measures how well the agents are doing.
#
#   Think of it like a sports statistician tracking a game in
#   real-time. As Yuki and Carl chat, this tool counts:
#   - How many tokens each turn uses (efficiency)
#   - How often they mention their bodies (embodiment)
#   - Whether they stay in character (Yuki's mysticism vs Carl's cynicism)
#   - Whether they remember earlier parts of the conversation
#
#   So this code does three things:
#     1. Watches the conversation logs as they happen
#     2. Counts meaningful patterns in what agents say
#     3. Generates a report comparing to baseline expectations
#
#   A scorekeeper for cognitive authenticity.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.live_comparison_test
# PURPOSE:  Real-time metrics collection during agent conversations
# LAYER:    Backend / Testing
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   LiveMetricsCollector    Tracks conversation metrics
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────
"""
Live Comparison Test - Happening NOW

Caity interacts with Yuki and Carl while we measure:
1. Tokens per turn (actual from LLM calls)
2. Embodiment consistency (count paw/tail/sniff references)
3. Character separation (Yuki mysticism vs Carl cynicism)
4. Memory retention (do they remember earlier turns?)

This runs ALONGSIDE the conversation, tracking metrics in real-time.
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
import re

class LiveMetricsCollector:
    """Tracks metrics from live noodleMUSH conversation."""

    def __init__(self):
        self.metrics = {
            'turns': 0,
            'total_tokens': 0,
            'tokens_per_turn': [],
            'yuki_embodiment_refs': 0,  # *paws*, *tail*, *sniffs*, etc.
            'carl_embodiment_refs': 0,
            'yuki_cultural_refs': 0,    # kami, shrine, ancient, etc.
            'carl_cultural_refs': 0,     # skeptical observations
            'memory_callbacks': 0,       # References to earlier turns
            'start_time': time.time(),
            'utterances': []
        }

        # Regex patterns for detection
        self.embodiment_patterns = {
            'yuki': r'\*(?:paws?|tail|sniff|ears?|yip|fox-laugh|mouths?)\*',
            'carl': r'\*(?:paws?|tail|bark|snort|wag|scratch|terrier)\*'
        }

        self.cultural_patterns = {
            'yuki': r'(?:kami|shrine|ancient|centuries|spirit|balance|harmony)',
            'carl': r'(?:absurd|question|authority|hypocr|cynical|skeptic)'
        }

    def analyze_log_file(self, log_path: str):
        """
        Analyze recent log entries for metrics.

        Looks for:
        - LLM token usage logs
        - Agent utterances
        - Embodiment references
        - Cultural markers
        """
        if not Path(log_path).exists():
            print(f"Log file not found: {log_path}")
            return

        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Analyze last 200 lines (recent activity)
        recent = lines[-200:]

        for line in recent:
            # Look for agent speech
            if 'YUKI' in line and 'says,' in line:
                self.metrics['turns'] += 1
                text = self._extract_text(line)
                self._analyze_utterance('yuki', text)

            elif 'CARL' in line and 'says,' in line:
                text = self._extract_text(line)
                self._analyze_utterance('carl', text)

            # Look for token usage (if logged)
            elif 'tokens' in line.lower() and 'response' in line.lower():
                tokens = self._extract_token_count(line)
                if tokens:
                    self.metrics['tokens_per_turn'].append(tokens)
                    self.metrics['total_tokens'] += tokens

        # Calculate summary
        self._calculate_summary()

    def _extract_text(self, log_line: str) -> str:
        """Extract utterance text from log line."""
        # Simple extraction - gets text after 'says,'
        if 'says,' in log_line:
            parts = log_line.split('says,', 1)
            if len(parts) > 1:
                return parts[1].strip()
        return ""

    def _extract_token_count(self, log_line: str) -> int:
        """Extract token count from log line."""
        # Look for patterns like "tokens: 234" or "234 tokens"
        match = re.search(r'(\d+)\s*tokens?', log_line, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def _analyze_utterance(self, agent: str, text: str):
        """Analyze utterance for embodiment and cultural markers."""
        if not text:
            return

        self.metrics['utterances'].append({
            'agent': agent,
            'text': text[:200],
            'timestamp': time.time()
        })

        # Count embodiment references
        pattern = self.embodiment_patterns.get(agent, '')
        if pattern:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if agent == 'yuki':
                self.metrics['yuki_embodiment_refs'] += len(matches)
            else:
                self.metrics['carl_embodiment_refs'] += len(matches)

        # Count cultural markers
        cultural = self.cultural_patterns.get(agent, '')
        if cultural:
            matches = re.findall(cultural, text, re.IGNORECASE)
            if agent == 'yuki':
                self.metrics['yuki_cultural_refs'] += len(matches)
            else:
                self.metrics['carl_cultural_refs'] += len(matches)

        # Detect memory callbacks ("remember", "earlier", "before")
        memory_words = ['remember', 'earlier', 'before', 'recall', 'last time']
        if any(word in text.lower() for word in memory_words):
            self.metrics['memory_callbacks'] += 1

    def _calculate_summary(self):
        """Calculate summary statistics."""
        turns = max(self.metrics['turns'], 1)

        self.metrics['avg_tokens_per_turn'] = (
            sum(self.metrics['tokens_per_turn']) / len(self.metrics['tokens_per_turn'])
            if self.metrics['tokens_per_turn'] else 0
        )

        self.metrics['embodiment_refs_per_turn'] = (
            (self.metrics['yuki_embodiment_refs'] + self.metrics['carl_embodiment_refs']) / turns
        )

        self.metrics['cultural_refs_per_turn'] = (
            (self.metrics['yuki_cultural_refs'] + self.metrics['carl_cultural_refs']) / turns
        )

    def generate_report(self):
        """Generate formatted report."""
        print()
        print("╔" + "═"*70 + "╗")
        print("║" + " "*20 + "LIVE TEST METRICS" + " "*31 + "║")
        print("╚" + "═"*70 + "╝")
        print()

        print(f"Conversation Duration: {time.time() - self.metrics['start_time']:.1f} seconds")
        print(f"Turns Detected: {self.metrics['turns']}")
        print()

        print("TOKEN EFFICIENCY:")
        if self.metrics['tokens_per_turn']:
            print(f"  Total tokens: {self.metrics['total_tokens']:,}")
            print(f"  Avg per turn: {self.metrics['avg_tokens_per_turn']:.0f}")
            print()
            print(f"  Estimated Standard Claude: {self.metrics['turns'] * 5000:,} tokens")
            print(f"  Actual noodleMUSH: {self.metrics['total_tokens']:,} tokens")
            if self.metrics['total_tokens'] > 0:
                efficiency = (self.metrics['turns'] * 5000) / self.metrics['total_tokens']
                print(f"  Efficiency gain: {efficiency:.1f}x")
        else:
            print("  (Token data not yet available in logs)")
            print(f"  Estimated per turn: ~700 tokens (2 agents)")
            print(f"  Estimated total: {self.metrics['turns'] * 700:,} tokens")
        print()

        print("EMBODIMENT ENFORCEMENT:")
        print(f"  Yuki embodiment refs: {self.metrics['yuki_embodiment_refs']}")
        print(f"  Carl embodiment refs: {self.metrics['carl_embodiment_refs']}")
        print(f"  Avg per turn: {self.metrics['embodiment_refs_per_turn']:.1f}")
        print(f"  Consistency: {self.metrics['embodiment_refs_per_turn'] > 1 and ' High' or '~ Moderate'}")
        print()

        print("CULTURAL LENS SEPARATION:")
        print(f"  Yuki mysticism markers: {self.metrics['yuki_cultural_refs']}")
        print(f"  Carl skepticism markers: {self.metrics['carl_cultural_refs']}")
        print(f"  Separation: {abs(self.metrics['yuki_cultural_refs'] - self.metrics['carl_cultural_refs']) > 2 and ' Distinct' or '~ Similar'}")
        print()

        print("MEMORY RETENTION:")
        print(f"  Memory callbacks detected: {self.metrics['memory_callbacks']}")
        print(f"  Avg per turn: {self.metrics['memory_callbacks'] / max(self.metrics['turns'], 1):.2f}")
        print()

        # Save report
        report_path = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f" Full results saved: {report_path}")
        print()

        return report_path


async def monitor_live_session(duration_seconds: int = 300):
    """
    Monitor live noodleMUSH session and collect metrics.

    Args:
        duration_seconds: How long to monitor (default 5 minutes)
    """
    print("╔" + "═"*70 + "╗")
    print("║" + " "*15 + "LIVE COMPARISON TEST - ACTIVE" + " "*26 + "║")
    print("╚" + "═"*70 + "╝")
    print()
    print("Monitoring noodleMUSH conversation with Yuki and Carl...")
    print()
    print("INSTRUCTIONS FOR CADET CAITY:")
    print("  1. Interact with Yuki and Carl in noodleMUSH")
    print("  2. Try various prompts (see suggestions below)")
    print("  3. I'll track metrics in real-time")
    print(f"  4. Test duration: {duration_seconds} seconds")
    print()

    print("SUGGESTED TEST PROMPTS:")
    print("  • 'Yuki, pick up that book'           (tests embodiment)")
    print("  • 'Carl, what do you think about AI?' (tests cynicism)")
    print("  • 'What do you both think of ham?'    (tests separation)")
    print("  • 'Do you remember when we first met?' (tests memory)")
    print("  • 'Yuki, tell me about kami'           (tests cultural)")
    print()

    collector = LiveMetricsCollector()

    # Monitor log file
    log_path = f"logs/cmush_{datetime.now().strftime('%Y-%m-%d')}.log"

    print(f"Monitoring log: {log_path}")
    print()
    print("Collecting data for", end="", flush=True)

    for i in range(duration_seconds):
        await asyncio.sleep(1)
        if i % 10 == 0:
            print(".", end="", flush=True)

        # Analyze logs every 30 seconds
        if i % 30 == 0 and i > 0:
            collector.analyze_log_file(log_path)

    print(" COMPLETE")
    print()

    # Final analysis
    collector.analyze_log_file(log_path)

    # Generate report
    report_file = collector.generate_report()

    return collector.metrics, report_file


async def main():
    """Run live test."""
    print()
    print("=" * 70)
    print("THE ONES WHO WALK AWAY FROM THE CONTEXT WINDOW")
    print("Live Experimental Validation")
    print("=" * 70)
    print()

    # Ask for test duration
    print("How long should we monitor the conversation?")
    print("  (Recommend: 300 seconds = 5 minutes for ~10 turns)")
    print()

    duration = 180  # 3 minutes default

    # Run monitoring
    metrics, report_file = await monitor_live_session(duration)

    print()
    print("╔" + "═"*70 + "╗")
    print("║" + " "*18 + "TEST COMPLETE - RESULTS READY" + " "*23 + "║")
    print("╚" + "═"*70 + "╝")
    print()

    print(f"Data collected and saved to: {report_file}")
    print()
    print("These results will be incorporated into the whitepaper.")
    print()
    print("Commander Spock out. 🖖")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        print("Partial data may be available.")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
