# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Action Parser Facet - Extract structured actions from text
#
#   Parses physical action descriptions and emits structured ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.action_parser_facet
# PURPOSE:  action parser facet facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ParsedAction, ActionParserFacet
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedAction:
    """Structured physical action."""
    action_type: str          # 'jump_on', 'bite', 'point', etc.
    target: Optional[str]     # Target agent/object name
    location: Optional[str]   # Body part or spatial location
    emote_text: str          # Formatted emote text
    metadata: Dict[str, Any] # Additional structured data


class ActionParserFacet:
    """
    Parse physical actions from text and emit structured events.

    Example:
        Input: "*jumps on Caity's shoulder cackling*"
        Output: ParsedAction(
            action_type='jump_on',
            target='caity',
            location='shoulder',
            emote_text='jumps on Caity's shoulder cackling',
            metadata={'contact': True, 'intensity': 'moderate'}
        )
    """

    def __init__(self, patterns: List[Dict[str, Any]]):
        """
        Initialize parser with regex patterns.

        Args:
            patterns: List of pattern definitions:
                {
                    'pattern': r'regex with named groups',
                    'action_type': 'jump_on',
                    'emote_template': 'jumps on {target}'s {location}',
                    'metadata': {'contact': True}
                }
        """
        self.patterns = patterns

    def parse(self, text: str) -> List[ParsedAction]:
        """
        Parse text for physical actions.

        Args:
            text: Text containing *action descriptions*

        Returns:
            List of ParsedAction objects
        """
        actions = []

        # Extract all *action* blocks
        action_blocks = re.findall(r'\*(.*?)\*', text, re.DOTALL)

        for block in action_blocks:
            # Try each pattern
            for pattern_def in self.patterns:
                regex = pattern_def['pattern']
                match = re.search(regex, block, re.IGNORECASE)

                if match:
                    # Extract named groups
                    groups = match.groupdict()

                    # Normalize target name (lowercase for agent lookup)
                    target = groups.get('target', '').lower() if groups.get('target') else None

                    # Format emote text (use original matched text for naturalness)
                    emote_text = block.strip()

                    action = ParsedAction(
                        action_type=pattern_def['action_type'],
                        target=target,
                        location=groups.get('location'),
                        emote_text=emote_text,
                        metadata=pattern_def.get('metadata', {})
                    )

                    actions.append(action)
                    logger.debug(f"Parsed action: {action.action_type} -> {emote_text}")
                    break  # Don't try other patterns for this block

        return actions


# Default patterns for fire imp embodiment
DEFAULT_FIRE_IMP_PATTERNS = [
    {
        'pattern': r'jumps? on (?P<target>\w+)\'?s? (?P<location>\w+)',
        'action_type': 'jump_on',
        'emote_template': 'jumps on {target}\'s {location}',
        'metadata': {'contact': True, 'intensity': 'moderate'}
    },
    {
        'pattern': r'bites? (?P<target>\w+)\'?s? (?P<location>\w+)',
        'action_type': 'bite',
        'emote_template': 'bites {target}\'s {location}',
        'metadata': {'contact': True, 'intensity': 'light', 'playful': True}
    },
    {
        'pattern': r'grabs? (?P<target>\w+)\'?s? (?P<item>\w+)',
        'action_type': 'grab',
        'emote_template': 'grabs {target}\'s {item}',
        'metadata': {'contact': True}
    },
    {
        'pattern': r'hugs? (?P<target>\w+)',
        'action_type': 'hug',
        'emote_template': 'hugs {target}',
        'metadata': {'contact': True, 'affection': True}
    },
    {
        'pattern': r'points? at (?P<target>\w+)(?: (?P<manner>accusingly|excitedly|nervously))?',
        'action_type': 'point',
        'emote_template': 'points at {target} {manner}',
        'metadata': {'contact': False}
    },
    {
        'pattern': r'backs? away(?: from (?P<target>\w+))?',
        'action_type': 'back_away',
        'emote_template': 'backs away from {target}',
        'metadata': {'contact': False, 'defensive': True}
    },
    {
        'pattern': r'approaches? (?P<target>\w+)(?: (?P<manner>cautiously|eagerly))?',
        'action_type': 'approach',
        'emote_template': 'approaches {target} {manner}',
        'metadata': {'contact': False}
    },
    {
        'pattern': r'flames (surge|flare|dim|flicker|spike)s?',
        'action_type': 'flame_expression',
        'emote_template': 'flames {0}',
        'metadata': {'contact': False, 'emotional_expression': True}
    },
    {
        'pattern': r'tail (snaps|lashes|whips)',
        'action_type': 'tail_gesture',
        'emote_template': 'tail {0}',
        'metadata': {'contact': False, 'emphasis': True}
    },
    {
        'pattern': r'bounces? on toes',
        'action_type': 'bounce',
        'emote_template': 'bounces on toes',
        'metadata': {'contact': False, 'excited': True}
    },
    {
        'pattern': r'paces? in circles',
        'action_type': 'pace',
        'emote_template': 'paces in circles',
        'metadata': {'contact': False, 'anxious': True}
    },
    {
        'pattern': r'sets? fire to (?:the )?(?P<target>\w+)',
        'action_type': 'set_fire',
        'emote_template': 'sets fire to the {target}',
        'metadata': {'contact': False, 'destructive': True, 'target_type': 'prim'}
    }
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
