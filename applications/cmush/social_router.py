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
#   Social Router - Conversation Response Decisions
#
#   In a room with multiple agents and humans, how does a Noodling
#   know when someone is talking TO them? This module uses simple
#   heuristics (not expensive LLM calls) to decide: Was my name
#   mentioned? Are we alone? Is this a question? Then affect
#   modulates the response probability - a bored agent is more
#   likely to jump into conversations than a content one.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.social_router
# PURPOSE:  Heuristic-based conversation response decisions
# LAYER:    Backend / Agent Response
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SocialRouter          Decides if agent should respond to message
#
# KEY FUNCTIONS:
#   agent_name_in_text()  Check if name appears in message
#   is_question()         Detect questions
#   is_command()          Detect imperative commands
#   recent_exchange()     Check for ongoing conversation thread
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# Author: Caitlyn + Claude
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Social Router - Simple heuristic-based conversation routing.

Replaces unreliable LLM-based addressee parsing with deterministic logic.
Uses affect to modulate response probability, not as a boolean gate.
"""

import re
from typing import Tuple, List, Dict, Optional, Any


class SocialRouter:
    """
    Dead-simple heuristics for "should I respond?"
    NO LLM CALLS. Just logic + affect modulation.
    """

    @staticmethod
    def should_respond(
        message: str,
        speaker_id: str,
        agent_name: str,
        stage: Optional[Any],
        agent_affect: Dict[str, float],
        conversation_history: List[Dict],
        agent_id: Optional[str] = None
    ) -> Tuple[bool, float, str]:
        """
        Decide if agent should respond to this message.

        Args:
            message: The incoming message text
            speaker_id: ID of the person who spoke
            agent_name: Name of this agent
            stage: Stage object (or None if not using stage model yet)
            agent_affect: Current affect state (valence, arousal, boredom, etc.)
            conversation_history: Recent conversation messages
            agent_id: Optional agent ID for history matching

        Returns:
            (should_respond: bool, confidence: float, reason: str)
        """

        # 1. Name mentioned? HIGH confidence
        if agent_name_in_text(message, agent_name):
            return (True, 0.95, "name_mentioned")

        # 2. One-on-one conversation? HIGH confidence (if we have stage data)
        if stage:
            try:
                agent_entity = stage.entities.get(agent_id or agent_name)
                if agent_entity and agent_entity.zone:
                    zone = agent_entity.zone
                    people_in_zone = [
                        eid for eid in stage.zones.get(zone, [])
                        if stage.entities[eid].entity_type in ['user', 'agent']
                    ]
                    # Just speaker and agent in the room
                    if len(people_in_zone) == 2:
                        return (True, 0.9, "one_on_one")
            except (AttributeError, KeyError):
                # Stage data incomplete, skip this check
                pass

        # 3. Recent conversation thread? MEDIUM confidence
        if recent_exchange(conversation_history, speaker_id, agent_id or agent_name, window=3):
            return (True, 0.6, "recent_thread")

        # 4. Question or command? Affect-modulated
        if is_question(message):
            # Bored/high arousal = more likely to respond
            arousal = agent_affect.get('arousal', 0.5)
            boredom = agent_affect.get('boredom', 0.0)

            response_probability = 0.3 + (arousal * 0.3) + (boredom * 0.2)
            if response_probability > 0.5:
                return (True, response_probability, "question_with_interest")

        if is_command(message):
            # Commands are more attention-grabbing
            arousal = agent_affect.get('arousal', 0.5)
            boredom = agent_affect.get('boredom', 0.0)

            response_probability = 0.4 + (arousal * 0.3) + (boredom * 0.2)
            if response_probability > 0.5:
                return (True, response_probability, "command_with_interest")

        # 5. Default: observe silently
        return (False, 0.0, "not_addressed")


def agent_name_in_text(text: str, agent_name: str) -> bool:
    """
    Check if agent name (or first word) appears in text.
    Handles multi-word names like "Red Fire Anklebiter".
    """
    text_lower = text.lower()
    name_lower = agent_name.lower()

    # Full name check
    if name_lower in text_lower:
        return True

    # First word (handle spaces and underscores)
    # "Red Fire Anklebiter" -> "red"
    # "red_fire_anklebiter" -> "red"
    first_word = name_lower.split(' ')[0].split('_')[0]

    # Word boundary check (avoid matching "red" in "bored")
    pattern = r'\b' + re.escape(first_word) + r'\b'
    if re.search(pattern, text_lower):
        return True

    return False


def is_question(text: str) -> bool:
    """Detect questions."""
    if '?' in text:
        return True

    text_lower = text.lower().strip()
    question_starters = (
        'who', 'what', 'when', 'where', 'why', 'how',
        'can', 'will', 'would', 'could', 'should',
        'do', 'does', 'did', 'is', 'are', 'was', 'were'
    )

    first_word = text_lower.split()[0] if text_lower else ''
    return first_word in question_starters


def is_command(text: str) -> bool:
    """Detect imperative commands."""
    imperatives = [
        'go', 'come', 'stop', 'wait', 'look', 'get', 'take', 'give',
        'sit', 'stand', 'roast', 'burn', 'light', 'tell', 'show',
        'help', 'move', 'open', 'close', 'say', 'speak'
    ]

    text_lower = text.lower().strip()
    first_word = text_lower.split()[0] if text_lower else ''
    return first_word in imperatives


def recent_exchange(
    history: List[Dict],
    speaker_id: str,
    agent_id: str,
    window: int = 3
) -> bool:
    """
    Check if recent messages involved both parties.
    Indicates an ongoing conversation thread.
    """
    if not history or len(history) < 2:
        return False

    recent = history[-window:]

    # Extract speaker IDs from various message formats
    speakers = set()
    for msg in recent:
        if isinstance(msg, dict):
            # Try different field names
            speaker = msg.get('speaker') or msg.get('user_id') or msg.get('agent_id')
            if speaker:
                speakers.add(speaker)

    # Check if both parties were involved
    return speaker_id in speakers and agent_id in speakers

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
