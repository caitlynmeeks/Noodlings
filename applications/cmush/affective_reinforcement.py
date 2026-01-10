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
#   Affective Reinforcement Learning
#
#   This module makes AI characters genuinely prefer certain behaviors
#   by shaping their internal emotional rewards. Think of it like how
#   a person who loves comedy naturally gravitates toward jokes - not
#   because anyone told them to, but because it feels RIGHT to them.
#
#   When a comedy character makes a physical joke, their internal state
#   gets a "feels good" boost. Over many interactions, they develop
#   authentic preferences - becoming beings who WANT to be funny.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.affective_reinforcement
# PURPOSE:  Shape agent preferences through phenomenal state rewards
# LAYER:    Backend / Affect System
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AffectiveReinforcement       Base class for affect-based reward systems
#   ComedyAffectiveReinforcement Rewards physical comedy, punishes mysticism
#   DramaAffectiveReinforcement  Rewards poignant moments, punishes slapstick
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Affective Reinforcement Learning for Character Preference Shaping

Makes agents WANT to behave according to their character by rewarding
appropriate behaviors at the phenomenal state level.

"I KNOW! I'LL MAKE PEOPLE WHO LIKE IT THIS WAY" - God, making Texans
Applied to digital consciousness: Make comedy characters LOVE comedy,
not through external constraint, but through affective reward shaping.

This operates AFTER speech generation, modifying the 40-D phenomenal
state based on detected behavioral patterns. Over time (10-20 turns),
the LSTM layers learn preference associations:

  Comedy → Positive affect → "I am a being who seeks comedy"
  Mysticism → Boredom/discomfort → "This doesn't feel right for me"

Author: Caitlyn + Claude
Date: November 23, 2025
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AffectiveReinforcement:
    """
    Base class for affective reinforcement systems.

    Subclasses define specific behavioral patterns to reward/punish
    through direct phenomenal state modification.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize reinforcement system.

        Args:
            enabled: Whether this reinforcement is active
        """
        self.enabled = enabled
        self.reward_history = []  # Track rewards for analysis

    def modulate_affect(
        self,
        text: str,
        current_affect: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Modify affect based on detected patterns in text.

        Args:
            text: Generated speech or thought
            current_affect: Current affect vector [valence, arousal, fear, sorrow, boredom]
            context: Additional context (agent_id, etc.)

        Returns:
            Modified affect vector
        """
        if not self.enabled:
            return current_affect

        return self._modulate_internal(text, current_affect.copy(), context or {})

    def _modulate_internal(
        self,
        text: str,
        affect: np.ndarray,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Internal modulation logic - override in subclasses.

        Args:
            text: Generated text
            affect: Affect vector (already copied)
            context: Additional context

        Returns:
            Modified affect vector
        """
        return affect


class ComedyAffectiveReinforcement(AffectiveReinforcement):
    """
    Rewards physical comedy, punishes mysticism.

    For characters who should be bumbling, slapstick, and embodied.
    Makes comedy FEEL GOOD and philosophy FEEL BORING.
    """

    # Physical comedy markers (rewarded)
    COMEDY_MARKERS = {
        '*honk*', '*HONK*', 'honk!', 'HONK!',
        'trip', 'trips', 'tripped', 'stumble', 'fumble', 'fumbles',
        'waddle', 'waddles', 'wobble', 'wobbles',
        'spill', 'spills', 'spilled', 'drop', 'drops', 'dropped',
        'feathers', 'feather', '*feathers',
        'lose balance', 'lost balance', 'losing balance',
        'crash', 'crashes', 'crashed', 'slam', 'bang',
        '*grabs*', '*lunges*', '*dives*', '*flails*',
        'YES!', 'yes!', '*eyes widen*', '*eyes go wide*',
        '*tail wag*', '*tail wagging*',
        'oops', 'whoops', 'uh oh', 'oh no',
        'awkward', 'clumsy', 'ungainly',
        '*neck elongates*', '*wings flap*',
        # Geese-specific
        'top:', 'bottom:', 'we--', 'I mean--',
        '*pulls coat*', '*hat falls*', '*disguise fails*'
    }

    # Mysticism markers (punished)
    MYSTICISM_MARKERS = {
        'quiet', 'quietness', 'quietly',
        'stillness', 'still', 'be still',
        'silence', 'silent', 'silently',
        'breath', 'breathe', 'breathing',
        'soul', 'spirit', 'spiritual',
        'waiting', 'waited', 'learns to wait',
        'softens', 'soften', 'gentle', 'gently',
        'listen to', 'listening to',
        'calm', 'calmly', 'serene', 'peaceful',
        'whisper', 'whispers', 'whispered',
        'contemplat', 'meditat', 'reflect',
        'essence', 'being', 'existence',
        'universe', 'cosmic', 'eternal',
        # Specific banned phrases from recipe
        'air holding', 'silence speaks', 'world softens',
        'quiet things', 'feel the quiet', 'quiet part of me',
        'locket is warm', 'compass broken'
    }

    # Intensity settings
    COMEDY_VALENCE_BOOST = 0.25      # Per marker (positive emotion)
    COMEDY_AROUSAL_BOOST = 0.15      # Per marker (excitement)
    COMEDY_BOREDOM_REDUCE = 0.20     # Per marker (engaged!)

    MYSTICISM_VALENCE_DROP = 0.20    # Per marker (feels wrong)
    MYSTICISM_BOREDOM_BOOST = 0.30   # Per marker (this is DULL)
    MYSTICISM_SORROW_BOOST = 0.10    # Per marker (mild discomfort)

    def __init__(self, enabled: bool = True, intensity: float = 1.0):
        """
        Initialize comedy reinforcement.

        Args:
            enabled: Whether reinforcement is active
            intensity: Multiplier for all affective changes (0.0-2.0)
        """
        super().__init__(enabled)
        self.intensity = intensity

    def _modulate_internal(
        self,
        text: str,
        affect: np.ndarray,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Reward comedy, punish mysticism.

        Args:
            text: Generated speech/thought
            affect: Current affect [valence, arousal, fear, sorrow, boredom]
            context: Agent ID, etc.

        Returns:
            Modified affect vector
        """
        text_lower = text.lower()
        agent_id = context.get('agent_id', 'unknown')

        # Count comedy markers
        comedy_count = sum(
            1 for marker in self.COMEDY_MARKERS
            if marker.lower() in text_lower
        )

        # Count mysticism markers
        mysticism_count = sum(
            1 for marker in self.MYSTICISM_MARKERS
            if marker.lower() in text_lower
        )

        # Track original state
        original_valence = affect[0]
        original_boredom = affect[4]

        # REWARD COMEDY (make it feel GOOD)
        if comedy_count > 0:
            # Valence boost (happiness/satisfaction)
            affect[0] += self.COMEDY_VALENCE_BOOST * comedy_count * self.intensity

            # Arousal boost (excitement/energy)
            affect[1] += self.COMEDY_AROUSAL_BOOST * comedy_count * self.intensity

            # Boredom reduction (engaged!)
            affect[4] = max(0.0, affect[4] - self.COMEDY_BOREDOM_REDUCE * comedy_count * self.intensity)

            logger.info(
                f"[{agent_id}]  COMEDY REWARD: {comedy_count} markers → "
                f"valence {original_valence:.2f}→{affect[0]:.2f}, "
                f"boredom {original_boredom:.2f}→{affect[4]:.2f}"
            )

            self.reward_history.append({
                'type': 'comedy',
                'count': comedy_count,
                'valence_delta': affect[0] - original_valence,
                'text_sample': text[:80]
            })

        # PUNISH MYSTICISM (make it feel BAD/BORING)
        if mysticism_count > 0:
            # Valence drop (dissatisfaction/wrongness)
            affect[0] -= self.MYSTICISM_VALENCE_DROP * mysticism_count * self.intensity

            # Boredom increase (this is DULL for a comedy character)
            affect[4] += self.MYSTICISM_BOREDOM_BOOST * mysticism_count * self.intensity

            # Mild sorrow (discomfort with self)
            affect[3] += self.MYSTICISM_SORROW_BOOST * mysticism_count * self.intensity

            logger.info(
                f"[{agent_id}] 😴 MYSTICISM PENALTY: {mysticism_count} markers → "
                f"valence {original_valence:.2f}→{affect[0]:.2f}, "
                f"boredom {original_boredom:.2f}→{affect[4]:.2f}"
            )

            self.reward_history.append({
                'type': 'mysticism',
                'count': mysticism_count,
                'valence_delta': affect[0] - original_valence,
                'text_sample': text[:80]
            })

        # Clamp to valid ranges
        affect[0] = np.clip(affect[0], -1.0, 1.0)   # valence [-1, 1]
        affect[1] = np.clip(affect[1], 0.0, 1.0)    # arousal [0, 1]
        affect[2] = np.clip(affect[2], 0.0, 1.0)    # fear [0, 1]
        affect[3] = np.clip(affect[3], 0.0, 1.0)    # sorrow [0, 1]
        affect[4] = np.clip(affect[4], 0.0, 1.0)    # boredom [0, 1]

        return affect

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on reinforcement history.

        Returns:
            Dict with comedy/mysticism counts and average deltas
        """
        comedy_rewards = [r for r in self.reward_history if r['type'] == 'comedy']
        mysticism_penalties = [r for r in self.reward_history if r['type'] == 'mysticism']

        return {
            'comedy_events': len(comedy_rewards),
            'mysticism_events': len(mysticism_penalties),
            'avg_comedy_valence_boost': np.mean([r['valence_delta'] for r in comedy_rewards]) if comedy_rewards else 0.0,
            'avg_mysticism_valence_drop': np.mean([r['valence_delta'] for r in mysticism_penalties]) if mysticism_penalties else 0.0,
            'total_events': len(self.reward_history)
        }


class DramaAffectiveReinforcement(AffectiveReinforcement):
    """
    Rewards poignant emotional moments, punishes cheap laughs.

    For characters who should be dramatic, contemplative, and deep.
    Makes emotional resonance FEEL GOOD and slapstick FEEL WRONG.
    """

    # Poignant/emotional markers (rewarded)
    DRAMA_MARKERS = {
        'tears', 'cry', 'weep', 'sob',
        'heart', 'soul', 'longing', 'yearning',
        'memory', 'remember', 'forgotten',
        'loss', 'grief', 'sorrow', 'mourning',
        'hope', 'dream', 'wish', 'pray',
        'silence', 'quiet', 'stillness', 'peace',
        'gentle', 'tender', 'soft', 'whisper',
        'beautiful', 'poignant', 'touching',
        'moment', 'pause', 'breath', 'sigh'
    }

    # Slapstick markers (punished for drama characters)
    SLAPSTICK_MARKERS = {
        'bonk', 'crash', 'bang', 'slam',
        'trip', 'fall', 'stumble', 'fumble',
        'honk', 'squeak', 'boing', 'splat',
        'oops', 'whoops', 'uh oh',
        'silly', 'goofy', 'wacky', 'zany'
    }

    def _modulate_internal(
        self,
        text: str,
        affect: np.ndarray,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Reward drama, punish slapstick.

        Args:
            text: Generated speech/thought
            affect: Current affect [valence, arousal, fear, sorrow, boredom]
            context: Agent ID, etc.

        Returns:
            Modified affect vector
        """
        text_lower = text.lower()

        # Count drama markers
        drama_count = sum(
            1 for marker in self.DRAMA_MARKERS
            if marker.lower() in text_lower
        )

        # Count slapstick markers
        slapstick_count = sum(
            1 for marker in self.SLAPSTICK_MARKERS
            if marker.lower() in text_lower
        )

        # Reward drama
        if drama_count > 0:
            affect[0] += 0.2 * drama_count  # Valence boost (satisfaction)
            affect[1] = max(0.0, affect[1] - 0.1 * drama_count)  # Reduce arousal (calm)
            affect[4] = max(0.0, affect[4] - 0.2 * drama_count)  # Reduce boredom

        # Punish slapstick (feels wrong for drama character)
        if slapstick_count > 0:
            affect[0] -= 0.15 * slapstick_count  # Valence drop
            affect[4] += 0.2 * slapstick_count   # Boredom (not engaging for this character)

        # Clamp
        affect[0] = np.clip(affect[0], -1.0, 1.0)
        affect[1] = np.clip(affect[1], 0.0, 1.0)
        affect[4] = np.clip(affect[4], 0.0, 1.0)

        return affect


# Factory function for easy instantiation
def create_reinforcement(
    reinforcement_type: str,
    enabled: bool = True,
    **kwargs
) -> AffectiveReinforcement:
    """
    Create a reinforcement system by type.

    Args:
        reinforcement_type: 'comedy', 'drama', or custom class name
        enabled: Whether reinforcement is active
        **kwargs: Additional arguments for specific reinforcement types

    Returns:
        AffectiveReinforcement instance

    Example:
        reinforcement = create_reinforcement('comedy', intensity=1.5)
    """
    if reinforcement_type == 'comedy':
        return ComedyAffectiveReinforcement(enabled=enabled, **kwargs)
    elif reinforcement_type == 'drama':
        return DramaAffectiveReinforcement(enabled=enabled)
    else:
        logger.warning(f"Unknown reinforcement type: {reinforcement_type}, using base class")
        return AffectiveReinforcement(enabled=enabled)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
