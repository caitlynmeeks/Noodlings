"""
Speech Gate Facet - Prevents speech spam with cooldown

Implements minimum interval between speech outputs.
Pure continuous dynamics - no discrete "can speak / can't speak" flags.

Author: Caitlyn + Claude
Date: December 4, 2025
"""

import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SpeechGateFacet:
    """
    Speech gate with cooldown to prevent spam.

    Uses continuous salience to implement smooth cooldown curve.
    Not a binary gate - gradual transition from "recently spoke" to "ready to speak again".
    """

    def __init__(self, min_interval: float = 15.0):
        """
        Initialize speech gate.

        Args:
            min_interval: Minimum seconds between speech outputs
        """
        self.min_interval = min_interval
        self.last_speech_time = 0.0

    def process(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gate speech based on cooldown.

        Args:
            inputs: Input dict with 'response' field
            context: Execution context with time info

        Returns:
            Output dict with gated response
        """
        response = inputs.get('response', '')

        # Calculate time since last speech
        current_time = time.time()
        time_since_speech = current_time - self.last_speech_time

        # Continuous salience curve (not binary!)
        # Sigmoid rises from 0 to 1 as time passes
        if time_since_speech < self.min_interval:
            # Too soon - suppress output
            # Use sigmoid for smooth transition
            cooldown_progress = time_since_speech / self.min_interval
            salience = self._sigmoid(cooldown_progress, 0.8, 10)

            if salience < 0.5:
                # Cooldown still active, suppress
                logger.debug(f"Speech gate: cooldown active ({time_since_speech:.1f}s < {self.min_interval}s), suppressing")
                return {'out': '[SUPPRESS]'}

        # Gate open - allow speech through
        if response and response != '[SUPPRESS]' and response != '[No output]':
            self.last_speech_time = current_time
            logger.debug(f"Speech gate: allowing speech through (cooldown: {time_since_speech:.1f}s)")

        return {'out': response}

    def _sigmoid(self, x: float, center: float = 0.5, steepness: float = 10) -> float:
        """Sigmoid function for smooth transitions."""
        import math
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

    def reset(self):
        """Reset gate state."""
        self.last_speech_time = 0.0
        logger.debug("Speech gate reset")
