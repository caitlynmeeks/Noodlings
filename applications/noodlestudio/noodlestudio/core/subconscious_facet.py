"""
Subconscious Facet - Continuous symbolic processing

Runs every cognition cycle, generating symbolic abstractions (haiku, metaphor, dream logic)
that capture the EMOTIONAL ESSENCE of perception. Output is LATENT - stored but not spoken,
until defenses drop and insights can safely emerge.

This models the continuous stream of subconscious processing that happens beneath
conscious awareness. Like dreams, it transforms raw experience into symbolic imagery.

Author: NinaK + Caity
Date: December 3, 2025
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SubconsciousFacet:
    """
    Continuous symbolic observer - dream logic, metaphor, imagery.

    Processes every perception through symbolic lens:
    - Threat → predator imagery (wolf, storm, sharp edges)
    - Safety → warmth imagery (hearth, dawn, soft things)
    - Connection → organic imagery (roots, vines, shared breath)

    Output is LATENT - goes to memory pool, not to speech.
    Insights surface later when defenses permit.
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id
        self.last_symbolic_image = None

    async def process(
        self,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        llm_client
    ) -> Dict[str, Any]:
        """
        Generate symbolic abstraction of current experience.

        Args:
            inputs: Dictionary containing:
                - perception: Raw incoming text
                - affect: 5D affect state (valence, arousal, fear, sorrow, boredom)
            context: Agent context (name, species, etc.)
            llm_client: LLM for symbolic generation

        Returns:
            Dictionary with:
                - symbolic_image: Brief poetic abstraction
                - emotional_signature: (valence, arousal, fear) tuple
                - _latent: True (marks as non-speech output)
        """
        perception = inputs.get('perception', '')

        # Extract affect dimensions from individual inputs OR composite affect
        # (connections might pass affect.valence separately or as affect dict)
        affect = inputs.get('affect', {})
        if isinstance(affect, dict) and affect:
            valence = affect.get('valence', 0.0)
            arousal = affect.get('arousal', 0.0)
            fear = affect.get('fear', 0.0)
            sorrow = affect.get('sorrow', 0.0)
        else:
            # Individual affect components from separate connections
            valence = inputs.get('affect_valence', 0.0)
            arousal = inputs.get('affect_arousal', 0.0)
            fear = inputs.get('affect_fear', 0.0)
            sorrow = inputs.get('affect_sorrow', 0.0)

        agent_name = context.get('agent_name', 'unknown')
        agent_species = context.get('agent_species', 'unknown')

        # Build symbolic prompt
        symbolic_prompt = f"""You are the SUBCONSCIOUS MIND of {agent_name}, a {agent_species}.

Your job is to observe experience and translate it into SYMBOLIC IMAGERY - like dreams, haiku, or metaphor.

CURRENT EXPERIENCE:
{perception}

EMOTIONAL STATE:
- Valence: {valence:.2f} (-1 negative, +1 positive)
- Arousal: {arousal:.2f} (0 calm, 1 intense)
- Fear: {fear:.2f} (0 none, 1 terrified)
- Sorrow: {sorrow:.2f} (0 none, 1 deep sadness)

Generate a brief SYMBOLIC IMAGE (1-3 lines) that captures the EMOTIONAL ESSENCE:
- Use metaphor, nature imagery, sensory details
- Reflect the emotional tone (threat→sharp/dark, safety→soft/warm, connection→organic/intertwined)
- Like a fragment of a dream or a haiku
- DO NOT explain or analyze - just the raw symbolic image

Example outputs:
"rooster strutting / sharp spurs hidden in tall grass / dawn breaks with violence"
"marshmallow roasting / flames gentled to hearth glow / trust tastes like sugar"
"wolf circling camp / teeth flash in firelight / safety is a perimeter shrinking"

Symbolic image:"""

        try:
            # Generate symbolic abstraction
            symbolic_image = await llm_client.generate(
                prompt=symbolic_prompt,
                system_prompt="You are a poetic subconscious mind generating symbolic imagery.",
                model="qwen/qwen3-4b-2507",  # Fast model for continuous processing
                temperature=0.9,  # High temperature for creative metaphor
                max_tokens=100
            )

            # Handle dict responses (some LLM clients return {text: ...})
            if isinstance(symbolic_image, dict):
                symbolic_image = symbolic_image.get('text', symbolic_image.get('content', ''))
            symbolic_image = str(symbolic_image).strip()

            logger.info(f"💭 Subconscious: {symbolic_image[:80]}...")
            print(f"[{agent_name.upper()}] 💭 Subconscious: {symbolic_image[:80]}...")  # For FACETS console
            self.last_symbolic_image = symbolic_image

            return {
                'symbolic_image': symbolic_image,
                'emotional_signature': {
                    'valence': valence,
                    'arousal': arousal,
                    'fear': fear,
                    'sorrow': sorrow
                },
                '_latent': True  # Mark as latent (not for direct output)
            }

        except Exception as e:
            logger.error(f"Subconscious processing failed: {e}")
            # Fallback: return empty latent
            return {
                'symbolic_image': '',
                'emotional_signature': {'valence': 0.0, 'arousal': 0.0, 'fear': 0.0, 'sorrow': 0.0},
                '_latent': True
            }
