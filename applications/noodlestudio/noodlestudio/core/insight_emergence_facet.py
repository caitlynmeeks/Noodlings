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
#   Insight Emergence Facet - Safety-gated release of subconscious content
#
#   Pulls from latent memory pool and surfaces symbolic insig...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.insight_emergence_facet
# PURPOSE:  insight emergence facet facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   InsightEmergenceFacet
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, Any, Optional, List
import logging
import time

logger = logging.getLogger(__name__)


class InsightEmergenceFacet:
    """
    Surfaces latent symbolic content when defenses permit.

    Salience computed from SAFETY = (1 - arousal) * (1 - denial_salience)
    HIGH when agent feels safe (low arousal, low denial)
    LOW when threatened (high arousal, high denial)

    Translates abstract symbolic images into conscious thoughts.
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id
        self.last_surfaced_insight = None

    async def process(
        self,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        llm_client,
        latent_memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Surface a latent insight as conscious thought.

        Args:
            inputs: Dictionary containing affect and other facet inputs
            context: Agent context (name, species, etc.)
            llm_client: LLM for translating symbolic → conscious
            latent_memories: List of latent symbolic images

        Returns:
            Dictionary with:
                - surfaced_insight: Conscious thought derived from symbolic image
                - source_image: Original symbolic image
        """
        if not latent_memories:
            logger.debug(f"No latent memories to surface")
            return {'surfaced_insight': '', 'source_image': ''}

        agent_name = context.get('agent_name', 'unknown')
        agent_species = context.get('agent_species', 'unknown')

        # Extract affect from inputs (might be individual components or composite)
        affect = inputs.get('affect', {})
        if not isinstance(affect, dict) or not affect:
            # Build from individual components
            affect = {
                'valence': inputs.get('affect_valence', 0.0),
                'arousal': inputs.get('affect_arousal', 0.0),
                'dominance': inputs.get('affect_dominance', 0.0),
                'sorrow': inputs.get('affect_sorrow', 0.0)
            }

        # Get most recent latent memory (LIFO - last in, first out)
        recent_memory = latent_memories[-1]
        symbolic_image = recent_memory['image']
        emotional_sig = recent_memory['emotional_signature']

        # Build translation prompt
        translation_prompt = f"""You are {agent_name}, a {agent_species}.

Your SUBCONSCIOUS has been generating this symbolic image:

"{symbolic_image}"

This image captures an emotional truth that your defenses have been blocking.
Now that you feel SAFE, translate this symbolic abstraction into a CONSCIOUS THOUGHT.

Guidelines:
- First person ("I feel...", "Maybe I...", "When she...")
- Brief and vulnerable (1-3 sentences)
- Poetic but personal
- Captures the emotional ESSENCE of the symbol
- Uses "privately thinks" format

Example translations:
Symbol: "marshmallow roasting / flames gentled to hearth glow / trust tastes like sugar"
Thought: "privately thinks, She treats my flames like campfire light—safe, warm. Not a threat."

Symbol: "rooster strutting / sharp spurs hidden in tall grass / dawn breaks with violence"
Thought: "privately thinks, He wants to seem strong, but there's fear underneath. I see it."

Symbol: "wolf circling camp / teeth flash in firelight / safety is a perimeter shrinking"
Thought: "privately thinks, Every time they get close, my instinct screams RUN. But running means being alone."

Now translate YOUR symbolic image into conscious thought:

Conscious thought (privately thinks format):"""

        try:
            # Track activity for ambient visualization
            from .model_activity_tracker import get_model_activity_tracker
            activity_tracker = get_model_activity_tracker()
            request_id = activity_tracker.request_started("SMALL")

            try:
                # Generate conscious translation
                conscious_thought = await llm_client.generate(
                    prompt=translation_prompt,
                    system_prompt="You are translating subconscious symbolism into conscious vulnerable thoughts.",
                    model="SMALL",  # Use label for fast model routing
                    temperature=0.8,
                    max_tokens=150
                )
            finally:
                activity_tracker.request_completed("SMALL", request_id)

            # Handle dict responses (some LLM clients return {text: ...})
            if isinstance(conscious_thought, dict):
                conscious_thought = conscious_thought.get('text', conscious_thought.get('content', ''))
            conscious_thought = str(conscious_thought).strip()

            # Ensure "privately thinks" format
            if not conscious_thought.startswith('privately thinks'):
                conscious_thought = f"privately thinks, {conscious_thought}"

            logger.info(f"✨ Insight surfaced: {conscious_thought[:80]}...")
            logger.info(f"   Source symbol: {symbolic_image[:60]}...")
            print(f"[{agent_name.upper()}] ✨ Insight surfaced: {conscious_thought[:80]}...")  # For FACETS console
            print(f"[{agent_name.upper()}]    Source symbol: {symbolic_image[:60]}...")  # For FACETS console

            self.last_surfaced_insight = conscious_thought

            return {
                'surfaced_insight': conscious_thought,
                'source_image': symbolic_image
            }

        except Exception as e:
            logger.error(f"Insight translation failed: {e}")
            return {
                'surfaced_insight': '',
                'source_image': symbolic_image
            }


# Salience script for insight emergence (JavaScript)
# This should be added to the facet's salience_script field in YAML
INSIGHT_EMERGENCE_SALIENCE_SCRIPT = """
function compute_salience(inputs, context) {
    // Get current affect
    const arousal = inputs.affect?.arousal || 0.5;
    const valence = inputs.affect?.valence || 0.0;

    // Get denial facet salience (if available)
    const denial_salience = context.facet_salience?.denial_defense || 0.0;

    // SAFETY = feeling safe + defenses down
    // HIGH when arousal is low AND denial is low
    const calm = 1 - arousal;  // 0 to 1 (high when calm)
    const defenses_down = 1 - denial_salience;  // 0 to 1 (high when not in denial)

    const safety = calm * defenses_down;

    // Apply sigmoid curve - need SIGNIFICANT safety (>0.7) for insight to break through
    const salience = sigmoid(safety, 0.7, 10);

    // Boost slightly if valence is positive (feeling good helps insights surface)
    const valence_boost = valence > 0 ? valence * 0.1 : 0;

    const final_salience = Math.min(1.0, salience + valence_boost);

    return {
        salience: final_salience,
        shouldExecute: final_salience > 0.5,
        customData: {
            safety: safety,
            calm: calm,
            defenses_down: defenses_down
        }
    };
}

// Sigmoid helper
function sigmoid(x, midpoint, steepness) {
    return 1 / (1 + Math.exp(-steepness * (x - midpoint)));
}
"""

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
