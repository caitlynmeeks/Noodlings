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
#   Subconscious Facet - Continuous symbolic processing
#
#   Runs every cognition cycle, generating symbolic abstracti...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.subconscious_facet
# PURPOSE:  subconscious facet facet implementation
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SubconsciousFacet
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, Any, Optional, Callable, List
import logging
import asyncio

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

    Visual Mode:
    When generate_visual=True, also generates actual images via ImageGenFacet.
    Images are stored in project's Generations folder.
    """

    def __init__(
        self,
        facet_id: str,
        generate_visual: bool = False,
        visual_style: str = "artistic",
        visual_probability: float = 0.3
    ):
        """
        Initialize SubconsciousFacet.

        Args:
            facet_id: Unique identifier
            generate_visual: If True, generate visual images from symbolic text
            visual_style: Style preset for image generation (artistic, fantasy, etc.)
            visual_probability: Probability of generating visual per cycle (0-1)
        """
        self.facet_id = facet_id
        self.last_symbolic_image = None

        # Visual generation settings
        self.generate_visual = generate_visual
        self.visual_style = visual_style
        self.visual_probability = visual_probability

        # Event subscribers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Reference to image generation (set by executor)
        self._image_gen_facet = None
        self._generations_manager = None

        # Stats
        self.visuals_generated = 0
        self.last_visual_path = None

    def set_image_gen_facet(self, facet):
        """Connect to ImageGenFacet for visual generation."""
        self._image_gen_facet = facet
        logger.info(f"[SubconsciousFacet] Connected to ImageGenFacet")

    def set_generations_manager(self, manager):
        """Connect to GenerationsManager for asset storage."""
        self._generations_manager = manager
        logger.info(f"[SubconsciousFacet] Connected to GenerationsManager")

    def on(self, event_type: str, callback: Callable):
        """Subscribe to events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(callback)

    async def emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event to subscribers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error ({event_type}): {e}")

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
            dominance = affect.get('dominance', 0.0)
            sorrow = affect.get('sorrow', 0.0)
        else:
            # Individual affect components from separate connections
            valence = inputs.get('affect_valence', 0.0)
            arousal = inputs.get('affect_arousal', 0.0)
            dominance = inputs.get('affect_dominance', 0.0)
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
- Dominance: {dominance:.2f} (0 submissive, 1 dominant)
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
            # Track activity for ambient visualization
            from .model_activity_tracker import get_model_activity_tracker
            activity_tracker = get_model_activity_tracker()
            request_id = activity_tracker.request_started("SMALL")

            try:
                # Generate symbolic abstraction
                symbolic_image = await llm_client.generate(
                    prompt=symbolic_prompt,
                    system_prompt="You are a poetic subconscious mind generating symbolic imagery.",
                    model="SMALL",  # Use label for fast model routing
                    temperature=0.9,  # High temperature for creative metaphor
                    max_tokens=100
                )
            finally:
                activity_tracker.request_completed("SMALL", request_id)

            # Handle dict responses (some LLM clients return {text: ...})
            if isinstance(symbolic_image, dict):
                symbolic_image = symbolic_image.get('text', symbolic_image.get('content', ''))
            symbolic_image = str(symbolic_image).strip()

            logger.info(f"💭 Subconscious: {symbolic_image[:80]}...")
            print(f"[{agent_name.upper()}] 💭 Subconscious: {symbolic_image[:80]}...")  # For FACETS console
            self.last_symbolic_image = symbolic_image

            result = {
                'symbolic_image': symbolic_image,
                'emotional_signature': {
                    'valence': valence,
                    'arousal': arousal,
                    'dominance': dominance,
                    'sorrow': sorrow
                },
                '_latent': True  # Mark as latent (not for direct output)
            }

            # Visual generation (probabilistic)
            if self.generate_visual and symbolic_image:
                import random
                if random.random() < self.visual_probability:
                    visual_result = await self._generate_visual_imagery(
                        symbolic_image,
                        agent_name,
                        {
                            'valence': valence,
                            'arousal': arousal,
                            'dominance': dominance,
                            'sorrow': sorrow
                        }
                    )
                    if visual_result:
                        result['visual_generated'] = True
                        result['visual_path'] = visual_result.get('path')

            return result

        except Exception as e:
            logger.error(f"Subconscious processing failed: {e}")
            # Fallback: return empty latent
            return {
                'symbolic_image': '',
                'emotional_signature': {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0, 'sorrow': 0.0},
                '_latent': True
            }

    async def _generate_visual_imagery(
        self,
        symbolic_text: str,
        agent_name: str,
        emotional_signature: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate visual imagery from symbolic text.

        Args:
            symbolic_text: The haiku/metaphor to visualize
            agent_name: Agent generating the imagery
            emotional_signature: Affect state for metadata

        Returns:
            Dict with generation result or None
        """
        if not self._image_gen_facet:
            logger.debug("[SubconsciousFacet] No ImageGenFacet connected")
            return None

        try:
            # Build visual prompt from symbolic text
            # Add artistic framing to make it more visual
            visual_prompt = f"Dreamlike surreal imagery: {symbolic_text}"

            # Adjust style based on emotional tone
            valence = emotional_signature.get('valence', 0)
            arousal = emotional_signature.get('arousal', 0.5)

            # Dark/light based on valence
            if valence < -0.3:
                visual_prompt += ", dark moody atmosphere, shadows"
            elif valence > 0.3:
                visual_prompt += ", warm golden light, ethereal glow"

            # Intensity based on arousal
            if arousal > 0.7:
                visual_prompt += ", dynamic movement, vivid colors"
            elif arousal < 0.3:
                visual_prompt += ", calm still, muted tones"

            logger.info(f"[SubconsciousFacet] Generating visual: {visual_prompt[:60]}...")

            # Queue generation (async, non-blocking)
            def on_generated(image):
                """Callback when image is ready."""
                self._on_visual_generated(image, agent_name, symbolic_text, emotional_signature)

            request_id = self._image_gen_facet.queue_generation(
                prompt=visual_prompt,
                style=self.visual_style,
                callback=on_generated
            )

            return {
                'request_id': request_id,
                'prompt': visual_prompt,
                'queued': True
            }

        except Exception as e:
            logger.error(f"[SubconsciousFacet] Visual generation error: {e}")
            return None

    def _on_visual_generated(
        self,
        image,
        agent_name: str,
        symbolic_text: str,
        emotional_signature: Dict[str, float]
    ):
        """Handle generated visual image."""
        try:
            self.visuals_generated += 1

            # Store in generations manager if available
            if self._generations_manager and image:
                stored_path = self._generations_manager.store_generation(
                    image_data=image.image_data,
                    metadata={
                        'source': 'subconscious',
                        'agent': agent_name,
                        'symbolic_text': symbolic_text,
                        'emotional_signature': emotional_signature,
                        'style': self.visual_style,
                        'prompt': image.revised_prompt or symbolic_text,
                        'width': image.width,
                        'height': image.height
                    }
                )
                self.last_visual_path = stored_path
                logger.info(f"[SubconsciousFacet] Visual stored: {stored_path}")

            # Emit event
            asyncio.create_task(self.emit('subconscious_imagery_generated', {
                'agent': agent_name,
                'symbolic_text': symbolic_text,
                'emotional_signature': emotional_signature,
                'path': self.last_visual_path,
                'width': image.width if image else 0,
                'height': image.height if image else 0
            }))

            print(f"[{agent_name.upper()}] 🎨 Subconscious visual generated")

        except Exception as e:
            logger.error(f"[SubconsciousFacet] Visual storage error: {e}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
