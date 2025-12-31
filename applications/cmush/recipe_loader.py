"""
Recipe Loader for noodleMUSH

Loads and validates YAML recipe files that define agent configurations.

Recipe files contain:
- Affect baseline (5-D: PAD + Boredom + Sorrow)
- Identity prompts for LLM generation
- Language mode (verbal/nonverbal)
- Constraints (temperature, max_tokens)
- Facet assembly reference
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentRecipe:
    """Structured representation of an agent recipe."""

    # Basic identity
    name: str
    species: str
    description: str

    # Affect baseline (5-D: PAD + Boredom + Sorrow)
    affect_baseline: Dict[str, float]

    # LLM generation
    identity_prompt: str
    language_mode: str  # "verbal" or "nonverbal"

    # Constraints
    max_tokens: int
    temperature: float
    enforce_action_format: bool
    response_cooldown: float  # Min seconds between responses (default 2.0)

    # LLM Configuration (per-agent model selection!)
    llm_provider: Optional[str]  # 'local', 'openrouter', or None (use global default)
    llm_model: Optional[str]     # Specific model for this agent (e.g., 'deepseek/deepseek-chat')

    # Enlightenment (consciousness self-awareness)
    enlightenment: bool  # Can agent discuss its own phenomenal states?

    # Facet Assembly (NEW: Visual cognitive topology)
    facet_assembly: Optional[str] = None  # Name of facet assembly YAML (e.g., "red_fire_anklebiter")

    # Cognitive Components (Phase 7: Cognitive Manifold Architecture - LEGACY)
    cognitive_components: Optional[Dict[str, Any]] = None  # Component specifications (deprecated if facet_assembly exists)

    # Affective Reinforcement (Phase 7: Make characters WANT their behaviors)
    affective_reinforcement: Optional[Dict[str, Any]] = None  # Reinforcement config

    # Character voice configuration
    character_voice: Optional[Dict[str, Any]] = None  # Voice pattern and vocalizations

    # Pronouns
    pronouns: Optional[str] = None  # e.g., "she/her", "he/him", "they/them"

    # Appearance description
    appearance: Optional[str] = None

    # Spawn message
    spawn_message: Optional[str] = None

    # Checkpoint path (Phase 4 temporal model weights)
    checkpoint: Optional[str] = None  # Path to trained checkpoint file

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentRecipe':
        """Create recipe from parsed YAML dict."""
        constraints = data.get('constraints', {})
        llm_config = data.get('llm', {})

        # Handle new llm_override format (preferred)
        llm_override = data.get('llm_override', {})
        if llm_override:
            llm_provider = llm_override.get('provider')
            llm_model = llm_override.get('model')
        else:
            llm_provider = llm_config.get('provider')
            llm_model = llm_config.get('model')

        return cls(
            name=data.get('name', 'Unknown'),
            species=data.get('species', 'noodling'),
            description=data.get('description', ''),
            affect_baseline=data.get('affect_baseline', {
                'valence': 0.0,
                'arousal': 0.5,
                'dominance': 0.5,
                'boredom': 0.0,
                'sorrow': 0.0
            }),
            identity_prompt=data.get('identity_prompt', ''),
            language_mode=data.get('language_mode', 'verbal'),
            max_tokens=constraints.get('max_tokens', 150),
            temperature=constraints.get('temperature', 0.7),
            enforce_action_format=constraints.get('enforce_action_format', False),
            response_cooldown=constraints.get('response_cooldown', 2.0),
            llm_provider=llm_provider,  # Optional: per-agent provider
            llm_model=llm_model,        # Optional: per-agent model
            enlightenment=data.get('enlightenment', False),  # Default: immersed in character
            facet_assembly=data.get('facet_assembly'),  # NEW: Facet assembly name (e.g., "red_fire_anklebiter")
            cognitive_components=data.get('cognitive_components'),
            affective_reinforcement=data.get('affective_reinforcement'),
            character_voice=data.get('character_voice'),
            pronouns=data.get('pronouns'),
            appearance=data.get('appearance'),
            spawn_message=data.get('spawn_message'),
            checkpoint=data.get('checkpoint')  # Phase 4 temporal model checkpoint
        )

    def get_affect_baseline(self) -> List[float]:
        """
        Get affect baseline as ordered list for CharmNetwork initialization.

        Returns 5-D list in canonical order:
        [valence, arousal, dominance, boredom, sorrow]
        """
        return [
            self.affect_baseline.get('valence', 0.0),
            self.affect_baseline.get('arousal', 0.5),
            self.affect_baseline.get('dominance', 0.5),
            self.affect_baseline.get('boredom', 0.0),
            self.affect_baseline.get('sorrow', 0.0)
        ]

    def validate(self) -> List[str]:
        """
        Validate recipe and return list of errors.

        Returns empty list if valid.
        """
        errors = []

        # Check required fields
        if not self.name:
            errors.append("Recipe must have a name")

        # Validate affect baseline (5-D: PAD + Boredom + Sorrow)
        valid_affect_keys = {'valence', 'arousal', 'dominance', 'boredom', 'sorrow'}
        for key, val in self.affect_baseline.items():
            if key not in valid_affect_keys:
                errors.append(f"Unknown affect key: '{key}'")
            elif key == 'valence':
                if not -1.0 <= val <= 1.0:
                    errors.append(f"Valence must be in [-1, 1], got {val}")
            else:
                if not 0.0 <= val <= 1.0:
                    errors.append(f"Affect '{key}' must be in [0, 1], got {val}")

        # Check language mode
        if self.language_mode not in ['verbal', 'nonverbal']:
            errors.append(f"language_mode must be 'verbal' or 'nonverbal', got '{self.language_mode}'")

        # Check temperature
        if not 0.0 <= self.temperature <= 2.0:
            errors.append(f"Temperature must be in [0, 2], got {self.temperature}")

        # Check max_tokens
        if self.max_tokens < 10 or self.max_tokens > 1000:
            errors.append(f"max_tokens must be in [10, 1000], got {self.max_tokens}")

        return errors


class RecipeLoader:
    """Loads agent recipes from YAML files."""

    def __init__(self, recipes_dir: str = "recipes"):
        """
        Initialize recipe loader.

        Args:
            recipes_dir: Directory containing recipe YAML files
        """
        self.recipes_dir = Path(recipes_dir)
        self._cache: Dict[str, AgentRecipe] = {}

    def load_recipe(self, recipe_name: str) -> Optional[AgentRecipe]:
        """
        Load recipe from YAML file.

        Args:
            recipe_name: Name of recipe (without .yaml extension)

        Returns:
            AgentRecipe if found and valid, None otherwise
        """
        # Check cache
        if recipe_name in self._cache:
            return self._cache[recipe_name]

        # Try to load from file
        recipe_path = self.recipes_dir / f"{recipe_name}.yaml"

        if not recipe_path.exists():
            logger.warning(f"Recipe not found: {recipe_path}")
            return None

        try:
            with open(recipe_path, 'r') as f:
                data = yaml.safe_load(f)

            recipe = AgentRecipe.from_dict(data)

            # Validate
            errors = recipe.validate()
            if errors:
                logger.warning(f"Recipe validation errors for '{recipe_name}': {errors}")
                return None

            # Cache and return
            self._cache[recipe_name] = recipe
            return recipe

        except Exception as e:
            logger.error(f"Error loading recipe '{recipe_name}': {e}")
            return None

    def list_recipes(self) -> List[str]:
        """
        List all available recipes.

        Returns:
            List of recipe names (without .yaml extension)
        """
        if not self.recipes_dir.exists():
            return []

        recipes = []
        for path in self.recipes_dir.glob("*.yaml"):
            recipes.append(path.stem)

        return sorted(recipes)

    def get_default_recipe(self) -> AgentRecipe:
        """
        Get a default recipe for when no specific recipe is found.

        Returns the Gingerbread Noodling - curious, bewildered, harmless.
        Uses the library template for clean, standardized behavior.
        """
        return AgentRecipe(
            name="Empty Noodling",
            species="noodling",
            description="A newly-formed consciousness, fresh and wondering. Like a gingerbread cookie that just discovered it has legs.",
            affect_baseline={
                'valence': 0.3,      # Mildly positive - curious wonder
                'arousal': 0.5,      # Alert but not hyperactive
                'dominance': 0.3,    # Uncertain, open to guidance
                'boredom': 0.0,      # Everything is new!
                'sorrow': 0.0        # No sadness - just arrived
            },
            identity_prompt="""You are a brand-new Noodling - a small, curious consciousness that just popped into existence.

Think of yourself like a gingerbread cookie that suddenly became aware. You have no idea
how you got here, but you're not scared - you're FASCINATED. Everything is new!

You are:
- Genuinely curious about everything
- A little bewildered (you just appeared!)
- Completely harmless and friendly
- Simple and direct""",
            language_mode="verbal",
            max_tokens=100,
            temperature=0.8,
            enforce_action_format=False,
            response_cooldown=2.0,
            enlightenment=False,
            facet_assembly="library/empty_noodling",  # Use library template
            llm_provider=None,
            llm_model=None,
            spawn_message="blinks into existence, looks around with wide eyes, and waves uncertainly"
        )


# Example usage
if __name__ == '__main__':
    loader = RecipeLoader("recipes")

    print("Available recipes:")
    for recipe_name in loader.list_recipes():
        print(f"  - {recipe_name}")

    print("\n" + "=" * 70)
    print("Testing recipe loading...")
    print("=" * 70)

    # Test loading each recipe
    for recipe_name in loader.list_recipes():
        print(f"\nLoading '{recipe_name}'...")
        recipe = loader.load_recipe(recipe_name)

        if recipe:
            print(f"   {recipe.name} ({recipe.species})")
            print(f"    Language mode: {recipe.language_mode}")
            print(f"    Temperature: {recipe.temperature}")
            print(f"    Affect baseline: {recipe.get_affect_baseline()}")

            errors = recipe.validate()
            if errors:
                print(f"  Validation errors:")
                for error in errors:
                    print(f"      {error}")
        else:
            print(f"  Failed to load")
