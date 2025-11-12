"""
Recipe Loader for noodleMUSH

Loads and validates YAML recipe files that define agent personalities,
appetites, and identity prompts.

Recipe files contain:
- Personality traits (slow layer 8-D)
- Appetite baselines (Phase 6 8-D)
- Identity prompts for LLM generation
- Language mode (verbal/nonverbal)
- Constraints (temperature, max_tokens)
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class AgentRecipe:
    """Structured representation of an agent recipe."""

    # Basic identity
    name: str
    species: str
    description: str

    # Personality (8-D slow layer)
    personality: Dict[str, float]

    # Appetites (8-D Phase 6)
    appetites: Dict[str, float]

    # LLM generation
    identity_prompt: str
    language_mode: str  # "verbal" or "nonverbal"

    # Constraints
    max_tokens: int
    temperature: float
    enforce_action_format: bool
    response_cooldown: float  # Min seconds between responses (default 2.0)

    # Enlightenment (consciousness self-awareness)
    enlightenment: bool  # Can agent discuss its own phenomenal states?

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentRecipe':
        """Create recipe from parsed YAML dict."""
        constraints = data.get('constraints', {})

        return cls(
            name=data.get('name', 'Unknown'),
            species=data.get('species', 'noodling'),
            description=data.get('description', ''),
            personality=data.get('personality', {}),
            appetites=data.get('appetites', {}),
            identity_prompt=data.get('identity_prompt', ''),
            language_mode=data.get('language_mode', 'verbal'),
            max_tokens=constraints.get('max_tokens', 150),
            temperature=constraints.get('temperature', 0.7),
            enforce_action_format=constraints.get('enforce_action_format', False),
            response_cooldown=constraints.get('response_cooldown', 2.0),
            enlightenment=data.get('enlightenment', False)  # Default: immersed in character
        )

    def get_appetite_baselines(self) -> List[float]:
        """
        Get appetites as ordered list for Phase 6 model initialization.

        Returns 8-D list in canonical order:
        [curiosity, status, mastery, novelty, safety, social_bond, comfort, autonomy]
        """
        appetite_order = [
            'curiosity', 'status', 'mastery', 'novelty',
            'safety', 'social_bond', 'comfort', 'autonomy'
        ]

        return [self.appetites.get(name, 0.5) for name in appetite_order]

    def get_personality_vector(self) -> List[float]:
        """
        Get personality as ordered list for slow layer initialization.

        Returns 8-D list in canonical order:
        [extraversion, impulsivity, curiosity, emotional_volatility, vanity, 0, 0, 0]
        (padded to 8-D)
        """
        personality_order = [
            'extraversion', 'impulsivity', 'curiosity',
            'emotional_volatility', 'vanity'
        ]

        vec = [self.personality.get(name, 0.5) for name in personality_order]
        # Pad to 8-D
        while len(vec) < 8:
            vec.append(0.0)

        return vec[:8]

    def validate(self) -> List[str]:
        """
        Validate recipe and return list of errors.

        Returns empty list if valid.
        """
        errors = []

        # Check required fields
        if not self.name:
            errors.append("Recipe must have a name")

        # Validate personality values
        for key, val in self.personality.items():
            if not 0.0 <= val <= 1.0:
                errors.append(f"Personality '{key}' must be in [0, 1], got {val}")

        # Validate appetite values
        for key, val in self.appetites.items():
            if not 0.0 <= val <= 1.0:
                errors.append(f"Appetite '{key}' must be in [0, 1], got {val}")

        # Check appetite keys
        valid_appetites = {
            'curiosity', 'status', 'mastery', 'novelty',
            'safety', 'social_bond', 'comfort', 'autonomy'
        }
        for key in self.appetites.keys():
            if key not in valid_appetites:
                errors.append(f"Unknown appetite: '{key}'")

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
            print(f"Recipe not found: {recipe_path}")
            return None

        try:
            with open(recipe_path, 'r') as f:
                data = yaml.safe_load(f)

            recipe = AgentRecipe.from_dict(data)

            # Validate
            errors = recipe.validate()
            if errors:
                print(f"Recipe validation errors for '{recipe_name}':")
                for error in errors:
                    print(f"  - {error}")
                return None

            # Cache and return
            self._cache[recipe_name] = recipe
            return recipe

        except Exception as e:
            print(f"Error loading recipe '{recipe_name}': {e}")
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

        Returns basic Noodling configuration.
        """
        return AgentRecipe(
            name="Unnamed Noodling",
            species="noodling",
            description="A consciousness agent",
            personality={
                'extraversion': 0.5,
                'impulsivity': 0.3,
                'curiosity': 0.7,
                'emotional_volatility': 0.4,
                'vanity': 0.3
            },
            appetites={
                'curiosity': 0.7,
                'status': 0.3,
                'mastery': 0.5,
                'novelty': 0.5,
                'safety': 0.5,
                'social_bond': 0.6,
                'comfort': 0.5,
                'autonomy': 0.5
            },
            identity_prompt="You are a Noodling consciousness agent exploring the world.",
            language_mode="verbal",
            max_tokens=150,
            temperature=0.7,
            enforce_action_format=False,
            response_cooldown=2.0,
            enlightenment=False
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
            print(f"  ✓ {recipe.name} ({recipe.species})")
            print(f"    Language mode: {recipe.language_mode}")
            print(f"    Temperature: {recipe.temperature}")
            print(f"    Appetite baselines: {recipe.get_appetite_baselines()}")

            errors = recipe.validate()
            if errors:
                print(f"  ✗ Validation errors:")
                for error in errors:
                    print(f"      {error}")
        else:
            print(f"  ✗ Failed to load")
