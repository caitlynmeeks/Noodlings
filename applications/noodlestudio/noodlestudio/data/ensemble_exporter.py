"""
Ensemble Exporter - Create .ens files from live Noodlings

Reads current Noodlings from noodleMUSH and exports them as ensemble packs.

Workflow:
1. Select Noodlings in Scene Hierarchy
2. File > Export Ensemble
3. Creates .ens file from their current state

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from pathlib import Path
from typing import List, Dict
import requests
import yaml
from .ensemble_packs import NoodlingArchetype, EnsemblePack
from .ensemble_format import EnsembleFormat


class EnsembleExporter:
    """Export live Noodlings as ensemble packs."""

    def __init__(self, api_base: str = "http://localhost:8081/api"):
        self.api_base = api_base

    def export_from_noodlings(
        self,
        noodling_ids: List[str],
        ensemble_name: str,
        ensemble_description: str,
        output_path: Path
    ) -> bool:
        """
        Export selected Noodlings as ensemble.

        Args:
            noodling_ids: List of agent IDs to export
            ensemble_name: Name for the ensemble
            ensemble_description: Description of ensemble
            output_path: Where to save .ens file

        Returns:
            True if export succeeded
        """
        archetypes = []

        for agent_id in noodling_ids:
            archetype = self._noodling_to_archetype(agent_id)
            if archetype:
                archetypes.append(archetype)

        if not archetypes:
            return False

        # Create ensemble pack
        pack = EnsemblePack(
            id=ensemble_name.lower().replace(' ', '_'),
            name=ensemble_name,
            description=ensemble_description,
            version="1.0.0",
            author="Custom",
            price=0.0,
            license_type="free",
            archetypes=archetypes,
            suggested_setting="",
            relationship_dynamics="",
            scene_suggestions=[],
            thumbnail_url="",
            preview_images=[],
            downloads=0,
            rating=0.0,
            tags=["custom"]
        )

        # Save to .ens file
        EnsembleFormat.save_ensemble(pack, output_path)
        return True

    def _noodling_to_archetype(self, agent_id: str) -> NoodlingArchetype:
        """
        Convert live Noodling to archetype.

        Reads recipe YAML and current state from API.
        """
        try:
            # Get current state from API
            resp = requests.get(f"{self.api_base}/agents/{agent_id}", timeout=2)
            if resp.status_code != 200:
                return None

            agent_data = resp.json()

            # Load recipe file for full personality data
            recipe_name = agent_id.replace('agent_', '')
            # Try multiple paths
            recipe_paths = [
                Path(__file__).parent.parent.parent.parent / "cmush" / "recipes" / f"{recipe_name}.yaml",
                Path.home() / "git" / "noodlings_clean" / "applications" / "cmush" / "recipes" / f"{recipe_name}.yaml",
                Path(f"../cmush/recipes/{recipe_name}.yaml"),
            ]

            recipe_path = None
            for path in recipe_paths:
                if path.exists():
                    recipe_path = path
                    break

            if not recipe_path:
                print(f"Recipe not found for {agent_id}: tried {[str(p) for p in recipe_paths]}")
                return None

            with open(recipe_path, 'r') as f:
                recipe = yaml.safe_load(f)

            # Extract personality (now with full Big Five)
            personality = recipe.get('personality', {})

            # Get current affect state (for defaults)
            affect = agent_data.get('affect', {})

            # Create archetype
            archetype = NoodlingArchetype(
                name=recipe.get('name', agent_id),
                species=recipe.get('species', 'noodling'),
                description=recipe.get('description', ''),

                # Big Five
                extraversion=personality.get('extraversion', 0.5),
                agreeableness=personality.get('agreeableness', 0.5),
                conscientiousness=personality.get('conscientiousness', 0.5),
                neuroticism=personality.get('neuroticism', 0.5),
                openness=personality.get('openness', 0.5),

                # Extensions
                curiosity=personality.get('curiosity', 0.5),
                impulsivity=personality.get('impulsivity', 0.5),
                emotional_volatility=personality.get('emotional_volatility', 0.5),

                backstory=recipe.get('identity_prompt', ''),

                # LLM config
                llm_provider=agent_data.get('llm_provider', 'local'),
                llm_model=agent_data.get('llm_model', ''),

                # Current affect as defaults
                default_valence=affect.get('valence', 0.0),
                default_arousal=affect.get('arousal', 0.5),
                default_fear=affect.get('fear', 0.0),
                default_sorrow=affect.get('sorrow', 0.0),
                default_boredom=affect.get('boredom', 0.0),

                tags=[recipe.get('species', 'noodling'), 'custom']
            )

            return archetype

        except Exception as e:
            import traceback
            print(f"Error converting {agent_id} to archetype: {e}")
            traceback.print_exc()
            return None
