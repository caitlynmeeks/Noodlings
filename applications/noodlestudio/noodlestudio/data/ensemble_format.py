"""
Ensemble Format (.ens) - Prefab system for Noodling groups

.ens files are JSON-based prefabs containing:
- Multiple Noodling archetypes
- Relationship dynamics
- Scene suggestions
- World building metadata

Like Unity prefabs, but for consciousness ensembles!

File format: JSON with .ens extension
Location: ~/.noodlestudio/ensembles/

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from pathlib import Path
from typing import Dict, List, Any
import json
from .ensemble_packs import EnsemblePack, NoodlingArchetype

# Alias for compatibility
EnsembleSpawner = None  # Will be set below
from dataclasses import asdict


class EnsembleFormat:
    """
    Read/write .ens ensemble prefab files.

    Format:
    {
        "format_version": "1.0",
        "ensemble": {
            "id": "commedia_dellarte",
            "name": "Commedia dell'Arte",
            "description": "...",
            "archetypes": [...],
            "metadata": {...}
        }
    }
    """

    FORMAT_VERSION = "1.0"

    @staticmethod
    def save_ensemble(pack: EnsemblePack, output_path: Path):
        """
        Save ensemble pack as .ens file.

        Args:
            pack: EnsemblePack to save
            output_path: Path to .ens file
        """
        ensemble_data = {
            "format_version": EnsembleFormat.FORMAT_VERSION,
            "ensemble": {
                "id": pack.id,
                "name": pack.name,
                "description": pack.description,
                "version": pack.version,
                "author": pack.author,
                "price": pack.price,
                "license_type": pack.license_type,
                "archetypes": [
                    {
                        "name": arch.name,
                        "species": arch.species,
                        "description": arch.description,
                        "personality": {
                            "extraversion": arch.extraversion,
                            "agreeableness": arch.agreeableness,
                            "conscientiousness": arch.conscientiousness,
                            "neuroticism": arch.neuroticism,
                            "openness": arch.openness,
                            "curiosity": arch.curiosity,
                            "impulsivity": arch.impulsivity,
                            "emotional_volatility": arch.emotional_volatility,
                        },
                        "backstory": arch.backstory,
                        "llm_provider": arch.llm_provider,
                        "llm_model": arch.llm_model,
                        "default_affect": {
                            "valence": arch.default_valence,
                            "arousal": arch.default_arousal,
                            "fear": arch.default_fear,
                            "sorrow": arch.default_sorrow,
                            "boredom": arch.default_boredom,
                        },
                        "tags": arch.tags,
                    }
                    for arch in pack.archetypes
                ],
                "world_building": {
                    "suggested_setting": pack.suggested_setting,
                    "relationship_dynamics": pack.relationship_dynamics,
                    "scene_suggestions": pack.scene_suggestions,
                },
                "metadata": {
                    "thumbnail_url": pack.thumbnail_url,
                    "preview_images": pack.preview_images,
                    "downloads": pack.downloads,
                    "rating": pack.rating,
                    "tags": pack.tags,
                },
            }
        }

        # Ensure .ens extension
        if output_path.suffix != '.ens':
            output_path = output_path.with_suffix('.ens')

        with open(output_path, 'w') as f:
            json.dump(ensemble_data, f, indent=2)

        print(f"Ensemble saved: {output_path}")

    @staticmethod
    def load_ensemble(ens_path: Path) -> EnsemblePack:
        """
        Load ensemble pack from .ens file.

        Args:
            ens_path: Path to .ens file

        Returns:
            EnsemblePack instance
        """
        with open(ens_path, 'r') as f:
            data = json.load(f)

        # Validate format version
        format_version = data.get('format_version', '1.0')
        if format_version != EnsembleFormat.FORMAT_VERSION:
            print(f"Warning: Ensemble format version mismatch ({format_version} vs {EnsembleFormat.FORMAT_VERSION})")

        ens = data['ensemble']

        # Reconstruct archetypes
        archetypes = []
        for arch_data in ens['archetypes']:
            pers = arch_data['personality']
            affect = arch_data['default_affect']

            archetype = NoodlingArchetype(
                name=arch_data['name'],
                species=arch_data['species'],
                description=arch_data['description'],
                extraversion=pers['extraversion'],
                agreeableness=pers['agreeableness'],
                conscientiousness=pers['conscientiousness'],
                neuroticism=pers['neuroticism'],
                openness=pers['openness'],
                curiosity=pers['curiosity'],
                impulsivity=pers['impulsivity'],
                emotional_volatility=pers['emotional_volatility'],
                backstory=arch_data['backstory'],
                llm_provider=arch_data['llm_provider'],
                llm_model=arch_data['llm_model'],
                default_valence=affect['valence'],
                default_arousal=affect['arousal'],
                default_fear=affect['fear'],
                default_sorrow=affect['sorrow'],
                default_boredom=affect['boredom'],
                tags=arch_data['tags'],
            )
            archetypes.append(archetype)

        world = ens['world_building']
        meta = ens['metadata']

        pack = EnsemblePack(
            id=ens['id'],
            name=ens['name'],
            description=ens['description'],
            version=ens['version'],
            author=ens['author'],
            price=ens['price'],
            license_type=ens['license_type'],
            archetypes=archetypes,
            suggested_setting=world['suggested_setting'],
            relationship_dynamics=world['relationship_dynamics'],
            scene_suggestions=world['scene_suggestions'],
            thumbnail_url=meta['thumbnail_url'],
            preview_images=meta['preview_images'],
            downloads=meta['downloads'],
            rating=meta['rating'],
            tags=meta['tags'],
        )

        print(f"Ensemble loaded: {pack.name} ({len(pack.archetypes)} archetypes)")
        return pack

    @staticmethod
    def export_built_in_ensembles():
        """
        Export all built-in ensembles to .ens files.

        Creates ~/.noodlestudio/ensembles/ with all default packs.
        """
        from .ensemble_packs import ENSEMBLE_LIBRARY

        ensembles_dir = Path.home() / ".noodlestudio" / "ensembles"
        ensembles_dir.mkdir(parents=True, exist_ok=True)

        for pack in ENSEMBLE_LIBRARY.list_packs():
            output_path = ensembles_dir / f"{pack.id}.ens"
            EnsembleFormat.save_ensemble(pack, output_path)

        print(f"Exported {len(ENSEMBLE_LIBRARY.packs)} ensembles to {ensembles_dir}")

    @staticmethod
    def list_available_ensembles() -> List[Path]:
        """
        List all .ens files in ensembles directory.

        Returns:
            List of .ens file paths
        """
        ensembles_dir = Path.home() / ".noodlestudio" / "ensembles"
        if not ensembles_dir.exists():
            return []

        return list(ensembles_dir.glob("*.ens"))


class EnsembleRezzer:
    """
    Rez entire ensembles into noodleMUSH.

    Like importing a Unity prefab!
    """

    @staticmethod
    def rez_ensemble(pack: EnsemblePack, room_id: str = "room_000") -> List[str]:
        """
        Rez all archetypes from ensemble into specified room.

        Args:
            pack: EnsemblePack to rez
            room_id: Target room ID

        Returns:
            List of rezzed Noodling IDs
        """
        rezzed_ids = []

        print(f"Rezzing ensemble: {pack.name}")
        print(f"Target room: {room_id}")
        print(f"Archetypes: {len(pack.archetypes)}")

        for archetype in pack.archetypes:
            print(f"  - Rezzing {archetype.name} ({archetype.species})...")

            # TODO: Send to noodleMUSH API
            # For now, just simulate
            noodling_id = f"agent_{archetype.name.lower().replace(' ', '_')}"

            noodling_config = {
                'id': noodling_id,
                'name': archetype.name,
                'species': archetype.species,
                'description': archetype.description,
                'personality': {
                    'extraversion': archetype.extraversion,
                    'agreeableness': archetype.agreeableness,
                    'conscientiousness': archetype.conscientiousness,
                    'neuroticism': archetype.neuroticism,
                    'openness': archetype.openness,
                    'curiosity': archetype.curiosity,
                    'impulsivity': archetype.impulsivity,
                    'emotional_volatility': archetype.emotional_volatility,
                },
                'backstory': archetype.backstory,
                'llm_provider': archetype.llm_provider,
                'llm_model': archetype.llm_model,
                'initial_affect': {
                    'valence': archetype.default_valence,
                    'arousal': archetype.default_arousal,
                    'fear': archetype.default_fear,
                    'sorrow': archetype.default_sorrow,
                    'boredom': archetype.default_boredom,
                },
                'room': room_id,
            }

            # TODO: POST to /api/agents/rez
            print(f"    Config: {noodling_config}")

            rezzed_ids.append(noodling_id)

        print(f"Rezzed {len(rezzed_ids)} Noodlings from ensemble: {pack.name}")
        print(f"Suggested scene: {pack.scene_suggestions[0] if pack.scene_suggestions else 'None'}")

        return rezzed_ids


# Alias for backwards compatibility
EnsembleSpawner = EnsembleRezzer
