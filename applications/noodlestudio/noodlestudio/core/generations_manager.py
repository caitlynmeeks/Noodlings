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
#   Generations Manager - Storage and retrieval of AI-generated assets.
#
#   Handles all AI-generated content (images, audio, etc.) wi...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.generations_manager
# PURPOSE:  Generations Manager
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   GenerationMetadata, GenerationsManager, get_generations_manager()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetadata:
    """Metadata for a generated asset."""
    id: str
    source: str  # 'subconscious', 'scripted', 'manual', etc.
    created_at: str
    type: str  # 'image', 'audio', etc.

    # Common fields
    agent: str = ""
    prompt: str = ""

    # Image-specific
    width: int = 0
    height: int = 0
    style: str = ""

    # Context
    emotional_signature: Dict[str, float] = field(default_factory=dict)
    symbolic_text: str = ""

    # File info
    filename: str = ""
    filepath: str = ""
    thumbnail_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationMetadata':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class GenerationsManager:
    """
    Manages storage and retrieval of AI-generated content.

    Provides organized storage in project's Generations folder with
    metadata tracking and thumbnail support.
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize GenerationsManager.

        Args:
            base_path: Base path for Generations folder (auto-detected if None)
        """
        self._base_path = base_path
        self._generations: List[GenerationMetadata] = []
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Ensure directories exist
        self._ensure_directories()

        # Load existing metadata
        self._load_metadata_index()

    def _get_base_path(self) -> Path:
        """Get or create base path for generations."""
        if self._base_path:
            return Path(self._base_path)

        # Default to library/Generations in noodlestudio
        lib_path = Path(__file__).parent.parent.parent / "library" / "Generations"
        return lib_path

    def _ensure_directories(self):
        """Create directory structure if needed."""
        base = self._get_base_path()

        # Create type subdirectories
        (base / "Images").mkdir(parents=True, exist_ok=True)
        (base / "Audio").mkdir(parents=True, exist_ok=True)
        (base / "Thumbnails").mkdir(parents=True, exist_ok=True)

        logger.info(f"[GenerationsManager] Using path: {base}")

    def _get_month_folder(self, gen_type: str = "Images") -> Path:
        """Get current month's folder, creating if needed."""
        base = self._get_base_path() / gen_type
        month_folder = base / datetime.now().strftime("%Y-%m")
        month_folder.mkdir(parents=True, exist_ok=True)
        return month_folder

    def _load_metadata_index(self):
        """Load metadata index from disk."""
        index_path = self._get_base_path() / "index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self._generations = [
                        GenerationMetadata.from_dict(item)
                        for item in data.get('generations', [])
                    ]
                logger.info(f"[GenerationsManager] Loaded {len(self._generations)} generations")
            except Exception as e:
                logger.error(f"[GenerationsManager] Failed to load index: {e}")
                self._generations = []

    def _save_metadata_index(self):
        """Save metadata index to disk."""
        index_path = self._get_base_path() / "index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump({
                    'generations': [g.to_dict() for g in self._generations],
                    'updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"[GenerationsManager] Failed to save index: {e}")

    # ========== Event System ==========

    def on(self, event_type: str, callback: Callable):
        """Subscribe to events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(callback)

    def _emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event to subscribers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error ({event_type}): {e}")

    # ========== Image Storage ==========

    def store_generation(
        self,
        image_data: bytes,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Store a generated image with metadata.

        Args:
            image_data: PNG image bytes
            metadata: Generation metadata dict

        Returns:
            Path to stored image
        """
        # Generate ID
        gen_id = f"img_{uuid.uuid4().hex[:12]}"
        filename = f"{gen_id}.png"

        # Get storage path
        folder = self._get_month_folder("Images")
        filepath = folder / filename

        # Write image
        with open(filepath, 'wb') as f:
            f.write(image_data)

        # Create metadata record
        meta = GenerationMetadata(
            id=gen_id,
            source=metadata.get('source', 'unknown'),
            created_at=datetime.now().isoformat(),
            type='image',
            agent=metadata.get('agent', ''),
            prompt=metadata.get('prompt', ''),
            width=metadata.get('width', 0),
            height=metadata.get('height', 0),
            style=metadata.get('style', ''),
            emotional_signature=metadata.get('emotional_signature', {}),
            symbolic_text=metadata.get('symbolic_text', ''),
            filename=filename,
            filepath=str(filepath)
        )

        # Write metadata sidecar
        meta_path = folder / f"{gen_id}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta.to_dict(), f, indent=2)

        # Generate thumbnail
        thumb_path = self._generate_thumbnail(filepath, gen_id)
        meta.thumbnail_path = str(thumb_path) if thumb_path else ""

        # Add to index
        self._generations.append(meta)
        self._save_metadata_index()

        logger.info(f"[GenerationsManager] Stored: {filepath}")

        # Emit event
        self._emit('generation_stored', {
            'id': gen_id,
            'path': str(filepath),
            'metadata': meta.to_dict()
        })

        return str(filepath)

    def _generate_thumbnail(self, image_path: Path, gen_id: str) -> Optional[Path]:
        """Generate thumbnail for image."""
        try:
            from PIL import Image

            thumb_folder = self._get_base_path() / "Thumbnails"
            thumb_path = thumb_folder / f"{gen_id}_thumb.png"

            # Create thumbnail (128x128)
            with Image.open(image_path) as img:
                img.thumbnail((128, 128), Image.Resampling.LANCZOS)
                img.save(thumb_path, "PNG")

            return thumb_path

        except ImportError:
            logger.debug("[GenerationsManager] PIL not available for thumbnails")
            return None
        except Exception as e:
            logger.error(f"[GenerationsManager] Thumbnail error: {e}")
            return None

    # ========== Retrieval ==========

    def get_all_generations(self, gen_type: Optional[str] = None) -> List[GenerationMetadata]:
        """
        Get all generations, optionally filtered by type.

        Args:
            gen_type: Filter by type ('image', 'audio', etc.)

        Returns:
            List of GenerationMetadata
        """
        if gen_type:
            return [g for g in self._generations if g.type == gen_type]
        return list(self._generations)

    def get_by_agent(self, agent_name: str) -> List[GenerationMetadata]:
        """Get all generations by specific agent."""
        return [g for g in self._generations if g.agent == agent_name]

    def get_by_source(self, source: str) -> List[GenerationMetadata]:
        """Get all generations from specific source (subconscious, etc.)."""
        return [g for g in self._generations if g.source == source]

    def get_recent(self, limit: int = 20) -> List[GenerationMetadata]:
        """Get most recent generations."""
        sorted_gens = sorted(
            self._generations,
            key=lambda g: g.created_at,
            reverse=True
        )
        return sorted_gens[:limit]

    def get_by_id(self, gen_id: str) -> Optional[GenerationMetadata]:
        """Get generation by ID."""
        for gen in self._generations:
            if gen.id == gen_id:
                return gen
        return None

    def search(self, query: str) -> List[GenerationMetadata]:
        """Search generations by prompt or symbolic text."""
        query_lower = query.lower()
        results = []
        for gen in self._generations:
            if (query_lower in gen.prompt.lower() or
                query_lower in gen.symbolic_text.lower() or
                query_lower in gen.agent.lower()):
                results.append(gen)
        return results

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        images = [g for g in self._generations if g.type == 'image']
        audio = [g for g in self._generations if g.type == 'audio']

        # Count by source
        by_source = {}
        for gen in self._generations:
            by_source[gen.source] = by_source.get(gen.source, 0) + 1

        # Count by agent
        by_agent = {}
        for gen in self._generations:
            if gen.agent:
                by_agent[gen.agent] = by_agent.get(gen.agent, 0) + 1

        return {
            'total': len(self._generations),
            'images': len(images),
            'audio': len(audio),
            'by_source': by_source,
            'by_agent': by_agent,
            'base_path': str(self._get_base_path())
        }

    # ========== Management ==========

    def delete_generation(self, gen_id: str) -> bool:
        """Delete a generation and its files."""
        gen = self.get_by_id(gen_id)
        if not gen:
            return False

        try:
            # Delete main file
            if gen.filepath and os.path.exists(gen.filepath):
                os.remove(gen.filepath)

            # Delete metadata sidecar
            meta_path = Path(gen.filepath).with_suffix('.json')
            if meta_path.exists():
                os.remove(meta_path)

            # Delete thumbnail
            if gen.thumbnail_path and os.path.exists(gen.thumbnail_path):
                os.remove(gen.thumbnail_path)

            # Remove from index
            self._generations = [g for g in self._generations if g.id != gen_id]
            self._save_metadata_index()

            logger.info(f"[GenerationsManager] Deleted: {gen_id}")
            return True

        except Exception as e:
            logger.error(f"[GenerationsManager] Delete error: {e}")
            return False

    def clear_all(self) -> int:
        """Clear all generations. Returns count deleted."""
        count = len(self._generations)

        for gen in list(self._generations):
            self.delete_generation(gen.id)

        self._emit('generations_cleared', {'count': count})

        return count


# ========== Global Singleton ==========

_generations_manager_instance = None


def get_generations_manager(base_path: Optional[str] = None) -> GenerationsManager:
    """Get global GenerationsManager singleton."""
    global _generations_manager_instance

    if _generations_manager_instance is None:
        _generations_manager_instance = GenerationsManager(base_path)

    return _generations_manager_instance

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
